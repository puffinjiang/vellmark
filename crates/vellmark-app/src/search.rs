//! Quick Open fuzzy search (`Cmd/Ctrl + P`, PRD 4.5 F-06) and Global Workspace Search (PRD 4.5 F-07).

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

use vellmark_core::{ByteOffset, FileTreeNode, detect_and_decode};

pub(crate) const GLOBAL_SEARCH_MAX_RESULTS: usize = 500;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuickOpenMatch {
    pub path: PathBuf,
    pub relative_path: String,
    pub score: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceSearchMatch {
    pub path: PathBuf,
    pub line_number: usize,
    pub line_byte_offset: ByteOffset,
    pub line_preview: String,
}

/// Fuzzy search workspace files in `tree` by `query` (case-insensitive, PRD 4.5 F-06).
pub fn quick_open_fuzzy_search(
    tree: &FileTreeNode,
    root: &Path,
    query: &str,
) -> Vec<QuickOpenMatch> {
    let mut files = Vec::new();
    collect_files(tree, root, &mut files);

    let trimmed = query.trim();
    if trimmed.is_empty() {
        return files
            .into_iter()
            .map(|(path, relative_path)| QuickOpenMatch {
                path,
                relative_path,
                score: 0,
            })
            .collect();
    }

    let q_lower = trimmed.to_lowercase();
    let mut matches: Vec<QuickOpenMatch> = files
        .into_iter()
        .filter_map(|(path, rel)| {
            fuzzy_score(&rel.to_lowercase(), &q_lower).map(|score| QuickOpenMatch {
                path,
                relative_path: rel,
                score,
            })
        })
        .collect();

    matches.sort_by(|a, b| {
        b.score
            .cmp(&a.score)
            .then_with(|| a.relative_path.len().cmp(&b.relative_path.len()))
            .then_with(|| a.relative_path.cmp(&b.relative_path))
    });
    matches
}

/// Unbounded convenience helper for unit tests; runtime callers use bounded/cancellable `WorkspaceSearchWorker`.
#[cfg(test)]
pub fn global_workspace_search(
    root: &Path,
    query: &str,
    case_sensitive: bool,
) -> io::Result<Vec<WorkspaceSearchMatch>> {
    global_workspace_search_with_cancel(
        root,
        query,
        case_sensitive,
        &AtomicBool::new(false),
        usize::MAX,
    )
}

#[derive(Default)]
struct CaseFoldScratch {
    folded: String,
    orig_offsets: Vec<(usize, usize)>,
}

fn push_folded_char(out: &mut String, ch: char) {
    if ch.is_ascii() {
        out.push(ch.to_ascii_lowercase());
    } else if ch == '\u{03C2}' {
        // Unify Greek final sigma (ς) with medial/non-final sigma (σ) so query and line
        // fold identically without context-sensitive Final_Sigma mismatches.
        out.push('\u{03C3}');
    } else {
        for lc in ch.to_lowercase() {
            out.push(if lc == '\u{03C2}' { '\u{03C3}' } else { lc });
        }
    }
}

fn fold_query_string(query: &str) -> String {
    let mut folded = String::with_capacity(query.len());
    for ch in query.chars() {
        push_folded_char(&mut folded, ch);
    }
    folded
}

fn find_query_in_line(
    clean_line: &str,
    needle: &str,
    case_sensitive: bool,
    scratch: &mut CaseFoldScratch,
) -> Option<usize> {
    if needle.is_empty() {
        return None;
    }
    if case_sensitive {
        return clean_line.find(needle);
    }
    if needle.is_ascii() && clean_line.is_ascii() {
        let line_bytes = clean_line.as_bytes();
        let needle_bytes = needle.as_bytes();
        return line_bytes
            .windows(needle_bytes.len())
            .position(|w| w.eq_ignore_ascii_case(needle_bytes));
    }

    scratch.folded.clear();
    scratch.orig_offsets.clear();
    for (orig_idx, ch) in clean_line.char_indices() {
        let folded_start = scratch.folded.len();
        scratch.orig_offsets.push((folded_start, orig_idx));
        push_folded_char(&mut scratch.folded, ch);
    }

    let match_pos = scratch.folded.find(needle)?;
    let idx = match scratch
        .orig_offsets
        .binary_search_by_key(&match_pos, |&(folded_start, _)| folded_start)
    {
        Ok(i) => i,
        Err(i) => i.saturating_sub(1),
    };
    scratch.orig_offsets.get(idx).map(|&(_, orig_idx)| orig_idx)
}

/// Cooperatively cancellable and result-bounded global workspace search for background threads.
pub fn global_workspace_search_with_cancel(
    root: &Path,
    query: &str,
    case_sensitive: bool,
    cancelled: &AtomicBool,
    max_results: usize,
) -> io::Result<Vec<WorkspaceSearchMatch>> {
    if query.is_empty() || max_results == 0 || cancelled.load(Ordering::Acquire) {
        return Ok(Vec::new());
    }
    let tree = FileTreeNode::scan_dir(root)?;
    if cancelled.load(Ordering::Acquire) {
        return Ok(Vec::new());
    }
    let mut files = Vec::new();
    collect_files(&tree, root, &mut files);

    let q_needle = if case_sensitive {
        query.to_string()
    } else {
        fold_query_string(query)
    };

    let mut results = Vec::new();
    let mut scratch = CaseFoldScratch::default();
    for (path, _) in files {
        if cancelled.load(Ordering::Relaxed) || results.len() >= max_results {
            break;
        }
        let Ok(raw) = fs::read(&path) else {
            continue;
        };
        if cancelled.load(Ordering::Relaxed) {
            break;
        }
        let Ok((text, _)) = detect_and_decode(&raw) else {
            continue;
        };
        let mut line_start_offset = 0usize;
        for (line_idx, raw_line) in text.split_inclusive('\n').enumerate() {
            if line_idx % 256 == 0 && cancelled.load(Ordering::Relaxed) {
                return Ok(Vec::new());
            }
            let clean_line = raw_line.trim_end_matches(&['\r', '\n'][..]);
            if let Some(rel_col) =
                find_query_in_line(clean_line, &q_needle, case_sensitive, &mut scratch)
            {
                results.push(WorkspaceSearchMatch {
                    path: path.clone(),
                    line_number: line_idx + 1,
                    line_byte_offset: line_start_offset + rel_col,
                    line_preview: clean_line.trim().to_string(),
                });
                if results.len() >= max_results {
                    break;
                }
            }
            line_start_offset += raw_line.len();
        }
    }

    if cancelled.load(Ordering::Relaxed) {
        return Ok(Vec::new());
    }
    Ok(results)
}

#[derive(Clone)]
pub(crate) struct WorkspaceSearchRequest {
    pub req_id: u64,
    pub root: PathBuf,
    pub query: String,
    pub case_sensitive: bool,
    pub cancel: Arc<AtomicBool>,
}

pub(crate) struct WorkspaceSearchResponse {
    pub req_id: u64,
    pub result: io::Result<Vec<WorkspaceSearchMatch>>,
}

pub(crate) struct SearchDrainOutcome {
    pub responses: Vec<WorkspaceSearchResponse>,
    pub disconnected: bool,
}

#[derive(Default)]
struct SearchQueueState {
    pending: Option<WorkspaceSearchRequest>,
    active_cancel: Option<Arc<AtomicBool>>,
    shutdown: bool,
}

struct ClearActiveCancel<'a> {
    queue: &'a Arc<(Mutex<SearchQueueState>, Condvar)>,
}

impl Drop for ClearActiveCancel<'_> {
    fn drop(&mut self) {
        let (lock, _) = &**self.queue;
        if let Ok(mut guard) = lock.lock() {
            guard.active_cancel = None;
        }
    }
}

type SearchNotifier = Arc<Mutex<Option<Arc<dyn Fn() + Send + Sync>>>>;

/// Single background worker thread for global workspace search with latest-request coalescing.
pub(crate) struct WorkspaceSearchWorker {
    queue: Arc<(Mutex<SearchQueueState>, Condvar)>,
    rx: Receiver<WorkspaceSearchResponse>,
    notifier: SearchNotifier,
    handle: Option<JoinHandle<()>>,
}

impl WorkspaceSearchWorker {
    pub fn spawn(notifier: Option<Arc<dyn Fn() + Send + Sync>>, max_results: usize) -> Self {
        let queue = Arc::new((Mutex::new(SearchQueueState::default()), Condvar::new()));
        let (tx, rx) = mpsc::channel();
        let notifier_slot: SearchNotifier = Arc::new(Mutex::new(notifier));

        let worker_queue = Arc::clone(&queue);
        let worker_notifier = Arc::clone(&notifier_slot);
        let handle = thread::Builder::new()
            .name("vellmark-ws-search".into())
            .spawn(move || loop {
                let req = {
                    let (lock, cv) = &*worker_queue;
                    let mut guard = match lock.lock() {
                        Ok(g) => g,
                        Err(e) => e.into_inner(),
                    };
                    loop {
                        if guard.shutdown {
                            return;
                        }
                        if let Some(req) = guard.pending.take() {
                            guard.active_cancel = Some(Arc::clone(&req.cancel));
                            break req;
                        }
                        guard = match cv.wait(guard) {
                            Ok(g) => g,
                            Err(e) => e.into_inner(),
                        };
                    }
                };
                let _clear_active = ClearActiveCancel {
                    queue: &worker_queue,
                };

                if req.cancel.load(Ordering::Acquire) {
                    continue;
                }
                let result = global_workspace_search_with_cancel(
                    &req.root,
                    &req.query,
                    req.case_sensitive,
                    &req.cancel,
                    max_results,
                );
                if req.cancel.load(Ordering::Acquire) {
                    continue;
                }
                if tx
                    .send(WorkspaceSearchResponse {
                        req_id: req.req_id,
                        result,
                    })
                    .is_ok()
                    && let Ok(guard) = worker_notifier.lock()
                        && let Some(cb) = &*guard {
                            cb();
                        }
            })
            .ok();

        Self {
            queue,
            rx,
            notifier: notifier_slot,
            handle,
        }
    }

    #[cfg(test)]
    pub fn disconnected_for_test() -> Self {
        let (tx, rx) = mpsc::channel();
        drop(tx);
        Self {
            queue: Arc::new((Mutex::new(SearchQueueState::default()), Condvar::new())),
            rx,
            notifier: Arc::new(Mutex::new(None)),
            handle: None,
        }
    }

    pub fn is_alive(&self) -> bool {
        self.handle.as_ref().is_some_and(|h| !h.is_finished())
    }

    pub fn set_notifier(&self, notifier: Option<Arc<dyn Fn() + Send + Sync>>) {
        if let Ok(mut guard) = self.notifier.lock() {
            *guard = notifier;
        }
    }

    pub fn schedule(&self, req: WorkspaceSearchRequest) -> bool {
        if !self.is_alive() {
            return false;
        }
        let (lock, cv) = &*self.queue;
        let Ok(mut guard) = lock.lock() else {
            return false;
        };
        if let Some(prev_pending) = guard.pending.take() {
            prev_pending.cancel.store(true, Ordering::Release);
        }
        if let Some(active) = &guard.active_cancel {
            active.store(true, Ordering::Release);
        }
        guard.pending = Some(req);
        cv.notify_one();
        true
    }

    pub fn drain_responses(&self) -> SearchDrainOutcome {
        let mut responses = Vec::new();
        let mut disconnected = false;
        loop {
            match self.rx.try_recv() {
                Ok(resp) => responses.push(resp),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }
        SearchDrainOutcome {
            responses,
            disconnected,
        }
    }
}

impl Drop for WorkspaceSearchWorker {
    fn drop(&mut self) {
        let (lock, cv) = &*self.queue;
        if let Ok(mut guard) = lock.lock() {
            guard.shutdown = true;
            if let Some(pending) = guard.pending.take() {
                pending.cancel.store(true, Ordering::Release);
            }
            if let Some(active) = &guard.active_cancel {
                active.store(true, Ordering::Release);
            }
            cv.notify_all();
        }
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn collect_files(node: &FileTreeNode, root: &Path, out: &mut Vec<(PathBuf, String)>) {
    if node.is_dir {
        for child in &node.children {
            collect_files(child, root, out);
        }
    } else {
        let rel = node
            .path
            .strip_prefix(root)
            .unwrap_or(&node.path)
            .to_string_lossy()
            .replace('\\', "/");
        out.push((node.path.clone(), rel));
    }
}

fn fuzzy_score(haystack: &str, needle: &str) -> Option<i64> {
    if let Some(pos) = haystack.find(needle) {
        // Exact substring match bonus; prefer basename/filename matches over deep parent dirs
        let after_slash = haystack.rfind('/').map(|s| pos > s).unwrap_or(true);
        return Some(1000 + if after_slash { 200 } else { 0 } - pos as i64);
    }
    let mut score = 0i64;
    let mut h_chars = haystack.char_indices().peekable();
    let mut prev_match_idx: Option<usize> = None;

    for nc in needle.chars() {
        let mut found = None;
        for (idx, hc) in h_chars.by_ref() {
            if hc == nc {
                found = Some(idx);
                break;
            }
        }
        let idx = found?;
        if let Some(prev) = prev_match_idx {
            if idx == prev + 1 {
                score += 15;
            } else {
                score += 5;
            }
        } else {
            score += 10;
        }
        if idx == 0 || haystack.as_bytes().get(idx - 1) == Some(&b'/') {
            score += 25;
        }
        prev_match_idx = Some(idx);
    }
    Some(score)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_dir(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("vellmark-search-{tag}-{nanos}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_prd_f06_and_f07_quick_open_and_global_search() {
        let root = unique_temp_dir("ws");
        fs::create_dir_all(root.join("docs")).unwrap();
        fs::write(
            root.join("docs/architecture.md"),
            "# Architecture\nVellmark GPU renderer\n",
        )
        .unwrap();
        fs::write(root.join("README.md"), "# Vellmark\nHello world\n").unwrap();

        let tree = FileTreeNode::scan_dir(&root).unwrap();
        let quick = quick_open_fuzzy_search(&tree, &root, "arch");
        assert_eq!(quick.len(), 1);
        assert_eq!(quick[0].relative_path, "docs/architecture.md");

        let hits = global_workspace_search(&root, "GPU renderer", false).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].line_number, 2);
        assert_eq!(hits[0].line_preview, "Vellmark GPU renderer");

        // Verify case-insensitive ASCII search reports exact match offset and obeys cancel/max_results
        let hits_ci = global_workspace_search(&root, "gpu", false).unwrap();
        assert_eq!(hits_ci.len(), 1);
        assert_eq!(
            hits_ci[0].line_byte_offset,
            "# Architecture\nVellmark ".len()
        );

        let cancelled = AtomicBool::new(true);
        let canceled_hits =
            global_workspace_search_with_cancel(&root, "Vellmark", false, &cancelled, 100)
                .unwrap();
        assert!(canceled_hits.is_empty());

        let capped = global_workspace_search_with_cancel(
            &root,
            "Vellmark",
            false,
            &AtomicBool::new(false),
            1,
        )
        .unwrap();
        assert_eq!(capped.len(), 1);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn test_worker_drop_cancels_in_flight_request_and_disconnected_reports_false() {
        let root = unique_temp_dir("worker-drop-cancel");
        for i in 0..8 {
            fs::write(
                root.join(format!("f{i}.md")),
                "alpha\nbeta\ngamma\n".repeat(200),
            )
            .unwrap();
        }

        // 1. Disconnected worker refuses schedule() and reports disconnected in drain_responses()
        let dead = WorkspaceSearchWorker::disconnected_for_test();
        assert!(!dead.is_alive());
        let cancel_dead = Arc::new(AtomicBool::new(false));
        assert!(!dead.schedule(WorkspaceSearchRequest {
            req_id: 1,
            root: root.clone(),
            query: "alpha".to_string(),
            case_sensitive: false,
            cancel: Arc::clone(&cancel_dead),
        }));
        let drained = dead.drain_responses();
        assert!(drained.disconnected);
        assert!(drained.responses.is_empty());

        // 2. Dropping live worker sets active/pending request cancel flag before join returns
        let worker = WorkspaceSearchWorker::spawn(None, GLOBAL_SEARCH_MAX_RESULTS);
        assert!(worker.is_alive());
        let cancel_live = Arc::new(AtomicBool::new(false));
        assert!(worker.schedule(WorkspaceSearchRequest {
            req_id: 2,
            root: root.clone(),
            query: "beta".to_string(),
            case_sensitive: false,
            cancel: Arc::clone(&cancel_live),
        }));
        drop(worker);
        assert!(cancel_live.load(Ordering::Acquire));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn test_case_insensitive_unicode_folding_with_ascii_queries_and_lines() {
        let root = unique_temp_dir("unicode-folding");
        fs::write(
            root.join("kelvin.md"),
            "Temperature \u{212A}\n\u{0130}stanbul\n\u{03A9}mega\nTemp K here\n\u{212A} foo CaseMatch\n",
        )
        .unwrap();

        // Both lowercase "k" and uppercase "K" must match Kelvin sign U+212A ("K") and ASCII "K" at exact byte offsets
        let hits_k_lower = global_workspace_search(&root, "k", false).unwrap();
        assert_eq!(hits_k_lower.len(), 3);
        assert_eq!(hits_k_lower[0].line_number, 1);
        assert_eq!(hits_k_lower[0].line_byte_offset, "Temperature ".len());
        assert_eq!(hits_k_lower[1].line_number, 4);
        assert_eq!(hits_k_lower[2].line_number, 5);

        let hits_k_upper = global_workspace_search(&root, "K", false).unwrap();
        assert_eq!(hits_k_upper.len(), 3);
        assert_eq!(hits_k_upper[0].line_byte_offset, "Temperature ".len());

        // Reverse: Unicode Kelvin sign query matches both Kelvin sign and ASCII 'K'
        let hits_kelvin_query = global_workspace_search(&root, "\u{212A}", false).unwrap();
        assert_eq!(hits_kelvin_query.len(), 3);

        // Turkish dotted I ("İ" -> "i\u{307}") and Greek ("Ω" -> "ω") case folding + exact offsets after length-changing prefix
        let ist_hits = global_workspace_search(&root, "i", false).unwrap();
        assert_eq!(ist_hits.len(), 1);
        assert_eq!(ist_hits[0].line_byte_offset, "Temperature \u{212A}\n".len());
        let stanbul_hits = global_workspace_search(&root, "STANBUL", false).unwrap();
        assert_eq!(stanbul_hits.len(), 1);
        assert_eq!(
            stanbul_hits[0].line_byte_offset,
            "Temperature \u{212A}\n\u{0130}".len()
        );
        assert_eq!(global_workspace_search(&root, "ω", false).unwrap().len(), 1);

        // CaseMatch after U+212A (3 bytes -> 1 folded byte) maps back to exact original UTF-8 byte offset
        let cm_after_k = global_workspace_search(&root, "casematch", false).unwrap();
        assert_eq!(cm_after_k.len(), 1);
        assert_eq!(
            cm_after_k[0].line_byte_offset,
            "Temperature \u{212A}\n\u{0130}stanbul\n\u{03A9}mega\nTemp K here\n\u{212A} foo ".len()
        );

        // Pure ASCII, mixed CJK + ASCII, and CJK queries all return exact UTF-8 byte offsets
        fs::write(
            root.join("cjk_mixed.md"),
            "# 标题\n第二行 CaseMatch 测试\n第三行 中文内容\n",
        )
        .unwrap();
        let hits_mixed = global_workspace_search(&root, "casematch", false).unwrap();
        assert_eq!(hits_mixed.len(), 2);
        let cjk_mixed_hit = hits_mixed
            .iter()
            .find(|m| m.path.ends_with("cjk_mixed.md"))
            .unwrap();
        assert_eq!(cjk_mixed_hit.line_byte_offset, 19);

        let hits_zh_cn = global_workspace_search(&root, "中文", false).unwrap();
        assert_eq!(hits_zh_cn.len(), 1);
        assert_eq!(hits_zh_cn[0].line_byte_offset, 46);

        let hits_title = global_workspace_search(&root, "标题", false).unwrap();
        assert_eq!(hits_title.len(), 1);
        assert_eq!(hits_title[0].line_byte_offset, 2);

        // Greek Final_Sigma consistency: "ΟΣ" query ("ος") must match uppercase "ΟΣ",
        // prefix inside "ΚΟΣΜΟΣ", and lowercase medial/final sigma ("οσ" / "ος") identically.
        fs::write(
            root.join("greek.md"),
            "ΟΣ\nΚΟΣΜΟΣ\nοσ\nκόσμος\n",
        )
        .unwrap();
        let greek_hits_upper = global_workspace_search(&root, "ΟΣ", false).unwrap();
        assert_eq!(greek_hits_upper.len(), 4);
        assert_eq!(greek_hits_upper[0].line_byte_offset, 0);
        assert_eq!(greek_hits_upper[1].line_byte_offset, "ΟΣ\nΚ".len());
        assert_eq!(greek_hits_upper[2].line_byte_offset, "ΟΣ\nΚΟΣΜΟΣ\n".len());
        assert_eq!(
            greek_hits_upper[3].line_byte_offset,
            "ΟΣ\nΚΟΣΜΟΣ\nοσ\nκόσμ".len()
        );

        let final_sigma_only = global_workspace_search(&root, "Σ", false).unwrap();
        assert_eq!(final_sigma_only.len(), 4);

        // Case-sensitive search keeps exact codepoint distinction
        let hits_cs = global_workspace_search(&root, "k", true).unwrap();
        assert!(hits_cs.is_empty());

        let _ = fs::remove_dir_all(&root);
    }
}
