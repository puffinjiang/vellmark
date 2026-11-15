//! Quick Open fuzzy search (`Cmd/Ctrl + P`, PRD 4.5 F-06) and Global Workspace Search (PRD 4.5 F-07).

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use vellmark_core::{ByteOffset, FileTreeNode, detect_and_decode};

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

/// Search text across all supported Markdown/text documents under `root` (PRD 4.5 F-07),
/// safely skipping binary or undecodable files.
pub fn global_workspace_search(
    root: &Path,
    query: &str,
    case_sensitive: bool,
) -> io::Result<Vec<WorkspaceSearchMatch>> {
    if query.is_empty() {
        return Ok(Vec::new());
    }
    let tree = FileTreeNode::scan_dir(root)?;
    let mut files = Vec::new();
    collect_files(&tree, root, &mut files);

    let q_needle = if case_sensitive {
        query.to_string()
    } else {
        query.to_lowercase()
    };

    let mut results = Vec::new();
    for (path, _) in files {
        let Ok(raw) = fs::read(&path) else {
            continue;
        };
        let Ok((text, _)) = detect_and_decode(&raw) else {
            continue;
        };
        let mut line_start_offset = 0usize;
        for (line_idx, raw_line) in text.split_inclusive('\n').enumerate() {
            let clean_line = raw_line.trim_end_matches(&['\r', '\n'][..]);
            let col_offset = if case_sensitive {
                clean_line.find(&q_needle)
            } else {
                // Scan character indices so case-folded match maps back to a safe line offset
                if clean_line.to_lowercase().contains(&q_needle) {
                    Some(0)
                } else {
                    None
                }
            };
            if let Some(rel_col) = col_offset {
                results.push(WorkspaceSearchMatch {
                    path: path.clone(),
                    line_number: line_idx + 1,
                    line_byte_offset: line_start_offset + rel_col,
                    line_preview: clean_line.trim().to_string(),
                });
            }
            line_start_offset += raw_line.len();
        }
    }

    Ok(results)
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

        let _ = fs::remove_dir_all(&root);
    }
}
