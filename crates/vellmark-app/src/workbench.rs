//! Multi-tab application workbench coordinating file lifecycle, editor actions, outlines,
//! virtualized layout, status-bar stats, themes, clipboard conversion, image assets, and HTML export.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static NEXT_WORKBENCH_INSTANCE: AtomicU64 = AtomicU64::new(1);

fn next_recovery_session_tag() -> String {
    let seq = NEXT_WORKBENCH_INSTANCE.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("pid{}-{nanos}-wb{seq}", std::process::id())
}

use vellmark_core::{
    ByteOffset, ConflictActionOutcome, DocState, DocumentSession, DocumentStats, EditorAction,
    EditorContext, EditorViewState, FileTreeNode, RecentFilesList, SaveOutcome, Selection,
    SessionError,
};
use vellmark_export::{
    HtmlExportMode, export_markdown_to_html_document, html_to_markdown,
    spreadsheet_text_to_markdown_table,
};
use vellmark_parser::{
    CopyPayload, IncrementalDoc, OutlineItem, active_outline_index, copy_selection,
    extract_outline,
};
use vellmark_render::{
    DocumentLayout, LayoutBlock, stabilize_cursor_line_scroll, typewriter_scroll_y,
};
use vellmark_theme::ThemeSheet;

use crate::assets::import_image_to_assets;
use crate::config::AppConfig;
use crate::search::{QuickOpenMatch, WorkspaceSearchMatch, global_workspace_search, quick_open_fuzzy_search};

/// An open editor tab binding one `DocumentSession` (source of truth + conflict state)
/// with its `EditorViewState` (caret/selection, WYSIWYG/Source mode, IME state) and scroll position.
pub struct DocumentTab {
    pub id: u64,
    pub session: DocumentSession,
    pub view: EditorViewState,
    pub scroll_y: f32,
    ast_cache: std::cell::RefCell<Option<(u64, vellmark_parser::IncrementalDoc)>>,
    last_zero_width_find: Option<(
        u64,
        vellmark_core::ByteOffset,
        String,
        vellmark_core::FindOptions,
    )>,
}

impl DocumentTab {
    pub fn title(&self) -> String {
        self.session
            .path
            .as_ref()
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "Untitled".to_string())
    }

    pub fn is_read_only(&self) -> bool {
        self.session.is_read_only()
    }

    pub fn edit(&mut self, f: impl FnOnce(&mut EditorContext<'_>)) -> Result<bool, SessionError> {
        self.session.with_editor_view(&mut self.view, f)
    }

    pub fn dispatch(&mut self, action: EditorAction) -> Result<bool, SessionError> {
        // Pure view/navigation actions remain allowed even on ReadOnly documents (PRD 4.5.2 & 4.1.1 E)
        match action {
            EditorAction::ToggleSourceMode => {
                self.view.ime_preedit = None;
                self.view.expansion_frozen = false;
                self.view.mode = match self.view.mode {
                    vellmark_core::ViewMode::Wysiwyg => vellmark_core::ViewMode::Source,
                    vellmark_core::ViewMode::Source => vellmark_core::ViewMode::Wysiwyg,
                };
                Ok(false)
            }
            mutating_action => self
                .session
                .with_editor_view(&mut self.view, move |ctx| ctx.dispatch(mutating_action)),
        }
    }

    pub fn stats(&self) -> DocumentStats {
        DocumentStats::compute(&self.session.buffer, self.view.selection)
    }

    pub fn outline(&self) -> Vec<OutlineItem> {
        extract_outline(&self.session.buffer.to_string())
    }

    pub fn active_outline_index(&self) -> Option<usize> {
        let outline = self.outline();
        active_outline_index(&outline, self.view.selection.head)
    }

    /// Jump caret to an outline heading (PRD 4.5 F-04).
    pub fn jump_to_outline_item(&mut self, item: &OutlineItem) {
        self.view.ime_preedit = None;
        self.view.expansion_frozen = false;
        self.last_zero_width_find = None;
        let clamped = item
            .content_range
            .start
            .min(self.session.buffer.len_bytes());
        self.view.selection = Selection::caret(clamped);
    }

    pub fn copy_selection_payload(&self) -> CopyPayload {
        copy_selection(&self.session.buffer.to_string(), self.view.selection)
    }

    /// Paste rich clipboard content (`text/html`, Excel/CSV spreadsheet text, or plain text, PRD 4.6 X-06 & 4.3 T-04).
    pub fn paste_rich_clipboard(
        &mut self,
        html: Option<&str>,
        plain_text: &str,
    ) -> Result<bool, SessionError> {
        let converted = if let Some(h) = html.filter(|s| !s.trim().is_empty()) {
            html_to_markdown(h)
        } else if let Some(table_md) = spreadsheet_text_to_markdown_table(plain_text) {
            table_md
        } else {
            plain_text.to_string()
        };
        self.edit(|ctx| ctx.insert_text(&converted))
    }

    pub fn compute_layout(&self, base_line_height_px: f32) -> DocumentLayout {
        let rev = self.session.buffer.revision();
        let mut cache = self.ast_cache.borrow_mut();
        match cache.as_mut() {
            Some((cached_rev, inc)) if *cached_rev == rev => {
                DocumentLayout::compute(inc.blocks(), self.view.selection, base_line_height_px)
            }
            Some((cached_rev, inc)) => {
                let source = self.session.buffer.to_string();
                inc.sync_source(source);
                *cached_rev = rev;
                DocumentLayout::compute(inc.blocks(), self.view.selection, base_line_height_px)
            }
            None => {
                let source = self.session.buffer.to_string();
                let inc = IncrementalDoc::new(source);
                let layout =
                    DocumentLayout::compute(inc.blocks(), self.view.selection, base_line_height_px);
                *cache = Some((rev, inc));
                layout
            }
        }
    }

    /// Return the buffer revision currently cached in this tab's incremental AST, if populated.
    pub fn cached_ast_revision(&self) -> Option<u64> {
        self.ast_cache.borrow().as_ref().map(|(rev, _)| *rev)
    }

    /// Search for all matches in the tab buffer without modifying cursor state (PRD E-11).
    pub fn find_all(
        &self,
        query: &str,
        options: vellmark_core::FindOptions,
    ) -> Result<Vec<vellmark_core::FindMatch>, vellmark_core::FindError> {
        vellmark_core::find_all_matches(&self.session.buffer.to_string(), query, options)
    }

    /// Move selection to the next match at or after the current selection end (PRD E-11).
    pub fn find_next(
        &mut self,
        query: &str,
        wrap: bool,
        options: vellmark_core::FindOptions,
    ) -> Result<Option<vellmark_core::FindMatch>, vellmark_core::FindError> {
        let text = self.session.buffer.to_string();
        let rev = self.session.buffer.revision();
        let sel_range = self.view.selection.range();
        let mut from = sel_range.end;
        if sel_range.is_empty() {
            if let Some((saved_rev, saved_pos, saved_query, saved_opts)) =
                &self.last_zero_width_find
            {
                if *saved_rev == rev
                    && *saved_pos == from
                    && saved_query == query
                    && *saved_opts == options
                {
                    if let Some(ch) = text.get(from..).and_then(|s| s.chars().next()) {
                        from += ch.len_utf8();
                    } else {
                        from = text.len().saturating_add(1);
                    }
                }
            }
        }
        let found = vellmark_core::find_next_match(&text, query, from, wrap, options)?;
        if let Some(m) = &found {
            self.view.selection = vellmark_core::Selection::new(m.range.start, m.range.end);
            self.last_zero_width_find = if m.range.is_empty() {
                Some((rev, m.range.end, query.to_string(), options))
            } else {
                None
            };
        }
        Ok(found)
    }

    /// Move selection to the previous match before the current selection start (PRD E-11).
    pub fn find_prev(
        &mut self,
        query: &str,
        wrap: bool,
        options: vellmark_core::FindOptions,
    ) -> Result<Option<vellmark_core::FindMatch>, vellmark_core::FindError> {
        let rev = self.session.buffer.revision();
        let from = self.view.selection.range().start;
        let found = vellmark_core::find_prev_match(
            &self.session.buffer.to_string(),
            query,
            from,
            wrap,
            options,
        )?;
        if let Some(m) = &found {
            self.view.selection = vellmark_core::Selection::new(m.range.start, m.range.end);
            self.last_zero_width_find = if m.range.is_empty() {
                Some((rev, m.range.end, query.to_string(), options))
            } else {
                None
            };
        }
        Ok(found)
    }

    /// Replace the currently selected match (if it matches `query`) or the next match (PRD E-11).
    pub fn replace_current_or_next(
        &mut self,
        query: &str,
        replacement: &str,
        wrap: bool,
        options: vellmark_core::FindOptions,
    ) -> Result<Result<vellmark_core::FindReplaceOutcome, vellmark_core::FindError>, SessionError> {
        let candidate = {
            let text = self.session.buffer.to_string();
            let sel_range = self.view.selection.range();
            match vellmark_core::find_next_match(&text, query, sel_range.start, wrap, options) {
                Ok(m) => m,
                Err(err) => return Ok(Err(err)),
            }
        };
        let Some(matched) = candidate else {
            return Ok(Ok(vellmark_core::FindReplaceOutcome {
                replaced_count: 0,
                selection_after: self.view.selection,
            }));
        };
        let mut outcome = vellmark_core::FindReplaceOutcome {
            replaced_count: 0,
            selection_after: self.view.selection,
        };
        self.edit(|ctx| {
            let res = vellmark_core::replace_match_in_buffer(
                ctx.buffer,
                ctx.undo,
                *ctx.selection,
                &matched,
                replacement,
                options.use_regex,
            );
            *ctx.selection = res.selection_after;
            outcome = res;
        })?;
        Ok(Ok(outcome))
    }

    /// Replace every match in the document in a single undoable transaction (PRD E-11).
    pub fn replace_all(
        &mut self,
        query: &str,
        replacement: &str,
        options: vellmark_core::FindOptions,
    ) -> Result<Result<vellmark_core::FindReplaceOutcome, vellmark_core::FindError>, SessionError> {
        let preview = {
            let text = self.session.buffer.to_string();
            vellmark_core::find_all_matches(&text, query, options)
        };
        match preview {
            Err(err) => return Ok(Err(err)),
            Ok(matches) if matches.is_empty() => {
                return Ok(Ok(vellmark_core::FindReplaceOutcome {
                    replaced_count: 0,
                    selection_after: self.view.selection,
                }));
            }
            Ok(_) => {}
        }
        let mut outcome: Result<vellmark_core::FindReplaceOutcome, vellmark_core::FindError> =
            Ok(vellmark_core::FindReplaceOutcome {
                replaced_count: 0,
                selection_after: self.view.selection,
            });
        self.edit(|ctx| {
            match vellmark_core::replace_all_in_buffer(
                ctx.buffer,
                ctx.undo,
                *ctx.selection,
                query,
                replacement,
                options,
            ) {
                Ok(res) => {
                    *ctx.selection = res.selection_after;
                    outcome = Ok(res);
                }
                Err(err) => {
                    outcome = Err(err);
                }
            }
        })?;
        Ok(outcome)
    }

    pub fn visible_blocks(
        &self,
        base_line_height_px: f32,
        viewport_height: f32,
        overscan: f32,
    ) -> Vec<LayoutBlock> {
        let layout = self.compute_layout(base_line_height_px);
        layout
            .visible_blocks(self.scroll_y, viewport_height, overscan)
            .to_vec()
    }

    /// Update scroll position after block expansion/collapse or in Typewriter Mode (PRD 4.1.1 A & 4.7 V-02).
    pub fn update_scroll_for_cursor(
        &mut self,
        old_cursor_doc_y: f32,
        new_cursor_doc_y: f32,
        line_height_px: f32,
        viewport_height: f32,
        typewriter_mode: bool,
    ) {
        self.scroll_y = if typewriter_mode {
            typewriter_scroll_y(new_cursor_doc_y, line_height_px, viewport_height)
        } else {
            stabilize_cursor_line_scroll(self.scroll_y, old_cursor_doc_y, new_cursor_doc_y)
        };
    }

    pub fn save(&mut self) -> SaveOutcome {
        self.session.save()
    }

    pub fn save_as(&mut self, new_path: impl AsRef<Path>) -> SaveOutcome {
        self.session.save_as(new_path)
    }

    pub fn on_external_modified(&mut self, new_disk_text: &str) {
        self.session
            .on_external_modified_with_view(new_disk_text, &mut self.view);
    }

    pub fn resolve_conflict_use_theirs(&mut self, captured_rev: u64) -> ConflictActionOutcome {
        self.session
            .resolve_conflict_use_theirs_with_view(captured_rev, &mut self.view)
    }
}

/// Top-level Vellmark multi-tab application workbench (PRD 4.5, 4.7, 4.8).
pub struct AppWorkbench {
    pub config: AppConfig,
    pub workspace_root: Option<PathBuf>,
    pub file_tree: Option<FileTreeNode>,
    pub recent_files: RecentFilesList,
    pub recovery_root: Option<PathBuf>,
    pub tabs: Vec<DocumentTab>,
    pub active_tab_idx: Option<usize>,
    pub system_prefers_dark: bool,
    pub theme: ThemeSheet,
    custom_theme_css: Option<String>,
    next_tab_id: u64,
    recovery_session_tag: String,
}

impl AppWorkbench {
    pub fn new(config: AppConfig, recovery_root: Option<PathBuf>) -> Self {
        let theme = config.effective_builtin_theme(false).load();
        Self {
            config,
            workspace_root: None,
            file_tree: None,
            recent_files: RecentFilesList::default(),
            recovery_root,
            tabs: Vec::new(),
            active_tab_idx: None,
            system_prefers_dark: false,
            theme,
            custom_theme_css: None,
            next_tab_id: 1,
            recovery_session_tag: next_recovery_session_tag(),
        }
    }

    pub fn active_tab(&self) -> Option<&DocumentTab> {
        self.active_tab_idx.and_then(|i| self.tabs.get(i))
    }

    pub fn active_tab_mut(&mut self) -> Option<&mut DocumentTab> {
        self.active_tab_idx.and_then(|i| self.tabs.get_mut(i))
    }

    pub fn new_document(&mut self) -> u64 {
        let id = self.next_tab_id;
        self.next_tab_id += 1;
        let rec_dir = self.recovery_dir_for_tab(id);
        let session = DocumentSession::new_untitled(rec_dir);
        self.tabs.push(DocumentTab {
            id,
            session,
            view: EditorViewState::default(),
            scroll_y: 0.0,
            ast_cache: std::cell::RefCell::new(None),
            last_zero_width_find: None,
        });
        self.active_tab_idx = Some(self.tabs.len() - 1);
        id
    }

    /// Open a document file, activating an existing tab if that canonical/matching path is already open (PRD F-09).
    pub fn open_document(&mut self, path: impl AsRef<Path>) -> Result<u64, SessionError> {
        let target = path.as_ref();
        let canonical = fs::canonicalize(target).unwrap_or_else(|_| target.to_path_buf());
        if let Some((idx, tab)) = self.tabs.iter().enumerate().find(|(_, t)| {
            t.session
                .path
                .as_ref()
                .map(|p| fs::canonicalize(p).unwrap_or_else(|_| p.clone()) == canonical)
                .unwrap_or(false)
        }) {
            let id = tab.id;
            self.active_tab_idx = Some(idx);
            self.recent_files.record_open(canonical);
            return Ok(id);
        }

        let id = self.next_tab_id;
        self.next_tab_id += 1;
        let rec_dir = self.recovery_dir_for_tab(id);
        let session = DocumentSession::open_file(target, rec_dir)?;
        self.recent_files.record_open(canonical);
        self.tabs.push(DocumentTab {
            id,
            session,
            view: EditorViewState::default(),
            scroll_y: 0.0,
            ast_cache: std::cell::RefCell::new(None),
            last_zero_width_find: None,
        });
        self.active_tab_idx = Some(self.tabs.len() - 1);
        Ok(id)
    }

    /// Close a tab cleanly. Refuses to silently discard Dirty/Conflict tabs unless `force` is true (PRD 4.5.1).
    pub fn close_tab(&mut self, tab_id: u64, force: bool) -> bool {
        let Some(pos) = self.tabs.iter().position(|t| t.id == tab_id) else {
            return false;
        };
        if !force && self.tabs[pos].session.state != DocState::Clean {
            return false;
        }
        self.tabs.remove(pos);
        self.active_tab_idx = if self.tabs.is_empty() {
            None
        } else {
            Some(pos.min(self.tabs.len() - 1))
        };
        true
    }

    pub fn open_folder(&mut self, root: impl AsRef<Path>) -> io::Result<()> {
        let root_path = root.as_ref().to_path_buf();
        let tree = FileTreeNode::scan_dir(&root_path)?;
        self.recent_files.record_open(&root_path);
        self.workspace_root = Some(root_path);
        self.file_tree = Some(tree);
        Ok(())
    }

    pub fn refresh_file_tree(&mut self) -> io::Result<()> {
        if let Some(root) = &self.workspace_root {
            self.file_tree = Some(FileTreeNode::scan_dir(root)?);
        }
        Ok(())
    }

    pub fn quick_open(&self, query: &str) -> Vec<QuickOpenMatch> {
        match (&self.file_tree, &self.workspace_root) {
            (Some(tree), Some(root)) => quick_open_fuzzy_search(tree, root, query),
            _ => Vec::new(),
        }
    }

    pub fn search_in_workspace(
        &self,
        query: &str,
        case_sensitive: bool,
    ) -> io::Result<Vec<WorkspaceSearchMatch>> {
        match &self.workspace_root {
            Some(root) => global_workspace_search(root, query, case_sensitive),
            None => Ok(Vec::new()),
        }
    }

    /// Paste/drop an image file into the active document's configured assets folder and insert Markdown syntax (PRD 4.4 I-02).
    pub fn paste_image_into_active(
        &mut self,
        source_image: &Path,
        alt_text: Option<&str>,
    ) -> Result<PathBuf, String> {
        let doc_path = self
            .active_tab()
            .and_then(|t| t.session.path.as_deref())
            .map(|p| p.to_path_buf());
        let fallback = self.workspace_root.clone();
        let rel_dir = self.config.image_assets_dir.clone();

        let (dest_path, md_snippet) = import_image_to_assets(
            doc_path.as_deref(),
            fallback.as_deref(),
            source_image,
            &rel_dir,
            alt_text,
        )
        .map_err(|e| e.to_string())?;

        let tab = self
            .active_tab_mut()
            .ok_or_else(|| "no active tab".to_string())?;
        tab.edit(|ctx| ctx.insert_text(&md_snippet))
            .map_err(|e| e.to_string())?;
        Ok(dest_path)
    }

    pub fn set_system_color_scheme(&mut self, prefers_dark: bool) {
        self.system_prefers_dark = prefers_dark;
        if self.custom_theme_css.is_none() {
            self.theme = self.config.effective_builtin_theme(prefers_dark).load();
        }
    }

    pub fn load_custom_theme_css(&mut self, css: &str) {
        self.custom_theme_css = Some(css.to_string());
        self.theme = ThemeSheet::parse_css(css);
    }

    /// Export active document to standalone HTML using the active theme CSS (PRD 4.6 X-01 & 4.8 S-04).
    pub fn export_active_html(&self, mode: HtmlExportMode) -> Option<String> {
        let tab = self.active_tab()?;
        let css = self.active_css_source();
        let md = tab.session.buffer.to_string();
        Some(export_markdown_to_html_document(
            &tab.title(),
            &md,
            Some(css),
            mode,
        ))
    }

    fn active_css_source(&self) -> &str {
        if let Some(custom) = &self.custom_theme_css {
            custom.as_str()
        } else {
            self.config
                .effective_builtin_theme(self.system_prefers_dark)
                .css_source()
        }
    }

    /// Scan `recovery_root` for unrecovered crash snapshots left behind by previous sessions,
    /// skipping any recovery directory owned by an active tab in this workbench (PRD 4.5.1 F).
    pub fn discover_pending_recoveries(&self) -> Vec<vellmark_core::PendingRecoveryEntry> {
        let Some(root) = &self.recovery_root else {
            return Vec::new();
        };
        vellmark_core::scan_recovery_root(root)
            .into_iter()
            .filter(|entry| {
                !self
                    .tabs
                    .iter()
                    .any(|tab| tab.session.recovery_dir.as_ref() == Some(&entry.recovery_dir))
            })
            .collect()
    }

    fn recovery_dir_for_tab(&self, tab_id: u64) -> Option<PathBuf> {
        self.recovery_root
            .as_ref()
            .map(|r| r.join(format!("{}-tab-{tab_id}", self.recovery_session_tag)))
    }
}

/// Expose caret offset convenience check for integration tests.
pub fn tab_caret_offset(tab: &DocumentTab) -> ByteOffset {
    tab.view.selection.head
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_temp_dir(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("vellmark-wb-{tag}-{nanos}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_end_to_end_workbench_workflow() {
        let root = unique_temp_dir("e2e");
        let rec_dir = root.join(".rec");
        let note_path = root.join("chapter.md");
        fs::write(&note_path, "# Intro\nHello ").unwrap();

        let img_src = root.join("shot.png");
        fs::write(&img_src, b"\x89PNG\r\n\x1a\ndata").unwrap();

        let mut wb = AppWorkbench::new(AppConfig::default(), Some(rec_dir));
        wb.open_folder(&root).unwrap();
        let tab_id = wb.open_document(&note_path).unwrap();

        // Opening the same file again reuses the existing tab instead of duplicating (F-09)
        assert_eq!(wb.open_document(&note_path).unwrap(), tab_id);
        assert_eq!(wb.tabs.len(), 1);

        // Unsaved dirty tab refuses normal close without force, preserving unsaved edits
        {
            let tab = wb.active_tab_mut().unwrap();
            tab.view.selection = Selection::caret(tab.session.buffer.len_bytes());
            tab.paste_rich_clipboard(
                Some("<p><strong>Vellmark</strong> &amp; Rust</p>"),
                "fallback",
            )
            .unwrap();
        }
        assert!(!wb.close_tab(tab_id, false));
        assert_eq!(
            wb.active_tab().unwrap().session.buffer.to_string(),
            "# Intro\nHello **Vellmark** & Rust"
        );

        // Paste image asset (I-02)
        let copied_img = wb.paste_image_into_active(&img_src, Some("Demo")).unwrap();
        assert!(copied_img.exists());
        assert!(
            wb.active_tab()
                .unwrap()
                .session
                .buffer
                .to_string()
                .contains("![Demo](./assets/shot.png)")
        );

        // Outline & stats & export HTML with system dark theme
        wb.set_system_color_scheme(true);
        let tab = wb.active_tab().unwrap();
        assert_eq!(tab.outline().len(), 1);
        assert!(tab.stats().word_count >= 4);
        let html = wb.export_active_html(HtmlExportMode::Safe).unwrap();
        assert!(html.contains("#0d1117")); // default-dark.css background applied (S-02 & S-04)
        assert!(html.contains("<strong>Vellmark</strong>"));

        // Save, then clean close succeeds
        assert_eq!(wb.active_tab_mut().unwrap().save(), SaveOutcome::SavedClean);
        assert!(wb.close_tab(tab_id, false));
        assert!(wb.tabs.is_empty());

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn test_p1_new_workbench_session_does_not_delete_unrecovered_tab1_snapshot() {
        let root = unique_temp_dir("rec-isolation");
        let rec_root = root.join("recovery");
        let other_path = root.join("other.md");
        fs::write(&other_path, "Original other").unwrap();

        // Session 1 creates an untitled document (tab_id = 1), edits it, and leaves unrecovered snapshot on disk.
        let session1_rec_dir;
        {
            let mut wb1 = AppWorkbench::new(AppConfig::default(), Some(rec_root.clone()));
            let t1 = wb1.new_document();
            assert_eq!(t1, 1);
            let tab = wb1.active_tab_mut().unwrap();
            tab.edit(|ctx| ctx.insert_text("Unsaved crash draft from session 1"))
                .unwrap();
            tab.session.flush_current_recovery_snapshot().unwrap();
            session1_rec_dir = tab.session.recovery_dir.clone().unwrap();
        }
        assert!(DocumentSession::read_latest_intact_snapshot(&session1_rec_dir).is_some());

        // Session 2 starts cleanly (next_tab_id resets to 1), opens an unrelated doc (also tab_id = 1),
        // edits it, and saves cleanly (clearing its own recovery snapshot).
        {
            let mut wb2 = AppWorkbench::new(AppConfig::default(), Some(rec_root.clone()));
            let t1_new = wb2.open_document(&other_path).unwrap();
            assert_eq!(t1_new, 1);
            let tab2 = wb2.active_tab_mut().unwrap();
            tab2.edit(|ctx| ctx.insert_char('!')).unwrap();
            tab2.session.flush_current_recovery_snapshot().unwrap();
            assert_eq!(tab2.save(), SaveOutcome::SavedClean);
        }

        // Session 1's unrecovered snapshot MUST remain intact on disk!
        let recovered = DocumentSession::read_latest_intact_snapshot(&session1_rec_dir)
            .expect("session 1 recovery snapshot must not be deleted by session 2");
        assert_eq!(recovered.content, "Unsaved crash draft from session 1");

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn test_document_tab_incremental_layout_cache_updates_with_edits() {
        let mut wb = AppWorkbench::new(AppConfig::default(), None);
        wb.new_document();
        let tab = wb.active_tab_mut().unwrap();
        assert_eq!(tab.cached_ast_revision(), None);

        tab.edit(|ctx| {
            ctx.insert_text("# Title\n\nFirst paragraph **bold**.\n\nSecond paragraph.\n")
        })
        .unwrap();
        let rev1 = tab.session.buffer.revision();
        let layout1 = tab.compute_layout(20.0);
        assert_eq!(tab.cached_ast_revision(), Some(rev1));
        assert_eq!(
            tab.ast_cache.borrow().as_ref().unwrap().1.blocks(),
            vellmark_parser::parse_blocks(&tab.session.buffer.to_string()).as_slice()
        );
        let fresh1 = DocumentLayout::compute(
            &vellmark_parser::parse_blocks(&tab.session.buffer.to_string()),
            tab.view.selection,
            20.0,
        );
        assert_eq!(layout1.blocks.len(), fresh1.blocks.len());
        assert_eq!(layout1.total_height, fresh1.total_height);

        // Moving caret without buffer edit keeps the same cached AST revision.
        tab.view.selection = vellmark_core::Selection::caret(2);
        let _ = tab.compute_layout(20.0);
        assert_eq!(tab.cached_ast_revision(), Some(rev1));

        // Buffer edit advances revision and stays identical to fresh full parse.
        tab.edit(|ctx| ctx.insert_text("## Subheading\n")).unwrap();
        let rev2 = tab.session.buffer.revision();
        assert!(rev2 > rev1);
        let layout2 = tab.compute_layout(20.0);
        assert_eq!(tab.cached_ast_revision(), Some(rev2));
        assert_eq!(
            tab.ast_cache.borrow().as_ref().unwrap().1.blocks(),
            vellmark_parser::parse_blocks(&tab.session.buffer.to_string()).as_slice()
        );
        let fresh2 = DocumentLayout::compute(
            &vellmark_parser::parse_blocks(&tab.session.buffer.to_string()),
            tab.view.selection,
            20.0,
        );
        assert_eq!(layout2.blocks.len(), fresh2.blocks.len());
        assert_eq!(layout2.total_height, fresh2.total_height);
    }

    #[test]
    fn test_document_tab_find_and_replace_integration() {
        let mut wb = AppWorkbench::new(AppConfig::default(), None);
        wb.new_document();
        let tab = wb.active_tab_mut().unwrap();
        tab.edit(|ctx| ctx.insert_text("TODO(a): fix\nkeep line\nTODO(b): polish\n"))
            .unwrap();
        tab.view.selection = vellmark_core::Selection::caret(0);

        let opts = vellmark_core::FindOptions {
            case_sensitive: true,
            whole_word: false,
            use_regex: true,
        };
        assert_eq!(tab.find_all(r"TODO\((\w+)\)", opts).unwrap().len(), 2);

        let m1 = tab.find_next(r"TODO\((\w+)\)", true, opts).unwrap().unwrap();
        assert_eq!(tab.view.selection, vellmark_core::Selection::new(m1.range.start, m1.range.end));

        let rep1 = tab
            .replace_current_or_next(r"TODO\((\w+)\)", "DONE[$1]", true, opts)
            .unwrap()
            .unwrap();
        assert_eq!(rep1.replaced_count, 1);
        assert_eq!(
            tab.session.buffer.to_string(),
            "DONE[a]: fix\nkeep line\nTODO(b): polish\n"
        );

        let prev = tab.find_prev(r"(DONE|TODO)", true, opts).unwrap().unwrap();
        assert_eq!(prev.range.start, 0);

        let rep_all = tab
            .replace_all(r"TODO\((\w+)\)", "DONE[$1]", opts)
            .unwrap()
            .unwrap();
        assert_eq!(rep_all.replaced_count, 1);
        assert_eq!(
            tab.session.buffer.to_string(),
            "DONE[a]: fix\nkeep line\nDONE[b]: polish\n"
        );

        // Single undo reverts replace_all cleanly
        tab.edit(|ctx| {
            ctx.undo();
        })
        .unwrap();
        assert_eq!(
            tab.session.buffer.to_string(),
            "DONE[a]: fix\nkeep line\nTODO(b): polish\n"
        );
    }

    #[test]
    fn test_app_workbench_startup_discovers_pending_crash_recoveries() {
        let root = unique_temp_dir("discover-rec");
        let rec_root = root.join("recovery");
        let doc_path = root.join("notes.md");
        fs::write(&doc_path, "Original disk").unwrap();

        {
            let mut wb1 = AppWorkbench::new(AppConfig::default(), Some(rec_root.clone()));
            wb1.open_document(&doc_path).unwrap();
            let t1 = wb1.active_tab_mut().unwrap();
            t1.edit(|ctx| ctx.insert_text(" + crashed edit")).unwrap();
            t1.session.flush_current_recovery_snapshot().unwrap();

            wb1.new_document();
            let t2 = wb1.active_tab_mut().unwrap();
            t2.edit(|ctx| ctx.insert_text("Untitled crash buffer")).unwrap();
            t2.session.flush_current_recovery_snapshot().unwrap();
        }

        let mut wb2 = AppWorkbench::new(AppConfig::default(), Some(rec_root.clone()));
        // Active tab in wb2 also flushes a snapshot; discover_pending_recoveries excludes active tabs.
        wb2.new_document();
        let active_t = wb2.active_tab_mut().unwrap();
        active_t.edit(|ctx| ctx.insert_text("Live session draft")).unwrap();
        active_t.session.flush_current_recovery_snapshot().unwrap();

        let pending = wb2.discover_pending_recoveries();
        assert_eq!(pending.len(), 2);
        assert!(
            pending
                .iter()
                .any(|p| p.source_path.as_ref() == Some(&doc_path)
                    && p.report.content.contains("crashed edit"))
        );
        assert!(
            pending
                .iter()
                .any(|p| p.source_path.is_none() && p.report.content == "Untitled crash buffer")
        );

        drop(wb2);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn test_find_next_advances_across_zero_width_matches_and_utf8() {
        let mut wb = AppWorkbench::new(AppConfig::default(), None);
        wb.new_document();
        let tab = wb.active_tab_mut().unwrap();
        tab.edit(|ctx| ctx.insert_text("你好\n世界\n末行")).unwrap();
        tab.view.selection = vellmark_core::Selection::caret(0);

        let opts = vellmark_core::FindOptions {
            case_sensitive: true,
            whole_word: false,
            use_regex: true,
        };
        let l2 = "你好\n".len();
        let l3 = "你好\n世界\n".len();

        let m1 = tab.find_next("^", false, opts).unwrap().unwrap();
        assert_eq!(m1.range, vellmark_core::ByteRange::empty(0));
        let m2 = tab.find_next("^", false, opts).unwrap().unwrap();
        assert_eq!(m2.range, vellmark_core::ByteRange::empty(l2));
        let m3 = tab.find_next("^", false, opts).unwrap().unwrap();
        assert_eq!(m3.range, vellmark_core::ByteRange::empty(l3));

        // Without wrap, staying at the final match yields None repeatedly.
        assert!(tab.find_next("^", false, opts).unwrap().is_none());
        assert!(tab.find_next("^", false, opts).unwrap().is_none());

        // With wrap=true, wraps cleanly back to 0..0 and continues forward.
        let wrapped = tab.find_next("^", true, opts).unwrap().unwrap();
        assert_eq!(wrapped.range, vellmark_core::ByteRange::empty(0));
        let m2_again = tab.find_next("^", true, opts).unwrap().unwrap();
        assert_eq!(m2_again.range, vellmark_core::ByteRange::empty(l2));
    }
}
