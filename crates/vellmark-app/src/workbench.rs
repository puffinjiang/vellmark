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
    CopyPayload, OutlineItem, active_outline_index, copy_selection, extract_outline, parse_blocks,
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
        let source = self.session.buffer.to_string();
        let blocks = parse_blocks(&source);
        DocumentLayout::compute(&blocks, self.view.selection, base_line_height_px)
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
}
