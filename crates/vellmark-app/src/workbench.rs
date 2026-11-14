//! Multi-tab application workbench coordinating file lifecycle, editor actions, outlines,
//! virtualized layout, status-bar stats, themes, clipboard conversion, image assets, and HTML export.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

static NEXT_WORKBENCH_INSTANCE: AtomicU64 = AtomicU64::new(1);

fn next_recovery_session_tag() -> String {
    let seq = NEXT_WORKBENCH_INSTANCE.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("pid{}-{nanos}-wb{seq}", std::process::id())
}

fn normalize_line_endings(text: &str, target: LineEnding) -> String {
    if !text.contains('\n') && !text.contains('\r') {
        return text.to_string();
    }
    let target_str = target.as_str();
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '\r' => {
                if chars.peek() == Some(&'\n') {
                    chars.next();
                }
                out.push_str(target_str);
            }
            '\n' => out.push_str(target_str),
            other => out.push(other),
        }
    }
    out
}

use vellmark_core::{
    AutoMergeOutcome, ByteOffset, ConflictActionOutcome, DEFAULT_ORPHAN_GRACE_SECS, DocState,
    DocumentSession, DocumentStats, EditorAction, EditorContext, EditorViewState,
    ExternalSyncResult, FileTreeNode, HISTORICAL_BACKUPS_DIR_NAME, HistoricalBackupEntry,
    HistoricalBackupPolicy, HistoricalBackupReason, LineEnding, MergeConflictHunk,
    PendingRecoveryEntry, RecentFilesList, RecoveryMaintenanceReport, SaveOutcome, Selection,
    SessionError, archive_pending_recovery_to_history, clear_conflict_backups_for_path,
    clear_historical_backups_for_path, delete_historical_backup, discard_pending_recovery_entry,
    list_historical_backups, list_historical_backups_for_path, maintain_recovery_root,
    paths_refer_to_same_file,
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

pub const AUTO_SAVE_RETRY_BACKOFF_SECS: [u64; 3] = [5, 15, 60];

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
    auto_save_tracked_rev: u64,
    last_edit_at: Option<Instant>,
    auto_save_next_retry_at: Option<Instant>,
    auto_save_retry_attempt: usize,
    pub last_write_error: Option<String>,
    pub conflict_theirs_updated: bool,
    pub conflict_backup_count: usize,
    pub conflict_backup_notice_dismissed: bool,
    pub last_merge_conflicts: Vec<MergeConflictHunk>,
    last_external_read_error: Option<String>,
}

impl DocumentTab {
    fn new(id: u64, session: DocumentSession, view: EditorViewState) -> Self {
        let rev = session.buffer.revision();
        let last_edit_at = (session.state == DocState::Dirty).then(Instant::now);
        Self {
            id,
            session,
            view,
            scroll_y: 0.0,
            ast_cache: std::cell::RefCell::new(None),
            last_zero_width_find: None,
            auto_save_tracked_rev: rev,
            last_edit_at,
            auto_save_next_retry_at: None,
            auto_save_retry_attempt: 0,
            last_write_error: None,
            conflict_theirs_updated: false,
            conflict_backup_count: 0,
            conflict_backup_notice_dismissed: false,
            last_merge_conflicts: Vec::new(),
            last_external_read_error: None,
        }
    }

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
        let changed = self.session.with_editor_view(&mut self.view, f)?;
        if changed {
            self.sync_edit_activity_at(Instant::now());
        }
        Ok(changed)
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
            mutating_action => {
                let changed = self
                    .session
                    .with_editor_view(&mut self.view, move |ctx| ctx.dispatch(mutating_action))?;
                if changed {
                    self.sync_edit_activity_at(Instant::now());
                }
                Ok(changed)
            }
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

    fn target_line_ending(&self) -> LineEnding {
        LineEnding::detect_dominant_or(
            &self.session.buffer.to_string(),
            self.session.format.dominant_line_ending,
        )
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
        let normalized = normalize_line_endings(&converted, self.target_line_ending());
        self.edit(|ctx| ctx.insert_text(&normalized))
    }

    /// Sync and borrow cached incremental AST blocks without allocating on unchanged frames (10 MB fast path).
    pub fn with_incremental_blocks<R>(&self, f: impl FnOnce(&[vellmark_parser::BlockNode]) -> R) -> R {
        let rev = self.session.buffer.revision();
        let mut cache = self.ast_cache.borrow_mut();
        match cache.as_mut() {
            Some((cached_rev, inc)) if *cached_rev == rev => f(inc.blocks()),
            Some((cached_rev, inc)) => {
                let source = self.session.buffer.to_string();
                inc.sync_source(source);
                *cached_rev = rev;
                f(inc.blocks())
            }
            None => {
                let source = self.session.buffer.to_string();
                let inc = IncrementalDoc::new(source);
                *cache = Some((rev, inc));
                let inc_ref = &cache.as_ref().expect("just inserted").1;
                f(inc_ref.blocks())
            }
        }
    }

    pub fn compute_layout(&self, base_line_height_px: f32) -> DocumentLayout {
        self.with_incremental_blocks(|blocks| {
            DocumentLayout::compute(blocks, self.view.selection, base_line_height_px)
        })
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
        let outcome = self.session.save();
        self.apply_save_outcome_at(&outcome, Instant::now());
        outcome
    }

    pub fn save_as(&mut self, new_path: impl AsRef<Path>) -> SaveOutcome {
        let outcome = self.session.save_as(new_path);
        self.apply_save_outcome_at(&outcome, Instant::now());
        outcome
    }

    pub fn on_external_modified(&mut self, new_disk_text: &str) {
        self.session
            .on_external_modified_with_view(new_disk_text, &mut self.view);
    }

    pub fn resolve_conflict_keep_mine(&mut self) -> ConflictActionOutcome {
        let rev = self.session.buffer.revision();
        let rope = self.session.buffer.snapshot_rope();
        let outcome = self.session.resolve_conflict_keep_mine(rev, rope);
        match outcome {
            ConflictActionOutcome::ResolvedClean | ConflictActionOutcome::ResolvedDirty => {
                self.last_write_error = None;
                self.auto_save_next_retry_at = None;
                self.auto_save_retry_attempt = 0;
                self.conflict_theirs_updated = false;
                self.last_merge_conflicts.clear();
                self.conflict_backup_notice_dismissed = false;
                self.sync_edit_activity_at(Instant::now());
            }
            ConflictActionOutcome::TransitionedToMissing => {
                self.last_write_error = None;
                self.conflict_theirs_updated = false;
                self.last_merge_conflicts.clear();
            }
            ConflictActionOutcome::AbortedStillConflict(_) => {}
        }
        outcome
    }

    pub fn resolve_conflict_use_theirs(&mut self, captured_rev: u64) -> ConflictActionOutcome {
        let outcome = self
            .session
            .resolve_conflict_use_theirs_with_view(captured_rev, &mut self.view);
        match outcome {
            ConflictActionOutcome::ResolvedClean | ConflictActionOutcome::ResolvedDirty => {
                self.last_write_error = None;
                self.auto_save_next_retry_at = None;
                self.auto_save_retry_attempt = 0;
                self.conflict_theirs_updated = false;
                self.last_merge_conflicts.clear();
                self.conflict_backup_notice_dismissed = false;
                self.sync_edit_activity_at(Instant::now());
            }
            ConflictActionOutcome::TransitionedToMissing => {
                self.last_write_error = None;
                self.conflict_theirs_updated = false;
                self.last_merge_conflicts.clear();
            }
            ConflictActionOutcome::AbortedStillConflict(_) => {}
        }
        outcome
    }

    pub fn resolve_conflict_auto_merge(&mut self, captured_rev: u64) -> AutoMergeOutcome {
        let outcome = self
            .session
            .resolve_conflict_auto_merge_with_view(captured_rev, &mut self.view);
        match &outcome {
            AutoMergeOutcome::MergedClean | AutoMergeOutcome::MergedDirty { .. } => {
                self.last_write_error = None;
                self.auto_save_next_retry_at = None;
                self.auto_save_retry_attempt = 0;
                self.conflict_theirs_updated = false;
                self.last_merge_conflicts.clear();
                self.conflict_backup_notice_dismissed = false;
                self.sync_edit_activity_at(Instant::now());
            }
            AutoMergeOutcome::HasOverlappingConflicts { conflicting_hunks } => {
                self.last_merge_conflicts = conflicting_hunks.clone();
            }
            AutoMergeOutcome::TransitionedToMissing => {
                self.last_write_error = None;
                self.conflict_theirs_updated = false;
                self.last_merge_conflicts.clear();
            }
            AutoMergeOutcome::AbortedStillConflict(_) => {}
        }
        outcome
    }

    pub fn is_auto_save_eligible(&self) -> bool {
        self.session.path.is_some()
            && !self.session.is_read_only()
            && self.session.state == DocState::Dirty
    }

    pub fn sync_edit_activity_at(&mut self, now: Instant) {
        let rev = self.session.buffer.revision();
        if rev != self.auto_save_tracked_rev {
            self.auto_save_tracked_rev = rev;
            if self.session.state == DocState::Dirty {
                self.last_edit_at = Some(now);
                self.auto_save_next_retry_at = None;
                self.auto_save_retry_attempt = 0;
            }
        }
        if self.session.state != DocState::Dirty {
            self.last_edit_at = None;
            self.auto_save_next_retry_at = None;
            self.auto_save_retry_attempt = 0;
            if self.session.state == DocState::Clean {
                self.last_write_error = None;
            }
        }
        if self.session.state != DocState::Conflict {
            self.conflict_theirs_updated = false;
            self.last_merge_conflicts.clear();
        }
    }

    pub fn next_auto_save_deadline(
        &self,
        auto_save_enabled: bool,
        idle_delay: Duration,
    ) -> Option<Instant> {
        if !auto_save_enabled
            || !self.is_auto_save_eligible()
            || self.session.is_async_save_in_flight()
        {
            return None;
        }
        if let Some(retry_at) = self.auto_save_next_retry_at {
            return Some(retry_at);
        }
        self.last_edit_at.map(|t| t + idle_delay)
    }

    pub(crate) fn apply_save_outcome_at(&mut self, outcome: &SaveOutcome, now: Instant) {
        match outcome {
            SaveOutcome::SavedClean => {
                self.last_write_error = None;
                self.auto_save_next_retry_at = None;
                self.auto_save_retry_attempt = 0;
                self.last_edit_at = None;
                self.conflict_theirs_updated = false;
                self.last_merge_conflicts.clear();
                self.auto_save_tracked_rev = self.session.buffer.revision();
            }
            SaveOutcome::SavedButStillDirty { .. } => {
                self.last_write_error = None;
                self.auto_save_next_retry_at = None;
                self.auto_save_retry_attempt = 0;
                self.last_edit_at = Some(now);
                self.auto_save_tracked_rev = self.session.buffer.revision();
            }
            SaveOutcome::EnteredConflict => {
                self.last_write_error = None;
                self.auto_save_next_retry_at = None;
                self.auto_save_retry_attempt = 0;
            }
            SaveOutcome::WriteFailed(err) => {
                if matches!(self.session.state, DocState::Missing { .. })
                    && err == "target file disappeared before write"
                {
                    self.last_write_error = None;
                    self.auto_save_next_retry_at = None;
                    self.auto_save_retry_attempt = 0;
                } else {
                    self.last_write_error = Some(err.clone());
                    if self.session.state == DocState::Dirty {
                        let idx = self
                            .auto_save_retry_attempt
                            .min(AUTO_SAVE_RETRY_BACKOFF_SECS.len() - 1);
                        let secs = AUTO_SAVE_RETRY_BACKOFF_SECS[idx];
                        self.auto_save_next_retry_at = Some(now + Duration::from_secs(secs));
                        self.auto_save_retry_attempt =
                            (self.auto_save_retry_attempt + 1).min(AUTO_SAVE_RETRY_BACKOFF_SECS.len());
                    } else {
                        self.auto_save_next_retry_at = None;
                        self.auto_save_retry_attempt = 0;
                    }
                }
            }
        }
    }

    pub fn sync_from_disk(&mut self, bypass_stat_cache: bool) -> ExternalSyncResult {
        let res = self
            .session
            .sync_from_disk_with_view(&mut self.view, bypass_stat_cache);
        match &res {
            ExternalSyncResult::NoChange => {}
            ExternalSyncResult::ReloadedClean
            | ExternalSyncResult::ReappearedClean { .. } => {
                self.last_write_error = None;
                self.auto_save_next_retry_at = None;
                self.auto_save_retry_attempt = 0;
                self.last_edit_at = None;
                self.conflict_theirs_updated = false;
                self.last_merge_conflicts.clear();
                self.last_external_read_error = None;
                self.auto_save_tracked_rev = self.session.buffer.revision();
            }
            ExternalSyncResult::EnteredConflict => {
                self.last_write_error = None;
                self.auto_save_next_retry_at = None;
                self.auto_save_retry_attempt = 0;
                self.conflict_theirs_updated = false;
                self.last_merge_conflicts.clear();
                self.last_external_read_error = None;
            }
            ExternalSyncResult::ConflictTheirsUpdated => {
                self.conflict_theirs_updated = true;
                self.last_merge_conflicts.clear();
                self.last_external_read_error = None;
            }
            ExternalSyncResult::BecameMissing => {
                self.last_write_error = None;
                self.auto_save_next_retry_at = None;
                self.auto_save_retry_attempt = 0;
                self.conflict_theirs_updated = false;
                self.last_merge_conflicts.clear();
                self.last_external_read_error = None;
            }
            ExternalSyncResult::ReappearedDirty => {
                self.last_write_error = None;
                self.last_edit_at = Some(Instant::now());
                self.auto_save_next_retry_at = None;
                self.auto_save_retry_attempt = 0;
                self.conflict_theirs_updated = false;
                self.last_merge_conflicts.clear();
                self.last_external_read_error = None;
            }
            ExternalSyncResult::Renamed { .. } => {
                self.last_external_read_error = None;
            }
            ExternalSyncResult::ReadError(_) => {}
        }
        res
    }
}

/// High-level runtime event produced by auto-save timers or external file polling (PRD 4.5.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkbenchRuntimeEvent {
    AutoSavedClean { tab_id: u64, title: String },
    EnteredConflict { tab_id: u64, title: String },
    ConflictTheirsUpdated { tab_id: u64, title: String },
    ReloadedClean { tab_id: u64, title: String },
    BecameMissing { tab_id: u64, title: String, has_local_edits: bool },
    Reappeared { tab_id: u64, title: String, reloaded: bool },
    Renamed { tab_id: u64, old_path: PathBuf, new_path: PathBuf },
    WriteFailed { tab_id: u64, title: String, error: String },
    ReadError { tab_id: u64, title: String, error: String },
}

const DEFAULT_MAINTENANCE_INTERVAL_SECS: u64 = 60;

struct MaintenanceJob {
    rx: mpsc::Receiver<(u64, RecoveryMaintenanceReport)>,
    handle: Option<thread::JoinHandle<()>>,
}

/// Top-level Vellmark multi-tab application workbench (PRD 4.5, 4.7, 4.8).
pub struct AppWorkbench {
    pub config: AppConfig,
    pub workspace_root: Option<PathBuf>,
    pub file_tree: Option<FileTreeNode>,
    pub recent_files: RecentFilesList,
    pub recovery_root: Option<PathBuf>,
    pub history_policy: HistoricalBackupPolicy,
    pub last_recovery_maintenance: Option<RecoveryMaintenanceReport>,
    pub tabs: Vec<DocumentTab>,
    pub active_tab_idx: Option<usize>,
    pub system_prefers_dark: bool,
    pub theme: ThemeSheet,
    custom_theme_css: Option<String>,
    next_tab_id: u64,
    recovery_session_tag: String,
    last_maintenance_unix_secs: Option<u64>,
    pending_maintenance: Option<MaintenanceJob>,
}

impl Drop for AppWorkbench {
    fn drop(&mut self) {
        self.finish_pending_recovery_maintenance();
    }
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
            history_policy: HistoricalBackupPolicy::default(),
            last_recovery_maintenance: None,
            tabs: Vec::new(),
            active_tab_idx: None,
            system_prefers_dark: false,
            theme,
            custom_theme_css: None,
            next_tab_id: 1,
            recovery_session_tag: next_recovery_session_tag(),
            last_maintenance_unix_secs: None,
            pending_maintenance: None,
        }
    }

    pub fn history_root(&self) -> Option<PathBuf> {
        self.recovery_root
            .as_ref()
            .map(|r| r.join(HISTORICAL_BACKUPS_DIR_NAME))
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
        let hist_root = self.history_root();
        let session = DocumentSession::new_untitled_with_recovery(
            rec_dir,
            hist_root,
            self.history_policy,
            std::time::Duration::from_millis(200),
        );
        self.tabs
            .push(DocumentTab::new(id, session, EditorViewState::default()));
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
                .as_deref()
                .is_some_and(|p| paths_refer_to_same_file(p, target))
        }) {
            let id = tab.id;
            self.active_tab_idx = Some(idx);
            self.recent_files.record_open(canonical);
            return Ok(id);
        }

        let id = self.next_tab_id;
        self.next_tab_id += 1;
        let rec_dir = self.recovery_dir_for_tab(id);
        let hist_root = self.history_root();
        let session = DocumentSession::open_file_with_recovery(
            target,
            rec_dir,
            hist_root,
            self.history_policy,
            std::time::Duration::from_millis(200),
        )?;
        self.recent_files.record_open(canonical);
        self.tabs
            .push(DocumentTab::new(id, session, EditorViewState::default()));
        self.refresh_tab_conflict_backup_count(id);
        self.active_tab_idx = Some(self.tabs.len() - 1);
        Ok(id)
    }

    /// Return `(tab_idx, tab_id, title)` if any open tab other than `exclude_tab_id` is already
    /// bound to the same canonical/equivalent file path (PRD 4.5.1 E single-document invariant).
    pub fn find_other_tab_with_path(
        &self,
        exclude_tab_id: u64,
        target: &Path,
    ) -> Option<(usize, u64, String)> {
        self.tabs
            .iter()
            .enumerate()
            .find(|(_, tab)| {
                tab.id != exclude_tab_id
                    && tab
                        .session
                        .path
                        .as_deref()
                        .is_some_and(|p| paths_refer_to_same_file(p, target))
            })
            .map(|(idx, tab)| (idx, tab.id, tab.title()))
    }

    /// Save tab `tab_id` to `new_path`, refusing to bind a second independent tab to a file path
    /// already open in another tab (PRD 4.5.1 E single-document invariant).
    pub fn save_tab_as(
        &mut self,
        tab_id: u64,
        new_path: impl AsRef<Path>,
    ) -> Result<SaveOutcome, String> {
        let dest = new_path.as_ref();
        if let Some((other_idx, _, other_title)) = self.find_other_tab_with_path(tab_id, dest) {
            return Err(format!(
                "'{other_title}' is already open in tab {} — close that tab first or pick another file",
                other_idx + 1
            ));
        }
        let Some(tab) = self.tabs.iter_mut().find(|t| t.id == tab_id) else {
            return Err("tab no longer exists".to_string());
        };
        Ok(tab.save_as(dest))
    }

    /// Close a tab, enforcing unsaved-change guards when `force == false` and archiving unsaved edits
    /// into bounded historical backups when `force == true` (refusing to close `Missing` if backup fails).
    pub fn try_close_tab(&mut self, tab_id: u64, force: bool) -> Result<bool, String> {
        let Some(pos) = self.tabs.iter().position(|t| t.id == tab_id) else {
            return Ok(false);
        };
        let state = self.tabs[pos].session.state;
        if !force && state != DocState::Clean {
            return Ok(false);
        }
        if state == DocState::Clean {
            self.tabs[pos].session.clear_current_recovery_snapshot();
        } else if let Err(err) = self.tabs[pos].session.archive_unsaved_on_close() {
            return Err(format!("recovery backup failed: {err}"));
        }
        self.tabs.remove(pos);
        self.active_tab_idx = if self.tabs.is_empty() {
            None
        } else {
            Some(pos.min(self.tabs.len() - 1))
        };
        self.maybe_maintain_recovery_storage_async();
        Ok(true)
    }

    /// Close a tab cleanly. Refuses to silently discard Dirty/Conflict tabs unless `force` is true (PRD 4.5.1).
    pub fn close_tab(&mut self, tab_id: u64, force: bool) -> bool {
        self.try_close_tab(tab_id, force).unwrap_or(false)
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

    /// Restore any unrecovered crash snapshots found under `recovery_root` into open workbench tabs,
    /// cleaning stale `Clean` entries and enforcing bounded history retention across restarts.
    /// Returns the number of restored tabs.
    pub fn restore_pending_recoveries(&mut self) -> usize {
        let entries = self.discover_pending_recoveries();
        let mut restored = 0usize;
        let hist_root = self.history_root();
        let policy = self.history_policy;

        for entry in entries {
            let meaningful = entry.report.state != DocState::Clean
                || entry.report.content != entry.report.base_content;
            if !meaningful {
                discard_pending_recovery_entry(&entry);
                continue;
            }
            let entry_canonical = entry
                .source_path
                .as_ref()
                .map(|p| fs::canonicalize(p).unwrap_or_else(|_| p.clone()));

            if let Some(target_canon) = &entry_canonical {
                if let Some((existing_idx, tab)) = self.tabs.iter_mut().enumerate().find(|(_, t)| {
                    t.session
                        .path
                        .as_ref()
                        .map(|p| fs::canonicalize(p).unwrap_or_else(|_| p.clone()) == *target_canon)
                        .unwrap_or(false)
                }) {
                    if tab.session.state == DocState::Clean {
                        let rec_dir = tab.session.recovery_dir.clone();
                        tab.session = DocumentSession::from_pending_recovery_with_recovery(
                            &entry,
                            rec_dir,
                            hist_root.clone(),
                            policy,
                        );
                        let caret = tab.session.buffer.len_bytes();
                        tab.view.selection = Selection::caret(caret);
                        *tab.ast_cache.borrow_mut() = None;
                        tab.sync_edit_activity_at(Instant::now());
                        let tab_id = tab.id;
                        self.refresh_tab_conflict_backup_count(tab_id);
                        self.active_tab_idx = Some(existing_idx);
                        restored += 1;
                    }
                    continue;
                }
            }

            let id = self.next_tab_id;
            self.next_tab_id += 1;
            let rec_dir = self.recovery_dir_for_tab(id);
            let session = DocumentSession::from_pending_recovery_with_recovery(
                &entry,
                rec_dir,
                hist_root.clone(),
                policy,
            );
            let caret = session.buffer.len_bytes();
            self.tabs.push(DocumentTab::new(
                id,
                session,
                EditorViewState {
                    selection: Selection::caret(caret),
                    ..Default::default()
                },
            ));
            self.refresh_tab_conflict_backup_count(id);
            self.active_tab_idx = Some(self.tabs.len() - 1);
            restored += 1;
        }
        self.maintain_recovery_storage();
        restored
    }

    /// Archive an unrecovered pending crash snapshot into bounded historical backups and clean up
    /// its protected current snapshot directory (PRD 4.5.1 F).
    pub fn dismiss_pending_recovery_to_history(
        &self,
        entry: &PendingRecoveryEntry,
    ) -> Result<Option<PathBuf>, SessionError> {
        let Some(hist_root) = self.history_root() else {
            discard_pending_recovery_entry(entry);
            return Ok(None);
        };
        let path = archive_pending_recovery_to_history(
            hist_root,
            entry,
            HistoricalBackupReason::UnrecoveredCrashSnapshot,
            self.history_policy,
        )?;
        Ok(Some(path))
    }

    /// Explicitly discard an unrecovered pending crash snapshot without creating a historical backup.
    pub fn discard_pending_recovery(&self, entry: &PendingRecoveryEntry) {
        discard_pending_recovery_entry(entry);
    }

    /// List verified historical backups stored under this workbench's recovery root (newest first).
    pub fn list_history_backups(&self) -> Vec<HistoricalBackupEntry> {
        match self.history_root() {
            Some(hist) => list_historical_backups(hist),
            None => Vec::new(),
        }
    }

    /// Maintain recovery root synchronously: prune expired/over-quota historical backups, clean
    /// stale temp/corrupt files, expired legacy backups, and orphan empty dirs, and audit
    /// protected snapshot capacity.
    pub fn maintain_recovery_storage(&mut self) -> Option<RecoveryMaintenanceReport> {
        self.maintain_recovery_storage_with_grace(DEFAULT_ORPHAN_GRACE_SECS)
    }

    /// Run `maintain_recovery_storage` synchronously at most once every
    /// `DEFAULT_MAINTENANCE_INTERVAL_SECS`.
    pub fn maybe_maintain_recovery_storage(&mut self) -> Option<RecoveryMaintenanceReport> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        if let Some(last) = self.last_maintenance_unix_secs {
            if now.saturating_sub(last) < DEFAULT_MAINTENANCE_INTERVAL_SECS {
                return None;
            }
        }
        self.maintain_recovery_storage()
    }

    /// Launch `maintain_recovery_root` on a background worker thread if one is not already active
    /// and the maintenance throttle window has elapsed.
    pub fn maybe_maintain_recovery_storage_async(&mut self) -> bool {
        self.poll_recovery_maintenance();
        if self.pending_maintenance.is_some() {
            return false;
        }
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        if let Some(last) = self.last_maintenance_unix_secs {
            if now.saturating_sub(last) < DEFAULT_MAINTENANCE_INTERVAL_SECS {
                return false;
            }
        }
        self.maintain_recovery_storage_async()
    }

    /// Unconditionally schedule a background recovery maintenance pass unless one is already running.
    pub fn maintain_recovery_storage_async(&mut self) -> bool {
        self.poll_recovery_maintenance();
        if self.pending_maintenance.is_some() {
            return false;
        }
        let Some(root) = self.recovery_root.clone() else {
            return false;
        };
        let active_dirs: Vec<PathBuf> = self
            .tabs
            .iter()
            .filter_map(|t| t.session.recovery_dir.clone())
            .collect();
        let policy = self.history_policy;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        self.last_maintenance_unix_secs = Some(now);

        let (tx, rx) = mpsc::channel();
        match thread::Builder::new()
            .name("vellmark-recovery-maint".into())
            .spawn(move || {
                let report = maintain_recovery_root(
                    &root,
                    &active_dirs,
                    policy,
                    DEFAULT_ORPHAN_GRACE_SECS,
                    now,
                );
                let _ = tx.send((now, report));
            }) {
            Ok(handle) => {
                self.pending_maintenance = Some(MaintenanceJob {
                    rx,
                    handle: Some(handle),
                });
                true
            }
            Err(_) => {
                self.maintain_recovery_storage();
                false
            }
        }
    }

    /// Poll a background recovery maintenance task without blocking the GUI event loop.
    pub fn poll_recovery_maintenance(&mut self) -> Option<RecoveryMaintenanceReport> {
        let recv_result = match self.pending_maintenance.as_ref() {
            Some(job) => job.rx.try_recv(),
            None => return None,
        };
        match recv_result {
            Ok((ts, report)) => {
                if let Some(mut job) = self.pending_maintenance.take() {
                    if let Some(h) = job.handle.take() {
                        let _ = h.join();
                    }
                }
                self.last_maintenance_unix_secs = Some(ts);
                self.last_recovery_maintenance = Some(report.clone());
                self.refresh_all_conflict_backup_counts();
                Some(report)
            }
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => {
                if let Some(mut job) = self.pending_maintenance.take() {
                    if let Some(h) = job.handle.take() {
                        let _ = h.join();
                    }
                }
                None
            }
        }
    }

    /// Block until any in-flight background recovery maintenance thread completes.
    pub fn finish_pending_recovery_maintenance(&mut self) -> Option<RecoveryMaintenanceReport> {
        let mut job = self.pending_maintenance.take()?;
        if let Some(handle) = job.handle.take() {
            let _ = handle.join();
        }
        if let Ok((ts, report)) = job.rx.try_recv() {
            self.last_maintenance_unix_secs = Some(ts);
            self.last_recovery_maintenance = Some(report.clone());
            return Some(report);
        }
        None
    }

    pub fn maintain_recovery_storage_with_grace(
        &mut self,
        orphan_grace_secs: u64,
    ) -> Option<RecoveryMaintenanceReport> {
        self.finish_pending_recovery_maintenance();
        let root = self.recovery_root.as_ref()?;
        let active_dirs: Vec<PathBuf> = self
            .tabs
            .iter()
            .filter_map(|t| t.session.recovery_dir.clone())
            .collect();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let report = maintain_recovery_root(
            root,
            &active_dirs,
            self.history_policy,
            orphan_grace_secs,
            now,
        );
        self.last_maintenance_unix_secs = Some(now);
        self.last_recovery_maintenance = Some(report.clone());
        self.refresh_all_conflict_backup_counts();
        Some(report)
    }

    /// Refresh cached count of conflict safety backups under `history/` for one open tab.
    pub fn refresh_tab_conflict_backup_count(&mut self, tab_id: u64) {
        let Some(hist) = self.history_root() else {
            if let Some(tab) = self.tabs.iter_mut().find(|t| t.id == tab_id) {
                tab.conflict_backup_count = 0;
            }
            return;
        };
        let Some(pos) = self.tabs.iter().position(|t| t.id == tab_id) else {
            return;
        };
        let Some(src_path) = self.tabs[pos].session.path.clone() else {
            self.tabs[pos].conflict_backup_count = 0;
            return;
        };
        let count = list_historical_backups_for_path(hist, Some(&src_path))
            .into_iter()
            .filter(|e| e.reason.is_conflict_backup())
            .count();
        self.tabs[pos].conflict_backup_count = count;
    }

    /// Refresh cached conflict safety backup counts across all open tabs.
    pub fn refresh_all_conflict_backup_counts(&mut self) {
        let ids: Vec<u64> = self.tabs.iter().map(|t| t.id).collect();
        for id in ids {
            self.refresh_tab_conflict_backup_count(id);
        }
    }

    /// Return the earliest scheduled auto-save deadline across all open tabs.
    pub fn next_auto_save_deadline(&self) -> Option<Instant> {
        let idle = Duration::from_millis(self.config.auto_save_delay_ms);
        self.tabs
            .iter()
            .filter_map(|t| t.next_auto_save_deadline(self.config.auto_save_enabled, idle))
            .min()
    }

    /// Harvest completed background auto-saves across all tabs without dispatching new ones.
    pub fn poll_async_saves_at(&mut self, now: Instant) -> Vec<WorkbenchRuntimeEvent> {
        let mut events = Vec::new();
        for tab in &mut self.tabs {
            if let Some(outcome) = tab.session.poll_async_save() {
                tab.apply_save_outcome_at(&outcome, now);
                push_runtime_save_event(tab, &outcome, &mut events);
            }
        }
        events
    }

    /// Advance auto-save timers at `now` (2s idle delay + 5s/15s/60s backoff retries, PRD 4.5.1 A & C).
    pub fn tick_auto_save_at(&mut self, now: Instant) -> Vec<WorkbenchRuntimeEvent> {
        let mut events = self.poll_async_saves_at(now);
        let idle_delay = Duration::from_millis(self.config.auto_save_delay_ms);
        for tab in &mut self.tabs {
            tab.sync_edit_activity_at(now);
            if let Some(deadline) =
                tab.next_auto_save_deadline(self.config.auto_save_enabled, idle_delay)
            {
                if now >= deadline {
                    tab.session.schedule_async_save();
                }
            }
        }
        events.extend(self.poll_async_saves_at(now));
        events
    }

    /// Immediately flush auto-save for a specific tab (e.g. on tab switch per PRD 4.5.1 A).
    pub fn flush_auto_save_for_tab(&mut self, tab_id: u64, now: Instant) -> Vec<WorkbenchRuntimeEvent> {
        if !self.config.auto_save_enabled {
            return Vec::new();
        }
        let mut events = self.poll_async_saves_at(now);
        let Some(tab) = self.tabs.iter_mut().find(|t| t.id == tab_id) else {
            return events;
        };
        if !tab.is_auto_save_eligible() {
            return events;
        }
        let outcome = tab.session.save();
        tab.apply_save_outcome_at(&outcome, now);
        push_runtime_save_event(tab, &outcome, &mut events);
        events
    }

    /// Immediately flush auto-save for every eligible Dirty named tab (on focus loss or window close).
    pub fn flush_all_dirty_auto_saves(&mut self, now: Instant) -> Vec<WorkbenchRuntimeEvent> {
        if !self.config.auto_save_enabled {
            return Vec::new();
        }
        let mut events = self.poll_async_saves_at(now);
        for tab in &mut self.tabs {
            if !tab.is_auto_save_eligible() {
                continue;
            }
            let outcome = tab.session.save();
            tab.apply_save_outcome_at(&outcome, now);
            push_runtime_save_event(tab, &outcome, &mut events);
        }
        events
    }

    /// Synchronize open tabs against external disk modifications, deletions, and renames (PRD 4.5.1 C & G).
    pub fn poll_external_changes(&mut self, bypass_stat_cache: bool) -> Vec<WorkbenchRuntimeEvent> {
        let now = Instant::now();
        let mut events = self.poll_async_saves_at(now);
        for idx in 0..self.tabs.len() {
            if self.tabs[idx].session.path.is_none()
                || self.tabs[idx].session.is_async_save_in_flight()
            {
                continue;
            }
            let tab_id = self.tabs[idx].id;
            let res = self.tabs[idx].sync_from_disk(bypass_stat_cache);
            if let ExternalSyncResult::Renamed { old_path, new_path } = res {
                let duplicate_open = self.tabs.iter().enumerate().any(|(j, other)| {
                    j != idx
                        && other
                            .session
                            .path
                            .as_deref()
                            .is_some_and(|p| paths_refer_to_same_file(p, &new_path))
                });
                if duplicate_open {
                    // Preserve single-document-instance invariant (PRD 4.5.1 E).
                    self.tabs[idx].session.path = Some(old_path);
                    self.tabs[idx].session.on_external_deleted();
                    let has_local_edits = matches!(
                        self.tabs[idx].session.state,
                        DocState::Missing {
                            has_local_edits: true
                        }
                    );
                    let t = self.tabs[idx].title();
                    events.push(WorkbenchRuntimeEvent::BecameMissing {
                        tab_id,
                        title: t,
                        has_local_edits,
                    });
                } else {
                    self.recent_files.record_open(&new_path);
                    self.refresh_tab_conflict_backup_count(tab_id);
                    events.push(WorkbenchRuntimeEvent::Renamed {
                        tab_id,
                        old_path,
                        new_path,
                    });
                    // Reconcile renamed target's file content right away so simultaneous rename +
                    // external edit updates buffer/Conflict immediately without waiting another poll.
                    let follow_up = self.tabs[idx].sync_from_disk(true);
                    push_non_rename_sync_event(&mut self.tabs[idx], follow_up, &mut events);
                }
            } else {
                push_non_rename_sync_event(&mut self.tabs[idx], res, &mut events);
            }
        }
        events
    }

    /// Explicitly delete a single historical / conflict safety backup file under `history/`.
    pub fn delete_history_backup(&mut self, backup_path: &Path) -> io::Result<bool> {
        let Some(hist) = self.history_root() else {
            return Ok(false);
        };
        let removed = delete_historical_backup(hist, backup_path)?;
        if removed {
            self.refresh_all_conflict_backup_counts();
        }
        Ok(removed)
    }

    /// Explicitly remove all conflict-resolution safety backups for `source_path`.
    pub fn clear_conflict_backups_for_path(&mut self, source_path: Option<&Path>) -> usize {
        let removed = match self.history_root() {
            Some(hist) => clear_conflict_backups_for_path(hist, source_path),
            None => 0,
        };
        if removed > 0 {
            self.refresh_all_conflict_backup_counts();
        }
        removed
    }

    /// Explicitly remove all historical backups (conflict + close + crash archive) for `source_path`.
    pub fn clear_history_backups_for_path(&mut self, source_path: Option<&Path>) -> usize {
        let removed = match self.history_root() {
            Some(hist) => clear_historical_backups_for_path(hist, source_path),
            None => 0,
        };
        if removed > 0 {
            self.refresh_all_conflict_backup_counts();
        }
        removed
    }

    pub fn protected_snapshots_exceed_limit(&self) -> bool {
        self.last_recovery_maintenance
            .as_ref()
            .is_some_and(|r| r.protected_exceeds_limit)
    }

    /// Synchronously flush recovery snapshots for all unsaved tabs in the workbench.
    pub fn flush_all_recoveries(&mut self) {
        for tab in &mut self.tabs {
            if tab.session.state != DocState::Clean {
                let _ = tab.session.flush_current_recovery_snapshot();
            }
        }
    }

    /// Explicitly discard persisted recovery snapshots for all tabs
    /// (invoked only after deliberate user discard confirmation).
    pub fn discard_all_recoveries(&mut self) {
        for tab in &mut self.tabs {
            tab.session.discard_recovery_snapshot();
        }
    }

    fn recovery_dir_for_tab(&self, tab_id: u64) -> Option<PathBuf> {
        self.recovery_root
            .as_ref()
            .map(|r| r.join(format!("{}-tab-{tab_id}", self.recovery_session_tag)))
    }
}

fn push_runtime_save_event(
    tab: &DocumentTab,
    outcome: &SaveOutcome,
    events: &mut Vec<WorkbenchRuntimeEvent>,
) {
    let tab_id = tab.id;
    let title = tab.title();
    match outcome {
        SaveOutcome::SavedClean => {
            events.push(WorkbenchRuntimeEvent::AutoSavedClean { tab_id, title });
        }
        SaveOutcome::SavedButStillDirty { .. } => {}
        SaveOutcome::EnteredConflict => {
            events.push(WorkbenchRuntimeEvent::EnteredConflict { tab_id, title });
        }
        SaveOutcome::WriteFailed(err) => {
            if matches!(tab.session.state, DocState::Missing { .. })
                && err == "target file disappeared before write"
            {
                let has_local_edits = matches!(
                    tab.session.state,
                    DocState::Missing {
                        has_local_edits: true
                    }
                );
                events.push(WorkbenchRuntimeEvent::BecameMissing {
                    tab_id,
                    title,
                    has_local_edits,
                });
            } else {
                events.push(WorkbenchRuntimeEvent::WriteFailed {
                    tab_id,
                    title,
                    error: err.clone(),
                });
            }
        }
    }
}

fn push_non_rename_sync_event(
    tab: &mut DocumentTab,
    res: ExternalSyncResult,
    events: &mut Vec<WorkbenchRuntimeEvent>,
) {
    let tab_id = tab.id;
    let title = tab.title();
    match res {
        ExternalSyncResult::NoChange => {
            tab.last_external_read_error = None;
        }
        ExternalSyncResult::ReloadedClean => {
            tab.last_external_read_error = None;
            events.push(WorkbenchRuntimeEvent::ReloadedClean { tab_id, title });
        }
        ExternalSyncResult::EnteredConflict => {
            tab.last_external_read_error = None;
            events.push(WorkbenchRuntimeEvent::EnteredConflict { tab_id, title });
        }
        ExternalSyncResult::ConflictTheirsUpdated => {
            tab.last_external_read_error = None;
            events.push(WorkbenchRuntimeEvent::ConflictTheirsUpdated { tab_id, title });
        }
        ExternalSyncResult::BecameMissing => {
            tab.last_external_read_error = None;
            let has_local_edits = matches!(
                tab.session.state,
                DocState::Missing {
                    has_local_edits: true
                }
            );
            events.push(WorkbenchRuntimeEvent::BecameMissing {
                tab_id,
                title,
                has_local_edits,
            });
        }
        ExternalSyncResult::ReappearedClean { reloaded } => {
            tab.last_external_read_error = None;
            events.push(WorkbenchRuntimeEvent::Reappeared {
                tab_id,
                title,
                reloaded,
            });
        }
        ExternalSyncResult::ReappearedDirty => {
            tab.last_external_read_error = None;
            events.push(WorkbenchRuntimeEvent::Reappeared {
                tab_id,
                title,
                reloaded: false,
            });
        }
        ExternalSyncResult::Renamed { .. } => {}
        ExternalSyncResult::ReadError(err) => {
            if tab.last_external_read_error.as_deref() != Some(err.as_str()) {
                tab.last_external_read_error = Some(err.clone());
                events.push(WorkbenchRuntimeEvent::ReadError {
                    tab_id,
                    title,
                    error: err,
                });
            }
        }
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

    #[test]
    fn test_recovery_dirs_do_not_accumulate_on_save_force_close_or_restore_maintenance() {
        let root = unique_temp_dir("no-recovery-leak");
        let rec_root = root.join("recovery");
        let doc1 = root.join("doc1.md");
        let doc2 = root.join("doc2.md");
        let missing_doc = root.join("missing.md");
        fs::write(&doc1, "initial 1").unwrap();
        fs::write(&doc2, "initial 2").unwrap();
        fs::write(&missing_doc, "initial missing").unwrap();

        // Session 1: edit -> save clean -> tab dir must be completely removed (no empty session dir!)
        let doc1_rec_dir;
        {
            let mut wb = AppWorkbench::new(AppConfig::default(), Some(rec_root.clone()));
            wb.open_document(&doc1).unwrap();
            let t1 = wb.active_tab_mut().unwrap();
            t1.edit(|ctx| ctx.insert_text(" edited")).unwrap();
            t1.session.flush_current_recovery_snapshot().unwrap();
            doc1_rec_dir = t1.session.recovery_dir.clone().unwrap();
            assert!(doc1_rec_dir.exists());

            assert_eq!(t1.save(), SaveOutcome::SavedClean);
            assert!(
                !doc1_rec_dir.exists(),
                "clean save must delete slot/meta files AND remove the empty per-session directory"
            );
        }

        // Session 2: force-close Missing tab with local edits archives historical backup and removes current snapshot dir
        let missing_rec_dir;
        {
            let mut wb = AppWorkbench::new(AppConfig::default(), Some(rec_root.clone()));
            let tid = wb.open_document(&missing_doc).unwrap();
            {
                let t = wb.active_tab_mut().unwrap();
                t.edit(|ctx| ctx.insert_text(" unsaved local edit"))
                    .unwrap();
                t.session.flush_current_recovery_snapshot().unwrap();
                missing_rec_dir = t.session.recovery_dir.clone().unwrap();
                fs::remove_file(&missing_doc).unwrap();
                t.session.on_external_deleted();
            }
            assert!(wb.try_close_tab(tid, true).unwrap());
            assert!(!missing_rec_dir.exists());
            let hist = wb.list_history_backups();
            assert_eq!(hist.len(), 1);
            assert_eq!(
                hist[0].reason,
                HistoricalBackupReason::ClosedMissingUnsaved
            );
        }

        // Session 3: crash snapshot from doc2 can be dismissed into history without leaving orphan crash dir,
        // and maintenance leaves recovery_root clean except `history/`.
        {
            let mut wb_crash = AppWorkbench::new(AppConfig::default(), Some(rec_root.clone()));
            wb_crash.open_document(&doc2).unwrap();
            let t = wb_crash.active_tab_mut().unwrap();
            t.edit(|ctx| ctx.insert_text(" crashed draft")).unwrap();
            t.session.flush_current_recovery_snapshot().unwrap();
        }
        {
            let mut wb3 = AppWorkbench::new(AppConfig::default(), Some(rec_root.clone()));
            let pending = wb3.discover_pending_recoveries();
            assert_eq!(pending.len(), 1);
            let backup_path = wb3.dismiss_pending_recovery_to_history(&pending[0]).unwrap().unwrap();
            assert!(backup_path.exists());
            assert!(vellmark_core::scan_recovery_root(&rec_root).is_empty());
            assert_eq!(wb3.list_history_backups().len(), 2);
            wb3.maintain_recovery_storage_with_grace(0);
        }

        // Only `history/` should remain under `rec_root`!
        let remaining_top_dirs: Vec<_> = fs::read_dir(&rec_root)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(remaining_top_dirs, vec![HISTORICAL_BACKUPS_DIR_NAME.to_string()]);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn test_clean_tabs_skip_snapshots_normal_save_creates_no_history_and_conflict_backups_clear() {
        let root = std::env::temp_dir().join(format!(
            "vellmark-wb-lifecycle-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let rec_root = root.join("recovery");
        let doc = root.join("doc.md");
        fs::create_dir_all(&root).unwrap();
        fs::write(&doc, "Original\n").unwrap();

        let mut wb = AppWorkbench::new(AppConfig::default(), Some(rec_root.clone()));
        let untitled_id = wb.new_document();
        let doc_id = wb.open_document(&doc).unwrap();
        wb.flush_all_recoveries();
        assert!(
            vellmark_core::scan_recovery_root(&rec_root).is_empty(),
            "Clean untitled and Clean opened tabs must never write recovery snapshots"
        );
        assert!(wb.try_close_tab(untitled_id, true).unwrap());
        assert!(wb.list_history_backups().is_empty());

        // Normal edit -> flush snapshot -> normal save: cleans snapshot and creates NO history backup
        {
            let tab = wb.active_tab_mut().unwrap();
            assert_eq!(tab.id, doc_id);
            tab.edit(|ctx| ctx.insert_text("Update 1\n")).unwrap();
            tab.session.flush_current_recovery_snapshot().unwrap();
            assert_eq!(vellmark_core::scan_recovery_root(&rec_root).len(), 1);
            assert_eq!(tab.save(), SaveOutcome::SavedClean);
            assert!(vellmark_core::scan_recovery_root(&rec_root).is_empty());
        }
        assert!(
            wb.list_history_backups().is_empty(),
            "normal save must never write historical backups"
        );

        // Conflict resolution creates safety backup (bounded by TTL/caps), and can be explicitly cleared
        {
            let tab = wb.active_tab_mut().unwrap();
            tab.edit(|ctx| ctx.insert_text("Local\n")).unwrap();
            let rev = tab.session.buffer.revision();
            let rope = tab.session.buffer.snapshot_rope();
            fs::write(&doc, "External conflict\n").unwrap();
            assert_eq!(tab.save(), SaveOutcome::EnteredConflict);
            assert_eq!(
                tab.session.resolve_conflict_keep_mine(rev, rope),
                ConflictActionOutcome::ResolvedClean
            );
        }
        assert_eq!(wb.list_history_backups().len(), 1);
        assert_eq!(
            wb.list_history_backups()[0].reason,
            HistoricalBackupReason::ConflictOverwrittenTheirs
        );
        assert_eq!(wb.clear_conflict_backups_for_path(Some(&doc)), 1);
        assert!(wb.list_history_backups().is_empty());

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn test_async_recovery_maintenance_runs_off_thread_and_updates_report() {
        let root = std::env::temp_dir().join(format!(
            "vellmark-wb-async-maint-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let rec_root = root.join("recovery");
        fs::create_dir_all(&rec_root).unwrap();

        let mut wb = AppWorkbench::new(AppConfig::default(), Some(rec_root.clone()));
        wb.new_document();
        {
            let tab = wb.active_tab_mut().unwrap();
            tab.edit(|ctx| ctx.insert_text("active unsaved content")).unwrap();
            tab.session.flush_current_recovery_snapshot().unwrap();
        }
        assert!(wb.last_recovery_maintenance.is_none());
        assert!(wb.maintain_recovery_storage_async());

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let report = loop {
            if let Some(r) = wb.poll_recovery_maintenance() {
                break r;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "timed out waiting for background maintenance worker"
            );
            std::thread::sleep(std::time::Duration::from_millis(5));
        };
        assert!(report.protected_snapshot_bytes > 0);
        assert!(wb.last_recovery_maintenance.is_some());

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn test_runtime_auto_save_2s_idle_retry_backoff_and_flush_on_switch() {
        let root = std::env::temp_dir().join(format!(
            "vellmark-wb-autosave-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let rec_root = root.join("recovery");
        let doc_a = root.join("a.md");
        let doc_b = root.join("b.md");
        fs::create_dir_all(&rec_root).unwrap();
        fs::write(&doc_a, "alpha\n").unwrap();
        fs::write(&doc_b, "beta\n").unwrap();

        let mut wb = AppWorkbench::new(AppConfig::default(), Some(rec_root));
        let id_a = wb.open_document(&doc_a).unwrap();
        let _id_b = wb.open_document(&doc_b).unwrap();
        wb.active_tab_idx = Some(0);

        let t0 = Instant::now();
        {
            let tab = wb.tabs.iter_mut().find(|t| t.id == id_a).unwrap();
            tab.edit(|ctx| ctx.insert_text("edited-")).unwrap();
            tab.last_edit_at = Some(t0);
        }

        // Before 2000ms idle threshold -> does NOT write yet
        let early = wb.tick_auto_save_at(t0 + Duration::from_millis(1500));
        assert!(early.is_empty());
        assert_eq!(fs::read_to_string(&doc_a).unwrap(), "alpha\n");

        // After 2000ms idle threshold -> queues async save and finishes cleanly
        let _ = wb.tick_auto_save_at(t0 + Duration::from_millis(2050));
        let deadline = Instant::now() + Duration::from_secs(5);
        while wb.tabs.iter().find(|t| t.id == id_a).unwrap().session.state != DocState::Clean {
            let _ = wb.tick_auto_save_at(Instant::now());
            assert!(Instant::now() < deadline, "timed out waiting for 2s auto-save");
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(fs::read_to_string(&doc_a).unwrap(), "edited-alpha\n");

        // Verify 5s -> 15s -> 60s write-failure retry backoff schedule
        {
            let tab = wb.tabs.iter_mut().find(|t| t.id == id_a).unwrap();
            tab.edit(|ctx| ctx.insert_text("x")).unwrap();
            let err = SaveOutcome::WriteFailed("simulated I/O error".to_string());
            let base = t0 + Duration::from_secs(10);
            tab.apply_save_outcome_at(&err, base);
            assert_eq!(tab.auto_save_retry_attempt, 1);
            assert_eq!(
                tab.next_auto_save_deadline(true, Duration::from_secs(2)),
                Some(base + Duration::from_secs(5))
            );
            tab.apply_save_outcome_at(&err, base + Duration::from_secs(5));
            assert_eq!(tab.auto_save_retry_attempt, 2);
            assert_eq!(
                tab.next_auto_save_deadline(true, Duration::from_secs(2)),
                Some(base + Duration::from_secs(20))
            );
            tab.apply_save_outcome_at(&err, base + Duration::from_secs(20));
            assert_eq!(tab.auto_save_retry_attempt, 3);
            assert_eq!(
                tab.next_auto_save_deadline(true, Duration::from_secs(2)),
                Some(base + Duration::from_secs(80))
            );
        }

        // Flush on switch writes Dirty named tab immediately regardless of 2s delay
        let evs = wb.flush_auto_save_for_tab(id_a, Instant::now());
        assert!(evs.iter().any(|e| matches!(e, WorkbenchRuntimeEvent::AutoSavedClean { tab_id, .. } if *tab_id == id_a)));
        assert_eq!(fs::read_to_string(&doc_a).unwrap(), "edited-xalpha\n");

        #[cfg(unix)]
        {
            // Simultaneous external edit + rename is reconciled immediately on a single poll
            let doc_c = root.join("c.md");
            fs::write(&doc_b, "beta-modified-and-renamed\n").unwrap();
            fs::rename(&doc_b, &doc_c).unwrap();
            let poll_evs = wb.poll_external_changes(false);
            assert!(poll_evs
                .iter()
                .any(|e| matches!(e, WorkbenchRuntimeEvent::Renamed { new_path, .. } if new_path == &doc_c)));
            assert!(poll_evs
                .iter()
                .any(|e| matches!(e, WorkbenchRuntimeEvent::ReloadedClean { .. })));
            let tab_b = wb
                .tabs
                .iter()
                .find(|t| t.session.path.as_deref() == Some(doc_c.as_path()))
                .unwrap();
            assert_eq!(tab_b.session.buffer.to_string(), "beta-modified-and-renamed\n");
            assert_eq!(tab_b.session.state, DocState::Clean);
        }

        // save_tab_as must refuse binding two tabs to the same file path
        let other_path = wb.tabs[1].session.path.clone().unwrap();
        let err = wb.save_tab_as(id_a, &other_path).unwrap_err();
        assert!(err.contains("already open"));
        assert_eq!(
            wb.tabs.iter().find(|t| t.id == id_a).unwrap().session.path.as_deref(),
            Some(doc_a.as_path())
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn test_paste_rich_clipboard_normalizes_newlines_to_dominant_document_eol() {
        let root = std::env::temp_dir().join(format!(
            "vellmark-paste-eol-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let crlf_path = root.join("crlf.md");
        fs::write(&crlf_path, b"# Title\r\n\r\nBody\r\n").unwrap();

        let mut wb = AppWorkbench::new(AppConfig::default(), None);
        let id_crlf = wb.open_document(&crlf_path).unwrap();
        let tab = wb.tabs.iter_mut().find(|t| t.id == id_crlf).unwrap();
        let insert_at = "# Title\r\n\r\n".len();
        tab.edit(|ctx| ctx.set_caret(insert_at)).unwrap();

        tab.paste_rich_clipboard(Some("<p>Para 1</p><p>Para 2</p>"), "Para 1\nPara 2")
            .unwrap();
        tab.paste_rich_clipboard(None, "Plain 1\nPlain 2").unwrap();
        tab.paste_rich_clipboard(None, "h1\th2\nv1\tv2").unwrap();

        let buf = tab.session.buffer.to_string();
        for (idx, &b) in buf.as_bytes().iter().enumerate() {
            if b == b'\n' {
                assert!(idx > 0 && buf.as_bytes()[idx - 1] == b'\r', "bare LF at {idx} in {buf:?}");
            }
        }
        assert_eq!(tab.save(), SaveOutcome::SavedClean);

        // LF document must normalize CRLF clipboard text to `\n`
        let lf_path = root.join("lf.md");
        fs::write(&lf_path, b"# Title\n\nBody\n").unwrap();
        let id_lf = wb.open_document(&lf_path).unwrap();
        let lf_tab = wb.tabs.iter_mut().find(|t| t.id == id_lf).unwrap();
        lf_tab.edit(|ctx| ctx.set_caret("# Title\n\n".len())).unwrap();
        lf_tab
            .paste_rich_clipboard(None, "Clip 1\r\nClip 2\r\n")
            .unwrap();
        let lf_buf = lf_tab.session.buffer.to_string();
        assert!(!lf_buf.contains('\r'), "unexpected CR in LF doc: {lf_buf:?}");
        assert!(lf_buf.contains("Clip 1\nClip 2\n"));

        let _ = fs::remove_dir_all(&root);
    }
}
