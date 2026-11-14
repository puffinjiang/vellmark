//! High-level WYSIWYG editor controller implementing PRD 4.1.1 A–G:
//! - Mode switching (`WYSIWYG <-> Source`) with exact byte-offset cursor preservation (E-05, 4.1.1 E)
//! - IME composition lifecycle (`ime_update`, `ime_commit`, `ime_cancel`) where pre-edit text is
//!   never written to source buffer during composition (E-08, 4.1.1 F)
//! - Structural editing (`Backspace`, `Enter`, `Tab`, `Shift+Tab`) for headings, nested lists,
//!   task lists, blockquotes, fenced code blocks, and tables (4.1.1 C & G)

use crate::buffer::{ByteOffset, ByteRange, Selection, TextBuffer};
use crate::encoding::LineEnding;
use crate::undo::{EditKind, EditOp, Transaction, UndoHistory};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewMode {
    Wysiwyg,
    Source,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditorAction {
    InsertChar(char),
    InsertText(String),
    Backspace,
    Enter,
    Tab,
    ShiftTab,
    SelectAll,
    ToggleSourceMode,
    ToggleBold,
    ToggleItalic,
    ToggleInlineCode,
    ToggleStrikethrough,
    SetHeadingLevel(u8),
    ToggleTaskCheckbox,
    AlignCurrentTable,
    Undo,
    Redo,
}

/// Pure view/selection/IME state attached to a document session without duplicating `TextBuffer` or `UndoHistory`.
#[derive(Debug, Clone)]
pub struct EditorViewState {
    pub selection: Selection,
    pub mode: ViewMode,
    /// Active IME pre-edit string (never written to `buffer` until `ime_commit`, PRD 4.1.1 F).
    pub ime_preedit: Option<String>,
    /// Full `Selection` (preserving `anchor` and `head` direction) where the active IME composition
    /// started, so if the user clicks/moves the cursor before `ime_commit` arrives, committed text is
    /// inserted at the original composition range, and `undo()` restores the exact directional selection.
    pub ime_composition_selection: Option<Selection>,
    /// Tracks whether block expansion is frozen due to active IME composition.
    pub expansion_frozen: bool,
}

impl Default for EditorViewState {
    fn default() -> Self {
        Self {
            selection: Selection::caret(0),
            mode: ViewMode::Wysiwyg,
            ime_preedit: None,
            ime_composition_selection: None,
            expansion_frozen: false,
        }
    }
}

impl EditorViewState {
    pub fn context<'a>(
        &'a mut self,
        buffer: &'a mut TextBuffer,
        undo: &'a mut UndoHistory,
    ) -> EditorContext<'a> {
        self.context_with_line_ending(buffer, undo, LineEnding::Lf)
    }

    pub fn context_with_line_ending<'a>(
        &'a mut self,
        buffer: &'a mut TextBuffer,
        undo: &'a mut UndoHistory,
        preferred_line_ending: LineEnding,
    ) -> EditorContext<'a> {
        EditorContext {
            buffer,
            undo,
            selection: &mut self.selection,
            mode: &mut self.mode,
            ime_preedit: &mut self.ime_preedit,
            ime_composition_selection: &mut self.ime_composition_selection,
            expansion_frozen: &mut self.expansion_frozen,
            preferred_line_ending,
        }
    }
}

/// Borrowed editing context shared by standalone `EditorController` and `DocumentSession`
/// so all structural editing, IME, and undo rules live in one single implementation (zero duplication).
pub struct EditorContext<'a> {
    pub buffer: &'a mut TextBuffer,
    pub undo: &'a mut UndoHistory,
    pub selection: &'a mut Selection,
    pub mode: &'a mut ViewMode,
    pub ime_preedit: &'a mut Option<String>,
    pub ime_composition_selection: &'a mut Option<Selection>,
    pub expansion_frozen: &'a mut bool,
    pub preferred_line_ending: LineEnding,
}

/// Stateful standalone editor controller wrapping `TextBuffer`, `Selection`, `UndoHistory`,
/// `ViewMode`, and active IME composition state (PRD 4.1.1).
#[derive(Debug, Clone)]
pub struct EditorController {
    pub buffer: TextBuffer,
    pub selection: Selection,
    pub undo: UndoHistory,
    pub mode: ViewMode,
    /// Active IME pre-edit string (never written to `buffer` until `ime_commit`, PRD 4.1.1 F).
    pub ime_preedit: Option<String>,
    /// Full `Selection` (preserving `anchor` and `head` direction) where the active IME composition
    /// started, so if the user clicks/moves the cursor before `ime_commit` arrives, committed text is
    /// inserted at the original composition range, and `undo()` restores the exact directional selection.
    pub ime_composition_selection: Option<Selection>,
    /// Tracks whether block expansion is frozen due to active IME composition.
    pub expansion_frozen: bool,
    /// Remembered document line-ending preference used as fallback when all line breaks are deleted.
    pub preferred_line_ending: LineEnding,
}

impl Default for EditorController {
    fn default() -> Self {
        Self::new()
    }
}

impl EditorController {
    pub fn new() -> Self {
        Self::with_line_ending("", LineEnding::Lf)
    }

    pub fn from_str(text: &str) -> Self {
        Self::with_line_ending(text, LineEnding::detect_dominant(text))
    }

    pub fn with_line_ending(text: &str, preferred_line_ending: LineEnding) -> Self {
        Self {
            buffer: TextBuffer::from_str(text),
            selection: Selection::caret(0),
            undo: UndoHistory::new(),
            mode: ViewMode::Wysiwyg,
            ime_preedit: None,
            ime_composition_selection: None,
            expansion_frozen: false,
            preferred_line_ending,
        }
    }

    pub fn ctx(&mut self) -> EditorContext<'_> {
        EditorContext {
            buffer: &mut self.buffer,
            undo: &mut self.undo,
            selection: &mut self.selection,
            mode: &mut self.mode,
            ime_preedit: &mut self.ime_preedit,
            ime_composition_selection: &mut self.ime_composition_selection,
            expansion_frozen: &mut self.expansion_frozen,
            preferred_line_ending: self.preferred_line_ending,
        }
    }

    pub fn dispatch(&mut self, action: EditorAction) {
        self.ctx().dispatch(action);
    }

    /// Toggle between WYSIWYG and Source mode while preserving exact source byte offset (PRD 4.1.1 E & G #11).
    pub fn toggle_mode(&mut self) {
        self.ctx().toggle_mode();
    }

    pub fn set_caret(&mut self, offset: ByteOffset) {
        self.ctx().set_caret(offset);
    }

    pub fn set_selection(&mut self, anchor: ByteOffset, head: ByteOffset) {
        self.ctx().set_selection(anchor, head);
    }

    pub fn ime_update(&mut self, preedit: &str) {
        self.ctx().ime_update(preedit);
    }

    pub fn ime_clear_preedit(&mut self) {
        self.ctx().ime_clear_preedit();
    }

    pub fn ime_cancel(&mut self) {
        self.ctx().ime_cancel();
    }

    pub fn abort_ime_preedit(&mut self) {
        self.ctx().abort_ime_preedit();
    }

    pub fn ime_commit(&mut self, committed: &str) {
        self.ctx().ime_commit(committed);
    }

    pub fn insert_char(&mut self, ch: char) {
        self.ctx().insert_char(ch);
    }

    pub fn insert_text(&mut self, text: &str) {
        self.ctx().insert_text(text);
    }

    pub fn toggle_inline_wrap(&mut self, marker: &str) {
        self.ctx().toggle_inline_wrap(marker);
    }

    pub fn insert_link(&mut self, url: &str) {
        self.ctx().insert_link(url);
    }

    pub fn set_heading_level(&mut self, level: u8) {
        self.ctx().set_heading_level(level);
    }

    pub fn toggle_task_checkbox(&mut self) -> bool {
        self.ctx().toggle_task_checkbox()
    }

    pub fn backspace(&mut self) {
        self.ctx().backspace();
    }

    pub fn tab(&mut self) {
        self.ctx().tab();
    }

    pub fn shift_tab(&mut self) {
        self.ctx().shift_tab();
    }

    pub fn enter(&mut self) {
        self.ctx().enter();
    }

    pub fn undo(&mut self) {
        self.ctx().undo();
    }

    pub fn redo(&mut self) {
        self.ctx().redo();
    }

    pub fn select_all(&mut self) {
        self.ctx().select_all();
    }

    pub fn align_current_table(&mut self) -> bool {
        self.ctx().align_current_table()
    }
}

impl EditorContext<'_> {
    pub fn dispatch(&mut self, action: EditorAction) {
        match action {
            EditorAction::InsertChar(ch) => self.insert_char(ch),
            EditorAction::InsertText(text) => self.insert_text(&text),
            EditorAction::Backspace => self.backspace(),
            EditorAction::Enter => self.enter(),
            EditorAction::Tab => self.tab(),
            EditorAction::ShiftTab => self.shift_tab(),
            EditorAction::SelectAll => self.select_all(),
            EditorAction::ToggleSourceMode => self.toggle_mode(),
            EditorAction::ToggleBold => self.toggle_inline_wrap("**"),
            EditorAction::ToggleItalic => self.toggle_inline_wrap("*"),
            EditorAction::ToggleInlineCode => self.toggle_inline_wrap("`"),
            EditorAction::ToggleStrikethrough => self.toggle_inline_wrap("~~"),
            EditorAction::SetHeadingLevel(lvl) => self.set_heading_level(lvl),
            EditorAction::ToggleTaskCheckbox => {
                self.toggle_task_checkbox();
            }
            EditorAction::AlignCurrentTable => {
                self.align_current_table();
            }
            EditorAction::Undo => self.undo(),
            EditorAction::Redo => self.redo(),
        }
    }

    /// Toggle between WYSIWYG and Source mode while preserving exact source byte offset (PRD 4.1.1 E & G #11).
    pub fn toggle_mode(&mut self) {
        *self.ime_preedit = None;
        *self.expansion_frozen = false;
        *self.mode = match *self.mode {
            ViewMode::Wysiwyg => ViewMode::Source,
            ViewMode::Source => ViewMode::Wysiwyg,
        };
    }

    pub fn set_caret(&mut self, offset: ByteOffset) {
        // Keep `ime_composition_selection` intact if an IME session was active so a subsequent `ime_commit`
        // triggered by the focus/click change still commits at the original composition range!
        *self.ime_preedit = None;
        *self.expansion_frozen = false;
        let clamped = offset.min(self.buffer.len_bytes());
        *self.selection = Selection::caret(clamped);
    }

    pub fn set_selection(&mut self, anchor: ByteOffset, head: ByteOffset) {
        *self.ime_preedit = None;
        *self.expansion_frozen = false;
        let max = self.buffer.len_bytes();
        *self.selection = Selection::new(anchor.min(max), head.min(max));
    }

    /// Start or update IME composition (PRD 4.1.1 F):
    /// Pre-edit string is stored in `ime_preedit`, `ime_composition_selection` captures the starting selection
    /// (including backward selection direction `anchor > head`), `buffer` revision and content remain untouched,
    /// and block expansion is frozen.
    pub fn ime_update(&mut self, preedit: &str) {
        if self.ime_composition_selection.is_none() {
            *self.ime_composition_selection = Some(*self.selection);
        }
        *self.ime_preedit = Some(preedit.to_string());
        *self.expansion_frozen = true;
    }

    /// Clear visible preedit string (e.g., empty `Preedit("", None)` emitted immediately before OS `Commit`)
    /// while preserving `ime_composition_selection` so a trailing commit still lands on the composition anchor.
    pub fn ime_clear_preedit(&mut self) {
        *self.ime_preedit = None;
        *self.expansion_frozen = false;
    }

    /// Cancel IME composition (e.g. Esc key handled by IME, PRD 4.1.1 F).
    pub fn ime_cancel(&mut self) {
        *self.ime_preedit = None;
        *self.ime_composition_selection = None;
        *self.expansion_frozen = false;
    }

    /// Clear unconfirmed IME preedit and composition anchor when a non-IME edit occurs.
    pub fn abort_ime_preedit(&mut self) {
        *self.ime_preedit = None;
        *self.ime_composition_selection = None;
        *self.expansion_frozen = false;
    }

    /// Commit IME composition as a single standalone undo transaction (PRD 4.1.1 F & G #9, #10).
    /// If `set_caret` / `set_selection` moved the cursor during composition before `ime_commit` arrived,
    /// inserts `committed` at the original `ime_composition_selection` (so `A|B` -> move to end -> commit `"你"` yields `A你B`)
    /// and shifts the new cursor position by the byte delta if needed.
    pub fn ime_commit(&mut self, committed: &str) {
        *self.ime_preedit = None;
        *self.expansion_frozen = false;
        let orig_selection = self
            .ime_composition_selection
            .take()
            .unwrap_or(*self.selection);
        let target_range = orig_selection.range();
        if committed.is_empty() {
            return;
        }
        let cursor_moved_away = *self.selection != orig_selection;
        let sel_before = orig_selection;
        let commit_end = target_range.start + committed.len();
        let sel_after = if cursor_moved_away {
            let shift_point = |pt: ByteOffset| -> ByteOffset {
                if pt <= target_range.start {
                    pt
                } else if pt >= target_range.end {
                    pt - target_range.len() + committed.len()
                } else {
                    commit_end
                }
            };
            Selection::new(
                shift_point(self.selection.anchor),
                shift_point(self.selection.head),
            )
        } else {
            Selection::caret(commit_end)
        };
        self.undo.apply_edit(
            self.buffer,
            target_range,
            committed,
            sel_before,
            sel_after,
            EditKind::ImeCommit,
        );
        *self.selection = sel_after;
    }

    fn dominant_line_ending(&self) -> &'static str {
        let full = self.buffer.to_string();
        LineEnding::detect_dominant_or(&full, self.preferred_line_ending).as_str()
    }

    /// Snap caret/selection endpoints out of the middle of `\r\n` so user edits never split CRLF.
    fn normalize_selection_out_of_crlf(&mut self) {
        let norm_anchor = self.buffer.snap_out_of_crlf(self.selection.anchor);
        let norm_head = self.buffer.snap_out_of_crlf(self.selection.head);
        if norm_anchor != self.selection.anchor || norm_head != self.selection.head {
            *self.selection = Selection::new(norm_anchor, norm_head);
        }
    }

    /// Insert an arbitrary text snippet (e.g. paste / rich-text conversion / image link) as a single transaction.
    pub fn insert_text(&mut self, text: &str) {
        self.abort_ime_preedit();
        self.normalize_selection_out_of_crlf();
        if text.is_empty() && self.selection.is_caret() {
            return;
        }
        let range = self.selection.range();
        let sel_before = *self.selection;
        let sel_after = Selection::caret(range.start + text.len());
        self.undo.apply_edit(
            self.buffer,
            range,
            text,
            sel_before,
            sel_after,
            EditKind::Structural,
        );
        *self.selection = sel_after;
    }

    /// Toggle inline Markdown formatting markers (`**`, `*`, `` ` ``, `~~`) around the current selection (PRD 4.10).
    pub fn toggle_inline_wrap(&mut self, marker: &str) {
        self.abort_ime_preedit();
        if marker.is_empty() {
            return;
        }
        let range = self.selection.range();
        let was_backward = self.selection.anchor > self.selection.head;
        let sel_before = *self.selection;
        let m_len = marker.len();
        let full = self.buffer.to_string();
        let selected = &full[range.start..range.end];
        let full_bytes = full.as_bytes();
        let single_emph = is_single_byte_emphasis_marker(marker);

        // Case 1: Selection itself includes matching opening and closing markers (`**text**` / `*text*`)
        let mut unwrap_inner = !range.is_empty()
            && range.len() >= m_len * 2
            && selected.starts_with(marker)
            && selected.ends_with(marker);
        if unwrap_inner && single_emph {
            let ch = marker.as_bytes()[0];
            let sel_bytes = selected.as_bytes();
            let inner_left = count_consecutive_byte_right(sel_bytes, 0, ch);
            let inner_right = count_consecutive_byte_left(sel_bytes, sel_bytes.len(), ch);
            let outer_left = count_consecutive_byte_left(full_bytes, range.start, ch);
            let outer_right = count_consecutive_byte_right(full_bytes, range.end, ch);
            unwrap_inner = inner_left % 2 == 1
                && inner_right % 2 == 1
                && (outer_left + inner_left) % 2 == 1
                && (inner_right + outer_right) % 2 == 1;
        }
        if unwrap_inner {
            let inner = &selected[m_len..selected.len() - m_len];
            let new_start = range.start;
            let new_end = new_start + inner.len();
            let sel_after = if was_backward {
                Selection::new(new_end, new_start)
            } else {
                Selection::new(new_start, new_end)
            };
            self.undo.apply_edit(
                self.buffer,
                range,
                inner,
                sel_before,
                sel_after,
                EditKind::Structural,
            );
            *self.selection = sel_after;
            return;
        }

        // Case 2: Outer markers immediately wrap the selection (`**` before and `**` after)
        let mut unwrap_outer = !range.is_empty()
            && range.start >= m_len
            && range.end + m_len <= full.len()
            && full_bytes.get(range.start - m_len..range.start) == Some(marker.as_bytes())
            && full_bytes.get(range.end..range.end + m_len) == Some(marker.as_bytes());
        if unwrap_outer && single_emph {
            let ch = marker.as_bytes()[0];
            let sel_bytes = selected.as_bytes();
            let outer_left = count_consecutive_byte_left(full_bytes, range.start, ch);
            let outer_right = count_consecutive_byte_right(full_bytes, range.end, ch);
            let inner_left = count_consecutive_byte_right(sel_bytes, 0, ch);
            let inner_right = count_consecutive_byte_left(sel_bytes, sel_bytes.len(), ch);
            unwrap_outer = outer_left % 2 == 1
                && outer_right % 2 == 1
                && (outer_left + inner_left) % 2 == 1
                && (inner_right + outer_right) % 2 == 1;
        }
        if unwrap_outer {
            let new_start = range.start - m_len;
            let new_end = new_start + selected.len();
            let sel_after = if was_backward {
                Selection::new(new_end, new_start)
            } else {
                Selection::new(new_start, new_end)
            };
            self.undo.apply_edit(
                self.buffer,
                ByteRange::new(new_start, range.end + m_len),
                selected,
                sel_before,
                sel_after,
                EditKind::Structural,
            );
            *self.selection = sel_after;
            return;
        }

        // Case 3: Wrap selection (or insert marker pair around empty caret)
        let wrapped = format!("{marker}{selected}{marker}");
        let sel_after = if range.is_empty() {
            Selection::caret(range.start + m_len)
        } else if was_backward {
            Selection::new(range.end + m_len, range.start + m_len)
        } else {
            Selection::new(range.start + m_len, range.end + m_len)
        };
        self.undo.apply_edit(
            self.buffer,
            range,
            &wrapped,
            sel_before,
            sel_after,
            EditKind::Structural,
        );
        *self.selection = sel_after;
    }

    /// Insert or wrap selection as a Markdown hyperlink `[label](url)` (PRD 4.10 `Cmd/Ctrl+K`).
    pub fn insert_link(&mut self, url: &str) {
        self.abort_ime_preedit();
        let range = self.selection.range();
        let sel_before = *self.selection;
        let selected = self.buffer.slice_to_string(range);
        let snippet = format!("[{selected}]({url})");
        let caret_pos = if selected.is_empty() {
            range.start + 1
        } else {
            range.start + snippet.len()
        };
        let sel_after = Selection::caret(caret_pos);
        self.undo.apply_edit(
            self.buffer,
            range,
            &snippet,
            sel_before,
            sel_after,
            EditKind::Structural,
        );
        *self.selection = sel_after;
    }

    /// Convert current line to Paragraph (`level == 0`) or Heading 1–6 (`1..=6`, PRD 4.10 `Cmd/Ctrl + 0..6`).
    pub fn set_heading_level(&mut self, level: u8) {
        self.abort_ime_preedit();
        let (line_start, line_end) = self.current_line_bounds();
        if *self.mode == ViewMode::Wysiwyg && self.is_offset_inside_fenced_code_block(line_start) {
            return;
        }
        let line = self.buffer.slice_to_string(ByteRange::new(line_start, line_end));
        let old_prefix_len = heading_prefix_len(&line).unwrap_or(0);
        let new_prefix = if level == 0 || level > 6 {
            String::new()
        } else {
            format!("{} ", "#".repeat(usize::from(level)))
        };
        if &line[..old_prefix_len] == new_prefix {
            return;
        }

        let sel_before = *self.selection;
        let rel_head = sel_before.head.saturating_sub(line_start);
        let new_rel_head = if rel_head <= old_prefix_len {
            new_prefix.len()
        } else {
            (rel_head - old_prefix_len) + new_prefix.len()
        };
        let sel_after = Selection::caret(line_start + new_rel_head);
        self.undo.apply_edit(
            self.buffer,
            ByteRange::new(line_start, line_start + old_prefix_len),
            &new_prefix,
            sel_before,
            sel_after,
            EditKind::Structural,
        );
        *self.selection = sel_after;
    }

    /// Toggle task-list checkbox (`- [ ] ` <-> `- [x] `, or `- ` -> `- [ ] `) on current line.
    pub fn toggle_task_checkbox(&mut self) -> bool {
        self.abort_ime_preedit();
        let (line_start, line_end) = self.current_line_bounds();
        if *self.mode == ViewMode::Wysiwyg && self.is_offset_inside_fenced_code_block(line_start) {
            return false;
        }
        let line = self.buffer.slice_to_string(ByteRange::new(line_start, line_end));
        let Some((indent, marker_len)) = list_prefix_parts(&line) else {
            return false;
        };
        let marker = &line[indent..indent + marker_len];
        let replacement = if marker == "- [ ] " {
            "- [x] "
        } else if marker == "- [x] " || marker == "- [X] " {
            "- [ ] "
        } else if marker == "- " || marker == "* " || marker == "+ " {
            "- [ ] "
        } else {
            return false;
        };
        let sel_before = *self.selection;
        let delta = replacement.len() as isize - marker.len() as isize;
        let head = if sel_before.head <= line_start + indent {
            sel_before.head
        } else {
            ((sel_before.head as isize) + delta).max(line_start as isize) as usize
        };
        let sel_after = Selection::caret(head);
        self.undo.apply_edit(
            self.buffer,
            ByteRange::new(line_start + indent, line_start + indent + marker_len),
            replacement,
            sel_before,
            sel_after,
            EditKind::Structural,
        );
        *self.selection = sel_after;
        true
    }

    /// Type a character in WYSIWYG/Source mode with automatic syntax trigger recording (PRD 4.1.1 B & G #1, #6).
    pub fn insert_char(&mut self, ch: char) {
        self.abort_ime_preedit();
        self.normalize_selection_out_of_crlf();

        if ch == '\n' {
            let nl = self.dominant_line_ending();
            let range = self.selection.range();
            let sel_before = *self.selection;
            let sel_after = Selection::caret(range.start + nl.len());
            self.undo.apply_edit(
                self.buffer,
                range,
                nl,
                sel_before,
                sel_after,
                EditKind::Typing,
            );
            *self.selection = sel_after;
            return;
        }

        let (line_start_before, _) = self.current_line_bounds();
        let in_code_fence = self.is_offset_inside_fenced_code_block(line_start_before);

        // PRD 4.1.1 G #6: Inside a Markdown table line (outside code fence), typing `|` in WYSIWYG mode writes escaped `\|` to source.
        if *self.mode == ViewMode::Wysiwyg
            && !in_code_fence
            && ch == '|'
            && self.is_caret_in_table_row()
        {
            let range = self.selection.range();
            let sel_before = *self.selection;
            let sel_after = Selection::caret(range.start + 2);
            self.undo.apply_edit(
                self.buffer,
                range,
                "\\|",
                sel_before,
                sel_after,
                EditKind::Typing,
            );
            *self.selection = sel_after;
            return;
        }

        let mut buf = [0u8; 4];
        let s = ch.encode_utf8(&mut buf);
        let range = self.selection.range();
        let new_caret = range.start + s.len();
        let sel_before = *self.selection;
        let sel_after = Selection::caret(new_caret);
        self.undo.apply_edit(
            self.buffer,
            range,
            s,
            sel_before,
            sel_after,
            EditKind::Typing,
        );
        *self.selection = sel_after;

        // P2 fix: Only record WYSIWYG syntax conversion triggers when in `ViewMode::Wysiwyg` AND outside fenced code blocks!
        if *self.mode == ViewMode::Wysiwyg && !in_code_fence {
            let (line_start, line_end, _) = self.current_line_content_bounds();
            let line = self.buffer.slice_to_string(ByteRange::new(line_start, line_end));
            if ch == ' ' && self.selection.head == line_end {
                if let Some(rel) = block_line_syntax_escape_offset(&line) {
                    self.undo
                        .record_syntax_trigger(line_start + rel, *self.selection);
                }
            } else if matches!(ch, '*' | '`' | '~') {
                let caret_in_line = self.selection.head.saturating_sub(line_start);
                let escapes = detect_inline_conversion_escapes(&line, caret_in_line, ch);
                if escapes.len() == 1 {
                    self.undo
                        .record_syntax_trigger(line_start + escapes[0], *self.selection);
                } else if escapes.len() > 1 {
                    let abs: Vec<ByteOffset> =
                        escapes.into_iter().map(|rel| line_start + rel).collect();
                    self.undo
                        .record_syntax_trigger_offsets(abs, *self.selection);
                }
            }
        }
    }

    /// Handle Backspace with PRD 4.1.1 C & G (#2, #8) structural rules:
    /// - At start of heading content (`## ‸标题`) -> demote heading to paragraph (`标题`) by removing `## `;
    ///   2nd Backspace at paragraph start (`‸标题`) merges with previous block.
    /// - At start of empty fenced code block first line (` ```\n‸\n``` `) -> convert code block to empty paragraph.
    /// - At start of top-level list item (`- ‸item`) -> remove list marker (`item`).
    /// - At start of nested list item (`  - ‸sub`) -> outdent one level (`- ‸sub`).
    pub fn backspace(&mut self) {
        self.abort_ime_preedit();

        if !self.selection.is_caret() {
            self.normalize_selection_out_of_crlf();
            let r = self.selection.range();
            let sel_before = *self.selection;
            let sel_after = Selection::caret(r.start);
            self.undo.apply_edit(
                self.buffer,
                r,
                "",
                sel_before,
                sel_after,
                EditKind::Deletion,
            );
            *self.selection = sel_after;
            return;
        }

        let caret = self.selection.head;
        let (line_start, line_end, _) = self.current_line_content_bounds();
        let line = self.buffer.slice_to_string(ByteRange::new(line_start, line_end));

        if *self.mode == ViewMode::Wysiwyg {
            // PRD 4.1.1 G #8: Backspace at start of empty code block content line converts code block to empty paragraph
            if caret == line_start && line.is_empty() {
                if let Some(fence_range) = self.enclosing_empty_code_fence_range(line_start) {
                    let sel_before = *self.selection;
                    let sel_after = Selection::caret(fence_range.start);
                    self.undo.apply_edit(
                        self.buffer,
                        fence_range,
                        "",
                        sel_before,
                        sel_after,
                        EditKind::Structural,
                    );
                    *self.selection = sel_after;
                    return;
                }
            }

            let in_code_fence = self.is_offset_inside_fenced_code_block(line_start);

            // P2 fix: Inside a fenced code block, lines like `# comment` or `- item` are raw code, NOT Markdown headings/lists!
            if !in_code_fence {
                // PRD 4.1.1 C & G #2: Backspace at start of heading text (either at `line_start` or right after `## `)
                if let Some(prefix_len) = heading_prefix_len(&line) {
                    if caret == line_start || caret == line_start + prefix_len {
                        let sel_before = *self.selection;
                        let sel_after = Selection::caret(line_start);
                        self.undo.apply_edit(
                            self.buffer,
                            ByteRange::new(line_start, line_start + prefix_len),
                            "",
                            sel_before,
                            sel_after,
                            EditKind::Structural,
                        );
                        *self.selection = sel_after;
                        return;
                    }
                }

                // PRD 4.1.1 C: Backspace at start of list item content
                if let Some((indent_bytes, marker_bytes)) = list_prefix_parts(&line) {
                    if caret == line_start + indent_bytes + marker_bytes {
                        let sel_before = *self.selection;
                        if indent_bytes >= 2 {
                            // Nested item -> outdent by 2 spaces
                            let sel_after = Selection::caret(caret - 2);
                            self.undo.apply_edit(
                                self.buffer,
                                ByteRange::new(line_start, line_start + 2),
                                "",
                                sel_before,
                                sel_after,
                                EditKind::Structural,
                            );
                            *self.selection = sel_after;
                        } else {
                            // Top-level list item -> strip list marker, convert to paragraph in place
                            let sel_after = Selection::caret(line_start);
                            self.undo.apply_edit(
                                self.buffer,
                                ByteRange::new(line_start, line_start + indent_bytes + marker_bytes),
                                "",
                                sel_before,
                                sel_after,
                                EditKind::Structural,
                            );
                            *self.selection = sel_after;
                        }
                        return;
                    }
                }

                // PRD 4.1.1 C: Backspace at start of blockquote content demotes one quote level (`> ‸text` -> `text`, `> > ‸text` -> `> text`)
                if let Some((content_start, strip_rel_range)) =
                    blockquote_demote_parts(&line)
                {
                    if caret == line_start + content_start {
                        let sel_before = *self.selection;
                        let del_range = ByteRange::new(
                            line_start + strip_rel_range.start,
                            line_start + strip_rel_range.end,
                        );
                        let sel_after = Selection::caret(caret - del_range.len());
                        self.undo.apply_edit(
                            self.buffer,
                            del_range,
                            "",
                            sel_before,
                            sel_after,
                            EditKind::Structural,
                        );
                        *self.selection = sel_after;
                        return;
                    }
                }
            }
        }

        // Standard character or newline deletion (merges with previous block when at `line_start > 0`)
        if caret > 0 {
            let del_range = if self.buffer.is_crlf_midpoint(caret) {
                // Caret somehow landed mid-CRLF: delete the entire `\r\n` atomically.
                ByteRange::new(caret - 1, caret + 1)
            } else if caret >= 2 && self.buffer.is_crlf_midpoint(caret - 1) {
                // Treat `\r\n` as a single atomic newline unit so Backspace never leaves orphan `\r`.
                ByteRange::new(caret - 2, caret)
            } else {
                let full = self.buffer.to_string();
                let prev_char_len = full[..caret]
                    .chars()
                    .next_back()
                    .map(|c| c.len_utf8())
                    .unwrap_or(1);
                ByteRange::new(caret - prev_char_len, caret)
            };
            let sel_before = *self.selection;
            let sel_after = Selection::caret(del_range.start);
            self.undo.apply_edit(
                self.buffer,
                del_range,
                "",
                sel_before,
                sel_after,
                EditKind::Deletion,
            );
            *self.selection = sel_after;
        }
    }

    /// Handle Tab (PRD 4.1.1 C, G #6 & 4.3 T-02):
    /// - In a list item: indent by 2 spaces.
    /// - In a table cell: jump to next cell on the row, or if at the last cell, jump to the first cell
    ///   of the next data row (or append a new empty row if at the bottom-right cell of the table).
    pub fn tab(&mut self) {
        self.abort_ime_preedit();
        let (line_start, line_end) = self.current_line_bounds();
        if *self.mode == ViewMode::Wysiwyg && !self.is_offset_inside_fenced_code_block(line_start) {
            let line = self.buffer.slice_to_string(ByteRange::new(line_start, line_end));
            if list_prefix_parts(&line).is_some() {
                let sel_before = *self.selection;
                let sel_after = Selection::caret(self.selection.head + 2);
                self.undo.apply_edit(
                    self.buffer,
                    ByteRange::empty(line_start),
                    "  ",
                    sel_before,
                    sel_after,
                    EditKind::Structural,
                );
                *self.selection = sel_after;
                return;
            }
            if self.is_caret_in_table_row() {
                self.tab_in_table(line_start, line_end, &line);
                return;
            }
        }
        // Fallback: insert 2 spaces
        let range = self.selection.range();
        let sel_before = *self.selection;
        let sel_after = Selection::caret(range.start + 2);
        self.undo.apply_edit(
            self.buffer,
            range,
            "  ",
            sel_before,
            sel_after,
            EditKind::Typing,
        );
        *self.selection = sel_after;
    }

    /// Handle Shift+Tab (PRD 4.1.1 C, G #4 & 4.3 T-02):
    /// - Outdent nested list item by one level (2 spaces).
    /// - Or move to previous cell in a Markdown table.
    pub fn shift_tab(&mut self) {
        self.abort_ime_preedit();
        let (line_start, line_end) = self.current_line_bounds();
        if self.is_offset_inside_fenced_code_block(line_start) {
            return;
        }
        let line = self.buffer.slice_to_string(ByteRange::new(line_start, line_end));
        if let Some((indent_bytes, _)) = list_prefix_parts(&line) {
            if indent_bytes >= 2 {
                let sel_before = *self.selection;
                let new_head = self.selection.head.saturating_sub(2).max(line_start);
                let sel_after = Selection::caret(new_head);
                self.undo.apply_edit(
                    self.buffer,
                    ByteRange::new(line_start, line_start + 2),
                    "",
                    sel_before,
                    sel_after,
                    EditKind::Structural,
                );
                *self.selection = sel_after;
            }
            return;
        }
        if *self.mode == ViewMode::Wysiwyg && self.is_caret_in_table_row() {
            let pipes = unescaped_pipe_offsets(&line);
            let rel_caret = self.selection.head.saturating_sub(line_start);
            if pipes.len() >= 2 {
                if let Some(cell_idx) = current_table_cell_index(&pipes, rel_caret) {
                    if cell_idx > 0 {
                        let prev_pipe = pipes[cell_idx - 1];
                        let next_pipe = pipes[cell_idx];
                        let cell_slice = &line[prev_pipe + 1..next_pipe];
                        let target_rel = prev_pipe + 1 + table_cell_caret_offset(cell_slice);
                        *self.selection = Selection::caret(line_start + target_rel);
                        return;
                    }
                    // Cell 0: jump backward to the last cell of the previous non-delimiter table row
                    let full = self.buffer.to_string();
                    let mut probe_end = line_start;
                    while probe_end > 0 {
                        let before = &full[..probe_end - 1];
                        let prev_start = before.rfind('\n').map(|i| i + 1).unwrap_or(0);
                        let prev_line = before[prev_start..].trim_end_matches('\r');
                        if !(prev_line.trim().starts_with('|') && prev_line.trim().ends_with('|')) {
                            break;
                        }
                        if is_table_delimiter_row(prev_line) {
                            probe_end = prev_start;
                            continue;
                        }
                        let prev_pipes = unescaped_pipe_offsets(prev_line);
                        if prev_pipes.len() >= 2 {
                            let last_idx = prev_pipes.len() - 2;
                            let c_slice = &prev_line[prev_pipes[last_idx] + 1..prev_pipes[last_idx + 1]];
                            *self.selection = Selection::caret(
                                prev_start
                                    + prev_pipes[last_idx]
                                    + 1
                                    + table_cell_caret_offset(c_slice),
                            );
                        }
                        break;
                    }
                }
            }
        }
    }

    /// Handle Enter with PRD 4.1.1 C & G (#5, #6, #7) and 4.3 (T-01, T-02):
    /// - Non-empty selection -> replaces the selected range with `\n` in a single transaction (P2 fix).
    /// - Empty top-level list item (`- ‸`) -> exit list and become empty paragraph.
    /// - Empty nested list item (`  - ‸`) -> outdent one level (`- ‸`).
    /// - Non-empty list item (`- item‸`) -> continue list on next line (`\n- `).
    /// - Table row (`| a | b |`):
    ///   - Standalone header row (T-01) -> generates separator + empty data row (`\n| --- | --- |\n|  |  |`).
    ///   - Existing table row (G #6 & T-02) -> inserts a new empty row below (`\n|  |  |`) and places caret in first cell.
    /// - Inside fenced code block at end of block with 2 trailing empty lines (3rd consecutive Enter, G #7) ->
    ///   remove trailing empty lines and exit code block to a new paragraph after closing ` ``` ` in a SINGLE undo transaction (P2 fix).
    pub fn enter(&mut self) {
        self.abort_ime_preedit();

        // Replaces non-empty selection with the document's dominant line ending (P2 fix).
        self.normalize_selection_out_of_crlf();
        if !self.selection.is_caret() {
            let nl = self.dominant_line_ending();
            let range = self.selection.range();
            let sel_before = *self.selection;
            let sel_after = Selection::caret(range.start + nl.len());
            self.undo.apply_edit(
                self.buffer,
                range,
                nl,
                sel_before,
                sel_after,
                EditKind::Typing,
            );
            *self.selection = sel_after;
            return;
        }

        let caret = self.selection.head;
        let (line_start, content_end, line_end) = self.current_line_content_bounds();
        let line = self
            .buffer
            .slice_to_string(ByteRange::new(line_start, content_end));

        if *self.mode == ViewMode::Wysiwyg {
            let in_code_fence = self.is_offset_inside_fenced_code_block(line_start);

            // PRD 4.1.1 G #5: List item handling (only outside fenced code blocks)
            if !in_code_fence {
                if let Some((indent_bytes, marker_bytes)) = list_prefix_parts(&line) {
                    let content_after_marker = line[indent_bytes + marker_bytes..].trim();
                    if content_after_marker.is_empty() {
                        let sel_before = *self.selection;
                        if indent_bytes >= 2 {
                            // Outdent nested empty item
                            let sel_after = Selection::caret(caret - 2);
                            self.undo.apply_edit(
                                self.buffer,
                                ByteRange::new(line_start, line_start + 2),
                                "",
                                sel_before,
                                sel_after,
                                EditKind::Structural,
                            );
                            *self.selection = sel_after;
                        } else {
                            // Exit top-level list -> empty paragraph
                            let sel_after = Selection::caret(line_start);
                            self.undo.apply_edit(
                                self.buffer,
                                ByteRange::new(line_start, content_end),
                                "",
                                sel_before,
                                sel_after,
                                EditKind::Structural,
                            );
                            *self.selection = sel_after;
                        }
                        return;
                    } else if caret == content_end {
                        // Continue list on next line
                        let nl = self.dominant_line_ending();
                        let prefix = &line[..indent_bytes + marker_bytes];
                        let insertion = format!("{nl}{prefix}");
                        let sel_before = *self.selection;
                        let sel_after = Selection::caret(caret + insertion.len());
                        self.undo.apply_edit(
                            self.buffer,
                            ByteRange::empty(caret),
                            &insertion,
                            sel_before,
                            sel_after,
                            EditKind::Structural,
                        );
                        *self.selection = sel_after;
                        return;
                    }
                }

                // PRD 4.1.1 G #6 & 4.3 T-01 / T-02: Table Enter handling
                if self.is_caret_in_table_row() {
                    let pipes = unescaped_pipe_offsets(&line);
                    if pipes.len() >= 2 {
                        let cols = pipes.len() - 1;
                        let empty_row = build_empty_table_row(cols);
                        let is_standalone_header = !self.has_adjacent_table_row(line_start, line_end)
                            && !is_table_delimiter_row(&line);

                        let full = self.buffer.to_string();
                        let nl = detect_line_ending_str(&full, self.preferred_line_ending);
                        let insert_pos = table_row_insert_pos(&full, line_start, line_end);

                        let (insertion, caret_offset_in_insertion) = if is_standalone_header {
                            // PRD 4.3 T-01: `| a | b |` + Enter -> create separator row + first empty data row
                            let sep_row = build_table_separator_row(cols);
                            let ins = format!("{nl}{sep_row}{nl}{empty_row}");
                            let caret_rel = nl.len() + sep_row.len() + nl.len() + 2; // inside `| ‸ |` of empty_row
                            (ins, caret_rel)
                        } else {
                            // PRD 4.1.1 G #6 & 4.3 T-02: Enter inside table inserts a new row below
                            let ins = format!("{nl}{empty_row}");
                            let caret_rel = nl.len() + 2; // after `<nl>| `
                            (ins, caret_rel)
                        };

                        let sel_before = *self.selection;
                        let sel_after = Selection::caret(insert_pos + caret_offset_in_insertion);
                        self.undo.apply_edit(
                            self.buffer,
                            ByteRange::empty(insert_pos),
                            &insertion,
                            sel_before,
                            sel_after,
                            EditKind::Structural,
                        );
                        *self.selection = sel_after;
                        return;
                    }
                }

                // PRD 4.1.1 C: BlockQuote Enter continuation, nested outdent, or empty exit
                if let Some((content_start, strip_rel_range)) = blockquote_demote_parts(&line) {
                    if caret >= line_start + content_start {
                        let sel_before = *self.selection;
                        if line[content_start..].trim().is_empty() {
                            let del_range = if strip_rel_range.start == 0 {
                                ByteRange::new(line_start, content_end)
                            } else {
                                ByteRange::new(
                                    line_start + strip_rel_range.start,
                                    line_start + strip_rel_range.end,
                                )
                            };
                            let sel_after = Selection::caret(del_range.start);
                            self.undo.apply_edit(
                                self.buffer,
                                del_range,
                                "",
                                sel_before,
                                sel_after,
                                EditKind::Structural,
                            );
                            *self.selection = sel_after;
                        } else {
                            let le = self.dominant_line_ending();
                            let prefix = &line[..content_start];
                            let continuation = if prefix.ends_with(' ') {
                                format!("{le}{prefix}")
                            } else {
                                format!("{le}{prefix} ")
                            };
                            let sel_after = Selection::caret(caret + continuation.len());
                            self.undo.apply_edit(
                                self.buffer,
                                ByteRange::empty(caret),
                                &continuation,
                                sel_before,
                                sel_after,
                                EditKind::Structural,
                            );
                            *self.selection = sel_after;
                        }
                        return;
                    }
                }

                // PRD 4.1.1 B.1: ``` / ~~~ + Enter inserts companion closing fence in single SyntaxConversion tx
                if caret == content_end {
                    if let Some((fence_ch, fence_len)) = parse_opening_code_fence(&line) {
                        let full_text = self.buffer.to_string();
                        if !has_unpaired_closing_fence_below(
                            &full_text,
                            content_end,
                            fence_ch,
                            fence_len,
                        ) {
                            let indent = line.len() - line.trim_start_matches(' ').len();
                            let le =
                                LineEnding::detect_dominant_or(&full_text, self.preferred_line_ending)
                                    .as_str();
                            let closer: String =
                                std::iter::repeat(fence_ch as char).take(fence_len).collect();
                            let insertion = format!("{le}{le}{closer}");
                            let sel_before = *self.selection;
                            let sel_after = Selection::caret(caret + le.len());
                            self.undo.apply_edit(
                                self.buffer,
                                ByteRange::empty(caret),
                                &insertion,
                                sel_before,
                                sel_after,
                                EditKind::SyntaxConversion {
                                    escape_insert_at: Some(line_start + indent),
                                },
                            );
                            *self.selection = sel_after;
                            return;
                        }
                    }

                    // PRD 4.1.1 B.1: --- / *** / ___ + Enter completes ThematicBreak with escape-undo
                    if let Some(escape_rel) = thematic_break_escape_offset(&line) {
                        let le = self.dominant_line_ending();
                        let sel_before = *self.selection;
                        let sel_after = Selection::caret(caret + le.len());
                        self.undo.apply_edit(
                            self.buffer,
                            ByteRange::empty(caret),
                            le,
                            sel_before,
                            sel_after,
                            EditKind::SyntaxConversion {
                                escape_insert_at: Some(line_start + escape_rel),
                            },
                        );
                        *self.selection = sel_after;
                        return;
                    }
                }
            }

            // PRD 4.1.1 G #7 & P2 fix: 3rd consecutive Enter at end of fenced code block exits the code block
            // in a SINGLE `Transaction` so a single `undo()` restores the exact pre-Enter state!
            if let Some((strip_range, closing_fence_end)) =
                self.check_third_enter_at_code_fence_end(caret)
            {
                let sel_before = *self.selection;
                let removed_len = strip_range.len();
                let deleted = self.buffer.replace_bytes(strip_range, "");
                let mut ops = vec![EditOp {
                    range: strip_range,
                    deleted,
                    inserted: String::new(),
                }];

                let after_fence = closing_fence_end - removed_len;
                let sel_after = if after_fence == self.buffer.len_bytes() {
                    let nl = self.dominant_line_ending();
                    self.buffer
                        .replace_bytes(ByteRange::empty(after_fence), nl);
                    ops.push(EditOp {
                        range: ByteRange::empty(after_fence),
                        deleted: String::new(),
                        inserted: nl.to_string(),
                    });
                    Selection::caret(after_fence + nl.len())
                } else {
                    Selection::caret(after_fence)
                };

                self.undo.record_transaction(Transaction {
                    ops,
                    selection_before: sel_before,
                    selection_after: sel_after,
                    kind: EditKind::Structural,
                });
                *self.selection = sel_after;
                return;
            }
        }

        // Default newline insertion
        let nl = self.dominant_line_ending();
        let sel_before = *self.selection;
        let sel_after = Selection::caret(caret + nl.len());
        self.undo.apply_edit(
            self.buffer,
            ByteRange::empty(caret),
            nl,
            sel_before,
            sel_after,
            EditKind::Typing,
        );
        *self.selection = sel_after;
    }

    pub fn undo(&mut self) {
        self.abort_ime_preedit();
        if let Some(sel) = self.undo.undo(self.buffer) {
            *self.selection = sel;
        }
    }

    pub fn redo(&mut self) {
        self.abort_ime_preedit();
        if let Some(sel) = self.undo.redo(self.buffer) {
            *self.selection = sel;
        }
    }

    /// Progressive Select-All (`Cmd/Ctrl + A`, PRD 4.1.1 D):
    /// - 1st invocation: selects the current block's content (or inside a fenced code block / table cell,
    ///   selects the current code block body / table cell content).
    /// - 2nd consecutive invocation (when selection already matches the current block target): selects the entire document.
    pub fn select_all(&mut self) {
        self.abort_ime_preedit();
        let doc_len = self.buffer.len_bytes();
        let (line_start, content_end, _) = self.current_line_content_bounds();

        let block_target = if let Some(body_range) =
            self.enclosing_fenced_code_block_body_range(line_start)
        {
            body_range
        } else if self.is_caret_in_table_row() {
            let line = self
                .buffer
                .slice_to_string(ByteRange::new(line_start, content_end));
            let pipes = unescaped_pipe_offsets(&line);
            let rel_caret = self.selection.head.saturating_sub(line_start);
            if let Some(cell_idx) = current_table_cell_index(&pipes, rel_caret) {
                let c_start = line_start + pipes[cell_idx] + 1;
                let c_end = line_start + pipes[cell_idx + 1];
                let raw_cell = self.buffer.slice_to_string(ByteRange::new(c_start, c_end));
                let trim_l = raw_cell.len() - raw_cell.trim_start().len();
                let trim_r = raw_cell.len() - raw_cell.trim_end().len();
                let inner_start = (c_start + trim_l).min(c_end);
                let inner_end = c_end.saturating_sub(trim_r).max(inner_start);
                ByteRange::new(inner_start, inner_end)
            } else {
                ByteRange::new(line_start, content_end)
            }
        } else {
            ByteRange::new(line_start, content_end)
        };

        if !block_target.is_empty() && self.selection.range() != block_target {
            *self.selection = Selection::new(block_target.start, block_target.end);
        } else {
            *self.selection = Selection::new(0, doc_len);
        }
    }

    /// Re-aligns ONLY the Markdown table containing the caret (PRD 4.3 T-03 & 5.2 #2) as a single
    /// undoable transaction. Returns `true` if a table was found and its source changed.
    pub fn align_current_table(&mut self) -> bool {
        self.abort_ime_preedit();
        let (line_start, _) = self.current_line_bounds();
        if self.is_offset_inside_fenced_code_block(line_start) || !self.is_caret_in_table_row() {
            return false;
        }
        let full = self.buffer.to_string();
        let Some(table_range) = find_enclosing_table_range(&full, line_start) else {
            return false;
        };
        let raw_table = &full[table_range.start..table_range.end];
        let aligned = format_markdown_table(raw_table);
        if aligned == raw_table {
            return false;
        }
        let sel_before = *self.selection;
        let mut rel_after = aligned
            .len()
            .min(sel_before.head.saturating_sub(table_range.start));
        while rel_after > 0 && !aligned.is_char_boundary(rel_after) {
            rel_after -= 1;
        }
        let sel_after = Selection::caret(table_range.start + rel_after);
        self.undo.apply_edit(
            self.buffer,
            table_range,
            &aligned,
            sel_before,
            sel_after,
            EditKind::Structural,
        );
        *self.selection = sel_after;
        true
    }

    fn tab_in_table(&mut self, line_start: ByteOffset, line_end: ByteOffset, line: &str) {
        let pipes = unescaped_pipe_offsets(line);
        if pipes.len() < 2 {
            return;
        }
        let cols = pipes.len() - 1;
        let rel_caret = self.selection.head.saturating_sub(line_start);
        let cell_idx = current_table_cell_index(&pipes, rel_caret).unwrap_or(0);

        if cell_idx + 1 < cols {
            // Move to next cell on the same row
            let p_start = pipes[cell_idx + 1];
            let p_end = pipes[cell_idx + 2];
            let cell_slice = &line[p_start + 1..p_end];
            let target_rel = p_start + 1 + table_cell_caret_offset(cell_slice);
            *self.selection = Selection::caret(line_start + target_rel);
            return;
        }

        // At last cell of current row: check if there is a next row in the table
        let full = self.buffer.to_string();
        let mut probe_start = if line_end < full.len() {
            line_end + 1
        } else {
            full.len()
        };
        while probe_start < full.len() {
            let rel_end = full[probe_start..]
                .find('\n')
                .unwrap_or(full.len() - probe_start);
            let next_line_end = probe_start + rel_end;
            let next_line = full[probe_start..next_line_end].trim_end_matches('\r');
            if !(next_line.trim().starts_with('|') && next_line.trim().ends_with('|')) {
                break;
            }
            if is_table_delimiter_row(next_line) {
                probe_start = if next_line_end < full.len() {
                    next_line_end + 1
                } else {
                    full.len()
                };
                continue;
            }
            let next_pipes = unescaped_pipe_offsets(next_line);
            if next_pipes.len() >= 2 {
                let c_slice = &next_line[next_pipes[0] + 1..next_pipes[1]];
                *self.selection =
                    Selection::caret(probe_start + next_pipes[0] + 1 + table_cell_caret_offset(c_slice));
                return;
            }
            break;
        }

        // Bottom-right cell of table: append a new empty row (PRD 4.1.1 G #6)
        let nl = detect_line_ending_str(&full, self.preferred_line_ending);
        let insert_pos = table_row_insert_pos(&full, line_start, line_end);
        let empty_row = build_empty_table_row(cols);
        let insertion = format!("{nl}{empty_row}");
        let sel_before = *self.selection;
        let sel_after = Selection::caret(insert_pos + nl.len() + 2); // inside `| ‸ |` of new row
        self.undo.apply_edit(
            self.buffer,
            ByteRange::empty(insert_pos),
            &insertion,
            sel_before,
            sel_after,
            EditKind::Structural,
        );
        *self.selection = sel_after;
    }

    fn has_adjacent_table_row(&self, line_start: ByteOffset, line_end: ByteOffset) -> bool {
        let full = self.buffer.to_string();
        if line_start > 0 {
            let before = &full[..line_start - 1];
            let prev_start = before.rfind('\n').map(|i| i + 1).unwrap_or(0);
            let prev_line = before[prev_start..].trim_end_matches('\r').trim();
            if prev_line.starts_with('|') && prev_line.ends_with('|') {
                return true;
            }
        }
        if line_end < full.len() {
            let after = &full[line_end + 1..];
            let next_len = after.find('\n').unwrap_or(after.len());
            let next_line = after[..next_len].trim_end_matches('\r').trim();
            if next_line.starts_with('|') && next_line.ends_with('|') {
                return true;
            }
        }
        false
    }

    fn enclosing_fenced_code_block_body_range(
        &self,
        target_line_start: ByteOffset,
    ) -> Option<ByteRange> {
        let full = self.buffer.to_string();
        let bytes = full.as_bytes();
        let mut pos = 0usize;
        let mut active_fence: Option<(u8, usize, ByteOffset)> = None;
        let mut passed_target = false;

        while pos < bytes.len() {
            let l_start = pos;
            let mut l_end = pos;
            while l_end < bytes.len() && bytes[l_end] != b'\n' {
                l_end += 1;
            }
            let next_pos = if l_end < bytes.len() { l_end + 1 } else { l_end };
            let line = full[l_start..l_end].trim_end_matches('\r');

            if l_start == target_line_start {
                if active_fence.is_none() {
                    return None;
                }
                passed_target = true;
            }

            if let Some((f_char, f_len, body_start)) = active_fence {
                if is_closing_code_fence(line, f_char, f_len) {
                    if passed_target {
                        let mut body_end = l_start.saturating_sub(1).max(body_start);
                        if body_end > body_start && bytes.get(body_end - 1) == Some(&b'\r') {
                            body_end -= 1;
                        }
                        return Some(ByteRange::new(body_start, body_end));
                    }
                    active_fence = None;
                }
            } else if let Some((f_char, f_len)) = parse_opening_code_fence(line) {
                active_fence = Some((f_char, f_len, next_pos));
            }
            pos = next_pos;
        }
        None
    }

    /// Return `(line_start, content_end_without_cr, lf_or_eof_end)` for the current line.
    fn current_line_content_bounds(&self) -> (ByteOffset, ByteOffset, ByteOffset) {
        let full = self.buffer.to_string();
        let bytes = full.as_bytes();
        let caret = self.selection.head.min(bytes.len());
        let mut start = caret;
        while start > 0 && bytes[start - 1] != b'\n' {
            start -= 1;
        }
        let mut lf_end = caret;
        while lf_end < bytes.len() && bytes[lf_end] != b'\n' {
            lf_end += 1;
        }
        let mut content_end = lf_end;
        while content_end > start && bytes[content_end - 1] == b'\r' {
            content_end -= 1;
        }
        (start, content_end, lf_end)
    }

    fn current_line_bounds(&self) -> (ByteOffset, ByteOffset) {
        let (start, _, lf_end) = self.current_line_content_bounds();
        (start, lf_end)
    }

    /// Returns `Some((fence_char, fence_len))` if the line starting at `target_line_start` is inside
    /// the body of an unclosed fenced code block (` ``` ` or `~~~`, tracking both fence character and
    /// minimum opening length per CommonMark spec).
    fn enclosing_fenced_code_block(&self, target_line_start: ByteOffset) -> Option<(u8, usize)> {
        let full = self.buffer.to_string();
        let bytes = full.as_bytes();
        let mut pos = 0usize;
        let mut active_fence: Option<(u8, usize)> = None;

        while pos < bytes.len() {
            let l_start = pos;
            let mut l_end = pos;
            while l_end < bytes.len() && bytes[l_end] != b'\n' {
                l_end += 1;
            }
            let next_pos = if l_end < bytes.len() { l_end + 1 } else { l_end };
            let line = full[l_start..l_end].trim_end_matches('\r');

            if l_start == target_line_start {
                return active_fence;
            }

            if let Some((f_char, f_len)) = active_fence {
                if is_closing_code_fence(line, f_char, f_len) {
                    active_fence = None;
                }
            } else if let Some((f_char, f_len)) = parse_opening_code_fence(line) {
                active_fence = Some((f_char, f_len));
            }
            pos = next_pos;
        }
        active_fence
    }

    fn is_offset_inside_fenced_code_block(&self, target_line_start: ByteOffset) -> bool {
        self.enclosing_fenced_code_block(target_line_start).is_some()
    }

    fn is_caret_in_table_row(&self) -> bool {
        let (start, end) = self.current_line_bounds();
        let line = self.buffer.slice_to_string(ByteRange::new(start, end));
        line.trim().starts_with('|') && line.trim().ends_with('|')
    }

    fn enclosing_empty_code_fence_range(&self, empty_line_start: ByteOffset) -> Option<ByteRange> {
        let full = self.buffer.to_string();
        if empty_line_start < 4 {
            return None;
        }
        let before = &full[..empty_line_start];
        let prev_line_start = before[..before.len() - 1]
            .rfind('\n')
            .map(|i| i + 1)
            .unwrap_or(0);
        let prev_line = full[prev_line_start..empty_line_start - 1].trim_end_matches('\r');
        let (f_char, f_len) = parse_opening_code_fence(prev_line)?;

        let after = &full[empty_line_start..];
        let (after_eol_len, after_rest) = if let Some(rest) = after.strip_prefix("\r\n") {
            (2, rest)
        } else if let Some(rest) = after.strip_prefix('\n') {
            (1, rest)
        } else {
            return None;
        };
        let next_line_rel_end = after_rest.find('\n').unwrap_or(after_rest.len());
        let next_line = after_rest[..next_line_rel_end].trim_end_matches('\r');
        if !is_closing_code_fence(next_line, f_char, f_len) {
            return None;
        }
        let close_end = empty_line_start + after_eol_len + next_line.len();
        Some(ByteRange::new(prev_line_start, close_end))
    }

    fn check_third_enter_at_code_fence_end(
        &self,
        caret: ByteOffset,
    ) -> Option<(ByteRange, ByteOffset)> {
        let full = self.buffer.to_string();
        // Caret must be on the 2nd consecutive empty line right before a valid closing code fence line
        let trailing_empty_bytes = if full[..caret].ends_with("\r\n\r\n") {
            4
        } else if full[..caret].ends_with("\n\n") {
            2
        } else {
            return None;
        };
        let (f_char, f_len) = self.enclosing_fenced_code_block(caret)?;
        let after_full = full.get(caret..)?;
        let (after_eol_len, after) = if let Some(rest) = after_full.strip_prefix("\r\n") {
            (2, rest)
        } else if let Some(rest) = after_full.strip_prefix('\n') {
            (1, rest)
        } else {
            return None;
        };
        let close_line_len = after.find('\n').unwrap_or(after.len());
        let close_line = after[..close_line_len].trim_end_matches('\r');
        if !is_closing_code_fence(close_line, f_char, f_len) {
            return None;
        }
        let closing_fence_end = caret + after_eol_len + close_line.len();
        Some((ByteRange::new(caret - trailing_empty_bytes, caret), closing_fence_end))
    }
}

/// Map a UTF-8 byte offset across an external file reload (`old_text` -> `new_text`)
/// using longest common prefix/suffix so the cursor stays anchored near the same text (PRD 4.1.1 E).
pub fn map_offset_across_reload(
    old_text: &str,
    new_text: &str,
    old_offset: ByteOffset,
) -> ByteOffset {
    let mut clamped_old = old_offset.min(old_text.len());
    while clamped_old > 0 && !old_text.is_char_boundary(clamped_old) {
        clamped_old -= 1;
    }

    let mut prefix_bytes = 0usize;
    for (oc, nc) in old_text.chars().zip(new_text.chars()) {
        if oc != nc {
            break;
        }
        prefix_bytes += oc.len_utf8();
    }

    if clamped_old <= prefix_bytes {
        return clamped_old;
    }

    let old_rem = &old_text[prefix_bytes..];
    let new_rem = &new_text[prefix_bytes..];
    let mut suffix_bytes = 0usize;
    for (oc, nc) in old_rem.chars().rev().zip(new_rem.chars().rev()) {
        if oc != nc {
            break;
        }
        suffix_bytes += oc.len_utf8();
    }

    let old_suffix_start = old_text.len() - suffix_bytes;
    let new_suffix_start = new_text.len() - suffix_bytes;
    if clamped_old >= old_suffix_start {
        new_suffix_start + (clamped_old - old_suffix_start)
    } else {
        new_suffix_start
    }
}

/// Parse a CommonMark opening code fence (` ``` ` or `~~~` with up to 3 leading spaces, length >= 3).
fn parse_opening_code_fence(line: &str) -> Option<(u8, usize)> {
    let bytes = line.as_bytes();
    let mut indent = 0usize;
    while indent < bytes.len() && bytes[indent] == b' ' && indent < 4 {
        indent += 1;
    }
    if indent >= 4 || indent >= bytes.len() {
        return None;
    }
    let ch = bytes[indent];
    if ch != b'`' && ch != b'~' {
        return None;
    }
    let mut run = 0usize;
    while indent + run < bytes.len() && bytes[indent + run] == ch {
        run += 1;
    }
    if run < 3 {
        return None;
    }
    // Info string of backtick fence cannot contain backticks (CommonMark spec)
    let info = &line[indent + run..];
    if ch == b'`' && info.contains('`') {
        return None;
    }
    Some((ch, run))
}

/// Check if `line` is a valid closing fence matching `(fence_char, min_len)`.
fn is_closing_code_fence(line: &str, fence_char: u8, min_len: usize) -> bool {
    let bytes = line.as_bytes();
    let mut indent = 0usize;
    while indent < bytes.len() && bytes[indent] == b' ' && indent < 4 {
        indent += 1;
    }
    if indent >= 4 || indent >= bytes.len() {
        return false;
    }
    let mut run = 0usize;
    while indent + run < bytes.len() && bytes[indent + run] == fence_char {
        run += 1;
    }
    if run < min_len {
        return false;
    }
    line[indent + run..].trim().is_empty()
}

fn is_single_byte_emphasis_marker(marker: &str) -> bool {
    matches!(marker, "*" | "_")
}

fn count_consecutive_byte_left(bytes: &[u8], idx: usize, target: u8) -> usize {
    let mut count = 0;
    let mut pos = idx;
    while pos > 0 && bytes[pos - 1] == target {
        count += 1;
        pos -= 1;
    }
    count
}

fn count_consecutive_byte_right(bytes: &[u8], idx: usize, target: u8) -> usize {
    let mut count = 0;
    let mut pos = idx;
    while pos < bytes.len() && bytes[pos] == target {
        count += 1;
        pos += 1;
    }
    count
}

fn table_cell_caret_offset(cell_slice: &str) -> usize {
    match cell_slice.chars().next() {
        Some(ch) if ch.is_whitespace() => ch.len_utf8(),
        _ => 0,
    }
}

fn detect_line_ending_str(full: &str, fallback: LineEnding) -> &'static str {
    match LineEnding::detect_dominant_or(full, fallback) {
        LineEnding::CrLf => "\r\n",
        LineEnding::Lf | LineEnding::Cr => "\n",
    }
}

fn table_row_insert_pos(full: &str, line_start: ByteOffset, line_end: ByteOffset) -> ByteOffset {
    if line_end > line_start && full.as_bytes().get(line_end - 1) == Some(&b'\r') {
        line_end - 1
    } else {
        line_end
    }
}

/// Relative escape offset for PRD 4.1.1 B.1 block-line space triggers (`# `, `> `, `- `, `1. `, `- [ ] `).
fn block_line_syntax_escape_offset(line: &str) -> Option<ByteOffset> {
    if is_just_heading_trigger(line) {
        return Some(0);
    }
    if let Some((indent, marker_len)) = list_prefix_parts(line) {
        if line.len() == indent + marker_len {
            let marker = &line[indent..];
            if marker.starts_with("- [") {
                // `- [ ] ` -> `- \[ ] ` (escape before `[` per PRD 4.1.1 B.1)
                return Some(indent + 2);
            }
            if matches!(marker, "- " | "* " | "+ ") {
                return Some(indent);
            }
            if let Some(dot_rel) = marker.find(". ") {
                // `1. ` -> `1\. `
                return Some(indent + dot_rel);
            }
        }
    }
    blockquote_trigger_offset(line)
}

fn blockquote_trigger_offset(line: &str) -> Option<ByteOffset> {
    let bytes = line.as_bytes();
    let mut pos = 0usize;
    while pos < bytes.len() && bytes[pos] == b' ' && pos < 3 {
        pos += 1;
    }
    if pos >= bytes.len() {
        return None;
    }
    let mut last_quote = None;
    while pos < bytes.len() && line[pos..].starts_with("> ") {
        last_quote = Some(pos);
        pos += 2;
    }
    if pos == line.len() {
        last_quote
    } else {
        None
    }
}

/// Parse single or nested blockquote line prefix (`> `, `> > `, optional <=3 leading spaces).
/// Returns `(content_start_in_line, strip_rel_range)` where deleting `strip_rel_range`
/// demotes one blockquote level.
fn blockquote_demote_parts(line: &str) -> Option<(usize, ByteRange)> {
    let bytes = line.as_bytes();
    let mut indent = 0usize;
    while indent < bytes.len() && bytes[indent] == b' ' && indent < 4 {
        indent += 1;
    }
    if indent >= 4 || indent >= bytes.len() || bytes[indent] != b'>' {
        return None;
    }
    let mut pos = indent;
    let mut level_count = 0usize;
    let mut last_level_start = indent;
    while pos < bytes.len() && bytes[pos] == b'>' {
        let level_start = pos;
        pos += 1;
        if pos < bytes.len() && bytes[pos] == b' ' {
            pos += 1;
        }
        level_count += 1;
        last_level_start = level_start;
    }
    if level_count == 1 {
        Some((pos, ByteRange::new(0, pos)))
    } else {
        Some((pos, ByteRange::new(last_level_start, pos)))
    }
}

/// Check if `line` is a CommonMark thematic break (`---`, `***`, `___` with optional spaces, count >= 3).
fn thematic_break_escape_offset(line: &str) -> Option<ByteOffset> {
    let bytes = line.as_bytes();
    let mut indent = 0usize;
    while indent < bytes.len() && bytes[indent] == b' ' && indent < 4 {
        indent += 1;
    }
    if indent >= 4 || indent >= bytes.len() {
        return None;
    }
    let marker = bytes[indent];
    if !matches!(marker, b'-' | b'*' | b'_') {
        return None;
    }
    let mut count = 0usize;
    for &b in &bytes[indent..] {
        if b == marker {
            count += 1;
        } else if b != b' ' && b != b'\t' {
            return None;
        }
    }
    if count >= 3 {
        Some(indent)
    } else {
        None
    }
}

/// Determine whether a newly typed opening fence line at `line_end` already pairs
/// with an existing unpaired closing fence below it.
fn has_unpaired_closing_fence_below(
    full_text: &str,
    line_end: ByteOffset,
    fence_ch: u8,
    fence_len: usize,
) -> bool {
    let mut rest = &full_text[line_end.min(full_text.len())..];
    if let Some(after) = rest.strip_prefix("\r\n") {
        rest = after;
    } else if let Some(after) = rest.strip_prefix('\n') {
        rest = after;
    } else {
        return false;
    }

    let mut inner: Option<(u8, usize)> = None;
    let mut bare_closers = 0usize;
    for raw_line in rest.split('\n') {
        let line = raw_line.trim_end_matches('\r');
        if let Some((ich, ilen)) = inner {
            if is_closing_code_fence(line, ich, ilen) {
                inner = None;
            }
            continue;
        }
        if is_closing_code_fence(line, fence_ch, fence_len) {
            bare_closers += 1;
        } else if let Some((och, olen)) = parse_opening_code_fence(line) {
            // Only enter a nested balanced block if its delimiter could validly exist outside
            // or has an info string (so shorter literal ``` lines inside ``` ` ```` ` don't swallow ` ```` `).
            let trimmed_start = line.trim_start_matches(' ');
            let info_non_empty = !trimmed_start[olen..].trim().is_empty();
            if info_non_empty || och != fence_ch || olen >= fence_len {
                inner = Some((och, olen));
            }
        }
    }
    bare_closers % 2 == 1
}

/// Detect whether typing `typed_char` ending at `caret_in_line` just closed a paired
/// inline formatting span (`**bold**`, `*italic*`, `` `code` ``, `~~strike~~`),
/// returning the relative byte offsets inside `line` that 1st Undo should escape.
fn detect_inline_conversion_escapes(
    line: &str,
    caret_in_line: usize,
    typed_char: char,
) -> Vec<ByteOffset> {
    if caret_in_line == 0
        || caret_in_line > line.len()
        || !line.is_char_boundary(caret_in_line)
        || line[caret_in_line..].starts_with(typed_char)
    {
        return Vec::new();
    }

    let prefix = &line[..caret_in_line];
    let bytes = prefix.as_bytes();
    let mut i = 0usize;
    let mut open_backtick: Option<(usize, usize)> = None;
    let mut open_single_star: Option<usize> = None;
    let mut open_double_star: Option<usize> = None;
    let mut open_double_tilde: Option<usize> = None;

    while i < bytes.len() {
        let b = bytes[i];
        if open_backtick.is_some() {
            if b == b'`' {
                let mut run = 0usize;
                while i + run < bytes.len() && bytes[i + run] == b'`' {
                    run += 1;
                }
                if let Some((open, open_len)) = open_backtick {
                    if run == open_len {
                        let inner = &prefix[open + run..i];
                        let is_fence = parse_opening_code_fence(line).is_some() && open <= 3 && run >= 3;
                        if i + run == caret_in_line
                            && typed_char == '`'
                            && !is_fence
                            && !inner.is_empty()
                            && inner.chars().any(|c| !c.is_whitespace())
                        {
                            return vec![open];
                        }
                        open_backtick = None;
                    }
                }
                i += run;
                continue;
            }
            let ch = prefix[i..].chars().next().unwrap();
            i += ch.len_utf8();
            continue;
        }

        if b == b'\\' {
            if i + 1 < bytes.len() {
                let next_ch = prefix[i + 1..].chars().next().unwrap();
                i += 1 + next_ch.len_utf8();
            } else {
                i += 1;
            }
            continue;
        }

        if b == b'`' {
            let mut run = 0usize;
            while i + run < bytes.len() && bytes[i + run] == b'`' {
                run += 1;
            }
            open_backtick = Some((i, run));
            i += run;
            continue;
        }

        if b == b'*' {
            let mut run = 0usize;
            while i + run < bytes.len() && bytes[i + run] == b'*' {
                run += 1;
            }
            if run == 2 {
                if let Some(open) = open_double_star {
                    let inner = &prefix[open + 2..i];
                    if !inner.is_empty()
                        && !inner.starts_with(char::is_whitespace)
                        && !inner.ends_with(char::is_whitespace)
                    {
                        if i + run == caret_in_line && typed_char == '*' {
                            return vec![open, open + 1];
                        }
                        open_double_star = None;
                    } else {
                        open_double_star = Some(i);
                    }
                } else {
                    open_double_star = Some(i);
                }
            } else if run == 1 {
                if let Some(open) = open_single_star {
                    let inner = &prefix[open + 1..i];
                    if !inner.is_empty()
                        && !inner.starts_with(char::is_whitespace)
                        && !inner.ends_with(char::is_whitespace)
                    {
                        if i + run == caret_in_line && typed_char == '*' {
                            return vec![open];
                        }
                        open_single_star = None;
                    } else {
                        open_single_star = Some(i);
                    }
                } else {
                    open_single_star = Some(i);
                }
            }
            i += run;
            continue;
        }

        if b == b'~' {
            let mut run = 0usize;
            while i + run < bytes.len() && bytes[i + run] == b'~' {
                run += 1;
            }
            if run == 2 {
                if let Some(open) = open_double_tilde {
                    let inner = &prefix[open + 2..i];
                    if !inner.is_empty()
                        && !inner.starts_with(char::is_whitespace)
                        && !inner.ends_with(char::is_whitespace)
                    {
                        if i + run == caret_in_line && typed_char == '~' {
                            return vec![open, open + 1];
                        }
                        open_double_tilde = None;
                    } else {
                        open_double_tilde = Some(i);
                    }
                } else {
                    open_double_tilde = Some(i);
                }
            }
            i += run;
            continue;
        }

        let ch = prefix[i..].chars().next().unwrap();
        i += ch.len_utf8();
    }

    Vec::new()
}

fn is_just_heading_trigger(line: &str) -> bool {
    matches!(
        line,
        "# " | "## " | "### " | "#### " | "##### " | "###### "
    )
}

fn heading_prefix_len(line: &str) -> Option<usize> {
    let bytes = line.as_bytes();
    let mut h = 0usize;
    while h < bytes.len() && bytes[h] == b'#' && h < 6 {
        h += 1;
    }
    if h > 0 && h < bytes.len() && bytes[h] == b' ' {
        Some(h + 1)
    } else {
        None
    }
}

fn list_prefix_parts(line: &str) -> Option<(usize, usize)> {
    let bytes = line.as_bytes();
    let mut indent = 0usize;
    while indent < bytes.len() && bytes[indent] == b' ' {
        indent += 1;
    }
    let rest = &line[indent..];
    for marker in ["- [ ] ", "- [x] ", "- [X] ", "- ", "* ", "+ "] {
        if rest.starts_with(marker) {
            return Some((indent, marker.len()));
        }
    }
    // Ordered list `1. `
    let mut digits = 0usize;
    while digits < rest.len() && rest.as_bytes()[digits].is_ascii_digit() {
        digits += 1;
    }
    if digits > 0 && rest[digits..].starts_with(". ") {
        return Some((indent, digits + 2));
    }
    None
}

fn unescaped_pipe_offsets(line: &str) -> Vec<usize> {
    let bytes = line.as_bytes();
    let mut pipes = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'\\' && i + 1 < bytes.len() {
            i += 2;
            continue;
        }
        if bytes[i] == b'|' {
            pipes.push(i);
        }
        i += 1;
    }
    pipes
}

fn current_table_cell_index(pipes: &[usize], rel_caret: usize) -> Option<usize> {
    if pipes.len() < 2 {
        return None;
    }
    for idx in 0..pipes.len() - 1 {
        if rel_caret > pipes[idx] && rel_caret <= pipes[idx + 1] {
            return Some(idx);
        }
    }
    if rel_caret <= pipes[0] {
        Some(0)
    } else {
        Some(pipes.len() - 2)
    }
}

fn build_empty_table_row(cols: usize) -> String {
    let mut out = String::with_capacity(cols * 3 + 1);
    out.push('|');
    for _ in 0..cols.max(1) {
        out.push_str("  |");
    }
    out
}

fn build_table_separator_row(cols: usize) -> String {
    let mut out = String::with_capacity(cols * 6 + 1);
    out.push('|');
    for _ in 0..cols.max(1) {
        out.push_str(" --- |");
    }
    out
}

fn is_table_delimiter_row(line: &str) -> bool {
    let pipes = unescaped_pipe_offsets(line);
    if pipes.len() < 2 {
        return false;
    }
    for idx in 0..pipes.len() - 1 {
        let cell = line[pipes[idx] + 1..pipes[idx + 1]].trim();
        let stripped = cell.trim_start_matches(':').trim_end_matches(':');
        if stripped.len() < 3 || !stripped.bytes().all(|b| b == b'-') {
            return false;
        }
    }
    true
}

fn find_enclosing_table_range(source: &str, current_line_start: ByteOffset) -> Option<ByteRange> {
    let bytes = source.as_bytes();
    let mut lines: Vec<(ByteOffset, ByteOffset)> = Vec::new();
    let mut pos = 0usize;
    let mut current_idx = None;

    while pos < bytes.len() {
        let l_start = pos;
        let mut l_end = pos;
        while l_end < bytes.len() && bytes[l_end] != b'\n' {
            l_end += 1;
        }
        if l_start == current_line_start {
            current_idx = Some(lines.len());
        }
        lines.push((l_start, l_end));
        pos = if l_end < bytes.len() { l_end + 1 } else { l_end };
    }

    let idx = current_idx?;
    let is_tbl_line = |(s, e): (ByteOffset, ByteOffset)| -> bool {
        let l = source[s..e].trim_end_matches('\r').trim();
        l.starts_with('|') && l.ends_with('|') && unescaped_pipe_offsets(l).len() >= 2
    };
    if !is_tbl_line(lines[idx]) {
        return None;
    }

    let mut first = idx;
    while first > 0 && is_tbl_line(lines[first - 1]) {
        first -= 1;
    }
    let mut last = idx;
    while last + 1 < lines.len() && is_tbl_line(lines[last + 1]) {
        last += 1;
    }
    Some(ByteRange::new(lines[first].0, lines[last].1))
}

fn format_markdown_table(raw_table: &str) -> String {
    let raw_lines: Vec<&str> = raw_table.split('\n').collect();
    let mut parsed_rows: Vec<(bool, Vec<String>)> = Vec::with_capacity(raw_lines.len());
    let mut max_cols = 0usize;

    for raw_line in &raw_lines {
        let trimmed_cr = raw_line.trim_end_matches('\r');
        let pipes = unescaped_pipe_offsets(trimmed_cr);
        let is_delim = is_table_delimiter_row(trimmed_cr);
        let mut cells = Vec::new();
        for i in 0..pipes.len().saturating_sub(1) {
            let cell = trimmed_cr[pipes[i] + 1..pipes[i + 1]].trim().to_string();
            cells.push(cell);
        }
        max_cols = max_cols.max(cells.len());
        parsed_rows.push((is_delim, cells));
    }

    if max_cols == 0 {
        return raw_table.to_string();
    }

    let mut col_widths = vec![3usize; max_cols];
    for (is_delim, cells) in &parsed_rows {
        if *is_delim {
            continue;
        }
        for (col, cell) in cells.iter().enumerate() {
            col_widths[col] = col_widths[col].max(cell.chars().count());
        }
    }

    let mut formatted_lines = Vec::with_capacity(raw_lines.len());
    for (row_idx, (is_delim, cells)) in parsed_rows.iter().enumerate() {
        let has_cr = raw_lines[row_idx].ends_with('\r');
        let mut row_str = String::from("|");
        for col in 0..max_cols {
            let width = col_widths[col];
            let cell = cells.get(col).map(|s| s.as_str()).unwrap_or("");
            if *is_delim {
                let left_colon = cell.starts_with(':');
                let right_colon = cell.ends_with(':');
                let dash_count = width
                    .saturating_sub(usize::from(left_colon) + usize::from(right_colon))
                    .max(1);
                row_str.push(' ');
                if left_colon {
                    row_str.push(':');
                }
                for _ in 0..dash_count {
                    row_str.push('-');
                }
                if right_colon {
                    row_str.push(':');
                }
                row_str.push_str(" |");
            } else {
                let char_len = cell.chars().count();
                let pad = width.saturating_sub(char_len);
                row_str.push(' ');
                row_str.push_str(cell);
                for _ in 0..pad {
                    row_str.push(' ');
                }
                row_str.push_str(" |");
            }
        }
        if has_cr {
            row_str.push('\r');
        }
        formatted_lines.push(row_str);
    }

    formatted_lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prd_g_case_2_heading_backspace_demotes_then_merges() {
        // PRD 4.1.1 G #2: Caret at start of `## 标题` -> 1st Backspace demotes to `标题`;
        // 2nd Backspace merges with previous paragraph `上一段`.
        let mut ed = EditorController::from_str("上一段\n## 标题");
        ed.set_caret("上一段\n## ".len());
        ed.backspace();
        assert_eq!(ed.buffer.to_string(), "上一段\n标题");
        assert_eq!(ed.selection, Selection::caret("上一段\n".len()));

        // 2nd Backspace merges with "上一段"
        ed.backspace();
        assert_eq!(ed.buffer.to_string(), "上一段标题");
        assert_eq!(ed.selection, Selection::caret("上一段".len()));
    }

    #[test]
    fn test_prd_g_case_4_and_5_nested_list_shift_tab_and_empty_item_enter() {
        // PRD 4.1.1 G #4: Nested list `  - 子项` -> Shift+Tab -> `- 子项`
        let mut ed = EditorController::from_str("- 父项\n  - 子项");
        ed.set_caret("- 父项\n  - ".len());
        ed.shift_tab();
        assert_eq!(ed.buffer.to_string(), "- 父项\n- 子项");

        // PRD 4.1.1 G #5: Empty top-level list item `- ‸` -> Enter -> exits list to empty paragraph
        let mut ed2 = EditorController::from_str("- 第一项\n- ");
        ed2.set_caret(ed2.buffer.len_bytes());
        ed2.enter();
        assert_eq!(ed2.buffer.to_string(), "- 第一项\n");
    }

    #[test]
    fn test_prd_g_case_6_table_pipe_input_is_escaped() {
        // PRD 4.1.1 G #6: Typing `|` inside a table cell writes `\|` to source
        let mut ed = EditorController::from_str("| a | b |");
        ed.set_caret(3); // inside first cell
        ed.insert_char('|');
        assert_eq!(ed.buffer.to_string(), "| a\\| | b |");
    }

    #[test]
    fn test_prd_g_case_7_and_8_code_fence_exit_and_empty_backspace() {
        // PRD 4.1.1 G #7 & P2 #8: 3 consecutive Enters at end of code block exits code block,
        // and a SINGLE `undo()` restores the exact pre-3rd-Enter state!
        let mut ed = EditorController::from_str("```rust\nlet x = 1;\n```");
        ed.set_caret("```rust\nlet x = 1;".len());
        ed.enter(); // 1st Enter: inserts empty line
        ed.enter(); // 2nd Enter: inserts 2nd empty line
        let before_third_enter = ed.buffer.to_string();

        ed.enter(); // 3rd Enter: strips trailing empty lines and exits code fence in 1 transaction
        assert_eq!(ed.buffer.to_string(), "```rust\nlet x = 1;\n```\n");
        assert_eq!(ed.selection, Selection::caret(ed.buffer.len_bytes()));

        // Single undo must restore `before_third_enter`
        ed.undo();
        assert_eq!(ed.buffer.to_string(), before_third_enter);

        // PRD 4.1.1 G #8: Backspace at start of empty code block converts to empty paragraph
        let mut ed2 = EditorController::from_str("```\n\n```");
        ed2.set_caret(4); // on the empty line inside ```\n‸\n```
        ed2.backspace();
        assert_eq!(ed2.buffer.to_string(), "");
    }

    #[test]
    fn test_prd_g_case_9_10_11_ime_composition_and_mode_switch_offset_invariant() {
        // PRD 4.1.1 G #9 & #10: IME preedit does NOT modify source buffer until commit;
        // commit is a single undo unit.
        let mut ed = EditorController::from_str("## ");
        ed.set_caret(3);
        let rev_before = ed.buffer.revision();

        ed.ime_update("biao'ti");
        assert_eq!(ed.buffer.to_string(), "## ");
        assert_eq!(ed.buffer.revision(), rev_before);
        assert!(ed.expansion_frozen);

        ed.ime_commit("标题");
        assert_eq!(ed.buffer.to_string(), "## 标题");
        assert!(!ed.expansion_frozen);

        // Single undo removes the entire committed CJK phrase "标题"
        ed.undo();
        assert_eq!(ed.buffer.to_string(), "## ");

        // PRD 4.1.1 G #11: Toggle WYSIWYG <-> Source mode preserves exact byte offset
        ed.ime_commit("标题");
        ed.set_caret(6);
        ed.toggle_mode();
        assert_eq!(ed.mode, ViewMode::Source);
        assert_eq!(ed.selection, Selection::caret(6));
        ed.toggle_mode();
        assert_eq!(ed.mode, ViewMode::Wysiwyg);
        assert_eq!(ed.selection, Selection::caret(6));
    }

    #[test]
    fn test_p1_opening_existing_hashes_typing_space_and_undoing_twice_restores_original() {
        // P1 #1: Open `##`, type space at end, undo twice -> must restore `##` (not `\#`).
        let mut ed = EditorController::from_str("##");
        ed.set_caret(2);
        ed.insert_char(' ');
        assert_eq!(ed.buffer.to_string(), "## ");

        ed.undo();
        assert_eq!(ed.buffer.to_string(), "\\## ");

        ed.undo();
        assert_eq!(ed.buffer.to_string(), "##");
        assert_eq!(ed.selection, Selection::caret(2));
    }

    #[test]
    fn test_p1_ime_update_then_toggle_mode_does_not_write_raw_preedit_before_commit() {
        // P1 #2: `ime_update("ni")` followed by `toggle_mode()` must NOT write `"ni"` into buffer,
        // so a subsequent `ime_commit("你")` produces `"你"` (not `"ni你"`).
        let mut ed = EditorController::new();
        ed.ime_update("ni");
        ed.toggle_mode();
        assert_eq!(ed.buffer.to_string(), "");
        ed.ime_commit("你");
        assert_eq!(ed.buffer.to_string(), "你");
    }

    #[test]
    fn test_p2_enter_replaces_non_empty_selection() {
        // P2 #5: Selecting "world" in "hello world" and pressing Enter must replace "world" with "\n".
        let mut ed = EditorController::from_str("hello world");
        ed.set_selection(6, 11);
        ed.enter();
        assert_eq!(ed.buffer.to_string(), "hello \n");
        assert_eq!(ed.selection, Selection::caret(7));
    }

    #[test]
    fn test_p2_hash_comment_inside_code_fence_is_not_treated_as_heading_on_backspace() {
        // P2 #6: Inside a fenced code block, `# comment` is a code comment; Backspace after `# `
        // must only delete the space `' '` (leaving `#comment`), NOT the entire `# ` heading prefix.
        let mut ed = EditorController::from_str("```sh\n# comment\n```");
        ed.set_caret("```sh\n# ".len());
        ed.backspace();
        assert_eq!(ed.buffer.to_string(), "```sh\n#comment\n```");
    }

    #[test]
    fn test_p2_source_mode_does_not_record_wysiwyg_syntax_conversion_step() {
        // P2 #7: In Source mode, typing `# ` and pressing Undo once must directly undo the typed characters,
        // NOT insert a backslash `\# `.
        let mut ed = EditorController::new();
        ed.toggle_mode();
        assert_eq!(ed.mode, ViewMode::Source);
        ed.insert_char('#');
        ed.insert_char(' ');
        assert_eq!(ed.buffer.to_string(), "# ");

        ed.undo();
        assert_eq!(ed.buffer.to_string(), "");
    }

    #[test]
    fn test_p1_ime_commit_after_cursor_move_commits_at_original_composition_position() {
        // P1 #1: Start composing at `A|B` (offset 1), move caret to end (offset 2), then commit "你" ->
        // must produce `A你B` (not `AB你`).
        let mut ed = EditorController::from_str("AB");
        ed.set_caret(1);
        ed.ime_update("ni");
        ed.set_caret(2);
        ed.ime_clear_preedit();
        ed.ime_commit("你");
        assert_eq!(ed.buffer.to_string(), "A你B");
    }

    #[test]
    fn test_p2_tilde_fence_and_four_backtick_fence_containing_three_backticks() {
        // P2 #2: Inside `~~~` fence, `# comment` is code; and inside ` ```` ` fence, an inner ` ``` `
        // does not close the fence, so `# comment` after ` ``` ` is still inside the code fence.
        let mut ed_tilde = EditorController::from_str("~~~\n# comment\n~~~");
        ed_tilde.set_caret("~~~\n# ".len());
        ed_tilde.backspace();
        assert_eq!(ed_tilde.buffer.to_string(), "~~~\n#comment\n~~~");

        let mut ed_quad = EditorController::from_str("````md\n```\n# comment\n````");
        ed_quad.set_caret("````md\n```\n# ".len());
        ed_quad.backspace();
        assert_eq!(
            ed_quad.buffer.to_string(),
            "````md\n```\n#comment\n````"
        );
    }

    #[test]
    fn test_three_enters_before_inner_three_backticks_in_four_backtick_fence_does_not_exit() {
        // Inside a 4-backtick code fence (` ````md `), an inner ` ``` ` line is content, not a closing fence.
        // Pressing Enter 3 times right before ` ``` ` must simply insert 3 newlines and NOT exit or delete blank lines.
        let mut ed = EditorController::from_str("````md\nline\n```\n````");
        ed.set_caret("````md\nline".len());
        ed.enter();
        ed.enter();
        ed.enter();
        assert_eq!(ed.buffer.to_string(), "````md\nline\n\n\n\n```\n````");
        assert_eq!(ed.selection, Selection::caret("````md\nline\n\n\n".len()));
    }

    #[test]
    fn test_ime_commit_and_undo_preserves_backward_selection_direction() {
        // When a backward selection (`anchor > head`) is replaced by IME commit and then undone,
        // both the original text and the exact backward `Selection` (`anchor > head`) must be restored.
        let mut ed = EditorController::from_str("hello world");
        ed.set_selection(5, 0); // backward selection of "hello" (anchor=5, head=0)
        ed.ime_update("ni");
        ed.ime_commit("你");
        assert_eq!(ed.buffer.to_string(), "你 world");

        ed.undo();
        assert_eq!(ed.buffer.to_string(), "hello world");
        assert_eq!(ed.selection, Selection::new(5, 0));
    }

    #[test]
    fn test_prd_t01_t02_t03_table_creation_tab_navigation_and_alignment() {
        // PRD 4.3 T-01: Typing `| a | b |` and pressing Enter creates separator + empty row
        let mut ed = EditorController::from_str("| a | b |");
        ed.set_caret(ed.buffer.len_bytes());
        ed.enter();
        assert_eq!(ed.buffer.to_string(), "| a | b |\n| --- | --- |\n|  |  |");

        // PRD 4.1.1 G #6 & 4.3 T-02: Tab in 1st cell jumps to 2nd cell; Tab in last cell appends a new row
        ed.insert_char('x');
        ed.tab(); // jump to 2nd cell of row 1
        ed.insert_char('y');
        ed.tab(); // at last cell of table -> appends new row and places caret in its 1st cell
        ed.insert_char('z');

        // PRD 4.3 T-03: Align ONLY the active table as a single undoable transaction
        assert!(ed.align_current_table());
        assert_eq!(
            ed.buffer.to_string(),
            "| a   | b   |\n| --- | --- |\n| x   | y   |\n| z   |     |"
        );

        // Undo reverts the alignment step
        ed.undo();
        assert_eq!(
            ed.buffer.to_string(),
            "| a | b |\n| --- | --- |\n| x | y |\n| z |  |"
        );

        // Shift+Tab navigates backwards across table cells and skips delimiter row
        ed.shift_tab(); // from cell 0 of row 2 ("z") -> cell 1 of row 1 ("y")
        let row1_y_pos = "| a | b |\n| --- | --- |\n| x | ".len();
        assert_eq!(ed.selection, Selection::caret(row1_y_pos));

        // CJK table alignment keeps caret on valid UTF-8 boundary
        let mut cjk_ed = EditorController::from_str("| 名称 | 值 |\n| --- | --- |\n| 中文 | 1 |");
        cjk_ed.set_caret("| 名称 | ".len() + 1);
        assert!(cjk_ed.align_current_table());
        assert!(std::str::from_utf8(cjk_ed.buffer.to_string().as_bytes()).is_ok());
        let after_head = cjk_ed.selection.head;
        assert!(cjk_ed.buffer.to_string().is_char_boundary(after_head));
    }

    #[test]
    fn test_prd_4_1_1_d_progressive_select_all_block_then_document() {
        // PRD 4.1.1 D: 1st Select-All inside a code block selects only the code body;
        // 2nd Select-All selects the entire document.
        let doc = "# Title\n```rust\nfn main() {}\n```\nFooter";
        let mut ed = EditorController::from_str(doc);
        let code_body_start = "# Title\n```rust\n".len();
        let code_body_end = code_body_start + "fn main() {}".len();
        ed.set_caret(code_body_start + 3);

        // 1st Cmd/Ctrl+A -> selects code block body
        ed.select_all();
        assert_eq!(ed.selection.range(), ByteRange::new(code_body_start, code_body_end));

        // 2nd Cmd/Ctrl+A -> selects full document
        ed.select_all();
        assert_eq!(ed.selection.range(), ByteRange::new(0, doc.len()));
    }

    #[test]
    fn test_inline_wrap_heading_level_task_checkbox_and_reload_offset_mapping() {
        // Inline bold wrap & unwrap with backward selection preserved
        let mut ed = EditorController::from_str("hello world");
        ed.set_selection(11, 6); // backward selection of "world"
        ed.toggle_inline_wrap("**");
        assert_eq!(ed.buffer.to_string(), "hello **world**");
        assert_eq!(ed.selection, Selection::new(13, 8));

        ed.toggle_inline_wrap("**");
        assert_eq!(ed.buffer.to_string(), "hello world");
        assert_eq!(ed.selection, Selection::new(11, 6));

        // Heading level conversion (Cmd+2 / Cmd+0) and task checkbox toggle
        let mut ed2 = EditorController::from_str("Title\n- [ ] Todo item");
        ed2.set_caret(2);
        ed2.set_heading_level(2);
        assert_eq!(ed2.buffer.to_string(), "## Title\n- [ ] Todo item");
        ed2.set_heading_level(0);
        assert_eq!(ed2.buffer.to_string(), "Title\n- [ ] Todo item");

        ed2.set_caret("Title\n- [ ] To".len());
        assert!(ed2.toggle_task_checkbox());
        assert_eq!(ed2.buffer.to_string(), "Title\n- [x] Todo item");

        // PRD 4.1.1 E: External reload preserves cursor position in unchanged suffix
        let old = "# Header\nKeep my cursor here!";
        let new = "# Header\nInserted externally!\nKeep my cursor here!";
        let old_caret = "# Header\nKeep my ".len();
        let mapped = map_offset_across_reload(old, new, old_caret);
        assert_eq!(&new[mapped..], "cursor here!");
    }

    #[test]
    fn test_p1_table_navigation_with_fullwidth_space_keeps_utf8_char_boundary() {
        // P1 #6: Full-width space `\u{3000}` is 3 bytes in UTF-8; Tab/Shift+Tab must land on a valid
        // UTF-8 char boundary so a subsequent Backspace never panics.
        let mut ed = EditorController::from_str("| a |\u{3000}b |\n| --- | --- |\n| c |\u{3000}d |");
        ed.set_caret(2); // inside `| a|`
        ed.tab();
        let caret = ed.selection.head;
        let full = ed.buffer.to_string();
        assert!(full.is_char_boundary(caret));
        assert_eq!(caret, "| a |\u{3000}".len());

        // Backspace removes the full 3-byte full-width space cleanly without panic
        ed.backspace();
        assert_eq!(ed.buffer.to_string(), "| a |b |\n| --- | --- |\n| c |\u{3000}d |");

        // Also test jumping forward across delimiter row to row 2, then Shift+Tab backward
        let row2_first_cell = ed.buffer.to_string().find("| c |").unwrap() + 2;
        ed.set_caret(row2_first_cell);
        ed.tab();
        assert!(ed.buffer.to_string().is_char_boundary(ed.selection.head));
        ed.shift_tab();
        assert!(ed.buffer.to_string().is_char_boundary(ed.selection.head));
        ed.shift_tab(); // back to row 0 cell 1 (`| a |b |`)
        assert!(ed.buffer.to_string().is_char_boundary(ed.selection.head));
    }

    #[test]
    fn test_p2_toggling_italic_twice_on_bold_preserves_bold() {
        // P2 #1a: Selecting inner text of `**bold**` and toggling `*` twice must produce
        // `***bold***` -> `**bold**`, preserving original bold markers.
        let mut ed = EditorController::from_str("say **bold** now");
        let start = "say **".len();
        let end = start + "bold".len();
        ed.set_selection(start, end);

        ed.toggle_inline_wrap("*");
        assert_eq!(ed.buffer.to_string(), "say ***bold*** now");
        ed.toggle_inline_wrap("*");
        assert_eq!(ed.buffer.to_string(), "say **bold** now");
        assert_eq!(ed.selection, Selection::new(start, end));

        // P2 #1b: Selecting full `**bold**` (including markers) and toggling `*` twice also preserves bold
        let full_start = "say ".len();
        let full_end = "say **bold**".len();
        ed.set_selection(full_start, full_end);
        ed.toggle_inline_wrap("*");
        assert_eq!(ed.buffer.to_string(), "say ***bold*** now");
        ed.toggle_inline_wrap("*");
        assert_eq!(ed.buffer.to_string(), "say **bold** now");
        assert_eq!(ed.selection, Selection::new(full_start, full_end));
    }

    #[test]
    fn test_p2_table_new_rows_preserve_crlf_without_mixed_lf() {
        // P2 #2: CRLF document creating separator+row on standalone header (T-01), appending via Enter (T-02),
        // and appending via Tab at bottom-right cell must all insert `\r\n` without producing lone `\n`.
        let mut ed = EditorController::from_str("| a | b |\r\nAfter");
        ed.set_caret(3);
        ed.enter();
        assert_eq!(
            ed.buffer.to_string(),
            "| a | b |\r\n| --- | --- |\r\n|  |  |\r\nAfter"
        );
        // Append row at last row via Tab (with no trailing CRLF case too)
        let mut ed2 = EditorController::from_str("| a | b |\r\n| --- | --- |\r\n| 1 | 2 |");
        let second_col = ed2.buffer.to_string().rfind("2 |").unwrap();
        ed2.set_caret(second_col);
        ed2.tab();
        let out = ed2.buffer.to_string();
        assert_eq!(out, "| a | b |\r\n| --- | --- |\r\n| 1 | 2 |\r\n|  |  |");
        // Verify ZERO bare LF (`\n` not preceded by `\r`)
        for (idx, b) in out.bytes().enumerate() {
            if b == b'\n' {
                assert!(idx > 0 && out.as_bytes()[idx - 1] == b'\r', "found bare LF at {idx}");
            }
        }

        // Mixed line-ending document with LF majority + minority CRLF must pick dominant LF (`\n`)
        let mut mixed_lf_dom = EditorController::from_str(
            "l1\nl2\nl3\n| a | b |\r\n| --- | --- |\r\n| 1 | 2 |",
        );
        let col = mixed_lf_dom.buffer.to_string().rfind("2 |").unwrap();
        mixed_lf_dom.set_caret(col);
        mixed_lf_dom.tab();
        assert_eq!(
            mixed_lf_dom.buffer.to_string(),
            "l1\nl2\nl3\n| a | b |\r\n| --- | --- |\r\n| 1 | 2 |\n|  |  |"
        );

        let mut mixed_lf_enter =
            EditorController::from_str("l1\nl2\nl3\n| a | b |\r\nAfter");
        mixed_lf_enter.set_caret("l1\nl2\nl3\n| ".len());
        mixed_lf_enter.enter();
        assert_eq!(
            mixed_lf_enter.buffer.to_string(),
            "l1\nl2\nl3\n| a | b |\n| --- | --- |\n|  |  |\r\nAfter"
        );

        // Mixed line-ending document with CRLF majority + minority LF must pick dominant CRLF (`\r\n`)
        let mut mixed_crlf_dom = EditorController::from_str(
            "l1\r\nl2\r\nl3\r\n| a | b |\n| --- | --- |\n| 1 | 2 |",
        );
        let col2 = mixed_crlf_dom.buffer.to_string().rfind("2 |").unwrap();
        mixed_crlf_dom.set_caret(col2);
        mixed_crlf_dom.tab();
        assert_eq!(
            mixed_crlf_dom.buffer.to_string(),
            "l1\r\nl2\r\nl3\r\n| a | b |\n| --- | --- |\n| 1 | 2 |\r\n|  |  |"
        );
    }

    fn type_str(ed: &mut EditorController, s: &str) {
        for ch in s.chars() {
            ed.dispatch(EditorAction::InsertChar(ch));
        }
    }

    #[test]
    fn test_prd_4_1_1_b1_block_and_inline_triggers_and_escape_undo() {
        // 1. Blockquote `> ` -> `\> ` -> ``
        let mut q = EditorController::new();
        type_str(&mut q, "> ");
        assert_eq!(q.buffer.to_string(), "> ");
        q.dispatch(EditorAction::Undo);
        assert_eq!(q.buffer.to_string(), "\\> ");
        q.dispatch(EditorAction::Undo);
        assert_eq!(q.buffer.to_string(), "");

        // 2. Unordered list `- ` / `* ` / `+ ` -> `\- ` -> ``
        for marker in ["- ", "* ", "+ "] {
            let mut ul = EditorController::new();
            type_str(&mut ul, marker);
            ul.dispatch(EditorAction::Undo);
            assert_eq!(ul.buffer.to_string(), format!("\\{marker}"));
            ul.dispatch(EditorAction::Undo);
            assert_eq!(ul.buffer.to_string(), "");
        }

        // 3. Ordered list `1. ` -> `1\. ` -> ``
        let mut ol = EditorController::new();
        type_str(&mut ol, "1. ");
        ol.dispatch(EditorAction::Undo);
        assert_eq!(ol.buffer.to_string(), "1\\. ");
        ol.dispatch(EditorAction::Undo);
        assert_eq!(ol.buffer.to_string(), "");

        // 4. Task list `- [ ] ` -> `- \[ ] ` -> `- `
        let mut task = EditorController::new();
        type_str(&mut task, "- [ ] ");
        task.dispatch(EditorAction::Undo);
        assert_eq!(task.buffer.to_string(), "- \\[ ] ");
        task.dispatch(EditorAction::Undo);
        assert_eq!(task.buffer.to_string(), "- ");

        // 5. Inline bold `**粗体**` -> `\*\*粗体**` -> ``, plus Redo cycle & italic/code/strike
        let mut bold = EditorController::new();
        type_str(&mut bold, "**粗体**");
        assert_eq!(bold.buffer.to_string(), "**粗体**");
        bold.dispatch(EditorAction::Undo);
        assert_eq!(bold.buffer.to_string(), "\\*\\*粗体**");
        bold.dispatch(EditorAction::Redo);
        assert_eq!(bold.buffer.to_string(), "**粗体**");
        bold.dispatch(EditorAction::Undo);
        assert_eq!(bold.buffer.to_string(), "\\*\\*粗体**");
        bold.dispatch(EditorAction::Undo);
        assert_eq!(bold.buffer.to_string(), "");

        let mut italic = EditorController::new();
        type_str(&mut italic, "*斜体*");
        italic.dispatch(EditorAction::Undo);
        assert_eq!(italic.buffer.to_string(), "\\*斜体*");
        italic.dispatch(EditorAction::Undo);
        assert_eq!(italic.buffer.to_string(), "");

        let mut code = EditorController::new();
        type_str(&mut code, "`code`");
        code.dispatch(EditorAction::Undo);
        assert_eq!(code.buffer.to_string(), "\\`code`");
        code.dispatch(EditorAction::Undo);
        assert_eq!(code.buffer.to_string(), "");

        let mut strike = EditorController::new();
        type_str(&mut strike, "~~del~~");
        strike.dispatch(EditorAction::Undo);
        assert_eq!(strike.buffer.to_string(), "\\~\\~del~~");
        strike.dispatch(EditorAction::Undo);
        assert_eq!(strike.buffer.to_string(), "");

        // Pre-existing `**bold*` + typing final `*` escapes both and 2nd Undo restores exact `**bold*`
        let mut pre_bold = EditorController::from_str("**bold*");
        pre_bold.set_caret("**bold*".len());
        type_str(&mut pre_bold, "*");
        pre_bold.dispatch(EditorAction::Undo);
        assert_eq!(pre_bold.buffer.to_string(), "\\*\\*bold**");
        pre_bold.dispatch(EditorAction::Undo);
        assert_eq!(pre_bold.buffer.to_string(), "**bold*");

        // 6. Code fence ``` + Enter inserts companion closing fence; 1st Undo -> `\``` `; 2nd Undo -> ``
        let mut fence = EditorController::new();
        type_str(&mut fence, "```");
        fence.dispatch(EditorAction::Enter);
        assert_eq!(fence.buffer.to_string(), "```\n\n```");
        assert_eq!(fence.selection, Selection::caret(4));
        fence.dispatch(EditorAction::Undo);
        assert_eq!(fence.buffer.to_string(), "\\```");
        fence.dispatch(EditorAction::Undo);
        assert_eq!(fence.buffer.to_string(), "");

        // 7. Thematic break `---` + Enter -> `---\n`; 1st Undo -> `\---`; 2nd Undo -> ``
        let mut hr = EditorController::new();
        type_str(&mut hr, "---");
        hr.dispatch(EditorAction::Enter);
        assert_eq!(hr.buffer.to_string(), "---\n");
        hr.dispatch(EditorAction::Undo);
        assert_eq!(hr.buffer.to_string(), "\\---");
        hr.dispatch(EditorAction::Undo);
        assert_eq!(hr.buffer.to_string(), "");

        // 8. Code fence ```rust + Enter above an existing code block still inserts its companion
        let mut above = EditorController::from_str("\n# Later\n```py\nprint(1)\n```\n");
        above.set_caret(0);
        type_str(&mut above, "```rust");
        above.dispatch(EditorAction::Enter);
        assert_eq!(
            above.buffer.to_string(),
            "```rust\n\n```\n# Later\n```py\nprint(1)\n```\n"
        );
    }

    #[test]
    fn test_prd_4_1_1_c_blockquote_enter_and_backspace_demotion() {
        let mut ed = EditorController::from_str("> first quote");
        ed.set_caret("> first quote".len());
        ed.dispatch(EditorAction::Enter);
        assert_eq!(ed.buffer.to_string(), "> first quote\n> ");

        // Empty blockquote Enter exits quote cleanly to paragraph
        ed.dispatch(EditorAction::Enter);
        assert_eq!(ed.buffer.to_string(), "> first quote\n");

        // Nested blockquote Backspace demotes one level at a time
        let mut nested = EditorController::from_str("> > text");
        nested.set_caret("> > ".len());
        nested.dispatch(EditorAction::Backspace);
        assert_eq!(nested.buffer.to_string(), "> text");
        nested.dispatch(EditorAction::Backspace);
        assert_eq!(nested.buffer.to_string(), "text");

        // Source mode and inside code fence ignore WYSIWYG triggers
        let mut src_ed = EditorController::new();
        src_ed.mode = ViewMode::Source;
        type_str(&mut src_ed, "```");
        src_ed.dispatch(EditorAction::Enter);
        assert_eq!(src_ed.buffer.to_string(), "```\n");

        let mut fence_ed = EditorController::from_str("```md\n> \n```");
        fence_ed.set_caret("```md\n> ".len());
        fence_ed.dispatch(EditorAction::Backspace);
        assert_eq!(fence_ed.buffer.to_string(), "```md\n>\n```");
    }

    #[test]
    fn test_p2_open_existing_code_fence_enter_then_two_undos_restores_fence() {
        let mut ed = EditorController::from_str("```");
        ed.set_caret(3);
        ed.dispatch(EditorAction::Enter);
        assert_eq!(ed.buffer.to_string(), "```\n\n```");

        // 1st Undo removes companion fence and escapes opening ```
        ed.dispatch(EditorAction::Undo);
        assert_eq!(ed.buffer.to_string(), "\\```");

        // 2nd Undo removes the escape cleanly back to original ` ``` `
        ed.dispatch(EditorAction::Undo);
        assert_eq!(ed.buffer.to_string(), "```");
        assert_eq!(ed.selection, Selection::caret(3));
    }

    fn assert_strict_crlf(text: &str) {
        let bytes = text.as_bytes();
        for (idx, &b) in bytes.iter().enumerate() {
            if b == b'\n' {
                assert!(
                    idx > 0 && bytes[idx - 1] == b'\r',
                    "found bare LF at byte {idx} in {text:?}"
                );
            }
            if b == b'\r' {
                assert!(
                    bytes.get(idx + 1) == Some(&b'\n'),
                    "found orphan CR at byte {idx} in {text:?}"
                );
            }
        }
    }

    #[test]
    fn test_crlf_enter_backspace_and_structural_edits_preserve_crlf_without_bare_lf() {
        // Exact N1 repro: CRLF document + Enter before Body line must insert `\r\n`, not bare `\n`.
        let mut ed = EditorController::from_str("# Title\r\n\r\nBody line\r\n");
        ed.set_caret("# Title\r\n\r\n".len());
        ed.enter();
        assert_eq!(ed.buffer.to_string(), "# Title\r\n\r\n\r\nBody line\r\n");
        assert_strict_crlf(&ed.buffer.to_string());

        // Selection-replace Enter preserves CRLF
        let mut sel_ed = EditorController::from_str("# Title\r\nReplaceMe line\r\n");
        let sel_start = "# Title\r\n".len();
        sel_ed.set_selection(sel_start, sel_start + "ReplaceMe".len());
        sel_ed.enter();
        assert_eq!(sel_ed.buffer.to_string(), "# Title\r\n\r\n line\r\n");
        assert_strict_crlf(&sel_ed.buffer.to_string());

        // List continuation at EOF and before `\r\n`, plus empty list exit
        let mut list_eof = EditorController::from_str("- item 1\r\n- item 2");
        list_eof.set_caret(list_eof.buffer.len_bytes());
        list_eof.enter();
        assert_eq!(list_eof.buffer.to_string(), "- item 1\r\n- item 2\r\n- ");
        assert_strict_crlf(&list_eof.buffer.to_string());
        list_eof.enter();
        assert_eq!(list_eof.buffer.to_string(), "- item 1\r\n- item 2\r\n");
        assert_strict_crlf(&list_eof.buffer.to_string());

        let mut list_mid = EditorController::from_str("- item 1\r\nAfter\r\n");
        list_mid.set_caret("- item 1".len());
        list_mid.enter();
        assert_eq!(list_mid.buffer.to_string(), "- item 1\r\n- \r\nAfter\r\n");
        assert_strict_crlf(&list_mid.buffer.to_string());
        list_mid.enter();
        assert_eq!(list_mid.buffer.to_string(), "- item 1\r\n\r\nAfter\r\n");
        assert_strict_crlf(&list_mid.buffer.to_string());

        // Blockquote continuation & exit before `\r\n`
        let mut quote_ed = EditorController::from_str("> quote\r\nAfter\r\n");
        quote_ed.set_caret("> quote".len());
        quote_ed.enter();
        assert_eq!(quote_ed.buffer.to_string(), "> quote\r\n> \r\nAfter\r\n");
        assert_strict_crlf(&quote_ed.buffer.to_string());
        quote_ed.enter();
        assert_eq!(quote_ed.buffer.to_string(), "> quote\r\n\r\nAfter\r\n");
        assert_strict_crlf(&quote_ed.buffer.to_string());

        // Code fence opening companion + 3rd Enter exit + single Undo + empty fence Backspace
        let mut fence_open = EditorController::from_str("Intro\r\n```rust");
        fence_open.set_caret(fence_open.buffer.len_bytes());
        fence_open.enter();
        assert_eq!(fence_open.buffer.to_string(), "Intro\r\n```rust\r\n\r\n```");
        assert_strict_crlf(&fence_open.buffer.to_string());

        let mut fence_exit = EditorController::from_str("```rust\r\nlet x = 1;\r\n```");
        fence_exit.set_caret("```rust\r\nlet x = 1;".len());
        fence_exit.enter();
        fence_exit.enter();
        let before_third = fence_exit.buffer.to_string();
        assert_eq!(before_third, "```rust\r\nlet x = 1;\r\n\r\n\r\n```");
        fence_exit.enter();
        assert_eq!(fence_exit.buffer.to_string(), "```rust\r\nlet x = 1;\r\n```\r\n");
        assert_strict_crlf(&fence_exit.buffer.to_string());
        fence_exit.undo();
        assert_eq!(fence_exit.buffer.to_string(), before_third);

        let mut empty_fence = EditorController::from_str("```\r\n\r\n```");
        empty_fence.set_caret("```\r\n".len());
        empty_fence.backspace();
        assert_eq!(empty_fence.buffer.to_string(), "");

        // Standard Backspace right after `\r\n` removes full `\r\n` without orphan `\r`
        let mut bs_ed = EditorController::from_str("A\r\nB");
        bs_ed.set_caret("A\r\n".len());
        bs_ed.backspace();
        assert_eq!(bs_ed.buffer.to_string(), "AB");
        assert_strict_crlf(&bs_ed.buffer.to_string());

        // insert_char('\n') and mid-CRLF defensive snapping keep CRLF intact
        let mut ch_ed = EditorController::from_str("A\r\nB\r\n");
        ch_ed.set_caret("A\r".len()); // mid-CRLF
        ch_ed.enter();
        assert_eq!(ch_ed.buffer.to_string(), "A\r\n\r\nB\r\n");
        assert_strict_crlf(&ch_ed.buffer.to_string());
        ch_ed.set_caret("A\r\n\r\nB".len());
        ch_ed.insert_char('\n');
        assert_eq!(ch_ed.buffer.to_string(), "A\r\n\r\nB\r\n\r\n");
        assert_strict_crlf(&ch_ed.buffer.to_string());

        // Progressive Select-All does not select `\r` of `\r\n`
        let mut sa_ed = EditorController::from_str("# Title\r\nBody\r\n");
        sa_ed.set_caret(2);
        sa_ed.select_all();
        assert_eq!(sa_ed.selection, Selection::new(0, "# Title".len()));
        sa_ed.insert_char('X');
        assert_eq!(sa_ed.buffer.to_string(), "X\r\nBody\r\n");
        assert_strict_crlf(&sa_ed.buffer.to_string());

        // Thematic break and code fence before `\r\n` preserve CRLF
        let mut hr_ed = EditorController::from_str("---\r\nAfter\r\n");
        hr_ed.set_caret("---".len());
        hr_ed.enter();
        assert_eq!(hr_ed.buffer.to_string(), "---\r\n\r\nAfter\r\n");
        assert_strict_crlf(&hr_ed.buffer.to_string());
        hr_ed.undo();
        assert_eq!(hr_ed.buffer.to_string(), "\\---\r\nAfter\r\n");

        let mut fence_mid = EditorController::from_str("```rust\r\nAfter\r\n");
        fence_mid.set_caret("```rust".len());
        fence_mid.enter();
        assert_eq!(fence_mid.buffer.to_string(), "```rust\r\n\r\n```\r\nAfter\r\n");
        assert_strict_crlf(&fence_mid.buffer.to_string());

        // Dominant CR document inserts `\r`
        let mut cr_ed = EditorController::from_str("Line 1\rLine 2\r");
        cr_ed.set_caret("Line 1".len());
        cr_ed.enter();
        assert_eq!(cr_ed.buffer.to_string(), "Line 1\r\rLine 2\r");
        assert!(!cr_ed.buffer.to_string().contains('\n'));

        // Deleting the last CRLF in a document remembers `preferred_line_ending == CrLf`
        let mut del_last = EditorController::from_str("A\r\nB");
        del_last.set_caret("A\r\n".len());
        del_last.backspace();
        assert_eq!(del_last.buffer.to_string(), "AB");
        del_last.enter();
        assert_eq!(del_last.buffer.to_string(), "A\r\nB");
        assert_strict_crlf(&del_last.buffer.to_string());
    }
}
