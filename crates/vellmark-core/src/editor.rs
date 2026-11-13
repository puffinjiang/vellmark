//! High-level WYSIWYG editor controller implementing PRD 4.1.1 A–G:
//! - Mode switching (`WYSIWYG <-> Source`) with exact byte-offset cursor preservation (E-05, 4.1.1 E)
//! - IME composition lifecycle (`ime_update`, `ime_commit`, `ime_cancel`) where pre-edit text is
//!   never written to source buffer during composition (E-08, 4.1.1 F)
//! - Structural editing (`Backspace`, `Enter`, `Tab`, `Shift+Tab`) for headings, nested lists,
//!   task lists, blockquotes, fenced code blocks, and tables (4.1.1 C & G)

use crate::buffer::{ByteOffset, ByteRange, Selection, TextBuffer};
use crate::undo::{EditKind, EditOp, Transaction, UndoHistory};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewMode {
    Wysiwyg,
    Source,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditorAction {
    InsertChar(char),
    Backspace,
    Enter,
    Tab,
    ShiftTab,
    ToggleSourceMode,
    Undo,
    Redo,
}

/// Stateful editor controller wrapping `TextBuffer`, `Selection`, `UndoHistory`,
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
}

impl Default for EditorController {
    fn default() -> Self {
        Self::new()
    }
}

impl EditorController {
    pub fn new() -> Self {
        Self {
            buffer: TextBuffer::new(),
            selection: Selection::caret(0),
            undo: UndoHistory::new(),
            mode: ViewMode::Wysiwyg,
            ime_preedit: None,
            ime_composition_selection: None,
            expansion_frozen: false,
        }
    }

    pub fn from_str(text: &str) -> Self {
        Self {
            buffer: TextBuffer::from_str(text),
            selection: Selection::caret(0),
            undo: UndoHistory::new(),
            mode: ViewMode::Wysiwyg,
            ime_preedit: None,
            ime_composition_selection: None,
            expansion_frozen: false,
        }
    }

    /// Toggle between WYSIWYG and Source mode while preserving exact source byte offset (PRD 4.1.1 E & G #11).
    pub fn toggle_mode(&mut self) {
        self.ime_preedit = None;
        self.expansion_frozen = false;
        self.mode = match self.mode {
            ViewMode::Wysiwyg => ViewMode::Source,
            ViewMode::Source => ViewMode::Wysiwyg,
        };
    }

    pub fn set_caret(&mut self, offset: ByteOffset) {
        // Keep `ime_composition_selection` intact if an IME session was active so a subsequent `ime_commit`
        // triggered by the focus/click change still commits at the original composition range!
        self.ime_preedit = None;
        self.expansion_frozen = false;
        let clamped = offset.min(self.buffer.len_bytes());
        self.selection = Selection::caret(clamped);
    }

    pub fn set_selection(&mut self, anchor: ByteOffset, head: ByteOffset) {
        self.ime_preedit = None;
        self.expansion_frozen = false;
        let max = self.buffer.len_bytes();
        self.selection = Selection::new(anchor.min(max), head.min(max));
    }

    /// Start or update IME composition (PRD 4.1.1 F):
    /// Pre-edit string is stored in `ime_preedit`, `ime_composition_selection` captures the starting selection
    /// (including backward selection direction `anchor > head`), `buffer` revision and content remain untouched,
    /// and block expansion is frozen.
    pub fn ime_update(&mut self, preedit: &str) {
        if self.ime_composition_selection.is_none() {
            self.ime_composition_selection = Some(self.selection);
        }
        self.ime_preedit = Some(preedit.to_string());
        self.expansion_frozen = true;
    }

    /// Cancel IME composition (e.g. Esc key handled by IME, PRD 4.1.1 F).
    pub fn ime_cancel(&mut self) {
        self.ime_preedit = None;
        self.ime_composition_selection = None;
        self.expansion_frozen = false;
    }

    /// Clear unconfirmed IME preedit and composition anchor when a non-IME edit occurs.
    pub fn abort_ime_preedit(&mut self) {
        self.ime_preedit = None;
        self.ime_composition_selection = None;
        self.expansion_frozen = false;
    }

    /// Commit IME composition as a single standalone undo transaction (PRD 4.1.1 F & G #9, #10).
    /// If `set_caret` / `set_selection` moved the cursor during composition before `ime_commit` arrived,
    /// inserts `committed` at the original `ime_composition_selection` (so `A|B` -> move to end -> commit `"你"` yields `A你B`)
    /// and shifts the new cursor position by the byte delta if needed.
    pub fn ime_commit(&mut self, committed: &str) {
        self.ime_preedit = None;
        self.expansion_frozen = false;
        let orig_selection = self
            .ime_composition_selection
            .take()
            .unwrap_or(self.selection);
        let target_range = orig_selection.range();
        if committed.is_empty() {
            return;
        }
        let cursor_moved_away = self.selection != orig_selection;
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
            &mut self.buffer,
            target_range,
            committed,
            sel_before,
            sel_after,
            EditKind::ImeCommit,
        );
        self.selection = sel_after;
    }

    /// Type a character in WYSIWYG/Source mode with automatic syntax trigger recording (PRD 4.1.1 B & G #1, #6).
    pub fn insert_char(&mut self, ch: char) {
        self.abort_ime_preedit();

        let (line_start_before, _) = self.current_line_bounds();
        let in_code_fence = self.is_offset_inside_fenced_code_block(line_start_before);

        // PRD 4.1.1 G #6: Inside a Markdown table line (outside code fence), typing `|` in WYSIWYG mode writes escaped `\|` to source.
        if self.mode == ViewMode::Wysiwyg
            && !in_code_fence
            && ch == '|'
            && self.is_caret_in_table_row()
        {
            let pos = self.selection.range().start;
            let sel_before = self.selection;
            let sel_after = Selection::caret(pos + 2);
            self.undo.apply_edit(
                &mut self.buffer,
                self.selection.range(),
                "\\|",
                sel_before,
                sel_after,
                EditKind::Typing,
            );
            self.selection = sel_after;
            return;
        }

        let mut buf = [0u8; 4];
        let s = ch.encode_utf8(&mut buf);
        let range = self.selection.range();
        let new_caret = range.start + s.len();
        let sel_before = self.selection;
        let sel_after = Selection::caret(new_caret);
        self.undo.apply_edit(
            &mut self.buffer,
            range,
            s,
            sel_before,
            sel_after,
            EditKind::Typing,
        );
        self.selection = sel_after;

        // P2 fix: Only record WYSIWYG syntax conversion triggers when in `ViewMode::Wysiwyg` AND outside fenced code blocks!
        if self.mode == ViewMode::Wysiwyg && !in_code_fence && ch == ' ' {
            let (line_start, line_end) = self.current_line_bounds();
            let line = self.buffer.slice_to_string(ByteRange::new(line_start, line_end));
            if is_just_heading_trigger(&line) && self.selection.head == line_end {
                self.undo
                    .record_syntax_trigger(line_start, self.selection);
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
            let r = self.selection.range();
            let sel_before = self.selection;
            let sel_after = Selection::caret(r.start);
            self.undo.apply_edit(
                &mut self.buffer,
                r,
                "",
                sel_before,
                sel_after,
                EditKind::Deletion,
            );
            self.selection = sel_after;
            return;
        }

        let caret = self.selection.head;
        let (line_start, line_end) = self.current_line_bounds();
        let line = self.buffer.slice_to_string(ByteRange::new(line_start, line_end));

        if self.mode == ViewMode::Wysiwyg {
            // PRD 4.1.1 G #8: Backspace at start of empty code block content line converts code block to empty paragraph
            if caret == line_start && line.is_empty() {
                if let Some(fence_range) = self.enclosing_empty_code_fence_range(line_start) {
                    let sel_before = self.selection;
                    let sel_after = Selection::caret(fence_range.start);
                    self.undo.apply_edit(
                        &mut self.buffer,
                        fence_range,
                        "",
                        sel_before,
                        sel_after,
                        EditKind::Structural,
                    );
                    self.selection = sel_after;
                    return;
                }
            }

            let in_code_fence = self.is_offset_inside_fenced_code_block(line_start);

            // P2 fix: Inside a fenced code block, lines like `# comment` or `- item` are raw code, NOT Markdown headings/lists!
            if !in_code_fence {
                // PRD 4.1.1 C & G #2: Backspace at start of heading text (either at `line_start` or right after `## `)
                if let Some(prefix_len) = heading_prefix_len(&line) {
                    if caret == line_start || caret == line_start + prefix_len {
                        let sel_before = self.selection;
                        let sel_after = Selection::caret(line_start);
                        self.undo.apply_edit(
                            &mut self.buffer,
                            ByteRange::new(line_start, line_start + prefix_len),
                            "",
                            sel_before,
                            sel_after,
                            EditKind::Structural,
                        );
                        self.selection = sel_after;
                        return;
                    }
                }

                // PRD 4.1.1 C: Backspace at start of list item content
                if let Some((indent_bytes, marker_bytes)) = list_prefix_parts(&line) {
                    if caret == line_start + indent_bytes + marker_bytes {
                        let sel_before = self.selection;
                        if indent_bytes >= 2 {
                            // Nested item -> outdent by 2 spaces
                            let sel_after = Selection::caret(caret - 2);
                            self.undo.apply_edit(
                                &mut self.buffer,
                                ByteRange::new(line_start, line_start + 2),
                                "",
                                sel_before,
                                sel_after,
                                EditKind::Structural,
                            );
                            self.selection = sel_after;
                        } else {
                            // Top-level list item -> strip list marker, convert to paragraph in place
                            let sel_after = Selection::caret(line_start);
                            self.undo.apply_edit(
                                &mut self.buffer,
                                ByteRange::new(line_start, line_start + indent_bytes + marker_bytes),
                                "",
                                sel_before,
                                sel_after,
                                EditKind::Structural,
                            );
                            self.selection = sel_after;
                        }
                        return;
                    }
                }
            }
        }

        // Standard character or newline deletion (merges with previous block when at `line_start > 0`)
        if caret > 0 {
            let full = self.buffer.to_string();
            let prev_char_len = full[..caret]
                .chars()
                .next_back()
                .map(|c| c.len_utf8())
                .unwrap_or(1);
            let del_range = ByteRange::new(caret - prev_char_len, caret);
            let sel_before = self.selection;
            let sel_after = Selection::caret(del_range.start);
            self.undo.apply_edit(
                &mut self.buffer,
                del_range,
                "",
                sel_before,
                sel_after,
                EditKind::Deletion,
            );
            self.selection = sel_after;
        }
    }

    /// Handle Shift+Tab (PRD 4.1.1 C & G #4): outdent nested list item by one level (2 spaces).
    pub fn shift_tab(&mut self) {
        self.abort_ime_preedit();
        let (line_start, line_end) = self.current_line_bounds();
        if self.is_offset_inside_fenced_code_block(line_start) {
            return;
        }
        let line = self.buffer.slice_to_string(ByteRange::new(line_start, line_end));
        if let Some((indent_bytes, _)) = list_prefix_parts(&line) {
            if indent_bytes >= 2 {
                let sel_before = self.selection;
                let new_head = self.selection.head.saturating_sub(2).max(line_start);
                let sel_after = Selection::caret(new_head);
                self.undo.apply_edit(
                    &mut self.buffer,
                    ByteRange::new(line_start, line_start + 2),
                    "",
                    sel_before,
                    sel_after,
                    EditKind::Structural,
                );
                self.selection = sel_after;
            }
        }
    }

    /// Handle Enter with PRD 4.1.1 C & G (#5, #7):
    /// - Non-empty selection -> replaces the selected range with `\n` in a single transaction (P2 fix).
    /// - Empty top-level list item (`- ‸`) -> exit list and become empty paragraph.
    /// - Empty nested list item (`  - ‸`) -> outdent one level (`- ‸`).
    /// - Non-empty list item (`- item‸`) -> continue list on next line (`\n- `).
    /// - Inside fenced code block at end of block with 2 trailing empty lines (3rd consecutive Enter, G #7) ->
    ///   remove trailing empty lines and exit code block to a new paragraph after closing ` ``` ` in a SINGLE undo transaction (P2 fix).
    pub fn enter(&mut self) {
        self.abort_ime_preedit();

        // P2 fix: When a non-empty selection exists, Enter replaces the selected range with `\n`.
        if !self.selection.is_caret() {
            let range = self.selection.range();
            let sel_before = self.selection;
            let sel_after = Selection::caret(range.start + 1);
            self.undo.apply_edit(
                &mut self.buffer,
                range,
                "\n",
                sel_before,
                sel_after,
                EditKind::Typing,
            );
            self.selection = sel_after;
            return;
        }

        let caret = self.selection.head;
        let (line_start, line_end) = self.current_line_bounds();
        let line = self.buffer.slice_to_string(ByteRange::new(line_start, line_end));

        if self.mode == ViewMode::Wysiwyg {
            let in_code_fence = self.is_offset_inside_fenced_code_block(line_start);

            // PRD 4.1.1 G #5: List item handling (only outside fenced code blocks)
            if !in_code_fence {
                if let Some((indent_bytes, marker_bytes)) = list_prefix_parts(&line) {
                    let content_after_marker = line[indent_bytes + marker_bytes..].trim();
                    if content_after_marker.is_empty() {
                        let sel_before = self.selection;
                        if indent_bytes >= 2 {
                            // Outdent nested empty item
                            let sel_after = Selection::caret(caret - 2);
                            self.undo.apply_edit(
                                &mut self.buffer,
                                ByteRange::new(line_start, line_start + 2),
                                "",
                                sel_before,
                                sel_after,
                                EditKind::Structural,
                            );
                            self.selection = sel_after;
                        } else {
                            // Exit top-level list -> empty paragraph
                            let sel_after = Selection::caret(line_start);
                            self.undo.apply_edit(
                                &mut self.buffer,
                                ByteRange::new(line_start, line_end),
                                "",
                                sel_before,
                                sel_after,
                                EditKind::Structural,
                            );
                            self.selection = sel_after;
                        }
                        return;
                    } else if caret == line_end {
                        // Continue list on next line
                        let prefix = &line[..indent_bytes + marker_bytes];
                        let insertion = format!("\n{prefix}");
                        let sel_before = self.selection;
                        let sel_after = Selection::caret(caret + insertion.len());
                        self.undo.apply_edit(
                            &mut self.buffer,
                            ByteRange::empty(caret),
                            &insertion,
                            sel_before,
                            sel_after,
                            EditKind::Structural,
                        );
                        self.selection = sel_after;
                        return;
                    }
                }
            }

            // PRD 4.1.1 G #7 & P2 fix: 3rd consecutive Enter at end of fenced code block exits the code block
            // in a SINGLE `Transaction` so a single `undo()` restores the exact pre-Enter state!
            if let Some((strip_range, closing_fence_end)) =
                self.check_third_enter_at_code_fence_end(caret)
            {
                let sel_before = self.selection;
                let removed_len = strip_range.len();
                let deleted = self.buffer.replace_bytes(strip_range, "");
                let mut ops = vec![EditOp {
                    range: strip_range,
                    deleted,
                    inserted: String::new(),
                }];

                let after_fence = closing_fence_end - removed_len;
                let sel_after = if after_fence == self.buffer.len_bytes() {
                    self.buffer
                        .replace_bytes(ByteRange::empty(after_fence), "\n");
                    ops.push(EditOp {
                        range: ByteRange::empty(after_fence),
                        deleted: String::new(),
                        inserted: "\n".to_string(),
                    });
                    Selection::caret(after_fence + 1)
                } else {
                    Selection::caret(after_fence)
                };

                self.undo.record_transaction(Transaction {
                    ops,
                    selection_before: sel_before,
                    selection_after: sel_after,
                    kind: EditKind::Structural,
                });
                self.selection = sel_after;
                return;
            }
        }

        // Default newline insertion
        let sel_before = self.selection;
        let sel_after = Selection::caret(caret + 1);
        self.undo.apply_edit(
            &mut self.buffer,
            ByteRange::empty(caret),
            "\n",
            sel_before,
            sel_after,
            EditKind::Typing,
        );
        self.selection = sel_after;
    }

    pub fn undo(&mut self) {
        self.abort_ime_preedit();
        if let Some(sel) = self.undo.undo(&mut self.buffer) {
            self.selection = sel;
        }
    }

    pub fn redo(&mut self) {
        self.abort_ime_preedit();
        if let Some(sel) = self.undo.redo(&mut self.buffer) {
            self.selection = sel;
        }
    }

    fn current_line_bounds(&self) -> (ByteOffset, ByteOffset) {
        let full = self.buffer.to_string();
        let bytes = full.as_bytes();
        let caret = self.selection.head.min(bytes.len());
        let mut start = caret;
        while start > 0 && bytes[start - 1] != b'\n' {
            start -= 1;
        }
        let mut end = caret;
        while end < bytes.len() && bytes[end] != b'\n' {
            end += 1;
        }
        (start, end)
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
        let next_line_rel_end = after.strip_prefix('\n')?.find('\n').unwrap_or(after.len() - 1);
        let next_line = after[1..1 + next_line_rel_end].trim_end_matches('\r');
        if !is_closing_code_fence(next_line, f_char, f_len) {
            return None;
        }
        let close_end = empty_line_start + 1 + next_line_rel_end;
        Some(ByteRange::new(prev_line_start, close_end))
    }

    fn check_third_enter_at_code_fence_end(
        &self,
        caret: ByteOffset,
    ) -> Option<(ByteRange, ByteOffset)> {
        let full = self.buffer.to_string();
        // Caret must be on the 2nd consecutive empty line right before a valid closing code fence line
        if caret < 2 || !full[..caret].ends_with("\n\n") {
            return None;
        }
        let (f_char, f_len) = self.enclosing_fenced_code_block(caret)?;
        let after = full.get(caret..)?.strip_prefix('\n')?;
        let close_line_len = after.find('\n').unwrap_or(after.len());
        let close_line = after[..close_line_len].trim_end_matches('\r');
        if !is_closing_code_fence(close_line, f_char, f_len) {
            return None;
        }
        let closing_fence_end = caret + 1 + close_line_len;
        Some((ByteRange::new(caret - 2, caret), closing_fence_end))
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
}
