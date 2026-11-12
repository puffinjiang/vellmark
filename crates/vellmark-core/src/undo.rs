use crate::buffer::{ByteOffset, ByteRange, Selection, TextBuffer};

/// Semantic category of a transaction for undo coalescing and syntax-conversion undo (PRD E-06 & 4.1.1 B.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditKind {
    /// Continuous typing that can be coalesced into a single undo unit.
    Typing,
    /// Deletion (Backspace / Delete).
    Deletion,
    /// IME composition commit (always a standalone undo unit, PRD 4.1.1 F).
    ImeCommit,
    /// Automatic syntax conversion or companion insertion (e.g., closing code fence, table separator row, PRD 4.1.1 B.1).
    /// When undone via `UndoHistory::undo_with_syntax_escape`, inserts minimal backslash escape instead of deleting user text.
    SyntaxConversion {
        /// Byte offset where a backslash `\` should be inserted if this conversion is undone.
        /// `None` for conversions like table separator insertion where removing the auto-inserted text suffices.
        escape_insert_at: Option<ByteOffset>,
    },
    /// Structural edit, paste, table alignment, or external reload.
    Structural,
}

/// A single atomic byte replacement operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditOp {
    pub range: ByteRange,
    pub deleted: String,
    pub inserted: String,
}

impl EditOp {
    pub fn invert(&self) -> Self {
        Self {
            range: ByteRange::new(self.range.start, self.range.start + self.inserted.len()),
            deleted: self.inserted.clone(),
            inserted: self.deleted.clone(),
        }
    }
}

/// A reversible transaction containing one or more operations plus before/after selections.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transaction {
    pub ops: Vec<EditOp>,
    pub selection_before: Selection,
    pub selection_after: Selection,
    pub kind: EditKind,
}

impl Transaction {
    pub fn apply(&self, buf: &mut TextBuffer) {
        for op in &self.ops {
            buf.replace_bytes(op.range, &op.inserted);
        }
    }

    pub fn apply_inverse(&self, buf: &mut TextBuffer) {
        for op in self.ops.iter().rev() {
            let inv = op.invert();
            buf.replace_bytes(inv.range, &inv.inserted);
        }
    }
}

/// Build the minimal backslash-escape operation for PRD 4.1.1 B.1 syntax conversion undo.
pub fn syntax_conversion_escape_op(escape_at: ByteOffset) -> EditOp {
    EditOp {
        range: ByteRange::empty(escape_at),
        deleted: String::new(),
        inserted: "\\".to_string(),
    }
}

/// Unlimited undo/redo stack with typing coalescing and PRD 4.1.1 B.1 syntax conversion undo.
#[derive(Debug, Default, Clone)]
pub struct UndoHistory {
    undo_stack: Vec<Transaction>,
    redo_stack: Vec<Transaction>,
}

impl UndoHistory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    /// Execute and record a single replacement operation.
    pub fn apply_edit(
        &mut self,
        buf: &mut TextBuffer,
        range: ByteRange,
        replacement: &str,
        selection_before: Selection,
        selection_after: Selection,
        kind: EditKind,
    ) {
        let deleted = buf.replace_bytes(range, replacement);
        let op = EditOp {
            range,
            deleted,
            inserted: replacement.to_string(),
        };
        self.record_transaction(Transaction {
            ops: vec![op],
            selection_before,
            selection_after,
            kind,
        });
    }

    /// Record a marker-only syntax conversion (where no extra text was auto-inserted, e.g. `## ` or `**bold**`)
    /// so that pressing `Cmd/Ctrl + Z` immediately escapes the trigger marker per PRD 4.1.1 B.1.
    pub fn record_syntax_trigger(
        &mut self,
        escape_insert_at: ByteOffset,
        selection: Selection,
    ) {
        self.redo_stack.clear();
        self.undo_stack.push(Transaction {
            ops: Vec::new(),
            selection_before: selection,
            selection_after: selection,
            kind: EditKind::SyntaxConversion {
                escape_insert_at: Some(escape_insert_at),
            },
        });
    }

    pub fn record_transaction(&mut self, tx: Transaction) {
        self.redo_stack.clear();

        // Coalesce consecutive single-insert Typing transactions when adjacent (PRD E-06).
        if tx.kind == EditKind::Typing && tx.ops.len() == 1 && tx.ops[0].deleted.is_empty() {
            if let Some(prev) = self.undo_stack.last_mut() {
                if prev.kind == EditKind::Typing && prev.ops.len() == 1 && prev.ops[0].deleted.is_empty() {
                    let prev_end = prev.ops[0].range.start + prev.ops[0].inserted.len();
                    if tx.ops[0].range.start == prev_end && !tx.ops[0].inserted.contains('\n') {
                        prev.ops[0].inserted.push_str(&tx.ops[0].inserted);
                        prev.selection_after = tx.selection_after;
                        return;
                    }
                }
            }
        }

        self.undo_stack.push(tx);
    }

    /// Undo the top transaction, implementing PRD 4.1.1 B.1 for `SyntaxConversion`:
    /// - 1st undo after syntax conversion: removes any auto-inserted companion text and inserts `\`
    ///   at `escape_insert_at` (if specified), folds that escape into the preceding user-input transaction
    ///   so that a 2nd undo removes both the `\` and the user's typed text cleanly.
    pub fn undo(&mut self, buf: &mut TextBuffer) -> Option<Selection> {
        let tx = self.undo_stack.pop()?;

        match tx.kind {
            EditKind::SyntaxConversion { escape_insert_at } => {
                // First revert any auto-inserted companion text (e.g. closing fence or table separator row).
                tx.apply_inverse(buf);

                if let Some(esc_pos) = escape_insert_at {
                    // Insert `\` at `esc_pos` in the buffer.
                    let esc_op = syntax_conversion_escape_op(esc_pos);
                    buf.replace_bytes(esc_op.range, &esc_op.inserted);

                    // Also fold the inserted `\` into the preceding Typing transaction on the undo stack
                    // so that a 2nd Undo removes both the `\` and the user's continuous input (PRD 4.1.1 B.1 Rule 3).
                    if let Some(prev) = self.undo_stack.last_mut() {
                        if let Some(first_op) = prev.ops.first_mut() {
                            if esc_pos >= first_op.range.start
                                && esc_pos <= first_op.range.start + first_op.inserted.len()
                            {
                                let rel = esc_pos - first_op.range.start;
                                first_op.inserted.insert(rel, '\\');
                                prev.selection_after = Selection::caret(
                                    prev.selection_after.head + 1,
                                );
                            }
                        }
                    }

                    // Push an inverse transaction onto redo_stack so Redo restores the unescaped conversion.
                    let mut redo_ops = vec![EditOp {
                        range: ByteRange::new(esc_pos, esc_pos + 1),
                        deleted: "\\".to_string(),
                        inserted: String::new(),
                    }];
                    redo_ops.extend(tx.ops.clone());
                    let new_sel = Selection::caret(tx.selection_before.head + 1);
                    self.redo_stack.push(Transaction {
                        ops: redo_ops,
                        selection_before: new_sel,
                        selection_after: tx.selection_after,
                        kind: EditKind::Structural,
                    });
                    Some(new_sel)
                } else {
                    let sel = tx.selection_before;
                    self.redo_stack.push(tx);
                    Some(sel)
                }
            }
            _ => {
                tx.apply_inverse(buf);
                let sel = tx.selection_before;
                self.redo_stack.push(tx);
                Some(sel)
            }
        }
    }

    pub fn redo(&mut self, buf: &mut TextBuffer) -> Option<Selection> {
        let tx = self.redo_stack.pop()?;
        tx.apply(buf);
        let sel = tx.selection_after;
        self.undo_stack.push(tx);
        Some(sel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heading_syntax_conversion_undo_and_second_undo() {
        // PRD 4.1.1 B.1: Typing `## ` -> `## ` -> 1st Undo -> `\## ` -> 2nd Undo -> ``
        let mut buf = TextBuffer::new();
        let mut hist = UndoHistory::new();

        for (i, ch) in ['#', '#', ' '].iter().enumerate() {
            let s = ch.to_string();
            hist.apply_edit(
                &mut buf,
                ByteRange::empty(i),
                &s,
                Selection::caret(i),
                Selection::caret(i + 1),
                EditKind::Typing,
            );
        }
        // Trigger heading syntax conversion at offset 0
        hist.record_syntax_trigger(0, Selection::caret(3));
        assert_eq!(buf.to_string(), "## ");

        // 1st Undo: escapes trigger -> `\## `
        let sel1 = hist.undo(&mut buf).unwrap();
        assert_eq!(buf.to_string(), "\\## ");
        assert_eq!(sel1, Selection::caret(4));

        // 2nd Undo: removes both the escape and the continuous typing -> empty
        let sel2 = hist.undo(&mut buf).unwrap();
        assert_eq!(buf.to_string(), "");
        assert_eq!(sel2, Selection::caret(0));
    }

    #[test]
    fn test_ordered_list_and_bold_escape_undo() {
        // PRD 4.1.1 B.1: `1. ` -> `1\. ` -> ``
        let mut buf = TextBuffer::new();
        let mut hist = UndoHistory::new();
        hist.apply_edit(
            &mut buf,
            ByteRange::empty(0),
            "1. ",
            Selection::caret(0),
            Selection::caret(3),
            EditKind::Typing,
        );
        // Escape goes before `.` at byte offset 1
        hist.record_syntax_trigger(1, Selection::caret(3));
        hist.undo(&mut buf);
        assert_eq!(buf.to_string(), "1\\. ");
        hist.undo(&mut buf);
        assert_eq!(buf.to_string(), "");
    }

    #[test]
    fn test_code_fence_with_companion_closing_fence_undo() {
        // PRD 4.1.1 B.1: ``` + Enter inserts `\n\n``` ` companion; 1st Undo removes companion and escapes first backtick.
        let mut buf = TextBuffer::new();
        let mut hist = UndoHistory::new();
        hist.apply_edit(
            &mut buf,
            ByteRange::empty(0),
            "```",
            Selection::caret(0),
            Selection::caret(3),
            EditKind::Typing,
        );
        // Auto-insert companion newline + closing fence
        hist.apply_edit(
            &mut buf,
            ByteRange::empty(3),
            "\n\n```",
            Selection::caret(3),
            Selection::caret(4),
            EditKind::SyntaxConversion {
                escape_insert_at: Some(0),
            },
        );
        assert_eq!(buf.to_string(), "```\n\n```");

        // 1st Undo: companion removed and `\` inserted before ```
        hist.undo(&mut buf);
        assert_eq!(buf.to_string(), "\\```");

        // 2nd Undo: back to empty
        hist.undo(&mut buf);
        assert_eq!(buf.to_string(), "");
    }
}
