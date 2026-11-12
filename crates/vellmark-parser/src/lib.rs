//! `vellmark-parser` provides block-level incremental Markdown parsing, exact source byte spans,
//! and the two-level marker expansion rule (`block prefix + active inline span`, PRD 4.1.1 A).

use serde::{Deserialize, Serialize};
use vellmark_core::{ByteOffset, ByteRange, Selection};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockKind {
    Heading {
        level: u8,
        /// Byte range of the `# ` prefix that expands when cursor is inside the heading block (PRD 4.1.1 A).
        prefix_range: ByteRange,
    },
    Paragraph,
    CodeFence {
        info_string: String,
    },
    BlockQuote,
    ListItem {
        ordered: bool,
        indent: usize,
        task_checked: Option<bool>,
    },
    ThematicBreak,
    Table,
    YamlFrontMatter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InlineKind {
    Strong,
    Emphasis,
    Strikethrough,
    InlineCode,
    Link,
    Image,
}

/// An inline element with exact source byte ranges for its opening/closing markers and full span.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InlineSpan {
    pub kind: InlineKind,
    pub full_range: ByteRange,
    pub open_marker: ByteRange,
    pub close_marker: ByteRange,
}

impl InlineSpan {
    /// PRD 4.1.1 A: An inline element expands its markers iff the cursor (with empty selection)
    /// lies within the closed interval `[full_range.start, full_range.end]`.
    pub fn should_expand_markers(&self, selection: Selection) -> bool {
        selection.is_caret() && self.full_range.contains_inclusive(selection.head)
    }
}

/// A parsed block node mapped to its exact source UTF-8 byte range.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockNode {
    pub kind: BlockKind,
    pub range: ByteRange,
    pub inlines: Vec<InlineSpan>,
}

impl BlockNode {
    /// PRD 4.1.1 A: Returns the block-level prefix range to reveal in source form when the caret is inside this block,
    /// provided the selection is a caret (non-empty selections never expand markers).
    pub fn expanded_block_prefix(&self, selection: Selection) -> Option<ByteRange> {
        if !selection.is_caret() || !self.range.contains_inclusive(selection.head) {
            return None;
        }
        match &self.kind {
            BlockKind::Heading { prefix_range, .. } => Some(*prefix_range),
            BlockKind::ThematicBreak => Some(self.range),
            _ => None,
        }
    }

    /// Returns all inline spans within this block whose markers should be expanded for the given selection (PRD 4.1.1 A).
    pub fn expanded_inline_spans(&self, selection: Selection) -> Vec<&InlineSpan> {
        if !selection.is_caret() {
            return Vec::new();
        }
        self.inlines
            .iter()
            .filter(|span| span.should_expand_markers(selection))
            .collect()
    }
}

/// Parse source Markdown into `BlockNode`s with exact byte offsets.
pub fn parse_blocks(source: &str) -> Vec<BlockNode> {
    let mut blocks = Vec::new();
    let bytes = source.as_bytes();
    let mut pos: ByteOffset = 0;

    while pos < bytes.len() {
        let line_start = pos;
        let mut line_end = pos;
        while line_end < bytes.len() && bytes[line_end] != b'\n' {
            line_end += 1;
        }
        let next_pos = if line_end < bytes.len() {
            line_end + 1
        } else {
            line_end
        };
        let line_trimmed = source[line_start..line_end].trim_end_matches('\r');

        // Fenced code block (` ``` `)
        if let Some(rest) = line_trimmed.strip_prefix("```") {
            let info = rest.trim().to_string();
            let mut fence_end = next_pos;
            while fence_end < bytes.len() {
                let l_start = fence_end;
                let mut l_end = fence_end;
                while l_end < bytes.len() && bytes[l_end] != b'\n' {
                    l_end += 1;
                }
                fence_end = if l_end < bytes.len() { l_end + 1 } else { l_end };
                if source[l_start..l_end].trim_end_matches('\r').trim() == "```" {
                    break;
                }
            }
            blocks.push(BlockNode {
                kind: BlockKind::CodeFence { info_string: info },
                range: ByteRange::new(line_start, fence_end),
                inlines: Vec::new(),
            });
            pos = fence_end;
            continue;
        }

        // ATX Heading (`# `..`###### `)
        if let Some((level, prefix_len)) = parse_atx_heading_prefix(line_trimmed) {
            let range = ByteRange::new(line_start, line_end);
            let prefix_range = ByteRange::new(line_start, line_start + prefix_len);
            let inlines = parse_inlines(source, line_start + prefix_len, line_end);
            blocks.push(BlockNode {
                kind: BlockKind::Heading {
                    level,
                    prefix_range,
                },
                range,
                inlines,
            });
            pos = next_pos;
            continue;
        }

        // Thematic break (`---`)
        if line_trimmed == "---" || line_trimmed == "***" || line_trimmed == "___" {
            blocks.push(BlockNode {
                kind: BlockKind::ThematicBreak,
                range: ByteRange::new(line_start, line_end),
                inlines: Vec::new(),
            });
            pos = next_pos;
            continue;
        }

        // Paragraph or line block
        let range = ByteRange::new(line_start, line_end);
        let inlines = parse_inlines(source, line_start, line_end);
        blocks.push(BlockNode {
            kind: BlockKind::Paragraph,
            range,
            inlines,
        });
        pos = next_pos;
    }

    blocks
}

fn parse_atx_heading_prefix(line: &str) -> Option<(u8, usize)> {
    let bytes = line.as_bytes();
    let mut hashes = 0usize;
    while hashes < bytes.len() && bytes[hashes] == b'#' && hashes < 6 {
        hashes += 1;
    }
    if hashes > 0 && hashes < bytes.len() && bytes[hashes] == b' ' {
        Some((hashes as u8, hashes + 1))
    } else {
        None
    }
}

/// Scan a line range `[start, end)` for paired inline markers (`**...**`, `*...*`, `` `...` ``, `~~...~~`).
pub fn parse_inlines(source: &str, start: ByteOffset, end: ByteOffset) -> Vec<InlineSpan> {
    let slice = &source[start..end];
    let bytes = slice.as_bytes();
    let mut spans = Vec::new();
    let mut i = 0usize;

    while i < bytes.len() {
        // Skip backslash-escaped characters (`\*`, `\#`, etc., PRD 4.1.1 B.1)
        if bytes[i] == b'\\' && i + 1 < bytes.len() {
            i += 2;
            continue;
        }

        // Strong `**...**` (CommonMark left-flanking: content after `**` cannot be empty or start with whitespace)
        if i + 1 < bytes.len() && bytes[i] == b'*' && bytes[i + 1] == b'*' {
            if i + 2 < bytes.len() && !bytes[i + 2].is_ascii_whitespace() {
                if let Some(rel_close) = find_closing_double(&bytes[i + 2..], b'*') {
                    if rel_close > 0 && !bytes[i + 2 + rel_close - 1].is_ascii_whitespace() {
                        let close_idx = i + 2 + rel_close;
                        spans.push(InlineSpan {
                            kind: InlineKind::Strong,
                            full_range: ByteRange::new(start + i, start + close_idx + 2),
                            open_marker: ByteRange::new(start + i, start + i + 2),
                            close_marker: ByteRange::new(start + close_idx, start + close_idx + 2),
                        });
                        // Also recursively scan inside the strong span for nested emphasis (PRD 4.1.1 A)
                        let nested = parse_inlines(source, start + i + 2, start + close_idx);
                        spans.extend(nested);
                        i = close_idx + 2;
                        continue;
                    }
                }
            }
            // Skip the entire run of consecutive `*` so `**` is not re-parsed as empty `*` + `*`
            while i < bytes.len() && bytes[i] == b'*' {
                i += 1;
            }
            continue;
        }

        // Strikethrough `~~...~~`
        if i + 1 < bytes.len() && bytes[i] == b'~' && bytes[i + 1] == b'~' {
            if i + 2 < bytes.len() && !bytes[i + 2].is_ascii_whitespace() {
                if let Some(rel_close) = find_closing_double(&bytes[i + 2..], b'~') {
                    if rel_close > 0 && !bytes[i + 2 + rel_close - 1].is_ascii_whitespace() {
                        let close_idx = i + 2 + rel_close;
                        spans.push(InlineSpan {
                            kind: InlineKind::Strikethrough,
                            full_range: ByteRange::new(start + i, start + close_idx + 2),
                            open_marker: ByteRange::new(start + i, start + i + 2),
                            close_marker: ByteRange::new(start + close_idx, start + close_idx + 2),
                        });
                        i = close_idx + 2;
                        continue;
                    }
                }
            }
            while i < bytes.len() && bytes[i] == b'~' {
                i += 1;
            }
            continue;
        }

        // Emphasis `*...*` (cannot be empty or start/end with whitespace)
        if bytes[i] == b'*' {
            if i + 1 < bytes.len() && bytes[i + 1] != b'*' && !bytes[i + 1].is_ascii_whitespace() {
                if let Some(rel_close) = find_closing_single(&bytes[i + 1..], b'*') {
                    if rel_close > 0 && !bytes[i + 1 + rel_close - 1].is_ascii_whitespace() {
                        let close_idx = i + 1 + rel_close;
                        spans.push(InlineSpan {
                            kind: InlineKind::Emphasis,
                            full_range: ByteRange::new(start + i, start + close_idx + 1),
                            open_marker: ByteRange::new(start + i, start + i + 1),
                            close_marker: ByteRange::new(start + close_idx, start + close_idx + 1),
                        });
                        i = close_idx + 1;
                        continue;
                    }
                }
            }
        }

        // Inline code `` `...` ``
        if bytes[i] == b'`' {
            if let Some(rel_close) = find_closing_single(&bytes[i + 1..], b'`') {
                let close_idx = i + 1 + rel_close;
                spans.push(InlineSpan {
                    kind: InlineKind::InlineCode,
                    full_range: ByteRange::new(start + i, start + close_idx + 1),
                    open_marker: ByteRange::new(start + i, start + i + 1),
                    close_marker: ByteRange::new(start + close_idx, start + close_idx + 1),
                });
                i = close_idx + 1;
                continue;
            }
        }

        i += 1;
    }

    spans
}

fn find_closing_double(bytes: &[u8], marker: u8) -> Option<usize> {
    let mut i = 0usize;
    while i + 1 < bytes.len() {
        if bytes[i] == b'\\' {
            i += 2;
            continue;
        }
        if bytes[i] == marker && bytes[i + 1] == marker {
            return Some(i);
        }
        i += 1;
    }
    None
}

fn find_closing_single(bytes: &[u8], marker: u8) -> Option<usize> {
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'\\' {
            i += 2;
            continue;
        }
        if bytes[i] == marker {
            return Some(i);
        }
        i += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prd_case_3_inline_marker_expansion_only_for_active_element() {
        // PRD 4.1.1 G #3: `这是 **粗体** 和 *斜体*`
        let src = "这是 **粗体** 和 *斜体*";
        let blocks = parse_blocks(src);
        assert_eq!(blocks.len(), 1);
        let b = &blocks[0];
        assert_eq!(b.inlines.len(), 2);

        // Caret after "这是" (byte offset 6, before space) -> no inline markers expanded
        assert!(b.expanded_inline_spans(Selection::caret(6)).is_empty());

        // Caret right before `**粗体**` (byte offset 7) -> ONLY `**粗体**` expands
        let expanded = b.expanded_inline_spans(Selection::caret(7));
        assert_eq!(expanded.len(), 1);
        assert_eq!(expanded[0].kind, InlineKind::Strong);

        // Non-empty selection across the paragraph -> NO markers expand (PRD 4.1.1 A)
        assert!(b.expanded_inline_spans(Selection::new(0, 15)).is_empty());
    }

    #[test]
    fn test_prd_case_1_heading_prefix_expands_only_when_caret_inside() {
        let src = "## 标题\n下一段";
        let blocks = parse_blocks(src);
        assert_eq!(blocks.len(), 2);
        // Caret inside heading -> `## ` prefix (0..3) expands
        assert_eq!(
            blocks[0].expanded_block_prefix(Selection::caret(3)),
            Some(ByteRange::new(0, 3))
        );
        // Caret in second block -> heading prefix hidden
        assert_eq!(blocks[0].expanded_block_prefix(Selection::caret(12)), None);
    }

    #[test]
    fn test_escaped_markers_do_not_parse_as_bold() {
        // PRD 4.1.1 B.1: `\*\*粗体**` after 1st undo should not parse as Strong
        let src = "\\*\\*粗体**";
        let blocks = parse_blocks(src);
        assert!(blocks[0].inlines.is_empty());
    }
}
