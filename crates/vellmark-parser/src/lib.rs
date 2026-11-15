//! `vellmark-parser` provides block-level incremental Markdown parsing, exact source byte spans,
//! CommonMark 0.31.2 / GFM 0.29-gfm spec-compliant parsing & HTML rendering (PRD 4.2.1),
//! and the two-level marker expansion rule (`block prefix + active inline span`, PRD 4.1.1 A).

use pulldown_cmark::{Options, Parser, html};
use serde::{Deserialize, Serialize};
use vellmark_core::{ByteOffset, ByteRange, Selection};

/// Render Markdown to spec-compliant HTML using CommonMark + GFM extensions
/// (tables, tasklists, strikethrough, YAML frontmatter) per PRD 4.2.1.
pub fn render_markdown_to_html(markdown: &str) -> String {
    let mut opts = Options::empty();
    opts.insert(Options::ENABLE_TABLES);
    opts.insert(Options::ENABLE_TASKLISTS);
    opts.insert(Options::ENABLE_STRIKETHROUGH);
    opts.insert(Options::ENABLE_FOOTNOTES);
    opts.insert(Options::ENABLE_YAML_STYLE_METADATA_BLOCKS);

    let parser = Parser::new_ext(markdown, opts);
    let mut html_output = String::with_capacity(markdown.len() * 3 / 2);
    html::push_html(&mut html_output, parser);
    html_output
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum TableAlign {
    #[default]
    Left,
    Center,
    Right,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableCellSpan {
    /// Byte range in `source` of the raw cell slice between surrounding unescaped `|` pipes.
    pub raw_range: ByteRange,
    /// Trimmed content byte range in `source` (or insertion point after `"| "` for empty space cells).
    pub content_range: ByteRange,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableRowParsed {
    /// Full line byte range in `source` (excluding trailing `\r`/`\n`).
    pub line_range: ByteRange,
    /// Whether this row is a valid GFM delimiter row (`| --- | :---: | ---: |`).
    pub is_delimiter: bool,
    /// Per-column alignments parsed when `is_delimiter` is true.
    pub alignments: Vec<TableAlign>,
    /// Parsed cell slices between unescaped pipes.
    pub cells: Vec<TableCellSpan>,
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

        // YAML Front Matter at byte 0 (`---` ... `---` / `...`, PRD 4.2 M-08)
        if line_start == 0 && line_trimmed == "---" {
            let mut fm_pos = next_pos;
            let mut fm_close_end = None;
            let mut fm_next_pos = next_pos;
            while fm_pos < bytes.len() {
                let l_start = fm_pos;
                let mut l_end = fm_pos;
                while l_end < bytes.len() && bytes[l_end] != b'\n' {
                    l_end += 1;
                }
                let l_next = if l_end < bytes.len() { l_end + 1 } else { l_end };
                let l_str = source[l_start..l_end].trim_end_matches('\r').trim();
                if l_str == "---" || l_str == "..." {
                    fm_close_end = Some(l_end);
                    fm_next_pos = l_next;
                    break;
                }
                fm_pos = l_next;
            }
            if let Some(close_end) = fm_close_end {
                blocks.push(BlockNode {
                    kind: BlockKind::YamlFrontMatter,
                    range: ByteRange::new(0, close_end),
                    inlines: Vec::new(),
                });
                pos = fm_next_pos;
                continue;
            }
        }

        // Fenced code block (` ``` ` or `~~~`, length >= 3, CommonMark spec)
        if let Some((f_char, f_len, info)) = parse_opening_fence_line(line_trimmed) {
            let mut fence_pos = next_pos;
            let mut block_end = line_end;
            let mut after_fence_pos = next_pos;
            while fence_pos < bytes.len() {
                let l_start = fence_pos;
                let mut l_end = fence_pos;
                while l_end < bytes.len() && bytes[l_end] != b'\n' {
                    l_end += 1;
                }
                let l_next = if l_end < bytes.len() { l_end + 1 } else { l_end };
                block_end = l_end;
                after_fence_pos = l_next;
                let candidate = source[l_start..l_end].trim_end_matches('\r');
                if is_closing_fence_line(candidate, f_char, f_len) {
                    break;
                }
                fence_pos = l_next;
            }
            blocks.push(BlockNode {
                kind: BlockKind::CodeFence { info_string: info },
                range: ByteRange::new(line_start, block_end),
                inlines: Vec::new(),
            });
            pos = after_fence_pos;
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

        // Thematic break (`---`, `***`, `___`)
        if line_trimmed == "---" || line_trimmed == "***" || line_trimmed == "___" {
            blocks.push(BlockNode {
                kind: BlockKind::ThematicBreak,
                range: ByteRange::new(line_start, line_end),
                inlines: Vec::new(),
            });
            pos = next_pos;
            continue;
        }

        // BlockQuote (`> ...`)
        if line_trimmed.trim_start().starts_with('>') {
            let inlines = parse_inlines(source, line_start, line_end);
            blocks.push(BlockNode {
                kind: BlockKind::BlockQuote,
                range: ByteRange::new(line_start, line_end),
                inlines,
            });
            pos = next_pos;
            continue;
        }

        // List Item (`- `, `* `, `+ `, `1. `, `- [ ] `)
        if let Some((ordered, indent, task_checked)) = parse_list_item_kind(line_trimmed) {
            let inlines = parse_inlines(source, line_start, line_end);
            blocks.push(BlockNode {
                kind: BlockKind::ListItem {
                    ordered,
                    indent,
                    task_checked,
                },
                range: ByteRange::new(line_start, line_end),
                inlines,
            });
            pos = next_pos;
            continue;
        }

        // GFM table block: non-delimiter header row followed immediately by a matching-column
        // delimiter row (`| --- | :---: | ---: |`, outer pipes optional), then 0+ body rows.
        if let Some((delim_end, delim_next_pos)) =
            parse_table_header_and_delim(source, bytes, line_start, line_end, next_pos)
        {
            let table_start = line_start;
            let mut table_end = delim_end;
            let mut table_next_pos = delim_next_pos;
            let mut inlines = Vec::new();

            if let Some(header_cells) = parse_table_cell_rel_ranges(line_trimmed) {
                for (c_start, c_end) in header_cells {
                    inlines.extend(parse_inlines(
                        source,
                        line_start + c_start,
                        line_start + c_end,
                    ));
                }
            }

            let mut table_pos = delim_next_pos;
            while table_pos < bytes.len() {
                let (b_start, b_end, b_next) = scan_line_bounds(bytes, table_pos);
                let raw_line = source[b_start..b_end].trim_end_matches('\r');
                if !is_table_body_row_candidate(raw_line) {
                    break;
                }
                if let Some(cells) = parse_table_cell_rel_ranges(raw_line) {
                    for (c_start, c_end) in cells {
                        inlines.extend(parse_inlines(source, b_start + c_start, b_start + c_end));
                    }
                }
                table_end = b_end;
                table_next_pos = b_next;
                table_pos = b_next;
            }

            blocks.push(BlockNode {
                kind: BlockKind::Table,
                range: ByteRange::new(table_start, table_end),
                inlines,
            });
            pos = table_next_pos;
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

/// Byte offsets of all unescaped `|` characters in a single table row string.
pub fn unescaped_pipe_offsets(line: &str) -> Vec<usize> {
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

fn scan_line_bounds(bytes: &[u8], start: usize) -> (usize, usize, usize) {
    let mut end = start;
    while end < bytes.len() && bytes[end] != b'\n' {
        end += 1;
    }
    let next = if end < bytes.len() { end + 1 } else { end };
    (start, end, next)
}

fn is_table_interrupting_block_opener(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.is_empty()
        || trimmed.starts_with('>')
        || parse_atx_heading_prefix(line).is_some()
        || parse_opening_fence_line(line).is_some()
        || parse_list_item_kind(line).is_some()
}

/// Split a single table row line (without trailing `\r`/`\n`) into cell `(start, end)`
/// byte offsets relative to `line`, following GFM 0.29-gfm pipe-table rules (outer pipes optional).
pub fn parse_table_cell_rel_ranges(line: &str) -> Option<Vec<(usize, usize)>> {
    let bytes = line.as_bytes();
    let ws_lead = bytes
        .iter()
        .take_while(|&&b| b == b' ' || b == b'\t')
        .count();
    let ws_tail = bytes.len()
        - bytes[ws_lead..]
            .iter()
            .rev()
            .take_while(|&&b| b == b' ' || b == b'\t')
            .count();
    if ws_lead >= ws_tail {
        return None;
    }

    let pipes = unescaped_pipe_offsets(line);
    if pipes.is_empty() {
        return None;
    }

    let has_leading_pipe = pipes[0] == ws_lead;
    // A standalone pipe (`|` with optional surrounding whitespace) defines zero cells.
    if pipes.len() == 1 && has_leading_pipe && ws_tail == pipes[0] + 1 {
        return None;
    }

    let has_trailing_pipe =
        pipes.last().copied() == Some(ws_tail - 1) && (pipes.len() >= 2 || !has_leading_pipe);

    let mut cells = Vec::with_capacity(pipes.len() + 1);
    let mut cursor = if has_leading_pipe {
        pipes[0] + 1
    } else {
        ws_lead
    };
    let pipe_slice = if has_leading_pipe {
        &pipes[1..]
    } else {
        &pipes[..]
    };
    for &p in pipe_slice {
        cells.push((cursor, p));
        cursor = p + 1;
    }
    if !has_trailing_pipe {
        cells.push((cursor, ws_tail));
    }

    if cells.is_empty() {
        None
    } else {
        Some(cells)
    }
}

fn parse_table_delimiter_alignments_for_cells(
    line: &str,
    cell_ranges: &[(usize, usize)],
) -> Option<Vec<TableAlign>> {
    if cell_ranges.is_empty() {
        return None;
    }
    let mut alignments = Vec::with_capacity(cell_ranges.len());
    for &(start, end) in cell_ranges {
        let cell = line[start..end].trim();
        let (left_colon, s) = match cell.strip_prefix(':') {
            Some(rest) => (true, rest),
            None => (false, cell),
        };
        let (right_colon, dashes) = match s.strip_suffix(':') {
            Some(rest) => (true, rest),
            None => (false, s),
        };
        if dashes.is_empty() || !dashes.bytes().all(|b| b == b'-') {
            return None;
        }
        let align = match (left_colon, right_colon) {
            (true, true) => TableAlign::Center,
            (false, true) => TableAlign::Right,
            _ => TableAlign::Left,
        };
        alignments.push(align);
    }
    Some(alignments)
}

fn is_table_body_row_candidate(line: &str) -> bool {
    !is_table_interrupting_block_opener(line) && parse_table_cell_rel_ranges(line).is_some()
}

fn parse_table_header_and_delim(
    source: &str,
    bytes: &[u8],
    header_start: usize,
    header_end: usize,
    delim_scan_start: usize,
) -> Option<(usize, usize)> {
    let header_raw = source[header_start..header_end].trim_end_matches('\r');
    if is_table_interrupting_block_opener(header_raw) {
        return None;
    }
    let header_cells = parse_table_cell_rel_ranges(header_raw)?;
    if parse_table_delimiter_alignments_for_cells(header_raw, &header_cells).is_some() {
        return None;
    }
    if delim_scan_start >= bytes.len() {
        return None;
    }
    let (delim_start, delim_end, delim_next_pos) = scan_line_bounds(bytes, delim_scan_start);
    let delim_raw = source[delim_start..delim_end].trim_end_matches('\r');
    if is_table_interrupting_block_opener(delim_raw) {
        return None;
    }
    let delim_cells = parse_table_cell_rel_ranges(delim_raw)?;
    let delim_aligns = parse_table_delimiter_alignments_for_cells(delim_raw, &delim_cells)?;
    if delim_aligns.len() != header_cells.len() {
        return None;
    }
    Some((delim_end, delim_next_pos))
}

/// Whether a single line (without trailing `\r`/`\n`) is a Markdown pipe-table row candidate.
pub fn is_pipe_table_line(line: &str) -> bool {
    parse_table_cell_rel_ranges(line).is_some()
}

/// Parse per-column alignment markers from a delimiter row (`| --- | :---: | ---: |`, outer pipes optional).
pub fn parse_table_delimiter_alignments(line: &str, _pipes: &[usize]) -> Option<Vec<TableAlign>> {
    let cells = parse_table_cell_rel_ranges(line)?;
    parse_table_delimiter_alignments_for_cells(line, &cells)
}

/// Parse all rows and cell byte ranges inside a `BlockKind::Table` block range.
pub fn parse_table_block(source: &str, block_range: ByteRange) -> Vec<TableRowParsed> {
    let end = block_range.end.min(source.len());
    let mut pos = block_range.start.min(end);
    let bytes = source.as_bytes();
    let mut rows = Vec::new();

    while pos < end {
        let l_start = pos;
        let mut l_end = pos;
        while l_end < end && bytes[l_end] != b'\n' {
            l_end += 1;
        }
        let next_pos = if l_end < end { l_end + 1 } else { l_end };
        let content_end = if l_end > l_start && bytes[l_end - 1] == b'\r' {
            l_end - 1
        } else {
            l_end
        };

        let line = &source[l_start..content_end];
        let cell_ranges = parse_table_cell_rel_ranges(line).unwrap_or_default();
        // Per GFM 0.29-gfm, only row index 1 (immediately beneath header row 0) is the delimiter row.
        let alignments = if rows.len() == 1 {
            parse_table_delimiter_alignments_for_cells(line, &cell_ranges)
        } else {
            None
        };
        let is_delimiter = alignments.is_some();
        let mut cells = Vec::with_capacity(cell_ranges.len());
        for (rel_start, rel_end) in cell_ranges {
            let raw_start = l_start + rel_start;
            let raw_end = l_start + rel_end;
            let raw = &source[raw_start..raw_end];
            let lead_ws = raw.bytes().take_while(|&b| b == b' ' || b == b'\t').count();
            let content_range = if raw_start + lead_ws >= raw_end {
                let caret_pos = if raw.is_empty() {
                    raw_start
                } else {
                    (raw_start + 1).min(raw_end)
                };
                ByteRange::empty(caret_pos)
            } else {
                let tail_ws = raw[lead_ws..]
                    .bytes()
                    .rev()
                    .take_while(|&b| b == b' ' || b == b'\t')
                    .count();
                ByteRange::new(raw_start + lead_ws, raw_end - tail_ws)
            };
            cells.push(TableCellSpan {
                raw_range: ByteRange::new(raw_start, raw_end),
                content_range,
            });
        }
        rows.push(TableRowParsed {
            line_range: ByteRange::new(l_start, content_end),
            is_delimiter,
            alignments: alignments.unwrap_or_default(),
            cells,
        });

        pos = next_pos;
    }

    rows
}

fn parse_opening_fence_line(line: &str) -> Option<(u8, usize, String)> {
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
    let info = &line[indent + run..];
    if ch == b'`' && info.contains('`') {
        return None;
    }
    Some((ch, run, info.trim().to_string()))
}

fn is_closing_fence_line(line: &str, fence_char: u8, min_len: usize) -> bool {
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

fn parse_list_item_kind(line: &str) -> Option<(bool, usize, Option<bool>)> {
    let bytes = line.as_bytes();
    let mut indent = 0usize;
    while indent < bytes.len() && bytes[indent] == b' ' {
        indent += 1;
    }
    let rest = &line[indent..];
    if rest.starts_with("- [ ] ") {
        return Some((false, indent, Some(false)));
    }
    if rest.starts_with("- [x] ") || rest.starts_with("- [X] ") {
        return Some((false, indent, Some(true)));
    }
    for m in ["- ", "* ", "+ "] {
        if rest.starts_with(m) {
            return Some((false, indent, None));
        }
    }
    let mut digits = 0usize;
    while digits < rest.len() && rest.as_bytes()[digits].is_ascii_digit() {
        digits += 1;
    }
    if digits > 0 && rest[digits..].starts_with(". ") {
        return Some((true, indent, None));
    }
    None
}

/// Sidebar Outline item extracted from document headings (PRD 4.5 F-04).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutlineItem {
    pub level: u8,
    pub title: String,
    pub block_range: ByteRange,
    pub content_range: ByteRange,
}

/// Extract document outline (`H1`..`H6`) with exact source byte ranges, ignoring `#` inside
/// fenced code blocks or YAML Front Matter (PRD 4.5 F-04).
pub fn extract_outline(source: &str) -> Vec<OutlineItem> {
    let blocks = parse_blocks(source);
    let mut outline = Vec::new();
    for block in blocks {
        if let BlockKind::Heading {
            level,
            prefix_range,
        } = block.kind
        {
            let content_range = ByteRange::new(prefix_range.end, block.range.end);
            let title = source[content_range.start..content_range.end]
                .trim()
                .to_string();
            outline.push(OutlineItem {
                level,
                title,
                block_range: block.range,
                content_range,
            });
        }
    }
    outline
}

/// Returns the index in `outline` of the heading section that currently contains `caret` (PRD 4.5 F-04).
pub fn active_outline_index(outline: &[OutlineItem], caret: ByteOffset) -> Option<usize> {
    let mut active = None;
    for (idx, item) in outline.iter().enumerate() {
        if item.block_range.start <= caret {
            active = Some(idx);
        } else {
            break;
        }
    }
    active
}

/// Clipboard copy payload containing both Markdown (`text/plain`) and HTML (`text/html`) with
/// automatically completed paired inline markers when a selection partially overlaps inline elements (PRD 4.1.1 D & 4.6 X-06).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CopyPayload {
    pub markdown: String,
    pub html: String,
}

/// Generates `CopyPayload` (`text/plain` + `text/html`) for `selection` in `source`, automatically
/// completing paired inline markers (`**`, `*`, `~~`, `` ` ``) when an inline element is partially selected (PRD 4.1.1 D).
pub fn copy_selection(source: &str, selection: Selection) -> CopyPayload {
    if selection.is_caret() {
        return CopyPayload {
            markdown: String::new(),
            html: String::new(),
        };
    }
    let sel_range = selection.range();
    let mut start = sel_range.start.min(source.len());
    while start > 0 && !source.is_char_boundary(start) {
        start -= 1;
    }
    let mut end = sel_range.end.min(source.len()).max(start);
    while end > start && !source.is_char_boundary(end) {
        end -= 1;
    }
    if start == end {
        return CopyPayload {
            markdown: String::new(),
            html: String::new(),
        };
    }

    let blocks = parse_blocks(source);
    let mut prefix_markers = String::new();
    let mut suffix_markers: Vec<&str> = Vec::new();

    for block in &blocks {
        for span in &block.inlines {
            let content_start = span.open_marker.end;
            let content_end = span.close_marker.start;
            // Check if the selection intersects the inner text of `span` `[content_start, content_end)`
            let overlaps_content = start < content_end && end > content_start;
            if !overlaps_content {
                continue;
            }
            if start > span.open_marker.start {
                let missing_open = if start < content_start {
                    &source[span.open_marker.start..start]
                } else {
                    &source[span.open_marker.start..span.open_marker.end]
                };
                prefix_markers.push_str(missing_open);
            }
            if end < span.close_marker.end {
                let missing_close = if end > content_end {
                    &source[end..span.close_marker.end]
                } else {
                    &source[span.close_marker.start..span.close_marker.end]
                };
                suffix_markers.push(missing_close);
            }
        }
    }

    suffix_markers.reverse();
    let mut markdown = String::with_capacity(
        prefix_markers.len() + (end - start) + suffix_markers.iter().map(|s| s.len()).sum::<usize>(),
    );
    markdown.push_str(&prefix_markers);
    markdown.push_str(&source[start..end]);
    for m in suffix_markers {
        markdown.push_str(m);
    }

    let html = render_markdown_to_html(&markdown);
    CopyPayload { markdown, html }
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

        // Inline code `` `...` `` (highest inline precedence per CommonMark; escapes are literal inside code spans)
        if bytes[i] == b'`' {
            let mut open_len = 1usize;
            while i + open_len < bytes.len() && bytes[i + open_len] == b'`' {
                open_len += 1;
            }
            if let Some(rel_close) = find_closing_backtick_run(&bytes[i + open_len..], open_len) {
                let close_idx = i + open_len + rel_close;
                spans.push(InlineSpan {
                    kind: InlineKind::InlineCode,
                    full_range: ByteRange::new(start + i, start + close_idx + open_len),
                    open_marker: ByteRange::new(start + i, start + i + open_len),
                    close_marker: ByteRange::new(start + close_idx, start + close_idx + open_len),
                });
                i = close_idx + open_len;
                continue;
            }
            i += open_len;
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

        // Image `![alt](url)` and Link `[text](url)`
        let link_open_len = if bytes[i] == b'!' && i + 1 < bytes.len() && bytes[i + 1] == b'[' {
            Some((InlineKind::Image, 2usize))
        } else if bytes[i] == b'[' {
            Some((InlineKind::Link, 1usize))
        } else {
            None
        };
        if let Some((kind, open_len)) = link_open_len {
            if let Some(rel_bracket) = find_closing_single(&bytes[i + open_len..], b']') {
                let close_bracket = i + open_len + rel_bracket;
                if close_bracket + 1 < bytes.len() && bytes[close_bracket + 1] == b'(' {
                    if let Some(close_paren) = find_matching_paren(&bytes[close_bracket + 2..]) {
                        let span_end = close_bracket + 2 + close_paren + 1;
                        spans.push(InlineSpan {
                            kind,
                            full_range: ByteRange::new(start + i, start + span_end),
                            open_marker: ByteRange::new(start + i, start + i + open_len),
                            close_marker: ByteRange::new(start + close_bracket, start + span_end),
                        });
                        if kind == InlineKind::Link && close_bracket > i + open_len {
                            let nested = parse_inlines(source, start + i + open_len, start + close_bracket);
                            spans.extend(nested);
                        }
                        i = span_end;
                        continue;
                    }
                }
            }
        }

        i += 1;
    }

    spans
}

fn find_closing_backtick_run(bytes: &[u8], run_len: usize) -> Option<usize> {
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'`' {
            let mut run = 0usize;
            while i + run < bytes.len() && bytes[i + run] == b'`' {
                run += 1;
            }
            if run == run_len {
                return Some(i);
            }
            i += run;
            continue;
        }
        i += 1;
    }
    None
}

fn find_matching_paren(bytes: &[u8]) -> Option<usize> {
    let mut depth = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'\\' {
            i += 2;
            continue;
        }
        if bytes[i] == b'(' {
            depth += 1;
        } else if bytes[i] == b')' {
            if depth == 0 {
                return Some(i);
            }
            depth -= 1;
        }
        i += 1;
    }
    None
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

fn shift_offset(off: ByteOffset, delta: isize) -> ByteOffset {
    debug_assert!(off as isize + delta >= 0);
    (off as isize + delta) as usize
}

fn shift_range(range: ByteRange, delta: isize) -> ByteRange {
    ByteRange::new(shift_offset(range.start, delta), shift_offset(range.end, delta))
}

impl BlockNode {
    /// Shift all byte offsets inside this block node and its inline spans by `delta`.
    pub fn shift_offsets(&mut self, delta: isize) {
        if delta == 0 {
            return;
        }
        self.range = shift_range(self.range, delta);
        if let BlockKind::Heading { prefix_range, .. } = &mut self.kind {
            *prefix_range = shift_range(*prefix_range, delta);
        }
        for span in &mut self.inlines {
            span.full_range = shift_range(span.full_range, delta);
            span.open_marker = shift_range(span.open_marker, delta);
            span.close_marker = shift_range(span.close_marker, delta);
        }
    }
}

/// Summary of work performed by an incremental re-parse (PRD 5.1 / 6.2 #2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IncrementalUpdate {
    /// Byte span in the new source passed to `parse_blocks`.
    pub reparsed_new_range: ByteRange,
    /// Number of untouched leading blocks kept as-is.
    pub reused_prefix_blocks: usize,
    /// Number of untouched trailing blocks kept via byte-offset shifting only.
    pub shifted_suffix_blocks: usize,
    /// Whether fence/front-matter/boundary rules required a full document parse.
    pub full_reparse: bool,
}

/// Cached block AST with O(dirty_region + suffix_blocks) incremental updates.
#[derive(Debug, Clone)]
pub struct IncrementalDoc {
    source: String,
    blocks: Vec<BlockNode>,
}

impl IncrementalDoc {
    /// Parse an initial Markdown document into an incremental cache.
    pub fn new(source: impl Into<String>) -> Self {
        let source = source.into();
        let blocks = parse_blocks(&source);
        Self { source, blocks }
    }

    /// Borrow the current source text.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Borrow the up-to-date block AST slice.
    pub fn blocks(&self) -> &[BlockNode] {
        &self.blocks
    }

    /// Consume the cache into its underlying `(source, blocks)`.
    pub fn into_parts(self) -> (String, Vec<BlockNode>) {
        (self.source, self.blocks)
    }

    /// Synchronize with an updated full document string by computing the dirty span.
    pub fn sync_source(&mut self, new_source: impl Into<String>) -> IncrementalUpdate {
        let new_source = new_source.into();
        if self.source == new_source {
            return IncrementalUpdate {
                reparsed_new_range: ByteRange::empty(0),
                reused_prefix_blocks: self.blocks.len(),
                shifted_suffix_blocks: 0,
                full_reparse: false,
            };
        }
        let (edit_start, old_end, new_end) = compute_dirty_byte_span(&self.source, &new_source);
        self.apply_edit(new_source, edit_start, old_end, new_end)
    }

    /// Apply a known byte-range replacement `[edit_start..old_end) -> [edit_start..new_end)`
    /// resulting in `new_source`, updating `blocks()` to match `parse_blocks(&new_source)`.
    pub fn apply_edit(
        &mut self,
        new_source: impl Into<String>,
        edit_start: ByteOffset,
        old_end: ByteOffset,
        new_end: ByteOffset,
    ) -> IncrementalUpdate {
        let new_source = new_source.into();
        let old_len = self.source.len();
        let new_len = new_source.len();

        let invalid_bounds = edit_start > old_end
            || edit_start > new_end
            || old_end > old_len
            || new_end > new_len
            || !self.source.is_char_boundary(edit_start)
            || !self.source.is_char_boundary(old_end)
            || !new_source.is_char_boundary(edit_start)
            || !new_source.is_char_boundary(new_end);

        if invalid_bounds || self.blocks.len() < 4 {
            return self.fallback_full_reparse(new_source);
        }

        let old_win = touched_lines(&self.source, edit_start, old_end);
        let new_win = touched_lines(&new_source, edit_start, new_end);

        // Code fence delimiter changes can open/close/split fences across arbitrary ranges.
        let touches_fence_delim =
            has_fence_delimiter_line(old_win) || has_fence_delimiter_line(new_win);

        // Left boundary: block(s) at or just before `edit_start`.
        let first_affected = self.blocks.partition_point(|b| b.range.end < edit_start);
        let mut start_idx = first_affected.min(self.blocks.len()).saturating_sub(2);
        if start_idx > 0 && self.blocks[start_idx - 1].range.end == edit_start {
            start_idx = start_idx.saturating_sub(1);
        }
        // Step back to a preceding blank-line boundary within a small lookback when
        // possible, ensuring setext/table/list continuations cannot leak leftward.
        let min_lookback = start_idx.saturating_sub(3);
        while start_idx > min_lookback && !preceded_by_blank_line(&new_source, self.blocks[start_idx].range.start) {
            start_idx -= 1;
        }
        // If `start_idx` begins right after a Table block (e.g. merged/split without blank line)
        // or immediately beneath an adjacent pipe-table candidate Paragraph row, include it.
        while start_idx > 0
            && (matches!(self.blocks[start_idx - 1].kind, BlockKind::Table)
                || (!preceded_by_blank_line(&new_source, self.blocks[start_idx].range.start)
                    && matches!(self.blocks[start_idx - 1].kind, BlockKind::Paragraph)
                    && self
                        .source
                        .get(self.blocks[start_idx - 1].range.start..self.blocks[start_idx - 1].range.end)
                        .is_some_and(|line| parse_table_cell_rel_ranges(line.trim_end_matches('\r')).is_some())))
        {
            start_idx -= 1;
        }

        // Front-matter at byte 0 can absorb or release content up to the next `---`/`...`
        // anywhere in the document, so front-matter boundary edits always full-reparse.
        let has_frontmatter = matches!(
            self.blocks.first().map(|b| &b.kind),
            Some(BlockKind::YamlFrontMatter)
        );
        let frontmatter_affected = (has_frontmatter
            && edit_start <= self.blocks[0].range.end.saturating_add(4))
            || (new_source.starts_with("---")
                && (edit_start <= 4
                    || has_frontmatter_marker_line(old_win)
                    || has_frontmatter_marker_line(new_win)))
            || (self.source.starts_with("---") && has_frontmatter_marker_line(old_win));
        if frontmatter_affected || touches_fence_delim {
            return self.fallback_full_reparse(new_source);
        }

        // Right boundary: pick an untouched anchor block strictly after `old_end`.
        let delta: isize = new_end as isize - old_end as isize;
        let after_old_end = self.blocks.partition_point(|b| b.range.start <= old_end);
        let mut suffix_idx = after_old_end + 1;

        while suffix_idx < self.blocks.len() {
            // Avoid stopping inside/on a CodeFence or Table as anchor when safer blocks follow nearby.
            if matches!(
                self.blocks[suffix_idx].kind,
                BlockKind::CodeFence { .. } | BlockKind::Table
            ) && suffix_idx + 1 < self.blocks.len()
            {
                suffix_idx += 1;
                continue;
            }
            let reparse_start = if start_idx == 0 {
                0
            } else {
                self.blocks[start_idx].range.start
            };
            let stop_byte = if suffix_idx + 1 < self.blocks.len() {
                let raw = self.blocks[suffix_idx + 1].range.start as isize + delta;
                if raw < reparse_start as isize || (raw as usize) > new_len {
                    return self.fallback_full_reparse(new_source);
                }
                raw as usize
            } else {
                new_len
            };

            if !new_source.is_char_boundary(reparse_start)
                || !new_source.is_char_boundary(stop_byte)
                || reparse_start >= stop_byte
            {
                return self.fallback_full_reparse(new_source);
            }

            let local_blocks = parse_blocks_slice(&new_source, reparse_start, stop_byte);
            let mut expected_anchor = self.blocks[suffix_idx].clone();
            expected_anchor.shift_offsets(delta);

            let slice_to_eof = stop_byte == new_len;
            let anchor_matches = if slice_to_eof {
                true
            } else {
                local_blocks.last() == Some(&expected_anchor)
                    && local_blocks
                        .iter()
                        .all(|b| b.range.end <= expected_anchor.range.end)
            };

            if anchor_matches {
                let reparsed_range = ByteRange::new(reparse_start, stop_byte);
                let tail_start = suffix_idx + 1;
                let shifted_suffix_blocks = self.blocks.len().saturating_sub(tail_start);
                if delta != 0 {
                    for b in &mut self.blocks[tail_start..] {
                        b.shift_offsets(delta);
                    }
                }
                self.blocks.splice(start_idx..=suffix_idx, local_blocks);
                self.source = new_source;
                return IncrementalUpdate {
                    reparsed_new_range: reparsed_range,
                    reused_prefix_blocks: start_idx,
                    shifted_suffix_blocks,
                    full_reparse: false,
                };
            }

            // Structural merge/split crossed our tentative anchor; widen exponentially.
            let remaining = self.blocks.len() - suffix_idx;
            if remaining <= 4 {
                break;
            }
            let step = (suffix_idx - after_old_end).max(2);
            suffix_idx = (suffix_idx + step).min(self.blocks.len() - 1);
        }

        // Fall back to full re-parse if anchor never stabilized before EOF (or reached EOF via step).
        let reparse_start = if start_idx == 0 {
            0
        } else {
            self.blocks[start_idx].range.start
        };
        if start_idx > 0 && new_source.is_char_boundary(reparse_start) {
            let local_blocks = parse_blocks_slice(&new_source, reparse_start, new_len);
            self.blocks.splice(start_idx.., local_blocks);
            self.source = new_source;
            return IncrementalUpdate {
                reparsed_new_range: ByteRange::new(reparse_start, new_len),
                reused_prefix_blocks: start_idx,
                shifted_suffix_blocks: 0,
                full_reparse: false,
            };
        }

        self.fallback_full_reparse(new_source)
    }

    fn fallback_full_reparse(&mut self, new_source: String) -> IncrementalUpdate {
        let len = new_source.len();
        self.blocks = parse_blocks(&new_source);
        self.source = new_source;
        IncrementalUpdate {
            reparsed_new_range: ByteRange::new(0, len),
            reused_prefix_blocks: 0,
            shifted_suffix_blocks: 0,
            full_reparse: true,
        }
    }
}

fn parse_blocks_slice(new_source: &str, reparse_start: usize, stop_byte: usize) -> Vec<BlockNode> {
    let slice = &new_source[reparse_start..stop_byte];
    if reparse_start > 0 && slice.starts_with("---") {
        // Prevent `parse_blocks` from treating a mid-document `---` at index 0 of
        // the sub-slice as `YamlFrontMatter`.
        const GUARD: &str = "p\n\n";
        let mut buf = String::with_capacity(GUARD.len() + slice.len());
        buf.push_str(GUARD);
        buf.push_str(slice);
        let adjust = reparse_start as isize - GUARD.len() as isize;
        let mut blocks = Vec::new();
        for mut b in parse_blocks(&buf) {
            if b.range.start >= GUARD.len() {
                b.shift_offsets(adjust);
                blocks.push(b);
            }
        }
        blocks
    } else {
        let mut blocks = parse_blocks(slice);
        if reparse_start > 0 {
            let shift = reparse_start as isize;
            for b in &mut blocks {
                b.shift_offsets(shift);
            }
        }
        blocks
    }
}

fn compute_dirty_byte_span(old: &str, new: &str) -> (ByteOffset, ByteOffset, ByteOffset) {
    let old_b = old.as_bytes();
    let new_b = new.as_bytes();
    let min_len = old_b.len().min(new_b.len());

    let mut prefix = 0usize;
    while prefix < min_len && old_b[prefix] == new_b[prefix] {
        prefix += 1;
    }
    while prefix > 0 && (!old.is_char_boundary(prefix) || !new.is_char_boundary(prefix)) {
        prefix -= 1;
    }

    let max_suffix = min_len - prefix;
    let mut suffix = 0usize;
    while suffix < max_suffix && old_b[old_b.len() - 1 - suffix] == new_b[new_b.len() - 1 - suffix]
    {
        suffix += 1;
    }
    while suffix > 0
        && (!old.is_char_boundary(old.len() - suffix)
            || !new.is_char_boundary(new.len() - suffix))
    {
        suffix -= 1;
    }

    (prefix, old.len() - suffix, new.len() - suffix)
}

fn touched_lines(src: &str, start: ByteOffset, end: ByteOffset) -> &str {
    let bytes = src.as_bytes();
    let mut ls = start.min(bytes.len());
    while ls > 0 && bytes[ls - 1] != b'\n' {
        ls -= 1;
    }
    let mut le = end.min(bytes.len());
    while le < bytes.len() && bytes[le] != b'\n' {
        le += 1;
    }
    &src[ls..le]
}

fn has_fence_delimiter_line(window: &str) -> bool {
    window.lines().any(|line| {
        let trimmed = line.trim_start_matches([' ', '\t']);
        trimmed.starts_with("```") || trimmed.starts_with("~~~")
    })
}

fn has_frontmatter_marker_line(window: &str) -> bool {
    window.lines().any(|line| {
        let t = line.trim_end_matches('\r');
        t == "---" || t == "..."
    })
}

fn preceded_by_blank_line(src: &str, block_start: ByteOffset) -> bool {
    let prefix = src[..block_start.min(src.len())].as_bytes();
    prefix.ends_with(b"\n\n") || prefix.ends_with(b"\r\n\r\n")
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

        // PRD 4.1.1 G #11: Link markers expand when caret is inside `[text](url)`
        let link_src = "See [Vellmark](https://example.com) now";
        let link_blocks = parse_blocks(link_src);
        let inside_url = "See [Vellmark](https://".len();
        let expanded_links = link_blocks[0].expanded_inline_spans(Selection::caret(inside_url));
        assert_eq!(expanded_links.len(), 1);
        assert_eq!(expanded_links[0].kind, InlineKind::Link);
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

    #[test]
    fn test_commonmark_and_gfm_html_rendering_compliance() {
        // PRD 4.2.1: CommonMark + GFM tables, tasklists, strikethrough
        let md = "# Title\n\n- [x] Done\n- [ ] Todo\n\n| A | B |\n| --- | --- |\n| 1 | ~~2~~ |\n";
        let out = render_markdown_to_html(md);
        assert!(out.contains("<h1>Title</h1>"));
        assert!(out.contains("type=\"checkbox\""));
        assert!(out.contains("<table>"));
        assert!(out.contains("<del>2</del>"));
    }

    #[test]
    fn test_prd_f04_extract_outline_ignores_frontmatter_and_code_fences() {
        let md = "---\n# not a heading\n---\n# Chapter 1\n```sh\n# comment\n```\n## Section 1.1\nBody";
        let outline = extract_outline(md);
        assert_eq!(outline.len(), 2);
        assert_eq!(outline[0].level, 1);
        assert_eq!(outline[0].title, "Chapter 1");
        assert_eq!(outline[1].level, 2);
        assert_eq!(outline[1].title, "Section 1.1");

        // Caret inside body of Section 1.1 highlights index 1
        assert_eq!(active_outline_index(&outline, md.len() - 2), Some(1));
    }

    #[test]
    fn test_prd_4_1_1_d_copy_selection_auto_balances_partial_inline_markers() {
        // PRD 4.1.1 D: Partially selecting "bold" inside "hello **bold text** world"
        // automatically completes paired `**` markers in both Markdown and HTML clipboard formats.
        let src = "hello **bold text** world";
        let start = "hello **".len();
        let end = start + "bold".len();
        let payload = copy_selection(src, Selection::new(start, end));
        assert_eq!(payload.markdown, "**bold**");
        assert_eq!(payload.html.trim(), "<p><strong>bold</strong></p>");
    }

    #[test]
    fn test_inline_code_precedence_and_partial_link_copy_selection() {
        // Inline code must suppress bold/link parsing inside backtick spans (multi-backtick included)
        let code_src = "Use ``**raw** [not](link)`` here and [click **here**](https://example.com)";
        let spans = parse_inlines(code_src, 0, code_src.len());
        let has_raw_code = spans.iter().any(|s| {
            s.kind == InlineKind::InlineCode
                && &code_src[s.full_range.start..s.full_range.end] == "``**raw** [not](link)``"
        });
        assert!(has_raw_code, "expected multi-backtick inline code span");

        // No bold or link span should be emitted for `**raw** [not](link)` inside inline code
        let code_span = spans
            .iter()
            .find(|s| s.kind == InlineKind::InlineCode)
            .expect("inline code span");
        let has_false_nested = spans.iter().any(|s| {
            s.kind != InlineKind::InlineCode
                && s.full_range.start >= code_span.full_range.start
                && s.full_range.end <= code_span.full_range.end
        });
        assert!(
            !has_false_nested,
            "inline code should suppress bold/link parsing inside it"
        );
        assert!(
            spans
                .iter()
                .any(|s| s.kind == InlineKind::Link
                    && &code_src[s.full_range.start..s.full_range.end]
                        == "[click **here**](https://example.com)"),
            "expected trailing link span"
        );

        // Partial selection inside `[hello world](https://example.com)` completes paired `[` and `](url)`
        let link_src = "Visit [hello world](https://example.com) now";
        let start = "Visit [".len();
        let end = start + "hello".len();
        let payload = copy_selection(link_src, Selection::new(start, end));
        assert_eq!(payload.markdown, "[hello](https://example.com)");
        assert_eq!(
            payload.html.trim(),
            "<p><a href=\"https://example.com\">hello</a></p>"
        );

        // Starting right at content_start inside emphasis still completes symmetric marker
        let em_src = "say *hello world*";
        let em_start = "say *".len();
        let em_end = em_start + "hello".len();
        let em_payload = copy_selection(em_src, Selection::new(em_start, em_end));
        assert_eq!(em_payload.markdown, "*hello*");
    }

    #[test]
    fn test_fenced_code_and_yaml_front_matter_boundaries_do_not_overlap() {
        let md = "---\ntitle: Test\n---\n````md\n```\n# inside fence\n```\n````\n# Real Heading\n";
        let blocks = parse_blocks(md);
        assert_eq!(blocks.len(), 3, "expected front matter, fence, and heading");

        assert_eq!(blocks[0].kind, BlockKind::YamlFrontMatter);
        assert_eq!(blocks[0].range.start, 0);
        assert!(&md[blocks[0].range.start..blocks[0].range.end].ends_with("---"));

        match &blocks[1].kind {
            BlockKind::CodeFence { info_string } => {
                assert_eq!(info_string, "md");
                let slice = &md[blocks[1].range.start..blocks[1].range.end];
                assert!(slice.starts_with("````md"));
                assert!(slice.ends_with("````"));
                assert!(slice.contains("# inside fence"));
            }
            other => panic!("expected CodeFence, got {other:?}"),
        }

        match &blocks[2].kind {
            BlockKind::Heading { level, .. } => {
                assert_eq!(*level, 1);
                let slice = &md[blocks[2].range.start..blocks[2].range.end];
                assert!(slice.contains("Real Heading"));
            }
            other => panic!("expected Heading, got {other:?}"),
        }

        // Strict monotonicity / non-overlapping check
        for pair in blocks.windows(2) {
            assert!(
                pair[0].range.end <= pair[1].range.start,
                "adjacent blocks overlapped: {} > {}",
                pair[0].range.end,
                pair[1].range.start
            );
        }
    }

    fn apply_text_patch(
        inc: &mut IncrementalDoc,
        old_start: usize,
        old_end: usize,
        replacement: &str,
    ) -> IncrementalUpdate {
        let mut next = inc.source().to_string();
        next.replace_range(old_start..old_end, replacement);
        let upd = inc.apply_edit(next.clone(), old_start, old_end, old_start + replacement.len());
        assert_eq!(
            inc.blocks(),
            parse_blocks(&next).as_slice(),
            "IncrementalDoc mismatch after edit ({old_start}..{old_end} -> {replacement:?})"
        );
        upd
    }

    #[test]
    fn test_incremental_doc_local_edit_reuses_prefix_and_shifts_suffix() {
        let doc = concat!(
            "# Heading 0\n\n",
            "Paragraph 1 with **bold**.\n\n",
            "Paragraph 2 to edit.\n\n",
            "- list item A\n",
            "- list item B\n\n",
            "## Trailing Heading\n\n",
            "Final paragraph *italic*.\n"
        );
        let mut inc = IncrementalDoc::new(doc);
        assert_eq!(inc.blocks(), parse_blocks(doc).as_slice());

        let needle = "Paragraph 2 to edit.";
        let pos = doc.find(needle).unwrap();
        let upd = apply_text_patch(
            &mut inc,
            pos,
            pos + needle.len(),
            "Paragraph 2 now has `inline code` and **strong**!",
        );
        assert!(!upd.full_reparse);
        assert!(upd.reused_prefix_blocks >= 1);
        assert!(upd.shifted_suffix_blocks >= 1);
    }

    #[test]
    fn test_incremental_doc_equivalence_across_complex_mutations() {
        let initial = concat!(
            "---\n",
            "title: Spec\n",
            "---\n\n",
            "# Chapter 1\n\n",
            "中文段落 **加粗** 和 [链接](https://a.example)\n\n",
            "---\n\n",
            "> 引用内容 line 1\n\n",
            "```rust\nlet x = 1;\n```\n\n",
            "| Col A | Col B |\n| --- | --- |\n| 1 | 2 |\n\n",
            "Tail paragraph A\n\n",
            "Tail paragraph B\n\n",
            "Tail paragraph C\n"
        );
        let mut inc = IncrementalDoc::new(initial);
        assert_eq!(inc.blocks(), parse_blocks(initial).as_slice());

        // 1. Insert mid-document ThematicBreak right at a slice start candidate
        let p = inc.source().find("中文段落").unwrap();
        apply_text_patch(&mut inc, p, p, "---\n\n新段 ");

        // 2. Edit inside a closed CodeFence without touching fence delimiters (partial reuse)
        let let_x = inc.source().find("let x = 1;").unwrap();
        let fence_upd = apply_text_patch(&mut inc, let_x, let_x + "let x = 1;".len(), "let y = 42;\nprintln!(\"{y}\");");
        assert!(!fence_upd.full_reparse);
        assert!(fence_upd.shifted_suffix_blocks >= 1);

        // 3. Delete blank line merging two paragraphs or splitting block
        let tail_a = inc.source().find("Tail paragraph A\n\nTail").unwrap() + "Tail paragraph A".len();
        apply_text_patch(&mut inc, tail_a, tail_a + 1, "");

        // 4. Open a code fence mid-document and later close it
        let quote_pos = inc.source().find("> 引用").unwrap();
        apply_text_patch(&mut inc, quote_pos, quote_pos, "```python\nprint(1)\n");
        let after_open = quote_pos + "```python\nprint(1)\n".len();
        apply_text_patch(&mut inc, after_open, after_open, "```\n\n");

        // 5. Mutate YAML front matter and verify full consistency
        apply_text_patch(&mut inc, 0, 4, "---\nauthor: Alice\n");

        // 6. Remove YAML front matter closing marker
        let close_fm = inc.source().find("\n---\n").unwrap();
        apply_text_patch(&mut inc, close_fm, close_fm + 4, "");

        // 6. Delete `\n` right before closing ``` of a fence (line merge removes closing fence!)
        let close_fence = inc.source().find("\n```\n\n").unwrap();
        apply_text_patch(&mut inc, close_fence, close_fence + 1, "");

        // 7. Re-check `sync_source`
        let synced = format!("{}\n\n# Extra End\n", inc.source());
        inc.sync_source(synced.clone());
        assert_eq!(inc.blocks(), parse_blocks(&synced).as_slice());
    }

    #[test]
    fn test_incremental_doc_large_5000_blocks_local_update() {
        let mut doc = String::with_capacity(180_000);
        for i in 0..5_000 {
            doc.push_str(&format!("Paragraph #{i} with **bold** content.\n\n"));
        }
        let mut inc = IncrementalDoc::new(doc.clone());
        assert_eq!(inc.blocks().len(), 10_000);

        // Edit right in the middle (#2500)
        let target = "Paragraph #2500";
        let pos = doc.find(target).unwrap();
        let upd = apply_text_patch(&mut inc, pos, pos + target.len(), "## Heading 2500 Replaced");
        assert!(!upd.full_reparse);
        assert!(upd.reused_prefix_blocks > 4_900);
        assert!(upd.shifted_suffix_blocks > 4_900);
        assert!(upd.reparsed_new_range.len() < 512);
    }

    #[test]
    fn test_multiline_table_parsing_inlines_and_incremental_updates() {
        let src = concat!(
            "# Header\n\n",
            "| Name | Role | Notes |\n",
            "| :--- | :---: | ---: |\n",
            "| Alice | **Lead** | 2*x |\n",
            "| Bob | *Dev* | y*3 |\n\n",
            "After paragraph\n"
        );
        let blocks = parse_blocks(src);
        assert_eq!(blocks.len(), 5);
        assert_eq!(blocks[2].kind, BlockKind::Table);
        // Inline spans must stay confined within each cell (`2*x` and `y*3` do not merge across `|`).
        assert_eq!(blocks[2].inlines.len(), 2);
        assert_eq!(
            src.get(blocks[2].inlines[0].full_range.start..blocks[2].inlines[0].full_range.end),
            Some("**Lead**")
        );
        assert_eq!(
            src.get(blocks[2].inlines[1].full_range.start..blocks[2].inlines[1].full_range.end),
            Some("*Dev*")
        );

        let rows = parse_table_block(src, blocks[2].range);
        assert_eq!(rows.len(), 4);
        assert!(!rows[0].is_delimiter);
        assert!(rows[1].is_delimiter);
        assert_eq!(
            rows[1].alignments,
            vec![TableAlign::Left, TableAlign::Center, TableAlign::Right]
        );
        assert_eq!(rows[2].cells.len(), 3);
        assert_eq!(
            src.get(rows[2].cells[0].content_range.start..rows[2].cells[0].content_range.end),
            Some("Alice")
        );

        // Incremental mutations: edit cell, append row, split with blank line, merge back.
        let doc = concat!(
            "Intro A\n\n",
            "Intro B\n\n",
            "| C1 | C2 |\n| --- | --- |\n| A | B |\n\n",
            "Outro A\n\n",
            "Outro B\n"
        );
        let mut inc = IncrementalDoc::new(doc);
        let a_pos = inc.source().find("| A | B |").unwrap() + 2;
        let upd = apply_text_patch(&mut inc, a_pos, a_pos + 1, "Updated");
        assert_eq!(inc.blocks(), parse_blocks(inc.source()).as_slice());
        assert!(!upd.full_reparse);

        let split_pos = inc.source().find("| Updated | B |").unwrap();
        apply_text_patch(&mut inc, split_pos, split_pos, "\n");
        assert_eq!(inc.blocks(), parse_blocks(inc.source()).as_slice());
        apply_text_patch(&mut inc, split_pos, split_pos + 1, "");
        assert_eq!(inc.blocks(), parse_blocks(inc.source()).as_slice());
    }

    #[test]
    fn test_gfm_strict_table_recognition_delimiters_and_incremental_merge() {
        // Standalone delimiter rows (`|--|`, `|:---|`) and single header lines MUST stay Paragraph.
        for sample in [
            "|--|",
            "|:---|",
            "| a | b |",
            "| a |\n| b |",
            "| a | b |\n| --- |",
            "| --- |\n| --- |",
            "| a | b |\n\n| --- | --- |",
            "| a | b |\n- | ---",
        ] {
            let blocks = parse_blocks(sample);
            assert!(
                blocks.iter().all(|b| b.kind != BlockKind::Table),
                "sample `{sample:?}` must NOT be parsed as Table: {blocks:?}"
            );
        }

        // Single-hyphen / minimal GFM delimiter (`|-|`, `|--|`) and outer-pipe-optional tables parse cleanly.
        let one_col = "| A |\n|--|\n| x |";
        let one_col_blocks = parse_blocks(one_col);
        assert_eq!(one_col_blocks.len(), 1);
        assert_eq!(one_col_blocks[0].kind, BlockKind::Table);
        let one_rows = parse_table_block(one_col, one_col_blocks[0].range);
        assert_eq!(one_rows.len(), 3);
        assert!(one_rows[1].is_delimiter);

        let no_outer_pipes = "Name | Role | Score\n:--- | :---: | -:\nAlice | **Dev** | 10";
        let nop_blocks = parse_blocks(no_outer_pipes);
        assert_eq!(nop_blocks.len(), 1);
        assert_eq!(nop_blocks[0].kind, BlockKind::Table);
        let nop_rows = parse_table_block(no_outer_pipes, nop_blocks[0].range);
        assert_eq!(nop_rows.len(), 3);
        assert_eq!(
            nop_rows[1].alignments,
            vec![TableAlign::Left, TableAlign::Center, TableAlign::Right]
        );
        assert_eq!(
            no_outer_pipes.get(nop_rows[0].cells[0].content_range.start..nop_rows[0].cells[0].content_range.end),
            Some("Name")
        );
        assert_eq!(
            no_outer_pipes.get(nop_rows[2].cells[1].content_range.start..nop_rows[2].cells[1].content_range.end),
            Some("**Dev**")
        );

        // Prose or block opener right beneath a table cleanly ends the table.
        let ended = "| A |\n|-|\n| 1 |\nNormal paragraph\n| C |\n| - |\n# Heading | x";
        let ended_blocks = parse_blocks(ended);
        assert_eq!(ended_blocks[0].kind, BlockKind::Table);
        assert_eq!(
            ended.get(ended_blocks[0].range.start..ended_blocks[0].range.end),
            Some("| A |\n|-|\n| 1 |")
        );
        assert_eq!(ended_blocks[1].kind, BlockKind::Paragraph);
        assert_eq!(ended_blocks[2].kind, BlockKind::Table);
        assert!(matches!(ended_blocks[3].kind, BlockKind::Heading { level: 1, .. }));

        // Incremental merge when typing delimiter row underneath contiguous paragraph lines.
        let base = "L1\nL2\nL3\nL4\n| A | B |\n";
        let mut inc = IncrementalDoc::new(base);
        let end = inc.source().len();
        apply_text_patch(&mut inc, end, end, "| --- | --- |");
        assert_eq!(inc.blocks(), parse_blocks(inc.source()).as_slice());
        assert!(inc.blocks().iter().any(|b| b.kind == BlockKind::Table));
    }
}
