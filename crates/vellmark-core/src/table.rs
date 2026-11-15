//! Canonical GFM pipe-table parsing and enclosing-table discovery.
//!
//! This module serves as the single source of truth for GFM §4.10 table recognition across
//! `vellmark-core` (editing/formatting), `vellmark-parser` (block AST), and `vellmark-render` (WYSIWYG shaping).

use serde::{Deserialize, Serialize};

use crate::{ByteOffset, ByteRange};

/// Column alignment parsed from a GFM table delimiter row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum TableAlign {
    #[default]
    Left,
    Center,
    Right,
}

/// Source byte spans for a single parsed cell inside a table row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableCellSpan {
    /// Byte range in `source` of the raw cell slice between surrounding unescaped `|` pipes.
    pub raw_range: ByteRange,
    /// Trimmed content byte range in `source` (or insertion point after `"| "` for empty space cells).
    pub content_range: ByteRange,
}

/// A single row parsed from a GFM pipe table block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableRowParsed {
    /// Full line byte range in `source` (excluding trailing `\r`/`\n`).
    pub line_range: ByteRange,
    /// Whether this row is the canonical GFM delimiter row (only possible at row index 1).
    pub is_delimiter: bool,
    /// Per-column alignments parsed when `is_delimiter` is true.
    pub alignments: Vec<TableAlign>,
    /// Parsed cell slices between unescaped pipes.
    pub cells: Vec<TableCellSpan>,
}

/// Complete GFM table scanned by [`try_scan_gfm_table_from_line`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScannedGfmTable {
    /// Full byte range of the table from row 0 start to the last row end (excluding trailing `\r`/`\n`).
    pub range: ByteRange,
    /// Byte offset immediately following the last row's `\n` (or `source.len()`).
    pub next_pos: usize,
    /// Parsed rows of the table (`rows[0]` header, `rows[1]` delimiter, `rows[2..]` body).
    pub rows: Vec<TableRowParsed>,
}

/// Canonical GFM table enclosing a queried caret offset or line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GfmTableContext {
    /// Full byte range of the table in `source` (excluding trailing `\r`/`\n`).
    pub range: ByteRange,
    /// Canonical column count defined by the delimiter row (`rows[1].cells.len()`).
    pub delim_cols: usize,
    /// Parsed rows of the table.
    pub rows: Vec<TableRowParsed>,
    /// Index into `rows` containing the queried byte offset.
    pub row_idx: usize,
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

/// Return `(start, end_before_lf, next_pos_after_lf)` for the line starting at `start`.
pub fn scan_line_bounds(bytes: &[u8], start: usize) -> (usize, usize, usize) {
    let mut end = start;
    while end < bytes.len() && bytes[end] != b'\n' {
        end += 1;
    }
    let next = if end < bytes.len() { end + 1 } else { end };
    (start, end, next)
}

/// Return `(start, content_end_without_cr_or_lf, next_pos_after_lf)` for the line starting at `start`.
pub fn scan_line_content_bounds(bytes: &[u8], start: usize) -> (usize, usize, usize) {
    let (_, mut end, next) = scan_line_bounds(bytes, start);
    while end > start && bytes[end - 1] == b'\r' {
        end -= 1;
    }
    (start, end, next)
}

/// Parse an ATX heading prefix (`# `..`###### `), returning `(level, prefix_byte_len)`.
pub fn parse_atx_heading_prefix(line: &str) -> Option<(u8, usize)> {
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

/// Parse a fenced code block opening line (CommonMark §4.5), returning `(fence_byte, run_len, info)`.
pub fn parse_opening_fence_line(line: &str) -> Option<(u8, usize, String)> {
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

/// Whether `line` closes a fenced code block started with `(fence_char, min_len)`.
pub fn is_closing_fence_line(line: &str, fence_char: u8, min_len: usize) -> bool {
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

/// Parse a list item marker, returning `(ordered, indent_bytes, task_checked)`.
pub fn parse_list_item_kind(line: &str) -> Option<(bool, usize, Option<bool>)> {
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

/// Whether `line` starts a block construct that interrupts a GFM table.
pub fn is_table_interrupting_block_opener(line: &str) -> bool {
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

/// Parse delimiter-row column alignments for pre-split `cell_ranges`.
pub fn parse_table_delimiter_alignments_for_cells(
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

/// Whether a single line (without trailing `\r`/`\n`) is a Markdown pipe-table row candidate.
pub fn is_pipe_table_line(line: &str) -> bool {
    parse_table_cell_rel_ranges(line).is_some()
}

/// Whether a line can appear as a body row inside an open GFM table.
pub fn is_table_body_row_candidate(line: &str) -> bool {
    !is_table_interrupting_block_opener(line) && parse_table_cell_rel_ranges(line).is_some()
}

/// Parse per-column alignment markers from a delimiter row (`| --- | :---: | ---: |`, outer pipes optional).
pub fn parse_table_delimiter_alignments(line: &str, _pipes: &[usize]) -> Option<Vec<TableAlign>> {
    let cells = parse_table_cell_rel_ranges(line)?;
    parse_table_delimiter_alignments_for_cells(line, &cells)
}

fn build_parsed_table_row(
    source: &str,
    l_start: usize,
    content_end: usize,
    allow_delimiter: bool,
) -> TableRowParsed {
    let line = &source[l_start..content_end];
    let cell_ranges = parse_table_cell_rel_ranges(line).unwrap_or_default();
    let alignments = if allow_delimiter {
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
    TableRowParsed {
        line_range: ByteRange::new(l_start, content_end),
        is_delimiter,
        alignments: alignments.unwrap_or_default(),
        cells,
    }
}

/// Attempt to parse a complete GFM table starting with header line `[line_start, line_end)`.
///
/// Requires a non-delimiter header row followed immediately by a delimiter row with matching
/// column count, then consumes zero or more contiguous non-interrupting body row candidates.
pub fn try_scan_gfm_table_from_line(
    source: &str,
    bytes: &[u8],
    line_start: usize,
    line_end: usize,
    next_pos: usize,
) -> Option<ScannedGfmTable> {
    let mut header_content_end = line_end.min(source.len());
    while header_content_end > line_start && bytes[header_content_end - 1] == b'\r' {
        header_content_end -= 1;
    }
    let header_raw = &source[line_start..header_content_end];
    if is_table_interrupting_block_opener(header_raw) {
        return None;
    }
    let header_cells = parse_table_cell_rel_ranges(header_raw)?;
    if parse_table_delimiter_alignments_for_cells(header_raw, &header_cells).is_some() {
        return None;
    }
    if next_pos >= bytes.len() {
        return None;
    }

    let (delim_start, delim_end, delim_next_pos) = scan_line_content_bounds(bytes, next_pos);
    let delim_raw = &source[delim_start..delim_end];
    if is_table_interrupting_block_opener(delim_raw) {
        return None;
    }
    let delim_cells = parse_table_cell_rel_ranges(delim_raw)?;
    let delim_aligns = parse_table_delimiter_alignments_for_cells(delim_raw, &delim_cells)?;
    if delim_aligns.len() != header_cells.len() {
        return None;
    }

    let mut rows = vec![
        build_parsed_table_row(source, line_start, header_content_end, false),
        build_parsed_table_row(source, delim_start, delim_end, true),
    ];
    let mut table_end = delim_end;
    let mut table_next_pos = delim_next_pos;
    let mut table_pos = delim_next_pos;

    while table_pos < bytes.len() {
        let (b_start, b_end, b_next) = scan_line_content_bounds(bytes, table_pos);
        let raw_line = &source[b_start..b_end];
        if !is_table_body_row_candidate(raw_line) {
            break;
        }
        rows.push(build_parsed_table_row(source, b_start, b_end, false));
        table_end = b_end;
        table_next_pos = b_next;
        table_pos = b_next;
    }

    Some(ScannedGfmTable {
        range: ByteRange::new(line_start, table_end),
        next_pos: table_next_pos,
        rows,
    })
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

        // Per GFM 0.29-gfm, only row index 1 (immediately beneath header row 0) is the delimiter row.
        rows.push(build_parsed_table_row(
            source,
            l_start,
            content_end,
            rows.len() == 1,
        ));
        pos = next_pos;
    }

    rows
}

/// Find the canonical GFM table enclosing `byte_offset` in `O(contiguous candidate lines)` time.
pub fn find_enclosing_gfm_table_at(
    source: &str,
    byte_offset: ByteOffset,
) -> Option<GfmTableContext> {
    let bytes = source.as_bytes();
    let pos = byte_offset.min(bytes.len());
    let line_start = bytes[..pos]
        .iter()
        .rposition(|&b| b == b'\n')
        .map_or(0, |i| i + 1);
    let (cur_start, cur_end, _) = scan_line_content_bounds(bytes, line_start);
    if !is_table_body_row_candidate(&source[cur_start..cur_end]) {
        return None;
    }

    // Walk backward across contiguous candidate lines so top-down table parsing
    // matches `parse_blocks` even when a non-table pipe line immediately precedes row 0.
    let mut run_start = cur_start;
    while run_start > 0 {
        let prev_lf = run_start - 1;
        let prev_start = bytes[..prev_lf]
            .iter()
            .rposition(|&b| b == b'\n')
            .map_or(0, |i| i + 1);
        let (_, prev_end, _) = scan_line_content_bounds(bytes, prev_start);
        if !is_table_body_row_candidate(&source[prev_start..prev_end]) {
            break;
        }
        run_start = prev_start;
    }

    let mut scan_pos = run_start;
    while scan_pos <= cur_start && scan_pos < bytes.len() {
        let (l_start, l_end, l_next) = scan_line_content_bounds(bytes, scan_pos);
        if l_start > cur_start {
            break;
        }
        if let Some(scanned) = try_scan_gfm_table_from_line(source, bytes, l_start, l_end, l_next) {
            if let Some(row_idx) = scanned
                .rows
                .iter()
                .position(|r| r.line_range.start == cur_start)
            {
                let delim_cols = scanned.rows[1].cells.len();
                return Some(GfmTableContext {
                    range: scanned.range,
                    delim_cols,
                    rows: scanned.rows,
                    row_idx,
                });
            }
            if scanned.next_pos <= scan_pos {
                break;
            }
            scan_pos = scanned.next_pos;
        } else {
            if l_next <= scan_pos {
                break;
            }
            scan_pos = l_next;
        }
    }

    None
}
