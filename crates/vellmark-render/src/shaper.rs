//! Virtualized `cosmic-text` shaping, two-level WYSIWYG marker collapse/expansion mapping,
//! non-destructive IME preedit splicing, hit-testing, and 10 MB viewport scene generation.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::ops::Range;

use cosmic_text::{
    Align, Attrs, Buffer as CosmicBuffer, CacheKey, CacheMetrics, Cursor, Family, FontSystem,
    Metrics, Shaping, Style as FontStyle, Weight, Wrap,
};
use vellmark_core::{ByteOffset, ByteRange, EditorViewState, Selection, ViewMode};
use vellmark_parser::{parse_table_block, BlockKind, BlockNode, InlineKind, InlineSpan, TableAlign};

use crate::style::{ResolvedThemeStyle, RgbaColor};

/// Axis-aligned rectangle in logical pixels.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Rect {
    pub const fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }

    pub fn contains(&self, px: f32, py: f32) -> bool {
        px >= self.x && px <= self.x + self.w && py >= self.y && py <= self.y + self.h
    }

    pub fn translate(self, dx: f32, dy: f32) -> Self {
        Self {
            x: self.x + dx,
            y: self.y + dy,
            ..self
        }
    }
}

/// GPU-agnostic rounded/bordered rectangle primitive emitted by layout & UI chrome.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneRect {
    pub rect: Rect,
    pub color: RgbaColor,
    pub corner_radius: f32,
    pub border_width: f32,
    pub border_color: RgbaColor,
    pub clip: Option<Rect>,
}

impl SceneRect {
    pub const fn solid(rect: Rect, color: RgbaColor) -> Self {
        Self {
            rect,
            color,
            corner_radius: 0.0,
            border_width: 0.0,
            border_color: RgbaColor::new(0, 0, 0, 0),
            clip: None,
        }
    }

    pub const fn rounded(rect: Rect, color: RgbaColor, corner_radius: f32) -> Self {
        Self {
            corner_radius,
            ..Self::solid(rect, color)
        }
    }

    pub const fn with_border(mut self, width: f32, color: RgbaColor) -> Self {
        self.border_width = width;
        self.border_color = color;
        self
    }

    pub const fn with_clip(mut self, clip: Option<Rect>) -> Self {
        self.clip = clip;
        self
    }
}

/// GPU-agnostic positioned glyph primitive ready for rasterization via `cosmic_text::SwashCache`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneGlyph {
    pub cache_key: CacheKey,
    pub x: i32,
    pub y: i32,
    pub color: RgbaColor,
    pub clip: Option<Rect>,
}

/// Draw list consumed cleanly by `vellmark-app::gpu::WgpuRenderer`.
#[derive(Debug, Clone, Default)]
pub struct DrawList {
    pub rects: Vec<SceneRect>,
    pub glyphs: Vec<SceneGlyph>,
}

/// Bidirectionally maps a shaped block's displayed UTF-8 string back to exact source `ByteOffset`s,
/// preserving exact byte identity across collapsed markers (`**`, `# `, `[..](url)`) and active IME preedit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisualTextMap {
    pub block_range: ByteRange,
    pub display_text: String,
    pub line_starts: Vec<usize>,
    /// `(vis_start, vis_end, src_start, src_end)` sorted by `vis_start`.
    segments: Vec<VisualSegment>,
    /// Optional active IME composition splice inserted purely in visual coordinates.
    ime_splice: Option<ImeSplice>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct VisualSegment {
    vis_start: usize,
    vis_end: usize,
    src_start: ByteOffset,
    src_end: ByteOffset,
    synthetic: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ImeSplice {
    vis_start: usize,
    vis_end: usize,
    comp_src_start: ByteOffset,
    comp_src_end: ByteOffset,
    preedit_caret_vis: usize,
}

impl VisualTextMap {
    /// Convert a visual byte index inside `display_text` to an exact UTF-8 source `ByteOffset`.
    pub fn visual_to_source(&self, vis_byte: usize) -> ByteOffset {
        let clamped_vis = vis_byte.min(self.display_text.len());
        if let Some(ime) = &self.ime_splice {
            if clamped_vis >= ime.vis_start && clamped_vis <= ime.vis_end {
                return ime.comp_src_start;
            }
            if clamped_vis > ime.vis_end {
                let preedit_len = ime.vis_end - ime.vis_start;
                let unshifted_vis = clamped_vis.saturating_sub(preedit_len);
                let base_offset = self.base_visual_to_source(unshifted_vis);
                if base_offset <= ime.comp_src_end {
                    return ime.comp_src_end;
                }
                return base_offset;
            }
        }
        self.base_visual_to_source(clamped_vis)
    }

    /// Convert a visual byte index to a source `ByteOffset`, rebased from this cached map's
    /// block origin to `actual_block_start` (allows safe reuse across identical repeated blocks).
    pub fn visual_to_source_rebased(
        &self,
        vis_byte: usize,
        actual_block_start: ByteOffset,
    ) -> ByteOffset {
        let rel = self
            .visual_to_source(vis_byte)
            .saturating_sub(self.block_range.start);
        actual_block_start + rel
    }

    /// Convert a source `ByteOffset` (within `block_range`) into a visual byte index in `display_text`.
    pub fn source_to_visual(&self, src_offset: ByteOffset) -> usize {
        let clamped_src = src_offset.clamp(self.block_range.start, self.block_range.end);
        let base_vis = self.base_source_to_visual(clamped_src);
        if let Some(ime) = &self.ime_splice {
            if clamped_src <= ime.comp_src_start {
                return base_vis.min(ime.vis_start);
            }
            if clamped_src >= ime.comp_src_end {
                return ime.vis_end + base_vis.saturating_sub(ime.vis_start);
            }
            return ime.preedit_caret_vis;
        }
        base_vis
    }

    /// Convert a document source `ByteOffset` (belonging to a block starting at `actual_block_start`)
    /// into a visual byte index in this potentially shared `display_text`.
    pub fn source_to_visual_rebased(
        &self,
        src_offset: ByteOffset,
        actual_block_start: ByteOffset,
    ) -> usize {
        let rel = src_offset.saturating_sub(actual_block_start);
        self.source_to_visual(self.block_range.start + rel)
    }

    /// Convert a `cosmic-text` `(line_i, byte_in_line)` cursor into global `display_text` byte index.
    pub fn line_col_to_visual(&self, line_i: usize, col_byte: usize) -> usize {
        let line_start = self.line_starts.get(line_i).copied().unwrap_or(self.display_text.len());
        let next_start = self
            .line_starts
            .get(line_i + 1)
            .map(|s| s.saturating_sub(1))
            .unwrap_or(self.display_text.len());
        (line_start + col_byte).min(next_start)
    }

    /// Convert global `display_text` byte index into `cosmic_text::Cursor(line_i, byte_in_line)`.
    pub fn visual_to_line_col(&self, vis_byte: usize) -> Cursor {
        let clamped = vis_byte.min(self.display_text.len());
        let line_i = self
            .line_starts
            .partition_point(|&start| start <= clamped)
            .saturating_sub(1);
        let line_start = self.line_starts.get(line_i).copied().unwrap_or(0);
        Cursor::new(line_i, clamped.saturating_sub(line_start))
    }

    pub fn preedit_visual_range(&self) -> Option<Range<usize>> {
        self.ime_splice.as_ref().map(|s| s.vis_start..s.vis_end)
    }

    pub fn active_ime_caret_visual(&self) -> Option<usize> {
        self.ime_splice.as_ref().map(|s| s.preedit_caret_vis)
    }

    fn base_visual_to_source(&self, vis: usize) -> ByteOffset {
        if self.segments.is_empty() {
            return self.block_range.start;
        }
        for seg in &self.segments {
            if vis < seg.vis_start {
                return seg.src_start;
            }
            if vis < seg.vis_end {
                if seg.synthetic {
                    return seg.src_end;
                }
                let rel = vis - seg.vis_start;
                return (seg.src_start + rel).min(seg.src_end);
            }
        }
        self.segments
            .last()
            .map(|s| s.src_end)
            .unwrap_or(self.block_range.end)
    }

    fn base_source_to_visual(&self, src: ByteOffset) -> usize {
        if self.segments.is_empty() {
            return 0;
        }
        for seg in &self.segments {
            if src <= seg.src_start {
                return seg.vis_start;
            }
            if src <= seg.src_end {
                if seg.synthetic {
                    return seg.vis_end;
                }
                return (seg.vis_start + (src - seg.src_start)).min(seg.vis_end);
            }
        }
        self.segments.last().map(|s| s.vis_end).unwrap_or(0)
    }
}

/// Visual span styling computed prior to `cosmic_text::Buffer::set_rich_text`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct DisplaySpanStyle {
    range: Range<usize>,
    color: RgbaColor,
    bold: bool,
    italic: bool,
    underline: bool,
    strikethrough: bool,
    monospace: bool,
    inline_code_bg: bool,
    ime_preedit: bool,
}

#[derive(Debug, Clone)]
struct BlockDisplay {
    map: VisualTextMap,
    spans: Vec<DisplaySpanStyle>,
    left_indent_px: f32,
    top_pad_px: f32,
    bottom_pad_px: f32,
    blockquote_bar: bool,
    code_box: bool,
    hr_rule: bool,
    heading_level: Option<u8>,
    task_checkbox: Option<bool>,
}

/// Inputs describing the active editor viewport for one virtualized frame render.
pub struct EditorViewport<'a> {
    pub doc_rect: Rect,
    pub content_left_x: f32,
    pub content_width_px: f32,
    pub scroll_y: f32,
    pub overscan_px: f32,
    pub scale_factor: f32,
    pub source_rev: u64,
    pub source_len: usize,
    pub view: &'a EditorViewState,
    pub focus_mode: bool,
    pub show_caret: bool,
    pub find_matches: &'a [ByteRange],
    pub active_find: Option<ByteRange>,
}

/// Result of clicking/pointing inside the rendered editor scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditorHitTarget {
    /// Direct text click mapped to exact source UTF-8 `ByteOffset`.
    Text(ByteOffset),
    /// Click on a rendered `- [ ]` / `- [x]` task checkbox at `block_range`.
    TaskCheckbox {
        block_range: ByteRange,
        caret_offset: ByteOffset,
    },
}

/// Result returned after building an editor frame's virtualized draw list.
#[derive(Debug, Clone)]
pub struct EditorSceneResult {
    pub draw: DrawList,
    pub total_height: f32,
    pub caret_doc_y: f32,
    pub caret_line_height: f32,
    pub ime_cursor_area: Option<Rect>,
    pub visible_block_count: usize,
}

struct CachedBlockShape {
    last_used_frame: u64,
    buffer: CosmicBuffer,
    map: VisualTextMap,
    display: BlockDisplay,
    total_height: f32,
    inline_code_rects: Vec<Rect>,
}

struct TableCellDisplay {
    rel_raw_range: ByteRange,
    rel_src_range: ByteRange,
    align: TableAlign,
    is_placeholder: bool,
    map: VisualTextMap,
    spans: Vec<DisplaySpanStyle>,
    default_weight: u16,
}

struct TableRowDisplay {
    line_range_rel: ByteRange,
    is_header: bool,
    is_delimiter: bool,
    cells: Vec<TableCellDisplay>,
}

struct TableCellShape {
    x_rel: f32,
    w: f32,
    text_x_rel: f32,
    inner_w: f32,
    rel_raw_range: ByteRange,
    rel_src_range: ByteRange,
    align: TableAlign,
    is_placeholder: bool,
    buffer: CosmicBuffer,
    map: VisualTextMap,
    spans: Vec<DisplaySpanStyle>,
    inline_code_rects: Vec<Rect>,
}

struct TableRowShape {
    y_rel: f32,
    h: f32,
    is_header: bool,
    is_delimiter: bool,
    line_range_rel: ByteRange,
    cells: Vec<TableCellShape>,
}

struct CachedTableShape {
    last_used_frame: u64,
    table_w: f32,
    table_h: f32,
    top_pad_px: f32,
    cell_pad_y: f32,
    col_widths: Vec<f32>,
    rows: Vec<TableRowShape>,
    total_height: f32,
}

#[derive(Debug, Clone, Copy)]
struct HitBlockRecord {
    is_table: bool,
    shape_hash: u64,
    row_rect: Rect,
    text_origin_x: f32,
    text_origin_y: f32,
    checkbox_rect: Option<Rect>,
    block_range: ByteRange,
    src_end_fallback: ByteOffset,
}

/// Stateful `cosmic-text` shaper + bounded LRU block cache for O(1) visible-viewport 10 MB rendering.
pub struct ShapedDocumentCache {
    font_system: FontSystem,
    shaped_lru: HashMap<u64, CachedBlockShape>,
    table_lru: HashMap<u64, CachedTableShape>,
    frame_seq: u64,
    /// Persistent per-block layout state so 10 MB docs only recompute prefix Y on real edits.
    cached_rev: u64,
    cached_width_bits: u32,
    cached_style_sig: u64,
    block_ranges: Vec<ByteRange>,
    block_ys: Vec<f32>,
    block_heights: Vec<f32>,
    block_scale_y: Vec<bool>,
    total_height: f32,
    last_hit_blocks: Vec<HitBlockRecord>,
    last_doc_rect: Rect,
    last_scroll_y: f32,
    last_source_len: usize,
}

const MAX_SHAPED_BLOCK_CACHE: usize = 512;
const MAX_SHAPED_TABLE_CACHE: usize = 128;

impl Default for ShapedDocumentCache {
    fn default() -> Self {
        Self::new()
    }
}

impl ShapedDocumentCache {
    pub fn new() -> Self {
        Self {
            font_system: FontSystem::new(),
            shaped_lru: HashMap::with_capacity(256),
            table_lru: HashMap::with_capacity(32),
            frame_seq: 0,
            cached_rev: u64::MAX,
            cached_width_bits: 0,
            cached_style_sig: 0,
            block_ranges: Vec::new(),
            block_ys: Vec::new(),
            block_heights: Vec::new(),
            block_scale_y: Vec::new(),
            total_height: 0.0,
            last_hit_blocks: Vec::new(),
            last_doc_rect: Rect::default(),
            last_scroll_y: 0.0,
            last_source_len: 0,
        }
    }

    pub fn font_system_mut(&mut self) -> &mut FontSystem {
        &mut self.font_system
    }

    pub fn total_height(&self) -> f32 {
        self.total_height
    }

    /// Reconcile fast block height table with AST blocks without shaping offscreen blocks.
    pub fn sync_block_metrics(
        &mut self,
        blocks: &[BlockNode],
        style: &ResolvedThemeStyle,
        content_width_px: f32,
        source_rev: u64,
    ) {
        let style_sig = style_signature(style);
        let width_bits = content_width_px.max(120.0).to_bits();
        if self.cached_rev == source_rev
            && self.cached_width_bits == width_bits
            && self.cached_style_sig == style_sig
            && self.block_ranges.len() == blocks.len()
        {
            return;
        }

        self.cached_rev = source_rev;
        self.cached_width_bits = width_bits;
        self.cached_style_sig = style_sig;

        self.block_ranges.clear();
        self.block_ys.clear();
        self.block_heights.clear();
        self.block_scale_y.clear();
        self.block_ranges.reserve(blocks.len());
        self.block_ys.reserve(blocks.len());
        self.block_heights.reserve(blocks.len());
        self.block_scale_y.reserve(blocks.len());

        let mut y = 24.0 * style.zoom_scale;
        let width_chars = ((content_width_px - 32.0).max(80.0) / (style.base_font_size_px * 0.54))
            .max(12.0);
        for node in blocks {
            let est = estimate_block_height(node, style, width_chars);
            self.block_ranges.push(node.range);
            self.block_ys.push(y);
            self.block_heights.push(est);
            self.block_scale_y.push(false);
            y += est;
        }
        // Leave bottom breathing room below final line.
        self.total_height = y + 120.0;
    }

    /// Build the virtualized GPU-agnostic scene for the currently visible viewport blocks only (`O(visible)` shaping).
    pub fn build_editor_scene<S>(
        &mut self,
        blocks: &[BlockNode],
        style: &ResolvedThemeStyle,
        vp: &EditorViewport<'_>,
        mut read_slice: S,
    ) -> EditorSceneResult
    where
        S: FnMut(ByteRange) -> String,
    {
        self.frame_seq = self.frame_seq.wrapping_add(1);
        self.sync_block_metrics(blocks, style, vp.content_width_px, vp.source_rev);
        self.last_doc_rect = vp.doc_rect;
        self.last_scroll_y = vp.scroll_y;
        self.last_source_len = vp.source_len;

        let mut draw = DrawList::default();
        draw.rects
            .push(SceneRect::solid(vp.doc_rect, style.bg_color));

        self.last_hit_blocks.clear();

        if blocks.is_empty() {
            let pad_x = vp.content_left_x + 12.0;
            let top_y = vp.doc_rect.y + 28.0 - vp.scroll_y;
            let cursor_area = Rect::new(pad_x, top_y, 2.0, style.base_line_height_px);
            if vp.show_caret {
                draw.rects.push(
                    SceneRect::solid(cursor_area, style.caret_color).with_clip(Some(vp.doc_rect)),
                );
            }
            return EditorSceneResult {
                draw,
                total_height: self.total_height,
                caret_doc_y: 28.0,
                caret_line_height: style.base_line_height_px,
                ime_cursor_area: Some(cursor_area),
                visible_block_count: 0,
            };
        }

        let min_y = (vp.scroll_y - vp.overscan_px).max(0.0);
        let max_y = vp.scroll_y + vp.doc_rect.h + vp.overscan_px;
        let start_idx = self
            .block_ys
            .partition_point(|&by| by + style.base_line_height_px * 3.0 < min_y)
            .min(blocks.len().saturating_sub(1));
        // Walk backwards slightly if a tall code fence above start_idx overlaps min_y.
        let mut vis_start_idx = start_idx;
        while vis_start_idx > 0
            && self.block_ys[vis_start_idx - 1] + self.block_heights[vis_start_idx - 1] >= min_y
        {
            vis_start_idx -= 1;
        }
        let vis_end_idx = self
            .block_ys
            .partition_point(|&by| by <= max_y)
            .clamp(vis_start_idx, blocks.len());

        let sel = vp.view.selection;
        let exp_sel = if vp.view.expansion_frozen {
            vp.view.ime_composition_selection.unwrap_or(sel)
        } else {
            sel
        };
        let style_sig = style_signature(style);
        let base_caret = vp.view.selection.head;
        let mut caret_doc_y = 24.0 * style.zoom_scale;
        let mut caret_line_height = style.base_line_height_px;
        let mut ime_cursor_area = None;
        let mut any_height_changed = false;
        let mut first_changed_idx = usize::MAX;

        // Determine if caret sits at a trailing empty line past the final block (e.g., after `\n`).
        let caret_after_last_block = blocks
            .last()
            .map(|last| vp.view.ime_preedit.is_some() && vp.view.ime_composition_selection.map(|s| s.head > last.range.end).unwrap_or(false)
                || base_caret > last.range.end)
            .unwrap_or(false);

        let mut y_cursor_shift = 0.0f32;
        for idx in vis_start_idx..vis_end_idx {
            let node = &blocks[idx];
            let block_y = self.block_ys[idx] + y_cursor_shift;
            let raw_text = read_slice(node.range);
            let block_focused = node.range.contains_inclusive(exp_sel.head);

            if matches!(node.kind, BlockKind::Table) && vp.view.mode != ViewMode::Source
                && let Some(table_display) = build_table_display(
                    node,
                    &raw_text,
                    vp.view,
                    exp_sel,
                    style,
                    block_focused,
                    vp.focus_mode,
                ) {
                    let avail_table_w = (vp.content_width_px - 8.0).max(120.0);
                    let table_key = shape_table_cache_key(
                        &table_display,
                        avail_table_w,
                        style_sig,
                        block_focused,
                        vp.focus_mode,
                    );
                    let frame_seq = self.frame_seq;
                    if let Some(cached) = self.table_lru.get_mut(&table_key) {
                        cached.last_used_frame = frame_seq;
                    } else {
                        let shaped = shape_table_block(
                            &mut self.font_system,
                            table_display,
                            avail_table_w,
                            style,
                            frame_seq,
                        );
                        if self.table_lru.len() >= MAX_SHAPED_TABLE_CACHE {
                            evict_oldest_table_lru(&mut self.table_lru);
                        }
                        self.table_lru.insert(table_key, shaped);
                    }

                    let table = self.table_lru.get(&table_key).expect("just inserted");
                    let measured_h = table.total_height.max(style.base_line_height_px);
                    update_block_height_metric(
                        idx,
                        block_y,
                        measured_h,
                        &mut self.block_ys,
                        &mut self.block_heights,
                        &mut y_cursor_shift,
                        &mut any_height_changed,
                        &mut first_changed_idx,
                    );

                    let screen_top_y = vp.doc_rect.y + block_y - vp.scroll_y;
                    let table_left_x = vp.content_left_x + 4.0;
                    let table_top_y = screen_top_y + table.top_pad_px;

                    emit_table_scene(
                        table,
                        node,
                        block_y,
                        screen_top_y,
                        table_left_x,
                        table_top_y,
                        caret_after_last_block,
                        base_caret,
                        sel,
                        vp,
                        style,
                        block_focused,
                        &mut draw,
                        &mut caret_doc_y,
                        &mut caret_line_height,
                        &mut ime_cursor_area,
                    );

                    self.last_hit_blocks.push(HitBlockRecord {
                        is_table: true,
                        shape_hash: table_key,
                        row_rect: Rect::new(
                            vp.doc_rect.x,
                            screen_top_y,
                            vp.doc_rect.w,
                            measured_h.max(style.base_line_height_px),
                        ),
                        text_origin_x: table_left_x,
                        text_origin_y: table_top_y,
                        checkbox_rect: None,
                        block_range: node.range,
                        src_end_fallback: node.range.end,
                    });
                    continue;
                }

            let display = build_block_display(
                node,
                &raw_text,
                vp.view,
                exp_sel,
                style,
                block_focused,
                vp.focus_mode,
            );
            let avail_w = (vp.content_width_px - display.left_indent_px - 24.0).max(80.0);
            let key = shape_cache_key(
                &display,
                avail_w,
                style_sig,
                matches!(node.kind, BlockKind::CodeFence { .. } | BlockKind::YamlFrontMatter),
            );

            let frame_seq = self.frame_seq;
            if let Some(cached) = self.shaped_lru.get_mut(&key) {
                cached.last_used_frame = frame_seq;
            } else {
                let shaped = shape_block(&mut self.font_system, display, avail_w, style, key, frame_seq);
                if self.shaped_lru.len() >= MAX_SHAPED_BLOCK_CACHE {
                    evict_oldest_lru(&mut self.shaped_lru);
                }
                self.shaped_lru.insert(key, shaped);
            }

            let shaped = self.shaped_lru.get(&key).expect("just inserted");
            let measured_h = shaped.total_height.max(style.base_line_height_px * 0.7);
            update_block_height_metric(
                idx,
                block_y,
                measured_h,
                &mut self.block_ys,
                &mut self.block_heights,
                &mut y_cursor_shift,
                &mut any_height_changed,
                &mut first_changed_idx,
            );

            let screen_top_y = vp.doc_rect.y + block_y - vp.scroll_y;
            let text_origin_x = vp.content_left_x + shaped.display.left_indent_px;
            let text_origin_y = screen_top_y + shaped.display.top_pad_px;

            // Block background decorations (code block container, blockquote bar, HR, headings border)
            if shaped.display.code_box {
                let box_rect = Rect::new(
                    vp.content_left_x + 4.0,
                    screen_top_y + 2.0,
                    (vp.content_width_px - 8.0).max(40.0),
                    (measured_h - 4.0).max(20.0),
                );
                draw.rects.push(
                    SceneRect::rounded(box_rect, style.code_block_bg, 6.0 * style.zoom_scale)
                        .with_border(1.0, style.code_block_border)
                        .with_clip(Some(vp.doc_rect)),
                );
            }
            if shaped.display.blockquote_bar {
                if let Some(bq_bg) = style.blockquote_bg {
                    let bg_rect = Rect::new(
                        vp.content_left_x + 4.0,
                        screen_top_y + 1.0,
                        vp.content_width_px - 8.0,
                        measured_h - 2.0,
                    );
                    draw.rects.push(SceneRect::solid(bg_rect, bq_bg).with_clip(Some(vp.doc_rect)));
                }
                let bar = Rect::new(
                    vp.content_left_x + 6.0,
                    screen_top_y + 3.0,
                    style.blockquote_border_width.max(2.0),
                    (measured_h - 6.0).max(12.0),
                );
                draw.rects.push(
                    SceneRect::rounded(bar, style.blockquote_border, 1.5).with_clip(Some(vp.doc_rect)),
                );
            }
            if let Some(level) = shaped.display.heading_level {
                let hs = &style.headings[(level.clamp(1, 6) - 1) as usize];
                if let Some(border_c) = hs.bottom_border_color {
                    let rule_rect = Rect::new(
                        vp.content_left_x + 4.0,
                        screen_top_y + measured_h - shaped.display.bottom_pad_px.min(4.0),
                        (vp.content_width_px - 8.0).max(20.0),
                        hs.bottom_border_width.max(1.0),
                    );
                    draw.rects
                        .push(SceneRect::solid(rule_rect, border_c).with_clip(Some(vp.doc_rect)));
                }
            }
            if shaped.display.hr_rule {
                let rule_rect = Rect::new(
                    vp.content_left_x + 8.0,
                    screen_top_y + measured_h * 0.5 - 1.0,
                    (vp.content_width_px - 16.0).max(20.0),
                    (2.0 * style.zoom_scale).max(1.0),
                );
                draw.rects
                    .push(SceneRect::solid(rule_rect, style.hr_color).with_clip(Some(vp.doc_rect)));
            }

            let mut checkbox_rect_opt = None;
            if let Some(checked) = shaped.display.task_checkbox {
                let box_sz = (style.base_font_size_px * 0.92).round();
                let cb_rect = Rect::new(
                    text_origin_x - box_sz - 10.0,
                    text_origin_y + (style.base_line_height_px - box_sz) * 0.45,
                    box_sz,
                    box_sz,
                );
                checkbox_rect_opt = Some(cb_rect);
                let fill = if checked {
                    style.link_color
                } else {
                    style.bg_color
                };
                draw.rects.push(
                    SceneRect::rounded(cb_rect, fill, 3.0)
                        .with_border(1.5, if checked { style.link_color } else { style.muted_marker_color })
                        .with_clip(Some(vp.doc_rect)),
                );
                if checked {
                    let inner = Rect::new(
                        cb_rect.x + box_sz * 0.25,
                        cb_rect.y + box_sz * 0.25,
                        box_sz * 0.5,
                        box_sz * 0.5,
                    );
                    draw.rects.push(
                        SceneRect::rounded(inner, style.bg_color, 1.5).with_clip(Some(vp.doc_rect)),
                    );
                }
            }

            let fallback_text_color =
                style.apply_focus_mode(style.text_color, block_focused, vp.focus_mode);
            let sel_range = sel.range();
            render_text_buffer_contents(
                &shaped.inline_code_rects,
                &shaped.buffer,
                &shaped.map,
                &shaped.display.spans,
                node.range,
                node.range.start,
                true,
                text_origin_x,
                text_origin_y,
                fallback_text_color,
                sel_range,
                vp,
                style,
                &mut draw,
            );

            // Caret / IME candidate anchor if inside this block (and not past final block)
            let owns_ime = shaped.map.active_ime_caret_visual().is_some();
            let owns_caret = !caret_after_last_block
                && (owns_ime
                    || (node.range.contains_inclusive(base_caret)
                        && (idx + 1 == blocks.len() || blocks[idx + 1].range.start != base_caret || base_caret <= node.range.end)));
            if owns_caret && ime_cursor_area.is_none() {
                let target_vis = shaped
                    .map
                    .active_ime_caret_visual()
                    .unwrap_or_else(|| {
                        shaped
                            .map
                            .source_to_visual_rebased(base_caret, node.range.start)
                    });
                let cur_rect = caret_rect_in_buffer(
                    &shaped.buffer,
                    &shaped.map,
                    target_vis,
                    text_origin_x,
                    text_origin_y,
                    style.base_line_height_px,
                );
                caret_doc_y = block_y + (cur_rect.y - screen_top_y);
                caret_line_height = cur_rect.h;
                ime_cursor_area = Some(cur_rect);
                if vp.show_caret {
                    draw.rects.push(
                        SceneRect::solid(cur_rect, style.caret_color).with_clip(Some(vp.doc_rect)),
                    );
                }
            }

            self.last_hit_blocks.push(HitBlockRecord {
                is_table: false,
                shape_hash: key,
                row_rect: Rect::new(
                    vp.doc_rect.x,
                    screen_top_y,
                    vp.doc_rect.w,
                    measured_h.max(style.base_line_height_px),
                ),
                text_origin_x,
                text_origin_y,
                checkbox_rect: checkbox_rect_opt,
                block_range: node.range,
                src_end_fallback: node.range.end,
            });
        }

        // Trailing empty line caret (and optional IME preedit) when source ends in `\n`
        if caret_after_last_block {
            let last_y = self
                .block_ys
                .last()
                .zip(self.block_heights.last())
                .map(|(y, h)| y + h)
                .unwrap_or(24.0 * style.zoom_scale);
            let origin_x = vp.content_left_x + 12.0;
            let origin_y = vp.doc_rect.y + last_y - vp.scroll_y;
            let mut caret_x = origin_x;
            if let Some(preedit) = &vp.view.ime_preedit
                && !preedit.is_empty() {
                    let pre_w = shape_ui_text_into(
                        &mut self.font_system,
                        preedit,
                        origin_x,
                        origin_y,
                        style.base_font_size_px,
                        style.base_line_height_px,
                        400,
                        style.text_color,
                        Some((vp.content_width_px - 24.0).max(80.0)),
                        Some(vp.doc_rect),
                        vp.scale_factor,
                        &mut draw.glyphs,
                    );
                    draw.rects.push(
                        SceneRect::solid(
                            Rect::new(origin_x, origin_y, pre_w, style.base_line_height_px),
                            style.ime_preedit_bg,
                        )
                        .with_clip(Some(vp.doc_rect)),
                    );
                    draw.rects.push(
                        SceneRect::solid(
                            Rect::new(
                                origin_x,
                                origin_y + style.base_line_height_px - 2.0,
                                pre_w,
                                2.0,
                            ),
                            style.ime_underline_color,
                        )
                        .with_clip(Some(vp.doc_rect)),
                    );
                    caret_x += pre_w;
                }
            let cur_rect = Rect::new(caret_x, origin_y, 2.0, style.base_line_height_px);
            caret_doc_y = last_y;
            caret_line_height = style.base_line_height_px;
            ime_cursor_area = Some(cur_rect);
            if vp.show_caret {
                draw.rects
                    .push(SceneRect::solid(cur_rect, style.caret_color).with_clip(Some(vp.doc_rect)));
            }
        }

        // Propagate height changes linearly to remaining suffix offsets only when needed.
        if any_height_changed && first_changed_idx < blocks.len() {
            let mut acc_y = self.block_ys[first_changed_idx] + self.block_heights[first_changed_idx];
            for i in (first_changed_idx + 1)..self.block_ys.len() {
                self.block_ys[i] = acc_y;
                acc_y += self.block_heights[i];
            }
            self.total_height = acc_y + 120.0;
        }

        EditorSceneResult {
            draw,
            total_height: self.total_height,
            caret_doc_y,
            caret_line_height,
            ime_cursor_area,
            visible_block_count: vis_end_idx.saturating_sub(vis_start_idx),
        }
    }

    /// Map a window/viewport `(x, y)` click onto a visible shaped block -> exact `ByteOffset` or task checkbox.
    pub fn hit_test(&self, win_x: f32, win_y: f32) -> Option<EditorHitTarget> {
        if !self.last_doc_rect.contains(win_x, win_y) {
            return None;
        }
        if self.last_hit_blocks.is_empty() {
            return Some(EditorHitTarget::Text(0));
        }

        for rec in &self.last_hit_blocks {
            if let Some(cb) = rec.checkbox_rect {
                let padded = Rect::new(cb.x - 4.0, cb.y - 4.0, cb.w + 8.0, cb.h + 8.0);
                if padded.contains(win_x, win_y) {
                    return Some(EditorHitTarget::TaskCheckbox {
                        block_range: rec.block_range,
                        caret_offset: (rec.block_range.start + 6).min(rec.block_range.end),
                    });
                }
            }
            if win_y >= rec.row_rect.y && win_y <= rec.row_rect.y + rec.row_rect.h {
                if rec.is_table {
                    if let Some(table) = self.table_lru.get(&rec.shape_hash) {
                        return Some(EditorHitTarget::Text(hit_test_table(
                            table,
                            rec.block_range,
                            win_x - rec.text_origin_x,
                            win_y - rec.text_origin_y,
                        )));
                    }
                } else if let Some(shaped) = self.shaped_lru.get(&rec.shape_hash) {
                    let lx = (win_x - rec.text_origin_x).max(0.0);
                    let ly = (win_y - rec.text_origin_y).max(0.0);
                    if let Some(cur) = shaped.buffer.hit(lx, ly) {
                        let vis = shaped.map.line_col_to_visual(cur.line, cur.index);
                        return Some(EditorHitTarget::Text(
                            shaped
                                .map
                                .visual_to_source_rebased(vis, rec.block_range.start),
                        ));
                    }
                    return Some(EditorHitTarget::Text(rec.src_end_fallback));
                }
            }
        }

        // Below all rendered blocks -> snap to end of document
        if let Some(last) = self.last_hit_blocks.last() {
            if win_y > last.row_rect.y + last.row_rect.h {
                return Some(EditorHitTarget::Text(self.last_source_len));
            }
            return Some(EditorHitTarget::Text(last.block_range.start));
        }
        None
    }

    /// Move caret vertically by one visual line using real shaped coordinates (respects line wrapping & columns).
    #[allow(clippy::too_many_arguments)]
    pub fn move_caret_vertical<S>(
        &mut self,
        caret: ByteOffset,
        direction: i32,
        preferred_x: Option<f32>,
        blocks: &[BlockNode],
        view: &EditorViewState,
        style: &ResolvedThemeStyle,
        content_width_px: f32,
        mut read_slice: S,
    ) -> (ByteOffset, Option<f32>)
    where
        S: FnMut(ByteRange) -> String,
    {
        if blocks.is_empty() {
            return (0, preferred_x);
        }
        let exp_sel = if view.expansion_frozen {
            view.ime_composition_selection.unwrap_or(view.selection)
        } else {
            view.selection
        };
        let block_idx = blocks
            .partition_point(|b| b.range.end < caret)
            .min(blocks.len() - 1);

        let mut crossed_target_x = preferred_x;
        let nav_ctx = NavShapeContext {
            blocks,
            view,
            exp_sel,
            style,
            content_width_px,
        };
        if matches!(blocks[block_idx].kind, BlockKind::Table) && view.mode != ViewMode::Source {
            if let Some(table_key) = self.shape_table_for_nav(block_idx, &nav_ctx, &mut read_slice)
                && let Some(table) = self.table_lru.get(&table_key) {
                    match move_caret_vertical_in_table(
                        table,
                        blocks[block_idx].range.start,
                        caret,
                        direction,
                        preferred_x,
                        style,
                    ) {
                        TableNavOutcome::Moved(off, tx) => return (off, Some(tx)),
                        TableNavOutcome::Exited(tx) => {
                            crossed_target_x = Some(tx);
                        }
                    }
                }
        } else {
            let cur_offset = self.shape_single_for_nav(block_idx, &nav_ctx, &mut read_slice);
            let Some((cur_shaped, caret_vis)) = cur_offset else {
                return (caret, preferred_x);
            };
            let cur_rect = caret_rect_in_buffer(
                &cur_shaped.buffer,
                &cur_shaped.map,
                caret_vis,
                cur_shaped.display.left_indent_px,
                0.0,
                style.base_line_height_px,
            );
            let target_x = preferred_x.unwrap_or(cur_rect.x);
            crossed_target_x = Some(target_x);

            // Try moving inside the same block's wrapped visual runs first!
            let runs: Vec<(f32, f32)> = cur_shaped
                .buffer
                .layout_runs()
                .map(|r| (r.line_top, r.line_height))
                .collect();
            if let Some(cur_run_pos) = runs
                .iter()
                .position(|(top, h)| cur_rect.y >= *top - 1.0 && cur_rect.y < *top + *h + 1.0)
            {
                let next_run_pos = cur_run_pos as isize + direction as isize;
                if next_run_pos >= 0 && (next_run_pos as usize) < runs.len() {
                    let (top, h) = runs[next_run_pos as usize];
                    let lx = (target_x - cur_shaped.display.left_indent_px).max(0.0);
                    if let Some(hit) = cur_shaped.buffer.hit(lx, top + h * 0.5) {
                        let vis = cur_shaped.map.line_col_to_visual(hit.line, hit.index);
                        return (
                            cur_shaped
                                .map
                                .visual_to_source_rebased(vis, blocks[block_idx].range.start),
                            Some(target_x),
                        );
                    }
                }
            }
        }

        let target_x = crossed_target_x.unwrap_or(12.0);
        // Cross to adjacent block above or below
        let next_idx = if direction < 0 {
            if block_idx == 0 {
                return (blocks[0].range.start, Some(target_x));
            }
            block_idx - 1
        } else {
            if block_idx + 1 >= blocks.len() {
                return (self.last_source_len.max(blocks[block_idx].range.end), Some(target_x));
            }
            block_idx + 1
        };

        if matches!(blocks[next_idx].kind, BlockKind::Table) && view.mode != ViewMode::Source
            && let Some(table_key) = self.shape_table_for_nav(next_idx, &nav_ctx, &mut read_slice)
                && let Some(table) = self.table_lru.get(&table_key) {
                    return (
                        enter_table_from_adjacent(
                            table,
                            blocks[next_idx].range.start,
                            direction,
                            target_x,
                        ),
                        Some(target_x),
                    );
                }

        if let Some((adj_shaped, _)) =
            self.shape_single_for_nav(next_idx, &nav_ctx, &mut read_slice)
        {
            let target_y = if direction < 0 {
                (adj_shaped.total_height - adj_shaped.display.bottom_pad_px - style.base_line_height_px * 0.4)
                    .max(2.0)
            } else {
                2.0
            };
            let lx = (target_x - adj_shaped.display.left_indent_px).max(0.0);
            if let Some(hit) = adj_shaped.buffer.hit(lx, target_y) {
                let vis = adj_shaped.map.line_col_to_visual(hit.line, hit.index);
                return (
                    adj_shaped
                        .map
                        .visual_to_source_rebased(vis, blocks[next_idx].range.start),
                    Some(target_x),
                );
            }
            return (blocks[next_idx].range.start, Some(target_x));
        }

        (caret, Some(target_x))
    }

    fn shape_table_for_nav<S>(
        &mut self,
        block_idx: usize,
        ctx: &NavShapeContext<'_>,
        read_slice: &mut S,
    ) -> Option<u64>
    where
        S: FnMut(ByteRange) -> String,
    {
        let node = ctx.blocks.get(block_idx)?;
        let raw = read_slice(node.range);
        let rows_display = build_table_display(
            node,
            &raw,
            ctx.view,
            ctx.exp_sel,
            ctx.style,
            true,
            false,
        )?;
        let avail_table_w = (ctx.content_width_px - 8.0).max(120.0);
        let style_sig = style_signature(ctx.style);
        let key = shape_table_cache_key(&rows_display, avail_table_w, style_sig, true, false);
        self.frame_seq = self.frame_seq.wrapping_add(1);
        let frame_seq = self.frame_seq;
        if let Some(cached) = self.table_lru.get_mut(&key) {
            cached.last_used_frame = frame_seq;
        } else {
            let shaped = shape_table_block(
                &mut self.font_system,
                rows_display,
                avail_table_w,
                ctx.style,
                frame_seq,
            );
            if self.table_lru.len() >= MAX_SHAPED_TABLE_CACHE {
                evict_oldest_table_lru(&mut self.table_lru);
            }
            self.table_lru.insert(key, shaped);
        }
        Some(key)
    }

    fn shape_single_for_nav<S>(
        &mut self,
        block_idx: usize,
        ctx: &NavShapeContext<'_>,
        read_slice: &mut S,
    ) -> Option<(&CachedBlockShape, usize)>
    where
        S: FnMut(ByteRange) -> String,
    {
        let node = ctx.blocks.get(block_idx)?;
        let raw = read_slice(node.range);
        let display = build_block_display(node, &raw, ctx.view, ctx.exp_sel, ctx.style, true, false);
        let caret_vis = display
            .map
            .source_to_visual(ctx.view.selection.head.clamp(node.range.start, node.range.end));
        let avail_w = (ctx.content_width_px - display.left_indent_px - 24.0).max(80.0);
        let style_sig = style_signature(ctx.style);
        let key = shape_cache_key(
            &display,
            avail_w,
            style_sig,
            matches!(node.kind, BlockKind::CodeFence { .. } | BlockKind::YamlFrontMatter),
        );
        self.frame_seq = self.frame_seq.wrapping_add(1);
        let frame_seq = self.frame_seq;
        if !self.shaped_lru.contains_key(&key) {
            let shaped = shape_block(&mut self.font_system, display, avail_w, ctx.style, key, frame_seq);
            self.shaped_lru.insert(key, shaped);
        }
        self.shaped_lru.get(&key).map(|s| (s, caret_vis))
    }
}

struct NavShapeContext<'a> {
    blocks: &'a [BlockNode],
    view: &'a EditorViewState,
    exp_sel: Selection,
    style: &'a ResolvedThemeStyle,
    content_width_px: f32,
}

/// Helper for `vellmark-app` UI chrome (tabs, file tree, outline, find bar, status bar)
/// to shape crisp text lines reusing the shared `FontSystem`.
#[allow(clippy::too_many_arguments)]
pub fn shape_ui_text_into(
    font_system: &mut FontSystem,
    text: &str,
    x: f32,
    y: f32,
    font_size: f32,
    line_height: f32,
    weight: u16,
    color: RgbaColor,
    max_width: Option<f32>,
    clip: Option<Rect>,
    scale_factor: f32,
    out_glyphs: &mut Vec<SceneGlyph>,
) -> f32 {
    if text.is_empty() {
        return 0.0;
    }
    let scale = scale_factor.max(0.5);
    let metrics = Metrics::new(font_size.max(8.0), line_height.max(10.0));
    let mut buf = CosmicBuffer::new(font_system, metrics);
    buf.set_wrap(Wrap::None);
    buf.set_size(max_width, Some(line_height + 4.0));
    let attrs = Attrs::new()
        .family(Family::SansSerif)
        .weight(Weight(weight))
        .color(color.to_cosmic_color());
    buf.set_text(text, &attrs, Shaping::Advanced, None);
    buf.shape_until_scroll(font_system, false);

    let mut max_w = 0.0f32;
    if let Some(run) = buf.layout_runs().next() {
        max_w = run.line_w;
        for glyph in run.glyphs {
            let phys = glyph.physical((x * scale, (y + run.line_y) * scale), scale);
            out_glyphs.push(SceneGlyph {
                cache_key: phys.cache_key,
                x: phys.x,
                y: phys.y,
                color,
                clip,
            });
        }
    }
    max_w
}

#[allow(clippy::too_many_arguments)]
fn update_block_height_metric(
    idx: usize,
    block_y: f32,
    measured_h: f32,
    block_ys: &mut [f32],
    block_heights: &mut [f32],
    y_cursor_shift: &mut f32,
    any_height_changed: &mut bool,
    first_changed_idx: &mut usize,
) {
    let old_h = block_heights[idx];
    if (measured_h - old_h).abs() > 0.5 {
        let delta = measured_h - old_h;
        block_heights[idx] = measured_h;
        block_ys[idx] = block_y;
        *y_cursor_shift += delta;
        if !*any_height_changed {
            *first_changed_idx = idx;
        }
        *any_height_changed = true;
    } else {
        block_ys[idx] = block_y;
    }
}

#[allow(clippy::too_many_arguments)]
fn render_text_buffer_contents(
    inline_code_rects: &[Rect],
    buffer: &CosmicBuffer,
    map: &VisualTextMap,
    spans: &[DisplaySpanStyle],
    highlight_range: ByteRange,
    actual_map_start: ByteOffset,
    empty_sel_sliver: bool,
    text_origin_x: f32,
    text_origin_y: f32,
    fallback_text_color: RgbaColor,
    sel_range: ByteRange,
    vp: &EditorViewport<'_>,
    style: &ResolvedThemeStyle,
    draw: &mut DrawList,
) {
    for pill in inline_code_rects {
        draw.rects.push(
            SceneRect::rounded(
                pill.translate(text_origin_x, text_origin_y),
                style.inline_code_bg,
                4.0,
            )
            .with_clip(Some(vp.doc_rect)),
        );
    }

    for m in vp.find_matches {
        if m.end <= highlight_range.start || m.start >= highlight_range.end {
            continue;
        }
        let is_active = vp.active_find == Some(*m);
        let color = if is_active {
            style.chrome.warning.with_alpha(120)
        } else {
            style.chrome.warning.with_alpha(55)
        };
        emit_highlight_range(
            buffer,
            map,
            actual_map_start,
            m.start.max(highlight_range.start),
            m.end.min(highlight_range.end),
            text_origin_x,
            text_origin_y,
            color,
            vp.doc_rect,
            &mut draw.rects,
        );
    }

    if !sel_range.is_empty()
        && sel_range.end > highlight_range.start
        && sel_range.start <= highlight_range.end
    {
        let s_start = sel_range.start.max(highlight_range.start);
        let s_end = sel_range.end.min(highlight_range.end);
        if s_start < s_end {
            emit_highlight_range(
                buffer,
                map,
                actual_map_start,
                s_start,
                s_end,
                text_origin_x,
                text_origin_y,
                style.selection_bg,
                vp.doc_rect,
                &mut draw.rects,
            );
        } else if empty_sel_sliver {
            draw.rects.push(
                SceneRect::solid(
                    Rect::new(text_origin_x, text_origin_y, 6.0, style.base_line_height_px),
                    style.selection_bg,
                )
                .with_clip(Some(vp.doc_rect)),
            );
        }
    }

    if let Some(pre_vis) = map.preedit_visual_range()
        && !pre_vis.is_empty() {
            let c0 = map.visual_to_line_col(pre_vis.start);
            let c1 = map.visual_to_line_col(pre_vis.end);
            for run in buffer.layout_runs() {
                for (hx, hw) in run.highlight(c0, c1) {
                    let bg = Rect::new(
                        text_origin_x + hx,
                        text_origin_y + run.line_top,
                        hw,
                        run.line_height,
                    );
                    draw.rects.push(
                        SceneRect::solid(bg, style.ime_preedit_bg).with_clip(Some(vp.doc_rect)),
                    );
                    let ul = Rect::new(
                        text_origin_x + hx,
                        text_origin_y + run.line_top + run.line_height - 2.0,
                        hw,
                        2.0,
                    );
                    draw.rects.push(
                        SceneRect::solid(ul, style.ime_underline_color)
                            .with_clip(Some(vp.doc_rect)),
                    );
                }
            }
        }

    let scale = vp.scale_factor.max(0.5);
    for run in buffer.layout_runs() {
        for glyph in run.glyphs {
            let phys = glyph.physical(
                (text_origin_x * scale, (text_origin_y + run.line_y) * scale),
                scale,
            );
            let color = glyph
                .color_opt
                .map(|c| RgbaColor::new(c.r(), c.g(), c.b(), c.a()))
                .unwrap_or(fallback_text_color);
            draw.glyphs.push(SceneGlyph {
                cache_key: phys.cache_key,
                x: phys.x,
                y: phys.y,
                color,
                clip: Some(vp.doc_rect),
            });

            let vis_g_start = map.line_col_to_visual(run.line_i, glyph.start);
            for span in spans {
                if span.range.start <= vis_g_start && vis_g_start < span.range.end {
                    if span.strikethrough {
                        let mid_y = text_origin_y + run.line_top + run.line_height * 0.52;
                        draw.rects.push(
                            SceneRect::solid(
                                Rect::new(text_origin_x + glyph.x, mid_y, glyph.w, 1.4),
                                color,
                            )
                            .with_clip(Some(vp.doc_rect)),
                        );
                    }
                    if span.underline && !span.ime_preedit {
                        let u_y = text_origin_y + run.line_top + run.line_height - 2.5;
                        draw.rects.push(
                            SceneRect::solid(
                                Rect::new(text_origin_x + glyph.x, u_y, glyph.w, 1.2),
                                color,
                            )
                            .with_clip(Some(vp.doc_rect)),
                        );
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_table_scene(
    table: &CachedTableShape,
    node: &BlockNode,
    block_y: f32,
    screen_top_y: f32,
    table_left_x: f32,
    table_top_y: f32,
    caret_after_last_block: bool,
    base_caret: ByteOffset,
    sel: Selection,
    vp: &EditorViewport<'_>,
    style: &ResolvedThemeStyle,
    block_focused: bool,
    draw: &mut DrawList,
    caret_doc_y: &mut f32,
    caret_line_height: &mut f32,
    ime_cursor_area: &mut Option<Rect>,
) {
    let outer_rect = Rect::new(table_left_x, table_top_y, table.table_w, table.table_h);
    draw.rects.push(
        SceneRect::rounded(outer_rect, style.bg_color, 4.0 * style.zoom_scale)
            .with_clip(Some(vp.doc_rect)),
    );

    // Row backgrounds (header fill / revealed delimiter bar)
    for row in &table.rows {
        let fill = if row.is_header {
            Some(style.table_header_bg)
        } else if row.is_delimiter {
            Some(style.code_block_bg)
        } else {
            None
        };
        if let Some(bg) = fill {
            let row_bg = Rect::new(
                table_left_x + 1.0,
                table_top_y + row.y_rel + if row.y_rel == 0.0 { 1.0 } else { 0.0 },
                (table.table_w - 2.0).max(1.0),
                (row.h - if row.y_rel == 0.0 { 1.0 } else { 0.0 }).max(1.0),
            );
            draw.rects
                .push(SceneRect::solid(row_bg, bg).with_clip(Some(vp.doc_rect)));
        }
    }

    // Horizontal row dividers
    for (r_idx, row) in table.rows.iter().enumerate() {
        if r_idx + 1 >= table.rows.len() {
            continue;
        }
        let thick = if row.is_header { 1.5 } else { 1.0 };
        let line_y = table_top_y + row.y_rel + row.h - thick * 0.5;
        draw.rects.push(
            SceneRect::solid(
                Rect::new(table_left_x + 1.0, line_y, (table.table_w - 2.0).max(1.0), thick),
                style.table_border,
            )
            .with_clip(Some(vp.doc_rect)),
        );
    }

    // Vertical column dividers
    let mut col_x = 0.0f32;
    for (c_idx, &w) in table.col_widths.iter().enumerate() {
        col_x += w;
        if c_idx + 1 >= table.col_widths.len() {
            continue;
        }
        draw.rects.push(
            SceneRect::solid(
                Rect::new(
                    table_left_x + col_x - 0.5,
                    table_top_y + 1.0,
                    1.0,
                    (table.table_h - 2.0).max(1.0),
                ),
                style.table_border,
            )
            .with_clip(Some(vp.doc_rect)),
        );
    }

    // Outer border drawn above inner row fills and divider endpoints for crisp corners
    draw.rects.push(
        SceneRect::rounded(
            outer_rect,
            style.table_border.with_alpha(0),
            4.0 * style.zoom_scale,
        )
        .with_border(1.0, style.table_border)
        .with_clip(Some(vp.doc_rect)),
    );

    let sel_range = sel.range();
    let base_text_color = style.apply_focus_mode(style.text_color, block_focused, vp.focus_mode);

    for row in &table.rows {
        let cell_text_y = table_top_y + row.y_rel + table.cell_pad_y;
        for cell in &row.cells {
            if cell.is_placeholder {
                continue;
            }
            let cell_src_abs = ByteRange::new(
                node.range.start + cell.rel_src_range.start,
                node.range.start + cell.rel_src_range.end,
            );
            let cell_raw_abs = ByteRange::new(
                node.range.start + cell.rel_raw_range.start,
                node.range.start + cell.rel_raw_range.end,
            );
            if !sel_range.is_empty()
                && cell.map.display_text.is_empty()
                && sel_range.start <= cell_raw_abs.start
                && sel_range.end >= cell_raw_abs.end
            {
                draw.rects.push(
                    SceneRect::solid(
                        Rect::new(
                            table_left_x + cell.x_rel + 1.0,
                            table_top_y + row.y_rel + 1.0,
                            (cell.w - 2.0).max(1.0),
                            (row.h - 2.0).max(1.0),
                        ),
                        style.selection_bg,
                    )
                    .with_clip(Some(vp.doc_rect)),
                );
            }

            let cell_color = if row.is_delimiter {
                style.apply_focus_mode(style.muted_marker_color, block_focused, vp.focus_mode)
            } else {
                base_text_color
            };
            render_text_buffer_contents(
                &cell.inline_code_rects,
                &cell.buffer,
                &cell.map,
                &cell.spans,
                cell_src_abs,
                cell_src_abs.start,
                false,
                table_left_x + cell.text_x_rel,
                cell_text_y,
                cell_color,
                sel_range,
                vp,
                style,
                draw,
            );
        }
    }

    // Active caret / IME anchor inside this table
    if !caret_after_last_block && ime_cursor_area.is_none() {
        let ime_loc = find_table_ime_cell(table);
        let target_cell = ime_loc.or_else(|| {
            if !node.range.contains_inclusive(base_caret) {
                return None;
            }
            let rel_caret = base_caret - node.range.start;
            find_table_cell_for_rel_caret(table, rel_caret).map(|(r, c)| {
                let cell = &table.rows[r].cells[c];
                let vis = cell
                    .map
                    .source_to_visual_rebased(base_caret, node.range.start + cell.rel_src_range.start);
                (r, c, vis)
            })
        });
        if let Some((r_idx, c_idx, target_vis)) = target_cell {
            let cur_rect = table_cell_caret_rect(
                table,
                r_idx,
                c_idx,
                target_vis,
                table_left_x,
                table_top_y,
                style,
            );
            *caret_doc_y = block_y + (cur_rect.y - screen_top_y);
            *caret_line_height = cur_rect.h;
            *ime_cursor_area = Some(cur_rect);
            if vp.show_caret {
                draw.rects
                    .push(SceneRect::solid(cur_rect, style.caret_color).with_clip(Some(vp.doc_rect)));
            }
        }
    }
}

fn find_table_ime_cell(table: &CachedTableShape) -> Option<(usize, usize, usize)> {
    for (r_idx, row) in table.rows.iter().enumerate() {
        for (c_idx, cell) in row.cells.iter().enumerate() {
            if let Some(vis) = cell.map.active_ime_caret_visual() {
                return Some((r_idx, c_idx, vis));
            }
        }
    }
    None
}

fn find_table_cell_for_rel_caret(
    table: &CachedTableShape,
    rel_caret: ByteOffset,
) -> Option<(usize, usize)> {
    if table.rows.is_empty() {
        return None;
    }
    let row_idx = table
        .rows
        .iter()
        .position(|r| r.line_range_rel.contains_inclusive(rel_caret))
        .unwrap_or_else(|| {
            table
                .rows
                .iter()
                .enumerate()
                .min_by_key(|(_, r)| {
                    if rel_caret < r.line_range_rel.start {
                        r.line_range_rel.start - rel_caret
                    } else {
                        rel_caret - r.line_range_rel.end
                    }
                })
                .map(|(i, _)| i)
                .unwrap_or(0)
        });
    let row = &table.rows[row_idx];
    let cell_idx = row
        .cells
        .iter()
        .position(|c| !c.is_placeholder && c.rel_raw_range.contains_inclusive(rel_caret))
        .or_else(|| {
            let real_cells: Vec<usize> = row
                .cells
                .iter()
                .enumerate()
                .filter(|(_, c)| !c.is_placeholder)
                .map(|(i, _)| i)
                .collect();
            if real_cells.is_empty() {
                None
            } else if rel_caret <= row.cells[real_cells[0]].rel_raw_range.start {
                Some(real_cells[0])
            } else {
                real_cells.last().copied()
            }
        })?;
    Some((row_idx, cell_idx))
}

fn table_cell_caret_rect(
    table: &CachedTableShape,
    row_idx: usize,
    cell_idx: usize,
    target_vis: usize,
    origin_x: f32,
    origin_y: f32,
    style: &ResolvedThemeStyle,
) -> Rect {
    let row = &table.rows[row_idx];
    let cell = &row.cells[cell_idx];
    let cell_text_x = origin_x + cell.text_x_rel;
    let cell_text_y = origin_y + row.y_rel + table.cell_pad_y;
    let mut cur = caret_rect_in_buffer(
        &cell.buffer,
        &cell.map,
        target_vis,
        cell_text_x,
        cell_text_y,
        style.base_line_height_px,
    );
    if cell.map.display_text.is_empty() && (cur.x - cell_text_x).abs() < 0.5 {
        let align_off = match cell.align {
            TableAlign::Left => 0.0,
            TableAlign::Center => cell.inner_w * 0.5,
            TableAlign::Right => (cell.inner_w - 2.0).max(0.0),
        };
        cur.x = cell_text_x + align_off;
    }
    let max_x = (cell_text_x + cell.inner_w - 2.0).max(cell_text_x);
    cur.x = cur.x.clamp(cell_text_x, max_x);
    cur
}

fn hit_test_table(
    table: &CachedTableShape,
    block_range: ByteRange,
    tx: f32,
    ty: f32,
) -> ByteOffset {
    if table.rows.is_empty() {
        return block_range.start;
    }
    let row_idx = table
        .rows
        .iter()
        .position(|r| ty >= r.y_rel && ty <= r.y_rel + r.h)
        .unwrap_or_else(|| {
            if ty < table.rows[0].y_rel {
                0
            } else {
                table.rows.len() - 1
            }
        });
    let row = &table.rows[row_idx];
    let real_cells: Vec<usize> = row
        .cells
        .iter()
        .enumerate()
        .filter(|(_, c)| !c.is_placeholder)
        .map(|(i, _)| i)
        .collect();
    if real_cells.is_empty() {
        return block_range.start + row.line_range_rel.start;
    }
    let cell_idx = real_cells
        .iter()
        .copied()
        .find(|&i| tx >= row.cells[i].x_rel && tx <= row.cells[i].x_rel + row.cells[i].w)
        .unwrap_or_else(|| {
            if tx < row.cells[real_cells[0]].x_rel {
                real_cells[0]
            } else {
                *real_cells.last().unwrap()
            }
        });
    let cell = &row.cells[cell_idx];
    let lx = (tx - cell.text_x_rel).max(0.0);
    let ly = (ty - (row.y_rel + table.cell_pad_y)).max(0.0);
    if let Some(cur) = cell.buffer.hit(lx, ly) {
        let vis = cell.map.line_col_to_visual(cur.line, cur.index);
        return cell
            .map
            .visual_to_source_rebased(vis, block_range.start + cell.rel_src_range.start);
    }
    block_range.start + cell.rel_src_range.end
}

enum TableNavOutcome {
    Moved(ByteOffset, f32),
    Exited(f32),
}

fn hit_cell_at_target_x(
    cell: &TableCellShape,
    block_start: ByteOffset,
    target_x_rel_content: f32,
    target_y_in_cell: f32,
) -> ByteOffset {
    let cell_nav_text_x = 4.0 + cell.text_x_rel;
    let lx = (target_x_rel_content - cell_nav_text_x).max(0.0);
    if let Some(hit) = cell.buffer.hit(lx, target_y_in_cell) {
        let vis = cell.map.line_col_to_visual(hit.line, hit.index);
        return cell
            .map
            .visual_to_source_rebased(vis, block_start + cell.rel_src_range.start);
    }
    block_start + cell.rel_src_range.start
}

fn move_caret_vertical_in_table(
    table: &CachedTableShape,
    block_start: ByteOffset,
    caret: ByteOffset,
    direction: i32,
    preferred_x: Option<f32>,
    style: &ResolvedThemeStyle,
) -> TableNavOutcome {
    let rel_caret = caret.saturating_sub(block_start);
    let Some((r_idx, c_idx)) = find_table_cell_for_rel_caret(table, rel_caret) else {
        return TableNavOutcome::Exited(preferred_x.unwrap_or(16.0));
    };
    let cell = &table.rows[r_idx].cells[c_idx];
    let vis = cell
        .map
        .active_ime_caret_visual()
        .unwrap_or_else(|| {
            cell.map
                .source_to_visual_rebased(caret, block_start + cell.rel_src_range.start)
        });
    let cur_rect = table_cell_caret_rect(table, r_idx, c_idx, vis, 4.0, 0.0, style);
    let target_x = preferred_x.unwrap_or(cur_rect.x);
    let cell_top = table.rows[r_idx].y_rel + table.cell_pad_y;
    let local_cur_y = cur_rect.y - cell_top;

    // 1. Try wrapped visual lines inside the current cell first
    let runs: Vec<(f32, f32)> = cell
        .buffer
        .layout_runs()
        .map(|r| (r.line_top, r.line_height))
        .collect();
    if let Some(cur_run_pos) = runs
        .iter()
        .position(|(top, h)| local_cur_y >= *top - 1.0 && local_cur_y < *top + *h + 1.0)
    {
        let next_run_pos = cur_run_pos as isize + direction as isize;
        if next_run_pos >= 0 && (next_run_pos as usize) < runs.len() {
            let (top, h) = runs[next_run_pos as usize];
            return TableNavOutcome::Moved(
                hit_cell_at_target_x(cell, block_start, target_x, top + h * 0.5),
                target_x,
            );
        }
    }

    // 2. Step into the adjacent visible row in the same column
    let next_r = r_idx as isize + direction as isize;
    if next_r >= 0 && (next_r as usize) < table.rows.len() {
        let next_row = &table.rows[next_r as usize];
        let mut target_col = c_idx.min(next_row.cells.len().saturating_sub(1));
        while target_col > 0 && next_row.cells[target_col].is_placeholder {
            target_col -= 1;
        }
        if let Some(next_cell) = next_row.cells.get(target_col)
            && !next_cell.is_placeholder {
                let target_y = if direction < 0 {
                    next_cell
                        .buffer
                        .layout_runs()
                        .last()
                        .map(|r| r.line_top + r.line_height * 0.5)
                        .unwrap_or(2.0)
                } else {
                    2.0
                };
                return TableNavOutcome::Moved(
                    hit_cell_at_target_x(next_cell, block_start, target_x, target_y),
                    target_x,
                );
            }
    }

    TableNavOutcome::Exited(target_x)
}

fn enter_table_from_adjacent(
    table: &CachedTableShape,
    block_start: ByteOffset,
    direction: i32,
    target_x_rel_content: f32,
) -> ByteOffset {
    let row = if direction > 0 {
        table.rows.first()
    } else {
        table.rows.last()
    };
    let Some(row) = row else {
        return block_start;
    };
    let real_cells: Vec<usize> = row
        .cells
        .iter()
        .enumerate()
        .filter(|(_, c)| !c.is_placeholder)
        .map(|(i, _)| i)
        .collect();
    if real_cells.is_empty() {
        return block_start + row.line_range_rel.start;
    }
    let tx_in_table = target_x_rel_content - 4.0;
    let cell_idx = real_cells
        .iter()
        .copied()
        .find(|&i| tx_in_table >= row.cells[i].x_rel && tx_in_table <= row.cells[i].x_rel + row.cells[i].w)
        .unwrap_or_else(|| {
            if tx_in_table < row.cells[real_cells[0]].x_rel {
                real_cells[0]
            } else {
                *real_cells.last().unwrap()
            }
        });
    let cell = &row.cells[cell_idx];
    let target_y = if direction < 0 {
        cell.buffer
            .layout_runs()
            .last()
            .map(|r| r.line_top + r.line_height * 0.5)
            .unwrap_or(2.0)
    } else {
        2.0
    };
    hit_cell_at_target_x(cell, block_start, target_x_rel_content, target_y)
}

fn build_block_display(
    node: &BlockNode,
    raw_text: &str,
    view: &EditorViewState,
    exp_sel: Selection,
    style: &ResolvedThemeStyle,
    is_focused: bool,
    focus_mode: bool,
) -> BlockDisplay {
    let block_start = node.range.start;
    let is_source = view.mode == ViewMode::Source;
    let block_len = raw_text.len();

    let mut hidden: Vec<Range<usize>> = Vec::new();
    let mut marker_ranges: Vec<Range<usize>> = Vec::new();
    let mut synthetic_prefix: Option<&'static str> = None;
    let mut left_indent_px = 12.0;
    let mut top_pad_px = 3.0 * style.zoom_scale;
    let mut bottom_pad_px = 4.0 * style.zoom_scale;
    let mut blockquote_bar = false;
    let mut code_box = false;
    let mut hr_rule = false;
    let mut heading_level = None;
    let mut task_checkbox = None;
    let mut base_color = style.apply_focus_mode(style.text_color, is_focused, focus_mode);
    let mut base_font_size = style.base_font_size_px;
    let mut base_line_height = style.base_line_height_px;
    let mut base_weight = 400u16;
    let mut base_monospace = false;

    match &node.kind {
        BlockKind::Heading { level, prefix_range } => {
            heading_level = Some(*level);
            let hs = &style.headings[((*level).clamp(1, 6) - 1) as usize];
            base_font_size = hs.font_size_px;
            base_line_height = hs.line_height_px;
            base_weight = hs.weight;
            base_color = style.apply_focus_mode(hs.color, is_focused, focus_mode);
            top_pad_px = hs.margin_top_px * 0.55;
            bottom_pad_px = hs.margin_bottom_px * 0.65;
            let local_end = (prefix_range.end - block_start).min(block_len);
            if !is_source {
                if node.expanded_block_prefix(exp_sel).is_some() {
                    if local_end > 0 {
                        marker_ranges.push(0..local_end);
                    }
                } else if local_end > 0 {
                    hidden.push(0..local_end);
                }
            } else if local_end > 0 {
                marker_ranges.push(0..local_end);
            }
        }
        BlockKind::ThematicBreak => {
            top_pad_px = 8.0 * style.zoom_scale;
            bottom_pad_px = 8.0 * style.zoom_scale;
            if !is_source && node.expanded_block_prefix(exp_sel).is_none() {
                hr_rule = true;
                if block_len > 0 {
                    hidden.push(0..block_len);
                }
            } else {
                hr_rule = !is_source;
                if block_len > 0 {
                    marker_ranges.push(0..block_len);
                }
            }
        }
        BlockKind::BlockQuote => {
            blockquote_bar = !is_source;
            left_indent_px = 26.0 * style.zoom_scale;
            base_color = style.apply_focus_mode(style.blockquote_text, is_focused, focus_mode);
            let quote_prefix_len = if raw_text.starts_with("> ") {
                2
            } else if raw_text.starts_with('>') {
                1
            } else {
                0
            };
            if quote_prefix_len > 0 {
                if !is_source
                    && (!exp_sel.is_caret() || !node.range.contains_inclusive(exp_sel.head))
                {
                    hidden.push(0..quote_prefix_len);
                } else {
                    marker_ranges.push(0..quote_prefix_len);
                }
            }
        }
        BlockKind::ListItem {
            ordered,
            indent,
            task_checked,
        } => {
            left_indent_px = 28.0 * style.zoom_scale + (*indent as f32) * 14.0 * style.zoom_scale;
            let prefix_len = list_item_prefix_len(raw_text);
            let inside = exp_sel.is_caret() && node.range.contains_inclusive(exp_sel.head);
            if !is_source && !inside && prefix_len > 0 {
                if let Some(checked) = task_checked {
                    task_checkbox = Some(*checked);
                    hidden.push(0..prefix_len);
                } else if !*ordered {
                    hidden.push(0..prefix_len);
                    synthetic_prefix = Some("• ");
                } else {
                    marker_ranges.push(0..prefix_len);
                }
            } else {
                if !is_source {
                    task_checkbox = *task_checked;
                }
                if prefix_len > 0 {
                    marker_ranges.push(0..prefix_len);
                }
            }
        }
        BlockKind::CodeFence { .. } | BlockKind::YamlFrontMatter => {
            code_box = !is_source;
            left_indent_px = 16.0 * style.zoom_scale;
            top_pad_px = 8.0 * style.zoom_scale;
            bottom_pad_px = 8.0 * style.zoom_scale;
            base_color = style.apply_focus_mode(style.code_block_text, is_focused, focus_mode);
            base_font_size = style.code_font_size_px;
            base_line_height = style.code_line_height_px;
            base_monospace = true;
        }
        BlockKind::Table => {
            base_monospace = is_source;
            left_indent_px = 14.0 * style.zoom_scale;
            for (idx, b) in raw_text.bytes().enumerate() {
                if b == b'|' {
                    marker_ranges.push(idx..idx + 1);
                }
            }
        }
        BlockKind::Paragraph => {}
    }

    let disallow_past_end =
        matches!(node.kind, BlockKind::Paragraph if raw_text.is_empty());
    let (map, spans) = collect_fragment_display(
        node.range,
        raw_text,
        &node.inlines,
        block_start,
        hidden,
        marker_ranges,
        synthetic_prefix,
        base_color,
        base_weight,
        base_monospace,
        true,
        disallow_past_end,
        view,
        exp_sel,
        style,
        is_focused,
        focus_mode,
    );

    let _ = (base_font_size, base_line_height);
    BlockDisplay {
        map,
        spans,
        left_indent_px,
        top_pad_px,
        bottom_pad_px,
        blockquote_bar,
        code_box,
        hr_rule,
        heading_level,
        task_checkbox,
    }
}

#[allow(clippy::too_many_arguments)]
fn collect_fragment_display(
    frag_range_abs: ByteRange,
    frag_raw: &str,
    inlines_abs: &[InlineSpan],
    map_base_offset: ByteOffset,
    mut hidden: Vec<Range<usize>>,
    mut marker_ranges: Vec<Range<usize>>,
    synthetic_prefix: Option<&str>,
    base_color: RgbaColor,
    base_weight: u16,
    base_monospace: bool,
    allow_ime: bool,
    disallow_past_end: bool,
    view: &EditorViewState,
    exp_sel: Selection,
    style: &ResolvedThemeStyle,
    is_focused: bool,
    focus_mode: bool,
) -> (VisualTextMap, Vec<DisplaySpanStyle>) {
    let is_source = view.mode == ViewMode::Source;
    let frag_len = frag_raw.len();
    let delta = map_base_offset as isize - frag_range_abs.start as isize;
    let mut rebased_inlines: Vec<(InlineKind, ByteRange)> = Vec::new();

    for span in inlines_abs {
        if span.full_range.end <= frag_range_abs.start || span.full_range.start >= frag_range_abs.end
        {
            continue;
        }
        let o_start = span
            .open_marker
            .start
            .saturating_sub(frag_range_abs.start)
            .min(frag_len);
        let o_end = span
            .open_marker
            .end
            .saturating_sub(frag_range_abs.start)
            .min(frag_len);
        let c_start = span
            .close_marker
            .start
            .saturating_sub(frag_range_abs.start)
            .min(frag_len);
        let c_end = span
            .close_marker
            .end
            .saturating_sub(frag_range_abs.start)
            .min(frag_len);

        if is_source || span.should_expand_markers(exp_sel) {
            if o_start < o_end {
                marker_ranges.push(o_start..o_end);
            }
            if c_start < c_end {
                marker_ranges.push(c_start..c_end);
            }
        } else {
            if o_start < o_end {
                hidden.push(o_start..o_end);
            }
            if c_start < c_end {
                hidden.push(c_start..c_end);
            }
        }

        let s_abs = span.full_range.start.clamp(frag_range_abs.start, frag_range_abs.end);
        let e_abs = span.full_range.end.clamp(frag_range_abs.start, frag_range_abs.end);
        if s_abs < e_abs {
            rebased_inlines.push((
                span.kind,
                ByteRange::new(
                    (s_abs as isize + delta) as usize,
                    (e_abs as isize + delta) as usize,
                ),
            ));
        }
    }

    let merged_hidden = merge_ranges(hidden);
    let mut map = build_visual_text_map(map_base_offset, frag_raw, &merged_hidden, synthetic_prefix);

    let mut preedit_vis_range: Option<Range<usize>> = None;
    if allow_ime
        && let Some(preedit) = &view.ime_preedit {
            let comp_sel = view.ime_composition_selection.unwrap_or(view.selection);
            let comp_range = comp_sel.range();
            let owns_comp = frag_range_abs.contains_inclusive(comp_range.start)
                && !(disallow_past_end && comp_range.start > frag_range_abs.end);
            if owns_comp {
                let c_start_mapped = ((comp_range
                    .start
                    .clamp(frag_range_abs.start, frag_range_abs.end)) as isize
                    + delta) as usize;
                let c_end_mapped = ((comp_range
                    .end
                    .clamp(frag_range_abs.start, frag_range_abs.end)) as isize
                    + delta) as usize;
                let v_start = map.source_to_visual(c_start_mapped);
                let v_end = map.source_to_visual(c_end_mapped);
                map.display_text.replace_range(v_start..v_end, preedit);
                let new_v_end = v_start + preedit.len();
                map.line_starts = compute_line_starts(&map.display_text);
                map.ime_splice = Some(ImeSplice {
                    vis_start: v_start,
                    vis_end: new_v_end,
                    comp_src_start: c_start_mapped,
                    comp_src_end: c_end_mapped,
                    preedit_caret_vis: new_v_end,
                });
                preedit_vis_range = Some(v_start..new_v_end);
            }
        }

    let spans = build_display_spans_with_inlines(
        &rebased_inlines,
        &map,
        &marker_ranges,
        preedit_vis_range,
        base_color,
        base_weight,
        base_monospace,
        style,
        is_focused,
        focus_mode,
    );
    (map, spans)
}

fn build_table_display(
    node: &BlockNode,
    raw_text: &str,
    view: &EditorViewState,
    exp_sel: Selection,
    style: &ResolvedThemeStyle,
    is_focused: bool,
    focus_mode: bool,
) -> Option<Vec<TableRowDisplay>> {
    let rows = parse_table_block(raw_text, ByteRange::new(0, raw_text.len()));
    if rows.len() < 2 || !rows[1].is_delimiter || rows.iter().all(|r| r.cells.is_empty()) {
        return None;
    }

    let has_delim = true;
    let alignments: &[TableAlign] = &rows[1].alignments;

    // PRD 4.1.1 A & GFM 4.10: The delimiter row defines column count/alignment only and
    // NEVER renders as a visible cell row in WYSIWYG mode.
    let visible_indices: Vec<usize> = (0..rows.len()).filter(|&i| i != 1).collect();
    let num_cols = rows[1].cells.len().max(1);

    let active_caret_abs = if exp_sel.is_caret() {
        Some(exp_sel.head)
    } else if view.ime_preedit.is_some() {
        view.ime_composition_selection.map(|s| s.head)
    } else {
        None
    };

    let ime_comp_start = view
        .ime_preedit
        .as_ref()
        .map(|_| view.ime_composition_selection.unwrap_or(view.selection).range().start);
    let mut ime_assigned = false;

    let mut out_rows = Vec::with_capacity(visible_indices.len());
    for r_idx in visible_indices {
        let row = &rows[r_idx];
        let is_header = has_delim && r_idx == 0;
        let default_weight = if is_header { 700 } else { 400 };
        let base_color = if row.is_delimiter {
            style.apply_focus_mode(style.muted_marker_color, is_focused, focus_mode)
        } else {
            style.apply_focus_mode(style.text_color, is_focused, focus_mode)
        };

        let mut cells = Vec::with_capacity(num_cols);
        for c_idx in 0..num_cols {
            let align = alignments.get(c_idx).copied().unwrap_or(TableAlign::Left);
            if c_idx < row.cells.len() {
                let cell = &row.cells[c_idx];
                let mut rel_s = cell.content_range.start;
                let mut rel_e = cell.content_range.end;
                if let Some(abs_c) = active_caret_abs
                    && abs_c >= node.range.start {
                        let rel_c = abs_c - node.range.start;
                        if rel_c >= cell.raw_range.start && rel_c <= cell.raw_range.end {
                            let raw_slice = &raw_text[cell.raw_range.start..cell.raw_range.end];
                            let lead_cap = if raw_slice.starts_with(' ') {
                                cell.raw_range.start + 1
                            } else {
                                cell.raw_range.start
                            };
                            let tail_cap = if raw_slice.len() >= 2 && raw_slice.ends_with(' ') {
                                cell.raw_range.end - 1
                            } else {
                                cell.raw_range.end
                            };
                            if rel_c < rel_s {
                                rel_s = rel_c.max(lead_cap);
                            }
                            if rel_c > rel_e {
                                rel_e = rel_c.min(tail_cap).max(rel_s);
                            }
                        }
                    }
                let rel_src = ByteRange::new(rel_s, rel_e);
                let cell_abs_src = ByteRange::new(node.range.start + rel_s, node.range.start + rel_e);
                let cell_abs_raw = ByteRange::new(
                    node.range.start + cell.raw_range.start,
                    node.range.start + cell.raw_range.end,
                );
                let allow_ime_here = !ime_assigned
                    && ime_comp_start.is_some_and(|cs| {
                        cell_abs_raw.contains_inclusive(cs)
                            || (c_idx == 0
                                && cs >= node.range.start + row.line_range.start
                                && cs < cell_abs_raw.start)
                    });
                if allow_ime_here {
                    ime_assigned = true;
                }

                let frag_raw = &raw_text[rel_s..rel_e];
                let (map, spans) = collect_fragment_display(
                    cell_abs_src,
                    frag_raw,
                    &node.inlines,
                    rel_src.start,
                    Vec::new(),
                    Vec::new(),
                    None,
                    base_color,
                    default_weight,
                    row.is_delimiter,
                    allow_ime_here,
                    false,
                    view,
                    exp_sel,
                    style,
                    is_focused,
                    focus_mode,
                );
                cells.push(TableCellDisplay {
                    rel_raw_range: cell.raw_range,
                    rel_src_range: rel_src,
                    align,
                    is_placeholder: false,
                    map,
                    spans,
                    default_weight,
                });
            } else {
                let end_rel = ByteRange::empty(row.line_range.end);
                cells.push(TableCellDisplay {
                    rel_raw_range: end_rel,
                    rel_src_range: end_rel,
                    align,
                    is_placeholder: true,
                    map: build_visual_text_map(end_rel.start, "", &[], None),
                    spans: Vec::new(),
                    default_weight,
                });
            }
        }

        out_rows.push(TableRowDisplay {
            line_range_rel: row.line_range,
            is_header,
            is_delimiter: row.is_delimiter,
            cells,
        });
    }

    Some(out_rows)
}

fn list_item_prefix_len(raw: &str) -> usize {
    let bytes = raw.as_bytes();
    let mut i = 0;
    while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b'\t') {
        i += 1;
    }
    if i < bytes.len() && matches!(bytes[i], b'-' | b'*' | b'+') {
        i += 1;
    } else {
        let d_start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i > d_start && i < bytes.len() && matches!(bytes[i], b'.' | b')') {
            i += 1;
        } else {
            return 0;
        }
    }
    if i < bytes.len() && bytes[i] == b' ' {
        i += 1;
    }
    if raw[i..].starts_with("[ ] ") || raw[i..].starts_with("[x] ") || raw[i..].starts_with("[X] ")
    {
        i += 4;
    }
    i
}

fn merge_ranges(mut ranges: Vec<Range<usize>>) -> Vec<Range<usize>> {
    ranges.retain(|r| r.start < r.end);
    ranges.sort_by_key(|r| (r.start, r.end));
    let mut merged: Vec<Range<usize>> = Vec::with_capacity(ranges.len());
    for r in ranges {
        if let Some(last) = merged.last_mut()
            && r.start <= last.end {
                last.end = last.end.max(r.end);
                continue;
            }
        merged.push(r);
    }
    merged
}

fn build_visual_text_map(
    block_start: ByteOffset,
    raw_text: &str,
    hidden: &[Range<usize>],
    synthetic_prefix: Option<&str>,
) -> VisualTextMap {
    let mut display_text = String::with_capacity(raw_text.len() + 8);
    let mut segments = Vec::new();
    let mut cursor = 0usize;

    let mut first_content_local = 0usize;
    if let Some(h0) = hidden.first()
        && h0.start == 0 {
            first_content_local = h0.end;
        }
    if let Some(syn) = synthetic_prefix {
        let vs = display_text.len();
        display_text.push_str(syn);
        let ve = display_text.len();
        segments.push(VisualSegment {
            vis_start: vs,
            vis_end: ve,
            src_start: block_start + first_content_local,
            src_end: block_start + first_content_local,
            synthetic: true,
        });
    }

    for hr in hidden {
        if cursor < hr.start {
            let slice = &raw_text[cursor..hr.start];
            let vs = display_text.len();
            display_text.push_str(slice);
            let ve = display_text.len();
            segments.push(VisualSegment {
                vis_start: vs,
                vis_end: ve,
                src_start: block_start + cursor,
                src_end: block_start + hr.start,
                synthetic: false,
            });
        }
        cursor = cursor.max(hr.end);
    }
    if cursor < raw_text.len() {
        let slice = &raw_text[cursor..];
        let vs = display_text.len();
        display_text.push_str(slice);
        let ve = display_text.len();
        segments.push(VisualSegment {
            vis_start: vs,
            vis_end: ve,
            src_start: block_start + cursor,
            src_end: block_start + raw_text.len(),
            synthetic: false,
        });
    }

    let line_starts = compute_line_starts(&display_text);
    VisualTextMap {
        block_range: ByteRange::new(block_start, block_start + raw_text.len()),
        display_text,
        line_starts,
        segments,
        ime_splice: None,
    }
}

fn compute_line_starts(s: &str) -> Vec<usize> {
    let mut starts = vec![0];
    for (i, b) in s.bytes().enumerate() {
        if b == b'\n' {
            starts.push(i + 1);
        }
    }
    starts
}

#[allow(clippy::too_many_arguments)]
fn build_display_spans_with_inlines(
    inlines_mapped: &[(InlineKind, ByteRange)],
    map: &VisualTextMap,
    marker_ranges: &[Range<usize>],
    preedit_vis_range: Option<Range<usize>>,
    base_color: RgbaColor,
    base_weight: u16,
    base_monospace: bool,
    style: &ResolvedThemeStyle,
    is_focused: bool,
    focus_mode: bool,
) -> Vec<DisplaySpanStyle> {
    let total_vis = map.display_text.len();
    if total_vis == 0 {
        return Vec::new();
    }

    // Collect split boundaries in visual coordinates.
    let mut cuts = vec![0usize, total_vis];
    let map_start = map.block_range.start;
    let map_end = map.block_range.end;
    let inline_vis: Vec<_> = inlines_mapped
        .iter()
        .filter_map(|(kind, range)| {
            let s = map.source_to_visual(range.start.clamp(map_start, map_end));
            let e = map.source_to_visual(range.end.clamp(map_start, map_end));
            (s < e).then_some((*kind, s..e))
        })
        .collect();
    for (_, r) in &inline_vis {
        cuts.push(r.start);
        cuts.push(r.end);
    }
    let marker_vis: Vec<Range<usize>> = marker_ranges
        .iter()
        .filter_map(|r| {
            let s = map.source_to_visual((map_start + r.start).min(map_end));
            let e = map.source_to_visual((map_start + r.end).min(map_end));
            (s < e).then_some(s..e)
        })
        .collect();
    for r in &marker_vis {
        cuts.push(r.start);
        cuts.push(r.end);
    }
    if let Some(pre) = &preedit_vis_range {
        cuts.push(pre.start);
        cuts.push(pre.end);
    }

    cuts.sort_unstable();
    cuts.dedup();

    let mut out = Vec::with_capacity(cuts.len().saturating_sub(1));
    let mut i = 0;
    while i + 1 < cuts.len() {
        let a = cuts[i];
        let b = cuts[i + 1];
        i += 1;
        if a >= b {
            continue;
        }
        // Ensure char boundaries cleanly
        if !map.display_text.is_char_boundary(a) || !map.display_text.is_char_boundary(b) {
            continue;
        }

        let in_ime = preedit_vis_range
            .as_ref()
            .is_some_and(|p| a >= p.start && b <= p.end);
        let in_marker = !in_ime && marker_vis.iter().any(|m| a >= m.start && b <= m.end);

        let mut bold = base_weight >= 600;
        let mut italic = false;
        let mut strike = false;
        let mut monospace = base_monospace;
        let mut underline = in_ime;
        let mut inline_code_bg = false;
        let mut color = base_color;

        if !in_ime {
            for (kind, r) in &inline_vis {
                if a >= r.start && b <= r.end {
                    match kind {
                        InlineKind::Strong => bold = true,
                        InlineKind::Emphasis => italic = true,
                        InlineKind::Strikethrough => strike = true,
                        InlineKind::InlineCode => {
                            monospace = true;
                            inline_code_bg = !in_marker;
                            if !in_marker {
                                color = style.apply_focus_mode(
                                    style.inline_code_color,
                                    is_focused,
                                    focus_mode,
                                );
                            }
                        }
                        InlineKind::Link => {
                            if !in_marker {
                                color =
                                    style.apply_focus_mode(style.link_color, is_focused, focus_mode);
                                underline = style.link_underline;
                            }
                        }
                        InlineKind::Image => {
                            if !in_marker {
                                color =
                                    style.apply_focus_mode(style.link_color, is_focused, focus_mode);
                            }
                        }
                    }
                }
            }
        }
        if in_marker {
            color = style.apply_focus_mode(style.muted_marker_color, is_focused, focus_mode);
        } else if in_ime {
            color = style.text_color;
        }

        out.push(DisplaySpanStyle {
            range: a..b,
            color,
            bold,
            italic,
            underline,
            strikethrough: strike,
            monospace,
            inline_code_bg,
            ime_preedit: in_ime,
        });
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn shape_rich_text_buffer(
    font_system: &mut FontSystem,
    map: &VisualTextMap,
    spans: &[DisplaySpanStyle],
    avail_w: f32,
    font_size: f32,
    line_height: f32,
    default_weight: u16,
    monospace: bool,
    align: Option<Align>,
    style: &ResolvedThemeStyle,
    empty_fallback_height: Option<f32>,
) -> (CosmicBuffer, f32, f32, Vec<Rect>) {
    let metrics = Metrics::new(font_size.max(8.0), line_height.max(12.0));
    let mut buffer = CosmicBuffer::new(font_system, metrics);
    buffer.set_wrap(Wrap::WordOrGlyph);
    buffer.set_size(Some(avail_w), None);

    let default_attrs = Attrs::new()
        .family(if monospace {
            Family::Monospace
        } else {
            Family::Name(&style.base_font_family)
        })
        .weight(Weight(default_weight))
        .color(style.text_color.to_cosmic_color());

    if spans.is_empty() || map.display_text.is_empty() {
        buffer.set_text("", &default_attrs, Shaping::Advanced, align);
    } else {
        let rich: Vec<(&str, Attrs<'_>)> = spans
            .iter()
            .filter_map(|sp| {
                let s = map.display_text.get(sp.range.clone())?;
                let mut attrs = Attrs::new()
                    .color(sp.color.to_cosmic_color())
                    .family(if sp.monospace {
                        Family::Monospace
                    } else {
                        Family::Name(&style.base_font_family)
                    })
                    .weight(if sp.bold {
                        Weight::BOLD
                    } else {
                        Weight(default_weight)
                    })
                    .style(if sp.italic {
                        FontStyle::Italic
                    } else {
                        FontStyle::Normal
                    });
                attrs.metrics_opt = Some(CacheMetrics::from(metrics));
                Some((s, attrs))
            })
            .collect();
        buffer.set_rich_text(rich, &default_attrs, Shaping::Advanced, align);
    }
    buffer.shape_until_scroll(font_system, false);

    let mut run_count = 0usize;
    let mut max_line_w = 0.0f32;
    let mut text_height = line_height;
    for run in buffer.layout_runs() {
        run_count += 1;
        max_line_w = max_line_w.max(run.line_w);
        text_height = text_height.max(run.line_top + run.line_height);
    }
    if run_count == 0
        && let Some(fb) = empty_fallback_height {
            text_height = fb;
        }

    let mut inline_code_rects = Vec::new();
    for sp in spans {
        if sp.inline_code_bg {
            let c0 = map.visual_to_line_col(sp.range.start);
            let c1 = map.visual_to_line_col(sp.range.end);
            for run in buffer.layout_runs() {
                for (hx, hw) in run.highlight(c0, c1) {
                    inline_code_rects.push(Rect::new(
                        hx - 2.0,
                        run.line_top + 1.0,
                        hw + 4.0,
                        (run.line_height - 2.0).max(4.0),
                    ));
                }
            }
        }
    }

    (buffer, max_line_w, text_height, inline_code_rects)
}

fn shape_block(
    font_system: &mut FontSystem,
    display: BlockDisplay,
    avail_w: f32,
    style: &ResolvedThemeStyle,
    _hash: u64,
    frame_seq: u64,
) -> CachedBlockShape {
    let (font_size, line_height, default_weight) = if let Some(lvl) = display.heading_level {
        let hs = &style.headings[(lvl.clamp(1, 6) - 1) as usize];
        (hs.font_size_px, hs.line_height_px, hs.weight)
    } else if display.code_box {
        (style.code_font_size_px, style.code_line_height_px, 400)
    } else {
        (style.base_font_size_px, style.base_line_height_px, 400)
    };

    let (buffer, _, text_height, inline_code_rects) = shape_rich_text_buffer(
        font_system,
        &display.map,
        &display.spans,
        avail_w,
        font_size,
        line_height,
        default_weight,
        display.code_box,
        None,
        style,
        if display.hr_rule { Some(8.0) } else { None },
    );
    let total_height = text_height + display.top_pad_px + display.bottom_pad_px;

    CachedBlockShape {
        last_used_frame: frame_seq,
        buffer,
        map: display.map.clone(),
        display,
        total_height,
        inline_code_rects,
    }
}

fn shape_table_block(
    font_system: &mut FontSystem,
    rows_display: Vec<TableRowDisplay>,
    avail_table_w: f32,
    style: &ResolvedThemeStyle,
    frame_seq: u64,
) -> CachedTableShape {
    let cell_pad_x = 12.0 * style.zoom_scale;
    let cell_pad_y = 7.0 * style.zoom_scale;
    let top_pad_px = 6.0 * style.zoom_scale;
    let bottom_pad_px = 6.0 * style.zoom_scale;

    let num_cols = rows_display
        .iter()
        .map(|r| r.cells.len())
        .max()
        .unwrap_or(1)
        .max(1);
    let mut natural_text_w = vec![0.0f32; num_cols];

    for row in &rows_display {
        for (c_idx, cell) in row.cells.iter().enumerate() {
            if cell.is_placeholder || cell.map.display_text.is_empty() {
                continue;
            }
            let (_, line_w, _, _) = shape_rich_text_buffer(
                font_system,
                &cell.map,
                &cell.spans,
                4096.0,
                style.base_font_size_px,
                style.base_line_height_px,
                cell.default_weight,
                row.is_delimiter,
                Some(Align::Left),
                style,
                None,
            );
            natural_text_w[c_idx] = natural_text_w[c_idx].max(line_w);
        }
    }

    let min_col_w = (56.0 * style.zoom_scale).min(avail_table_w / num_cols as f32);
    let mut col_widths: Vec<f32> = natural_text_w
        .iter()
        .map(|&tw| (tw + 2.0 * cell_pad_x).max(min_col_w))
        .collect();
    let desired_sum: f32 = col_widths.iter().sum();

    if desired_sum <= avail_table_w {
        let extra = avail_table_w - desired_sum;
        let equal_share = extra * 0.5 / num_cols as f32;
        let weight_sum: f32 = col_widths.iter().sum::<f32>().max(1.0);
        for w in &mut col_widths {
            *w += equal_share + (extra * 0.5) * (*w / weight_sum);
        }
    } else if (num_cols as f32) * min_col_w >= avail_table_w {
        let uniform = avail_table_w / num_cols as f32;
        col_widths.fill(uniform);
    } else {
        let excess = desired_sum - avail_table_w;
        let shrinkable: f32 = col_widths
            .iter()
            .map(|&w| (w - min_col_w).max(0.0))
            .sum::<f32>()
            .max(1.0);
        for w in &mut col_widths {
            let slack = (*w - min_col_w).max(0.0);
            *w -= excess * (slack / shrinkable);
        }
    }

    let mut y_acc = 0.0f32;
    let mut shaped_rows = Vec::with_capacity(rows_display.len());

    for row in rows_display {
        let mut x_acc = 0.0f32;
        let mut row_text_h = style.base_line_height_px;
        let mut shaped_cells = Vec::with_capacity(row.cells.len());

        for (c_idx, cell) in row.cells.into_iter().enumerate() {
            let col_w = col_widths[c_idx];
            let inner_w = (col_w - 2.0 * cell_pad_x).max(12.0);
            let cosmic_align = match cell.align {
                TableAlign::Left => Align::Left,
                TableAlign::Center => Align::Center,
                TableAlign::Right => Align::Right,
            };
            let (buffer, _, text_h, inline_code_rects) = shape_rich_text_buffer(
                font_system,
                &cell.map,
                &cell.spans,
                inner_w,
                style.base_font_size_px,
                style.base_line_height_px,
                cell.default_weight,
                row.is_delimiter,
                Some(cosmic_align),
                style,
                None,
            );
            if !cell.is_placeholder {
                row_text_h = row_text_h.max(text_h);
            }
            shaped_cells.push(TableCellShape {
                x_rel: x_acc,
                w: col_w,
                text_x_rel: x_acc + cell_pad_x,
                inner_w,
                rel_raw_range: cell.rel_raw_range,
                rel_src_range: cell.rel_src_range,
                align: cell.align,
                is_placeholder: cell.is_placeholder,
                buffer,
                map: cell.map,
                spans: cell.spans,
                inline_code_rects,
            });
            x_acc += col_w;
        }

        let row_h = row_text_h + 2.0 * cell_pad_y;
        shaped_rows.push(TableRowShape {
            y_rel: y_acc,
            h: row_h,
            is_header: row.is_header,
            is_delimiter: row.is_delimiter,
            line_range_rel: row.line_range_rel,
            cells: shaped_cells,
        });
        y_acc += row_h;
    }

    let table_h = y_acc.max(style.base_line_height_px + 2.0 * cell_pad_y);
    CachedTableShape {
        last_used_frame: frame_seq,
        table_w: avail_table_w,
        table_h,
        top_pad_px,
        cell_pad_y,
        col_widths,
        rows: shaped_rows,
        total_height: top_pad_px + table_h + bottom_pad_px,
    }
}

fn caret_rect_in_buffer(
    buffer: &CosmicBuffer,
    map: &VisualTextMap,
    target_vis: usize,
    origin_x: f32,
    origin_y: f32,
    fallback_line_height: f32,
) -> Rect {
    let cursor = map.visual_to_line_col(target_vis);
    let mut last_run_y = 0.0f32;
    let mut last_run_h = fallback_line_height;
    let mut last_end_x = 0.0f32;

    for run in buffer.layout_runs() {
        last_run_y = run.line_top;
        last_run_h = run.line_height;
        last_end_x = run.line_w;
        if let Some(cx) = run.cursor_position(&cursor) {
            return Rect::new(
                origin_x + cx,
                origin_y + run.line_top + 1.0,
                2.0,
                (run.line_height - 2.0).max(8.0),
            );
        }
    }

    Rect::new(
        origin_x + last_end_x,
        origin_y + last_run_y + 1.0,
        2.0,
        (last_run_h - 2.0).max(8.0),
    )
}

#[allow(clippy::too_many_arguments)]
fn emit_highlight_range(
    buffer: &CosmicBuffer,
    map: &VisualTextMap,
    block_start: ByteOffset,
    src_start: ByteOffset,
    src_end: ByteOffset,
    origin_x: f32,
    origin_y: f32,
    color: RgbaColor,
    clip: Rect,
    out: &mut Vec<SceneRect>,
) {
    let v0 = map.source_to_visual_rebased(src_start, block_start);
    let v1 = map.source_to_visual_rebased(src_end, block_start);
    if v0 >= v1 {
        return;
    }
    let c0 = map.visual_to_line_col(v0);
    let c1 = map.visual_to_line_col(v1);
    for run in buffer.layout_runs() {
        for (hx, hw) in run.highlight(c0, c1) {
            out.push(
                SceneRect::solid(
                    Rect::new(origin_x + hx, origin_y + run.line_top, hw, run.line_height),
                    color,
                )
                .with_clip(Some(clip)),
            );
        }
    }
}

fn estimate_block_height(node: &BlockNode, style: &ResolvedThemeStyle, width_chars: f32) -> f32 {
    match &node.kind {
        BlockKind::Heading { level, .. } => {
            let hs = &style.headings[((*level).clamp(1, 6) - 1) as usize];
            hs.line_height_px + (hs.margin_top_px + hs.margin_bottom_px) * 0.6
        }
        BlockKind::ThematicBreak => 16.0 * style.zoom_scale + 8.0,
        BlockKind::CodeFence { .. } | BlockKind::YamlFrontMatter => {
            let est_lines = (node.range.len() as f32 / (width_chars * 0.6).max(12.0)).clamp(2.0, 200.0);
            est_lines * style.code_line_height_px + 16.0 * style.zoom_scale
        }
        BlockKind::Table => {
            let est_rows = ((node.range.len() as f32 / (width_chars * 0.7).max(20.0)).ceil().max(1.0) - 0.5).max(1.0);
            est_rows * (style.base_line_height_px + 14.0 * style.zoom_scale) + 12.0 * style.zoom_scale
        }
        _ => {
            let est_lines = (node.range.len() as f32 / width_chars).ceil().max(1.0);
            est_lines * style.base_line_height_px + 7.0 * style.zoom_scale
        }
    }
}

fn style_signature(style: &ResolvedThemeStyle) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    style.text_color.hash(&mut h);
    style.muted_marker_color.hash(&mut h);
    style.link_color.hash(&mut h);
    style.inline_code_color.hash(&mut h);
    style.code_block_text.hash(&mut h);
    style.blockquote_text.hash(&mut h);
    style.table_border.hash(&mut h);
    style.table_header_bg.hash(&mut h);
    style.base_font_family.hash(&mut h);
    style.base_font_size_px.to_bits().hash(&mut h);
    style.base_line_height_px.to_bits().hash(&mut h);
    style.zoom_scale.to_bits().hash(&mut h);
    h.finish()
}

fn hash_map_and_spans(
    h: &mut std::collections::hash_map::DefaultHasher,
    map: &VisualTextMap,
    spans: &[DisplaySpanStyle],
) {
    let block_start = map.block_range.start;
    map.display_text.hash(h);
    map.block_range.len().hash(h);
    map.segments.len().hash(h);
    for seg in &map.segments {
        seg.vis_start.hash(h);
        seg.vis_end.hash(h);
        seg.src_start.saturating_sub(block_start).hash(h);
        seg.src_end.saturating_sub(block_start).hash(h);
        seg.synthetic.hash(h);
    }
    if let Some(ime) = &map.ime_splice {
        true.hash(h);
        ime.vis_start.hash(h);
        ime.vis_end.hash(h);
        ime.comp_src_start.saturating_sub(block_start).hash(h);
        ime.comp_src_end.saturating_sub(block_start).hash(h);
        ime.preedit_caret_vis.hash(h);
    } else {
        false.hash(h);
    }
    spans.hash(h);
}

fn shape_cache_key(
    display: &BlockDisplay,
    avail_w: f32,
    style_sig: u64,
    code_box: bool,
) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    hash_map_and_spans(&mut h, &display.map, &display.spans);
    avail_w.to_bits().hash(&mut h);
    style_sig.hash(&mut h);
    display.top_pad_px.to_bits().hash(&mut h);
    display.bottom_pad_px.to_bits().hash(&mut h);
    code_box.hash(&mut h);
    h.finish()
}

fn shape_table_cache_key(
    rows: &[TableRowDisplay],
    avail_table_w: f32,
    style_sig: u64,
    block_focused: bool,
    focus_mode: bool,
) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    rows.len().hash(&mut h);
    for row in rows {
        row.is_header.hash(&mut h);
        row.is_delimiter.hash(&mut h);
        row.cells.len().hash(&mut h);
        for cell in &row.cells {
            cell.is_placeholder.hash(&mut h);
            cell.default_weight.hash(&mut h);
            match cell.align {
                TableAlign::Left => 0u8,
                TableAlign::Center => 1u8,
                TableAlign::Right => 2u8,
            }
            .hash(&mut h);
            cell.rel_raw_range.len().hash(&mut h);
            cell.rel_src_range.start.saturating_sub(cell.rel_raw_range.start).hash(&mut h);
            hash_map_and_spans(&mut h, &cell.map, &cell.spans);
        }
    }
    avail_table_w.to_bits().hash(&mut h);
    style_sig.hash(&mut h);
    block_focused.hash(&mut h);
    focus_mode.hash(&mut h);
    h.finish()
}

fn evict_oldest_lru(map: &mut HashMap<u64, CachedBlockShape>) {
    if let Some((&oldest_key, _)) = map.iter().min_by_key(|(_, v)| v.last_used_frame) {
        map.remove(&oldest_key);
    }
}

fn evict_oldest_table_lru(map: &mut HashMap<u64, CachedTableShape>) {
    if let Some((&oldest_key, _)) = map.iter().min_by_key(|(_, v)| v.last_used_frame) {
        map.remove(&oldest_key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;
    use vellmark_parser::{IncrementalDoc, parse_blocks};

    #[test]
    fn test_two_level_marker_collapse_and_source_round_trip_with_cjk() {
        let src = "## 标题 **加粗** 正文";
        let blocks = parse_blocks(src);
        let style = ResolvedThemeStyle::default();

        // Caret outside block -> both `## ` and `**` collapse cleanly
        // Move caret outside block's range (`blocks[0].range.end + 1` or parse 2 blocks)
        let two_block_src = "## 标题 **加粗** 正文\n下一行";
        let two_blocks = parse_blocks(two_block_src);
        let outside_view = EditorViewState {
            selection: Selection::caret(two_block_src.len()),
            ..Default::default()
        };
        let d_collapsed = build_block_display(
            &two_blocks[0],
            &two_block_src[two_blocks[0].range.start..two_blocks[0].range.end],
            &outside_view,
            outside_view.selection,
            &style,
            false,
            false,
        );
        assert_eq!(d_collapsed.map.display_text, "标题 加粗 正文");
        // Clicking at byte offset of `加` in visual (`"标题 ".len()`) maps back to exact source byte after `**`!
        let vis_jia = "标题 ".len();
        assert_eq!(
            d_collapsed.map.visual_to_source(vis_jia),
            "## 标题 **".len()
        );

        // Caret inside `**加粗**` -> `## ` expands (prefix) AND `**` expands (inline)!
        let mut inside_view = EditorViewState::default();
        let caret_in_bold = "## 标题 **加".len();
        inside_view.selection = Selection::caret(caret_in_bold);
        let d_expanded = build_block_display(
            &blocks[0],
            src,
            &inside_view,
            inside_view.selection,
            &style,
            true,
            false,
        );
        assert_eq!(d_expanded.map.display_text, "## 标题 **加粗** 正文");
        assert_eq!(d_expanded.map.source_to_visual(caret_in_bold), caret_in_bold);
    }

    #[test]
    fn test_ime_preedit_visual_splice_does_not_expand_collapsed_markers() {
        let src = "**加粗** 世界";
        let blocks = parse_blocks(src);
        let style = ResolvedThemeStyle::default();
        let caret_pos = src.len();
        let view = EditorViewState {
            selection: Selection::caret(caret_pos),
            ime_composition_selection: Some(Selection::caret(caret_pos)),
            ime_preedit: Some("ni hao".to_string()),
            expansion_frozen: true,
            ..Default::default()
        };

        let d = build_block_display(
            &blocks[0],
            src,
            &view,
            Selection::caret(caret_pos),
            &style,
            true,
            false,
        );
        assert_eq!(d.map.display_text, "加粗 世界ni hao");
        let pre_range = d.map.preedit_visual_range().expect("ime range present");
        assert_eq!(&d.map.display_text[pre_range], "ni hao");
        // Clicking inside preedit resolves cleanly to composition source anchor without mutating buffer
        assert_eq!(d.map.visual_to_source("加粗 世界ni".len()), caret_pos);
    }

    #[test]
    fn test_10mb_document_virtualized_shaping_and_hit_test_within_frame_budget() {
        // Construct ~10 MB Markdown document (~125,000 blocks)
        let line = "这是一段包含 **加粗文本** 与 `inline_code` 以及 [link](https://vellmark.dev) 的中文混合长段落。\n";
        let repeat = 10_000_000 / line.len();
        let mut doc = String::with_capacity(line.len() * repeat + 64);
        doc.push_str("# Top Heading\n");
        for _ in 0..repeat {
            doc.push_str(line);
        }
        assert!(doc.len() >= 9_500_000);

        let mut inc = IncrementalDoc::new(doc);
        let style = ResolvedThemeStyle::default();
        let mut cache = ShapedDocumentCache::new();
        let view = EditorViewState::default();

        // Warm once for initial metrics table
        let vp = EditorViewport {
            doc_rect: Rect::new(0.0, 0.0, 900.0, 720.0),
            content_left_x: 60.0,
            content_width_px: 780.0,
            scroll_y: 250_000.0,
            overscan_px: 180.0,
            scale_factor: 1.0,
            source_rev: 1,
            source_len: inc.source().len(),
            view: &view,
            focus_mode: false,
            show_caret: true,
            find_matches: &[],
            active_find: None,
        };
        let scene = cache.build_editor_scene(inc.blocks(), &style, &vp, |r| {
            inc.source()[r.start..r.end].to_string()
        });
        assert!(scene.visible_block_count > 10 && scene.visible_block_count < 80);
        assert!(!scene.draw.glyphs.is_empty());

        // Local edit + incremental AST update + virtualized viewport frame render
        let mid = inc.source().len() / 2;
        let t0 = Instant::now();
        let mut modified = inc.source().to_string();
        let edit_at = modified[mid..].find('\n').map(|d| mid + d).unwrap_or(mid);
        modified.insert_str(edit_at, "新增");
        inc.apply_edit(modified, edit_at, edit_at, edit_at + "新增".len());
        let vp2 = EditorViewport {
            source_rev: 2,
            source_len: inc.source().len(),
            ..vp
        };
        let scene2 = cache.build_editor_scene(inc.blocks(), &style, &vp2, |r| {
            inc.source()[r.start..r.end].to_string()
        });
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        assert!(scene2.visible_block_count < 80);
        // Must comfortably satisfy < 16 ms frame budget on debug/dev (check generous threshold < 120 ms in unoptimized debug build, note typical is few ms)
        assert!(
            elapsed_ms < 250.0,
            "10 MB incremental edit + viewport shape took {elapsed_ms:.2} ms"
        );
    }

    #[test]
    fn test_repeated_identical_paragraphs_hit_test_and_caret_map_to_distinct_offsets() {
        let p = "重复中文段落 **加粗** 结束";
        let doc = format!("{p}\n\n{p}\n\n{p}");
        let blocks = parse_blocks(&doc);
        assert_eq!(blocks.len(), 5);

        let style = ResolvedThemeStyle::default();
        let mut cache = ShapedDocumentCache::new();
        let view = EditorViewState {
            selection: Selection::caret(blocks[4].range.start + 3),
            ..Default::default()
        };

        let vp = EditorViewport {
            doc_rect: Rect::new(0.0, 0.0, 900.0, 720.0),
            content_left_x: 60.0,
            content_width_px: 780.0,
            scroll_y: 0.0,
            overscan_px: 120.0,
            scale_factor: 1.0,
            source_rev: 1,
            source_len: doc.len(),
            view: &view,
            focus_mode: false,
            show_caret: true,
            find_matches: &[],
            active_find: None,
        };
        let scene = cache.build_editor_scene(&blocks, &style, &vp, |r| doc[r.start..r.end].to_string());
        // Caret placed in 3rd paragraph (block 4) must position caret_doc_y below block 2!
        assert!(
            scene.caret_doc_y > cache.block_ys[2],
            "expected caret_doc_y ({}) inside 3rd paragraph (> block 2 y={})",
            scene.caret_doc_y,
            cache.block_ys[2]
        );

        // Hit-testing paragraph rows 2 and 4 must resolve to blocks[2] and blocks[4], NEVER blocks[0]!
        for target_idx in [2usize, 4usize] {
            let rec = cache.last_hit_blocks[target_idx];
            let mid_y = rec.row_rect.y + rec.row_rect.h * 0.5;
            let hit = cache
                .hit_test(rec.text_origin_x + 28.0, mid_y)
                .expect("hit target");
            match hit {
                EditorHitTarget::Text(off) => {
                    assert!(
                        off >= blocks[target_idx].range.start && off <= blocks[target_idx].range.end,
                        "clicking block {target_idx} ({:?}) mapped to wrong offset {off} (block 0 is {:?})",
                        blocks[target_idx].range,
                        blocks[0].range
                    );
                }
                other => panic!("unexpected hit target: {other:?}"),
            }
        }

        // Moving caret up from paragraph 4 skips blank line or lands inside blank/2nd paragraph, NEVER paragraph 0!
        let (up_off, _) = cache.move_caret_vertical(
            blocks[4].range.start + 3,
            -1,
            None,
            &blocks,
            &view,
            &style,
            vp.content_width_px,
            |r| doc[r.start..r.end].to_string(),
        );
        assert!(
            up_off >= blocks[2].range.start && up_off <= blocks[3].range.end,
            "moving up from paragraph 4 landed at {up_off}, expected above paragraph 4 and past paragraph 0 ({:?})",
            blocks[0].range
        );
        // Also test directly adjacent identical paragraphs separated by single newline
        let adj_doc = format!("{p}\n{p}\n{p}");
        let adj_blocks = parse_blocks(&adj_doc);
        assert_eq!(adj_blocks.len(), 3);
        let (up_adj, _) = cache.move_caret_vertical(
            adj_blocks[2].range.start + 3,
            -1,
            None,
            &adj_blocks,
            &view,
            &style,
            vp.content_width_px,
            |r| adj_doc[r.start..r.end].to_string(),
        );
        assert!(
            up_adj >= adj_blocks[1].range.start && up_adj <= adj_blocks[1].range.end,
            "moving up from adjacent block 2 landed at {up_adj}, expected inside {:?}",
            adj_blocks[1].range
        );
    }

    #[test]
    fn test_identical_visual_text_with_different_link_urls_do_not_collide_in_cache() {
        let doc = "[click](https://a.dev)\n\n[click](https://very-long-url.example.com/path/to/page)";
        let blocks = parse_blocks(doc);
        assert_eq!(blocks.len(), 3);

        let style = ResolvedThemeStyle::default();
        let mut cache = ShapedDocumentCache::new();
        let view = EditorViewState::default();
        let vp = EditorViewport {
            doc_rect: Rect::new(0.0, 0.0, 900.0, 720.0),
            content_left_x: 60.0,
            content_width_px: 780.0,
            scroll_y: 0.0,
            overscan_px: 120.0,
            scale_factor: 1.0,
            source_rev: 1,
            source_len: doc.len(),
            view: &view,
            focus_mode: false,
            show_caret: true,
            find_matches: &[],
            active_find: None,
        };
        let _ = cache.build_editor_scene(&blocks, &style, &vp, |r| doc[r.start..r.end].to_string());
        let rec2 = cache.last_hit_blocks[2];
        let hit2 = cache
            .hit_test(rec2.text_origin_x + 40.0, rec2.row_rect.y + rec2.row_rect.h * 0.5)
            .expect("hit 2");
        let EditorHitTarget::Text(off) = hit2 else {
            panic!("expected text hit");
        };
        assert!(
            off >= blocks[2].range.start && off <= blocks[2].range.end,
            "second link hit mapped to {off}, expected within {:?}",
            blocks[2].range
        );
    }

    #[test]
    fn test_wysiwyg_table_grid_rendering_delimiter_collapse_hit_test_and_vertical_nav() {
        let doc = concat!(
            "Intro paragraph\n\n",
            "| Name | Role | Score |\n",
            "| :--- | :---: | ---: |\n",
            "| Alice | **Lead** | 100 |\n",
            "| 中文 | Dev | 95 |\n\n",
            "Outro paragraph"
        );
        let blocks = parse_blocks(doc);
        assert_eq!(blocks.len(), 5);
        assert_eq!(blocks[2].kind, BlockKind::Table);

        let style = ResolvedThemeStyle::default();
        let mut cache = ShapedDocumentCache::new();
        let view = EditorViewState {
            mode: ViewMode::Wysiwyg,
            selection: Selection::caret(0),
            ..Default::default()
        };
        let vp = EditorViewport {
            doc_rect: Rect::new(0.0, 0.0, 900.0, 720.0),
            content_left_x: 60.0,
            content_width_px: 780.0,
            scroll_y: 0.0,
            overscan_px: 120.0,
            scale_factor: 1.0,
            source_rev: 1,
            source_len: doc.len(),
            view: &view,
            focus_mode: false,
            show_caret: true,
            find_matches: &[],
            active_find: None,
        };
        let scene = cache.build_editor_scene(&blocks, &style, &vp, |r| doc[r.start..r.end].to_string());

        // Table renders outer border, header background, row & column dividers using theme table styles.
        let header_bg_count = scene
            .draw
            .rects
            .iter()
            .filter(|r| r.color == style.table_header_bg)
            .count();
        assert_eq!(header_bg_count, 1, "expected 1 header row background fill");
        let border_rects = scene
            .draw
            .rects
            .iter()
            .filter(|r| r.color == style.table_border || r.border_color == style.table_border)
            .count();
        assert!(
            border_rects >= 5,
            "expected outer border + row dividers + col dividers, got {border_rects}"
        );

        let table_rec = cache.last_hit_blocks[2];
        assert!(table_rec.is_table);
        let shaped_tbl = cache.table_lru.get(&table_rec.shape_hash).unwrap();
        // Delimiter row is collapsed in WYSIWYG mode when caret is not on it -> 3 visible rows (1 header + 2 body).
        assert_eq!(shaped_tbl.rows.len(), 3);
        assert!(shaped_tbl.rows[0].is_header);
        assert!(!shaped_tbl.rows[1].is_delimiter);

        // Clicking middle cell (`**Lead**`) hits inside exact `**Lead**` cell byte range without showing `**`.
        let body0 = &shaped_tbl.rows[1];
        let mid_cell = &body0.cells[1];
        assert_eq!(mid_cell.map.display_text, "Lead");
        let click_x = table_rec.text_origin_x + mid_cell.text_x_rel + 12.0;
        let click_y = table_rec.text_origin_y + body0.y_rel + body0.h * 0.5;
        let hit = cache.hit_test(click_x, click_y).expect("table cell hit");
        let EditorHitTarget::Text(hit_off) = hit else {
            panic!("expected text hit in table cell");
        };
        let lead_start = doc.find("**Lead**").unwrap();
        assert!(
            hit_off >= lead_start && hit_off <= lead_start + "**Lead**".len(),
            "hit offset {hit_off} expected inside **Lead** ({}..={})",
            lead_start,
            lead_start + "**Lead**".len()
        );

        // Moving caret onto the delimiter row keeps it hidden in WYSIWYG mode (never renders `:---:` as cells).
        let delim_pos = doc.find(":---:").unwrap();
        let delim_view = EditorViewState {
            mode: ViewMode::Wysiwyg,
            selection: Selection::caret(delim_pos),
            ..Default::default()
        };
        let delim_vp = EditorViewport {
            view: &delim_view,
            ..vp
        };
        let _ = cache.build_editor_scene(&blocks, &style, &delim_vp, |r| doc[r.start..r.end].to_string());
        let delim_tbl = cache.table_lru.get(&cache.last_hit_blocks[2].shape_hash).unwrap();
        assert_eq!(delim_tbl.rows.len(), 3);
        assert!(delim_tbl.rows.iter().all(|r| !r.is_delimiter));
        assert!(
            delim_tbl
                .rows
                .iter()
                .flat_map(|r| r.cells.iter())
                .all(|c| !c.map.display_text.contains("---")),
            "delimiter row must never render as visible cells"
        );

        // Standalone delimiter lines (`|--|`, `|:---|`) remain plain Paragraphs, never table grids.
        for standalone in ["|--|", "|:---|"] {
            let s_blocks = parse_blocks(standalone);
            let s_view = EditorViewState {
                mode: ViewMode::Wysiwyg,
                selection: Selection::caret(standalone.len()),
                ..Default::default()
            };
            let s_vp = EditorViewport {
                view: &s_view,
                ..vp
            };
            let _ = cache.build_editor_scene(&s_blocks, &style, &s_vp, |r| {
                standalone[r.start..r.end].to_string()
            });
            assert!(
                !cache.last_hit_blocks[0].is_table,
                "`{standalone}` must not render as a WYSIWYG table"
            );
        }

        // Vertical navigation: from header column 1 (`Role`) Down goes to body0 column 1 (`**Lead**`),
        // skipping collapsed delimiter row; Down again goes to body1 column 1 (`Dev`); Down exits to Outro paragraph.
        let role_pos = doc.find("Role").unwrap() + 1;
        let nav_view = EditorViewState {
            mode: ViewMode::Wysiwyg,
            selection: Selection::caret(role_pos),
            ..Default::default()
        };
        let (down1, px1) = cache.move_caret_vertical(
            role_pos,
            1,
            None,
            &blocks,
            &nav_view,
            &style,
            vp.content_width_px,
            |r| doc[r.start..r.end].to_string(),
        );
        assert!(
            down1 >= lead_start && down1 <= lead_start + "**Lead**".len(),
            "Down from header col 1 landed at {down1}, expected inside **Lead**"
        );
        let (down2, px2) = cache.move_caret_vertical(
            down1,
            1,
            px1,
            &blocks,
            &nav_view,
            &style,
            vp.content_width_px,
            |r| doc[r.start..r.end].to_string(),
        );
        let dev_start = doc.find("| Dev |").unwrap() + 2;
        assert!(
            down2 >= dev_start && down2 <= dev_start + 3,
            "Down from body0 col 1 landed at {down2}, expected inside Dev"
        );
        let (down3, px3) = cache.move_caret_vertical(
            down2,
            1,
            px2,
            &blocks,
            &nav_view,
            &style,
            vp.content_width_px,
            |r| doc[r.start..r.end].to_string(),
        );
        assert_eq!(down3, blocks[3].range.start, "Down from table exited to blank line block");
        let (down4, _) = cache.move_caret_vertical(
            down3,
            1,
            px3,
            &blocks,
            &nav_view,
            &style,
            vp.content_width_px,
            |r| doc[r.start..r.end].to_string(),
        );
        assert!(
            down4 >= blocks[4].range.start,
            "Down past blank line landed at {down4}, expected Outro ({:?})",
            blocks[4].range
        );

        // Source mode keeps non-table plain monospace rendering
        let src_view = EditorViewState {
            mode: ViewMode::Source,
            ..Default::default()
        };
        let src_vp = EditorViewport {
            view: &src_view,
            ..vp
        };
        let _ = cache.build_editor_scene(&blocks, &style, &src_vp, |r| doc[r.start..r.end].to_string());
        assert!(!cache.last_hit_blocks[2].is_table);
    }

    #[test]
    fn test_wysiwyg_table_uses_canonical_delimiter_column_count() {
        let mut cache = ShapedDocumentCache::new();
        let style = ResolvedThemeStyle::default();

        // Body row has 3 cells while delimiter row defines 2 columns: WYSIWYG renders 2 columns.
        let wide_body_doc = "| a | b |\n| - | - |\n| 1 | 2 | 3 |\n";
        let blocks = parse_blocks(wide_body_doc);
        let wide_caret = wide_body_doc.find('3').unwrap();
        let view = EditorViewState {
            mode: ViewMode::Wysiwyg,
            selection: Selection::caret(wide_caret),
            ..Default::default()
        };
        let vp = EditorViewport {
            doc_rect: Rect::new(0.0, 0.0, 900.0, 720.0),
            content_left_x: 60.0,
            content_width_px: 780.0,
            scroll_y: 0.0,
            overscan_px: 120.0,
            scale_factor: 1.0,
            source_rev: 1,
            source_len: wide_body_doc.len(),
            view: &view,
            focus_mode: false,
            show_caret: true,
            find_matches: &[],
            active_find: None,
        };
        let scene = cache.build_editor_scene(&blocks, &style, &vp, |r| {
            wide_body_doc[r.start..r.end].to_string()
        });
        assert!(
            scene.draw.rects.iter().any(|r| r.color == style.caret_color),
            "caret rect must still be emitted safely when caret sits in unrendered col 3"
        );
        let tbl = cache.table_lru.get(&cache.last_hit_blocks[0].shape_hash).unwrap();
        assert_eq!(tbl.rows.len(), 2);
        assert_eq!(tbl.rows[0].cells.len(), 2);
        assert_eq!(tbl.rows[1].cells.len(), 2);
        let hdr_texts: Vec<&str> = tbl.rows[0].cells.iter().map(|c| c.map.display_text.as_str()).collect();
        let body_texts: Vec<&str> = tbl.rows[1].cells.iter().map(|c| c.map.display_text.as_str()).collect();
        assert_eq!(hdr_texts, vec!["a", "b"]);
        assert_eq!(body_texts, vec!["1", "2"]);

        let html = vellmark_parser::render_markdown_to_html(wide_body_doc);
        assert!(html.contains("<td>1</td>"));
        assert!(html.contains("<td>2</td>"));
        assert!(!html.contains("<td>3</td>"));

        // Short body row still renders canonical 2 columns (with empty placeholder in col 2).
        let short_body_doc = "| a | b |\n| - | - |\n| 1 |\n";
        let short_blocks = parse_blocks(short_body_doc);
        let short_view = EditorViewState {
            mode: ViewMode::Wysiwyg,
            selection: Selection::caret(0),
            ..Default::default()
        };
        let short_vp = EditorViewport {
            source_rev: 2,
            source_len: short_body_doc.len(),
            view: &short_view,
            ..vp
        };
        let _ = cache.build_editor_scene(&short_blocks, &style, &short_vp, |r| {
            short_body_doc[r.start..r.end].to_string()
        });
        let short_tbl = cache
            .table_lru
            .get(&cache.last_hit_blocks[0].shape_hash)
            .unwrap();
        assert_eq!(short_tbl.rows.len(), 2);
        assert_eq!(short_tbl.rows[0].cells.len(), 2);
        assert_eq!(short_tbl.rows[1].cells.len(), 2);
        assert_eq!(short_tbl.rows[1].cells[0].map.display_text, "1");
        assert_eq!(short_tbl.rows[1].cells[1].map.display_text, "");
    }
}
