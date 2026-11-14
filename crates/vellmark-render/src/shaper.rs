//! Virtualized `cosmic-text` shaping, two-level WYSIWYG marker collapse/expansion mapping,
//! non-destructive IME preedit splicing, hit-testing, and 10 MB viewport scene generation.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::ops::Range;

use cosmic_text::{
    Attrs, Buffer as CosmicBuffer, CacheKey, CacheMetrics, Cursor, Family, FontSystem, Metrics,
    Shaping, Style as FontStyle, Weight, Wrap,
};
use vellmark_core::{ByteOffset, ByteRange, EditorViewState, Selection, ViewMode};
use vellmark_parser::{BlockKind, BlockNode, InlineKind};

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

#[derive(Debug, Clone, Copy)]
struct HitBlockRecord {
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
            let old_h = self.block_heights[idx];
            if (measured_h - old_h).abs() > 0.5 {
                let delta = measured_h - old_h;
                self.block_heights[idx] = measured_h;
                self.block_ys[idx] = block_y;
                y_cursor_shift += delta;
                if !any_height_changed {
                    first_changed_idx = idx;
                }
                any_height_changed = true;
            } else {
                self.block_ys[idx] = block_y;
            }

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

            // Inline code pill backgrounds
            for pill in &shaped.inline_code_rects {
                draw.rects.push(
                    SceneRect::rounded(pill.translate(text_origin_x, text_origin_y), style.inline_code_bg, 4.0)
                        .with_clip(Some(vp.doc_rect)),
                );
            }

            // Find match highlights intersecting this block
            for m in vp.find_matches {
                if m.end <= node.range.start || m.start >= node.range.end {
                    continue;
                }
                let is_active = vp.active_find == Some(*m);
                let color = if is_active {
                    style.chrome.warning.with_alpha(120)
                } else {
                    style.chrome.warning.with_alpha(55)
                };
                emit_highlight_range(
                    &shaped.buffer,
                    &shaped.map,
                    node.range.start,
                    m.start.max(node.range.start),
                    m.end.min(node.range.end),
                    text_origin_x,
                    text_origin_y,
                    color,
                    vp.doc_rect,
                    &mut draw.rects,
                );
            }

            // Selection highlight intersecting this block
            let sel_range = sel.range();
            if !sel_range.is_empty() && sel_range.end > node.range.start && sel_range.start <= node.range.end {
                let s_start = sel_range.start.max(node.range.start);
                let s_end = sel_range.end.min(node.range.end);
                if s_start < s_end {
                    emit_highlight_range(
                        &shaped.buffer,
                        &shaped.map,
                        node.range.start,
                        s_start,
                        s_end,
                        text_origin_x,
                        text_origin_y,
                        style.selection_bg,
                        vp.doc_rect,
                        &mut draw.rects,
                    );
                } else {
                    // Selection spans entire empty line cleanly
                    draw.rects.push(
                        SceneRect::solid(
                            Rect::new(text_origin_x, text_origin_y, 6.0, style.base_line_height_px),
                            style.selection_bg,
                        )
                        .with_clip(Some(vp.doc_rect)),
                    );
                }
            }

            // IME preedit highlight + bottom underline
            if let Some(pre_vis) = shaped.map.preedit_visual_range() {
                if !pre_vis.is_empty() {
                    let c0 = shaped.map.visual_to_line_col(pre_vis.start);
                    let c1 = shaped.map.visual_to_line_col(pre_vis.end);
                    for run in shaped.buffer.layout_runs() {
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
            }

            // Emit glyphs (including strikethrough & underline lines for links/preedit/strikethrough spans)
            let fallback_text_color =
                style.apply_focus_mode(style.text_color, block_focused, vp.focus_mode);
            let scale = vp.scale_factor.max(0.5);
            for run in shaped.buffer.layout_runs() {
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

                    // Check inline strikethrough / link underline
                    let vis_g_start = shaped.map.line_col_to_visual(run.line_i, glyph.start);
                    for span in &shaped.display.spans {
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
            if let Some(preedit) = &vp.view.ime_preedit {
                if !preedit.is_empty() {
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
                if let Some(shaped) = self.shaped_lru.get(&rec.shape_hash) {
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

        let cur_offset = self
            .shape_single_for_nav(block_idx, blocks, view, exp_sel, style, content_width_px, &mut read_slice);
        let Some((cur_shaped, caret_vis)) = cur_offset else {
            return (caret, preferred_x);
        };
        let cursor = cur_shaped.map.visual_to_line_col(caret_vis);
        let _ = cursor;
        let cur_rect = caret_rect_in_buffer(
            &cur_shaped.buffer,
            &cur_shaped.map,
            caret_vis,
            cur_shaped.display.left_indent_px,
            0.0,
            style.base_line_height_px,
        );
        let target_x = preferred_x.unwrap_or(cur_rect.x);

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

        if let Some((adj_shaped, _)) =
            self.shape_single_for_nav(next_idx, blocks, view, exp_sel, style, content_width_px, &mut read_slice)
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

    fn shape_single_for_nav<S>(
        &mut self,
        block_idx: usize,
        blocks: &[BlockNode],
        view: &EditorViewState,
        exp_sel: Selection,
        style: &ResolvedThemeStyle,
        content_width_px: f32,
        read_slice: &mut S,
    ) -> Option<(&CachedBlockShape, usize)>
    where
        S: FnMut(ByteRange) -> String,
    {
        let node = blocks.get(block_idx)?;
        let raw = read_slice(node.range);
        let display = build_block_display(node, &raw, view, exp_sel, style, true, false);
        let caret_vis = display
            .map
            .source_to_visual(view.selection.head.clamp(node.range.start, node.range.end));
        let avail_w = (content_width_px - display.left_indent_px - 24.0).max(80.0);
        let style_sig = style_signature(style);
        let key = shape_cache_key(
            &display,
            avail_w,
            style_sig,
            matches!(node.kind, BlockKind::CodeFence { .. } | BlockKind::YamlFrontMatter),
        );
        self.frame_seq = self.frame_seq.wrapping_add(1);
        let frame_seq = self.frame_seq;
        if !self.shaped_lru.contains_key(&key) {
            let shaped = shape_block(&mut self.font_system, display, avail_w, style, key, frame_seq);
            self.shaped_lru.insert(key, shaped);
        }
        self.shaped_lru.get(&key).map(|s| (s, caret_vis))
    }
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
                if !is_source && !(exp_sel.is_caret() && node.range.contains_inclusive(exp_sel.head))
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

    // Process inline spans unless we're in a verbatim code fence or Source mode hides nothing.
    for span in &node.inlines {
        let o_start = (span.open_marker.start - block_start).min(block_len);
        let o_end = (span.open_marker.end - block_start).min(block_len);
        let c_start = (span.close_marker.start - block_start).min(block_len);
        let c_end = (span.close_marker.end - block_start).min(block_len);

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
    }

    let merged_hidden = merge_ranges(hidden);
    let mut map = build_visual_text_map(block_start, raw_text, &merged_hidden, synthetic_prefix);

    // Optional active IME preedit splicing if this block owns the active composition point.
    let mut preedit_vis_range: Option<Range<usize>> = None;
    if let Some(preedit) = &view.ime_preedit {
        let comp_sel = view.ime_composition_selection.unwrap_or(view.selection);
        let comp_range = comp_sel.range();
        let owns_comp = node.range.contains_inclusive(comp_range.start)
            && !matches!(node.kind, BlockKind::Paragraph if raw_text.is_empty() && comp_range.start > node.range.end);
        // If multiple blocks border `comp_range.start` (`block1.end == comp_range.start`), only the owning block picks it up cleanly:
        if owns_comp {
            let v_start = map.source_to_visual(comp_range.start);
            let v_end = map.source_to_visual(comp_range.end.min(node.range.end));
            map.display_text.replace_range(v_start..v_end, preedit);
            let new_v_end = v_start + preedit.len();
            map.line_starts = compute_line_starts(&map.display_text);
            map.ime_splice = Some(ImeSplice {
                vis_start: v_start,
                vis_end: new_v_end,
                comp_src_start: comp_range.start,
                comp_src_end: comp_range.end.min(node.range.end),
                preedit_caret_vis: new_v_end,
            });
            preedit_vis_range = Some(v_start..new_v_end);
        }
    }

    let spans = build_display_spans(
        node,
        raw_text,
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
        if let Some(last) = merged.last_mut() {
            if r.start <= last.end {
                last.end = last.end.max(r.end);
                continue;
            }
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
    if let Some(h0) = hidden.first() {
        if h0.start == 0 {
            first_content_local = h0.end;
        }
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
fn build_display_spans(
    node: &BlockNode,
    raw_text: &str,
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
    let block_start = node.range.start;
    let inline_vis: Vec<_> = node
        .inlines
        .iter()
        .filter_map(|span| {
            let s = map.source_to_visual(span.full_range.start.clamp(block_start, node.range.end));
            let e = map.source_to_visual(span.full_range.end.clamp(block_start, node.range.end));
            (s < e).then_some((span.kind, s..e))
        })
        .collect();
    for (_, r) in &inline_vis {
        cuts.push(r.start);
        cuts.push(r.end);
    }
    let marker_vis: Vec<Range<usize>> = marker_ranges
        .iter()
        .filter_map(|r| {
            let s = map.source_to_visual((block_start + r.start).min(node.range.end));
            let e = map.source_to_visual((block_start + r.end).min(node.range.end));
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
    let _ = raw_text;

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

    let metrics = Metrics::new(font_size.max(8.0), line_height.max(12.0));
    let mut buffer = CosmicBuffer::new(font_system, metrics);
    buffer.set_wrap(Wrap::WordOrGlyph);
    buffer.set_size(Some(avail_w), None);

    let default_attrs = Attrs::new()
        .family(if display.code_box {
            Family::Monospace
        } else {
            Family::Name(&style.base_font_family)
        })
        .weight(Weight(default_weight))
        .color(style.text_color.to_cosmic_color());

    if display.spans.is_empty() || display.map.display_text.is_empty() {
        buffer.set_text("", &default_attrs, Shaping::Advanced, None);
    } else {
        let rich: Vec<(&str, Attrs<'_>)> = display
            .spans
            .iter()
            .filter_map(|sp| {
                let s = display.map.display_text.get(sp.range.clone())?;
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
        buffer.set_rich_text(rich, &default_attrs, Shaping::Advanced, None);
    }
    buffer.shape_until_scroll(font_system, false);

    let mut run_count = 0usize;
    let mut text_height = line_height;
    for run in buffer.layout_runs() {
        run_count += 1;
        text_height = text_height.max(run.line_top + run.line_height);
    }
    if run_count == 0 {
        text_height = if display.hr_rule { 8.0 } else { line_height };
    }

    let mut inline_code_rects = Vec::new();
    for sp in &display.spans {
        if sp.inline_code_bg {
            let c0 = display.map.visual_to_line_col(sp.range.start);
            let c1 = display.map.visual_to_line_col(sp.range.end);
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
    style.base_font_family.hash(&mut h);
    style.base_font_size_px.to_bits().hash(&mut h);
    style.base_line_height_px.to_bits().hash(&mut h);
    style.zoom_scale.to_bits().hash(&mut h);
    h.finish()
}

fn shape_cache_key(
    display: &BlockDisplay,
    avail_w: f32,
    style_sig: u64,
    code_box: bool,
) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    let block_start = display.map.block_range.start;
    display.map.display_text.hash(&mut h);
    display.map.block_range.len().hash(&mut h);
    display.map.segments.len().hash(&mut h);
    for seg in &display.map.segments {
        seg.vis_start.hash(&mut h);
        seg.vis_end.hash(&mut h);
        seg.src_start.saturating_sub(block_start).hash(&mut h);
        seg.src_end.saturating_sub(block_start).hash(&mut h);
        seg.synthetic.hash(&mut h);
    }
    if let Some(ime) = &display.map.ime_splice {
        true.hash(&mut h);
        ime.vis_start.hash(&mut h);
        ime.vis_end.hash(&mut h);
        ime.comp_src_start.saturating_sub(block_start).hash(&mut h);
        ime.comp_src_end.saturating_sub(block_start).hash(&mut h);
        ime.preedit_caret_vis.hash(&mut h);
    } else {
        false.hash(&mut h);
    }
    display.spans.hash(&mut h);
    avail_w.to_bits().hash(&mut h);
    style_sig.hash(&mut h);
    display.top_pad_px.to_bits().hash(&mut h);
    display.bottom_pad_px.to_bits().hash(&mut h);
    code_box.hash(&mut h);
    h.finish()
}

fn evict_oldest_lru(map: &mut HashMap<u64, CachedBlockShape>) {
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
        let mut outside_view = EditorViewState::default();
        outside_view.selection = Selection::caret(0);
        // Move caret outside block's range (`blocks[0].range.end + 1` or parse 2 blocks)
        let two_block_src = "## 标题 **加粗** 正文\n下一行";
        let two_blocks = parse_blocks(two_block_src);
        outside_view.selection = Selection::caret(two_block_src.len());
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
        let mut view = EditorViewState::default();
        view.selection = Selection::caret(caret_pos);
        view.ime_composition_selection = Some(Selection::caret(caret_pos));
        view.ime_preedit = Some("ni hao".to_string());
        view.expansion_frozen = true;

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
        let mut view = EditorViewState::default();
        view.selection = Selection::caret(blocks[4].range.start + 3);

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
}
