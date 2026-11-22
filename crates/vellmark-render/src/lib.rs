//! `vellmark-render` computes virtualized block layout, viewport culling (PRD 6.2 #3),
//! Focus Mode & Typewriter Mode scroll positioning (PRD 4.7 V-01 & V-02), and cursor-line
//! vertical anchor stabilization when markers expand or collapse (PRD 4.1.1 A).

use vellmark_core::{ByteRange, Selection};
use vellmark_parser::{BlockKind, BlockNode};

pub mod shaper;
pub mod style;
pub mod syntax;

pub use shaper::{
    CodeBlockButtonHit, CodeBlockButtonKind, DrawList, EditorHitTarget, EditorSceneResult,
    EditorViewport, ImageLoadStatus, ImageStatusResolver, Rect, SceneGlyph, SceneImage, SceneRect,
    ShapedDocumentCache, TableButtonHit, TableButtonKind, VisualTextMap, shape_ui_text_into,
};
pub use style::{HeadingStyle, ResolvedThemeStyle, RgbaColor, UiChromeColors};
pub use syntax::{
    CodeLanguageEntry, CodeSyntaxSpan, code_language_catalog, code_language_display_label,
    highlight_code_slice, normalize_fence_language_token,
};

/// Computes the new scroll offset `scroll_y` so that the cursor line's screen Y coordinate
/// remains invariant when marker expansion changes the height of blocks before the cursor line (PRD 4.1.1 A).
pub fn stabilize_cursor_line_scroll(
    old_scroll_y: f32,
    old_cursor_doc_y: f32,
    new_cursor_doc_y: f32,
) -> f32 {
    let screen_y = old_cursor_doc_y - old_scroll_y;
    (new_cursor_doc_y - screen_y).max(0.0)
}

/// Computes the target `scroll_y` in Typewriter Mode so the active cursor line remains vertically
/// centered in the viewport (PRD 4.7 V-02).
pub fn typewriter_scroll_y(cursor_doc_y: f32, line_height: f32, viewport_height: f32) -> f32 {
    let target = cursor_doc_y + (line_height * 0.5) - (viewport_height * 0.5);
    target.max(0.0)
}

/// A laid-out block with document Y coordinates, height, marker expansion state, and Focus Mode state.
#[derive(Debug, Clone, PartialEq)]
pub struct LayoutBlock {
    pub block_index: usize,
    pub source_range: ByteRange,
    pub y: f32,
    pub height: f32,
    /// True if block-level or inline syntax markers are expanded in this block (PRD 4.1.1 A).
    pub has_expanded_markers: bool,
    /// True if this is the active block highlighted in Focus Mode (PRD 4.7 V-01); all other blocks are dimmed.
    pub is_focused: bool,
}

/// Full document block layout supporting O(log N) / virtualized viewport culling (PRD 6.2 #3).
#[derive(Debug, Clone, Default)]
pub struct DocumentLayout {
    pub blocks: Vec<LayoutBlock>,
    pub total_height: f32,
}

impl DocumentLayout {
    /// Build block layout from parsed `BlockNode`s and current `Selection`.
    pub fn compute(
        parsed_blocks: &[BlockNode],
        selection: Selection,
        base_line_height: f32,
    ) -> Self {
        let mut blocks = Vec::with_capacity(parsed_blocks.len());
        let mut y = 0.0f32;
        let line_h = base_line_height.max(16.0);

        for (idx, node) in parsed_blocks.iter().enumerate() {
            let block_scale = match node.kind {
                BlockKind::Heading { level: 1, .. } => 2.2,
                BlockKind::Heading { level: 2, .. } => 1.8,
                BlockKind::Heading { level: 3, .. } => 1.5,
                BlockKind::Heading { .. } => 1.25,
                BlockKind::CodeFence { .. } => 1.4,
                BlockKind::ThematicBreak => 0.8,
                _ => 1.0,
            };
            let height = line_h * block_scale;
            let has_expanded_markers = node.expanded_block_prefix(selection).is_some()
                || !node.expanded_inline_spans(selection).is_empty();
            let is_focused = node.range.contains_inclusive(selection.head);

            blocks.push(LayoutBlock {
                block_index: idx,
                source_range: node.range,
                y,
                height,
                has_expanded_markers,
                is_focused,
            });
            y += height;
        }

        Self {
            blocks,
            total_height: y,
        }
    }

    /// Return only the blocks intersecting `[scroll_y - overscan, scroll_y + viewport_height + overscan]`
    /// for virtualized GPU rendering (PRD 6.2 #3).
    pub fn visible_blocks(
        &self,
        scroll_y: f32,
        viewport_height: f32,
        overscan: f32,
    ) -> &[LayoutBlock] {
        if self.blocks.is_empty() {
            return &[];
        }
        let min_y = (scroll_y - overscan).max(0.0);
        let max_y = scroll_y + viewport_height + overscan;

        let start_idx = self.blocks.partition_point(|b| b.y + b.height < min_y);
        let end_idx = self.blocks.partition_point(|b| b.y <= max_y).max(start_idx);
        &self.blocks[start_idx..end_idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vellmark_parser::parse_blocks;

    #[test]
    fn test_anchor_stabilization_preserves_cursor_screen_y() {
        let old_scroll = 200.0;
        let old_cursor_y = 350.0; // screen_y = 150.0
        let new_cursor_y = 380.0; // block above grew by 30px
        let new_scroll = stabilize_cursor_line_scroll(old_scroll, old_cursor_y, new_cursor_y);
        assert_eq!(new_cursor_y - new_scroll, old_cursor_y - old_scroll);
    }

    #[test]
    fn test_virtualized_viewport_culling_and_focus_typewriter_modes() {
        let mut md = String::new();
        for i in 0..100 {
            md.push_str(&format!("Paragraph line {i}\n"));
        }
        let parsed = parse_blocks(&md);
        let sel = Selection::caret(25); // in 2nd paragraph
        let layout = DocumentLayout::compute(&parsed, sel, 24.0);

        assert_eq!(layout.blocks.len(), 100);
        assert!(layout.blocks[1].is_focused);
        assert!(!layout.blocks[0].is_focused);

        // Viewport showing y=[240.0..480.0] with 0 overscan only renders ~11 blocks out of 100
        let visible = layout.visible_blocks(240.0, 240.0, 0.0);
        assert!(visible.len() <= 12);
        assert!(
            visible
                .iter()
                .all(|b| b.y + b.height >= 240.0 && b.y <= 480.0)
        );

        // Typewriter mode centers cursor vertically
        let tw_scroll = typewriter_scroll_y(500.0, 24.0, 600.0);
        assert_eq!(tw_scroll, 212.0);
    }
}
