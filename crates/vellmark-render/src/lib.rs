//! `vellmark-render` computes block layout and implements cursor-line vertical anchor
//! stabilization when markers expand or collapse (PRD 4.1.1 A).

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anchor_stabilization_preserves_cursor_screen_y() {
        let old_scroll = 200.0;
        let old_cursor_y = 350.0; // screen_y = 150.0
        let new_cursor_y = 380.0; // block above grew by 30px
        let new_scroll = stabilize_cursor_line_scroll(old_scroll, old_cursor_y, new_cursor_y);
        assert_eq!(new_cursor_y - new_scroll, old_cursor_y - old_scroll);
    }
}
