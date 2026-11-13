//! Document and selection statistics for the status bar (PRD 4.7 V-04):
//! - Word count (CJK characters counted individually + contiguous ASCII/Latin words)
//! - Character count (with and without whitespace)
//! - Line count
//! - Estimated reading time
//! - Active selection word and character counts

use crate::buffer::{ByteRange, Selection, TextBuffer};
use serde::{Deserialize, Serialize};

/// Reading speed baseline (words / CJK characters per minute) for estimated reading time.
const WORDS_PER_MINUTE: usize = 300;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DocumentStats {
    pub word_count: usize,
    pub char_count: usize,
    pub char_count_no_spaces: usize,
    pub line_count: usize,
    /// Estimated reading time in seconds (at 300 WPM / CPM).
    pub reading_time_secs: usize,
    pub selection_word_count: usize,
    pub selection_char_count: usize,
}

impl DocumentStats {
    /// Compute full document and active selection statistics from a `TextBuffer` and `Selection` (PRD 4.7 V-04).
    pub fn compute(buffer: &TextBuffer, selection: Selection) -> Self {
        let full_text = buffer.to_string();
        let (word_count, char_count, char_count_no_spaces) = count_text_metrics(&full_text);
        let line_count = if full_text.is_empty() {
            1
        } else {
            full_text.bytes().filter(|&b| b == b'\n').count() + 1
        };
        let reading_time_secs = if word_count == 0 {
            0
        } else {
            word_count.saturating_mul(60).div_ceil(WORDS_PER_MINUTE)
        };

        let (selection_word_count, selection_char_count) = if selection.is_caret() {
            (0, 0)
        } else {
            let range: ByteRange = selection.range();
            let clamped_end = range.end.min(buffer.len_bytes());
            let clamped_start = range.start.min(clamped_end);
            let sel_text = buffer.slice_to_string(ByteRange::new(clamped_start, clamped_end));
            let (w, c, _) = count_text_metrics(&sel_text);
            (w, c)
        };

        Self {
            word_count,
            char_count,
            char_count_no_spaces,
            line_count,
            reading_time_secs,
            selection_word_count,
            selection_char_count,
        }
    }
}

/// Counts `(words, chars, chars_without_whitespace)` where each CJK ideograph/kana/hangul counts as 1 word
/// and each contiguous run of non-CJK alphanumeric characters counts as 1 word.
pub fn count_text_metrics(text: &str) -> (usize, usize, usize) {
    let mut words = 0usize;
    let mut chars = 0usize;
    let mut chars_no_space = 0usize;
    let mut in_word = false;

    for ch in text.chars() {
        chars += 1;
        if !ch.is_whitespace() {
            chars_no_space += 1;
        }

        if is_cjk_character(ch) {
            if in_word {
                in_word = false;
            }
            words += 1;
        } else if ch.is_alphanumeric() {
            if !in_word {
                words += 1;
                in_word = true;
            }
        } else if ch == '\'' || ch == '-' || ch == '_' {
            // Keep `in_word` state for hyphenated/contracted words like `don't` or `well-known`
        } else {
            in_word = false;
        }
    }

    (words, chars, chars_no_space)
}

fn is_cjk_character(ch: char) -> bool {
    let u = ch as u32;
    matches!(
        u,
        0x4E00..=0x9FFF   // CJK Unified Ideographs
        | 0x3400..=0x4DBF // CJK Unified Ideographs Extension A
        | 0x20000..=0x2A6DF // CJK Unified Ideographs Extension B
        | 0xF900..=0xFAFF // CJK Compatibility Ideographs
        | 0x3040..=0x309F // Hiragana
        | 0x30A0..=0x30FF // Katakana
        | 0xAC00..=0xD7AF // Hangul Syllables
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_stats_mixed_cjk_and_english_with_selection() {
        let buf = TextBuffer::from_str("你好世界 Hello world!\n第二行");
        // "你好世界" (4 CJK) + "Hello" (1) + "world" (1) + "第二行" (3 CJK) = 9 words
        let sel = Selection::new(0, "你好世界".len());
        let stats = DocumentStats::compute(&buf, sel);
        assert_eq!(stats.word_count, 9);
        assert_eq!(stats.line_count, 2);
        assert_eq!(stats.selection_word_count, 4);
        assert_eq!(stats.selection_char_count, 4);
        assert!(stats.reading_time_secs >= 1);
    }
}
