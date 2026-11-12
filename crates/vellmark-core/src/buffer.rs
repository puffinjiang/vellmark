use ropey::Rope;
use serde::{Deserialize, Serialize};

/// UTF-8 byte offset in the source document (PRD 4.1.1 A).
pub type ByteOffset = usize;

/// Half-open byte range `[start, end)` in UTF-8 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ByteRange {
    pub start: ByteOffset,
    pub end: ByteOffset,
}

impl ByteRange {
    pub fn new(start: ByteOffset, end: ByteOffset) -> Self {
        assert!(start <= end, "ByteRange start ({start}) > end ({end})");
        Self { start, end }
    }

    pub fn empty(at: ByteOffset) -> Self {
        Self { start: at, end: at }
    }

    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub fn contains_inclusive(&self, offset: ByteOffset) -> bool {
        offset >= self.start && offset <= self.end
    }
}

/// Cursor and selection represented strictly in source UTF-8 byte offsets (PRD 4.1.1 A).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Selection {
    pub anchor: ByteOffset,
    pub head: ByteOffset,
}

impl Selection {
    pub fn caret(offset: ByteOffset) -> Self {
        Self {
            anchor: offset,
            head: offset,
        }
    }

    pub fn new(anchor: ByteOffset, head: ByteOffset) -> Self {
        Self { anchor, head }
    }

    pub fn is_caret(&self) -> bool {
        self.anchor == self.head
    }

    pub fn range(&self) -> ByteRange {
        if self.anchor <= self.head {
            ByteRange::new(self.anchor, self.head)
        } else {
            ByteRange::new(self.head, self.anchor)
        }
    }
}

/// Rope-backed source text buffer with monotonic revision counter.
#[derive(Debug, Clone)]
pub struct TextBuffer {
    rope: Rope,
    revision: u64,
}

impl Default for TextBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl TextBuffer {
    pub fn new() -> Self {
        Self {
            rope: Rope::new(),
            revision: 0,
        }
    }

    pub fn from_str(text: &str) -> Self {
        Self {
            rope: Rope::from_str(text),
            revision: 0,
        }
    }

    pub fn rope(&self) -> &Rope {
        &self.rope
    }

    /// Snapshot the current rope (O(1) structural sharing, PRD 4.5.1 A).
    pub fn snapshot_rope(&self) -> Rope {
        self.rope.clone()
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn len_bytes(&self) -> usize {
        self.rope.len_bytes()
    }

    pub fn is_empty(&self) -> bool {
        self.rope.len_bytes() == 0
    }

    pub fn slice_to_string(&self, range: ByteRange) -> String {
        let start_char = self.rope.byte_to_char(range.start);
        let end_char = self.rope.byte_to_char(range.end);
        self.rope.slice(start_char..end_char).to_string()
    }

    /// Replace byte range `[range.start, range.end)` with `replacement`, returning the deleted text.
    pub fn replace_bytes(&mut self, range: ByteRange, replacement: &str) -> String {
        let start_char = self.rope.byte_to_char(range.start);
        let end_char = self.rope.byte_to_char(range.end);
        let deleted = self.rope.slice(start_char..end_char).to_string();
        if start_char != end_char {
            self.rope.remove(start_char..end_char);
        }
        if !replacement.is_empty() {
            self.rope.insert(start_char, replacement);
        }
        self.revision += 1;
        deleted
    }

    /// Replace the entire buffer content and advance revision.
    pub fn reset_content(&mut self, new_text: &str) {
        self.rope = Rope::from_str(new_text);
        self.revision += 1;
    }

    pub fn crc32(&self) -> u32 {
        let mut hasher = crc32fast::Hasher::new();
        for chunk in self.rope.chunks() {
            hasher.update(chunk.as_bytes());
        }
        hasher.finalize()
    }
}

impl std::fmt::Display for TextBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for chunk in self.rope.chunks() {
            f.write_str(chunk)?;
        }
        Ok(())
    }
}
