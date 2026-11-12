//! `vellmark-core` implements the core document model for Vellmark:
//! - Source-of-truth Rope text buffer (`buffer`)
//! - Transactions, undo/redo stack, and syntax-conversion escape undo (`undo`, PRD 4.1.1 B.1)
//! - Encoding & binary detection and round-trip preservation (`encoding`, PRD 4.5.2 & 5.2)
//! - Document state machine (`Clean / Dirty / Conflict / Missing`), versioned writes,
//!   and dual-slot atomic crash recovery (`state`, PRD 4.5.1)

pub mod buffer;
pub mod encoding;
pub mod state;
pub mod undo;

pub use buffer::{ByteOffset, ByteRange, Selection, TextBuffer};
pub use encoding::{
    DetectedEncoding, EncodingDecision, EncodingError, FileFormatMeta, LineEnding,
    detect_and_decode, encode_for_save,
};
pub use state::{
    ConflictActionOutcome, DiskSnapshot, DocState, DocumentSession, RecoverySlotReport,
    SaveOutcome,
};
pub use undo::{EditKind, EditOp, Transaction, UndoHistory, syntax_conversion_escape_op};
