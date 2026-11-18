//! `vellmark-core` implements the core document model for Vellmark:
//! - Source-of-truth Rope text buffer (`buffer`)
//! - Transactions, undo/redo stack, and syntax-conversion escape undo (`undo`, PRD 4.1.1 B.1)
//! - Encoding & binary detection and round-trip preservation (`encoding`, PRD 4.5.2 & 5.2)
//! - Document state machine (`Clean / Dirty / Conflict / Missing`), versioned writes,
//!   and dual-slot atomic crash recovery (`state`, PRD 4.5.1)

pub mod buffer;
pub mod editor;
pub mod encoding;
pub mod find_replace;
pub mod recovery_history;
pub mod state;
pub mod stats;
pub mod table;
pub mod undo;
pub mod workspace;

pub use buffer::{ByteOffset, ByteRange, Selection, TextBuffer};
pub use editor::{
    EditorAction, EditorContext, EditorController, EditorViewState, ViewMode,
    map_offset_across_reload,
};
pub use table::{
    GfmTableContext, ScannedGfmTable, TableAlign, TableCellSpan, TableRowParsed,
    find_enclosing_gfm_table_at, is_pipe_table_line, is_table_body_row_candidate,
    parse_table_block, parse_table_cell_rel_ranges, parse_table_delimiter_alignments,
    parse_table_delimiter_alignments_for_cells, scan_line_bounds, scan_line_content_bounds,
    try_scan_gfm_table_from_line, unescaped_pipe_offsets,
};
pub use encoding::{
    DetectedEncoding, EncodingDecision, EncodingError, FileFormatMeta, LineEnding,
    detect_and_decode, encode_for_save, normalize_line_endings,
};
pub use recovery_history::{
    DEFAULT_BACKUP_MAX_AGE_SECS, DEFAULT_MAX_BACKUPS_PER_FILE, DEFAULT_MAX_TOTAL_HISTORY_BYTES,
    DEFAULT_ORPHAN_GRACE_SECS, HISTORICAL_BACKUPS_DIR_NAME, HistoricalBackupData,
    HistoricalBackupEntry, HistoricalBackupPolicy, HistoricalBackupReason, PruneHistoryReport,
    RecoveryMaintenanceReport, archive_pending_recovery_to_history,
    archive_pending_recovery_to_history_at, clear_conflict_backups_for_path,
    clear_historical_backups_for_path, delete_historical_backup, discard_pending_recovery_entry,
    doc_history_key, doc_history_key_for_draft, list_historical_backups,
    list_historical_backups_for_path, maintain_recovery_root, measure_protected_snapshots_bytes,
    prune_historical_backups, prune_historical_backups_at, read_historical_backup,
};
pub use state::{
    AutoMergeOutcome, ConflictActionOutcome, ConflictDiffKind, ConflictDiffLine, DiskSnapshot,
    DocState, DocumentSession, ExternalSyncResult, MergeConflictHunk, PendingRecoveryEntry,
    RECOVERY_META_FILENAME, RecoveryMeta, RecoveryPayload, RecoverySlotReport, RecoveryWorker,
    SaveOutcome, SessionError, atomic_write_file, paths_refer_to_same_file,
    paths_share_underlying_file, scan_recovery_root,
};
pub use stats::{DocumentStats, count_text_metrics};
pub use find_replace::{
    FindError, FindMatch, FindOptions, FindReplaceOutcome, find_all_matches, find_next_match,
    find_prev_match, replace_all_in_buffer, replace_all_in_buffer_with_line_ending,
    replace_match_in_buffer, replace_match_in_buffer_with_line_ending,
};
pub use undo::{EditKind, EditOp, Transaction, UndoHistory, syntax_conversion_escape_op};
pub use workspace::{FileTreeNode, RecentFilesList, is_supported_document};
