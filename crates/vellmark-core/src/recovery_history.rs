//! Managed historical recovery backups (`history/`), metadata headers, checksum
//! verification, and bounded retention/eviction per PRD 4.5.1 F:
//! - Protected current crash snapshots (`current.slot{0,1}.rec`) never participate in
//!   count, age, or capacity eviction and are measured separately.
//! - Historical backups are stored atomically with CRC32 verification and metadata
//!   keyed by stable canonical document identity (`doc_key`), enforcing:
//!   - at most 20 historical backups per document (`DEFAULT_MAX_BACKUPS_PER_FILE`),
//!   - at most 14 days retention (`DEFAULT_BACKUP_MAX_AGE_SECS`),
//!   - at most 500 MB total history size (`DEFAULT_MAX_TOTAL_HISTORY_BYTES`), evicting oldest first.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::UNIX_EPOCH;

use crate::state::{
    BoundedFileHeader, DocState, MAX_RECOVERY_HEADER_BYTES, PendingRecoveryEntry,
    RECOVERY_META_FILENAME, SessionError, SlotInspect, SlotVerify, atomic_write_internal_file,
    canonicalize_lossy, compute_bytes_crc32, current_unix_secs, doc_state_from_tag,
    doc_state_to_tag, inspect_slot_report_header, json_escape_str, parse_json_string_at,
    read_file_bounded_header, remove_recovery_dir_and_empty_dir,
    slot_content_equals_base_streaming, verify_slot_report_crc_streaming,
};

pub const HISTORICAL_BACKUPS_DIR_NAME: &str = "history";
pub const DEFAULT_MAX_BACKUPS_PER_FILE: usize = 20;
pub const DEFAULT_BACKUP_MAX_AGE_SECS: u64 = 14 * 24 * 60 * 60;
pub const DEFAULT_MAX_TOTAL_HISTORY_BYTES: u64 = 500 * 1024 * 1024;
pub const DEFAULT_ORPHAN_GRACE_SECS: u64 = 300;
const HIST_MAGIC: &str = "VELLMARK_HIST_V1";

static NEXT_HISTORY_SEQ: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HistoricalBackupReason {
    ConflictOverwrittenTheirs,
    ConflictDiscardedMine,
    ConflictPreMergeMine,
    ClosedUnsaved,
    ClosedMissingUnsaved,
    UnrecoveredCrashSnapshot,
    Custom(String),
}

impl HistoricalBackupReason {
    /// Whether this backup was produced as a conflict-resolution safety backup
    /// (overwritten external disk version, discarded local edits, or pre-merge local edits).
    pub fn is_conflict_backup(&self) -> bool {
        matches!(
            self,
            Self::ConflictOverwrittenTheirs
                | Self::ConflictDiscardedMine
                | Self::ConflictPreMergeMine
        )
    }

    pub fn as_slug(&self) -> &str {
        match self {
            Self::ConflictOverwrittenTheirs => "conflict-overwritten-theirs",
            Self::ConflictDiscardedMine => "conflict-discarded-mine",
            Self::ConflictPreMergeMine => "conflict-pre-merge-mine",
            Self::ClosedUnsaved => "closed-unsaved",
            Self::ClosedMissingUnsaved => "closed-missing-unsaved",
            Self::UnrecoveredCrashSnapshot => "unrecovered-crash-snapshot",
            Self::Custom(s) => s.as_str(),
        }
    }

    pub fn from_slug(s: &str) -> Self {
        match s {
            "conflict-overwritten-theirs" => Self::ConflictOverwrittenTheirs,
            "conflict-discarded-mine" => Self::ConflictDiscardedMine,
            "conflict-pre-merge-mine" => Self::ConflictPreMergeMine,
            "closed-unsaved" => Self::ClosedUnsaved,
            "closed-missing-unsaved" => Self::ClosedMissingUnsaved,
            "unrecovered-crash-snapshot" => Self::UnrecoveredCrashSnapshot,
            other => Self::Custom(other.to_string()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HistoricalBackupPolicy {
    pub max_backups_per_file: usize,
    pub max_age_secs: u64,
    pub max_total_bytes: u64,
}

impl Default for HistoricalBackupPolicy {
    fn default() -> Self {
        Self {
            max_backups_per_file: DEFAULT_MAX_BACKUPS_PER_FILE,
            max_age_secs: DEFAULT_BACKUP_MAX_AGE_SECS,
            max_total_bytes: DEFAULT_MAX_TOTAL_HISTORY_BYTES,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistoricalBackupEntry {
    pub path: PathBuf,
    pub doc_key: String,
    pub source_path: Option<PathBuf>,
    pub created_unix_secs: u64,
    pub created_seq: u64,
    pub doc_state: DocState,
    pub reason: HistoricalBackupReason,
    pub base_same_as_payload: bool,
    pub base_len: usize,
    pub payload_len: usize,
    pub file_size_bytes: u64,
    pub header_crc32: u32,
    pub base_crc32: u32,
    pub payload_crc32: u32,
    data_offset: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistoricalBackupData {
    pub entry: HistoricalBackupEntry,
    pub base_content: Option<String>,
    pub payload: Vec<u8>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PruneHistoryReport {
    pub removed_expired: usize,
    pub removed_per_file_cap: usize,
    pub removed_global_cap: usize,
    pub removed_corrupt_or_tmp: usize,
    pub remaining_count: usize,
    pub remaining_bytes: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RecoveryMaintenanceReport {
    pub removed_clean_pending: usize,
    pub removed_corrupt_slots: usize,
    pub removed_orphan_dirs: usize,
    pub removed_legacy_backups: usize,
    pub history_prune: PruneHistoryReport,
    pub protected_snapshot_bytes: u64,
    pub protected_exceeds_limit: bool,
}

/// Stable per-document directory key used to enforce per-file retention across sessions.
pub fn doc_history_key(source_path: Option<&Path>) -> String {
    doc_history_key_for_draft(source_path, None)
}

/// Per-document history bucket key, distinguishing separate untitled drafts by `draft_id`.
pub fn doc_history_key_for_draft(source_path: Option<&Path>, draft_id: Option<&str>) -> String {
    if let Some(path) = source_path {
        let canonical = canonicalize_lossy(path);
        let bytes = canonical.to_string_lossy();
        let (h1, h2) = fnv1a_128(bytes.as_bytes());
        return format!("doc-{h1:016x}{h2:016x}");
    }
    if let Some(id) = draft_id.map(str::trim).filter(|id| !id.is_empty()) {
        let (h1, h2) = fnv1a_128(id.as_bytes());
        return format!("untitled-{h1:016x}{h2:016x}");
    }
    "untitled".to_string()
}

fn fnv1a_128(data: &[u8]) -> (u64, u64) {
    let mut h1: u64 = 0xcbf2_9ce4_8422_2325;
    let mut h2: u64 = 0xaf63_bd4c_8601_b7df;
    for &b in data {
        h1 ^= u64::from(b);
        h1 = h1.wrapping_mul(0x0100_0000_01b3);
        h2 ^= u64::from(b).wrapping_add(0x9e);
        h2 = h2.wrapping_mul(0x0100_0000_01b3).rotate_left(7);
    }
    (h1, h2)
}

fn sanitize_reason_slug(reason: &str) -> String {
    let sanitized: String = reason
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .take(48)
        .collect();
    if sanitized.is_empty() {
        "backup".to_string()
    } else {
        sanitized
    }
}

struct HistoryHeaderCrcInput<'a> {
    doc_key: &'a str,
    created_unix_secs: u64,
    created_seq: u64,
    doc_state: DocState,
    reason_slug: &'a str,
    source_path: Option<&'a Path>,
    base_same_as_payload: bool,
    base_len: usize,
    payload_len: usize,
    base_crc32: u32,
    payload_crc32: u32,
}

fn compute_history_header_crc(input: &HistoryHeaderCrcInput<'_>) -> u32 {
    let mut h = crc32fast::Hasher::new();
    h.update(HIST_MAGIC.as_bytes());
    h.update(b"\0");
    h.update(input.doc_key.as_bytes());
    h.update(b"\0");
    h.update(&input.created_unix_secs.to_le_bytes());
    h.update(&input.created_seq.to_le_bytes());
    h.update(doc_state_to_tag(input.doc_state).as_bytes());
    h.update(b"\0");
    h.update(input.reason_slug.as_bytes());
    h.update(b"\0");
    if let Some(p) = input.source_path {
        h.update(p.to_string_lossy().as_bytes());
    }
    h.update(b"\0");
    h.update(&[u8::from(input.base_same_as_payload)]);
    h.update(&(input.base_len as u64).to_le_bytes());
    h.update(&(input.payload_len as u64).to_le_bytes());
    h.update(&input.base_crc32.to_le_bytes());
    h.update(&input.payload_crc32.to_le_bytes());
    h.finalize()
}

pub(crate) struct WriteHistoricalBackupRequest<'a> {
    pub history_root: &'a Path,
    pub source_path: Option<&'a Path>,
    pub draft_id: Option<&'a str>,
    pub doc_state: DocState,
    pub reason: HistoricalBackupReason,
    pub base_content: Option<&'a str>,
    pub payload: &'a [u8],
    pub policy: HistoricalBackupPolicy,
    pub now_unix_secs: u64,
}

/// Atomically write a checksummed historical backup into `history_root/<doc_key>/`,
/// then enforce per-file/age/total-capacity limits while keeping the newly written backup.
pub(crate) fn write_historical_backup_entry(
    req: WriteHistoricalBackupRequest<'_>,
) -> std::io::Result<PathBuf> {
    let canonical_source = req.source_path.map(canonicalize_lossy);
    let doc_key = doc_history_key_for_draft(canonical_source.as_deref(), req.draft_id);
    let doc_dir = req.history_root.join(&doc_key);
    fs::create_dir_all(&doc_dir)?;

    let created_seq = NEXT_HISTORY_SEQ.fetch_add(1, Ordering::Relaxed);
    let reason_slug = sanitize_reason_slug(req.reason.as_slug());
    let pid = std::process::id();
    let filename = format!(
        "h-{:012}-p{pid}-s{created_seq:08}-{reason_slug}.hbak",
        req.now_unix_secs
    );
    let target = doc_dir.join(filename);

    let payload_crc32 = compute_bytes_crc32(req.payload);
    let (base_same_as_payload, stored_base_bytes, base_crc32) = match req.base_content {
        Some(base) if base.as_bytes() == req.payload => (true, &[][..], 0u32),
        Some(base) => (false, base.as_bytes(), compute_bytes_crc32(base.as_bytes())),
        None => (false, &[][..], 0u32),
    };
    let base_len = stored_base_bytes.len();
    let payload_len = req.payload.len();
    let header_crc32 = compute_history_header_crc(&HistoryHeaderCrcInput {
        doc_key: &doc_key,
        created_unix_secs: req.now_unix_secs,
        created_seq,
        doc_state: req.doc_state,
        reason_slug: &reason_slug,
        source_path: canonical_source.as_deref(),
        base_same_as_payload,
        base_len,
        payload_len,
        base_crc32,
        payload_crc32,
    });
    let source_path_json = match &canonical_source {
        Some(p) => json_escape_str(&p.to_string_lossy()),
        None => "null".to_string(),
    };

    let header = format!(
        "{HIST_MAGIC}\ndoc_key={doc_key}\ncreated_unix_secs={}\ncreated_seq={created_seq}\nstate={}\nreason={reason_slug}\nsource_path_json={source_path_json}\nbase_same_as_payload={}\nbase_len={base_len}\npayload_len={payload_len}\nbase_crc32={base_crc32}\npayload_crc32={payload_crc32}\nheader_crc32={header_crc32}\n---\n",
        req.now_unix_secs,
        doc_state_to_tag(req.doc_state),
        u8::from(base_same_as_payload),
    );

    let mut out = Vec::with_capacity(header.len() + base_len + payload_len);
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(stored_base_bytes);
    out.extend_from_slice(req.payload);

    atomic_write_internal_file(&target, &out)?;
    prune_historical_backups_at(
        req.history_root,
        req.policy,
        req.now_unix_secs,
        Some(&target),
        DEFAULT_ORPHAN_GRACE_SECS,
    );
    Ok(target)
}

enum HistoryHeaderInspect {
    Valid(HistoricalBackupEntry),
    Corrupt,
    IoError(std::io::ErrorKind),
}

fn inspect_historical_entry_header(path: &Path) -> HistoryHeaderInspect {
    let (file_size_bytes, data_offset, body_header) =
        match read_file_bounded_header(path, MAX_RECOVERY_HEADER_BYTES) {
            BoundedFileHeader::Missing => {
                return HistoryHeaderInspect::IoError(std::io::ErrorKind::NotFound);
            }
            BoundedFileHeader::Corrupt => return HistoryHeaderInspect::Corrupt,
            BoundedFileHeader::IoError(kind) => return HistoryHeaderInspect::IoError(kind),
            BoundedFileHeader::Valid {
                file_size_bytes,
                data_offset,
                body_header,
            } => (file_size_bytes, data_offset, body_header),
        };
    match parse_historical_header_fields(path, &body_header, data_offset, file_size_bytes) {
        Some(entry) => HistoryHeaderInspect::Valid(entry),
        None => HistoryHeaderInspect::Corrupt,
    }
}

fn parse_historical_header_fields(
    path: &Path,
    body_header: &str,
    data_offset: usize,
    file_size_bytes: u64,
) -> Option<HistoricalBackupEntry> {
    let mut lines = body_header.lines();
    if lines.next()? != HIST_MAGIC {
        return None;
    }

    let mut doc_key: Option<String> = None;
    let mut created_unix_secs: Option<u64> = None;
    let mut created_seq: Option<u64> = None;
    let mut doc_state: Option<DocState> = None;
    let mut reason_slug: Option<String> = None;
    let mut source_path: Option<PathBuf> = None;
    let mut has_source_field = false;
    let mut base_same_as_payload = false;
    let mut base_len: Option<usize> = None;
    let mut payload_len: Option<usize> = None;
    let mut base_crc32: Option<u32> = None;
    let mut payload_crc32: Option<u32> = None;
    let mut header_crc32: Option<u32> = None;

    for raw_line in lines {
        let (k, v) = raw_line.split_once('=')?;
        match k {
            "doc_key" => doc_key = Some(v.to_string()),
            "created_unix_secs" => created_unix_secs = v.parse().ok(),
            "created_seq" => created_seq = v.parse().ok(),
            "state" => doc_state = doc_state_from_tag(v),
            "reason" => reason_slug = Some(v.to_string()),
            "source_path_json" => {
                has_source_field = true;
                if v == "null" {
                    source_path = None;
                } else {
                    let (parsed, after) = parse_json_string_at(v, 0)?;
                    if after != v.len() {
                        return None;
                    }
                    source_path = Some(PathBuf::from(parsed));
                }
            }
            "base_same_as_payload" => base_same_as_payload = v == "1",
            "base_len" => base_len = v.parse().ok(),
            "payload_len" => payload_len = v.parse().ok(),
            "base_crc32" => base_crc32 = v.parse().ok(),
            "payload_crc32" => payload_crc32 = v.parse().ok(),
            "header_crc32" => header_crc32 = v.parse().ok(),
            _ => {}
        }
    }

    if !has_source_field {
        return None;
    }

    let doc_key = doc_key?;
    let created_unix_secs = created_unix_secs?;
    let created_seq = created_seq?;
    let doc_state = doc_state?;
    let reason_slug = reason_slug?;
    let base_len = base_len?;
    let payload_len = payload_len?;
    let base_crc32 = base_crc32?;
    let payload_crc32 = payload_crc32?;
    let header_crc32 = header_crc32?;

    let expected_total = (data_offset as u64)
        .checked_add(base_len as u64)?
        .checked_add(payload_len as u64)?;
    if expected_total != file_size_bytes {
        return None;
    }

    let expected_hdr_crc = compute_history_header_crc(&HistoryHeaderCrcInput {
        doc_key: &doc_key,
        created_unix_secs,
        created_seq,
        doc_state,
        reason_slug: &reason_slug,
        source_path: source_path.as_deref(),
        base_same_as_payload,
        base_len,
        payload_len,
        base_crc32,
        payload_crc32,
    });
    if expected_hdr_crc != header_crc32 {
        return None;
    }

    Some(HistoricalBackupEntry {
        path: path.to_path_buf(),
        doc_key,
        source_path,
        created_unix_secs,
        created_seq,
        doc_state,
        reason: HistoricalBackupReason::from_slug(&reason_slug),
        base_same_as_payload,
        base_len,
        payload_len,
        file_size_bytes,
        header_crc32,
        base_crc32,
        payload_crc32,
        data_offset,
    })
}

/// Read and CRC32-verify a historical backup entry's payload and optional `base` snapshot.
pub fn read_historical_backup(
    path: impl AsRef<Path>,
) -> Result<HistoricalBackupData, SessionError> {
    let path = path.as_ref();
    let entry = match inspect_historical_entry_header(path) {
        HistoryHeaderInspect::Valid(entry) => entry,
        HistoryHeaderInspect::Corrupt => return Err(SessionError::CorruptedSnapshot),
        HistoryHeaderInspect::IoError(kind) => {
            return Err(SessionError::Io(std::io::Error::from(kind)));
        }
    };
    let mut f = File::open(path)?;
    let mut raw = Vec::new();
    f.read_to_end(&mut raw)?;

    let base_end = entry
        .data_offset
        .checked_add(entry.base_len)
        .ok_or(SessionError::CorruptedSnapshot)?;
    let payload_end = base_end
        .checked_add(entry.payload_len)
        .ok_or(SessionError::CorruptedSnapshot)?;
    if raw.len() != payload_end {
        return Err(SessionError::CorruptedSnapshot);
    }

    let base_slice = &raw[entry.data_offset..base_end];
    if entry.base_len > 0 && compute_bytes_crc32(base_slice) != entry.base_crc32 {
        return Err(SessionError::CorruptedSnapshot);
    }
    let payload_slice = &raw[base_end..payload_end];
    if compute_bytes_crc32(payload_slice) != entry.payload_crc32 {
        return Err(SessionError::CorruptedSnapshot);
    }

    let base_content = if entry.base_same_as_payload {
        String::from_utf8(payload_slice.to_vec()).ok()
    } else if entry.base_len > 0 {
        Some(String::from_utf8(base_slice.to_vec()).map_err(|_| SessionError::CorruptedSnapshot)?)
    } else {
        None
    };

    Ok(HistoricalBackupData {
        entry,
        base_content,
        payload: payload_slice.to_vec(),
    })
}

fn entry_mtime_age_secs(path: &Path, now_unix_secs: u64) -> Option<u64> {
    let meta = fs::symlink_metadata(path).ok()?;
    let modified = meta.modified().ok()?;
    let mtime_secs = modified.duration_since(UNIX_EPOCH).ok()?.as_secs();
    Some(now_unix_secs.saturating_sub(mtime_secs))
}

/// Scan `history_root` for valid historical backups, ordered newest-first.
pub fn list_historical_backups(history_root: impl AsRef<Path>) -> Vec<HistoricalBackupEntry> {
    let mut entries = Vec::new();
    collect_history_entries(history_root.as_ref(), &mut entries, None);
    entries.sort_by(compare_newest_first);
    entries
}

/// List verified historical backups belonging to `source_path` (ordered newest-first).
/// For untitled drafts (`source_path == None`), aggregates across all per-draft untitled buckets.
pub fn list_historical_backups_for_path(
    history_root: impl AsRef<Path>,
    source_path: Option<&Path>,
) -> Vec<HistoricalBackupEntry> {
    let root = history_root.as_ref();
    let mut entries = Vec::new();
    if source_path.is_some() {
        let target_dir = root.join(doc_history_key(source_path));
        collect_dir_history_entries(&target_dir, &mut entries, None);
    } else {
        collect_history_entries(root, &mut entries, None);
        entries.retain(|entry| entry.source_path.is_none());
    }
    entries.sort_by(compare_newest_first);
    entries
}

fn compare_newest_first(
    a: &HistoricalBackupEntry,
    b: &HistoricalBackupEntry,
) -> std::cmp::Ordering {
    b.created_unix_secs
        .cmp(&a.created_unix_secs)
        .then_with(|| b.created_seq.cmp(&a.created_seq))
        .then_with(|| a.path.cmp(&b.path))
}

fn collect_history_entries(
    history_root: &Path,
    out: &mut Vec<HistoricalBackupEntry>,
    mut corrupt_or_tmp: Option<&mut Vec<PathBuf>>,
) {
    if !history_root.is_dir() {
        return;
    }
    collect_dir_history_entries(history_root, out, corrupt_or_tmp.as_deref_mut());
    let Ok(read_dir) = fs::read_dir(history_root) else {
        return;
    };
    for entry in read_dir.flatten() {
        if entry
            .file_type()
            .is_ok_and(|t| !t.is_symlink() && t.is_dir())
        {
            collect_dir_history_entries(&entry.path(), out, corrupt_or_tmp.as_deref_mut());
        }
    }
}

fn collect_dir_history_entries(
    dir: &Path,
    out: &mut Vec<HistoricalBackupEntry>,
    mut corrupt_or_tmp: Option<&mut Vec<PathBuf>>,
) {
    let Ok(read_dir) = fs::read_dir(dir) else {
        return;
    };
    for item in read_dir.flatten() {
        if !item
            .file_type()
            .is_ok_and(|t| !t.is_symlink() && t.is_file())
        {
            continue;
        }
        let path = item.path();
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if name.ends_with(".vellmark.tmp") {
            if let Some(bad) = corrupt_or_tmp.as_deref_mut() {
                bad.push(path);
            }
            continue;
        }
        if name.ends_with(".hbak") {
            match inspect_historical_entry_header(&path) {
                HistoryHeaderInspect::Valid(parsed) => out.push(parsed),
                HistoryHeaderInspect::Corrupt => {
                    if let Some(bad) = corrupt_or_tmp.as_deref_mut() {
                        bad.push(path);
                    }
                }
                HistoryHeaderInspect::IoError(_) => {}
            }
        }
    }
}

/// Enforce PRD 4.5.1 F history eviction policy (`max_age_secs`, `max_backups_per_file`,
/// `max_total_bytes`), cleaning stale `.vellmark.tmp`, corrupt `.hbak`, and empty bucket directories.
pub fn prune_historical_backups(
    history_root: impl AsRef<Path>,
    policy: HistoricalBackupPolicy,
) -> PruneHistoryReport {
    prune_historical_backups_at(
        history_root.as_ref(),
        policy,
        current_unix_secs(),
        None,
        DEFAULT_ORPHAN_GRACE_SECS,
    )
}

pub fn prune_historical_backups_at(
    history_root: &Path,
    policy: HistoricalBackupPolicy,
    now_unix_secs: u64,
    protected_just_written: Option<&Path>,
    corrupt_tmp_grace_secs: u64,
) -> PruneHistoryReport {
    let mut report = PruneHistoryReport::default();
    if !history_root.is_dir() {
        return report;
    }

    let mut entries = Vec::new();
    let mut bad_files = Vec::new();
    collect_history_entries(history_root, &mut entries, Some(&mut bad_files));

    // 0. Clean stale temporary or corrupted `.hbak` files older than grace window.
    for bad in bad_files {
        let should_remove = corrupt_tmp_grace_secs == 0
            || entry_mtime_age_secs(&bad, now_unix_secs)
                .is_some_and(|age| age >= corrupt_tmp_grace_secs);
        if should_remove && fs::remove_file(&bad).is_ok() {
            report.removed_corrupt_or_tmp += 1;
        }
    }

    // 1. Evict entries exceeding TTL (`max_age_secs`, default 14 days).
    let mut live_after_ttl = Vec::with_capacity(entries.len());
    for entry in entries {
        let is_protected_new = protected_just_written.is_some_and(|p| p == entry.path);
        let expired = now_unix_secs.saturating_sub(entry.created_unix_secs) > policy.max_age_secs;
        if expired && !is_protected_new && fs::remove_file(&entry.path).is_ok() {
            report.removed_expired += 1;
        } else {
            live_after_ttl.push(entry);
        }
    }

    // 2. Evict oldest entries per document key beyond `max_backups_per_file` (default 20).
    let mut by_doc: HashMap<String, Vec<HistoricalBackupEntry>> = HashMap::new();
    for entry in live_after_ttl {
        by_doc.entry(entry.doc_key.clone()).or_default().push(entry);
    }

    let mut live_after_per_file = Vec::new();
    for (_key, mut group) in by_doc {
        group.sort_by(compare_newest_first);
        let keep_limit = if protected_just_written.is_some() {
            policy.max_backups_per_file.max(1)
        } else {
            policy.max_backups_per_file
        };
        for (idx, entry) in group.into_iter().enumerate() {
            let is_protected_new = protected_just_written.is_some_and(|p| p == entry.path);
            if idx >= keep_limit && !is_protected_new && fs::remove_file(&entry.path).is_ok() {
                report.removed_per_file_cap += 1;
            } else {
                live_after_per_file.push(entry);
            }
        }
    }

    // 3. Enforce global capacity cap (`max_total_bytes`, default 500 MB), oldest-first across all docs.
    live_after_per_file.sort_by(|a, b| compare_newest_first(b, a)); // oldest first
    let mut total_bytes: u64 = live_after_per_file.iter().map(|e| e.file_size_bytes).sum();
    let mut kept = Vec::with_capacity(live_after_per_file.len());
    for entry in live_after_per_file {
        let is_protected_new = protected_just_written.is_some_and(|p| p == entry.path);
        if total_bytes > policy.max_total_bytes
            && !is_protected_new
            && fs::remove_file(&entry.path).is_ok()
        {
            total_bytes = total_bytes.saturating_sub(entry.file_size_bytes);
            report.removed_global_cap += 1;
            continue;
        }
        kept.push(entry);
    }

    // 4. Clean empty document subdirectories under `history_root`.
    if let Ok(read_dir) = fs::read_dir(history_root) {
        for sub in read_dir.flatten() {
            if sub.file_type().is_ok_and(|t| !t.is_symlink() && t.is_dir()) {
                let _ = fs::remove_dir(sub.path());
            }
        }
    }

    report.remaining_count = kept.len();
    report.remaining_bytes = total_bytes;
    report
}

/// Measure total byte size of protected files under `recovery_root` (excluding `history/`)
/// in O(stat) without reading snapshot bodies. Includes corrupt slot files so storage accounting
/// remains accurate until maintenance cleans them.
pub fn measure_protected_snapshots_bytes(recovery_root: impl AsRef<Path>) -> u64 {
    fn sum_dir_recursive(dir: &Path, depth: usize, total: &mut u64) {
        if depth > 4 || !dir.is_dir() {
            return;
        }
        let Ok(read_dir) = fs::read_dir(dir) else {
            return;
        };
        for item in read_dir.flatten() {
            let Ok(ft) = item.file_type() else {
                continue;
            };
            if ft.is_symlink() {
                continue;
            }
            let path = item.path();
            if ft.is_dir() {
                if path.file_name().and_then(|s| s.to_str()) == Some(HISTORICAL_BACKUPS_DIR_NAME) {
                    continue;
                }
                sum_dir_recursive(&path, depth + 1, total);
            } else if ft.is_file()
                && let Ok(meta) = fs::symlink_metadata(&path)
            {
                *total = total.saturating_add(meta.len());
            }
        }
    }

    let mut total = 0u64;
    sum_dir_recursive(recovery_root.as_ref(), 0, &mut total);
    total
}

/// Archive an unrecovered crash snapshot (`PendingRecoveryEntry`) into historical backups and
/// remove its protected current snapshot directory (PRD 4.5.1 F).
/// Uses `current_unix_secs()` as archive creation time so old crash snapshots remain inspectable
/// for the full retention window after archiving.
pub fn archive_pending_recovery_to_history(
    history_root: impl AsRef<Path>,
    entry: &PendingRecoveryEntry,
    reason: HistoricalBackupReason,
    policy: HistoricalBackupPolicy,
) -> Result<PathBuf, SessionError> {
    archive_pending_recovery_to_history_at(history_root, entry, reason, policy, current_unix_secs())
}

/// Deterministic clock-injected variant of [`archive_pending_recovery_to_history`].
pub fn archive_pending_recovery_to_history_at(
    history_root: impl AsRef<Path>,
    entry: &PendingRecoveryEntry,
    reason: HistoricalBackupReason,
    policy: HistoricalBackupPolicy,
    now_unix_secs: u64,
) -> Result<PathBuf, SessionError> {
    let fallback_draft_id;
    let draft_id = if entry.source_path.is_none() {
        match entry.draft_id.as_deref().map(str::trim) {
            Some(id) if !id.is_empty() => Some(id),
            _ => {
                fallback_draft_id = canonicalize_lossy(&entry.recovery_dir)
                    .to_string_lossy()
                    .into_owned();
                Some(fallback_draft_id.as_str())
            }
        }
    } else {
        None
    };
    let path = write_historical_backup_entry(WriteHistoricalBackupRequest {
        history_root: history_root.as_ref(),
        source_path: entry.source_path.as_deref(),
        draft_id,
        doc_state: entry.report.state,
        reason,
        base_content: Some(&entry.report.base_content),
        payload: entry.report.content.as_bytes(),
        policy,
        now_unix_secs,
    })?;
    remove_recovery_dir_and_empty_dir(&entry.recovery_dir);
    Ok(path)
}

/// Explicitly delete one managed historical backup file under `history_root`
/// and prune its document directory if now empty.
pub fn delete_historical_backup(
    history_root: impl AsRef<Path>,
    backup_path: impl AsRef<Path>,
) -> std::io::Result<bool> {
    let root = history_root.as_ref();
    let target = backup_path.as_ref();
    if !root.is_dir() {
        return Ok(false);
    }
    let canon_root = fs::canonicalize(root)?;
    let canon_target = match fs::canonicalize(target) {
        Ok(p) => p,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(e) => return Err(e),
    };
    if !canon_target.starts_with(&canon_root)
        || canon_target.extension().and_then(|e| e.to_str()) != Some("hbak")
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "path is not a managed historical backup",
        ));
    }
    fs::remove_file(&canon_target)?;
    if let Some(parent) = canon_target.parent()
        && parent != canon_root
    {
        let _ = fs::remove_dir(parent);
    }
    Ok(true)
}

fn clear_historical_backups_filtered(
    history_root: &Path,
    source_path: Option<&Path>,
    only_conflict: bool,
) -> usize {
    if !history_root.is_dir() {
        return 0;
    }
    let mut removed = 0usize;
    let mut candidate_dirs = HashSet::new();
    for entry in list_historical_backups_for_path(history_root, source_path) {
        if only_conflict && !entry.reason.is_conflict_backup() {
            continue;
        }
        if let Some(parent) = entry.path.parent()
            && parent != history_root
        {
            candidate_dirs.insert(parent.to_path_buf());
        }
        if fs::remove_file(&entry.path).is_ok() {
            removed += 1;
        }
    }
    if source_path.is_some() {
        let doc_dir = history_root.join(doc_history_key(source_path));
        candidate_dirs.insert(doc_dir);
    }
    for dir in candidate_dirs {
        let _ = fs::remove_dir(dir);
    }
    removed
}

/// Explicitly remove all conflict-resolution safety backups associated with `source_path`
/// (e.g., once the user confirms the resolved document is satisfactory).
pub fn clear_conflict_backups_for_path(
    history_root: impl AsRef<Path>,
    source_path: Option<&Path>,
) -> usize {
    clear_historical_backups_filtered(history_root.as_ref(), source_path, true)
}

/// Explicitly remove all historical backups (conflict + close + unrecovered snapshots)
/// associated with `source_path`.
pub fn clear_historical_backups_for_path(
    history_root: impl AsRef<Path>,
    source_path: Option<&Path>,
) -> usize {
    clear_historical_backups_filtered(history_root.as_ref(), source_path, false)
}

/// Permanently discard an unrecovered crash snapshot (`current.slot*.rec` + metadata + empty dir)
/// without creating a historical backup (when the user explicitly chooses "Discard").
pub fn discard_pending_recovery_entry(entry: &PendingRecoveryEntry) {
    remove_recovery_dir_and_empty_dir(&entry.recovery_dir);
}

struct ActiveDirSet {
    raw: HashSet<PathBuf>,
    canonical: HashSet<PathBuf>,
}

impl ActiveDirSet {
    fn new(dirs: &[PathBuf]) -> Self {
        let mut raw = HashSet::with_capacity(dirs.len());
        let mut canonical = HashSet::with_capacity(dirs.len());
        for d in dirs {
            raw.insert(d.clone());
            canonical.insert(canonicalize_lossy(d));
        }
        Self { raw, canonical }
    }

    fn contains(&self, dir: &Path) -> bool {
        if self.raw.is_empty() {
            return false;
        }
        self.raw.contains(dir) || self.canonical.contains(&canonicalize_lossy(dir))
    }
}

fn is_past_grace(path: &Path, grace_secs: u64, now_unix_secs: u64) -> bool {
    grace_secs == 0
        || entry_mtime_age_secs(path, now_unix_secs).is_some_and(|age| age >= grace_secs)
}

fn entry_might_exist(path: &Path) -> bool {
    !matches!(fs::symlink_metadata(path), Err(e) if e.kind() == std::io::ErrorKind::NotFound)
}

fn maintain_dir_recursive(
    dir: &Path,
    depth: usize,
    active_dirs: &ActiveDirSet,
    policy: HistoricalBackupPolicy,
    orphan_grace_secs: u64,
    now_unix_secs: u64,
    report: &mut RecoveryMaintenanceReport,
) {
    if depth > 4 || !dir.is_dir() {
        return;
    }
    if depth >= 1 && active_dirs.contains(dir) {
        return;
    }

    let dir_old_enough = depth >= 1 && is_past_grace(dir, orphan_grace_secs, now_unix_secs);

    // 1. Walk child entries: recurse into non-history subdirs first and clean loose tmp/legacy files.
    let dir_readable = match fs::read_dir(dir) {
        Ok(read_dir) => {
            for item in read_dir.flatten() {
                let Ok(ft) = item.file_type() else {
                    continue;
                };
                if ft.is_symlink() {
                    continue;
                }
                let cp = item.path();
                if ft.is_dir() {
                    if cp.file_name().and_then(|s| s.to_str()) == Some(HISTORICAL_BACKUPS_DIR_NAME)
                    {
                        continue;
                    }
                    maintain_dir_recursive(
                        &cp,
                        depth + 1,
                        active_dirs,
                        policy,
                        orphan_grace_secs,
                        now_unix_secs,
                        report,
                    );
                    continue;
                }
                if !ft.is_file() {
                    continue;
                }
                let Some(name) = cp.file_name().and_then(|s| s.to_str()) else {
                    continue;
                };
                if name.ends_with(".vellmark.tmp") {
                    if is_past_grace(&cp, orphan_grace_secs, now_unix_secs) {
                        let _ = fs::remove_file(&cp);
                    }
                } else if name.starts_with("history-")
                    && name.ends_with(".bak")
                    && entry_mtime_age_secs(&cp, now_unix_secs)
                        .is_some_and(|age| age > policy.max_age_secs)
                    && fs::remove_file(&cp).is_ok()
                {
                    report.removed_legacy_backups += 1;
                }
            }
            true
        }
        Err(_) => false,
    };

    // 2. Inspect recovery slots inside non-active directory without loading multi-MB bodies:
    //    - Bounded 64 KB header check
    //    - Fixed 64 KB buffer streaming CRC32 verification
    //    - Streaming content == base comparison only when state == Clean
    //    Slots that fail with I/O/permission errors (`Unreadable`) are NEVER treated as corrupt.
    let mut intact_slots = Vec::with_capacity(2);
    let mut corrupt_slots = Vec::with_capacity(2);
    let mut unreadable_slots = 0usize;
    for slot in [0u8, 1u8] {
        let sp = dir.join(format!("current.slot{slot}.rec"));
        match inspect_slot_report_header(&sp) {
            SlotInspect::Missing => {}
            SlotInspect::Corrupt => corrupt_slots.push(sp),
            SlotInspect::Unreadable => unreadable_slots += 1,
            SlotInspect::HeaderValid(summary) => {
                match verify_slot_report_crc_streaming(&sp, &summary) {
                    SlotVerify::Valid => intact_slots.push((sp, summary)),
                    SlotVerify::Corrupt => corrupt_slots.push(sp),
                    SlotVerify::Unreadable => unreadable_slots += 1,
                }
            }
        }
    }

    if let Some((latest_path, latest_summary)) = intact_slots
        .iter()
        .max_by_key(|(_, s)| (s.sequence.unwrap_or(0), s.revision))
    {
        let meaningful = latest_summary.state != DocState::Clean
            || slot_content_equals_base_streaming(latest_path, latest_summary) != Some(true);
        if !meaningful && unreadable_slots == 0 && dir_readable {
            remove_recovery_dir_and_empty_dir(dir);
            report.removed_clean_pending += 1;
            return;
        }
        // Keep intact (and any unreadable) slots, purging only confirmed corrupt sibling slots once aged.
        for bad_slot in corrupt_slots {
            if is_past_grace(&bad_slot, orphan_grace_secs, now_unix_secs)
                && fs::remove_file(&bad_slot).is_ok()
            {
                report.removed_corrupt_slots += 1;
            }
        }
    } else {
        for bad_slot in corrupt_slots {
            if is_past_grace(&bad_slot, orphan_grace_secs, now_unix_secs)
                && fs::remove_file(&bad_slot).is_ok()
            {
                report.removed_corrupt_slots += 1;
            }
        }
        if unreadable_slots == 0 && dir_readable {
            let has_any_slot = entry_might_exist(&dir.join("current.slot0.rec"))
                || entry_might_exist(&dir.join("current.slot1.rec"));
            if !has_any_slot {
                let meta_path = dir.join(RECOVERY_META_FILENAME);
                if fs::symlink_metadata(&meta_path).is_ok()
                    && is_past_grace(&meta_path, orphan_grace_secs, now_unix_secs)
                {
                    let _ = fs::remove_file(&meta_path);
                }
            }
        }
    }

    // 3. Remove empty orphaned session directory if aged and we verified directory/slot readability.
    if depth >= 1
        && dir_old_enough
        && dir_readable
        && unreadable_slots == 0
        && fs::remove_dir(dir).is_ok()
    {
        report.removed_orphan_dirs += 1;
    }
}

/// Full maintenance pass for `recovery_root` + `history/`:
/// 1. Skips `active_recovery_dirs` completely without opening their slot files.
/// 2. Inspects inactive recovery directories with bounded headers + streaming CRC32 / base compare
///    to remove stale `Clean` snapshots, corrupt recovery slots, orphan `recovery-meta.json`,
///    leftover `.vellmark.tmp`, expired legacy `history-*.bak`, and empty orphaned session folders.
/// 3. Runs bounded history pruning (`20` per file, `14 days`, `500 MB`).
/// 4. Audits protected snapshot byte usage in O(stat) without ever deleting protected snapshots.
pub fn maintain_recovery_root(
    recovery_root: impl AsRef<Path>,
    active_recovery_dirs: &[PathBuf],
    policy: HistoricalBackupPolicy,
    orphan_grace_secs: u64,
    now_unix_secs: u64,
) -> RecoveryMaintenanceReport {
    let root = recovery_root.as_ref();
    let mut report = RecoveryMaintenanceReport::default();
    if !root.is_dir() {
        return report;
    }

    let active_set = ActiveDirSet::new(active_recovery_dirs);
    maintain_dir_recursive(
        root,
        0,
        &active_set,
        policy,
        orphan_grace_secs,
        now_unix_secs,
        &mut report,
    );

    // Enforce historical backup limits.
    let history_root = root.join(HISTORICAL_BACKUPS_DIR_NAME);
    report.history_prune = prune_historical_backups_at(
        &history_root,
        policy,
        now_unix_secs,
        None,
        orphan_grace_secs,
    );

    // Audit protected snapshots size (warn-only; never evict protected snapshots).
    report.protected_snapshot_bytes = measure_protected_snapshots_bytes(root);
    report.protected_exceeds_limit = report.protected_snapshot_bytes > policy.max_total_bytes;
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;

    fn unique_temp_dir(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = std::env::temp_dir().join(format!(
            "vellmark-hist-test-{tag}-{}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn test_historical_backup_roundtrip_crc_tamper_detection_and_base_dedup() {
        let root = unique_temp_dir("roundtrip");
        let hist_root = root.join("history");
        let doc = root.join("note.md");
        fs::write(&doc, "same bytes").unwrap();

        let created = write_historical_backup_entry(WriteHistoricalBackupRequest {
            history_root: &hist_root,
            source_path: Some(&doc),
            draft_id: None,
            doc_state: DocState::Conflict,
            reason: HistoricalBackupReason::ConflictDiscardedMine,
            base_content: Some("same bytes"),
            payload: b"same bytes",
            policy: HistoricalBackupPolicy::default(),
            now_unix_secs: 1_700_000_000,
        })
        .unwrap();

        let listed = list_historical_backups(&hist_root);
        assert_eq!(listed.len(), 1);
        assert!(listed[0].base_same_as_payload);
        assert_eq!(listed[0].base_len, 0);

        let loaded = read_historical_backup(&created).unwrap();
        assert_eq!(loaded.base_content.as_deref(), Some("same bytes"));
        assert_eq!(loaded.payload, b"same bytes");

        // Corrupt last payload byte -> must fail CRC verification
        let mut corrupted = fs::read(&created).unwrap();
        let last = corrupted.len() - 1;
        corrupted[last] ^= 0xFF;
        fs::write(&created, &corrupted).unwrap();
        assert!(matches!(
            read_historical_backup(&created),
            Err(SessionError::CorruptedSnapshot)
        ));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn test_prune_enforces_per_file_cap_14day_ttl_and_global_capacity_oldest_first() {
        let root = unique_temp_dir("prune-rules");
        let hist_root = root.join("history");
        let doc_a = root.join("a.md");
        let doc_b = root.join("b.md");
        let doc_c = root.join("c.md");
        fs::write(&doc_a, "a").unwrap();
        fs::write(&doc_b, "b").unwrap();
        fs::write(&doc_c, "c").unwrap();

        let policy = HistoricalBackupPolicy {
            max_backups_per_file: 3,
            max_age_secs: 14 * 24 * 60 * 60,
            max_total_bytes: u64::MAX,
        };

        // 5 backups for doc_a -> automatic pruning on write keeps newest 3 immediately (seqs for i=2,3,4)
        for i in 0..5u64 {
            write_historical_backup_entry(WriteHistoricalBackupRequest {
                history_root: &hist_root,
                source_path: Some(&doc_a),
                doc_state: DocState::Conflict,
                draft_id: None,
                reason: HistoricalBackupReason::ConflictOverwrittenTheirs,
                base_content: None,
                payload: format!("payload-a-{i}").as_bytes(),
                policy,
                now_unix_secs: 10_000_000 + i,
            })
            .unwrap();
        }
        let doc_a_kept = list_historical_backups_for_path(&hist_root, Some(&doc_a));
        assert_eq!(doc_a_kept.len(), 3);
        let newest_a = read_historical_backup(&doc_a_kept[0].path).unwrap();
        assert_eq!(newest_a.payload, b"payload-a-4");

        // Write 1 backup for doc_b at t=10_000_005 (not yet expired at t=10_000_020)
        write_historical_backup_entry(WriteHistoricalBackupRequest {
            history_root: &hist_root,
            source_path: Some(&doc_b),
            doc_state: DocState::Dirty,
            draft_id: None,
            reason: HistoricalBackupReason::ClosedUnsaved,
            base_content: None,
            payload: b"old-b",
            policy,
            now_unix_secs: 10_000_005,
        })
        .unwrap();
        // 2 backups for doc_c written 16 days later -> expires ONLY doc_b (>14d)? No, doc_a is also >14d if +16d;
        // instead write doc_c fresh at 10_000_010..12, then test explicit TTL sweep 15 days after doc_b (keep doc_c <14d!).
        for i in 0..2u64 {
            write_historical_backup_entry(WriteHistoricalBackupRequest {
                history_root: &hist_root,
                source_path: Some(&doc_c),
                doc_state: DocState::Dirty,
                draft_id: None,
                reason: HistoricalBackupReason::ClosedUnsaved,
                base_content: None,
                payload: &[b'x'; 200],
                policy,
                now_unix_secs: 10_000_100 + i * 200_000, // 10_000_100 and 10_200_100
            })
            .unwrap();
        }

        // At now = 10_000_005 + 14d + 1 = 11_209_606:
        // - doc_a (10_000_002..4): expired (> 14d) -> 3 removed
        // - doc_b (10_000_005): expired (> 14d) -> 1 removed
        // - doc_c[0] (10_000_100): 1_209_506s < 14d (1_209_600s) -> kept even through check_now + 5!
        // - doc_c[1] (10_200_100): fresh -> kept!
        let check_now = 10_000_005 + 14 * 24 * 60 * 60 + 1;
        let prune1 = prune_historical_backups_at(&hist_root, policy, check_now, None, 0);
        assert_eq!(prune1.removed_expired, 4);
        assert!(list_historical_backups_for_path(&hist_root, Some(&doc_a)).is_empty());
        assert!(list_historical_backups_for_path(&hist_root, Some(&doc_b)).is_empty());
        assert_eq!(
            list_historical_backups_for_path(&hist_root, Some(&doc_c)).len(),
            2
        );

        // Now test global cap across multiple documents + protection of newer entries
        let doc_c_entries = list_historical_backups_for_path(&hist_root, Some(&doc_c));
        let doc_c_total: u64 = doc_c_entries.iter().map(|e| e.file_size_bytes).sum();

        // Add 2 newer backups for doc_a at check_now + 1..2; global budget == doc_a_new_total keeps only doc_a's 2 newest!
        let mut new_a_total = 0u64;
        for i in 1..=2u64 {
            let p = write_historical_backup_entry(WriteHistoricalBackupRequest {
                history_root: &hist_root,
                source_path: Some(&doc_a),
                doc_state: DocState::Dirty,
                draft_id: None,
                reason: HistoricalBackupReason::ClosedUnsaved,
                base_content: None,
                payload: &[b'a'; 150],
                policy,
                now_unix_secs: check_now + i,
            })
            .unwrap();
            new_a_total += fs::metadata(p).unwrap().len();
        }
        assert!(doc_c_total > 0);
        let tight_policy = HistoricalBackupPolicy {
            max_backups_per_file: 20,
            max_age_secs: 14 * 24 * 60 * 60,
            max_total_bytes: new_a_total,
        };
        let prune2 = prune_historical_backups_at(&hist_root, tight_policy, check_now + 5, None, 0);
        assert_eq!(prune2.removed_global_cap, 2);
        assert!(prune2.remaining_bytes <= tight_policy.max_total_bytes);
        assert_eq!(list_historical_backups(&hist_root).len(), 2);
        assert!(list_historical_backups_for_path(&hist_root, Some(&doc_c)).is_empty());

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn test_maintain_recovery_root_protects_current_snapshots_and_cleans_orphans_and_corrupt_files()
    {
        use crate::state::DocumentSession;

        let root = unique_temp_dir("maintain-root");
        let rec_root = root.join("recovery");
        let hist_root = rec_root.join(HISTORICAL_BACKUPS_DIR_NAME);

        // Active dirty session (must be protected even if over budget!)
        let dirty_dir = rec_root.join("session-dirty");
        let mut dirty_doc = DocumentSession::new_untitled(Some(dirty_dir.clone()));
        dirty_doc.edit(
            crate::buffer::ByteRange::empty(0),
            &"important unsaved text ".repeat(50),
            crate::buffer::Selection::caret(0),
            crate::buffer::Selection::caret(10),
            crate::undo::EditKind::Typing,
        );
        dirty_doc.flush_current_recovery_snapshot().unwrap();
        drop(dirty_doc);

        // Stale clean session directory (left over on disk) + completely empty orphan session dir + tmp debris
        let clean_dir = rec_root.join("session-stale-clean");
        let clean_doc = DocumentSession::new_untitled(Some(clean_dir.clone()));
        drop(clean_doc);
        fs::create_dir_all(&clean_dir).unwrap();
        let report_clean = crate::state::RecoverySlotReport {
            slot: 0,
            sequence: 1,
            revision: 0,
            state: DocState::Clean,
            content: String::new(),
            base_content: String::new(),
            theirs_content: None,
            crc32: 0,
        };
        // Actually write a valid Clean snapshot via a temp dirty+save or direct helper? Simplest: empty orphan + tmp
        let _ = report_clean;
        let orphan_dir = rec_root.join("session-empty-orphan");
        fs::create_dir_all(&orphan_dir).unwrap();
        fs::write(
            orphan_dir.join(".current.slot0.rec.1.2.vellmark.tmp"),
            b"stale",
        )
        .unwrap();

        fs::create_dir_all(&hist_root).unwrap();
        fs::write(hist_root.join("corrupt.hbak"), b"not valid history").unwrap();
        fs::write(hist_root.join(".partial.vellmark.tmp"), b"tmp").unwrap();

        // Tiny byte budget -> protected current snapshot exceeds limit, warns ONLY and NEVER deletes it!
        let small_policy = HistoricalBackupPolicy {
            max_backups_per_file: 20,
            max_age_secs: 14 * 24 * 60 * 60,
            max_total_bytes: 16,
        };
        let maint = maintain_recovery_root(&rec_root, &[], small_policy, 0, current_unix_secs());
        assert!(maint.protected_exceeds_limit);
        assert!(maint.protected_snapshot_bytes > 16);
        assert_eq!(maint.history_prune.removed_corrupt_or_tmp, 2);
        assert!(
            !orphan_dir.exists(),
            "empty orphan session dir must be cleaned"
        );
        assert!(
            DocumentSession::read_latest_intact_snapshot(&dirty_dir).is_some(),
            "protected dirty snapshot must NEVER be evicted even when exceeding budget"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn test_conflict_safety_backups_explicit_cleanup_and_legacy_bak_expiration() {
        let root = unique_temp_dir("conflict-cleanup");
        let rec_root = root.join("recovery");
        let hist_root = rec_root.join(HISTORICAL_BACKUPS_DIR_NAME);
        let doc_a = root.join("alpha.md");
        fs::create_dir_all(&root).unwrap();
        fs::write(&doc_a, "alpha base").unwrap();

        let policy = HistoricalBackupPolicy {
            max_backups_per_file: 20,
            max_age_secs: 14 * 24 * 60 * 60,
            max_total_bytes: 500 * 1024 * 1024,
        };
        let now = current_unix_secs();

        // Create 2 conflict safety backups + 1 close backup for doc_a
        let conflict1 = write_historical_backup_entry(WriteHistoricalBackupRequest {
            history_root: &hist_root,
            source_path: Some(&doc_a),
            draft_id: None,
            doc_state: DocState::Conflict,
            reason: HistoricalBackupReason::ConflictOverwrittenTheirs,
            base_content: Some("alpha base"),
            payload: b"theirs-1",
            policy,
            now_unix_secs: now,
        })
        .unwrap();
        let conflict2 = write_historical_backup_entry(WriteHistoricalBackupRequest {
            history_root: &hist_root,
            source_path: Some(&doc_a),
            draft_id: None,
            doc_state: DocState::Conflict,
            reason: HistoricalBackupReason::ConflictDiscardedMine,
            base_content: Some("alpha base"),
            payload: b"mine-2",
            policy,
            now_unix_secs: now + 1,
        })
        .unwrap();
        let close1 = write_historical_backup_entry(WriteHistoricalBackupRequest {
            history_root: &hist_root,
            source_path: Some(&doc_a),
            draft_id: None,
            doc_state: DocState::Dirty,
            reason: HistoricalBackupReason::ClosedUnsaved,
            base_content: Some("alpha base"),
            payload: b"close-draft",
            policy,
            now_unix_secs: now + 2,
        })
        .unwrap();

        assert!(conflict1.exists() && conflict2.exists() && close1.exists());
        // Single-entry delete rejects paths outside history_root
        assert!(delete_historical_backup(&hist_root, &doc_a).is_err());

        // Clearing ONLY conflict safety backups preserves close-archive backup
        assert_eq!(clear_conflict_backups_for_path(&hist_root, Some(&doc_a)), 2);
        assert!(!conflict1.exists() && !conflict2.exists());
        assert!(close1.exists());

        // Single-entry delete removes the remaining entry and cleans empty doc subdir
        assert!(delete_historical_backup(&hist_root, &close1).unwrap());
        assert!(list_historical_backups_for_path(&hist_root, Some(&doc_a)).is_empty());

        // Legacy pre-v1 `history-rev*.bak` left inside an orphan dir expires when past max_age_secs
        let legacy_dir = rec_root.join("legacy-tab-1");
        fs::create_dir_all(&legacy_dir).unwrap();
        let legacy_bak = legacy_dir.join("history-rev7.bak");
        fs::write(&legacy_bak, b"old legacy conflict backup").unwrap();

        // Not yet expired at `now + 1`
        let early = maintain_recovery_root(&rec_root, &[], policy, 0, now + 1);
        assert_eq!(early.removed_legacy_backups, 0);
        assert!(legacy_bak.exists());

        // Past 14-day TTL -> legacy conflict backup + empty dir are both cleaned!
        let after_ttl =
            maintain_recovery_root(&rec_root, &[], policy, 0, now + policy.max_age_secs + 60);
        assert_eq!(after_ttl.removed_legacy_backups, 1);
        assert!(!legacy_bak.exists());
        assert!(!legacy_dir.exists());

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn test_untitled_drafts_do_not_evict_each_other_past_20_and_single_draft_stays_bounded() {
        use crate::buffer::Selection;
        use crate::state::DocumentSession;
        use crate::undo::EditKind;

        let root = unique_temp_dir("untitled-isolation");
        let rec_root = root.join("recovery");
        let hist_root = rec_root.join(HISTORICAL_BACKUPS_DIR_NAME);
        let policy = HistoricalBackupPolicy::default();

        // 1. 25 separate untitled drafts closed unsaved must keep all 25 backups (no eviction at #21).
        for i in 0..25usize {
            let tab_rec = rec_root.join(format!("draft-tab-{i}"));
            let mut session = DocumentSession::new_untitled_with_recovery(
                Some(tab_rec),
                Some(hist_root.clone()),
                policy,
                std::time::Duration::from_millis(50),
            );
            session.edit(
                crate::buffer::ByteRange::empty(0),
                &format!("unique untitled draft content #{i}"),
                Selection::caret(0),
                Selection::caret(10),
                EditKind::Typing,
            );
            session.archive_unsaved_on_close().unwrap();
        }
        assert_eq!(list_historical_backups(&hist_root).len(), 25);
        assert_eq!(list_historical_backups_for_path(&hist_root, None).len(), 25);

        // 2. Single untitled draft writing > max_backups_per_file backups still stays bounded to cap.
        let tight_policy = HistoricalBackupPolicy {
            max_backups_per_file: 3,
            max_age_secs: DEFAULT_BACKUP_MAX_AGE_SECS,
            max_total_bytes: DEFAULT_MAX_TOTAL_HISTORY_BYTES,
        };
        let single_hist = root.join("single-history");
        for i in 0..5u64 {
            write_historical_backup_entry(WriteHistoricalBackupRequest {
                history_root: &single_hist,
                source_path: None,
                draft_id: Some("draft-same-single"),
                doc_state: DocState::Dirty,
                reason: HistoricalBackupReason::ClosedUnsaved,
                base_content: None,
                payload: format!("rev-{i}").as_bytes(),
                policy: tight_policy,
                now_unix_secs: 1_700_000_000 + i,
            })
            .unwrap();
        }
        assert_eq!(
            list_historical_backups_for_path(&single_hist, None).len(),
            3
        );
        assert_eq!(clear_historical_backups_for_path(&hist_root, None), 25);
        assert!(list_historical_backups(&hist_root).is_empty());

        // 3. 25 separate crashed untitled snapshots archived via `archive_pending_recovery_to_history`
        //    (including legacy entries lacking `draft_id`) must also retain all 25 entries.
        let crash_rec_root = root.join("crash-recovery");
        let crash_hist_root = crash_rec_root.join(HISTORICAL_BACKUPS_DIR_NAME);
        for i in 0..25usize {
            let tab_rec = crash_rec_root.join(format!("crash-draft-{i}"));
            let mut session = DocumentSession::new_untitled_with_recovery(
                Some(tab_rec),
                Some(crash_hist_root.clone()),
                policy,
                std::time::Duration::from_millis(50),
            );
            session.edit(
                crate::buffer::ByteRange::empty(0),
                &format!("crashed untitled draft #{i}"),
                Selection::caret(0),
                Selection::caret(10),
                EditKind::Typing,
            );
            session.flush_current_recovery_snapshot().unwrap();
        }
        let mut crashed = crate::state::scan_recovery_root(&crash_rec_root);
        assert_eq!(crashed.len(), 25);
        // Simulate legacy entry on #0 missing draft_id -> still isolated via recovery_dir fallback.
        crashed[0].draft_id = None;
        for entry in &crashed {
            archive_pending_recovery_to_history(
                &crash_hist_root,
                entry,
                HistoricalBackupReason::UnrecoveredCrashSnapshot,
                policy,
            )
            .unwrap();
        }
        assert_eq!(
            list_historical_backups_for_path(&crash_hist_root, None).len(),
            25
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn test_old_crash_snapshot_archived_with_current_time_survives_next_maintenance() {
        use crate::buffer::Selection;
        use crate::state::{DocumentSession, RECOVERY_META_FILENAME, scan_recovery_root};
        use crate::undo::EditKind;

        let root = unique_temp_dir("archive-fresh-ttl");
        let rec_root = root.join("recovery");
        let hist_root = rec_root.join(HISTORICAL_BACKUPS_DIR_NAME);
        let tab_rec = rec_root.join("crashed-old-tab");
        let policy = HistoricalBackupPolicy::default();

        {
            let mut session = DocumentSession::new_untitled_with_recovery(
                Some(tab_rec.clone()),
                Some(hist_root.clone()),
                policy,
                std::time::Duration::from_millis(50),
            );
            session.edit(
                crate::buffer::ByteRange::empty(0),
                "old crash draft never opened for 20 days",
                Selection::caret(0),
                Selection::caret(10),
                EditKind::Typing,
            );
            session.flush_current_recovery_snapshot().unwrap();
        }

        // Backdate `updated_unix_secs` in recovery-meta.json to 20 days ago (> 14-day TTL).
        let now = current_unix_secs();
        let old_ts = now.saturating_sub(20 * 24 * 60 * 60);
        let meta_path = tab_rec.join(RECOVERY_META_FILENAME);
        let meta_json = fs::read_to_string(&meta_path).unwrap();
        let start = meta_json.find("\"updated_unix_secs\":").unwrap();
        let rest = &meta_json[start..];
        let colon = rest.find(':').unwrap();
        let end = rest.find([',', '}']).unwrap();
        let updated_meta = format!("{}:{}{}", &meta_json[..start + colon], old_ts, &rest[end..]);
        fs::write(&meta_path, updated_meta).unwrap();

        let pending = scan_recovery_root(&rec_root);
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].updated_unix_secs, Some(old_ts));

        let archived = archive_pending_recovery_to_history(
            &hist_root,
            &pending[0],
            HistoricalBackupReason::UnrecoveredCrashSnapshot,
            policy,
        )
        .unwrap();
        assert!(archived.exists());

        // Subsequent maintenance must NOT evict the newly archived backup.
        let maint = maintain_recovery_root(&rec_root, &[], policy, 0, current_unix_secs() + 10);
        assert_eq!(maint.history_prune.removed_expired, 0);
        let kept = list_historical_backups(&hist_root);
        assert_eq!(kept.len(), 1);
        assert!(kept[0].created_unix_secs >= now.saturating_sub(5));
        assert_eq!(
            read_historical_backup(&archived).unwrap().payload,
            b"old crash draft never opened for 20 days"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn test_corrupt_recovery_slots_and_orphan_meta_counted_and_cleaned() {
        use crate::buffer::Selection;
        use crate::state::{DocumentSession, RECOVERY_META_FILENAME, scan_recovery_root};
        use crate::undo::EditKind;

        let root = unique_temp_dir("corrupt-slots");
        let rec_root = root.join("recovery");
        let policy = HistoricalBackupPolicy::default();

        // 1. Both slots corrupt + orphan recovery-meta.json -> counted in bytes and cleaned after grace.
        let corrupt_only = rec_root.join("tab-both-corrupt");
        fs::create_dir_all(&corrupt_only).unwrap();
        let junk0 = vec![b'C'; 4096];
        let junk1 = vec![b'D'; 2048];
        fs::write(corrupt_only.join("current.slot0.rec"), &junk0).unwrap();
        fs::write(corrupt_only.join("current.slot1.rec"), &junk1).unwrap();
        fs::write(
            corrupt_only.join(RECOVERY_META_FILENAME),
            b"{\"version\":1,\"source_path\":null,\"draft_id\":\"x\",\"updated_unix_secs\":1}",
        )
        .unwrap();

        assert!(scan_recovery_root(&rec_root).is_empty());
        let measured = measure_protected_snapshots_bytes(&rec_root);
        assert!(
            measured >= (junk0.len() + junk1.len()) as u64,
            "corrupt slot files must be counted in protected snapshot bytes (got {measured})"
        );

        let cleaned = maintain_recovery_root(&rec_root, &[], policy, 0, current_unix_secs());
        assert_eq!(cleaned.removed_corrupt_slots, 2);
        assert_eq!(cleaned.removed_orphan_dirs, 1);
        assert!(!corrupt_only.exists());
        assert_eq!(measure_protected_snapshots_bytes(&rec_root), 0);

        // 2. Mixed: one intact Dirty slot + one corrupt sibling slot -> keeps intact slot, cleans corrupt slot.
        let mixed_dir = rec_root.join("tab-mixed");
        {
            let mut session = DocumentSession::new_untitled_with_recovery(
                Some(mixed_dir.clone()),
                None,
                policy,
                std::time::Duration::from_millis(50),
            );
            session.edit(
                crate::buffer::ByteRange::empty(0),
                "keep my intact dirty draft",
                Selection::caret(0),
                Selection::caret(10),
                EditKind::Typing,
            );
            session.flush_current_recovery_snapshot().unwrap();
        }
        let corrupt_sibling = if mixed_dir.join("current.slot0.rec").exists() {
            mixed_dir.join("current.slot1.rec")
        } else {
            mixed_dir.join("current.slot0.rec")
        };
        fs::write(&corrupt_sibling, b"corrupt half-written sibling").unwrap();

        let mixed_maint = maintain_recovery_root(&rec_root, &[], policy, 0, current_unix_secs());
        assert_eq!(mixed_maint.removed_corrupt_slots, 1);
        assert!(!corrupt_sibling.exists());
        assert!(DocumentSession::read_latest_intact_snapshot(&mixed_dir).is_some());

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn test_oversized_corrupt_header_without_newline_rejected_bounded() {
        use crate::state::{DocumentSession, SlotInspect};

        let root = unique_temp_dir("oversized-hdr");
        let rec_root = root.join("recovery");
        let hist_root = rec_root.join(HISTORICAL_BACKUPS_DIR_NAME);
        let doc_bucket = hist_root.join("untitled");
        fs::create_dir_all(&doc_bucket).unwrap();

        // 256 KB single line without '\n' or terminator exceeds 64 KB limit.
        let oversized_hbak = doc_bucket.join("h-oversized.hbak");
        let big_junk = vec![b'A'; 256 * 1024];
        fs::write(&oversized_hbak, &big_junk).unwrap();

        assert!(matches!(
            inspect_historical_entry_header(&oversized_hbak),
            HistoryHeaderInspect::Corrupt
        ));
        assert!(list_historical_backups(&hist_root).is_empty());
        assert!(matches!(
            read_historical_backup(&oversized_hbak),
            Err(SessionError::CorruptedSnapshot)
        ));

        let slot_dir = rec_root.join("oversized-slot-dir");
        fs::create_dir_all(&slot_dir).unwrap();
        let oversized_slot = slot_dir.join("current.slot0.rec");
        fs::write(&oversized_slot, &big_junk).unwrap();
        assert_eq!(
            inspect_slot_report_header(&oversized_slot),
            SlotInspect::Corrupt
        );
        assert!(DocumentSession::read_latest_intact_snapshot(&slot_dir).is_none());

        let maint = maintain_recovery_root(
            &rec_root,
            &[],
            HistoricalBackupPolicy::default(),
            0,
            current_unix_secs(),
        );
        assert_eq!(maint.history_prune.removed_corrupt_or_tmp, 1);
        assert_eq!(maint.removed_corrupt_slots, 1);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn test_large_32mb_snapshot_maintenance_uses_header_and_streaming_fast_path() {
        use crate::buffer::Selection;
        use crate::state::DocumentSession;
        use crate::undo::EditKind;

        let root = unique_temp_dir("fast-32mb");
        let rec_root = root.join("recovery");
        let active_dir = rec_root.join("active-32mb");
        let policy = HistoricalBackupPolicy::default();

        let big_text = "M".repeat(32 * 1024 * 1024);
        {
            let mut session = DocumentSession::new_untitled_with_recovery(
                Some(active_dir.clone()),
                None,
                policy,
                std::time::Duration::from_millis(50),
            );
            session.edit(
                crate::buffer::ByteRange::empty(0),
                &big_text,
                Selection::caret(0),
                Selection::caret(10),
                EditKind::Typing,
            );
            session.flush_current_recovery_snapshot().unwrap();
        }
        drop(big_text);

        // Also create an inactive multi-chunk (512 KB) stale Clean snapshot (content == base, state == Clean)
        // to exercise bounded header + streaming CRC32 + streaming content==base cleanup path.
        let clean_inactive_dir = rec_root.join("stale-clean-512kb");
        fs::create_dir_all(&clean_inactive_dir).unwrap();
        let payload = "Z".repeat(512 * 1024);
        let rev = 1u64;
        let seq = 1u64;
        let mut h = crc32fast::Hasher::new();
        h.update(&rev.to_le_bytes());
        h.update(b"seq=");
        h.update(&seq.to_le_bytes());
        h.update(b"\0");
        h.update(payload.as_bytes());
        h.update(b"\0");
        h.update(payload.as_bytes());
        let crc = h.finalize();
        let slot_header = format!(
            "VELLMARK_REC_V1\nslot=0\nseq={seq}\nrev={rev}\nstate={}\nhas_theirs=0\ncontent_len={}\nbase_len={}\ntheirs_len=0\ncrc32={crc}\n---\n",
            doc_state_to_tag(DocState::Clean),
            payload.len(),
            payload.len()
        );
        let mut raw = Vec::with_capacity(slot_header.len() + payload.len() * 2);
        raw.extend_from_slice(slot_header.as_bytes());
        raw.extend_from_slice(payload.as_bytes());
        raw.extend_from_slice(payload.as_bytes());
        fs::write(clean_inactive_dir.join("current.slot0.rec"), &raw).unwrap();
        assert!(DocumentSession::read_latest_intact_snapshot(&clean_inactive_dir).is_some());

        let started = std::time::Instant::now();
        let report = maintain_recovery_root(
            &rec_root,
            std::slice::from_ref(&active_dir),
            policy,
            0,
            current_unix_secs(),
        );
        let elapsed = started.elapsed();
        eprintln!(
            "32 MiB active + 512 KiB clean maintenance elapsed: {:?}",
            elapsed
        );

        assert!(active_dir.exists());
        assert!(!clean_inactive_dir.exists());
        assert_eq!(report.removed_clean_pending, 1);
        assert!(report.protected_snapshot_bytes >= 32 * 1024 * 1024);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn test_legacy_v1_snapshot_without_seq_recovers_and_survives_maintenance() {
        use crate::state::{
            DocumentSession, RECOVERY_META_FILENAME, SlotInspect, SlotVerify, scan_recovery_root,
        };

        fn write_legacy_v1_slot(
            path: &Path,
            slot: u8,
            revision: u64,
            state: DocState,
            content: &str,
            base: &str,
            theirs: Option<&str>,
        ) {
            // Original V1 format (before `seq=` was introduced): CRC omits `seq=`.
            let mut h = crc32fast::Hasher::new();
            h.update(&revision.to_le_bytes());
            h.update(content.as_bytes());
            h.update(b"\0");
            h.update(base.as_bytes());
            if let Some(t) = theirs {
                h.update(b"\0");
                h.update(t.as_bytes());
            }
            let crc = h.finalize();
            let theirs_len = theirs.map(|t| t.len()).unwrap_or(0);
            let has_theirs = u8::from(theirs.is_some());
            let mut raw = format!(
                "VELLMARK_REC_V1\nslot={slot}\nrev={revision}\nstate={}\ncrc32={crc}\ncontent_len={}\nbase_len={}\nhas_theirs={has_theirs}\ntheirs_len={theirs_len}\n---\n",
                doc_state_to_tag(state),
                content.len(),
                base.len(),
            );
            raw.push_str(content);
            raw.push_str(base);
            if let Some(t) = theirs {
                raw.push_str(t);
            }
            fs::write(path, raw.as_bytes()).unwrap();
        }

        let root = unique_temp_dir("legacy-v1-no-seq");
        let rec_root = root.join("recovery");

        let legacy_dirty_dir = rec_root.join("legacy-dirty");
        fs::create_dir_all(&legacy_dirty_dir).unwrap();
        let dirty_slot = legacy_dirty_dir.join("current.slot0.rec");
        write_legacy_v1_slot(
            &dirty_slot,
            0,
            7,
            DocState::Dirty,
            "legacy dirty content",
            "legacy base content",
            None,
        );
        fs::write(
            legacy_dirty_dir.join(RECOVERY_META_FILENAME),
            b"{\"version\":1,\"source_path\":null,\"updated_unix_secs\":1700000000}",
        )
        .unwrap();

        let legacy_conflict_dir = rec_root.join("legacy-conflict");
        fs::create_dir_all(&legacy_conflict_dir).unwrap();
        let conflict_slot = legacy_conflict_dir.join("current.slot1.rec");
        write_legacy_v1_slot(
            &conflict_slot,
            1,
            9,
            DocState::Conflict,
            "ours",
            "base",
            Some("theirs"),
        );

        let legacy_clean_dir = rec_root.join("legacy-clean");
        fs::create_dir_all(&legacy_clean_dir).unwrap();
        write_legacy_v1_slot(
            &legacy_clean_dir.join("current.slot0.rec"),
            0,
            2,
            DocState::Clean,
            "same",
            "same",
            None,
        );

        // 1. Header inspect & streaming CRC accept legacy V1 without `seq=`.
        let summary = match inspect_slot_report_header(&dirty_slot) {
            SlotInspect::HeaderValid(s) => s,
            other => panic!("expected HeaderValid for legacy slot, got {other:?}"),
        };
        assert_eq!(summary.sequence, None);
        assert_eq!(
            verify_slot_report_crc_streaming(&dirty_slot, &summary),
            SlotVerify::Valid
        );

        // 2. Startup scan & recovery work identically on legacy V1 slots.
        let restored_dirty =
            DocumentSession::read_latest_intact_snapshot(&legacy_dirty_dir).unwrap();
        assert_eq!(restored_dirty.sequence, 0);
        assert_eq!(restored_dirty.revision, 7);
        assert_eq!(restored_dirty.content, "legacy dirty content");

        let restored_conflict =
            DocumentSession::read_latest_intact_snapshot(&legacy_conflict_dir).unwrap();
        assert_eq!(restored_conflict.theirs_content.as_deref(), Some("theirs"));

        // 3. Maintenance preserves valid legacy Dirty/Conflict snapshots and cleans only Clean dir.
        let maint = maintain_recovery_root(
            &rec_root,
            &[],
            HistoricalBackupPolicy::default(),
            0,
            current_unix_secs(),
        );
        assert_eq!(maint.removed_corrupt_slots, 0);
        assert_eq!(maint.removed_clean_pending, 1);
        assert!(!legacy_clean_dir.exists());
        assert!(dirty_slot.exists());
        assert!(conflict_slot.exists());
        assert_eq!(scan_recovery_root(&rec_root).len(), 2);

        let _ = fs::remove_dir_all(&root);
    }

    #[cfg(unix)]
    #[test]
    fn test_unreadable_snapshot_and_history_are_preserved_during_maintenance() {
        use std::os::unix::fs::PermissionsExt;

        use crate::buffer::Selection;
        use crate::state::{DocumentSession, RECOVERY_META_FILENAME, SlotInspect};
        use crate::undo::EditKind;

        let root = unique_temp_dir("unreadable-preserve");
        let rec_root = root.join("recovery");
        let hist_root = rec_root.join(HISTORICAL_BACKUPS_DIR_NAME);
        let policy = HistoricalBackupPolicy::default();

        // 1. Valid dirty snapshot whose file loses read permission (chmod 000).
        let dirty_dir = rec_root.join("unreadable-dirty");
        {
            let mut session = DocumentSession::new_untitled_with_recovery(
                Some(dirty_dir.clone()),
                Some(hist_root.clone()),
                policy,
                std::time::Duration::from_millis(50),
            );
            session.edit(
                crate::buffer::ByteRange::empty(0),
                "precious temporarily unreadable draft",
                Selection::caret(0),
                Selection::caret(10),
                EditKind::Typing,
            );
            session.flush_current_recovery_snapshot().unwrap();
        }
        let dirty_slot = if dirty_dir.join("current.slot0.rec").exists() {
            dirty_dir.join("current.slot0.rec")
        } else {
            dirty_dir.join("current.slot1.rec")
        };

        // 2. Two-slot directory: slot0 is intact Clean, slot1 is newer Dirty made unreadable.
        //    Maintenance MUST NOT delete the directory based on slot0 alone.
        let mixed_dir = rec_root.join("mixed-clean-and-unreadable");
        let doc_path = root.join("mixed.md");
        fs::write(&doc_path, "base text").unwrap();
        {
            let mut session = DocumentSession::open_file_with_recovery(
                &doc_path,
                Some(mixed_dir.clone()),
                Some(hist_root.clone()),
                policy,
                std::time::Duration::from_millis(50),
            )
            .unwrap();
            session.flush_current_recovery_snapshot().unwrap(); // slot0 or slot1 Clean? note Clean skips write
            session.edit(
                crate::buffer::ByteRange::empty(0),
                "edit 1",
                Selection::caret(0),
                Selection::caret(6),
                EditKind::Typing,
            );
            session.flush_current_recovery_snapshot().unwrap();
        }
        // Explicitly place an intact Clean slot0 alongside an unreadable Dirty slot1.
        fs::write(
            mixed_dir.join("current.slot0.rec"),
            {
                let mut h = crc32fast::Hasher::new();
                h.update(&0u64.to_le_bytes());
                h.update(b"seq=");
                h.update(&1u64.to_le_bytes());
                h.update(b"\0");
                h.update(b"same");
                h.update(b"\0");
                h.update(b"same");
                let crc = h.finalize();
                format!(
                    "VELLMARK_REC_V1\nslot=0\nseq=1\nrev=0\nstate={}\ncrc32={crc}\ncontent_len=4\nbase_len=4\nhas_theirs=0\ntheirs_len=0\n---\nsamesame",
                    doc_state_to_tag(DocState::Clean)
                )
            },
        )
        .unwrap();
        let mixed_dirty_slot = mixed_dir.join("current.slot1.rec");
        if !mixed_dirty_slot.exists() {
            let existing = mixed_dir.join("current.slot0.rec");
            assert!(existing.exists());
        }
        let mut h2 = crc32fast::Hasher::new();
        h2.update(&5u64.to_le_bytes());
        h2.update(b"seq=");
        h2.update(&2u64.to_le_bytes());
        h2.update(b"\0");
        h2.update(b"unsaved-newer");
        h2.update(b"\0");
        h2.update(b"base");
        let crc2 = h2.finalize();
        fs::write(
            &mixed_dirty_slot,
            format!(
                "VELLMARK_REC_V1\nslot=1\nseq=2\nrev=5\nstate={}\ncrc32={crc2}\ncontent_len=13\nbase_len=4\nhas_theirs=0\ntheirs_len=0\n---\nunsaved-newerbase",
                doc_state_to_tag(DocState::Dirty)
            ),
        )
        .unwrap();

        // 3. Valid historical backup file made unreadable.
        let hbak_path = write_historical_backup_entry(WriteHistoricalBackupRequest {
            history_root: &hist_root,
            source_path: Some(&doc_path),
            draft_id: None,
            doc_state: DocState::Conflict,
            reason: HistoricalBackupReason::ConflictOverwrittenTheirs,
            base_content: Some("base"),
            payload: b"precious unreadable history",
            policy,
            now_unix_secs: current_unix_secs(),
        })
        .unwrap();

        // 4. Genuinely corrupted slot in another folder MUST still be removed.
        let confirmed_corrupt_dir = rec_root.join("confirmed-corrupt");
        fs::create_dir_all(&confirmed_corrupt_dir).unwrap();
        fs::write(
            confirmed_corrupt_dir.join("current.slot0.rec"),
            b"definitely corrupt data",
        )
        .unwrap();

        fs::set_permissions(&dirty_slot, fs::Permissions::from_mode(0o000)).unwrap();
        fs::set_permissions(&mixed_dirty_slot, fs::Permissions::from_mode(0o000)).unwrap();
        fs::set_permissions(&hbak_path, fs::Permissions::from_mode(0o000)).unwrap();

        // Skip assertion if running in a privileged environment (e.g. uid 0) that ignores 0o000.
        if !matches!(
            File::open(&dirty_slot),
            Err(ref e) if e.kind() == std::io::ErrorKind::PermissionDenied
        ) {
            let _ = fs::set_permissions(&dirty_slot, fs::Permissions::from_mode(0o644));
            let _ = fs::set_permissions(&mixed_dirty_slot, fs::Permissions::from_mode(0o644));
            let _ = fs::set_permissions(&hbak_path, fs::Permissions::from_mode(0o644));
            let _ = fs::remove_dir_all(&root);
            return;
        }

        assert_eq!(
            inspect_slot_report_header(&dirty_slot),
            SlotInspect::Unreadable
        );
        assert!(matches!(
            read_historical_backup(&hbak_path),
            Err(SessionError::Io(_))
        ));

        let maint = maintain_recovery_root(&rec_root, &[], policy, 0, current_unix_secs());
        assert_eq!(maint.removed_corrupt_slots, 1);
        assert_eq!(maint.removed_clean_pending, 0);
        assert_eq!(maint.history_prune.removed_corrupt_or_tmp, 0);
        assert!(!confirmed_corrupt_dir.exists());
        assert!(
            dirty_slot.exists(),
            "unreadable recovery slot must NEVER be deleted during maintenance"
        );
        assert!(dirty_dir.join(RECOVERY_META_FILENAME).exists());
        assert!(
            mixed_dirty_slot.exists(),
            "unreadable newer Dirty slot must NOT be deleted because older sibling slot is Clean"
        );
        assert!(
            hbak_path.exists(),
            "unreadable historical backup must NEVER be deleted as corrupt"
        );

        // Restore permissions -> all unreadable files remain intact and recoverable.
        fs::set_permissions(&dirty_slot, fs::Permissions::from_mode(0o644)).unwrap();
        fs::set_permissions(&mixed_dirty_slot, fs::Permissions::from_mode(0o644)).unwrap();
        fs::set_permissions(&hbak_path, fs::Permissions::from_mode(0o644)).unwrap();

        let rec1 = DocumentSession::read_latest_intact_snapshot(&dirty_dir).unwrap();
        assert_eq!(rec1.content, "precious temporarily unreadable draft");

        let rec2 = DocumentSession::read_latest_intact_snapshot(&mixed_dir).unwrap();
        assert_eq!(rec2.slot, 1);
        assert_eq!(rec2.content, "unsaved-newer");

        let hbak_data = read_historical_backup(&hbak_path).unwrap();
        assert_eq!(hbak_data.payload, b"precious unreadable history");

        let _ = fs::remove_dir_all(&root);
    }
}
