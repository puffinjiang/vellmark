use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use ropey::Rope;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::buffer::{ByteRange, Selection, TextBuffer};
use crate::encoding::{FileFormatMeta, detect_and_decode, encode_for_save};
use crate::undo::{EditKind, UndoHistory};

/// Document state machine per PRD 4.5.1 B.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DocState {
    Clean,
    Dirty,
    Conflict,
    Missing { has_local_edits: bool },
}

/// Immutable snapshot of a disk version (`base` or `theirs`, PRD 4.5.1 A).
#[derive(Debug, Clone)]
pub struct DiskSnapshot {
    pub rope: Rope,
    pub crc32: u32,
    pub len_bytes: usize,
}

impl DiskSnapshot {
    pub fn from_rope(rope: Rope) -> Self {
        let mut hasher = crc32fast::Hasher::new();
        for chunk in rope.chunks() {
            hasher.update(chunk.as_bytes());
        }
        let len_bytes = rope.len_bytes();
        Self {
            rope,
            crc32: hasher.finalize(),
            len_bytes,
        }
    }

    pub fn from_str(text: &str) -> Self {
        Self::from_rope(Rope::from_str(text))
    }

    pub fn content_string(&self) -> String {
        self.rope.to_string()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SaveOutcome {
    /// Saved and no new edits occurred during write -> Clean.
    SavedClean,
    /// Saved snapshot `written_revision`, but user edited during write -> base updated, stays Dirty.
    SavedButStillDirty { written_revision: u64 },
    /// Pre-write check failed because disk != base -> transitioned to Conflict.
    EnteredConflict,
    /// Write failed (I/O error) -> stays Dirty.
    WriteFailed(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConflictActionOutcome {
    ResolvedClean,
    ResolvedDirty,
    AbortedStillConflict(String),
    TransitionedToMissing,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RecoverySlotReport {
    pub slot: u8,
    pub revision: u64,
    pub state: DocState,
    pub content: String,
    pub base_content: String,
    pub theirs_content: Option<String>,
    pub crc32: u32,
}

#[derive(Debug, Error)]
pub enum SessionError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("encoding error: {0}")]
    Encoding(#[from] crate::encoding::EncodingError),
    #[error("corrupted recovery snapshot")]
    CorruptedSnapshot,
}

/// Complete document session managing buffer, undo stack, `base` / `theirs` snapshots,
/// state transitions, and dual-slot atomic crash recovery (PRD 4.5.1).
pub struct DocumentSession {
    pub path: Option<PathBuf>,
    pub buffer: TextBuffer,
    pub undo: UndoHistory,
    pub format: FileFormatMeta,
    pub state: DocState,
    pub base: DiskSnapshot,
    pub theirs: Option<DiskSnapshot>,
    pub recovery_dir: Option<PathBuf>,
    active_recovery_slot: u8,
    pub recovery_degraded: bool,
}

impl DocumentSession {
    /// Open an existing file from disk following PRD 4.5.2 encoding rules.
    pub fn open_file(path: impl AsRef<Path>, recovery_dir: Option<PathBuf>) -> Result<Self, SessionError> {
        let path = path.as_ref().to_path_buf();
        let raw = fs::read(&path)?;
        let (text, format) = detect_and_decode(&raw)?;
        let buffer = TextBuffer::from_str(&text);
        let base = DiskSnapshot::from_rope(buffer.snapshot_rope());
        Ok(Self {
            path: Some(path),
            buffer,
            undo: UndoHistory::new(),
            format,
            state: DocState::Clean,
            base,
            theirs: None,
            recovery_dir,
            active_recovery_slot: 0,
            recovery_degraded: false,
        })
    }

    /// Perform a local edit and transition state per PRD 4.5.1 C.
    pub fn edit(
        &mut self,
        range: ByteRange,
        replacement: &str,
        sel_before: Selection,
        sel_after: Selection,
        kind: EditKind,
    ) {
        self.undo
            .apply_edit(&mut self.buffer, range, replacement, sel_before, sel_after, kind);
        self.state = match self.state {
            DocState::Clean | DocState::Dirty => DocState::Dirty,
            DocState::Conflict => DocState::Conflict,
            DocState::Missing { .. } => DocState::Missing {
                has_local_edits: true,
            },
        };
        let _ = self.write_current_recovery_snapshot();
    }

    /// Handle external file modification event per PRD 4.5.1 C.
    pub fn on_external_modified(&mut self, new_disk_text: &str) {
        let disk_snap = DiskSnapshot::from_str(new_disk_text);
        match self.state {
            DocState::Clean => {
                self.buffer.reset_content(new_disk_text);
                self.base = disk_snap;
                self.state = DocState::Clean;
            }
            DocState::Dirty => {
                if disk_snap.crc32 != self.base.crc32 || disk_snap.len_bytes != self.base.len_bytes {
                    self.theirs = Some(disk_snap);
                    self.state = DocState::Conflict;
                    let _ = self.write_current_recovery_snapshot();
                }
            }
            DocState::Conflict => {
                // Disk modified again during Conflict: refresh `theirs`, stay Conflict (PRD 4.5.1 C).
                self.theirs = Some(disk_snap);
                let _ = self.write_current_recovery_snapshot();
            }
            DocState::Missing { has_local_edits } => {
                // Original path reappeared (PRD 4.5.1 C).
                let matches_base =
                    disk_snap.crc32 == self.base.crc32 && disk_snap.len_bytes == self.base.len_bytes;
                if matches_base {
                    self.state = if has_local_edits {
                        DocState::Dirty
                    } else {
                        DocState::Clean
                    };
                } else if !has_local_edits {
                    self.buffer.reset_content(new_disk_text);
                    self.base = disk_snap;
                    self.state = DocState::Clean;
                } else {
                    self.theirs = Some(disk_snap);
                    self.state = DocState::Conflict;
                }
            }
        }
    }

    /// Handle external file deletion per PRD 4.5.1 C.
    pub fn on_external_deleted(&mut self) {
        let has_local_edits = match self.state {
            DocState::Clean => false,
            DocState::Dirty | DocState::Conflict => true,
            DocState::Missing { has_local_edits } => has_local_edits,
        };
        self.theirs = None;
        self.state = DocState::Missing { has_local_edits };
        let _ = self.write_current_recovery_snapshot();
    }

    /// Execute a version-bound save: snapshots `written_revision` and `written_rope`,
    /// verifies disk == `base` before atomic write, and re-evaluates state upon completion
    /// against `self.buffer.revision()` so concurrent edits during write are never marked Clean (PRD 4.5.1 A & C).
    pub fn save_with_captured_snapshot(
        &mut self,
        written_revision: u64,
        written_rope: Rope,
    ) -> SaveOutcome {
        let Some(path) = self.path.clone() else {
            return SaveOutcome::WriteFailed("no file path bound".to_string());
        };

        // Pre-write verification against `base` (PRD 4.5.1 C)
        if let Ok(disk_bytes) = fs::read(&path) {
            if let Ok((disk_text, _)) = detect_and_decode(&disk_bytes) {
                let disk_snap = DiskSnapshot::from_str(&disk_text);
                if disk_snap.crc32 != self.base.crc32 || disk_snap.len_bytes != self.base.len_bytes {
                    self.theirs = Some(disk_snap);
                    self.state = DocState::Conflict;
                    let _ = self.write_current_recovery_snapshot();
                    return SaveOutcome::EnteredConflict;
                }
            }
        } else {
            self.on_external_deleted();
            return SaveOutcome::WriteFailed("target file disappeared before write".to_string());
        }

        let written_text = written_rope.to_string();
        let bytes = match encode_for_save(&written_text, &self.format) {
            Ok(b) => b,
            Err(e) => return SaveOutcome::WriteFailed(e.to_string()),
        };

        if let Err(e) = atomic_write_file(&path, &bytes) {
            return SaveOutcome::WriteFailed(e.to_string());
        }

        // Update `base` to the snapshot that was actually written (PRD 4.5.1 A)
        self.base = DiskSnapshot::from_rope(written_rope);

        // Compare current buffer revision with the written snapshot revision
        if self.buffer.revision() == written_revision
            && self.buffer.crc32() == self.base.crc32
        {
            self.state = DocState::Clean;
            self.clear_current_recovery_snapshot();
            SaveOutcome::SavedClean
        } else {
            self.state = DocState::Dirty;
            let _ = self.write_current_recovery_snapshot();
            SaveOutcome::SavedButStillDirty { written_revision }
        }
    }

    /// Convenience synchronous save of current buffer state.
    pub fn save(&mut self) -> SaveOutcome {
        let rev = self.buffer.revision();
        let snap = self.buffer.snapshot_rope();
        self.save_with_captured_snapshot(rev, snap)
    }

    /// Conflict resolution: "Keep My Version" (PRD 4.5.1 D).
    /// Precondition: persists `theirs` to recovery backup FIRST; if backup fails, aborts and stays Conflict.
    pub fn resolve_conflict_keep_mine(
        &mut self,
        captured_revision: u64,
        captured_rope: Rope,
    ) -> ConflictActionOutcome {
        let Some(path) = self.path.clone() else {
            return ConflictActionOutcome::AbortedStillConflict("no path".to_string());
        };
        let Some(expected_theirs) = self.theirs.clone() else {
            return ConflictActionOutcome::AbortedStillConflict("not in conflict".to_string());
        };

        // 1. Verify disk still matches the `theirs` the user saw
        let disk_bytes = match fs::read(&path) {
            Ok(b) => b,
            Err(_) => {
                self.on_external_deleted();
                return ConflictActionOutcome::TransitionedToMissing;
            }
        };
        if let Ok((disk_text, _)) = detect_and_decode(&disk_bytes) {
            let latest_disk = DiskSnapshot::from_str(&disk_text);
            if latest_disk.crc32 != expected_theirs.crc32
                || latest_disk.len_bytes != expected_theirs.len_bytes
            {
                self.theirs = Some(latest_disk);
                return ConflictActionOutcome::AbortedStillConflict(
                    "disk changed again; refreshed theirs".to_string(),
                );
            }
        }

        // 2. Precondition: save `theirs` to historical recovery backup BEFORE overwriting (PRD 4.5.1 D)
        if let Err(e) = self.write_historical_backup("conflict-overwritten-theirs", &expected_theirs.content_string()) {
            self.recovery_degraded = true;
            return ConflictActionOutcome::AbortedStillConflict(format!(
                "recovery backup failed ({e}); refusing to overwrite disk version"
            ));
        }

        // 3. Write captured snapshot to original file
        let written_text = captured_rope.to_string();
        let bytes = match encode_for_save(&written_text, &self.format) {
            Ok(b) => b,
            Err(e) => return ConflictActionOutcome::AbortedStillConflict(e.to_string()),
        };
        if let Err(e) = atomic_write_file(&path, &bytes) {
            return ConflictActionOutcome::AbortedStillConflict(e.to_string());
        }

        self.base = DiskSnapshot::from_rope(captured_rope);
        self.theirs = None;

        // Re-evaluate Clean vs Dirty per PRD 4.5.1 A
        if self.buffer.revision() == captured_revision && self.buffer.crc32() == self.base.crc32 {
            self.state = DocState::Clean;
            self.clear_current_recovery_snapshot();
            ConflictActionOutcome::ResolvedClean
        } else {
            self.state = DocState::Dirty;
            let _ = self.write_current_recovery_snapshot();
            ConflictActionOutcome::ResolvedDirty
        }
    }

    /// Conflict resolution: "Use Disk Version" (PRD 4.5.1 D).
    /// Precondition: persists my version at `captured_revision` to recovery backup FIRST;
    /// if new edits occurred before replacement completes, aborts reload so new input is never lost.
    pub fn resolve_conflict_use_theirs(&mut self, captured_revision: u64) -> ConflictActionOutcome {
        let Some(path) = self.path.clone() else {
            return ConflictActionOutcome::AbortedStillConflict("no path".to_string());
        };
        let Some(expected_theirs) = self.theirs.clone() else {
            return ConflictActionOutcome::AbortedStillConflict("not in conflict".to_string());
        };

        // Check if file still exists on disk
        if !path.exists() {
            self.on_external_deleted();
            return ConflictActionOutcome::TransitionedToMissing;
        }

        // 1. Precondition: save my version to historical backup first
        let my_content = self.buffer.to_string();
        if let Err(e) = self.write_historical_backup("conflict-discarded-mine", &my_content) {
            self.recovery_degraded = true;
            return ConflictActionOutcome::AbortedStillConflict(format!(
                "recovery backup failed ({e}); refusing to discard local edits"
            ));
        }

        // 2. Verify no new edits occurred since user triggered the action (PRD 4.5.1 D)
        if self.buffer.revision() != captured_revision {
            return ConflictActionOutcome::AbortedStillConflict(
                "new edits occurred while saving backup; aborted reload".to_string(),
            );
        }

        let theirs_text = expected_theirs.content_string();
        let len = self.buffer.len_bytes();
        self.undo.apply_edit(
            &mut self.buffer,
            ByteRange::new(0, len),
            &theirs_text,
            Selection::caret(0),
            Selection::caret(0),
            EditKind::Structural,
        );
        self.base = expected_theirs;
        self.theirs = None;
        self.state = DocState::Clean;
        self.clear_current_recovery_snapshot();
        ConflictActionOutcome::ResolvedClean
    }

    /// Dual-slot atomic current recovery snapshot writer (PRD 4.5.1 F).
    /// Alternates between `current.slot0.rec` and `current.slot1.rec` via temp-file + fsync + atomic rename,
    /// so an interrupted write never corrupts the previous intact snapshot.
    pub fn write_current_recovery_snapshot(&mut self) -> Result<(), SessionError> {
        let Some(dir) = self.recovery_dir.clone() else {
            return Ok(());
        };
        if let Err(e) = fs::create_dir_all(&dir) {
            self.recovery_degraded = true;
            return Err(SessionError::Io(e));
        }

        let next_slot = 1 - self.active_recovery_slot;
        let content = self.buffer.to_string();
        let base_content = self.base.content_string();
        let theirs_content = self.theirs.as_ref().map(|t| t.content_string());

        let crc32 = compute_snapshot_crc(
            self.buffer.revision(),
            &content,
            &base_content,
            theirs_content.as_deref(),
        );
        let report = RecoverySlotReport {
            slot: next_slot,
            revision: self.buffer.revision(),
            state: self.state,
            content,
            base_content,
            theirs_content,
            crc32,
        };
        let encoded = serialize_slot_report(&report);
        let target = dir.join(format!("current.slot{next_slot}.rec"));
        match atomic_write_file(&target, encoded.as_bytes()) {
            Ok(()) => {
                self.active_recovery_slot = next_slot;
                self.recovery_degraded = false;
                Ok(())
            }
            Err(e) => {
                self.recovery_degraded = true;
                Err(SessionError::Io(e))
            }
        }
    }

    /// Read the latest intact current recovery snapshot across slot 0 and slot 1,
    /// falling back to the older slot if the newer slot was corrupted mid-write (PRD 4.5.1 F).
    pub fn read_latest_intact_snapshot(dir: impl AsRef<Path>) -> Option<RecoverySlotReport> {
        let dir = dir.as_ref();
        let mut valid_slots = Vec::new();
        for slot in [0u8, 1u8] {
            let p = dir.join(format!("current.slot{slot}.rec"));
            if let Ok(raw) = fs::read_to_string(&p) {
                if let Some(rep) = deserialize_and_verify_slot_report(&raw) {
                    valid_slots.push(rep);
                }
            }
        }
        valid_slots.into_iter().max_by_key(|r| r.revision)
    }

    fn write_historical_backup(&self, reason: &str, content: &str) -> std::io::Result<()> {
        let Some(dir) = &self.recovery_dir else {
            return Ok(());
        };
        fs::create_dir_all(dir)?;
        let rev = self.buffer.revision();
        let backup_path = dir.join(format!("history-{reason}-rev{rev}.bak"));
        atomic_write_file(&backup_path, content.as_bytes())
    }

    fn clear_current_recovery_snapshot(&self) {
        if let Some(dir) = &self.recovery_dir {
            for slot in [0u8, 1u8] {
                let _ = fs::remove_file(dir.join(format!("current.slot{slot}.rec")));
            }
        }
    }
}

/// Write to temporary sibling file, `fsync`, and atomically rename (PRD 4.5.1 G).
pub fn atomic_write_file(target: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let parent = target.parent().unwrap_or_else(|| Path::new("."));
    let file_name = target
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("vellmark");
    let tmp_path = parent.join(format!(".{file_name}.vellmark.tmp"));
    {
        let mut f = File::create(&tmp_path)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    fs::rename(&tmp_path, target)
}

fn compute_snapshot_crc(
    revision: u64,
    content: &str,
    base: &str,
    theirs: Option<&str>,
) -> u32 {
    let mut h = crc32fast::Hasher::new();
    h.update(&revision.to_le_bytes());
    h.update(content.as_bytes());
    h.update(b"\0");
    h.update(base.as_bytes());
    if let Some(t) = theirs {
        h.update(b"\0");
        h.update(t.as_bytes());
    }
    h.finalize()
}

fn serialize_slot_report(r: &RecoverySlotReport) -> String {
    let state_tag = match r.state {
        DocState::Clean => "Clean",
        DocState::Dirty => "Dirty",
        DocState::Conflict => "Conflict",
        DocState::Missing {
            has_local_edits: true,
        } => "MissingDirty",
        DocState::Missing {
            has_local_edits: false,
        } => "MissingClean",
    };
    let theirs_len = r.theirs_content.as_ref().map(|s| s.len()).unwrap_or(0);
    let has_theirs = u8::from(r.theirs_content.is_some());
    let mut out = format!(
        "VELLMARK_REC_V1\nslot={}\nrev={}\nstate={}\ncrc32={}\ncontent_len={}\nbase_len={}\nhas_theirs={}\ntheirs_len={}\n---\n",
        r.slot,
        r.revision,
        state_tag,
        r.crc32,
        r.content.len(),
        r.base_content.len(),
        has_theirs,
        theirs_len
    );
    out.push_str(&r.content);
    out.push_str(&r.base_content);
    if let Some(t) = &r.theirs_content {
        out.push_str(t);
    }
    out
}

fn deserialize_and_verify_slot_report(raw: &str) -> Option<RecoverySlotReport> {
    let (header, body) = raw.split_once("\n---\n")?;
    let mut lines = header.lines();
    if lines.next()? != "VELLMARK_REC_V1" {
        return None;
    }
    let mut slot = 0u8;
    let mut revision = 0u64;
    let mut state = DocState::Dirty;
    let mut crc32 = 0u32;
    let mut content_len = 0usize;
    let mut base_len = 0usize;
    let mut has_theirs = false;
    let mut theirs_len = 0usize;

    for line in lines {
        let (k, v) = line.split_once('=')?;
        match k {
            "slot" => slot = v.parse().ok()?,
            "rev" => revision = v.parse().ok()?,
            "state" => {
                state = match v {
                    "Clean" => DocState::Clean,
                    "Dirty" => DocState::Dirty,
                    "Conflict" => DocState::Conflict,
                    "MissingDirty" => DocState::Missing {
                        has_local_edits: true,
                    },
                    "MissingClean" => DocState::Missing {
                        has_local_edits: false,
                    },
                    _ => return None,
                };
            }
            "crc32" => crc32 = v.parse().ok()?,
            "content_len" => content_len = v.parse().ok()?,
            "base_len" => base_len = v.parse().ok()?,
            "has_theirs" => has_theirs = v == "1",
            "theirs_len" => theirs_len = v.parse().ok()?,
            _ => {}
        }
    }

    if body.len() != content_len + base_len + theirs_len {
        return None;
    }
    let content = body.get(0..content_len)?.to_string();
    let base_content = body.get(content_len..content_len + base_len)?.to_string();
    let theirs_content = if has_theirs {
        Some(
            body.get(content_len + base_len..content_len + base_len + theirs_len)?
                .to_string(),
        )
    } else {
        None
    };

    let expected_crc = compute_snapshot_crc(
        revision,
        &content,
        &base_content,
        theirs_content.as_deref(),
    );
    if expected_crc != crc32 {
        return None;
    }

    Some(RecoverySlotReport {
        slot,
        revision,
        state,
        content,
        base_content,
        theirs_content,
        crc32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_dir(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("vellmark-test-{tag}-{nanos}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_version_bound_save_during_concurrent_edit_stays_dirty() {
        // PRD 4.5.1 A & C: Saving version A while user edits to version B updates `base` to A and stays Dirty.
        let dir = unique_temp_dir("save-concurrent");
        let file_path = dir.join("note.md");
        let rec_dir = dir.join("recovery");
        fs::write(&file_path, "Initial\n").unwrap();

        let mut session = DocumentSession::open_file(&file_path, Some(rec_dir.clone())).unwrap();
        assert_eq!(session.state, DocState::Clean);

        // Edit to version A
        session.edit(
            ByteRange::new(8, 8),
            "Version A\n",
            Selection::caret(8),
            Selection::caret(18),
            EditKind::Typing,
        );
        let rev_a = session.buffer.revision();
        let rope_a = session.buffer.snapshot_rope();

        // User continues typing to form Version B before save of Version A completes
        session.edit(
            ByteRange::new(18, 18),
            "Version B\n",
            Selection::caret(18),
            Selection::caret(28),
            EditKind::Typing,
        );

        let outcome = session.save_with_captured_snapshot(rev_a, rope_a);
        assert_eq!(
            outcome,
            SaveOutcome::SavedButStillDirty {
                written_revision: rev_a
            }
        );
        assert_eq!(session.state, DocState::Dirty);
        assert_eq!(session.base.content_string(), "Initial\nVersion A\n");

        // Recovery snapshot for Version B must still be preserved
        let latest = DocumentSession::read_latest_intact_snapshot(&rec_dir).unwrap();
        assert_eq!(latest.content, "Initial\nVersion A\nVersion B\n");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_conflict_detection_and_dual_slot_recovery_fallback_on_corruption() {
        // PRD 4.5.1 F: If the newest recovery slot is corrupted mid-write, fallback to the previous intact slot.
        let dir = unique_temp_dir("dual-slot");
        let file_path = dir.join("doc.md");
        let rec_dir = dir.join("recovery");
        fs::write(&file_path, "Base\n").unwrap();

        let mut session = DocumentSession::open_file(&file_path, Some(rec_dir.clone())).unwrap();
        session.edit(
            ByteRange::new(5, 5),
            "Edit 1\n",
            Selection::caret(5),
            Selection::caret(12),
            EditKind::Typing,
        );
        session.edit(
            ByteRange::new(12, 12),
            "Edit 2\n",
            Selection::caret(12),
            Selection::caret(19),
            EditKind::Typing,
        );

        // Corrupt the latest slot file to simulate power loss mid-write
        let newest_slot = session.active_recovery_slot;
        fs::write(
            rec_dir.join(format!("current.slot{newest_slot}.rec")),
            "CORRUPTED_PARTIAL_WRITE",
        )
        .unwrap();

        // Reader must automatically recover the previous intact slot ("Base\nEdit 1\n")
        let recovered = DocumentSession::read_latest_intact_snapshot(&rec_dir).unwrap();
        assert_eq!(recovered.content, "Base\nEdit 1\n");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_use_theirs_aborts_if_user_edited_during_backup() {
        // PRD 4.5.1 D: If user edits after triggering `use_theirs`, abort reload so new edits are not lost.
        let dir = unique_temp_dir("use-theirs");
        let file_path = dir.join("conflict.md");
        let rec_dir = dir.join("recovery");
        fs::write(&file_path, "Base\n").unwrap();

        let mut session = DocumentSession::open_file(&file_path, Some(rec_dir)).unwrap();
        session.edit(
            ByteRange::new(5, 5),
            "Mine\n",
            Selection::caret(5),
            Selection::caret(10),
            EditKind::Typing,
        );
        let captured_rev = session.buffer.revision();

        // External modification triggers Conflict
        fs::write(&file_path, "Base\nTheirs\n").unwrap();
        session.on_external_modified("Base\nTheirs\n");
        assert_eq!(session.state, DocState::Conflict);

        // Simulate user typing one more character before `resolve_conflict_use_theirs` executes with old revision
        session.edit(
            ByteRange::new(10, 10),
            "!",
            Selection::caret(10),
            Selection::caret(11),
            EditKind::Typing,
        );

        let res = session.resolve_conflict_use_theirs(captured_rev);
        assert!(matches!(res, ConflictActionOutcome::AbortedStillConflict(_)));
        assert_eq!(session.state, DocState::Conflict);
        assert_eq!(session.buffer.to_string(), "Base\nMine\n!");

        let _ = fs::remove_dir_all(&dir);
    }
}
