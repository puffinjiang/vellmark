use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use ropey::Rope;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::buffer::{ByteRange, Selection, TextBuffer};
use crate::editor::{EditorContext, EditorViewState, map_offset_across_reload};
use crate::encoding::{
    DetectedEncoding, EncodingDecision, FileFormatMeta, LineEnding, detect_and_decode,
    encode_for_save,
};
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
/// When the external disk file is undecodable/binary, `raw_bytes` stores the exact
/// original bytes so historical backups ("Keep My Version") can restore it byte-for-byte without lossy corruption.
#[derive(Debug, Clone)]
pub struct DiskSnapshot {
    pub rope: Rope,
    pub raw_bytes: Option<Arc<[u8]>>,
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
            raw_bytes: None,
            crc32: hasher.finalize(),
            len_bytes,
        }
    }

    pub fn from_str(text: &str) -> Self {
        Self::from_rope(Rope::from_str(text))
    }

    /// Create a snapshot when the external file was modified into undecodable/binary bytes,
    /// preserving the exact raw bytes (`Arc<[u8]>`) as well as a lossy Rope for preview/diff,
    /// so overwriting it in "Keep My Version" writes the exact original bytes to the recovery backup.
    pub fn from_undecodable_bytes(raw: &[u8]) -> Self {
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(raw);
        let lossy = String::from_utf8_lossy(raw);
        Self {
            rope: Rope::from_str(&lossy),
            raw_bytes: Some(Arc::from(raw)),
            crc32: hasher.finalize(),
            len_bytes: raw.len(),
        }
    }

    pub fn content_string(&self) -> String {
        self.rope.to_string()
    }

    /// Return the exact bytes to persist when backing up this snapshot to `recovery/`.
    pub fn to_backup_bytes(&self) -> Vec<u8> {
        if let Some(raw) = &self.raw_bytes {
            raw.to_vec()
        } else {
            self.rope.to_string().into_bytes()
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SaveOutcome {
    /// Saved and no new edits occurred during write -> Clean.
    SavedClean,
    /// Saved snapshot `written_revision`, but user edited during write -> base updated, stays Dirty.
    SavedButStillDirty { written_revision: u64 },
    /// Pre-write check failed because disk != base (or became undecodable/binary) -> transitioned to Conflict.
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
    pub sequence: u64,
    pub revision: u64,
    pub state: DocState,
    pub content: String,
    pub base_content: String,
    pub theirs_content: Option<String>,
    pub crc32: u32,
}

/// Lightweight O(1) Rope-cloned payload sent from the UI/editor thread to the background recovery worker (PRD 4.5.1 F).
#[derive(Debug, Clone)]
pub struct RecoveryPayload {
    pub source_path: Option<PathBuf>,
    pub revision: u64,
    pub state: DocState,
    pub content_rope: Rope,
    pub base_rope: Rope,
    pub theirs_rope: Option<Rope>,
}

enum WorkerCommand {
    Schedule(RecoveryPayload),
    FlushNow(RecoveryPayload, Sender<Result<(), String>>),
    Clear(Sender<()>),
    Shutdown,
}

/// Background recovery snapshot worker with <= 1s throttling and dual-slot atomic `fsync` writes,
/// ensuring `DocumentSession::edit` never blocks the UI thread on serialization or disk I/O (PRD 4.5.1 F).
pub struct RecoveryWorker {
    tx: Sender<WorkerCommand>,
    degraded: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl RecoveryWorker {
    pub fn spawn(recovery_dir: PathBuf, throttle_window: Duration) -> Self {
        let (tx, rx) = mpsc::channel::<WorkerCommand>();
        let degraded = Arc::new(AtomicBool::new(false));
        let degraded_clone = Arc::clone(&degraded);

        let handle = thread::spawn(move || {
            worker_loop(recovery_dir, throttle_window, rx, degraded_clone);
        });

        Self {
            tx,
            degraded,
            handle: Some(handle),
        }
    }

    pub fn schedule(&self, payload: RecoveryPayload) {
        let _ = self.tx.send(WorkerCommand::Schedule(payload));
    }

    pub fn flush_sync(&self, payload: RecoveryPayload) -> Result<(), SessionError> {
        let (ack_tx, ack_rx) = mpsc::channel();
        let _ = self.tx.send(WorkerCommand::FlushNow(payload, ack_tx));
        match ack_rx.recv() {
            Ok(Ok(())) => Ok(()),
            Ok(Err(msg)) => Err(SessionError::Io(std::io::Error::other(msg))),
            Err(_) => Err(SessionError::Io(std::io::Error::other(
                "recovery worker disconnected",
            ))),
        }
    }

    pub fn clear_sync(&self) {
        let (ack_tx, ack_rx) = mpsc::channel();
        let _ = self.tx.send(WorkerCommand::Clear(ack_tx));
        let _ = ack_rx.recv();
    }

    pub fn is_degraded(&self) -> bool {
        self.degraded.load(Ordering::Acquire)
    }
}

impl Drop for RecoveryWorker {
    fn drop(&mut self) {
        let _ = self.tx.send(WorkerCommand::Shutdown);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn initial_worker_slot_state(recovery_dir: &Path) -> (u8, u64) {
    if let Some(latest) = DocumentSession::read_latest_intact_snapshot(recovery_dir) {
        (latest.slot, latest.sequence)
    } else {
        (0, 0)
    }
}

fn worker_loop(
    recovery_dir: PathBuf,
    throttle_window: Duration,
    rx: Receiver<WorkerCommand>,
    degraded: Arc<AtomicBool>,
) {
    let (mut active_slot, mut write_seq) = initial_worker_slot_state(&recovery_dir);
    let mut pending: Option<RecoveryPayload> = None;
    let mut deadline: Option<Instant> = None;

    loop {
        let cmd = if let Some(dl) = deadline {
            let now = Instant::now();
            if now >= dl {
                if let Some(p) = pending.take() {
                    let ok = write_payload_to_next_slot(
                        &recovery_dir,
                        &mut active_slot,
                        &mut write_seq,
                        &p,
                    )
                    .is_ok();
                    degraded.store(!ok, Ordering::Release);
                }
                deadline = None;
                continue;
            }
            match rx.recv_timeout(dl - now) {
                Ok(c) => c,
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        } else {
            match rx.recv() {
                Ok(c) => c,
                Err(_) => break,
            }
        };

        match cmd {
            WorkerCommand::Schedule(mut latest) => {
                // Coalesce any additional queued Schedule commands immediately
                loop {
                    match rx.try_recv() {
                        Ok(WorkerCommand::Schedule(newer)) => latest = newer,
                        Ok(other) => {
                            // Write or queue `latest` before handling `other`
                            pending = Some(latest);
                            if deadline.is_none() {
                                deadline = Some(Instant::now() + throttle_window);
                            }
                            if handle_non_schedule_cmd(
                                other,
                                &recovery_dir,
                                &mut active_slot,
                                &mut write_seq,
                                &mut pending,
                                &mut deadline,
                                &degraded,
                            ) {
                                return;
                            }
                            break;
                        }
                        Err(TryRecvError::Empty) => {
                            pending = Some(latest);
                            if deadline.is_none() {
                                deadline = Some(Instant::now() + throttle_window);
                            }
                            break;
                        }
                        Err(TryRecvError::Disconnected) => return,
                    }
                }
            }
            other => {
                if handle_non_schedule_cmd(
                    other,
                    &recovery_dir,
                    &mut active_slot,
                    &mut write_seq,
                    &mut pending,
                    &mut deadline,
                    &degraded,
                ) {
                    break;
                }
            }
        }
    }
}

fn handle_non_schedule_cmd(
    cmd: WorkerCommand,
    recovery_dir: &Path,
    active_slot: &mut u8,
    write_seq: &mut u64,
    pending: &mut Option<RecoveryPayload>,
    deadline: &mut Option<Instant>,
    degraded: &Arc<AtomicBool>,
) -> bool {
    match cmd {
        WorkerCommand::FlushNow(payload, ack) => {
            *pending = None;
            *deadline = None;
            let res = write_payload_to_next_slot(recovery_dir, active_slot, write_seq, &payload)
                .map_err(|e| e.to_string());
            degraded.store(res.is_err(), Ordering::Release);
            let _ = ack.send(res);
            false
        }
        WorkerCommand::Clear(ack) => {
            *pending = None;
            *deadline = None;
            remove_recovery_dir_files(recovery_dir);
            let _ = ack.send(());
            false
        }
        WorkerCommand::Shutdown => {
            if let Some(p) = pending.take() {
                let _ = write_payload_to_next_slot(recovery_dir, active_slot, write_seq, &p);
            }
            true
        }
        WorkerCommand::Schedule(_) => false,
    }
}

fn write_payload_to_next_slot(
    dir: &Path,
    active_slot: &mut u8,
    write_seq: &mut u64,
    payload: &RecoveryPayload,
) -> std::io::Result<()> {
    fs::create_dir_all(dir)?;
    let next_slot = 1 - *active_slot;
    *write_seq = write_seq.saturating_add(1);
    let sequence = *write_seq;
    let content = payload.content_rope.to_string();
    let base_content = payload.base_rope.to_string();
    let theirs_content = payload.theirs_rope.as_ref().map(|r| r.to_string());

    let crc32 = compute_snapshot_crc(
        payload.revision,
        Some(sequence),
        &content,
        &base_content,
        theirs_content.as_deref(),
    );
    let report = RecoverySlotReport {
        slot: next_slot,
        sequence,
        revision: payload.revision,
        state: payload.state,
        content,
        base_content,
        theirs_content,
        crc32,
    };
    let encoded = serialize_slot_report(&report);
    let target = dir.join(format!("current.slot{next_slot}.rec"));
    atomic_write_file(&target, encoded.as_bytes())?;
    let meta = RecoveryMeta {
        source_path: payload.source_path.clone(),
        updated_unix_secs: current_unix_secs(),
    };
    let meta_json = serialize_recovery_meta_json(&meta);
    atomic_write_file(&dir.join(RECOVERY_META_FILENAME), meta_json.as_bytes())?;
    *active_slot = next_slot;
    Ok(())
}

#[derive(Debug, Error)]
pub enum SessionError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("encoding error: {0}")]
    Encoding(#[from] crate::encoding::EncodingError),
    #[error("document is opened read-only and edits are disabled")]
    ReadOnlyDocument,
    #[error("corrupted recovery snapshot")]
    CorruptedSnapshot,
}

/// Complete document session managing buffer, undo stack, `base` / `theirs` snapshots,
/// state transitions, and non-blocking throttled dual-slot atomic crash recovery (PRD 4.5.1).
pub struct DocumentSession {
    pub path: Option<PathBuf>,
    pub buffer: TextBuffer,
    pub undo: UndoHistory,
    pub format: FileFormatMeta,
    pub state: DocState,
    pub base: DiskSnapshot,
    pub theirs: Option<DiskSnapshot>,
    pub recovery_dir: Option<PathBuf>,
    recovery_worker: Option<RecoveryWorker>,
    pub recovery_degraded: bool,
    /// Revision whose recovery snapshot has been scheduled/dirty since last flush.
    pending_recovery_revision: Option<u64>,
}

impl DocumentSession {
    /// Open an existing file from disk following PRD 4.5.2 encoding rules.
    /// Spawns a background recovery worker with a 200ms throttle window (<= 1s per PRD 4.5.1 F)
    /// so `edit()` never blocks the UI thread on disk I/O.
    pub fn open_file(
        path: impl AsRef<Path>,
        recovery_dir: Option<PathBuf>,
    ) -> Result<Self, SessionError> {
        Self::open_file_with_throttle(path, recovery_dir, Duration::from_millis(200))
    }

    pub fn open_file_with_throttle(
        path: impl AsRef<Path>,
        recovery_dir: Option<PathBuf>,
        throttle_window: Duration,
    ) -> Result<Self, SessionError> {
        let path = normalize_session_path(path.as_ref());
        let raw = fs::read(&path)?;
        let (text, format) = detect_and_decode(&raw)?;
        let buffer = TextBuffer::from_str(&text);
        let base = DiskSnapshot::from_rope(buffer.snapshot_rope());
        let recovery_worker = recovery_dir
            .clone()
            .map(|dir| RecoveryWorker::spawn(dir, throttle_window));

        Ok(Self {
            path: Some(path),
            buffer,
            undo: UndoHistory::new(),
            format,
            state: DocState::Clean,
            base,
            theirs: None,
            recovery_dir,
            recovery_worker,
            recovery_degraded: false,
            pending_recovery_revision: None,
        })
    }

    /// Create a brand-new untitled UTF-8 LF document session (PRD 4.5 F-01).
    pub fn new_untitled(recovery_dir: Option<PathBuf>) -> Self {
        Self::new_untitled_with_throttle(recovery_dir, Duration::from_millis(200))
    }

    pub fn new_untitled_with_throttle(
        recovery_dir: Option<PathBuf>,
        throttle_window: Duration,
    ) -> Self {
        let format = FileFormatMeta {
            decision: EncodingDecision::Editable(DetectedEncoding::Utf8),
            dominant_line_ending: LineEnding::Lf,
            has_trailing_newline: false,
        };
        let buffer = TextBuffer::new();
        let base = DiskSnapshot::from_rope(buffer.snapshot_rope());
        let recovery_worker = recovery_dir
            .clone()
            .map(|dir| RecoveryWorker::spawn(dir, throttle_window));

        Self {
            path: None,
            buffer,
            undo: UndoHistory::new(),
            format,
            state: DocState::Clean,
            base,
            theirs: None,
            recovery_dir,
            recovery_worker,
            recovery_degraded: false,
            pending_recovery_revision: None,
        }
    }

    /// True if the file encoding is ReadOnly (non-UTF-8/ambiguous, PRD 4.5.2).
    pub fn is_read_only(&self) -> bool {
        matches!(self.format.decision, EncodingDecision::ReadOnly { .. })
    }

    fn buffers_match_base(&self) -> bool {
        self.buffer.crc32() == self.base.crc32
            && self.buffer.len_bytes() == self.base.len_bytes
            && self.buffer.rope().bytes().eq(self.base.rope.bytes())
    }

    fn sync_state_after_buffer_mutation(&mut self) {
        match self.state {
            DocState::Clean | DocState::Dirty => {
                if self.buffers_match_base() {
                    self.state = DocState::Clean;
                    self.clear_current_recovery_snapshot();
                } else {
                    self.state = DocState::Dirty;
                    self.schedule_recovery_snapshot();
                }
            }
            DocState::Conflict => {
                self.schedule_recovery_snapshot();
            }
            DocState::Missing { .. } => {
                self.state = DocState::Missing {
                    has_local_edits: true,
                };
                self.schedule_recovery_snapshot();
            }
        }
    }

    /// Execute an interactive editing closure using `EditorViewState` directly against this session's
    /// single-source-of-truth `TextBuffer` and `UndoHistory`.
    /// Refuses edits cleanly if `self.is_read_only()` (PRD 4.5.2), transitions `Clean <-> Dirty`,
    /// and schedules non-blocking throttled recovery snapshots when `buffer.revision()` changes.
    pub fn with_editor_view(
        &mut self,
        view: &mut EditorViewState,
        f: impl FnOnce(&mut EditorContext<'_>),
    ) -> Result<bool, SessionError> {
        if self.is_read_only() {
            return Err(SessionError::ReadOnlyDocument);
        }
        let rev_before = self.buffer.revision();
        {
            let mut ctx = view.context(&mut self.buffer, &mut self.undo);
            f(&mut ctx);
        }
        let changed = self.buffer.revision() != rev_before;
        if changed {
            self.sync_state_after_buffer_mutation();
        }
        Ok(changed)
    }

    /// Handle external modification while preserving cursor position across Clean reloads and discarding
    /// unconfirmed IME preedit text so external events never commit raw phonetic preedit into buffer (PRD 4.1.1 E & G).
    pub fn on_external_modified_with_view(
        &mut self,
        new_disk_text: &str,
        view: &mut EditorViewState,
    ) {
        view.ime_preedit = None;
        view.ime_composition_selection = None;
        view.expansion_frozen = false;

        let old_text = self.buffer.to_string();
        let mut old_anchor = view.selection.anchor.min(old_text.len());
        while old_anchor > 0 && !old_text.is_char_boundary(old_anchor) {
            old_anchor -= 1;
        }
        let mut old_head = view.selection.head.min(old_text.len());
        while old_head > 0 && !old_text.is_char_boundary(old_head) {
            old_head -= 1;
        }
        let old_sel = Selection::new(old_anchor, old_head);
        let mapped_sel = Selection::new(
            map_offset_across_reload(&old_text, new_disk_text, old_sel.anchor),
            map_offset_across_reload(&old_text, new_disk_text, old_sel.head),
        );

        let rev_before = self.buffer.revision();
        self.on_external_modified_inner(new_disk_text, Some((old_sel, mapped_sel)));
        if self.buffer.revision() != rev_before {
            view.selection = mapped_sel;
        }
    }

    /// Perform "Use Disk Version" conflict resolution while mapping `view.selection` across the reloaded text.
    pub fn resolve_conflict_use_theirs_with_view(
        &mut self,
        captured_revision: u64,
        view: &mut EditorViewState,
    ) -> ConflictActionOutcome {
        let old_text = self.buffer.to_string();
        let mut old_anchor = view.selection.anchor.min(old_text.len());
        while old_anchor > 0 && !old_text.is_char_boundary(old_anchor) {
            old_anchor -= 1;
        }
        let mut old_head = view.selection.head.min(old_text.len());
        while old_head > 0 && !old_text.is_char_boundary(old_head) {
            old_head -= 1;
        }
        let old_sel = Selection::new(old_anchor, old_head);

        let (outcome, mapped_opt) =
            self.resolve_conflict_use_theirs_inner(captured_revision, Some((&old_text, old_sel)));
        if outcome == ConflictActionOutcome::ResolvedClean {
            if let Some(mapped_sel) = mapped_opt {
                view.selection = mapped_sel;
            }
            view.ime_preedit = None;
            view.ime_composition_selection = None;
            view.expansion_frozen = false;
        }
        outcome
    }

    /// Save the document to a target path (`Save As`, PRD 4.5.1 A & 4.5 F-01),
    /// refusing to bypass `Conflict` / external-divergence checks when targeting the currently open path.
    pub fn save_as(&mut self, new_path: impl AsRef<Path>) -> SaveOutcome {
        if self.is_read_only() {
            return SaveOutcome::WriteFailed("read-only document cannot be saved".to_string());
        }
        let target = new_path.as_ref().to_path_buf();

        if let Some(current_path) = &self.path {
            if paths_refer_to_same_file(current_path, &target) {
                if self.state == DocState::Conflict {
                    match fs::read(&target) {
                        Ok(disk_bytes) => {
                            let snap = match detect_and_decode(&disk_bytes) {
                                Ok((disk_text, _)) => DiskSnapshot::from_str(&disk_text),
                                Err(_) => DiskSnapshot::from_undecodable_bytes(&disk_bytes),
                            };
                            self.theirs = Some(snap);
                            let _ = self.flush_current_recovery_snapshot();
                            return SaveOutcome::EnteredConflict;
                        }
                        Err(_) => {
                            self.on_external_deleted();
                            return SaveOutcome::WriteFailed(
                                "target file disappeared before write".to_string(),
                            );
                        }
                    }
                }
                if !matches!(self.state, DocState::Missing { .. })
                    || fs::symlink_metadata(&target).is_ok()
                {
                    return self.save();
                }
            }
        }

        let written_revision = self.buffer.revision();
        let written_rope = self.buffer.snapshot_rope();
        let written_text = written_rope.to_string();
        let bytes = match encode_for_save(&written_text, &self.format) {
            Ok(b) => b,
            Err(e) => return SaveOutcome::WriteFailed(e.to_string()),
        };

        if let Some(parent) = target.parent() {
            if !parent.as_os_str().is_empty() {
                if let Err(e) = fs::create_dir_all(parent) {
                    return SaveOutcome::WriteFailed(e.to_string());
                }
            }
        }
        if let Err(e) = atomic_write_file(&target, &bytes) {
            return SaveOutcome::WriteFailed(e.to_string());
        }

        self.path = Some(normalize_session_path(&target));
        self.base = DiskSnapshot::from_rope(written_rope);
        self.theirs = None;

        if self.buffer.revision() == written_revision && self.buffer.crc32() == self.base.crc32 {
            self.state = DocState::Clean;
            self.clear_current_recovery_snapshot();
            SaveOutcome::SavedClean
        } else {
            self.state = DocState::Dirty;
            let _ = self.flush_current_recovery_snapshot();
            SaveOutcome::SavedButStillDirty { written_revision }
        }
    }

    /// Perform a local edit and transition state per PRD 4.5.1 C.
    /// Non-blocking: schedules an O(1) Rope-snapshot recovery payload on the background worker
    /// instead of performing synchronous serialization or `fsync` on the input thread.
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
        self.schedule_recovery_snapshot();
    }

    /// Handle external file modification event per PRD 4.5.1 C.
    pub fn on_external_modified(&mut self, new_disk_text: &str) {
        self.on_external_modified_inner(new_disk_text, None);
    }

    fn on_external_modified_inner(
        &mut self,
        new_disk_text: &str,
        selection_pair: Option<(Selection, Selection)>,
    ) {
        let disk_snap = DiskSnapshot::from_str(new_disk_text);
        match self.state {
            DocState::Clean => {
                self.apply_external_reload_to_buffer(new_disk_text, selection_pair);
                self.base = disk_snap;
                self.state = DocState::Clean;
            }
            DocState::Dirty => {
                if disk_snap.crc32 != self.base.crc32 || disk_snap.len_bytes != self.base.len_bytes {
                    self.theirs = Some(disk_snap);
                    self.state = DocState::Conflict;
                    self.schedule_recovery_snapshot();
                }
            }
            DocState::Conflict => {
                // Disk modified again during Conflict: refresh `theirs`, stay Conflict (PRD 4.5.1 C).
                self.theirs = Some(disk_snap);
                self.schedule_recovery_snapshot();
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
                    self.apply_external_reload_to_buffer(new_disk_text, selection_pair);
                    self.base = disk_snap;
                    self.state = DocState::Clean;
                } else {
                    self.theirs = Some(disk_snap);
                    self.state = DocState::Conflict;
                    self.schedule_recovery_snapshot();
                }
            }
        }
    }

    fn apply_external_reload_to_buffer(
        &mut self,
        new_disk_text: &str,
        selection_pair: Option<(Selection, Selection)>,
    ) {
        if self.buffer.rope().bytes().eq(new_disk_text.bytes()) {
            return;
        }
        let len = self.buffer.len_bytes();
        let (sel_before, sel_after) =
            selection_pair.unwrap_or((Selection::caret(0), Selection::caret(0)));
        self.undo.apply_edit(
            &mut self.buffer,
            ByteRange::new(0, len),
            new_disk_text,
            sel_before,
            sel_after,
            EditKind::Structural,
        );
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
        self.schedule_recovery_snapshot();
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

        // Pre-write verification against `base` (PRD 4.5.1 C).
        // Issue 1 fix: If disk file was modified into undecodable/binary bytes (e.g. containing NUL),
        // `detect_and_decode` returns `Err` — we MUST treat this as a conflict and refuse to overwrite!
        if let Ok(disk_bytes) = fs::read(&path) {
            match detect_and_decode(&disk_bytes) {
                Ok((disk_text, _)) => {
                    let disk_snap = DiskSnapshot::from_str(&disk_text);
                    if disk_snap.crc32 != self.base.crc32
                        || disk_snap.len_bytes != self.base.len_bytes
                    {
                        self.theirs = Some(disk_snap);
                        self.state = DocState::Conflict;
                        let _ = self.flush_current_recovery_snapshot();
                        return SaveOutcome::EnteredConflict;
                    }
                }
                Err(_) => {
                    self.theirs = Some(DiskSnapshot::from_undecodable_bytes(&disk_bytes));
                    self.state = DocState::Conflict;
                    let _ = self.flush_current_recovery_snapshot();
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
            let _ = self.flush_current_recovery_snapshot();
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
    /// Precondition: verifies disk still matches `theirs` (and is decodable), then persists `theirs`
    /// to recovery backup FIRST; if backup fails, aborts and stays Conflict.
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
        let latest_disk = match detect_and_decode(&disk_bytes) {
            Ok((disk_text, _)) => DiskSnapshot::from_str(&disk_text),
            Err(_) => DiskSnapshot::from_undecodable_bytes(&disk_bytes),
        };
        if latest_disk.crc32 != expected_theirs.crc32
            || latest_disk.len_bytes != expected_theirs.len_bytes
        {
            self.theirs = Some(latest_disk);
            let _ = self.flush_current_recovery_snapshot();
            return ConflictActionOutcome::AbortedStillConflict(
                "disk changed again; refreshed theirs".to_string(),
            );
        }

        // 2. Precondition: save `theirs` exact bytes to historical recovery backup BEFORE overwriting (PRD 4.5.1 D)
        if let Err(e) = self.write_historical_backup_bytes(
            "conflict-overwritten-theirs",
            &expected_theirs.to_backup_bytes(),
        ) {
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
            let _ = self.flush_current_recovery_snapshot();
            ConflictActionOutcome::ResolvedDirty
        }
    }

    /// Conflict resolution: "Use Disk Version" (PRD 4.5.1 D).
    /// Precondition:
    /// 1. Persists my version at `captured_revision` to recovery backup FIRST;
    /// 2. Re-reads disk and verifies it is still decodable and matches the `expected_theirs` snapshot
    ///    (Issue 2 fix: if disk changed from A to B before file watcher delivered event, refreshes `theirs` to B and aborts);
    /// 3. Verifies no new local edits occurred since `captured_revision`.
    pub fn resolve_conflict_use_theirs(&mut self, captured_revision: u64) -> ConflictActionOutcome {
        self.resolve_conflict_use_theirs_inner(captured_revision, None)
            .0
    }

    fn resolve_conflict_use_theirs_inner(
        &mut self,
        captured_revision: u64,
        view_selection: Option<(&str, Selection)>,
    ) -> (ConflictActionOutcome, Option<Selection>) {
        let Some(path) = self.path.clone() else {
            return (
                ConflictActionOutcome::AbortedStillConflict("no path".to_string()),
                None,
            );
        };
        let Some(expected_theirs) = self.theirs.clone() else {
            return (
                ConflictActionOutcome::AbortedStillConflict("not in conflict".to_string()),
                None,
            );
        };

        // Check if file still exists on disk
        let initial_disk_bytes = match fs::read(&path) {
            Ok(b) => b,
            Err(_) => {
                self.on_external_deleted();
                return (ConflictActionOutcome::TransitionedToMissing, None);
            }
        };

        // Verify disk before backup
        match detect_and_decode(&initial_disk_bytes) {
            Ok((disk_text, _)) => {
                let latest_disk = DiskSnapshot::from_str(&disk_text);
                if latest_disk.crc32 != expected_theirs.crc32
                    || latest_disk.len_bytes != expected_theirs.len_bytes
                {
                    self.theirs = Some(latest_disk);
                    let _ = self.flush_current_recovery_snapshot();
                    return (
                        ConflictActionOutcome::AbortedStillConflict(
                            "disk content changed again before reload; refreshed theirs".to_string(),
                        ),
                        None,
                    );
                }
            }
            Err(e) => {
                self.theirs = Some(DiskSnapshot::from_undecodable_bytes(&initial_disk_bytes));
                let _ = self.flush_current_recovery_snapshot();
                return (
                    ConflictActionOutcome::AbortedStillConflict(format!(
                        "disk content is not valid text ({e}); cannot load disk version"
                    )),
                    None,
                );
            }
        }

        // 1. Precondition: save my version to historical backup first
        let my_content = self.buffer.to_string();
        if let Err(e) = self.write_historical_backup("conflict-discarded-mine", &my_content) {
            self.recovery_degraded = true;
            return (
                ConflictActionOutcome::AbortedStillConflict(format!(
                    "recovery backup failed ({e}); refusing to discard local edits"
                )),
                None,
            );
        }

        // 2. Re-verify both local revision AND disk content after backup completes (PRD 4.5.1 D)
        if self.buffer.revision() != captured_revision {
            return (
                ConflictActionOutcome::AbortedStillConflict(
                    "new edits occurred while saving backup; aborted reload".to_string(),
                ),
                None,
            );
        }

        let post_backup_bytes = match fs::read(&path) {
            Ok(b) => b,
            Err(_) => {
                self.on_external_deleted();
                return (ConflictActionOutcome::TransitionedToMissing, None);
            }
        };
        match detect_and_decode(&post_backup_bytes) {
            Ok((disk_text, _)) => {
                let latest_disk = DiskSnapshot::from_str(&disk_text);
                if latest_disk.crc32 != expected_theirs.crc32
                    || latest_disk.len_bytes != expected_theirs.len_bytes
                {
                    self.theirs = Some(latest_disk);
                    let _ = self.flush_current_recovery_snapshot();
                    return (
                        ConflictActionOutcome::AbortedStillConflict(
                            "disk content changed while saving backup; refreshed theirs".to_string(),
                        ),
                        None,
                    );
                }
            }
            Err(e) => {
                self.theirs = Some(DiskSnapshot::from_undecodable_bytes(&post_backup_bytes));
                let _ = self.flush_current_recovery_snapshot();
                return (
                    ConflictActionOutcome::AbortedStillConflict(format!(
                        "disk content became undecodable ({e}); aborted reload"
                    )),
                    None,
                );
            }
        }

        let theirs_text = expected_theirs.content_string();
        let (sel_before, sel_after, mapped_ret) = match view_selection {
            Some((old_text, old_sel)) => {
                let mapped = Selection::new(
                    map_offset_across_reload(old_text, &theirs_text, old_sel.anchor),
                    map_offset_across_reload(old_text, &theirs_text, old_sel.head),
                );
                (old_sel, mapped, Some(mapped))
            }
            None => (Selection::caret(0), Selection::caret(0), None),
        };
        let len = self.buffer.len_bytes();
        self.undo.apply_edit(
            &mut self.buffer,
            ByteRange::new(0, len),
            &theirs_text,
            sel_before,
            sel_after,
            EditKind::Structural,
        );
        self.base = expected_theirs;
        self.theirs = None;
        self.state = DocState::Clean;
        self.clear_current_recovery_snapshot();
        (ConflictActionOutcome::ResolvedClean, mapped_ret)
    }

    fn build_recovery_payload(&self) -> RecoveryPayload {
        RecoveryPayload {
            source_path: self.path.as_ref().map(|p| normalize_session_path(p)),
            revision: self.buffer.revision(),
            state: self.state,
            content_rope: self.buffer.snapshot_rope(),
            base_rope: self.base.rope.clone(),
            theirs_rope: self.theirs.as_ref().map(|t| t.rope.clone()),
        }
    }

    /// Non-blocking: schedules a throttled recovery snapshot on the background worker thread.
    pub fn schedule_recovery_snapshot(&mut self) {
        if let Some(worker) = &self.recovery_worker {
            let payload = self.build_recovery_payload();
            self.pending_recovery_revision = Some(payload.revision);
            worker.schedule(payload);
            self.recovery_degraded = worker.is_degraded();
        }
    }

    /// Synchronously flush the current recovery snapshot to disk via the background worker
    /// (used in tests, explicit flushes, or state boundaries).
    pub fn flush_current_recovery_snapshot(&mut self) -> Result<(), SessionError> {
        if let Some(worker) = &self.recovery_worker {
            let payload = self.build_recovery_payload();
            let res = worker.flush_sync(payload);
            self.recovery_degraded = worker.is_degraded();
            if res.is_ok() {
                self.pending_recovery_revision = None;
            }
            res
        } else {
            Ok(())
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
        valid_slots
            .into_iter()
            .max_by_key(|r| (r.sequence, r.revision))
    }

    fn write_historical_backup(&self, reason: &str, content: &str) -> std::io::Result<()> {
        self.write_historical_backup_bytes(reason, content.as_bytes())
    }

    fn write_historical_backup_bytes(&self, reason: &str, bytes: &[u8]) -> std::io::Result<()> {
        let Some(dir) = &self.recovery_dir else {
            return Ok(());
        };
        fs::create_dir_all(dir)?;
        let rev = self.buffer.revision();
        let backup_path = dir.join(format!("history-{reason}-rev{rev}.bak"));
        atomic_write_file(&backup_path, bytes)
    }

    fn clear_current_recovery_snapshot(&mut self) {
        self.pending_recovery_revision = None;
        if let Some(worker) = &self.recovery_worker {
            worker.clear_sync();
        } else if let Some(dir) = &self.recovery_dir {
            remove_recovery_dir_files(dir);
        }
    }
}

fn canonicalize_lossy(p: &Path) -> PathBuf {
    if let Ok(c) = fs::canonicalize(p) {
        return c;
    }
    if let (Some(parent), Some(name)) = (p.parent(), p.file_name()) {
        let resolved_parent = if parent.as_os_str().is_empty() {
            fs::canonicalize(".").unwrap_or_else(|_| PathBuf::from("."))
        } else {
            fs::canonicalize(parent).unwrap_or_else(|_| parent.to_path_buf())
        };
        return resolved_parent.join(name);
    }
    p.to_path_buf()
}

fn normalize_session_path(p: &Path) -> PathBuf {
    let has_relative_segments = p.is_relative()
        || p.components().any(|c| {
            matches!(
                c,
                std::path::Component::CurDir | std::path::Component::ParentDir
            )
        });
    if has_relative_segments {
        canonicalize_lossy(p)
    } else {
        p.to_path_buf()
    }
}

fn paths_refer_to_same_file(a: &Path, b: &Path) -> bool {
    a == b || canonicalize_lossy(a) == canonicalize_lossy(b)
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
    sequence: Option<u64>,
    content: &str,
    base: &str,
    theirs: Option<&str>,
) -> u32 {
    let mut h = crc32fast::Hasher::new();
    h.update(&revision.to_le_bytes());
    if let Some(seq) = sequence {
        h.update(b"seq=");
        h.update(&seq.to_le_bytes());
        h.update(b"\0");
    }
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
        "VELLMARK_REC_V1\nslot={}\nseq={}\nrev={}\nstate={}\ncrc32={}\ncontent_len={}\nbase_len={}\nhas_theirs={}\ntheirs_len={}\n---\n",
        r.slot,
        r.sequence,
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
    let mut seq: Option<u64> = None;
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
            "seq" => seq = Some(v.parse().ok()?),
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
        seq,
        &content,
        &base_content,
        theirs_content.as_deref(),
    );
    if expected_crc != crc32 {
        return None;
    }

    Some(RecoverySlotReport {
        slot,
        sequence: seq.unwrap_or(0),
        revision,
        state,
        content,
        base_content,
        theirs_content,
        crc32,
    })
}

/// Sidecar JSON filename stored alongside `current.slot{0,1}.rec` inside each recovery dir.
pub const RECOVERY_META_FILENAME: &str = "recovery-meta.json";

/// Metadata sidecar identifying the document path and last recovery snapshot timestamp (PRD 4.5.1 F).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoveryMeta {
    pub source_path: Option<PathBuf>,
    pub updated_unix_secs: u64,
}

/// Unrecovered crash snapshot discovered under `recovery_root` on application startup (PRD 4.5.1 F).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingRecoveryEntry {
    pub recovery_dir: PathBuf,
    pub source_path: Option<PathBuf>,
    pub updated_unix_secs: Option<u64>,
    pub report: RecoverySlotReport,
}

fn current_unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn remove_recovery_dir_files(dir: &Path) {
    for slot in [0u8, 1u8] {
        let _ = fs::remove_file(dir.join(format!("current.slot{slot}.rec")));
    }
    let _ = fs::remove_file(dir.join(RECOVERY_META_FILENAME));
}

fn json_escape_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn serialize_recovery_meta_json(meta: &RecoveryMeta) -> String {
    let path_json = match &meta.source_path {
        Some(p) => json_escape_str(&p.to_string_lossy()),
        None => "null".to_string(),
    };
    format!(
        "{{\"source_path\":{path_json},\"updated_unix_secs\":{}}}\n",
        meta.updated_unix_secs
    )
}

fn parse_json_string_at(raw: &str, start: usize) -> Option<(String, usize)> {
    let bytes = raw.as_bytes();
    if bytes.get(start) != Some(&b'"') {
        return None;
    }
    let mut out = String::new();
    let mut i = start + 1;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'"' {
            return Some((out, i + 1));
        }
        if b == b'\\' {
            i += 1;
            let esc = *bytes.get(i)?;
            match esc {
                b'"' => out.push('"'),
                b'\\' => out.push('\\'),
                b'/' => out.push('/'),
                b'n' => out.push('\n'),
                b'r' => out.push('\r'),
                b't' => out.push('\t'),
                b'b' => out.push('\u{0008}'),
                b'f' => out.push('\u{000C}'),
                b'u' => {
                    let hex = raw.get(i + 1..i + 5)?;
                    let code = u32::from_str_radix(hex, 16).ok()?;
                    let ch = char::from_u32(code)?;
                    out.push(ch);
                    i += 4;
                }
                _ => return None,
            }
            i += 1;
            continue;
        }
        let ch = raw[i..].chars().next()?;
        out.push(ch);
        i += ch.len_utf8();
    }
    None
}

fn deserialize_recovery_meta_json(raw: &str) -> Option<RecoveryMeta> {
    let trimmed = raw.trim();
    if !trimmed.starts_with('{') || !trimmed.ends_with('}') {
        return None;
    }
    let bytes = trimmed.as_bytes();
    let mut i = 1usize;
    let end = bytes.len() - 1;
    let mut source_path: Option<PathBuf> = None;
    let mut updated_unix_secs: Option<u64> = None;

    while i < end {
        while i < end && (bytes[i].is_ascii_whitespace() || bytes[i] == b',') {
            i += 1;
        }
        if i >= end {
            break;
        }
        let (key, after_key) = parse_json_string_at(trimmed, i)?;
        i = after_key;
        while i < end && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if bytes.get(i) != Some(&b':') {
            return None;
        }
        i += 1;
        while i < end && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if trimmed[i..end].starts_with("null") {
            if key == "source_path" {
                source_path = None;
            }
            i += 4;
        } else if bytes.get(i) == Some(&b'"') {
            let (val, after_val) = parse_json_string_at(trimmed, i)?;
            if key == "source_path" {
                source_path = Some(PathBuf::from(val));
            }
            i = after_val;
        } else {
            let num_start = i;
            while i < end && bytes[i].is_ascii_digit() {
                i += 1;
            }
            if i == num_start {
                return None;
            }
            let num = trimmed[num_start..i].parse::<u64>().ok()?;
            if key == "updated_unix_secs" {
                updated_unix_secs = Some(num);
            }
        }
    }

    Some(RecoveryMeta {
        source_path,
        updated_unix_secs: updated_unix_secs.unwrap_or(0),
    })
}

/// Recursively scan `recovery_root` for unrecovered crash snapshots (`current.slot*.rec` +
/// optional `recovery-meta.json`) up to bounded subdirectory depth (PRD 4.5.1 F).
pub fn scan_recovery_root(recovery_root: impl AsRef<Path>) -> Vec<PendingRecoveryEntry> {
    let mut out = Vec::new();
    scan_recovery_dir_recursive(recovery_root.as_ref(), 0, &mut out);
    out.sort_by(|a, b| {
        b.updated_unix_secs
            .unwrap_or(0)
            .cmp(&a.updated_unix_secs.unwrap_or(0))
            .then_with(|| a.recovery_dir.cmp(&b.recovery_dir))
    });
    out
}

fn scan_recovery_dir_recursive(dir: &Path, depth: usize, out: &mut Vec<PendingRecoveryEntry>) {
    if depth > 4 || !dir.is_dir() {
        return;
    }
    if let Some(report) = DocumentSession::read_latest_intact_snapshot(dir) {
        let meta = fs::read_to_string(dir.join(RECOVERY_META_FILENAME))
            .ok()
            .and_then(|s| deserialize_recovery_meta_json(&s));
        let (source_path, updated_unix_secs) = match meta {
            Some(m) => (m.source_path, Some(m.updated_unix_secs)),
            None => (None, None),
        };
        out.push(PendingRecoveryEntry {
            recovery_dir: dir.to_path_buf(),
            source_path,
            updated_unix_secs,
            report,
        });
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        if entry.file_type().is_ok_and(|t| t.is_dir()) {
            scan_recovery_dir_recursive(&entry.path(), depth + 1, out);
        }
    }
}

/// A line-level conflicting region where local and disk edits overlapped relative to `base`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeConflictHunk {
    /// 0-based starting line index in `base`.
    pub base_start_line: usize,
    /// 0-based exclusive ending line index in `base`.
    pub base_end_line: usize,
    /// Conflicting line content from the local editor buffer.
    pub ours_lines: Vec<String>,
    /// Conflicting line content from the external disk file.
    pub theirs_lines: Vec<String>,
}

/// Outcome of attempting automatic three-way conflict resolution (PRD 4.5.1 D).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AutoMergeOutcome {
    /// Non-overlapping three-way merge succeeded and produced dirty local buffer state.
    MergedDirty { merged_bytes: usize },
    /// Both sides converged identically to disk; session returned to `Clean`.
    MergedClean,
    /// Overlapping changes detected; buffer and `DocState::Conflict` left untouched.
    HasOverlappingConflicts {
        conflicting_hunks: Vec<MergeConflictHunk>,
    },
    /// File disappeared on disk; transitioned to `MissingFile`.
    TransitionedToMissing,
    /// Precondition, TOCTOU, encoding, or I/O failure; remains in `Conflict`.
    AbortedStillConflict(String),
}

impl DocumentSession {
    /// Attempt line-level three-way auto-merge between `self.base`, current buffer (`ours`),
    /// and live disk (`theirs`) per PRD 4.5.1 D.
    ///
    /// Succeeds cleanly when hunks do not overlap; aborts without altering the buffer and returns
    /// `HasOverlappingConflicts` (staying in `Conflict`) if any hunks conflict.
    pub fn resolve_conflict_auto_merge(&mut self, captured_revision: u64) -> AutoMergeOutcome {
        if self.buffer.revision() != captured_revision {
            return AutoMergeOutcome::AbortedStillConflict(
                "document changed before auto-merge executed".to_string(),
            );
        }
        let Some(path) = self.path.clone() else {
            return AutoMergeOutcome::AbortedStillConflict("no path".to_string());
        };
        if self.state != DocState::Conflict {
            return AutoMergeOutcome::AbortedStillConflict("not in conflict".to_string());
        }
        let Some(expected_theirs) = self.theirs.clone() else {
            return AutoMergeOutcome::AbortedStillConflict("missing external snapshot".to_string());
        };

        let disk_bytes = match fs::read(&path) {
            Ok(b) => b,
            Err(_) => {
                self.on_external_deleted();
                return AutoMergeOutcome::TransitionedToMissing;
            }
        };

        let disk_text = match detect_and_decode(&disk_bytes) {
            Ok((text, _)) => text,
            Err(e) => {
                self.theirs = Some(DiskSnapshot::from_undecodable_bytes(&disk_bytes));
                let _ = self.flush_current_recovery_snapshot();
                return AutoMergeOutcome::AbortedStillConflict(format!(
                    "disk content is not valid text ({e}); cannot auto-merge"
                ));
            }
        };

        let latest_disk = DiskSnapshot::from_str(&disk_text);
        if expected_theirs.raw_bytes.is_some()
            || latest_disk.crc32 != expected_theirs.crc32
            || latest_disk.len_bytes != expected_theirs.len_bytes
        {
            self.theirs = Some(latest_disk);
            let _ = self.flush_current_recovery_snapshot();
            return AutoMergeOutcome::AbortedStillConflict(
                "disk content changed again before auto-merge; refreshed theirs".to_string(),
            );
        }

        let base_text = self.base.rope.to_string();
        let ours_text = self.buffer.to_string();
        match three_way_merge_lines(&base_text, &ours_text, &disk_text) {
            Ok(merged) => {
                if let Err(e) = self.write_historical_backup("conflict-pre-merge-mine", &ours_text)
                {
                    return AutoMergeOutcome::AbortedStillConflict(format!(
                        "failed to write historical backup: {e}"
                    ));
                }
                let old_len = self.buffer.len_bytes();
                let merged_len = merged.len();
                self.undo.apply_edit(
                    &mut self.buffer,
                    ByteRange::new(0, old_len),
                    &merged,
                    Selection::caret(0),
                    Selection::caret(merged_len),
                    EditKind::Structural,
                );
                self.base = latest_disk;
                self.theirs = None;
                if merged == disk_text {
                    self.state = DocState::Clean;
                    self.clear_current_recovery_snapshot();
                    AutoMergeOutcome::MergedClean
                } else {
                    self.state = DocState::Dirty;
                    let _ = self.flush_current_recovery_snapshot();
                    AutoMergeOutcome::MergedDirty {
                        merged_bytes: merged_len,
                    }
                }
            }
            Err(conflicting_hunks) => AutoMergeOutcome::HasOverlappingConflicts {
                conflicting_hunks,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct DiffHunk {
    base_start: usize,
    base_end: usize,
    side_start: usize,
    side_end: usize,
}

#[derive(Debug, Clone, Copy)]
struct MergeLine<'a> {
    content: &'a str,
    eol: &'a str,
    raw: &'a str,
}

fn split_lines_with_eol(text: &str) -> Vec<MergeLine<'_>> {
    if text.is_empty() {
        return Vec::new();
    }
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'\n' {
            let content_end = if i > start && bytes[i - 1] == b'\r' {
                i - 1
            } else {
                i
            };
            out.push(MergeLine {
                content: &text[start..content_end],
                eol: &text[content_end..i + 1],
                raw: &text[start..i + 1],
            });
            start = i + 1;
        }
        i += 1;
    }
    if start < bytes.len() {
        out.push(MergeLine {
            content: &text[start..],
            eol: "",
            raw: &text[start..],
        });
    }
    out
}

fn lcs_matching_lines<'a>(a: &[&'a str], b: &[&'a str]) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    let min_len = a.len().min(b.len());
    let mut prefix = 0usize;
    while prefix < min_len && a[prefix] == b[prefix] {
        pairs.push((prefix, prefix));
        prefix += 1;
    }
    let mut suffix = 0usize;
    while prefix + suffix < min_len && a[a.len() - 1 - suffix] == b[b.len() - 1 - suffix] {
        suffix += 1;
    }

    let a_end = a.len() - suffix;
    let b_end = b.len() - suffix;
    if prefix < a_end && prefix < b_end {
        let anchors = unique_line_anchors(&a[prefix..a_end], &b[prefix..b_end]);
        let mut prev_a = prefix;
        let mut prev_b = prefix;
        for (rel_a, rel_b) in anchors {
            let abs_a = prefix + rel_a;
            let abs_b = prefix + rel_b;
            lcs_small_gap(a, prev_a, abs_a, b, prev_b, abs_b, &mut pairs);
            pairs.push((abs_a, abs_b));
            prev_a = abs_a + 1;
            prev_b = abs_b + 1;
        }
        lcs_small_gap(a, prev_a, a_end, b, prev_b, b_end, &mut pairs);
    }

    for k in (0..suffix).rev() {
        pairs.push((a.len() - 1 - k, b.len() - 1 - k));
    }
    pairs
}

fn unique_line_anchors<'a>(a: &[&'a str], b: &[&'a str]) -> Vec<(usize, usize)> {
    use std::collections::HashMap;
    let mut count_a: HashMap<&'a str, usize> = HashMap::with_capacity(a.len());
    let mut pos_a: HashMap<&'a str, usize> = HashMap::with_capacity(a.len());
    for (idx, &line) in a.iter().enumerate() {
        *count_a.entry(line).or_insert(0) += 1;
        pos_a.entry(line).or_insert(idx);
    }
    let mut count_b: HashMap<&'a str, usize> = HashMap::with_capacity(b.len());
    let mut pos_b: HashMap<&'a str, usize> = HashMap::with_capacity(b.len());
    for (idx, &line) in b.iter().enumerate() {
        *count_b.entry(line).or_insert(0) += 1;
        pos_b.entry(line).or_insert(idx);
    }

    let mut candidates: Vec<(usize, usize)> = Vec::new();
    for (line, &ca) in &count_a {
        if ca == 1 && count_b.get(line) == Some(&1) {
            candidates.push((pos_a[line], pos_b[line]));
        }
    }
    candidates.sort_unstable_by_key(|&(ai, _)| ai);
    if candidates.is_empty() {
        return Vec::new();
    }

    // Standard O(K log K) LIS by `bi`
    let mut tails: Vec<usize> = Vec::new();
    let mut tail_indices: Vec<usize> = Vec::new();
    let mut prev = vec![usize::MAX; candidates.len()];
    for (idx, &(_, bi)) in candidates.iter().enumerate() {
        let pos = tails.partition_point(|&x| x < bi);
        if pos == tails.len() {
            tails.push(bi);
            tail_indices.push(idx);
        } else {
            tails[pos] = bi;
            tail_indices[pos] = idx;
        }
        if pos > 0 {
            prev[idx] = tail_indices[pos - 1];
        }
    }

    let mut lis = Vec::new();
    let mut cur = *tail_indices.last().unwrap();
    while cur != usize::MAX {
        lis.push(candidates[cur]);
        cur = prev[cur];
    }
    lis.reverse();
    lis
}

fn lcs_small_gap<'a>(
    a: &[&'a str],
    mut a_lo: usize,
    mut a_hi: usize,
    b: &[&'a str],
    mut b_lo: usize,
    mut b_hi: usize,
    out: &mut Vec<(usize, usize)>,
) {
    while a_lo < a_hi && b_lo < b_hi && a[a_lo] == b[b_lo] {
        out.push((a_lo, b_lo));
        a_lo += 1;
        b_lo += 1;
    }
    let mut suffix_pairs = Vec::new();
    while a_lo < a_hi && b_lo < b_hi && a[a_hi - 1] == b[b_hi - 1] {
        a_hi -= 1;
        b_hi -= 1;
        suffix_pairs.push((a_hi, b_hi));
    }

    let m = a_hi - a_lo;
    let n = b_hi - b_lo;
    if m > 0 && n > 0 && m.saturating_mul(n) <= 2_500_000 {
        let cols = n + 1;
        let mut dp = vec![0u32; (m + 1) * cols];
        for i in (0..m).rev() {
            for j in (0..n).rev() {
                dp[i * cols + j] = if a[a_lo + i] == b[b_lo + j] {
                    dp[(i + 1) * cols + (j + 1)] + 1
                } else {
                    dp[(i + 1) * cols + j].max(dp[i * cols + (j + 1)])
                };
            }
        }
        let mut i = 0usize;
        let mut j = 0usize;
        while i < m && j < n {
            if a[a_lo + i] == b[b_lo + j] {
                out.push((a_lo + i, b_lo + j));
                i += 1;
                j += 1;
            } else if dp[(i + 1) * cols + j] >= dp[i * cols + (j + 1)] {
                i += 1;
            } else {
                j += 1;
            }
        }
    }

    while let Some(pair) = suffix_pairs.pop() {
        out.push(pair);
    }
}

fn diff_line_hunks(base: &[MergeLine<'_>], side: &[MergeLine<'_>]) -> Vec<DiffHunk> {
    let base_raw: Vec<&str> = base.iter().map(|l| l.raw).collect();
    let side_raw: Vec<&str> = side.iter().map(|l| l.raw).collect();
    let mut matches = lcs_matching_lines(&base_raw, &side_raw);
    matches.push((base.len(), side.len()));

    let mut hunks = Vec::new();
    let mut cur_b = 0usize;
    let mut cur_s = 0usize;
    for (mb, ms) in matches {
        if cur_b < mb || cur_s < ms {
            hunks.push(DiffHunk {
                base_start: cur_b,
                base_end: mb,
                side_start: cur_s,
                side_end: ms,
            });
        }
        cur_b = mb + 1;
        cur_s = ms + 1;
    }
    hunks
}

fn hunks_conflict_or_touch(
    a: DiffHunk,
    b: DiffHunk,
    side_a: &[MergeLine<'_>],
    side_b: &[MergeLine<'_>],
) -> Option<bool> {
    // Returns Some(true) if strictly conflict, Some(false) if identical changes, None if disjoint.
    let a_empty = a.base_start == a.base_end;
    let b_empty = b.base_start == b.base_end;
    let overlaps = if a_empty && b_empty {
        a.base_start == b.base_start
    } else if a_empty {
        b.base_start < a.base_start && a.base_start < b.base_end
    } else if b_empty {
        a.base_start < b.base_start && b.base_start < a.base_end
    } else {
        a.base_start < b.base_end && b.base_start < a.base_end
    };

    if !overlaps {
        return None;
    }
    let same = a.base_start == b.base_start
        && a.base_end == b.base_end
        && side_a[a.side_start..a.side_end]
            .iter()
            .map(|l| l.raw)
            .eq(side_b[b.side_start..b.side_end].iter().map(|l| l.raw));
    Some(!same)
}

fn hunk_before(a: DiffHunk, b: DiffHunk) -> bool {
    if a.base_start != b.base_start {
        return a.base_start < b.base_start;
    }
    // Same start: pure insertion at `pos` precedes replacement `[pos..end)` when not overlapping
    a.base_start == a.base_end && b.base_start < b.base_end
}

fn reconstruct_side_slice_in_span(
    base: &[MergeLine<'_>],
    side: &[MergeLine<'_>],
    hunks: &[DiffHunk],
    span_start: usize,
    span_end: usize,
) -> Vec<String> {
    let mut lines = Vec::new();
    let mut cursor = span_start;
    for &h in hunks {
        while cursor < h.base_start {
            lines.push(base[cursor].content.to_string());
            cursor += 1;
        }
        for l in &side[h.side_start..h.side_end] {
            lines.push(l.content.to_string());
        }
        cursor = h.base_end;
    }
    while cursor < span_end {
        lines.push(base[cursor].content.to_string());
        cursor += 1;
    }
    lines
}

fn three_way_merge_lines(
    base_str: &str,
    ours_str: &str,
    theirs_str: &str,
) -> Result<String, Vec<MergeConflictHunk>> {
    let base = split_lines_with_eol(base_str);
    let ours = split_lines_with_eol(ours_str);
    let theirs = split_lines_with_eol(theirs_str);
    let dominant_eol = crate::encoding::LineEnding::detect_dominant(base_str).as_str();

    let ours_hunks = diff_line_hunks(&base, &ours);
    let theirs_hunks = diff_line_hunks(&base, &theirs);

    let mut merged_lines: Vec<MergeLine<'_>> = Vec::new();
    let mut conflicts: Vec<MergeConflictHunk> = Vec::new();
    let mut base_cursor = 0usize;
    let mut oi = 0usize;
    let mut ti = 0usize;

    while oi < ours_hunks.len() || ti < theirs_hunks.len() {
        match (ours_hunks.get(oi).copied(), theirs_hunks.get(ti).copied()) {
            (Some(oh), None) => {
                merged_lines.extend_from_slice(&base[base_cursor..oh.base_start]);
                merged_lines.extend_from_slice(&ours[oh.side_start..oh.side_end]);
                base_cursor = oh.base_end;
                oi += 1;
            }
            (None, Some(th)) => {
                merged_lines.extend_from_slice(&base[base_cursor..th.base_start]);
                merged_lines.extend_from_slice(&theirs[th.side_start..th.side_end]);
                base_cursor = th.base_end;
                ti += 1;
            }
            (Some(oh), Some(th)) => match hunks_conflict_or_touch(oh, th, &ours, &theirs) {
                None => {
                    if hunk_before(oh, th) {
                        merged_lines.extend_from_slice(&base[base_cursor..oh.base_start]);
                        merged_lines.extend_from_slice(&ours[oh.side_start..oh.side_end]);
                        base_cursor = oh.base_end;
                        oi += 1;
                    } else {
                        merged_lines.extend_from_slice(&base[base_cursor..th.base_start]);
                        merged_lines.extend_from_slice(&theirs[th.side_start..th.side_end]);
                        base_cursor = th.base_end;
                        ti += 1;
                    }
                }
                Some(false) => {
                    // Identical modification in both `ours` and `theirs`
                    merged_lines.extend_from_slice(&base[base_cursor..oh.base_start]);
                    merged_lines.extend_from_slice(&ours[oh.side_start..oh.side_end]);
                    base_cursor = oh.base_end;
                    oi += 1;
                    ti += 1;
                }
                Some(true) => {
                    // Grow transitive overlapping conflict cluster
                    let c_start = oh.base_start.min(th.base_start);
                    let mut c_end = oh.base_end.max(th.base_end);
                    let oi_start = oi;
                    let ti_start = ti;
                    oi += 1;
                    ti += 1;
                    loop {
                        let mut extended = false;
                        if let Some(&next_o) = ours_hunks.get(oi) {
                            if next_o.base_start < c_end {
                                c_end = c_end.max(next_o.base_end);
                                oi += 1;
                                extended = true;
                            }
                        }
                        if let Some(&next_t) = theirs_hunks.get(ti) {
                            if next_t.base_start < c_end {
                                c_end = c_end.max(next_t.base_end);
                                ti += 1;
                                extended = true;
                            }
                        }
                        if !extended {
                            break;
                        }
                    }
                    conflicts.push(MergeConflictHunk {
                        base_start_line: c_start,
                        base_end_line: c_end,
                        ours_lines: reconstruct_side_slice_in_span(
                            &base,
                            &ours,
                            &ours_hunks[oi_start..oi],
                            c_start,
                            c_end,
                        ),
                        theirs_lines: reconstruct_side_slice_in_span(
                            &base,
                            &theirs,
                            &theirs_hunks[ti_start..ti],
                            c_start,
                            c_end,
                        ),
                    });
                    base_cursor = c_end;
                }
            },
            (None, None) => break,
        }
    }

    if !conflicts.is_empty() {
        return Err(conflicts);
    }

    merged_lines.extend_from_slice(&base[base_cursor..]);
    let mut out = String::new();
    for (idx, line) in merged_lines.iter().enumerate() {
        out.push_str(line.content);
        if idx + 1 < merged_lines.len() {
            if line.eol.is_empty() {
                out.push_str(dominant_eol);
            } else {
                out.push_str(line.eol);
            }
        } else if !line.eol.is_empty() {
            out.push_str(line.eol);
        }
    }
    Ok(out)
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
        session.flush_current_recovery_snapshot().unwrap(); // Writes to slot 1

        session.edit(
            ByteRange::new(12, 12),
            "Edit 2\n",
            Selection::caret(12),
            Selection::caret(19),
            EditKind::Typing,
        );
        session.flush_current_recovery_snapshot().unwrap(); // Writes to slot 0

        let latest_before = DocumentSession::read_latest_intact_snapshot(&rec_dir).unwrap();
        assert_eq!(latest_before.content, "Base\nEdit 1\nEdit 2\n");
        let newest_slot = latest_before.slot;

        // Corrupt the latest slot file to simulate power loss mid-write
        fs::write(
            rec_dir.join(format!("current.slot{newest_slot}.rec")),
            "CORRUPTED_PARTIAL_WRITE",
        )
        .unwrap();

        // Reader must automatically recover the previous intact slot ("Base\nEdit 1\n")
        let recovered = DocumentSession::read_latest_intact_snapshot(&rec_dir).unwrap();
        assert_eq!(recovered.content, "Base\nEdit 1\n");

        drop(session);
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

        drop(session);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_issue_1_undecodable_binary_disk_change_blocks_save_and_enters_conflict() {
        // Issue 1: External process overwrites file with binary NUL bytes while local doc is Dirty.
        // Save must NOT let the pre-write check pass; it must enter Conflict and preserve disk bytes.
        let dir = unique_temp_dir("binary-disk-conflict");
        let file_path = dir.join("doc.md");
        let rec_dir = dir.join("recovery");
        fs::write(&file_path, "Base\n").unwrap();

        let mut session = DocumentSession::open_file(&file_path, Some(rec_dir)).unwrap();
        session.edit(
            ByteRange::new(5, 5),
            "Local edit\n",
            Selection::caret(5),
            Selection::caret(16),
            EditKind::Typing,
        );

        // External program writes binary content containing NUL byte
        let binary_disk = b"corrupted\x00binary\x00payload";
        fs::write(&file_path, binary_disk).unwrap();

        let outcome = session.save();
        assert_eq!(outcome, SaveOutcome::EnteredConflict);
        assert_eq!(session.state, DocState::Conflict);
        // Disk file must NOT have been overwritten
        assert_eq!(fs::read(&file_path).unwrap(), binary_disk);

        drop(session);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_issue_2_use_theirs_rechecks_disk_and_aborts_if_disk_changed_from_a_to_b() {
        // Issue 2: Cached `theirs` is version A, but disk has already changed to version B before
        // file watcher event arrived. `resolve_conflict_use_theirs` must re-verify disk, update `theirs` to B,
        // and stay in Conflict rather than loading stale A and marking Clean.
        let dir = unique_temp_dir("use-theirs-stale");
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

        // External modification A arrives via watcher
        fs::write(&file_path, "Theirs A\n").unwrap();
        session.on_external_modified("Theirs A\n");
        assert_eq!(session.state, DocState::Conflict);

        // Disk changes again to "Theirs B\n" BEFORE watcher notifies session
        fs::write(&file_path, "Theirs B\n").unwrap();

        let outcome = session.resolve_conflict_use_theirs(captured_rev);
        assert!(matches!(
            outcome,
            ConflictActionOutcome::AbortedStillConflict(_)
        ));
        assert_eq!(session.state, DocState::Conflict);
        assert_eq!(
            session.theirs.as_ref().unwrap().content_string(),
            "Theirs B\n"
        );
        // Local buffer must still hold "Base\nMine\n"
        assert_eq!(session.buffer.to_string(), "Base\nMine\n");

        drop(session);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_issue_3_throttled_background_recovery_coalesces_rapid_edits() {
        // Issue 3: Rapid edits are coalesced by the background worker within the throttle window
        // without blocking the caller thread on every keystroke, and flushed within the window.
        let dir = unique_temp_dir("throttled-worker");
        let file_path = dir.join("fast.md");
        let rec_dir = dir.join("recovery");
        fs::write(&file_path, "").unwrap();

        let mut session = DocumentSession::open_file_with_throttle(
            &file_path,
            Some(rec_dir.clone()),
            Duration::from_millis(60),
        )
        .unwrap();

        for i in 0..20usize {
            session.edit(
                ByteRange::new(i, i),
                "x",
                Selection::caret(i),
                Selection::caret(i + 1),
                EditKind::Typing,
            );
        }

        // Wait for the 60ms throttle timer to fire on the background worker
        thread::sleep(Duration::from_millis(120));

        let snap = DocumentSession::read_latest_intact_snapshot(&rec_dir).unwrap();
        assert_eq!(snap.revision, 20);
        assert_eq!(snap.content, "x".repeat(20));

        drop(session);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_schedule_immediately_followed_by_shutdown_does_not_hang_and_flushes() {
        // Regression test: scheduling snapshots with a long throttle window (60s) and immediately
        // dropping the session must NOT hang in `JoinHandle::join()` when `Shutdown` is coalesced
        // inside the inner `try_recv` loop, and must flush the pending snapshot before exiting.
        let dir = unique_temp_dir("immediate-shutdown");
        let file_path = dir.join("shutdown.md");
        let rec_dir = dir.join("recovery");
        fs::write(&file_path, "Start\n").unwrap();

        let start_time = Instant::now();
        {
            let mut session = DocumentSession::open_file_with_throttle(
                &file_path,
                Some(rec_dir.clone()),
                Duration::from_secs(60),
            )
            .unwrap();

            session.edit(
                ByteRange::new(6, 6),
                "Edited before immediate drop\n",
                Selection::caret(6),
                Selection::caret(35),
                EditKind::Typing,
            );
            // Drop happens immediately here while `Schedule` is still in the channel or inner loop
        }
        assert!(
            start_time.elapsed() < Duration::from_secs(2),
            "Dropping session hung waiting for background recovery worker"
        );

        let snap = DocumentSession::read_latest_intact_snapshot(&rec_dir).unwrap();
        assert_eq!(snap.content, "Start\nEdited before immediate drop\n");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_keep_mine_preserves_exact_undecodable_bytes_in_recovery_backup() {
        // Regression test: When disk is modified into undecodable/binary bytes (invalid UTF-8 + NUL)
        // and user chooses "Keep My Version", the historical backup file must preserve the exact
        // original bytes rather than a lossy UTF-8 replacement string.
        let dir = unique_temp_dir("keep-mine-raw-bytes");
        let file_path = dir.join("raw.md");
        let rec_dir = dir.join("recovery");
        fs::write(&file_path, "Base\n").unwrap();

        let mut session = DocumentSession::open_file(&file_path, Some(rec_dir.clone())).unwrap();
        session.edit(
            ByteRange::new(5, 5),
            "My local version\n",
            Selection::caret(5),
            Selection::caret(22),
            EditKind::Typing,
        );
        let rev = session.buffer.revision();
        let rope = session.buffer.snapshot_rope();

        // External process writes invalid UTF-8 + NUL bytes
        let undecodable_theirs: Vec<u8> = vec![0xFF, 0xFE, 0xFD, 0x00, 0x80, 0x9F, 0x41];
        fs::write(&file_path, &undecodable_theirs).unwrap();

        // Save triggers pre-write check -> Conflict with undecodable `theirs`
        assert_eq!(session.save(), SaveOutcome::EnteredConflict);
        assert_eq!(session.state, DocState::Conflict);

        // Resolve with "Keep My Version"
        let outcome = session.resolve_conflict_keep_mine(rev, rope);
        assert_eq!(outcome, ConflictActionOutcome::ResolvedClean);
        assert_eq!(
            fs::read_to_string(&file_path).unwrap(),
            "Base\nMy local version\n"
        );

        // Historical backup must match `undecodable_theirs` byte-for-byte
        let backup_path = rec_dir.join(format!("history-conflict-overwritten-theirs-rev{rev}.bak"));
        let backed_up_bytes = fs::read(&backup_path).unwrap();
        assert_eq!(backed_up_bytes, undecodable_theirs);

        drop(session);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_untitled_save_as_read_only_guard_and_external_reload_cursor_mapping() {
        let dir = unique_temp_dir("untitled-save-as");
        let target = dir.join("draft.md");
        let rec_dir = dir.join("recovery");

        // 1. Untitled draft + EditorViewState editing + Save As
        let mut session = DocumentSession::new_untitled(Some(rec_dir));
        let mut view = EditorViewState::default();
        session
            .with_editor_view(&mut view, |ctx| {
                ctx.insert_text("# Title\nHello world");
            })
            .unwrap();
        assert_eq!(session.state, DocState::Dirty);
        assert_eq!(session.save_as(&target), SaveOutcome::SavedClean);
        assert_eq!(fs::read_to_string(&target).unwrap(), "# Title\nHello world");

        // Undo back to base -> returns cleanly to DocState::Clean
        session
            .with_editor_view(&mut view, |ctx| {
                ctx.insert_char('!');
            })
            .unwrap();
        assert_eq!(session.state, DocState::Dirty);
        session
            .with_editor_view(&mut view, |ctx| {
                ctx.undo();
            })
            .unwrap();
        assert_eq!(session.state, DocState::Clean);

        // 2. External modification during Clean preserves cursor offset inside unchanged suffix
        view.selection = Selection::caret("# Title\nHello ".len());
        session.on_external_modified_with_view("# Title\nNew intro\nHello world", &mut view);
        assert_eq!(view.selection, Selection::caret("# Title\nNew intro\nHello ".len()));

        // 3. Non-UTF8 GBK file opens ReadOnly and blocks with_editor_view mutations
        let gbk_path = dir.join("gbk.md");
        fs::write(&gbk_path, [0xC4, 0xE3, 0xBA, 0xC3]).unwrap();
        let mut ro_session = DocumentSession::open_file(&gbk_path, None).unwrap();
        assert!(ro_session.is_read_only());
        let mut ro_view = EditorViewState::default();
        assert!(matches!(
            ro_session.with_editor_view(&mut ro_view, |ctx| ctx.insert_char('x')),
            Err(SessionError::ReadOnlyDocument)
        ));

        drop(session);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_prd_5_2_round_trip_byte_fidelity_unmodified_open_save() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let uniq = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "vellmark-prd-5-2-roundtrip-{}-{uniq}",
            std::process::id()
        ));
        let recovery_dir = root.join("recovery");
        fs::create_dir_all(&recovery_dir).expect("create temp dirs");

        let long_line = "中文长行Word ".repeat(900); // > 10KB UTF-8 line
        assert!(long_line.len() > 10_000);

        let crlf_no_trailing_nl = format!(
            "\u{FEFF}# 标题\r\n\r\n第一段包含混合换行：\nsecond line with CRLF\r\n{long_line}end-no-nl"
        );
        let mixed_crlf_lf_raw = crlf_no_trailing_nl.into_bytes();

        let cases: &[(&str, &[u8])] = &[
            (
                "bom_crlf_no_trailing_nl.md",
                mixed_crlf_lf_raw.as_slice(),
            ),
            (
                "pure_lf_no_trailing_nl.md",
                "第一段\n\n```rust\nlet x = 1;\n```\n结尾无换行".as_bytes(),
            ),
            (
                "pure_crlf_trailing_nl.md",
                "\u{FEFF}Title\r\n\r\nBody 中文\r\n".as_bytes(),
            ),
        ];

        for (name, raw_bytes) in cases {
            let file_path = root.join(name);
            fs::write(&file_path, raw_bytes).expect("write fixture");

            let mut session =
                DocumentSession::open_file(&file_path, Some(recovery_dir.clone()))
                    .expect("open fixture");
            let outcome = session.save();
            assert_eq!(
                outcome,
                crate::state::SaveOutcome::SavedClean,
                "expected clean save for {name}"
            );

            let saved_bytes = fs::read(&file_path).expect("read saved bytes");
            assert_eq!(
                saved_bytes.as_slice(),
                *raw_bytes,
                "PRD 5.2 round-trip byte mismatch for {name}"
            );
            drop(session);
        }

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn test_p1_save_as_current_path_in_conflict_does_not_overwrite_external_version() {
        let dir = unique_temp_dir("save-as-conflict");
        let file_path = dir.join("doc.md");
        let rec_dir = dir.join("recovery");
        fs::write(&file_path, "Base text\n").unwrap();

        let mut session = DocumentSession::open_file(&file_path, Some(rec_dir)).unwrap();
        let mut view = EditorViewState::default();
        session
            .with_editor_view(&mut view, |ctx| {
                ctx.insert_text("My edit\n");
            })
            .unwrap();

        // External process overwrites file -> triggers Conflict
        fs::write(&file_path, "External precious edit\n").unwrap();
        assert_eq!(session.save(), SaveOutcome::EnteredConflict);
        assert_eq!(session.state, DocState::Conflict);

        // Save As targeting the exact same path (via relative `./doc.md` style or direct path) MUST NOT overwrite!
        let same_via_dot = dir.join(".").join("doc.md");
        assert_eq!(session.save_as(&same_via_dot), SaveOutcome::EnteredConflict);
        assert_eq!(session.state, DocState::Conflict);
        assert_eq!(
            fs::read_to_string(&file_path).unwrap(),
            "External precious edit\n"
        );
        assert!(session.theirs.is_some());

        // Saving As a DIFFERENT path writes a copy safely without touching original conflicted file
        let copy_path = dir.join("doc-copy.md");
        assert_eq!(session.save_as(&copy_path), SaveOutcome::SavedClean);
        assert_eq!(
            fs::read_to_string(&file_path).unwrap(),
            "External precious edit\n"
        );
        assert_eq!(
            fs::read_to_string(&copy_path).unwrap(),
            "My edit\nBase text\n"
        );

        drop(session);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_p1_undo_after_external_reload_to_shorter_file_does_not_panic() {
        let dir = unique_temp_dir("reload-undo");
        let file_path = dir.join("long.md");
        let rec_dir = dir.join("recovery");
        let long_prefix = "A".repeat(400);
        let initial = format!("{long_prefix} tail");
        fs::write(&file_path, &initial).unwrap();

        let mut session = DocumentSession::open_file(&file_path, Some(rec_dir)).unwrap();
        let mut view = EditorViewState::default();
        view.selection = Selection::caret(initial.len());

        // Edit near offset 405, then save cleanly
        session
            .with_editor_view(&mut view, |ctx| {
                ctx.insert_text(" more");
            })
            .unwrap();
        assert_eq!(session.save(), SaveOutcome::SavedClean);

        // External process truncates file to much shorter content ("short")
        session.on_external_modified_with_view("short", &mut view);
        assert_eq!(session.buffer.to_string(), "short");
        assert_eq!(session.state, DocState::Clean);

        // Undoing after external shrinkage MUST NOT panic with out-of-bounds range!
        session.with_editor_view(&mut view, |ctx| ctx.undo()).unwrap();
        assert_eq!(session.buffer.to_string(), format!("{long_prefix} tail more"));
        assert_eq!(session.state, DocState::Dirty);

        // Second undo reverts the pre-save edit cleanly too
        session.with_editor_view(&mut view, |ctx| ctx.undo()).unwrap();
        assert_eq!(session.buffer.to_string(), initial);

        drop(session);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_p1_external_modified_during_ime_preedit_does_not_commit_raw_preedit() {
        let dir = unique_temp_dir("ime-external-reload");
        let file_path = dir.join("note.md");
        let rec_dir = dir.join("recovery");
        fs::write(&file_path, "Hello").unwrap();

        let mut session = DocumentSession::open_file(&file_path, Some(rec_dir)).unwrap();
        let mut view = EditorViewState::default();
        view.selection = Selection::caret(5);

        // User is composing `"ni"` in IME (not yet committed)
        session
            .with_editor_view(&mut view, |ctx| {
                ctx.ime_update("ni");
            })
            .unwrap();
        assert_eq!(session.state, DocState::Clean);
        assert_eq!(view.ime_preedit.as_deref(), Some("ni"));

        // External modification arrives while preedit is open:
        // must discard unconfirmed preedit instead of committing `"ni"` into buffer, and reload Clean!
        session.on_external_modified_with_view("Hello from disk", &mut view);
        assert_eq!(view.ime_preedit, None);
        assert!(!view.expansion_frozen);
        assert_eq!(session.state, DocState::Clean);
        assert_eq!(session.buffer.to_string(), "Hello from disk");

        // Subsequent committed Chinese character does NOT form `"ni你"`
        let end = session.buffer.len_bytes();
        session
            .with_editor_view(&mut view, |ctx| {
                ctx.set_caret(end);
                ctx.ime_commit("你");
            })
            .unwrap();
        assert_eq!(session.buffer.to_string(), "Hello from disk你");

        drop(session);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_prd_4_5_1_f_recovery_meta_and_scan_recovery_root() {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("vellmark_rec_scan_{nanos}"));
        let rec_root = root.join("recovery");
        let docs_dir = root.join("docs");
        fs::create_dir_all(&docs_dir).unwrap();

        let note_a = docs_dir.join("a \"draft\"\\note.md");
        fs::create_dir_all(note_a.parent().unwrap()).unwrap();
        fs::write(&note_a, "Original A").unwrap();

        let note_clean = docs_dir.join("clean.md");
        fs::write(&note_clean, "Clean").unwrap();

        // Session 1 (crashed file-backed doc) + Session 2 (crashed untitled) + Session 3 (cleanly saved)
        {
            let mut s1 = DocumentSession::open_file(&note_a, Some(rec_root.join("s1-tab-1"))).unwrap();
            let mut v1 = EditorViewState::default();
            s1.with_editor_view(&mut v1, |ctx| ctx.insert_text("Unsaved edit A")).unwrap();
            s1.flush_current_recovery_snapshot().unwrap();

            let mut s2 = DocumentSession::new_untitled(Some(rec_root.join("nested").join("s2-tab-1")));
            let mut v2 = EditorViewState::default();
            s2.with_editor_view(&mut v2, |ctx| ctx.insert_text("Untitled crash draft")).unwrap();
            s2.flush_current_recovery_snapshot().unwrap();

            let mut s3 = DocumentSession::open_file(&note_clean, Some(rec_root.join("s3-tab-1"))).unwrap();
            let mut v3 = EditorViewState::default();
            s3.with_editor_view(&mut v3, |ctx| ctx.insert_text(" saved")).unwrap();
            s3.flush_current_recovery_snapshot().unwrap();
            assert_eq!(s3.save(), SaveOutcome::SavedClean);
        }

        let found = scan_recovery_root(&rec_root);
        assert_eq!(found.len(), 2);
        let file_entry = found
            .iter()
            .find(|e| e.source_path.as_ref() == Some(&note_a))
            .expect("file-backed recovery entry must preserve exact source_path");
        assert!(file_entry.report.content.contains("Unsaved edit A"));
        assert!(file_entry.updated_unix_secs.unwrap_or(0) > 0);

        let untitled_entry = found
            .iter()
            .find(|e| e.source_path.is_none())
            .expect("untitled recovery entry must be discovered with source_path: None");
        assert_eq!(untitled_entry.report.content, "Untitled crash draft");

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn test_prd_4_5_1_d_three_way_auto_merge_disjoint_and_overlapping_conflicts() {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("vellmark_3way_merge_{nanos}"));
        fs::create_dir_all(&dir).unwrap();

        // 1. Disjoint CRLF modifications auto-merge cleanly
        let merge_path = dir.join("disjoint.md");
        let base_text = "# Header\r\nLine 1 base\r\nLine 2 shared\r\nLine 3 base\r\n";
        fs::write(&merge_path, base_text).unwrap();

        let mut session = DocumentSession::open_file(&merge_path, Some(dir.join("rec1"))).unwrap();
        let mut view = EditorViewState::default();
        // Local edit on Line 1
        session
            .with_editor_view(&mut view, |ctx| {
                ctx.set_caret("# Header\r\n".len());
                ctx.insert_text("Local ");
            })
            .unwrap();
        // External disk edit on Line 3
        let theirs_text = "# Header\r\nLine 1 base\r\nLine 2 shared\r\nLine 3 from disk\r\n";
        fs::write(&merge_path, theirs_text).unwrap();

        assert_eq!(session.save(), SaveOutcome::EnteredConflict);
        let rev = session.buffer.revision();
        let outcome = session.resolve_conflict_auto_merge(rev);
        assert!(matches!(outcome, AutoMergeOutcome::MergedDirty { .. }));
        assert_eq!(session.state, DocState::Dirty);
        assert!(session.theirs.is_none());
        assert_eq!(
            session.buffer.to_string(),
            "# Header\r\nLocal Line 1 base\r\nLine 2 shared\r\nLine 3 from disk\r\n"
        );
        assert_eq!(session.save(), SaveOutcome::SavedClean);

        // 2. Overlapping edits stay in Conflict and report conflicting hunk
        let conflict_path = dir.join("overlap.md");
        fs::write(&conflict_path, "Line 0\nTarget base\nLine 2\n").unwrap();
        let mut s_overlap =
            DocumentSession::open_file(&conflict_path, Some(dir.join("rec2"))).unwrap();
        let mut v_overlap = EditorViewState::default();
        let pos = "Line 0\n".len();
        s_overlap
            .with_editor_view(&mut v_overlap, |ctx| {
                ctx.set_caret(pos);
                ctx.insert_text("Ours: ");
            })
            .unwrap();
        fs::write(&conflict_path, "Line 0\nTheirs edit\nLine 2\n").unwrap();
        assert_eq!(s_overlap.save(), SaveOutcome::EnteredConflict);
        let snap_before = s_overlap.buffer.to_string();
        let rev2 = s_overlap.buffer.revision();

        let err_outcome = s_overlap.resolve_conflict_auto_merge(rev2);
        match err_outcome {
            AutoMergeOutcome::HasOverlappingConflicts { conflicting_hunks } => {
                assert_eq!(conflicting_hunks.len(), 1);
                assert_eq!(conflicting_hunks[0].base_start_line, 1);
                assert_eq!(conflicting_hunks[0].base_end_line, 2);
                assert_eq!(conflicting_hunks[0].ours_lines, vec!["Ours: Target base"]);
                assert_eq!(conflicting_hunks[0].theirs_lines, vec!["Theirs edit"]);
            }
            other => panic!("expected HasOverlappingConflicts, got {other:?}"),
        }
        assert_eq!(s_overlap.state, DocState::Conflict);
        assert_eq!(s_overlap.buffer.to_string(), snap_before);

        drop(session);
        drop(s_overlap);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_three_way_merge_preserves_trailing_newline_deletion_and_eol_changes() {
        // Local edits line 1 while external disk deletes trailing newline: merge must NOT restore `\n`.
        let merged = three_way_merge_lines(
            "Line 1 base\nLine 2 base\n",
            "Line 1 edited\nLine 2 base\n",
            "Line 1 base\nLine 2 base",
        )
        .expect("disjoint edit + trailing newline deletion must merge cleanly");
        assert_eq!(merged, "Line 1 edited\nLine 2 base");

        // External disk adds missing trailing newline while local edits line 1.
        let merged_added = three_way_merge_lines(
            "Line 1 base\nLine 2 base",
            "Line 1 edited\nLine 2 base",
            "Line 1 base\nLine 2 base\n",
        )
        .expect("disjoint edit + trailing newline addition must merge cleanly");
        assert_eq!(merged_added, "Line 1 edited\nLine 2 base\n");

        // Conflicting edit vs trailing newline deletion on the same line reports conflict rather than silent drop.
        let conflict = three_way_merge_lines("Only line\n", "Only line", "Only line\r\n");
        assert!(conflict.is_err());
    }

    #[test]
    fn test_recovery_scan_prefers_conflict_over_earlier_dirty_with_same_revision() {
        let dir = unique_temp_dir("same-rev-slot-order");
        let file_path = dir.join("doc.md");
        let rec_dir = dir.join("recovery");
        fs::write(&file_path, "Base\n").unwrap();

        let mut session = DocumentSession::open_file(&file_path, Some(rec_dir.clone())).unwrap();
        session.edit(
            ByteRange::new(5, 5),
            "Mine\n",
            Selection::caret(5),
            Selection::caret(10),
            EditKind::Typing,
        );
        let rev_after_edit = session.buffer.revision();
        // Flush 1: Dirty snapshot written to slot 1
        session.flush_current_recovery_snapshot().unwrap();
        let snap1 = DocumentSession::read_latest_intact_snapshot(&rec_dir).unwrap();
        assert_eq!(snap1.slot, 1);
        assert_eq!(snap1.state, DocState::Dirty);
        assert_eq!(snap1.revision, rev_after_edit);

        // External modification moves session to Conflict with the exact same buffer revision
        session.on_external_modified("Base\nDisk version\n");
        assert_eq!(session.state, DocState::Conflict);
        assert_eq!(session.buffer.revision(), rev_after_edit);
        session.flush_current_recovery_snapshot().unwrap(); // Written to slot 0

        let latest = DocumentSession::read_latest_intact_snapshot(&rec_dir).unwrap();
        assert_eq!(latest.slot, 0);
        assert_eq!(latest.state, DocState::Conflict);
        assert_eq!(latest.theirs_content.as_deref(), Some("Base\nDisk version\n"));

        let scanned = scan_recovery_root(&rec_dir);
        assert_eq!(scanned.len(), 1);
        assert_eq!(scanned[0].report.state, DocState::Conflict);
        assert!(scanned[0].report.sequence > snap1.sequence);

        drop(session);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_recovery_meta_stabilizes_relative_source_path() {
        let root = unique_temp_dir("rel-recovery-path");
        let rec_root = root.join("recovery");
        let rec_dir = rec_root.join("tab-1");
        let rel_path = Path::new("src/../Cargo.toml");
        let expected_canonical = fs::canonicalize("Cargo.toml").unwrap();

        let mut session = DocumentSession::open_file(rel_path, Some(rec_dir)).unwrap();
        assert!(
            session.path.as_ref().unwrap().is_absolute(),
            "relative open path must normalize to stable absolute path"
        );
        assert_eq!(session.path.as_ref().unwrap(), &expected_canonical);

        session.edit(
            ByteRange::empty(0),
            "# draft\n",
            Selection::caret(0),
            Selection::caret(8),
            EditKind::Typing,
        );
        session.flush_current_recovery_snapshot().unwrap();

        let scanned = scan_recovery_root(&rec_root);
        assert_eq!(scanned.len(), 1);
        assert_eq!(scanned[0].source_path.as_ref(), Some(&expected_canonical));

        drop(session);
        let _ = fs::remove_dir_all(&root);
    }
}
