//! Native cross-platform desktop GUI shell (`winit 0.30` + `wgpu` Metal/Vulkan/DX12),
//! coordinating virtualized editor rendering, OS IME composition, mouse/keyboard interaction,
//! sidebar outline & file tree navigation, multi-tab header, find overlay, and live themes.

use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use vellmark_core::{
    AutoMergeOutcome, ByteRange, ConflictActionOutcome, ConflictDiffKind, DocState, DocumentStats,
    EditorAction, EncodingDecision, ExternalSyncResult, FindMatch, FindOptions, SaveOutcome,
    Selection, ViewMode,
};
use vellmark_parser::{OutlineItem, active_outline_index};
use vellmark_render::{
    DrawList, EditorHitTarget, EditorSceneResult, EditorViewport, Rect, ResolvedThemeStyle,
    SceneRect, ShapedDocumentCache, shape_ui_text_into,
};
use vellmark_theme::BuiltinTheme;
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalPosition, LogicalSize, PhysicalPosition};
use winit::event::{ElementState, Ime, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, ModifiersState, NamedKey};
use winit::window::{CursorIcon, Theme as WindowTheme, Window, WindowAttributes, WindowId};

use crate::clipboard::{ClipboardBackend, NativeClipboard};
use crate::gpu::WgpuRenderer;
use crate::workbench::{AppWorkbench, WorkbenchRuntimeEvent};

const TAB_BAR_H: f32 = 36.0;
const STATUS_BAR_H: f32 = 26.0;
const SIDEBAR_W: f32 = 248.0;
const FIND_BAR_H: f32 = 38.0;
const ALERT_BANNER_H: f32 = 42.0;
const CLEANUP_BANNER_H: f32 = 30.0;
const BLINK_INTERVAL: Duration = Duration::from_millis(530);
const EXTERNAL_POLL_INTERVAL: Duration = Duration::from_millis(1500);

/// Persistent warning displayed when the recovery directory cannot be written (PRD 4.5.1 F).
pub const RECOVERY_DEGRADED_NOTICE: &str =
    "Crash recovery unavailable (dir unwritable) — 3s recovery target degraded";

/// Abstract Save-As file picker backend (`osascript`/native dialog at runtime; deterministic mock in tests).
pub trait SaveDialogBackend: Send + Sync {
    fn pick_save_as_path(&self, current_path: Option<&Path>, default_name: &str) -> Option<PathBuf>;
}

/// Non-interactive Save-As backend for headless tests (`with_clipboard` default).
#[derive(Debug, Default)]
pub struct NoopSaveDialog;

impl SaveDialogBackend for NoopSaveDialog {
    fn pick_save_as_path(&self, _current_path: Option<&Path>, _default_name: &str) -> Option<PathBuf> {
        None
    }
}

/// Deterministic in-memory Save-As dialog backend for unit tests.
#[derive(Debug, Default)]
pub struct MemorySaveDialog {
    next_path: Mutex<Option<PathBuf>>,
}

impl MemorySaveDialog {
    pub fn new(path: Option<PathBuf>) -> Self {
        Self {
            next_path: Mutex::new(path),
        }
    }

    pub fn set_next_path(&self, path: Option<PathBuf>) {
        if let Ok(mut g) = self.next_path.lock() {
            *g = path;
        }
    }
}

impl SaveDialogBackend for MemorySaveDialog {
    fn pick_save_as_path(&self, _current_path: Option<&Path>, _default_name: &str) -> Option<PathBuf> {
        self.next_path.lock().ok().and_then(|g| g.clone())
    }
}

fn parse_dialog_stdout_path(stdout: &[u8]) -> Option<PathBuf> {
    // Strip ONLY the single trailing LF appended by the Unix dialog CLI (`osascript`/`zenity`).
    // Never call `.trim()` or strip CRLF, so valid filenames ending in spaces or `\r` (e.g. `Icon\r`)
    // are preserved verbatim instead of silently targeting a different file.
    let raw_bytes = stdout.strip_suffix(b"\n").unwrap_or(stdout);
    if raw_bytes.is_empty() {
        return None;
    }
    #[cfg(unix)]
    {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;
        Some(PathBuf::from(OsStr::from_bytes(raw_bytes)))
    }
    #[cfg(not(unix))]
    {
        let s = String::from_utf8(raw_bytes.to_vec()).ok()?;
        Some(PathBuf::from(s))
    }
}

fn existing_default_dialog_dir(current_path: Option<&Path>) -> Option<&Path> {
    let start = current_path.and_then(|p| p.parent())?;
    start
        .ancestors()
        .find(|p| !p.as_os_str().is_empty() && p.is_dir())
}

/// Native desktop Save-As dialog adapter.
#[derive(Debug, Default)]
pub struct NativeSaveDialog;

impl SaveDialogBackend for NativeSaveDialog {
    fn pick_save_as_path(&self, current_path: Option<&Path>, default_name: &str) -> Option<PathBuf> {
        #[cfg(target_os = "macos")]
        {
            let dir_arg = existing_default_dialog_dir(current_path)
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_default();
            let script = "on run argv\n\
                set defName to item 1 of argv\n\
                set defDir to item 2 of argv\n\
                if defDir is not \"\" then\n\
                    set chosen to choose file name with prompt \"Save Markdown As\" default name defName default location (POSIX file defDir)\n\
                else\n\
                    set chosen to choose file name with prompt \"Save Markdown As\" default name defName\n\
                end if\n\
                return POSIX path of chosen\n\
            end run";
            let out = Command::new("osascript")
                .arg("-e")
                .arg(script)
                .arg(default_name)
                .arg(dir_arg)
                .output()
                .ok()?;
            if !out.status.success() {
                return None;
            }
            parse_dialog_stdout_path(&out.stdout)
        }
        #[cfg(all(unix, not(target_os = "macos")))]
        {
            let mut cmd = Command::new("zenity");
            cmd.arg("--file-selection")
                .arg("--save")
                .arg("--confirm-overwrite");
            if let Some(cur) = current_path {
                cmd.arg("--filename").arg(cur);
            } else {
                cmd.arg("--filename").arg(default_name);
            }
            let out = cmd.output().ok()?;
            if !out.status.success() {
                return None;
            }
            parse_dialog_stdout_path(&out.stdout)
        }
        #[cfg(not(unix))]
        {
            let _ = (current_path, default_name);
            None
        }
    }
}

pub const WELCOME_MARKDOWN: &str = "\
# Vellmark — Native WYSIWYG Markdown Studio\n\
\n\
Welcome to **Vellmark M0**! Move your cursor inside any formatted element like **bold text**, *italic emphasis*, ~~strikethrough~~, `inline code`, or [this link](https://vellmark.dev) to reveal its source syntax markers automatically.\n\
\n\
## Interactive Architecture Highlights\n\
\n\
> Two-level syntax expansion: collapsed cleanly while reading, revealed accurately under the caret, and frozen smoothly during **OS IME preedit composition**.\n\
\n\
- [x] Native GPU acceleration via `wgpu` + `cosmic-text` SDF glyph atlas\n\
- [x] Virtualized O(log N) viewport culling for **10 MB+ Markdown** files\n\
- [ ] Click checkboxes, Outline headings, or toggle **Source Mode** (`Cmd/Ctrl + /`)\n\
\n\
### Quick Shortcuts\n\
\n\
1. `Cmd/Ctrl + S` — Save with atomic CRC32 conflict guard\n\
2. `Cmd/Ctrl + F` — Regex & case-sensitive Find bar\n\
3. `Cmd/Ctrl + B / I / E` — Toggle Bold / Italic / Inline Code\n\
4. `Cmd/Ctrl + \\` — Toggle Sidebar (Files & Outline)\n\
5. `Cmd/Ctrl + + / - / 0` — Zoom in / out / reset\n\
\n\
```rust\n\
fn hello_vellmark() {\n\
    println!(\"Typora-grade WYSIWYG on Rust + wgpu!\");\n\
}\n\
```\n\
\n\
---\n\
\n\
Start typing anywhere in Chinese IME or English to try live editing.\n";

/// Active section shown in the sidebar when a folder is open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SidebarSection {
    Outline,
    Files,
}

#[derive(Debug, Clone)]
enum UiAction {
    SelectTab(usize),
    CloseTab(u64),
    NewTab,
    ToggleMode,
    ToggleFocusMode,
    CycleTheme,
    ToggleSidebar,
    SwitchSidebarSection(SidebarSection),
    JumpOutline(usize),
    OpenWorkspaceFile(PathBuf),
    ToggleFindCase,
    ToggleFindRegex,
    FindPrev,
    FindNext,
    CloseFind,
    FocusFindInput,
    ToggleConflictDiff,
    ResolveConflictKeepMine,
    ResolveConflictUseTheirs,
    ResolveConflictAutoMerge,
    SaveAsActive,
    ResaveMissingFile,
    RetrySaveActive,
    ForceCloseActiveMissing,
    ClearConflictBackups,
    DismissConflictBackupNotice,
    SaveActiveNow,
}

#[derive(Debug, Clone)]
struct UiHitZone {
    rect: Rect,
    action: UiAction,
    pointer_cursor: bool,
}

#[derive(Debug, Clone)]
struct FlatTreeRow {
    depth: usize,
    name: String,
    path: PathBuf,
    is_dir: bool,
}

#[derive(Debug, Clone)]
enum DocBannerKind {
    Conflict {
        theirs_updated: bool,
    },
    Missing {
        has_local_edits: bool,
        write_error: Option<String>,
    },
    WriteError {
        error: String,
    },
    RecoveryDegraded,
    ConflictBackupCleanup {
        count: usize,
    },
}

#[derive(Debug, Clone, Copy)]
struct LayoutRegions {
    tab_bar: Rect,
    sidebar: Option<Rect>,
    alert_banner: Option<Rect>,
    diff_panel: Option<Rect>,
    editor: Rect,
    status_bar: Rect,
}

struct CachedTabMeta {
    tab_id: u64,
    rev: u64,
    stats: DocumentStats,
    outline: Vec<OutlineItem>,
    matches_query: String,
    matches_opts: FindOptions,
    find_matches: Vec<FindMatch>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum ImeTarget {
    #[default]
    None,
    Tab(u64),
    FindBox,
}

/// Warning shown in the status bar whenever protected crash snapshots exceed the 500 MB budget.
pub const PROTECTED_SNAPSHOT_LIMIT_NOTICE: &str =
    "Protected recovery snapshots exceed 500 MB — please save or discard old drafts";

/// Platform-independent GUI controller owning layout state, shortcuts, IME state,
/// find overlay, and virtualized scene construction (fully testable headlessly).
pub struct GuiController {
    pub workbench: AppWorkbench,
    pub shaper: ShapedDocumentCache,
    pub style: ResolvedThemeStyle,
    pub window_size: (f32, f32),
    pub scale_factor: f32,
    pub sidebar_section: SidebarSection,
    pub find_open: bool,
    pub find_focused: bool,
    pub find_query: String,
    pub find_options: FindOptions,
    pub active_find_range: Option<ByteRange>,
    pub preferred_caret_x: Option<f32>,
    pub dragging_selection: bool,
    pub pointer_pos: (f32, f32),
    pub ime_enabled: bool,
    pub caret_visible: bool,
    pub last_caret_blink: Instant,
    pub last_external_poll: Instant,
    pub clipboard_text: String,
    pub status_notice: Option<(String, Instant)>,
    pub last_ime_cursor_area: Option<Rect>,
    pub untitled_save_dir: Option<PathBuf>,
    pub close_confirm_armed: bool,
    pub conflict_diff_open: bool,
    clipboard: Box<dyn ClipboardBackend>,
    save_dialog: Box<dyn SaveDialogBackend>,
    ime_target: ImeTarget,
    cached_meta: Option<CachedTabMeta>,
    ui_hit_zones: Vec<UiHitZone>,
    last_editor_rect: Rect,
    last_total_doc_height: f32,
    collapsed_dirs: HashSet<PathBuf>,
    pending_scroll_ensure_caret: bool,
    protected_limit_warned: bool,
}

impl GuiController {
    pub fn new(workbench: AppWorkbench) -> Self {
        Self::with_backends(
            workbench,
            Box::new(NativeClipboard::new()),
            Box::new(NativeSaveDialog),
        )
    }

    pub fn with_clipboard(workbench: AppWorkbench, clipboard: Box<dyn ClipboardBackend>) -> Self {
        Self::with_backends(workbench, clipboard, Box::new(NoopSaveDialog))
    }

    pub fn with_backends(
        workbench: AppWorkbench,
        clipboard: Box<dyn ClipboardBackend>,
        save_dialog: Box<dyn SaveDialogBackend>,
    ) -> Self {
        let style = style_from_workbench(&workbench);
        let sidebar_section = if workbench.file_tree.is_some() {
            SidebarSection::Files
        } else {
            SidebarSection::Outline
        };
        let protected_limit_warned = workbench.protected_snapshots_exceed_limit();
        let now = Instant::now();
        Self {
            workbench,
            shaper: ShapedDocumentCache::new(),
            style,
            window_size: (1220.0, 820.0),
            scale_factor: 1.0,
            sidebar_section,
            find_open: false,
            find_focused: false,
            find_query: String::new(),
            find_options: FindOptions::default(),
            active_find_range: None,
            preferred_caret_x: None,
            dragging_selection: false,
            pointer_pos: (0.0, 0.0),
            ime_enabled: false,
            caret_visible: true,
            last_caret_blink: now,
            last_external_poll: now,
            clipboard_text: String::new(),
            status_notice: None,
            last_ime_cursor_area: None,
            untitled_save_dir: None,
            close_confirm_armed: false,
            conflict_diff_open: false,
            clipboard,
            save_dialog,
            ime_target: ImeTarget::None,
            cached_meta: None,
            ui_hit_zones: Vec::new(),
            last_editor_rect: Rect::default(),
            last_total_doc_height: 0.0,
            collapsed_dirs: HashSet::new(),
            pending_scroll_ensure_caret: false,
            protected_limit_warned,
        }
    }

    pub fn set_save_dialog(&mut self, save_dialog: Box<dyn SaveDialogBackend>) {
        self.save_dialog = save_dialog;
    }

    pub fn refresh_style(&mut self) {
        self.style = style_from_workbench(&self.workbench);
    }

    pub fn set_notice(&mut self, msg: impl Into<String>) {
        self.status_notice = Some((msg.into(), Instant::now()));
    }

    /// Return the active status-bar notice text, keeping persistent warnings
    /// visible beyond the 3-second transient notice timeout until resolved.
    pub fn active_status_notice(&self) -> Option<String> {
        let transient = self
            .status_notice
            .as_ref()
            .and_then(|(msg, ts)| (ts.elapsed() < Duration::from_secs(3)).then_some(msg.as_str()));
        let mut persistent: Vec<&str> = Vec::new();
        if self
            .workbench
            .active_tab()
            .is_some_and(|t| t.session.recovery_degraded)
        {
            persistent.push(RECOVERY_DEGRADED_NOTICE);
        }
        if self.workbench.protected_snapshots_exceed_limit() {
            persistent.push(PROTECTED_SNAPSHOT_LIMIT_NOTICE);
        }
        let persistent_joined = if persistent.is_empty() {
            None
        } else {
            Some(persistent.join("   |   "))
        };
        match (transient, persistent_joined) {
            (Some(msg), Some(pers))
                if !persistent.contains(&msg) && !pers.contains(msg) =>
            {
                Some(format!("{msg}   |   {pers}"))
            }
            (Some(msg), _) => Some(msg.to_string()),
            (None, Some(pers)) => Some(pers),
            (None, None) => None,
        }
    }

    /// Synchronize transient notice state after a recovery-maintenance audit completes.
    pub fn sync_protected_snapshot_warning(&mut self) -> bool {
        let exceeds = self.workbench.protected_snapshots_exceed_limit();
        if exceeds {
            let newly_exceeded = !self.protected_limit_warned;
            self.protected_limit_warned = true;
            if newly_exceeded {
                self.set_notice(PROTECTED_SNAPSHOT_LIMIT_NOTICE);
            }
            newly_exceeded
        } else {
            let cleared = self.protected_limit_warned;
            self.protected_limit_warned = false;
            if self
                .status_notice
                .as_ref()
                .is_some_and(|(msg, _)| msg == PROTECTED_SNAPSHOT_LIMIT_NOTICE)
            {
                self.status_notice = None;
            }
            cleared
        }
    }

    /// Harvest any completed background maintenance worker result and refresh runtime limit warnings.
    pub fn poll_background_maintenance(&mut self) -> bool {
        if self.workbench.poll_recovery_maintenance().is_some() {
            self.sync_protected_snapshot_warning();
            return true;
        }
        false
    }

    fn trigger_background_maintenance(&mut self, force_when_warned: bool) {
        self.poll_background_maintenance();
        let started = if force_when_warned && self.protected_limit_warned {
            self.workbench.maintain_recovery_storage_async()
        } else {
            self.workbench.maybe_maintain_recovery_storage_async()
        };
        if !started {
            self.sync_protected_snapshot_warning();
        }
    }

    fn apply_runtime_events(&mut self, events: Vec<WorkbenchRuntimeEvent>) -> bool {
        if events.is_empty() {
            return false;
        }
        let active_id = self.workbench.active_tab().map(|t| t.id);
        let mut changed = false;
        for ev in events {
            changed = true;
            match ev {
                WorkbenchRuntimeEvent::AutoSavedClean { .. } => {
                    self.trigger_background_maintenance(true);
                }
                WorkbenchRuntimeEvent::EnteredConflict { tab_id, title } => {
                    if Some(tab_id) == active_id {
                        self.set_notice(format!(
                            "Conflict detected in '{title}' — review Diff / Keep Mine / Use Disk"
                        ));
                    } else {
                        self.set_notice(format!("Conflict detected in '{title}'"));
                    }
                }
                WorkbenchRuntimeEvent::ConflictTheirsUpdated { title, .. } => {
                    self.set_notice(format!(
                        "Disk content changed again for '{title}' — refreshed external snapshot"
                    ));
                }
                WorkbenchRuntimeEvent::ReloadedClean { title, .. } => {
                    self.set_notice(format!(
                        "Reloaded '{title}' from disk (Undo with Cmd/Ctrl+Z)"
                    ));
                }
                WorkbenchRuntimeEvent::BecameMissing {
                    title,
                    has_local_edits,
                    ..
                } => {
                    if has_local_edits {
                        self.set_notice(format!(
                            "'{title}' was deleted or moved (local changes kept)"
                        ));
                    } else {
                        self.set_notice(format!("'{title}' was deleted or moved"));
                    }
                }
                WorkbenchRuntimeEvent::Reappeared {
                    title, reloaded, ..
                } => {
                    if reloaded {
                        self.set_notice(format!("'{title}' reappeared and reloaded from disk"));
                    } else {
                        self.set_notice(format!("'{title}' restored on disk"));
                    }
                }
                WorkbenchRuntimeEvent::Renamed { new_path, .. } => {
                    let name = new_path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("renamed file");
                    self.set_notice(format!("File renamed externally -> {name}"));
                }
                WorkbenchRuntimeEvent::WriteFailed { title, error, .. } => {
                    self.set_notice(format!("Save failed for '{title}': {error}"));
                }
                WorkbenchRuntimeEvent::ReadError { title, error, .. } => {
                    self.set_notice(format!("Read error for '{title}': {error}"));
                }
            }
        }
        changed
    }

    /// Drive 2s idle auto-save, 5s/15s/60s failure retry backoff, and periodic external disk polling.
    pub fn tick_runtime_at(&mut self, now: Instant) -> bool {
        let auto_events = self.workbench.tick_auto_save_at(now);
        let mut changed = self.apply_runtime_events(auto_events);
        if now.duration_since(self.last_external_poll) >= EXTERNAL_POLL_INTERVAL {
            self.last_external_poll = now;
            let poll_events = self.workbench.poll_external_changes(false);
            changed |= self.apply_runtime_events(poll_events);
        }
        changed
    }

    /// Earliest wake deadline for the event loop combining caret blink, external poll, and auto-save.
    pub fn next_wake_deadline(&self) -> Instant {
        let mut next = self.last_caret_blink + BLINK_INTERVAL;
        next = next.min(self.last_external_poll + EXTERNAL_POLL_INTERVAL);
        if let Some(auto_dl) = self.workbench.next_auto_save_deadline() {
            next = next.min(auto_dl);
        }
        next
    }

    /// Immediately auto-save all eligible dirty named tabs when window loses focus (PRD 4.5.1 A).
    pub fn handle_focus_lost(&mut self) -> bool {
        let events = self.workbench.flush_all_dirty_auto_saves(Instant::now());
        self.apply_runtime_events(events)
    }

    /// Re-verify open files bypassing stat cache when window regains focus (PRD 4.5.1 G).
    pub fn handle_focus_gained(&mut self) -> bool {
        self.last_external_poll = Instant::now();
        let events = self.workbench.poll_external_changes(true);
        let changed = self.apply_runtime_events(events);
        self.trigger_background_maintenance(false);
        self.reset_blink();
        changed
    }

    fn flush_active_tab_before_switch(&mut self) {
        self.conflict_diff_open = false;
        if let Some(id) = self.workbench.active_tab().map(|t| t.id) {
            let events = self.workbench.flush_auto_save_for_tab(id, Instant::now());
            self.apply_runtime_events(events);
        }
    }

    fn reset_blink(&mut self) {
        self.caret_visible = true;
        self.last_caret_blink = Instant::now();
    }

    pub fn tick_blink(&mut self, now: Instant) -> bool {
        if now.duration_since(self.last_caret_blink) >= BLINK_INTERVAL {
            self.caret_visible = !self.caret_visible;
            self.last_caret_blink = now;
            return true;
        }
        false
    }

    fn active_doc_banner_kind(&self) -> Option<DocBannerKind> {
        let tab = self.workbench.active_tab()?;
        match tab.session.state {
            DocState::Conflict => Some(DocBannerKind::Conflict {
                theirs_updated: tab.conflict_theirs_updated,
            }),
            DocState::Missing { has_local_edits } => Some(DocBannerKind::Missing {
                has_local_edits,
                write_error: tab.last_write_error.clone(),
            }),
            DocState::Dirty if tab.last_write_error.is_some() => Some(DocBannerKind::WriteError {
                error: tab.last_write_error.clone().unwrap_or_default(),
            }),
            _ if tab.session.recovery_degraded => Some(DocBannerKind::RecoveryDegraded),
            _ if tab.conflict_backup_count > 0 && !tab.conflict_backup_notice_dismissed => {
                Some(DocBannerKind::ConflictBackupCleanup {
                    count: tab.conflict_backup_count,
                })
            }
            _ => None,
        }
    }

    fn layout_regions(&self) -> LayoutRegions {
        let (w, h) = (self.window_size.0.max(320.0), self.window_size.1.max(200.0));
        let tab_bar = Rect::new(0.0, 0.0, w, TAB_BAR_H);
        let status_bar = Rect::new(0.0, (h - STATUS_BAR_H).max(TAB_BAR_H), w, STATUS_BAR_H);
        let body_h = (status_bar.y - TAB_BAR_H).max(40.0);
        let sidebar = if self.workbench.config.sidebar_visible && w > 540.0 {
            Some(Rect::new(0.0, TAB_BAR_H, SIDEBAR_W, body_h))
        } else {
            None
        };
        let ed_x = sidebar.map(|s| s.w).unwrap_or(0.0);
        let ed_w = (w - ed_x).max(80.0);
        let mut cursor_y = TAB_BAR_H;
        let mut rem_h = body_h;

        let banner_h = match self.active_doc_banner_kind() {
            Some(DocBannerKind::ConflictBackupCleanup { .. }) => CLEANUP_BANNER_H,
            Some(_) => ALERT_BANNER_H,
            None => 0.0,
        }
        .min((rem_h - 40.0).max(0.0));
        let alert_banner = if banner_h > 0.0 {
            let r = Rect::new(ed_x, cursor_y, ed_w, banner_h);
            cursor_y += banner_h;
            rem_h -= banner_h;
            Some(r)
        } else {
            None
        };

        let show_diff = self.conflict_diff_open
            && self
                .workbench
                .active_tab()
                .is_some_and(|t| t.session.state == DocState::Conflict);
        let diff_h = if show_diff {
            (body_h * 0.36).clamp(110.0, 230.0).min((rem_h - 40.0).max(0.0))
        } else {
            0.0
        };
        let diff_panel = if diff_h > 0.0 {
            let r = Rect::new(ed_x, cursor_y, ed_w, diff_h);
            cursor_y += diff_h;
            rem_h -= diff_h;
            Some(r)
        } else {
            None
        };

        let editor = Rect::new(ed_x, cursor_y, ed_w, rem_h.max(40.0));
        LayoutRegions {
            tab_bar,
            sidebar,
            alert_banner,
            diff_panel,
            editor,
            status_bar,
        }
    }

    /// Recompute layout regions for `(win_w, win_h)` logical window dimensions.
    pub fn regions(&self) -> (Rect, Option<Rect>, Rect, Rect) {
        let l = self.layout_regions();
        (l.tab_bar, l.sidebar, l.editor, l.status_bar)
    }

    /// Build layered GPU-agnostic draw lists (`[editor_layer, chrome_layer]`) and hit zones.
    pub fn build_frame(&mut self) -> (DrawList, DrawList) {
        self.refresh_style();
        self.ui_hit_zones.clear();

        let layout = self.layout_regions();
        self.last_editor_rect = layout.editor;

        let editor_layer = self.build_editor_layer(layout.editor);
        let mut chrome_layer = DrawList::default();

        self.build_tab_bar(&mut chrome_layer, layout.tab_bar);
        if let Some(sb_rect) = layout.sidebar {
            self.build_sidebar(&mut chrome_layer, sb_rect);
        }
        if let (Some(banner_rect), Some(kind)) = (layout.alert_banner, self.active_doc_banner_kind())
        {
            self.build_doc_alert_banner(&mut chrome_layer, banner_rect, &kind);
        }
        if let Some(diff_rect) = layout.diff_panel {
            self.build_conflict_diff_panel(&mut chrome_layer, diff_rect);
        }
        if self.find_open {
            self.build_find_bar(&mut chrome_layer, layout.editor);
        }
        self.build_status_bar(&mut chrome_layer, layout.status_bar);

        (editor_layer, chrome_layer)
    }

    fn sync_cached_meta(&mut self) {
        let find_active = self.find_open && !self.find_query.is_empty();
        let query = if find_active {
            self.find_query.clone()
        } else {
            String::new()
        };
        let opts = self.find_options;

        let Some(tab) = self.workbench.active_tab() else {
            self.cached_meta = None;
            return;
        };
        let tab_id = tab.id;
        let rev = tab.session.buffer.revision();

        let needs_doc_recompute = self
            .cached_meta
            .as_ref()
            .is_none_or(|m| m.tab_id != tab_id || m.rev != rev);
        let needs_find_recompute = needs_doc_recompute
            || self
                .cached_meta
                .as_ref()
                .is_none_or(|m| m.matches_query != query || m.matches_opts != opts);

        if needs_doc_recompute {
            let stats = tab.stats();
            let outline = tab.outline();
            let find_matches = if find_active {
                tab.find_all(&query, opts).unwrap_or_default()
            } else {
                Vec::new()
            };
            self.cached_meta = Some(CachedTabMeta {
                tab_id,
                rev,
                stats,
                outline,
                matches_query: query,
                matches_opts: opts,
                find_matches,
            });
        } else if needs_find_recompute {
            let find_matches = if find_active {
                tab.find_all(&query, opts).unwrap_or_default()
            } else {
                Vec::new()
            };
            if let Some(m) = self.cached_meta.as_mut() {
                m.matches_query = query;
                m.matches_opts = opts;
                m.find_matches = find_matches;
            }
        }
    }

    fn build_editor_layer(&mut self, editor_rect: Rect) -> DrawList {
        self.sync_cached_meta();

        let horiz_pad = 36.0;
        let max_w = self
            .style
            .max_content_width_px
            .min((editor_rect.w - horiz_pad * 2.0).max(200.0));
        let content_left_x = editor_rect.x + ((editor_rect.w - max_w) * 0.5).max(16.0);

        let focus_mode = self.workbench.config.focus_mode;
        let typewriter_mode = self.workbench.config.typewriter_mode;
        let scale_factor = self.scale_factor;
        let show_caret = self.caret_visible && !self.find_focused;
        let active_find = self.active_find_range;

        let find_ranges: Vec<ByteRange> = self
            .cached_meta
            .as_ref()
            .map(|m| m.find_matches.iter().map(|fm| fm.range).collect())
            .unwrap_or_default();

        let scene_opt: Option<EditorSceneResult> = if let Some(tab) = self.workbench.active_tab() {
            let rev = tab.session.buffer.revision();
            let src_len = tab.session.buffer.len_bytes();
            let vp = EditorViewport {
                doc_rect: editor_rect,
                content_left_x,
                content_width_px: max_w,
                scroll_y: tab.scroll_y,
                overscan_px: 220.0,
                scale_factor,
                source_rev: rev,
                source_len: src_len,
                view: &tab.view,
                focus_mode,
                show_caret,
                find_matches: &find_ranges,
                active_find,
            };
            Some(tab.with_incremental_blocks(|blocks| {
                self.shaper.build_editor_scene(blocks, &self.style, &vp, |r| {
                    tab.session.buffer.slice_to_string(r)
                })
            }))
        } else {
            None
        };

        let Some(mut scene) = scene_opt else {
            self.last_ime_cursor_area = None;
            let mut empty = DrawList::default();
            empty.rects.push(SceneRect::solid(editor_rect, self.style.bg_color));
            shape_ui_text_into(
                self.shaper.font_system_mut(),
                "No open document. Press Cmd/Ctrl+N to create one.",
                editor_rect.x + 40.0,
                editor_rect.y + 40.0,
                self.style.base_font_size_px,
                self.style.base_line_height_px,
                400,
                self.style.chrome.muted_text,
                Some(editor_rect.w - 80.0),
                Some(editor_rect),
                self.scale_factor,
                &mut empty.glyphs,
            );
            return empty;
        };

        self.last_total_doc_height = scene.total_height;
        self.last_ime_cursor_area = scene.ime_cursor_area;

        // Keep active cursor line visible or vertically centered in Typewriter Mode
        if self.pending_scroll_ensure_caret || typewriter_mode {
            let need_reframe = if let Some(tab) = self.workbench.active_tab_mut() {
                let prev_scroll = tab.scroll_y;
                let max_scroll = (scene.total_height - editor_rect.h).max(0.0);
                if typewriter_mode {
                    tab.update_scroll_for_cursor(
                        scene.caret_doc_y,
                        scene.caret_doc_y,
                        scene.caret_line_height,
                        editor_rect.h,
                        true,
                    );
                } else {
                    let top_margin = 24.0;
                    let bot_margin = 28.0;
                    let caret_top = scene.caret_doc_y;
                    let caret_bot = scene.caret_doc_y + scene.caret_line_height;
                    if caret_top < tab.scroll_y + top_margin {
                        tab.scroll_y = (caret_top - top_margin).max(0.0);
                    } else if caret_bot > tab.scroll_y + editor_rect.h - bot_margin {
                        tab.scroll_y = (caret_bot - editor_rect.h + bot_margin).clamp(0.0, max_scroll);
                    }
                }
                tab.scroll_y = tab.scroll_y.clamp(0.0, max_scroll);
                (tab.scroll_y - prev_scroll).abs() > 0.5
            } else {
                false
            };
            self.pending_scroll_ensure_caret = false;
            if need_reframe {
                if let Some(tab) = self.workbench.active_tab() {
                    let rev = tab.session.buffer.revision();
                    let src_len = tab.session.buffer.len_bytes();
                    let vp2 = EditorViewport {
                        doc_rect: editor_rect,
                        content_left_x,
                        content_width_px: max_w,
                        scroll_y: tab.scroll_y,
                        overscan_px: 220.0,
                        scale_factor,
                        source_rev: rev,
                        source_len: src_len,
                        view: &tab.view,
                        focus_mode,
                        show_caret,
                        find_matches: &find_ranges,
                        active_find,
                    };
                    scene = tab.with_incremental_blocks(|blocks| {
                        self.shaper.build_editor_scene(blocks, &self.style, &vp2, |r| {
                            tab.session.buffer.slice_to_string(r)
                        })
                    });
                    self.last_ime_cursor_area = scene.ime_cursor_area;
                }
            }
        }

        // Scrollbar thumb indicator on right edge when document exceeds viewport height
        if scene.total_height > editor_rect.h {
            let ratio = (editor_rect.h / scene.total_height).clamp(0.05, 1.0);
            let thumb_h = (editor_rect.h * ratio).max(26.0);
            let max_scroll = (scene.total_height - editor_rect.h).max(1.0);
            let scroll_frac = self
                .workbench
                .active_tab()
                .map(|t| (t.scroll_y / max_scroll).clamp(0.0, 1.0))
                .unwrap_or(0.0);
            let thumb_y = editor_rect.y + scroll_frac * (editor_rect.h - thumb_h);
            let sb_rect = Rect::new(editor_rect.x + editor_rect.w - 8.0, thumb_y + 2.0, 5.0, thumb_h - 4.0);
            scene.draw.rects.push(
                SceneRect::rounded(sb_rect, self.style.chrome.muted_text.with_alpha(95), 2.5)
                    .with_clip(Some(editor_rect)),
            );
        }

        scene.draw
    }

    fn build_tab_bar(&mut self, dl: &mut DrawList, bar: Rect) {
        let chrome = self.style.chrome;
        dl.rects.push(SceneRect::solid(bar, chrome.tab_bar_bg));
        dl.rects.push(SceneRect::solid(
            Rect::new(bar.x, bar.y + bar.h - 1.0, bar.w, 1.0),
            chrome.border,
        ));

        // Sidebar toggle button on far left
        let sb_btn = Rect::new(8.0, 6.0, 28.0, 24.0);
        let sb_bg = if self.workbench.config.sidebar_visible {
            chrome.hover_bg
        } else {
            chrome.panel_bg
        };
        dl.rects
            .push(SceneRect::rounded(sb_btn, sb_bg, 5.0).with_border(1.0, chrome.border));
        self.push_ui_text(dl, "|||", sb_btn.x + 7.0, sb_btn.y + 4.0, 11.5, 600, chrome.text, None, Some(bar));
        self.ui_hit_zones.push(UiHitZone {
            rect: sb_btn,
            action: UiAction::ToggleSidebar,
            pointer_cursor: true,
        });

        // Right-side action pills (Theme, Focus, WYSIWYG/Source)
        let mode_label = match self.workbench.active_tab().map(|t| t.view.mode) {
            Some(ViewMode::Source) => "Source",
            _ => "WYSIWYG",
        };
        let mode_w = 78.0;
        let focus_w = 64.0;
        let theme_w = 68.0;
        let right_margin = 10.0;

        let mode_rect = Rect::new(bar.w - right_margin - mode_w, 6.0, mode_w, 24.0);
        let focus_rect = Rect::new(mode_rect.x - 6.0 - focus_w, 6.0, focus_w, 24.0);
        let theme_rect = Rect::new(focus_rect.x - 6.0 - theme_w, 6.0, theme_w, 24.0);

        let mode_bg = if matches!(
            self.workbench.active_tab().map(|t| t.view.mode),
            Some(ViewMode::Source)
        ) {
            chrome.accent.with_alpha(45)
        } else {
            chrome.panel_bg
        };
        dl.rects
            .push(SceneRect::rounded(mode_rect, mode_bg, 5.0).with_border(1.0, chrome.border));
        self.push_ui_text(
            dl,
            mode_label,
            mode_rect.x + 11.0,
            mode_rect.y + 4.0,
            11.5,
            600,
            chrome.text,
            Some(mode_w - 8.0),
            Some(bar),
        );
        self.ui_hit_zones.push(UiHitZone {
            rect: mode_rect,
            action: UiAction::ToggleMode,
            pointer_cursor: true,
        });

        let focus_bg = if self.workbench.config.focus_mode {
            chrome.accent.with_alpha(45)
        } else {
            chrome.panel_bg
        };
        dl.rects
            .push(SceneRect::rounded(focus_rect, focus_bg, 5.0).with_border(1.0, chrome.border));
        self.push_ui_text(
            dl,
            "Focus",
            focus_rect.x + 13.0,
            focus_rect.y + 4.0,
            11.5,
            500,
            chrome.text,
            Some(focus_w - 8.0),
            Some(bar),
        );
        self.ui_hit_zones.push(UiHitZone {
            rect: focus_rect,
            action: UiAction::ToggleFocusMode,
            pointer_cursor: true,
        });

        dl.rects
            .push(SceneRect::rounded(theme_rect, chrome.panel_bg, 5.0).with_border(1.0, chrome.border));
        self.push_ui_text(
            dl,
            "Theme",
            theme_rect.x + 12.0,
            theme_rect.y + 4.0,
            11.5,
            500,
            chrome.text,
            Some(theme_w - 8.0),
            Some(bar),
        );
        self.ui_hit_zones.push(UiHitZone {
            rect: theme_rect,
            action: UiAction::CycleTheme,
            pointer_cursor: true,
        });

        // Document tabs
        let max_tabs_right = (theme_rect.x - 40.0).max(120.0);
        let tab_count = self.workbench.tabs.len();
        let tabs_meta: Vec<(usize, u64, String, bool)> = self
            .workbench
            .tabs
            .iter()
            .enumerate()
            .map(|(idx, tab)| {
                let is_active = self.workbench.active_tab_idx == Some(idx);
                let title = tab.title();
                let prefix = match tab.session.state {
                    DocState::Dirty => "• ",
                    DocState::Conflict => "! ",
                    DocState::Missing { .. } => "? ",
                    DocState::Clean => "",
                };
                (idx, tab.id, format!("{prefix}{title}"), is_active)
            })
            .collect();

        let mut x = sb_btn.x + sb_btn.w + 10.0;
        for (idx, tab_id, label, is_active) in tabs_meta {
            if x + 64.0 > max_tabs_right {
                break;
            }
            let tab_w = (label.len() as f32 * 7.5 + 42.0).clamp(96.0, 200.0).min(max_tabs_right - x);
            let tab_rect = Rect::new(x, 5.0, tab_w, 27.0);
            let bg = if is_active {
                chrome.active_tab_bg
            } else {
                chrome.tab_bar_bg
            };
            dl.rects.push(
                SceneRect::rounded(tab_rect, bg, 6.0)
                    .with_border(if is_active { 1.0 } else { 0.0 }, chrome.border),
            );
            if is_active {
                dl.rects.push(SceneRect::solid(
                    Rect::new(tab_rect.x + 8.0, tab_rect.y + tab_rect.h - 2.0, tab_w - 16.0, 2.0),
                    chrome.accent,
                ));
            }
            let col = if is_active { chrome.text } else { chrome.muted_text };
            self.push_ui_text(
                dl,
                &label,
                tab_rect.x + 10.0,
                tab_rect.y + 5.0,
                12.0,
                if is_active { 600 } else { 400 },
                col,
                Some(tab_w - 28.0),
                Some(tab_rect),
            );
            self.ui_hit_zones.push(UiHitZone {
                rect: Rect::new(tab_rect.x, tab_rect.y, (tab_w - 22.0).max(30.0), tab_rect.h),
                action: UiAction::SelectTab(idx),
                pointer_cursor: true,
            });

            if tab_count > 1 {
                let close_r = Rect::new(tab_rect.x + tab_w - 20.0, tab_rect.y + 5.0, 16.0, 16.0);
                self.push_ui_text(
                    dl,
                    "×",
                    close_r.x + 3.0,
                    close_r.y - 1.0,
                    12.5,
                    500,
                    chrome.muted_text,
                    None,
                    Some(tab_rect),
                );
                self.ui_hit_zones.push(UiHitZone {
                    rect: close_r,
                    action: UiAction::CloseTab(tab_id),
                    pointer_cursor: true,
                });
            }
            x += tab_w + 6.0;
        }

        let new_btn = Rect::new(x, 6.0, 24.0, 24.0);
        if new_btn.x + new_btn.w < max_tabs_right + 28.0 {
            dl.rects.push(SceneRect::rounded(new_btn, chrome.panel_bg, 5.0));
            self.push_ui_text(dl, "+", new_btn.x + 7.0, new_btn.y + 3.0, 14.0, 600, chrome.text, None, Some(bar));
            self.ui_hit_zones.push(UiHitZone {
                rect: new_btn,
                action: UiAction::NewTab,
                pointer_cursor: true,
            });
        }
    }

    fn build_sidebar(&mut self, dl: &mut DrawList, sb: Rect) {
        let chrome = self.style.chrome;
        dl.rects.push(SceneRect::solid(sb, chrome.sidebar_bg));
        dl.rects.push(SceneRect::solid(
            Rect::new(sb.x + sb.w - 1.0, sb.y, 1.0, sb.h),
            chrome.border,
        ));

        // Header switch: Outline | Files
        let header_h = 32.0;
        let half_w = (sb.w - 20.0) * 0.5;
        let outline_btn = Rect::new(sb.x + 8.0, sb.y + 6.0, half_w, 22.0);
        let files_btn = Rect::new(outline_btn.x + half_w + 4.0, sb.y + 6.0, half_w, 22.0);

        for (btn, sec, label) in [
            (outline_btn, SidebarSection::Outline, "Outline"),
            (files_btn, SidebarSection::Files, "Files"),
        ] {
            let active = self.sidebar_section == sec;
            dl.rects.push(SceneRect::rounded(
                btn,
                if active { chrome.active_tab_bg } else { chrome.sidebar_bg },
                4.0,
            ));
            self.push_ui_text(
                dl,
                label,
                btn.x + 14.0,
                btn.y + 3.0,
                11.5,
                if active { 600 } else { 400 },
                if active { chrome.text } else { chrome.muted_text },
                Some(btn.w - 8.0),
                Some(sb),
            );
            self.ui_hit_zones.push(UiHitZone {
                rect: btn,
                action: UiAction::SwitchSidebarSection(sec),
                pointer_cursor: true,
            });
        }

        let list_clip = Rect::new(sb.x, sb.y + header_h, sb.w - 1.0, (sb.h - header_h).max(10.0));
        let mut row_y = sb.y + header_h + 6.0;
        let row_h = 24.0;

        match self.sidebar_section {
            SidebarSection::Outline => {
                let outline = self
                    .cached_meta
                    .as_ref()
                    .map(|m| m.outline.clone())
                    .unwrap_or_default();
                let caret = self
                    .workbench
                    .active_tab()
                    .map(|t| t.view.selection.head)
                    .unwrap_or(0);
                let active_idx = active_outline_index(&outline, caret);

                if outline.is_empty() {
                    self.push_ui_text(
                        dl,
                        "No headings",
                        sb.x + 14.0,
                        row_y + 4.0,
                        12.0,
                        400,
                        chrome.muted_text,
                        Some(sb.w - 24.0),
                        Some(list_clip),
                    );
                    return;
                }

                for (i, item) in outline.iter().enumerate() {
                    if row_y + row_h > sb.y + sb.h {
                        break;
                    }
                    let row_rect = Rect::new(sb.x + 6.0, row_y, sb.w - 13.0, row_h);
                    let is_cur = active_idx == Some(i);
                    if is_cur {
                        dl.rects.push(
                            SceneRect::rounded(row_rect, chrome.accent.with_alpha(38), 4.0)
                                .with_clip(Some(list_clip)),
                        );
                    }
                    let indent = ((item.level.saturating_sub(1) as f32) * 12.0).min(60.0);
                    let col = if is_cur { chrome.accent } else { chrome.text };
                    self.push_ui_text(
                        dl,
                        &item.title,
                        row_rect.x + 8.0 + indent,
                        row_y + 4.0,
                        12.0,
                        if item.level <= 2 { 600 } else { 400 },
                        col,
                        Some((row_rect.w - 14.0 - indent).max(30.0)),
                        Some(list_clip),
                    );
                    self.ui_hit_zones.push(UiHitZone {
                        rect: row_rect,
                        action: UiAction::JumpOutline(i),
                        pointer_cursor: true,
                    });
                    row_y += row_h + 2.0;
                }
            }
            SidebarSection::Files => {
                let Some(tree) = self.workbench.file_tree.clone() else {
                    self.push_ui_text(
                        dl,
                        "Pass a folder arg to browse files",
                        sb.x + 14.0,
                        row_y + 4.0,
                        11.5,
                        400,
                        chrome.muted_text,
                        Some(sb.w - 24.0),
                        Some(list_clip),
                    );
                    return;
                };
                let mut rows = Vec::new();
                flatten_file_tree(&tree, 0, &self.collapsed_dirs, &mut rows);
                let active_path = self
                    .workbench
                    .active_tab()
                    .and_then(|t| t.session.path.as_ref())
                    .cloned();

                for row in rows {
                    if row_y + row_h > sb.y + sb.h {
                        break;
                    }
                    let row_rect = Rect::new(sb.x + 6.0, row_y, sb.w - 13.0, row_h);
                    let is_open = active_path.as_ref() == Some(&row.path);
                    if is_open {
                        dl.rects.push(
                            SceneRect::rounded(row_rect, chrome.accent.with_alpha(38), 4.0)
                                .with_clip(Some(list_clip)),
                        );
                    }
                    let prefix = if row.is_dir {
                        if self.collapsed_dirs.contains(&row.path) {
                            "▸ "
                        } else {
                            "▾ "
                        }
                    } else {
                        "  "
                    };
                    let label = format!("{prefix}{}", row.name);
                    let indent = (row.depth as f32 * 12.0).min(60.0);
                    self.push_ui_text(
                        dl,
                        &label,
                        row_rect.x + 8.0 + indent,
                        row_y + 4.0,
                        12.0,
                        if row.is_dir || is_open { 600 } else { 400 },
                        if is_open { chrome.accent } else { chrome.text },
                        Some((row_rect.w - 12.0 - indent).max(30.0)),
                        Some(list_clip),
                    );
                    self.ui_hit_zones.push(UiHitZone {
                        rect: row_rect,
                        action: UiAction::OpenWorkspaceFile(row.path),
                        pointer_cursor: true,
                    });
                    row_y += row_h + 2.0;
                }
            }
        }
    }

    fn build_find_bar(&mut self, dl: &mut DrawList, editor_rect: Rect) {
        let chrome = self.style.chrome;
        let bar_w = 420.0_f32.min((editor_rect.w - 24.0).max(240.0));
        let bar = Rect::new(
            editor_rect.x + editor_rect.w - bar_w - 16.0,
            editor_rect.y + 8.0,
            bar_w,
            FIND_BAR_H,
        );
        dl.rects.push(
            SceneRect::rounded(bar, chrome.panel_bg, 7.0)
                .with_border(1.0, chrome.border),
        );

        let input_w = (bar.w - 210.0).max(90.0);
        let input_rect = Rect::new(bar.x + 8.0, bar.y + 6.0, input_w, 26.0);
        dl.rects.push(
            SceneRect::rounded(input_rect, chrome.active_tab_bg, 4.0).with_border(
                1.0,
                if self.find_focused {
                    chrome.accent
                } else {
                    chrome.border
                },
            ),
        );
        let q_display = if self.find_query.is_empty() {
            "Find...".to_string()
        } else {
            self.find_query.clone()
        };
        let q_w = self.push_ui_text(
            dl,
            &q_display,
            input_rect.x + 7.0,
            input_rect.y + 5.0,
            12.0,
            400,
            if self.find_query.is_empty() {
                chrome.muted_text
            } else {
                chrome.text
            },
            Some(input_w - 12.0),
            Some(input_rect),
        );
        if self.find_focused && self.caret_visible {
            let cx = input_rect.x + 7.0 + if self.find_query.is_empty() { 0.0 } else { q_w };
            dl.rects.push(SceneRect::solid(
                Rect::new(cx.min(input_rect.x + input_rect.w - 4.0), input_rect.y + 5.0, 1.5, 16.0),
                chrome.accent,
            ));
        }
        self.ui_hit_zones.push(UiHitZone {
            rect: input_rect,
            action: UiAction::FocusFindInput,
            pointer_cursor: false,
        });

        let total_matches = self
            .cached_meta
            .as_ref()
            .map(|m| m.find_matches.len())
            .unwrap_or(0);
        let cur_idx = self.active_find_range.and_then(|active| {
            self.cached_meta
                .as_ref()
                .and_then(|m| m.find_matches.iter().position(|fm| fm.range == active))
                .map(|i| i + 1)
        });
        let count_text = match (cur_idx, total_matches) {
            (_, 0) if !self.find_query.is_empty() => "0/0".to_string(),
            (Some(c), t) => format!("{c}/{t}"),
            (None, t) if t > 0 => format!("{t}"),
            _ => String::new(),
        };
        let count_x = input_rect.x + input_w + 6.0;
        self.push_ui_text(dl, &count_text, count_x, bar.y + 11.0, 11.0, 500, chrome.muted_text, Some(42.0), Some(bar));

        let mut bx = bar.x + bar.w - 148.0;
        for (label, bw, active, action) in [
            ("Aa", 26.0, self.find_options.case_sensitive, UiAction::ToggleFindCase),
            (".*", 26.0, self.find_options.use_regex, UiAction::ToggleFindRegex),
            ("<", 24.0, false, UiAction::FindPrev),
            (">", 24.0, false, UiAction::FindNext),
            ("×", 24.0, false, UiAction::CloseFind),
        ] {
            let r = Rect::new(bx, bar.y + 7.0, bw, 24.0);
            dl.rects.push(
                SceneRect::rounded(
                    r,
                    if active { chrome.accent.with_alpha(48) } else { chrome.tab_bar_bg },
                    4.0,
                )
                .with_border(1.0, if active { chrome.accent } else { chrome.border }),
            );
            self.push_ui_text(dl, label, r.x + 6.0, r.y + 4.0, 11.5, 600, chrome.text, None, Some(bar));
            self.ui_hit_zones.push(UiHitZone {
                rect: r,
                action,
                pointer_cursor: true,
            });
            bx += bw + 4.0;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn push_action_button(
        &mut self,
        dl: &mut DrawList,
        right_edge: f32,
        y: f32,
        h: f32,
        label: &str,
        bg: vellmark_render::RgbaColor,
        border: vellmark_render::RgbaColor,
        fg: vellmark_render::RgbaColor,
        action: UiAction,
        clip: Rect,
    ) -> f32 {
        let bw = (label.chars().count() as f32 * 6.4 + 14.0).max(24.0);
        let x = right_edge - bw;
        if x < clip.x + 8.0 {
            return right_edge;
        }
        let rect = Rect::new(x, y, bw, h);
        dl.rects
            .push(SceneRect::rounded(rect, bg, 5.0).with_border(1.0, border));
        self.push_ui_text(
            dl,
            label,
            rect.x + 7.0,
            rect.y + ((h - 14.0) * 0.5).max(1.0),
            11.0,
            600,
            fg,
            Some((bw - 10.0).max(10.0)),
            Some(rect),
        );
        self.ui_hit_zones.push(UiHitZone {
            rect,
            action,
            pointer_cursor: true,
        });
        x - 6.0
    }

    fn build_doc_alert_banner(&mut self, dl: &mut DrawList, banner: Rect, kind: &DocBannerKind) {
        let chrome = self.style.chrome;
        let diff_label = if self.conflict_diff_open {
            "Hide Diff"
        } else {
            "Diff"
        };
        let (accent, tint_alpha, message, buttons): (
            vellmark_render::RgbaColor,
            u8,
            String,
            Vec<(&str, UiAction, vellmark_render::RgbaColor, vellmark_render::RgbaColor, vellmark_render::RgbaColor)>,
        ) = match kind {
            DocBannerKind::Conflict { theirs_updated } => (
                chrome.danger,
                26,
                if *theirs_updated {
                    "External conflict (disk changed again) — compare Diff or choose Merge / Keep Mine / Use Disk"
                        .to_string()
                } else {
                    "External file conflict detected — local buffer differs from file on disk"
                        .to_string()
                },
                vec![
                    ("Save As...", UiAction::SaveAsActive, chrome.panel_bg, chrome.border, chrome.text),
                    ("Use Disk", UiAction::ResolveConflictUseTheirs, chrome.panel_bg, chrome.border, chrome.text),
                    ("Keep Mine", UiAction::ResolveConflictKeepMine, chrome.danger.with_alpha(38), chrome.danger, chrome.text),
                    ("Auto Merge", UiAction::ResolveConflictAutoMerge, chrome.accent.with_alpha(38), chrome.accent, chrome.text),
                    (
                        diff_label,
                        UiAction::ToggleConflictDiff,
                        if self.conflict_diff_open { chrome.accent.with_alpha(55) } else { chrome.panel_bg },
                        if self.conflict_diff_open { chrome.accent } else { chrome.border },
                        chrome.text,
                    ),
                ],
            ),
            DocBannerKind::Missing {
                has_local_edits,
                write_error,
            } => (
                chrome.danger,
                24,
                if let Some(err) = write_error {
                    format!("File missing on disk ({err}) — Re-save, Save As..., or Close Tab")
                } else if *has_local_edits {
                    "File deleted or moved externally (unsaved local edits kept)".to_string()
                } else {
                    "File deleted or moved externally".to_string()
                },
                vec![
                    ("Close Tab", UiAction::ForceCloseActiveMissing, chrome.panel_bg, chrome.border, chrome.text),
                    ("Save As...", UiAction::SaveAsActive, chrome.panel_bg, chrome.border, chrome.text),
                    ("Re-save", UiAction::ResaveMissingFile, chrome.danger.with_alpha(38), chrome.danger, chrome.text),
                ],
            ),
            DocBannerKind::WriteError { error } => (
                chrome.danger,
                24,
                format!("Auto-save write failed: {error}"),
                vec![
                    ("Save As...", UiAction::SaveAsActive, chrome.panel_bg, chrome.border, chrome.text),
                    ("Retry", UiAction::RetrySaveActive, chrome.danger.with_alpha(38), chrome.danger, chrome.text),
                ],
            ),
            DocBannerKind::RecoveryDegraded => (
                chrome.warning,
                24,
                RECOVERY_DEGRADED_NOTICE.to_string(),
                vec![
                    ("Save As...", UiAction::SaveAsActive, chrome.panel_bg, chrome.border, chrome.text),
                    ("Save Now", UiAction::SaveActiveNow, chrome.warning.with_alpha(38), chrome.warning, chrome.text),
                ],
            ),
            DocBannerKind::ConflictBackupCleanup { count } => (
                chrome.warning,
                18,
                format!("{count} conflict backup copy(s) stored in history for this file"),
                vec![
                    ("×", UiAction::DismissConflictBackupNotice, chrome.panel_bg, chrome.border, chrome.muted_text),
                    ("Clean Backups", UiAction::ClearConflictBackups, chrome.warning.with_alpha(34), chrome.warning, chrome.text),
                ],
            ),
        };

        dl.rects.push(SceneRect::solid(banner, chrome.panel_bg));
        dl.rects
            .push(SceneRect::solid(banner, accent.with_alpha(tint_alpha)));
        dl.rects.push(SceneRect::solid(
            Rect::new(banner.x, banner.y, 4.0, banner.h),
            accent,
        ));
        dl.rects.push(SceneRect::solid(
            Rect::new(banner.x, banner.y + banner.h - 1.0, banner.w, 1.0),
            chrome.border,
        ));

        let btn_h = (banner.h - 12.0).clamp(20.0, 24.0);
        let btn_y = banner.y + (banner.h - btn_h) * 0.5;
        let mut right_edge = banner.x + banner.w - 10.0;
        for (label, action, bg, border, fg) in buttons {
            right_edge = self.push_action_button(
                dl,
                right_edge,
                btn_y,
                btn_h,
                label,
                bg,
                border,
                fg,
                action,
                banner,
            );
        }

        let text_max_w = (right_edge - (banner.x + 14.0)).max(40.0);
        self.push_ui_text(
            dl,
            &message,
            banner.x + 12.0,
            banner.y + ((banner.h - 15.0) * 0.5).max(3.0),
            11.5,
            600,
            chrome.text,
            Some(text_max_w),
            Some(banner),
        );
    }

    fn build_conflict_diff_panel(&mut self, dl: &mut DrawList, diff_rect: Rect) {
        let chrome = self.style.chrome;
        dl.rects.push(SceneRect::solid(diff_rect, chrome.tab_bar_bg));
        dl.rects.push(SceneRect::solid(
            Rect::new(diff_rect.x, diff_rect.y + diff_rect.h - 1.0, diff_rect.w, 1.0),
            chrome.border,
        ));

        let header_h = 24.0;
        dl.rects
            .push(SceneRect::solid(Rect::new(diff_rect.x, diff_rect.y, diff_rect.w, header_h), chrome.panel_bg));
        dl.rects.push(SceneRect::solid(
            Rect::new(diff_rect.x, diff_rect.y + header_h - 1.0, diff_rect.w, 1.0),
            chrome.border,
        ));
        self.push_ui_text(
            dl,
            "Conflict Diff (Mine vs Disk)",
            diff_rect.x + 12.0,
            diff_rect.y + 4.0,
            11.5,
            600,
            chrome.text,
            Some((diff_rect.w - 56.0).max(40.0)),
            Some(diff_rect),
        );
        self.push_action_button(
            dl,
            diff_rect.x + diff_rect.w - 8.0,
            diff_rect.y + 2.0,
            20.0,
            "×",
            chrome.panel_bg,
            chrome.border,
            chrome.muted_text,
            UiAction::ToggleConflictDiff,
            diff_rect,
        );

        let lines = self
            .workbench
            .active_tab()
            .map(|t| t.session.compute_conflict_diff_lines())
            .unwrap_or_default();

        let body_y = diff_rect.y + header_h + 4.0;
        let body_clip = Rect::new(
            diff_rect.x,
            diff_rect.y + header_h,
            diff_rect.w,
            (diff_rect.h - header_h).max(0.0),
        );
        if lines.is_empty() {
            self.push_ui_text(
                dl,
                "No text differences detected between local buffer and disk snapshot.",
                diff_rect.x + 12.0,
                body_y,
                11.5,
                400,
                chrome.muted_text,
                Some((diff_rect.w - 24.0).max(40.0)),
                Some(body_clip),
            );
            return;
        }

        let line_h = 17.0;
        let max_lines = ((diff_rect.y + diff_rect.h - body_y - 4.0) / line_h)
            .floor()
            .max(0.0) as usize;
        if max_lines == 0 {
            return;
        }
        let truncated = lines.len() > max_lines && max_lines > 1;
        let visible_count = if truncated { max_lines - 1 } else { max_lines.min(lines.len()) };

        for (i, line) in lines.iter().take(visible_count).enumerate() {
            let y = body_y + i as f32 * line_h;
            let (fg, bg_opt, weight) = match line.kind {
                ConflictDiffKind::SectionHeader => (chrome.accent, Some(chrome.accent.with_alpha(18)), 600),
                ConflictDiffKind::Context => (chrome.muted_text, None, 400),
                ConflictDiffKind::Mine => (chrome.warning, Some(chrome.warning.with_alpha(24)), 500),
                ConflictDiffKind::Theirs => (chrome.success, Some(chrome.success.with_alpha(22)), 500),
            };
            if let Some(bg) = bg_opt {
                dl.rects.push(SceneRect::solid(
                    Rect::new(diff_rect.x + 8.0, y, (diff_rect.w - 16.0).max(20.0), line_h),
                    bg,
                ));
            }
            self.push_ui_text(
                dl,
                &line.text,
                diff_rect.x + 14.0,
                y + 1.0,
                11.5,
                weight,
                fg,
                Some((diff_rect.w - 28.0).max(40.0)),
                Some(body_clip),
            );
        }
        if truncated {
            let y = body_y + visible_count as f32 * line_h;
            let omitted = lines.len() - visible_count;
            let footer = format!("... ({omitted} more diff lines)");
            self.push_ui_text(
                dl,
                &footer,
                diff_rect.x + 14.0,
                y + 1.0,
                11.0,
                500,
                chrome.muted_text,
                Some((diff_rect.w - 28.0).max(40.0)),
                Some(body_clip),
            );
        }
    }

    fn build_status_bar(&mut self, dl: &mut DrawList, sb: Rect) {
        let chrome = self.style.chrome;
        dl.rects.push(SceneRect::solid(sb, chrome.status_bar_bg));
        dl.rects.push(SceneRect::solid(
            Rect::new(sb.x, sb.y, sb.w, 1.0),
            chrome.border,
        ));

        let (state_text, state_col, eol_text, ime_badge) = match self.workbench.active_tab() {
            Some(tab) => {
                let (st, col) = match tab.session.state {
                    DocState::Clean => ("Clean", chrome.success),
                    DocState::Dirty => ("Modified", chrome.warning),
                    DocState::Conflict => ("Conflict", chrome.danger),
                    DocState::Missing { .. } => ("Missing", chrome.danger),
                };
                let ro = if matches!(tab.session.format.decision, EncodingDecision::ReadOnly { .. }) {
                    " [Read-Only]"
                } else {
                    ""
                };
                let eol = match tab.session.format.dominant_line_ending {
                    vellmark_core::LineEnding::Lf => "UTF-8 • LF",
                    vellmark_core::LineEnding::CrLf => "UTF-8 • CRLF",
                    vellmark_core::LineEnding::Cr => "UTF-8 • CR",
                };
                let ime = if tab.view.ime_preedit.is_some() {
                    "IME: Composing"
                } else if self.ime_enabled {
                    "IME: Ready"
                } else {
                    "IME: Idle"
                };
                (format!("{st}{ro}"), col, eol, ime)
            }
            None => ("No Document".to_string(), chrome.muted_text, "", ""),
        };

        let active_notice = self.active_status_notice();

        dl.rects.push(SceneRect::rounded(
            Rect::new(10.0, sb.y + 8.0, 8.0, 8.0),
            state_col,
            4.0,
        ));
        let left_str = match active_notice {
            Some(note) => format!("{state_text}   |   {note}"),
            None => state_text,
        };
        self.push_ui_text(
            dl,
            &left_str,
            24.0,
            sb.y + 5.0,
            11.0,
            500,
            chrome.text,
            Some((sb.w * 0.45).max(120.0)),
            Some(sb),
        );

        let right_stats = if let Some(m) = &self.cached_meta {
            format!(
                "{} words • {} chars • {} lines   |   {}   |   {}%   |   {}",
                m.stats.word_count,
                m.stats.char_count,
                m.stats.line_count,
                eol_text,
                self.workbench.config.zoom_percent,
                ime_badge,
            )
        } else {
            format!("{}%", self.workbench.config.zoom_percent)
        };
        let rx = (sb.w - right_stats.len() as f32 * 6.4 - 18.0).max(sb.w * 0.48);
        self.push_ui_text(
            dl,
            &right_stats,
            rx,
            sb.y + 5.0,
            11.0,
            400,
            chrome.muted_text,
            Some((sb.w - rx - 10.0).max(80.0)),
            Some(sb),
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn push_ui_text(
        &mut self,
        dl: &mut DrawList,
        text: &str,
        x: f32,
        y: f32,
        font_size: f32,
        weight: u16,
        color: vellmark_render::RgbaColor,
        max_w: Option<f32>,
        clip: Option<Rect>,
    ) -> f32 {
        shape_ui_text_into(
            self.shaper.font_system_mut(),
            text,
            x,
            y,
            font_size,
            font_size * 1.45,
            weight,
            color,
            max_w,
            clip,
            self.scale_factor,
            &mut dl.glyphs,
        )
    }

    fn cancel_abandoned_tab_ime(&mut self, active_tab_id: Option<u64>) {
        if let ImeTarget::Tab(ime_tab_id) = self.ime_target {
            if active_tab_id != Some(ime_tab_id) || (self.find_open && self.find_focused) {
                if let Some(tab) = self.workbench.tabs.iter_mut().find(|t| t.id == ime_tab_id) {
                    let _ = tab.edit(|ctx| ctx.ime_cancel());
                }
                self.ime_target = ImeTarget::None;
            }
        }
    }

    /// Handle OS IME lifecycle event (`Enabled`, `Preedit`, `Commit`, `Disabled`).
    pub fn handle_ime_event(&mut self, ime: &Ime) {
        self.reset_blink();
        match ime {
            Ime::Enabled => {
                self.ime_enabled = true;
            }
            Ime::Preedit(text, _) => {
                self.ime_enabled = true;
                self.close_confirm_armed = false;
                if text.is_empty() {
                    // Do NOT treat empty preedit (`Preedit("", None)`) as `ime_cancel()`, because
                    // winit emits it immediately before `Ime::Commit` (especially on Cocoa insertText).
                    // Clearing only the visible preedit preserves `ime_composition_selection` and `ime_target`.
                    match self.ime_target {
                        ImeTarget::FindBox => {}
                        ImeTarget::Tab(target_id) => {
                            if let Some(tab) =
                                self.workbench.tabs.iter_mut().find(|t| t.id == target_id)
                            {
                                let _ = tab.edit(|ctx| ctx.ime_clear_preedit());
                            }
                        }
                        ImeTarget::None => {
                            if !(self.find_open && self.find_focused) {
                                if let Some(tab) = self.workbench.active_tab_mut() {
                                    let _ = tab.edit(|ctx| ctx.ime_clear_preedit());
                                }
                            }
                        }
                    }
                    return;
                }

                let target = match self.ime_target {
                    ImeTarget::Tab(id)
                        if self.workbench.tabs.iter().any(|tab| tab.id == id) =>
                    {
                        ImeTarget::Tab(id)
                    }
                    ImeTarget::FindBox if self.find_open => ImeTarget::FindBox,
                    _ if self.find_open && self.find_focused => ImeTarget::FindBox,
                    _ => match self.workbench.active_tab() {
                        Some(tab) => ImeTarget::Tab(tab.id),
                        None => ImeTarget::None,
                    },
                };
                self.ime_target = target;

                if let ImeTarget::Tab(id) = target {
                    if let Some(tab) = self.workbench.tabs.iter_mut().find(|t| t.id == id) {
                        let _ = tab.edit(|ctx| ctx.ime_update(text));
                    }
                    if self.workbench.active_tab().map(|tab| tab.id) == Some(id) {
                        self.pending_scroll_ensure_caret = true;
                    }
                }
            }
            Ime::Commit(committed) => {
                self.close_confirm_armed = false;
                let target = self.ime_target;
                self.ime_target = ImeTarget::None;

                match target {
                    ImeTarget::FindBox => {
                        self.find_query.push_str(committed);
                        self.run_find_next(false);
                    }
                    ImeTarget::Tab(target_id) => {
                        let active_id = self.workbench.active_tab().map(|t| t.id);
                        if let Some(tab) =
                            self.workbench.tabs.iter_mut().find(|t| t.id == target_id)
                        {
                            let _ = tab.edit(|ctx| ctx.ime_commit(committed));
                        }
                        if active_id == Some(target_id) {
                            self.preferred_caret_x = None;
                            self.pending_scroll_ensure_caret = true;
                        }
                    }
                    ImeTarget::None => {
                        if self.find_open && self.find_focused {
                            self.find_query.push_str(committed);
                            self.run_find_next(false);
                            return;
                        }
                        if let Some(tab) = self.workbench.active_tab_mut() {
                            let _ = tab.edit(|ctx| ctx.ime_commit(committed));
                        }
                        self.preferred_caret_x = None;
                        self.pending_scroll_ensure_caret = true;
                    }
                }
            }
            Ime::Disabled => {
                self.ime_enabled = false;
                let prev = self.ime_target;
                self.ime_target = ImeTarget::None;
                if let ImeTarget::Tab(target_id) = prev {
                    if let Some(tab) = self.workbench.tabs.iter_mut().find(|t| t.id == target_id) {
                        let _ = tab.edit(|ctx| ctx.ime_cancel());
                    }
                }
                if let Some(tab) = self.workbench.active_tab_mut() {
                    let _ = tab.edit(|ctx| ctx.ime_cancel());
                }
            }
        }
    }

    /// Close guard for window shutdown: auto-saves eligible dirty named files, flushes recovery
    /// snapshots synchronously, blocks first close when any tab remains unsaved/conflicted/missing,
    /// archives conflicted/missing unsaved tabs or discards untitled drafts on second confirmation.
    pub fn handle_close_requested(&mut self) -> bool {
        if !self.close_confirm_armed {
            let auto_events = self.workbench.flush_all_dirty_auto_saves(Instant::now());
            let _ = self.apply_runtime_events(auto_events);
        }
        self.workbench.flush_all_recoveries();
        let first_unsaved = self
            .workbench
            .tabs
            .iter()
            .position(|t| t.session.state != DocState::Clean);
        let Some(idx) = first_unsaved else {
            self.workbench.maintain_recovery_storage();
            return true;
        };
        if !self.close_confirm_armed {
            self.workbench.active_tab_idx = Some(idx);
            self.close_confirm_armed = true;
            self.find_focused = false;
            let state = self.workbench.tabs[idx].session.state;
            let title = self.workbench.tabs[idx].title();
            match state {
                DocState::Conflict => {
                    self.conflict_diff_open = true;
                    self.set_notice(format!(
                        "External conflict in '{title}' — resolve or close again to archive backup & exit"
                    ));
                }
                DocState::Missing {
                    has_local_edits: true,
                } => {
                    self.set_notice(format!(
                        "Missing file '{title}' has unsaved changes — Save As or close again to backup & exit"
                    ));
                }
                _ => {
                    self.set_notice(format!(
                        "Unsaved changes in '{title}' — Save (Cmd/Ctrl+S) or close again to discard"
                    ));
                }
            }
            return false;
        }
        for i in 0..self.workbench.tabs.len() {
            let state = self.workbench.tabs[i].session.state;
            let title = self.workbench.tabs[i].title();
            match state {
                DocState::Conflict
                | DocState::Missing {
                    has_local_edits: true,
                } => {
                    if let Err(err) = self.workbench.tabs[i].session.archive_unsaved_on_close() {
                        self.workbench.active_tab_idx = Some(i);
                        self.set_notice(format!(
                            "Backup failed for '{title}' ({err}) — please Save As first"
                        ));
                        return false;
                    }
                }
                _ => {
                    self.workbench.tabs[i].session.discard_recovery_snapshot();
                }
            }
        }
        self.workbench.maintain_recovery_storage();
        true
    }

    /// Handle OS keyboard press (`primary_mod` = `Cmd` on macOS, `Ctrl` on Win/Linux).
    pub fn handle_key_press(&mut self, event: &KeyEvent, mods: ModifiersState) -> bool {
        if event.state != ElementState::Pressed {
            return false;
        }
        self.reset_blink();

        let primary = if cfg!(target_os = "macos") {
            mods.super_key()
        } else {
            mods.control_key()
        };
        let shift = mods.shift_key();
        let alt = mods.alt_key();

        // When OS IME composition is active, let IME handle printable input; only allow Escape to cancel safely.
        let ime_composing = self
            .workbench
            .active_tab()
            .is_some_and(|t| t.view.ime_preedit.is_some());
        if ime_composing {
            if event.logical_key == NamedKey::Escape {
                if let Some(tab) = self.workbench.active_tab_mut() {
                    let _ = tab.edit(|ctx| ctx.ime_cancel());
                }
                self.ime_target = ImeTarget::None;
                return true;
            }
            return false;
        }
        let active_id = self.workbench.active_tab().map(|t| t.id);
        self.cancel_abandoned_tab_ime(active_id);

        // Primary shortcuts
        if primary && !alt {
            if let Key::Character(ch) = &event.logical_key {
                let lower = ch.to_ascii_lowercase();
                match lower.as_str() {
                    "s" if shift => {
                        self.save_as_active_tab();
                        return true;
                    }
                    "s" => {
                        self.save_active_tab();
                        return true;
                    }
                    "n" => {
                        self.close_confirm_armed = false;
                        self.flush_active_tab_before_switch();
                        self.workbench.new_document();
                        self.find_focused = false;
                        self.pending_scroll_ensure_caret = true;
                        self.set_notice("New Document");
                        return true;
                    }
                    "w" => {
                        self.close_confirm_armed = false;
                        if let Some(id) = self.workbench.active_tab().map(|t| t.id) {
                            self.close_tab_checked(id, "Unsaved changes — save first or keep open");
                        }
                        return true;
                    }
                    "z" => {
                        self.close_confirm_armed = false;
                        if let Some(tab) = self.workbench.active_tab_mut() {
                            let act = if shift {
                                EditorAction::Redo
                            } else {
                                EditorAction::Undo
                            };
                            let _ = tab.dispatch(act);
                            self.pending_scroll_ensure_caret = true;
                        }
                        return true;
                    }
                    "b" => {
                        self.dispatch_editor(EditorAction::ToggleBold);
                        return true;
                    }
                    "i" => {
                        self.dispatch_editor(EditorAction::ToggleItalic);
                        return true;
                    }
                    "e" => {
                        self.dispatch_editor(EditorAction::ToggleInlineCode);
                        return true;
                    }
                    "x" if shift => {
                        self.dispatch_editor(EditorAction::ToggleStrikethrough);
                        return true;
                    }
                    "x" => {
                        self.copy_selection(true);
                        return true;
                    }
                    "c" => {
                        self.copy_selection(false);
                        return true;
                    }
                    "v" => {
                        self.paste_clipboard();
                        return true;
                    }
                    "a" => {
                        if !self.find_focused {
                            self.dispatch_editor(EditorAction::SelectAll);
                        }
                        return true;
                    }
                    "f" if shift => {
                        self.workbench.config.focus_mode = !self.workbench.config.focus_mode;
                        let on = self.workbench.config.focus_mode;
                        self.set_notice(if on { "Focus Mode: ON" } else { "Focus Mode: OFF" });
                        return true;
                    }
                    "f" => {
                        self.find_open = true;
                        self.find_focused = true;
                        return true;
                    }
                    "/" => {
                        self.dispatch_editor(EditorAction::ToggleSourceMode);
                        self.preferred_caret_x = None;
                        return true;
                    }
                    "\\" => {
                        self.workbench.config.sidebar_visible = !self.workbench.config.sidebar_visible;
                        return true;
                    }
                    "t" if shift => {
                        self.cycle_theme();
                        return true;
                    }
                    "=" | "+" => {
                        self.workbench.config.zoom_in();
                        self.set_notice(format!("Zoom {}%", self.workbench.config.zoom_percent));
                        return true;
                    }
                    "-" => {
                        self.workbench.config.zoom_out();
                        self.set_notice(format!("Zoom {}%", self.workbench.config.zoom_percent));
                        return true;
                    }
                    "0" => {
                        self.workbench.config.reset_zoom();
                        self.set_notice("Zoom 100%");
                        return true;
                    }
                    "1" | "2" | "3" | "4" | "5" | "6" => {
                        let lvl = lower.parse::<u8>().unwrap_or(1);
                        self.dispatch_editor(EditorAction::SetHeadingLevel(lvl));
                        return true;
                    }
                    _ => {}
                }
            }
        }

        // Find overlay focused handling
        if self.find_open && self.find_focused {
            match &event.logical_key {
                Key::Named(NamedKey::Escape) => {
                    self.find_open = false;
                    self.find_focused = false;
                    self.active_find_range = None;
                    return true;
                }
                Key::Named(NamedKey::Enter) => {
                    if shift {
                        self.run_find_prev();
                    } else {
                        self.run_find_next(true);
                    }
                    return true;
                }
                Key::Named(NamedKey::Backspace) => {
                    self.find_query.pop();
                    self.run_find_next(false);
                    return true;
                }
                _ => {
                    if !primary && !mods.control_key() {
                        if let Some(txt) = &event.text {
                            if !txt.chars().all(|c| c.is_control()) {
                                self.find_query.push_str(txt);
                                self.run_find_next(false);
                                return true;
                            }
                        }
                    }
                }
            }
        }

        // Standard editor keys
        match &event.logical_key {
            Key::Named(NamedKey::Escape) => {
                if self.close_confirm_armed {
                    self.close_confirm_armed = false;
                    self.set_notice("Close cancelled");
                    return true;
                }
                if self.find_open {
                    self.find_open = false;
                    self.find_focused = false;
                    self.active_find_range = None;
                    return true;
                }
                false
            }
            Key::Named(NamedKey::Backspace) => {
                self.preferred_caret_x = None;
                self.dispatch_editor(EditorAction::Backspace);
                true
            }
            Key::Named(NamedKey::Delete) => {
                self.preferred_caret_x = None;
                self.delete_forward();
                true
            }
            Key::Named(NamedKey::Enter) => {
                self.preferred_caret_x = None;
                self.dispatch_editor(EditorAction::Enter);
                true
            }
            Key::Named(NamedKey::Tab) => {
                self.preferred_caret_x = None;
                self.dispatch_editor(if shift {
                    EditorAction::ShiftTab
                } else {
                    EditorAction::Tab
                });
                true
            }
            Key::Named(NamedKey::ArrowLeft) => {
                self.preferred_caret_x = None;
                self.move_caret_horizontal(-1, shift);
                true
            }
            Key::Named(NamedKey::ArrowRight) => {
                self.preferred_caret_x = None;
                self.move_caret_horizontal(1, shift);
                true
            }
            Key::Named(NamedKey::ArrowUp) => {
                self.move_caret_vertical(-1, shift);
                true
            }
            Key::Named(NamedKey::ArrowDown) => {
                self.move_caret_vertical(1, shift);
                true
            }
            Key::Named(NamedKey::Home) => {
                self.preferred_caret_x = None;
                self.move_caret_line_edge(true, shift);
                true
            }
            Key::Named(NamedKey::End) => {
                self.preferred_caret_x = None;
                self.move_caret_line_edge(false, shift);
                true
            }
            Key::Named(NamedKey::PageUp) => {
                self.scroll_by(-self.last_editor_rect.h * 0.85);
                true
            }
            Key::Named(NamedKey::PageDown) => {
                self.scroll_by(self.last_editor_rect.h * 0.85);
                true
            }
            Key::Named(NamedKey::Space) if !primary && !mods.control_key() => {
                self.preferred_caret_x = None;
                self.dispatch_editor(EditorAction::InsertChar(' '));
                true
            }
            _ => {
                if !primary && !mods.control_key() {
                    if let Some(txt) = &event.text {
                        let clean: String = txt.chars().filter(|c| !c.is_control()).collect();
                        if !clean.is_empty() {
                            self.preferred_caret_x = None;
                            if clean.chars().count() == 1 {
                                self.dispatch_editor(EditorAction::InsertChar(
                                    clean.chars().next().expect("1 char"),
                                ));
                            } else {
                                self.dispatch_editor(EditorAction::InsertText(clean));
                            }
                            return true;
                        }
                    }
                }
                false
            }
        }
    }

    pub fn handle_pointer_down(&mut self, x: f32, y: f32) -> bool {
        self.reset_blink();
        self.pointer_pos = (x, y);

        // 1. Check chrome buttons/tabs/sidebar/find bar first
        if let Some(hit) = self
            .ui_hit_zones
            .iter()
            .rev()
            .find(|z| z.rect.contains(x, y))
            .cloned()
        {
            self.dragging_selection = false;
            return self.execute_ui_action(hit.action);
        }

        // 2. Check editor document viewport click
        if self.last_editor_rect.contains(x, y) {
            self.find_focused = false;
            self.close_confirm_armed = false;
            let active_id = self.workbench.active_tab().map(|t| t.id);
            self.cancel_abandoned_tab_ime(active_id);
            if let Some(target) = self.shaper.hit_test(x, y) {
                self.preferred_caret_x = None;
                match target {
                    EditorHitTarget::Text(offset) => {
                        if let Some(tab) = self.workbench.active_tab_mut() {
                            let safe_offset = tab.session.buffer.snap_out_of_crlf(offset);
                            let _ = tab.edit(|ctx| ctx.set_caret(safe_offset));
                            self.dragging_selection = true;
                        }
                        return true;
                    }
                    EditorHitTarget::TaskCheckbox { caret_offset, .. } => {
                        if let Some(tab) = self.workbench.active_tab_mut() {
                            let _ = tab.edit(|ctx| {
                                ctx.set_caret(caret_offset);
                                ctx.toggle_task_checkbox();
                            });
                        }
                        return true;
                    }
                }
            }
        }
        false
    }

    pub fn handle_pointer_move(&mut self, x: f32, y: f32) -> (CursorIcon, bool) {
        self.pointer_pos = (x, y);
        let mut changed = false;
        if self.dragging_selection {
            if let Some(EditorHitTarget::Text(offset)) = self.shaper.hit_test(x, y) {
                if let Some(tab) = self.workbench.active_tab_mut() {
                    let anchor = tab.session.buffer.snap_out_of_crlf(tab.view.selection.anchor);
                    let safe_offset = tab.session.buffer.snap_out_of_crlf(offset);
                    let _ = tab.edit(|ctx| ctx.set_selection(anchor, safe_offset));
                    changed = true;
                }
            }
        }
        let icon = if let Some(zone) = self.ui_hit_zones.iter().rev().find(|z| z.rect.contains(x, y)) {
            if zone.pointer_cursor {
                CursorIcon::Pointer
            } else {
                CursorIcon::Text
            }
        } else if self.last_editor_rect.contains(x, y) {
            CursorIcon::Text
        } else {
            CursorIcon::Default
        };
        (icon, changed)
    }

    pub fn handle_pointer_up(&mut self) {
        self.dragging_selection = false;
    }

    pub fn handle_wheel(&mut self, delta: MouseScrollDelta) -> bool {
        let dy = match delta {
            MouseScrollDelta::LineDelta(_, y) => -y * (self.style.base_line_height_px * 2.4),
            MouseScrollDelta::PixelDelta(pos) => -(pos.y as f32),
        };
        self.scroll_by(dy)
    }

    pub fn scroll_by(&mut self, delta_y: f32) -> bool {
        let max_scroll = (self.last_total_doc_height - self.last_editor_rect.h).max(0.0);
        if let Some(tab) = self.workbench.active_tab_mut() {
            let next = (tab.scroll_y + delta_y).clamp(0.0, max_scroll);
            if (next - tab.scroll_y).abs() > 0.1 {
                tab.scroll_y = next;
                return true;
            }
        }
        false
    }

    fn execute_ui_action(&mut self, action: UiAction) -> bool {
        self.close_confirm_armed = false;
        match action {
            UiAction::SelectTab(idx) => {
                if idx < self.workbench.tabs.len() {
                    if self.workbench.active_tab_idx != Some(idx) {
                        self.flush_active_tab_before_switch();
                    }
                    self.workbench.active_tab_idx = Some(idx);
                    self.find_focused = false;
                    self.preferred_caret_x = None;
                }
                true
            }
            UiAction::CloseTab(id) => {
                self.close_tab_checked(id, "Unsaved changes — press Save first");
                true
            }
            UiAction::NewTab => {
                self.flush_active_tab_before_switch();
                self.workbench.new_document();
                self.find_focused = false;
                true
            }
            UiAction::ToggleMode => {
                self.dispatch_editor(EditorAction::ToggleSourceMode);
                true
            }
            UiAction::ToggleFocusMode => {
                self.workbench.config.focus_mode = !self.workbench.config.focus_mode;
                true
            }
            UiAction::CycleTheme => {
                self.cycle_theme();
                true
            }
            UiAction::ToggleSidebar => {
                self.workbench.config.sidebar_visible = !self.workbench.config.sidebar_visible;
                true
            }
            UiAction::SwitchSidebarSection(sec) => {
                self.sidebar_section = sec;
                true
            }
            UiAction::JumpOutline(idx) => {
                let item_opt = self
                    .cached_meta
                    .as_ref()
                    .and_then(|m| m.outline.get(idx).cloned());
                if let Some(item) = item_opt {
                    if let Some(tab) = self.workbench.active_tab_mut() {
                        tab.jump_to_outline_item(&item);
                        self.pending_scroll_ensure_caret = true;
                        self.find_focused = false;
                    }
                }
                true
            }
            UiAction::OpenWorkspaceFile(path) => {
                if path.is_dir() {
                    if !self.collapsed_dirs.remove(&path) {
                        self.collapsed_dirs.insert(path);
                    }
                } else {
                    self.flush_active_tab_before_switch();
                    match self.workbench.open_document(&path) {
                        Ok(_) => self.find_focused = false,
                        Err(e) => self.set_notice(format!("Open failed: {e}")),
                    }
                }
                true
            }
            UiAction::ToggleFindCase => {
                self.find_options.case_sensitive = !self.find_options.case_sensitive;
                self.run_find_next(false);
                true
            }
            UiAction::ToggleFindRegex => {
                self.find_options.use_regex = !self.find_options.use_regex;
                self.run_find_next(false);
                true
            }
            UiAction::FindPrev => {
                self.run_find_prev();
                true
            }
            UiAction::FindNext => {
                self.run_find_next(true);
                true
            }
            UiAction::CloseFind => {
                self.find_open = false;
                self.find_focused = false;
                self.active_find_range = None;
                true
            }
            UiAction::FocusFindInput => {
                self.find_focused = true;
                true
            }
            UiAction::ToggleConflictDiff => {
                self.conflict_diff_open = !self.conflict_diff_open;
                true
            }
            UiAction::ResolveConflictKeepMine => {
                self.resolve_active_conflict_keep_mine();
                true
            }
            UiAction::ResolveConflictUseTheirs => {
                self.resolve_active_conflict_use_theirs();
                true
            }
            UiAction::ResolveConflictAutoMerge => {
                self.resolve_active_conflict_auto_merge();
                true
            }
            UiAction::SaveAsActive => {
                self.save_as_active_tab();
                true
            }
            UiAction::ResaveMissingFile | UiAction::RetrySaveActive | UiAction::SaveActiveNow => {
                self.save_active_tab();
                true
            }
            UiAction::ForceCloseActiveMissing => {
                if let Some(id) = self.workbench.active_tab().map(|t| t.id) {
                    match self.workbench.try_close_tab(id, true) {
                        Ok(true) => {
                            if self.ime_target == ImeTarget::Tab(id) {
                                self.ime_target = ImeTarget::None;
                            }
                            self.conflict_diff_open = false;
                            self.trigger_background_maintenance(true);
                        }
                        Ok(false) => {}
                        Err(e) => {
                            self.set_notice(format!("Cannot close tab: {e} — please Save As first"));
                        }
                    }
                }
                true
            }
            UiAction::ClearConflictBackups => {
                let path_opt = self
                    .workbench
                    .active_tab()
                    .and_then(|t| t.session.path.clone());
                let removed = self
                    .workbench
                    .clear_conflict_backups_for_path(path_opt.as_deref());
                if removed > 0 {
                    self.trigger_background_maintenance(true);
                    self.set_notice(format!("Removed {removed} conflict backup copy(s)"));
                } else {
                    self.set_notice("No conflict backup copies found to clean");
                }
                true
            }
            UiAction::DismissConflictBackupNotice => {
                if let Some(tab) = self.workbench.active_tab_mut() {
                    tab.conflict_backup_notice_dismissed = true;
                }
                true
            }
        }
    }

    fn cycle_theme(&mut self) {
        let all = BuiltinTheme::ALL;
        let current = self.workbench.config.theme;
        let idx = all.iter().position(|&t| t == current).unwrap_or(0);
        let next = all[(idx + 1) % all.len()];
        self.workbench.config.follow_system_theme = false;
        self.workbench.config.theme = next;
        self.workbench.theme = next.load();
        self.refresh_style();
        self.set_notice(format!("Theme: {next:?}"));
    }

    fn close_tab_checked(&mut self, id: u64, unsaved_msg: &str) {
        if self.workbench.close_tab(id, false) {
            if self.ime_target == ImeTarget::Tab(id) {
                self.ime_target = ImeTarget::None;
            }
            self.conflict_diff_open = false;
            self.trigger_background_maintenance(true);
        } else {
            self.set_notice(unsaved_msg);
        }
    }

    fn handle_conflict_action_outcome(
        &mut self,
        tab_id: u64,
        outcome: ConflictActionOutcome,
        resolved_msg: &str,
    ) {
        match outcome {
            ConflictActionOutcome::ResolvedClean | ConflictActionOutcome::ResolvedDirty => {
                self.conflict_diff_open = false;
                self.workbench.refresh_tab_conflict_backup_count(tab_id);
                self.set_notice(resolved_msg);
                self.trigger_background_maintenance(true);
            }
            ConflictActionOutcome::TransitionedToMissing => {
                self.conflict_diff_open = false;
                self.set_notice("Disk file missing — choose Re-save, Save As..., or Close Tab");
            }
            ConflictActionOutcome::AbortedStillConflict(msg) => {
                self.conflict_diff_open = true;
                self.set_notice(format!("Conflict not resolved: {msg}"));
            }
        }
    }

    fn resolve_active_conflict_keep_mine(&mut self) {
        self.close_confirm_armed = false;
        let Some(tab) = self.workbench.active_tab_mut() else {
            return;
        };
        let tab_id = tab.id;
        let outcome = tab.resolve_conflict_keep_mine();
        self.handle_conflict_action_outcome(
            tab_id,
            outcome,
            "Kept local edits and saved (disk backup created)",
        );
    }

    fn resolve_active_conflict_use_theirs(&mut self) {
        self.close_confirm_armed = false;
        let Some(tab) = self.workbench.active_tab_mut() else {
            return;
        };
        let tab_id = tab.id;
        let rev = tab.session.buffer.revision();
        let outcome = tab.resolve_conflict_use_theirs(rev);
        self.handle_conflict_action_outcome(
            tab_id,
            outcome,
            "Loaded disk version into editor (Undo with Cmd/Ctrl+Z)",
        );
    }

    fn resolve_active_conflict_auto_merge(&mut self) {
        self.close_confirm_armed = false;
        let Some(tab) = self.workbench.active_tab_mut() else {
            return;
        };
        let tab_id = tab.id;
        let rev = tab.session.buffer.revision();
        let outcome = tab.resolve_conflict_auto_merge(rev);
        match outcome {
            AutoMergeOutcome::MergedClean | AutoMergeOutcome::MergedDirty { .. } => {
                self.conflict_diff_open = false;
                self.workbench.refresh_tab_conflict_backup_count(tab_id);
                self.set_notice("Auto-merged non-overlapping changes (Undo with Cmd/Ctrl+Z)");
                self.trigger_background_maintenance(true);
            }
            AutoMergeOutcome::HasOverlappingConflicts { conflicting_hunks } => {
                self.conflict_diff_open = true;
                self.set_notice(format!(
                    "Auto Merge found {} overlapping hunk(s) — inspect Diff or choose Keep Mine / Use Disk",
                    conflicting_hunks.len()
                ));
            }
            AutoMergeOutcome::TransitionedToMissing => {
                self.conflict_diff_open = false;
                self.set_notice("Disk file missing — choose Re-save, Save As..., or Close Tab");
            }
            AutoMergeOutcome::AbortedStillConflict(msg) => {
                self.conflict_diff_open = true;
                self.set_notice(format!("Auto Merge blocked: {msg}"));
            }
        }
    }

    fn finish_save_as(
        &mut self,
        tab_id: u64,
        dest: &Path,
        outcome: SaveOutcome,
        remove_reserved_on_failure: bool,
        notice_with_path: bool,
    ) {
        match outcome {
            SaveOutcome::SavedClean | SaveOutcome::SavedButStillDirty { .. } => {
                self.conflict_diff_open = false;
                self.workbench.recent_files.record_open(dest);
                self.workbench.refresh_tab_conflict_backup_count(tab_id);
                if notice_with_path {
                    self.set_notice(format!("Saved {}", dest.display()));
                } else {
                    self.set_notice(format!("Saved as {}", dest.display()));
                }
                self.trigger_background_maintenance(true);
            }
            SaveOutcome::EnteredConflict => {
                if remove_reserved_on_failure {
                    let _ = fs::remove_file(dest);
                }
                self.conflict_diff_open = true;
                self.set_notice(
                    "Conflict at target file — choose Diff, Auto Merge, Keep Mine, or Use Disk",
                );
            }
            SaveOutcome::WriteFailed(e) => {
                if remove_reserved_on_failure {
                    let _ = fs::remove_file(dest);
                }
                self.set_notice(format!("Save error: {e}"));
            }
        }
    }

    fn save_as_active_tab(&mut self) {
        self.close_confirm_armed = false;
        let Some((tab_id, current_path, default_name)) = self.workbench.active_tab().map(|t| {
            let default_name = t
                .session
                .path
                .as_deref()
                .and_then(|p| p.file_name())
                .and_then(|n| n.to_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| "Untitled.md".to_string());
            (t.id, t.session.path.clone(), default_name)
        }) else {
            return;
        };
        let Some(dest) = self
            .save_dialog
            .pick_save_as_path(current_path.as_deref(), &default_name)
        else {
            self.set_notice("Save As cancelled");
            return;
        };
        let outcome = match self.workbench.save_tab_as(tab_id, &dest) {
            Ok(outcome) => outcome,
            Err(err) => {
                self.set_notice(err);
                return;
            }
        };
        self.finish_save_as(tab_id, &dest, outcome, false, false);
    }

    fn save_active_tab(&mut self) {
        self.close_confirm_armed = false;
        let untitled_dir = self.untitled_save_dir.clone();
        let Some((tab_id, state, is_untitled)) = self
            .workbench
            .active_tab()
            .map(|t| (t.id, t.session.state, t.session.path.is_none()))
        else {
            return;
        };

        // PRD 4.5.1 C: `Conflict` Cmd/Ctrl+S opens conflict options instead of overwriting disk.
        if state == DocState::Conflict {
            let first = match self.workbench.tabs.iter_mut().find(|t| t.id == tab_id) {
                Some(tab) => tab.sync_from_disk(true),
                None => return,
            };
            let sync = if let ExternalSyncResult::Renamed { old_path, new_path } = first {
                if let Some((_, _, other_title)) =
                    self.workbench.find_other_tab_with_path(tab_id, &new_path)
                {
                    self.set_notice(format!(
                        "Conflict rename target already open in '{other_title}' — keeping current path"
                    ));
                    if let Some(cur) = self.workbench.tabs.iter_mut().find(|t| t.id == tab_id) {
                        cur.session.path = Some(old_path);
                        cur.session.on_external_deleted();
                    }
                    ExternalSyncResult::BecameMissing
                } else {
                    self.workbench.recent_files.record_open(&new_path);
                    self.workbench.refresh_tab_conflict_backup_count(tab_id);
                    match self.workbench.tabs.iter_mut().find(|t| t.id == tab_id) {
                        Some(cur) => cur.sync_from_disk(true),
                        None => return,
                    }
                }
            } else {
                first
            };
            let new_state = self
                .workbench
                .tabs
                .iter()
                .find(|t| t.id == tab_id)
                .map(|t| t.session.state);
            match (sync, new_state) {
                (ExternalSyncResult::BecameMissing, _) => {
                    self.conflict_diff_open = false;
                    self.set_notice(
                        "File deleted or moved externally — choose Re-save, Save As..., or Close Tab",
                    );
                }
                (ExternalSyncResult::ReloadedClean, _)
                | (ExternalSyncResult::ReappearedClean { .. }, _) => {
                    self.conflict_diff_open = false;
                    self.set_notice("Disk content already matches local buffer (Clean)");
                    self.trigger_background_maintenance(true);
                }
                (ExternalSyncResult::ConflictTheirsUpdated, _) => {
                    self.conflict_diff_open = true;
                    self.set_notice(
                        "Disk file changed again — conflict diff updated; choose an action",
                    );
                }
                (ExternalSyncResult::ReadError(e), _) => {
                    self.conflict_diff_open = true;
                    self.set_notice(format!("Read error while re-checking conflict: {e}"));
                }
                (_, Some(DocState::Conflict)) => {
                    self.conflict_diff_open = true;
                    self.set_notice(
                        "Conflict active — choose Diff, Auto Merge, Keep Mine, Use Disk, or Save As...",
                    );
                }
                (_, Some(DocState::Clean)) => {
                    self.conflict_diff_open = false;
                    self.set_notice("Disk content matches local buffer (Clean)");
                    self.trigger_background_maintenance(true);
                }
                _ => {
                    self.conflict_diff_open = true;
                    self.set_notice("Conflict active — choose resolution option");
                }
            }
            return;
        }

        if is_untitled {
            if let Some(dir) = untitled_dir {
                let reserved = match reserve_unique_untitled_path(&dir, tab_id) {
                    Ok(p) => p,
                    Err(e) => {
                        self.set_notice(format!("Save error: {e}"));
                        return;
                    }
                };
                let outcome = match self.workbench.tabs.iter_mut().find(|t| t.id == tab_id) {
                    Some(tab) => tab.save_as(&reserved),
                    None => return,
                };
                self.finish_save_as(tab_id, &reserved, outcome, true, true);
                return;
            }
            if let Some(dest) = self.save_dialog.pick_save_as_path(None, "Untitled.md") {
                let outcome = match self.workbench.save_tab_as(tab_id, &dest) {
                    Ok(outcome) => outcome,
                    Err(err) => {
                        self.set_notice(err);
                        return;
                    }
                };
                self.finish_save_as(tab_id, &dest, outcome, false, true);
            } else {
                self.set_notice("Save cancelled — use Save As to pick a location");
            }
            return;
        }

        let outcome = match self.workbench.tabs.iter_mut().find(|t| t.id == tab_id) {
            Some(tab) => tab.save(),
            None => return,
        };
        match outcome {
            SaveOutcome::SavedClean | SaveOutcome::SavedButStillDirty { .. } => {
                self.conflict_diff_open = false;
                self.workbench.refresh_tab_conflict_backup_count(tab_id);
                self.set_notice("Saved");
                self.trigger_background_maintenance(true);
            }
            SaveOutcome::EnteredConflict => {
                self.conflict_diff_open = true;
                self.set_notice(
                    "External modification conflict detected — choose Diff, Auto Merge, Keep Mine, Use Disk, or Save As...",
                );
            }
            SaveOutcome::WriteFailed(e) => {
                self.set_notice(format!("Save error: {e}"));
            }
        }
    }

    fn dispatch_editor(&mut self, action: EditorAction) {
        self.find_focused = false;
        self.close_confirm_armed = false;
        let active_id = self.workbench.active_tab().map(|t| t.id);
        self.cancel_abandoned_tab_ime(active_id);
        if let Some(tab) = self.workbench.active_tab_mut() {
            if let Err(e) = tab.dispatch(action) {
                self.set_notice(format!("Read-only or blocked: {e}"));
            } else {
                self.pending_scroll_ensure_caret = true;
            }
        }
    }

    fn delete_forward(&mut self) {
        self.close_confirm_armed = false;
        if let Some(tab) = self.workbench.active_tab_mut() {
            let _ = tab.edit(|ctx| {
                ctx.abort_ime_preedit();
                if !ctx.selection.is_caret() {
                    ctx.backspace();
                    return;
                }
                let head = ctx.selection.head;
                let total = ctx.buffer.len_bytes();
                if head >= total {
                    return;
                }
                let (start, next) = {
                    let rope = ctx.buffer.rope();
                    let ch_idx = rope.byte_to_char(head.min(total));
                    let aligned = rope.char_to_byte(ch_idx);
                    if ctx.buffer.is_crlf_midpoint(aligned) {
                        (aligned - 1, aligned + 1)
                    } else if ctx.buffer.is_crlf_midpoint(aligned + 1) {
                        (aligned, aligned + 2)
                    } else {
                        let next_byte = if ch_idx < rope.len_chars() {
                            rope.char_to_byte(ch_idx + 1)
                        } else {
                            total
                        };
                        (aligned, next_byte)
                    }
                };
                if next > start {
                    ctx.set_selection(start, next);
                    ctx.backspace();
                }
            });
            self.pending_scroll_ensure_caret = true;
        }
    }

    fn move_caret_horizontal(&mut self, dir: i32, extend: bool) {
        if let Some(tab) = self.workbench.active_tab_mut() {
            let sel = tab.view.selection;
            let total = tab.session.buffer.len_bytes();
            let target = if !extend && !sel.is_caret() {
                if dir < 0 {
                    sel.range().start
                } else {
                    sel.range().end
                }
            } else {
                let rope = tab.session.buffer.rope();
                let clamped = sel.head.min(total);
                let ch_idx = rope.byte_to_char(clamped);
                if dir < 0 {
                    let prev = if ch_idx > 0 && rope.char_to_byte(ch_idx) == clamped {
                        rope.char_to_byte(ch_idx - 1)
                    } else {
                        rope.char_to_byte(ch_idx)
                    };
                    if tab.session.buffer.is_crlf_midpoint(prev) {
                        prev - 1
                    } else {
                        prev
                    }
                } else {
                    let next = if ch_idx < rope.len_chars() {
                        rope.char_to_byte(ch_idx + 1)
                    } else {
                        total
                    };
                    if tab.session.buffer.is_crlf_midpoint(next) {
                        next + 1
                    } else {
                        next
                    }
                }
            };
            let _ = tab.edit(|ctx| {
                if extend {
                    ctx.set_selection(sel.anchor, target);
                } else {
                    ctx.set_caret(target);
                }
            });
            self.pending_scroll_ensure_caret = true;
        }
    }

    fn move_caret_line_edge(&mut self, to_start: bool, extend: bool) {
        if let Some(tab) = self.workbench.active_tab_mut() {
            let sel = tab.view.selection;
            let total = tab.session.buffer.len_bytes();
            let target = tab.with_incremental_blocks(|blocks| {
                let idx = blocks
                    .partition_point(|b| b.range.end < sel.head)
                    .min(blocks.len().saturating_sub(1));
                if blocks.is_empty() {
                    0
                } else if sel.head > blocks[blocks.len() - 1].range.end {
                    total
                } else if to_start {
                    blocks[idx].range.start
                } else {
                    blocks[idx].range.end
                }
            });
            let _ = tab.edit(|ctx| {
                let target = ctx.buffer.snap_out_of_crlf(target);
                if extend {
                    ctx.set_selection(sel.anchor, target);
                } else {
                    ctx.set_caret(target);
                }
            });
            self.pending_scroll_ensure_caret = true;
        }
    }

    fn move_caret_vertical(&mut self, dir: i32, extend: bool) {
        let (_, _, editor_rect, _) = self.regions();
        let max_w = self
            .style
            .max_content_width_px
            .min((editor_rect.w - 72.0).max(200.0));
        let preferred_x = self.preferred_caret_x;

        if let Some(tab) = self.workbench.active_tab_mut() {
            let sel = tab.view.selection;
            let (next_pos, next_pref) = tab.with_incremental_blocks(|blocks| {
                self.shaper.move_caret_vertical(
                    sel.head,
                    dir,
                    preferred_x,
                    blocks,
                    &tab.view,
                    &self.style,
                    max_w,
                    |r| tab.session.buffer.slice_to_string(r),
                )
            });
            self.preferred_caret_x = next_pref;
            let _ = tab.edit(|ctx| {
                let next_pos = ctx.buffer.snap_out_of_crlf(next_pos);
                if extend {
                    ctx.set_selection(sel.anchor, next_pos);
                } else {
                    ctx.set_caret(next_pos);
                }
            });
            self.pending_scroll_ensure_caret = true;
        }
    }

    fn copy_selection(&mut self, cut: bool) {
        if self.find_focused {
            if !self.find_query.is_empty() {
                let q = self.find_query.clone();
                match self.clipboard.write(&q, None) {
                    Ok(()) => {
                        self.clipboard_text = q;
                        self.set_notice(if cut {
                            "Cut search query"
                        } else {
                            "Copied search query"
                        });
                    }
                    Err(err) => {
                        self.set_notice(format!("Clipboard error: {err}"));
                        return;
                    }
                }
                if cut {
                    self.find_query.clear();
                    self.run_find_next(false);
                }
            }
            return;
        }

        let Some(tab) = self.workbench.active_tab() else {
            return;
        };
        if tab.view.selection.is_caret() {
            // Never trigger Backspace deletion or overwrite clipboard when selection is empty.
            return;
        }
        let payload = tab.copy_selection_payload();
        if payload.markdown.is_empty() {
            return;
        }

        match self.clipboard.write(&payload.markdown, Some(&payload.html)) {
            Ok(()) => {
                self.clipboard_text = payload.markdown;
                self.set_notice(if cut { "Cut selection" } else { "Copied Markdown" });
            }
            Err(err) => {
                self.set_notice(format!("Clipboard error: {err}"));
                return;
            }
        }
        if cut {
            self.dispatch_editor(EditorAction::Backspace);
        }
    }

    fn paste_clipboard(&mut self) {
        let content = self.clipboard.read();
        let plain = content.text.unwrap_or_default();
        let has_html = content
            .html
            .as_deref()
            .is_some_and(|h| !h.trim().is_empty());
        if plain.is_empty() && !has_html {
            return;
        }
        if !plain.is_empty() {
            self.clipboard_text = plain.clone();
        }
        if self.find_focused {
            if !plain.is_empty() {
                self.find_query.push_str(&plain);
                self.run_find_next(false);
            }
            return;
        }
        self.close_confirm_armed = false;
        let active_id = self.workbench.active_tab().map(|t| t.id);
        self.cancel_abandoned_tab_ime(active_id);
        if let Some(tab) = self.workbench.active_tab_mut() {
            let _ = tab.paste_rich_clipboard(content.html.as_deref(), &plain);
            self.pending_scroll_ensure_caret = true;
        }
    }

    fn run_find_next(&mut self, advance: bool) {
        if self.find_query.is_empty() {
            self.active_find_range = None;
            return;
        }
        let query = self.find_query.clone();
        let opts = self.find_options;
        if let Some(tab) = self.workbench.active_tab_mut() {
            if !advance {
                // Re-anchor near start of current selection so live incremental typing doesn't skip matches.
                let start = tab.view.selection.range().start;
                tab.view.selection = Selection::caret(start);
            }
            match tab.find_next(&query, true, opts) {
                Ok(Some(m)) => {
                    self.active_find_range = Some(m.range);
                    self.pending_scroll_ensure_caret = true;
                }
                Ok(None) => {
                    self.active_find_range = None;
                }
                Err(e) => {
                    self.active_find_range = None;
                    self.set_notice(format!("Regex error: {e}"));
                }
            }
        }
    }

    fn run_find_prev(&mut self) {
        if self.find_query.is_empty() {
            return;
        }
        let query = self.find_query.clone();
        let opts = self.find_options;
        if let Some(tab) = self.workbench.active_tab_mut() {
            if let Ok(Some(m)) = tab.find_prev(&query, true, opts) {
                self.active_find_range = Some(m.range);
                self.pending_scroll_ensure_caret = true;
            }
        }
    }
}

/// Active native `winit` + `wgpu` application handler.
pub struct VellmarkApp {
    controller: GuiController,
    window: Option<Arc<Window>>,
    gpu: Option<WgpuRenderer>,
    modifiers: ModifiersState,
}

impl VellmarkApp {
    pub fn new(controller: GuiController) -> Self {
        Self {
            controller,
            window: None,
            gpu: None,
            modifiers: ModifiersState::default(),
        }
    }

    pub fn run(controller: GuiController) -> Result<(), String> {
        let event_loop = EventLoop::new().map_err(|e| format!("failed to create event loop: {e}"))?;
        let mut app = Self::new(controller);
        event_loop
            .run_app(&mut app)
            .map_err(|e| format!("window event loop exited with error: {e}"))
    }

    fn render_now(&mut self) {
        let (Some(win), Some(gpu)) = (&self.window, &mut self.gpu) else {
            return;
        };
        let scale = win.scale_factor() as f32;
        let phys = win.inner_size();
        self.controller.scale_factor = scale;
        self.controller.window_size = (
            phys.width as f32 / scale.max(0.5),
            phys.height as f32 / scale.max(0.5),
        );

        let (editor_dl, chrome_dl) = self.controller.build_frame();
        if let Some(tab) = self.controller.workbench.active_tab() {
            let dirty_mark = if tab.session.state != DocState::Clean {
                "• "
            } else {
                ""
            };
            let title = format!("{dirty_mark}{} — Vellmark", tab.title());
            win.set_title(&title);
        }
        if let Some(area) = self.controller.last_ime_cursor_area {
            win.set_ime_cursor_area(
                LogicalPosition::new(area.x, area.y),
                LogicalSize::new(area.w.max(2.0), area.h.max(16.0)),
            );
        }
        let clear = self.controller.style.bg_color;
        let _ = gpu.render_layers(
            clear,
            scale,
            self.controller.shaper.font_system_mut(),
            &[&editor_dl, &chrome_dl],
        );
    }

    fn request_redraw(&self) {
        if let Some(win) = &self.window {
            win.request_redraw();
        }
    }
}

impl ApplicationHandler for VellmarkApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = WindowAttributes::default()
            .with_title("Vellmark — Markdown Studio")
            .with_inner_size(LogicalSize::new(1220.0, 820.0));
        match event_loop.create_window(attrs) {
            Ok(win) => {
                win.set_ime_allowed(true);
                if let Some(theme) = win.theme() {
                    self.controller
                        .workbench
                        .set_system_color_scheme(matches!(theme, WindowTheme::Dark));
                }
                let arc = Arc::new(win);
                match WgpuRenderer::new(arc.clone()) {
                    Ok(gpu) => {
                        let scale = arc.scale_factor() as f32;
                        let phys = arc.inner_size();
                        self.controller.scale_factor = scale;
                        self.controller.window_size = (
                            phys.width as f32 / scale.max(0.5),
                            phys.height as f32 / scale.max(0.5),
                        );
                        self.window = Some(arc);
                        self.gpu = Some(gpu);
                        self.request_redraw();
                    }
                    Err(e) => {
                        eprintln!("Failed to initialize wgpu renderer: {e}");
                        event_loop.exit();
                    }
                }
            }
            Err(e) => {
                eprintln!("Failed to create native window: {e}");
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                if self.controller.handle_close_requested() {
                    event_loop.exit();
                } else {
                    self.request_redraw();
                }
            }
            WindowEvent::Resized(new_size) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.resize(new_size.width, new_size.height);
                }
                self.request_redraw();
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                self.controller.scale_factor = scale_factor as f32;
                self.request_redraw();
            }
            WindowEvent::ThemeChanged(theme) => {
                self.controller
                    .workbench
                    .set_system_color_scheme(matches!(theme, WindowTheme::Dark));
                self.controller.refresh_style();
                self.request_redraw();
            }
            WindowEvent::ModifiersChanged(m) => {
                self.modifiers = m.state();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if self.controller.handle_key_press(&event, self.modifiers) {
                    self.request_redraw();
                }
            }
            WindowEvent::Ime(ime) => {
                self.controller.handle_ime_event(&ime);
                self.request_redraw();
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                match state {
                    ElementState::Pressed => {
                        let (px, py) = self.controller.pointer_pos;
                        if self.controller.handle_pointer_down(px, py) {
                            self.request_redraw();
                        }
                    }
                    ElementState::Released => {
                        self.controller.handle_pointer_up();
                    }
                }
            }
            WindowEvent::CursorMoved {
                position: PhysicalPosition { x, y },
                ..
            } => {
                let scale = self.controller.scale_factor.max(0.5);
                let lx = (x as f32) / scale;
                let ly = (y as f32) / scale;
                let (icon, changed) = self.controller.handle_pointer_move(lx, ly);
                if let Some(win) = &self.window {
                    win.set_cursor(icon);
                }
                if changed {
                    self.request_redraw();
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if self.controller.handle_wheel(delta) {
                    self.request_redraw();
                }
            }
            WindowEvent::DroppedFile(path) => {
                self.controller.close_confirm_armed = false;
                let res = if path.is_dir() {
                    self.controller.workbench.open_folder(&path).map(|_| ())
                } else {
                    self.controller.flush_active_tab_before_switch();
                    self.controller
                        .workbench
                        .open_document(&path)
                        .map(|_| ())
                        .map_err(|e| std::io::Error::other(e.to_string()))
                };
                if let Err(e) = res {
                    self.controller.set_notice(format!("Drop open error: {e}"));
                }
                self.request_redraw();
            }
            WindowEvent::Focused(false) => {
                if self.controller.handle_focus_lost() {
                    self.request_redraw();
                }
            }
            WindowEvent::Focused(true) => {
                if let Some(win) = &self.window {
                    win.set_ime_allowed(true);
                }
                self.controller.handle_focus_gained();
                self.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                self.render_now();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.controller.poll_background_maintenance() {
            self.request_redraw();
        }
        self.controller.workbench.maybe_maintain_recovery_storage_async();
        let now = Instant::now();
        if self.controller.tick_runtime_at(now) {
            self.request_redraw();
        }
        if self.controller.tick_blink(now) {
            self.request_redraw();
        }
        let next_wake = self.controller.next_wake_deadline();
        event_loop.set_control_flow(ControlFlow::WaitUntil(next_wake));
    }
}

fn reserve_unique_untitled_path(dir: &Path, tab_id: u64) -> std::io::Result<PathBuf> {
    fs::create_dir_all(dir)?;
    let start_seq = tab_id.max(1);
    for seq in start_seq..start_seq + 10_000 {
        let candidate = dir.join(format!("vellmark-note-{seq}.md"));
        if fs::symlink_metadata(&candidate).is_ok() {
            continue;
        }
        match OpenOptions::new().write(true).create_new(true).open(&candidate) {
            Ok(_) => return Ok(candidate),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id();
    let candidate = dir.join(format!("vellmark-note-{nanos}-{pid}-{tab_id}.md"));
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&candidate)?;
    Ok(candidate)
}

fn style_from_workbench(wb: &AppWorkbench) -> ResolvedThemeStyle {
    ResolvedThemeStyle::from_sheet(
        &wb.theme,
        wb.config.font_size_px,
        wb.config.line_height,
        wb.config.zoom_percent,
        wb.config.max_content_width_px as f32,
    )
}

fn flatten_file_tree(
    node: &vellmark_core::FileTreeNode,
    depth: usize,
    collapsed: &HashSet<PathBuf>,
    out: &mut Vec<FlatTreeRow>,
) {
    out.push(FlatTreeRow {
        depth,
        name: node.name.clone(),
        path: node.path.clone(),
        is_dir: node.is_dir,
    });
    if node.is_dir && !collapsed.contains(&node.path) {
        for child in &node.children {
            flatten_file_tree(child, depth + 1, collapsed, out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clipboard::{ClipboardContent, MemoryClipboard};
    use crate::config::AppConfig;

    fn controller_with_doc(md: &str) -> GuiController {
        let mut wb = AppWorkbench::new(AppConfig::default(), None);
        wb.new_document();
        if let Some(tab) = wb.active_tab_mut() {
            tab.session.buffer.reset_content(md);
            tab.view.selection = Selection::caret(0);
        }
        GuiController::with_clipboard(wb, Box::new(MemoryClipboard::default()))
    }

    #[test]
    fn test_ime_lifecycle_preserves_buffer_until_commit_and_reports_cursor_area() {
        let mut gui = controller_with_doc("## Title\nHello world");
        let caret = gui.workbench.active_tab().unwrap().session.buffer.len_bytes();
        gui.workbench
            .active_tab_mut()
            .unwrap()
            .edit(|ctx| ctx.set_caret(caret))
            .unwrap();
        let rev_before = gui.workbench.active_tab().unwrap().session.buffer.revision();

        gui.handle_ime_event(&Ime::Enabled);
        gui.handle_ime_event(&Ime::Preedit("zhongwen".to_string(), Some((8, 8))));
        {
            let tab = gui.workbench.active_tab().unwrap();
            assert_eq!(tab.session.buffer.revision(), rev_before);
            assert_eq!(tab.view.ime_preedit.as_deref(), Some("zhongwen"));
            assert!(tab.view.expansion_frozen);
        }

        let (editor_dl, _chrome_dl) = gui.build_frame();
        assert!(!editor_dl.glyphs.is_empty());
        assert!(gui.last_ime_cursor_area.is_some());

        gui.handle_ime_event(&Ime::Commit("中文".to_string()));
        let tab = gui.workbench.active_tab().unwrap();
        assert_eq!(tab.session.buffer.to_string(), "## Title\nHello world中文");
        assert!(tab.view.ime_preedit.is_none());
        assert!(!tab.view.expansion_frozen);
    }

    #[test]
    fn test_sidebar_outline_click_and_mode_toggle_ui_actions() {
        let mut gui = controller_with_doc("# Top\n\nBody text\n\n## Second Section\n\nEnd line");
        let _ = gui.build_frame();

        // Click the Mode toggle zone in the top bar
        let mode_rect = gui
            .ui_hit_zones
            .iter()
            .find(|z| matches!(z.action, UiAction::ToggleMode))
            .expect("mode toggle button")
            .rect;
        assert!(gui.handle_pointer_down(mode_rect.x + 5.0, mode_rect.y + 5.0));
        assert_eq!(
            gui.workbench.active_tab().unwrap().view.mode,
            ViewMode::Source
        );

        // Click the second Outline entry -> jumps caret directly to `Second Section`
        let jump_rect = gui
            .ui_hit_zones
            .iter()
            .find(|z| matches!(z.action, UiAction::JumpOutline(1)))
            .expect("second outline entry")
            .rect;
        assert!(gui.handle_pointer_down(jump_rect.x + 5.0, jump_rect.y + 5.0));
        let caret = gui.workbench.active_tab().unwrap().view.selection.head;
        let doc = gui.workbench.active_tab().unwrap().session.buffer.to_string();
        assert!(doc[caret..].starts_with("Second Section"));
    }

    #[test]
    fn test_editor_pointer_hit_test_and_vertical_navigation_respects_cjk() {
        let mut gui = controller_with_doc("**粗体段落**\n第二行内容");
        let _ = gui.build_frame();
        // Move down from line 0 to line 1 using virtualized shaped navigation
        gui.move_caret_vertical(1, false);
        let head = gui.workbench.active_tab().unwrap().view.selection.head;
        let line1_start = "**粗体段落**\n".len();
        assert!(head >= line1_start);
    }

    #[test]
    fn test_ime_commit_after_tab_switch_routes_to_original_tab_and_preserves_new_tab() {
        let mut gui = controller_with_doc("TabA ");
        let tab_a_id = gui.workbench.tabs[0].id;
        let end_a = gui.workbench.tabs[0].session.buffer.len_bytes();
        gui.workbench.tabs[0]
            .edit(|ctx| ctx.set_caret(end_a))
            .unwrap();

        let _tab_b_id = gui.workbench.new_document();
        gui.workbench.tabs[1]
            .session
            .buffer
            .reset_content("TabB untouched");
        // Switch back to TabA and start composing
        assert!(gui.execute_ui_action(UiAction::SelectTab(0)));
        gui.handle_ime_event(&Ime::Preedit("zhongwen".to_string(), Some((8, 8))));
        assert_eq!(gui.workbench.tabs[0].view.ime_preedit.as_deref(), Some("zhongwen"));

        // Switch to TabB before IME commits
        assert!(gui.execute_ui_action(UiAction::SelectTab(1)));
        assert_eq!(gui.workbench.active_tab().unwrap().id, gui.workbench.tabs[1].id);
        gui.handle_ime_event(&Ime::Preedit(String::new(), None));
        gui.handle_ime_event(&Ime::Commit("中文".to_string()));

        assert_eq!(gui.workbench.tabs[0].session.buffer.to_string(), "TabA 中文");
        assert_eq!(
            gui.workbench.tabs[1].session.buffer.to_string(),
            "TabB untouched"
        );
        assert!(gui.workbench.tabs[0].view.ime_preedit.is_none());
        let _ = tab_a_id;
    }

    #[test]
    fn test_empty_ime_preedit_before_commit_keeps_composition_anchor_after_cursor_move() {
        let mut gui = controller_with_doc("AB");
        gui.workbench
            .active_tab_mut()
            .unwrap()
            .edit(|ctx| ctx.set_caret(1))
            .unwrap();

        gui.handle_ime_event(&Ime::Preedit("ni".to_string(), Some((2, 2))));
        // Simulate clicking end of text during composition
        gui.workbench
            .active_tab_mut()
            .unwrap()
            .edit(|ctx| ctx.set_caret(2))
            .unwrap();
        // Simulate winit/macOS clearing markedText with empty preedit immediately before Commit
        gui.handle_ime_event(&Ime::Preedit(String::new(), None));
        gui.handle_ime_event(&Ime::Commit("你".to_string()));

        assert_eq!(
            gui.workbench.active_tab().unwrap().session.buffer.to_string(),
            "A你B"
        );
    }

    #[test]
    fn test_cjk_left_arrow_navigation_stays_on_char_boundaries() {
        let mut gui = controller_with_doc("一二三");
        let end = "一二三".len();
        assert_eq!(end, 9);
        gui.workbench
            .active_tab_mut()
            .unwrap()
            .edit(|ctx| ctx.set_caret(end))
            .unwrap();

        gui.move_caret_horizontal(-1, false);
        let pos1 = gui.workbench.active_tab().unwrap().view.selection.head;
        assert_eq!(pos1, 6);
        assert!("一二三".is_char_boundary(pos1));

        gui.move_caret_horizontal(-1, false);
        assert_eq!(gui.workbench.active_tab().unwrap().view.selection.head, 3);

        gui.move_caret_horizontal(-1, false);
        assert_eq!(gui.workbench.active_tab().unwrap().view.selection.head, 0);
    }

    #[test]
    fn test_cut_empty_selection_does_not_delete_or_overwrite_clipboard() {
        let initial_clip = ClipboardContent {
            text: Some("preserved".to_string()),
            html: None,
        };
        let shared_clip = std::sync::Arc::new(MemoryClipboard::with_initial(initial_clip));
        struct SharedClip(std::sync::Arc<MemoryClipboard>);
        impl ClipboardBackend for SharedClip {
            fn read(&self) -> ClipboardContent {
                self.0.read()
            }
            fn write(&self, text: &str, html: Option<&str>) -> Result<(), String> {
                self.0.write(text, html)
            }
        }

        let mut wb = AppWorkbench::new(AppConfig::default(), None);
        wb.new_document();
        wb.active_tab_mut()
            .unwrap()
            .session
            .buffer
            .reset_content("hello");
        wb.active_tab_mut()
            .unwrap()
            .edit(|ctx| ctx.set_caret(5))
            .unwrap();
        let mut gui = GuiController::with_clipboard(wb, Box::new(SharedClip(shared_clip.clone())));

        gui.copy_selection(true);
        assert_eq!(gui.workbench.active_tab().unwrap().session.buffer.to_string(), "hello");
        assert_eq!(shared_clip.get_content().text.as_deref(), Some("preserved"));
    }

    #[test]
    fn test_copy_cut_paste_exchanges_with_clipboard_backend_including_html() {
        let shared_clip = std::sync::Arc::new(MemoryClipboard::default());
        struct SharedClip(std::sync::Arc<MemoryClipboard>);
        impl ClipboardBackend for SharedClip {
            fn read(&self) -> ClipboardContent {
                self.0.read()
            }
            fn write(&self, text: &str, html: Option<&str>) -> Result<(), String> {
                self.0.write(text, html)
            }
        }

        let mut wb = AppWorkbench::new(AppConfig::default(), None);
        wb.new_document();
        wb.active_tab_mut()
            .unwrap()
            .session
            .buffer
            .reset_content("**hello** world");
        wb.active_tab_mut()
            .unwrap()
            .edit(|ctx| ctx.set_selection(0, "**hello**".len()))
            .unwrap();

        let mut gui = GuiController::with_clipboard(wb, Box::new(SharedClip(shared_clip.clone())));
        gui.copy_selection(false);
        assert_eq!(shared_clip.get_content().text.as_deref(), Some("**hello**"));
        assert!(shared_clip
            .get_content()
            .html
            .as_deref()
            .is_some_and(|h| h.contains("<strong>hello</strong>")));

        // External app puts rich HTML into clipboard -> Paste converts HTML to Markdown!
        shared_clip.set_content(ClipboardContent {
            text: Some("heading".to_string()),
            html: Some("<h2>External Heading</h2>".to_string()),
        });
        let end = gui.workbench.active_tab().unwrap().session.buffer.len_bytes();
        gui.workbench
            .active_tab_mut()
            .unwrap()
            .edit(|ctx| ctx.set_caret(end))
            .unwrap();
        gui.paste_clipboard();
        assert!(gui
            .workbench
            .active_tab()
            .unwrap()
            .session
            .buffer
            .to_string()
            .contains("## External Heading"));
    }

    #[test]
    fn test_untitled_save_does_not_overwrite_existing_note_file() {
        let dir = std::env::temp_dir().join(format!(
            "vellmark-untitled-save-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        let existing = dir.join("vellmark-note-1.md");
        fs::write(&existing, "previous session note").unwrap();

        let mut gui = controller_with_doc("");
        gui.untitled_save_dir = Some(dir.clone());
        gui.workbench
            .active_tab_mut()
            .unwrap()
            .edit(|ctx| ctx.insert_text("new session note"))
            .unwrap();

        gui.save_active_tab();
        let saved_path = gui
            .workbench
            .active_tab()
            .unwrap()
            .session
            .path
            .clone()
            .expect("path assigned");
        assert_ne!(saved_path, existing);
        assert_eq!(fs::read_to_string(&existing).unwrap(), "previous session note");
        assert_eq!(fs::read_to_string(&saved_path).unwrap(), "new session note");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_close_requested_blocks_when_dirty_flushes_recovery_and_saves_cleanly() {
        let dir = std::env::temp_dir().join(format!(
            "vellmark-close-guard-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let rec_root = dir.join("rec");
        let note_dir = dir.join("notes");
        fs::create_dir_all(&rec_root).unwrap();

        let mut wb = AppWorkbench::new(AppConfig::default(), Some(rec_root.clone()));
        wb.new_document();
        wb.active_tab_mut()
            .unwrap()
            .edit(|ctx| ctx.insert_text("unsaved close guard test"))
            .unwrap();
        let mut gui = GuiController::with_clipboard(wb, Box::new(MemoryClipboard::default()));
        gui.untitled_save_dir = Some(note_dir.clone());

        // First CloseRequested must block close AND flush crash recovery snapshot synchronously
        assert!(!gui.handle_close_requested());
        assert!(gui.close_confirm_armed);
        assert!(!vellmark_core::scan_recovery_root(&rec_root).is_empty());

        // Saving disarms close guard and turns document Clean; subsequent close exits immediately
        gui.save_active_tab();
        assert!(!gui.close_confirm_armed);
        assert!(gui.handle_close_requested());

        // Deliberate double close when Dirty discards recovery on exit
        let mut wb2 = AppWorkbench::new(AppConfig::default(), Some(rec_root.clone()));
        wb2.new_document();
        wb2.active_tab_mut()
            .unwrap()
            .edit(|ctx| ctx.insert_text("discard me"))
            .unwrap();
        let mut gui2 = GuiController::with_clipboard(wb2, Box::new(MemoryClipboard::default()));
        assert!(!gui2.handle_close_requested());
        assert!(gui2.handle_close_requested());
        assert!(vellmark_core::scan_recovery_root(&rec_root).is_empty());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_continuing_ime_preedit_after_tab_switch_preserves_original_tab_ownership() {
        let mut gui = controller_with_doc("TabA ");
        let tab_a_id = gui.workbench.tabs[0].id;
        let end_a = gui.workbench.tabs[0].session.buffer.len_bytes();
        gui.workbench.tabs[0]
            .edit(|ctx| ctx.set_caret(end_a))
            .unwrap();

        let tab_b_id = gui.workbench.new_document();
        gui.workbench.tabs[1]
            .session
            .buffer
            .reset_content("TabB untouched");

        assert!(gui.execute_ui_action(UiAction::SelectTab(0)));
        gui.handle_ime_event(&Ime::Preedit("ni".to_string(), Some((2, 2))));
        assert_eq!(gui.ime_target, ImeTarget::Tab(tab_a_id));

        assert!(gui.execute_ui_action(UiAction::SelectTab(1)));
        assert_eq!(gui.workbench.active_tab().unwrap().id, tab_b_id);

        // Continuing non-empty preedit must stay bound to TabA until commit or explicit cancel
        gui.handle_ime_event(&Ime::Preedit("nihao".to_string(), Some((5, 5))));
        assert_eq!(gui.ime_target, ImeTarget::Tab(tab_a_id));
        assert_eq!(gui.workbench.tabs[0].view.ime_preedit.as_deref(), Some("nihao"));
        assert!(gui.workbench.tabs[1].view.ime_preedit.is_none());

        gui.handle_ime_event(&Ime::Commit("你好".to_string()));
        assert_eq!(gui.workbench.tabs[0].session.buffer.to_string(), "TabA 你好");
        assert_eq!(
            gui.workbench.tabs[1].session.buffer.to_string(),
            "TabB untouched"
        );
        assert_eq!(gui.ime_target, ImeTarget::None);

        // After commit, a fresh composition binds to the currently active TabB
        gui.handle_ime_event(&Ime::Preedit("a".to_string(), Some((1, 1))));
        assert_eq!(gui.ime_target, ImeTarget::Tab(tab_b_id));
    }

    #[test]
    fn test_cut_preserves_document_and_find_query_when_clipboard_write_fails() {
        let shared_clip = std::sync::Arc::new(MemoryClipboard::with_initial(ClipboardContent {
            text: Some("prior".to_string()),
            html: None,
        }));
        struct FailingClip(std::sync::Arc<MemoryClipboard>);
        impl ClipboardBackend for FailingClip {
            fn read(&self) -> ClipboardContent {
                self.0.read()
            }
            fn write(&self, _text: &str, _html: Option<&str>) -> Result<(), String> {
                Err("clipboard unavailable".to_string())
            }
        }

        let mut wb = AppWorkbench::new(AppConfig::default(), None);
        wb.new_document();
        wb.active_tab_mut()
            .unwrap()
            .session
            .buffer
            .reset_content("hello world");
        wb.active_tab_mut()
            .unwrap()
            .edit(|ctx| ctx.set_selection(0, 5))
            .unwrap();

        let mut gui = GuiController::with_clipboard(wb, Box::new(FailingClip(shared_clip.clone())));
        gui.clipboard_text = "prior".to_string();

        gui.copy_selection(true);
        assert_eq!(
            gui.workbench.active_tab().unwrap().session.buffer.to_string(),
            "hello world"
        );
        assert_eq!(
            gui.workbench.active_tab().unwrap().view.selection,
            Selection::new(0, 5)
        );
        assert_eq!(gui.clipboard_text, "prior");
        assert_eq!(shared_clip.get_content().text.as_deref(), Some("prior"));
        assert!(gui
            .status_notice
            .as_ref()
            .is_some_and(|(n, _)| n.contains("Clipboard error")));

        gui.find_open = true;
        gui.find_focused = true;
        gui.find_query = "world".to_string();
        gui.copy_selection(true);
        assert_eq!(gui.find_query, "world");
        assert_eq!(gui.clipboard_text, "prior");
        assert!(gui
            .status_notice
            .as_ref()
            .is_some_and(|(n, _)| n.contains("Clipboard error")));
    }

    #[test]
    fn test_gui_close_runs_recovery_maintenance_and_prunes_conflict_backups() {
        let dir = std::env::temp_dir().join(format!(
            "vellmark-gui-close-maint-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let rec_root = dir.join("rec");
        let doc = dir.join("doc.md");
        fs::create_dir_all(&rec_root).unwrap();
        fs::write(&doc, "Base\n").unwrap();

        let mut wb = AppWorkbench::new(AppConfig::default(), Some(rec_root.clone()));
        wb.open_document(&doc).unwrap();
        {
            let tab = wb.active_tab_mut().unwrap();
            tab.edit(|ctx| ctx.insert_text("Local\n")).unwrap();
            let rev = tab.session.buffer.revision();
            let rope = tab.session.buffer.snapshot_rope();
            fs::write(&doc, "External\n").unwrap();
            assert_eq!(tab.save(), SaveOutcome::EnteredConflict);
            assert_eq!(
                tab.session.resolve_conflict_keep_mine(rev, rope),
                vellmark_core::ConflictActionOutcome::ResolvedClean
            );
        }
        assert_eq!(wb.list_history_backups().len(), 1);

        let mut gui = GuiController::with_clipboard(wb, Box::new(MemoryClipboard::default()));
        // Tighten quota to 0 to simulate expired/over-budget conflict backups at clean exit time
        gui.workbench.history_policy.max_total_bytes = 0;

        assert!(gui.handle_close_requested());
        let report = gui
            .workbench
            .last_recovery_maintenance
            .as_ref()
            .expect("clean window exit must run recovery storage maintenance");
        assert_eq!(report.history_prune.removed_global_cap, 1);
        assert!(gui.workbench.list_history_backups().is_empty());
        assert!(
            vellmark_core::scan_recovery_root(&rec_root).is_empty(),
            "clean window exit must leave zero current snapshots"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_runtime_protected_snapshot_limit_warning_updates_and_persists_until_resolved() {
        let dir = std::env::temp_dir().join(format!(
            "vellmark-gui-runtime-limit-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let rec_root = dir.join("rec");
        let save_dir = dir.join("saved");
        fs::create_dir_all(&rec_root).unwrap();
        fs::create_dir_all(&save_dir).unwrap();

        let mut wb = AppWorkbench::new(AppConfig::default(), Some(rec_root.clone()));
        wb.history_policy.max_total_bytes = 256;
        // Initial startup maintenance starts cleanly under budget.
        let initial = wb.maintain_recovery_storage_with_grace(0).unwrap();
        assert!(!initial.protected_exceeds_limit);

        let mut gui = GuiController::with_clipboard(wb, Box::new(MemoryClipboard::default()));
        gui.untitled_save_dir = Some(save_dir.clone());
        assert!(gui.active_status_notice().is_none());

        // During runtime, create an unsaved draft exceeding the limit and flush its snapshot.
        gui.workbench.new_document();
        {
            let tab = gui.workbench.active_tab_mut().unwrap();
            tab.edit(|ctx| ctx.insert_text(&"X".repeat(2048))).unwrap();
            tab.session.flush_current_recovery_snapshot().unwrap();
        }

        // Background maintenance runs and crosses limit during runtime (no restart!).
        assert!(gui.workbench.maintain_recovery_storage_async());
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !gui.poll_background_maintenance() {
            assert!(
                std::time::Instant::now() < deadline,
                "timed out waiting for runtime background maintenance"
            );
            std::thread::sleep(std::time::Duration::from_millis(5));
        }

        assert!(gui.workbench.protected_snapshots_exceed_limit());
        assert_eq!(
            gui.active_status_notice().as_deref(),
            Some(PROTECTED_SNAPSHOT_LIMIT_NOTICE)
        );

        // Even after a 3-second transient notice timeout expires, warning remains visible.
        gui.status_notice = Some((
            "Transient notice".to_string(),
            Instant::now() - Duration::from_secs(5),
        ));
        assert_eq!(
            gui.active_status_notice().as_deref(),
            Some(PROTECTED_SNAPSHOT_LIMIT_NOTICE)
        );

        // Transient notices during active overflow still retain the protected-limit warning.
        gui.set_notice("Saved something else");
        let combined = gui.active_status_notice().unwrap();
        assert!(combined.contains(PROTECTED_SNAPSHOT_LIMIT_NOTICE));

        // Saving the large untitled draft removes its protected snapshot and forces re-audit,
        // clearing the limit warning once background maintenance finishes.
        gui.save_active_tab();
        let clear_deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while gui.workbench.protected_snapshots_exceed_limit() {
            gui.poll_background_maintenance();
            assert!(
                std::time::Instant::now() < clear_deadline,
                "timed out waiting for warning to clear after save"
            );
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!gui.workbench.protected_snapshots_exceed_limit());
        assert!(!gui
            .active_status_notice()
            .as_deref()
            .is_some_and(|n| n.contains(PROTECTED_SNAPSHOT_LIMIT_NOTICE)));

        let _ = fs::remove_dir_all(&dir);
    }

    fn click_hit_action(gui: &mut GuiController, predicate: impl Fn(&UiAction) -> bool) -> bool {
        let _ = gui.build_frame();
        let Some(zone) = gui
            .ui_hit_zones
            .iter()
            .rev()
            .find(|z| predicate(&z.action))
            .cloned()
        else {
            return false;
        };
        gui.handle_pointer_down(
            zone.rect.x + zone.rect.w * 0.5,
            zone.rect.y + zone.rect.h * 0.5,
        )
    }

    #[test]
    fn test_conflict_banner_diff_automerge_keepmine_usedisk_saveas_and_cmd_s_behavior() {
        let dir = std::env::temp_dir().join(format!(
            "vellmark-gui-conflict-actions-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let rec_root = dir.join("rec");
        let doc = dir.join("note.md");
        let save_as_doc = dir.join("note-copy.md");
        fs::create_dir_all(&rec_root).unwrap();
        fs::write(&doc, "line1\nline2\nline3\n").unwrap();

        let mut wb = AppWorkbench::new(AppConfig::default(), Some(rec_root));
        wb.open_document(&doc).unwrap();
        let mut gui = GuiController::with_clipboard(wb, Box::new(MemoryClipboard::default()));
        gui.window_size = (1100.0, 720.0);

        // Local edit on line 1; external disk edit on line 3 (disjoint -> Auto Merge succeeds)
        {
            let tab = gui.workbench.active_tab_mut().unwrap();
            tab.edit(|ctx| {
                ctx.set_selection(0, 5);
                ctx.insert_text("mine1");
            })
            .unwrap();
        }
        fs::write(&doc, "line1\nline2\ndisk3\n").unwrap();
        assert!(gui.handle_focus_gained());
        assert_eq!(
            gui.workbench.active_tab().unwrap().session.state,
            DocState::Conflict
        );

        // Cmd/Ctrl+S under Conflict opens diff options without overwriting disk
        gui.save_active_tab();
        assert!(gui.conflict_diff_open);
        assert_eq!(fs::read_to_string(&doc).unwrap(), "line1\nline2\ndisk3\n");
        assert_eq!(
            gui.workbench.active_tab().unwrap().session.state,
            DocState::Conflict
        );

        // Toggle Diff off via banner button, then back on
        assert!(click_hit_action(&mut gui, |a| matches!(a, UiAction::ToggleConflictDiff)));
        assert!(!gui.conflict_diff_open);
        assert!(click_hit_action(&mut gui, |a| matches!(a, UiAction::ToggleConflictDiff)));
        assert!(gui.conflict_diff_open);

        // Click Auto Merge -> resolves disjoint changes, exits Conflict, leaves merged content
        assert!(click_hit_action(&mut gui, |a| matches!(a, UiAction::ResolveConflictAutoMerge)));
        assert!(!gui.conflict_diff_open);
        assert_eq!(
            gui.workbench.active_tab().unwrap().session.state,
            DocState::Dirty
        );
        assert_eq!(
            gui.workbench.active_tab().unwrap().session.buffer.to_string(),
            "mine1\nline2\ndisk3\n"
        );
        gui.save_active_tab();
        assert_eq!(fs::read_to_string(&doc).unwrap(), "mine1\nline2\ndisk3\n");

        // Overlapping conflict -> Auto Merge opens Diff & remains Conflict, Use Disk loads disk cleanly (undoable)
        {
            let tab = gui.workbench.active_tab_mut().unwrap();
            tab.edit(|ctx| {
                ctx.set_selection(6, 11);
                ctx.insert_text("local2");
            })
            .unwrap();
        }
        fs::write(&doc, "mine1\ndisk2\ndisk3\n").unwrap();
        assert!(gui.handle_focus_gained());
        gui.conflict_diff_open = false;
        assert!(click_hit_action(&mut gui, |a| matches!(a, UiAction::ResolveConflictAutoMerge)));
        assert!(gui.conflict_diff_open);
        assert_eq!(
            gui.workbench.active_tab().unwrap().session.state,
            DocState::Conflict
        );
        assert!(click_hit_action(&mut gui, |a| matches!(a, UiAction::ResolveConflictUseTheirs)));
        assert_eq!(
            gui.workbench.active_tab().unwrap().session.state,
            DocState::Clean
        );
        assert_eq!(
            gui.workbench.active_tab().unwrap().session.buffer.to_string(),
            "mine1\ndisk2\ndisk3\n"
        );

        // New conflict -> Keep Mine creates backup & surfaces Clean Backups banner button
        {
            let tab = gui.workbench.active_tab_mut().unwrap();
            tab.edit(|ctx| ctx.insert_text("top\n")).unwrap();
        }
        fs::write(&doc, "external-v2\n").unwrap();
        assert!(gui.handle_focus_gained());
        assert!(click_hit_action(&mut gui, |a| matches!(a, UiAction::ResolveConflictKeepMine)));
        assert_eq!(
            gui.workbench.active_tab().unwrap().session.state,
            DocState::Clean
        );
        // Auto Merge, Use Disk, and Keep Mine each left a safety backup for `doc`
        assert!(
            gui.workbench.active_tab().unwrap().conflict_backup_count >= 1,
            "expected at least 1 conflict backup after resolutions"
        );
        assert!(click_hit_action(&mut gui, |a| matches!(a, UiAction::ClearConflictBackups)));
        assert_eq!(
            gui.workbench.active_tab().unwrap().conflict_backup_count,
            0
        );

        // Create another conflict so Save As... hit zone is rendered on the banner, then resolve via MemorySaveDialog
        {
            let tab = gui.workbench.active_tab_mut().unwrap();
            tab.edit(|ctx| ctx.insert_text("local-copy-")).unwrap();
        }
        fs::write(&doc, "external-v3\n").unwrap();
        assert!(gui.handle_focus_gained());
        gui.set_save_dialog(Box::new(MemorySaveDialog::new(Some(save_as_doc.clone()))));
        assert!(click_hit_action(&mut gui, |a| matches!(a, UiAction::SaveAsActive)));
        assert_eq!(
            gui.workbench.active_tab().unwrap().session.path.as_deref(),
            Some(save_as_doc.as_path())
        );
        assert_eq!(
            gui.workbench.active_tab().unwrap().session.state,
            DocState::Clean
        );
        assert!(fs::read_to_string(&save_as_doc)
            .unwrap()
            .contains("local-copy-"));

        // Tab switch immediately flushes named Dirty document
        gui.workbench.new_document();
        let dirty_idx = 0usize;
        assert!(click_hit_action(&mut gui, |a| matches!(a, UiAction::SelectTab(i) if *i == dirty_idx)));
        {
            let tab = gui.workbench.active_tab_mut().unwrap();
            tab.edit(|ctx| {
                ctx.set_selection(0, 0);
                ctx.insert_text("tab-switch-");
            })
            .unwrap();
        }
        assert!(click_hit_action(&mut gui, |a| matches!(a, UiAction::SelectTab(1))));
        assert!(fs::read_to_string(&save_as_doc)
            .unwrap()
            .starts_with("tab-switch-"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_missing_banner_resave_and_close_and_write_error_banner_and_focus_lost() {
        let dir = std::env::temp_dir().join(format!(
            "vellmark-gui-missing-writeerr-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let rec_root = dir.join("rec");
        let doc = dir.join("target.md");
        fs::create_dir_all(&rec_root).unwrap();
        fs::write(&doc, "start\n").unwrap();

        let mut wb = AppWorkbench::new(AppConfig::default(), Some(rec_root));
        wb.open_document(&doc).unwrap();
        let mut gui = GuiController::with_clipboard(wb, Box::new(MemoryClipboard::default()));
        gui.window_size = (1000.0, 700.0);

        // Focus loss immediately flushes named Dirty document
        {
            let tab = gui.workbench.active_tab_mut().unwrap();
            tab.edit(|ctx| ctx.insert_text("focus-")).unwrap();
        }
        assert!(gui.handle_focus_lost());
        assert_eq!(fs::read_to_string(&doc).unwrap(), "focus-start\n");
        assert_eq!(
            gui.workbench.active_tab().unwrap().session.state,
            DocState::Clean
        );

        // Simulated write failure shows WriteError banner with Retry button
        {
            let tab = gui.workbench.active_tab_mut().unwrap();
            tab.edit(|ctx| ctx.insert_text("retry-")).unwrap();
            tab.last_write_error = Some("Simulated EIO".to_string());
        }
        assert!(matches!(
            gui.active_doc_banner_kind(),
            Some(DocBannerKind::WriteError { .. })
        ));
        assert!(click_hit_action(&mut gui, |a| matches!(a, UiAction::RetrySaveActive)));
        assert_eq!(
            gui.workbench.active_tab().unwrap().session.state,
            DocState::Clean
        );
        assert!(gui.workbench.active_tab().unwrap().last_write_error.is_none());
        assert_eq!(fs::read_to_string(&doc).unwrap(), "focus-retry-start\n");

        // Deleting file triggers Missing banner; Re-save recreates original file; Close Tab removes it
        fs::remove_file(&doc).unwrap();
        assert!(gui.handle_focus_gained());
        assert!(matches!(
            gui.active_doc_banner_kind(),
            Some(DocBannerKind::Missing { .. })
        ));
        assert!(click_hit_action(&mut gui, |a| matches!(a, UiAction::ResaveMissingFile)));
        assert_eq!(
            gui.workbench.active_tab().unwrap().session.state,
            DocState::Clean
        );
        assert!(doc.exists());

        fs::remove_file(&doc).unwrap();
        assert!(gui.handle_focus_gained());
        let deleted_id = gui.workbench.active_tab().unwrap().id;
        assert!(click_hit_action(&mut gui, |a| matches!(a, UiAction::ForceCloseActiveMissing)));
        assert!(!gui.workbench.tabs.iter().any(|t| t.id == deleted_id));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_save_as_refuses_path_already_open_in_another_tab() {
        let dir = std::env::temp_dir().join(format!(
            "vellmark-gui-saveas-dup-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let rec_root = dir.join("rec");
        let doc_a = dir.join("a.md");
        let doc_b = dir.join("b.md");
        fs::create_dir_all(&rec_root).unwrap();
        fs::write(&doc_a, "content A\n").unwrap();
        fs::write(&doc_b, "content B\n").unwrap();

        let mut wb = AppWorkbench::new(AppConfig::default(), Some(rec_root));
        wb.open_document(&doc_a).unwrap();
        wb.open_document(&doc_b).unwrap();
        wb.active_tab_idx = Some(0);

        let mut gui = GuiController::with_clipboard(wb, Box::new(MemoryClipboard::default()));
        {
            let tab = gui.workbench.active_tab_mut().unwrap();
            tab.edit(|ctx| ctx.insert_text("edited-")).unwrap();
        }
        gui.set_save_dialog(Box::new(MemorySaveDialog::new(Some(doc_b.clone()))));
        gui.save_as_active_tab();

        // Tab 0 remains bound to doc_a, doc_b on disk stays intact, no duplicate tab paths exist
        assert_eq!(gui.workbench.tabs.len(), 2);
        assert_eq!(
            gui.workbench.tabs[0].session.path.as_deref(),
            Some(doc_a.as_path())
        );
        assert_eq!(
            gui.workbench.tabs[1].session.path.as_deref(),
            Some(doc_b.as_path())
        );
        assert_eq!(fs::read_to_string(&doc_b).unwrap(), "content B\n");
        assert!(gui
            .status_notice
            .as_ref()
            .is_some_and(|(msg, _)| msg.contains("already open")));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_parse_dialog_stdout_path_preserves_trailing_spaces_cr_and_existing_ancestor_dir() {
        assert_eq!(
            parse_dialog_stdout_path(b"/tmp/note .md\n"),
            Some(PathBuf::from("/tmp/note .md"))
        );
        // Must strip ONLY one trailing LF so filenames ending in `\r` (e.g. macOS `Icon\r`)
        // or `\n` are not rewritten into another filename.
        assert_eq!(
            parse_dialog_stdout_path(b"/tmp/Icon\r\n"),
            Some(PathBuf::from("/tmp/Icon\r"))
        );
        assert_eq!(
            parse_dialog_stdout_path(b"/tmp/trailing-lf\n\n"),
            Some(PathBuf::from("/tmp/trailing-lf\n"))
        );
        assert_eq!(parse_dialog_stdout_path(b"\n"), None);
        assert_eq!(parse_dialog_stdout_path(b""), None);

        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            let raw = b"/tmp/nonutf8-\xff-name \n";
            let parsed = parse_dialog_stdout_path(raw).unwrap();
            assert_eq!(parsed.as_os_str().as_bytes(), b"/tmp/nonutf8-\xff-name ");
        }

        let dir = std::env::temp_dir().join(format!(
            "vellmark-gui-dialog-dir-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        let deleted_child = dir.join("missing_subdir").join("file.md");
        assert_eq!(existing_default_dialog_dir(Some(&deleted_child)), Some(dir.as_path()));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_crlf_enter_navigation_delete_and_autosave_preserves_crlf_without_bare_lf() {
        let dir = std::env::temp_dir().join(format!(
            "vellmark-gui-crlf-enter-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let rec_dir = dir.join("recovery");
        fs::create_dir_all(&rec_dir).unwrap();
        let doc = dir.join("win.md");
        fs::write(&doc, b"# Title\r\n\r\nBody line\r\n").unwrap();

        let mut wb = AppWorkbench::new(AppConfig::default(), Some(rec_dir));
        wb.open_document(&doc).unwrap();
        let mut gui = GuiController::with_clipboard(wb, Box::new(MemoryClipboard::default()));

        let body_start = "# Title\r\n\r\n".len();
        gui.workbench
            .active_tab_mut()
            .unwrap()
            .edit(|ctx| ctx.set_caret(body_start))
            .unwrap();

        // Press Enter via GUI editor dispatch
        gui.dispatch_editor(EditorAction::Enter);
        assert_eq!(
            gui.workbench.active_tab().unwrap().session.buffer.to_string(),
            "# Title\r\n\r\n\r\nBody line\r\n"
        );

        // Wait past the 2s auto-save idle window and drain SaveWorker through GuiController runtime
        let future = Instant::now() + Duration::from_secs(3);
        let _ = gui.tick_runtime_at(future);
        let deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < deadline {
            if gui
                .workbench
                .active_tab()
                .is_some_and(|t| t.session.state == DocState::Clean)
            {
                break;
            }
            let _ = gui.tick_runtime_at(future);
            std::thread::sleep(Duration::from_millis(5));
        }

        assert_eq!(
            gui.workbench.active_tab().unwrap().session.state,
            DocState::Clean
        );
        assert!(gui.active_doc_banner_kind().is_none());
        let saved_bytes = fs::read(&doc).unwrap();
        assert_eq!(saved_bytes, b"# Title\r\n\r\n\r\nBody line\r\n");
        let bare_lf = saved_bytes
            .iter()
            .copied()
            .enumerate()
            .filter(|&(idx, b)| b == b'\n' && (idx == 0 || saved_bytes[idx - 1] != b'\r'))
            .count();
        assert_eq!(bare_lf, 0);

        // ArrowRight / ArrowLeft across `\r\n` must skip the 2-byte CRLF pair atomically,
        // and Delete forward before `\r\n` must delete both `\r` and `\n`.
        gui.workbench
            .active_tab_mut()
            .unwrap()
            .edit(|ctx| ctx.set_caret("# Title".len()))
            .unwrap();
        gui.move_caret_horizontal(1, false);
        assert_eq!(
            gui.workbench.active_tab().unwrap().view.selection.head,
            "# Title\r\n".len()
        );
        gui.move_caret_horizontal(-1, false);
        assert_eq!(
            gui.workbench.active_tab().unwrap().view.selection.head,
            "# Title".len()
        );
        gui.delete_forward();
        let after_del = gui.workbench.active_tab().unwrap().session.buffer.to_string();
        assert_eq!(after_del, "# Title\r\n\r\nBody line\r\n");

        // Deleting the last `\r\n` in a CRLF file (`A\r\nB`) still preserves `CrLf` on next Enter + auto-save
        let single_eol = dir.join("single_eol.md");
        fs::write(&single_eol, b"A\r\nB").unwrap();
        gui.workbench.open_document(&single_eol).unwrap();
        gui.workbench
            .active_tab_mut()
            .unwrap()
            .edit(|ctx| ctx.set_caret("A\r\n".len()))
            .unwrap();
        gui.dispatch_editor(EditorAction::Backspace);
        assert_eq!(
            gui.workbench.active_tab().unwrap().session.buffer.to_string(),
            "AB"
        );
        gui.dispatch_editor(EditorAction::Enter);
        assert_eq!(
            gui.workbench.active_tab().unwrap().session.buffer.to_string(),
            "A\r\nB"
        );
        let future2 = Instant::now() + Duration::from_secs(3);
        let _ = gui.tick_runtime_at(future2);
        let dl2 = Instant::now() + Duration::from_secs(2);
        while Instant::now() < dl2 {
            if gui
                .workbench
                .active_tab()
                .is_some_and(|t| t.session.state == DocState::Clean)
            {
                break;
            }
            let _ = gui.tick_runtime_at(future2);
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(fs::read(&single_eol).unwrap(), b"A\r\nB");

        let _ = fs::remove_dir_all(&dir);
    }
}
