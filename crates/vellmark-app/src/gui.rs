//! Native cross-platform desktop GUI shell (`winit 0.30` + `wgpu` Metal/Vulkan/DX12),
//! coordinating virtualized editor rendering, OS IME composition, mouse/keyboard interaction,
//! sidebar outline & file tree navigation, multi-tab header, find overlay, and live themes.

use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};
#[cfg(unix)]
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use vellmark_core::{
    AutoMergeOutcome, ByteRange, ConflictActionOutcome, ConflictDiffKind, DetectedEncoding,
    DocState, DocumentStats, EditorAction, EncodingDecision, ExternalSyncResult, FindMatch,
    FindOptions, HistoricalBackupEntry, LineEnding, SaveOutcome, Selection, TableAlign, ViewMode,
    atomic_write_file, count_text_metrics, paths_share_underlying_file,
};
use vellmark_export::HtmlExportMode;
use vellmark_parser::{
    BlockKind, InlineKind, OutlineItem, active_outline_index, extract_link_destination,
    parse_table_block,
};
use vellmark_render::{
    CodeBlockButtonKind, CodeLanguageEntry, DrawList, EditorHitTarget, EditorSceneResult,
    EditorViewport, ImageLoadStatus, Rect, ResolvedThemeStyle, SceneRect, ShapedDocumentCache,
    TableButtonKind, code_language_catalog, code_language_display_label,
    normalize_fence_language_token, shape_ui_text_into,
};
use vellmark_theme::BuiltinTheme;
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalPosition, LogicalSize, PhysicalPosition};
use winit::event::{ElementState, Ime, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::keyboard::{Key, KeyCode, ModifiersState, NamedKey, PhysicalKey};
use winit::window::{
    CursorIcon, Fullscreen, Theme as WindowTheme, Window, WindowAttributes, WindowId,
};

use crate::assets::is_supported_image_extension;
use crate::clipboard::{ClipboardBackend, NativeClipboard};
use crate::gpu::WgpuRenderer;
use crate::images::ImageCache;
use crate::search::{
    GLOBAL_SEARCH_MAX_RESULTS, QuickOpenMatch, WorkspaceSearchMatch, WorkspaceSearchRequest,
    WorkspaceSearchWorker,
};
use crate::workbench::{AppWorkbench, DocumentTab, WorkbenchRuntimeEvent};

const TAB_BAR_H: f32 = 36.0;
const STATUS_BAR_H: f32 = 26.0;
const SIDEBAR_W: f32 = 248.0;
const FIND_BAR_H: f32 = 38.0;
const ALERT_BANNER_H: f32 = 42.0;
const CLEANUP_BANNER_H: f32 = 30.0;
const BLINK_INTERVAL: Duration = Duration::from_millis(530);
const EXTERNAL_POLL_INTERVAL: Duration = Duration::from_millis(1500);
const GLOBAL_SEARCH_DEBOUNCE: Duration = Duration::from_millis(120);

/// Persistent warning displayed when the recovery directory cannot be written (PRD 4.5.1 F).
pub const RECOVERY_DEGRADED_NOTICE: &str =
    "Crash recovery unavailable (dir unwritable) — 3s recovery target degraded";

/// Abstract native file/folder dialog backend (`osascript`/native dialog at runtime; deterministic mock in tests).
pub trait SaveDialogBackend: Send + Sync {
    fn pick_save_as_path(&self, current_path: Option<&Path>, default_name: &str)
    -> Option<PathBuf>;

    /// Pick an existing Markdown/text document to open (`Cmd/Ctrl + O`).
    fn pick_open_path(&self, _current_dir: Option<&Path>) -> Option<PathBuf> {
        None
    }

    /// Pick a workspace folder to open (`Cmd/Ctrl + Shift + O`).
    fn pick_open_folder(&self, _current_dir: Option<&Path>) -> Option<PathBuf> {
        None
    }
}

/// Non-interactive Save-As backend for headless tests (`with_clipboard` default).
#[derive(Debug, Default)]
pub struct NoopSaveDialog;

impl SaveDialogBackend for NoopSaveDialog {
    fn pick_save_as_path(
        &self,
        _current_path: Option<&Path>,
        _default_name: &str,
    ) -> Option<PathBuf> {
        None
    }
}

/// Deterministic in-memory file/folder dialog backend for unit tests.
#[derive(Debug, Default)]
pub struct MemorySaveDialog {
    next_path: Mutex<Option<PathBuf>>,
    next_open_path: Mutex<Option<PathBuf>>,
    next_open_folder: Mutex<Option<PathBuf>>,
}

impl MemorySaveDialog {
    pub fn new(path: Option<PathBuf>) -> Self {
        Self {
            next_path: Mutex::new(path),
            next_open_path: Mutex::new(None),
            next_open_folder: Mutex::new(None),
        }
    }

    pub fn set_next_path(&self, path: Option<PathBuf>) {
        if let Ok(mut g) = self.next_path.lock() {
            *g = path;
        }
    }

    /// Configure the path returned by the next `pick_open_path` call.
    pub fn set_next_open_path(&self, path: Option<PathBuf>) {
        if let Ok(mut g) = self.next_open_path.lock() {
            *g = path;
        }
    }

    /// Configure the folder returned by the next `pick_open_folder` call.
    pub fn set_next_open_folder(&self, path: Option<PathBuf>) {
        if let Ok(mut g) = self.next_open_folder.lock() {
            *g = path;
        }
    }
}

impl SaveDialogBackend for MemorySaveDialog {
    fn pick_save_as_path(
        &self,
        _current_path: Option<&Path>,
        _default_name: &str,
    ) -> Option<PathBuf> {
        self.next_path.lock().ok().and_then(|g| g.clone())
    }

    fn pick_open_path(&self, _current_dir: Option<&Path>) -> Option<PathBuf> {
        self.next_open_path.lock().ok().and_then(|g| g.clone())
    }

    fn pick_open_folder(&self, _current_dir: Option<&Path>) -> Option<PathBuf> {
        self.next_open_folder.lock().ok().and_then(|g| g.clone())
    }
}

#[cfg(any(unix, test))]
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

#[cfg(any(target_os = "macos", test))]
fn existing_default_dialog_dir(current_path: Option<&Path>) -> Option<&Path> {
    existing_dir_or_ancestor(current_path, true)
}

#[cfg(any(unix, test))]
fn existing_dir_or_ancestor(path: Option<&Path>, start_from_parent: bool) -> Option<&Path> {
    let start = if start_from_parent {
        path.and_then(|p| p.parent())?
    } else {
        path?
    };
    start
        .ancestors()
        .find(|p| !p.as_os_str().is_empty() && p.is_dir())
}

#[cfg(target_os = "macos")]
fn run_osascript_pick(script: &str, args: &[&str]) -> Option<PathBuf> {
    let mut cmd = Command::new("osascript");
    cmd.arg("-e").arg(script);
    for arg in args {
        cmd.arg(arg);
    }
    let out = cmd.output().ok()?;
    if !out.status.success() {
        return None;
    }
    parse_dialog_stdout_path(&out.stdout)
}

#[cfg(all(unix, not(target_os = "macos")))]
fn run_zenity_pick(
    extra_args: &[&str],
    current_path: Option<&Path>,
    fallback_name: Option<&str>,
) -> Option<PathBuf> {
    let mut cmd = Command::new("zenity");
    cmd.arg("--file-selection");
    for arg in extra_args {
        cmd.arg(arg);
    }
    if let Some(cur) = current_path {
        cmd.arg("--filename").arg(cur);
    } else if let Some(name) = fallback_name {
        cmd.arg("--filename").arg(name);
    }
    let out = cmd.output().ok()?;
    if !out.status.success() {
        return None;
    }
    parse_dialog_stdout_path(&out.stdout)
}

/// Native desktop file & folder picker adapter.
#[derive(Debug, Default)]
pub struct NativeSaveDialog;

impl SaveDialogBackend for NativeSaveDialog {
    fn pick_save_as_path(
        &self,
        current_path: Option<&Path>,
        default_name: &str,
    ) -> Option<PathBuf> {
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
            run_osascript_pick(script, &[default_name, &dir_arg])
        }
        #[cfg(all(unix, not(target_os = "macos")))]
        {
            run_zenity_pick(
                &["--save", "--confirm-overwrite"],
                current_path,
                Some(default_name),
            )
        }
        #[cfg(not(unix))]
        {
            let _ = (current_path, default_name);
            None
        }
    }

    fn pick_open_path(&self, current_dir: Option<&Path>) -> Option<PathBuf> {
        #[cfg(target_os = "macos")]
        {
            let dir_arg = existing_dir_or_ancestor(current_dir, false)
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_default();
            let script = "on run argv\n\
                set defDir to item 1 of argv\n\
                if defDir is not \"\" then\n\
                    set chosen to choose file of type {\"md\", \"markdown\", \"txt\", \"public.plain-text\"} with prompt \"Open Markdown File\" default location (POSIX file defDir)\n\
                else\n\
                    set chosen to choose file of type {\"md\", \"markdown\", \"txt\", \"public.plain-text\"} with prompt \"Open Markdown File\"\n\
                end if\n\
                return POSIX path of chosen\n\
            end run";
            run_osascript_pick(script, &[&dir_arg])
        }
        #[cfg(all(unix, not(target_os = "macos")))]
        {
            let dir = existing_dir_or_ancestor(current_dir, false);
            run_zenity_pick(
                &["--file-filter=Markdown | *.md *.markdown *.txt"],
                dir,
                None,
            )
        }
        #[cfg(not(unix))]
        {
            let _ = current_dir;
            None
        }
    }

    fn pick_open_folder(&self, current_dir: Option<&Path>) -> Option<PathBuf> {
        #[cfg(target_os = "macos")]
        {
            let dir_arg = existing_dir_or_ancestor(current_dir, false)
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_default();
            let script = "on run argv\n\
                set defDir to item 1 of argv\n\
                if defDir is not \"\" then\n\
                    set chosen to choose folder with prompt \"Open Folder\" default location (POSIX file defDir)\n\
                else\n\
                    set chosen to choose folder with prompt \"Open Folder\"\n\
                end if\n\
                return POSIX path of chosen\n\
            end run";
            run_osascript_pick(script, &[&dir_arg])
        }
        #[cfg(all(unix, not(target_os = "macos")))]
        {
            let dir = existing_dir_or_ancestor(current_dir, false);
            run_zenity_pick(&["--directory"], dir, None)
        }
        #[cfg(not(unix))]
        {
            let _ = current_dir;
            None
        }
    }
}

/// Abstract external URL/mailto opener (`open`/`xdg-open`/`start` natively; mock in tests).
pub trait LinkOpenerBackend: Send + Sync {
    fn open_url(&self, url: &str) -> Result<(), String>;
}

/// No-op link opener used as the default in headless tests.
#[derive(Debug, Default)]
pub struct NoopLinkOpener;

impl LinkOpenerBackend for NoopLinkOpener {
    fn open_url(&self, _url: &str) -> Result<(), String> {
        Ok(())
    }
}

/// Deterministic in-memory link opener recording opened external URLs for unit tests.
#[derive(Debug, Default)]
pub struct MemoryLinkOpener {
    opened: Mutex<Vec<String>>,
    fail_with: Mutex<Option<String>>,
}

impl MemoryLinkOpener {
    pub fn opened_urls(&self) -> Vec<String> {
        self.opened.lock().map(|g| g.clone()).unwrap_or_default()
    }

    pub fn set_error(&self, err: Option<String>) {
        if let Ok(mut g) = self.fail_with.lock() {
            *g = err;
        }
    }
}

impl LinkOpenerBackend for MemoryLinkOpener {
    fn open_url(&self, url: &str) -> Result<(), String> {
        if let Some(err) = self.fail_with.lock().ok().and_then(|g| g.clone()) {
            return Err(err);
        }
        if let Ok(mut g) = self.opened.lock() {
            g.push(url.to_string());
        }
        Ok(())
    }
}

/// Native OS external link launcher (`open` on macOS, `xdg-open` on Linux, `ShellExecuteW` on Windows).
#[derive(Debug, Default)]
pub struct NativeLinkOpener;

impl LinkOpenerBackend for NativeLinkOpener {
    fn open_url(&self, url: &str) -> Result<(), String> {
        validate_native_open_target(url)?;
        #[cfg(target_os = "macos")]
        let spawn_res = Command::new("open").arg(url).spawn();
        #[cfg(all(unix, not(target_os = "macos")))]
        let spawn_res = Command::new("xdg-open").arg(url).spawn();
        #[cfg(windows)]
        return open_with_windows_shell_execute(url);
        #[cfg(not(windows))]
        spawn_res
            .map(|_| ())
            .map_err(|e| format!("Failed to open link: {e}"))
    }
}

/// Extract an RFC 3986 URI scheme when `value` starts with `scheme:` (requiring >= 2 chars
/// so single-letter Windows drive prefixes like `C:\...` are never misclassified as schemes).
fn extract_uri_scheme(value: &str) -> Option<&str> {
    let bytes = value.as_bytes();
    let colon = bytes.iter().position(|&b| b == b':')?;
    if colon < 2 || !bytes[0].is_ascii_alphabetic() {
        return None;
    }
    if bytes[1..colon]
        .iter()
        .all(|&b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'-' | b'.'))
    {
        Some(&value[..colon])
    } else {
        None
    }
}

fn is_supported_external_scheme(scheme: &str) -> bool {
    scheme.eq_ignore_ascii_case("http")
        || scheme.eq_ignore_ascii_case("https")
        || scheme.eq_ignore_ascii_case("mailto")
}

fn contains_uri_unwise_or_control(url: &str) -> bool {
    url.chars().any(|ch| {
        ch.is_control() || matches!(ch, '<' | '>' | '"' | '{' | '}' | '|' | '\\' | '^' | '`')
    })
}

fn is_valid_http_authority(authority: &str) -> bool {
    if authority.is_empty() {
        return false;
    }
    // Reject whitespace and shell/delimiter metacharacters inside host/userinfo authority
    // (defense-in-depth for `https://x&calc`), while keeping query-string `&` (`?a=1&b=2`).
    if authority.chars().any(|ch| {
        ch.is_whitespace()
            || matches!(
                ch,
                '&' | '|'
                    | '<'
                    | '>'
                    | '^'
                    | '!'
                    | ';'
                    | '`'
                    | '\\'
                    | '"'
                    | '\''
                    | '$'
                    | '('
                    | ')'
                    | '*'
                    | ','
                    | '='
            )
    }) {
        return false;
    }
    let host_port = match authority.split_once('@') {
        Some((userinfo, rest)) => {
            if userinfo.is_empty() || rest.contains('@') {
                return false;
            }
            rest
        }
        None => authority,
    };
    if host_port.is_empty() {
        return false;
    }
    if host_port.starts_with('[') {
        let Some(end_bracket) = host_port.find(']') else {
            return false;
        };
        if end_bracket <= 1 {
            return false;
        }
        let after = &host_port[end_bracket + 1..];
        return after.is_empty()
            || after
                .strip_prefix(':')
                .is_some_and(|port| !port.is_empty() && port.bytes().all(|b| b.is_ascii_digit()));
    }
    if let Some((host, port)) = host_port.rsplit_once(':') {
        return !host.is_empty() && !port.is_empty() && port.bytes().all(|b| b.is_ascii_digit());
    }
    true
}

fn has_valid_percent_encoding(s: &str) -> bool {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            if i + 2 >= bytes.len()
                || !bytes[i + 1].is_ascii_hexdigit()
                || !bytes[i + 2].is_ascii_hexdigit()
            {
                return false;
            }
            i += 3;
        } else {
            i += 1;
        }
    }
    true
}

fn is_valid_mailto_url(after_scheme: &str) -> bool {
    // Product policy: reject bare `mailto:` with neither recipient nor query headers
    // (RFC 6068 §2 grammar technically allows both to be omitted).
    // Also reject unencoded `#` anywhere in `mailto:` because `#` starts a URI fragment
    // and would silently truncate recipient/header data when opened by mail clients.
    if after_scheme.is_empty() || after_scheme.contains('#') {
        return false;
    }
    let (to, query_opt) = match after_scheme.split_once('?') {
        Some((to, query)) => (to, Some(query)),
        None => (after_scheme, None),
    };
    // Per RFC 6068 §2, valid mailbox characters like `'`, `!`, `$`, `+`, `,` are allowed
    // unencoded in `to`, whereas whitespace, invalid `%` escapes, and unencoded URI delimiters
    // (`&`, `=`, `;`, `/`, `[`, `]`) are rejected.
    if !to.is_empty()
        && (!has_valid_percent_encoding(to)
            || to
                .chars()
                .any(|ch| ch.is_whitespace() || matches!(ch, '&' | '=' | ';' | '/' | '[' | ']')))
    {
        return false;
    }
    if let Some(query) = query_opt {
        if query.is_empty() || !has_valid_percent_encoding(query) {
            return false;
        }
        return query.split('&').all(|field| {
            field.split_once('=').is_some_and(|(name, _value)| {
                !name.is_empty()
                    && !name
                        .chars()
                        .any(|ch| ch.is_whitespace() || matches!(ch, '?' | '/' | '[' | ']' | ';'))
            })
        });
    }
    true
}

fn is_allowed_external_url(url: &str) -> bool {
    if contains_uri_unwise_or_control(url) {
        return false;
    }
    for prefix in ["http://", "https://"] {
        if has_ascii_prefix_ignore_case(url, prefix) {
            let rest = &url[prefix.len()..];
            let auth_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
            return is_valid_http_authority(&rest[..auth_end]);
        }
    }
    if has_ascii_prefix_ignore_case(url, "mailto:") {
        return is_valid_mailto_url(&url["mailto:".len()..]);
    }
    false
}

fn is_blocked_executable_path(path: &Path) -> bool {
    let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
        return false;
    };
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "exe"
            | "bat"
            | "cmd"
            | "com"
            | "msi"
            | "scr"
            | "hta"
            | "vbs"
            | "vbe"
            | "js"
            | "jse"
            | "wsf"
            | "wsh"
            | "ps1"
            | "psm1"
            | "pif"
            | "lnk"
            | "reg"
    )
}

fn parse_local_file_uri_path(after_scheme: &str) -> Option<String> {
    let local_slash = if after_scheme.starts_with('/') {
        after_scheme
    } else if has_ascii_prefix_ignore_case(after_scheme, "localhost/") {
        &after_scheme["localhost".len()..]
    } else {
        return None;
    };
    if local_slash.len() <= 1 {
        return None;
    }
    #[cfg(windows)]
    {
        let b = local_slash.as_bytes();
        if b.len() >= 3 && b[0] == b'/' && b[1].is_ascii_alphabetic() && b[2] == b':' {
            return Some(local_slash[1..].to_string());
        }
    }
    Some(local_slash.to_string())
}

fn validate_native_open_target(target: &str) -> Result<(), String> {
    if target.is_empty() || target.chars().any(char::is_control) {
        return Err("Invalid link target".to_string());
    }
    if let Some(scheme) = extract_uri_scheme(target) {
        if !is_supported_external_scheme(scheme) {
            return Err(format!("Blocked unsupported URL scheme: {scheme}"));
        }
        if !is_allowed_external_url(target) {
            return Err("Blocked unsafe or malformed external URL".to_string());
        }
        return Ok(());
    }
    if is_blocked_executable_path(Path::new(target)) {
        return Err("Blocked opening executable file directly".to_string());
    }
    Ok(())
}

#[cfg(windows)]
fn open_with_windows_shell_execute(target: &str) -> Result<(), String> {
    use std::ffi::OsStr;
    use std::os::windows::ffi::OsStrExt;

    #[link(name = "shell32")]
    unsafe extern "system" {
        fn ShellExecuteW(
            hwnd: *mut core::ffi::c_void,
            lp_operation: *const u16,
            lp_file: *const u16,
            lp_parameters: *const u16,
            lp_directory: *const u16,
            n_show_cmd: i32,
        ) -> isize;
    }

    const SW_SHOWNORMAL: i32 = 1;
    let verb: Vec<u16> = OsStr::new("open\0").encode_wide().collect();
    let mut file_wide: Vec<u16> = OsStr::new(target).encode_wide().collect();
    file_wide.push(0);

    let res = unsafe {
        ShellExecuteW(
            core::ptr::null_mut(),
            verb.as_ptr(),
            file_wide.as_ptr(),
            core::ptr::null(),
            core::ptr::null(),
            SW_SHOWNORMAL,
        )
    };
    if res > 32 {
        Ok(())
    } else {
        Err(format!("Failed to open link (ShellExecuteW error {res})"))
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
| Feature | Shortcut / Syntax | Status |\n\
| :--- | :---: | ---: |\n\
| Bold & Italic | `**bold**` / `*italic*` | **Ready** |\n\
| Live Table Grid | `Tab` / `Enter` in cells | **Ready** |\n\
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
    Recent,
}

#[derive(Debug, Clone)]
enum UiAction {
    SelectTab(usize),
    CloseTab(u64),
    NewTab,
    ToggleMode,
    ToggleFocusMode,
    ToggleTypewriterMode,
    CycleTheme,
    ToggleSidebar,
    SwitchSidebarSection(SidebarSection),
    JumpOutline(usize),
    OpenWorkspaceFile(PathBuf),
    OpenRecentFile(PathBuf),
    OpenFolderPrompt,
    OpenNewFilePrompt,
    OpenRenameFilePrompt,
    ArmOrConfirmDeleteFile,
    ConfirmSimplePrompt,
    CloseSimplePrompt,
    FocusSimplePromptInput,
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
    CloseLinkPrompt,
    ConfirmLinkPrompt,
    FocusLinkInput,
    CloseQuickOpen,
    SelectQuickOpenMatch(usize),
    FocusQuickOpenInput,
    OpenCodeLangPicker {
        block_range: ByteRange,
        current_language: String,
    },
    ToggleCodeLineNumbers,
    CloseCodeLangPicker,
    SelectCodeLangMatch(usize),
    FocusCodeLangInput,
    TableToolbarAction(TableButtonKind),
    CloseGlobalSearch,
    SelectGlobalSearchMatch(usize),
    FocusGlobalSearchInput,
    ToggleGlobalSearchCase,
    ToggleFindWholeWord,
    ToggleReplaceRow,
    FocusReplaceInput,
    ReplaceNext,
    ReplaceAll,
    SaveUtf8Copy,
    CloseCommandPalette,
    SelectCommandPaletteMatch(usize),
    FocusCommandPaletteInput,
    CloseInspectModal,
    FocusInspectModalInput,
    SelectInspectModalRow(usize),
    ReloadCustomTheme,
    DeleteSelectedRecoveryBackup,
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
enum FilePromptMode {
    CreateFile { parent_dir: PathBuf },
    RenameFile { source_path: PathBuf },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InspectModalKind {
    ThemeDiagnostics,
    RecoveryBackups,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PaletteCommandId {
    FileNew,
    FileOpen,
    FileOpenFolder,
    FileSave,
    FileSaveAs,
    ManageRecoveryBackups,
    ExportHtmlSafe,
    ExportHtmlRaw,
    QuickOpen,
    GlobalSearch,
    Find,
    FindAndReplace,
    ToggleSourceMode,
    ToggleFocusMode,
    ToggleTypewriterMode,
    ToggleFullscreen,
    ToggleSidebar,
    InsertTable,
    InsertCodeBlock,
    InsertLink,
    ToggleTableAutoAlign,
    ToggleCodeLineNumbers,
    ToggleNetworkImages,
    ThemeFollowSystem,
    ThemeLight,
    ThemeDark,
    ThemeAcademic,
    ThemeGithub,
    LoadCustomThemeCss,
    ShowThemeDiagnostics,
    ProseWidthNarrow,
    ProseWidthBalanced,
    ProseWidthWide,
    ZoomIn,
    ZoomOut,
    ZoomReset,
}

#[derive(Debug, Clone)]
struct PaletteCommandEntry {
    id: PaletteCommandId,
    label: &'static str,
    shortcut: &'static str,
}

const PALETTE_COMMANDS: &[PaletteCommandEntry] = &[
    PaletteCommandEntry {
        id: PaletteCommandId::FileNew,
        label: "File: New Document",
        shortcut: "Cmd/Ctrl+N",
    },
    PaletteCommandEntry {
        id: PaletteCommandId::FileOpen,
        label: "File: Open File...",
        shortcut: "Cmd/Ctrl+O",
    },
    PaletteCommandEntry {
        id: PaletteCommandId::FileOpenFolder,
        label: "File: Open Folder...",
        shortcut: "Cmd/Ctrl+Shift+O",
    },
    PaletteCommandEntry {
        id: PaletteCommandId::FileSave,
        label: "File: Save",
        shortcut: "Cmd/Ctrl+S",
    },
    PaletteCommandEntry {
        id: PaletteCommandId::FileSaveAs,
        label: "File: Save As...",
        shortcut: "Cmd/Ctrl+Shift+S",
    },
    PaletteCommandEntry {
        id: PaletteCommandId::ManageRecoveryBackups,
        label: "File: Manage Recovery & Safety Backups...",
        shortcut: "",
    },
    PaletteCommandEntry {
        id: PaletteCommandId::ExportHtmlSafe,
        label: "Export: Standalone HTML (Safe Mode)...",
        shortcut: "",
    },
    PaletteCommandEntry {
        id: PaletteCommandId::ExportHtmlRaw,
        label: "Export: Standalone HTML (Raw)...",
        shortcut: "",
    },
    PaletteCommandEntry {
        id: PaletteCommandId::QuickOpen,
        label: "Go: Quick Open File...",
        shortcut: "Cmd/Ctrl+P",
    },
    PaletteCommandEntry {
        id: PaletteCommandId::GlobalSearch,
        label: "Search: Find in Workspace...",
        shortcut: "Cmd/Ctrl+Shift+F",
    },
    PaletteCommandEntry {
        id: PaletteCommandId::Find,
        label: "Edit: Find in Document",
        shortcut: "Cmd/Ctrl+F",
    },
    PaletteCommandEntry {
        id: PaletteCommandId::FindAndReplace,
        label: "Edit: Find and Replace",
        shortcut: "Cmd+Option+F / Ctrl+H",
    },
    PaletteCommandEntry {
        id: PaletteCommandId::InsertTable,
        label: "Insert: Table...",
        shortcut: "Cmd/Ctrl+T",
    },
    PaletteCommandEntry {
        id: PaletteCommandId::InsertCodeBlock,
        label: "Insert / Edit: Code Block",
        shortcut: "Cmd+Option+C / Ctrl+Shift+K",
    },
    PaletteCommandEntry {
        id: PaletteCommandId::InsertLink,
        label: "Insert: Hyperlink...",
        shortcut: "Cmd/Ctrl+K",
    },
    PaletteCommandEntry {
        id: PaletteCommandId::ToggleSourceMode,
        label: "View: Toggle Source / WYSIWYG Mode",
        shortcut: "Cmd/Ctrl+/",
    },
    PaletteCommandEntry {
        id: PaletteCommandId::ToggleFocusMode,
        label: "View: Toggle Focus Mode",
        shortcut: "F8",
    },
    PaletteCommandEntry {
        id: PaletteCommandId::ToggleTypewriterMode,
        label: "View: Toggle Typewriter Mode",
        shortcut: "F9",
    },
    PaletteCommandEntry {
        id: PaletteCommandId::ToggleFullscreen,
        label: "View: Toggle Fullscreen",
        shortcut: "F11",
    },
    PaletteCommandEntry {
        id: PaletteCommandId::ToggleSidebar,
        label: "View: Toggle Sidebar",
        shortcut: "Cmd/Ctrl+\\",
    },
    PaletteCommandEntry {
        id: PaletteCommandId::ToggleTableAutoAlign,
        label: "Table: Toggle Auto-Align Source on Edit",
        shortcut: "",
    },
    PaletteCommandEntry {
        id: PaletteCommandId::ToggleCodeLineNumbers,
        label: "Editor: Toggle Code Block Line Numbers",
        shortcut: "Cmd/Ctrl+Shift+L",
    },
    PaletteCommandEntry {
        id: PaletteCommandId::ToggleNetworkImages,
        label: "Images: Toggle Allow Network Images",
        shortcut: "",
    },
    PaletteCommandEntry {
        id: PaletteCommandId::ThemeFollowSystem,
        label: "Theme: Follow System Appearance",
        shortcut: "",
    },
    PaletteCommandEntry {
        id: PaletteCommandId::ThemeLight,
        label: "Theme: Light",
        shortcut: "Cmd/Ctrl+Shift+T",
    },
    PaletteCommandEntry {
        id: PaletteCommandId::ThemeDark,
        label: "Theme: Dark",
        shortcut: "",
    },
    PaletteCommandEntry {
        id: PaletteCommandId::ThemeAcademic,
        label: "Theme: Academic",
        shortcut: "",
    },
    PaletteCommandEntry {
        id: PaletteCommandId::ThemeGithub,
        label: "Theme: GitHub",
        shortcut: "",
    },
    PaletteCommandEntry {
        id: PaletteCommandId::LoadCustomThemeCss,
        label: "Theme: Load Custom CSS File...",
        shortcut: "",
    },
    PaletteCommandEntry {
        id: PaletteCommandId::ShowThemeDiagnostics,
        label: "Theme: Show Theme Diagnostics...",
        shortcut: "",
    },
    PaletteCommandEntry {
        id: PaletteCommandId::ProseWidthNarrow,
        label: "Layout: Prose Width Compact (720px)",
        shortcut: "",
    },
    PaletteCommandEntry {
        id: PaletteCommandId::ProseWidthBalanced,
        label: "Layout: Prose Width Balanced (860px)",
        shortcut: "",
    },
    PaletteCommandEntry {
        id: PaletteCommandId::ProseWidthWide,
        label: "Layout: Prose Width Wide (1080px)",
        shortcut: "",
    },
    PaletteCommandEntry {
        id: PaletteCommandId::ZoomIn,
        label: "View: Zoom In",
        shortcut: "Cmd/Ctrl+=",
    },
    PaletteCommandEntry {
        id: PaletteCommandId::ZoomOut,
        label: "View: Zoom Out",
        shortcut: "Cmd/Ctrl+-",
    },
    PaletteCommandEntry {
        id: PaletteCommandId::ZoomReset,
        label: "View: Reset Zoom (100%)",
        shortcut: "Cmd/Ctrl+0",
    },
];

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
    ReadOnlyEncoding {
        candidate: DetectedEncoding,
    },
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
    selection_range: ByteRange,
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
    ReplaceBox,
    LinkPromptBox,
    QuickOpenBox,
    CodeLangBox,
    GlobalSearchBox,
    CommandPaletteBox,
    InspectModalBox,
    InsertTableBox,
    FileNameBox,
}

/// Warning shown in the status bar whenever protected crash snapshots exceed the 500 MB budget.
pub const PROTECTED_SNAPSHOT_LIMIT_NOTICE: &str =
    "Protected recovery snapshots exceed 500 MB — please save or discard old drafts";

/// Platform-independent GUI controller owning layout state, shortcuts, IME state,
/// find/quick-open/link overlays, and virtualized scene construction (fully testable headlessly).
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
    pub replace_open: bool,
    pub replace_focused: bool,
    pub replace_text: String,
    pub link_prompt_open: bool,
    pub link_prompt_url: String,
    pub quick_open_open: bool,
    pub quick_open_query: String,
    pub quick_open_matches: Vec<QuickOpenMatch>,
    pub quick_open_selected: usize,
    pub global_search_open: bool,
    pub global_search_query: String,
    pub global_search_case_sensitive: bool,
    pub global_search_matches: Vec<WorkspaceSearchMatch>,
    pub global_search_selected: usize,
    pub global_search_searching: bool,
    global_search_debounce_deadline: Option<Instant>,
    global_search_req_id: u64,
    global_search_cancel: Option<Arc<AtomicBool>>,
    global_search_worker: Option<WorkspaceSearchWorker>,
    redraw_notifier: Option<Arc<dyn Fn() + Send + Sync>>,
    pub command_palette_open: bool,
    pub command_palette_query: String,
    command_palette_matches: Vec<PaletteCommandEntry>,
    command_palette_selected: usize,
    pub inspect_modal: Option<InspectModalKind>,
    pub inspect_modal_query: String,
    inspect_modal_selected: usize,
    inspect_recovery_backups: Vec<HistoricalBackupEntry>,
    insert_table_open: bool,
    insert_table_input: String,
    file_prompt_mode: Option<FilePromptMode>,
    file_prompt_input: String,
    selected_workspace_path: Option<PathBuf>,
    delete_confirm_path: Option<PathBuf>,
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
    link_opener: Box<dyn LinkOpenerBackend>,
    image_cache: ImageCache,
    code_lang_open: bool,
    code_lang_block_range: Option<ByteRange>,
    code_lang_query: String,
    code_lang_matches: Vec<&'static CodeLanguageEntry>,
    code_lang_selected: usize,
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
        let mut ctrl = Self::with_backends(
            workbench,
            Box::new(NativeClipboard::new()),
            Box::new(NativeSaveDialog),
        );
        ctrl.link_opener = Box::new(NativeLinkOpener);
        ctrl
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
            replace_open: false,
            replace_focused: false,
            replace_text: String::new(),
            link_prompt_open: false,
            link_prompt_url: String::new(),
            quick_open_open: false,
            quick_open_query: String::new(),
            quick_open_matches: Vec::new(),
            quick_open_selected: 0,
            global_search_open: false,
            global_search_query: String::new(),
            global_search_case_sensitive: false,
            global_search_matches: Vec::new(),
            global_search_selected: 0,
            global_search_searching: false,
            global_search_debounce_deadline: None,
            global_search_req_id: 0,
            global_search_cancel: None,
            global_search_worker: None,
            redraw_notifier: None,
            command_palette_open: false,
            command_palette_query: String::new(),
            command_palette_matches: Vec::new(),
            command_palette_selected: 0,
            inspect_modal: None,
            inspect_modal_query: String::new(),
            inspect_modal_selected: 0,
            inspect_recovery_backups: Vec::new(),
            insert_table_open: false,
            insert_table_input: String::new(),
            file_prompt_mode: None,
            file_prompt_input: String::new(),
            selected_workspace_path: None,
            delete_confirm_path: None,
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
            link_opener: Box::new(NoopLinkOpener),
            image_cache: ImageCache::new(None),
            code_lang_open: false,
            code_lang_block_range: None,
            code_lang_query: String::new(),
            code_lang_matches: Vec::new(),
            code_lang_selected: 0,
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

    pub(crate) fn set_image_redraw_notifier(
        &mut self,
        notifier: Option<Arc<dyn Fn() + Send + Sync>>,
    ) {
        self.redraw_notifier = notifier.clone();
        if let Some(worker) = &self.global_search_worker {
            worker.set_notifier(notifier.clone());
        }
        self.image_cache = ImageCache::new(notifier);
    }

    pub(crate) fn active_image_base_dir(&self) -> Option<PathBuf> {
        self.workbench
            .active_tab()
            .and_then(|t| t.session.path.as_deref())
            .and_then(|p| p.parent())
            .filter(|p| !p.as_os_str().is_empty())
            .map(Path::to_path_buf)
            .or_else(|| self.workbench.workspace_root.clone())
    }

    pub fn set_save_dialog(&mut self, save_dialog: Box<dyn SaveDialogBackend>) {
        self.save_dialog = save_dialog;
    }

    /// Replace the external link opener backend (used by headless tests via `MemoryLinkOpener`).
    pub fn set_link_opener(&mut self, link_opener: Box<dyn LinkOpenerBackend>) {
        self.link_opener = link_opener;
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
            (Some(msg), Some(pers)) if !persistent.contains(&msg) && !pers.contains(msg) => {
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

    /// Drive 2s idle auto-save, 5s/15s/60s failure retry backoff, global search debounce/poll, and periodic external disk polling.
    pub fn tick_runtime_at(&mut self, now: Instant) -> bool {
        let auto_events = self.workbench.tick_auto_save_at(now);
        let mut changed = self.apply_runtime_events(auto_events);
        changed |= self.poll_global_search(now);
        if now.duration_since(self.last_external_poll) >= EXTERNAL_POLL_INTERVAL {
            self.last_external_poll = now;
            let poll_events = self.workbench.poll_external_changes(false);
            changed |= self.apply_runtime_events(poll_events);
            changed |= self.poll_custom_theme_hot_reload(false);
            let _ = self.workbench.flush_pending_persistence();
        }
        changed
    }

    /// Earliest wake deadline for the event loop combining caret blink, search debounce, external poll, and auto-save.
    pub fn next_wake_deadline(&self) -> Instant {
        let mut next = self.last_caret_blink + BLINK_INTERVAL;
        next = next.min(self.last_external_poll + EXTERNAL_POLL_INTERVAL);
        if let Some(auto_dl) = self.workbench.next_auto_save_deadline() {
            next = next.min(auto_dl);
        }
        if let Some(search_dl) = self.global_search_debounce_deadline {
            next = next.min(search_dl);
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
        let mut changed = self.apply_runtime_events(events);
        changed |= self.poll_custom_theme_hot_reload(true);
        let _ = self.workbench.flush_pending_persistence();
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
        match (&tab.session.state, &tab.session.format.decision) {
            (DocState::Conflict, _) => Some(DocBannerKind::Conflict {
                theirs_updated: tab.conflict_theirs_updated,
            }),
            (DocState::Missing { has_local_edits }, _) => Some(DocBannerKind::Missing {
                has_local_edits: *has_local_edits,
                write_error: tab.last_write_error.clone(),
            }),
            (DocState::Dirty, _) if tab.last_write_error.is_some() => {
                Some(DocBannerKind::WriteError {
                    error: tab.last_write_error.clone().unwrap_or_default(),
                })
            }
            (_, EncodingDecision::ReadOnly { candidate }) => {
                Some(DocBannerKind::ReadOnlyEncoding {
                    candidate: *candidate,
                })
            }
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
            (body_h * 0.36)
                .clamp(110.0, 230.0)
                .min((rem_h - 40.0).max(0.0))
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
        if let (Some(banner_rect), Some(kind)) =
            (layout.alert_banner, self.active_doc_banner_kind())
        {
            self.build_doc_alert_banner(&mut chrome_layer, banner_rect, &kind);
        }
        if let Some(diff_rect) = layout.diff_panel {
            self.build_conflict_diff_panel(&mut chrome_layer, diff_rect);
        }
        if self.find_open {
            self.build_find_bar(&mut chrome_layer, layout.editor);
        }
        if self.link_prompt_open {
            self.build_link_prompt(&mut chrome_layer, layout.editor);
        }
        if self.quick_open_open {
            self.build_quick_open_modal(&mut chrome_layer, layout.editor);
        }
        if self.code_lang_open {
            self.build_code_lang_modal(&mut chrome_layer, layout.editor);
        }
        if self.global_search_open {
            self.build_global_search_modal(&mut chrome_layer, layout.editor);
        }
        if self.command_palette_open {
            self.build_command_palette_modal(&mut chrome_layer, layout.editor);
        }
        if self.inspect_modal.is_some() {
            self.build_inspect_modal(&mut chrome_layer, layout.editor);
        }
        if self.insert_table_open || self.file_prompt_mode.is_some() {
            self.build_simple_prompt_modal(&mut chrome_layer, layout.editor);
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
        let selection = tab.view.selection;
        let selection_range = selection.range();

        let needs_doc_recompute = self
            .cached_meta
            .as_ref()
            .is_none_or(|m| m.tab_id != tab_id || m.rev != rev);
        let needs_find_recompute = needs_doc_recompute
            || self
                .cached_meta
                .as_ref()
                .is_none_or(|m| m.matches_query != query || m.matches_opts != opts);
        let needs_selection_recompute = !needs_doc_recompute
            && self
                .cached_meta
                .as_ref()
                .is_some_and(|m| m.selection_range != selection_range);

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
                selection_range,
                stats,
                outline,
                matches_query: query,
                matches_opts: opts,
                find_matches,
            });
        } else {
            if needs_find_recompute {
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
            if needs_selection_recompute {
                let (sel_words, sel_chars) = if selection.is_caret() {
                    (0, 0)
                } else {
                    let total = tab.session.buffer.len_bytes();
                    let end = selection_range.end.min(total);
                    let start = selection_range.start.min(end);
                    let sel_text = tab
                        .session
                        .buffer
                        .slice_to_string(ByteRange::new(start, end));
                    let (w, c, _) = count_text_metrics(&sel_text);
                    (w, c)
                };
                if let Some(m) = self.cached_meta.as_mut() {
                    m.selection_range = selection_range;
                    m.stats.selection_word_count = sel_words;
                    m.stats.selection_char_count = sel_chars;
                }
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
        let show_caret = self.caret_visible && !self.has_focused_text_input();
        let active_find = self.active_find_range;

        let find_ranges: Vec<ByteRange> = self
            .cached_meta
            .as_ref()
            .map(|m| m.find_matches.iter().map(|fm| fm.range).collect())
            .unwrap_or_default();

        let _ = self.image_cache.drain_completed();
        let image_base_dir = self.active_image_base_dir();
        let allow_network_images = self.workbench.config.allow_network_images;
        let show_code_line_numbers = self.workbench.config.show_code_line_numbers;
        let status_resolver = |src: &str| -> ImageLoadStatus {
            self.image_cache
                .status_for(image_base_dir.as_deref(), src, allow_network_images)
        };

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
                show_code_line_numbers,
                find_matches: &find_ranges,
                active_find,
                image_status: Some(&status_resolver),
            };
            Some(tab.with_incremental_blocks(|blocks| {
                self.shaper
                    .build_editor_scene(blocks, &self.style, &vp, |r| {
                        tab.session.buffer.slice_to_string(r)
                    })
            }))
        } else {
            None
        };

        let Some(mut scene) = scene_opt else {
            self.last_ime_cursor_area = None;
            let mut empty = DrawList::default();
            empty
                .rects
                .push(SceneRect::solid(editor_rect, self.style.bg_color));
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
                        tab.scroll_y =
                            (caret_bot - editor_rect.h + bot_margin).clamp(0.0, max_scroll);
                    }
                }
                tab.scroll_y = tab.scroll_y.clamp(0.0, max_scroll);
                (tab.scroll_y - prev_scroll).abs() > 0.5
            } else {
                false
            };
            self.pending_scroll_ensure_caret = false;
            if need_reframe && let Some(tab) = self.workbench.active_tab() {
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
                    show_code_line_numbers,
                    find_matches: &find_ranges,
                    active_find,
                    image_status: Some(&status_resolver),
                };
                scene = tab.with_incremental_blocks(|blocks| {
                    self.shaper
                        .build_editor_scene(blocks, &self.style, &vp2, |r| {
                            tab.session.buffer.slice_to_string(r)
                        })
                });
                self.last_ime_cursor_area = scene.ime_cursor_area;
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
            let sb_rect = Rect::new(
                editor_rect.x + editor_rect.w - 8.0,
                thumb_y + 2.0,
                5.0,
                thumb_h - 4.0,
            );
            scene.draw.rects.push(
                SceneRect::rounded(sb_rect, self.style.chrome.muted_text.with_alpha(95), 2.5)
                    .with_clip(Some(editor_rect)),
            );
        }

        if !scene.requested_images.is_empty() {
            self.image_cache.request_images(
                image_base_dir.as_deref(),
                &scene.requested_images,
                allow_network_images,
            );
        }
        for btn in &scene.code_block_buttons {
            if let Some(rect) = intersect_rect(btn.rect, editor_rect) {
                let action = match btn.kind {
                    CodeBlockButtonKind::LanguagePicker => UiAction::OpenCodeLangPicker {
                        block_range: btn.block_range,
                        current_language: btn.current_language.clone(),
                    },
                    CodeBlockButtonKind::ToggleLineNumbers => UiAction::ToggleCodeLineNumbers,
                };
                self.ui_hit_zones.push(UiHitZone {
                    rect,
                    action,
                    pointer_cursor: true,
                });
            }
        }
        for btn in &scene.table_buttons {
            if let Some(rect) = intersect_rect(btn.rect, editor_rect) {
                self.ui_hit_zones.push(UiHitZone {
                    rect,
                    action: UiAction::TableToolbarAction(btn.kind),
                    pointer_cursor: true,
                });
            }
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
        self.push_ui_text(
            dl,
            "|||",
            sb_btn.x + 7.0,
            sb_btn.y + 4.0,
            11.5,
            600,
            chrome.text,
            None,
            Some(bar),
        );
        self.ui_hit_zones.push(UiHitZone {
            rect: sb_btn,
            action: UiAction::ToggleSidebar,
            pointer_cursor: true,
        });

        // Right-side action pills (Theme, TW, Focus, WYSIWYG/Source)
        let mode_label = match self.workbench.active_tab().map(|t| t.view.mode) {
            Some(ViewMode::Source) => "Source",
            _ => "WYSIWYG",
        };
        let mode_w = 78.0;
        let focus_w = 64.0;
        let tw_w = 38.0;
        let theme_w = 68.0;
        let right_margin = 10.0;

        let mode_rect = Rect::new(bar.w - right_margin - mode_w, 6.0, mode_w, 24.0);
        let focus_rect = Rect::new(mode_rect.x - 6.0 - focus_w, 6.0, focus_w, 24.0);
        let tw_rect = Rect::new(focus_rect.x - 6.0 - tw_w, 6.0, tw_w, 24.0);
        let theme_rect = Rect::new(tw_rect.x - 6.0 - theme_w, 6.0, theme_w, 24.0);

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

        let tw_bg = if self.workbench.config.typewriter_mode {
            chrome.accent.with_alpha(45)
        } else {
            chrome.panel_bg
        };
        dl.rects
            .push(SceneRect::rounded(tw_rect, tw_bg, 5.0).with_border(1.0, chrome.border));
        self.push_ui_text(
            dl,
            "TW",
            tw_rect.x + 9.0,
            tw_rect.y + 4.0,
            11.5,
            500,
            chrome.text,
            Some(tw_w - 6.0),
            Some(bar),
        );
        self.ui_hit_zones.push(UiHitZone {
            rect: tw_rect,
            action: UiAction::ToggleTypewriterMode,
            pointer_cursor: true,
        });

        dl.rects.push(
            SceneRect::rounded(theme_rect, chrome.panel_bg, 5.0).with_border(1.0, chrome.border),
        );
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
            let tab_w = (label.len() as f32 * 7.5 + 42.0)
                .clamp(96.0, 200.0)
                .min(max_tabs_right - x);
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
                    Rect::new(
                        tab_rect.x + 8.0,
                        tab_rect.y + tab_rect.h - 2.0,
                        tab_w - 16.0,
                        2.0,
                    ),
                    chrome.accent,
                ));
            }
            let col = if is_active {
                chrome.text
            } else {
                chrome.muted_text
            };
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
            dl.rects
                .push(SceneRect::rounded(new_btn, chrome.panel_bg, 5.0));
            self.push_ui_text(
                dl,
                "+",
                new_btn.x + 7.0,
                new_btn.y + 3.0,
                14.0,
                600,
                chrome.text,
                None,
                Some(bar),
            );
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

        // Header switch: Outline | Files | Recent
        let header_h = 32.0;
        let third_w = (sb.w - 24.0) / 3.0;
        let outline_btn = Rect::new(sb.x + 8.0, sb.y + 6.0, third_w, 22.0);
        let files_btn = Rect::new(outline_btn.x + third_w + 4.0, sb.y + 6.0, third_w, 22.0);
        let recent_btn = Rect::new(files_btn.x + third_w + 4.0, sb.y + 6.0, third_w, 22.0);

        for (btn, sec, label) in [
            (outline_btn, SidebarSection::Outline, "Outline"),
            (files_btn, SidebarSection::Files, "Files"),
            (recent_btn, SidebarSection::Recent, "Recent"),
        ] {
            let active = self.sidebar_section == sec;
            dl.rects.push(SceneRect::rounded(
                btn,
                if active {
                    chrome.active_tab_bg
                } else {
                    chrome.sidebar_bg
                },
                4.0,
            ));
            self.push_ui_text(
                dl,
                label,
                btn.x + 10.0,
                btn.y + 3.0,
                11.5,
                if active { 600 } else { 400 },
                if active {
                    chrome.text
                } else {
                    chrome.muted_text
                },
                Some(btn.w - 8.0),
                Some(sb),
            );
            self.ui_hit_zones.push(UiHitZone {
                rect: btn,
                action: UiAction::SwitchSidebarSection(sec),
                pointer_cursor: true,
            });
        }

        let list_clip = Rect::new(
            sb.x,
            sb.y + header_h,
            sb.w - 1.0,
            (sb.h - header_h).max(10.0),
        );
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
                let delete_armed = self.delete_confirm_path.is_some();
                let del_label = if delete_armed { "Del?" } else { "Del" };
                let bar_y = row_y;
                let mut btn_x = sb.x + 8.0;
                for (bw, label, action, is_danger) in [
                    (48.0, "+File", UiAction::OpenNewFilePrompt, false),
                    (38.0, "Ren", UiAction::OpenRenameFilePrompt, false),
                    (
                        42.0,
                        del_label,
                        UiAction::ArmOrConfirmDeleteFile,
                        delete_armed,
                    ),
                ] {
                    let rect = Rect::new(btn_x, bar_y, bw, 22.0);
                    let bg = if is_danger {
                        chrome.danger.with_alpha(48)
                    } else {
                        chrome.panel_bg
                    };
                    let border = if is_danger {
                        chrome.danger
                    } else {
                        chrome.border
                    };
                    dl.rects.push(
                        SceneRect::rounded(rect, bg, 4.0)
                            .with_border(1.0, border)
                            .with_clip(Some(list_clip)),
                    );
                    self.push_ui_text(
                        dl,
                        label,
                        rect.x + 7.0,
                        rect.y + 4.0,
                        11.0,
                        600,
                        if is_danger {
                            chrome.danger
                        } else {
                            chrome.text
                        },
                        Some(bw - 8.0),
                        Some(list_clip),
                    );
                    self.ui_hit_zones.push(UiHitZone {
                        rect,
                        action,
                        pointer_cursor: true,
                    });
                    btn_x += bw + 4.0;
                }
                let folder_w = 56.0;
                let folder_rect = Rect::new(sb.x + sb.w - folder_w - 9.0, bar_y, folder_w, 22.0);
                if folder_rect.x >= btn_x {
                    dl.rects.push(
                        SceneRect::rounded(folder_rect, chrome.panel_bg, 4.0)
                            .with_border(1.0, chrome.border)
                            .with_clip(Some(list_clip)),
                    );
                    self.push_ui_text(
                        dl,
                        "Folder",
                        folder_rect.x + 8.0,
                        folder_rect.y + 4.0,
                        11.0,
                        500,
                        chrome.text,
                        Some(folder_w - 10.0),
                        Some(list_clip),
                    );
                    self.ui_hit_zones.push(UiHitZone {
                        rect: folder_rect,
                        action: UiAction::OpenFolderPrompt,
                        pointer_cursor: true,
                    });
                }
                row_y += 28.0;

                let Some(tree) = self.workbench.file_tree.clone() else {
                    self.push_ui_text(
                        dl,
                        "Open a folder to browse files",
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
                let selected_path = self.selected_workspace_path.clone();

                for row in rows {
                    if row_y + row_h > sb.y + sb.h {
                        break;
                    }
                    let row_rect = Rect::new(sb.x + 6.0, row_y, sb.w - 13.0, row_h);
                    let is_open = active_path.as_ref() == Some(&row.path);
                    let is_selected = selected_path.as_ref() == Some(&row.path);
                    if is_open {
                        dl.rects.push(
                            SceneRect::rounded(row_rect, chrome.accent.with_alpha(38), 4.0)
                                .with_clip(Some(list_clip)),
                        );
                    } else if is_selected {
                        dl.rects.push(
                            SceneRect::rounded(row_rect, chrome.hover_bg, 4.0)
                                .with_border(1.0, chrome.accent.with_alpha(120))
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
            SidebarSection::Recent => {
                let entries = self.workbench.recent_files.entries.clone();
                if entries.is_empty() {
                    self.push_ui_text(
                        dl,
                        "No recent files",
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
                let active_path = self
                    .workbench
                    .active_tab()
                    .and_then(|t| t.session.path.as_ref())
                    .cloned();
                for path in entries {
                    if row_y + row_h > sb.y + sb.h {
                        break;
                    }
                    let row_rect = Rect::new(sb.x + 6.0, row_y, sb.w - 13.0, row_h);
                    let is_open = active_path.as_ref() == Some(&path);
                    if is_open {
                        dl.rects.push(
                            SceneRect::rounded(row_rect, chrome.accent.with_alpha(38), 4.0)
                                .with_clip(Some(list_clip)),
                        );
                    }
                    let name = path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .map(str::to_string)
                        .unwrap_or_else(|| path.display().to_string());
                    let display = if path.is_dir() {
                        format!("[dir] {name}")
                    } else {
                        name
                    };
                    self.push_ui_text(
                        dl,
                        &display,
                        row_rect.x + 8.0,
                        row_y + 4.0,
                        12.0,
                        if is_open { 600 } else { 400 },
                        if is_open { chrome.accent } else { chrome.text },
                        Some((row_rect.w - 16.0).max(30.0)),
                        Some(list_clip),
                    );
                    self.ui_hit_zones.push(UiHitZone {
                        rect: row_rect,
                        action: UiAction::OpenRecentFile(path),
                        pointer_cursor: true,
                    });
                    row_y += row_h + 2.0;
                }
            }
        }
    }

    fn draw_overlay_text_input(
        &mut self,
        dl: &mut DrawList,
        input_rect: Rect,
        value: &str,
        placeholder: &str,
        focused: bool,
        action: UiAction,
    ) {
        let chrome = self.style.chrome;
        dl.rects.push(
            SceneRect::rounded(input_rect, chrome.active_tab_bg, 4.0).with_border(
                1.0,
                if focused {
                    chrome.accent
                } else {
                    chrome.border
                },
            ),
        );
        let display = if value.is_empty() { placeholder } else { value };
        let text_w = self.push_ui_text(
            dl,
            display,
            input_rect.x + 7.0,
            input_rect.y + 5.0,
            12.0,
            400,
            if value.is_empty() {
                chrome.muted_text
            } else {
                chrome.text
            },
            Some((input_rect.w - 12.0).max(16.0)),
            Some(input_rect),
        );
        if focused && self.caret_visible {
            let cx = input_rect.x + 7.0 + if value.is_empty() { 0.0 } else { text_w };
            dl.rects.push(SceneRect::solid(
                Rect::new(
                    cx.min(input_rect.x + input_rect.w - 4.0),
                    input_rect.y + 5.0,
                    1.5,
                    16.0,
                ),
                chrome.accent,
            ));
        }
        self.ui_hit_zones.push(UiHitZone {
            rect: input_rect,
            action,
            pointer_cursor: false,
        });
    }

    fn build_find_bar(&mut self, dl: &mut DrawList, editor_rect: Rect) {
        let chrome = self.style.chrome;
        let bar_w = 490.0_f32.min((editor_rect.w - 24.0).max(280.0));
        let bar_h = if self.replace_open {
            FIND_BAR_H + 34.0
        } else {
            FIND_BAR_H
        };
        let bar = Rect::new(
            editor_rect.x + editor_rect.w - bar_w - 16.0,
            editor_rect.y + 8.0,
            bar_w,
            bar_h,
        );
        dl.rects
            .push(SceneRect::rounded(bar, chrome.panel_bg, 7.0).with_border(1.0, chrome.border));
        self.ui_hit_zones.push(UiHitZone {
            rect: bar,
            action: if self.replace_open && self.replace_focused {
                UiAction::FocusReplaceInput
            } else {
                UiAction::FocusFindInput
            },
            pointer_cursor: false,
        });

        let mut bx = bar.x + bar.w - 208.0;
        let count_x = bx - 48.0;
        let input_w = (count_x - 6.0 - (bar.x + 8.0)).max(84.0);
        let input_rect = Rect::new(bar.x + 8.0, bar.y + 6.0, input_w, 26.0);
        let query = self.find_query.clone();
        self.draw_overlay_text_input(
            dl,
            input_rect,
            &query,
            "Find...",
            self.find_focused,
            UiAction::FocusFindInput,
        );

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
        self.push_ui_text(
            dl,
            &count_text,
            count_x,
            bar.y + 11.0,
            11.0,
            500,
            chrome.muted_text,
            Some(44.0),
            Some(bar),
        );

        for (label, bw, active, action) in [
            (
                "W",
                26.0,
                self.find_options.whole_word,
                UiAction::ToggleFindWholeWord,
            ),
            (
                "Aa",
                26.0,
                self.find_options.case_sensitive,
                UiAction::ToggleFindCase,
            ),
            (
                ".*",
                26.0,
                self.find_options.use_regex,
                UiAction::ToggleFindRegex,
            ),
            ("⇄", 26.0, self.replace_open, UiAction::ToggleReplaceRow),
            ("<", 24.0, false, UiAction::FindPrev),
            (">", 24.0, false, UiAction::FindNext),
            ("×", 24.0, false, UiAction::CloseFind),
        ] {
            let r = Rect::new(bx, bar.y + 7.0, bw, 24.0);
            dl.rects.push(
                SceneRect::rounded(
                    r,
                    if active {
                        chrome.accent.with_alpha(48)
                    } else {
                        chrome.tab_bar_bg
                    },
                    4.0,
                )
                .with_border(1.0, if active { chrome.accent } else { chrome.border }),
            );
            self.push_ui_text(
                dl,
                label,
                r.x + 6.0,
                r.y + 4.0,
                11.5,
                600,
                chrome.text,
                None,
                Some(bar),
            );
            self.ui_hit_zones.push(UiHitZone {
                rect: r,
                action,
                pointer_cursor: true,
            });
            bx += bw + 4.0;
        }

        if self.replace_open {
            let row2_y = bar.y + FIND_BAR_H;
            let mut right_edge = bar.x + bar.w - 8.0;
            right_edge = self.push_action_button(
                dl,
                right_edge,
                row2_y + 1.0,
                24.0,
                "All",
                chrome.tab_bar_bg,
                chrome.border,
                chrome.text,
                UiAction::ReplaceAll,
                bar,
            );
            right_edge = self.push_action_button(
                dl,
                right_edge,
                row2_y + 1.0,
                24.0,
                "Rep",
                chrome.accent.with_alpha(48),
                chrome.accent,
                chrome.text,
                UiAction::ReplaceNext,
                bar,
            );
            let rep_w = (right_edge - (bar.x + 8.0)).max(90.0);
            let rep_rect = Rect::new(bar.x + 8.0, row2_y, rep_w, 26.0);
            let rep_val = self.replace_text.clone();
            self.draw_overlay_text_input(
                dl,
                rep_rect,
                &rep_val,
                "Replace...",
                self.replace_focused,
                UiAction::FocusReplaceInput,
            );
        }
    }

    fn build_link_prompt(&mut self, dl: &mut DrawList, editor_rect: Rect) {
        let chrome = self.style.chrome;
        let bar_w = 430.0_f32.min((editor_rect.w - 24.0).max(240.0));
        let top_offset = if self.find_open {
            let find_h = if self.replace_open {
                FIND_BAR_H + 34.0
            } else {
                FIND_BAR_H
            };
            find_h + 14.0
        } else {
            8.0
        };
        let bar = Rect::new(
            editor_rect.x + editor_rect.w - bar_w - 16.0,
            editor_rect.y + top_offset,
            bar_w,
            FIND_BAR_H,
        );
        dl.rects
            .push(SceneRect::rounded(bar, chrome.panel_bg, 7.0).with_border(1.0, chrome.accent));
        self.ui_hit_zones.push(UiHitZone {
            rect: bar,
            action: UiAction::FocusLinkInput,
            pointer_cursor: false,
        });

        let mut right_edge = bar.x + bar.w - 6.0;
        right_edge = self.push_action_button(
            dl,
            right_edge,
            bar.y + 7.0,
            24.0,
            "×",
            chrome.tab_bar_bg,
            chrome.border,
            chrome.text,
            UiAction::CloseLinkPrompt,
            bar,
        );
        right_edge = self.push_action_button(
            dl,
            right_edge,
            bar.y + 7.0,
            24.0,
            "Insert",
            chrome.accent.with_alpha(48),
            chrome.accent,
            chrome.text,
            UiAction::ConfirmLinkPrompt,
            bar,
        );
        let input_w = (right_edge - (bar.x + 8.0)).max(90.0);
        let input_rect = Rect::new(bar.x + 8.0, bar.y + 6.0, input_w, 26.0);
        let url_val = self.link_prompt_url.clone();
        self.draw_overlay_text_input(
            dl,
            input_rect,
            &url_val,
            "https://... or path.md or #heading",
            true,
            UiAction::FocusLinkInput,
        );
    }

    fn build_quick_open_modal(&mut self, dl: &mut DrawList, editor_rect: Rect) {
        let query = self.quick_open_query.clone();
        let selected = self.quick_open_selected;
        let empty_msg = if self.workbench.workspace_root.is_none() {
            "No workspace folder open — press Cmd/Ctrl+Shift+O to open a folder"
        } else {
            "No matching Markdown/text files"
        };
        let rows: Vec<ListPickerRow> = self
            .quick_open_matches
            .iter()
            .enumerate()
            .map(|(idx, m)| ListPickerRow {
                label: m.relative_path.clone(),
                badge: None,
                action: UiAction::SelectQuickOpenMatch(idx),
            })
            .collect();
        self.draw_list_picker_modal(
            dl,
            editor_rect,
            &query,
            "Quick Open — type file name or path...",
            empty_msg,
            selected,
            UiAction::FocusQuickOpenInput,
            UiAction::CloseQuickOpen,
            None,
            rows,
        );
    }

    fn build_global_search_modal(&mut self, dl: &mut DrawList, editor_rect: Rect) {
        let query = self.global_search_query.clone();
        let selected = self.global_search_selected;
        let empty_msg = if self.workbench.workspace_root.is_none() {
            "No workspace folder open — press Cmd/Ctrl+Shift+O to open a folder"
        } else if query.trim().is_empty() {
            "Type text to search across workspace files..."
        } else if self.global_search_searching {
            "Searching workspace..."
        } else {
            "No matching lines in workspace"
        };
        let root_opt = self.workbench.workspace_root.as_ref();
        let rows: Vec<ListPickerRow> = self
            .global_search_matches
            .iter()
            .enumerate()
            .map(|(idx, m)| {
                let rel = root_opt
                    .and_then(|root| m.path.strip_prefix(root).ok())
                    .unwrap_or(&m.path)
                    .to_string_lossy();
                ListPickerRow {
                    label: format!("{rel}  —  {}", m.line_preview.trim()),
                    badge: Some(format!("L{}", m.line_number)),
                    action: UiAction::SelectGlobalSearchMatch(idx),
                }
            })
            .collect();
        self.draw_list_picker_modal(
            dl,
            editor_rect,
            &query,
            "Global Search — type text across workspace...",
            empty_msg,
            selected,
            UiAction::FocusGlobalSearchInput,
            UiAction::CloseGlobalSearch,
            Some((
                "Aa",
                self.global_search_case_sensitive,
                UiAction::ToggleGlobalSearchCase,
            )),
            rows,
        );
    }

    fn build_command_palette_modal(&mut self, dl: &mut DrawList, editor_rect: Rect) {
        let query = self.command_palette_query.clone();
        let selected = self.command_palette_selected;
        let rows: Vec<ListPickerRow> = self
            .command_palette_matches
            .iter()
            .enumerate()
            .map(|(idx, entry)| ListPickerRow {
                label: entry.label.to_string(),
                badge: (!entry.shortcut.is_empty()).then(|| entry.shortcut.to_string()),
                action: UiAction::SelectCommandPaletteMatch(idx),
            })
            .collect();
        self.draw_list_picker_modal(
            dl,
            editor_rect,
            &query,
            "Command Palette — type a command...",
            "No matching commands",
            selected,
            UiAction::FocusCommandPaletteInput,
            UiAction::CloseCommandPalette,
            None,
            rows,
        );
    }

    fn build_inspect_modal(&mut self, dl: &mut DrawList, editor_rect: Rect) {
        let Some(kind) = self.inspect_modal else {
            return;
        };
        let query = self.inspect_modal_query.clone();
        let selected = self.inspect_modal_selected;
        match kind {
            InspectModalKind::ThemeDiagnostics => {
                let indices = self.filtered_theme_diagnostic_indices();
                let rows: Vec<ListPickerRow> = indices
                    .iter()
                    .enumerate()
                    .filter_map(|(slot_idx, &diag_idx)| {
                        let d = self.workbench.theme.diagnostics.get(diag_idx)?;
                        Some(ListPickerRow {
                            label: format!("{} — {}", d.item, d.reason),
                            badge: Some(d.selector.clone()),
                            action: UiAction::SelectInspectModalRow(slot_idx),
                        })
                    })
                    .collect();
                let empty_msg = if query.trim().is_empty() {
                    "No unsupported CSS rules — active theme matches Vellmark CSS subset"
                } else {
                    "No matching theme diagnostics"
                };
                let header_btn = self
                    .workbench
                    .config
                    .custom_theme_path
                    .as_ref()
                    .map(|_| ("Reload", false, UiAction::ReloadCustomTheme));
                self.draw_list_picker_modal(
                    dl,
                    editor_rect,
                    &query,
                    "Theme Diagnostics — filter unsupported CSS rules...",
                    empty_msg,
                    selected,
                    UiAction::FocusInspectModalInput,
                    UiAction::CloseInspectModal,
                    header_btn,
                    rows,
                );
            }
            InspectModalKind::RecoveryBackups => {
                let indices = self.filtered_recovery_backup_indices();
                let rows: Vec<ListPickerRow> = indices
                    .iter()
                    .enumerate()
                    .filter_map(|(slot_idx, &entry_idx)| {
                        let entry = self.inspect_recovery_backups.get(entry_idx)?;
                        let source_name = entry
                            .source_path
                            .as_deref()
                            .and_then(|p| p.file_name())
                            .and_then(|n| n.to_str())
                            .unwrap_or("(untitled draft)");
                        Some(ListPickerRow {
                            label: format!("{source_name} — {}", entry.reason.as_slug()),
                            badge: Some(format!("{} B", entry.payload_len)),
                            action: UiAction::SelectInspectModalRow(slot_idx),
                        })
                    })
                    .collect();
                let empty_msg = if query.trim().is_empty() {
                    "No historical recovery or conflict safety backups found"
                } else {
                    "No matching recovery backups"
                };
                self.draw_list_picker_modal(
                    dl,
                    editor_rect,
                    &query,
                    "Recovery Backups — Enter to restore draft, Del/Cmd+Backspace to delete...",
                    empty_msg,
                    selected,
                    UiAction::FocusInspectModalInput,
                    UiAction::CloseInspectModal,
                    Some(("Del", false, UiAction::DeleteSelectedRecoveryBackup)),
                    rows,
                );
            }
        }
    }

    fn build_code_lang_modal(&mut self, dl: &mut DrawList, editor_rect: Rect) {
        let query = self.code_lang_query.clone();
        let selected = self.code_lang_selected;
        let trimmed = query.trim();
        let custom_empty;
        let empty_msg = if !trimmed.is_empty() {
            custom_empty = format!("Press Enter to use custom language \"{trimmed}\"");
            custom_empty.as_str()
        } else {
            "No matching language"
        };
        let rows: Vec<ListPickerRow> = self
            .code_lang_matches
            .iter()
            .enumerate()
            .map(|(idx, entry)| ListPickerRow {
                label: entry.label.to_string(),
                badge: Some(if entry.token.is_empty() {
                    "(none)".to_string()
                } else {
                    entry.token.to_string()
                }),
                action: UiAction::SelectCodeLangMatch(idx),
            })
            .collect();
        self.draw_list_picker_modal(
            dl,
            editor_rect,
            &query,
            "Code Language — type language name or token...",
            empty_msg,
            selected,
            UiAction::FocusCodeLangInput,
            UiAction::CloseCodeLangPicker,
            None,
            rows,
        );
    }

    fn build_simple_prompt_modal(&mut self, dl: &mut DrawList, editor_rect: Rect) {
        let (title, value, placeholder, confirm_label) = if self.insert_table_open {
            (
                "Insert Table (rows x cols)",
                self.insert_table_input.clone(),
                "3x4",
                "Insert",
            )
        } else if let Some(mode) = &self.file_prompt_mode {
            match mode {
                FilePromptMode::CreateFile { .. } => (
                    "New Document File",
                    self.file_prompt_input.clone(),
                    "note.md",
                    "Create",
                ),
                FilePromptMode::RenameFile { .. } => (
                    "Rename File",
                    self.file_prompt_input.clone(),
                    "renamed.md",
                    "Rename",
                ),
            }
        } else {
            return;
        };

        let chrome = self.style.chrome;
        let bar_w = 430.0_f32.min((editor_rect.w - 24.0).max(240.0));
        let modal = Rect::new(
            editor_rect.x + ((editor_rect.w - bar_w) * 0.5).max(8.0),
            editor_rect.y + 18.0,
            bar_w,
            64.0,
        );
        dl.rects
            .push(SceneRect::rounded(modal, chrome.panel_bg, 8.0).with_border(1.0, chrome.accent));
        self.ui_hit_zones.push(UiHitZone {
            rect: modal,
            action: UiAction::FocusSimplePromptInput,
            pointer_cursor: false,
        });
        self.push_ui_text(
            dl,
            title,
            modal.x + 10.0,
            modal.y + 6.0,
            11.0,
            600,
            chrome.muted_text,
            Some(modal.w - 20.0),
            Some(modal),
        );

        let mut right_edge = modal.x + modal.w - 8.0;
        let input_y = modal.y + 28.0;
        right_edge = self.push_action_button(
            dl,
            right_edge,
            input_y + 1.0,
            24.0,
            "×",
            chrome.tab_bar_bg,
            chrome.border,
            chrome.text,
            UiAction::CloseSimplePrompt,
            modal,
        );
        right_edge = self.push_action_button(
            dl,
            right_edge,
            input_y + 1.0,
            24.0,
            confirm_label,
            chrome.accent.with_alpha(48),
            chrome.accent,
            chrome.text,
            UiAction::ConfirmSimplePrompt,
            modal,
        );
        let input_w = (right_edge - (modal.x + 8.0)).max(90.0);
        let input_rect = Rect::new(modal.x + 8.0, input_y, input_w, 26.0);
        self.draw_overlay_text_input(
            dl,
            input_rect,
            &value,
            placeholder,
            true,
            UiAction::FocusSimplePromptInput,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_list_picker_modal(
        &mut self,
        dl: &mut DrawList,
        editor_rect: Rect,
        query: &str,
        placeholder: &str,
        empty_msg: &str,
        selected: usize,
        focus_action: UiAction,
        close_action: UiAction,
        header_toggle: Option<(&str, bool, UiAction)>,
        rows: Vec<ListPickerRow>,
    ) {
        let chrome = self.style.chrome;
        let modal_w = 540.0_f32.min((editor_rect.w - 24.0).max(260.0));
        let modal_x = editor_rect.x + ((editor_rect.w - modal_w) * 0.5).max(8.0);
        let modal_y = editor_rect.y + 18.0;

        let max_visible = 8usize;
        let row_h = 28.0_f32;
        let visible_rows = rows.len().clamp(1, max_visible);
        let list_top = modal_y + 42.0;
        let modal_h = (48.0 + visible_rows as f32 * row_h).min((editor_rect.h - 24.0).max(80.0));
        let modal = Rect::new(modal_x, modal_y, modal_w, modal_h);

        dl.rects
            .push(SceneRect::rounded(modal, chrome.panel_bg, 8.0).with_border(1.0, chrome.accent));
        self.ui_hit_zones.push(UiHitZone {
            rect: modal,
            action: focus_action.clone(),
            pointer_cursor: false,
        });

        let close_w = 24.0;
        let close_rect = Rect::new(
            modal.x + modal.w - close_w - 8.0,
            modal.y + 8.0,
            close_w,
            24.0,
        );
        dl.rects.push(
            SceneRect::rounded(close_rect, chrome.tab_bar_bg, 4.0).with_border(1.0, chrome.border),
        );
        self.push_ui_text(
            dl,
            "×",
            close_rect.x + 7.0,
            close_rect.y + 4.0,
            12.0,
            600,
            chrome.text,
            None,
            Some(modal),
        );
        self.ui_hit_zones.push(UiHitZone {
            rect: close_rect,
            action: close_action,
            pointer_cursor: true,
        });

        let input_right = if let Some((toggle_label, toggle_on, toggle_action)) = header_toggle {
            let toggle_w = (toggle_label.chars().count() as f32 * 7.2 + 12.0).max(28.0);
            let toggle_rect =
                Rect::new(close_rect.x - toggle_w - 6.0, modal.y + 8.0, toggle_w, 24.0);
            dl.rects.push(
                SceneRect::rounded(
                    toggle_rect,
                    if toggle_on {
                        chrome.accent.with_alpha(48)
                    } else {
                        chrome.tab_bar_bg
                    },
                    4.0,
                )
                .with_border(
                    1.0,
                    if toggle_on {
                        chrome.accent
                    } else {
                        chrome.border
                    },
                ),
            );
            self.push_ui_text(
                dl,
                toggle_label,
                toggle_rect.x + 6.0,
                toggle_rect.y + 4.0,
                11.0,
                600,
                chrome.text,
                None,
                Some(modal),
            );
            self.ui_hit_zones.push(UiHitZone {
                rect: toggle_rect,
                action: toggle_action,
                pointer_cursor: true,
            });
            toggle_rect.x - 6.0
        } else {
            close_rect.x - 6.0
        };

        let input_rect = Rect::new(
            modal.x + 8.0,
            modal.y + 7.0,
            (input_right - (modal.x + 8.0)).max(120.0),
            26.0,
        );
        self.draw_overlay_text_input(dl, input_rect, query, placeholder, true, focus_action);

        if rows.is_empty() {
            self.push_ui_text(
                dl,
                empty_msg,
                modal.x + 14.0,
                list_top + 6.0,
                12.0,
                400,
                chrome.muted_text,
                Some(modal.w - 28.0),
                Some(modal),
            );
            return;
        }

        let total = rows.len();
        let selected = selected.min(total - 1);
        let start_idx = selected.saturating_sub(max_visible - 1);
        let end_idx = (start_idx + max_visible).min(total);

        for (slot, (idx, row)) in rows
            .into_iter()
            .enumerate()
            .skip(start_idx)
            .take(end_idx - start_idx)
            .enumerate()
        {
            let y = list_top + slot as f32 * row_h;
            if y + row_h > modal.y + modal.h - 2.0 {
                break;
            }
            let row_rect = Rect::new(modal.x + 6.0, y, modal.w - 12.0, row_h - 2.0);
            let is_sel = idx == selected;
            if is_sel {
                dl.rects.push(
                    SceneRect::rounded(row_rect, chrome.accent.with_alpha(40), 5.0)
                        .with_border(1.0, chrome.accent.with_alpha(140)),
                );
            }
            let badge_reserve = row
                .badge
                .as_deref()
                .map(|b| (b.chars().count() as f32 * 6.5 + 18.0).min(120.0))
                .unwrap_or(0.0);
            self.push_ui_text(
                dl,
                &row.label,
                row_rect.x + 10.0,
                row_rect.y + 5.0,
                12.0,
                if is_sel { 600 } else { 400 },
                if is_sel {
                    chrome.text
                } else {
                    chrome.muted_text
                },
                Some((row_rect.w - 20.0 - badge_reserve).max(40.0)),
                Some(modal),
            );
            if let Some(badge) = &row.badge {
                let bx = row_rect.x + row_rect.w - badge_reserve + 4.0;
                self.push_ui_text(
                    dl,
                    badge,
                    bx,
                    row_rect.y + 5.5,
                    11.0,
                    500,
                    chrome.muted_text,
                    Some((badge_reserve - 10.0).max(20.0)),
                    Some(modal),
                );
            }
            self.ui_hit_zones.push(UiHitZone {
                rect: row_rect,
                action: row.action,
                pointer_cursor: true,
            });
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
        type BannerButton<'a> = (
            &'a str,
            UiAction,
            vellmark_render::RgbaColor,
            vellmark_render::RgbaColor,
            vellmark_render::RgbaColor,
        );
        let (accent, tint_alpha, message, buttons): (
            vellmark_render::RgbaColor,
            u8,
            String,
            Vec<BannerButton<'_>>,
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
                    (
                        "Save As...",
                        UiAction::SaveAsActive,
                        chrome.panel_bg,
                        chrome.border,
                        chrome.text,
                    ),
                    (
                        "Use Disk",
                        UiAction::ResolveConflictUseTheirs,
                        chrome.panel_bg,
                        chrome.border,
                        chrome.text,
                    ),
                    (
                        "Keep Mine",
                        UiAction::ResolveConflictKeepMine,
                        chrome.danger.with_alpha(38),
                        chrome.danger,
                        chrome.text,
                    ),
                    (
                        "Auto Merge",
                        UiAction::ResolveConflictAutoMerge,
                        chrome.accent.with_alpha(38),
                        chrome.accent,
                        chrome.text,
                    ),
                    (
                        diff_label,
                        UiAction::ToggleConflictDiff,
                        if self.conflict_diff_open {
                            chrome.accent.with_alpha(55)
                        } else {
                            chrome.panel_bg
                        },
                        if self.conflict_diff_open {
                            chrome.accent
                        } else {
                            chrome.border
                        },
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
                    (
                        "Close Tab",
                        UiAction::ForceCloseActiveMissing,
                        chrome.panel_bg,
                        chrome.border,
                        chrome.text,
                    ),
                    (
                        "Save As...",
                        UiAction::SaveAsActive,
                        chrome.panel_bg,
                        chrome.border,
                        chrome.text,
                    ),
                    (
                        "Re-save",
                        UiAction::ResaveMissingFile,
                        chrome.danger.with_alpha(38),
                        chrome.danger,
                        chrome.text,
                    ),
                ],
            ),
            DocBannerKind::WriteError { error } => (
                chrome.danger,
                24,
                format!("Auto-save write failed: {error}"),
                vec![
                    (
                        "Save As...",
                        UiAction::SaveAsActive,
                        chrome.panel_bg,
                        chrome.border,
                        chrome.text,
                    ),
                    (
                        "Retry",
                        UiAction::RetrySaveActive,
                        chrome.danger.with_alpha(38),
                        chrome.danger,
                        chrome.text,
                    ),
                ],
            ),
            DocBannerKind::RecoveryDegraded => (
                chrome.warning,
                24,
                RECOVERY_DEGRADED_NOTICE.to_string(),
                vec![
                    (
                        "Save As...",
                        UiAction::SaveAsActive,
                        chrome.panel_bg,
                        chrome.border,
                        chrome.text,
                    ),
                    (
                        "Save Now",
                        UiAction::SaveActiveNow,
                        chrome.warning.with_alpha(38),
                        chrome.warning,
                        chrome.text,
                    ),
                ],
            ),
            DocBannerKind::ReadOnlyEncoding { candidate } => (
                chrome.warning,
                24,
                format!(
                    "Read-only ({}) — cannot overwrite original non-UTF-8 file directly",
                    encoding_label(*candidate)
                ),
                vec![(
                    "Save UTF-8 Copy...",
                    UiAction::SaveUtf8Copy,
                    chrome.warning.with_alpha(38),
                    chrome.warning,
                    chrome.text,
                )],
            ),
            DocBannerKind::ConflictBackupCleanup { count } => (
                chrome.warning,
                18,
                format!("{count} conflict backup copy(s) stored in history for this file"),
                vec![
                    (
                        "×",
                        UiAction::DismissConflictBackupNotice,
                        chrome.panel_bg,
                        chrome.border,
                        chrome.muted_text,
                    ),
                    (
                        "Clean Backups",
                        UiAction::ClearConflictBackups,
                        chrome.warning.with_alpha(34),
                        chrome.warning,
                        chrome.text,
                    ),
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
                dl, right_edge, btn_y, btn_h, label, bg, border, fg, action, banner,
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
        dl.rects
            .push(SceneRect::solid(diff_rect, chrome.tab_bar_bg));
        dl.rects.push(SceneRect::solid(
            Rect::new(
                diff_rect.x,
                diff_rect.y + diff_rect.h - 1.0,
                diff_rect.w,
                1.0,
            ),
            chrome.border,
        ));

        let header_h = 24.0;
        dl.rects.push(SceneRect::solid(
            Rect::new(diff_rect.x, diff_rect.y, diff_rect.w, header_h),
            chrome.panel_bg,
        ));
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
        let visible_count = if truncated {
            max_lines - 1
        } else {
            max_lines.min(lines.len())
        };

        for (i, line) in lines.iter().take(visible_count).enumerate() {
            let y = body_y + i as f32 * line_h;
            let (fg, bg_opt, weight) = match line.kind {
                ConflictDiffKind::SectionHeader => {
                    (chrome.accent, Some(chrome.accent.with_alpha(18)), 600)
                }
                ConflictDiffKind::Context => (chrome.muted_text, None, 400),
                ConflictDiffKind::Mine => {
                    (chrome.warning, Some(chrome.warning.with_alpha(24)), 500)
                }
                ConflictDiffKind::Theirs => {
                    (chrome.success, Some(chrome.success.with_alpha(22)), 500)
                }
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
                let (enc, ro) = match tab.session.format.decision {
                    EncodingDecision::Editable(enc) => (encoding_label(enc), ""),
                    EncodingDecision::ReadOnly { candidate } => {
                        (encoding_label(candidate), " [Read-Only]")
                    }
                };
                let eol_name = match tab.session.format.dominant_line_ending {
                    LineEnding::Lf => "LF",
                    LineEnding::CrLf => "CRLF",
                    LineEnding::Cr => "CR",
                };
                let eol = format!("{enc} • {eol_name}");
                let ime = if tab.view.ime_preedit.is_some() {
                    "IME: Composing"
                } else if self.ime_enabled {
                    "IME: Ready"
                } else {
                    "IME: Idle"
                };
                (format!("{st}{ro}"), col, eol, ime)
            }
            None => (
                "No Document".to_string(),
                chrome.muted_text,
                String::new(),
                "",
            ),
        };

        let active_notice = self.active_status_notice();

        dl.rects.push(SceneRect::rounded(
            Rect::new(10.0, sb.y + 8.0, 8.0, 8.0),
            state_col,
            4.0,
        ));
        let right_stats = if let Some(m) = &self.cached_meta {
            let sel_part = if m.stats.selection_char_count > 0 {
                format!(
                    " (sel {}w / {}c)",
                    m.stats.selection_word_count, m.stats.selection_char_count
                )
            } else {
                String::new()
            };
            let read_part = format_reading_time(m.stats.reading_time_secs);
            format!(
                "{} words • {} chars • {} lines{sel_part} • {read_part}   |   {}   |   {}%   |   {}",
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
        let rx = (sb.w - right_stats.len() as f32 * 6.4 - 18.0).max(sb.w * 0.38);

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
            Some((rx - 32.0).max(100.0)),
            Some(sb),
        );

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

    fn focused_overlay_ime_target(&self) -> Option<ImeTarget> {
        if self.code_lang_open {
            Some(ImeTarget::CodeLangBox)
        } else if self.insert_table_open {
            Some(ImeTarget::InsertTableBox)
        } else if self.file_prompt_mode.is_some() {
            Some(ImeTarget::FileNameBox)
        } else if self.command_palette_open {
            Some(ImeTarget::CommandPaletteBox)
        } else if self.inspect_modal.is_some() {
            Some(ImeTarget::InspectModalBox)
        } else if self.global_search_open {
            Some(ImeTarget::GlobalSearchBox)
        } else if self.quick_open_open {
            Some(ImeTarget::QuickOpenBox)
        } else if self.link_prompt_open {
            Some(ImeTarget::LinkPromptBox)
        } else if self.find_open && self.replace_open && self.replace_focused {
            Some(ImeTarget::ReplaceBox)
        } else if self.find_open && self.find_focused {
            Some(ImeTarget::FindBox)
        } else {
            None
        }
    }

    fn has_focused_text_input(&self) -> bool {
        self.focused_overlay_ime_target().is_some()
    }

    fn overlay_target_still_open(&self, target: ImeTarget) -> bool {
        match target {
            ImeTarget::CodeLangBox => self.code_lang_open,
            ImeTarget::InsertTableBox => self.insert_table_open,
            ImeTarget::FileNameBox => self.file_prompt_mode.is_some(),
            ImeTarget::CommandPaletteBox => self.command_palette_open,
            ImeTarget::InspectModalBox => self.inspect_modal.is_some(),
            ImeTarget::GlobalSearchBox => self.global_search_open,
            ImeTarget::QuickOpenBox => self.quick_open_open,
            ImeTarget::LinkPromptBox => self.link_prompt_open,
            ImeTarget::ReplaceBox => self.find_open && self.replace_open,
            ImeTarget::FindBox => self.find_open,
            _ => false,
        }
    }

    fn commit_overlay_text(&mut self, target: ImeTarget, text: &str) -> bool {
        match target {
            ImeTarget::FindBox => {
                self.find_query.push_str(text);
                self.run_find_next(false);
                true
            }
            ImeTarget::ReplaceBox => {
                self.replace_text.push_str(text);
                true
            }
            ImeTarget::LinkPromptBox => {
                self.link_prompt_url.push_str(text);
                true
            }
            ImeTarget::QuickOpenBox => {
                self.quick_open_query.push_str(text);
                self.refresh_quick_open_matches();
                true
            }
            ImeTarget::CodeLangBox => {
                self.code_lang_query.push_str(text);
                self.refresh_code_lang_matches();
                true
            }
            ImeTarget::GlobalSearchBox => {
                self.global_search_query.push_str(text);
                self.schedule_global_search(false);
                true
            }
            ImeTarget::CommandPaletteBox => {
                self.command_palette_query.push_str(text);
                self.refresh_command_palette_matches();
                true
            }
            ImeTarget::InspectModalBox => {
                self.inspect_modal_query.push_str(text);
                self.inspect_modal_selected = 0;
                true
            }
            ImeTarget::InsertTableBox => {
                self.insert_table_input.push_str(text);
                true
            }
            ImeTarget::FileNameBox => {
                self.file_prompt_input.push_str(text);
                true
            }
            ImeTarget::None | ImeTarget::Tab(_) => false,
        }
    }

    fn cancel_abandoned_tab_ime(&mut self, active_tab_id: Option<u64>) {
        if let ImeTarget::Tab(ime_tab_id) = self.ime_target
            && (active_tab_id != Some(ime_tab_id) || self.has_focused_text_input())
        {
            if let Some(tab) = self.workbench.tabs.iter_mut().find(|t| t.id == ime_tab_id) {
                let _ = tab.edit(|ctx| ctx.ime_cancel());
            }
            self.ime_target = ImeTarget::None;
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
                        ImeTarget::FindBox
                        | ImeTarget::ReplaceBox
                        | ImeTarget::LinkPromptBox
                        | ImeTarget::QuickOpenBox
                        | ImeTarget::CodeLangBox
                        | ImeTarget::GlobalSearchBox
                        | ImeTarget::CommandPaletteBox
                        | ImeTarget::InspectModalBox
                        | ImeTarget::InsertTableBox
                        | ImeTarget::FileNameBox => {}
                        ImeTarget::Tab(target_id) => {
                            if let Some(tab) =
                                self.workbench.tabs.iter_mut().find(|t| t.id == target_id)
                            {
                                let _ = tab.edit(|ctx| ctx.ime_clear_preedit());
                            }
                        }
                        ImeTarget::None if !self.has_focused_text_input() => {
                            if let Some(tab) = self.workbench.active_tab_mut() {
                                let _ = tab.edit(|ctx| ctx.ime_clear_preedit());
                            }
                        }
                        ImeTarget::None => {}
                    }
                    return;
                }

                let target = match self.ime_target {
                    ImeTarget::Tab(id) if self.workbench.tabs.iter().any(|tab| tab.id == id) => {
                        ImeTarget::Tab(id)
                    }
                    t if self.overlay_target_still_open(t) => t,
                    _ => self.focused_overlay_ime_target().unwrap_or_else(|| {
                        match self.workbench.active_tab() {
                            Some(tab) => ImeTarget::Tab(tab.id),
                            None => ImeTarget::None,
                        }
                    }),
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
                    ImeTarget::FindBox
                    | ImeTarget::ReplaceBox
                    | ImeTarget::LinkPromptBox
                    | ImeTarget::QuickOpenBox
                    | ImeTarget::CodeLangBox
                    | ImeTarget::GlobalSearchBox
                    | ImeTarget::CommandPaletteBox
                    | ImeTarget::InspectModalBox
                    | ImeTarget::InsertTableBox
                    | ImeTarget::FileNameBox => {
                        self.commit_overlay_text(target, committed);
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
                        if let Some(overlay) = self.focused_overlay_ime_target() {
                            self.commit_overlay_text(overlay, committed);
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
                if let ImeTarget::Tab(target_id) = prev
                    && let Some(tab) = self.workbench.tabs.iter_mut().find(|t| t.id == target_id)
                {
                    let _ = tab.edit(|ctx| ctx.ime_cancel());
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
        let _ = self
            .workbench
            .flush_pending_persistence_with_timeout(crate::config::CLOSE_PERSISTENCE_LOCK_TIMEOUT);
        let first_unsaved = self
            .workbench
            .tabs
            .iter()
            .position(|t| t.session.state != DocState::Clean);
        let Some(idx) = first_unsaved else {
            if self.workbench.has_pending_persistence() && !self.close_confirm_armed {
                self.close_confirm_armed = true;
                self.find_focused = false;
                self.set_notice(
                    "Settings or recent files are locked by another instance — close again to retry or exit without saving them",
                );
                return false;
            }
            self.workbench.maintain_recovery_storage();
            return true;
        };
        if !self.close_confirm_armed {
            self.workbench.active_tab_idx = Some(idx);
            self.close_confirm_armed = true;
            self.find_focused = false;
            let state = self.workbench.tabs[idx].session.state;
            let title = self.workbench.tabs[idx].title();
            let pending_suffix = if self.workbench.has_pending_persistence() {
                " (settings/recent files also locked by another instance)"
            } else {
                ""
            };
            match state {
                DocState::Conflict => {
                    self.conflict_diff_open = true;
                    self.set_notice(format!(
                        "External conflict in '{title}' — resolve or close again to archive backup & exit{pending_suffix}"
                    ));
                }
                DocState::Missing {
                    has_local_edits: true,
                } => {
                    self.set_notice(format!(
                        "Missing file '{title}' has unsaved changes — Save As or close again to backup & exit{pending_suffix}"
                    ));
                }
                _ => {
                    self.set_notice(format!(
                        "Unsaved changes in '{title}' — Save (Cmd/Ctrl+S) or close again to discard{pending_suffix}"
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
        self.handle_key_input_with_physical(
            &event.logical_key,
            Some(event.physical_key),
            event.text.as_deref(),
            mods,
        )
    }

    #[cfg(test)]
    pub(crate) fn handle_key_input(
        &mut self,
        logical_key: &Key,
        text: Option<&str>,
        mods: ModifiersState,
    ) -> bool {
        self.handle_key_input_with_physical(logical_key, None, text, mods)
    }

    fn handle_key_input_with_physical(
        &mut self,
        logical_key: &Key,
        physical_key: Option<PhysicalKey>,
        text: Option<&str>,
        mods: ModifiersState,
    ) -> bool {
        self.reset_blink();

        let primary = is_primary_mod(mods);
        let shift = mods.shift_key();
        let alt = mods.alt_key();
        let ctrl = mods.control_key();

        // When OS IME composition is active, let IME handle printable input; only allow Escape to cancel safely.
        let ime_composing = self
            .workbench
            .active_tab()
            .is_some_and(|t| t.view.ime_preedit.is_some());
        if ime_composing {
            if *logical_key == NamedKey::Escape {
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

        // macOS native Fullscreen shortcut (`Cmd+Ctrl+F`, PRD 4.10)
        if cfg!(target_os = "macos")
            && mods.super_key()
            && ctrl
            && !alt
            && !shift
            && (physical_key == Some(PhysicalKey::Code(KeyCode::KeyF))
                || matches!(logical_key, Key::Character(ch) if ch.eq_ignore_ascii_case("f")))
        {
            self.toggle_fullscreen();
            return true;
        }

        // macOS-only Cmd+Option shortcuts (`Cmd+Option+C` / `Cmd+Option+T` / `Cmd+Option+F`).
        // Unreachable on Linux/Windows because `primary` is `Ctrl` (so `primary && !ctrl` is false);
        // non-macOS platforms use `Ctrl+Shift+K`, `Ctrl+T`, and `Ctrl+H` below.
        if primary && alt && !shift && !ctrl {
            if physical_key == Some(PhysicalKey::Code(KeyCode::KeyC))
                || matches!(
                    logical_key,
                    Key::Character(ch) if matches!(ch.as_str(), "c" | "C" | "ç" | "Ç")
                )
            {
                self.insert_or_pick_code_block();
                return true;
            }
            if physical_key == Some(PhysicalKey::Code(KeyCode::KeyT))
                || matches!(
                    logical_key,
                    Key::Character(ch) if matches!(ch.as_str(), "t" | "T" | "†")
                )
            {
                self.open_insert_table_prompt();
                return true;
            }
            if physical_key == Some(PhysicalKey::Code(KeyCode::KeyF))
                || matches!(
                    logical_key,
                    Key::Character(ch) if matches!(ch.as_str(), "f" | "F" | "ƒ")
                )
            {
                self.open_find_bar(true);
                return true;
            }
        }

        // Function key view toggles (`F8` Focus Mode, `F9` Typewriter Mode, `F11` Fullscreen, PRD 4.7 / 4.10)
        if !primary && !alt && !ctrl {
            match logical_key {
                Key::Named(NamedKey::F8) => {
                    self.toggle_focus_mode();
                    return true;
                }
                Key::Named(NamedKey::F9) => {
                    self.toggle_typewriter_mode();
                    return true;
                }
                Key::Named(NamedKey::F11) => {
                    self.toggle_fullscreen();
                    return true;
                }
                _ => {}
            }
        }

        // Primary shortcuts
        if primary
            && !alt
            && let Key::Character(ch) = logical_key
        {
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
                "o" if shift => {
                    self.open_folder_dialog();
                    return true;
                }
                "o" => {
                    self.open_file_dialog();
                    return true;
                }
                "p" if shift => {
                    self.open_command_palette();
                    return true;
                }
                "p" => {
                    self.open_quick_open();
                    return true;
                }
                "k" if shift && !cfg!(target_os = "macos") => {
                    self.insert_or_pick_code_block();
                    return true;
                }
                "k" => {
                    self.open_link_prompt();
                    return true;
                }
                "l" if shift => {
                    self.execute_ui_action(UiAction::ToggleCodeLineNumbers);
                    return true;
                }
                "n" => {
                    self.close_confirm_armed = false;
                    self.close_transient_prompts();
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
                    if !self.has_focused_text_input() {
                        self.dispatch_editor(EditorAction::SelectAll);
                    }
                    return true;
                }
                "f" if shift => {
                    self.open_global_search();
                    return true;
                }
                "f" => {
                    self.open_find_bar(false);
                    return true;
                }
                "h" if !shift => {
                    self.open_find_bar(true);
                    return true;
                }
                "/" => {
                    self.dispatch_editor(EditorAction::ToggleSourceMode);
                    self.preferred_caret_x = None;
                    return true;
                }
                "\\" => {
                    self.workbench.config.sidebar_visible = !self.workbench.config.sidebar_visible;
                    self.persist_user_config();
                    return true;
                }
                "t" if shift => {
                    self.cycle_theme();
                    return true;
                }
                "t" => {
                    self.open_insert_table_prompt();
                    return true;
                }
                "=" | "+" => {
                    self.workbench.config.zoom_in();
                    self.refresh_style();
                    self.persist_user_config();
                    self.set_notice(format!("Zoom {}%", self.workbench.config.zoom_percent));
                    return true;
                }
                "-" => {
                    self.workbench.config.zoom_out();
                    self.refresh_style();
                    self.persist_user_config();
                    self.set_notice(format!("Zoom {}%", self.workbench.config.zoom_percent));
                    return true;
                }
                "0" => {
                    self.workbench.config.reset_zoom();
                    self.refresh_style();
                    self.persist_user_config();
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

        // Focused overlay / modal input handling (Quick Open -> Link Prompt -> Find bar)
        if self.handle_overlay_key_press(logical_key, text, primary, shift, mods.control_key()) {
            return true;
        }

        // Standard editor keys
        match logical_key {
            Key::Named(NamedKey::Escape) => {
                if self.close_confirm_armed {
                    self.close_confirm_armed = false;
                    self.set_notice("Close cancelled");
                    return true;
                }
                if self.find_open {
                    self.close_find_bar();
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
                if !primary
                    && !mods.control_key()
                    && let Some(txt) = text
                {
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
                false
            }
        }
    }

    fn handle_overlay_key_press(
        &mut self,
        logical_key: &Key,
        text: Option<&str>,
        primary: bool,
        shift: bool,
        ctrl: bool,
    ) -> bool {
        if self.code_lang_open {
            match decode_list_modal_key(logical_key, text, primary, ctrl) {
                ListModalAction::Close => self.close_code_lang_picker(),
                ListModalAction::MoveSelection(delta) => {
                    move_list_selection(
                        &mut self.code_lang_selected,
                        self.code_lang_matches.len(),
                        delta,
                    );
                }
                ListModalAction::Confirm => {
                    self.confirm_code_lang_match(self.code_lang_selected);
                }
                ListModalAction::QueryBackspace => {
                    self.code_lang_query.pop();
                    self.refresh_code_lang_matches();
                }
                ListModalAction::QueryAppend(txt) => {
                    self.code_lang_query.push_str(&txt);
                    self.refresh_code_lang_matches();
                }
                ListModalAction::None => {}
            }
            return true;
        }

        if self.insert_table_open || self.file_prompt_mode.is_some() {
            match logical_key {
                Key::Named(NamedKey::Escape) => self.close_simple_prompt(),
                Key::Named(NamedKey::Enter) => self.confirm_simple_prompt(),
                Key::Named(NamedKey::Backspace) => {
                    if self.insert_table_open {
                        self.insert_table_input.pop();
                    } else {
                        self.file_prompt_input.pop();
                    }
                }
                _ => {
                    if let Some(txt) = (!primary && !ctrl)
                        .then(|| overlay_printable_key_text(logical_key, text))
                        .flatten()
                    {
                        if self.insert_table_open {
                            self.insert_table_input.push_str(&txt);
                        } else {
                            self.file_prompt_input.push_str(&txt);
                        }
                    }
                }
            }
            return true;
        }

        if self.command_palette_open {
            match decode_list_modal_key(logical_key, text, primary, ctrl) {
                ListModalAction::Close => self.close_command_palette(),
                ListModalAction::MoveSelection(delta) => {
                    move_list_selection(
                        &mut self.command_palette_selected,
                        self.command_palette_matches.len(),
                        delta,
                    );
                }
                ListModalAction::Confirm => {
                    self.confirm_command_palette_match(self.command_palette_selected);
                }
                ListModalAction::QueryBackspace => {
                    self.command_palette_query.pop();
                    self.refresh_command_palette_matches();
                }
                ListModalAction::QueryAppend(txt) => {
                    self.command_palette_query.push_str(&txt);
                    self.refresh_command_palette_matches();
                }
                ListModalAction::None => {}
            }
            return true;
        }

        if let Some(kind) = self.inspect_modal {
            if kind == InspectModalKind::RecoveryBackups
                && (matches!(logical_key, Key::Named(NamedKey::Delete))
                    || ((primary || ctrl)
                        && matches!(logical_key, Key::Named(NamedKey::Backspace))))
            {
                self.delete_selected_recovery_backup();
                return true;
            }
            match decode_list_modal_key(logical_key, text, primary, ctrl) {
                ListModalAction::Close => self.close_inspect_modal(),
                ListModalAction::MoveSelection(delta) => {
                    let len = self.filtered_inspect_modal_len();
                    move_list_selection(&mut self.inspect_modal_selected, len, delta);
                }
                ListModalAction::Confirm => {
                    self.confirm_inspect_modal_row(self.inspect_modal_selected);
                }
                ListModalAction::QueryBackspace => {
                    self.inspect_modal_query.pop();
                    self.inspect_modal_selected = 0;
                }
                ListModalAction::QueryAppend(txt) => {
                    self.inspect_modal_query.push_str(&txt);
                    self.inspect_modal_selected = 0;
                }
                ListModalAction::None => {}
            }
            return true;
        }

        if self.global_search_open {
            match decode_list_modal_key(logical_key, text, primary, ctrl) {
                ListModalAction::Close => self.close_global_search(),
                ListModalAction::MoveSelection(delta) => {
                    move_list_selection(
                        &mut self.global_search_selected,
                        self.global_search_matches.len(),
                        delta,
                    );
                }
                ListModalAction::Confirm => {
                    self.confirm_global_search_match(self.global_search_selected);
                }
                ListModalAction::QueryBackspace => {
                    self.global_search_query.pop();
                    self.schedule_global_search(false);
                }
                ListModalAction::QueryAppend(txt) => {
                    self.global_search_query.push_str(&txt);
                    self.schedule_global_search(false);
                }
                ListModalAction::None => {}
            }
            return true;
        }

        if self.quick_open_open {
            match decode_list_modal_key(logical_key, text, primary, ctrl) {
                ListModalAction::Close => self.close_quick_open(),
                ListModalAction::MoveSelection(delta) => {
                    move_list_selection(
                        &mut self.quick_open_selected,
                        self.quick_open_matches.len(),
                        delta,
                    );
                }
                ListModalAction::Confirm => {
                    self.confirm_quick_open_match(self.quick_open_selected);
                }
                ListModalAction::QueryBackspace => {
                    self.quick_open_query.pop();
                    self.refresh_quick_open_matches();
                }
                ListModalAction::QueryAppend(txt) => {
                    self.quick_open_query.push_str(&txt);
                    self.refresh_quick_open_matches();
                }
                ListModalAction::None => {}
            }
            return true;
        }

        if self.link_prompt_open {
            match logical_key {
                Key::Named(NamedKey::Escape) => {
                    self.close_link_prompt();
                    return true;
                }
                Key::Named(NamedKey::Enter) => {
                    self.confirm_link_prompt();
                    return true;
                }
                Key::Named(NamedKey::Backspace) => {
                    self.link_prompt_url.pop();
                    return true;
                }
                _ => {
                    if let Some(txt) = (!primary && !ctrl)
                        .then(|| overlay_printable_key_text(logical_key, text))
                        .flatten()
                    {
                        self.link_prompt_url.push_str(&txt);
                        return true;
                    }
                }
            }
            return true;
        }

        if self.find_open && (self.find_focused || (self.replace_open && self.replace_focused)) {
            if matches!(logical_key, Key::Named(NamedKey::Escape)) {
                self.close_find_bar();
                return true;
            }
            if matches!(logical_key, Key::Named(NamedKey::Tab)) && self.replace_open {
                let to_replace = !self.replace_focused;
                self.find_focused = !to_replace;
                self.replace_focused = to_replace;
                if matches!(self.ime_target, ImeTarget::FindBox | ImeTarget::ReplaceBox) {
                    self.ime_target = ImeTarget::None;
                }
                return true;
            }
            if self.replace_open && self.replace_focused {
                match logical_key {
                    Key::Named(NamedKey::Enter) => {
                        if primary || shift {
                            self.run_replace_all();
                        } else {
                            self.run_replace_next();
                        }
                        return true;
                    }
                    Key::Named(NamedKey::Backspace) => {
                        self.replace_text.pop();
                        return true;
                    }
                    _ => {
                        if let Some(txt) = (!primary && !ctrl)
                            .then(|| overlay_printable_key_text(logical_key, text))
                            .flatten()
                        {
                            self.replace_text.push_str(&txt);
                            return true;
                        }
                    }
                }
            } else {
                match logical_key {
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
                        if let Some(txt) = (!primary && !ctrl)
                            .then(|| overlay_printable_key_text(logical_key, text))
                            .flatten()
                        {
                            self.find_query.push_str(&txt);
                            self.run_find_next(false);
                            return true;
                        }
                    }
                }
            }
        }

        false
    }

    pub fn handle_pointer_down(&mut self, x: f32, y: f32) -> bool {
        self.handle_pointer_down_with_mods(x, y, ModifiersState::empty())
    }

    /// Handle left mouse press with keyboard modifier state (supports `Cmd/Ctrl + Click` link navigation, PRD V-06).
    pub fn handle_pointer_down_with_mods(&mut self, x: f32, y: f32, mods: ModifiersState) -> bool {
        self.reset_blink();
        self.pointer_pos = (x, y);

        // 1. Check chrome buttons/tabs/sidebar/overlays first
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
            self.close_transient_prompts();
            self.close_confirm_armed = false;
            let active_id = self.workbench.active_tab().map(|t| t.id);
            self.cancel_abandoned_tab_ime(active_id);
            if let Some(target) = self.shaper.hit_test(x, y) {
                self.preferred_caret_x = None;
                match target {
                    EditorHitTarget::Text(offset) => {
                        if is_primary_mod(mods) && self.try_open_link_at_offset(offset) {
                            self.dragging_selection = false;
                            return true;
                        }
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
        self.handle_pointer_move_with_mods(x, y, ModifiersState::empty())
    }

    /// Update pointer hover/drag state with modifier awareness (shows pointer cursor over links while `Cmd/Ctrl` is held).
    pub fn handle_pointer_move_with_mods(
        &mut self,
        x: f32,
        y: f32,
        mods: ModifiersState,
    ) -> (CursorIcon, bool) {
        self.pointer_pos = (x, y);
        let mut changed = false;
        if self.dragging_selection
            && let Some(EditorHitTarget::Text(offset)) = self.shaper.hit_test(x, y)
            && let Some(tab) = self.workbench.active_tab_mut()
        {
            let anchor = tab
                .session
                .buffer
                .snap_out_of_crlf(tab.view.selection.anchor);
            let safe_offset = tab.session.buffer.snap_out_of_crlf(offset);
            let _ = tab.edit(|ctx| ctx.set_selection(anchor, safe_offset));
            changed = true;
        }
        let icon = if let Some(zone) = self
            .ui_hit_zones
            .iter()
            .rev()
            .find(|z| z.rect.contains(x, y))
        {
            if zone.pointer_cursor {
                CursorIcon::Pointer
            } else {
                CursorIcon::Text
            }
        } else if self.last_editor_rect.contains(x, y) {
            if is_primary_mod(mods)
                && !self.quick_open_open
                && !self.link_prompt_open
                && matches!(
                    self.shaper.hit_test(x, y),
                    Some(EditorHitTarget::Text(offset)) if self.has_link_at_offset(offset)
                )
            {
                CursorIcon::Pointer
            } else {
                CursorIcon::Text
            }
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
        if !matches!(action, UiAction::ArmOrConfirmDeleteFile) {
            self.delete_confirm_path = None;
        }
        match action {
            UiAction::SelectTab(idx) => {
                if idx < self.workbench.tabs.len() {
                    if self.workbench.active_tab_idx != Some(idx) {
                        self.flush_active_tab_before_switch();
                    }
                    self.workbench.active_tab_idx = Some(idx);
                    self.find_focused = false;
                    self.close_transient_prompts();
                    self.preferred_caret_x = None;
                }
                true
            }
            UiAction::CloseTab(id) => {
                self.close_transient_prompts();
                self.close_tab_checked(id, "Unsaved changes — press Save first");
                true
            }
            UiAction::NewTab => {
                self.flush_active_tab_before_switch();
                self.workbench.new_document();
                self.find_focused = false;
                self.close_transient_prompts();
                true
            }
            UiAction::ToggleMode => {
                self.dispatch_editor(EditorAction::ToggleSourceMode);
                true
            }
            UiAction::ToggleFocusMode => {
                self.toggle_focus_mode();
                true
            }
            UiAction::ToggleTypewriterMode => {
                self.toggle_typewriter_mode();
                true
            }
            UiAction::CycleTheme => {
                self.cycle_theme();
                true
            }
            UiAction::ToggleSidebar => {
                self.workbench.config.sidebar_visible = !self.workbench.config.sidebar_visible;
                self.persist_user_config();
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
                if let Some(item) = item_opt
                    && let Some(tab) = self.workbench.active_tab_mut()
                {
                    tab.jump_to_outline_item(&item);
                    self.pending_scroll_ensure_caret = true;
                    self.find_focused = false;
                    self.close_transient_prompts();
                }
                true
            }
            UiAction::OpenWorkspaceFile(path) => {
                self.selected_workspace_path = Some(path.clone());
                if path.is_dir() {
                    if !self.collapsed_dirs.remove(&path) {
                        self.collapsed_dirs.insert(path);
                    }
                } else {
                    self.flush_active_tab_before_switch();
                    match self.workbench.open_document(&path) {
                        Ok(_) => {
                            self.find_focused = false;
                            self.close_transient_prompts();
                            self.preferred_caret_x = None;
                            self.pending_scroll_ensure_caret = true;
                        }
                        Err(e) => self.set_notice(format!("Open failed: {e}")),
                    }
                }
                true
            }
            UiAction::OpenRecentFile(path) => {
                self.close_transient_prompts();
                self.find_focused = false;
                if path.is_dir() {
                    let normalized: PathBuf = path.components().collect();
                    match self.workbench.open_folder(&normalized) {
                        Ok(()) => {
                            if !self.workbench.config.sidebar_visible {
                                self.workbench.config.sidebar_visible = true;
                                self.persist_user_config();
                            }
                            self.sidebar_section = SidebarSection::Files;
                            self.collapsed_dirs.clear();
                            self.selected_workspace_path = None;
                            let name = normalized
                                .file_name()
                                .and_then(|n| n.to_str())
                                .unwrap_or("folder");
                            self.set_notice(format!("Opened folder: {name}"));
                        }
                        Err(e) => self.set_notice(format!("Open folder failed: {e}")),
                    }
                } else {
                    self.flush_active_tab_before_switch();
                    match self.workbench.open_document(&path) {
                        Ok(_) => {
                            self.ime_target = ImeTarget::None;
                            self.preferred_caret_x = None;
                            self.pending_scroll_ensure_caret = true;
                            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("file");
                            self.set_notice(format!("Opened {name}"));
                        }
                        Err(e) => self.set_notice(format!("Open failed: {e}")),
                    }
                }
                true
            }
            UiAction::OpenFolderPrompt => {
                self.open_folder_dialog();
                true
            }
            UiAction::OpenNewFilePrompt => {
                self.open_new_file_prompt();
                true
            }
            UiAction::OpenRenameFilePrompt => {
                self.open_rename_file_prompt();
                true
            }
            UiAction::ArmOrConfirmDeleteFile => {
                self.arm_or_confirm_delete_file();
                true
            }
            UiAction::ConfirmSimplePrompt => {
                self.confirm_simple_prompt();
                true
            }
            UiAction::CloseSimplePrompt => {
                self.close_simple_prompt();
                true
            }
            UiAction::FocusSimplePromptInput => true,
            UiAction::ToggleFindWholeWord => {
                self.find_options.whole_word = !self.find_options.whole_word;
                self.run_find_next(false);
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
            UiAction::ToggleReplaceRow => {
                self.replace_open = !self.replace_open;
                if self.replace_open {
                    if self.find_query.is_empty() {
                        self.find_focused = true;
                        self.replace_focused = false;
                    } else {
                        self.find_focused = false;
                        self.replace_focused = true;
                    }
                } else {
                    self.replace_focused = false;
                    self.find_focused = true;
                }
                if matches!(self.ime_target, ImeTarget::FindBox | ImeTarget::ReplaceBox) {
                    self.ime_target = ImeTarget::None;
                }
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
            UiAction::ReplaceNext => {
                self.run_replace_next();
                true
            }
            UiAction::ReplaceAll => {
                self.run_replace_all();
                true
            }
            UiAction::CloseFind => {
                self.close_find_bar();
                true
            }
            UiAction::FocusFindInput => {
                self.close_transient_prompts();
                self.find_focused = true;
                self.replace_focused = false;
                if self.ime_target == ImeTarget::ReplaceBox {
                    self.ime_target = ImeTarget::None;
                }
                true
            }
            UiAction::FocusReplaceInput => {
                self.close_transient_prompts();
                self.replace_open = true;
                self.find_focused = false;
                self.replace_focused = true;
                if self.ime_target == ImeTarget::FindBox {
                    self.ime_target = ImeTarget::None;
                }
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
                            self.set_notice(format!(
                                "Cannot close tab: {e} — please Save As first"
                            ));
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
            UiAction::CloseLinkPrompt => {
                self.close_link_prompt();
                true
            }
            UiAction::ConfirmLinkPrompt => {
                self.confirm_link_prompt();
                true
            }
            UiAction::FocusLinkInput
            | UiAction::FocusQuickOpenInput
            | UiAction::FocusCodeLangInput
            | UiAction::FocusGlobalSearchInput
            | UiAction::FocusCommandPaletteInput
            | UiAction::FocusInspectModalInput => true,
            UiAction::CloseQuickOpen => {
                self.close_quick_open();
                true
            }
            UiAction::SelectQuickOpenMatch(idx) => {
                self.confirm_quick_open_match(idx);
                true
            }
            UiAction::OpenCodeLangPicker {
                block_range,
                current_language,
            } => {
                self.open_code_lang_picker(block_range, current_language);
                true
            }
            UiAction::ToggleCodeLineNumbers => {
                self.workbench.config.show_code_line_numbers =
                    !self.workbench.config.show_code_line_numbers;
                self.persist_user_config();
                let on = self.workbench.config.show_code_line_numbers;
                self.set_notice(if on {
                    "Code Line Numbers: ON"
                } else {
                    "Code Line Numbers: OFF"
                });
                true
            }
            UiAction::CloseCodeLangPicker => {
                self.close_code_lang_picker();
                true
            }
            UiAction::SelectCodeLangMatch(idx) => {
                self.confirm_code_lang_match(idx);
                true
            }
            UiAction::TableToolbarAction(kind) => {
                self.execute_table_toolbar_action(kind);
                true
            }
            UiAction::CloseGlobalSearch => {
                self.close_global_search();
                true
            }
            UiAction::SelectGlobalSearchMatch(idx) => {
                self.confirm_global_search_match(idx);
                true
            }
            UiAction::ToggleGlobalSearchCase => {
                self.global_search_case_sensitive = !self.global_search_case_sensitive;
                self.schedule_global_search(true);
                true
            }
            UiAction::SaveUtf8Copy => {
                self.save_utf8_copy_dialog();
                true
            }
            UiAction::CloseCommandPalette => {
                self.close_command_palette();
                true
            }
            UiAction::SelectCommandPaletteMatch(idx) => {
                self.confirm_command_palette_match(idx);
                true
            }
            UiAction::CloseInspectModal => {
                self.close_inspect_modal();
                true
            }
            UiAction::SelectInspectModalRow(slot_idx) => {
                self.confirm_inspect_modal_row(slot_idx);
                true
            }
            UiAction::ReloadCustomTheme => {
                self.reload_custom_theme_now();
                true
            }
            UiAction::DeleteSelectedRecoveryBackup => {
                self.delete_selected_recovery_backup();
                true
            }
        }
    }

    fn cycle_theme(&mut self) {
        let all = BuiltinTheme::ALL;
        let current = self.workbench.config.theme;
        let idx = all.iter().position(|&t| t == current).unwrap_or(0);
        let next = all[(idx + 1) % all.len()];
        self.apply_builtin_theme_selection(false, Some(next));
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
                self.workbench.record_recent_path(dest);
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
                    self.workbench.record_recent_path(&new_path);
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

    fn dispatch_editor(&mut self, action: EditorAction) -> Option<bool> {
        self.find_focused = false;
        self.replace_focused = false;
        self.close_transient_prompts();
        self.close_confirm_armed = false;
        self.delete_confirm_path = None;
        let active_id = self.workbench.active_tab().map(|t| t.id);
        self.cancel_abandoned_tab_ime(active_id);
        let auto_align = self.workbench.config.table_auto_align;
        let tab = self.workbench.active_tab_mut()?;
        tab.view.table_auto_align = auto_align;
        match tab.dispatch(action) {
            Ok(changed) => {
                self.pending_scroll_ensure_caret = true;
                Some(changed)
            }
            Err(e) => {
                self.set_notice(format!("Read-only or blocked: {e}"));
                None
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
            let raw_target = if !extend && !sel.is_caret() {
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
            let target = if !extend && tab.view.mode == ViewMode::Wysiwyg {
                Self::skip_wysiwyg_table_delimiter(tab, raw_target, dir, sel.head)
            } else {
                raw_target
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

    fn skip_wysiwyg_table_delimiter(
        tab: &DocumentTab,
        target: usize,
        dir: i32,
        prev_head: usize,
    ) -> usize {
        let source = tab.session.buffer.to_string();
        tab.with_incremental_blocks(|blocks| {
            let idx = blocks.partition_point(|b| b.range.end < target);
            let Some(block) = blocks.get(idx) else {
                return target;
            };
            if !matches!(block.kind, BlockKind::Table) {
                return target;
            }
            let rows = parse_table_block(&source, block.range);
            if rows.len() < 2 || !rows[1].is_delimiter {
                return target;
            }
            let delim = &rows[1].line_range;
            if target < delim.start || target > delim.end {
                return target;
            }
            let header_end = rows[0]
                .cells
                .last()
                .map(|c| c.content_range.end)
                .unwrap_or(rows[0].line_range.end);
            if dir < 0 {
                return header_end;
            }
            if let Some(body0) = rows.get(2) {
                return body0
                    .cells
                    .first()
                    .map(|c| c.content_range.start)
                    .unwrap_or(body0.line_range.start);
            }
            // Two-row table (header + delimiter only): step to the following line if present,
            // otherwise keep caret out of the hidden delimiter row.
            let total = source.len();
            if delim.end < total {
                let mut after = delim.end;
                if source.as_bytes().get(after) == Some(&b'\r') {
                    after += 1;
                }
                if source.as_bytes().get(after) == Some(&b'\n') {
                    after += 1;
                }
                return after.min(total);
            }
            prev_head.min(rows[0].line_range.end)
        })
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

    fn on_overlay_input_changed(&mut self, target: ImeTarget) {
        match target {
            ImeTarget::FindBox => self.run_find_next(false),
            ImeTarget::QuickOpenBox => self.refresh_quick_open_matches(),
            ImeTarget::CodeLangBox => self.refresh_code_lang_matches(),
            ImeTarget::GlobalSearchBox => self.schedule_global_search(false),
            ImeTarget::CommandPaletteBox => self.refresh_command_palette_matches(),
            ImeTarget::InspectModalBox => self.inspect_modal_selected = 0,
            ImeTarget::ReplaceBox
            | ImeTarget::LinkPromptBox
            | ImeTarget::InsertTableBox
            | ImeTarget::FileNameBox
            | ImeTarget::None
            | ImeTarget::Tab(_) => {}
        }
    }

    fn overlay_input_text(&self, target: ImeTarget) -> &str {
        match target {
            ImeTarget::FindBox => &self.find_query,
            ImeTarget::ReplaceBox => &self.replace_text,
            ImeTarget::LinkPromptBox => &self.link_prompt_url,
            ImeTarget::QuickOpenBox => &self.quick_open_query,
            ImeTarget::CodeLangBox => &self.code_lang_query,
            ImeTarget::GlobalSearchBox => &self.global_search_query,
            ImeTarget::CommandPaletteBox => &self.command_palette_query,
            ImeTarget::InspectModalBox => &self.inspect_modal_query,
            ImeTarget::InsertTableBox => &self.insert_table_input,
            ImeTarget::FileNameBox => &self.file_prompt_input,
            ImeTarget::None | ImeTarget::Tab(_) => "",
        }
    }

    fn clear_overlay_input_text(&mut self, target: ImeTarget) {
        match target {
            ImeTarget::FindBox => self.find_query.clear(),
            ImeTarget::ReplaceBox => self.replace_text.clear(),
            ImeTarget::LinkPromptBox => self.link_prompt_url.clear(),
            ImeTarget::QuickOpenBox => self.quick_open_query.clear(),
            ImeTarget::CodeLangBox => self.code_lang_query.clear(),
            ImeTarget::GlobalSearchBox => self.global_search_query.clear(),
            ImeTarget::CommandPaletteBox => self.command_palette_query.clear(),
            ImeTarget::InspectModalBox => self.inspect_modal_query.clear(),
            ImeTarget::InsertTableBox => self.insert_table_input.clear(),
            ImeTarget::FileNameBox => self.file_prompt_input.clear(),
            ImeTarget::None | ImeTarget::Tab(_) => {}
        }
    }

    fn copy_selection(&mut self, cut: bool) {
        if let Some(target) = self.focused_overlay_ime_target() {
            let text = self.overlay_input_text(target).to_string();
            if !text.is_empty() {
                match self.clipboard.write(&text, None) {
                    Ok(()) => {
                        self.clipboard_text = text;
                        let noun = match target {
                            ImeTarget::FindBox => "search query",
                            ImeTarget::ReplaceBox => "replacement text",
                            _ => "text",
                        };
                        self.set_notice(if cut {
                            format!("Cut {noun}")
                        } else {
                            format!("Copied {noun}")
                        });
                    }
                    Err(err) => {
                        self.set_notice(format!("Clipboard error: {err}"));
                        return;
                    }
                }
                if cut {
                    self.clear_overlay_input_text(target);
                    self.on_overlay_input_changed(target);
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
                self.set_notice(if cut {
                    "Cut selection"
                } else {
                    "Copied Markdown"
                });
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
        let has_image =
            !self.has_focused_text_input() && plain.is_empty() && content.image_png.is_some();
        if plain.is_empty() && !has_html && !has_image {
            return;
        }
        if has_image && let Some(png) = content.image_png.as_deref() {
            self.close_confirm_armed = false;
            let active_id = self.workbench.active_tab().map(|t| t.id);
            self.cancel_abandoned_tab_ime(active_id);
            match self.workbench.paste_image_bytes_into_active(
                png,
                "pasted-image",
                "png",
                Some("image"),
            ) {
                Ok(dest) => {
                    self.pending_scroll_ensure_caret = true;
                    let name = dest
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("image.png");
                    self.set_notice(format!("Pasted image -> {name}"));
                }
                Err(e) => self.set_notice(format!("Image paste failed: {e}")),
            }
            return;
        }
        if !plain.is_empty() {
            self.clipboard_text = plain.clone();
        }
        if let Some(target) = self.focused_overlay_ime_target() {
            let clean = match target {
                ImeTarget::FindBox | ImeTarget::ReplaceBox => plain.as_str(),
                ImeTarget::LinkPromptBox
                | ImeTarget::QuickOpenBox
                | ImeTarget::CodeLangBox
                | ImeTarget::GlobalSearchBox
                | ImeTarget::CommandPaletteBox
                | ImeTarget::InspectModalBox
                | ImeTarget::InsertTableBox
                | ImeTarget::FileNameBox => plain.trim_matches(&['\r', '\n'][..]),
                ImeTarget::None | ImeTarget::Tab(_) => "",
            };
            if !clean.is_empty() {
                self.commit_overlay_text(target, clean);
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

    fn close_transient_prompts(&mut self) {
        self.replace_focused = false;
        self.link_prompt_open = false;
        self.quick_open_open = false;
        self.code_lang_open = false;
        self.code_lang_block_range = None;
        self.global_search_open = false;
        self.cancel_active_global_search();
        self.command_palette_open = false;
        self.inspect_modal = None;
        self.insert_table_open = false;
        self.file_prompt_mode = None;
        if matches!(
            self.ime_target,
            ImeTarget::LinkPromptBox
                | ImeTarget::QuickOpenBox
                | ImeTarget::CodeLangBox
                | ImeTarget::GlobalSearchBox
                | ImeTarget::CommandPaletteBox
                | ImeTarget::InspectModalBox
                | ImeTarget::InsertTableBox
                | ImeTarget::FileNameBox
        ) {
            self.ime_target = ImeTarget::None;
        }
    }

    fn active_dialog_dir(&self) -> Option<PathBuf> {
        self.workbench
            .active_tab()
            .and_then(|t| t.session.path.as_deref())
            .and_then(|p| p.parent())
            .map(Path::to_path_buf)
            .or_else(|| self.untitled_save_dir.clone())
            .or_else(|| self.workbench.workspace_root.clone())
    }

    /// Open a Markdown/text document picked via the configured dialog backend (`Cmd/Ctrl + O`).
    pub fn open_file_dialog(&mut self) {
        self.close_confirm_armed = false;
        self.close_transient_prompts();
        let dir = self.active_dialog_dir();
        let Some(path) = self.save_dialog.pick_open_path(dir.as_deref()) else {
            return;
        };
        self.flush_active_tab_before_switch();
        match self.workbench.open_document(&path) {
            Ok(_) => {
                self.find_focused = false;
                self.ime_target = ImeTarget::None;
                self.preferred_caret_x = None;
                self.pending_scroll_ensure_caret = true;
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("file");
                self.set_notice(format!("Opened {name}"));
            }
            Err(e) => self.set_notice(format!("Open failed: {e}")),
        }
    }

    /// Open a workspace folder picked via the configured dialog backend (`Cmd/Ctrl + Shift + O`).
    pub fn open_folder_dialog(&mut self) {
        self.close_confirm_armed = false;
        self.close_transient_prompts();
        let dir = self.active_dialog_dir();
        let Some(folder) = self.save_dialog.pick_open_folder(dir.as_deref()) else {
            return;
        };
        let normalized: PathBuf = folder.components().collect();
        match self.workbench.open_folder(&normalized) {
            Ok(()) => {
                if !self.workbench.config.sidebar_visible {
                    self.workbench.config.sidebar_visible = true;
                    self.persist_user_config();
                }
                self.sidebar_section = SidebarSection::Files;
                self.collapsed_dirs.clear();
                self.selected_workspace_path = None;
                self.delete_confirm_path = None;
                self.find_focused = false;
                let name = normalized
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("folder");
                self.set_notice(format!("Opened folder: {name}"));
            }
            Err(e) => self.set_notice(format!("Open folder failed: {e}")),
        }
    }

    /// Show Quick Open fuzzy file switcher (`Cmd/Ctrl + P`, PRD F-06).
    pub fn open_quick_open(&mut self) {
        self.close_confirm_armed = false;
        self.close_transient_prompts();
        self.find_focused = false;
        let active_id = self.workbench.active_tab().map(|t| t.id);
        self.cancel_abandoned_tab_ime(active_id);
        self.quick_open_open = true;
        self.quick_open_query.clear();
        self.refresh_quick_open_matches();
    }

    fn close_quick_open(&mut self) {
        self.quick_open_open = false;
        if self.ime_target == ImeTarget::QuickOpenBox {
            self.ime_target = ImeTarget::None;
        }
    }

    fn refresh_quick_open_matches(&mut self) {
        self.quick_open_matches = self.workbench.quick_open(&self.quick_open_query);
        self.quick_open_selected = 0;
    }

    fn confirm_quick_open_match(&mut self, idx: usize) {
        let Some(picked) = self.quick_open_matches.get(idx).cloned() else {
            self.close_quick_open();
            return;
        };
        self.close_quick_open();
        self.find_focused = false;
        self.flush_active_tab_before_switch();
        match self.workbench.open_document(&picked.path) {
            Ok(_) => {
                self.ime_target = ImeTarget::None;
                self.preferred_caret_x = None;
                self.pending_scroll_ensure_caret = true;
                self.set_notice(format!("Opened {}", picked.relative_path));
            }
            Err(e) => self.set_notice(format!("Open failed: {e}")),
        }
    }

    /// Show Command Palette (`Cmd/Ctrl + Shift + P`, PRD 4.9 P-02).
    pub fn open_command_palette(&mut self) {
        self.close_confirm_armed = false;
        self.close_transient_prompts();
        self.find_focused = false;
        self.replace_focused = false;
        let active_id = self.workbench.active_tab().map(|t| t.id);
        self.cancel_abandoned_tab_ime(active_id);
        self.command_palette_open = true;
        self.command_palette_query.clear();
        self.refresh_command_palette_matches();
    }

    fn close_command_palette(&mut self) {
        self.command_palette_open = false;
        if self.ime_target == ImeTarget::CommandPaletteBox {
            self.ime_target = ImeTarget::None;
        }
    }

    fn refresh_command_palette_matches(&mut self) {
        let terms: Vec<String> = self
            .command_palette_query
            .split_whitespace()
            .map(|t| t.to_ascii_lowercase())
            .collect();
        self.command_palette_matches = PALETTE_COMMANDS
            .iter()
            .filter(|entry| {
                let label = entry.label.to_ascii_lowercase();
                let shortcut = entry.shortcut.to_ascii_lowercase();
                terms
                    .iter()
                    .all(|term| label.contains(term) || shortcut.contains(term))
            })
            .cloned()
            .collect();
        self.command_palette_selected = 0;
    }

    fn confirm_command_palette_match(&mut self, idx: usize) {
        let Some(picked) = self.command_palette_matches.get(idx).cloned() else {
            self.close_command_palette();
            return;
        };
        self.close_command_palette();
        self.execute_palette_command(picked.id);
    }

    fn persist_user_config(&mut self) {
        let _ = self.workbench.persist_config();
    }

    fn poll_custom_theme_hot_reload(&mut self, force: bool) -> bool {
        let Some((name, diag_count)) = self.workbench.poll_custom_theme_reload(force) else {
            return false;
        };
        self.refresh_style();
        if diag_count > 0 {
            self.set_notice(format!(
                "Hot-reloaded theme CSS: {name} ({diag_count} unsupported rule(s) ignored)"
            ));
        } else {
            self.set_notice(format!("Hot-reloaded theme CSS: {name}"));
        }
        true
    }

    fn reload_custom_theme_now(&mut self) {
        let Some(path) = self.workbench.config.custom_theme_path.clone() else {
            self.set_notice("No custom theme CSS configured");
            return;
        };
        match self.workbench.load_custom_theme_file(&path) {
            Ok(diag_count) => {
                self.refresh_style();
                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("custom.css");
                if diag_count > 0 {
                    self.set_notice(format!(
                        "Reloaded theme CSS: {name} ({diag_count} unsupported rule(s) ignored)"
                    ));
                } else {
                    self.set_notice(format!("Reloaded theme CSS: {name}"));
                }
            }
            Err(err) => self.set_notice(format!("Failed to reload CSS {}: {err}", path.display())),
        }
    }

    /// Open an inspect/management modal (`ThemeDiagnostics` or `RecoveryBackups`).
    pub fn open_inspect_modal(&mut self, kind: InspectModalKind) {
        self.close_confirm_armed = false;
        self.close_transient_prompts();
        self.find_focused = false;
        self.replace_focused = false;
        let active_id = self.workbench.active_tab().map(|t| t.id);
        self.cancel_abandoned_tab_ime(active_id);
        self.inspect_modal = Some(kind);
        self.inspect_modal_query.clear();
        self.inspect_modal_selected = 0;
        if kind == InspectModalKind::RecoveryBackups {
            self.inspect_recovery_backups = self.workbench.list_history_backups();
        }
    }

    fn close_inspect_modal(&mut self) {
        self.inspect_modal = None;
        if self.ime_target == ImeTarget::InspectModalBox {
            self.ime_target = ImeTarget::None;
        }
    }

    fn filtered_theme_diagnostic_indices(&self) -> Vec<usize> {
        let terms: Vec<String> = self
            .inspect_modal_query
            .split_whitespace()
            .map(|t| t.to_ascii_lowercase())
            .collect();
        self.workbench
            .theme
            .diagnostics
            .iter()
            .enumerate()
            .filter_map(|(idx, d)| {
                if terms.is_empty() {
                    return Some(idx);
                }
                let hay = format!("{} {} {}", d.selector, d.item, d.reason).to_ascii_lowercase();
                terms.iter().all(|term| hay.contains(term)).then_some(idx)
            })
            .collect()
    }

    fn filtered_recovery_backup_indices(&self) -> Vec<usize> {
        let terms: Vec<String> = self
            .inspect_modal_query
            .split_whitespace()
            .map(|t| t.to_ascii_lowercase())
            .collect();
        self.inspect_recovery_backups
            .iter()
            .enumerate()
            .filter_map(|(idx, entry)| {
                if terms.is_empty() {
                    return Some(idx);
                }
                let source = entry
                    .source_path
                    .as_deref()
                    .map(|p| p.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "untitled".to_string());
                let hay = format!(
                    "{source} {} {}",
                    entry.reason.as_slug(),
                    entry.path.to_string_lossy()
                )
                .to_ascii_lowercase();
                terms.iter().all(|term| hay.contains(term)).then_some(idx)
            })
            .collect()
    }

    fn filtered_inspect_modal_len(&self) -> usize {
        match self.inspect_modal {
            Some(InspectModalKind::ThemeDiagnostics) => {
                self.filtered_theme_diagnostic_indices().len()
            }
            Some(InspectModalKind::RecoveryBackups) => {
                self.filtered_recovery_backup_indices().len()
            }
            None => 0,
        }
    }

    fn confirm_inspect_modal_row(&mut self, slot_idx: usize) {
        let Some(kind) = self.inspect_modal else {
            return;
        };
        match kind {
            InspectModalKind::ThemeDiagnostics => {
                let indices = self.filtered_theme_diagnostic_indices();
                let Some(&diag_idx) = indices.get(slot_idx) else {
                    return;
                };
                self.inspect_modal_selected = slot_idx;
                if let Some(d) = self.workbench.theme.diagnostics.get(diag_idx) {
                    self.set_notice(format!("CSS [{}] {}: {}", d.selector, d.item, d.reason));
                }
            }
            InspectModalKind::RecoveryBackups => {
                let indices = self.filtered_recovery_backup_indices();
                let Some(&entry_idx) = indices.get(slot_idx) else {
                    return;
                };
                let Some(entry) = self.inspect_recovery_backups.get(entry_idx).cloned() else {
                    return;
                };
                self.close_inspect_modal();
                self.flush_active_tab_before_switch();
                match self.workbench.restore_history_backup_as_draft(&entry.path) {
                    Ok(_) => {
                        self.find_focused = false;
                        self.ime_target = ImeTarget::None;
                        self.preferred_caret_x = None;
                        self.pending_scroll_ensure_caret = true;
                        let source_name = entry
                            .source_path
                            .as_deref()
                            .and_then(|p| p.file_name())
                            .and_then(|n| n.to_str())
                            .unwrap_or("untitled draft");
                        self.set_notice(format!(
                            "Restored backup ({}) from {source_name} as new draft tab",
                            entry.reason.as_slug()
                        ));
                    }
                    Err(err) => self.set_notice(format!("Restore backup failed: {err}")),
                }
            }
        }
    }

    fn delete_selected_recovery_backup(&mut self) {
        if self.inspect_modal != Some(InspectModalKind::RecoveryBackups) {
            return;
        }
        let indices = self.filtered_recovery_backup_indices();
        if indices.is_empty() {
            self.set_notice("No recovery backup selected");
            return;
        }
        let slot = self.inspect_modal_selected.min(indices.len() - 1);
        let Some(&entry_idx) = indices.get(slot) else {
            return;
        };
        let Some(entry) = self.inspect_recovery_backups.get(entry_idx).cloned() else {
            return;
        };
        match self.workbench.delete_history_backup(&entry.path) {
            Ok(deleted) => {
                self.inspect_recovery_backups = self.workbench.list_history_backups();
                let new_len = self.filtered_recovery_backup_indices().len();
                self.inspect_modal_selected = if new_len == 0 {
                    0
                } else {
                    slot.min(new_len - 1)
                };
                self.trigger_background_maintenance(true);
                let name = entry
                    .source_path
                    .as_deref()
                    .and_then(|p| p.file_name())
                    .and_then(|n| n.to_str())
                    .unwrap_or("backup");
                if deleted {
                    self.set_notice(format!("Deleted recovery backup for {name}"));
                } else {
                    self.set_notice(format!("Recovery backup for {name} was already removed"));
                }
            }
            Err(err) => self.set_notice(format!("Delete backup failed: {err}")),
        }
    }

    fn apply_builtin_theme_selection(
        &mut self,
        follow_system: bool,
        theme_opt: Option<BuiltinTheme>,
    ) {
        self.workbench
            .select_builtin_theme(follow_system, theme_opt);
        self.refresh_style();
        self.persist_user_config();
        let effective = self
            .workbench
            .config
            .effective_builtin_theme(self.workbench.system_prefers_dark);
        if follow_system {
            self.set_notice("Theme: Follow System Appearance");
        } else {
            self.set_notice(format!("Theme: {effective:?}"));
        }
    }

    fn set_prose_width_px(&mut self, width_px: u32, label: &str) {
        self.workbench.config.max_content_width_px = width_px;
        self.refresh_style();
        self.persist_user_config();
        self.set_notice(format!("Prose width: {label} ({width_px}px)"));
    }

    fn execute_palette_command(&mut self, id: PaletteCommandId) {
        self.close_confirm_armed = false;
        match id {
            PaletteCommandId::FileNew => {
                self.close_transient_prompts();
                self.flush_active_tab_before_switch();
                self.workbench.new_document();
                self.find_focused = false;
                self.replace_focused = false;
                self.pending_scroll_ensure_caret = true;
                self.set_notice("New Document");
            }
            PaletteCommandId::FileOpen => self.open_file_dialog(),
            PaletteCommandId::FileOpenFolder => self.open_folder_dialog(),
            PaletteCommandId::FileSave => self.save_active_tab(),
            PaletteCommandId::FileSaveAs => self.save_as_active_tab(),
            PaletteCommandId::ExportHtmlSafe => {
                self.export_active_html_dialog(HtmlExportMode::Safe)
            }
            PaletteCommandId::ExportHtmlRaw => self.export_active_html_dialog(HtmlExportMode::Raw),
            PaletteCommandId::ManageRecoveryBackups => {
                self.open_inspect_modal(InspectModalKind::RecoveryBackups);
            }
            PaletteCommandId::QuickOpen => self.open_quick_open(),
            PaletteCommandId::GlobalSearch => self.open_global_search(),
            PaletteCommandId::Find => self.open_find_bar(false),
            PaletteCommandId::FindAndReplace => self.open_find_bar(true),
            PaletteCommandId::ToggleSourceMode => {
                self.dispatch_editor(EditorAction::ToggleSourceMode);
                self.preferred_caret_x = None;
            }
            PaletteCommandId::ToggleFocusMode => self.toggle_focus_mode(),
            PaletteCommandId::ToggleTypewriterMode => self.toggle_typewriter_mode(),
            PaletteCommandId::ToggleFullscreen => self.toggle_fullscreen(),
            PaletteCommandId::ToggleSidebar => {
                self.workbench.config.sidebar_visible = !self.workbench.config.sidebar_visible;
                self.persist_user_config();
            }
            PaletteCommandId::InsertTable => self.open_insert_table_prompt(),
            PaletteCommandId::InsertCodeBlock => self.insert_or_pick_code_block(),
            PaletteCommandId::InsertLink => self.open_link_prompt(),
            PaletteCommandId::ToggleTableAutoAlign => {
                self.workbench.config.table_auto_align = !self.workbench.config.table_auto_align;
                self.persist_user_config();
                let on = self.workbench.config.table_auto_align;
                if let Some(tab) = self.workbench.active_tab_mut() {
                    tab.view.table_auto_align = on;
                }
                self.set_notice(if on {
                    "Table Auto-Align: ON"
                } else {
                    "Table Auto-Align: OFF"
                });
            }
            PaletteCommandId::ToggleCodeLineNumbers => {
                let _ = self.execute_ui_action(UiAction::ToggleCodeLineNumbers);
            }
            PaletteCommandId::ToggleNetworkImages => {
                self.workbench.config.allow_network_images =
                    !self.workbench.config.allow_network_images;
                self.persist_user_config();
                self.image_cache.invalidate_remote_entries();
                let on = self.workbench.config.allow_network_images;
                self.set_notice(if on {
                    "Network Images: ON"
                } else {
                    "Network Images: OFF"
                });
            }
            PaletteCommandId::ThemeFollowSystem => {
                self.apply_builtin_theme_selection(true, None);
            }
            PaletteCommandId::ThemeLight => {
                self.apply_builtin_theme_selection(false, Some(BuiltinTheme::Light));
            }
            PaletteCommandId::ThemeDark => {
                self.apply_builtin_theme_selection(false, Some(BuiltinTheme::Dark));
            }
            PaletteCommandId::ThemeAcademic => {
                self.apply_builtin_theme_selection(false, Some(BuiltinTheme::Academic));
            }
            PaletteCommandId::ThemeGithub => {
                self.apply_builtin_theme_selection(false, Some(BuiltinTheme::GitHub));
            }
            PaletteCommandId::LoadCustomThemeCss => self.load_custom_theme_dialog(),
            PaletteCommandId::ShowThemeDiagnostics => {
                self.open_inspect_modal(InspectModalKind::ThemeDiagnostics);
            }
            PaletteCommandId::ProseWidthNarrow => self.set_prose_width_px(720, "Compact"),
            PaletteCommandId::ProseWidthBalanced => self.set_prose_width_px(860, "Balanced"),
            PaletteCommandId::ProseWidthWide => self.set_prose_width_px(1080, "Wide"),
            PaletteCommandId::ZoomIn => {
                self.workbench.config.zoom_in();
                self.refresh_style();
                self.persist_user_config();
                self.set_notice(format!("Zoom {}%", self.workbench.config.zoom_percent));
            }
            PaletteCommandId::ZoomOut => {
                self.workbench.config.zoom_out();
                self.refresh_style();
                self.persist_user_config();
                self.set_notice(format!("Zoom {}%", self.workbench.config.zoom_percent));
            }
            PaletteCommandId::ZoomReset => {
                self.workbench.config.reset_zoom();
                self.refresh_style();
                self.persist_user_config();
                self.set_notice("Zoom 100%");
            }
        }
    }

    fn export_active_html_dialog(&mut self, mode: HtmlExportMode) {
        self.close_confirm_armed = false;
        let Some((current_path, default_name)) = self.workbench.active_tab().map(|t| {
            let stem = t
                .session
                .path
                .as_deref()
                .and_then(|p| p.file_stem())
                .and_then(|n| n.to_str())
                .filter(|s| !s.is_empty())
                .unwrap_or("Untitled");
            (t.session.path.clone(), format!("{stem}.html"))
        }) else {
            return;
        };
        let Some(html) = self.workbench.export_active_html(mode) else {
            return;
        };
        let Some(dest) = self
            .save_dialog
            .pick_save_as_path(current_path.as_deref(), &default_name)
        else {
            self.set_notice("HTML export cancelled");
            return;
        };
        if let Some(parent) = dest.parent()
            && !parent.as_os_str().is_empty()
            && let Err(err) = fs::create_dir_all(parent)
        {
            self.set_notice(format!("HTML export failed: {err}"));
            return;
        }
        match atomic_write_file(&dest, html.as_bytes()) {
            Ok(()) => {
                let name = dest
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("export.html");
                let mode_label = match mode {
                    HtmlExportMode::Safe => "Safe",
                    HtmlExportMode::Raw => "Raw",
                };
                self.set_notice(format!("Exported HTML ({mode_label}): {name}"));
            }
            Err(err) => self.set_notice(format!("HTML export failed: {err}")),
        }
    }

    fn load_custom_theme_dialog(&mut self) {
        self.close_confirm_armed = false;
        self.close_transient_prompts();
        let dir = self
            .workbench
            .config
            .custom_theme_path
            .as_deref()
            .and_then(|p| p.parent())
            .map(Path::to_path_buf)
            .or_else(|| self.active_dialog_dir());
        let Some(path) = self.save_dialog.pick_open_path(dir.as_deref()) else {
            self.set_notice("Load Custom CSS cancelled");
            return;
        };
        let diag_count = match self.workbench.load_custom_theme_file(&path) {
            Ok(count) => count,
            Err(err) => {
                self.set_notice(format!("Failed to read CSS {}: {err}", path.display()));
                return;
            }
        };
        self.refresh_style();
        self.persist_user_config();
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("custom.css");
        if diag_count > 0 {
            self.set_notice(format!(
                "Loaded theme CSS: {name} ({diag_count} unsupported rule(s) — open Theme Diagnostics)"
            ));
        } else {
            self.set_notice(format!("Loaded theme CSS: {name}"));
        }
    }

    fn save_utf8_copy_dialog(&mut self) {
        self.close_confirm_armed = false;
        let Some((tab_id, orig_path, content, default_name)) =
            self.workbench.active_tab().map(|t| {
                let p = t.session.path.clone();
                let default_name = match p.as_deref() {
                    Some(path) => {
                        let stem = path
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .filter(|s| !s.is_empty())
                            .unwrap_or("Untitled");
                        match path.extension().and_then(|e| e.to_str()) {
                            Some(ext) if !ext.is_empty() => format!("{stem}-utf8.{ext}"),
                            _ => format!("{stem}-utf8.md"),
                        }
                    }
                    None => "Untitled-utf8.md".to_string(),
                };
                (t.id, p, t.session.buffer.to_string(), default_name)
            })
        else {
            return;
        };
        let Some(dest) = self
            .save_dialog
            .pick_save_as_path(orig_path.as_deref(), &default_name)
        else {
            self.set_notice("Save UTF-8 Copy cancelled");
            return;
        };
        if let Some(orig) = orig_path.as_deref()
            && paths_share_underlying_file(orig, &dest)
        {
            self.set_notice(
                "Choose a new path — cannot overwrite original non-UTF-8 file directly",
            );
            return;
        }
        if let Some((other_idx, _, other_title)) =
            self.workbench.find_other_tab_sharing_file(tab_id, &dest)
        {
            self.set_notice(format!(
                "'{other_title}' is already open in tab {} — close that tab first or pick another file",
                other_idx + 1
            ));
            return;
        }
        if let Some(parent) = dest.parent()
            && !parent.as_os_str().is_empty()
            && let Err(err) = fs::create_dir_all(parent)
        {
            self.set_notice(format!("Save UTF-8 Copy failed: {err}"));
            return;
        }
        if let Err(err) = atomic_write_file(&dest, content.as_bytes()) {
            self.set_notice(format!("Save UTF-8 Copy failed: {err}"));
            return;
        }
        self.flush_active_tab_before_switch();
        match self.workbench.open_document(&dest) {
            Ok(_) => {
                self.ime_target = ImeTarget::None;
                self.preferred_caret_x = None;
                self.pending_scroll_ensure_caret = true;
                let name = dest
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("copy.md");
                self.set_notice(format!("Saved editable UTF-8 copy: {name}"));
            }
            Err(err) => self.set_notice(format!("Saved copy, but open failed: {err}")),
        }
    }

    /// Show Global Workspace Search modal (`Cmd/Ctrl + Shift + F`, PRD 4.5 F-07).
    pub fn open_global_search(&mut self) {
        self.close_confirm_armed = false;
        self.close_transient_prompts();
        self.find_focused = false;
        let active_id = self.workbench.active_tab().map(|t| t.id);
        self.cancel_abandoned_tab_ime(active_id);
        self.global_search_open = true;
        self.global_search_query.clear();
        self.schedule_global_search(false);
    }

    fn close_global_search(&mut self) {
        self.global_search_open = false;
        self.cancel_active_global_search();
        if self.ime_target == ImeTarget::GlobalSearchBox {
            self.ime_target = ImeTarget::None;
        }
    }

    fn cancel_active_global_search(&mut self) {
        if let Some(cancel) = self.global_search_cancel.take() {
            cancel.store(true, Ordering::Release);
        }
        self.global_search_debounce_deadline = None;
        self.global_search_searching = false;
    }

    fn schedule_global_search(&mut self, immediate: bool) {
        self.global_search_req_id = self.global_search_req_id.wrapping_add(1);
        self.cancel_active_global_search();
        self.global_search_matches.clear();
        self.global_search_selected = 0;

        if self.global_search_query.trim().is_empty() || self.workbench.workspace_root.is_none() {
            return;
        }
        self.global_search_searching = true;
        if immediate {
            self.dispatch_global_search_worker();
        } else {
            self.global_search_debounce_deadline = Some(Instant::now() + GLOBAL_SEARCH_DEBOUNCE);
        }
    }

    fn dispatch_global_search_worker(&mut self) {
        self.global_search_debounce_deadline = None;
        if !self.global_search_open || self.global_search_query.trim().is_empty() {
            self.global_search_searching = false;
            return;
        }
        let Some(root) = self.workbench.workspace_root.clone() else {
            self.global_search_searching = false;
            return;
        };
        if self
            .global_search_worker
            .as_ref()
            .is_some_and(|w| !w.is_alive())
        {
            self.global_search_worker = None;
        }
        let cancel = Arc::new(AtomicBool::new(false));
        self.global_search_cancel = Some(Arc::clone(&cancel));
        let notifier = self.redraw_notifier.clone();
        let scheduled = {
            let worker = self.global_search_worker.get_or_insert_with(|| {
                WorkspaceSearchWorker::spawn(notifier, GLOBAL_SEARCH_MAX_RESULTS)
            });
            worker.schedule(WorkspaceSearchRequest {
                req_id: self.global_search_req_id,
                root,
                query: self.global_search_query.clone(),
                case_sensitive: self.global_search_case_sensitive,
                cancel,
            })
        };
        if !scheduled {
            self.global_search_worker = None;
            self.global_search_searching = false;
            self.global_search_cancel = None;
            self.set_notice("Workspace search unavailable (failed to start worker thread)");
        }
    }

    fn poll_global_search(&mut self, now: Instant) -> bool {
        let mut changed = false;
        if let Some(dl) = self.global_search_debounce_deadline
            && now >= dl
        {
            self.dispatch_global_search_worker();
            changed = true;
        }
        let outcome = self
            .global_search_worker
            .as_ref()
            .map(|w| w.drain_responses());
        if let Some(outcome) = outcome {
            for resp in outcome.responses {
                if self.global_search_open && resp.req_id == self.global_search_req_id {
                    self.global_search_searching = false;
                    self.global_search_cancel = None;
                    changed = true;
                    match resp.result {
                        Ok(matches) => {
                            self.global_search_matches = matches;
                            self.global_search_selected = 0;
                        }
                        Err(e) => {
                            self.global_search_matches.clear();
                            self.global_search_selected = 0;
                            self.set_notice(format!("Workspace search failed: {e}"));
                        }
                    }
                }
            }
            if outcome.disconnected {
                self.global_search_worker = None;
                if self.global_search_searching {
                    self.global_search_searching = false;
                    self.global_search_cancel = None;
                    changed = true;
                    self.set_notice("Workspace search worker stopped unexpectedly");
                }
            }
        }
        changed
    }

    fn confirm_global_search_match(&mut self, idx: usize) {
        if self.global_search_matches.is_empty() && self.global_search_searching {
            if self.global_search_debounce_deadline.is_some() {
                self.dispatch_global_search_worker();
            }
            return;
        }
        let Some(picked) = self.global_search_matches.get(idx).cloned() else {
            self.close_global_search();
            return;
        };
        self.close_global_search();
        self.find_focused = false;
        self.flush_active_tab_before_switch();
        match self.workbench.open_document(&picked.path) {
            Ok(_) => {
                self.ime_target = ImeTarget::None;
                self.preferred_caret_x = None;
                if let Some(tab) = self.workbench.active_tab_mut() {
                    let safe = tab.session.buffer.snap_out_of_crlf(
                        picked.line_byte_offset.min(tab.session.buffer.len_bytes()),
                    );
                    let _ = tab.edit(|ctx| ctx.set_caret(safe));
                }
                self.pending_scroll_ensure_caret = true;
                let name = picked
                    .path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("file");
                self.set_notice(format!("Opened {name}:{}", picked.line_number));
            }
            Err(e) => self.set_notice(format!("Open failed: {e}")),
        }
    }

    /// Show Insert Table dimensions prompt (`Cmd+Option+T` / `Cmd/Ctrl+T`, PRD 4.3 T-01).
    pub fn open_insert_table_prompt(&mut self) {
        if self.workbench.active_tab().is_none() {
            self.set_notice("No active document");
            return;
        }
        self.close_confirm_armed = false;
        self.close_transient_prompts();
        self.find_focused = false;
        let active_id = self.workbench.active_tab().map(|t| t.id);
        self.cancel_abandoned_tab_ime(active_id);
        self.insert_table_input = "3x4".to_string();
        self.insert_table_open = true;
    }

    fn target_for_workspace_entry(&self) -> Option<PathBuf> {
        self.selected_workspace_path.clone().or_else(|| {
            self.workbench
                .active_tab()
                .and_then(|t| t.session.path.clone())
        })
    }

    fn open_new_file_prompt(&mut self) {
        let parent_dir = match self.target_for_workspace_entry() {
            Some(p) if p.is_dir() => Some(p),
            Some(p) => p.parent().map(Path::to_path_buf),
            None => None,
        }
        .or_else(|| self.workbench.workspace_root.clone())
        .or_else(|| self.active_dialog_dir());
        let Some(parent_dir) = parent_dir else {
            self.set_notice("Open a folder first to create files");
            return;
        };
        self.close_confirm_armed = false;
        self.close_transient_prompts();
        self.find_focused = false;
        let active_id = self.workbench.active_tab().map(|t| t.id);
        self.cancel_abandoned_tab_ime(active_id);
        self.file_prompt_input.clear();
        self.file_prompt_mode = Some(FilePromptMode::CreateFile { parent_dir });
    }

    fn open_rename_file_prompt(&mut self) {
        let Some(src) = self.target_for_workspace_entry() else {
            self.set_notice("Select a file in the sidebar first");
            return;
        };
        if self.workbench.workspace_root.as_ref() == Some(&src) {
            self.set_notice("Cannot rename workspace root folder here");
            return;
        }
        self.close_confirm_armed = false;
        self.close_transient_prompts();
        self.find_focused = false;
        let active_id = self.workbench.active_tab().map(|t| t.id);
        self.cancel_abandoned_tab_ime(active_id);
        self.file_prompt_input = src
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_string();
        self.file_prompt_mode = Some(FilePromptMode::RenameFile { source_path: src });
    }

    fn arm_or_confirm_delete_file(&mut self) {
        let Some(target) = self.target_for_workspace_entry() else {
            self.set_notice("Select a file in the sidebar first");
            return;
        };
        if target.is_dir() {
            self.delete_confirm_path = None;
            self.set_notice("Only files can be deleted from the sidebar");
            return;
        }
        let name = target
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("file")
            .to_string();
        if self.delete_confirm_path.as_ref() == Some(&target) {
            self.delete_confirm_path = None;
            match self.workbench.delete_workspace_file(&target) {
                Ok(()) => {
                    if self.selected_workspace_path.as_ref() == Some(&target) {
                        self.selected_workspace_path = None;
                    }
                    self.set_notice(format!("Deleted {name}"));
                }
                Err(e) => self.set_notice(format!("Delete failed: {e}")),
            }
        } else {
            self.delete_confirm_path = Some(target);
            self.set_notice(format!("Click Del again to delete {name}"));
        }
    }

    fn close_simple_prompt(&mut self) {
        self.insert_table_open = false;
        self.file_prompt_mode = None;
        if matches!(
            self.ime_target,
            ImeTarget::InsertTableBox | ImeTarget::FileNameBox
        ) {
            self.ime_target = ImeTarget::None;
        }
    }

    fn confirm_simple_prompt(&mut self) {
        if self.insert_table_open {
            let Some((rows, cols)) = parse_table_dimensions(&self.insert_table_input) else {
                self.set_notice("Enter table dimensions as rows x cols (e.g. 3x4)");
                return;
            };
            self.close_simple_prompt();
            match self.dispatch_editor(EditorAction::InsertTable { rows, cols }) {
                Some(true) => self.set_notice(format!("Inserted {rows}×{cols} table")),
                Some(false) => self.set_notice("Cannot insert table inside a code block"),
                None => {}
            }
            return;
        }

        if let Some(mode) = self.file_prompt_mode.clone() {
            let trimmed = self.file_prompt_input.trim().to_string();
            if trimmed.is_empty() {
                self.set_notice("Filename cannot be empty");
                return;
            }
            match mode {
                FilePromptMode::CreateFile { parent_dir } => {
                    let final_name = if trimmed.contains('.') {
                        trimmed
                    } else {
                        format!("{trimmed}.md")
                    };
                    self.close_simple_prompt();
                    match self
                        .workbench
                        .create_workspace_document(&parent_dir, &final_name)
                    {
                        Ok(created) => {
                            self.flush_active_tab_before_switch();
                            self.selected_workspace_path = Some(created.clone());
                            match self.workbench.open_document(&created) {
                                Ok(_) => {
                                    self.find_focused = false;
                                    self.ime_target = ImeTarget::None;
                                    self.preferred_caret_x = None;
                                    self.pending_scroll_ensure_caret = true;
                                    self.set_notice(format!("Created {final_name}"));
                                }
                                Err(e) => {
                                    self.set_notice(format!("Created {final_name} (open err: {e})"))
                                }
                            }
                        }
                        Err(e) => self.set_notice(format!("Create failed: {e}")),
                    }
                }
                FilePromptMode::RenameFile { source_path } => {
                    self.close_simple_prompt();
                    match self
                        .workbench
                        .rename_workspace_entry(&source_path, &trimmed)
                    {
                        Ok(dest) => {
                            self.selected_workspace_path = Some(dest);
                            self.set_notice(format!("Renamed -> {trimmed}"));
                        }
                        Err(e) => self.set_notice(format!("Rename failed: {e}")),
                    }
                }
            }
        }
    }

    fn execute_table_toolbar_action(&mut self, kind: TableButtonKind) {
        let action = match kind {
            TableButtonKind::AddRowAbove => EditorAction::InsertTableRowAbove,
            TableButtonKind::AddRowBelow => EditorAction::InsertTableRowBelow,
            TableButtonKind::DeleteRow => EditorAction::DeleteCurrentTableRow,
            TableButtonKind::AddColLeft => EditorAction::InsertTableColLeft,
            TableButtonKind::AddColRight => EditorAction::InsertTableColRight,
            TableButtonKind::DeleteCol => EditorAction::DeleteCurrentTableCol,
            TableButtonKind::AlignLeft => EditorAction::SetTableColumnAlign(TableAlign::Left),
            TableButtonKind::AlignCenter => EditorAction::SetTableColumnAlign(TableAlign::Center),
            TableButtonKind::AlignRight => EditorAction::SetTableColumnAlign(TableAlign::Right),
            TableButtonKind::AlignFormat => EditorAction::AlignCurrentTable,
        };
        self.dispatch_editor(action);
    }

    pub fn toggle_focus_mode(&mut self) {
        self.workbench.config.focus_mode = !self.workbench.config.focus_mode;
        self.persist_user_config();
        let on = self.workbench.config.focus_mode;
        self.set_notice(if on {
            "Focus Mode: ON"
        } else {
            "Focus Mode: OFF"
        });
    }

    pub fn toggle_typewriter_mode(&mut self) {
        self.workbench.config.typewriter_mode = !self.workbench.config.typewriter_mode;
        self.persist_user_config();
        let on = self.workbench.config.typewriter_mode;
        if on {
            self.pending_scroll_ensure_caret = true;
        }
        self.set_notice(if on {
            "Typewriter Mode: ON"
        } else {
            "Typewriter Mode: OFF"
        });
    }

    pub fn toggle_fullscreen(&mut self) {
        self.workbench.config.fullscreen = !self.workbench.config.fullscreen;
        self.persist_user_config();
        let on = self.workbench.config.fullscreen;
        self.set_notice(if on {
            "Fullscreen: ON"
        } else {
            "Fullscreen: OFF"
        });
    }

    /// Insert a fenced code block, or open the Code Language Picker if the caret is already inside one.
    pub fn insert_or_pick_code_block(&mut self) {
        if self.code_lang_open {
            self.close_code_lang_picker();
            return;
        }
        if let Some((block_range, current_language)) = self.active_caret_code_fence() {
            self.open_code_lang_picker(block_range, current_language);
        } else {
            self.dispatch_editor(EditorAction::InsertCodeBlock(String::new()));
        }
    }

    fn active_caret_code_fence(&self) -> Option<(ByteRange, String)> {
        let tab = self.workbench.active_tab()?;
        let caret = tab.view.selection.head;
        let (found_range, info_string) = tab.with_incremental_blocks(|blocks| {
            blocks.iter().find_map(|b| match &b.kind {
                BlockKind::CodeFence { info_string }
                    if caret >= b.range.start && caret <= b.range.end =>
                {
                    Some((b.range, info_string.clone()))
                }
                _ => None,
            })
        })?;
        let block_text = tab.session.buffer.slice_to_string(found_range);
        if caret == found_range.end && block_text.ends_with('\n') {
            return None;
        }
        Some((found_range, info_string))
    }

    /// Show the Code Language Picker modal bound to a specific fenced code block.
    pub fn open_code_lang_picker(&mut self, block_range: ByteRange, current_language: String) {
        self.close_confirm_armed = false;
        self.close_transient_prompts();
        self.find_focused = false;
        let active_id = self.workbench.active_tab().map(|t| t.id);
        self.cancel_abandoned_tab_ime(active_id);
        self.code_lang_block_range = Some(block_range);
        self.code_lang_query.clear();
        self.refresh_code_lang_matches();
        let normalized = normalize_fence_language_token(&current_language);
        if !normalized.is_empty()
            && let Some(pos) = self
                .code_lang_matches
                .iter()
                .position(|e| e.token == normalized)
        {
            self.code_lang_selected = pos;
        }
        self.code_lang_open = true;
    }

    fn close_code_lang_picker(&mut self) {
        self.code_lang_open = false;
        self.code_lang_block_range = None;
        if self.ime_target == ImeTarget::CodeLangBox {
            self.ime_target = ImeTarget::None;
        }
    }

    fn refresh_code_lang_matches(&mut self) {
        let q = self.code_lang_query.trim().to_ascii_lowercase();
        self.code_lang_matches = code_language_catalog()
            .iter()
            .filter(|entry| {
                q.is_empty()
                    || entry.label.to_ascii_lowercase().contains(&q)
                    || entry.token.to_ascii_lowercase().contains(&q)
            })
            .collect();
        self.code_lang_selected = 0;
    }

    fn confirm_code_lang_match(&mut self, idx: usize) {
        let Some(block_range) = self.code_lang_block_range else {
            self.close_code_lang_picker();
            return;
        };
        let selected_lang = if let Some(entry) = self.code_lang_matches.get(idx) {
            entry.token.clone()
        } else {
            let custom = self.code_lang_query.trim();
            if custom.is_empty() {
                self.close_code_lang_picker();
                return;
            }
            custom.to_string()
        };
        self.close_code_lang_picker();
        self.find_focused = false;
        self.close_confirm_armed = false;
        let active_id = self.workbench.active_tab().map(|t| t.id);
        self.cancel_abandoned_tab_ime(active_id);
        if let Some(tab) = self.workbench.active_tab_mut() {
            match tab.dispatch(EditorAction::SetCodeFenceLanguage {
                block_range,
                language: selected_lang.clone(),
            }) {
                Ok(_) => {
                    self.pending_scroll_ensure_caret = true;
                    let label = code_language_display_label(&selected_lang);
                    self.set_notice(format!("Code block language: {label}"));
                }
                Err(e) => self.set_notice(format!("Action error: {e}")),
            }
        }
    }

    /// Handle an OS dropped directory, image asset (`PRD I-02`), or Markdown/text document.
    pub fn handle_dropped_path(&mut self, path: &Path) {
        self.close_confirm_armed = false;
        if path.is_dir() {
            self.close_transient_prompts();
            if let Err(e) = self.workbench.open_folder(path) {
                self.set_notice(format!("Drop open error: {e}"));
            }
            return;
        }
        if is_supported_image_extension(path) {
            self.close_transient_prompts();
            self.find_focused = false;
            let active_id = self.workbench.active_tab().map(|t| t.id);
            self.cancel_abandoned_tab_ime(active_id);
            let alt = path.file_stem().and_then(|s| s.to_str());
            match self.workbench.paste_image_into_active(path, alt) {
                Ok(dest) => {
                    self.pending_scroll_ensure_caret = true;
                    let name = dest.file_name().and_then(|n| n.to_str()).unwrap_or("image");
                    self.set_notice(format!("Inserted image -> {name}"));
                }
                Err(e) => self.set_notice(format!("Image drop error: {e}")),
            }
            return;
        }
        self.close_transient_prompts();
        self.flush_active_tab_before_switch();
        match self.workbench.open_document(path) {
            Ok(_) => {
                self.find_focused = false;
                self.ime_target = ImeTarget::None;
                self.preferred_caret_x = None;
                self.pending_scroll_ensure_caret = true;
            }
            Err(e) => self.set_notice(format!("Drop open error: {e}")),
        }
    }

    /// Show hyperlink prompt (`Cmd/Ctrl + K`), pre-filling clipboard URL when applicable.
    pub fn open_link_prompt(&mut self) {
        if self.workbench.active_tab().is_none() {
            self.set_notice("No active document");
            return;
        }
        self.close_confirm_armed = false;
        self.close_transient_prompts();
        self.find_focused = false;
        let active_id = self.workbench.active_tab().map(|t| t.id);
        self.cancel_abandoned_tab_ime(active_id);

        let prefill = self
            .clipboard
            .read()
            .text
            .map(|t| t.trim().to_string())
            .filter(|t| {
                !t.is_empty()
                    && !t.contains(char::is_whitespace)
                    && (has_ascii_prefix_ignore_case(t, "http://")
                        || has_ascii_prefix_ignore_case(t, "https://")
                        || has_ascii_prefix_ignore_case(t, "mailto:"))
            })
            .unwrap_or_default();

        self.link_prompt_url = prefill;
        self.link_prompt_open = true;
    }

    fn close_link_prompt(&mut self) {
        self.link_prompt_open = false;
        if self.ime_target == ImeTarget::LinkPromptBox {
            self.ime_target = ImeTarget::None;
        }
    }

    fn confirm_link_prompt(&mut self) {
        let url = self.link_prompt_url.trim().to_string();
        self.close_link_prompt();
        if url.is_empty() {
            self.set_notice("Empty URL — link not inserted");
            return;
        }
        self.dispatch_editor(EditorAction::InsertLink(url));
    }

    fn has_link_at_offset(&self, offset: usize) -> bool {
        self.workbench
            .active_tab()
            .and_then(|tab| link_destination_at_offset(tab, offset))
            .is_some()
    }

    fn try_open_link_at_offset(&mut self, offset: usize) -> bool {
        let dest = self
            .workbench
            .active_tab()
            .and_then(|tab| link_destination_at_offset(tab, offset));
        let Some(dest) = dest else {
            return false;
        };
        self.open_resolved_link_destination(&dest);
        true
    }

    fn jump_active_tab_to_anchor(&mut self, anchor: &str) {
        let Some(item) = self
            .workbench
            .active_tab()
            .and_then(|tab| find_matching_outline_item(&tab.outline(), anchor))
        else {
            self.set_notice(format!("Heading anchor not found: #{anchor}"));
            return;
        };
        if let Some(tab) = self.workbench.active_tab_mut() {
            tab.jump_to_outline_item(&item);
            self.preferred_caret_x = None;
            self.pending_scroll_ensure_caret = true;
            self.set_notice(format!("Jumped to: {}", item.title));
        }
    }

    fn open_resolved_link_destination(&mut self, raw_dest: &str) {
        let trimmed = raw_dest.trim();
        if trimmed.is_empty() || trimmed.chars().any(char::is_control) {
            self.set_notice(if trimmed.is_empty() {
                "Empty link target"
            } else {
                "Invalid link target"
            });
            return;
        }
        if let Some(anchor) = trimmed.strip_prefix('#') {
            self.jump_active_tab_to_anchor(anchor);
            return;
        }

        let local_raw: String = if let Some(scheme) = extract_uri_scheme(trimmed) {
            if is_supported_external_scheme(scheme) {
                if !is_allowed_external_url(trimmed) {
                    self.set_notice("Blocked unsafe or malformed external URL");
                    return;
                }
                match self.link_opener.open_url(trimmed) {
                    Ok(()) => self.set_notice(format!("Opened link: {trimmed}")),
                    Err(e) => self.set_notice(e),
                }
                return;
            }
            if scheme.eq_ignore_ascii_case("file")
                && has_ascii_prefix_ignore_case(trimmed, "file://")
            {
                let Some(local_path) = parse_local_file_uri_path(&trimmed[7..]) else {
                    self.set_notice("Blocked non-local file URI");
                    return;
                };
                local_path
            } else {
                self.set_notice(format!("Unsupported link scheme: {scheme}"));
                return;
            }
        } else {
            trimmed.to_string()
        };

        let (file_part, frag_part) = match local_raw.split_once('#') {
            Some((p, f)) => (p, Some(f)),
            None => (local_raw.as_str(), None),
        };
        let resolved = self.resolve_local_link_path(file_part);
        let Some(target) = resolved else {
            self.set_notice(format!("Link target not found: {file_part}"));
            return;
        };

        let is_doc = is_supported_document_path(&target);
        if is_doc {
            self.flush_active_tab_before_switch();
            match self.workbench.open_document(&target) {
                Ok(_) => {
                    self.ime_target = ImeTarget::None;
                    self.find_focused = false;
                    self.preferred_caret_x = None;
                    self.pending_scroll_ensure_caret = true;
                    if let Some(frag) = frag_part.filter(|s| !s.is_empty()) {
                        self.jump_active_tab_to_anchor(frag);
                    } else {
                        let name = target
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("document");
                        self.set_notice(format!("Opened {name}"));
                    }
                }
                Err(e) => self.set_notice(format!("Open failed: {e}")),
            }
        } else if is_blocked_executable_path(&target) {
            let name = target
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("executable");
            self.set_notice(format!("Blocked opening executable file: {name}"));
        } else {
            let target_str = target.to_string_lossy().into_owned();
            match self.link_opener.open_url(&target_str) {
                Ok(()) => self.set_notice(format!("Opened {target_str}")),
                Err(e) => self.set_notice(e),
            }
        }
    }

    fn resolve_local_link_path(&self, raw_path: &str) -> Option<PathBuf> {
        let mut candidates = Vec::new();
        let decoded = decode_percent_spaces(raw_path);
        for candidate_str in [raw_path, decoded.as_str()] {
            if candidate_str.is_empty() {
                continue;
            }
            let p = Path::new(candidate_str);
            if p.is_absolute() {
                candidates.push(normalize_lexical_path(p));
                continue;
            }
            if let Some(doc_dir) = self
                .workbench
                .active_tab()
                .and_then(|t| t.session.path.as_deref())
                .and_then(|p| p.parent())
            {
                candidates.push(normalize_lexical_path(&doc_dir.join(p)));
            }
            if let Some(root) = &self.workbench.workspace_root {
                candidates.push(normalize_lexical_path(&root.join(p)));
            }
        }
        candidates.into_iter().find(|p| p.is_file())
    }

    fn open_find_bar(&mut self, show_replace: bool) {
        self.close_confirm_armed = false;
        self.close_transient_prompts();
        self.find_open = true;
        self.replace_open = show_replace;
        if show_replace && !self.find_query.is_empty() {
            self.find_focused = false;
            self.replace_focused = true;
        } else {
            self.find_focused = true;
            self.replace_focused = false;
        }
        if matches!(self.ime_target, ImeTarget::FindBox | ImeTarget::ReplaceBox) {
            self.ime_target = ImeTarget::None;
        }
    }

    fn close_find_bar(&mut self) {
        self.find_open = false;
        self.find_focused = false;
        self.replace_focused = false;
        self.active_find_range = None;
        if matches!(self.ime_target, ImeTarget::FindBox | ImeTarget::ReplaceBox) {
            self.ime_target = ImeTarget::None;
        }
    }

    fn replace_in_active_tab(&mut self, all: bool) {
        if self.find_query.is_empty() {
            self.set_notice("Enter text to find");
            return;
        }
        let query = self.find_query.clone();
        let replacement = self.replace_text.clone();
        let opts = self.find_options;
        let outcome = {
            let Some(tab) = self.workbench.active_tab_mut() else {
                return;
            };
            if all {
                tab.replace_all(&query, &replacement, opts)
            } else {
                tab.replace_current_or_next(&query, &replacement, true, opts)
            }
        };
        match outcome {
            Ok(Ok(res)) if res.replaced_count == 0 => {
                self.active_find_range = None;
                self.set_notice(if all {
                    "No matches replaced"
                } else {
                    "No matches found"
                });
            }
            Ok(Ok(res)) => {
                self.pending_scroll_ensure_caret = true;
                if all {
                    self.active_find_range = None;
                    self.set_notice(format!("Replaced {} match(es)", res.replaced_count));
                } else {
                    self.run_find_next(false);
                    self.set_notice("Replaced 1 match");
                }
            }
            Ok(Err(err)) => {
                self.active_find_range = None;
                self.set_notice(format!("Replace error: {err}"));
            }
            Err(err) => {
                self.set_notice(format!("Read-only or blocked: {err}"));
            }
        }
    }

    fn run_replace_next(&mut self) {
        self.replace_in_active_tab(false);
    }

    fn run_replace_all(&mut self) {
        self.replace_in_active_tab(true);
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
        if let Some(tab) = self.workbench.active_tab_mut()
            && let Ok(Some(m)) = tab.find_prev(&query, true, opts)
        {
            self.active_find_range = Some(m.range);
            self.pending_scroll_ensure_caret = true;
        }
    }
}

/// Active native `winit` + `wgpu` application handler.
fn encoding_label(enc: DetectedEncoding) -> &'static str {
    match enc {
        DetectedEncoding::Utf8 => "UTF-8",
        DetectedEncoding::Utf8Bom => "UTF-8 BOM",
        DetectedEncoding::Utf16LeBom => "UTF-16 LE BOM",
        DetectedEncoding::Utf16BeBom => "UTF-16 BE BOM",
        DetectedEncoding::Utf16LeNoBom => "UTF-16 LE",
        DetectedEncoding::Utf16BeNoBom => "UTF-16 BE",
        DetectedEncoding::NonUtf8Ambiguous => "Non-UTF-8",
    }
}

fn format_reading_time(secs: usize) -> String {
    if secs == 0 {
        "0m read".to_string()
    } else if secs < 60 {
        "< 1 min read".to_string()
    } else {
        format!("{} min read", secs.div_ceil(60))
    }
}

fn parse_table_dimensions(input: &str) -> Option<(usize, usize)> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return None;
    }
    let parts: Vec<&str> = trimmed
        .split(['x', 'X', '×', ',', ' ', '*'])
        .filter(|s| !s.is_empty())
        .collect();
    if parts.len() != 2 {
        return None;
    }
    let rows = parts[0].parse::<usize>().ok()?.clamp(1, 50);
    let cols = parts[1].parse::<usize>().ok()?.clamp(1, 20);
    Some((rows, cols))
}

pub struct VellmarkApp {
    controller: GuiController,
    window: Option<Arc<Window>>,
    gpu: Option<WgpuRenderer>,
    modifiers: ModifiersState,
    last_fullscreen_config: bool,
}

impl VellmarkApp {
    pub fn new(controller: GuiController) -> Self {
        let last_fullscreen_config = controller.workbench.config.fullscreen;
        Self {
            controller,
            window: None,
            gpu: None,
            modifiers: ModifiersState::default(),
            last_fullscreen_config,
        }
    }

    fn sync_window_fullscreen(&mut self) {
        let desired = self.controller.workbench.config.fullscreen;
        if desired != self.last_fullscreen_config {
            self.last_fullscreen_config = desired;
            if let Some(win) = &self.window {
                win.set_fullscreen(desired.then_some(Fullscreen::Borderless(None)));
            }
        }
    }

    pub fn run(mut controller: GuiController) -> Result<(), String> {
        let event_loop = EventLoop::<()>::with_user_event()
            .build()
            .map_err(|e| format!("failed to create event loop: {e}"))?;
        let proxy: EventLoopProxy<()> = event_loop.create_proxy();
        controller.set_image_redraw_notifier(Some(Arc::new(move || {
            let _ = proxy.send_event(());
        })));
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
        let base_dir = self.controller.active_image_base_dir();
        let clear = self.controller.style.bg_color;
        let shaper = &mut self.controller.shaper;
        let image_cache = &mut self.controller.image_cache;
        let _ = gpu.render_layers(
            clear,
            scale,
            shaper.font_system_mut(),
            &[&editor_dl, &chrome_dl],
            |src| image_cache.get_decoded(base_dir.as_deref(), src),
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
                        if self.last_fullscreen_config {
                            arc.set_fullscreen(Some(Fullscreen::Borderless(None)));
                        }
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
                let (px, py) = self.controller.pointer_pos;
                let (icon, _) =
                    self.controller
                        .handle_pointer_move_with_mods(px, py, self.modifiers);
                if let Some(win) = &self.window {
                    win.set_cursor(icon);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if self.controller.handle_key_press(&event, self.modifiers) {
                    self.sync_window_fullscreen();
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
            } => match state {
                ElementState::Pressed => {
                    let (px, py) = self.controller.pointer_pos;
                    if self
                        .controller
                        .handle_pointer_down_with_mods(px, py, self.modifiers)
                    {
                        self.sync_window_fullscreen();
                        self.request_redraw();
                    }
                }
                ElementState::Released => {
                    self.controller.handle_pointer_up();
                }
            },
            WindowEvent::CursorMoved {
                position: PhysicalPosition { x, y },
                ..
            } => {
                let scale = self.controller.scale_factor.max(0.5);
                let lx = (x as f32) / scale;
                let ly = (y as f32) / scale;
                let (icon, changed) =
                    self.controller
                        .handle_pointer_move_with_mods(lx, ly, self.modifiers);
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
                self.controller.handle_dropped_path(&path);
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

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, _event: ()) {
        let _ = self.controller.image_cache.drain_completed();
        let _ = self.controller.poll_global_search(Instant::now());
        self.request_redraw();
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.controller.image_cache.drain_completed() {
            self.request_redraw();
        }
        if self.controller.poll_background_maintenance() {
            self.request_redraw();
        }
        self.controller
            .workbench
            .maybe_maintain_recovery_storage_async();
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

struct ListPickerRow {
    label: String,
    badge: Option<String>,
    action: UiAction,
}

enum ListModalAction {
    None,
    Close,
    MoveSelection(isize),
    Confirm,
    QueryBackspace,
    QueryAppend(String),
}

fn decode_list_modal_key(
    logical_key: &Key,
    text: Option<&str>,
    primary: bool,
    ctrl: bool,
) -> ListModalAction {
    match logical_key {
        Key::Named(NamedKey::Escape) => ListModalAction::Close,
        Key::Named(NamedKey::ArrowUp) => ListModalAction::MoveSelection(-1),
        Key::Named(NamedKey::ArrowDown) => ListModalAction::MoveSelection(1),
        Key::Named(NamedKey::Enter) => ListModalAction::Confirm,
        Key::Named(NamedKey::Backspace) => ListModalAction::QueryBackspace,
        _ => (!primary && !ctrl)
            .then(|| overlay_printable_key_text(logical_key, text))
            .flatten()
            .map_or(ListModalAction::None, ListModalAction::QueryAppend),
    }
}

fn move_list_selection(selected: &mut usize, len: usize, delta: isize) {
    if len == 0 {
        *selected = 0;
        return;
    }
    let next = (*selected as isize + delta).clamp(0, len as isize - 1);
    *selected = next as usize;
}

fn intersect_rect(a: Rect, b: Rect) -> Option<Rect> {
    let x0 = a.x.max(b.x);
    let y0 = a.y.max(b.y);
    let x1 = (a.x + a.w).min(b.x + b.w);
    let y1 = (a.y + a.h).min(b.y + b.h);
    (x1 > x0 && y1 > y0).then(|| Rect::new(x0, y0, x1 - x0, y1 - y0))
}

fn reserve_unique_untitled_path(dir: &Path, tab_id: u64) -> std::io::Result<PathBuf> {
    fs::create_dir_all(dir)?;
    let start_seq = tab_id.max(1);
    for seq in start_seq..start_seq + 10_000 {
        let candidate = dir.join(format!("vellmark-note-{seq}.md"));
        if fs::symlink_metadata(&candidate).is_ok() {
            continue;
        }
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
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

fn is_primary_mod(mods: ModifiersState) -> bool {
    if cfg!(target_os = "macos") {
        mods.super_key()
    } else {
        mods.control_key()
    }
}

fn overlay_printable_key_text(logical_key: &Key, text: Option<&str>) -> Option<String> {
    if let Some(txt) = text {
        let clean: String = txt.chars().filter(|c| !c.is_control()).collect();
        if !clean.is_empty() {
            return Some(clean);
        }
    }
    if *logical_key == NamedKey::Space {
        return Some(" ".to_string());
    }
    if let Key::Character(ch) = logical_key {
        let clean: String = ch.chars().filter(|c| !c.is_control()).collect();
        if !clean.is_empty() {
            return Some(clean);
        }
    }
    None
}

fn has_ascii_prefix_ignore_case(value: &str, prefix: &str) -> bool {
    value
        .get(..prefix.len())
        .is_some_and(|head| head.eq_ignore_ascii_case(prefix))
}

fn is_supported_document_path(path: &Path) -> bool {
    match path.extension().and_then(|e| e.to_str()) {
        None => true,
        Some(ext) => matches!(
            ext.to_ascii_lowercase().as_str(),
            "md" | "markdown" | "mdown" | "mkd" | "txt" | "text"
        ),
    }
}

fn hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn decode_percent_spaces(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let decoded = (bytes[i] == b'%' && i + 2 < bytes.len())
            .then(|| hex_nibble(bytes[i + 1]).zip(hex_nibble(bytes[i + 2])))
            .flatten();
        if let Some((hi, lo)) = decoded {
            out.push((hi << 4) | lo);
            i += 3;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(out).unwrap_or_else(|_| input.to_string())
}

fn normalize_lexical_path(path: &Path) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                if !out.pop() {
                    out.push(comp.as_os_str());
                }
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

fn link_destination_at_offset(tab: &DocumentTab, offset: usize) -> Option<String> {
    let source = tab.session.buffer.to_string();
    tab.with_incremental_blocks(|blocks| {
        let block = blocks.iter().find(|b| b.range.contains_inclusive(offset))?;
        let span = block.inlines.iter().find(|s| {
            s.kind == InlineKind::Link && offset >= s.full_range.start && offset < s.full_range.end
        })?;
        let close = source.get(span.close_marker.start..span.close_marker.end)?;
        let inner = close.strip_prefix("](")?.strip_suffix(')')?;
        let dest = extract_markdown_link_destination(inner);
        (!dest.is_empty()).then_some(dest)
    })
}

fn extract_markdown_link_destination(raw_inner: &str) -> String {
    extract_link_destination(raw_inner).unwrap_or_default()
}

fn heading_to_github_slug(title: &str) -> String {
    let mut slug = String::new();
    for ch in title.trim().chars() {
        if ch == ' ' || ch == '-' {
            slug.push('-');
        } else if ch == '_' {
            slug.push('_');
        } else if ch.is_alphanumeric() {
            for lc in ch.to_lowercase() {
                slug.push(lc);
            }
        }
    }
    slug
}

fn find_matching_outline_item(outline: &[OutlineItem], anchor: &str) -> Option<OutlineItem> {
    let decoded = decode_percent_spaces(anchor);
    let target = decoded.trim().trim_start_matches('#').to_lowercase();
    if target.is_empty() {
        return None;
    }

    let mut seen: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for item in outline {
        let base = heading_to_github_slug(&item.title);
        let count = seen.entry(base.clone()).or_insert(0);
        let unique = if *count == 0 {
            base
        } else {
            format!("{base}-{}", *count)
        };
        *count += 1;
        if unique == target {
            return Some(item.clone());
        }
    }

    let slugified_target = heading_to_github_slug(&decoded);
    if let Some(item) = (!slugified_target.is_empty())
        .then(|| {
            outline
                .iter()
                .find(|item| heading_to_github_slug(&item.title) == slugified_target)
        })
        .flatten()
    {
        return Some(item.clone());
    }

    outline
        .iter()
        .find(|item| item.title.trim().eq_ignore_ascii_case(decoded.trim()))
        .cloned()
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
        let caret = gui
            .workbench
            .active_tab()
            .unwrap()
            .session
            .buffer
            .len_bytes();
        gui.workbench
            .active_tab_mut()
            .unwrap()
            .edit(|ctx| ctx.set_caret(caret))
            .unwrap();
        let rev_before = gui
            .workbench
            .active_tab()
            .unwrap()
            .session
            .buffer
            .revision();

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
        let doc = gui
            .workbench
            .active_tab()
            .unwrap()
            .session
            .buffer
            .to_string();
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
        assert_eq!(
            gui.workbench.tabs[0].view.ime_preedit.as_deref(),
            Some("zhongwen")
        );

        // Switch to TabB before IME commits
        assert!(gui.execute_ui_action(UiAction::SelectTab(1)));
        assert_eq!(
            gui.workbench.active_tab().unwrap().id,
            gui.workbench.tabs[1].id
        );
        gui.handle_ime_event(&Ime::Preedit(String::new(), None));
        gui.handle_ime_event(&Ime::Commit("中文".to_string()));

        assert_eq!(
            gui.workbench.tabs[0].session.buffer.to_string(),
            "TabA 中文"
        );
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
            gui.workbench
                .active_tab()
                .unwrap()
                .session
                .buffer
                .to_string(),
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
            image_png: None,
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
        assert_eq!(
            gui.workbench
                .active_tab()
                .unwrap()
                .session
                .buffer
                .to_string(),
            "hello"
        );
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
        assert!(
            shared_clip
                .get_content()
                .html
                .as_deref()
                .is_some_and(|h| h.contains("<strong>hello</strong>"))
        );

        // External app puts rich HTML into clipboard -> Paste converts HTML to Markdown!
        shared_clip.set_content(ClipboardContent {
            text: Some("heading".to_string()),
            html: Some("<h2>External Heading</h2>".to_string()),
            image_png: None,
        });
        let end = gui
            .workbench
            .active_tab()
            .unwrap()
            .session
            .buffer
            .len_bytes();
        gui.workbench
            .active_tab_mut()
            .unwrap()
            .edit(|ctx| ctx.set_caret(end))
            .unwrap();
        gui.paste_clipboard();
        assert!(
            gui.workbench
                .active_tab()
                .unwrap()
                .session
                .buffer
                .to_string()
                .contains("## External Heading")
        );
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
        assert_eq!(
            fs::read_to_string(&existing).unwrap(),
            "previous session note"
        );
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
        assert_eq!(
            gui.workbench.tabs[0].view.ime_preedit.as_deref(),
            Some("nihao")
        );
        assert!(gui.workbench.tabs[1].view.ime_preedit.is_none());

        gui.handle_ime_event(&Ime::Commit("你好".to_string()));
        assert_eq!(
            gui.workbench.tabs[0].session.buffer.to_string(),
            "TabA 你好"
        );
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
            image_png: None,
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
            gui.workbench
                .active_tab()
                .unwrap()
                .session
                .buffer
                .to_string(),
            "hello world"
        );
        assert_eq!(
            gui.workbench.active_tab().unwrap().view.selection,
            Selection::new(0, 5)
        );
        assert_eq!(gui.clipboard_text, "prior");
        assert_eq!(shared_clip.get_content().text.as_deref(), Some("prior"));
        assert!(
            gui.status_notice
                .as_ref()
                .is_some_and(|(n, _)| n.contains("Clipboard error"))
        );

        gui.find_open = true;
        gui.find_focused = true;
        gui.find_query = "world".to_string();
        gui.copy_selection(true);
        assert_eq!(gui.find_query, "world");
        assert_eq!(gui.clipboard_text, "prior");
        assert!(
            gui.status_notice
                .as_ref()
                .is_some_and(|(n, _)| n.contains("Clipboard error"))
        );
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
        assert!(
            !gui.active_status_notice()
                .as_deref()
                .is_some_and(|n| n.contains(PROTECTED_SNAPSHOT_LIMIT_NOTICE))
        );

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
        assert!(click_hit_action(&mut gui, |a| matches!(
            a,
            UiAction::ToggleConflictDiff
        )));
        assert!(!gui.conflict_diff_open);
        assert!(click_hit_action(&mut gui, |a| matches!(
            a,
            UiAction::ToggleConflictDiff
        )));
        assert!(gui.conflict_diff_open);

        // Click Auto Merge -> resolves disjoint changes, exits Conflict, leaves merged content
        assert!(click_hit_action(&mut gui, |a| matches!(
            a,
            UiAction::ResolveConflictAutoMerge
        )));
        assert!(!gui.conflict_diff_open);
        assert_eq!(
            gui.workbench.active_tab().unwrap().session.state,
            DocState::Dirty
        );
        assert_eq!(
            gui.workbench
                .active_tab()
                .unwrap()
                .session
                .buffer
                .to_string(),
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
        assert!(click_hit_action(&mut gui, |a| matches!(
            a,
            UiAction::ResolveConflictAutoMerge
        )));
        assert!(gui.conflict_diff_open);
        assert_eq!(
            gui.workbench.active_tab().unwrap().session.state,
            DocState::Conflict
        );
        assert!(click_hit_action(&mut gui, |a| matches!(
            a,
            UiAction::ResolveConflictUseTheirs
        )));
        assert_eq!(
            gui.workbench.active_tab().unwrap().session.state,
            DocState::Clean
        );
        assert_eq!(
            gui.workbench
                .active_tab()
                .unwrap()
                .session
                .buffer
                .to_string(),
            "mine1\ndisk2\ndisk3\n"
        );

        // New conflict -> Keep Mine creates backup & surfaces Clean Backups banner button
        {
            let tab = gui.workbench.active_tab_mut().unwrap();
            tab.edit(|ctx| ctx.insert_text("top\n")).unwrap();
        }
        fs::write(&doc, "external-v2\n").unwrap();
        assert!(gui.handle_focus_gained());
        assert!(click_hit_action(&mut gui, |a| matches!(
            a,
            UiAction::ResolveConflictKeepMine
        )));
        assert_eq!(
            gui.workbench.active_tab().unwrap().session.state,
            DocState::Clean
        );
        // Auto Merge, Use Disk, and Keep Mine each left a safety backup for `doc`
        assert!(
            gui.workbench.active_tab().unwrap().conflict_backup_count >= 1,
            "expected at least 1 conflict backup after resolutions"
        );
        assert!(click_hit_action(&mut gui, |a| matches!(
            a,
            UiAction::ClearConflictBackups
        )));
        assert_eq!(gui.workbench.active_tab().unwrap().conflict_backup_count, 0);

        // Create another conflict so Save As... hit zone is rendered on the banner, then resolve via MemorySaveDialog
        {
            let tab = gui.workbench.active_tab_mut().unwrap();
            tab.edit(|ctx| ctx.insert_text("local-copy-")).unwrap();
        }
        fs::write(&doc, "external-v3\n").unwrap();
        assert!(gui.handle_focus_gained());
        gui.set_save_dialog(Box::new(MemorySaveDialog::new(Some(save_as_doc.clone()))));
        assert!(click_hit_action(&mut gui, |a| matches!(
            a,
            UiAction::SaveAsActive
        )));
        assert_eq!(
            gui.workbench.active_tab().unwrap().session.path.as_deref(),
            Some(save_as_doc.as_path())
        );
        assert_eq!(
            gui.workbench.active_tab().unwrap().session.state,
            DocState::Clean
        );
        assert!(
            fs::read_to_string(&save_as_doc)
                .unwrap()
                .contains("local-copy-")
        );

        // Tab switch immediately flushes named Dirty document
        gui.workbench.new_document();
        let dirty_idx = 0usize;
        assert!(click_hit_action(
            &mut gui,
            |a| matches!(a, UiAction::SelectTab(i) if *i == dirty_idx)
        ));
        {
            let tab = gui.workbench.active_tab_mut().unwrap();
            tab.edit(|ctx| {
                ctx.set_selection(0, 0);
                ctx.insert_text("tab-switch-");
            })
            .unwrap();
        }
        assert!(click_hit_action(&mut gui, |a| matches!(
            a,
            UiAction::SelectTab(1)
        )));
        assert!(
            fs::read_to_string(&save_as_doc)
                .unwrap()
                .starts_with("tab-switch-")
        );

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
        assert!(click_hit_action(&mut gui, |a| matches!(
            a,
            UiAction::RetrySaveActive
        )));
        assert_eq!(
            gui.workbench.active_tab().unwrap().session.state,
            DocState::Clean
        );
        assert!(
            gui.workbench
                .active_tab()
                .unwrap()
                .last_write_error
                .is_none()
        );
        assert_eq!(fs::read_to_string(&doc).unwrap(), "focus-retry-start\n");

        // Deleting file triggers Missing banner; Re-save recreates original file; Close Tab removes it
        fs::remove_file(&doc).unwrap();
        assert!(gui.handle_focus_gained());
        assert!(matches!(
            gui.active_doc_banner_kind(),
            Some(DocBannerKind::Missing { .. })
        ));
        assert!(click_hit_action(&mut gui, |a| matches!(
            a,
            UiAction::ResaveMissingFile
        )));
        assert_eq!(
            gui.workbench.active_tab().unwrap().session.state,
            DocState::Clean
        );
        assert!(doc.exists());

        fs::remove_file(&doc).unwrap();
        assert!(gui.handle_focus_gained());
        let deleted_id = gui.workbench.active_tab().unwrap().id;
        assert!(click_hit_action(&mut gui, |a| matches!(
            a,
            UiAction::ForceCloseActiveMissing
        )));
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
        assert!(
            gui.status_notice
                .as_ref()
                .is_some_and(|(msg, _)| msg.contains("already open"))
        );

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
        assert_eq!(
            existing_default_dialog_dir(Some(&deleted_child)),
            Some(dir.as_path())
        );
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
            gui.workbench
                .active_tab()
                .unwrap()
                .session
                .buffer
                .to_string(),
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
        let after_del = gui
            .workbench
            .active_tab()
            .unwrap()
            .session
            .buffer
            .to_string();
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
            gui.workbench
                .active_tab()
                .unwrap()
                .session
                .buffer
                .to_string(),
            "AB"
        );
        gui.dispatch_editor(EditorAction::Enter);
        assert_eq!(
            gui.workbench
                .active_tab()
                .unwrap()
                .session
                .buffer
                .to_string(),
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

        // Horizontal caret movement in WYSIWYG skips collapsed table delimiter row cleanly (CRLF & LF)
        let tbl_doc = dir.join("tbl_nav.md");
        fs::write(&tbl_doc, b"| H1 | H2 |\r\n| --- | --- |\r\n| B1 | B2 |").unwrap();
        gui.workbench.open_document(&tbl_doc).unwrap();
        let hdr_end = "| H1 | H2 |".len();
        let b1_start = "| H1 | H2 |\r\n| --- | --- |\r\n| ".len();
        let body_row_start = "| H1 | H2 |\r\n| --- | --- |\r\n".len();
        let h2_content_end = "| H1 | H2".len();
        gui.workbench
            .active_tab_mut()
            .unwrap()
            .edit(|ctx| ctx.set_caret(hdr_end))
            .unwrap();
        gui.move_caret_horizontal(1, false);
        assert_eq!(
            gui.workbench.active_tab().unwrap().view.selection.head,
            b1_start
        );
        gui.workbench
            .active_tab_mut()
            .unwrap()
            .edit(|ctx| ctx.set_caret(body_row_start))
            .unwrap();
        gui.move_caret_horizontal(-1, false);
        assert_eq!(
            gui.workbench.active_tab().unwrap().view.selection.head,
            h2_content_end
        );

        let _ = fs::remove_dir_all(&dir);
    }

    fn primary_test_mod() -> ModifiersState {
        if cfg!(target_os = "macos") {
            ModifiersState::SUPER
        } else {
            ModifiersState::CONTROL
        }
    }

    fn unique_test_dir(prefix: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "{prefix}-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn test_cmd_o_and_cmd_shift_o_open_file_and_folder_dialogs() {
        let dir = unique_test_dir("vellmark-gui-open-dialog");
        let rec_root = dir.join("rec");
        let docs_dir = dir.join("workspace");
        fs::create_dir_all(&docs_dir).unwrap();
        let file_a = docs_dir.join("notes.md");
        fs::write(&file_a, "# Notes\n\nHello open dialog.").unwrap();

        let dialog = Arc::new(MemorySaveDialog::new(None));
        struct SharedDialog(Arc<MemorySaveDialog>);
        impl SaveDialogBackend for SharedDialog {
            fn pick_save_as_path(&self, cur: Option<&Path>, def: &str) -> Option<PathBuf> {
                self.0.pick_save_as_path(cur, def)
            }
            fn pick_open_path(&self, cur_dir: Option<&Path>) -> Option<PathBuf> {
                self.0.pick_open_path(cur_dir)
            }
            fn pick_open_folder(&self, cur_dir: Option<&Path>) -> Option<PathBuf> {
                self.0.pick_open_folder(cur_dir)
            }
        }

        let wb = AppWorkbench::new(AppConfig::default(), Some(rec_root));
        let mut gui = GuiController::with_backends(
            wb,
            Box::new(MemoryClipboard::default()),
            Box::new(SharedDialog(dialog.clone())),
        );
        let p_mod = primary_test_mod();

        // Cancelled Open File leaves tabs untouched
        assert!(gui.handle_key_input(&Key::Character("o".into()), None, p_mod));
        assert!(gui.workbench.tabs.is_empty());

        // Configured Open File loads document into active tab
        dialog.set_next_open_path(Some(file_a.clone()));
        assert!(gui.handle_key_input(&Key::Character("o".into()), None, p_mod));
        assert_eq!(gui.workbench.tabs.len(), 1);
        assert_eq!(
            gui.workbench
                .active_tab()
                .unwrap()
                .session
                .buffer
                .to_string(),
            "# Notes\n\nHello open dialog."
        );

        // Configured Cmd/Ctrl+Shift+O opens workspace folder & reveals Files sidebar
        gui.workbench.config.sidebar_visible = false;
        gui.sidebar_section = SidebarSection::Outline;
        dialog.set_next_open_folder(Some(docs_dir.clone()));
        assert!(gui.handle_key_input(
            &Key::Character("o".into()),
            None,
            p_mod | ModifiersState::SHIFT
        ));
        assert_eq!(
            gui.workbench.workspace_root.as_deref(),
            Some(docs_dir.as_path())
        );
        assert!(gui.workbench.file_tree.is_some());
        assert!(gui.workbench.config.sidebar_visible);
        assert_eq!(gui.sidebar_section, SidebarSection::Files);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_cmd_p_quick_open_fuzzy_search_keyboard_ime_and_click() {
        let dir = unique_test_dir("vellmark-gui-quick-open");
        let rec_root = dir.join("rec");
        let ws = dir.join("ws");
        fs::create_dir_all(ws.join("docs")).unwrap();
        fs::create_dir_all(ws.join("notes")).unwrap();
        fs::create_dir_all(ws.join("assets")).unwrap();
        fs::write(ws.join("README.md"), "# Root").unwrap();
        fs::write(ws.join("docs/architecture.md"), "# Arch").unwrap();
        fs::write(ws.join("notes/todo.txt"), "Todo list").unwrap();
        fs::write(ws.join("assets/logo.png"), b"PNG").unwrap();

        let mut wb = AppWorkbench::new(AppConfig::default(), Some(rec_root));
        wb.new_document();
        wb.active_tab_mut()
            .unwrap()
            .session
            .buffer
            .reset_content("Keep me intact");

        let mut gui = GuiController::with_clipboard(wb, Box::new(MemoryClipboard::default()));
        let p_mod = primary_test_mod();

        // Empty-workspace Quick Open renders clean prompt without panicking
        assert!(gui.handle_key_input(&Key::Character("p".into()), None, p_mod));
        assert!(gui.quick_open_open);
        assert!(gui.quick_open_matches.is_empty());
        let _ = gui.build_frame();
        assert!(gui.handle_key_input(&Key::Named(NamedKey::Escape), None, ModifiersState::empty()));
        assert!(!gui.quick_open_open);

        // Open workspace and launch Cmd/Ctrl + P
        gui.workbench.open_folder(&ws).unwrap();
        assert!(gui.handle_key_input(&Key::Character("p".into()), None, p_mod));
        assert!(gui.quick_open_open);
        assert_eq!(gui.quick_open_matches.len(), 3);

        // Filter via keyboard + IME without touching active tab buffer
        assert!(gui.handle_key_input(
            &Key::Character("a".into()),
            Some("a"),
            ModifiersState::empty()
        ));
        assert!(gui.handle_key_input(
            &Key::Character("r".into()),
            Some("r"),
            ModifiersState::empty()
        ));
        gui.handle_ime_event(&Ime::Preedit("ch".to_string(), None));
        assert!(
            gui.workbench
                .active_tab()
                .unwrap()
                .view
                .ime_preedit
                .is_none()
        );
        gui.handle_ime_event(&Ime::Preedit(String::new(), None));
        gui.handle_ime_event(&Ime::Commit("ch".to_string()));
        assert_eq!(gui.quick_open_query, "arch");
        assert!(
            gui.quick_open_matches
                .iter()
                .any(|m| m.relative_path.ends_with("architecture.md"))
        );
        assert_eq!(
            gui.workbench.tabs[0].session.buffer.to_string(),
            "Keep me intact"
        );

        // Confirm via Enter opens matching document
        assert!(gui.handle_key_input(&Key::Named(NamedKey::Enter), None, ModifiersState::empty()));
        assert!(!gui.quick_open_open);
        assert_eq!(
            gui.workbench
                .active_tab()
                .unwrap()
                .session
                .buffer
                .to_string(),
            "# Arch"
        );

        // Re-open Quick Open, clear query, ArrowDown + Enter / UI hit zone click opens another file
        assert!(gui.handle_key_input(&Key::Character("p".into()), None, p_mod));
        assert_eq!(gui.quick_open_matches.len(), 3);
        let _ = gui.build_frame();
        let target_idx = gui
            .quick_open_matches
            .iter()
            .position(|m| m.relative_path.ends_with("todo.txt"))
            .unwrap();
        assert!(click_hit_action(&mut gui, |a| {
            matches!(a, UiAction::SelectQuickOpenMatch(i) if *i == target_idx)
        }));
        assert!(!gui.quick_open_open);
        assert_eq!(
            gui.workbench
                .active_tab()
                .unwrap()
                .session
                .buffer
                .to_string(),
            "Todo list"
        );

        // ArrowDown + Enter selection works
        assert!(gui.handle_key_input(&Key::Character("p".into()), None, p_mod));
        assert!(gui.handle_key_input(
            &Key::Named(NamedKey::ArrowDown),
            None,
            ModifiersState::empty()
        ));
        assert_eq!(gui.quick_open_selected, 1);
        let expected_path = gui.quick_open_matches[1].path.clone();
        assert!(gui.handle_key_input(&Key::Named(NamedKey::Enter), None, ModifiersState::empty()));
        assert_eq!(
            gui.workbench.active_tab().unwrap().session.path.as_deref(),
            Some(expected_path.as_path())
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_cmd_k_link_prompt_prefill_typing_ime_cancel_insert_and_clipboard() {
        let mut gui = controller_with_doc("Click here please");
        let p_mod = primary_test_mod();

        // 1. Non-URL clipboard does not prefill; IME commit + Esc cancels cleanly
        gui.clipboard.write("not a url just text", None).unwrap();
        gui.workbench
            .active_tab_mut()
            .unwrap()
            .edit(|ctx| ctx.set_selection("Click ".len(), "Click here".len()))
            .unwrap();
        assert!(gui.handle_key_input(&Key::Character("k".into()), None, p_mod));
        assert!(gui.link_prompt_open);
        assert_eq!(gui.link_prompt_url, "");
        gui.handle_ime_event(&Ime::Commit("guide.md".to_string()));
        assert_eq!(gui.link_prompt_url, "guide.md");
        assert!(gui.handle_key_input(&Key::Named(NamedKey::Escape), None, ModifiersState::empty()));
        assert!(!gui.link_prompt_open);
        assert_eq!(
            gui.workbench
                .active_tab()
                .unwrap()
                .session
                .buffer
                .to_string(),
            "Click here please"
        );

        // 2. HTTPS URL clipboard prefills cleanly and Enter inserts [here](url)
        gui.clipboard
            .write("  https://vellmark.dev  ", None)
            .unwrap();
        assert!(gui.handle_key_input(&Key::Character("k".into()), None, p_mod));
        assert!(gui.link_prompt_open);
        assert_eq!(gui.link_prompt_url, "https://vellmark.dev");
        assert!(gui.handle_key_input(&Key::Named(NamedKey::Enter), None, ModifiersState::empty()));
        assert!(!gui.link_prompt_open);
        assert_eq!(
            gui.workbench
                .active_tab()
                .unwrap()
                .session
                .buffer
                .to_string(),
            "Click [here](https://vellmark.dev) please"
        );

        // 3. Cmd+K at caret + typed/pasted URL + Insert button click places caret inside []
        let end = gui
            .workbench
            .active_tab()
            .unwrap()
            .session
            .buffer
            .len_bytes();
        gui.workbench
            .active_tab_mut()
            .unwrap()
            .edit(|ctx| ctx.set_caret(end))
            .unwrap();
        gui.clipboard.write("ignored text", None).unwrap();
        assert!(gui.handle_key_input(&Key::Character("k".into()), None, p_mod));
        assert_eq!(gui.link_prompt_url, "");
        for ch in "https://".chars() {
            let s = ch.to_string();
            assert!(gui.handle_key_input(
                &Key::Character(s.clone().into()),
                Some(&s),
                ModifiersState::empty()
            ));
        }
        gui.clipboard.write("example.org", None).unwrap();
        assert!(gui.handle_key_input(&Key::Character("v".into()), None, p_mod));
        assert_eq!(gui.link_prompt_url, "https://example.org");
        let _ = gui.build_frame();
        assert!(click_hit_action(&mut gui, |a| matches!(
            a,
            UiAction::ConfirmLinkPrompt
        )));
        assert!(!gui.link_prompt_open);
        assert_eq!(
            gui.workbench
                .active_tab()
                .unwrap()
                .session
                .buffer
                .to_string(),
            "Click [here](https://vellmark.dev) please[](https://example.org)"
        );
        assert_eq!(
            gui.workbench.active_tab().unwrap().view.selection.head,
            "Click [here](https://vellmark.dev) please[".len()
        );
    }

    #[test]
    fn test_cmd_ctrl_click_navigation_external_relative_and_heading_anchors() {
        let dir = unique_test_dir("vellmark-gui-link-nav");
        let rec_root = dir.join("rec");
        let ws = dir.join("ws");
        fs::create_dir_all(ws.join("docs")).unwrap();
        let arch_md = ws.join("arch.md");
        fs::write(
            &arch_md,
            "# Architecture\n\nIntro paragraph.\n\n## Deep Dive\n\nNested architecture details.\n",
        )
        .unwrap();
        let intro_md = ws.join("docs/intro.md");
        let intro_src = "\
# Intro\n\
\n\
Plain line before links\n\
\n\
[Vellmark](https://vellmark.dev \"Home\")\n\
\n\
[Anchor First](#second-heading)\n\
\n\
[Anchor Second](#second-heading-1)\n\
\n\
[Relative Arch](<../arch.md#deep-dive> \"Arch\")\n\
\n\
[Broken Rel](missing-note.md)\n\
\n\
## Second Heading\n\
\n\
First duplicate body\n\
\n\
## Second Heading\n\
\n\
Second duplicate body\n";
        fs::write(&intro_md, intro_src).unwrap();

        let opener = Arc::new(MemoryLinkOpener::default());
        struct SharedOpener(Arc<MemoryLinkOpener>);
        impl LinkOpenerBackend for SharedOpener {
            fn open_url(&self, url: &str) -> Result<(), String> {
                self.0.open_url(url)
            }
        }

        let mut wb = AppWorkbench::new(AppConfig::default(), Some(rec_root));
        wb.open_folder(&ws).unwrap();
        wb.open_document(&intro_md).unwrap();
        let mut gui = GuiController::with_clipboard(wb, Box::new(MemoryClipboard::default()));
        gui.set_link_opener(Box::new(SharedOpener(opener.clone())));

        let p_mod = primary_test_mod();

        // Helper: find screen pixel inside a given byte offset while link markers are collapsed
        let caret_point_for_offset =
            |gui: &mut GuiController, caret_off: usize, collapse_off: usize| {
                gui.workbench
                    .active_tab_mut()
                    .unwrap()
                    .edit(|ctx| ctx.set_caret(caret_off))
                    .unwrap();
                let _ = gui.build_frame();
                let rect = gui.last_ime_cursor_area.expect("caret rect");
                gui.workbench
                    .active_tab_mut()
                    .unwrap()
                    .edit(|ctx| ctx.set_caret(collapse_off))
                    .unwrap();
                let _ = gui.build_frame();
                (rect.x + 8.0, rect.y + rect.h * 0.5)
            };

        // Plain line does not trigger pointer cursor or link open on Cmd/Ctrl+Click
        let plain_off = intro_src.find("Plain line").unwrap() + 2;
        let (px, py) = caret_point_for_offset(&mut gui, plain_off, 0);
        let (icon_mod, _) = gui.handle_pointer_move_with_mods(px, py, p_mod);
        assert_eq!(icon_mod, CursorIcon::Text);
        assert!(gui.handle_pointer_down_with_mods(px, py, p_mod));
        assert!(opener.opened_urls().is_empty());

        // 1. External URL link: Pointer cursor under Cmd/Ctrl, Text cursor without; Click opens URL
        let ext_off = intro_src.find("Vellmark").unwrap() + 2;
        let (ex, ey) = caret_point_for_offset(&mut gui, ext_off, 0);
        assert_eq!(
            gui.handle_pointer_move_with_mods(ex, ey, ModifiersState::empty())
                .0,
            CursorIcon::Text
        );
        assert_eq!(
            gui.handle_pointer_move_with_mods(ex, ey, p_mod).0,
            CursorIcon::Pointer
        );
        assert!(gui.handle_pointer_down_with_mods(ex, ey, p_mod));
        assert!(!gui.dragging_selection);
        assert_eq!(
            opener.opened_urls(),
            vec!["https://vellmark.dev".to_string()]
        );

        // 2. Same-document heading anchor jump & duplicate GitHub slug disambiguation (-1)
        let a1_off = intro_src.find("Anchor First").unwrap() + 2;
        let (ax1, ay1) = caret_point_for_offset(&mut gui, a1_off, 0);
        assert!(gui.handle_pointer_down_with_mods(ax1, ay1, p_mod));
        let first_heading_pos = intro_src.find("## Second Heading").unwrap();
        let caret1 = gui.workbench.active_tab().unwrap().view.selection.head;
        assert!(
            caret1 >= first_heading_pos && caret1 < intro_src.rfind("## Second Heading").unwrap(),
            "caret1={caret1} should jump to first Second Heading"
        );

        let a2_off = intro_src.find("Anchor Second").unwrap() + 2;
        let (ax2, ay2) = caret_point_for_offset(&mut gui, a2_off, 0);
        assert!(gui.handle_pointer_down_with_mods(ax2, ay2, p_mod));
        let second_heading_pos = intro_src.rfind("## Second Heading").unwrap();
        let caret2 = gui.workbench.active_tab().unwrap().view.selection.head;
        assert!(
            caret2 >= second_heading_pos,
            "caret2={caret2} should jump to second Second Heading at {second_heading_pos}"
        );

        // 3. Broken relative link sets status notice safely without drag
        let broken_off = intro_src.find("Broken Rel").unwrap() + 2;
        let (bx, by) = caret_point_for_offset(&mut gui, broken_off, 0);
        assert!(gui.handle_pointer_down_with_mods(bx, by, p_mod));
        assert!(!gui.dragging_selection);
        assert!(
            gui.active_status_notice()
                .is_some_and(|n| n.contains("not found")),
            "expected not found notice, got {:?}",
            gui.active_status_notice()
        );

        // 4. Relative `.md#heading` angle-bracket link opens target file and jumps to `## Deep Dive`
        let rel_off = intro_src.find("Relative Arch").unwrap() + 2;
        let (rx, ry) = caret_point_for_offset(&mut gui, rel_off, 0);
        assert!(gui.handle_pointer_down_with_mods(rx, ry, p_mod));
        let active = gui.workbench.active_tab().unwrap();
        assert_eq!(active.session.path.as_deref(), Some(arch_md.as_path()));
        let arch_text = active.session.buffer.to_string();
        let deep_idx = arch_text.find("Deep Dive").unwrap();
        assert_eq!(active.view.selection.head, deep_idx);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_link_security_scheme_allowlist_and_escaped_link_roundtrip() {
        // 1. `validate_native_open_target` enforces scheme allowlist, validates HTTP(S)/mailto
        // authorities as defense-in-depth (`https://x&calc`), and keeps query `&` & `%` intact.
        assert!(validate_native_open_target("https://example.com/search?q=1&b=2%203").is_ok());
        assert!(
            validate_native_open_target(
                "https://en.wikipedia.org/wiki/Rust_(programming_language)"
            )
            .is_ok()
        );
        assert!(validate_native_open_target("https://[::1]:8080/docs?x=1&y=2").is_ok());
        assert!(validate_native_open_target("mailto:test@example.com").is_ok());
        assert!(validate_native_open_target("mailto:test@example.com?subject=hi&body=bye").is_ok());
        assert!(validate_native_open_target("mailto:?subject=Hello&body=World").is_ok());
        assert!(validate_native_open_target("mailto:?subject=").is_ok());
        assert!(validate_native_open_target("mailto:o'connor@example.com").is_ok());
        assert!(
            validate_native_open_target(
                "mailto:user+tag!name$1@example.com,second@example.com?subject=咨询"
            )
            .is_ok()
        );
        assert!(
            validate_native_open_target("mailto:user%26tag@example.com?subject=100%25").is_ok()
        );
        assert!(validate_native_open_target("mailto:?subject=Hello World").is_ok());
        assert!(validate_native_open_target("mailto:?subject=a=b").is_ok());
        assert!(validate_native_open_target("mailto:?subject=How are you?").is_ok());
        assert!(validate_native_open_target("mailto:?subject=Hello%23tag").is_ok());
        assert!(validate_native_open_target("/tmp/report.pdf").is_ok());
        assert!(validate_native_open_target(r"C:\docs\report.pdf").is_ok());

        for (blocked, expected_err) in [
            ("", "Invalid link target"),
            ("https://example.com\ncalc", "Invalid link target"),
            ("https://example.com\0", "Invalid link target"),
            ("https://x&calc", "Blocked unsafe or malformed external URL"),
            (
                "http://evil.com&calc",
                "Blocked unsafe or malformed external URL",
            ),
            (
                "https://evil.com|calc",
                "Blocked unsafe or malformed external URL",
            ),
            (
                "https://example.com/a|b",
                "Blocked unsafe or malformed external URL",
            ),
            (
                "https://:8080/path",
                "Blocked unsafe or malformed external URL",
            ),
            ("https://", "Blocked unsafe or malformed external URL"),
            ("http:foo", "Blocked unsafe or malformed external URL"),
            ("mailto:", "Blocked unsafe or malformed external URL"),
            ("mailto:?", "Blocked unsafe or malformed external URL"),
            ("mailto:?abc", "Blocked unsafe or malformed external URL"),
            ("mailto:?&", "Blocked unsafe or malformed external URL"),
            ("mailto:?=foo", "Blocked unsafe or malformed external URL"),
            (
                "mailto:?subject=100%",
                "Blocked unsafe or malformed external URL",
            ),
            (
                "mailto:?subject=Hello#fragment",
                "Blocked unsafe or malformed external URL",
            ),
            (
                "mailto:?sub?ject=Hello",
                "Blocked unsafe or malformed external URL",
            ),
            (
                "mailto:evil@example.com&calc",
                "Blocked unsafe or malformed external URL",
            ),
            (
                "mailto:evil@example.com;calc",
                "Blocked unsafe or malformed external URL",
            ),
            (
                "mailto:evil@example.com=1",
                "Blocked unsafe or malformed external URL",
            ),
            (
                "mailto:evil@example.com/path",
                "Blocked unsafe or malformed external URL",
            ),
            (
                "mailto:evil@example.com#frag",
                "Blocked unsafe or malformed external URL",
            ),
            (
                "mailto:evil@[127.0.0.1]",
                "Blocked unsafe or malformed external URL",
            ),
            (
                "mailto:evil%2@example.com",
                "Blocked unsafe or malformed external URL",
            ),
            (
                "ftp://example.com/file",
                "Blocked unsupported URL scheme: ftp",
            ),
            (
                "smb://attacker/share",
                "Blocked unsupported URL scheme: smb",
            ),
            (
                "search-ms:displayname=evil",
                "Blocked unsupported URL scheme: search-ms",
            ),
            ("ms-msdt:/id", "Blocked unsupported URL scheme: ms-msdt"),
            (
                "javascript:alert(1)//http://example.com",
                "Blocked unsupported URL scheme: javascript",
            ),
            (
                "file:///tmp/secret.txt",
                "Blocked unsupported URL scheme: file",
            ),
            (
                r"C:\notes\payload.bat",
                "Blocked opening executable file directly",
            ),
            ("/tmp/setup.exe", "Blocked opening executable file directly"),
        ] {
            assert_eq!(
                validate_native_open_target(blocked),
                Err(expected_err.to_string()),
                "unexpected validate_native_open_target result for `{blocked}`"
            );
        }

        // 2. `open_resolved_link_destination` blocks arbitrary URI schemes, malformed external URLs,
        // remote `file://host/...` SMB/UNC, and executable local files while allowing `FILE:///...`.
        let dir = unique_test_dir("vellmark-gui-link-sec");
        let ws = dir.join("ws");
        fs::create_dir_all(&ws).unwrap();
        let note = ws.join("note.md");
        fs::write(&note, "# Hello\n").unwrap();
        let script = ws.join("run.bat");
        fs::write(&script, "@echo hi\n").unwrap();

        let opener = Arc::new(MemoryLinkOpener::default());
        struct SharedOpener(Arc<MemoryLinkOpener>);
        impl LinkOpenerBackend for SharedOpener {
            fn open_url(&self, url: &str) -> Result<(), String> {
                self.0.open_url(url)
            }
        }
        let mut wb = AppWorkbench::new(AppConfig::default(), Some(dir.join("rec")));
        wb.open_folder(&ws).unwrap();
        let mut gui = GuiController::with_clipboard(wb, Box::new(MemoryClipboard::default()));
        gui.set_link_opener(Box::new(SharedOpener(opener.clone())));

        for (disallowed, expected_notice) in [
            ("ftp://example.com/a", "Unsupported link scheme: ftp"),
            ("smb://evil.example/share", "Unsupported link scheme: smb"),
            ("ms-msdt:/id", "Unsupported link scheme: ms-msdt"),
            (
                "javascript:alert(1)//http://x",
                "Unsupported link scheme: javascript",
            ),
            ("https://x&calc", "Blocked unsafe or malformed external URL"),
            ("mailto:?abc", "Blocked unsafe or malformed external URL"),
            (
                "file://attacker/share/secret.md",
                "Blocked non-local file URI",
            ),
            ("run.bat", "Blocked opening executable file: run.bat"),
        ] {
            gui.open_resolved_link_destination(disallowed);
            assert!(
                opener.opened_urls().is_empty(),
                "`{disallowed}` must never reach LinkOpenerBackend"
            );
            assert_eq!(
                gui.active_status_notice().as_deref(),
                Some(expected_notice),
                "unexpected notice for `{disallowed}`"
            );
        }

        for allowed_mailto in [
            "mailto:?subject=Hello&body=World",
            "mailto:o'connor@example.com",
        ] {
            gui.open_resolved_link_destination(allowed_mailto);
        }
        assert_eq!(
            opener.opened_urls(),
            vec![
                "mailto:?subject=Hello&body=World".to_string(),
                "mailto:o'connor@example.com".to_string(),
            ]
        );

        // Case-insensitive `FILE:///...` resolves local note cleanly inside Vellmark.
        let note_display = note.to_string_lossy().replace('\\', "/");
        let file_uri = if note_display.starts_with('/') {
            format!("FILE://{note_display}")
        } else {
            format!("FILE:///{note_display}")
        };
        gui.open_resolved_link_destination(&file_uri);
        assert_eq!(
            gui.workbench.active_tab().unwrap().session.path.as_deref(),
            Some(note.as_path())
        );

        // 3. End-to-end `Cmd+K` (`InsertLink`) -> HTML export -> WYSIWYG parser -> `link_destination_at_offset`.
        let cases = [
            ("a]b", "http://example.com/path"),
            ("text", "http://example.com/a b"),
            ("text", "http://example.com/a(b"),
            ("text", "http://example.com/a)"),
            ("text", r"http://example.com/a\b"),
        ];
        for (idx, (label, url)) in cases.iter().enumerate() {
            let tab = gui.workbench.active_tab_mut().unwrap();
            let full = tab.session.buffer.len_bytes();
            tab.edit(|ctx| {
                ctx.set_selection(0, full);
                ctx.insert_text(label);
                ctx.set_selection(0, label.len());
                ctx.insert_link(url);
            })
            .unwrap();
            let md = gui
                .workbench
                .active_tab()
                .unwrap()
                .session
                .buffer
                .to_string();
            let html = vellmark_parser::render_markdown_to_html(&md);
            assert!(
                html.contains("<a href="),
                "case {idx} (`{md}`) did not export an HTML link: {html}"
            );
            let extracted = link_destination_at_offset(gui.workbench.active_tab().unwrap(), 1);
            assert_eq!(
                extracted.as_deref(),
                Some(*url),
                "case {idx} (`{md}`) failed round-trip extraction"
            );
        }

        // 4. Bare invalid space/backslash-space links do NOT parse or open.
        for bad_md in ["[a](my file.md)", "[a](my\\ file.md)"] {
            let tab = gui.workbench.active_tab_mut().unwrap();
            let full = tab.session.buffer.len_bytes();
            tab.edit(|ctx| {
                ctx.set_selection(0, full);
                ctx.insert_text(bad_md);
            })
            .unwrap();
            assert_eq!(
                link_destination_at_offset(gui.workbench.active_tab().unwrap(), 1),
                None,
                "`{bad_md}` should not produce an interactive link target"
            );
        }

        let _ = fs::remove_dir_all(&dir);
    }

    fn test_primary_modifier() -> ModifiersState {
        if cfg!(target_os = "macos") {
            ModifiersState::SUPER
        } else {
            ModifiersState::CONTROL
        }
    }

    fn test_encode_png_2x2() -> Vec<u8> {
        let img = image::RgbaImage::from_pixel(2, 2, image::Rgba([24, 144, 255, 255]));
        let mut cursor = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(img)
            .write_to(&mut cursor, image::ImageFormat::Png)
            .unwrap();
        cursor.into_inner()
    }

    #[test]
    fn test_code_block_language_picker_and_line_numbers_toggle() {
        let mut wb = AppWorkbench::new(AppConfig::default(), None);
        wb.new_document();
        wb.active_tab_mut()
            .unwrap()
            .session
            .buffer
            .reset_content("hello\n\n```rust\nfn main() {}\n```\n");
        let mut gui = GuiController::new(wb);
        assert!(gui.workbench.config.show_code_line_numbers);

        let _ = gui.build_frame();
        let ln_rect = gui
            .ui_hit_zones
            .iter()
            .find(|z| matches!(z.action, UiAction::ToggleCodeLineNumbers))
            .map(|z| z.rect)
            .expect("expected code block line-number toggle pill");
        let lang_rect = gui
            .ui_hit_zones
            .iter()
            .find(|z| matches!(z.action, UiAction::OpenCodeLangPicker { .. }))
            .map(|z| z.rect)
            .expect("expected code block language picker pill");

        // Click LN toggle pill -> disables line numbers; Cmd/Ctrl+Shift+L re-enables.
        assert!(gui.handle_pointer_down(ln_rect.x + ln_rect.w * 0.5, ln_rect.y + ln_rect.h * 0.5));
        assert!(!gui.workbench.config.show_code_line_numbers);
        assert!(gui.handle_key_input(
            &Key::Character("l".into()),
            None,
            test_primary_modifier() | ModifiersState::SHIFT
        ));
        assert!(gui.workbench.config.show_code_line_numbers);

        // Click Language Picker pill -> search "py" -> Enter -> fence language becomes python.
        assert!(gui.handle_pointer_down(
            lang_rect.x + lang_rect.w * 0.5,
            lang_rect.y + lang_rect.h * 0.5
        ));
        assert!(gui.code_lang_open);
        assert!(gui.handle_key_input(
            &Key::Character("p".into()),
            Some("p"),
            ModifiersState::empty()
        ));
        assert!(gui.handle_key_input(
            &Key::Character("y".into()),
            Some("y"),
            ModifiersState::empty()
        ));
        assert!(!gui.code_lang_matches.is_empty());
        assert!(gui.handle_key_input(&Key::Named(NamedKey::Enter), None, ModifiersState::empty()));
        assert!(!gui.code_lang_open);
        assert!(
            gui.workbench
                .active_tab()
                .unwrap()
                .session
                .buffer
                .to_string()
                .contains("```python\nfn main() {}")
        );
    }

    #[test]
    fn test_code_block_shortcut_inserts_or_opens_language_picker() {
        let mut wb = AppWorkbench::new(AppConfig::default(), None);
        wb.new_document();
        wb.active_tab_mut()
            .unwrap()
            .session
            .buffer
            .reset_content("paragraph");
        wb.active_tab_mut()
            .unwrap()
            .edit(|ctx| ctx.set_caret("paragraph".len()))
            .unwrap();
        let mut gui = GuiController::new(wb);

        let (insert_key, picker_key, code_mod) = if cfg!(target_os = "macos") {
            ("ç", "c", ModifiersState::SUPER | ModifiersState::ALT)
        } else {
            ("K", "k", ModifiersState::CONTROL | ModifiersState::SHIFT)
        };

        // Outside a code block, the platform shortcut inserts a fence and places the caret inside.
        assert!(gui.handle_key_input(&Key::Character(insert_key.into()), None, code_mod));
        let md = gui
            .workbench
            .active_tab()
            .unwrap()
            .session
            .buffer
            .to_string();
        assert!(
            md.contains("```"),
            "expected code fence inserted, got: {md}"
        );
        assert!(!gui.code_lang_open);

        // Inside the new code block, the same shortcut opens the language picker.
        assert!(gui.handle_key_input(&Key::Character(picker_key.into()), None, code_mod));
        assert!(gui.code_lang_open);
        assert!(gui.handle_key_input(
            &Key::Character("r".into()),
            Some("r"),
            ModifiersState::empty()
        ));
        assert!(gui.handle_key_input(
            &Key::Character("u".into()),
            Some("u"),
            ModifiersState::empty()
        ));
        assert!(gui.handle_key_input(
            &Key::Character("s".into()),
            Some("s"),
            ModifiersState::empty()
        ));
        assert!(gui.handle_key_input(
            &Key::Character("t".into()),
            Some("t"),
            ModifiersState::empty()
        ));
        assert!(gui.handle_key_input(&Key::Named(NamedKey::Enter), None, ModifiersState::empty()));
        assert!(!gui.code_lang_open);
        assert!(
            gui.workbench
                .active_tab()
                .unwrap()
                .session
                .buffer
                .to_string()
                .contains("```rust\n")
        );
    }

    #[test]
    fn test_clipboard_png_paste_imports_asset_and_skips_overlay_input() {
        let dir = std::env::temp_dir().join(format!(
            "vellmark-gui-clip-img-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        let note = dir.join("note.md");
        fs::write(&note, "intro\n").unwrap();

        let png = test_encode_png_2x2();
        let clip = MemoryClipboard::with_initial(ClipboardContent {
            text: None,
            html: None,
            image_png: Some(png.clone()),
        });
        let mut wb = AppWorkbench::new(AppConfig::default(), Some(dir.clone()));
        wb.open_document(&note).unwrap();
        let end = wb.active_tab().unwrap().session.buffer.len_bytes();
        wb.active_tab_mut()
            .unwrap()
            .edit(|ctx| ctx.set_caret(end))
            .unwrap();
        let mut gui = GuiController::with_clipboard(wb, Box::new(clip));

        gui.paste_clipboard();
        let expected_asset = dir.join("assets").join("pasted-image.png");
        assert_eq!(fs::read(&expected_asset).unwrap(), png);
        assert!(
            gui.workbench
                .active_tab()
                .unwrap()
                .session
                .buffer
                .to_string()
                .contains("![image](./assets/pasted-image.png)")
        );
        assert!(
            gui.active_status_notice()
                .as_deref()
                .is_some_and(|n| n.contains("pasted-image.png"))
        );

        // Pasting while a text overlay is focused must never import an image into assets.
        gui.find_open = true;
        gui.find_focused = true;
        gui.paste_clipboard();
        assert!(gui.find_query.is_empty());
        assert!(!dir.join("assets").join("pasted-image-2.png").exists());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_handle_dropped_image_imports_to_assets_while_md_drop_opens_tab() {
        let dir = std::env::temp_dir().join(format!(
            "vellmark-gui-drop-img-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        let note = dir.join("note.md");
        let other = dir.join("other.md");
        let shot = dir.join("shot.png");
        let png = test_encode_png_2x2();
        fs::write(&note, "# Start\n").unwrap();
        fs::write(&other, "# Other\n").unwrap();
        fs::write(&shot, &png).unwrap();

        let mut wb = AppWorkbench::new(AppConfig::default(), Some(dir.clone()));
        wb.open_document(&note).unwrap();
        let end = wb.active_tab().unwrap().session.buffer.len_bytes();
        wb.active_tab_mut()
            .unwrap()
            .edit(|ctx| ctx.set_caret(end))
            .unwrap();
        let mut gui = GuiController::new(wb);

        gui.handle_dropped_path(&shot);
        assert_eq!(
            gui.workbench.active_tab().unwrap().session.path.as_deref(),
            Some(note.as_path())
        );
        let copied_asset = dir.join("assets").join("shot.png");
        assert_eq!(fs::read(&copied_asset).unwrap(), png);
        assert!(
            gui.workbench
                .active_tab()
                .unwrap()
                .session
                .buffer
                .to_string()
                .contains("![shot](./assets/shot.png)")
        );

        // Non-image Markdown drop still opens as a document tab.
        gui.handle_dropped_path(&other);
        assert_eq!(
            gui.workbench.active_tab().unwrap().session.path.as_deref(),
            Some(other.as_path())
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_insert_table_prompt_and_toolbar_pill_clicks_sync_auto_align() {
        let mut wb = AppWorkbench::new(AppConfig::default(), None);
        wb.new_document();
        let mut gui = GuiController::new(wb);
        let mods = test_primary_modifier();

        // Trigger Insert Table prompt with Cmd/Ctrl+T, replace default "3x4" with "2x3", press Enter
        assert!(gui.handle_key_input(&Key::Character("t".into()), Some("t"), mods));
        assert!(gui.insert_table_open);
        assert_eq!(gui.insert_table_input, "3x4");
        for _ in 0..3 {
            assert!(gui.handle_key_input(
                &Key::Named(NamedKey::Backspace),
                None,
                ModifiersState::empty()
            ));
        }
        for ch in ["2", "x", "3"] {
            assert!(gui.handle_key_input(
                &Key::Character(ch.into()),
                Some(ch),
                ModifiersState::empty()
            ));
        }
        assert!(gui.handle_key_input(&Key::Named(NamedKey::Enter), None, ModifiersState::empty()));
        assert!(!gui.insert_table_open);

        let doc = gui
            .workbench
            .active_tab()
            .unwrap()
            .session
            .buffer
            .to_string();
        let lines: Vec<&str> = doc.lines().collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0].matches('|').count(), 4);

        // Build frame in WYSIWYG mode with caret inside a table cell -> table toolbar pills appear
        let _ = gui.build_frame();
        let add_row_rect = gui
            .ui_hit_zones
            .iter()
            .find(|z| {
                matches!(
                    z.action,
                    UiAction::TableToolbarAction(TableButtonKind::AddRowBelow)
                )
            })
            .map(|z| z.rect)
            .expect("expected +Row↓ table toolbar pill");
        assert!(gui.handle_pointer_down(add_row_rect.x + 2.0, add_row_rect.y + 2.0));
        let doc_after_row = gui
            .workbench
            .active_tab()
            .unwrap()
            .session
            .buffer
            .to_string();
        assert_eq!(doc_after_row.lines().count(), doc.lines().count() + 1);

        let _ = gui.build_frame();
        let align_right_rect = gui
            .ui_hit_zones
            .iter()
            .find(|z| {
                matches!(
                    z.action,
                    UiAction::TableToolbarAction(TableButtonKind::AlignRight)
                )
            })
            .map(|z| z.rect)
            .expect("expected R alignment pill");
        assert!(gui.handle_pointer_down(align_right_rect.x + 2.0, align_right_rect.y + 2.0));
        let doc_aligned = gui
            .workbench
            .active_tab()
            .unwrap()
            .session
            .buffer
            .to_string();
        let delim = doc_aligned.lines().nth(1).unwrap();
        assert!(delim.contains(":"));

        let _ = gui.build_frame();
        let del_row_rect = gui
            .ui_hit_zones
            .iter()
            .find(|z| {
                matches!(
                    z.action,
                    UiAction::TableToolbarAction(TableButtonKind::DeleteRow)
                )
            })
            .map(|z| z.rect)
            .expect("expected -Row pill");
        assert!(gui.handle_pointer_down(del_row_rect.x + 2.0, del_row_rect.y + 2.0));
        assert!(
            gui.workbench
                .active_tab()
                .unwrap()
                .session
                .buffer
                .to_string()
                .lines()
                .count()
                < doc_after_row.lines().count()
        );

        // Changing config.table_auto_align dynamically syncs into tab.view on dispatch_editor
        gui.workbench.config.table_auto_align = false;
        assert!(gui.handle_key_input(&Key::Named(NamedKey::Enter), None, ModifiersState::empty()));
        assert!(!gui.workbench.active_tab().unwrap().view.table_auto_align);

        // Confirming Insert Table prompt inside a fenced code block reports rejection instead of success
        let code_doc = "```rust\nlet x = 1;\n```\n";
        let tab = gui.workbench.active_tab_mut().unwrap();
        tab.session.buffer.reset_content(code_doc);
        tab.view.selection = Selection::caret(code_doc.find("let").unwrap());
        gui.open_insert_table_prompt();
        gui.insert_table_input = "2x2".to_string();
        assert!(gui.handle_key_input(&Key::Named(NamedKey::Enter), None, ModifiersState::empty()));
        assert_eq!(
            gui.workbench
                .active_tab()
                .unwrap()
                .session
                .buffer
                .to_string(),
            code_doc
        );
        assert!(
            gui.status_notice
                .as_ref()
                .is_some_and(|(m, _)| m.contains("Cannot insert table"))
        );
    }

    #[test]
    fn test_global_workspace_search_modal_case_toggle_and_jump_to_line_offset() {
        let dir = std::env::temp_dir().join(format!(
            "vellmark-gui-gsearch-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        let alpha = dir.join("alpha.md");
        let beta = dir.join("beta.md");
        fs::write(&alpha, "# Title\nSecond CaseMatch line\n").unwrap();
        fs::write(&beta, "Hello\ncasematch here\n").unwrap();

        let mut wb = AppWorkbench::new(AppConfig::default(), Some(dir.clone()));
        wb.open_folder(&dir).unwrap();
        let mut gui = GuiController::new(wb);

        // Cmd/Ctrl+Shift+F opens Global Search without enabling Focus Mode
        let shift_primary = test_primary_modifier() | ModifiersState::SHIFT;
        assert!(gui.handle_key_input(&Key::Character("f".into()), None, shift_primary));
        assert!(gui.global_search_open);
        assert!(!gui.workbench.config.focus_mode);

        for ch in "CaseMatch".chars() {
            let s = ch.to_string();
            assert!(gui.handle_key_input(
                &Key::Character(s.as_str().into()),
                Some(&s),
                ModifiersState::empty()
            ));
        }
        // Key input must stay non-blocking & debounced (not scanned synchronously per keystroke)
        assert!(gui.global_search_searching);
        assert!(gui.global_search_debounce_deadline.is_some());
        assert!(gui.global_search_matches.is_empty());

        // Advance past debounce window and wait for background search worker to deliver matches
        let deadline = Instant::now() + Duration::from_secs(5);
        gui.tick_runtime_at(Instant::now() + GLOBAL_SEARCH_DEBOUNCE + Duration::from_millis(5));
        while gui.global_search_searching {
            assert!(Instant::now() < deadline, "global search worker timed out");
            std::thread::sleep(Duration::from_millis(5));
            let _ = gui.tick_runtime_at(Instant::now());
        }
        assert_eq!(gui.global_search_matches.len(), 2);

        // Simulate worker exit/panic while searching -> poll_global_search clears `global_search_searching`
        gui.global_search_worker = Some(WorkspaceSearchWorker::disconnected_for_test());
        gui.global_search_searching = true;
        assert!(gui.tick_runtime_at(Instant::now()));
        assert!(!gui.global_search_searching);
        assert!(gui.global_search_worker.is_none());

        // Click "Aa" pill to toggle case sensitivity -> respawns worker & searches immediately -> only alpha.md remains
        let _ = gui.build_frame();
        let case_rect = gui
            .ui_hit_zones
            .iter()
            .find(|z| matches!(z.action, UiAction::ToggleGlobalSearchCase))
            .map(|z| z.rect)
            .expect("expected Aa case-sensitivity toggle pill");
        assert!(gui.handle_pointer_down(case_rect.x + 2.0, case_rect.y + 2.0));
        assert!(gui.global_search_case_sensitive);
        while gui.global_search_searching {
            assert!(Instant::now() < deadline, "case-sensitive search timed out");
            std::thread::sleep(Duration::from_millis(5));
            let _ = gui.tick_runtime_at(Instant::now());
        }
        assert_eq!(gui.global_search_matches.len(), 1);
        assert_eq!(gui.global_search_matches[0].line_number, 2);

        // Confirm with Enter -> opens alpha.md and moves caret to matching offset
        assert!(gui.handle_key_input(&Key::Named(NamedKey::Enter), None, ModifiersState::empty()));
        assert!(!gui.global_search_open);
        assert!(!gui.global_search_searching);
        let tab = gui.workbench.active_tab().unwrap();
        assert_eq!(
            fs::canonicalize(tab.session.path.as_ref().unwrap()).unwrap(),
            fs::canonicalize(&alpha).unwrap()
        );
        assert_eq!(tab.view.selection.head, "# Title\nSecond ".len());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_sidebar_recent_tab_and_file_create_rename_delete_flow() {
        let dir = std::env::temp_dir().join(format!(
            "vellmark-gui-files-recent-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        let existing = dir.join("existing.md");
        fs::write(&existing, "# Existing\n").unwrap();

        let mut wb = AppWorkbench::new(AppConfig::default(), Some(dir.clone()));
        wb.open_folder(&dir).unwrap();
        let existing_id = wb.open_document(&existing).unwrap();
        assert!(wb.close_tab(existing_id, false));
        let mut gui = GuiController::new(wb);
        gui.workbench.config.sidebar_visible = true;

        // Switch to Recent sidebar tab and reopen existing.md by clicking it
        let _ = gui.build_frame();
        let recent_tab_rect = gui
            .ui_hit_zones
            .iter()
            .find(|z| {
                matches!(
                    z.action,
                    UiAction::SwitchSidebarSection(SidebarSection::Recent)
                )
            })
            .map(|z| z.rect)
            .expect("expected Recent sidebar tab");
        assert!(gui.handle_pointer_down(recent_tab_rect.x + 2.0, recent_tab_rect.y + 2.0));
        assert_eq!(gui.sidebar_section, SidebarSection::Recent);

        let _ = gui.build_frame();
        let canonical_existing = fs::canonicalize(&existing).unwrap();
        let recent_item_rect = gui
            .ui_hit_zones
            .iter()
            .find(|z| matches!(&z.action, UiAction::OpenRecentFile(p) if p == &canonical_existing))
            .map(|z| z.rect)
            .expect("expected existing.md entry in Recent");
        assert!(gui.handle_pointer_down(recent_item_rect.x + 2.0, recent_item_rect.y + 2.0));
        assert_eq!(
            fs::canonicalize(
                gui.workbench
                    .active_tab()
                    .unwrap()
                    .session
                    .path
                    .as_ref()
                    .unwrap()
            )
            .unwrap(),
            canonical_existing
        );

        // Switch to Files, create "draft" -> creates draft.md and opens it
        let _ = gui.build_frame();
        let files_tab_rect = gui
            .ui_hit_zones
            .iter()
            .find(|z| {
                matches!(
                    z.action,
                    UiAction::SwitchSidebarSection(SidebarSection::Files)
                )
            })
            .map(|z| z.rect)
            .expect("expected Files sidebar tab");
        assert!(gui.handle_pointer_down(files_tab_rect.x + 2.0, files_tab_rect.y + 2.0));

        let _ = gui.build_frame();
        let new_btn_rect = gui
            .ui_hit_zones
            .iter()
            .find(|z| matches!(z.action, UiAction::OpenNewFilePrompt))
            .map(|z| z.rect)
            .expect("expected +File button");
        assert!(gui.handle_pointer_down(new_btn_rect.x + 2.0, new_btn_rect.y + 2.0));
        for ch in "draft".chars() {
            let s = ch.to_string();
            assert!(gui.handle_key_input(
                &Key::Character(s.as_str().into()),
                Some(&s),
                ModifiersState::empty()
            ));
        }
        assert!(gui.handle_key_input(&Key::Named(NamedKey::Enter), None, ModifiersState::empty()));
        let draft_path = dir.join("draft.md");
        assert!(draft_path.exists());
        assert_eq!(
            fs::canonicalize(
                gui.workbench
                    .active_tab()
                    .unwrap()
                    .session
                    .path
                    .as_ref()
                    .unwrap()
            )
            .unwrap(),
            fs::canonicalize(&draft_path).unwrap()
        );

        // Rename draft.md -> notes.md via Ren prompt
        let _ = gui.build_frame();
        let ren_btn_rect = gui
            .ui_hit_zones
            .iter()
            .find(|z| matches!(z.action, UiAction::OpenRenameFilePrompt))
            .map(|z| z.rect)
            .expect("expected Ren button");
        assert!(gui.handle_pointer_down(ren_btn_rect.x + 2.0, ren_btn_rect.y + 2.0));
        assert_eq!(gui.file_prompt_input, "draft.md");
        for _ in 0..8 {
            assert!(gui.handle_key_input(
                &Key::Named(NamedKey::Backspace),
                None,
                ModifiersState::empty()
            ));
        }
        for ch in "notes.md".chars() {
            let s = ch.to_string();
            assert!(gui.handle_key_input(
                &Key::Character(s.as_str().into()),
                Some(&s),
                ModifiersState::empty()
            ));
        }
        assert!(gui.handle_key_input(&Key::Named(NamedKey::Enter), None, ModifiersState::empty()));
        let notes_path = dir.join("notes.md");
        assert!(!draft_path.exists());
        assert!(notes_path.exists());
        assert_eq!(
            fs::canonicalize(
                gui.workbench
                    .active_tab()
                    .unwrap()
                    .session
                    .path
                    .as_ref()
                    .unwrap()
            )
            .unwrap(),
            fs::canonicalize(&notes_path).unwrap()
        );

        // Two-click delete safety: first click arms Del?, second click deletes file & marks open tab Missing
        let _ = gui.build_frame();
        let del_rect_1 = gui
            .ui_hit_zones
            .iter()
            .find(|z| matches!(z.action, UiAction::ArmOrConfirmDeleteFile))
            .map(|z| z.rect)
            .expect("expected Del button");
        assert!(gui.handle_pointer_down(del_rect_1.x + 2.0, del_rect_1.y + 2.0));
        assert!(notes_path.exists());
        assert!(gui.delete_confirm_path.is_some());

        let _ = gui.build_frame();
        let del_rect_2 = gui
            .ui_hit_zones
            .iter()
            .find(|z| matches!(z.action, UiAction::ArmOrConfirmDeleteFile))
            .map(|z| z.rect)
            .expect("expected armed Del? button");
        assert!(gui.handle_pointer_down(del_rect_2.x + 2.0, del_rect_2.y + 2.0));
        assert!(!notes_path.exists());
        assert!(matches!(
            gui.workbench.active_tab().unwrap().session.state,
            DocState::Missing { .. }
        ));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_view_modes_shortcuts_typewriter_pill_and_status_bar_selection_stats() {
        let mut wb = AppWorkbench::new(AppConfig::default(), None);
        wb.new_document();
        wb.active_tab_mut()
            .unwrap()
            .session
            .buffer
            .reset_content("hello world from vellmark\n");
        let mut gui = GuiController::new(wb);

        // Initial cached meta & reading time without selection
        let _ = gui.build_frame();
        let meta = gui.cached_meta.as_ref().unwrap();
        assert_eq!(meta.stats.word_count, 4);
        assert_eq!(meta.stats.reading_time_secs, 1);
        assert_eq!(meta.stats.selection_word_count, 0);
        assert_eq!(meta.stats.selection_char_count, 0);

        // Update selection to "world from" without changing buffer revision -> selection stats update
        gui.workbench
            .active_tab_mut()
            .unwrap()
            .edit(|ctx| ctx.set_selection(6, 16))
            .unwrap();
        let _ = gui.build_frame();
        let meta2 = gui.cached_meta.as_ref().unwrap();
        assert_eq!(meta2.stats.selection_word_count, 2);
        assert_eq!(meta2.stats.selection_char_count, 10);

        // F8 Focus Mode, F9 & TW pill Typewriter Mode, F11 Fullscreen
        assert!(!gui.workbench.config.focus_mode);
        assert!(gui.handle_key_input(&Key::Named(NamedKey::F8), None, ModifiersState::empty()));
        assert!(gui.workbench.config.focus_mode);

        assert!(!gui.workbench.config.typewriter_mode);
        assert!(gui.handle_key_input(&Key::Named(NamedKey::F9), None, ModifiersState::empty()));
        assert!(gui.workbench.config.typewriter_mode);

        let _ = gui.build_frame();
        let tw_rect = gui
            .ui_hit_zones
            .iter()
            .find(|z| matches!(z.action, UiAction::ToggleTypewriterMode))
            .map(|z| z.rect)
            .expect("expected TW typewriter pill in tab bar");
        assert!(gui.handle_pointer_down(tw_rect.x + 2.0, tw_rect.y + 2.0));
        assert!(!gui.workbench.config.typewriter_mode);

        assert!(!gui.workbench.config.fullscreen);
        assert!(gui.handle_key_input(&Key::Named(NamedKey::F11), None, ModifiersState::empty()));
        assert!(gui.workbench.config.fullscreen);
        if cfg!(target_os = "macos") {
            assert!(gui.handle_key_input(
                &Key::Character("f".into()),
                None,
                ModifiersState::SUPER | ModifiersState::CONTROL,
            ));
            assert!(!gui.workbench.config.fullscreen);
        }
    }

    #[test]
    fn test_find_bar_whole_word_toggle_and_replace_next_replace_all() {
        let mut wb = AppWorkbench::new(AppConfig::default(), None);
        wb.new_document();
        wb.active_tab_mut()
            .unwrap()
            .session
            .buffer
            .reset_content("foo foobar foo foo\n");
        let mut gui = GuiController::new(wb);
        let p_mod = primary_test_mod();

        // Cmd/Ctrl+H opens Find + Replace bar focused on Find when query is empty
        assert!(gui.handle_key_input(&Key::Character("h".into()), None, p_mod));
        assert!(gui.find_open);
        assert!(gui.replace_open);
        assert!(gui.find_focused);
        assert!(!gui.replace_focused);

        for ch in "foo".chars() {
            let s = ch.to_string();
            assert!(gui.handle_key_input(
                &Key::Character(s.as_str().into()),
                Some(&s),
                ModifiersState::empty(),
            ));
        }
        let _ = gui.build_frame();
        assert_eq!(gui.cached_meta.as_ref().unwrap().find_matches.len(), 4);

        // Toggle whole-word ("W") button: "foobar" must no longer match (4 -> 3 matches)
        let whole_word_rect = gui
            .ui_hit_zones
            .iter()
            .find(|z| matches!(z.action, UiAction::ToggleFindWholeWord))
            .map(|z| z.rect)
            .expect("expected W whole-word toggle button");
        assert!(gui.handle_pointer_down(whole_word_rect.x + 2.0, whole_word_rect.y + 2.0));
        assert!(gui.find_options.whole_word);
        let _ = gui.build_frame();
        assert_eq!(gui.cached_meta.as_ref().unwrap().find_matches.len(), 3);

        // Tab cycles focus from Find input to Replace input
        assert!(gui.handle_key_input(&Key::Named(NamedKey::Tab), None, ModifiersState::empty()));
        assert!(!gui.find_focused);
        assert!(gui.replace_focused);

        for ch in "baz".chars() {
            let s = ch.to_string();
            assert!(gui.handle_key_input(
                &Key::Character(s.as_str().into()),
                Some(&s),
                ModifiersState::empty(),
            ));
        }
        assert_eq!(gui.replace_text, "baz");

        // Enter in Replace input replaces current/next match
        assert!(gui.handle_key_input(&Key::Named(NamedKey::Enter), None, ModifiersState::empty()));
        assert_eq!(
            gui.workbench
                .active_tab()
                .unwrap()
                .session
                .buffer
                .to_string(),
            "baz foobar foo foo\n"
        );
        assert!(gui.active_find_range.is_some());

        // Clicking Replace All ("All") replaces remaining whole-word matches without touching "foobar"
        let _ = gui.build_frame();
        let all_rect = gui
            .ui_hit_zones
            .iter()
            .find(|z| matches!(z.action, UiAction::ReplaceAll))
            .map(|z| z.rect)
            .expect("expected Replace All button");
        assert!(gui.handle_pointer_down(all_rect.x + 2.0, all_rect.y + 2.0));
        assert_eq!(
            gui.workbench
                .active_tab()
                .unwrap()
                .session
                .buffer
                .to_string(),
            "baz foobar baz baz\n"
        );

        // IME commits route into ReplaceBox while Replace input is focused
        let rep_input = gui
            .ui_hit_zones
            .iter()
            .find(|z| matches!(z.action, UiAction::FocusReplaceInput))
            .map(|z| z.rect)
            .expect("expected Replace input hit zone");
        assert!(gui.handle_pointer_down(rep_input.x + 4.0, rep_input.y + 4.0));
        assert!(gui.replace_focused);
        gui.handle_ime_event(&Ime::Preedit("!".to_string(), Some((0, 1))));
        assert_eq!(gui.ime_target, ImeTarget::ReplaceBox);
        gui.handle_ime_event(&Ime::Commit("!".to_string()));
        assert_eq!(gui.replace_text, "baz!");

        // Escape closes Find & Replace; the platform shortcut reopens it.
        assert!(gui.handle_key_input(&Key::Named(NamedKey::Escape), None, ModifiersState::empty()));
        assert!(!gui.find_open);
        assert!(!gui.find_focused);
        assert!(!gui.replace_focused);

        if cfg!(target_os = "macos") {
            assert!(gui.handle_key_input_with_physical(
                &Key::Character("ƒ".into()),
                Some(PhysicalKey::Code(KeyCode::KeyF)),
                None,
                p_mod | ModifiersState::ALT,
            ));
        } else {
            assert!(gui.handle_key_input(&Key::Character("h".into()), None, p_mod));
        }
        assert!(gui.find_open);
        assert!(gui.replace_open);
    }

    #[test]
    fn test_command_palette_filter_export_html_custom_css_and_prose_width() {
        let dir = unique_test_dir("vellmark-gui-palette");
        fs::create_dir_all(&dir).unwrap();
        let doc_path = dir.join("note.md");
        let export_path = dir.join("out").join("note.html");
        let css_path = dir.join("custom.css");
        fs::write(&doc_path, "# Heading\n\nBody copy.\n").unwrap();
        fs::write(
            &css_path,
            ":root { --accent: #ff5500; }\nh1 { color: #123456; }\n",
        )
        .unwrap();

        let dialog = Arc::new(MemorySaveDialog::new(None));
        struct SharedDialog(Arc<MemorySaveDialog>);
        impl SaveDialogBackend for SharedDialog {
            fn pick_save_as_path(&self, cur: Option<&Path>, def: &str) -> Option<PathBuf> {
                self.0.pick_save_as_path(cur, def)
            }
            fn pick_open_path(&self, cur_dir: Option<&Path>) -> Option<PathBuf> {
                self.0.pick_open_path(cur_dir)
            }
            fn pick_open_folder(&self, cur_dir: Option<&Path>) -> Option<PathBuf> {
                self.0.pick_open_folder(cur_dir)
            }
        }

        let mut wb = AppWorkbench::new(AppConfig::default(), None);
        wb.open_document(&doc_path).unwrap();
        let mut gui = GuiController::with_backends(
            wb,
            Box::new(MemoryClipboard::default()),
            Box::new(SharedDialog(dialog.clone())),
        );
        let p_mod = primary_test_mod();

        // Cmd/Ctrl+Shift+P opens Command Palette (distinct from Cmd/Ctrl+P Quick Open)
        assert!(gui.handle_key_input(
            &Key::Character("p".into()),
            None,
            p_mod | ModifiersState::SHIFT,
        ));
        assert!(gui.command_palette_open);
        assert!(!gui.quick_open_open);

        for ch in "html safe".chars() {
            let s = ch.to_string();
            assert!(gui.handle_key_input(
                &Key::Character(s.as_str().into()),
                Some(&s),
                ModifiersState::empty(),
            ));
        }
        assert_eq!(gui.command_palette_matches.len(), 1);
        assert_eq!(
            gui.command_palette_matches[0].id,
            PaletteCommandId::ExportHtmlSafe
        );

        let _ = gui.build_frame();
        assert!(
            gui.ui_hit_zones
                .iter()
                .any(|z| matches!(z.action, UiAction::SelectCommandPaletteMatch(0)))
        );

        dialog.set_next_path(Some(export_path.clone()));
        assert!(gui.handle_key_input(&Key::Named(NamedKey::Enter), None, ModifiersState::empty()));
        assert!(!gui.command_palette_open);
        let exported = fs::read_to_string(&export_path).unwrap();
        assert!(exported.contains("<!DOCTYPE html"));
        assert!(exported.contains("Body copy."));

        // Load Custom CSS via Command Palette
        assert!(gui.handle_key_input(
            &Key::Character("p".into()),
            None,
            p_mod | ModifiersState::SHIFT,
        ));
        for ch in "custom css".chars() {
            let s = ch.to_string();
            assert!(gui.handle_key_input(
                &Key::Character(s.as_str().into()),
                Some(&s),
                ModifiersState::empty(),
            ));
        }
        dialog.set_next_open_path(Some(css_path.clone()));
        assert!(gui.handle_key_input(&Key::Named(NamedKey::Enter), None, ModifiersState::empty()));
        assert_eq!(gui.workbench.config.custom_theme_path, Some(css_path));
        assert_eq!(
            gui.workbench
                .theme
                .variables
                .get("--accent")
                .map(String::as_str),
            Some("#ff5500")
        );

        // Prose width preset (Wide 1080px) via Command Palette updates style.max_content_width_px
        assert!(gui.handle_key_input(
            &Key::Character("p".into()),
            None,
            p_mod | ModifiersState::SHIFT,
        ));
        for ch in "1080".chars() {
            let s = ch.to_string();
            assert!(gui.handle_key_input(
                &Key::Character(s.as_str().into()),
                Some(&s),
                ModifiersState::empty(),
            ));
        }
        assert!(gui.handle_key_input(&Key::Named(NamedKey::Enter), None, ModifiersState::empty()));
        assert_eq!(gui.workbench.config.max_content_width_px, 1080);
        assert!((gui.style.max_content_width_px - 1080.0).abs() < 0.01);

        // Switching back to built-in Dark theme clears custom_theme_path and custom CSS override
        assert!(gui.handle_key_input(
            &Key::Character("p".into()),
            None,
            p_mod | ModifiersState::SHIFT,
        ));
        for ch in "theme: dark".chars() {
            let s = ch.to_string();
            assert!(gui.handle_key_input(
                &Key::Character(s.as_str().into()),
                Some(&s),
                ModifiersState::empty(),
            ));
        }
        assert!(gui.handle_key_input(&Key::Named(NamedKey::Enter), None, ModifiersState::empty()));
        assert!(gui.workbench.config.custom_theme_path.is_none());
        assert_eq!(gui.workbench.config.theme, BuiltinTheme::Dark);
        let dark_html = gui
            .workbench
            .export_active_html(HtmlExportMode::Safe)
            .unwrap();
        assert!(!dark_html.contains("#ff5500"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_read_only_encoding_banner_status_label_and_save_utf8_copy() {
        let dir = unique_test_dir("vellmark-gui-readonly-enc");
        fs::create_dir_all(&dir).unwrap();
        let orig_path = dir.join("utf16.md");
        let copy_path = dir.join("utf16-utf8.md");

        let mut utf16_bytes = vec![0xFFu8, 0xFEu8];
        for unit in "# UTF16 Title\n\nBody text.\n".encode_utf16() {
            utf16_bytes.extend_from_slice(&unit.to_le_bytes());
        }
        fs::write(&orig_path, &utf16_bytes).unwrap();

        let dialog = Arc::new(MemorySaveDialog::new(None));
        struct SharedDialog(Arc<MemorySaveDialog>);
        impl SaveDialogBackend for SharedDialog {
            fn pick_save_as_path(&self, cur: Option<&Path>, def: &str) -> Option<PathBuf> {
                self.0.pick_save_as_path(cur, def)
            }
        }

        let mut wb = AppWorkbench::new(AppConfig::default(), None);
        wb.open_document(&orig_path).unwrap();
        let mut gui = GuiController::with_backends(
            wb,
            Box::new(MemoryClipboard::default()),
            Box::new(SharedDialog(dialog.clone())),
        );

        assert!(matches!(
            gui.active_doc_banner_kind(),
            Some(DocBannerKind::ReadOnlyEncoding {
                candidate: DetectedEncoding::Utf16LeBom
            })
        ));
        assert_eq!(
            encoding_label(DetectedEncoding::Utf16LeBom),
            "UTF-16 LE BOM"
        );

        // Replace All must fail gracefully without modifying ReadOnly buffer
        gui.find_query = "Body".to_string();
        gui.replace_text = "Edited".to_string();
        gui.run_replace_all();
        assert!(
            gui.workbench
                .active_tab()
                .unwrap()
                .session
                .buffer
                .to_string()
                .contains("Body text.")
        );
        assert!(
            gui.status_notice
                .as_ref()
                .is_some_and(|(msg, _)| msg.contains("Read-only"))
        );

        // Frame exposes Save UTF-8 Copy... button
        let _ = gui.build_frame();
        let copy_btn = gui
            .ui_hit_zones
            .iter()
            .find(|z| matches!(z.action, UiAction::SaveUtf8Copy))
            .map(|z| z.rect)
            .expect("expected Save UTF-8 Copy... banner button");

        // Refuses to overwrite original non-UTF-8 file in-place
        dialog.set_next_path(Some(orig_path.clone()));
        assert!(gui.handle_pointer_down(copy_btn.x + 2.0, copy_btn.y + 2.0));
        assert_eq!(fs::read(&orig_path).unwrap(), utf16_bytes);
        assert!(
            gui.status_notice
                .as_ref()
                .is_some_and(|(msg, _)| msg.contains("Choose a new path"))
        );

        // Refuses to overwrite a hard link aliasing the original non-UTF-8 inode
        let hardlink_path = dir.join("utf16-hardlink.md");
        fs::hard_link(&orig_path, &hardlink_path).unwrap();
        dialog.set_next_path(Some(hardlink_path.clone()));
        assert!(gui.handle_pointer_down(copy_btn.x + 2.0, copy_btn.y + 2.0));
        assert_eq!(fs::read(&orig_path).unwrap(), utf16_bytes);
        assert_eq!(fs::read(&hardlink_path).unwrap(), utf16_bytes);
        assert!(
            gui.status_notice
                .as_ref()
                .is_some_and(|(msg, _)| msg.contains("Choose a new path"))
        );

        // Refuses to overwrite another already-open tab's file (even if its tab holds stale ReadOnly state)
        let other_ro_path = dir.join("other-ro.md");
        let mut other_utf16 = vec![0xFFu8, 0xFEu8];
        for unit in "# Other ReadOnly\n".encode_utf16() {
            other_utf16.extend_from_slice(&unit.to_le_bytes());
        }
        fs::write(&other_ro_path, &other_utf16).unwrap();
        gui.workbench.open_document(&other_ro_path).unwrap();
        // Switch back to the first tab (`orig_path`) before triggering Save UTF-8 Copy...
        gui.workbench.active_tab_idx = Some(0);
        dialog.set_next_path(Some(other_ro_path.clone()));
        let _ = gui.build_frame();
        let copy_btn_again = gui
            .ui_hit_zones
            .iter()
            .find(|z| matches!(z.action, UiAction::SaveUtf8Copy))
            .map(|z| z.rect)
            .expect("expected Save UTF-8 Copy... banner button");
        assert!(gui.handle_pointer_down(copy_btn_again.x + 2.0, copy_btn_again.y + 2.0));
        assert_eq!(fs::read(&other_ro_path).unwrap(), other_utf16);
        assert_eq!(gui.workbench.active_tab_idx, Some(0));
        assert!(
            gui.status_notice
                .as_ref()
                .is_some_and(|(msg, _)| msg.contains("already open in tab"))
        );

        // Writing to a new path opens an editable UTF-8 tab while leaving original UTF-16LE file untouched
        dialog.set_next_path(Some(copy_path.clone()));
        assert!(gui.handle_pointer_down(copy_btn.x + 2.0, copy_btn.y + 2.0));
        assert_eq!(fs::read(&orig_path).unwrap(), utf16_bytes);
        assert_eq!(
            fs::read_to_string(&copy_path).unwrap(),
            "# UTF16 Title\n\nBody text.\n"
        );

        let active = gui.workbench.active_tab().unwrap();
        assert_eq!(
            active.session.format.decision,
            EncodingDecision::Editable(DetectedEncoding::Utf8)
        );
        assert_eq!(active.session.state, DocState::Clean);
        assert!(gui.active_doc_banner_kind().is_none());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_find_bar_regex_replace_normalizes_crlf_line_endings() {
        let dir = unique_test_dir("vellmark-gui-replace-crlf");
        fs::create_dir_all(&dir).unwrap();
        let doc_path = dir.join("crlf.md");
        fs::write(&doc_path, b"A\r\nB\r\n").unwrap();

        let mut wb = AppWorkbench::new(AppConfig::default(), None);
        wb.open_document(&doc_path).unwrap();
        let mut gui = GuiController::new(wb);
        let p_mod = primary_test_mod();

        assert!(gui.handle_key_input(&Key::Character("h".into()), None, p_mod));
        let _ = gui.build_frame();
        let regex_btn = gui
            .ui_hit_zones
            .iter()
            .find(|z| matches!(z.action, UiAction::ToggleFindRegex))
            .map(|z| z.rect)
            .expect("expected .* regex toggle button");
        assert!(gui.handle_pointer_down(regex_btn.x + 2.0, regex_btn.y + 2.0));
        assert!(gui.find_options.use_regex);

        gui.find_query = "A".to_string();
        gui.replace_text = r"X\nY".to_string();
        let _ = gui.build_frame();
        let rep_btn = gui
            .ui_hit_zones
            .iter()
            .find(|z| matches!(z.action, UiAction::ReplaceNext))
            .map(|z| z.rect)
            .expect("expected Rep button");
        assert!(gui.handle_pointer_down(rep_btn.x + 2.0, rep_btn.y + 2.0));

        assert_eq!(
            gui.workbench
                .active_tab()
                .unwrap()
                .session
                .buffer
                .to_string(),
            "X\r\nY\r\nB\r\n"
        );
        assert!(gui.handle_key_input(&Key::Character("s".into()), None, p_mod));
        assert_eq!(fs::read(&doc_path).unwrap(), b"X\r\nY\r\nB\r\n");

        // Mixed-EOL document: replacing `C\nD` via `$0` preserves its captured LF while
        // template-inserted `\n` still normalizes to CRLF.
        gui.workbench
            .active_tab_mut()
            .unwrap()
            .session
            .buffer
            .reset_content("A\r\nB\r\nC\nD\r\n");
        gui.find_query = r"C\nD".to_string();
        gui.replace_text = "$0".to_string();
        gui.run_replace_all();
        assert_eq!(
            gui.workbench
                .active_tab()
                .unwrap()
                .session
                .buffer
                .to_string(),
            "A\r\nB\r\nC\nD\r\n"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_config_persistence_custom_css_hot_reload_and_theme_diagnostics_modal() {
        let dir = unique_test_dir("vellmark-gui-css-hotreload");
        fs::create_dir_all(&dir).unwrap();
        let config_path = dir.join("config.toml");
        let recent_path = dir.join("recent-files.txt");
        let css_path = dir.join("theme.css");
        let doc_path = dir.join("note.md");

        fs::write(&doc_path, "# Hello\n").unwrap();
        fs::write(
            &css_path,
            ":root { --bg-color: #112233; }\n#write { background-color: var(--bg-color); }\nbody { position: fixed; }\n",
        )
        .unwrap();

        let dialog = Arc::new(MemorySaveDialog::new(None));
        struct SharedDialog(Arc<MemorySaveDialog>);
        impl SaveDialogBackend for SharedDialog {
            fn pick_save_as_path(&self, cur: Option<&Path>, def: &str) -> Option<PathBuf> {
                self.0.pick_save_as_path(cur, def)
            }
            fn pick_open_path(&self, cur: Option<&Path>) -> Option<PathBuf> {
                self.0.pick_open_path(cur)
            }
        }

        let mut wb = AppWorkbench::new(AppConfig::default(), None);
        wb.set_persistence_paths(Some(config_path.clone()), Some(recent_path.clone()));
        wb.open_document(&doc_path).unwrap();

        let mut gui = GuiController::with_backends(
            wb,
            Box::new(MemoryClipboard::default()),
            Box::new(SharedDialog(dialog.clone())),
        );
        let p_mod = primary_test_mod();

        // Recent file open is persisted to disk automatically
        let loaded_recent =
            vellmark_core::RecentFilesList::load_from_file(&recent_path, 10).unwrap();
        assert_eq!(loaded_recent.entries.len(), 1);
        assert!(paths_share_underlying_file(
            &loaded_recent.entries[0],
            &doc_path
        ));

        // Load custom CSS via Command Palette -> persists config + updates style + records diagnostics
        dialog.set_next_open_path(Some(css_path.clone()));
        gui.execute_palette_command(PaletteCommandId::LoadCustomThemeCss);
        assert_eq!(
            gui.workbench
                .theme
                .variables
                .get("--bg-color")
                .map(String::as_str),
            Some("#112233")
        );
        assert!(!gui.workbench.theme.diagnostics.is_empty());
        let saved_cfg = AppConfig::load_from_file(&config_path).unwrap();
        assert_eq!(
            saved_cfg.custom_theme_path.as_deref(),
            Some(css_path.as_path())
        );
        assert!(!saved_cfg.follow_system_theme);

        // Zoom shortcut updates style & persists zoom_percent
        assert!(gui.handle_key_input(&Key::Character("+".into()), None, p_mod));
        let saved_cfg_zoom = AppConfig::load_from_file(&config_path).unwrap();
        assert_eq!(saved_cfg_zoom.zoom_percent, 110);

        // Hot-reload custom CSS on disk via tick_runtime_at (even with same byte length!)
        fs::write(
            &css_path,
            ":root { --bg-color: #445566; }\n#write { background-color: var(--bg-color); }\n.card { animation: spin; }\n",
        )
        .unwrap();
        let bg_before = gui.style.bg_color;
        assert_eq!(bg_before, vellmark_render::RgbaColor::rgb(0x11, 0x22, 0x33));
        assert!(
            gui.tick_runtime_at(
                Instant::now() + EXTERNAL_POLL_INTERVAL + Duration::from_millis(50)
            )
        );
        assert_eq!(
            gui.workbench
                .theme
                .variables
                .get("--bg-color")
                .map(String::as_str),
            Some("#445566")
        );
        assert_eq!(
            gui.style.bg_color,
            vellmark_render::RgbaColor::rgb(0x44, 0x55, 0x66)
        );
        assert_ne!(gui.style.bg_color, bg_before);
        assert!(
            gui.status_notice
                .as_ref()
                .is_some_and(|(m, _)| m.contains("Hot-reloaded theme CSS"))
        );

        // Open Theme Diagnostics modal via Command Palette, filter rules, select row, and click Reload
        gui.execute_palette_command(PaletteCommandId::ShowThemeDiagnostics);
        assert_eq!(gui.inspect_modal, Some(InspectModalKind::ThemeDiagnostics));
        let _ = gui.build_frame();
        assert!(
            gui.ui_hit_zones
                .iter()
                .any(|z| matches!(z.action, UiAction::ReloadCustomTheme))
        );
        assert!(gui.handle_key_input(
            &Key::Character("card".into()),
            Some("card"),
            ModifiersState::empty(),
        ));
        assert_eq!(gui.filtered_theme_diagnostic_indices().len(), 1);
        assert!(gui.handle_key_input(&Key::Named(NamedKey::Enter), None, ModifiersState::empty()));
        assert!(
            gui.status_notice
                .as_ref()
                .is_some_and(|(m, _)| m.contains("CSS ["))
        );

        // Manual Reload button updates diagnostics and style when CSS is cleaned up
        fs::write(
            &css_path,
            ":root { --bg-color: #778899; }\n#write { background-color: var(--bg-color); }\n",
        )
        .unwrap();
        let _ = gui.build_frame();
        let reload_btn = gui
            .ui_hit_zones
            .iter()
            .find(|z| matches!(z.action, UiAction::ReloadCustomTheme))
            .map(|z| z.rect)
            .expect("expected Reload button in Theme Diagnostics modal");
        assert!(gui.handle_pointer_down(reload_btn.x + 2.0, reload_btn.y + 2.0));
        assert_eq!(
            gui.workbench
                .theme
                .variables
                .get("--bg-color")
                .map(String::as_str),
            Some("#778899")
        );
        assert_eq!(
            gui.style.bg_color,
            vellmark_render::RgbaColor::rgb(0x77, 0x88, 0x99)
        );
        assert!(gui.workbench.theme.diagnostics.is_empty());

        assert!(gui.handle_key_input(&Key::Named(NamedKey::Escape), None, ModifiersState::empty()));
        assert!(gui.inspect_modal.is_none());

        // Toggle Network Images via Command Palette persists to config.toml
        assert!(gui.workbench.config.allow_network_images);
        gui.execute_palette_command(PaletteCommandId::ToggleNetworkImages);
        assert!(!gui.workbench.config.allow_network_images);
        assert!(
            !AppConfig::load_from_file(&config_path)
                .unwrap()
                .allow_network_images
        );
        gui.execute_palette_command(PaletteCommandId::ToggleNetworkImages);
        assert!(gui.workbench.config.allow_network_images);
        assert!(
            AppConfig::load_from_file(&config_path)
                .unwrap()
                .allow_network_images
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_recovery_backups_modal_restore_draft_and_delete() {
        let dir = unique_test_dir("vellmark-gui-recovery-modal");
        let rec_root = dir.join("recovery");
        fs::create_dir_all(&rec_root).unwrap();

        let doc_path = dir.join("chapter.md");
        fs::write(&doc_path, "base version\n").unwrap();

        let mut wb = AppWorkbench::new(AppConfig::default(), Some(rec_root.clone()));
        wb.open_document(&doc_path).unwrap();
        wb.active_tab_mut()
            .unwrap()
            .edit(|ctx| {
                ctx.set_selection(0, ctx.buffer.len_bytes());
                ctx.insert_text("local conflict edits\n");
            })
            .unwrap();
        // Create external modification with mixed EOL (`A\r\nB\nC\r\n`) & keep mine -> conflict safety backup #1
        fs::write(&doc_path, "A\r\nB\nC\r\n").unwrap();
        let _ = wb.active_tab_mut().unwrap().sync_from_disk(true);
        assert_eq!(wb.active_tab().unwrap().session.state, DocState::Conflict);
        let outcome = wb.active_tab_mut().unwrap().resolve_conflict_keep_mine();
        assert!(matches!(outcome, ConflictActionOutcome::ResolvedClean));

        // Create a second backup to test both plain `Del` and `Cmd/Ctrl + Backspace`
        wb.active_tab_mut()
            .unwrap()
            .edit(|ctx| {
                ctx.set_selection(0, ctx.buffer.len_bytes());
                ctx.insert_text("local conflict edits v2\n");
            })
            .unwrap();
        fs::write(&doc_path, "second external disk version\n").unwrap();
        let _ = wb.active_tab_mut().unwrap().sync_from_disk(true);
        assert_eq!(wb.active_tab().unwrap().session.state, DocState::Conflict);
        let outcome2 = wb.active_tab_mut().unwrap().resolve_conflict_keep_mine();
        assert!(matches!(outcome2, ConflictActionOutcome::ResolvedClean));

        let backups_before = wb.list_history_backups();
        assert_eq!(backups_before.len(), 2);

        let mut gui = GuiController::new(wb);
        let p_mod = primary_test_mod();

        // Open Recovery & Safety Backups modal via Command Palette
        gui.execute_palette_command(PaletteCommandId::ManageRecoveryBackups);
        assert_eq!(gui.inspect_modal, Some(InspectModalKind::RecoveryBackups));
        let _ = gui.build_frame();
        assert!(
            gui.ui_hit_zones
                .iter()
                .any(|z| matches!(z.action, UiAction::DeleteSelectedRecoveryBackup))
        );

        // Select the mixed-EOL backup (may be at index 0 or 1 depending on sort order)
        let mixed_slot = gui
            .inspect_recovery_backups
            .iter()
            .position(|e| e.payload_len == "A\r\nB\nC\r\n".len())
            .expect("expected mixed-EOL backup entry");
        gui.inspect_modal_selected = mixed_slot;

        // Pressing Enter restores backup verbatim without rewriting mixed line endings or altering chapter.md
        let tab_count_before = gui.workbench.tabs.len();
        assert!(gui.handle_key_input(&Key::Named(NamedKey::Enter), None, ModifiersState::empty()));
        assert!(gui.inspect_modal.is_none());
        assert_eq!(gui.workbench.tabs.len(), tab_count_before + 1);
        let restored_tab = gui.workbench.active_tab().unwrap();
        assert!(restored_tab.session.path.is_none());
        assert_eq!(restored_tab.session.state, DocState::Dirty);
        assert_eq!(
            restored_tab.session.format.dominant_line_ending,
            LineEnding::CrLf
        );
        assert_eq!(restored_tab.session.buffer.to_string(), "A\r\nB\nC\r\n");
        assert_eq!(
            fs::read_to_string(&doc_path).unwrap(),
            "local conflict edits v2\n"
        );

        // Re-open modal: delete first backup via plain `Delete` (no modifier), then second via `Cmd/Ctrl + Backspace`
        gui.execute_palette_command(PaletteCommandId::ManageRecoveryBackups);
        assert_eq!(gui.inspect_recovery_backups.len(), 2);
        assert!(gui.handle_key_input(&Key::Named(NamedKey::Delete), None, ModifiersState::empty()));
        assert_eq!(gui.inspect_recovery_backups.len(), 1);
        assert_eq!(gui.workbench.list_history_backups().len(), 1);
        assert!(
            gui.status_notice
                .as_ref()
                .is_some_and(|(m, _)| m.contains("Deleted recovery backup"))
        );

        assert!(gui.handle_key_input(&Key::Named(NamedKey::Backspace), None, p_mod));
        assert!(gui.inspect_recovery_backups.is_empty());
        assert!(gui.workbench.list_history_backups().is_empty());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_multi_instance_exit_and_open_preserve_newer_config_and_recent_files() {
        let dir = unique_test_dir("vellmark-gui-multi-instance");
        fs::create_dir_all(&dir).unwrap();
        let config_path = dir.join("config.toml");
        let recent_path = dir.join("recent-files.txt");
        let file_a = dir.join("a.md");
        let file_b = dir.join("b.md");
        let file_c = dir.join("c.md");
        fs::write(&file_a, "# A\n").unwrap();
        fs::write(&file_b, "# B\n").unwrap();
        fs::write(&file_c, "# C\n").unwrap();

        // Instance A hides sidebar, then opens a folder from recent files (forcing sidebar_visible=true
        // and persisting immediately), BEFORE Instance B switches theme to Dark.
        let folder_a = dir.join("folder_a");
        fs::create_dir_all(&folder_a).unwrap();
        let mut wb_a = AppWorkbench::new(AppConfig::default(), None);
        wb_a.set_persistence_paths(Some(config_path.clone()), Some(recent_path.clone()));
        wb_a.open_document(&file_a).unwrap();
        let mut gui_a = GuiController::new(wb_a);
        assert!(gui_a.execute_ui_action(UiAction::ToggleSidebar));
        assert!(!gui_a.workbench.config.sidebar_visible);
        assert!(gui_a.execute_ui_action(UiAction::OpenRecentFile(folder_a.clone())));
        assert!(gui_a.workbench.config.sidebar_visible);
        assert!(
            AppConfig::load_from_file(&config_path)
                .unwrap()
                .sidebar_visible
        );

        // Instance B starts with same persisted files, changes theme + zoom and opens b.md
        let mut wb_b = AppWorkbench::new(AppConfig::default(), None);
        wb_b.set_persistence_paths(Some(config_path.clone()), Some(recent_path.clone()));
        wb_b.open_document(&file_b).unwrap();
        let mut gui_b = GuiController::new(wb_b);
        gui_b.execute_palette_command(PaletteCommandId::ThemeDark);
        gui_b.execute_palette_command(PaletteCommandId::ZoomIn);

        // Older Instance A exits: must NOT overwrite B's config or B's recent-files entry
        assert!(gui_a.handle_close_requested());
        let saved_cfg = AppConfig::load_from_file(&config_path).unwrap();
        assert_eq!(saved_cfg.theme, BuiltinTheme::Dark);
        assert!(!saved_cfg.follow_system_theme);
        assert_eq!(saved_cfg.zoom_percent, 110);

        let recent_after_a_exit =
            vellmark_core::RecentFilesList::load_from_file(&recent_path, 10).unwrap();
        assert_eq!(recent_after_a_exit.entries.len(), 3);
        assert!(paths_share_underlying_file(
            &recent_after_a_exit.entries[0],
            &file_b
        ));
        assert!(paths_share_underlying_file(
            &recent_after_a_exit.entries[1],
            &folder_a
        ));
        assert!(paths_share_underlying_file(
            &recent_after_a_exit.entries[2],
            &file_a
        ));

        // Instance C (started before B opened b.md) opens c.md: must merge with disk rather than wiping b.md
        let mut wb_c = AppWorkbench::new(AppConfig::default(), None);
        wb_c.set_persistence_paths(Some(config_path.clone()), Some(recent_path.clone()));
        wb_c.open_document(&file_c).unwrap();
        let recent_after_c_open =
            vellmark_core::RecentFilesList::load_from_file(&recent_path, 10).unwrap();
        assert_eq!(recent_after_c_open.entries.len(), 4);
        assert!(paths_share_underlying_file(
            &recent_after_c_open.entries[0],
            &file_c
        ));
        assert!(paths_share_underlying_file(
            &recent_after_c_open.entries[1],
            &file_b
        ));
        assert!(paths_share_underlying_file(
            &recent_after_c_open.entries[2],
            &folder_a
        ));
        assert!(paths_share_underlying_file(
            &recent_after_c_open.entries[3],
            &file_a
        ));

        // Issue 1 repro: Instance D (started earlier with Light theme) modifies ONLY zoom & focus mode
        // AFTER Instance B saved Dark theme — D's save must merge fields and preserve B's Dark theme!
        let mut wb_d = AppWorkbench::new(AppConfig::default(), None);
        wb_d.set_persistence_paths(Some(config_path.clone()), Some(recent_path.clone()));
        let mut gui_d = GuiController::new(wb_d);
        gui_d.execute_palette_command(PaletteCommandId::ZoomIn);
        gui_d.execute_palette_command(PaletteCommandId::ZoomIn);
        gui_d.execute_palette_command(PaletteCommandId::ToggleFocusMode);
        assert!(gui_d.handle_close_requested());

        let merged_cfg = AppConfig::load_from_file(&config_path).unwrap();
        assert_eq!(merged_cfg.theme, BuiltinTheme::Dark);
        assert!(!merged_cfg.follow_system_theme);
        assert_eq!(merged_cfg.zoom_percent, 120);
        assert!(merged_cfg.focus_mode);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_close_warns_when_pending_persistence_is_locked_and_allows_retry_or_second_close() {
        let dir = unique_test_dir("vellmark-gui-close-locked-persist");
        fs::create_dir_all(&dir).unwrap();
        let config_path = dir.join("config.toml");
        let recent_path = dir.join("recent-files.txt");

        let cfg_lock = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(dir.join(".config.toml.vellmark.lock"))
            .unwrap();
        cfg_lock.lock().unwrap();

        let mut wb = AppWorkbench::new(AppConfig::default(), None);
        wb.set_persistence_paths(Some(config_path.clone()), Some(recent_path.clone()));
        let mut gui = GuiController::new(wb);

        // User changes Focus Mode while peer holds config lock -> deferred, not yet on disk.
        gui.execute_palette_command(PaletteCommandId::ToggleFocusMode);
        assert!(gui.workbench.has_pending_persistence());
        assert!(!config_path.exists());

        // First close attempt warns instead of silently losing settings.
        assert!(!gui.handle_close_requested());
        assert!(gui.close_confirm_armed);
        assert!(
            gui.status_notice
                .as_ref()
                .is_some_and(|(m, _)| m.contains("locked"))
        );

        // Peer releases the lock before second close -> retry succeeds and persists before exit.
        let _ = cfg_lock.unlock();
        drop(cfg_lock);
        assert!(gui.handle_close_requested());
        assert!(!gui.workbench.has_pending_persistence());
        assert!(AppConfig::load_from_file(&config_path).unwrap().focus_mode);

        // Combined case: unsaved untitled draft AND locked pending settings -> first close notice
        // must warn about BOTH the unsaved draft and the locked settings before second close exits.
        let cfg_lock2 = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(dir.join(".config.toml.vellmark.lock"))
            .unwrap();
        cfg_lock2.lock().unwrap();
        let mut wb2 = AppWorkbench::new(AppConfig::default(), None);
        wb2.set_persistence_paths(Some(config_path.clone()), Some(recent_path));
        wb2.new_document();
        wb2.active_tab_mut()
            .unwrap()
            .edit(|ctx| ctx.insert_text("unsaved draft text"))
            .unwrap();
        let mut gui2 = GuiController::new(wb2);
        gui2.execute_palette_command(PaletteCommandId::ZoomIn);
        assert!(!gui2.handle_close_requested());
        assert!(
            gui2.status_notice
                .as_ref()
                .is_some_and(|(m, _)| m.contains("Unsaved changes") && m.contains("locked"))
        );
        assert!(gui2.handle_close_requested());
        let _ = cfg_lock2.unlock();
        drop(cfg_lock2);

        let _ = fs::remove_dir_all(&dir);
    }
}
