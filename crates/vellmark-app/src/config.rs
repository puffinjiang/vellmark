//! Persistent application configuration and view preferences (PRD 4.7, 4.8, 5.2, 6.3).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, TryLockError};
use std::time::{Duration, Instant};
use vellmark_theme::BuiltinTheme;

static PERSISTENCE_IO_LOCKS: OnceLock<Mutex<HashMap<PathBuf, Arc<Mutex<()>>>>> = OnceLock::new();

/// Short lock wait budget on the GUI main thread so normal contention (including a couple of
/// `fsync` round-trips) finishes synchronously while a hung peer instance never freezes the window.
pub const DEFAULT_PERSISTENCE_LOCK_TIMEOUT: Duration = Duration::from_millis(80);

/// Slightly longer bounded lock wait budget when flushing pending settings/MRU on window exit.
pub const CLOSE_PERSISTENCE_LOCK_TIMEOUT: Duration = Duration::from_millis(250);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UiLanguage {
    ZhCn,
    EnUs,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppConfig {
    pub theme: BuiltinTheme,
    pub follow_system_theme: bool,
    pub custom_theme_path: Option<PathBuf>,
    pub font_family: String,
    pub code_font_family: String,
    pub font_size_px: f32,
    pub line_height: f32,
    pub max_content_width_px: u32,
    pub zoom_percent: u16,
    pub focus_mode: bool,
    pub typewriter_mode: bool,
    pub fullscreen: bool,
    pub sidebar_visible: bool,
    pub auto_save_enabled: bool,
    pub auto_save_delay_ms: u64,
    pub table_auto_align: bool,
    pub show_code_line_numbers: bool,
    pub image_assets_dir: String,
    pub allow_network_images: bool,
    pub language: UiLanguage,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            theme: BuiltinTheme::Light,
            follow_system_theme: true,
            custom_theme_path: None,
            font_family: "-apple-system, BlinkMacSystemFont, \"PingFang SC\", \"Microsoft YaHei\", sans-serif".to_string(),
            code_font_family: "\"JetBrains Mono\", \"SF Mono\", Menlo, Consolas, monospace".to_string(),
            font_size_px: 16.0,
            line_height: 1.75,
            max_content_width_px: 860,
            zoom_percent: 100,
            focus_mode: false,
            typewriter_mode: false,
            fullscreen: false,
            sidebar_visible: true,
            auto_save_enabled: true,
            auto_save_delay_ms: 2000, // PRD 4.5.1 A: 2s pause auto-save
            table_auto_align: true,   // PRD 4.3 T-03: enabled by default
            show_code_line_numbers: true,
            image_assets_dir: "./assets".to_string(), // PRD 4.4 I-02
            allow_network_images: true,
            language: UiLanguage::ZhCn,
        }
    }
}

impl AppConfig {
    /// Resolve active built-in theme honoring `follow_system_theme` (PRD 4.8 S-02).
    pub fn effective_builtin_theme(&self, system_prefers_dark: bool) -> BuiltinTheme {
        if self.follow_system_theme {
            BuiltinTheme::for_system_color_scheme(system_prefers_dark)
        } else {
            self.theme
        }
    }

    pub fn zoom_in(&mut self) {
        self.zoom_percent = (self.zoom_percent + 10).min(300);
    }

    pub fn zoom_out(&mut self) {
        self.zoom_percent = self.zoom_percent.saturating_sub(10).max(50);
    }

    pub fn reset_zoom(&mut self) {
        self.zoom_percent = 100;
    }

    pub fn effective_line_height_px(&self) -> f32 {
        self.font_size_px * self.line_height * (f32::from(self.zoom_percent) / 100.0)
    }

    pub fn to_toml_string(&self) -> String {
        let theme_name = match self.theme {
            BuiltinTheme::Light => "light",
            BuiltinTheme::Dark => "dark",
            BuiltinTheme::Academic => "academic",
            BuiltinTheme::GitHub => "github",
        };
        let lang_name = match self.language {
            UiLanguage::ZhCn => "zh-CN",
            UiLanguage::EnUs => "en-US",
        };
        let custom_theme = self
            .custom_theme_path
            .as_ref()
            .map(|p| format!("\"{}\"", escape_toml_str(&p.to_string_lossy())))
            .unwrap_or_else(|| "\"\"".to_string());

        format!(
            "theme = \"{theme_name}\"\n\
follow_system_theme = {}\n\
custom_theme_path = {custom_theme}\n\
font_family = \"{}\"\n\
code_font_family = \"{}\"\n\
font_size_px = {:.2}\n\
line_height = {:.2}\n\
max_content_width_px = {}\n\
zoom_percent = {}\n\
focus_mode = {}\n\
typewriter_mode = {}\n\
fullscreen = {}\n\
sidebar_visible = {}\n\
auto_save_enabled = {}\n\
auto_save_delay_ms = {}\n\
table_auto_align = {}\n\
show_code_line_numbers = {}\n\
image_assets_dir = \"{}\"\n\
allow_network_images = {}\n\
language = \"{lang_name}\"\n",
            self.follow_system_theme,
            escape_toml_str(&self.font_family),
            escape_toml_str(&self.code_font_family),
            self.font_size_px,
            self.line_height,
            self.max_content_width_px,
            self.zoom_percent,
            self.focus_mode,
            self.typewriter_mode,
            self.fullscreen,
            self.sidebar_visible,
            self.auto_save_enabled,
            self.auto_save_delay_ms,
            self.table_auto_align,
            self.show_code_line_numbers,
            escape_toml_str(&self.image_assets_dir),
            self.allow_network_images,
        )
    }

    pub fn from_toml_str(src: &str) -> Self {
        let mut cfg = Self::default();
        for raw_line in src.lines() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((k, v)) = line.split_once('=') else {
                continue;
            };
            let k = k.trim();
            let v = v.trim();
            match k {
                "theme" => {
                    cfg.theme = match unquote_toml_str(v).as_str() {
                        "dark" => BuiltinTheme::Dark,
                        "academic" => BuiltinTheme::Academic,
                        "github" => BuiltinTheme::GitHub,
                        _ => BuiltinTheme::Light,
                    };
                }
                "follow_system_theme" => {
                    if let Ok(b) = v.parse() {
                        cfg.follow_system_theme = b;
                    }
                }
                "custom_theme_path" => {
                    let s = unquote_toml_str(v);
                    cfg.custom_theme_path = if s.is_empty() {
                        None
                    } else {
                        Some(PathBuf::from(s))
                    };
                }
                "font_family" => cfg.font_family = unquote_toml_str(v),
                "code_font_family" => cfg.code_font_family = unquote_toml_str(v),
                "font_size_px" => {
                    if let Ok(n) = v.parse::<f32>() {
                        cfg.font_size_px = n.clamp(8.0, 72.0);
                    }
                }
                "line_height" => {
                    if let Ok(n) = v.parse::<f32>() {
                        cfg.line_height = n.clamp(1.0, 3.0);
                    }
                }
                "max_content_width_px" => {
                    if let Ok(n) = v.parse::<u32>() {
                        cfg.max_content_width_px = n.clamp(400, 2400);
                    }
                }
                "zoom_percent" => {
                    if let Ok(n) = v.parse::<u16>() {
                        cfg.zoom_percent = n.clamp(50, 300);
                    }
                }
                "focus_mode" => {
                    if let Ok(b) = v.parse() {
                        cfg.focus_mode = b;
                    }
                }
                "typewriter_mode" => {
                    if let Ok(b) = v.parse() {
                        cfg.typewriter_mode = b;
                    }
                }
                "fullscreen" => {
                    if let Ok(b) = v.parse() {
                        cfg.fullscreen = b;
                    }
                }
                "sidebar_visible" => {
                    if let Ok(b) = v.parse() {
                        cfg.sidebar_visible = b;
                    }
                }
                "auto_save_enabled" => {
                    if let Ok(b) = v.parse() {
                        cfg.auto_save_enabled = b;
                    }
                }
                "auto_save_delay_ms" => {
                    if let Ok(n) = v.parse::<u64>() {
                        cfg.auto_save_delay_ms = n.clamp(200, 60_000);
                    }
                }
                "table_auto_align" => {
                    if let Ok(b) = v.parse() {
                        cfg.table_auto_align = b;
                    }
                }
                "show_code_line_numbers" => {
                    if let Ok(b) = v.parse() {
                        cfg.show_code_line_numbers = b;
                    }
                }
                "image_assets_dir" => {
                    let s = unquote_toml_str(v);
                    if !s.is_empty() {
                        cfg.image_assets_dir = s;
                    }
                }
                "allow_network_images" => {
                    if let Ok(b) = v.parse() {
                        cfg.allow_network_images = b;
                    }
                }
                "language" => {
                    cfg.language = match unquote_toml_str(v).as_str() {
                        "en-US" => UiLanguage::EnUs,
                        _ => UiLanguage::ZhCn,
                    };
                }
                _ => {}
            }
        }
        cfg
    }

    pub fn save_to_file(&self, path: &Path) -> io::Result<()> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        vellmark_core::atomic_write_file(path, self.to_toml_string().as_bytes())
    }

    pub fn load_from_file(path: &Path) -> io::Result<Self> {
        let raw = fs::read_to_string(path)?;
        Ok(Self::from_toml_str(&raw))
    }

    /// Overlay only the fields modified in `self` relative to `baseline` onto `disk`,
    /// preserving any unrelated settings updated on disk by another running instance.
    pub fn merge_modified_fields(&self, baseline: &Self, disk: &Self) -> Self {
        let mut out = disk.clone();
        if self.theme != baseline.theme
            || self.follow_system_theme != baseline.follow_system_theme
            || self.custom_theme_path != baseline.custom_theme_path
        {
            out.theme = self.theme;
            out.follow_system_theme = self.follow_system_theme;
            out.custom_theme_path = self.custom_theme_path.clone();
        }
        if self.font_family != baseline.font_family {
            out.font_family = self.font_family.clone();
        }
        if self.code_font_family != baseline.code_font_family {
            out.code_font_family = self.code_font_family.clone();
        }
        if (self.font_size_px - baseline.font_size_px).abs() > f32::EPSILON {
            out.font_size_px = self.font_size_px;
        }
        if (self.line_height - baseline.line_height).abs() > f32::EPSILON {
            out.line_height = self.line_height;
        }
        if self.max_content_width_px != baseline.max_content_width_px {
            out.max_content_width_px = self.max_content_width_px;
        }
        if self.zoom_percent != baseline.zoom_percent {
            out.zoom_percent = self.zoom_percent;
        }
        if self.focus_mode != baseline.focus_mode {
            out.focus_mode = self.focus_mode;
        }
        if self.typewriter_mode != baseline.typewriter_mode {
            out.typewriter_mode = self.typewriter_mode;
        }
        if self.fullscreen != baseline.fullscreen {
            out.fullscreen = self.fullscreen;
        }
        if self.sidebar_visible != baseline.sidebar_visible {
            out.sidebar_visible = self.sidebar_visible;
        }
        if self.auto_save_enabled != baseline.auto_save_enabled {
            out.auto_save_enabled = self.auto_save_enabled;
        }
        if self.auto_save_delay_ms != baseline.auto_save_delay_ms {
            out.auto_save_delay_ms = self.auto_save_delay_ms;
        }
        if self.table_auto_align != baseline.table_auto_align {
            out.table_auto_align = self.table_auto_align;
        }
        if self.show_code_line_numbers != baseline.show_code_line_numbers {
            out.show_code_line_numbers = self.show_code_line_numbers;
        }
        if self.image_assets_dir != baseline.image_assets_dir {
            out.image_assets_dir = self.image_assets_dir.clone();
        }
        if self.allow_network_images != baseline.allow_network_images {
            out.allow_network_images = self.allow_network_images;
        }
        if self.language != baseline.language {
            out.language = self.language;
        }
        out
    }
}

/// Serialize read-modify-write persistence operations across both threads and processes using
/// an in-process mutex plus an OS-managed exclusive advisory file lock (`std::fs::File::try_lock`)
/// with a bounded timeout.
///
/// The lock file is never unlinked (preventing unlink-and-recreate races), the OS kernel
/// releases the lock automatically if a holder crashes, and `f` is never invoked if the lock
/// cannot be acquired within `timeout` (returns `ErrorKind::WouldBlock`).
pub fn with_persistence_file_lock_timeout<T>(
    target: &Path,
    timeout: Duration,
    f: impl FnOnce() -> io::Result<T>,
) -> io::Result<T> {
    let deadline = Instant::now() + timeout;
    let parent = match target.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    fs::create_dir_all(parent)?;
    let name = target
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("vellmark");
    let lock_path = parent.join(format!(".{name}.vellmark.lock"));
    let lock_key = fs::canonicalize(parent)
        .unwrap_or_else(|_| parent.to_path_buf())
        .join(format!(".{name}.vellmark.lock"));

    // Per-file in-process mutex so threads targeting different files never block one another,
    // and threads targeting the same file share the same overall `deadline`.
    let path_mutex = {
        let map_lock = PERSISTENCE_IO_LOCKS.get_or_init(|| Mutex::new(HashMap::new()));
        let mut map = map_lock.lock().unwrap_or_else(|e| e.into_inner());
        Arc::clone(
            map.entry(lock_key)
                .or_insert_with(|| Arc::new(Mutex::new(()))),
        )
    };
    let _in_proc = loop {
        match path_mutex.try_lock() {
            Ok(guard) => break guard,
            Err(TryLockError::Poisoned(p)) => break p.into_inner(),
            Err(TryLockError::WouldBlock) => {
                let now = Instant::now();
                if now >= deadline {
                    return Err(io::Error::new(
                        io::ErrorKind::WouldBlock,
                        "persistence file lock timed out",
                    ));
                }
                std::thread::sleep(Duration::from_millis(2).min(deadline - now));
            }
        }
    };

    let lock_file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)?;

    loop {
        match lock_file.try_lock() {
            Ok(()) => break,
            Err(fs::TryLockError::WouldBlock) => {
                let now = Instant::now();
                if now >= deadline {
                    return Err(io::Error::new(
                        io::ErrorKind::WouldBlock,
                        "persistence file lock timed out",
                    ));
                }
                std::thread::sleep(Duration::from_millis(2).min(deadline - now));
            }
            Err(fs::TryLockError::Error(err)) => return Err(err),
        }
    }

    let res = f();
    let _ = lock_file.unlock();
    res
}

/// Convenience wrapper around [`with_persistence_file_lock_timeout`] using
/// [`DEFAULT_PERSISTENCE_LOCK_TIMEOUT`].
pub fn with_persistence_file_lock<T>(
    target: &Path,
    f: impl FnOnce() -> io::Result<T>,
) -> io::Result<T> {
    with_persistence_file_lock_timeout(target, DEFAULT_PERSISTENCE_LOCK_TIMEOUT, f)
}

/// Base persistent application-data directory across desktop platforms.
pub fn default_app_data_root() -> PathBuf {
    if let Some(dir) = std::env::var_os("VELLMARK_DATA_DIR").filter(|v| !v.is_empty()) {
        return PathBuf::from(dir);
    }
    #[cfg(target_os = "macos")]
    if let Some(home) = std::env::var_os("HOME").filter(|v| !v.is_empty()) {
        return PathBuf::from(home)
            .join("Library")
            .join("Application Support")
            .join("vellmark");
    }
    #[cfg(target_os = "windows")]
    if let Some(appdata) = std::env::var_os("APPDATA").filter(|v| !v.is_empty()) {
        return PathBuf::from(appdata).join("vellmark");
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        if let Some(xdg) = std::env::var_os("XDG_DATA_HOME").filter(|v| !v.is_empty()) {
            return PathBuf::from(xdg).join("vellmark");
        }
        if let Some(home) = std::env::var_os("HOME").filter(|v| !v.is_empty()) {
            return PathBuf::from(home)
                .join(".local")
                .join("share")
                .join("vellmark");
        }
    }
    std::env::temp_dir().join("vellmark")
}

/// Default persistent directory storing crash-recovery snapshots across GUI restarts.
pub fn default_recovery_root() -> PathBuf {
    if let Some(dir) = std::env::var_os("VELLMARK_RECOVERY_DIR").filter(|v| !v.is_empty()) {
        return PathBuf::from(dir);
    }
    default_app_data_root().join("recovery")
}

/// Default path for persistent user settings (`config.toml`).
pub fn default_config_path() -> PathBuf {
    if let Some(p) = std::env::var_os("VELLMARK_CONFIG_PATH").filter(|v| !v.is_empty()) {
        return PathBuf::from(p);
    }
    default_app_data_root().join("config.toml")
}

/// Default path for persistent recent-files MRU history (`recent-files.txt`).
pub fn default_recent_files_path() -> PathBuf {
    default_config_path()
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(|dir| dir.join("recent-files.txt"))
        .unwrap_or_else(|| default_app_data_root().join("recent-files.txt"))
}

fn escape_toml_str(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn unquote_toml_str(s: &str) -> String {
    let trimmed = s.trim();
    if let Some(inner) = trimmed.strip_prefix('"').and_then(|x| x.strip_suffix('"')) {
        let mut out = String::with_capacity(inner.len());
        let mut chars = inner.chars();
        while let Some(c) = chars.next() {
            if c == '\\' {
                if let Some(next) = chars.next() {
                    out.push(next);
                }
            } else {
                out.push(c);
            }
        }
        out
    } else {
        trimmed.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_roundtrip_and_system_theme_and_zoom() {
        let mut cfg = AppConfig {
            follow_system_theme: false,
            theme: BuiltinTheme::Academic,
            ..Default::default()
        };
        cfg.zoom_in();
        cfg.zoom_in();
        assert_eq!(cfg.zoom_percent, 120);
        cfg.font_family = "\"Noto Serif CJK SC\", serif".to_string();
        cfg.show_code_line_numbers = false;

        let serialized = cfg.to_toml_string();
        let parsed = AppConfig::from_toml_str(&serialized);
        assert_eq!(parsed, cfg);

        cfg.follow_system_theme = true;
        assert_eq!(cfg.effective_builtin_theme(true), BuiltinTheme::Dark);
        assert_eq!(cfg.effective_builtin_theme(false), BuiltinTheme::Light);
    }

    #[test]
    fn test_persistence_file_lock_waits_for_os_holder_and_aborts_on_lock_error() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

        let dir = std::env::temp_dir().join(format!(
            "vellmark-cfg-oslock-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        let target = dir.join("recent-files.txt");
        let lock_path = dir.join(".recent-files.txt.vellmark.lock");

        // 1. Simulate another process holding the OS exclusive lock on the sidecar lock file.
        let holder_file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
            .unwrap();
        holder_file.lock().unwrap();

        let target_for_holder = target.clone();
        let holder_thread = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(120));
            fs::write(&target_for_holder, "from-holder\n").unwrap();
            let _ = holder_file.unlock();
            drop(holder_file);
        });

        let start = Instant::now();
        with_persistence_file_lock_timeout(&target, Duration::from_millis(500), || {
            let mut cur = fs::read_to_string(&target).unwrap_or_default();
            cur.push_str("from-waiter\n");
            fs::write(&target, cur)
        })
        .unwrap();
        let elapsed = start.elapsed();
        holder_thread.join().unwrap();

        assert!(
            elapsed >= Duration::from_millis(100),
            "expected lock waiter to block until holder released OS lock, elapsed={elapsed:?}"
        );
        assert_eq!(
            fs::read_to_string(&target).unwrap(),
            "from-holder\nfrom-waiter\n"
        );

        // 2. When the lock stays held beyond timeout, return WouldBlock without invoking `f`.
        let held_file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
            .unwrap();
        held_file.lock().unwrap();
        let timed_out_called = AtomicBool::new(false);
        let timeout_start = Instant::now();
        let timeout_err =
            with_persistence_file_lock_timeout(&target, Duration::from_millis(30), || {
                timed_out_called.store(true, Ordering::SeqCst);
                Ok(())
            })
            .unwrap_err();
        let timeout_elapsed = timeout_start.elapsed();
        assert_eq!(timeout_err.kind(), io::ErrorKind::WouldBlock);
        assert!(!timed_out_called.load(Ordering::SeqCst));
        assert!(
            timeout_elapsed < Duration::from_millis(250),
            "lock timeout should return promptly, elapsed={timeout_elapsed:?}"
        );
        let _ = held_file.unlock();
        drop(held_file);

        // 3. If acquiring the lock fails, `f` must NEVER be invoked.
        let blocked_target = dir.join("blocked.toml");
        let blocked_lock = dir.join(".blocked.toml.vellmark.lock");
        fs::create_dir_all(&blocked_lock).unwrap();
        let called = AtomicBool::new(false);
        let err = with_persistence_file_lock(&blocked_target, || {
            called.store(true, Ordering::SeqCst);
            Ok(())
        })
        .unwrap_err();
        assert!(!called.load(Ordering::SeqCst));
        assert!(err.kind() != io::ErrorKind::WouldBlock);

        // 4. Per-file in-process mutex: another file does not block, and same-file waits honor timeout.
        let file_a = dir.join("a.toml");
        let file_b = dir.join("b.toml");
        let entered = Arc::new(std::sync::Barrier::new(2));
        let entered_clone = Arc::clone(&entered);
        let file_a_clone = file_a.clone();
        let slow_holder = std::thread::spawn(move || {
            with_persistence_file_lock_timeout(&file_a_clone, Duration::from_millis(500), || {
                entered_clone.wait();
                std::thread::sleep(Duration::from_millis(220));
                Ok(())
            })
            .unwrap();
        });
        entered.wait();

        let other_start = Instant::now();
        with_persistence_file_lock_timeout(&file_b, Duration::from_millis(50), || Ok(())).unwrap();
        assert!(
            other_start.elapsed() < Duration::from_millis(120),
            "different target file must not block on another file's in-process lock"
        );

        let same_start = Instant::now();
        let same_err =
            with_persistence_file_lock_timeout(&file_a, Duration::from_millis(40), || Ok(()))
                .unwrap_err();
        assert_eq!(same_err.kind(), io::ErrorKind::WouldBlock);
        assert!(
            same_start.elapsed() < Duration::from_millis(160),
            "same-file in-process lock wait must honor timeout, elapsed={:?}",
            same_start.elapsed()
        );
        slow_holder.join().unwrap();

        let _ = fs::remove_dir_all(&dir);
    }
}
