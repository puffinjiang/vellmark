//! Cross-platform OS clipboard adapter (`NSPasteboard` on macOS, CLI tools on Linux,
//! PowerShell bridge on Windows) with an in-memory backend for isolated headless tests.

use std::sync::Mutex;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ClipboardContent {
    pub text: Option<String>,
    pub html: Option<String>,
}

/// Abstract clipboard transport so GUI controllers interact with the OS clipboard at runtime
/// while tests swap in an isolated in-memory implementation without mutating the host clipboard.
pub trait ClipboardBackend: Send + Sync {
    fn read(&self) -> ClipboardContent;
    fn write(&self, text: &str, html: Option<&str>) -> Result<(), String>;
}

/// Isolated in-memory clipboard for deterministic unit testing.
#[derive(Debug, Default)]
pub struct MemoryClipboard {
    content: Mutex<ClipboardContent>,
}

impl MemoryClipboard {
    pub fn with_initial(content: ClipboardContent) -> Self {
        Self {
            content: Mutex::new(content),
        }
    }

    pub fn set_content(&self, content: ClipboardContent) {
        if let Ok(mut guard) = self.content.lock() {
            *guard = content;
        }
    }

    pub fn get_content(&self) -> ClipboardContent {
        self.content.lock().map(|g| g.clone()).unwrap_or_default()
    }
}

impl ClipboardBackend for MemoryClipboard {
    fn read(&self) -> ClipboardContent {
        self.content.lock().map(|g| g.clone()).unwrap_or_default()
    }

    fn write(&self, text: &str, html: Option<&str>) -> Result<(), String> {
        let mut guard = self
            .content
            .lock()
            .map_err(|_| "clipboard mutex poisoned".to_string())?;
        *guard = ClipboardContent {
            text: Some(text.to_string()),
            html: html.map(str::to_string),
        };
        Ok(())
    }
}

/// Native OS clipboard integration.
///
/// Stores our last written `(os_change_count, markdown)` so intra-app copies preserve exact
/// raw Markdown syntax even while also exporting `public.html` to rich-text apps.
#[derive(Debug, Default)]
pub struct NativeClipboard {
    #[cfg(target_os = "macos")]
    last_self_write: Mutex<Option<(isize, String)>>,
    #[cfg(not(target_os = "macos"))]
    fallback: Mutex<ClipboardContent>,
}

impl NativeClipboard {
    pub fn new() -> Self {
        Self::default()
    }
}

#[cfg(target_os = "macos")]
impl ClipboardBackend for NativeClipboard {
    fn read(&self) -> ClipboardContent {
        use objc2_app_kit::{NSPasteboard, NSPasteboardTypeHTML, NSPasteboardTypeString};

        unsafe {
            let pb = NSPasteboard::generalPasteboard();
            let change_count = pb.changeCount();
            if let Ok(guard) = self.last_self_write.lock()
                && let Some((saved_cc, ref md)) = *guard
                    && saved_cc == change_count {
                        return ClipboardContent {
                            text: Some(md.clone()),
                            html: None,
                        };
                    }
            let text = pb.stringForType(NSPasteboardTypeString).map(|s| s.to_string());
            let html = pb.stringForType(NSPasteboardTypeHTML).map(|s| s.to_string());
            ClipboardContent { text, html }
        }
    }

    fn write(&self, text: &str, html: Option<&str>) -> Result<(), String> {
        use objc2_app_kit::{NSPasteboard, NSPasteboardTypeHTML, NSPasteboardTypeString};
        use objc2_foundation::{NSArray, NSCopying, NSString};

        unsafe {
            let pb = NSPasteboard::generalPasteboard();
            let mut types = vec![NSPasteboardTypeString.copy()];
            if html.is_some() {
                types.push(NSPasteboardTypeHTML.copy());
            }
            let types_arr = NSArray::from_vec(types);
            pb.clearContents();
            let _ = pb.declareTypes_owner(&types_arr, None);
            let ns_text = NSString::from_str(text);
            let ok_text = pb.setString_forType(&ns_text, NSPasteboardTypeString);
            if let Some(h) = html {
                let ns_html = NSString::from_str(h);
                let _ = pb.setString_forType(&ns_html, NSPasteboardTypeHTML);
            }
            if !ok_text {
                return Err("failed to write text to NSPasteboard".to_string());
            }
            let actual_cc = pb.changeCount();
            if let Ok(mut guard) = self.last_self_write.lock() {
                *guard = Some((actual_cc, text.to_string()));
            }
            Ok(())
        }
    }
}

#[cfg(all(unix, not(target_os = "macos")))]
impl ClipboardBackend for NativeClipboard {
    fn read(&self) -> ClipboardContent {
        use std::process::Command;

        let read_cmd = |program: &str, args: &[&str]| -> Option<String> {
            let out = Command::new(program).args(args).output().ok()?;
            if out.status.success() {
                Some(String::from_utf8_lossy(&out.stdout).into_owned())
            } else {
                None
            }
        };

        let text = read_cmd("wl-paste", &["--no-newline"])
            .or_else(|| read_cmd("xclip", &["-o", "-selection", "clipboard"]))
            .or_else(|| read_cmd("xsel", &["--clipboard", "--output"]));
        let html = read_cmd("wl-paste", &["--no-newline", "--type", "text/html"]).or_else(|| {
            read_cmd("xclip", &["-o", "-selection", "clipboard", "-t", "text/html"])
        });

        if text.is_some() || html.is_some() {
            ClipboardContent { text, html }
        } else {
            self.fallback.lock().map(|g| g.clone()).unwrap_or_default()
        }
    }

    fn write(&self, text: &str, html: Option<&str>) -> Result<(), String> {
        use std::io::Write;
        use std::process::{Command, Stdio};

        let try_pipe = |program: &str, args: &[&str]| -> bool {
            let Ok(mut child) = Command::new(program)
                .args(args)
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
            else {
                return false;
            };
            if let Some(mut stdin) = child.stdin.take() {
                let _ = stdin.write_all(text.as_bytes());
            }
            child.wait().map(|s| s.success()).unwrap_or(false)
        };

        if let Ok(mut guard) = self.fallback.lock() {
            *guard = ClipboardContent {
                text: Some(text.to_string()),
                html: html.map(str::to_string),
            };
        }

        if try_pipe("wl-copy", &[])
            || try_pipe("xclip", &["-selection", "clipboard"])
            || try_pipe("xsel", &["--clipboard", "--input"])
        {
            Ok(())
        } else {
            Err("no OS clipboard helper (wl-copy/xclip/xsel) available".to_string())
        }
    }
}

#[cfg(target_os = "windows")]
impl ClipboardBackend for NativeClipboard {
    fn read(&self) -> ClipboardContent {
        use std::process::Command;

        if let Ok(out) = Command::new("powershell")
            .args(["-NoProfile", "-Command", "Get-Clipboard -Raw"])
            .output()
        {
            if out.status.success() {
                let text = String::from_utf8_lossy(&out.stdout).into_owned();
                if !text.is_empty() {
                    return ClipboardContent {
                        text: Some(text),
                        html: None,
                    };
                }
            }
        }
        self.fallback.lock().map(|g| g.clone()).unwrap_or_default()
    }

    fn write(&self, text: &str, html: Option<&str>) -> Result<(), String> {
        use std::io::Write;
        use std::process::{Command, Stdio};

        if let Ok(mut guard) = self.fallback.lock() {
            *guard = ClipboardContent {
                text: Some(text.to_string()),
                html: html.map(str::to_string),
            };
        }

        let mut child = Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                "$input | Out-String -Stream | Set-Clipboard",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("failed to launch clipboard bridge: {e}"))?;
        if let Some(mut stdin) = child.stdin.take() {
            let _ = stdin.write_all(text.as_bytes());
        }
        if child.wait().map(|s| s.success()).unwrap_or(false) {
            Ok(())
        } else {
            Err("Set-Clipboard command failed".to_string())
        }
    }
}
