//! Cross-platform OS clipboard adapter (`NSPasteboard` on macOS, CLI tools on Linux,
//! PowerShell bridge on Windows) with an in-memory backend for isolated headless tests.

use std::sync::Mutex;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ClipboardContent {
    pub text: Option<String>,
    pub html: Option<String>,
    pub image_png: Option<Vec<u8>>,
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
            image_png: None,
        };
        Ok(())
    }
}

const MAX_CLIPBOARD_IMAGE_BYTES: usize = 32 * 1024 * 1024;

#[cfg(target_os = "macos")]
fn decode_to_png_bytes(raw: &[u8], fmt: image::ImageFormat) -> Option<Vec<u8>> {
    if raw.is_empty() || raw.len() > MAX_CLIPBOARD_IMAGE_BYTES {
        return None;
    }
    let img = image::load_from_memory_with_format(raw, fmt).ok()?;
    let mut png_out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut png_out, image::ImageFormat::Png).ok()?;
    Some(png_out.into_inner())
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
        use objc2_app_kit::{
            NSPasteboard, NSPasteboardTypeHTML, NSPasteboardTypePNG, NSPasteboardTypeString,
            NSPasteboardTypeTIFF,
        };

        unsafe {
            let pb = NSPasteboard::generalPasteboard();
            let change_count = pb.changeCount();
            if let Ok(guard) = self.last_self_write.lock()
                && let Some((saved_cc, ref md)) = *guard
                && saved_cc == change_count
            {
                return ClipboardContent {
                    text: Some(md.clone()),
                    html: None,
                    image_png: None,
                };
            }
            let text = pb
                .stringForType(NSPasteboardTypeString)
                .map(|s| s.to_string());
            let html = pb
                .stringForType(NSPasteboardTypeHTML)
                .map(|s| s.to_string());
            // Prefer text/HTML whenever present; read bitmap only when no text payload exists.
            let image_png = if text.is_none() {
                pb.dataForType(NSPasteboardTypePNG)
                    .map(|d| d.bytes().to_vec())
                    .filter(|b| !b.is_empty() && b.len() <= MAX_CLIPBOARD_IMAGE_BYTES)
                    .or_else(|| {
                        pb.dataForType(NSPasteboardTypeTIFF)
                            .and_then(|d| decode_to_png_bytes(d.bytes(), image::ImageFormat::Tiff))
                    })
            } else {
                None
            };
            ClipboardContent {
                text,
                html,
                image_png,
            }
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

        let read_cmd_bytes = |program: &str, args: &[&str]| -> Option<Vec<u8>> {
            let out = Command::new(program).args(args).output().ok()?;
            if out.status.success()
                && !out.stdout.is_empty()
                && out.stdout.len() <= MAX_CLIPBOARD_IMAGE_BYTES
            {
                Some(out.stdout)
            } else {
                None
            }
        };

        let text = read_cmd("wl-paste", &["--no-newline"])
            .or_else(|| read_cmd("xclip", &["-o", "-selection", "clipboard"]))
            .or_else(|| read_cmd("xsel", &["--clipboard", "--output"]));
        let html = read_cmd("wl-paste", &["--no-newline", "--type", "text/html"]).or_else(|| {
            read_cmd(
                "xclip",
                &["-o", "-selection", "clipboard", "-t", "text/html"],
            )
        });
        let image_png = if text.is_none() {
            read_cmd_bytes("wl-paste", &["--no-newline", "--type", "image/png"]).or_else(|| {
                read_cmd_bytes(
                    "xclip",
                    &["-o", "-selection", "clipboard", "-t", "image/png"],
                )
            })
        } else {
            None
        };

        if text.is_some() || html.is_some() || image_png.is_some() {
            ClipboardContent {
                text,
                html,
                image_png,
            }
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
                image_png: None,
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
            && out.status.success()
        {
            let text = String::from_utf8_lossy(&out.stdout).into_owned();
            if !text.is_empty() {
                return ClipboardContent {
                    text: Some(text),
                    html: None,
                    image_png: None,
                };
            }
        }
        if let Ok(out) = Command::new("powershell")
            .args([
                "-NoProfile",
                "-STA",
                "-Command",
                "Add-Type -AssemblyName System.Windows.Forms,System.Drawing; if ([System.Windows.Forms.Clipboard]::ContainsImage()) { $img = [System.Windows.Forms.Clipboard]::GetImage(); $ms = New-Object System.IO.MemoryStream; $img.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png); [Convert]::ToBase64String($ms.ToArray()) }",
            ])
            .output()
            && out.status.success()
        {
            let b64 = String::from_utf8_lossy(&out.stdout);
            if let Some(png) = crate::images::decode_base64_str(b64.trim(), MAX_CLIPBOARD_IMAGE_BYTES)
            {
                return ClipboardContent {
                    text: None,
                    html: None,
                    image_png: Some(png),
                };
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
                image_png: None,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_clipboard_image_and_text_roundtrip() {
        let clip = MemoryClipboard::with_initial(ClipboardContent {
            text: None,
            html: None,
            image_png: Some(vec![0x89, b'P', b'N', b'G']),
        });
        assert_eq!(
            clip.read().image_png.as_deref(),
            Some(&[0x89, b'P', b'N', b'G'][..])
        );
        clip.write("hello", Some("<b>hello</b>")).unwrap();
        let after = clip.read();
        assert_eq!(after.text.as_deref(), Some("hello"));
        assert!(after.image_png.is_none());
    }
}
