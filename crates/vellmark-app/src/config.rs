//! Persistent application configuration and view preferences (PRD 4.7, 4.8, 5.2, 6.3).

use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use vellmark_theme::BuiltinTheme;

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
            && !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        fs::write(path, self.to_toml_string())
    }

    pub fn load_from_file(path: &Path) -> io::Result<Self> {
        let raw = fs::read_to_string(path)?;
        Ok(Self::from_toml_str(&raw))
    }
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
}
