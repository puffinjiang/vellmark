//! CSS-subset visual style cascade translating `vellmark_theme::ThemeSheet` rules
//! and user configuration into resolved colors, typography, and UI chrome palettes.

use cosmic_text::Color as CosmicColor;
use vellmark_theme::ThemeSheet;

/// Premultiplied/straight sRGB RGBA color `[0..=255]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RgbaColor {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl RgbaColor {
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    /// Convert to normalized linear/sRGB shader floats `[r, g, b, a]` in `0.0..=1.0`.
    pub fn to_f32_array(self) -> [f32; 4] {
        [
            self.r as f32 / 255.0,
            self.g as f32 / 255.0,
            self.b as f32 / 255.0,
            self.a as f32 / 255.0,
        ]
    }

    /// Convert to `cosmic_text::Color`.
    pub const fn to_cosmic_color(self) -> CosmicColor {
        CosmicColor::rgba(self.r, self.g, self.b, self.a)
    }

    pub const fn with_alpha(self, a: u8) -> Self {
        Self { a, ..self }
    }

    /// Blend `self` over `bg` using `factor` (`0.0` = `bg`, `1.0` = `self`).
    pub fn lerp(self, bg: Self, factor: f32) -> Self {
        let t = factor.clamp(0.0, 1.0);
        let inv = 1.0 - t;
        Self {
            r: (self.r as f32 * t + bg.r as f32 * inv).round().clamp(0.0, 255.0) as u8,
            g: (self.g as f32 * t + bg.g as f32 * inv).round().clamp(0.0, 255.0) as u8,
            b: (self.b as f32 * t + bg.b as f32 * inv).round().clamp(0.0, 255.0) as u8,
            a: (self.a as f32 * t + bg.a as f32 * inv).round().clamp(0.0, 255.0) as u8,
        }
    }

    /// Relative luminance approximation in `0.0..=1.0` for dark/light theme detection.
    pub fn luminance(self) -> f32 {
        (0.2126 * self.r as f32 + 0.7152 * self.g as f32 + 0.0722 * self.b as f32) / 255.0
    }

    /// Shift brightness by signed delta `-255..=255` clamped per channel.
    pub fn adjust(self, delta: i16) -> Self {
        let shift = |v: u8| -> u8 { (v as i16 + delta).clamp(0, 255) as u8 };
        Self {
            r: shift(self.r),
            g: shift(self.g),
            b: shift(self.b),
            a: self.a,
        }
    }
}

/// Resolved heading typography (`h1`..`h6`).
#[derive(Debug, Clone, PartialEq)]
pub struct HeadingStyle {
    pub font_size_px: f32,
    pub line_height_px: f32,
    pub weight: u16,
    pub color: RgbaColor,
    pub bottom_border_color: Option<RgbaColor>,
    pub bottom_border_width: f32,
    pub margin_top_px: f32,
    pub margin_bottom_px: f32,
}

/// Adaptive desktop window chrome colors harmonized with the active `#write` theme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UiChromeColors {
    pub is_dark: bool,
    pub window_bg: RgbaColor,
    pub sidebar_bg: RgbaColor,
    pub tab_bar_bg: RgbaColor,
    pub active_tab_bg: RgbaColor,
    pub status_bar_bg: RgbaColor,
    pub panel_bg: RgbaColor,
    pub border: RgbaColor,
    pub hover_bg: RgbaColor,
    pub text: RgbaColor,
    pub muted_text: RgbaColor,
    pub accent: RgbaColor,
    pub warning: RgbaColor,
    pub danger: RgbaColor,
    pub success: RgbaColor,
}

/// Complete cascaded visual theme for document rendering and GUI chrome.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedThemeStyle {
    pub bg_color: RgbaColor,
    pub text_color: RgbaColor,
    pub muted_marker_color: RgbaColor,
    pub focus_dimmed_color: RgbaColor,
    pub selection_bg: RgbaColor,
    pub caret_color: RgbaColor,
    pub ime_preedit_bg: RgbaColor,
    pub ime_underline_color: RgbaColor,
    pub link_color: RgbaColor,
    pub link_underline: bool,
    pub inline_code_color: RgbaColor,
    pub inline_code_bg: RgbaColor,
    pub code_block_bg: RgbaColor,
    pub code_block_text: RgbaColor,
    pub code_block_border: RgbaColor,
    pub code_font_size_px: f32,
    pub code_line_height_px: f32,
    pub blockquote_border: RgbaColor,
    pub blockquote_border_width: f32,
    pub blockquote_text: RgbaColor,
    pub blockquote_bg: Option<RgbaColor>,
    pub hr_color: RgbaColor,
    pub table_border: RgbaColor,
    pub table_header_bg: RgbaColor,
    pub base_font_family: String,
    pub mono_font_family: String,
    pub base_font_size_px: f32,
    pub base_line_height_px: f32,
    pub zoom_scale: f32,
    pub max_content_width_px: f32,
    pub headings: [HeadingStyle; 6],
    pub chrome: UiChromeColors,
}

impl Default for ResolvedThemeStyle {
    fn default() -> Self {
        Self::from_sheet(&ThemeSheet::default(), 16.0, 1.7, 100, 860.0)
    }
}

impl ResolvedThemeStyle {
    /// Resolve CSS subset rules from `sheet` combined with user font size, line height, zoom %, and page max-width.
    pub fn from_sheet(
        sheet: &ThemeSheet,
        config_font_size_px: f32,
        config_line_height_ratio: f32,
        zoom_percent: u16,
        config_max_content_width_px: f32,
    ) -> Self {
        let zoom_scale = (zoom_percent.clamp(50, 300) as f32) / 100.0;

        let bg_color = resolve_color_any(
            sheet,
            &[("#write", "background-color"), ("#write", "background"), ("body", "background-color"), ("body", "background")],
        )
        .unwrap_or(RgbaColor::rgb(255, 255, 255));
        let is_dark = bg_color.luminance() < 0.45;

        let default_text = if is_dark {
            RgbaColor::rgb(228, 232, 240)
        } else {
            RgbaColor::rgb(31, 35, 40)
        };
        let text_color = resolve_color_any(
            sheet,
            &[("#write", "color"), ("body", "color")],
        )
        .unwrap_or(default_text);

        // User config provides baseline font size; CSS `#write` overrides only if explicit, or scales via zoom.
        let unzoomed_font_px = resolve_prop_any(sheet, &[("#write", "font-size"), ("body", "font-size")])
            .and_then(|v| parse_css_length_px(&v, config_font_size_px))
            .unwrap_or(config_font_size_px)
            .clamp(10.0, 48.0);
        let base_font_size_px = unzoomed_font_px * zoom_scale;

        let line_height_ratio = resolve_prop_any(sheet, &[("#write", "line-height"), ("body", "line-height")])
            .and_then(|v| parse_css_line_height_ratio(&v, unzoomed_font_px))
            .unwrap_or(config_line_height_ratio)
            .clamp(1.1, 3.0);
        let base_line_height_px = (base_font_size_px * line_height_ratio).max(16.0);

        let unzoomed_max_w = resolve_prop_any(sheet, &[("#write", "max-width")])
            .and_then(|v| parse_css_length_px(&v, unzoomed_font_px))
            .unwrap_or(config_max_content_width_px)
            .clamp(360.0, 2200.0);
        let max_content_width_px = unzoomed_max_w * zoom_scale;

        let base_font_family = resolve_prop_any(sheet, &[("#write", "font-family"), ("body", "font-family")])
            .map(|s| first_css_font_name(&s))
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "sans-serif".to_string());
        let mono_font_family = resolve_prop_any(sheet, &[("pre", "font-family"), ("code", "font-family")])
            .map(|s| first_css_font_name(&s))
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "monospace".to_string());

        let link_color = resolve_color_any(sheet, &[("a", "color")])
            .unwrap_or(if is_dark {
                RgbaColor::rgb(110, 179, 255)
            } else {
                RgbaColor::rgb(9, 105, 218)
            });
        let link_underline = resolve_prop_any(sheet, &[("a", "text-decoration")])
            .map(|v| v.contains("underline"))
            .unwrap_or(true);

        let selection_bg = resolve_color_any(
            sheet,
            &[("::selection", "background-color"), ("::selection", "background")],
        )
        .map(|c| if c.a == 255 { c.with_alpha(96) } else { c })
        .unwrap_or_else(|| link_color.with_alpha(if is_dark { 90 } else { 70 }));

        let inline_code_color = resolve_color_any(sheet, &[("code", "color")]).unwrap_or(
            if is_dark {
                RgbaColor::rgb(240, 168, 180)
            } else {
                RgbaColor::rgb(185, 58, 92)
            },
        );
        let inline_code_bg = resolve_color_any(
            sheet,
            &[("code", "background-color"), ("code", "background")],
        )
        .unwrap_or(if is_dark {
            RgbaColor::rgba(110, 118, 129, 72)
        } else {
            RgbaColor::rgba(175, 184, 193, 56)
        });

        let code_block_bg = resolve_color_any(
            sheet,
            &[("pre", "background-color"), ("pre", "background")],
        )
        .unwrap_or(if is_dark {
            bg_color.adjust(10)
        } else {
            bg_color.adjust(-10)
        });
        let code_block_text =
            resolve_color_any(sheet, &[("pre", "color")]).unwrap_or(text_color);
        let code_block_border = resolve_color_any(
            sheet,
            &[("pre", "border-color"), ("pre", "border")],
        )
        .unwrap_or_else(|| bg_color.lerp(text_color, 0.85));

        let code_font_size_px = resolve_prop_any(sheet, &[("pre", "font-size"), ("code", "font-size")])
            .and_then(|v| parse_css_length_px(&v, unzoomed_font_px))
            .map(|px| px * zoom_scale)
            .unwrap_or(base_font_size_px * 0.88)
            .max(11.0);
        let code_line_height_px = (code_font_size_px * 1.55).max(16.0);

        let (bq_width, bq_border_from_shorthand) = resolve_prop_any(
            sheet,
            &[("blockquote", "border-left"), ("blockquote", "border")],
        )
        .map(|v| parse_css_border(&v, unzoomed_font_px))
        .unwrap_or((None, None));
        let blockquote_border = resolve_color_any(
            sheet,
            &[("blockquote", "border-left-color")],
        )
        .or(bq_border_from_shorthand)
        .unwrap_or_else(|| text_color.lerp(bg_color, 0.28));
        let blockquote_border_width = resolve_prop_any(
            sheet,
            &[("blockquote", "border-left-width")],
        )
        .and_then(|v| parse_css_length_px(&v, unzoomed_font_px))
        .or(bq_width)
        .unwrap_or(3.5)
        * zoom_scale;
        let blockquote_text = resolve_color_any(sheet, &[("blockquote", "color")])
            .unwrap_or_else(|| text_color.lerp(bg_color, 0.72));
        let blockquote_bg = resolve_color_any(
            sheet,
            &[("blockquote", "background-color"), ("blockquote", "background")],
        );

        let hr_color = resolve_color_any(
            sheet,
            &[("hr", "background-color"), ("hr", "border-color"), ("hr", "color")],
        )
        .unwrap_or_else(|| text_color.lerp(bg_color, 0.22));
        let table_border = resolve_color_any(
            sheet,
            &[("td", "border-color"), ("th", "border-color"), ("table", "border-color")],
        )
        .unwrap_or_else(|| text_color.lerp(bg_color, 0.25));
        let table_header_bg = resolve_color_any(
            sheet,
            &[("th", "background-color"), ("th", "background"), ("thead", "background-color")],
        )
        .unwrap_or_else(|| if is_dark { bg_color.adjust(12) } else { bg_color.adjust(-8) });

        let default_heading_scales = [2.0_f32, 1.55, 1.32, 1.15, 1.05, 0.96];
        let headings = std::array::from_fn(|idx| {
            let sel = format!("h{}", idx + 1);
            let default_px = unzoomed_font_px * default_heading_scales[idx];
            let font_px = resolve_prop_any(sheet, &[(&sel, "font-size")])
                .and_then(|v| parse_css_length_px(&v, unzoomed_font_px))
                .unwrap_or(default_px)
                .clamp(12.0, 72.0)
                * zoom_scale;
            let lh_ratio = resolve_prop_any(sheet, &[(&sel, "line-height")])
                .and_then(|v| parse_css_line_height_ratio(&v, font_px / zoom_scale))
                .unwrap_or(1.32);
            let weight = resolve_prop_any(sheet, &[(&sel, "font-weight")])
                .map(|v| parse_css_weight(&v))
                .unwrap_or(if idx <= 2 { 700 } else { 600 });
            let color = resolve_color_any(sheet, &[(&sel, "color")]).unwrap_or(text_color);
            let (bb_w, bb_c) = resolve_prop_any(sheet, &[(&sel, "border-bottom")])
                .map(|v| parse_css_border(&v, unzoomed_font_px))
                .unwrap_or((None, None));
            let bottom_border_color = resolve_color_any(sheet, &[(&sel, "border-bottom-color")])
                .or(bb_c);
            let bottom_border_width = if bottom_border_color.is_some() {
                bb_w.unwrap_or(1.0) * zoom_scale
            } else {
                0.0
            };
            let margin_top_px = resolve_prop_any(sheet, &[(&sel, "margin-top")])
                .and_then(|v| parse_css_length_px(&v, unzoomed_font_px))
                .unwrap_or(base_font_size_px / zoom_scale * (if idx == 0 { 0.8 } else { 0.65 }))
                * zoom_scale;
            let margin_bottom_px = resolve_prop_any(sheet, &[(&sel, "margin-bottom")])
                .and_then(|v| parse_css_length_px(&v, unzoomed_font_px))
                .unwrap_or(base_font_size_px / zoom_scale * 0.35)
                * zoom_scale;

            HeadingStyle {
                font_size_px: font_px,
                line_height_px: (font_px * lh_ratio).max(font_px + 4.0),
                weight,
                color,
                bottom_border_color,
                bottom_border_width,
                margin_top_px,
                margin_bottom_px,
            }
        });

        let chrome = derive_ui_chrome(bg_color, text_color, link_color, is_dark);

        Self {
            bg_color,
            text_color,
            muted_marker_color: text_color.lerp(bg_color, 0.42),
            focus_dimmed_color: text_color.lerp(bg_color, 0.28),
            selection_bg,
            caret_color: link_color,
            ime_preedit_bg: link_color.with_alpha(if is_dark { 45 } else { 32 }),
            ime_underline_color: link_color,
            link_color,
            link_underline,
            inline_code_color,
            inline_code_bg,
            code_block_bg,
            code_block_text,
            code_block_border,
            code_font_size_px,
            code_line_height_px,
            blockquote_border,
            blockquote_border_width,
            blockquote_text,
            blockquote_bg,
            hr_color,
            table_border,
            table_header_bg,
            base_font_family,
            mono_font_family,
            base_font_size_px,
            base_line_height_px,
            zoom_scale,
            max_content_width_px,
            headings,
            chrome,
        }
    }

    /// Dim inactive block colors smoothly in Focus Mode (`PRD 4.7 V-01`).
    pub fn apply_focus_mode(&self, color: RgbaColor, is_focused: bool, focus_mode: bool) -> RgbaColor {
        if !focus_mode || is_focused {
            color
        } else {
            color.lerp(self.bg_color, 0.28)
        }
    }
}

fn derive_ui_chrome(
    doc_bg: RgbaColor,
    doc_text: RgbaColor,
    accent: RgbaColor,
    is_dark: bool,
) -> UiChromeColors {
    let sidebar_bg = if is_dark { doc_bg.adjust(-8) } else { doc_bg.adjust(-9) };
    let tab_bar_bg = if is_dark { doc_bg.adjust(-12) } else { doc_bg.adjust(-14) };
    let status_bar_bg = doc_bg.adjust(-10);
    let panel_bg = if is_dark { doc_bg.adjust(8) } else { doc_bg.adjust(-4) };
    let border = doc_text.lerp(doc_bg, 0.16);
    let hover_bg = if is_dark { doc_bg.adjust(14) } else { doc_bg.adjust(-18) };
    UiChromeColors {
        is_dark,
        window_bg: doc_bg,
        sidebar_bg,
        tab_bar_bg,
        active_tab_bg: doc_bg,
        status_bar_bg,
        panel_bg,
        border,
        hover_bg,
        text: doc_text,
        muted_text: doc_text.lerp(doc_bg, 0.56),
        accent,
        warning: RgbaColor::rgb(217, 152, 42),
        danger: RgbaColor::rgb(216, 76, 76),
        success: RgbaColor::rgb(58, 166, 98),
    }
}

fn resolve_prop_any(sheet: &ThemeSheet, queries: &[(&str, &str)]) -> Option<String> {
    queries
        .iter()
        .find_map(|(sel, prop)| sheet.resolve_property(sel, prop))
}

fn resolve_color_any(sheet: &ThemeSheet, queries: &[(&str, &str)]) -> Option<RgbaColor> {
    queries
        .iter()
        .find_map(|(sel, prop)| sheet.resolve_property(sel, prop).and_then(|v| parse_css_color(&v)))
}

fn first_css_font_name(raw: &str) -> String {
    for part in raw.split(',') {
        let trimmed = part.trim().trim_matches('"').trim_matches('\'').trim();
        if !trimmed.is_empty()
            && !trimmed.starts_with('-')
            && !trimmed.eq_ignore_ascii_case("system-ui")
        {
            return trimmed.to_string();
        }
    }
    String::new()
}

/// Parse hex (`#rgb`, `#rgba`, `#rrggbb`, `#rrggbbaa`) or `rgb(...)`/`rgba(...)` color literals.
pub fn parse_css_color(raw: &str) -> Option<RgbaColor> {
    let s = raw.trim();
    if let Some(hex) = s.strip_prefix('#') {
        return match hex.len() {
            3 => {
                let r = u8::from_str_radix(&hex[0..1], 16).ok()? * 17;
                let g = u8::from_str_radix(&hex[1..2], 16).ok()? * 17;
                let b = u8::from_str_radix(&hex[2..3], 16).ok()? * 17;
                Some(RgbaColor::rgb(r, g, b))
            }
            4 => {
                let r = u8::from_str_radix(&hex[0..1], 16).ok()? * 17;
                let g = u8::from_str_radix(&hex[1..2], 16).ok()? * 17;
                let b = u8::from_str_radix(&hex[2..3], 16).ok()? * 17;
                let a = u8::from_str_radix(&hex[3..4], 16).ok()? * 17;
                Some(RgbaColor::new(r, g, b, a))
            }
            6 => {
                let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
                let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
                let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
                Some(RgbaColor::rgb(r, g, b))
            }
            8 => {
                let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
                let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
                let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
                let a = u8::from_str_radix(&hex[6..8], 16).ok()?;
                Some(RgbaColor::new(r, g, b, a))
            }
            _ => None,
        };
    }

    let lower = s.to_ascii_lowercase();
    if let Some(inner) = lower.strip_prefix("rgba(").and_then(|t| t.strip_suffix(')')) {
        let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
        if parts.len() == 4 {
            let r = parts[0].parse::<f32>().ok()?.round().clamp(0.0, 255.0) as u8;
            let g = parts[1].parse::<f32>().ok()?.round().clamp(0.0, 255.0) as u8;
            let b = parts[2].parse::<f32>().ok()?.round().clamp(0.0, 255.0) as u8;
            let a_raw = parts[3].parse::<f32>().ok()?;
            let a = if a_raw <= 1.0 {
                (a_raw * 255.0).round()
            } else {
                a_raw.round()
            }
            .clamp(0.0, 255.0) as u8;
            return Some(RgbaColor::new(r, g, b, a));
        }
    }
    if let Some(inner) = lower.strip_prefix("rgb(").and_then(|t| t.strip_suffix(')')) {
        let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
        if parts.len() == 3 {
            let r = parts[0].parse::<f32>().ok()?.round().clamp(0.0, 255.0) as u8;
            let g = parts[1].parse::<f32>().ok()?.round().clamp(0.0, 255.0) as u8;
            let b = parts[2].parse::<f32>().ok()?.round().clamp(0.0, 255.0) as u8;
            return Some(RgbaColor::rgb(r, g, b));
        }
    }
    match lower.as_str() {
        "black" => Some(RgbaColor::rgb(0, 0, 0)),
        "white" => Some(RgbaColor::rgb(255, 255, 255)),
        "transparent" => Some(RgbaColor::new(0, 0, 0, 0)),
        _ => None,
    }
}

/// Parse a CSS length (`px`, `em`, `rem`, `pt`, or unitless multiplier) into logical pixels.
pub fn parse_css_length_px(raw: &str, base_px: f32) -> Option<f32> {
    let s = raw.trim().to_ascii_lowercase();
    if let Some(n) = s.strip_suffix("px") {
        return n.trim().parse::<f32>().ok();
    }
    if let Some(n) = s.strip_suffix("rem") {
        return n.trim().parse::<f32>().ok().map(|v| v * base_px);
    }
    if let Some(n) = s.strip_suffix("em") {
        return n.trim().parse::<f32>().ok().map(|v| v * base_px);
    }
    if let Some(n) = s.strip_suffix("pt") {
        return n.trim().parse::<f32>().ok().map(|v| v * (96.0 / 72.0));
    }
    if let Some(n) = s.strip_suffix('%') {
        return n.trim().parse::<f32>().ok().map(|v| base_px * (v / 100.0));
    }
    s.parse::<f32>().ok().map(|v| v * base_px)
}

fn parse_css_line_height_ratio(raw: &str, font_size_px: f32) -> Option<f32> {
    let s = raw.trim().to_ascii_lowercase();
    if s.ends_with("px") || s.ends_with("em") || s.ends_with("rem") || s.ends_with('%') {
        return parse_css_length_px(&s, font_size_px).map(|px| px / font_size_px.max(1.0));
    }
    s.parse::<f32>().ok()
}

fn parse_css_weight(raw: &str) -> u16 {
    let s = raw.trim().to_ascii_lowercase();
    match s.as_str() {
        "normal" => 400,
        "medium" => 500,
        "semibold" | "semi-bold" => 600,
        "bold" => 700,
        "extrabold" => 800,
        _ => s.parse::<u16>().unwrap_or(600).clamp(100, 900),
    }
}

fn parse_css_border(raw: &str, base_px: f32) -> (Option<f32>, Option<RgbaColor>) {
    let mut width = None;
    let mut color = None;
    // Handle `rgba(...)` or `rgb(...)` without splitting inside parens.
    if let Some(rgb_start) = raw.find("rgb")
        && let Some(close) = raw[rgb_start..].find(')') {
            color = parse_css_color(&raw[rgb_start..=rgb_start + close]);
        }
    for token in raw.split_whitespace() {
        if width.is_none() && (token.ends_with("px") || token.ends_with("em")) {
            width = parse_css_length_px(token, base_px);
        }
        if color.is_none() && token.starts_with('#') {
            color = parse_css_color(token);
        }
    }
    (width, color)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vellmark_theme::BuiltinTheme;

    #[test]
    fn test_css_color_and_length_parsing() {
        assert_eq!(parse_css_color("#abc"), Some(RgbaColor::rgb(0xaa, 0xbb, 0xcc)));
        assert_eq!(
            parse_css_color("rgba(10, 20, 30, 0.5)"),
            Some(RgbaColor::new(10, 20, 30, 128))
        );
        assert_eq!(parse_css_length_px("1.5em", 16.0), Some(24.0));
        assert_eq!(parse_css_length_px("820px", 16.0), Some(820.0));
    }

    #[test]
    fn test_all_builtin_themes_resolve_consistent_palettes_and_zoom() {
        for builtin in BuiltinTheme::ALL {
            let sheet = builtin.load();
            let style = ResolvedThemeStyle::from_sheet(&sheet, 16.0, 1.7, 125, 860.0);
            assert!(style.headings[0].font_size_px > style.headings[1].font_size_px);
            assert!(style.base_font_size_px >= 19.0);
            assert_eq!(style.chrome.is_dark, matches!(builtin, BuiltinTheme::Dark));
            assert_ne!(style.bg_color, style.text_color);
        }

        let custom = ThemeSheet::parse_css(
            ":root { --acc: #cc3355; } #write { background-color: #111827; color: #f8fafc; max-width: 720px; } a { color: var(--acc); } ::selection { background-color: rgba(204, 51, 85, 0.4); }",
        );
        let style = ResolvedThemeStyle::from_sheet(&custom, 16.0, 1.7, 100, 860.0);
        assert_eq!(style.bg_color, RgbaColor::rgb(0x11, 0x18, 0x27));
        assert_eq!(style.link_color, RgbaColor::rgb(0xcc, 0x33, 0x55));
        assert_eq!(style.selection_bg, RgbaColor::new(204, 51, 85, 102));
        assert_eq!(style.max_content_width_px, 720.0);
        assert!(style.chrome.is_dark);
    }
}
