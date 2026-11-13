//! `vellmark-theme` parses the Vellmark CSS subset (PRD 6.6) and reports unsupported
//! selectors or properties for the Theme Diagnostics panel (PRD S-03 & S-08).

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

const SUPPORTED_SELECTORS: &[&str] = &[
    ":root",
    "#write",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "blockquote",
    "code",
    "pre",
    "table",
    "th",
    "td",
    "ul",
    "ol",
    "li",
    "a",
    "hr",
    "::selection",
];

const SUPPORTED_PROPERTIES: &[&str] = &[
    "font-family",
    "font-size",
    "font-weight",
    "font-style",
    "line-height",
    "color",
    "letter-spacing",
    "text-decoration",
    "text-align",
    "margin",
    "margin-top",
    "margin-bottom",
    "margin-left",
    "margin-right",
    "padding",
    "padding-top",
    "padding-bottom",
    "padding-left",
    "padding-right",
    "border",
    "border-radius",
    "background-color",
    "max-width",
    "width",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BuiltinTheme {
    Light,
    Dark,
    Academic,
    GitHub,
}

impl BuiltinTheme {
    pub const ALL: [BuiltinTheme; 4] = [
        BuiltinTheme::Light,
        BuiltinTheme::Dark,
        BuiltinTheme::Academic,
        BuiltinTheme::GitHub,
    ];

    /// Follow system dark/light preference (PRD 4.8 S-02).
    pub fn for_system_color_scheme(prefers_dark: bool) -> Self {
        if prefers_dark {
            Self::Dark
        } else {
            Self::Light
        }
    }

    pub fn css_source(self) -> &'static str {
        match self {
            Self::Light => include_str!("../../../themes/default-light.css"),
            Self::Dark => include_str!("../../../themes/default-dark.css"),
            Self::Academic => include_str!("../../../themes/academic-newsprint.css"),
            Self::GitHub => include_str!("../../../themes/github.css"),
        }
    }

    pub fn load(self) -> ThemeSheet {
        ThemeSheet::parse_css(self.css_source())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThemeDiagnostic {
    pub selector: String,
    pub item: String,
    pub reason: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThemeSheet {
    pub variables: HashMap<String, String>,
    pub rules: HashMap<String, HashMap<String, String>>,
    pub diagnostics: Vec<ThemeDiagnostic>,
}

impl ThemeSheet {
    /// Parse a CSS stylesheet using the Vellmark v1.0 subset rules, silently ignoring
    /// unsupported selectors/properties while recording them in `diagnostics` (PRD 6.6).
    pub fn parse_css(css: &str) -> Self {
        let mut sheet = ThemeSheet::default();
        for block in css.split('}') {
            let Some((raw_sel, raw_body)) = block.split_once('{') else {
                continue;
            };
            let selector = strip_css_comments(raw_sel).trim().to_string();
            if selector.is_empty() {
                continue;
            }

            if !is_supported_selector(&selector) {
                sheet.diagnostics.push(ThemeDiagnostic {
                    selector: selector.clone(),
                    item: selector,
                    reason: "unsupported selector in Vellmark CSS subset".to_string(),
                });
                continue;
            }

            let body = strip_css_comments(raw_body);
            for decl in body.split(';') {
                let decl = decl.trim();
                if decl.is_empty() {
                    continue;
                }
                let Some((prop, val)) = decl.split_once(':') else {
                    continue;
                };
                let prop = prop.trim();
                let val = val.trim();

                if let Some(var_name) = prop.strip_prefix("--") {
                    sheet.variables.insert(format!("--{var_name}"), val.to_string());
                    continue;
                }

                if SUPPORTED_PROPERTIES.contains(&prop) {
                    sheet
                        .rules
                        .entry(selector.clone())
                        .or_default()
                        .insert(prop.to_string(), val.to_string());
                } else {
                    sheet.diagnostics.push(ThemeDiagnostic {
                        selector: selector.clone(),
                        item: prop.to_string(),
                        reason: "unsupported CSS property (layout controlled by editor)".to_string(),
                    });
                }
            }
        }
        sheet
    }

    /// Resolve a property for `selector` (falling back to `#write` for inherited text properties)
    /// and expand any `var(--custom-prop)` references against `:root` variables (PRD 6.6).
    pub fn resolve_property(&self, selector: &str, property: &str) -> Option<String> {
        let raw = self
            .rules
            .get(selector)
            .and_then(|props| props.get(property))
            .or_else(|| {
                if matches!(
                    property,
                    "color" | "font-family" | "font-size" | "line-height" | "letter-spacing"
                ) {
                    self.rules
                        .get("#write")
                        .and_then(|props| props.get(property))
                } else {
                    None
                }
            })?;
        Some(self.expand_css_variables(raw))
    }

    fn expand_css_variables(&self, raw: &str) -> String {
        let mut out = String::with_capacity(raw.len());
        let mut rest = raw;
        while let Some(idx) = rest.find("var(") {
            out.push_str(&rest[..idx]);
            let after_var = &rest[idx + 4..];
            if let Some(close) = after_var.find(')') {
                let inner = after_var[..close].trim();
                let (var_name, fallback) = match inner.split_once(',') {
                    Some((v, fb)) => (v.trim(), Some(fb.trim())),
                    None => (inner, None),
                };
                if let Some(val) = self.variables.get(var_name) {
                    out.push_str(val);
                } else if let Some(fb) = fallback {
                    out.push_str(fb);
                }
                rest = &after_var[close + 1..];
            } else {
                out.push_str(&rest[idx..]);
                return out;
            }
        }
        out.push_str(rest);
        out
    }
}

fn is_supported_selector(sel: &str) -> bool {
    // Support descendant (`a b`) and child (`a > b`) combinators of supported simple selectors (PRD 6.6)
    if sel.contains('(') || sel.contains('[') || sel.contains('+') || sel.contains('~') {
        return false;
    }
    let parts: Vec<&str> = sel
        .split(|c: char| c.is_whitespace() || c == '>')
        .filter(|s| !s.is_empty())
        .collect();
    if parts.is_empty() {
        return false;
    }
    parts.into_iter().all(|part| {
        let base = part
            .trim_end_matches(":hover")
            .trim_end_matches(":first-child")
            .trim();
        SUPPORTED_SELECTORS.contains(&base) || base.starts_with('.')
    })
}

fn strip_css_comments(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    while let Some(start) = rest.find("/*") {
        out.push_str(&rest[..start]);
        if let Some(end) = rest[start + 2..].find("*/") {
            rest = &rest[start + 2 + end + 2..];
        } else {
            return out;
        }
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_css_subset_accepts_supported_and_reports_unsupported_layout_props() {
        let css = r#"
        :root { --text-color: #222; }
        #write { max-width: 860px; display: flex; }
        h1 { color: var(--text-color); font-size: 2rem; }
        div:has(> p) { color: red; }
        "#;
        let sheet = ThemeSheet::parse_css(css);
        assert_eq!(sheet.variables.get("--text-color").unwrap(), "#222");
        assert_eq!(sheet.rules["#write"]["max-width"], "860px");
        assert!(!sheet.rules["#write"].contains_key("display"));
        assert_eq!(sheet.diagnostics.len(), 2);
        assert_eq!(
            sheet.resolve_property("h1", "color").as_deref(),
            Some("#222")
        );
    }

    #[test]
    fn test_prd_s01_all_four_builtin_themes_parse_cleanly_and_resolve_vars() {
        for theme in BuiltinTheme::ALL {
            let sheet = theme.load();
            assert!(
                sheet.diagnostics.is_empty(),
                "Builtin theme {theme:?} produced diagnostics: {:?}",
                sheet.diagnostics
            );
            let bg = sheet.resolve_property("#write", "background-color").unwrap();
            let fg = sheet.resolve_property("#write", "color").unwrap();
            assert!(bg.starts_with('#'));
            assert!(fg.starts_with('#'));
        }
    }
}
