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
}

fn is_supported_selector(sel: &str) -> bool {
    let base = sel
        .trim_end_matches(":hover")
        .trim_end_matches(":first-child")
        .trim();
    SUPPORTED_SELECTORS.contains(&base) || base.starts_with('.')
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
    }
}
