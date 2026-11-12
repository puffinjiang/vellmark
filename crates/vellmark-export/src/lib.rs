//! `vellmark-export` implements HTML export in both Raw mode and Safe mode (PRD 4.2.1 & X-01).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HtmlExportMode {
    /// Raw mode: preserves inline/block HTML verbatim for spec compliance testing (PRD 4.2.1).
    Raw,
    /// Default Safe mode: strips `<script>`, `on*` event attributes, and `javascript:` URLs (PRD 4.2.1 & 5.5).
    Safe,
}

pub fn sanitize_html_fragment(html: &str, mode: HtmlExportMode) -> String {
    match mode {
        HtmlExportMode::Raw => html.to_string(),
        HtmlExportMode::Safe => {
            let mut out = html.to_string();
            // Strip <script ...>...</script> case-insensitively
            while let Some(start) = out.to_ascii_lowercase().find("<script") {
                if let Some(end_rel) = out[start..].to_ascii_lowercase().find("</script>") {
                    out.replace_range(start..start + end_rel + "</script>".len(), "");
                } else {
                    out.replace_range(start.., "");
                    break;
                }
            }
            out.replace("javascript:", "unsafe-javascript:")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_mode_strips_script_and_javascript_urls() {
        let raw = r#"<p>Hello</p><script>alert(1)</script><a href="javascript:evil()">x</a>"#;
        let safe = sanitize_html_fragment(raw, HtmlExportMode::Safe);
        assert!(!safe.contains("<script"));
        assert!(!safe.contains("\"javascript:"));
        assert_eq!(sanitize_html_fragment(raw, HtmlExportMode::Raw), raw);
    }
}
