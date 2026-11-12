//! `vellmark-export` implements HTML export in both Raw mode and Safe mode (PRD 4.2.1 & X-01).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HtmlExportMode {
    /// Raw mode: preserves inline/block HTML verbatim for spec compliance testing (PRD 4.2.1).
    Raw,
    /// Default Safe mode: strips `<script>`, `on*` event attributes, and `javascript:` / `vbscript:` URLs (PRD 4.2.1 & 5.5).
    Safe,
}

pub fn sanitize_html_fragment(html: &str, mode: HtmlExportMode) -> String {
    match mode {
        HtmlExportMode::Raw => html.to_string(),
        HtmlExportMode::Safe => {
            let without_scripts = strip_script_elements(html);
            sanitize_html_tags(&without_scripts)
        }
    }
}

fn strip_script_elements(html: &str) -> String {
    let mut out = html.to_string();
    let mut search_from = 0usize;
    while search_from < out.len() {
        let Some(rel_start) = out[search_from..].to_ascii_lowercase().find("<script") else {
            break;
        };
        let start = search_from + rel_start;
        // Ensure `<script` is followed by whitespace, `>`, or `/` (or end of string),
        // rather than a longer tag name prefix like `<scripture>`.
        let after = out.as_bytes().get(start + 7).copied();
        if matches!(
            after,
            None | Some(b'>') | Some(b'/') | Some(b' ') | Some(b'\t') | Some(b'\n') | Some(b'\r')
        ) {
            // Find matching closing `</script>` where `</script` is followed by whitespace or `>`
            let mut close_search = start + 7;
            let mut removed = false;
            while close_search < out.len() {
                if let Some(end_rel) = out[close_search..].to_ascii_lowercase().find("</script") {
                    let close_pos = close_search + end_rel;
                    let after_close = out.as_bytes().get(close_pos + 8).copied();
                    if matches!(
                        after_close,
                        Some(b'>') | Some(b' ') | Some(b'\t') | Some(b'\n') | Some(b'\r')
                    ) {
                        if let Some(close_gt) = out[close_pos + 8..].find('>') {
                            out.replace_range(start..close_pos + 8 + close_gt + 1, "");
                            removed = true;
                            break;
                        }
                    }
                    close_search = close_pos + 8;
                } else {
                    break;
                }
            }
            if !removed {
                out.replace_range(start.., "");
                break;
            }
        } else {
            // Not a `<script>` tag (e.g. `<scripture>`): advance `search_from` and continue scanning!
            search_from = start + 7;
        }
    }
    out
}

/// Scans HTML tags and removes:
/// - Any `on*` event handler attribute (`onclick`, `onerror`, `onload`, etc., case-insensitive)
/// - Any `href`, `src`, `xlink:href`, `action`, or `formaction` attribute whose value starts with
///   `javascript:` or `vbscript:` (ignoring leading whitespace, control chars, and case).
fn sanitize_html_tags(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let bytes = input.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i] == b'<' {
            // Find matching '>' respecting quoted attribute values
            if let Some(tag_end) = find_tag_end(input, i) {
                let tag_slice = &input[i..=tag_end];
                out.push_str(&sanitize_single_tag(tag_slice));
                i = tag_end + 1;
                continue;
            }
        }
        let ch = input[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }

    out
}

fn find_tag_end(input: &str, start: usize) -> Option<usize> {
    let bytes = input.as_bytes();
    let mut i = start + 1;
    let mut quote: Option<u8> = None;
    while i < bytes.len() {
        let b = bytes[i];
        if let Some(q) = quote {
            if b == q {
                quote = None;
            }
        } else if b == b'"' || b == b'\'' {
            quote = Some(b);
        } else if b == b'>' {
            return Some(i);
        }
        i += 1;
    }
    None
}

fn sanitize_single_tag(tag: &str) -> String {
    // Pass through comments, declarations, or closing tags unchanged
    if tag.starts_with("</") || tag.starts_with("<!") || tag.starts_with("<?") {
        return tag.to_string();
    }

    let self_closing = tag.ends_with("/>");
    let inner_end = if self_closing {
        tag.len() - 2
    } else {
        tag.len() - 1
    };
    let inner = &tag[1..inner_end];

    // Split tag name and attribute region
    let mut chars = inner.char_indices();
    let mut split_idx = inner.len();
    for (idx, ch) in &mut chars {
        if ch.is_ascii_whitespace() {
            split_idx = idx;
            break;
        }
    }

    let tag_name = &inner[..split_idx];
    let attr_region = &inner[split_idx..];

    let mut out = String::with_capacity(tag.len());
    out.push('<');
    out.push_str(tag_name);

    let bytes = attr_region.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        // Skip whitespace
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }

        // Parse attribute name
        let name_start = i;
        while i < bytes.len()
            && !bytes[i].is_ascii_whitespace()
            && bytes[i] != b'='
            && bytes[i] != b'>'
            && bytes[i] != b'/'
        {
            i += 1;
        }
        let attr_name = &attr_region[name_start..i];
        if attr_name.is_empty() {
            i += 1;
            continue;
        }

        // Check for optional `= value`
        let mut ws_after_name = i;
        while ws_after_name < bytes.len() && bytes[ws_after_name].is_ascii_whitespace() {
            ws_after_name += 1;
        }

        let mut val_raw = "";
        let mut unquoted_val = "";
        if ws_after_name < bytes.len() && bytes[ws_after_name] == b'=' {
            i = ws_after_name + 1;
            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            let val_start = i;
            if i < bytes.len() && (bytes[i] == b'"' || bytes[i] == b'\'') {
                let q = bytes[i];
                i += 1;
                let content_start = i;
                while i < bytes.len() && bytes[i] != q {
                    i += 1;
                }
                unquoted_val = &attr_region[content_start..i];
                if i < bytes.len() {
                    i += 1; // include closing quote
                }
                val_raw = &attr_region[val_start..i];
            } else {
                while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
                    i += 1;
                }
                unquoted_val = &attr_region[val_start..i];
                val_raw = unquoted_val;
            }
        } else {
            i = ws_after_name;
        }

        let lower_name = attr_name.to_ascii_lowercase();

        // 1. Strip any `on*` event handler attribute (PRD 4.2.1)
        if lower_name.starts_with("on") && lower_name.len() > 2 {
            continue;
        }

        // 2. Strip dangerous `javascript:` / `vbscript:` URLs on URL-bearing attributes
        if matches!(
            lower_name.as_str(),
            "href" | "src" | "xlink:href" | "action" | "formaction"
        ) && is_dangerous_url(unquoted_val)
        {
            continue;
        }

        out.push(' ');
        out.push_str(attr_name);
        if !val_raw.is_empty() {
            out.push('=');
            out.push_str(val_raw);
        }
    }

    if self_closing {
        out.push_str(" />");
    } else {
        out.push('>');
    }
    out
}

fn is_dangerous_url(val: &str) -> bool {
    let cleaned: String = val
        .chars()
        .filter(|c| !c.is_ascii_whitespace() && !c.is_ascii_control())
        .collect::<String>()
        .to_ascii_lowercase();
    cleaned.starts_with("javascript:") || cleaned.starts_with("vbscript:")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_mode_strips_script_event_handlers_and_javascript_urls() {
        let raw = r#"<p onclick="evil()" ONMOUSEOVER='x()' class="ok">Hello</p><script>alert(1)</script><img src="x.png" onerror=alert(1) /><a href="  JaVaScRiPt:evil()" title="link">x</a>"#;
        let safe = sanitize_html_fragment(raw, HtmlExportMode::Safe);
        assert!(!safe.contains("<script"));
        assert!(!safe.to_ascii_lowercase().contains("onclick"));
        assert!(!safe.to_ascii_lowercase().contains("onmouseover"));
        assert!(!safe.to_ascii_lowercase().contains("onerror"));
        assert!(!safe.to_ascii_lowercase().contains("javascript:"));
        assert!(safe.contains(r#"class="ok""#));
        assert!(safe.contains(r#"src="x.png""#));
        assert!(safe.contains(r#"title="link""#));
        assert_eq!(sanitize_html_fragment(raw, HtmlExportMode::Raw), raw);
    }

    #[test]
    fn test_script_prefix_tag_like_scripture_does_not_bypass_subsequent_script_removal() {
        let raw = "<scripture>keep me</scripture><script>alert(1)</script><p>after</p>";
        let safe = sanitize_html_fragment(raw, HtmlExportMode::Safe);
        assert_eq!(safe, "<scripture>keep me</scripture><p>after</p>");
    }
}
