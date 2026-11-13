//! `vellmark-export` implements HTML export in both Raw mode and Safe mode (PRD 4.2.1 & X-01),
//! including optional embedded theme CSS (PRD S-04).

use vellmark_parser::render_markdown_to_html;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HtmlExportMode {
    /// Raw mode: preserves inline/block HTML verbatim for spec compliance testing (PRD 4.2.1).
    Raw,
    /// Default Safe mode: strips `<script>` / `<iframe>` / `<object>` / `<embed>`, `on*` event attributes,
    /// `srcdoc` attributes, and `javascript:` / `vbscript:` / `data:text/html` URLs (including HTML entity encoded forms) (PRD 4.2.1 & 5.5).
    Safe,
}

/// Export Markdown source to an HTML fragment (Raw or Safe mode, PRD 4.2.1).
pub fn export_markdown_to_html_fragment(markdown: &str, mode: HtmlExportMode) -> String {
    let raw_html = render_markdown_to_html(markdown);
    sanitize_html_fragment(&raw_html, mode)
}

/// Export Markdown source to a standalone HTML document with optional embedded theme CSS (PRD X-01 & S-04).
/// Escapes the `<title>` text and neutralizes any `</style` sequence in `theme_css` so embedded styles
/// can never break out of the `<style>` container and inject `<script>` tags (P1 #3 fix).
pub fn export_markdown_to_html_document(
    title: &str,
    markdown: &str,
    theme_css: Option<&str>,
    mode: HtmlExportMode,
) -> String {
    let body = export_markdown_to_html_fragment(markdown, mode);
    let safe_title = escape_html_text(title);
    let style_block = match theme_css {
        Some(css) => {
            let safe_css = sanitize_embedded_style_css(css);
            format!("<style>\n{safe_css}\n</style>\n")
        }
        None => String::new(),
    };
    format!(
        "<!DOCTYPE html>\n<html>\n<head>\n<meta charset=\"utf-8\">\n<title>{safe_title}</title>\n{style_block}</head>\n<body>\n<div id=\"write\">\n{body}</div>\n</body>\n</html>\n"
    )
}

fn escape_html_text(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(ch),
        }
    }
    out
}

/// Neutralize any case-insensitive `</style` closing tag sequence inside embedded CSS so a theme file
/// cannot break out of `<style>...</style>` and inject active HTML/scripts (P1 #3 fix).
pub fn sanitize_embedded_style_css(css: &str) -> String {
    let mut out = css.to_string();
    while let Some(pos) = out.to_ascii_lowercase().find("</style") {
        out.replace_range(pos..pos + 7, "<\\/style");
    }
    out
}

pub fn sanitize_html_fragment(html: &str, mode: HtmlExportMode) -> String {
    match mode {
        HtmlExportMode::Raw => html.to_string(),
        HtmlExportMode::Safe => {
            let mut cleaned = html.to_string();
            for forbidden_tag in ["script", "iframe", "object", "embed", "applet"] {
                cleaned = strip_forbidden_element(&cleaned, forbidden_tag);
            }
            sanitize_html_tags(&cleaned)
        }
    }
}

fn strip_forbidden_element(html: &str, tag_lower: &str) -> String {
    let open_pat = format!("<{tag_lower}");
    let close_pat = format!("</{tag_lower}");
    let mut out = html.to_string();
    let mut search_from = 0usize;

    while search_from < out.len() {
        let Some(rel_start) = out[search_from..].to_ascii_lowercase().find(&open_pat) else {
            break;
        };
        let start = search_from + rel_start;
        let after = out.as_bytes().get(start + open_pat.len()).copied();
        if matches!(
            after,
            None | Some(b'>') | Some(b'/') | Some(b' ') | Some(b'\t') | Some(b'\n') | Some(b'\r')
        ) {
            // Check if self-closing `<iframe ... />` or `<embed ...>` without closing tag
            if let Some(open_tag_end) = find_tag_end(&out, start) {
                let open_slice = &out[start..=open_tag_end];
                if open_slice.ends_with("/>") || tag_lower == "embed" {
                    out.replace_range(start..=open_tag_end, "");
                    continue;
                }
            }

            let mut close_search = start + open_pat.len();
            let mut removed = false;
            while close_search < out.len() {
                if let Some(end_rel) = out[close_search..].to_ascii_lowercase().find(&close_pat) {
                    let close_pos = close_search + end_rel;
                    let after_close = out.as_bytes().get(close_pos + close_pat.len()).copied();
                    if matches!(
                        after_close,
                        Some(b'>') | Some(b' ') | Some(b'\t') | Some(b'\n') | Some(b'\r')
                    ) {
                        if let Some(close_gt) = out[close_pos + close_pat.len()..].find('>') {
                            out.replace_range(start..close_pos + close_pat.len() + close_gt + 1, "");
                            removed = true;
                            break;
                        }
                    }
                    close_search = close_pos + close_pat.len();
                } else {
                    break;
                }
            }
            if !removed {
                out.replace_range(start.., "");
                break;
            }
        } else {
            // Prefix of a longer valid tag name (e.g. `<scripture>`): advance `search_from` and continue scanning
            search_from = start + open_pat.len();
        }
    }
    out
}

/// Scans HTML tags and removes:
/// - Any `on*` event handler attribute (`onclick`, `onerror`, `onload`, etc., case-insensitive)
/// - Any `srcdoc` attribute
/// - Any `href`, `src`, `xlink:href`, `action`, `formaction`, or `data` attribute whose value
///   (after decoding numeric/named HTML character references and stripping whitespace/control chars)
///   starts with `javascript:`, `vbscript:`, or `data:text/html`.
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

        // 1. Strip any `on*` event handler attribute or `srcdoc` attribute (PRD 4.2.1 & P1 #4)
        if (lower_name.starts_with("on") && lower_name.len() > 2) || lower_name == "srcdoc" {
            continue;
        }

        // 2. Strip dangerous `javascript:` / `vbscript:` / `data:text/html` URLs (including HTML entity encoded forms)
        if matches!(
            lower_name.as_str(),
            "href" | "src" | "xlink:href" | "action" | "formaction" | "data"
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

/// Decodes numeric (`&#106;`, `&#x6a;`, with or without trailing semicolon) and common named HTML entities,
/// strips whitespace and control characters, and checks for dangerous URL schemes (`javascript:`, `vbscript:`, `data:text/html`).
fn is_dangerous_url(val: &str) -> bool {
    let decoded = decode_html_entities(val);
    let cleaned: String = decoded
        .chars()
        .filter(|c| !c.is_ascii_whitespace() && !c.is_ascii_control() && *c != '\0')
        .collect::<String>()
        .to_ascii_lowercase();
    cleaned.starts_with("javascript:")
        || cleaned.starts_with("vbscript:")
        || cleaned.starts_with("data:text/html")
}

fn decode_html_entities(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let bytes = input.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i] == b'&' {
            if let Some((ch, consumed)) = parse_entity_at(&input[i..]) {
                out.push(ch);
                i += consumed;
                continue;
            }
        }
        let ch = input[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

fn parse_entity_at(s: &str) -> Option<(char, usize)> {
    // s starts with '&'
    let rest = s.strip_prefix('&')?;
    if let Some(hex_rest) = rest.strip_prefix("#x").or_else(|| rest.strip_prefix("#X")) {
        let mut digits = 0;
        while digits < hex_rest.len() && hex_rest.as_bytes()[digits].is_ascii_hexdigit() {
            digits += 1;
        }
        if digits > 0 {
            let code = u32::from_str_radix(&hex_rest[..digits], 16).ok()?;
            let ch = char::from_u32(code).unwrap_or('\0');
            let semi = usize::from(hex_rest.as_bytes().get(digits) == Some(&b';'));
            return Some((ch, 1 + 2 + digits + semi));
        }
    } else if let Some(dec_rest) = rest.strip_prefix('#') {
        let mut digits = 0;
        while digits < dec_rest.len() && dec_rest.as_bytes()[digits].is_ascii_digit() {
            digits += 1;
        }
        if digits > 0 {
            let code = dec_rest[..digits].parse::<u32>().ok()?;
            let ch = char::from_u32(code).unwrap_or('\0');
            let semi = usize::from(dec_rest.as_bytes().get(digits) == Some(&b';'));
            return Some((ch, 1 + 1 + digits + semi));
        }
    } else {
        // Named entities relevant to scheme obfuscation (`&colon;`, `&tab;`, `&newline;`).
        // Compare on `rest.as_bytes()` first so we never slice inside a multi-byte UTF-8 character (e.g. `&你好`)!
        for (name, ch) in [
            (b"colon;".as_slice(), ':'),
            (b"tab;".as_slice(), '\t'),
            (b"newline;".as_slice(), '\n'),
            (b"amp;".as_slice(), '&'),
        ] {
            if rest.as_bytes().len() >= name.len()
                && rest.as_bytes()[..name.len()].eq_ignore_ascii_case(name)
            {
                return Some((ch, 1 + name.len()));
            }
        }
    }
    None
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

    #[test]
    fn test_p1_theme_css_cannot_break_out_of_style_tag() {
        // P1 #3: Malicious theme CSS containing `</style><script>alert(1)</script>` must not break out of <style>.
        let malicious_css = "body { color: red; } </STYLE><script>alert(1)</script>";
        let doc = export_markdown_to_html_document(
            "Title </title><script>",
            "# Hello",
            Some(malicious_css),
            HtmlExportMode::Safe,
        );
        assert!(!doc.contains("</STYLE><script>"));
        assert!(!doc.contains("</title><script>"));
        assert!(doc.contains("<\\/style>"));
    }

    #[test]
    fn test_p1_entity_encoded_javascript_url_and_iframe_srcdoc_are_stripped() {
        // P1 #4: Entity-encoded `javascript:` (`&#106;avascript:alert(1)`, `java&#x09;script&colon;alert(1)`)
        // and `<iframe srcdoc="<script>alert(1)</script>"></iframe>` must be stripped in Safe mode.
        let raw = r#"<a href="&#106;&#x61;vascript&colon;alert(1)" title="t">link</a><iframe srcdoc="<script>alert(1)</script>"></iframe><object data="javascript:alert(1)"></object>"#;
        let safe = sanitize_html_fragment(raw, HtmlExportMode::Safe);
        assert!(!safe.contains("alert(1)"));
        assert!(!safe.to_ascii_lowercase().contains("<iframe"));
        assert!(!safe.to_ascii_lowercase().contains("<object"));
        assert!(!safe.to_ascii_lowercase().contains("href="));
        assert!(safe.contains(r#"title="t""#));
    }

    #[test]
    fn test_p2_non_ascii_url_after_ampersand_does_not_panic_in_safe_export() {
        // P2 #3: Safe export containing `href="https://example.com/?q=&你好"` must not panic
        // when checking named entities after `&`.
        let raw = r#"<a href="https://example.com/?q=&你好">中文链接</a>"#;
        let safe = sanitize_html_fragment(raw, HtmlExportMode::Safe);
        assert_eq!(safe, raw);
    }
}
