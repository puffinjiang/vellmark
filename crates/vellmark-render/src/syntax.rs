//! Syntect-powered TextMate syntax highlighting and language catalog for fenced code blocks (PRD M-04).

use std::borrow::Cow;
use std::collections::HashSet;
use std::ops::Range;
use std::sync::OnceLock;

use syntect::easy::HighlightLines;
use syntect::highlighting::{FontStyle as SyntectFontStyle, Theme, ThemeSet};
use syntect::parsing::{SyntaxReference, SyntaxSet};
use syntect::util::LinesWithEndings;

use crate::style::RgbaColor;

/// Maximum raw code fence byte length for token-level highlighting so huge minified blobs
/// inside 10 MB Markdown files never stall frame shaping.
const MAX_HIGHLIGHT_BYTES: usize = 256 * 1024;
/// Maximum single line byte length for regex-based TextMate tokenization.
const MAX_HIGHLIGHT_LINE_BYTES: usize = 4096;

/// Highlighted token span relative to the code fence body slice.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CodeSyntaxSpan {
    pub range: Range<usize>,
    pub color: RgbaColor,
    pub bold: bool,
    pub italic: bool,
}

/// Entry exposed in the Code Block Language Selector UI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodeLanguageEntry {
    /// Fence token written after ```` ``` ```` (e.g., `"rust"`, `"python"`, or `""` for Plain Text).
    pub token: String,
    /// Human-readable display label (e.g., `"Rust"`, `"Python"`, `"Plain Text"`).
    pub label: String,
}

struct SyntaxEngine {
    syntax_set: SyntaxSet,
    theme_set: ThemeSet,
    catalog: Vec<CodeLanguageEntry>,
}

static SYNTAX_ENGINE: OnceLock<SyntaxEngine> = OnceLock::new();

fn engine() -> &'static SyntaxEngine {
    SYNTAX_ENGINE.get_or_init(|| {
        let syntax_set = SyntaxSet::load_defaults_newlines();
        let theme_set = ThemeSet::load_defaults();
        let catalog = build_language_catalog(&syntax_set);
        SyntaxEngine {
            syntax_set,
            theme_set,
            catalog,
        }
    })
}

/// Return the full catalog of supported fenced-code languages (`Plain Text` + 100+ tokens/syntaxes).
pub fn code_language_catalog() -> &'static [CodeLanguageEntry] {
    &engine().catalog
}

/// Normalize the primary language token from a Markdown code fence info string (`"rust,no_run"` -> `"rust"`).
pub fn normalize_fence_language_token(info_string: &str) -> String {
    let first_word = info_string.split_whitespace().next().unwrap_or("");
    let raw_tok = first_word
        .split(['{', ',', ';'])
        .next()
        .unwrap_or("")
        .trim();
    raw_tok.to_ascii_lowercase()
}

/// Display label rendered on the top-right badge of a fenced code block.
pub fn code_language_display_label(info_string: &str) -> String {
    let token = normalize_fence_language_token(info_string);
    if token.is_empty() {
        return "Plain Text".to_string();
    }
    if let Some(syn) = resolve_syntax_for_token(&engine().syntax_set, &token) {
        if syn.name == "Plain Text" {
            return "Plain Text".to_string();
        }
        return token;
    }
    token
}

/// Highlight `code` according to `info_string` using light/dark TextMate themes.
///
/// Returned `range` byte indices are exact UTF-8 byte offsets within `code`.
pub fn highlight_code_slice(
    code: &str,
    info_string: &str,
    is_dark: bool,
    fallback_color: RgbaColor,
    bg_color: RgbaColor,
) -> Vec<CodeSyntaxSpan> {
    if code.is_empty() || code.len() > MAX_HIGHLIGHT_BYTES {
        return Vec::new();
    }
    let token = normalize_fence_language_token(info_string);
    if token.is_empty() || matches!(token.as_str(), "plain" | "plaintext" | "text" | "txt") {
        return Vec::new();
    }

    let eng = engine();
    let Some(syntax) = resolve_syntax_for_token(&eng.syntax_set, &token) else {
        return Vec::new();
    };
    if syntax.name == "Plain Text" {
        return Vec::new();
    }

    let theme = select_theme(&eng.theme_set, is_dark);
    let mut highlighter = HighlightLines::new(syntax, theme);
    let mut out: Vec<CodeSyntaxSpan> = Vec::with_capacity(code.len() / 8 + 4);
    let mut line_base = 0usize;

    for raw_line in LinesWithEndings::from(code) {
        let raw_len = raw_line.len();
        if raw_len > MAX_HIGHLIGHT_LINE_BYTES {
            line_base += raw_len;
            continue;
        }
        let has_crlf = raw_line.ends_with("\r\n");
        let content_limit = if has_crlf { raw_len - 2 } else { raw_len };
        let normalized_line: Cow<'_, str> = if has_crlf {
            let mut s = String::with_capacity(raw_len);
            s.push_str(&raw_line[..content_limit]);
            s.push('\n');
            Cow::Owned(s)
        } else {
            Cow::Borrowed(raw_line)
        };

        let Ok(regions) = highlighter.highlight_line(normalized_line.as_ref(), &eng.syntax_set)
        else {
            line_base += raw_len;
            continue;
        };

        let mut cursor = 0usize;
        for (syn_style, slice) in regions {
            let seg_len = slice.len();
            let seg_start = cursor.min(content_limit);
            let seg_end = (cursor + seg_len).min(content_limit);
            cursor += seg_len;
            if seg_start >= seg_end {
                continue;
            }
            let span_start = line_base + seg_start;
            let span_end = line_base + seg_end;
            if !code.is_char_boundary(span_start) || !code.is_char_boundary(span_end) {
                continue;
            }

            let raw_color = RgbaColor::rgba(
                syn_style.foreground.r,
                syn_style.foreground.g,
                syn_style.foreground.b,
                syn_style.foreground.a,
            );
            let color = ensure_readable_contrast(raw_color, bg_color, fallback_color);
            let bold = syn_style.font_style.contains(SyntectFontStyle::BOLD);
            let italic = syn_style.font_style.contains(SyntectFontStyle::ITALIC);

            if let Some(prev) = out.last_mut()
                && prev.range.end == span_start
                && prev.color == color
                && prev.bold == bold
                && prev.italic == italic
            {
                prev.range.end = span_end;
                continue;
            }
            out.push(CodeSyntaxSpan {
                range: span_start..span_end,
                color,
                bold,
                italic,
            });
        }

        line_base += raw_len;
    }

    out
}

fn select_theme(theme_set: &ThemeSet, is_dark: bool) -> &Theme {
    let preferred = if is_dark {
        ["base16-ocean.dark", "base16-eighties.dark", "Solarized (dark)"]
    } else {
        ["InspiredGitHub", "Solarized (light)", "base16-ocean.light"]
    };
    for name in preferred {
        if let Some(theme) = theme_set.themes.get(name) {
            return theme;
        }
    }
    theme_set
        .themes
        .values()
        .next()
        .expect("syntect default ThemeSet always contains at least one theme")
}

fn ensure_readable_contrast(
    fg: RgbaColor,
    bg: RgbaColor,
    fallback: RgbaColor,
) -> RgbaColor {
    if fg.a < 16 {
        return fallback;
    }
    let fg_l = fg.luminance();
    let bg_l = bg.luminance();
    if (fg_l - bg_l).abs() < 0.22 {
        fg.lerp(fallback, 0.72).with_alpha(fg.a)
    } else {
        fg
    }
}

fn canonical_alias(token: &str) -> &str {
    match token {
        "rs" => "rust",
        "py" | "python3" => "python",
        "js" | "mjs" | "cjs" => "javascript",
        "ts" | "mts" | "cts" | "tsx" | "jsx" => "typescript",
        "sh" | "zsh" | "shell" | "console" => "bash",
        "yml" => "yaml",
        "md" => "markdown",
        "c++" | "cxx" | "cc" | "hpp" => "cpp",
        "c#" | "csharp" => "cs",
        "objc" | "objective-c" | "objectivec" => "mm",
        "dockerfile" => "Dockerfile",
        "makefile" => "Makefile",
        "toml" => "ini",
        _ => token,
    }
}

fn resolve_syntax_for_token<'a>(
    ss: &'a SyntaxSet,
    raw_token: &str,
) -> Option<&'a SyntaxReference> {
    let mapped = canonical_alias(raw_token);
    if let Some(syn) = ss
        .find_syntax_by_token(mapped)
        .or_else(|| ss.find_syntax_by_token(raw_token))
        .or_else(|| ss.find_syntax_by_name(mapped))
    {
        return Some(syn);
    }

    let lower = mapped.to_ascii_lowercase();
    ss.syntaxes().iter().find(|syn| {
        syn.name.to_ascii_lowercase() == lower
            || syn
                .file_extensions
                .iter()
                .any(|ext| ext.eq_ignore_ascii_case(&lower))
    })
}

fn build_language_catalog(ss: &SyntaxSet) -> Vec<CodeLanguageEntry> {
    let mut entries = Vec::new();
    let mut seen_tokens: HashSet<String> = HashSet::new();

    entries.push(CodeLanguageEntry {
        token: String::new(),
        label: "Plain Text".to_string(),
    });
    seen_tokens.insert(String::new());

    let popular = [
        ("rust", "Rust"),
        ("python", "Python"),
        ("javascript", "JavaScript"),
        ("typescript", "TypeScript"),
        ("bash", "Bash / Shell"),
        ("json", "JSON"),
        ("yaml", "YAML"),
        ("toml", "TOML"),
        ("markdown", "Markdown"),
        ("html", "HTML"),
        ("css", "CSS"),
        ("sql", "SQL"),
        ("c", "C"),
        ("cpp", "C++"),
        ("go", "Go"),
        ("java", "Java"),
        ("xml", "XML"),
        ("diff", "Diff"),
    ];
    for (tok, label) in popular {
        if seen_tokens.insert(tok.to_string()) {
            entries.push(CodeLanguageEntry {
                token: tok.to_string(),
                label: format!("{label} ({tok})"),
            });
        }
    }

    let mut rest: Vec<CodeLanguageEntry> = Vec::new();
    for syn in ss.syntaxes() {
        if syn.name == "Plain Text" {
            continue;
        }
        let tok = syn
            .file_extensions
            .first()
            .map(|e| e.to_ascii_lowercase())
            .unwrap_or_else(|| syn.name.to_ascii_lowercase().replace(' ', "-"));
        if !tok.is_empty() && seen_tokens.insert(tok.clone()) {
            rest.push(CodeLanguageEntry {
                label: format!("{} ({tok})", syn.name),
                token: tok,
            });
        }
    }
    rest.sort_by(|a, b| a.label.cmp(&b.label));
    entries.extend(rest);
    entries
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syntect_highlighting_rust_python_json_light_and_dark() {
        let rust_code = "fn main() {\n    let x = \"hello\"; // comment\n}\n";
        let bg_light = RgbaColor::rgb(246, 248, 250);
        let bg_dark = RgbaColor::rgb(30, 34, 42);
        let fg_light = RgbaColor::rgb(36, 41, 47);
        let fg_dark = RgbaColor::rgb(212, 212, 212);

        let light_spans = highlight_code_slice(rust_code, "rust", false, fg_light, bg_light);
        let dark_spans = highlight_code_slice(rust_code, "rs", true, fg_dark, bg_dark);
        assert!(
            light_spans.len() >= 4,
            "expected multi-token syntax highlights, got {:?}",
            light_spans
        );
        assert!(dark_spans.len() >= 4);

        let distinct_colors: HashSet<RgbaColor> =
            light_spans.iter().map(|s| s.color).collect();
        assert!(
            distinct_colors.len() >= 2,
            "expected distinct token colors for keywords/strings/comments"
        );
        assert!(
            light_spans
                .iter()
                .all(|s| rust_code.is_char_boundary(s.range.start)
                    && rust_code.is_char_boundary(s.range.end))
        );

        // CRLF + UTF-8 preservation
        let py_crlf = "# 中文注释\r\ndef greet():\r\n    return \"你好\"\r\n";
        let py_spans = highlight_code_slice(py_crlf, "python,ignore", false, fg_light, bg_light);
        assert!(!py_spans.is_empty());
        assert!(
            py_spans
                .iter()
                .all(|s| py_crlf.is_char_boundary(s.range.start)
                    && py_crlf.is_char_boundary(s.range.end))
        );

        // Language catalog includes syntect built-ins + aliases
        let catalog = code_language_catalog();
        assert!(
            catalog.len() >= 40,
            "expected rich language catalog, got {}",
            catalog.len()
        );
        assert_eq!(code_language_display_label(""), "Plain Text");
        assert_eq!(code_language_display_label("rust,no_run"), "rust");
    }
}
