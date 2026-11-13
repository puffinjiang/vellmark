use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use vellmark_core::detect_and_decode;
use vellmark_export::{HtmlExportMode, export_markdown_to_html_document};
use vellmark_theme::BuiltinTheme;

fn print_help() {
    println!("vellmark-cli v{}", env!("CARGO_PKG_VERSION"));
    println!("Usage:");
    println!("  vellmark <file.md>                                  Inspect/open Markdown file");
    println!("  vellmark export <file.md> -o <out.html> [--raw]     Export Markdown to HTML");
    println!("  vellmark export <file.md> -o <out.html> --theme <light|dark|academic|github|theme.css>");
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return ExitCode::SUCCESS;
    }

    if args[0] == "export" {
        return run_export(&args[1..]);
    }

    // Open/inspect file mode
    let path = PathBuf::from(&args[0]);
    match fs::read(&path) {
        Ok(raw) => match detect_and_decode(&raw) {
            Ok((text, meta)) => {
                println!(
                    "Loaded {} ({} bytes, encoding: {:?}, line_ending: {:?})",
                    path.display(),
                    text.len(),
                    meta.decision,
                    meta.dominant_line_ending
                );
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("Error opening {}: {e}", path.display());
                ExitCode::FAILURE
            }
        },
        Err(e) => {
            eprintln!("Failed to read {}: {e}", path.display());
            ExitCode::FAILURE
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct ExportOptions {
    input_path: PathBuf,
    output_path: Option<PathBuf>,
    theme_spec: Option<PathBuf>,
    mode: HtmlExportMode,
}

fn parse_export_args(args: &[String]) -> Result<ExportOptions, String> {
    if args.is_empty() {
        return Err("Missing input file for `export` command.".to_string());
    }
    let input_path = PathBuf::from(&args[0]);
    let mut output_path: Option<PathBuf> = None;
    let mut theme_spec: Option<PathBuf> = None;
    let mut mode = HtmlExportMode::Safe;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => {
                let val = args
                    .get(i + 1)
                    .ok_or_else(|| format!("Missing path after `{}`", args[i]))?;
                output_path = Some(PathBuf::from(val));
                i += 2;
            }
            "--theme" => {
                let val = args
                    .get(i + 1)
                    .ok_or_else(|| "Missing theme name or path after `--theme`".to_string())?;
                theme_spec = Some(PathBuf::from(val));
                i += 2;
            }
            "--raw" => {
                mode = HtmlExportMode::Raw;
                i += 1;
            }
            unknown => {
                return Err(format!("Unknown export option: {unknown}"));
            }
        }
    }

    Ok(ExportOptions {
        input_path,
        output_path,
        theme_spec,
        mode,
    })
}

fn resolve_builtin_theme(name: &str) -> Option<BuiltinTheme> {
    match name.to_ascii_lowercase().as_str() {
        "light" | "default-light" => Some(BuiltinTheme::Light),
        "dark" | "default-dark" => Some(BuiltinTheme::Dark),
        "academic" | "academic-newsprint" => Some(BuiltinTheme::Academic),
        "github" => Some(BuiltinTheme::GitHub),
        _ => None,
    }
}

fn resolve_theme_css(spec: &Path) -> Result<String, String> {
    if !spec.exists() {
        if let Some(name) = spec.to_str() {
            if spec.components().count() == 1 {
                if let Some(builtin) = resolve_builtin_theme(name) {
                    return Ok(builtin.css_source().to_string());
                }
            }
        }
    }
    fs::read_to_string(spec).map_err(|e| format!("Failed to read theme {}: {e}", spec.display()))
}

fn export_document(
    input_path: &Path,
    theme_spec: Option<&Path>,
    mode: HtmlExportMode,
) -> Result<String, String> {
    let raw = fs::read(input_path)
        .map_err(|e| format!("Failed to read {}: {e}", input_path.display()))?;
    let (markdown, _) = detect_and_decode(&raw)
        .map_err(|e| format!("Encoding error in {}: {e}", input_path.display()))?;

    let theme_css = match theme_spec {
        Some(spec) => Some(resolve_theme_css(spec)?),
        None => None,
    };

    let title = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Document");
    Ok(export_markdown_to_html_document(
        title,
        &markdown,
        theme_css.as_deref(),
        mode,
    ))
}

fn run_export(args: &[String]) -> ExitCode {
    let opts = match parse_export_args(args) {
        Ok(o) => o,
        Err(msg) => {
            eprintln!("{msg}");
            return ExitCode::FAILURE;
        }
    };

    let html_doc = match export_document(&opts.input_path, opts.theme_spec.as_deref(), opts.mode) {
        Ok(html) => html,
        Err(msg) => {
            eprintln!("{msg}");
            return ExitCode::FAILURE;
        }
    };

    if let Some(out) = opts.output_path {
        if let Err(e) = fs::write(&out, html_doc) {
            eprintln!("Failed to write {}: {e}", out.display());
            return ExitCode::FAILURE;
        }
        println!("Exported {} -> {}", opts.input_path.display(), out.display());
    } else {
        print!("{html_doc}");
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn test_parse_export_args_supports_flags_and_reports_errors() {
        let args = vec![
            "note.md".to_string(),
            "-o".to_string(),
            "out.html".to_string(),
            "--theme".to_string(),
            "dark".to_string(),
            "--raw".to_string(),
        ];
        let opts = parse_export_args(&args).expect("parse export args");
        assert_eq!(opts.input_path, PathBuf::from("note.md"));
        assert_eq!(opts.output_path, Some(PathBuf::from("out.html")));
        assert_eq!(opts.theme_spec, Some(PathBuf::from("dark")));
        assert_eq!(opts.mode, HtmlExportMode::Raw);

        assert!(parse_export_args(&[]).is_err());
        assert!(parse_export_args(&["note.md".to_string(), "--theme".to_string()]).is_err());
        assert!(parse_export_args(&["note.md".to_string(), "--unknown".to_string()]).is_err());
    }

    #[test]
    fn test_resolve_theme_css_builtin_aliases_and_custom_file() {
        for (alias, expected) in [
            ("light", BuiltinTheme::Light),
            ("default-light", BuiltinTheme::Light),
            ("dark", BuiltinTheme::Dark),
            ("academic", BuiltinTheme::Academic),
            ("github", BuiltinTheme::GitHub),
        ] {
            let css = resolve_theme_css(Path::new(alias)).expect("resolve builtin theme");
            assert_eq!(css, expected.css_source());
        }

        let uniq = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "vellmark-cli-test-{}-{uniq}",
            std::process::id()
        ));
        fs::create_dir_all(&root).expect("create temp dir");

        let custom_css_path = root.join("custom.css");
        fs::write(&custom_css_path, "body { color: #123456; }").expect("write custom css");
        let resolved_custom = resolve_theme_css(&custom_css_path).expect("read custom css");
        assert!(resolved_custom.contains("#123456"));

        let missing = root.join("does-not-exist.css");
        assert!(resolve_theme_css(&missing).is_err());

        let md_path = root.join("测试文档.md");
        fs::write(&md_path, "# 标题\n\n<script>alert(1)</script>\n正文 **加粗**").expect("write md");

        let safe_html = export_document(&md_path, Some(Path::new("dark")), HtmlExportMode::Safe)
            .expect("safe export");
        assert!(safe_html.contains("<title>测试文档</title>"));
        assert!(safe_html.contains("<strong>加粗</strong>"));
        assert!(!safe_html.contains("<script>"));
        assert!(safe_html.contains(BuiltinTheme::Dark.css_source().lines().next().unwrap_or("")));

        let raw_html = export_document(&md_path, Some(&custom_css_path), HtmlExportMode::Raw)
            .expect("raw export");
        assert!(raw_html.contains("<script>alert(1)</script>"));
        assert!(raw_html.contains("#123456"));

        let _ = fs::remove_dir_all(&root);
    }
}
