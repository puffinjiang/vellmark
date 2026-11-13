use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use vellmark_core::detect_and_decode;
use vellmark_export::{HtmlExportMode, export_markdown_to_html_document};

fn print_help() {
    println!("vellmark-cli v{}", env!("CARGO_PKG_VERSION"));
    println!("Usage:");
    println!("  vellmark <file.md>                                  Inspect/open Markdown file");
    println!("  vellmark export <file.md> -o <out.html> [--raw]     Export Markdown to HTML");
    println!("  vellmark export <file.md> -o <out.html> --theme <theme.css>");
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

fn run_export(args: &[String]) -> ExitCode {
    if args.is_empty() {
        eprintln!("Missing input file for `export` command.");
        return ExitCode::FAILURE;
    }
    let input_path = PathBuf::from(&args[0]);
    let mut output_path: Option<PathBuf> = None;
    let mut theme_path: Option<PathBuf> = None;
    let mut mode = HtmlExportMode::Safe;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => {
                if i + 1 < args.len() {
                    output_path = Some(PathBuf::from(&args[i + 1]));
                    i += 2;
                    continue;
                }
            }
            "--theme" => {
                if i + 1 < args.len() {
                    theme_path = Some(PathBuf::from(&args[i + 1]));
                    i += 2;
                    continue;
                }
            }
            "--raw" => {
                mode = HtmlExportMode::Raw;
            }
            _ => {}
        }
        i += 1;
    }

    let raw = match fs::read(&input_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Failed to read {}: {e}", input_path.display());
            return ExitCode::FAILURE;
        }
    };
    let (markdown, _) = match detect_and_decode(&raw) {
        Ok(res) => res,
        Err(e) => {
            eprintln!("Encoding error in {}: {e}", input_path.display());
            return ExitCode::FAILURE;
        }
    };

    let theme_css = match &theme_path {
        Some(tp) => match fs::read_to_string(tp) {
            Ok(css) => Some(css),
            Err(e) => {
                eprintln!("Failed to read theme {}: {e}", tp.display());
                return ExitCode::FAILURE;
            }
        },
        None => None,
    };

    let title = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Document");
    let html_doc = export_markdown_to_html_document(title, &markdown, theme_css.as_deref(), mode);

    if let Some(out) = output_path {
        if let Err(e) = fs::write(&out, html_doc) {
            eprintln!("Failed to write {}: {e}", out.display());
            return ExitCode::FAILURE;
        }
        println!("Exported {} -> {}", input_path.display(), out.display());
    } else {
        print!("{html_doc}");
    }
    ExitCode::SUCCESS
}
