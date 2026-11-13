pub mod assets;
pub mod config;
pub mod search;
pub mod workbench;

use std::path::PathBuf;
use std::process::ExitCode;

use config::AppConfig;
use workbench::AppWorkbench;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut workbench = AppWorkbench::new(AppConfig::default(), None);

    if let Some(first) = args.first() {
        let path = PathBuf::from(first);
        if path.is_dir() {
            if let Err(e) = workbench.open_folder(&path) {
                eprintln!("Failed to open folder {}: {e}", path.display());
                return ExitCode::FAILURE;
            }
            println!(
                "Vellmark v{} — opened folder {} ({} top-level items)",
                env!("CARGO_PKG_VERSION"),
                path.display(),
                workbench
                    .file_tree
                    .as_ref()
                    .map(|t| t.children.len())
                    .unwrap_or(0)
            );
            return ExitCode::SUCCESS;
        }
        if path.exists() {
            match workbench.open_document(&path) {
                Ok(_) => {
                    let stats = workbench.active_tab().unwrap().stats();
                    println!(
                        "Vellmark v{} — opened {} ({} words, {} chars, {} lines)",
                        env!("CARGO_PKG_VERSION"),
                        path.display(),
                        stats.word_count,
                        stats.char_count,
                        stats.line_count
                    );
                    return ExitCode::SUCCESS;
                }
                Err(e) => {
                    eprintln!("Failed to open {}: {e}", path.display());
                    return ExitCode::FAILURE;
                }
            }
        }
    }

    println!(
        "Vellmark v{} — Native Rust WYSIWYG Markdown Editor",
        env!("CARGO_PKG_VERSION")
    );
    ExitCode::SUCCESS
}
