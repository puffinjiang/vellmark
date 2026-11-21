pub mod assets;
pub mod clipboard;
pub mod config;
pub mod gpu;
pub mod gui;
pub mod images;
pub mod search;
pub mod workbench;

use std::path::{Path, PathBuf};

pub use config::default_recovery_root;
use config::{AppConfig, default_config_path, default_recent_files_path};
use gui::{GuiController, PROTECTED_SNAPSHOT_LIMIT_NOTICE, WELCOME_MARKDOWN, VellmarkApp};
use vellmark_core::{RecentFilesList, Selection};
use workbench::AppWorkbench;

fn build_initial_workbench_with_recovery(
    paths: &[PathBuf],
    recovery_root: Option<PathBuf>,
) -> (AppWorkbench, usize) {
    build_initial_workbench_with_persistence(paths, recovery_root, None, None)
}

fn build_initial_workbench_with_persistence(
    paths: &[PathBuf],
    recovery_root: Option<PathBuf>,
    config_path: Option<PathBuf>,
    recent_files_path: Option<PathBuf>,
) -> (AppWorkbench, usize) {
    let config = config_path
        .as_deref()
        .and_then(|p| AppConfig::load_from_file(p).ok())
        .unwrap_or_default();
    let mut workbench = AppWorkbench::new(config, recovery_root);
    workbench.restore_configured_theme();
    if let Some(recent_path) = recent_files_path.as_deref()
        && let Ok(loaded) = RecentFilesList::load_from_file(recent_path, workbench.recent_files.max_entries)
    {
        workbench.recent_files = loaded;
        workbench.recent_files.prune_missing();
    }
    workbench.set_persistence_paths(config_path, recent_files_path);

    for path in paths {
        if path.is_dir() {
            if let Err(err) = workbench.open_folder(path) {
                eprintln!("vellmark: failed to open folder {}: {err}", path.display());
            }
        } else if let Err(err) = workbench.open_document(path) {
            eprintln!("vellmark: failed to open file {}: {err}", path.display());
        }
    }

    let recovered = workbench.restore_pending_recoveries();

    if workbench.tabs.is_empty() {
        workbench.new_document();
        if let Some(tab) = workbench.active_tab_mut() {
            tab.session.buffer.reset_content(WELCOME_MARKDOWN);
            tab.view.selection = Selection::caret(0);
        }
    }

    (workbench, recovered)
}

fn run_headless(paths: &[PathBuf]) {
    let (mut workbench, _) = build_initial_workbench_with_recovery(paths, None);
    if workbench.workspace_root.is_none()
        && let Ok(cwd) = std::env::current_dir()
            && Path::new(&cwd).exists() {
                let _ = workbench.open_folder(cwd);
            }

    let stats = workbench.active_tab().map(|t| t.stats());
    println!(
        "vellmark headless ready (tabs={}, active={:?}, stats={:?})",
        workbench.tabs.len(),
        workbench.active_tab().map(|t| t.title()),
        stats.map(|s| (s.word_count, s.char_count, s.line_count)),
    );
}

fn main() {
    let raw_args: Vec<String> = std::env::args().skip(1).collect();
    let mut headless = false;
    let mut paths = Vec::new();
    for arg in &raw_args {
        match arg.as_str() {
            "--headless" => headless = true,
            "--help" | "-h" => {
                println!(
                    "Vellmark Desktop Markdown Editor\n\nUSAGE:\n  vellmark-app [OPTIONS] [FILES_OR_FOLDERS...]\n\nOPTIONS:\n      --headless    Print workspace inspection summary without launching GUI window\n  -h, --help        Print help"
                );
                return;
            }
            other => paths.push(PathBuf::from(other)),
        }
    }

    if headless {
        run_headless(&paths);
        return;
    }

    let (workbench, recovered) = build_initial_workbench_with_persistence(
        &paths,
        Some(default_recovery_root()),
        Some(default_config_path()),
        Some(default_recent_files_path()),
    );
    let protected_warn = workbench.protected_snapshots_exceed_limit();
    let mut controller = GuiController::new(workbench);
    if protected_warn {
        controller.set_notice(PROTECTED_SNAPSHOT_LIMIT_NOTICE);
    } else if recovered > 0 {
        controller.set_notice(format!("Recovered {recovered} unsaved document(s)"));
    }
    if let Err(err) = VellmarkApp::run(controller) {
        eprintln!("vellmark GUI error: {err}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_gui_workbench_configures_recovery_root_and_restores_snapshots() {
        let rec_root = std::env::temp_dir().join(format!(
            "vellmark-main-rec-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&rec_root).unwrap();

        {
            let mut wb = AppWorkbench::new(AppConfig::default(), Some(rec_root.clone()));
            wb.new_document();
            let tab = wb.active_tab_mut().unwrap();
            tab.edit(|ctx| ctx.insert_text("crash protected content")).unwrap();
            tab.session.flush_current_recovery_snapshot().unwrap();
        }

        let (wb2, recovered) =
            build_initial_workbench_with_recovery(&[], Some(rec_root.clone()));
        assert_eq!(wb2.recovery_root.as_deref(), Some(rec_root.as_path()));
        assert_eq!(recovered, 1);
        assert_eq!(wb2.tabs[0].session.buffer.to_string(), "crash protected content");
        assert!(!default_recovery_root().as_os_str().is_empty());

        let _ = std::fs::remove_dir_all(&rec_root);
    }

    #[test]
    fn test_startup_restores_persisted_config_custom_css_and_recent_files() {
        let dir = std::env::temp_dir().join(format!(
            "vellmark-main-persist-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();

        let config_path = dir.join("config.toml");
        let recent_path = dir.join("recent-files.txt");
        let css_path = dir.join("custom.css");
        let doc_existing = dir.join("existing.md");
        let doc_missing = dir.join("deleted.md");

        std::fs::write(&doc_existing, "# Existing\n").unwrap();
        std::fs::write(
            &css_path,
            ":root { --bg-color: #102030; }\n.unsupported { position: fixed; }\n",
        )
        .unwrap();

        let cfg = AppConfig {
            follow_system_theme: false,
            theme: vellmark_theme::BuiltinTheme::Academic,
            max_content_width_px: 1080,
            zoom_percent: 120,
            custom_theme_path: Some(css_path.clone()),
            ..AppConfig::default()
        };
        cfg.save_to_file(&config_path).unwrap();

        let mut recent = RecentFilesList::new(10);
        recent.record_open(&doc_missing);
        recent.record_open(&doc_existing);
        recent.save_to_file(&recent_path).unwrap();

        let (wb, _) = build_initial_workbench_with_persistence(
            &[],
            None,
            Some(config_path.clone()),
            Some(recent_path.clone()),
        );
        assert_eq!(wb.config.max_content_width_px, 1080);
        assert_eq!(wb.config.zoom_percent, 120);
        assert_eq!(wb.config.custom_theme_path.as_deref(), Some(css_path.as_path()));
        assert_eq!(
            wb.theme.variables.get("--bg-color").map(String::as_str),
            Some("#102030")
        );
        assert!(
            !wb.theme.diagnostics.is_empty(),
            "startup should capture custom CSS diagnostics"
        );
        assert_eq!(wb.recent_files.entries, vec![doc_existing.clone()]);

        // Missing custom CSS file on startup safely falls back to configured built-in theme
        std::fs::remove_file(&css_path).unwrap();
        let (wb_fallback, _) = build_initial_workbench_with_persistence(
            &[],
            None,
            Some(config_path),
            Some(recent_path),
        );
        let academic = vellmark_theme::BuiltinTheme::Academic.load();
        assert_eq!(
            wb_fallback.theme.variables.get("--bg-color"),
            academic.variables.get("--bg-color")
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
