pub mod assets;
pub mod clipboard;
pub mod config;
pub mod gpu;
pub mod gui;
pub mod images;
pub mod search;
pub mod workbench;

use std::path::{Path, PathBuf};

use config::AppConfig;
use gui::{GuiController, PROTECTED_SNAPSHOT_LIMIT_NOTICE, WELCOME_MARKDOWN, VellmarkApp};
use vellmark_core::Selection;
use workbench::AppWorkbench;

/// Default persistent directory storing crash-recovery snapshots across GUI restarts.
pub fn default_recovery_root() -> PathBuf {
    if let Some(dir) = std::env::var_os("VELLMARK_RECOVERY_DIR").filter(|v| !v.is_empty()) {
        return PathBuf::from(dir);
    }
    #[cfg(target_os = "macos")]
    if let Some(home) = std::env::var_os("HOME").filter(|v| !v.is_empty()) {
        return PathBuf::from(home)
            .join("Library")
            .join("Application Support")
            .join("vellmark")
            .join("recovery");
    }
    #[cfg(target_os = "windows")]
    if let Some(appdata) = std::env::var_os("APPDATA").filter(|v| !v.is_empty()) {
        return PathBuf::from(appdata).join("vellmark").join("recovery");
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        if let Some(xdg) = std::env::var_os("XDG_DATA_HOME").filter(|v| !v.is_empty()) {
            return PathBuf::from(xdg).join("vellmark").join("recovery");
        }
        if let Some(home) = std::env::var_os("HOME").filter(|v| !v.is_empty()) {
            return PathBuf::from(home)
                .join(".local")
                .join("share")
                .join("vellmark")
                .join("recovery");
        }
    }
    std::env::temp_dir().join("vellmark-recovery")
}

fn build_initial_workbench_with_recovery(
    paths: &[PathBuf],
    recovery_root: Option<PathBuf>,
) -> (AppWorkbench, usize) {
    let config = AppConfig::default();
    let mut workbench = AppWorkbench::new(config, recovery_root);

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

    let (workbench, recovered) =
        build_initial_workbench_with_recovery(&paths, Some(default_recovery_root()));
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
}
