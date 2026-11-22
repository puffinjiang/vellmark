//! Workspace file tree (PRD 4.5 F-03) and Recent Files manager (PRD 4.5 F-08).

use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Supported document file extensions in the Vellmark sidebar file tree (PRD 4.5 F-01 & F-03).
const SUPPORTED_EXTENSIONS: &[&str] = &["md", "markdown", "txt"];

/// Directories ignored when scanning a workspace folder.
const IGNORED_DIRS: &[&str] = &[
    ".git",
    ".hg",
    ".svn",
    "target",
    "node_modules",
    ".idea",
    ".vscode",
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileTreeNode {
    pub name: String,
    pub path: PathBuf,
    pub is_dir: bool,
    pub children: Vec<FileTreeNode>,
}

impl FileTreeNode {
    /// Scan a root folder into a sorted tree of directories and Markdown/text files (PRD 4.5 F-03).
    pub fn scan_dir(root: &Path) -> io::Result<Self> {
        let name = root
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| root.display().to_string());
        let mut children = Vec::new();

        if root.is_dir() {
            for entry in fs::read_dir(root)? {
                let entry = entry?;
                let path = entry.path();
                let file_name = entry.file_name().to_string_lossy().to_string();

                // Skip hidden dotfiles/directories and ignored build directories
                if file_name.starts_with('.') || IGNORED_DIRS.contains(&file_name.as_str()) {
                    continue;
                }

                let file_type = entry.file_type()?;
                if file_type.is_dir() {
                    let node = Self::scan_dir(&path)?;
                    children.push(node);
                } else if file_type.is_file() && is_supported_document(&path) {
                    children.push(FileTreeNode {
                        name: file_name,
                        path,
                        is_dir: false,
                        children: Vec::new(),
                    });
                }
            }

            // Sort directories first (case-insensitive), then files alphabetically
            children.sort_by(|a, b| match (a.is_dir, b.is_dir) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => a.name.to_lowercase().cmp(&b.name.to_lowercase()),
            });
        }

        Ok(FileTreeNode {
            name,
            path: root.to_path_buf(),
            is_dir: root.is_dir(),
            children,
        })
    }

    /// Create a new empty Markdown file inside `parent_dir` (PRD 4.5 F-03).
    /// Validates that `file_name` is a single normal filename and fails with `AlreadyExists`
    /// if the target file already exists so existing files are never overwritten.
    pub fn create_document(parent_dir: &Path, file_name: &str) -> io::Result<PathBuf> {
        let mut components = Path::new(file_name).components();
        match (components.next(), components.next()) {
            (Some(std::path::Component::Normal(_)), None) => {}
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("invalid document filename: {file_name}"),
                ));
            }
        }
        let target = parent_dir.join(file_name);
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&target)?;
        use std::io::Write;
        file.write_all(b"")?;
        file.sync_all()?;
        Ok(target)
    }

    /// Rename or move a file or folder within the workspace (PRD 4.5 F-03).
    /// Refuses to overwrite an existing destination path (including broken/dangling symlinks).
    pub fn rename_or_move(from: &Path, to: &Path) -> io::Result<()> {
        match fs::symlink_metadata(to) {
            Ok(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    format!("destination already exists: {}", to.display()),
                ));
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }
        fs::rename(from, to)
    }

    /// Delete a document file from disk (PRD 4.5 F-03).
    pub fn delete_document(path: &Path) -> io::Result<()> {
        fs::remove_file(path)
    }
}

pub fn is_supported_document(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|ext| SUPPORTED_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// Tracks recently opened files and folders in MRU order (PRD 4.5 F-08).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecentFilesList {
    pub max_entries: usize,
    pub entries: Vec<PathBuf>,
}

impl Default for RecentFilesList {
    fn default() -> Self {
        Self::new(20)
    }
}

impl RecentFilesList {
    pub fn new(max_entries: usize) -> Self {
        Self {
            max_entries: max_entries.max(1),
            entries: Vec::new(),
        }
    }

    /// Record a file or folder as most-recently opened, deduplicating existing entries.
    pub fn record_open(&mut self, path: impl Into<PathBuf>) {
        let path = path.into();
        self.entries.retain(|p| p != &path);
        self.entries.insert(0, path);
        if self.entries.len() > self.max_entries {
            self.entries.truncate(self.max_entries);
        }
    }

    /// Prune entries that no longer exist on disk.
    pub fn prune_missing(&mut self) {
        self.entries.retain(|p| p.exists());
    }

    /// Serialize recent file paths using line-oriented escaping (`\\`, `\n`, `\r`) so paths
    /// remain round-trippable across platforms without external dependencies.
    pub fn to_persisted_str(&self) -> String {
        let mut out = String::new();
        for p in &self.entries {
            let s = p.to_string_lossy();
            if s.is_empty() {
                continue;
            }
            for ch in s.chars() {
                match ch {
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    other => out.push(other),
                }
            }
            out.push('\n');
        }
        out
    }

    pub fn from_persisted_str(raw: &str, max_entries: usize) -> Self {
        let mut list = Self::new(max_entries);
        for line in raw.lines() {
            if line.is_empty() {
                continue;
            }
            let mut decoded = String::with_capacity(line.len());
            let mut chars = line.chars();
            while let Some(ch) = chars.next() {
                if ch == '\\' {
                    match chars.next() {
                        Some('\\') => decoded.push('\\'),
                        Some('n') => decoded.push('\n'),
                        Some('r') => decoded.push('\r'),
                        Some(other) => {
                            decoded.push('\\');
                            decoded.push(other);
                        }
                        None => decoded.push('\\'),
                    }
                } else {
                    decoded.push(ch);
                }
            }
            if !decoded.is_empty() {
                let pb = PathBuf::from(decoded);
                if !list.entries.contains(&pb) {
                    list.entries.push(pb);
                    if list.entries.len() >= list.max_entries {
                        break;
                    }
                }
            }
        }
        list
    }

    pub fn save_to_file(&self, path: &Path) -> io::Result<()> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        crate::state::atomic_write_file(path, self.to_persisted_str().as_bytes())
    }

    pub fn load_from_file(path: &Path, max_entries: usize) -> io::Result<Self> {
        match fs::read_to_string(path) {
            Ok(raw) => Ok(Self::from_persisted_str(&raw, max_entries)),
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(Self::new(max_entries)),
            Err(err) => Err(err),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static TEST_DIR_SEQ: AtomicU64 = AtomicU64::new(0);

    fn unique_temp_dir(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let seq = TEST_DIR_SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "vellmark-ws-{tag}-{}-{nanos}-{seq}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_file_tree_scan_sorts_dirs_first_and_filters_extensions() {
        let root = unique_temp_dir("tree");

        fs::create_dir(root.join("docs")).unwrap();
        fs::create_dir(root.join(".git")).unwrap();
        FileTreeNode::create_document(&root.join("docs"), "guide.md").unwrap();
        FileTreeNode::create_document(&root, "README.md").unwrap();
        FileTreeNode::create_document(&root, "notes.txt").unwrap();
        fs::write(root.join("binary.bin"), b"\x00\x01").unwrap();

        assert!(FileTreeNode::create_document(&root, "../escape.md").is_err());

        let tree = FileTreeNode::scan_dir(&root).unwrap();
        assert_eq!(tree.children.len(), 3);
        assert!(tree.children[0].is_dir);
        assert_eq!(tree.children[0].name, "docs");
        assert_eq!(tree.children[0].children.len(), 1);
        assert_eq!(tree.children[0].children[0].name, "guide.md");
        assert_eq!(tree.children[1].name, "notes.txt");
        assert_eq!(tree.children[2].name, "README.md");

        // Test rename & RecentFilesList
        let renamed = root.join("CHANGELOG.md");
        FileTreeNode::rename_or_move(&root.join("notes.txt"), &renamed).unwrap();
        assert!(renamed.exists());

        let mut recent = RecentFilesList::new(3);
        recent.record_open(root.join("README.md"));
        recent.record_open(&renamed);
        recent.record_open(root.join("README.md"));
        assert_eq!(recent.entries[0], root.join("README.md"));
        assert_eq!(recent.entries.len(), 2);

        let _ = fs::remove_dir_all(&root);
    }

    #[cfg(unix)]
    #[test]
    fn test_p2_rename_or_move_refuses_to_overwrite_dangling_symlink() {
        use std::os::unix::fs::symlink;

        let root = unique_temp_dir("dangling-symlink");
        let src = root.join("source.md");
        fs::write(&src, "important source content").unwrap();

        let dangling_dest = root.join("dangling_link.md");
        symlink(root.join("missing_target.md"), &dangling_dest).unwrap();
        assert!(
            !dangling_dest.exists(),
            "dangling symlink target should not exist"
        );
        assert!(fs::symlink_metadata(&dangling_dest).is_ok());

        let err = FileTreeNode::rename_or_move(&src, &dangling_dest)
            .expect_err("must refuse overwriting dangling symlink");
        assert_eq!(err.kind(), io::ErrorKind::AlreadyExists);
        assert!(src.exists(), "source file should remain intact");
        assert!(
            fs::symlink_metadata(&dangling_dest)
                .map(|m| m.file_type().is_symlink())
                .unwrap_or(false),
            "dangling symlink should remain intact"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn test_recent_files_persistence_roundtrip() {
        let root = unique_temp_dir("recent-persist");
        let recent_file = root.join("nested").join("recent-files.txt");

        let mut list = RecentFilesList::new(3);
        let p1 = root.join("with space.md");
        let p2 = root.join(r"back\slash.md");
        let p3 = root.join("third.md");
        let p4 = root.join("fourth.md");
        list.record_open(&p1);
        list.record_open(&p2);
        list.record_open(&p3);
        list.record_open(&p4);
        // Max entries = 3 -> p1 evicted, order is p4, p3, p2
        assert_eq!(list.entries, vec![p4.clone(), p3.clone(), p2.clone()]);

        list.save_to_file(&recent_file).unwrap();
        let loaded = RecentFilesList::load_from_file(&recent_file, 3).unwrap();
        assert_eq!(loaded.entries, vec![p4.clone(), p3.clone(), p2.clone()]);

        // Truncating to smaller max_entries preserves most recent entries and deduplicates
        let loaded_small = RecentFilesList::load_from_file(&recent_file, 2).unwrap();
        assert_eq!(loaded_small.entries, vec![p4, p3]);

        // Missing file loads as empty list without error
        let missing = RecentFilesList::load_from_file(&root.join("does-not-exist.txt"), 5).unwrap();
        assert!(missing.entries.is_empty());

        let _ = fs::remove_dir_all(&root);
    }
}
