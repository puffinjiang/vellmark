//! Image asset import helper (PRD 4.4 I-02):
//! Copies pasted or dropped images into the configured relative assets folder (default `./assets`)
//! and formats a clean relative Markdown image link.

use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};

/// Returns whether `path` has a supported raster image extension decoded by `vellmark`.
pub fn is_supported_image_extension(path: &Path) -> bool {
    let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
        return false;
    };
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "png" | "jpg" | "jpeg" | "gif" | "webp" | "bmp" | "ico" | "tif" | "tiff"
    )
}

/// Imports `source_image` into `<base_dir>/<assets_rel_dir>/<unique_file_name>` and returns
/// `(copied_destination_path, markdown_snippet)` (e.g. `![diagram](./assets/diagram.png)`).
pub fn import_image_to_assets(
    doc_path: Option<&Path>,
    fallback_dir: Option<&Path>,
    source_image: &Path,
    assets_rel_dir: &str,
    alt_text: Option<&str>,
) -> io::Result<(PathBuf, String)> {
    let bytes = fs::read(source_image)?;
    let raw_name = source_image
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "invalid source image name"))?;
    let stem = Path::new(raw_name)
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("image");
    let ext = Path::new(raw_name)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    import_image_bytes_to_assets(
        doc_path,
        fallback_dir,
        &bytes,
        stem,
        ext,
        assets_rel_dir,
        alt_text,
    )
}

/// Writes raw image `bytes` into `<base_dir>/<assets_rel_dir>/<unique_file_name>` and returns
/// `(written_destination_path, markdown_snippet)` (PRD 4.4 I-02).
pub fn import_image_bytes_to_assets(
    doc_path: Option<&Path>,
    fallback_dir: Option<&Path>,
    bytes: &[u8],
    stem: &str,
    ext: &str,
    assets_rel_dir: &str,
    alt_text: Option<&str>,
) -> io::Result<(PathBuf, String)> {
    if bytes.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "cannot import empty image bytes",
        ));
    }
    let base_dir = doc_path
        .and_then(|p| p.parent())
        .filter(|p| !p.as_os_str().is_empty())
        .or(fallback_dir)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "cannot resolve target directory for image assets",
            )
        })?;

    let rel_clean = sanitize_relative_assets_dir(assets_rel_dir)?;
    let dest_dir = base_dir.join(&rel_clean);
    fs::create_dir_all(&dest_dir)?;

    let clean_stem = sanitize_filename_component(stem, "image");
    let clean_ext: String = ext
        .trim_start_matches('.')
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect();

    let (dest_path, final_name) =
        write_unique_destination(&dest_dir, &clean_stem, &clean_ext, bytes)?;
    let rel_posix = format_relative_markdown_path(assets_rel_dir, &rel_clean, &final_name);
    let alt = alt_text.unwrap_or(&clean_stem);
    let markdown = format!("![{alt}]({rel_posix})");
    Ok((dest_path, markdown))
}

fn sanitize_filename_component(raw: &str, fallback: &str) -> String {
    let clean: String = raw
        .chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '\0' => '-',
            _ if c.is_control() => '-',
            _ => c,
        })
        .collect();
    let trimmed = clean.trim().trim_matches('.');
    if trimmed.is_empty() {
        fallback.to_string()
    } else {
        trimmed.to_string()
    }
}

fn sanitize_relative_assets_dir(rel: &str) -> io::Result<PathBuf> {
    let mut clean = PathBuf::new();
    for comp in Path::new(rel).components() {
        match comp {
            Component::CurDir => {}
            Component::Normal(seg) => clean.push(seg),
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("unsafe assets directory path: {rel}"),
                ));
            }
        }
    }
    if clean.as_os_str().is_empty() {
        clean.push("assets");
    }
    Ok(clean)
}

fn write_unique_destination(
    dest_dir: &Path,
    stem: &str,
    ext: &str,
    bytes: &[u8],
) -> io::Result<(PathBuf, String)> {
    let mut suffix = 0usize;
    loop {
        let candidate_name = match (suffix, ext.is_empty()) {
            (0, true) => stem.to_string(),
            (0, false) => format!("{stem}.{ext}"),
            (n, true) => format!("{stem}-{n}"),
            (n, false) => format!("{stem}-{n}.{ext}"),
        };
        let dest_path = dest_dir.join(&candidate_name);
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&dest_path)
        {
            Ok(mut f) => {
                use std::io::Write;
                f.write_all(bytes)?;
                f.sync_all()?;
                return Ok((dest_path, candidate_name));
            }
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
                suffix += 1;
            }
            Err(e) => return Err(e),
        }
    }
}

fn format_relative_markdown_path(raw_rel: &str, clean_rel: &Path, file_name: &str) -> String {
    let dir_str = clean_rel
        .components()
        .map(|c| c.as_os_str().to_string_lossy().to_string())
        .collect::<Vec<_>>()
        .join("/");
    let encoded_name = file_name.replace(' ', "%20");
    if raw_rel.trim().starts_with("./") {
        format!("./{dir_str}/{encoded_name}")
    } else {
        format!("{dir_str}/{encoded_name}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_dir(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("vellmark-assets-{tag}-{nanos}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_prd_i02_import_image_copies_to_assets_and_disambiguates_collisions() {
        let dir = unique_temp_dir("copy");
        let doc_path = dir.join("note.md");
        fs::write(&doc_path, "# Note").unwrap();

        let src_img = dir.join("diagram.png");
        fs::write(&src_img, b"\x89PNG\r\n\x1a\nfakepng").unwrap();

        let (p1, md1) =
            import_image_to_assets(Some(&doc_path), None, &src_img, "./assets", None).unwrap();
        assert_eq!(md1, "![diagram](./assets/diagram.png)");
        assert!(p1.exists());

        // Second paste of same filename must disambiguate to `diagram-1.png`
        let (p2, md2) =
            import_image_to_assets(Some(&doc_path), None, &src_img, "./assets", Some("v2"))
                .unwrap();
        assert_eq!(md2, "![v2](./assets/diagram-1.png)");
        assert!(p2.exists());

        // Reject traversal in assets_rel_dir
        assert!(import_image_to_assets(Some(&doc_path), None, &src_img, "../outside", None).is_err());

        // Raw byte import (e.g. clipboard PNG) + supported extensions
        assert!(is_supported_image_extension(Path::new("snap.PNG")));
        assert!(is_supported_image_extension(Path::new("photo.jpeg")));
        assert!(!is_supported_image_extension(Path::new("notes.md")));

        let (p3, md3) = import_image_bytes_to_assets(
            Some(&doc_path),
            None,
            b"\x89PNG\r\n\x1a\nclip",
            "clipboard/../clip",
            "png",
            "./assets",
            Some("Pasted"),
        )
        .unwrap();
        assert_eq!(md3, "![Pasted](./assets/clipboard-..-clip.png)");
        assert!(p3.exists());

        let _ = fs::remove_dir_all(&dir);
    }
}
