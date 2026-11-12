use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Detected or assigned file encoding (PRD 4.5.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DetectedEncoding {
    Utf8,
    Utf8Bom,
    Utf16LeBom,
    Utf16BeBom,
    Utf16LeNoBom,
    Utf16BeNoBom,
    NonUtf8Ambiguous,
}

/// Whether the file can be edited immediately or must open read-only first (PRD 4.5.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EncodingDecision {
    /// Explicit evidence (UTF-8 BOM) or default product policy (strict UTF-8 validation passed).
    Editable(DetectedEncoding),
    /// Non-UTF-8 or ambiguous encoding: must open read-only until user confirms/selects encoding (PRD 4.5.2).
    ReadOnly { candidate: DetectedEncoding },
}

/// Dominant line ending style for newly inserted newlines while preserving existing line endings verbatim (PRD 4.5.2 & 5.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LineEnding {
    Lf,
    CrLf,
    Cr,
}

impl LineEnding {
    pub fn as_str(&self) -> &'static str {
        match self {
            LineEnding::Lf => "\n",
            LineEnding::CrLf => "\r\n",
            LineEnding::Cr => "\r",
        }
    }

    pub fn detect_dominant(text: &str) -> Self {
        let mut lf = 0usize;
        let mut crlf = 0usize;
        let mut cr = 0usize;
        let bytes = text.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'\r' {
                if i + 1 < bytes.len() && bytes[i + 1] == b'\n' {
                    crlf += 1;
                    i += 2;
                    continue;
                } else {
                    cr += 1;
                }
            } else if bytes[i] == b'\n' {
                lf += 1;
            }
            i += 1;
        }
        if crlf > lf && crlf >= cr {
            LineEnding::CrLf
        } else if cr > lf && cr > crlf {
            LineEnding::Cr
        } else {
            LineEnding::Lf
        }
    }
}

/// Format metadata captured on open so save preserves BOM and line endings byte-for-byte (PRD 5.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileFormatMeta {
    pub decision: EncodingDecision,
    pub dominant_line_ending: LineEnding,
    pub has_trailing_newline: bool,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum EncodingError {
    #[error("binary file detected (contains NUL code point or non-UTF-16 NUL bytes)")]
    BinaryFile,
    #[error("document is opened read-only and cannot be encoded for saving")]
    ReadOnlyDocument,
}

/// Strictly ordered encoding & binary detection per PRD 4.5.2:
/// 1. Check BOM (`EF BB BF`, `FF FE`, `FE FF`).
/// 2. Check aligned even/odd zero-byte pattern for BOM-less UTF-16 LE/BE; otherwise check first 8 KB for `0x00` binary NUL.
/// 3. Verify decoded UTF-16 does not contain `U+0000`.
/// 4. Validate strict UTF-8 (default policy -> Editable); otherwise classify as `ReadOnly` ambiguous non-UTF-8.
pub fn detect_and_decode(raw: &[u8]) -> Result<(String, FileFormatMeta), EncodingError> {
    // Step 1: BOM check
    if raw.starts_with(&[0xEF, 0xBB, 0xBF]) {
        let body = &raw[3..];
        if let Ok(text) = std::str::from_utf8(body) {
            if text.contains('\0') {
                return Err(EncodingError::BinaryFile);
            }
            let meta = FileFormatMeta {
                decision: EncodingDecision::Editable(DetectedEncoding::Utf8Bom),
                dominant_line_ending: LineEnding::detect_dominant(text),
                has_trailing_newline: text.ends_with('\n') || text.ends_with('\r'),
            };
            return Ok((text.to_string(), meta));
        }
    }

    if raw.starts_with(&[0xFF, 0xFE]) && raw.len() % 2 == 0 {
        if let Some(text) = decode_utf16(&raw[2..], true) {
            if text.contains('\0') {
                return Err(EncodingError::BinaryFile);
            }
            // P0: UTF-16 with BOM is recognized as valid text (not rejected as binary!) and opened ReadOnly in P0.
            let meta = FileFormatMeta {
                decision: EncodingDecision::ReadOnly {
                    candidate: DetectedEncoding::Utf16LeBom,
                },
                dominant_line_ending: LineEnding::detect_dominant(&text),
                has_trailing_newline: text.ends_with('\n') || text.ends_with('\r'),
            };
            return Ok((text, meta));
        }
    }

    if raw.starts_with(&[0xFE, 0xFF]) && raw.len() % 2 == 0 {
        if let Some(text) = decode_utf16(&raw[2..], false) {
            if text.contains('\0') {
                return Err(EncodingError::BinaryFile);
            }
            let meta = FileFormatMeta {
                decision: EncodingDecision::ReadOnly {
                    candidate: DetectedEncoding::Utf16BeBom,
                },
                dominant_line_ending: LineEnding::detect_dominant(&text),
                has_trailing_newline: text.ends_with('\n') || text.ends_with('\r'),
            };
            return Ok((text, meta));
        }
    }

    // Step 2: BOM-less UTF-16 pattern check BEFORE raw NUL binary rejection
    if raw.len() >= 4 && raw.len() % 2 == 0 {
        let pairs = raw.len() / 2;
        let mut even_zeros = 0usize;
        let mut odd_zeros = 0usize;
        for chunk in raw.chunks_exact(2) {
            if chunk[0] == 0 {
                even_zeros += 1;
            }
            if chunk[1] == 0 {
                odd_zeros += 1;
            }
        }
        // >= 30% zero bytes on odd indices -> BOM-less UTF-16 LE candidate
        if odd_zeros * 100 / pairs >= 30 && even_zeros == 0 {
            if let Some(text) = decode_utf16(raw, true) {
                if !text.contains('\0') {
                    let meta = FileFormatMeta {
                        decision: EncodingDecision::ReadOnly {
                            candidate: DetectedEncoding::Utf16LeNoBom,
                        },
                        dominant_line_ending: LineEnding::detect_dominant(&text),
                        has_trailing_newline: text.ends_with('\n') || text.ends_with('\r'),
                    };
                    return Ok((text, meta));
                }
            }
        }
        // >= 30% zero bytes on even indices -> BOM-less UTF-16 BE candidate
        if even_zeros * 100 / pairs >= 30 && odd_zeros == 0 {
            if let Some(text) = decode_utf16(raw, false) {
                if !text.contains('\0') {
                    let meta = FileFormatMeta {
                        decision: EncodingDecision::ReadOnly {
                            candidate: DetectedEncoding::Utf16BeNoBom,
                        },
                        dominant_line_ending: LineEnding::detect_dominant(&text),
                        has_trailing_newline: text.ends_with('\n') || text.ends_with('\r'),
                    };
                    return Ok((text, meta));
                }
            }
        }
    }

    // Binary check on first 8 KB if not UTF-16
    let probe_len = raw.len().min(8192);
    if raw[..probe_len].contains(&0x00) {
        return Err(EncodingError::BinaryFile);
    }

    // Step 3: Strict UTF-8 check (Default judgement policy -> Editable)
    if let Ok(text) = std::str::from_utf8(raw) {
        let meta = FileFormatMeta {
            decision: EncodingDecision::Editable(DetectedEncoding::Utf8),
            dominant_line_ending: LineEnding::detect_dominant(text),
            has_trailing_newline: text.ends_with('\n') || text.ends_with('\r'),
        };
        return Ok((text.to_string(), meta));
    }

    // Non-UTF-8 ambiguous text -> ReadOnly with lossy preview
    let preview = String::from_utf8_lossy(raw).into_owned();
    let meta = FileFormatMeta {
        decision: EncodingDecision::ReadOnly {
            candidate: DetectedEncoding::NonUtf8Ambiguous,
        },
        dominant_line_ending: LineEnding::detect_dominant(&preview),
        has_trailing_newline: preview.ends_with('\n') || preview.ends_with('\r'),
    };
    Ok((preview, meta))
}

/// Encode buffer text back to bytes for saving, preserving BOM and exact content (PRD 5.2).
pub fn encode_for_save(text: &str, meta: &FileFormatMeta) -> Result<Vec<u8>, EncodingError> {
    match meta.decision {
        EncodingDecision::Editable(DetectedEncoding::Utf8) => Ok(text.as_bytes().to_vec()),
        EncodingDecision::Editable(DetectedEncoding::Utf8Bom) => {
            let mut out = Vec::with_capacity(3 + text.len());
            out.extend_from_slice(&[0xEF, 0xBB, 0xBF]);
            out.extend_from_slice(text.as_bytes());
            Ok(out)
        }
        _ => Err(EncodingError::ReadOnlyDocument),
    }
}

fn decode_utf16(bytes: &[u8], little_endian: bool) -> Option<String> {
    if bytes.len() % 2 != 0 {
        return None;
    }
    let u16s: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|c| {
            if little_endian {
                u16::from_le_bytes([c[0], c[1]])
            } else {
                u16::from_be_bytes([c[0], c[1]])
            }
        })
        .collect();
    String::from_utf16(&u16s).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_trip_utf8_with_bom_and_mixed_line_endings() {
        let mut raw = vec![0xEF, 0xBB, 0xBF];
        raw.extend_from_slice(b"# Title\r\nParagraph 1\nParagraph 2\r\n");
        let (decoded, meta) = detect_and_decode(&raw).unwrap();
        assert_eq!(
            meta.decision,
            EncodingDecision::Editable(DetectedEncoding::Utf8Bom)
        );
        assert_eq!(meta.dominant_line_ending, LineEnding::CrLf);
        let saved = encode_for_save(&decoded, &meta).unwrap();
        assert_eq!(saved, raw, "Round-trip save must be byte-for-byte identical");
    }

    #[test]
    fn test_utf16_with_nul_bytes_is_not_rejected_as_binary() {
        // PRD 4.5.2: UTF-16 LE contains 0x00 for ASCII characters; must NOT be rejected as BinaryFile.
        let text = "# Hello Markdown\nThis is UTF-16 LE.\n";
        let mut raw = vec![0xFF, 0xFE]; // UTF-16 LE BOM
        for u in text.encode_utf16() {
            raw.extend_from_slice(&u.to_le_bytes());
        }
        let (decoded, meta) = detect_and_decode(&raw).unwrap();
        assert_eq!(decoded, text);
        assert_eq!(
            meta.decision,
            EncodingDecision::ReadOnly {
                candidate: DetectedEncoding::Utf16LeBom
            }
        );
    }

    #[test]
    fn test_utf16_containing_actual_nul_codepoint_is_rejected_as_binary() {
        let mut raw = vec![0xFF, 0xFE];
        raw.extend_from_slice(&0x0041u16.to_le_bytes()); // 'A'
        raw.extend_from_slice(&0x0000u16.to_le_bytes()); // U+0000 NUL code point
        assert_eq!(detect_and_decode(&raw), Err(EncodingError::BinaryFile));
    }

    #[test]
    fn test_non_utf8_gbk_opens_readonly_and_refuses_save() {
        // GBK bytes for "中文" (0xD6, 0xD0, 0xCE, 0xC4) - invalid UTF-8
        let raw = [0xD6u8, 0xD0, 0xCE, 0xC4];
        let (_preview, meta) = detect_and_decode(&raw).unwrap();
        assert_eq!(
            meta.decision,
            EncodingDecision::ReadOnly {
                candidate: DetectedEncoding::NonUtf8Ambiguous
            }
        );
        assert_eq!(
            encode_for_save("中文", &meta),
            Err(EncodingError::ReadOnlyDocument)
        );
    }
}
