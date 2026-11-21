//! In-document Find & Replace engine for `vellmark-core` (PRD 4.1 E-11).
//!
//! Supports literal search and regex search (`use_regex`), case sensitivity
//! (`case_sensitive`), whole-word boundaries aware of Latin + CJK mixed text
//! (`whole_word`), capture-group replacement (`$0`/`$&`, `$1`..`$9`, `$$`),
//! next/previous navigation, and single-undo `replace_all`.

use std::fmt;

use crate::buffer::{ByteOffset, ByteRange, Selection, TextBuffer};
use crate::encoding::{LineEnding, normalize_line_endings};
use crate::undo::{EditKind, EditOp, Transaction, UndoHistory};

/// Configuration options for in-document Find & Replace (PRD E-11).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FindOptions {
    /// Distinguish upper/lower case when matching.
    pub case_sensitive: bool,
    /// Require word boundaries around the match (Latin/digit words vs delimiters & CJK).
    pub whole_word: bool,
    /// Interpret query as a regular expression.
    pub use_regex: bool,
}

/// Errors produced when compiling a search pattern.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FindError {
    /// Regular expression syntax error or complexity budget exceeded.
    InvalidRegex(String),
}

impl fmt::Display for FindError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FindError::InvalidRegex(msg) => write!(f, "invalid regex: {msg}"),
        }
    }
}

impl std::error::Error for FindError {}

/// A single search match in the document along with optional capture groups.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FindMatch {
    /// Full match byte range `[start..end)` in the source string.
    pub range: ByteRange,
    /// Capture slots where `captures[0]` is `Some(range)` and `captures[1..]` are `(...)` groups.
    pub captures: Vec<Option<ByteRange>>,
}

/// Outcome of replacing one or more matches in a document buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FindReplaceOutcome {
    /// Number of matches replaced.
    pub replaced_count: usize,
    /// Selection after the replacement operation.
    pub selection_after: Selection,
}

/// Find all non-overlapping matches for `query` in `text` under `options`.
pub fn find_all_matches(
    text: &str,
    query: &str,
    options: FindOptions,
) -> Result<Vec<FindMatch>, FindError> {
    if query.is_empty() {
        return Ok(Vec::new());
    }
    if options.use_regex {
        let re = CompiledRegex::compile(query, options.case_sensitive)?;
        Ok(re.find_all(text, options.whole_word))
    } else {
        Ok(find_all_literal(text, query, options))
    }
}

/// Find the next match at or after `from_offset`, optionally wrapping to the start of the document.
pub fn find_next_match(
    text: &str,
    query: &str,
    from_offset: ByteOffset,
    wrap: bool,
    options: FindOptions,
) -> Result<Option<FindMatch>, FindError> {
    let matches = find_all_matches(text, query, options)?;
    if matches.is_empty() {
        return Ok(None);
    }
    let found = matches.iter().find(|m| m.range.start >= from_offset).cloned();
    if found.is_some() || !wrap {
        Ok(found)
    } else {
        Ok(matches.into_iter().next())
    }
}

/// Find the previous match strictly before (or ending at/before) `from_offset`, optionally wrapping to EOF.
pub fn find_prev_match(
    text: &str,
    query: &str,
    from_offset: ByteOffset,
    wrap: bool,
    options: FindOptions,
) -> Result<Option<FindMatch>, FindError> {
    let matches = find_all_matches(text, query, options)?;
    if matches.is_empty() {
        return Ok(None);
    }
    let found = matches
        .iter()
        .rfind(|m| m.range.end <= from_offset && m.range.start < from_offset)
        .or_else(|| matches.iter().rfind(|m| m.range.start < from_offset))
        .cloned();
    if found.is_some() || !wrap {
        Ok(found)
    } else {
        Ok(matches.into_iter().last())
    }
}

fn flush_template_chunk(out: &mut String, chunk: &mut String, target_eol: Option<LineEnding>) {
    if chunk.is_empty() {
        return;
    }
    if let Some(eol) = target_eol {
        out.push_str(&normalize_line_endings(chunk, eol));
    } else {
        out.push_str(chunk);
    }
    chunk.clear();
}

fn expand_replacement_inner(
    text: &str,
    matched: &FindMatch,
    replacement: &str,
    use_regex: bool,
    target_eol: Option<LineEnding>,
) -> String {
    if !use_regex {
        return match target_eol {
            Some(eol) => normalize_line_endings(replacement, eol).into_owned(),
            None => replacement.to_string(),
        };
    }
    let mut out = String::with_capacity(replacement.len());
    // Accumulate template-authored text/escapes separately so newline normalization
    // applies only to template newlines (and handles `\r\n` as a single break) without
    // altering existing line endings captured via `$0`/`$&`/`$1..$9`.
    let mut template_chunk = String::new();
    let bytes = replacement.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'$' && i + 1 < bytes.len() {
            let next = bytes[i + 1];
            if next == b'$' {
                template_chunk.push('$');
                i += 2;
                continue;
            }
            if next == b'&' || next.is_ascii_digit() {
                let idx = if next == b'&' {
                    0
                } else {
                    (next - b'0') as usize
                };
                if let Some(Some(r)) = matched.captures.get(idx)
                    && let Some(sub) = text.get(r.start..r.end)
                    && !sub.is_empty()
                {
                    flush_template_chunk(&mut out, &mut template_chunk, target_eol);
                    out.push_str(sub);
                }
                i += 2;
                continue;
            }
        }
        if bytes[i] == b'\\' && i + 1 < bytes.len() {
            let esc = match bytes[i + 1] {
                b'n' => Some('\n'),
                b'r' => Some('\r'),
                b't' => Some('\t'),
                b'\\' => Some('\\'),
                _ => None,
            };
            if let Some(ch) = esc {
                template_chunk.push(ch);
                i += 2;
                continue;
            }
        }
        let ch = replacement[i..].chars().next().unwrap();
        template_chunk.push(ch);
        i += ch.len_utf8();
    }
    flush_template_chunk(&mut out, &mut template_chunk, target_eol);
    out
}

/// Expand replacement template (`$0`/`$&`, `$1`..`$9`, `$$`, `\n`, `\t`, `\r`, `\\`) when `use_regex` is true.
pub fn expand_replacement(
    text: &str,
    matched: &FindMatch,
    replacement: &str,
    use_regex: bool,
) -> String {
    expand_replacement_inner(text, matched, replacement, use_regex, None)
}

fn expand_and_normalize_replacement(
    text: &str,
    matched: &FindMatch,
    replacement: &str,
    use_regex: bool,
    target_eol: LineEnding,
) -> String {
    expand_replacement_inner(text, matched, replacement, use_regex, Some(target_eol))
}

/// Replace a single match in `buffer` cleanly with undo history, normalizing inserted newlines
/// to the document's dominant line ending.
#[deprecated(
    note = "use replace_match_in_buffer_with_line_ending to preserve single-line document EOL preferences"
)]
pub fn replace_match_in_buffer(
    buffer: &mut TextBuffer,
    undo: &mut UndoHistory,
    selection_before: Selection,
    matched: &FindMatch,
    replacement: &str,
    use_regex: bool,
) -> FindReplaceOutcome {
    replace_match_in_buffer_with_line_ending(
        buffer,
        undo,
        selection_before,
        matched,
        replacement,
        use_regex,
        LineEnding::Lf,
    )
}

/// Replace a single match in `buffer`, falling back to `preferred_eol` when normalizing newlines
/// in single-line or empty documents.
pub fn replace_match_in_buffer_with_line_ending(
    buffer: &mut TextBuffer,
    undo: &mut UndoHistory,
    selection_before: Selection,
    matched: &FindMatch,
    replacement: &str,
    use_regex: bool,
    preferred_eol: LineEnding,
) -> FindReplaceOutcome {
    let snapshot = buffer.to_string();
    let target_eol = LineEnding::detect_dominant_or(&snapshot, preferred_eol);
    let expanded =
        expand_and_normalize_replacement(&snapshot, matched, replacement, use_regex, target_eol);
    let sel_after = Selection::caret(matched.range.start + expanded.len());
    undo.apply_edit(
        buffer,
        matched.range,
        &expanded,
        selection_before,
        sel_after,
        EditKind::Structural,
    );
    FindReplaceOutcome {
        replaced_count: 1,
        selection_after: sel_after,
    }
}

/// Replace all matches in `buffer` inside a **single undo transaction** (PRD E-11), normalizing
/// inserted newlines to the document's dominant line ending.
#[deprecated(
    note = "use replace_all_in_buffer_with_line_ending to preserve single-line document EOL preferences"
)]
pub fn replace_all_in_buffer(
    buffer: &mut TextBuffer,
    undo: &mut UndoHistory,
    selection_before: Selection,
    query: &str,
    replacement: &str,
    options: FindOptions,
) -> Result<FindReplaceOutcome, FindError> {
    replace_all_in_buffer_with_line_ending(
        buffer,
        undo,
        selection_before,
        query,
        replacement,
        options,
        LineEnding::Lf,
    )
}

/// Replace all matches in `buffer`, falling back to `preferred_eol` when normalizing newlines
/// in single-line or empty documents.
pub fn replace_all_in_buffer_with_line_ending(
    buffer: &mut TextBuffer,
    undo: &mut UndoHistory,
    selection_before: Selection,
    query: &str,
    replacement: &str,
    options: FindOptions,
    preferred_eol: LineEnding,
) -> Result<FindReplaceOutcome, FindError> {
    let snapshot = buffer.to_string();
    let matches = find_all_matches(&snapshot, query, options)?;
    if matches.is_empty() {
        return Ok(FindReplaceOutcome {
            replaced_count: 0,
            selection_after: selection_before,
        });
    }

    let target_eol = LineEnding::detect_dominant_or(&snapshot, preferred_eol);
    let replacements: Vec<String> = matches
        .iter()
        .map(|m| {
            expand_and_normalize_replacement(
                &snapshot,
                m,
                replacement,
                options.use_regex,
                target_eol,
            )
        })
        .collect();

    // Compute mapped caret offset before mutating buffer.
    let orig_caret = selection_before.head.min(snapshot.len());
    let mut mapped_caret = orig_caret as isize;
    for (m, rep) in matches.iter().zip(replacements.iter()) {
        let delta = rep.len() as isize - m.range.len() as isize;
        if orig_caret >= m.range.end {
            mapped_caret += delta;
        } else if orig_caret > m.range.start {
            mapped_caret += rep.len() as isize - (orig_caret - m.range.start) as isize;
        }
    }

    // Apply in reverse document order so each `op` holds exact pre-edit coordinates,
    // making both forward redo and `.rev()` undo offset-exact with zero shifting.
    let mut ops = Vec::with_capacity(matches.len());
    for (m, rep) in matches.iter().zip(replacements).rev() {
        let deleted = buffer.replace_bytes(m.range, &rep);
        ops.push(EditOp {
            range: m.range,
            deleted,
            inserted: rep,
        });
    }

    let updated = buffer.to_string();
    let mut final_offset = (mapped_caret.max(0) as usize).min(updated.len());
    while final_offset > 0 && !updated.is_char_boundary(final_offset) {
        final_offset -= 1;
    }
    let sel_after = Selection::caret(final_offset);
    undo.record_transaction(Transaction {
        ops,
        selection_before,
        selection_after: sel_after,
        kind: EditKind::Structural,
    });

    Ok(FindReplaceOutcome {
        replaced_count: matches.len(),
        selection_after: sel_after,
    })
}

// ---------------------------------------------------------------------------
// Literal & Whole-Word Search
// ---------------------------------------------------------------------------

fn find_all_literal(text: &str, query: &str, options: FindOptions) -> Vec<FindMatch> {
    let mut out = Vec::new();
    let mut pos = 0usize;

    if options.case_sensitive {
        while pos <= text.len() {
            let Some(rel) = text[pos..].find(query) else {
                break;
            };
            let start = pos + rel;
            let end = start + query.len();
            let range = ByteRange::new(start, end);
            if !options.whole_word || is_whole_word_match(text, range) {
                out.push(FindMatch {
                    range,
                    captures: vec![Some(range)],
                });
                pos = end;
            } else {
                pos = advance_one_char(text, start);
            }
        }
        return out;
    }

    let query_chars: Vec<char> = query.chars().collect();
    while pos < text.len() {
        if let Some(match_bytes) = match_literal_ci_at(&text[pos..], &query_chars) {
            let range = ByteRange::new(pos, pos + match_bytes);
            if !options.whole_word || is_whole_word_match(text, range) {
                out.push(FindMatch {
                    range,
                    captures: vec![Some(range)],
                });
                pos += match_bytes;
                continue;
            }
        }
        pos = advance_one_char(text, pos);
    }

    out
}

fn match_literal_ci_at(haystack: &str, query_chars: &[char]) -> Option<usize> {
    let mut h_iter = haystack.char_indices();
    let mut last_end = 0usize;
    for &qc in query_chars {
        let (idx, hc) = h_iter.next()?;
        if !chars_eq_ci(hc, qc) {
            return None;
        }
        last_end = idx + hc.len_utf8();
    }
    Some(last_end)
}

fn advance_one_char(text: &str, pos: usize) -> usize {
    if pos >= text.len() {
        return text.len() + 1;
    }
    let ch = text[pos..].chars().next().unwrap();
    pos + ch.len_utf8()
}

fn chars_eq_ci(a: char, b: char) -> bool {
    if a == b {
        return true;
    }
    if a.is_ascii() && b.is_ascii() {
        return a.eq_ignore_ascii_case(&b);
    }
    a.to_lowercase().eq(b.to_lowercase())
}

fn is_cjk(c: char) -> bool {
    matches!(
        c,
        '\u{1100}'..='\u{11FF}'
            | '\u{3040}'..='\u{30FF}'
            | '\u{31F0}'..='\u{31FF}'
            | '\u{3400}'..='\u{4DBF}'
            | '\u{4E00}'..='\u{9FFF}'
            | '\u{AC00}'..='\u{D7AF}'
            | '\u{F900}'..='\u{FAFF}'
            | '\u{20000}'..='\u{2FA1F}'
    )
}

/// Latin/Unicode word character distinct from CJK ideographs and punctuation.
fn is_word_char(c: char) -> bool {
    (c.is_alphanumeric() || c == '_') && !is_cjk(c)
}

fn is_whole_word_match(text: &str, range: ByteRange) -> bool {
    if range.is_empty() {
        return false;
    }
    let slice = &text[range.start..range.end];
    let first = slice.chars().next();
    let last = slice.chars().next_back();
    let prev = text[..range.start].chars().next_back();
    let next = text[range.end..].chars().next();

    let left_ok = !matches!(
        (prev, first),
        (Some(p), Some(f)) if is_word_char(p) && is_word_char(f)
    );
    let right_ok = !matches!(
        (last, next),
        (Some(l), Some(n)) if is_word_char(l) && is_word_char(n)
    );
    left_ok && right_ok
}

// ---------------------------------------------------------------------------
// Pure-Rust Linear-Time Regex Engine (AST -> Bytecode -> Pike VM)
// ---------------------------------------------------------------------------

const MAX_REPEAT_EXPAND: usize = 512;
const MAX_NFA_INSTS: usize = 8_192;
const MAX_REGEX_DEPTH: usize = 64;

#[derive(Debug, Clone)]
enum ClassItem {
    Range(char, char),
    Digit,
    NotDigit,
    Word,
    NotWord,
    Space,
    NotSpace,
}

#[derive(Debug, Clone)]
enum CharMatcher {
    AnyNoNewline,
    AnyCharIncludingNewline,
    Single(char),
    Class { negated: bool, items: Vec<ClassItem> },
}

impl CharMatcher {
    fn matches(&self, ch: char, case_sensitive: bool) -> bool {
        match self {
            CharMatcher::AnyNoNewline => ch != '\n' && ch != '\r',
            CharMatcher::AnyCharIncludingNewline => true,
            CharMatcher::Single(target) => {
                if case_sensitive {
                    ch == *target
                } else {
                    chars_eq_ci(ch, *target)
                }
            }
            CharMatcher::Class { negated, items } => {
                let hit = items.iter().any(|item| match item {
                    ClassItem::Range(lo, hi) => char_in_range(ch, *lo, *hi, case_sensitive),
                    ClassItem::Digit => ch.is_ascii_digit(),
                    ClassItem::NotDigit => !ch.is_ascii_digit(),
                    ClassItem::Word => is_word_char(ch),
                    ClassItem::NotWord => !is_word_char(ch),
                    ClassItem::Space => ch.is_whitespace(),
                    ClassItem::NotSpace => !ch.is_whitespace(),
                });
                if *negated { !hit } else { hit }
            }
        }
    }
}

fn char_in_range(ch: char, lo: char, hi: char, case_sensitive: bool) -> bool {
    if ch >= lo && ch <= hi {
        return true;
    }
    if case_sensitive {
        return false;
    }
    if ch.is_ascii() && lo.is_ascii() && hi.is_ascii() {
        let u = ch.to_ascii_uppercase();
        let l = ch.to_ascii_lowercase();
        (u >= lo && u <= hi) || (l >= lo && l <= hi)
    } else {
        ch.to_lowercase().all(|lc| {
            let mut lol = lo.to_lowercase();
            let mut hil = hi.to_lowercase();
            match (lol.next(), hil.next()) {
                (Some(a), Some(b)) => lc >= a && lc <= b,
                _ => false,
            }
        })
    }
}

#[derive(Debug, Clone, Copy)]
enum AnchorKind {
    LineStart,
    LineEnd,
    WordBoundary,
    NotWordBoundary,
}

impl AnchorKind {
    fn matches(&self, text: &str, pos: usize) -> bool {
        let bytes = text.as_bytes();
        match self {
            AnchorKind::LineStart => pos == 0 || bytes.get(pos.saturating_sub(1)) == Some(&b'\n'),
            AnchorKind::LineEnd => {
                pos == text.len()
                    || (bytes.get(pos) == Some(&b'\r') && bytes.get(pos + 1) == Some(&b'\n'))
                    || (bytes.get(pos) == Some(&b'\n')
                        && (pos == 0 || bytes.get(pos - 1) != Some(&b'\r')))
            }
            AnchorKind::WordBoundary => is_at_word_boundary(text, pos),
            AnchorKind::NotWordBoundary => !is_at_word_boundary(text, pos),
        }
    }
}

fn is_at_word_boundary(text: &str, pos: usize) -> bool {
    let prev = text[..pos].chars().next_back().is_some_and(is_word_char);
    let next = text[pos..].chars().next().is_some_and(is_word_char);
    prev != next
}

#[derive(Debug, Clone)]
enum Ast {
    Empty,
    Matcher(CharMatcher),
    Anchor(AnchorKind),
    Concat(Vec<Ast>),
    Alt(Vec<Ast>),
    Group {
        capture_idx: Option<usize>,
        inner: Box<Ast>,
    },
    Repeat {
        inner: Box<Ast>,
        min: usize,
        max: Option<usize>,
        greedy: bool,
    },
}

struct RegexParser<'a> {
    src: &'a str,
    pos: usize,
    capture_count: usize,
    depth: usize,
}

impl<'a> RegexParser<'a> {
    fn new(src: &'a str) -> Self {
        Self {
            src,
            pos: 0,
            capture_count: 1, // group 0 reserved for full match
            depth: 0,
        }
    }

    fn peek(&self) -> Option<char> {
        self.src[self.pos..].chars().next()
    }

    fn bump(&mut self) -> Option<char> {
        let ch = self.peek()?;
        self.pos += ch.len_utf8();
        Some(ch)
    }

    fn parse_all(&mut self) -> Result<(Ast, usize), FindError> {
        let ast = self.parse_alt()?;
        if let Some(ch) = self.peek() {
            return Err(FindError::InvalidRegex(format!(
                "unexpected character '{ch}' at byte {}",
                self.pos
            )));
        }
        Ok((ast, self.capture_count))
    }

    fn parse_alt(&mut self) -> Result<Ast, FindError> {
        let mut branches = vec![self.parse_concat()?];
        while self.peek() == Some('|') {
            self.bump();
            branches.push(self.parse_concat()?);
        }
        if branches.len() == 1 {
            Ok(branches.pop().unwrap())
        } else {
            Ok(Ast::Alt(branches))
        }
    }

    fn parse_concat(&mut self) -> Result<Ast, FindError> {
        let mut items = Vec::new();
        while let Some(ch) = self.peek() {
            if ch == ')' || ch == '|' {
                break;
            }
            items.push(self.parse_quantified()?);
        }
        match items.len() {
            0 => Ok(Ast::Empty),
            1 => Ok(items.pop().unwrap()),
            _ => Ok(Ast::Concat(items)),
        }
    }

    fn parse_quantified(&mut self) -> Result<Ast, FindError> {
        let atom = self.parse_atom()?;
        if matches!(atom, Ast::Anchor(_)) && matches!(self.peek(), Some('*' | '+' | '?' | '{')) {
            return Err(FindError::InvalidRegex(
                "quantifier cannot follow an anchor".to_string(),
            ));
        }
        let mut node = atom;
        if let Some(q) = self.peek() {
            let quant = match q {
                '*' => {
                    self.bump();
                    Some((0usize, None))
                }
                '+' => {
                    self.bump();
                    Some((1usize, None))
                }
                '?' => {
                    self.bump();
                    Some((0usize, Some(1usize)))
                }
                '{' => Some(self.parse_brace_quantifier()?),
                _ => None,
            };
            if let Some((min, max)) = quant {
                let greedy = if self.peek() == Some('?') {
                    self.bump();
                    false
                } else {
                    true
                };
                if matches!(self.peek(), Some('*' | '+' | '?' | '{')) {
                    return Err(FindError::InvalidRegex(
                        "nested quantifier without grouping".to_string(),
                    ));
                }
                node = Ast::Repeat {
                    inner: Box::new(node),
                    min,
                    max,
                    greedy,
                };
            }
        }
        Ok(node)
    }

    fn parse_brace_quantifier(&mut self) -> Result<(usize, Option<usize>), FindError> {
        self.bump(); // consume '{'
        let start = self.pos;
        let min = self.read_usize().ok_or_else(|| {
            FindError::InvalidRegex(format!("invalid repetition bound at byte {start}"))
        })?;
        let max = if self.peek() == Some(',') {
            self.bump();
            if self.peek() == Some('}') {
                None
            } else {
                let m = self.read_usize().ok_or_else(|| {
                    FindError::InvalidRegex(format!("invalid max bound at byte {}", self.pos))
                })?;
                if m < min {
                    return Err(FindError::InvalidRegex(format!(
                        "repetition min ({min}) > max ({m})"
                    )));
                }
                Some(m)
            }
        } else {
            Some(min)
        };
        if self.bump() != Some('}') {
            return Err(FindError::InvalidRegex(
                "unclosed '{...}' quantifier".to_string(),
            ));
        }
        let check_count = max.unwrap_or(min);
        if min > MAX_REPEAT_EXPAND || check_count > MAX_REPEAT_EXPAND {
            return Err(FindError::InvalidRegex(format!(
                "repetition bound exceeds limit ({MAX_REPEAT_EXPAND})"
            )));
        }
        Ok((min, max))
    }

    fn read_usize(&mut self) -> Option<usize> {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                self.bump();
            } else {
                break;
            }
        }
        if self.pos == start {
            None
        } else {
            self.src[start..self.pos].parse::<usize>().ok()
        }
    }

    fn parse_atom(&mut self) -> Result<Ast, FindError> {
        let ch = self.bump().ok_or_else(|| {
            FindError::InvalidRegex("unexpected end of regex".to_string())
        })?;
        match ch {
            '*' | '+' | '?' | '{' => Err(FindError::InvalidRegex(format!(
                "nothing to repeat before '{ch}'"
            ))),
            '.' => Ok(Ast::Matcher(CharMatcher::AnyNoNewline)),
            '^' => Ok(Ast::Anchor(AnchorKind::LineStart)),
            '$' => Ok(Ast::Anchor(AnchorKind::LineEnd)),
            '(' => self.parse_group(),
            '[' => self.parse_char_class(),
            '\\' => self.parse_escape(),
            other => Ok(Ast::Matcher(CharMatcher::Single(other))),
        }
    }

    fn parse_group(&mut self) -> Result<Ast, FindError> {
        if self.depth >= MAX_REGEX_DEPTH {
            return Err(FindError::InvalidRegex(format!(
                "regex nesting depth exceeds limit ({MAX_REGEX_DEPTH})"
            )));
        }
        let capture_idx = if self.src[self.pos..].starts_with("?:") {
            self.pos += 2;
            None
        } else {
            let idx = self.capture_count;
            self.capture_count += 1;
            Some(idx)
        };
        self.depth += 1;
        let inner = self.parse_alt();
        self.depth -= 1;
        let inner = inner?;
        if self.bump() != Some(')') {
            return Err(FindError::InvalidRegex("unclosed '(' group".to_string()));
        }
        Ok(Ast::Group {
            capture_idx,
            inner: Box::new(inner),
        })
    }

    fn parse_escape(&mut self) -> Result<Ast, FindError> {
        let ch = self.bump().ok_or_else(|| {
            FindError::InvalidRegex("trailing '\\' in pattern".to_string())
        })?;
        match ch {
            'd' => Ok(shorthand_class(false, ClassItem::Digit)),
            'D' => Ok(shorthand_class(false, ClassItem::NotDigit)),
            'w' => Ok(shorthand_class(false, ClassItem::Word)),
            'W' => Ok(shorthand_class(false, ClassItem::NotWord)),
            's' => Ok(shorthand_class(false, ClassItem::Space)),
            'S' => Ok(shorthand_class(false, ClassItem::NotSpace)),
            'b' => Ok(Ast::Anchor(AnchorKind::WordBoundary)),
            'B' => Ok(Ast::Anchor(AnchorKind::NotWordBoundary)),
            'n' => Ok(Ast::Matcher(CharMatcher::Single('\n'))),
            'r' => Ok(Ast::Matcher(CharMatcher::Single('\r'))),
            't' => Ok(Ast::Matcher(CharMatcher::Single('\t'))),
            other => Ok(Ast::Matcher(CharMatcher::Single(other))),
        }
    }

    fn parse_char_class(&mut self) -> Result<Ast, FindError> {
        let negated = if self.peek() == Some('^') {
            self.bump();
            true
        } else {
            false
        };
        let mut items = Vec::new();
        let mut first = true;
        loop {
            let Some(ch) = self.bump() else {
                return Err(FindError::InvalidRegex("unclosed '[' character class".to_string()));
            };
            if ch == ']' && !first {
                break;
            }
            first = false;
            if ch == '\\' {
                let esc = self.bump().ok_or_else(|| {
                    FindError::InvalidRegex("trailing '\\' inside '[...]'".to_string())
                })?;
                if let Some(shorthand) = escape_class_item(esc) {
                    items.push(shorthand);
                    continue;
                }
                let lit = unescape_char(esc);
                self.finish_class_char(lit, &mut items)?;
            } else {
                self.finish_class_char(ch, &mut items)?;
            }
        }
        Ok(Ast::Matcher(CharMatcher::Class { negated, items }))
    }

    fn finish_class_char(&mut self, start: char, items: &mut Vec<ClassItem>) -> Result<(), FindError> {
        if self.peek() == Some('-') && !self.src[self.pos + 1..].starts_with(']') {
            self.bump(); // consume '-'
            let Some(next) = self.bump() else {
                return Err(FindError::InvalidRegex("unclosed '[' character class".to_string()));
            };
            let end = if next == '\\' {
                let esc = self.bump().ok_or_else(|| {
                    FindError::InvalidRegex("trailing '\\' in class range".to_string())
                })?;
                if escape_class_item(esc).is_some() {
                    return Err(FindError::InvalidRegex(
                        "shorthand escape cannot end character range".to_string(),
                    ));
                }
                unescape_char(esc)
            } else {
                next
            };
            if start > end {
                return Err(FindError::InvalidRegex(format!(
                    "character class range '{start}-{end}' out of order"
                )));
            }
            items.push(ClassItem::Range(start, end));
        } else {
            items.push(ClassItem::Range(start, start));
        }
        Ok(())
    }
}

fn shorthand_class(negated: bool, item: ClassItem) -> Ast {
    Ast::Matcher(CharMatcher::Class {
        negated,
        items: vec![item],
    })
}

fn escape_class_item(esc: char) -> Option<ClassItem> {
    match esc {
        'd' => Some(ClassItem::Digit),
        'D' => Some(ClassItem::NotDigit),
        'w' => Some(ClassItem::Word),
        'W' => Some(ClassItem::NotWord),
        's' => Some(ClassItem::Space),
        'S' => Some(ClassItem::NotSpace),
        _ => None,
    }
}

fn unescape_char(esc: char) -> char {
    match esc {
        'n' => '\n',
        'r' => '\r',
        't' => '\t',
        other => other,
    }
}

#[derive(Debug, Clone)]
enum Inst {
    Match,
    Char(CharMatcher),
    Assert(AnchorKind),
    Jump(usize),
    Split(usize, usize),
    Save(usize),
}

struct CompiledRegex {
    insts: Vec<Inst>,
    num_slots: usize,
    case_sensitive: bool,
}

impl CompiledRegex {
    fn compile(pattern: &str, case_sensitive: bool) -> Result<Self, FindError> {
        let mut parser = RegexParser::new(pattern);
        let (ast, capture_count) = parser.parse_all()?;
        // Unanchored lazy prefix: 0: Split(3, 1), 1: AnyCharIncludingNewline, 2: Jump(0)
        // followed by Save(0), <ast>, Save(1), Match.
        let mut insts = vec![
            Inst::Split(3, 1),
            Inst::Char(CharMatcher::AnyCharIncludingNewline),
            Inst::Jump(0),
            Inst::Save(0),
        ];
        compile_ast(&ast, &mut insts)?;
        if insts.len() + 2 > MAX_NFA_INSTS {
            return Err(FindError::InvalidRegex(
                "compiled regex exceeds instruction budget".to_string(),
            ));
        }
        insts.push(Inst::Save(1));
        insts.push(Inst::Match);
        Ok(Self {
            insts,
            num_slots: capture_count * 2,
            case_sensitive,
        })
    }

    fn find_all(&self, text: &str, whole_word: bool) -> Vec<FindMatch> {
        let mut matches = Vec::new();
        let mut pos = 0usize;
        while pos <= text.len() {
            let Some(m) = self.find_earliest_at(text, pos) else {
                break;
            };
            if whole_word && !is_whole_word_match(text, m.range) {
                if m.range.start >= text.len() {
                    break;
                }
                pos = advance_one_char(text, m.range.start);
                continue;
            }
            let next_pos = if m.range.is_empty() {
                if m.range.end >= text.len() {
                    matches.push(m);
                    break;
                }
                advance_one_char(text, m.range.end)
            } else {
                m.range.end
            };
            matches.push(m);
            pos = next_pos;
        }
        matches
    }

    fn find_earliest_at(&self, text: &str, start_pos: usize) -> Option<FindMatch> {
        let mut curr: Vec<(usize, Vec<Option<usize>>)> = Vec::new();
        let mut next: Vec<(usize, Vec<Option<usize>>)> = Vec::new();
        let mut seen = vec![0usize; self.insts.len()];
        let mut generation = 1usize;

        let initial_slots = vec![None; self.num_slots];
        self.add_thread(
            &mut curr,
            0,
            initial_slots,
            text,
            start_pos,
            generation,
            &mut seen,
        );

        let mut best: Option<Vec<Option<usize>>> = None;
        let mut pos = start_pos;

        loop {
            let opt_ch = text[pos..].chars().next();
            let char_len = opt_ch.map_or(0, char::len_utf8);
            generation += 1;

            for (pc, slots) in curr.drain(..) {
                // Once we already recorded a match starting at some earlier offset,
                // stop advancing the unanchored background scanner (pc == 1).
                if best.is_some() && pc == 1 {
                    continue;
                }
                match &self.insts[pc] {
                    Inst::Match => {
                        best = Some(slots);
                        // Leftmost-first: discard lower-priority threads at this same pos.
                        break;
                    }
                    Inst::Char(matcher) => {
                        if let Some(ch) = opt_ch
                            && matcher.matches(ch, self.case_sensitive) {
                                self.add_thread(
                                    &mut next,
                                    pc + 1,
                                    slots,
                                    text,
                                    pos + char_len,
                                    generation,
                                    &mut seen,
                                );
                            }
                    }
                    _ => {}
                }
            }

            std::mem::swap(&mut curr, &mut next);
            if curr.is_empty() {
                break;
            }
            match opt_ch {
                Some(_) => pos += char_len,
                None => break,
            }
        }

        best.map(|slots| {
            let start = slots.first().copied().flatten().unwrap_or(start_pos);
            let end = slots.get(1).copied().flatten().unwrap_or(start);
            let range = ByteRange::new(start, end);
            let captures = slots
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&[s, e]| match (s, e) {
                    (Some(s), Some(e)) => Some(ByteRange::new(s, e)),
                    _ => None,
                })
                .collect();
            FindMatch { range, captures }
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn add_thread(
        &self,
        list: &mut Vec<(usize, Vec<Option<usize>>)>,
        pc: usize,
        slots: Vec<Option<usize>>,
        text: &str,
        pos: usize,
        generation: usize,
        seen: &mut [usize],
    ) {
        let mut stack = vec![(pc, slots)];
        while let Some((mut cur_pc, mut cur_slots)) = stack.pop() {
            loop {
                if seen[cur_pc] == generation {
                    break;
                }
                seen[cur_pc] = generation;
                match &self.insts[cur_pc] {
                    Inst::Jump(target) => {
                        cur_pc = *target;
                    }
                    Inst::Split(a, b) => {
                        stack.push((*b, cur_slots.clone()));
                        cur_pc = *a;
                    }
                    Inst::Save(slot) => {
                        if *slot < cur_slots.len() {
                            cur_slots[*slot] = Some(pos);
                        }
                        cur_pc += 1;
                    }
                    Inst::Assert(anchor) => {
                        if anchor.matches(text, pos) {
                            cur_pc += 1;
                        } else {
                            break;
                        }
                    }
                    Inst::Char(_) | Inst::Match => {
                        list.push((cur_pc, cur_slots));
                        break;
                    }
                }
            }
        }
    }
}

fn compile_ast(ast: &Ast, insts: &mut Vec<Inst>) -> Result<(), FindError> {
    if insts.len() > MAX_NFA_INSTS {
        return Err(FindError::InvalidRegex(
            "compiled regex exceeds instruction budget".to_string(),
        ));
    }
    match ast {
        Ast::Empty => Ok(()),
        Ast::Matcher(m) => {
            insts.push(Inst::Char(m.clone()));
            Ok(())
        }
        Ast::Anchor(a) => {
            insts.push(Inst::Assert(*a));
            Ok(())
        }
        Ast::Concat(items) => {
            for item in items {
                compile_ast(item, insts)?;
            }
            Ok(())
        }
        Ast::Alt(branches) => compile_alt(branches, insts),
        Ast::Group { capture_idx, inner } => {
            if let Some(idx) = capture_idx {
                insts.push(Inst::Save(idx * 2));
                compile_ast(inner, insts)?;
                insts.push(Inst::Save(idx * 2 + 1));
                Ok(())
            } else {
                compile_ast(inner, insts)
            }
        }
        Ast::Repeat {
            inner,
            min,
            max,
            greedy,
        } => {
            for _ in 0..*min {
                compile_ast(inner, insts)?;
            }
            match max {
                None => compile_star(inner, *greedy, insts),
                Some(m) => {
                    let optional_count = m.saturating_sub(*min);
                    for _ in 0..optional_count {
                        compile_optional(inner, *greedy, insts)?;
                    }
                    Ok(())
                }
            }
        }
    }
}

fn compile_alt(branches: &[Ast], insts: &mut Vec<Inst>) -> Result<(), FindError> {
    debug_assert!(!branches.is_empty());
    if branches.len() == 1 {
        return compile_ast(&branches[0], insts);
    }
    let mut jumps_to_end = Vec::with_capacity(branches.len() - 1);
    for branch in &branches[..branches.len() - 1] {
        let split_pc = insts.len();
        insts.push(Inst::Split(split_pc + 1, 0));
        compile_ast(branch, insts)?;
        let jmp_pc = insts.len();
        insts.push(Inst::Jump(0));
        jumps_to_end.push(jmp_pc);
        let next_branch = insts.len();
        insts[split_pc] = Inst::Split(split_pc + 1, next_branch);
    }
    compile_ast(branches.last().unwrap(), insts)?;
    let end_pc = insts.len();
    for jmp_pc in jumps_to_end {
        insts[jmp_pc] = Inst::Jump(end_pc);
    }
    Ok(())
}

fn compile_optional(inner: &Ast, greedy: bool, insts: &mut Vec<Inst>) -> Result<(), FindError> {
    let split_pc = insts.len();
    insts.push(Inst::Split(0, 0));
    let body_pc = insts.len();
    compile_ast(inner, insts)?;
    let end_pc = insts.len();
    insts[split_pc] = if greedy {
        Inst::Split(body_pc, end_pc)
    } else {
        Inst::Split(end_pc, body_pc)
    };
    Ok(())
}

fn compile_star(inner: &Ast, greedy: bool, insts: &mut Vec<Inst>) -> Result<(), FindError> {
    let split_pc = insts.len();
    insts.push(Inst::Split(0, 0));
    let body_pc = insts.len();
    compile_ast(inner, insts)?;
    insts.push(Inst::Jump(split_pc));
    let end_pc = insts.len();
    insts[split_pc] = if greedy {
        Inst::Split(body_pc, end_pc)
    } else {
        Inst::Split(end_pc, body_pc)
    };
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literal_case_insensitive_and_whole_word_with_cjk_boundaries() {
        let text = "Hello helloworld, 在hello中说 HELLO 和 cafÉ vs mycafé!";

        // Case-insensitive without whole_word hits all 4 occurrences of "hello".
        let all_ci = find_all_matches(text, "hello", FindOptions::default()).unwrap();
        assert_eq!(all_ci.len(), 4);

        // Case-sensitive matches exact case only.
        let cs = find_all_matches(
            text,
            "hello",
            FindOptions {
                case_sensitive: true,
                whole_word: false,
                use_regex: false,
            },
        )
        .unwrap();
        assert_eq!(cs.len(), 2);

        // Whole-word case-insensitive excludes "helloworld", keeps "Hello", "hello" (inside 在hello中), "HELLO".
        let ww = find_all_matches(
            text,
            "hello",
            FindOptions {
                case_sensitive: false,
                whole_word: true,
                use_regex: false,
            },
        )
        .unwrap();
        let matched: Vec<&str> = ww
            .iter()
            .map(|m| &text[m.range.start..m.range.end])
            .collect();
        assert_eq!(matched, vec!["Hello", "hello", "HELLO"]);

        // Accented Latin whole word excludes "mycafé", matches "cafÉ".
        let cafe = find_all_matches(
            text,
            "café",
            FindOptions {
                case_sensitive: false,
                whole_word: true,
                use_regex: false,
            },
        )
        .unwrap();
        assert_eq!(cafe.len(), 1);
        assert_eq!(&text[cafe[0].range.start..cafe[0].range.end], "cafÉ");
    }

    #[test]
    fn test_regex_pike_vm_quantifiers_groups_anchors_and_empty_progression() {
        let md = "item-12: **alpha**\nitem-345: **beta**\ntail";
        let opts = FindOptions {
            case_sensitive: true,
            whole_word: false,
            use_regex: true,
        };

        let found = find_all_matches(md, r"^item-(\d{2,3}): \*\*(\w+?)\*\*$", opts).unwrap();
        assert_eq!(found.len(), 2);
        assert_eq!(
            &md[found[0].captures[1].unwrap().start..found[0].captures[1].unwrap().end],
            "12"
        );
        assert_eq!(
            &md[found[0].captures[2].unwrap().start..found[0].captures[2].unwrap().end],
            "alpha"
        );
        assert_eq!(
            &md[found[1].captures[1].unwrap().start..found[1].captures[1].unwrap().end],
            "345"
        );
        assert_eq!(
            &md[found[1].captures[2].unwrap().start..found[1].captures[2].unwrap().end],
            "beta"
        );

        // Zero-length regex must advance cleanly across characters instead of looping forever.
        let line_starts = find_all_matches("a\nb\nc", "^", opts).unwrap();
        assert_eq!(line_starts.len(), 3);

        // Adversarial ReDoS pattern finishes instantaneously in linear Pike VM time.
        let redos_text = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa!";
        let redos = find_all_matches(redos_text, "(a+)+b", opts).unwrap();
        assert!(redos.is_empty());

        // Invalid regex reports structured error.
        assert!(find_all_matches("abc", "(unclosed", opts).is_err());
        assert!(find_all_matches("abc", "[z-a]", opts).is_err());
    }

    #[test]
    fn test_next_prev_navigation_and_single_undo_replace_all() {
        let mut buf = TextBuffer::from_str("foo-1 bar foo-2 baz foo-30");
        let mut undo = UndoHistory::new();
        let opts = FindOptions {
            case_sensitive: true,
            whole_word: false,
            use_regex: true,
        };

        let text = buf.to_string();
        let next1 = find_next_match(&text, r"foo-(\d+)", 0, true, opts)
            .unwrap()
            .unwrap();
        assert_eq!(&text[next1.range.start..next1.range.end], "foo-1");
        let next2 = find_next_match(&text, r"foo-(\d+)", next1.range.end, true, opts)
            .unwrap()
            .unwrap();
        assert_eq!(&text[next2.range.start..next2.range.end], "foo-2");

        let wrapped_next = find_next_match(&text, r"foo-(\d+)", text.len(), true, opts)
            .unwrap()
            .unwrap();
        assert_eq!(wrapped_next.range, next1.range);

        let prev_from_2 = find_prev_match(&text, r"foo-(\d+)", next2.range.start, true, opts)
            .unwrap()
            .unwrap();
        assert_eq!(prev_from_2.range, next1.range);
        let wrapped_prev = find_prev_match(&text, r"foo-(\d+)", 0, true, opts)
            .unwrap()
            .unwrap();
        assert_eq!(&text[wrapped_prev.range.start..wrapped_prev.range.end], "foo-30");

        // Replace All expands captures and collapses into ONE undoable transaction.
        let out = replace_all_in_buffer_with_line_ending(
            &mut buf,
            &mut undo,
            Selection::caret(10),
            r"foo-(\d+)",
            "item[$1]",
            opts,
            LineEnding::Lf,
        )
        .unwrap();
        assert_eq!(out.replaced_count, 3);
        assert_eq!(buf.to_string(), "item[1] bar item[2] baz item[30]");

        undo.undo(&mut buf).unwrap();
        assert_eq!(buf.to_string(), "foo-1 bar foo-2 baz foo-30");
        undo.redo(&mut buf).unwrap();
        assert_eq!(buf.to_string(), "item[1] bar item[2] baz item[30]");
    }

    #[test]
    fn test_line_end_anchor_does_not_split_crlf() {
        let mut buf = TextBuffer::from_str("foo\r\nbar\r\nbaz");
        let mut undo = UndoHistory::new();
        let opts = FindOptions {
            case_sensitive: true,
            whole_word: false,
            use_regex: true,
        };

        let text = buf.to_string();
        let anchors = find_all_matches(&text, "$", opts).unwrap();
        let positions: Vec<usize> = anchors.iter().map(|m| m.range.start).collect();
        // Matches only before \r\n (at 3 and 8) and at EOF (13), never between \r and \n.
        assert_eq!(positions, vec![3, 8, 13]);

        let out = replace_all_in_buffer_with_line_ending(
            &mut buf,
            &mut undo,
            Selection::caret(0),
            "$",
            "!",
            opts,
            LineEnding::Lf,
        )
        .unwrap();
        assert_eq!(out.replaced_count, 3);
        assert_eq!(buf.to_string(), "foo!\r\nbar!\r\nbaz!");

        let words = find_all_matches("foo\r\nbar", r"\w+$", opts).unwrap();
        let word_strs: Vec<&str> = words
            .iter()
            .map(|m| &"foo\r\nbar"[m.range.start..m.range.end])
            .collect();
        assert_eq!(word_strs, vec!["foo", "bar"]);
    }

    #[test]
    fn test_deeply_nested_regex_returns_error_instead_of_stack_overflow() {
        let opts = FindOptions {
            case_sensitive: true,
            whole_word: false,
            use_regex: true,
        };
        let nested_5000 = format!("{}a{}", "(".repeat(5000), ")".repeat(5000));
        assert_eq!(nested_5000.len(), 10_001);
        let err = find_all_matches("a", &nested_5000, opts).unwrap_err();
        assert!(matches!(err, FindError::InvalidRegex(_)));

        let boundary_ok = format!("{}a{}", "(?:".repeat(32), ")".repeat(32));
        let ok = find_all_matches("a", &boundary_ok, opts).unwrap();
        assert_eq!(ok.len(), 1);
    }

    #[test]
    fn test_replace_normalizes_newlines_to_dominant_or_preferred_line_ending() {
        let opts = FindOptions {
            case_sensitive: true,
            whole_word: false,
            use_regex: true,
        };
        let mut crlf_buf = TextBuffer::from_str("A\r\nB\r\n");
        let mut undo = UndoHistory::new();
        let out = replace_all_in_buffer_with_line_ending(
            &mut crlf_buf,
            &mut undo,
            Selection::caret(0),
            "A",
            r"X\nY",
            opts,
            LineEnding::CrLf,
        )
        .unwrap();
        assert_eq!(out.replaced_count, 1);
        assert_eq!(crlf_buf.to_string(), "X\r\nY\r\nB\r\n");

        // Single-line CRLF session without prior newlines uses preferred_eol fallback
        let mut single_buf = TextBuffer::from_str("A");
        let m = find_all_matches("A", "A", opts).unwrap().pop().unwrap();
        replace_match_in_buffer_with_line_ending(
            &mut single_buf,
            &mut undo,
            Selection::caret(0),
            &m,
            r"X\nY",
            true,
            LineEnding::CrLf,
        );
        assert_eq!(single_buf.to_string(), "X\r\nY");

        // Mixed-EOL document: `$0` / `$1` preserve captured original `\n` bytes while only
        // template-inserted newlines normalize to dominant CRLF (and `\r\n` yields one break).
        let mut mixed_buf = TextBuffer::from_str("A\r\nB\r\nC\nD\r\n");
        replace_all_in_buffer_with_line_ending(
            &mut mixed_buf,
            &mut undo,
            Selection::caret(0),
            r"(C\nD)",
            r"$1\nE\r\nF",
            opts,
            LineEnding::CrLf,
        )
        .unwrap();
        assert_eq!(mixed_buf.to_string(), "A\r\nB\r\nC\nD\r\nE\r\nF\r\n");

        let mut preserve_only_buf = TextBuffer::from_str("A\r\nB\r\nC\nD\r\n");
        replace_all_in_buffer_with_line_ending(
            &mut preserve_only_buf,
            &mut undo,
            Selection::caret(0),
            r"C\nD",
            "$0",
            opts,
            LineEnding::CrLf,
        )
        .unwrap();
        assert_eq!(preserve_only_buf.to_string(), "A\r\nB\r\nC\nD\r\n");

        // LF document: `\r\n` template collapses to single LF, never doubled
        let mut lf_buf = TextBuffer::from_str("foo\nbar\n");
        replace_all_in_buffer_with_line_ending(
            &mut lf_buf,
            &mut undo,
            Selection::caret(0),
            "foo",
            r"X\r\nY",
            opts,
            LineEnding::Lf,
        )
        .unwrap();
        assert_eq!(lf_buf.to_string(), "X\nY\nbar\n");

        // Unmatched or empty optional capture group between `\r` and `\n` must not split
        // `\r\n` into two separate line breaks.
        let mut opt_crlf = TextBuffer::from_str("bar\r\nbaz\r\n");
        replace_all_in_buffer_with_line_ending(
            &mut opt_crlf,
            &mut undo,
            Selection::caret(0),
            r"(foo)?bar",
            r"X\r$1\nY",
            opts,
            LineEnding::CrLf,
        )
        .unwrap();
        assert_eq!(opt_crlf.to_string(), "X\r\nY\r\nbaz\r\n");

        let mut empty_cap_lf = TextBuffer::from_str("bar\nbaz\n");
        replace_all_in_buffer_with_line_ending(
            &mut empty_cap_lf,
            &mut undo,
            Selection::caret(0),
            r"()bar",
            r"X\r$1\nY",
            opts,
            LineEnding::Lf,
        )
        .unwrap();
        assert_eq!(empty_cap_lf.to_string(), "X\nY\nbaz\n");

        let mut matched_opt_crlf = TextBuffer::from_str("Zbar\r\nbaz\r\n");
        replace_all_in_buffer_with_line_ending(
            &mut matched_opt_crlf,
            &mut undo,
            Selection::caret(0),
            r"(Z)?bar",
            r"X\r$1\nY",
            opts,
            LineEnding::CrLf,
        )
        .unwrap();
        assert_eq!(matched_opt_crlf.to_string(), "X\r\nZ\r\nY\r\nbaz\r\n");
    }
}
