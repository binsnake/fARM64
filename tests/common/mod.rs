//! Shared helpers for the fARM64 validation harness (golden + llvm_diff).
//!
//! This module is `std`-only test support; the library itself is `no_std`. It
//! provides:
//!
//! * [`normalize`] — a conservative, documented text-normalization pass that
//!   makes stylistically-equivalent disassembly strings compare equal.
//! * [`disasm_farm64`] — decode + format one A64 word with fARM64.
//! * corpus streaming/parsing helpers shared by both differential tests.
//!
//! Nothing here is authoritative: correctness is defined against the ARM ARM.
//! These oracles (the binja corpus and LLVM) are development guides only.

#![allow(dead_code)] // each test binary uses a different subset of helpers.

use std::path::PathBuf;

/// Address every case is decoded at. Matches the corpus's PC-relative targets
/// in `test_cases.txt` (the same anchor the original scaffold used).
pub const ADDRESS_TEST: u64 = 0x8000_0000_0000_0004;

// ---------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------

/// Normalize a disassembly string so that stylistically-equivalent renderings
/// compare equal, without erasing real semantic differences.
///
/// The pass is deliberately conservative — it only smooths *formatting* noise:
///
/// 1. Drop a trailing comment introduced by `//` or `;` (oracle annotations).
/// 2. Lower-case everything (mnemonics/registers are case-insensitive in UAL).
/// 3. Turn tabs into spaces (the mnemonic/operand separator differs per tool —
///    fARM64 pads with spaces, `llvm-mc` uses a tab).
/// 4. Collapse every run of whitespace to a single space and trim the ends.
/// 5. Remove a space *immediately before* a `,` and ensure exactly one space
///    *after* a `,` (operand-separator spacing is purely cosmetic).
/// 6. Remove spaces just inside brackets/braces `[ x ]` -> `[x]`, `{ x }` ->
///    `{x}` (bracket padding is cosmetic).
///
/// It intentionally does **not** rewrite immediate radices, expand register
/// ranges, or canonicalize alias spellings: those can mask genuine decoder
/// bugs, so we keep them as visible mismatches for now.
pub fn normalize(s: &str) -> String {
    // (1) Drop trailing `//` or `;` comment.
    let mut body = s;
    if let Some(idx) = body.find("//") {
        body = &body[..idx];
    }
    if let Some(idx) = body.find(';') {
        body = &body[..idx];
    }

    // (2)+(3) Lower-case and turn tabs into spaces in one pass.
    let lowered: String = body
        .chars()
        .map(|c| if c == '\t' { ' ' } else { c.to_ascii_lowercase() })
        .collect();

    // (4)+(5)+(6) in one pass. `pending_space` means "at least one whitespace
    // char was seen since the last emitted glyph"; it is only materialized as a
    // single space when the next glyph actually wants one.
    let mut out = String::with_capacity(lowered.len());
    let mut pending_space = false;
    for ch in lowered.chars() {
        if ch.is_whitespace() {
            pending_space = true;
            continue;
        }

        // (5)/(6): drop any pending space before a separator/closer.
        let suppress_before = matches!(ch, ',' | ']' | '}' | ')');
        // (6): drop any pending space right after an opener.
        let after_opener = out.ends_with(['[', '{', '(']);

        if pending_space && !out.is_empty() && !suppress_before && !after_opener {
            out.push(' ');
        }
        pending_space = false;

        out.push(ch);

        // After a comma we always want exactly one space before the next glyph.
        // Represent that as a pending space (collapsed with any literal space
        // that follows), so we never emit a double space.
        if ch == ',' {
            pending_space = true;
        }
    }

    // A trailing comma leaves a pending space that was never emitted; nothing to
    // trim in `out` itself, but guard against any stray edge whitespace.
    out.trim().to_string()
}

// ---------------------------------------------------------------------------
// fARM64 decode + format
// ---------------------------------------------------------------------------

/// Decode `word` with fARM64 at [`ADDRESS_TEST`] and return
/// `(is_invalid, formatted_text)` using the default [`FmtFormatter`].
///
/// `is_invalid` is `true` when the case was *not attempted* — either the decoder
/// produced [`Code::Invalid`] OR the (currently stubbed) group decoder panicked
/// via `todo!()`. Panics are caught so a single unimplemented group cannot abort
/// the whole corpus sweep; the caller treats both as "not attempted".
///
/// Install [`silence_panics`] once before a bulk sweep to suppress the default
/// panic-hook backtrace spam from the many `todo!()` stubs.
pub fn disasm_farm64(word: u32) -> (bool, String) {
    use fARM64::format::{FmtFormatter, Formatter};
    use fARM64::{Decoder, DecoderOptions};

    let result = std::panic::catch_unwind(|| {
        let bytes = word.to_le_bytes();
        let mut dec = Decoder::new(&bytes, ADDRESS_TEST, DecoderOptions::default());
        let insn = dec.decode();

        let mut text = String::new();
        FmtFormatter::new().format(&insn, &mut text);
        (insn.is_invalid(), text)
    });

    match result {
        Ok(pair) => pair,
        // A stubbed group decoder panicked (`todo!()`): treat as "not attempted".
        Err(_) => (true, String::from("<unimplemented>")),
    }
}

/// Replace the panic hook with a no-op so that the flood of `todo!()` panics
/// raised during a corpus sweep does not print thousands of backtraces. Call
/// once at the start of a bulk run. Idempotent enough for test use.
pub fn silence_panics() {
    std::panic::set_hook(Box::new(|_info| {}));
}

// ---------------------------------------------------------------------------
// Corpus parsing
// ---------------------------------------------------------------------------

/// One parsed corpus case: the instruction word, its expected disassembly text,
/// and the encoding-group key it belongs to (first token after `//`).
#[derive(Debug, Clone)]
pub struct Case {
    pub word: u32,
    pub expected: String,
    pub group: String,
}

impl Case {
    /// The "base group": the group key with any trailing `_..._` form suffix
    /// trimmed to the leading alphabetic mnemonic-ish prefix, for coarser
    /// bucketing in the summary (e.g. `REVD_Z_P_Z_` -> `REVD`).
    pub fn base_group(&self) -> &str {
        let g = self.group.as_str();
        match g.find('_') {
            Some(idx) if idx > 0 => &g[..idx],
            _ => g,
        }
    }
}

/// Locate the differential corpus relative to the crate root, honoring the
/// `FARM64_CORPUS` environment override.
pub fn corpus_path() -> PathBuf {
    if let Ok(p) = std::env::var("FARM64_CORPUS") {
        return PathBuf::from(p);
    }
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("refs");
    p.push("arch-arm64-master");
    p.push("disassembler");
    p.push("test_cases.txt");
    p
}

/// Heuristic: does `tok` look like the bit-pattern field of a KEY header line
/// (e.g. `00000101|size=00|101110100|Pg=xxx|...`)? Such tokens contain a `|` or
/// `=`, or consist solely of the encoding alphabet `0/1/x`. The operand-syntax
/// comment line's second token is an operand like `<Zd>.q,` and fails this.
fn is_bitpattern(tok: &str) -> bool {
    if tok.is_empty() {
        return false;
    }
    if tok.contains('|') || tok.contains('=') {
        return true;
    }
    tok.chars().all(|c| matches!(c, '0' | '1' | 'x'))
}

/// Stream-parse the corpus file line by line, invoking `f` for each data case
/// as it is parsed (so the whole corpus is never held in memory at once by the
/// caller). Comment lines (`//`) update the running group label.
///
/// Returns the total number of data cases seen.
pub fn stream_corpus<F: FnMut(Case)>(path: &std::path::Path, mut f: F) -> std::io::Result<usize> {
    use std::io::{BufRead, BufReader};

    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut group = String::new();
    let mut count = 0usize;

    for line in reader.lines() {
        let line = line?;
        // Strip a leading UTF-8 BOM (the corpus file begins with one, which
        // would otherwise prevent the first `// KEY` line from matching).
        let line = line.strip_prefix('\u{feff}').unwrap_or(&line);
        let line = line.trim_end();

        if let Some(rest) = line.strip_prefix("//") {
            // Each encoding has TWO comment lines:
            //   `// KEY 00000101|size=00|...`   (the encoding-header / KEY line)
            //   `// MNEMONIC <Zd>.Q, <Zn>.Q`    (the operand-SYNTAX line)
            // The group key is the first token of the *KEY* line only; we must
            // not let the SYNTAX line (whose first token is the bare mnemonic)
            // overwrite it. A KEY line is identified by its second token being a
            // bitpattern (contains `|` or `=`, or is made only of 0/1/x/`|`).
            let mut toks = rest.split_whitespace();
            let first = toks.next().unwrap_or("");
            let second = toks.next().unwrap_or("");
            if is_bitpattern(second) {
                group = first.to_string();
            }
            continue;
        }
        // Data line: `WWWWWWWW text` (word is 8 big-endian hex digits).
        if line.len() < 9 {
            continue;
        }
        let (hex, expected) = line.split_at(8);
        let hex = hex.trim();
        if hex.len() != 8 {
            continue;
        }
        if let Ok(word) = u32::from_str_radix(hex, 16) {
            f(Case {
                word,
                expected: expected.trim().to_string(),
                group: group.clone(),
            });
            count += 1;
        }
    }
    Ok(count)
}

/// Read an optional `usize` env var (e.g. `FARM64_LIMIT`).
pub fn env_limit(name: &str) -> Option<usize> {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok())
}

/// Read an optional substring filter env var (e.g. `FARM64_GROUP`).
pub fn env_filter(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|s| !s.is_empty())
}
