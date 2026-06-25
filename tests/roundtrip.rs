//! Encoder round-trip harness — decode->encode->compare over the binja corpus.
//!
//! For each corpus word `W`, decode it to an [`Instruction`] `I`, then call
//! `I.encode()` and require it to return exactly `W`. This proves the encoder is
//! the inverse of the decoder for every encoding it claims to support.
//!
//! The encoder reconstructs the word purely from `I`'s *semantics* (code,
//! mnemonic, operands, ip) — it never reads `I.word()` — so a match is a genuine
//! semantic round-trip, not a copy.
//!
//! Like [`golden`](../golden.rs), the big sweep is `#[ignore]`d and NON-failing:
//! it prints overall + per-base-group round-trip stats and the biggest gaps, and
//! dumps mismatches to `target/roundtrip-mismatches.txt`. A small NON-ignored
//! unit test round-trips a few hand-built dp_imm words so normal CI still
//! exercises the encoder.
//!
//! Env knobs (shared with golden): `FARM64_CORPUS`, `FARM64_GROUP`,
//! `FARM64_LIMIT`.

#![cfg(feature = "std")]

mod common;

use common::{corpus_path, env_filter, env_limit, stream_corpus, Case};
use fARM64::{Decoder, DecoderOptions, EncodeError};
use std::collections::BTreeMap;
use std::io::Write as _;

/// Per-bucket tally: total cases, how many were attempted (decoded non-Invalid
/// AND the encoder did not return `Unsupported`), how many reproduced the exact
/// corpus word (`matched`), and how many are a true *semantic* round-trip — the
/// re-encoded word decodes back to an equal [`Instruction`] (`semantic`).
///
/// `matched <= semantic`. They diverge only for inputs whose raw encoding
/// carries bits the decoder legitimately discards (e.g. a 32-bit logical
/// immediate whose `immr` field is `>= esize`: the architecture masks it, so
/// `immr` and `immr mod esize` are the *same instruction*, but the corpus word
/// keeps the non-canonical high bits which are irrecoverable from semantics).
#[derive(Debug, Default, Clone, Copy)]
struct Tally {
    total: usize,
    decoded: usize,
    attempted: usize,
    matched: usize,
    semantic: usize,
}

impl Tally {
    fn match_rate(&self) -> f64 {
        if self.attempted == 0 {
            0.0
        } else {
            self.matched as f64 / self.attempted as f64
        }
    }

    fn semantic_rate(&self) -> f64 {
        if self.attempted == 0 {
            0.0
        } else {
            self.semantic as f64 / self.attempted as f64
        }
    }
}

/// A recorded round-trip mismatch / error, for the debug dump.
struct Mismatch {
    word: u32,
    group: String,
    /// The re-encoded word (if `Ok`) or the error spelling (if `Err`).
    got: String,
}

/// Decode one word at `ip = 0` with all features enabled.
fn decode_all(word: u32) -> fARM64::Instruction {
    let bytes = word.to_le_bytes();
    // Default options accept FeatureSet::ALL.
    let mut dec = Decoder::new(&bytes, 0, DecoderOptions::default());
    dec.decode()
}

/// `true` if two instructions are the same *semantically* — equal code,
/// mnemonic, ip, and operand list — ignoring the raw `word`. Used to recognize a
/// re-encoding that the decoder maps back to the identical instruction even when
/// the raw word differs in bits the decoder discards.
fn same_semantics(a: &fARM64::Instruction, b: &fARM64::Instruction) -> bool {
    if a.code() != b.code()
        || a.mnemonic() != b.mnemonic()
        || a.ip() != b.ip()
        || a.op_count() != b.op_count()
    {
        return false;
    }
    (0..a.op_count()).all(|i| a.op(i) == b.op(i))
}

/// Run the round-trip sweep, returning overall + per-base-group tallies and the
/// mismatch list.
fn run_sweep() -> (Tally, BTreeMap<String, Tally>, Vec<Mismatch>) {
    let path = corpus_path();
    let group_filter = env_filter("FARM64_GROUP");
    let limit = env_limit("FARM64_LIMIT");

    let mut overall = Tally::default();
    let mut by_group: BTreeMap<String, Tally> = BTreeMap::new();
    let mut mismatches: Vec<Mismatch> = Vec::new();
    let mut seen = 0usize;

    let parse_result = stream_corpus(&path, |case: Case| {
        if let Some(n) = limit {
            if seen >= n {
                return;
            }
        }
        if let Some(ref f) = group_filter {
            if !case.group.contains(f.as_str()) {
                return;
            }
        }
        seen += 1;

        let g = case.base_group().to_string();
        let bucket = by_group.entry(g).or_default();
        bucket.total += 1;
        overall.total += 1;

        let insn = decode_all(case.word);
        if insn.is_invalid() {
            return; // not decodable -> not part of the round-trip set.
        }
        bucket.decoded += 1;
        overall.decoded += 1;

        match insn.encode() {
            // `Unsupported` means a group encoder is not implemented yet — not a
            // failure, just out of scope (do not count as attempted).
            Err(EncodeError::Unsupported) => {}
            Ok(w) => {
                bucket.attempted += 1;
                overall.attempted += 1;
                if w == case.word {
                    bucket.matched += 1;
                    overall.matched += 1;
                    bucket.semantic += 1;
                    overall.semantic += 1;
                } else {
                    // Not an exact word match: is it nonetheless a true semantic
                    // round-trip? (the re-encoded word decodes to a
                    // semantically-equal Instruction). This catches raw bits the
                    // decoder legitimately discards. We compare everything EXCEPT
                    // the raw `word` (which is what differs by construction).
                    let re = decode_all(w);
                    if !re.is_invalid() && same_semantics(&re, &insn) {
                        bucket.semantic += 1;
                        overall.semantic += 1;
                    }
                    mismatches.push(Mismatch {
                        word: case.word,
                        group: case.group.clone(),
                        got: format!("{w:08X}"),
                    });
                }
            }
            Err(e) => {
                // An encoder that is supposed to handle this code but errored is
                // a genuine round-trip failure.
                bucket.attempted += 1;
                overall.attempted += 1;
                mismatches.push(Mismatch {
                    word: case.word,
                    group: case.group.clone(),
                    got: format!("ERR:{e:?}"),
                });
            }
        }
    });

    if let Err(e) = parse_result {
        panic!("cannot stream corpus at {}: {e}", path.display());
    }

    (overall, by_group, mismatches)
}

/// Dump the full mismatch list to `target/roundtrip-mismatches.txt`.
fn dump_mismatches(mismatches: &[Mismatch]) {
    let out_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("roundtrip-mismatches.txt");
    if let Some(dir) = out_path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    match std::fs::File::create(&out_path) {
        Ok(mut f) => {
            let _ = writeln!(
                f,
                "# fARM64 round-trip mismatches: {} entries\n# format: WORD\tGROUP\tGOT-or-ERR",
                mismatches.len()
            );
            for m in mismatches {
                let _ = writeln!(f, "{:08X}\t{}\t{}", m.word, m.group, m.got);
            }
            eprintln!(
                "[roundtrip] wrote {} mismatches to {}",
                mismatches.len(),
                out_path.display()
            );
        }
        Err(e) => eprintln!("[roundtrip] could not write mismatch dump: {e}"),
    }
}

/// Print a readable summary to stderr.
fn print_summary(overall: &Tally, by_group: &BTreeMap<String, Tally>) {
    let pct = |n: usize, d: usize| -> f64 {
        if d == 0 {
            0.0
        } else {
            100.0 * n as f64 / d as f64
        }
    };

    eprintln!();
    eprintln!("=== fARM64 encoder round-trip ===");
    eprintln!(
        "overall: total={} decoded={} attempted={} ({:.2}% of decoded)",
        overall.total,
        overall.decoded,
        overall.attempted,
        pct(overall.attempted, overall.decoded),
    );
    eprintln!(
        "  exact-word matched={} ({:.2}% of attempted); semantic round-trip={} ({:.2}% of attempted)",
        overall.matched,
        overall.match_rate() * 100.0,
        overall.semantic,
        overall.semantic_rate() * 100.0,
    );
    eprintln!("groups: {}", by_group.len());

    // Per-base-group rows that the encoder ATTEMPTED, sorted worst-rate first.
    let mut attempted: Vec<(&String, &Tally)> =
        by_group.iter().filter(|(_, t)| t.attempted > 0).collect();
    attempted.sort_by(|a, b| {
        a.1.match_rate()
            .partial_cmp(&b.1.match_rate())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.1.attempted.cmp(&a.1.attempted))
    });

    eprintln!();
    eprintln!("--- per-base-group round-trip (attempted>0) ---");
    eprintln!(
        "{:<16} {:>7} {:>8} {:>9} {:>7} {:>7} {:>8} {:>8}",
        "GROUP", "total", "decoded", "attempt", "match", "sem", "word%", "sem%"
    );
    for (g, t) in attempted.iter() {
        eprintln!(
            "{:<16} {:>7} {:>8} {:>9} {:>7} {:>7} {:>7.2} {:>7.2}",
            g,
            t.total,
            t.decoded,
            t.attempted,
            t.matched,
            t.semantic,
            t.match_rate() * 100.0,
            t.semantic_rate() * 100.0,
        );
    }

    // Biggest gaps: decoded but not yet attempted by the encoder (other groups).
    let mut gaps: Vec<(&String, usize, &Tally)> = by_group
        .iter()
        .map(|(g, t)| (g, t.decoded.saturating_sub(t.attempted), t))
        .filter(|(_, gap, _)| *gap > 0)
        .collect();
    gaps.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    eprintln!();
    eprintln!("--- biggest gaps (decoded but encoder Unsupported) ---");
    eprintln!("{:<16} {:>8} {:>9} {:>7}", "GROUP", "decoded", "attempt", "gap");
    for (g, gap, t) in gaps.iter().take(30) {
        eprintln!("{:<16} {:>8} {:>9} {:>7}", g, t.decoded, t.attempted, gap);
    }
    eprintln!("=================================");
}

#[test]
#[ignore = "large corpus sweep; run with `--ignored`. Non-failing report."]
fn roundtrip_corpus() {
    let (overall, by_group, mismatches) = run_sweep();
    assert!(
        overall.total > 0,
        "corpus produced zero cases (path: {})",
        corpus_path().display()
    );
    print_summary(&overall, &by_group);
    dump_mismatches(&mismatches);
    // Non-failing report (matches golden.rs style); no threshold gate yet.
}

/// NON-ignored: round-trip a handful of known dp_imm words through
/// decode->encode and require the exact word back. Runs in normal CI.
#[test]
fn roundtrip_known_dp_imm_words() {
    // (word, human description) — a representative spread of the dp_imm group
    // including canonical encodings and the alias-resolving forms.
    let words: &[(u32, &str)] = &[
        (0x1100_0420, "ADD w0, w1, #1"),
        (0x9100_0420, "ADD x0, x1, #1"),
        (0x9140_0420, "ADD x0, x1, #1, lsl #12"),
        (0xD100_0420, "SUB x0, x1, #1"),
        (0xF100_043F, "CMP x1, #1 (SUBS xzr,...)"),
        (0x9100_03E0, "MOV x0, sp (ADD x0, sp, #0)"),
        (0x9240_1C00, "AND x0, x0, #0xff"),
        (0x3200_0000, "ORR w0, w0, #1"),
        (0xF240_001F, "TST x0, #1 (ANDS xzr,...)"),
        (0xD280_0020, "MOVZ x0, #1"),
        (0xD2A0_0020, "MOVZ x0, #1, lsl #16 -> MOV x0, #0x10000"),
        (0x7280_0020, "MOVK w0, #1"),
        (0x9280_0000, "MOVN x0, #0 -> MOV x0, #-1"),
        (0x9344_FC20, "ASR x0, x1, #4 (SBFM)"),
        (0xD37C_EC20, "LSL x0, x1, #4 (UBFM)"),
        (0xD344_FC20, "LSR x0, x1, #4 (UBFM)"),
        (0xB37C_0C20, "BFI x0, x1, #4, #4 (BFM)"),
        (0x93C2_1020, "EXTR x0, x1, x2, #4"),
        (0x1000_0000, "ADR x0, .+0"),
        (0x9000_0000, "ADRP x0, page"),
    ];

    for &(word, desc) in words {
        let insn = decode_all(word);
        assert!(!insn.is_invalid(), "{desc}: {word:#010x} did not decode");
        let got = insn
            .encode()
            .unwrap_or_else(|e| panic!("{desc}: encode failed: {e:?}"));
        assert_eq!(
            got, word,
            "{desc}: round-trip mismatch {word:#010x} -> {got:#010x} (code={:?}, mnem={:?})",
            insn.code(),
            insn.mnemonic()
        );
    }
}

/// NON-ignored: exhaustively round-trip the FEAT_MOPS Memory Copy / Memory Set
/// space. Every allocated `(family, op1, op2)` with a register triple that
/// satisfies the distinctness constraints must decode, re-encode to the exact
/// same word, and re-decode to an equal instruction.
#[test]
fn roundtrip_mops() {
    // Build a MOPS word from its fields (mirrors the encoder layout).
    fn mops(family: u32, op1: u32, op2: u32, rn: u32, rs: u32, rd: u32) -> u32 {
        (0b00011 << 27)
            | (family << 26)
            | (0b01 << 24)
            | (op1 << 21)
            | (rn << 16)
            | (op2 << 12)
            | (0b01 << 10)
            | (rs << 5)
            | rd
    }

    let mut checked = 0usize;
    for family in 0..2u32 {
        // Copy stages (op1 0/2/4) have 16 option variants; Set (op1 6) has 12.
        let stage_opts: &[(u32, u32)] = &[(0, 16), (2, 16), (4, 16), (6, 12)];
        for &(op1, n_opts) in stage_opts {
            for op2 in 0..n_opts {
                // A register triple that is pairwise distinct and avoids the
                // illegal destination `31`, with a sweep so each field varies.
                for &(rd, rs, rn) in &[(5u32, 6u32, 7u32), (1, 2, 3), (10, 20, 30)] {
                    let word = mops(family, op1, op2, rn, rs, rd);
                    let insn = decode_all(word);
                    assert!(
                        !insn.is_invalid(),
                        "MOPS {word:#010x} (family={family} op1={op1} op2={op2}) did not decode"
                    );
                    let got = insn
                        .encode()
                        .unwrap_or_else(|e| panic!("MOPS {word:#010x} encode failed: {e:?}"));
                    assert_eq!(
                        got, word,
                        "MOPS round-trip word mismatch {word:#010x} -> {got:#010x} (code={:?})",
                        insn.code()
                    );
                    // Re-decode the re-encoded word: must equal the original.
                    let insn2 = decode_all(got);
                    assert_eq!(
                        insn, insn2,
                        "MOPS re-decode mismatch for {word:#010x} (code={:?})",
                        insn.code()
                    );
                    checked += 1;
                }
            }
        }
    }
    // 2 families * (16+16+16+12) options * 3 register triples.
    assert_eq!(checked, 2 * (16 + 16 + 16 + 12) * 3);
}
