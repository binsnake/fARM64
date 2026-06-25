//! Differential harness — the binja-corpus oracle (NOT authoritative).
//!
//! Correctness for fARM64 is defined against the ARM Architecture Reference
//! Manual. This harness uses a third-party disassembler's published
//! `test_cases.txt` corpus (read locally from `refs/`, never shipped) as *one*
//! differential oracle / development guide. Where the corpus deliberately
//! diverges from the ARM ARM, fARM64 follows the spec; for now every divergence
//! is simply reported, never failed.
//!
//! It streams the corpus, decodes each instruction at [`common::ADDRESS_TEST`],
//! formats with the default [`fARM64::format::FmtFormatter`], normalizes both
//! sides ([`common::normalize`]) and compares.
//!
//! The big test is `#[ignore]`d (run with `--ignored`) and currently
//! NON-FAILING: it always passes and just prints a coverage/parity summary. Flip
//! [`MATCH_THRESHOLD`] above 0.0 once real decoders land to turn it into a
//! regression gate.
//!
//! Env knobs:
//! * `FARM64_CORPUS=<path>` — override the corpus location.
//! * `FARM64_GROUP=<substr>` — only run cases whose group key contains `substr`.
//! * `FARM64_LIMIT=<n>` — stop after `n` cases (post-filter).

#![cfg(feature = "std")]

mod common;

use common::{
    corpus_path, env_filter, env_limit, normalize, stream_corpus, Case,
};
use std::collections::BTreeMap;
use std::io::Write as _;

/// Fraction of *attempted* cases that must match for the gated `#[test]` to be
/// considered a pass. While decoders are stubbed this is `0.0` (always passes).
/// Raise it (e.g. to `0.95`) to turn the golden run into a regression gate.
const MATCH_THRESHOLD: f64 = 0.0;

/// Per-bucket tally: how many cases, how many were attempted (decoded to a
/// non-`Invalid` instruction), and how many of those matched the oracle.
#[derive(Debug, Default, Clone, Copy)]
struct Tally {
    total: usize,
    attempted: usize,
    matched: usize,
}

impl Tally {
    fn record(&mut self, attempted: bool, matched: bool) {
        self.total += 1;
        if attempted {
            self.attempted += 1;
            if matched {
                self.matched += 1;
            }
        }
    }

    /// Match rate among attempted cases (`0.0` if none attempted).
    fn match_rate(&self) -> f64 {
        if self.attempted == 0 {
            0.0
        } else {
            self.matched as f64 / self.attempted as f64
        }
    }
}

/// A recorded mismatch, for the debug dump.
struct Mismatch {
    word: u32,
    group: String,
    expected: String,
    got: String,
}

/// Run the differential sweep, returning the overall tally, per-base-group
/// tallies, and the mismatch list. Shared by the gated test and an explicit
/// runner.
fn run_sweep() -> (Tally, BTreeMap<String, Tally>, Vec<Mismatch>) {
    // The group decoders are still `todo!()` stubs that panic; silence the panic
    // hook so the sweep does not print thousands of backtraces (each decode is
    // wrapped in `catch_unwind` inside `disasm_farm64`).
    common::silence_panics();

    let path = corpus_path();
    let group_filter = env_filter("FARM64_GROUP");
    let limit = env_limit("FARM64_LIMIT");

    let mut overall = Tally::default();
    let mut by_group: BTreeMap<String, Tally> = BTreeMap::new();
    let mut mismatches: Vec<Mismatch> = Vec::new();
    let mut seen = 0usize;

    let parse_result = stream_corpus(&path, |case: Case| {
        // Honor the post-filter limit.
        if let Some(n) = limit {
            if seen >= n {
                return;
            }
        }
        // Group substring filter.
        if let Some(ref f) = group_filter {
            if !case.group.contains(f.as_str()) {
                return;
            }
        }
        seen += 1;

        let (is_invalid, got_raw) = common::disasm_farm64(case.word);
        let attempted = !is_invalid;

        let norm_got = normalize(&got_raw);
        let norm_exp = normalize(&case.expected);
        let matched = attempted && norm_got == norm_exp;

        overall.record(attempted, matched);
        by_group
            .entry(case.base_group().to_string())
            .or_default()
            .record(attempted, matched);

        // Record mismatches only among attempted cases (a stubbed decode is not
        // a mismatch, it is simply "not yet implemented").
        if attempted && !matched {
            mismatches.push(Mismatch {
                word: case.word,
                group: case.group.clone(),
                expected: case.expected.clone(),
                got: got_raw,
            });
        }
    });

    if let Err(e) = parse_result {
        panic!("cannot stream corpus at {}: {e}", path.display());
    }

    (overall, by_group, mismatches)
}

/// Write the full mismatch list to `target/golden-mismatches.txt` for debugging.
fn dump_mismatches(mismatches: &[Mismatch]) {
    let out_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("golden-mismatches.txt");
    if let Some(dir) = out_path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    match std::fs::File::create(&out_path) {
        Ok(mut f) => {
            let _ = writeln!(
                f,
                "# fARM64 golden mismatches: {} entries\n# format: WORD\tGROUP\tEXPECTED\t||\tGOT",
                mismatches.len()
            );
            for m in mismatches {
                let _ = writeln!(
                    f,
                    "{:08X}\t{}\t{}\t||\t{}",
                    m.word, m.group, m.expected, m.got
                );
            }
            eprintln!("[golden] wrote {} mismatches to {}", mismatches.len(), out_path.display());
        }
        Err(e) => eprintln!("[golden] could not write mismatch dump: {e}"),
    }
}

/// Print a readable summary table to stderr.
fn print_summary(overall: &Tally, by_group: &BTreeMap<String, Tally>) {
    let pct = |n: usize, d: usize| -> f64 {
        if d == 0 {
            0.0
        } else {
            100.0 * n as f64 / d as f64
        }
    };

    eprintln!();
    eprintln!("=== fARM64 golden corpus parity ===");
    eprintln!(
        "overall: total={} attempted={} ({:.2}%) matched={} ({:.2}% of attempted)",
        overall.total,
        overall.attempted,
        pct(overall.attempted, overall.total),
        overall.matched,
        overall.match_rate() * 100.0,
    );
    eprintln!("groups: {}", by_group.len());

    // Worst groups: those with at least one attempted case but a low match
    // rate, sorted by (match_rate asc, attempted desc). Only meaningful once
    // decoders land; harmless (empty) while everything is stubbed.
    let mut worst: Vec<(&String, &Tally)> = by_group
        .iter()
        .filter(|(_, t)| t.attempted > 0)
        .collect();
    worst.sort_by(|a, b| {
        a.1.match_rate()
            .partial_cmp(&b.1.match_rate())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.1.attempted.cmp(&a.1.attempted))
    });

    eprintln!();
    eprintln!("--- worst groups (attempted>0, lowest match rate) ---");
    if worst.is_empty() {
        eprintln!("(none attempted yet — decoders are stubbed)");
    } else {
        eprintln!("{:<16} {:>7} {:>9} {:>7} {:>8}", "GROUP", "total", "attempt", "match", "rate%");
        for (g, t) in worst.iter().take(25) {
            eprintln!(
                "{:<16} {:>7} {:>9} {:>7} {:>7.2}",
                g,
                t.total,
                t.attempted,
                t.matched,
                t.match_rate() * 100.0
            );
        }
    }

    // Per-base-group table (full), sorted by total desc, capped for readability.
    let mut all: Vec<(&String, &Tally)> = by_group.iter().collect();
    all.sort_by(|a, b| b.1.total.cmp(&a.1.total).then(a.0.cmp(b.0)));
    eprintln!();
    eprintln!("--- per-base-group (top 40 by size) ---");
    eprintln!("{:<16} {:>7} {:>9} {:>7} {:>8}", "GROUP", "total", "attempt", "match", "rate%");
    for (g, t) in all.iter().take(40) {
        eprintln!(
            "{:<16} {:>7} {:>9} {:>7} {:>7.2}",
            g,
            t.total,
            t.attempted,
            t.matched,
            t.match_rate() * 100.0
        );
    }
    // Biggest coverage gaps: base-groups with the most NOT-yet-decoded cases
    // (total - attempted), so the next decode work can be targeted precisely.
    let mut gaps: Vec<(&String, usize, &Tally)> = by_group
        .iter()
        .map(|(g, t)| (g, t.total - t.attempted, t))
        .filter(|(_, gap, _)| *gap > 0)
        .collect();
    gaps.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    eprintln!();
    eprintln!("--- biggest coverage gaps (total - attempted) ---");
    eprintln!("{:<16} {:>7} {:>9} {:>7}", "GROUP", "total", "attempt", "gap");
    for (g, gap, t) in gaps.iter().take(30) {
        eprintln!("{:<16} {:>7} {:>9} {:>7}", g, t.total, t.attempted, gap);
    }
    eprintln!("===================================");
}

#[test]
#[ignore = "large corpus sweep; run with `--ignored`. Non-failing report until MATCH_THRESHOLD is raised."]
fn golden_corpus_parity() {
    let (overall, by_group, mismatches) = run_sweep();

    assert!(
        overall.total > 0,
        "corpus produced zero cases (path: {})",
        corpus_path().display()
    );

    print_summary(&overall, &by_group);
    dump_mismatches(&mismatches);

    // Regression gate, currently disabled (threshold 0.0 => always passes).
    // Once decoders land, raise MATCH_THRESHOLD to enforce parity.
    let rate = overall.match_rate();
    assert!(
        rate >= MATCH_THRESHOLD,
        "match rate {:.4} fell below threshold {:.4} ({} matched / {} attempted)",
        rate,
        MATCH_THRESHOLD,
        overall.matched,
        overall.attempted
    );
}

/// Smoke test that the corpus file is present and parses to a non-trivial count.
/// Does not touch the decoder, so it runs in normal CI.
#[test]
fn corpus_is_present_and_parses() {
    let path = corpus_path();
    let mut first: Option<Case> = None;
    let mut count = 0usize;
    let total = stream_corpus(&path, |c| {
        if first.is_none() {
            first = Some(c.clone());
        }
        count += 1;
    })
    .unwrap_or_else(|e| panic!("cannot read corpus at {}: {e}", path.display()));

    assert_eq!(total, count);
    assert!(count > 40_000, "expected >40k corpus cases, parsed {count}");
    // The corpus opens with the SVE REVD encoding (architectural anchor).
    let first = first.expect("no cases parsed");
    assert_eq!(first.word, 0x052E_93FD, "unexpected first corpus word");
    assert_eq!(first.group, "REVD_Z_P_Z_", "unexpected first group key");
}

/// Prove the harness comparison logic itself is correct, independent of the
/// (currently stubbed) decoder. We hand-build the normalized comparison on a few
/// (got, expected) pairs and assert `normalize` collapses cosmetic differences
/// while preserving semantic ones. This test is NOT ignored — it runs in CI.
#[test]
fn normalize_makes_equivalent_strings_equal() {
    // Cosmetic-only differences must normalize equal.
    let equivalent_pairs = [
        // tab vs space-padding between mnemonic and operands (fARM64 pads,
        // llvm-mc tabs).
        ("add\tw0, w1, #1", "add     w0, w1, #1"),
        // case-insensitivity.
        ("ADD X0, X1, X2", "add x0, x1, x2"),
        // operand-separator spacing.
        ("add x0,x1,x2", "add x0, x1, x2"),
        // bracket padding.
        ("ldr x0, [ x1 ]", "ldr x0, [x1]"),
        // trailing comment stripped.
        ("ret // return", "ret"),
        ("nop ; filler", "nop"),
        // a space before a comma is removed.
        ("cmp x0 , x1", "cmp x0, x1"),
        // collapse multiple internal spaces.
        ("mov   x0,    x1", "mov x0, x1"),
    ];
    for (a, b) in equivalent_pairs {
        assert_eq!(
            normalize(a),
            normalize(b),
            "expected equivalent after normalize: {a:?} vs {b:?}"
        );
    }

    // Genuinely different strings must stay different (normalize is
    // conservative — it does NOT rewrite radices, aliases, or register names).
    let different_pairs = [
        ("add x0, x1, x2", "add x0, x1, x3"), // different register
        ("add x0, x1, #1", "sub x0, x1, #1"), // different mnemonic
        ("mov x0, #0x10", "mov x0, #16"),     // radix NOT normalized (kept visible)
        ("ldr x0, [x1]", "ldr x0, [x1, #8]"), // different operand
        ("b.eq 0x1000", "b.ne 0x1000"),       // different condition
    ];
    for (a, b) in different_pairs {
        assert_ne!(
            normalize(a),
            normalize(b),
            "expected NOT equal after normalize: {a:?} vs {b:?}"
        );
    }

    // Spot-check the exact normalized form so the rules are pinned.
    assert_eq!(normalize("ADD\tW0 ,  W1,#1  // c"), "add w0, w1, #1");
    assert_eq!(normalize("LDR  x0, [ x1 , #8 ]!"), "ldr x0, [x1, #8]!");
}
