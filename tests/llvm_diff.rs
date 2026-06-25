//! LLVM-differential discovery + validation harness for fARM64.
//!
//! The binja corpus reached 100% parity, so it is exhausted as an oracle. This
//! test uses **LLVM 21** (`llvm-mc --disassemble`) as the oracle to discover
//! which A64 encodings LLVM decodes that fARM64 still returns `Invalid` for —
//! the prioritized list that drives new-extension implementation work.
//!
//! It is a **report-only** sweep: it NEVER hard-fails on a decode difference
//! (only on harness/IO bugs), and it does not touch any decode/encode logic.
//!
//! ## Oracle invocation
//!
//! llvm-objdump in LLVM 21 cannot ingest a headerless raw blob (no `-b binary`),
//! so we use `llvm-mc --disassemble --triple=aarch64 --mattr=+all`, which reads
//! a stream of `0xNN,0xNN,0xNN,0xNN` lines from stdin and disassembles the whole
//! batch in ONE process. `--mattr=+all` enables every extension the installed
//! LLVM 21 supports (verified to accept SVE/SME/MOPS/CSSC/... encodings).
//! Aliases are left ON (no `-M no-aliases`) so output matches our preferred
//! disassembly style; the comparison is lenient (valid-vs-invalid + first-token).
//!
//! ## Word→text alignment (the subtle part)
//!
//! llvm-mc emits exactly ONE tab-led instruction line on **stdout** per VALID
//! word, in order, and emits NOTHING on stdout for an invalid word — instead it
//! writes `<stdin>:LINE:COL: warning: invalid instruction encoding` to
//! **stderr**, where LINE is the 1-based input line. We put one word per input
//! line, collect the set of invalid line numbers from stderr, then walk the
//! words in order: every non-invalid index consumes the next stdout line. This
//! reconstructs the word→text mapping exactly even when invalids interleave.
//!
//! ## Run
//!
//! ```text
//! cargo test --features "std full" -- --ignored --nocapture llvm_diff
//! ```
//!
//! Env knobs:
//! * `FARM64_LLVM_MC=<exe>` — override the `llvm-mc` binary path.
//! * `FARM64_DIFF_WORDS=<n>` — cap the de-duped sample size (default ~6M).

#![cfg(feature = "std")]

mod common;

use std::collections::HashMap;
use std::io::Write as _;
use std::process::{Command, Stdio};

use fARM64::decode::decode;
use fARM64::FeatureSet;

const ADDRESS: u64 = 0x8000_0000_0000_0004;

// ---------------------------------------------------------------------------
// Oracle discovery
// ---------------------------------------------------------------------------

/// Default `llvm-mc` locations to probe (the user's install is at LLVM21).
const LLVM_MC_CANDIDATES: &[&str] = &[
    "C:/Program Files/LLVM21/bin/llvm-mc.exe",
    "C:/Program Files/LLVM21/bin/llvm-mc",
    "/c/Program Files/LLVM21/bin/llvm-mc",
    "llvm-mc",
    "llvm-mc-21",
];

/// Locate the `llvm-mc` binary: honor `FARM64_LLVM_MC`/`FARM64_LLVM`, else probe.
fn find_llvm_mc() -> Option<String> {
    for var in ["FARM64_LLVM_MC", "FARM64_LLVM"] {
        if let Ok(p) = std::env::var(var) {
            if !p.is_empty() && probe(&p) {
                return Some(p);
            }
        }
    }
    for cand in LLVM_MC_CANDIDATES {
        if probe(cand) {
            return Some((*cand).to_string());
        }
    }
    None
}

fn probe(exe: &str) -> bool {
    Command::new(exe)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Pick the maximal `--mattr` string the installed llvm-mc accepts. Prefers
/// `+all`; falls back to an explicit modern-extension list if `+all` is
/// rejected. Verified by round-tripping a known SVE word that requires features.
fn pick_mattr(exe: &str) -> String {
    // A word that only decodes with extensions on (SVE `ptrue p0.b`): if the
    // mattr string is accepted AND enables features, this disassembles cleanly.
    let probe_word = 0x2518_E3E0u32; // ptrue p0.b
    let candidates = [
        "+all",
        // Explicit maximal list (used only if +all is unknown to this build).
        "+v9.5a,+sve2,+sme2,+mops,+cssc,+lse128,+rcpc3,+the,+d128,+sve2p1,\
         +sme2p1,+fp8,+faminmax,+lut,+gcs,+pauth-lr,+ite,+sve-b16b16,+sme-f16f16,\
         +sme-f64f64,+sme-i16i64,+bf16,+i8mm,+crypto,+sha3,+sm4,+ls64,+flagm2,\
         +frintts,+rcpc,+rcpc-immo,+altnzcv,+predres,+specres2,+ssbs,+mte,+pauth",
    ];
    for m in candidates {
        let mattr = format!("--mattr={m}");
        let out = Command::new(exe)
            .args(["--disassemble", "--triple=aarch64", &mattr])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .and_then(|mut c| {
                let b = probe_word.to_le_bytes();
                let line = format!("0x{:02x},0x{:02x},0x{:02x},0x{:02x}\n", b[0], b[1], b[2], b[3]);
                c.stdin.take().unwrap().write_all(line.as_bytes())?;
                c.wait_with_output()
            });
        if let Ok(out) = out {
            let text = String::from_utf8_lossy(&out.stdout);
            // Accepted if it produced a real instruction line (ptrue) — i.e. the
            // mattr string parsed AND enabled the extension.
            if text.lines().any(|l| l.trim_start().starts_with("ptrue")) {
                return m.to_string();
            }
        }
    }
    // Last resort: bare triple (base ISA only) — still useful, just narrower.
    "+v8a".to_string()
}

/// Batch-disassemble `words` with llvm-mc in ONE process. Returns a vec parallel
/// to `words`: `Some(text)` for words LLVM decoded, `None` for invalid words.
///
/// Splits into chunks to keep any single stdin/stdout buffer bounded, but each
/// chunk is still hundreds of thousands of words per process.
fn llvm_disasm_batch(exe: &str, mattr: &str, words: &[u32]) -> Vec<Option<String>> {
    const CHUNK: usize = 500_000;
    let mut out = Vec::with_capacity(words.len());
    let mattr_arg = format!("--mattr={mattr}");
    for chunk in words.chunks(CHUNK) {
        out.extend(llvm_disasm_chunk(exe, &mattr_arg, chunk));
    }
    out
}

fn llvm_disasm_chunk(exe: &str, mattr_arg: &str, words: &[u32]) -> Vec<Option<String>> {
    // Build the stdin payload: one `0xNN,0xNN,0xNN,0xNN` line per word.
    let mut input = String::with_capacity(words.len() * 20);
    for &w in words {
        let b = w.to_le_bytes();
        input.push_str("0x");
        push_hex(&mut input, b[0]);
        input.push_str(",0x");
        push_hex(&mut input, b[1]);
        input.push_str(",0x");
        push_hex(&mut input, b[2]);
        input.push_str(",0x");
        push_hex(&mut input, b[3]);
        input.push('\n');
    }

    let mut child = Command::new(exe)
        .args(["--disassemble", "--triple=aarch64", mattr_arg])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn llvm-mc");

    // Write stdin on a thread so a large payload cannot deadlock against full
    // stdout/stderr pipes.
    let mut stdin = child.stdin.take().expect("llvm-mc stdin");
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(input.as_bytes());
        drop(stdin);
    });

    let output = child.wait_with_output().expect("wait llvm-mc");
    writer.join().expect("stdin writer thread");

    // (1) Invalid line numbers from stderr (1-based input line -> 0-based index).
    //
    // CRITICAL: llvm-mc emits TWO diagnostic flavours and they behave
    // differently w.r.t. stdout:
    //   * "invalid instruction encoding"            -> NO stdout line (skip).
    //   * "potentially undefined instruction encoding" -> STILL emits a stdout
    //     instruction line (it decoded it, just flags it). These must NOT be
    //     treated as invalid, or the stdout cursor drifts and every subsequent
    //     word is mis-attributed.
    // Only the former marks a word as not-decoded.
    let stderr = String::from_utf8_lossy(&output.stderr);
    let mut invalid = vec![false; words.len()];
    for line in stderr.lines() {
        // `<stdin>:LINE:COL: warning: invalid instruction encoding`
        if let Some(rest) = line.strip_prefix("<stdin>:") {
            if line.contains("invalid instruction encoding") {
                if let Some(num) = rest.split(':').next().and_then(|n| n.trim().parse::<usize>().ok())
                {
                    if num >= 1 && num <= words.len() {
                        invalid[num - 1] = true;
                    }
                }
            }
        }
    }

    // (2) Valid instruction lines from stdout, in order. Skip directive/comment
    // lines (`.text`, `.cfi_*`, blank). Each remaining line is one instruction.
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut insn_lines: Vec<&str> = Vec::with_capacity(words.len());
    for line in stdout.lines() {
        let l = line.trim_end_matches('\r');
        let t = l.trim();
        if t.is_empty() || t.starts_with('.') || t.starts_with('#') {
            continue;
        }
        insn_lines.push(l);
    }

    // (3) Reconstruct word -> Option<text>. Non-invalid words consume the next
    // stdout instruction line, in order.
    let mut result = Vec::with_capacity(words.len());
    let mut next = 0usize;
    for &is_invalid in invalid.iter() {
        if is_invalid {
            result.push(None);
        } else if next < insn_lines.len() {
            result.push(Some(insn_lines[next].trim().to_string()));
            next += 1;
        } else {
            // Defensive: stdout ran short (should not happen). Treat as invalid.
            result.push(None);
        }
    }
    result
}

fn push_hex(s: &mut String, b: u8) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    s.push(HEX[(b >> 4) as usize] as char);
    s.push(HEX[(b & 0xf) as usize] as char);
}

// ---------------------------------------------------------------------------
// Word sample generation
// ---------------------------------------------------------------------------

/// Simple xorshift64* LCG-ish PRNG — fixed seed, zero deps, deterministic.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    #[inline]
    fn next_u32(&mut self) -> u32 {
        // xorshift64*
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }
}

/// Build the word sample: a large uniform-random set PLUS structured sweeps that
/// bias toward likely-new-extension regions, then de-dup.
fn build_sample(cap: usize) -> Vec<u32> {
    let mut words: Vec<u32> = Vec::with_capacity(cap.min(8_000_000) + 1_000_000);

    // --- (A) Uniform random bulk (the majority of the sample). ---
    let mut rng = Rng::new(0x0BADC0DE_F00DFACE);
    let random_target = (cap as f64 * 0.55) as usize;
    for _ in 0..random_target {
        words.push(rng.next_u32());
    }

    // --- (B) Structured sweeps over high opcode fields of likely-new regions. ---
    // Helper: emit `count` words formed by `base | (rng-masked field bits)`.
    let sweep = |words: &mut Vec<u32>, base: u32, vary_mask: u32, count: usize, rng: &mut Rng| {
        for _ in 0..count {
            words.push(base | (rng.next_u32() & vary_mask));
        }
    };

    // SVE space: op0 = word<28:25> = 0b0010 -> bits 28..25 = 0010, i.e. 0x04..
    // Cover the whole SVE encoding quadrant densely (SVE2/SVE2p1/SME-vector).
    sweep(&mut words, 0x0400_0000, 0x03FF_FFFF, 400_000, &mut rng);
    sweep(&mut words, 0x0500_0000, 0x00FF_FFFF, 200_000, &mut rng); // SVE perm/mem subspace
    sweep(&mut words, 0x0600_0000, 0x01FF_FFFF, 200_000, &mut rng);
    sweep(&mut words, 0x0700_0000, 0x01FF_FFFF, 200_000, &mut rng);

    // SME space: top byte 0xC0..0xC7 region (SME outer-product / move / ZA) +
    // 0x8x SME2 multi-vector forms. Sweep low 25 bits.
    sweep(&mut words, 0xC000_0000, 0x00FF_FFFF, 250_000, &mut rng);
    sweep(&mut words, 0x8000_0000, 0x00FF_FFFF, 200_000, &mut rng);
    sweep(&mut words, 0xC100_0000, 0x00FF_FFFF, 150_000, &mut rng);

    // Loads/stores: op0 = x1x0 (0100/0110/1100/1110). MOPS (CPY*/SET*), LSE128
    // (LDCLRP/LDSETP/SWPP), RCPC3 (LDAPUR/STLUR SIMD), LS64 live here.
    for top in [0x1900_0000u32, 0x1D00_0000, 0x3800_0000, 0x3900_0000, 0x7800_0000,
                0x7900_0000, 0xB800_0000, 0xB900_0000, 0xF800_0000, 0xF900_0000,
                0x4800_0000, 0x0800_0000, 0xC800_0000,
                // FEAT_THE unprivileged translation-enhanced pairs (opc=11, V=0):
                // LDTP/STTP/LDTNP/STTNP live at 0xE8.. (load/store-pair, no SIMD).
                0xE800_0000, 0xE900_0000] {
        sweep(&mut words, top, 0x00FF_FFFF, 70_000, &mut rng);
    }
    // MOPS specifically: 0001_1001_... CPYP/SETP family base 0x19xxxxxx with
    // option bits; sweep the option/size fields densely.
    sweep(&mut words, 0x1900_0400, 0x00FF_FC1F, 120_000, &mut rng);
    sweep(&mut words, 0x1980_0400, 0x007F_FC1F, 120_000, &mut rng);

    // Data-processing register: op0 = x101 (0101/1101). CSSC (ABS/CNT/SMAX/UMAX
    // /SMIN/UMIN/CTZ), RNG, etc. live in DP-1src/2src high fields.
    sweep(&mut words, 0x1A00_0000, 0x00FF_FFFF, 150_000, &mut rng); // 32-bit DP-2src/1src
    sweep(&mut words, 0x9A00_0000, 0x00FF_FFFF, 150_000, &mut rng); // 64-bit
    sweep(&mut words, 0x5A00_0000, 0x00FF_FFFF, 120_000, &mut rng);
    sweep(&mut words, 0xDA00_0000, 0x00FF_FFFF, 120_000, &mut rng);
    // CSSC immediate (SMAX/UMAX/SMIN/UMIN imm) base ~0x11C0.. region in DP-imm.
    sweep(&mut words, 0x1100_0000, 0x00FF_FFFF, 100_000, &mut rng);
    sweep(&mut words, 0x9100_0000, 0x00FF_FFFF, 100_000, &mut rng);

    // Data-processing immediate: op0 = 100x. PAuth-LR / misc.
    sweep(&mut words, 0x9000_0000, 0x00FF_FFFF, 80_000, &mut rng);

    // System / branch space: op0 = 101x. GCS (GCSPUSHM/GCSPOPM/GCSSS*), barriers,
    // hints (CHKFEAT/CLRBHB), THE (RCWxx is in ldst), system register moves.
    sweep(&mut words, 0xD500_0000, 0x00FF_FFFF, 200_000, &mut rng); // system
    sweep(&mut words, 0xD503_0000, 0x0000_FFFF, 120_000, &mut rng); // hints/barriers
    sweep(&mut words, 0xD508_0000, 0x0007_FFFF, 80_000, &mut rng); // sys/sysl
    sweep(&mut words, 0xD400_0000, 0x00FF_FFFF, 60_000, &mut rng); // exceptions

    // SIMD/FP scalar+vector: op0 = x111. FP8 (FMLALB/FCVTN FP8), LUT (LUTI),
    // FAMINMAX (FAMAX/FAMIN), modern crypto.
    sweep(&mut words, 0x0E00_0000, 0x01FF_FFFF, 200_000, &mut rng);
    sweep(&mut words, 0x4E00_0000, 0x01FF_FFFF, 200_000, &mut rng);
    sweep(&mut words, 0x2E00_0000, 0x01FF_FFFF, 150_000, &mut rng);
    sweep(&mut words, 0x6E00_0000, 0x01FF_FFFF, 150_000, &mut rng);
    sweep(&mut words, 0x5E00_0000, 0x01FF_FFFF, 100_000, &mut rng);
    sweep(&mut words, 0x7E00_0000, 0x01FF_FFFF, 100_000, &mut rng);

    // --- De-dup (sort + dedup; stable order not required). ---
    words.sort_unstable();
    words.dedup();

    // Respect the cap (keep a deterministic strided subset if we overshot).
    if words.len() > cap {
        let stride = words.len() / cap + 1;
        let mut trimmed = Vec::with_capacity(cap);
        let mut i = 0;
        while i < words.len() && trimmed.len() < cap {
            trimmed.push(words[i]);
            i += stride;
        }
        // top up sequentially if striding left us short
        let mut j = 0;
        while trimmed.len() < cap && j < words.len() {
            trimmed.push(words[j]);
            j += 1;
        }
        trimmed.sort_unstable();
        trimmed.dedup();
        return trimmed;
    }
    words
}

// ---------------------------------------------------------------------------
// Reporting helpers
// ---------------------------------------------------------------------------

/// First whitespace-separated token of a disassembly line, lowercased.
fn first_token(s: &str) -> String {
    s.split_whitespace().next().unwrap_or("").to_ascii_lowercase()
}

/// A per-mnemonic bucket: count + one representative example.
#[derive(Default)]
struct Bucket {
    count: usize,
    example: Option<(u32, String, String)>, // (word, a, b)
}
impl Bucket {
    fn record(&mut self, word: u32, a: &str, b: &str) {
        self.count += 1;
        if self.example.is_none() {
            self.example = Some((word, a.to_string(), b.to_string()));
        }
    }
}

/// Sort buckets by count desc and return top `n` as (mnemonic, bucket).
fn top(map: &HashMap<String, Bucket>, n: usize) -> Vec<(&String, &Bucket)> {
    let mut v: Vec<_> = map.iter().collect();
    v.sort_by(|a, b| b.1.count.cmp(&a.1.count).then(a.0.cmp(b.0)));
    v.truncate(n);
    v
}

// ---------------------------------------------------------------------------
// The sweep
// ---------------------------------------------------------------------------

#[test]
#[ignore = "needs LLVM 21 llvm-mc; report-only discovery sweep. Run with `--ignored --nocapture`."]
fn llvm_diff() {
    let Some(exe) = find_llvm_mc() else {
        eprintln!("[llvm_diff] llvm-mc not found (set FARM64_LLVM_MC); skipping.");
        return;
    };
    let mattr = pick_mattr(&exe);
    eprintln!("[llvm_diff] oracle: {exe}");
    eprintln!("[llvm_diff] invocation: llvm-mc --disassemble --triple=aarch64 --mattr={mattr}");

    let cap = common::env_limit("FARM64_DIFF_WORDS").unwrap_or(6_000_000);
    eprintln!("[llvm_diff] building word sample (cap {cap}) ...");
    let words = build_sample(cap);
    eprintln!("[llvm_diff] de-duped sample size: {}", words.len());

    eprintln!("[llvm_diff] disassembling with LLVM (batched) ...");
    let llvm = llvm_disasm_batch(&exe, &mattr, &words);
    assert_eq!(llvm.len(), words.len(), "llvm result count must match sample");

    eprintln!("[llvm_diff] decoding with fARM64 (FeatureSet::ALL) ...");

    let mut llvm_valid = 0usize;
    let mut farm_valid = 0usize;
    let mut gap_count = 0usize; // LLVM valid, fARM64 invalid
    let mut disagree_count = 0usize; // both valid, first token differs
    let mut reverse_count = 0usize; // fARM64 valid, LLVM invalid

    let mut gaps: HashMap<String, Bucket> = HashMap::new();
    let mut disagrees: HashMap<String, Bucket> = HashMap::new();
    let mut reverses: HashMap<String, Bucket> = HashMap::new();

    for (i, &word) in words.iter().enumerate() {
        let insn = decode(word, ADDRESS, FeatureSet::ALL);
        let farm_invalid = insn.is_invalid();
        let farm_mnem = insn.mnemonic().name();
        if !farm_invalid {
            farm_valid += 1;
        }

        match &llvm[i] {
            Some(text) => {
                llvm_valid += 1;
                let lmnem = first_token(text);
                if farm_invalid {
                    // GAP: LLVM decodes a real instruction, fARM64 returns Invalid.
                    gap_count += 1;
                    gaps.entry(lmnem).or_default().record(word, text, "");
                } else {
                    // Both valid: compare first token leniently.
                    let fmnem = farm_mnem.to_ascii_lowercase();
                    if fmnem != lmnem {
                        disagree_count += 1;
                        disagrees
                            .entry(farm_mnem.to_string())
                            .or_default()
                            .record(word, text, farm_mnem);
                    }
                }
            }
            None => {
                if !farm_invalid {
                    // REVERSE: fARM64 decodes, LLVM calls it invalid.
                    reverse_count += 1;
                    reverses
                        .entry(farm_mnem.to_string())
                        .or_default()
                        .record(word, "<llvm-invalid>", farm_mnem);
                }
            }
        }
    }

    // ---- Write full report to target/llvm-diff.txt ----
    let out_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("llvm-diff.txt");
    if let Some(dir) = out_path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Ok(mut f) = std::fs::File::create(&out_path) {
        let _ = writeln!(f, "# fARM64 vs LLVM 21 differential discovery sweep");
        let _ = writeln!(f, "# oracle: {exe} --disassemble --triple=aarch64 --mattr={mattr}");
        let _ = writeln!(f, "# sample size (de-duped): {}", words.len());
        let _ = writeln!(f, "# llvm-valid: {llvm_valid}  farm64-valid: {farm_valid}");
        let _ = writeln!(f, "# gaps: {gap_count}  disagreements: {disagree_count}  reverse: {reverse_count}");
        let _ = writeln!(f);

        let _ = writeln!(f, "## (a) GAPS — LLVM decodes, fARM64 returns Invalid (by LLVM mnemonic)");
        let _ = writeln!(f, "# count\tmnemonic\texample_word\tllvm_text");
        for (m, b) in top(&gaps, 200) {
            let ex = b.example.as_ref().map(|(w, t, _)| format!("{w:08X}\t{t}")).unwrap_or_default();
            let _ = writeln!(f, "{}\t{m}\t{ex}", b.count);
        }
        let _ = writeln!(f);

        let _ = writeln!(f, "## (b) DISAGREEMENTS — both decode, first token differs (by fARM64 mnemonic)");
        let _ = writeln!(f, "# count\tfarm64_mnem\texample_word\tllvm_text\tfarm64_mnem");
        for (m, b) in top(&disagrees, 200) {
            let ex = b
                .example
                .as_ref()
                .map(|(w, l, fa)| format!("{w:08X}\tllvm=[{l}]\tfarm64_mnem={fa}"))
                .unwrap_or_default();
            let _ = writeln!(f, "{}\t{m}\t{ex}", b.count);
        }
        let _ = writeln!(f);

        let _ = writeln!(f, "## (c) REVERSE — fARM64 decodes, LLVM Invalid (by fARM64 mnemonic)");
        let _ = writeln!(f, "# count\tfarm64_mnem\texample_word");
        for (m, b) in top(&reverses, 200) {
            let ex = b.example.as_ref().map(|(w, _, _)| format!("{w:08X}")).unwrap_or_default();
            let _ = writeln!(f, "{}\t{m}\t{ex}", b.count);
        }
    }

    // ---- Concise summary to stderr ----
    eprintln!();
    eprintln!("================ fARM64 vs LLVM 21 ================");
    eprintln!("sample size (de-duped) : {}", words.len());
    eprintln!("llvm-valid             : {llvm_valid}");
    eprintln!("farm64-valid           : {farm_valid}");
    eprintln!("GAPS (llvm>farm)       : {gap_count}");
    eprintln!("DISAGREEMENTS          : {disagree_count}");
    eprintln!("REVERSE (farm>llvm)    : {reverse_count}");
    eprintln!("full report            : {}", out_path.display());
    eprintln!("---------------------------------------------------");
    eprintln!("TOP GAP MNEMONICS (LLVM decodes, fARM64 missing):");
    for (m, b) in top(&gaps, 50) {
        let ex = b
            .example
            .as_ref()
            .map(|(w, t, _)| format!("{w:08X}  {t}"))
            .unwrap_or_default();
        eprintln!("  {:>8}  {:<16} e.g. {ex}", b.count, m);
    }
    eprintln!("---------------------------------------------------");
    eprintln!("TOP DISAGREEMENTS (by fARM64 mnemonic; many are alias/radix noise):");
    for (m, b) in top(&disagrees, 30) {
        let ex = b
            .example
            .as_ref()
            .map(|(w, l, _)| format!("{w:08X}  llvm=[{l}]"))
            .unwrap_or_default();
        eprintln!("  {:>8}  {:<16} {ex}", b.count, m);
    }
    eprintln!("---------------------------------------------------");
    eprintln!("TOP REVERSE (fARM64 over-decodes vs LLVM):");
    for (m, b) in top(&reverses, 20) {
        let ex = b.example.as_ref().map(|(w, _, _)| format!("{w:08X}")).unwrap_or_default();
        eprintln!("  {:>8}  {:<16} e.g. {ex}", b.count, m);
    }
    eprintln!("===================================================");

    // Report-only: never hard-fail on a decode difference.
}

// ---------------------------------------------------------------------------
// Non-ignored sanity checks (do not require llvm-mc).
// ---------------------------------------------------------------------------

#[test]
fn sample_is_large_and_deduped() {
    let words = build_sample(200_000);
    assert!(words.len() > 50_000, "sample should be substantial: {}", words.len());
    // De-dup invariant: sorted + unique.
    let mut sorted = words.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), words.len(), "sample must be de-duped");
}

#[test]
fn rng_is_deterministic() {
    let mut a = Rng::new(0x0BADC0DE_F00DFACE);
    let mut b = Rng::new(0x0BADC0DE_F00DFACE);
    for _ in 0..1000 {
        assert_eq!(a.next_u32(), b.next_u32());
    }
}

#[test]
fn input_encoding_le() {
    // `add w0, w1, #1` is 0x11000420; little-endian bytes are 20 04 00 11.
    let mut s = String::new();
    for b in 0x1100_0420u32.to_le_bytes() {
        push_hex(&mut s, b);
        s.push(',');
    }
    assert_eq!(s, "20,04,00,11,");
}
