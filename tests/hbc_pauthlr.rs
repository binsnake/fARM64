//! FEAT_HBC `BC.<cond>` and FEAT_PAuth_LR `*SPPC` branch forms: decode + render
//! + round-trip coverage.
//!
//! Two extensions are exercised:
//!
//! * **FEAT_HBC** — the consistent/hinted conditional branch `BC.<cond>`. It is
//!   the `bit4 == 1` sibling of the ordinary `B.<cond>` (`0101010 0 imm19 o0
//!   cond`): `o0 == 0` is `B.<cond>`, `o0 == 1` is `BC.<cond>`. The condition
//!   suffix list is identical to `B.cond`; the two fuse the condition into the
//!   mnemonic (`b.ne` / `bc.ne`).
//! * **FEAT_PAuth_LR** — the PC-relative authenticate/return branches
//!   `RETAASPPC`/`RETABSPPC` (`0101010 1 00 M imm16 11111`, in the branch group)
//!   and `AUTIASPPC`/`AUTIBSPPC` (`1111001110 M imm16 11111`, in the
//!   data-processing-immediate group). The `M` bit selects the signing key; the
//!   16-bit `imm16` is the *negated* offset, so the PC-relative target is
//!   `ip - (imm16:00)` (a backward branch toward the signing instruction).
//!
//! Example words are LLVM (`clang` / `llvm-objdump --mattr=+all`) oracle
//! encodings. The tests confirm the expected mnemonic/text, prove a bit-exact
//! encoder round-trip, pin the `B.cond`-vs-`BC.cond` bit4 boundary, and check the
//! per-extension feature gating.

#![cfg(feature = "std")]

use fARM64::decode::decode;
use fARM64::format::{format_to_string, FmtFormatter};
use fARM64::{encode, Feature, FeatureSet};

/// Render `word` (decoded at `ip`) to its disassembly text.
fn text(word: u32, ip: u64) -> String {
    let insn = decode(word, ip, FeatureSet::ALL);
    format_to_string(&FmtFormatter::new(), &insn)
}

/// Collapse runs of whitespace so the fixed mnemonic/operand padding does not
/// affect the comparison.
fn norm(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Decode `word` at `ip`, assert the disassembly matches `expected`, then prove
/// a bit-for-bit encoder round-trip.
fn check(word: u32, ip: u64, expected: &str) {
    let insn = decode(word, ip, FeatureSet::ALL);
    assert!(!insn.is_invalid(), "{:08X} decoded Invalid (want `{}`)", word, expected);
    assert_eq!(norm(&text(word, ip)), norm(expected), "{:08X} disasm mismatch", word);
    let enc = encode(&insn)
        .unwrap_or_else(|e| panic!("{:08X} ({}) encode error {:?}", word, insn.mnemonic().name(), e));
    assert_eq!(enc, word, "{:08X} round-trip produced {:08X}", word, enc);
}

/// `FeatureSet::ALL` minus one feature bit (in both words).
fn without(fs: FeatureSet, f: Feature) -> FeatureSet {
    let bit = f as u32;
    FeatureSet {
        features0: fs.features0 & !(1u64 << bit),
        features1: fs.features1 & !(1u64 << bit),
    }
}

// ---------------------------------------------------------------------------
// FEAT_HBC BC.<cond>.
// ---------------------------------------------------------------------------

#[test]
fn bc_cond_examples() {
    // Canonical LLVM (+all) oracle encodings: `bc.ne 0x6ef34`.
    check(0x543779B1, 0, "bc.ne 0x6ef34");
    // bit4 == 1 over the full condition-suffix list (cond == word<3:0>), offset 0.
    let conds = [
        "eq", "ne", "hs", "lo", "mi", "pl", "vs", "vc", "hi", "ls", "ge", "lt", "gt", "le", "al",
        "nv",
    ];
    for (cond, name) in conds.iter().enumerate() {
        let w = 0x5400_0010u32 | (cond as u32); // 0101010 0 imm19=0 o0=1 cond
        check(w, 0, &format!("bc.{} 0x0", name));
    }
}

#[test]
fn b_cond_unchanged_examples() {
    // bit4 == 0 stays the ordinary B.<cond> with the same suffix list.
    check(0x54000000, 0, "b.eq 0x0");
    let conds = [
        "eq", "ne", "hs", "lo", "mi", "pl", "vs", "vc", "hi", "ls", "ge", "lt", "gt", "le", "al",
        "nv",
    ];
    for (cond, name) in conds.iter().enumerate() {
        let w = 0x5400_0000u32 | (cond as u32); // o0 == 0
        check(w, 0, &format!("b.{} 0x0", name));
    }
}

/// The `B.cond`-vs-`BC.cond` boundary is exactly `word<4>` (`o0`): bit4 == 0
/// renders `b.<cond>`, bit4 == 1 renders `bc.<cond>`, for every condition.
#[test]
fn bcond_vs_bccond_bit4_boundary() {
    for cond in 0..16u32 {
        let base = 0x5400_0000u32 | cond;
        let b = decode(base, 0, FeatureSet::ALL); // o0 == 0
        let bc = decode(base | (1 << 4), 0, FeatureSet::ALL); // o0 == 1
        assert!(!b.is_invalid() && !bc.is_invalid(), "cond {:#x} should both decode", cond);
        let bt = format_to_string(&FmtFormatter::new(), &b);
        let bct = format_to_string(&FmtFormatter::new(), &bc);
        assert!(bt.starts_with("b."), "bit4==0 should be b.<cond>, got `{}`", bt);
        assert!(bct.starts_with("bc."), "bit4==1 should be bc.<cond>, got `{}`", bct);
        // Same condition suffix, different mnemonic stem.
        let bsuf = bt.split_whitespace().next().unwrap().trim_start_matches("b.");
        let bcsuf = bct.split_whitespace().next().unwrap().trim_start_matches("bc.");
        assert_eq!(bsuf, bcsuf, "cond {:#x}: condition suffix must match", cond);
    }
}

#[test]
fn bc_cond_roundtrip_with_offsets() {
    // Exercise the imm19 field both directions (forward and backward) at a
    // non-zero ip, for a couple of conditions, with bit4 == 1.
    let ip = 0x1_0000u64;
    for &cond in &[0u32, 1, 6, 7, 13, 15] {
        for &off in &[0i64, 4, -4, 0xff_ffc, -0x10_0000] {
            let imm19 = ((off >> 2) as u32) & 0x7_FFFF;
            let w = 0x5400_0010u32 | (imm19 << 5) | cond;
            let insn = decode(w, ip, FeatureSet::ALL);
            assert_eq!(insn.code(), fARM64::Code::BcCond, "{:08X} should be BcCond", w);
            let enc = encode(&insn).expect("BcCond encode");
            assert_eq!(enc, w, "{:08X} round-trip", w);
        }
    }
}

// ---------------------------------------------------------------------------
// FEAT_PAuth_LR RETAASPPC / RETABSPPC / AUTIASPPC / AUTIBSPPC.
// ---------------------------------------------------------------------------

#[test]
fn sppc_examples() {
    // Canonical LLVM (+all) oracle encodings (ip == 0; target == ip - imm16:00).
    check(0x552F577F, 0, "retabsppc 0xfffffffffffe1514");
    check(0x551E8D9F, 0, "retaasppc 0xfffffffffffc2e50");
    check(0xF3983E9F, 0, "autiasppc 0xfffffffffffcf830");
    check(0xF3B9D25F, 0, "autibsppc 0xfffffffffffcc5b8");
    // Offset-0 forms: target == ip (a backward branch of 0).
    check(0x5500001F, 0x1000, "retaasppc 0x1000"); // key A
    check(0x5520001F, 0x1000, "retabsppc 0x1000"); // key B
    check(0xF380001F, 0x1000, "autiasppc 0x1000"); // key A
    check(0xF3A0001F, 0x1000, "autibsppc 0x1000"); // key B
}

/// The backward PC-relative immediate: `target = ip - (imm16 << 2)`. Sweep
/// imm16 across both keys and both classes and require a bit-exact round-trip.
#[test]
fn sppc_imm16_roundtrip_sweep() {
    let ip = 0x10_0000u64;
    // (base word with imm16==0 and key==A, key-bit position is word<21>).
    let bases = [0x5500_001Fu32, 0xF380_001Fu32];
    for base in bases {
        for key in 0..2u32 {
            for &imm16 in &[0u32, 1, 2, 0x40, 0x1234, 0x7FFF, 0xFFFF] {
                let w = base | (key << 21) | (imm16 << 5);
                let insn = decode(w, ip, FeatureSet::ALL);
                assert!(!insn.is_invalid(), "{:08X} should decode", w);
                // Target must be the backward offset.
                let enc = encode(&insn)
                    .unwrap_or_else(|e| panic!("{:08X} ({}) encode error {:?}", w, insn.mnemonic().name(), e));
                assert_eq!(enc, w, "{:08X} round-trip produced {:08X}", w, enc);
            }
        }
    }
}

/// The fixed structural bits of the SPPC forms are enforced: `RET*SPPC` requires
/// `word<23:22> == 00` and `word<4:0> == 11111`; `AUTI*SPPC` requires
/// `word<4:0> == 11111` (and the exact `word<31:22>` mask). Neighbours are
/// UNDEFINED rather than over-decoded.
#[test]
fn sppc_reserved_neighbours_invalid() {
    // RET*SPPC: word<23:22> must be 00.
    assert!(decode(0x5500_001F | (1 << 22), 0, FeatureSet::ALL).is_invalid(), "RET word<22>==1");
    assert!(decode(0x5500_001F | (1 << 23), 0, FeatureSet::ALL).is_invalid(), "RET word<23>==1");
    // RET*SPPC: word<4:0> must be 11111.
    assert!(decode(0x5500_001E, 0, FeatureSet::ALL).is_invalid(), "RET Rd!=31 (0x...1e)");
    assert!(decode(0x5500_000F, 0, FeatureSet::ALL).is_invalid(), "RET Rd!=31 (0x...0f)");
    // AUTI*SPPC: word<4:0> must be 11111; word<22>==1 leaves the mask.
    assert!(decode(0xF380_001E, 0, FeatureSet::ALL).is_invalid(), "AUTI Rd!=31");
    assert!(decode(0xF3C0_001F, 0, FeatureSet::ALL).is_invalid(), "AUTI word<22>==1");
}

// ---------------------------------------------------------------------------
// Feature gating.
// ---------------------------------------------------------------------------

#[test]
fn hbc_feature_gated() {
    let no_hbc = without(FeatureSet::ALL, Feature::Hbc);
    // BC.cond (bit4==1) requires FEAT_HBC.
    let bc = 0x5400_0010u32;
    assert!(decode(bc, 0, no_hbc).is_invalid(), "BC.cond should require FEAT_HBC");
    assert!(!decode(bc, 0, FeatureSet::ALL).is_invalid(), "BC.cond should decode with FEAT_HBC");
    // B.cond (bit4==0) is base ISA — never gated.
    let b = 0x5400_0000u32;
    assert!(!decode(b, 0, no_hbc).is_invalid(), "B.cond must stay base ISA (no FEAT_HBC)");
}

#[test]
fn pauth_lr_feature_gated() {
    let no_plr = without(FeatureSet::ALL, Feature::PauthLr);
    for &w in &[0x5500_001Fu32, 0x5520_001F, 0xF380_001F, 0xF3A0_001F] {
        assert!(decode(w, 0, no_plr).is_invalid(), "{:08X} should require FEAT_PAuth_LR", w);
        assert!(!decode(w, 0, FeatureSet::ALL).is_invalid(), "{:08X} should decode with FEAT_PAuth_LR", w);
    }
    // Disabling FEAT_PAuth_LR must not perturb the neighbouring B.cond / EXTR
    // encodings that share these dispatch paths.
    assert!(!decode(0x5400_0000, 0, no_plr).is_invalid(), "b.eq must stay valid");
    // EXTR x0, x1, x2, #0 (1001_0011_110... ) shares the dp-imm Extract slot.
    assert!(!decode(0x93C20020, 0, no_plr).is_invalid(), "extr must stay valid");
}
