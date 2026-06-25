//! FEAT_THE / FEAT_LSE128 atomic coverage: exhaustive decode + round-trip.
//!
//! These instructions live in two encoding majors that share their bit layout
//! with unrelated families (memory tagging, the LSE atomics), so the tests sweep
//! the whole sub-space and check that:
//!   * every encoding fARM64 decodes also re-encodes and re-decodes identically
//!     (semantic round-trip), and
//!   * the canonical example words from each family decode to the expected
//!     mnemonic with the expected operand count.
//!
//! The families covered are LDTADD/LDTCLR/LDTSET and SWPT (FEAT_THE
//! unprivileged), RCWCAS/RCWCASP and the single/pair RCW RMW ops plus their
//! RCWS* check variants (FEAT_THE read-check-write), and LDCLRP/LDSETP/SWPP
//! (FEAT_LSE128 128-bit load-op-pair).

#![cfg(feature = "std")]

use fARM64::decode::decode;
use fARM64::{encode, FeatureSet};

/// Decode `word`, re-encode, re-decode; require identical mnemonic + operands.
fn assert_roundtrip(word: u32) {
    let insn = decode(word, 0, FeatureSet::ALL);
    assert!(!insn.is_invalid(), "{:08X} decoded Invalid", word);
    let enc = encode(&insn).unwrap_or_else(|e| panic!("{:08X} ({}) encode error {:?}", word, insn.mnemonic().name(), e));
    let insn2 = decode(enc, 0, FeatureSet::ALL);
    assert_eq!(
        insn.mnemonic(),
        insn2.mnemonic(),
        "{:08X} ({}) re-decoded as {} (re-encoded {:08X})",
        word,
        insn.mnemonic().name(),
        insn2.mnemonic().name(),
        enc
    );
    assert_eq!(insn.op_count(), insn2.op_count(), "{:08X} operand-count drift", word);
    for i in 0..insn.op_count() {
        assert_eq!(
            format!("{:?}", insn.op(i)),
            format!("{:?}", insn2.op(i)),
            "{:08X} ({}) operand {} differs",
            word,
            insn.mnemonic().name(),
            i
        );
    }
}

/// Word in the `011001` major: `size 011 0 01 A R 1 Rs o3 opc op2 Rn Rt`.
#[allow(clippy::too_many_arguments)]
fn mk_tag_major(sz: u32, a: u32, r: u32, rs: u32, o3: u32, opc: u32, op2: u32, rn: u32, rt: u32) -> u32 {
    (sz << 30)
        | (0b011 << 27)
        | (0b01 << 24)
        | (a << 23)
        | (r << 22)
        | (1 << 21)
        | (rs << 16)
        | (o3 << 15)
        | (opc << 12)
        | (op2 << 10)
        | (rn << 5)
        | rt
}

/// Word in the LSE-atomic major: `size 111 0 00 A R 1 Rs o3 opc 00 Rn Rt`.
#[allow(clippy::too_many_arguments)]
fn mk_lse_major(sz: u32, a: u32, r: u32, rs: u32, o3: u32, opc: u32, rn: u32, rt: u32) -> u32 {
    (sz << 30)
        | (0b111 << 27)
        | (a << 23)
        | (r << 22)
        | (1 << 21)
        | (rs << 16)
        | (o3 << 15)
        | (opc << 12)
        | (rn << 5)
        | rt
}

#[test]
fn examples_decode_as_expected() {
    // (word, mnemonic, operand_count) — canonical encodings (LLVM 21 oracle).
    let cases: &[(u32, &str, usize)] = &[
        (0x19200401, "ldtadd", 3),
        (0x19600401, "ldtaddl", 3),
        (0x19A00400, "ldtadda", 3),
        (0x19E00400, "ldtaddal", 3),
        (0x19201400, "ldtclr", 3),
        (0x19203404, "ldtset", 3),
        (0x19208405, "swpt", 3),
        (0x19E08400, "swptal", 3),
        (0x19200829, "rcwcas", 3),
        (0x19E008F3, "rcwcasal", 3),
        (0x19200C06, "rcwcasp", 5),
        (0x19E00C02, "rcwcaspal", 5),
        (0x1920908F, "rcwclrp", 3),
        (0x1920A144, "rcwswpp", 3),
        (0x1920B044, "rcwsetp", 3),
        (0x192011E9, "ldclrp", 3),
        (0x19203052, "ldsetp", 3),
        (0x192080A0, "swpp", 3),
        // Single-register RCW RMW (LSE-atomic major).
        (0x38209284, "rcwclr", 3),
        (0x3820B032, "rcwset", 3),
        (0x7820A1B8, "rcwsswp", 3),
        (0x78E0B22D, "rcwssetal", 3),
    ];
    for &(w, m, n) in cases {
        let insn = decode(w, 0, FeatureSet::ALL);
        assert!(!insn.is_invalid(), "{:08X} decoded Invalid (expected {})", w, m);
        assert_eq!(insn.mnemonic().name(), m, "{:08X} mnemonic", w);
        assert_eq!(insn.op_count(), n, "{:08X} operand count", w);
        assert_roundtrip(w);
    }
}

#[test]
fn exhaustive_roundtrip() {
    let regs = [
        (0u32, 0u32, 0u32),
        (4, 5, 6),
        (5, 7, 9),
        (2, 4, 6),
        (31, 5, 7),
        (5, 7, 31),
        (30, 5, 4),
    ];
    let mut decoded = 0usize;

    // 011001 major: LDT*/SWPT, RCWCAS/CASP, pair load-op ops.
    for sz in 0..2 {
        for a in 0..2 {
            for r in 0..2 {
                for o3 in 0..2 {
                    for opc in 0..8 {
                        for op2 in 0..4 {
                            for &(rs, rn, rt) in &regs {
                                let w = mk_tag_major(sz, a, r, rs, o3, opc, op2, rn, rt);
                                if decode(w, 0, FeatureSet::ALL).is_invalid() {
                                    continue;
                                }
                                decoded += 1;
                                assert_roundtrip(w);
                            }
                        }
                    }
                }
            }
        }
    }

    // LSE-atomic major: only the single-register RCW RMW ops are new here.
    for sz in 0..4 {
        for a in 0..2 {
            for r in 0..2 {
                for o3 in 0..2 {
                    for opc in 0..8 {
                        for &(rs, rn, rt) in &regs {
                            let w = mk_lse_major(sz, a, r, rs, o3, opc, rn, rt);
                            let insn = decode(w, 0, FeatureSet::ALL);
                            if insn.is_invalid() || !insn.mnemonic().name().starts_with("rcw") {
                                continue;
                            }
                            decoded += 1;
                            assert_roundtrip(w);
                        }
                    }
                }
            }
        }
    }

    assert!(decoded > 400, "expected >400 THE/LSE128 encodings, got {}", decoded);
}

#[test]
fn pair_ops_reject_reg31() {
    // The load-op-pair forms reserve register 31 in both Rs and Rt.
    // rcwclrp with Rt=31 (Rs=5) and with Rs=31 (Rt=5) must be Invalid.
    let rt31 = mk_tag_major(0, 0, 0, 5, 1, 0b001, 0b00, 7, 31);
    let rs31 = mk_tag_major(0, 0, 0, 31, 1, 0b001, 0b00, 7, 5);
    assert!(decode(rt31, 0, FeatureSet::ALL).is_invalid(), "rcwclrp Rt=31 should be Invalid");
    assert!(decode(rs31, 0, FeatureSet::ALL).is_invalid(), "rcwclrp Rs=31 should be Invalid");
    // But the single-register forms permit register 31 (xzr).
    let single = mk_lse_major(0, 0, 0, 31, 1, 0b001, 7, 5); // rcwclr xzr, x5, [x7]
    assert!(!decode(single, 0, FeatureSet::ALL).is_invalid(), "rcwclr Rs=31 should decode");
}

#[test]
fn casp_requires_even_registers() {
    // RCWCASP: odd Rs or odd Rt is UNDEFINED.
    let odd_rs = mk_tag_major(0, 0, 0, 5, 0, 0b000, 0b11, 7, 6);
    let odd_rt = mk_tag_major(0, 0, 0, 4, 0, 0b000, 0b11, 7, 7);
    let even = mk_tag_major(0, 0, 0, 4, 0, 0b000, 0b11, 7, 6);
    assert!(decode(odd_rs, 0, FeatureSet::ALL).is_invalid(), "rcwcasp odd Rs should be Invalid");
    assert!(decode(odd_rt, 0, FeatureSet::ALL).is_invalid(), "rcwcasp odd Rt should be Invalid");
    assert!(!decode(even, 0, FeatureSet::ALL).is_invalid(), "rcwcasp even pair should decode");
}

#[test]
fn the_gated_by_feature() {
    // With FEAT_THE absent, the THE encodings must not decode.
    let no_the = FeatureSet::ALL.without_the();
    let ldtadd = 0x19200401u32;
    assert!(decode(ldtadd, 0, no_the).is_invalid(), "ldtadd should require FEAT_THE");
    assert!(!decode(ldtadd, 0, FeatureSet::ALL).is_invalid());
}

/// Helper trait so the feature-gating test can clear just `The`.
trait WithoutThe {
    fn without_the(self) -> Self;
}
impl WithoutThe for FeatureSet {
    fn without_the(self) -> Self {
        // Rebuild ALL minus the Feature::The bit in both words.
        let bit = fARM64::Feature::The as u32;
        FeatureSet {
            features0: self.features0 & !(1u64 << bit),
            features1: self.features1 & !(1u64 << bit),
        }
    }
}
