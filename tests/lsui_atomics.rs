//! FEAT_LSUI unprivileged atomics: decode + round-trip coverage.
//!
//! Covers the unprivileged load/store-exclusive register forms
//! (`LDTXR`/`LDATXR`/`STTXR`/`STLTXR`, W and X) and the unprivileged
//! compare-and-swap forms (`CAST`/`CASAT`/`CASLT`/`CASALT`, 64-bit only, and the
//! pair `CASPT`/`CASPAT`/`CASPLT`/`CASPALT`, 64-bit only). All share the
//! exclusive/CAS bit layout `sz 001001 o2 L o1 Rs o0 Rt2 Rn Rt` — the standard
//! exclusive class with the group's low bit (`word<24>`) set.
//!
//! The canonical example words are the LLVM 21 (`llvm-mc --mattr=+all`) oracle
//! encodings. The tests confirm fARM64 decodes them to the expected
//! mnemonic/operand-count, re-encodes to the identical word, sweeps the whole
//! sub-space for semantic round-trip stability, and that the reserved/undefined
//! slots stay `Invalid`.

#![cfg(feature = "std")]

use fARM64::decode::decode;
use fARM64::{encode, Feature, FeatureSet};

/// Build a word in the LSUI major: `sz 001001 o2 L o1 Rs o0 Rt2 Rn Rt`.
#[allow(clippy::too_many_arguments)]
fn mk(sz: u32, o2: u32, l: u32, o1: u32, rs: u32, o0: u32, rt2: u32, rn: u32, rt: u32) -> u32 {
    (sz << 30)
        | (0b001001 << 24)
        | (o2 << 23)
        | (l << 22)
        | (o1 << 21)
        | (rs << 16)
        | (o0 << 15)
        | (rt2 << 10)
        | (rn << 5)
        | rt
}

/// Decode `word`, re-encode, re-decode; require identical mnemonic + operands.
fn assert_roundtrip(word: u32) {
    let insn = decode(word, 0, FeatureSet::ALL);
    assert!(!insn.is_invalid(), "{:08X} decoded Invalid", word);
    let enc = encode(&insn)
        .unwrap_or_else(|e| panic!("{:08X} ({}) encode error {:?}", word, insn.mnemonic().name(), e));
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

#[test]
fn examples_decode_as_expected() {
    // (word, mnemonic, operand_count) — canonical encodings (LLVM 21 oracle).
    let cases: &[(u32, &str, usize)] = &[
        // Unprivileged load/store-exclusive register (W and X).
        (0x895F7C20, "ldtxr", 2),   // ldtxr  w0,  [x1]
        (0xC95F7C20, "ldtxr", 2),   // ldtxr  x0,  [x1]
        (0x895FFEB9, "ldatxr", 2),  // ldatxr w25, [x21]
        (0xC95FFEB9, "ldatxr", 2),  // ldatxr x25, [x21]
        (0x89007FE9, "sttxr", 3),   // sttxr  w0, w9, [sp]
        (0xC9007FE9, "sttxr", 3),   // sttxr  w0, x9, [sp]
        (0x8900FFE9, "stltxr", 3),  // stltxr w0, w9, [sp]
        (0xC900FFE9, "stltxr", 3),  // stltxr w0, x9, [sp]
        // Unprivileged compare-and-swap (64-bit only).
        (0xC9807FC9, "cast", 3),    // cast   x0, x9, [x30]
        (0xC9C07FC9, "casat", 3),   // casat  x0, x9, [x30]
        (0xC980FFC9, "caslt", 3),   // caslt  x0, x9, [x30]
        (0xC9C0FFC9, "casalt", 3),  // casalt x0, x9, [x30]
        // Unprivileged compare-and-swap pair (64-bit only).
        (0x49807C82, "caspt", 5),   // caspt   x0, x1, x2, x3, [x4]
        (0x49C07C82, "caspat", 5),  // caspat  x0, x1, x2, x3, [x4]
        (0x4980FC82, "casplt", 5),  // casplt  x0, x1, x2, x3, [x4]
        (0x49C0FC82, "caspalt", 5), // caspalt x0, x1, x2, x3, [x4]
    ];
    for &(w, m, n) in cases {
        let insn = decode(w, 0, FeatureSet::ALL);
        assert!(!insn.is_invalid(), "{:08X} decoded Invalid (expected {})", w, m);
        assert_eq!(insn.mnemonic().name(), m, "{:08X} mnemonic", w);
        assert_eq!(insn.op_count(), n, "{:08X} operand count", w);
        // Canonical words must re-encode to the identical bit pattern.
        let enc = encode(&insn).expect("encode canonical");
        assert_eq!(enc, w, "{:08X} ({}) did not re-encode identically (got {:08X})", w, m, enc);
        assert_roundtrip(w);
    }
}

#[test]
fn exhaustive_roundtrip() {
    let regs = [
        (0u32, 0u32, 0u32),
        (4, 5, 6),
        (2, 4, 6),
        (5, 7, 9),
        (31, 5, 7),
        (5, 7, 31),
        (30, 1, 0),
    ];
    let mut decoded = 0usize;
    for sz in 0..4 {
        for o2 in 0..2 {
            for l in 0..2 {
                for o1 in 0..2 {
                    for o0 in 0..2 {
                        for &(rs, rn, rt) in &regs {
                            let w = mk(sz, o2, l, o1, rs, o0, 0b11111, rn, rt);
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
    assert!(decoded >= 16, "expected the LSUI forms to decode, got {}", decoded);
}

#[test]
fn reserved_forms_are_invalid() {
    // Byte/half single-exclusive (sz 0/1, o2=0) are not allocated for LSUI.
    for sz in [0u32, 1] {
        for l in 0..2 {
            for o0 in 0..2 {
                let w = mk(sz, 0, l, 0, 0b11111, o0, 0b11111, 1, 0);
                assert!(decode(w, 0, FeatureSet::ALL).is_invalid(), "{:08X} sz={} excl should be Invalid", w, sz);
            }
        }
    }
    // 32-bit CAS (sz=2, o2=1) and 32-bit CASP (sz=0, o2=1) are UNDEFINED.
    for l in 0..2 {
        for o0 in 0..2 {
            let cas32 = mk(2, 1, l, 0, 0, o0, 0b11111, 5, 6);
            let casp32 = mk(0, 1, l, 0, 0, o0, 0b11111, 5, 6);
            assert!(decode(cas32, 0, FeatureSet::ALL).is_invalid(), "{:08X} 32-bit CAS should be Invalid", cas32);
            assert!(decode(casp32, 0, FeatureSet::ALL).is_invalid(), "{:08X} 32-bit CASP should be Invalid", casp32);
        }
    }
    // o1=1 has no LSUI form (no unprivileged exclusive pair). Sweep o2/sz.
    for sz in 0..4 {
        for o2 in 0..2 {
            for l in 0..2 {
                for o0 in 0..2 {
                    let w = mk(sz, o2, l, 1, 0b11111, o0, 0b11111, 1, 0);
                    assert!(decode(w, 0, FeatureSet::ALL).is_invalid(), "{:08X} o1=1 should be Invalid", w);
                }
            }
        }
    }
    // Unlike standard CAS/CASP, the LSUI CAS/CASP forms treat Rt2 (bits<14:10>)
    // as should-be-one but IGNORED: LLVM still decodes a non-all-ones Rt2 (just
    // flags it "potentially undefined"), so fARM64 must decode it too and
    // round-trip it (the encoder re-canonicalizes Rt2 to all-ones).
    let cas_badrt2 = mk(3, 1, 0, 0, 0, 0, 0b01110, 30, 9);
    let casp_badrt2 = mk(1, 1, 0, 0, 0, 0, 0b01110, 4, 2);
    assert_eq!(decode(cas_badrt2, 0, FeatureSet::ALL).mnemonic().name(), "cast", "CAST Rt2 is SBO-ignored");
    assert_eq!(decode(casp_badrt2, 0, FeatureSet::ALL).mnemonic().name(), "caspt", "CASPT Rt2 is SBO-ignored");
    assert_roundtrip(cas_badrt2);
    assert_roundtrip(casp_badrt2);
}

#[test]
fn casp_requires_even_registers() {
    // CASPT: odd Rs or odd Rt is UNDEFINED; an even pair decodes.
    let odd_rs = mk(1, 1, 0, 0, 5, 0, 0b11111, 7, 6);
    let odd_rt = mk(1, 1, 0, 0, 4, 0, 0b11111, 7, 7);
    let even = mk(1, 1, 0, 0, 4, 0, 0b11111, 7, 6);
    assert!(decode(odd_rs, 0, FeatureSet::ALL).is_invalid(), "caspt odd Rs should be Invalid");
    assert!(decode(odd_rt, 0, FeatureSet::ALL).is_invalid(), "caspt odd Rt should be Invalid");
    assert!(!decode(even, 0, FeatureSet::ALL).is_invalid(), "caspt even pair should decode");
}

#[test]
fn gated_by_feature() {
    // With FEAT_LSUI absent, none of the unprivileged atomics decode.
    let no_lsui = without_lsui(FeatureSet::ALL);
    let words = [
        0x895F7C20u32, // ldtxr
        0xC9007FE9,    // sttxr
        0xC9807FC9,    // cast
        0x49807C82,    // caspt
    ];
    for w in words {
        assert!(decode(w, 0, no_lsui).is_invalid(), "{:08X} should require FEAT_LSUI", w);
        assert!(!decode(w, 0, FeatureSet::ALL).is_invalid(), "{:08X} should decode with FEAT_LSUI", w);
    }
}

/// `FeatureSet::ALL` minus the `Lsui` bit (in both words).
fn without_lsui(fs: FeatureSet) -> FeatureSet {
    let bit = Feature::Lsui as u32;
    FeatureSet {
        features0: fs.features0 & !(1u64 << bit),
        features1: fs.features1 & !(1u64 << bit),
    }
}
