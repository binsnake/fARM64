//! FEAT_SVE2p1 group G2: decode + encode round-trip coverage for
//!
//! 1. quadword (`.q`) contiguous structured + gather/scatter loads/stores
//!    (`LD1Q`/`ST1Q` gather/scatter, `LD{2,3,4}Q`/`ST{2,3,4}Q` structured);
//! 2. `REVD` zeroing (`/z`) plus the merging-form size-reserved over-decode fix;
//! 3. `WHILE<cc>` predicate-pair (`{p0.b, p1.b}`) and predicate-as-counter
//!    (`pn8.b, ..., vlx{2,4}`) forms;
//! 4. SVE FP unary predicated convert/round, zeroing (`/z`).
//!
//! Every canonical word here is an LLVM (`clang` `.inst` + `llvm-objdump
//! --mattr=+all`) oracle encoding. The tests confirm fARM64 decodes them to the
//! expected mnemonic / operand-count, re-encodes bit-identically, sweeps the
//! sub-spaces for semantic round-trip stability, and that reserved / feature-
//! gated slots stay `Invalid`.

#![cfg(feature = "std")]

use fARM64::decode::decode;
use fARM64::format::{format_to_string, FmtFormatter};
use fARM64::{encode, Feature, FeatureSet};

/// Decode `word`, re-encode, re-decode; require identical mnemonic + operands.
fn assert_roundtrip(word: u32) {
    let insn = decode(word, 0, FeatureSet::ALL);
    assert!(!insn.is_invalid(), "{:08X} decoded Invalid", word);
    let enc = encode(&insn)
        .unwrap_or_else(|e| panic!("{:08X} ({}) encode error {:?}", word, insn.mnemonic().name(), e));
    assert_eq!(enc, word, "{:08X} ({}) re-encoded to {:08X}", word, insn.mnemonic().name(), enc);
    let insn2 = decode(enc, 0, FeatureSet::ALL);
    assert_eq!(insn.mnemonic(), insn2.mnemonic(), "{:08X} mnemonic drift", word);
    assert_eq!(insn.op_count(), insn2.op_count(), "{:08X} operand-count drift", word);
}

fn text(word: u32) -> String {
    let insn = decode(word, 0, FeatureSet::ALL);
    format_to_string(&FmtFormatter::new(), &insn)
}

// ---------------------------------------------------------------------------
// 1. Quadword load/store.
// ---------------------------------------------------------------------------

#[test]
fn quadword_examples() {
    // (word, mnemonic, op_count) — canonical LLVM (+all) oracle encodings.
    let cases: &[(u32, &str, usize)] = &[
        (0xc41fa000, "ld1q", 3),  // ld1q {z0.q}, p0/z, [z0.d]
        (0xc401a000, "ld1q", 3),  // ld1q {z0.q}, p0/z, [z0.d, x1]
        (0xe43f2000, "st1q", 3),  // st1q {z0.q}, p0, [z0.d]
        (0xe4212000, "st1q", 3),  // st1q {z0.q}, p0, [z0.d, x1]
        (0xa5208101, "ld3q", 3),  // ld3q {z1.q-z3.q}, p0/z, [x8, x0, lsl #4]
        (0xa4a18000, "ld2q", 3),  // ld2q ... ss
        (0xa498e000, "ld2q", 3),  // ld2q ... imm
        (0xa5a18000, "ld4q", 3),  // ld4q ... ss
        (0xe4610000, "st2q", 3),  // st2q ... ss
        (0xe4480000, "st2q", 3),  // st2q ... imm
        (0xe4a10000, "st3q", 3),  // st3q ... ss
        (0xe4e10000, "st4q", 3),  // st4q ... ss
    ];
    for &(w, m, n) in cases {
        let insn = decode(w, 0, FeatureSet::ALL);
        assert_eq!(insn.mnemonic().name(), m, "{:08X} mnemonic", w);
        assert_eq!(insn.op_count(), n, "{:08X} operand count", w);
        assert_roundtrip(w);
    }
    // Spot-check a couple of full renderings (house style: comma list, hex lsl).
    assert_eq!(text(0xa5208101), "ld3q    {z1.q, z2.q, z3.q}, p0/z, [x8, x0, lsl #0x4]");
    assert_eq!(text(0xc41fa000), "ld1q    {z0.q}, p0/z, [z0.d]");
    assert_eq!(text(0xc401a000), "ld1q    {z0.q}, p0/z, [z0.d, x1]");
}

#[test]
fn quadword_struct_roundtrip_sweep() {
    let mut decoded = 0usize;
    // Loads (0xa4/0xa5) and stores (0xe4) structured forms, both ss + imm.
    for nreg in 2u8..=4 {
        for store in [false, true] {
            // ss form: `[Xn, Xm, lsl #4]`, Xm != 31.
            for &(rn, rm, zt, pg) in &[(0u32, 1u32, 0u32, 0u32), (10, 9, 5, 3), (31, 30, 8, 7)] {
                let top: u32;
                let nfield: u32;
                if store {
                    top = 0xe4;
                    nfield = (nreg as u32 - 1) << 22;
                } else {
                    top = if nreg == 2 { 0xa4 } else { 0xa5 };
                    nfield = (nreg as u32 - 1) << 23;
                }
                let sel = if store { 0b000 } else { 0b100 };
                let w = (top << 24) | nfield | (1 << 21) | (rm << 16) | (sel << 13) | (pg << 10) | (rn << 5) | zt;
                if !decode(w, 0, FeatureSet::ALL).is_invalid() {
                    decoded += 1;
                    assert_roundtrip(w);
                }
                // imm form.
                let isel = if store { 0b000 } else { 0b111 };
                let b20 = if store { 0 } else { 1 };
                let i4 = 7u32; // imm4 = 7 -> #(7*nreg), mul vl
                let wi = (top << 24) | nfield | (b20 << 20) | (i4 << 16) | (isel << 13) | (pg << 10) | (rn << 5) | zt;
                if !decode(wi, 0, FeatureSet::ALL).is_invalid() {
                    decoded += 1;
                    assert_roundtrip(wi);
                }
            }
        }
    }
    assert!(decoded >= 30, "expected many quadword structured forms, got {}", decoded);
}

#[test]
fn quadword_gather_roundtrip_and_reserved() {
    // LD1Q gather + ST1Q scatter, both `[Zn.d]` (Rm=31) and `[Zn.d, Xm]`.
    for &(load, rm) in &[(true, 31u32), (true, 1), (false, 31), (false, 3)] {
        for &(zn, pg, zt) in &[(0u32, 0u32, 0u32), (10, 3, 5), (31, 7, 31)] {
            let w = if load {
                0xc400_0000 | (0b101 << 13) | (rm << 16) | (pg << 10) | (zn << 5) | zt
            } else {
                0xe400_0000 | (0b001 << 21) | (0b001 << 13) | (rm << 16) | (pg << 10) | (zn << 5) | zt
            };
            assert!(!decode(w, 0, FeatureSet::ALL).is_invalid(), "{:08X} should decode", w);
            assert_roundtrip(w);
        }
    }
    // Structured ss form with Xm==31 (xzr) is UNDEFINED.
    let bad = (0xa4u32 << 24) | (1 << 23) | (1 << 21) | (31 << 16) | (0b100 << 13);
    assert!(decode(bad, 0, FeatureSet::ALL).is_invalid(), "{:08X} ss xzr should be Invalid", bad);
}

// ---------------------------------------------------------------------------
// 2. REVD /z + merging size over-decode fix.
// ---------------------------------------------------------------------------

#[test]
fn revd_merging_and_zeroing() {
    // Merging (/m) is base; zeroing (/z) is SVE2.1.
    assert_eq!(text(0x052e8020), "revd    z0.q, p0/m, z1.q");
    assert_eq!(text(0x052ea020), "revd    z0.q, p0/z, z1.q");
    assert_eq!(text(0x052e8d45), "revd    z5.q, p3/m, z10.q");
    assert_eq!(text(0x052ead45), "revd    z5.q, p3/z, z10.q");
    assert_roundtrip(0x052e8020);
    assert_roundtrip(0x052ea020);
    assert_roundtrip(0x052e8d45);
    assert_roundtrip(0x052ead45);
}

#[test]
fn revd_reserved_slots_invalid() {
    // size != 00 is reserved (UNDEFINED) for both /m and /z.
    for size in 1u32..=3 {
        for mbit in 0u32..=1 {
            let w = (0x05u32 << 24) | (size << 22) | (1 << 21) | (0b01110 << 16) | ((0b100 | mbit) << 13) | (1 << 5);
            assert!(decode(w, 0, FeatureSet::ALL).is_invalid(), "{:08X} size={} should be Invalid", w, size);
        }
    }
    // `<21>` must be 1 (the merging-form over-decode that was fixed).
    let no21 = (0x05u32 << 24) | (0b01110 << 16) | (0b100 << 13) | (1 << 5);
    assert!(decode(no21, 0, FeatureSet::ALL).is_invalid(), "{:08X} <21>=0 should be Invalid", no21);
}

// ---------------------------------------------------------------------------
// 3. WHILE predicate-pair / predicate-as-counter.
// ---------------------------------------------------------------------------

#[test]
fn while_pair_pn_examples() {
    // (word, mnemonic, op_count, rendered).
    let cases: &[(u32, &str, usize, &str)] = &[
        (0x25225110, "whilege", 3, "whilege {p0.b, p1.b}, x8, x2"),
        (0x25225911, "whilehi", 3, "whilehi {p0.b, p1.b}, x8, x2"),
        (0x25225d11, "whilels", 3, "whilels {p0.b, p1.b}, x8, x2"),
        (0x2522591f, "whilehi", 3, "whilehi {p14.b, p15.b}, x8, x2"),
        (0x25625911, "whilehi", 3, "whilehi {p0.h, p1.h}, x8, x2"),
        (0x25224918, "whilehi", 4, "whilehi pn8.b, x8, x2, vlx2"),
        (0x25226918, "whilehi", 4, "whilehi pn8.b, x8, x2, vlx4"),
        (0x2522691f, "whilehi", 4, "whilehi pn15.b, x8, x2, vlx4"),
        (0x25e24918, "whilehi", 4, "whilehi pn8.d, x8, x2, vlx2"),
    ];
    for &(w, m, n, t) in cases {
        let insn = decode(w, 0, FeatureSet::ALL);
        assert_eq!(insn.mnemonic().name(), m, "{:08X} mnemonic", w);
        assert_eq!(insn.op_count(), n, "{:08X} op count", w);
        assert_eq!(format_to_string(&FmtFormatter::new(), &insn), t, "{:08X} render", w);
        assert_roundtrip(w);
    }
}

#[test]
fn while_pair_pn_roundtrip_sweep() {
    let mut n = 0usize;
    for size in 0u32..=3 {
        for cc in 0u32..8 {
            // cc -> (U, lt, eq).
            let (u, lt, eq) = ((cc >> 2) & 1, (cc >> 1) & 1, cc & 1);
            for &(rn, rm) in &[(8u32, 2u32), (0, 31), (16, 5)] {
                // pred-pair: bit12=1, eq at bit0, Pd<3:1>.
                let k = 3u32; // pair index -> p6/p7
                let pair = (0x25u32 << 24)
                    | (size << 22)
                    | (1 << 21)
                    | (rm << 16)
                    | (0b010 << 13)
                    | (1 << 12)
                    | (u << 11)
                    | (lt << 10)
                    | (rn << 5)
                    | (1 << 4)
                    | (k << 1)
                    | eq;
                if !decode(pair, 0, FeatureSet::ALL).is_invalid() {
                    n += 1;
                    assert_roundtrip(pair);
                }
                // pred-as-counter: bit12=0, eq at bit3, PN<2:0>, vlx via bit13.
                for vl in 0u32..=1 {
                    let pn = 3u32; // pn11
                    let cnt = (0x25u32 << 24)
                        | (size << 22)
                        | (1 << 21)
                        | (rm << 16)
                        | (0b01 << 14)
                        | (vl << 13)
                        | (u << 11)
                        | (lt << 10)
                        | (rn << 5)
                        | (1 << 4)
                        | (eq << 3)
                        | pn;
                    if !decode(cnt, 0, FeatureSet::ALL).is_invalid() {
                        n += 1;
                        assert_roundtrip(cnt);
                    }
                }
            }
        }
    }
    assert!(n >= 200, "expected many WHILE pair/pn forms, got {}", n);
}

// ---------------------------------------------------------------------------
// 4. FP unary predicated convert/round, zeroing (/z).
// ---------------------------------------------------------------------------

#[test]
fn fp_unary_zeroing_examples() {
    let cases: &[(u32, &str, &str)] = &[
        (0x64588020, "frintn", "frintn  z0.h, p0/z, z1.h"),
        (0x6498a020, "frintp", "frintp  z0.s, p0/z, z1.s"),
        (0x64d8c020, "frintm", "frintm  z0.d, p0/z, z1.d"),
        (0x649ba020, "fsqrt", "fsqrt   z0.s, p0/z, z1.s"),
        (0x649b8020, "frecpx", "frecpx  z0.s, p0/z, z1.s"),
        (0x649a8020, "fcvt", "fcvt    z0.h, p0/z, z1.s"),
        (0x641ac020, "fcvtx", "fcvtx   z0.s, p0/z, z1.d"),
        (0x649d8020, "scvtf", "scvtf   z0.s, p0/z, z1.s"),
        (0x649fa020, "fcvtzu", "fcvtzu  z0.s, p0/z, z1.s"),
        (0x64dea000, "fcvtzu", "fcvtzu  z0.s, p0/z, z0.d"),
        (0x649ac020, "bfcvt", "bfcvt   z0.h, p0/z, z1.s"),
        (0x641ec020, "flogb", "flogb   z0.s, p0/z, z1.s"),
    ];
    for &(w, m, t) in cases {
        let insn = decode(w, 0, FeatureSet::ALL);
        assert_eq!(insn.mnemonic().name(), m, "{:08X} mnemonic", w);
        assert_eq!(insn.op_count(), 3, "{:08X} op count", w);
        assert_eq!(format_to_string(&FmtFormatter::new(), &insn), t, "{:08X} render", w);
        assert_roundtrip(w);
    }
}

#[test]
fn fp_unary_zeroing_sweep() {
    // Sweep the whole 0x64, <21>=0, <20:19>=11, <15>=1 region; every form that
    // decodes must round-trip (and the slot must never panic).
    let mut n = 0usize;
    for size in 0u32..=3 {
        for opc in 0u32..8 {
            for sel in [4u32, 5, 6, 7] {
                let w = (0x64u32 << 24) | (size << 22) | (0b11 << 19) | (opc << 16) | (sel << 13) | (1 << 5);
                if !decode(w, 0, FeatureSet::ALL).is_invalid() {
                    n += 1;
                    assert_roundtrip(w);
                }
            }
        }
    }
    assert!(n >= 40, "expected many FP /z forms, got {}", n);
}

// ---------------------------------------------------------------------------
// Feature gating: every G2 family requires FEAT_SVE2p1.
// ---------------------------------------------------------------------------

#[test]
fn gated_by_sve2p1() {
    let no = without_sve2p1(FeatureSet::ALL);
    let words = [
        0xc41fa000u32, // ld1q
        0xa5208101,    // ld3q
        0xe4e10000,    // st4q
        0x052ea020,    // revd /z
        0x25225911,    // whilehi pair
        0x25224918,    // whilehi pn
        0x64588020,    // frintn /z
        0x64dea000,    // fcvtzu /z
    ];
    for w in words {
        assert!(decode(w, 0, no).is_invalid(), "{:08X} should require FEAT_SVE2p1", w);
        assert!(!decode(w, 0, FeatureSet::ALL).is_invalid(), "{:08X} should decode with FEAT_SVE2p1", w);
    }
    // The REVD *merging* form is base SVE and must still decode without SVE2.1.
    assert!(!decode(0x052e8020, 0, no).is_invalid(), "REVD /m must remain base-SVE");
}

/// `FeatureSet::ALL` minus the `Sve2p1` bit (in both words).
fn without_sve2p1(fs: FeatureSet) -> FeatureSet {
    let bit = Feature::Sve2p1 as u32;
    FeatureSet {
        features0: fs.features0 & !(1u64 << bit),
        features1: fs.features1 & !(1u64 << bit),
    }
}
