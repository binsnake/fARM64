//! FEAT_I8MM / F32MM / F64MM / SSVE_FP8FMA / SVE_B16B16 / SME2 multi-vector
//! coverage (the G1 simd_fp area): exhaustive decode + round-trip plus
//! example-word and over-decode checks.
//!
//! Families covered:
//!   * NEON integer matrix-multiply (FEAT_I8MM): `SMMLA`/`UMMLA`/`USMMLA`
//!     (`<Vd>.4S, <Vn>.16B, <Vm>.16B`).
//!   * SVE matrix-multiply: `FMMLA` (`.s`/`.d`) and SVE `SMMLA`/`UMMLA`/`USMMLA`
//!     (`<Zda>.s, <Zn>.b, <Zm>.b`).  (Pre-existing; here for regression.)
//!   * SVE FP8 widening MLAL, indexed z-form (FEAT_SSVE_FP8FMA): `FMLALB`/
//!     `FMLALT` (`<Zda>.h, <Zn>.b, <Zm>.b[i]`) and `FMLALLBB`/`BT`/`TB`/`TT`
//!     (`<Zda>.s, <Zn>.b, <Zm>.b[i]`).
//!   * SVE BF16 multiply-add / multiply, indexed (FEAT_SVE_B16B16): `BFMLA`/
//!     `BFMLS`/`BFMUL` (`<Zda>.h, <Zn>.h, <Zm>.h[i]`).
//!   * SME2/SVE2 multi-vector `FMUL` (`{Zd..}, {Zn..}, {Zm..}`, vgx2 / vgx4).
//!
//! All example words are LLVM-21/22 (`llvm-mc` / clang + llvm-objdump) verified.

#![cfg(feature = "std")]

use fARM64::decode::decode;
use fARM64::format::{format_to_string, FmtFormatter};
use fARM64::{encode, FeatureSet};

/// Lower-case, collapse whitespace, strip spaces around `,`/brackets so
/// stylistic differences (tab vs spaces, brace padding) compare equal.
fn norm(s: &str) -> String {
    let s = s.to_ascii_lowercase();
    let mut out = String::with_capacity(s.len());
    let mut pending = false;
    for ch in s.chars() {
        if ch.is_whitespace() {
            pending = true;
            continue;
        }
        let suppress = matches!(ch, ',' | ']' | '}' | ')');
        let after_open = out.ends_with(['[', '{', '(']);
        if pending && !out.is_empty() && !suppress && !after_open {
            out.push(' ');
        }
        pending = false;
        out.push(ch);
        if ch == ',' {
            pending = true;
        }
    }
    out.trim().to_string()
}

fn text(word: u32) -> String {
    let insn = decode(word, 0, FeatureSet::ALL);
    format_to_string(&FmtFormatter::new(), &insn)
}

/// Decode `word`, assert it matches `expected` text, then prove an exact
/// (bit-for-bit) encoder round-trip.
fn check(word: u32, expected: &str) {
    let insn = decode(word, 0, FeatureSet::ALL);
    assert!(!insn.is_invalid(), "{:08X} decoded Invalid (want `{}`)", word, expected);
    assert_eq!(
        norm(&text(word)),
        norm(expected),
        "{:08X} disasm mismatch",
        word
    );
    let enc = encode(&insn)
        .unwrap_or_else(|e| panic!("{:08X} ({}) encode error {:?}", word, insn.mnemonic().name(), e));
    assert_eq!(enc, word, "{:08X} round-trip produced {:08X}", word, enc);
}

/// Require `word` to decode to `Code::Invalid` (no over-decode).
fn check_invalid(word: u32) {
    let insn = decode(word, 0, FeatureSet::ALL);
    assert!(
        insn.is_invalid(),
        "{:08X} over-decoded as `{}`",
        word,
        text(word)
    );
}

// ---------------------------------------------------------------------------
// Example-word decode + round-trip.
// ---------------------------------------------------------------------------

#[test]
fn neon_i8mm_examples() {
    check(0x4E80A45F, "smmla v31.4s, v2.16b, v0.16b");
    check(0x4E80AC5F, "usmmla v31.4s, v2.16b, v0.16b");
    check(0x6E80A45F, "ummla v31.4s, v2.16b, v0.16b");
    // a few more register combinations
    check(0x4E80A400, "smmla v0.4s, v0.16b, v0.16b");
    check(0x4E9FA400, "smmla v0.4s, v0.16b, v31.16b");
    check(0x4E81A400, "smmla v0.4s, v0.16b, v1.16b");
    check(0x4E80A420, "smmla v0.4s, v1.16b, v0.16b");
}

#[test]
fn sve_matmul_examples() {
    // FMMLA .s / .d (F32MM / F64MM).
    check(0x64A0E400, "fmmla z0.s, z0.s, z0.s");
    check(0x64E0E400, "fmmla z0.d, z0.d, z0.d");
    check(0x64A0E7E0, "fmmla z0.s, z31.s, z0.s");
    check(0x64BFE400, "fmmla z0.s, z0.s, z31.s");
    // SVE integer matmul (FEAT_I8MM).
    check(0x45029820, "smmla z0.s, z1.b, z2.b");
    check(0x45829820, "usmmla z0.s, z1.b, z2.b");
    check(0x45C29820, "ummla z0.s, z1.b, z2.b");
}

#[test]
fn sve_fp8_mlal_examples() {
    check(0x6420518E, "fmlalb z14.h, z12.b, z0.b[0]");
    check(0x64A0518E, "fmlalt z14.h, z12.b, z0.b[0]");
    check(0x64285C00, "fmlalb z0.h, z0.b, z0.b[7]");
    check(0x64385C00, "fmlalb z0.h, z0.b, z0.b[15]");
    check(0x64275000, "fmlalb z0.h, z0.b, z7.b[0]");
    check(0x6420C000, "fmlallbb z0.s, z0.b, z0.b[0]");
    check(0x6460C000, "fmlallbt z0.s, z0.b, z0.b[0]");
    check(0x64A0C000, "fmlalltb z0.s, z0.b, z0.b[0]");
    check(0x64E0C000, "fmlalltt z0.s, z0.b, z0.b[0]");
}

#[test]
fn sve_bf16_indexed_examples() {
    check(0x64220960, "bfmla z0.h, z11.h, z2.h[0]");
    check(0x64200800, "bfmla z0.h, z0.h, z0.h[0]");
    check(0x64200C00, "bfmls z0.h, z0.h, z0.h[0]");
    check(0x64202800, "bfmul z0.h, z0.h, z0.h[0]");
    check(0x64780800, "bfmla z0.h, z0.h, z0.h[7]");
    check(0x64270800, "bfmla z0.h, z0.h, z7.h[0]");
}

#[test]
fn sme2_multivector_fmul_examples() {
    check(0xC160E798, "fmul {z24.h, z25.h}, {z28.h, z29.h}, {z0.h, z1.h}");
    check(0xC160E400, "fmul {z0.h, z1.h}, {z0.h, z1.h}, {z0.h, z1.h}");
    check(0xC1A0E400, "fmul {z0.s, z1.s}, {z0.s, z1.s}, {z0.s, z1.s}");
    check(0xC1E0E400, "fmul {z0.d, z1.d}, {z0.d, z1.d}, {z0.d, z1.d}");
    check(0xC161E400, "fmul {z0.h - z3.h}, {z0.h - z3.h}, {z0.h - z3.h}");
    check(0xC1A1E400, "fmul {z0.s - z3.s}, {z0.s - z3.s}, {z0.s - z3.s}");
}

// ---------------------------------------------------------------------------
// Exhaustive round-trip over each family's encodable register/index space.
// ---------------------------------------------------------------------------

fn assert_rt(word: u32) {
    let insn = decode(word, 0, FeatureSet::ALL);
    assert!(!insn.is_invalid(), "{:08X} Invalid", word);
    let enc = encode(&insn)
        .unwrap_or_else(|e| panic!("{:08X} ({}) encode error {:?}", word, insn.mnemonic().name(), e));
    assert_eq!(enc, word, "{:08X} round-trip produced {:08X}", word, enc);
}

#[test]
fn neon_i8mm_roundtrip_sweep() {
    // 0 1 1 0 1110 1 0 0 0 Rm 1010 B 1 Rn Rd, with U=bit29, B=bit11.
    // (U,B): (0,0)=smmla (0,1)=usmmla (1,0)=ummla; (1,1) unallocated.
    for &(u, b) in &[(0u32, 0u32), (0, 1), (1, 0)] {
        for rm in [0u32, 1, 15, 31] {
            for rn in [0u32, 7, 31] {
                for rd in [0u32, 2, 31] {
                    let w = (0b0100_1110_1000_0000u32 << 16)
                        | (u << 29)
                        | (rm << 16)
                        | (0b1010 << 12)
                        | (b << 11)
                        | (1 << 10)
                        | (rn << 5)
                        | rd;
                    assert_rt(w);
                }
            }
        }
    }
}

#[test]
fn sve_fp8_mlal_roundtrip_sweep() {
    // FMLALB/T (to .h): base 0x64205000; T=bit23; index{ih<20:19>,il<11:10>};
    // Zm z0..z7 (<18:16>).
    for t in [0u32, 1] {
        for idx in 0u32..16 {
            for zm in 0u32..8 {
                for &(zn, zda) in &[(0u32, 0u32), (1, 2), (31, 31)] {
                    let w = 0x6420_5000
                        | (t << 23)
                        | ((idx >> 2) << 19)
                        | (zm << 16)
                        | ((idx & 3) << 10)
                        | (zn << 5)
                        | zda;
                    assert_rt(w);
                }
            }
        }
    }
    // FMLALLBB/BT/TB/TT (to .s): base 0x6420C000; B/T pair in <23:22>.
    for bt in 0u32..4 {
        for idx in [0u32, 1, 7, 15] {
            for zm in [0u32, 3, 7] {
                let w = 0x6420_C000
                    | (bt << 22)
                    | ((idx >> 2) << 19)
                    | (zm << 16)
                    | ((idx & 3) << 10);
                assert_rt(w);
            }
        }
    }
}

#[test]
fn sve_bf16_indexed_roundtrip_sweep() {
    // base 0x64200800 (bfmla); <15:10>: bfmla=000010 bfmls=000011 bfmul=001010.
    for sub in [0b000010u32, 0b000011, 0b001010] {
        for idx in 0u32..8 {
            for zm in 0u32..8 {
                for &(zn, zda) in &[(0u32, 0u32), (11, 0), (31, 31)] {
                    let w = (0b0110_0100u32 << 24)
                        | (1 << 21)
                        | ((idx >> 2) << 22) // i3h -> bit22
                        | ((idx & 3) << 19) // i2l -> <20:19>
                        | (zm << 16)
                        | (sub << 10)
                        | (zn << 5)
                        | zda;
                    assert_rt(w);
                }
            }
        }
    }
}

#[test]
fn sme2_multivector_fmul_roundtrip_sweep() {
    // vgx2: base 0xC120E400; vgx4: base 0xC121E400. size <23:22> = 01/10/11.
    for size in [1u32, 2, 3] {
        // vgx2: Zd<4:1> Zn<9:6> Zm<20:17>, even bases.
        for zd in [0u32, 2, 30] {
            for zn in [0u32, 4, 28] {
                for zm in [0u32, 2, 6] {
                    let w = 0xC120_E400
                        | (size << 22)
                        | ((zm / 2) << 17)
                        | ((zn / 2) << 6)
                        | ((zd / 2) << 1);
                    assert_rt(w);
                }
            }
        }
        // vgx4: Zd<4:2> Zn<9:7> Zm<20:18>, bases multiple of 4.
        for zd in [0u32, 4, 28] {
            for zn in [0u32, 8, 12] {
                for zm in [0u32, 4, 8] {
                    let w = 0xC121_E400
                        | (size << 22)
                        | ((zm / 4) << 18)
                        | ((zn / 4) << 7)
                        | ((zd / 4) << 2);
                    assert_rt(w);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Over-decode: words LLVM rejects in these slots must stay Invalid.
// ---------------------------------------------------------------------------

#[test]
fn no_over_decode() {
    // NEON i8mm slot: only Q=1, size=10 is valid.
    check_invalid(0x0E80A400); // Q=0
    check_invalid(0x4E00A400); // size 00
    check_invalid(0x4EC0A400); // size 11
    check_invalid(0x4E80A800); // wrong opcode bit
    check_invalid(0x6E80AC00); // (U=1,B=1) unallocated
    // SVE FP8 fmlalb requires <22>==0 (with <22>==1 the slot is unallocated).
    check_invalid(0x64605022);
    // multi-vector FMUL: size 00 is the (unimplemented) BF16 BFMUL neighbour,
    // bit5 / bit0 set, and a 4-register form with a stray <17> set are all
    // not FMUL.
    check_invalid(0xC160E420); // bit5 set
    check_invalid(0xC160E401); // bit0 set
    check_invalid(0xC163E408); // vgx4 with stray <17>
    check_invalid(0xC161E440); // vgx4 with stray <6>
}
