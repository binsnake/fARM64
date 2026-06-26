//! NEON / Advanced-SIMD over-decode hardening, batch K2.
//!
//! Four reserved-encoding families in the Advanced-SIMD floating-point space,
//! each derived from an LLVM field sweep (`oracle.py dec/enc`, i.e. `clang
//! .inst` + `llvm-objdump --mattr=+all`) and proven 0-regression with a pre/post
//! differential over the affected top bytes (`0x0E`/`0x2E`/`0x4E`/`0x6E`/`0x5E`/
//! `0x7E` and their by-element `0x_F` siblings): a 96,835-word structured+random
//! sweep eliminated 1,829 over-decoded words with **0** LLVM-valid words newly
//! rejected, **0** invalid→valid flips, and **0** valid-word text changes.
//!
//! Families hardened (all in `src/decode/simd_fp/simd_arith.rs`):
//!
//!  1. **NEON three-same (FP16)** — the "Advanced SIMD three same (FP16)" format
//!     fixes `word<23:21> = a:1:0`; `word<21>==0` is already enforced by the
//!     caller, but `word<22>` must be `1`. A `word<22>==0` word is reserved
//!     (`size==00` is the copy family, `size==10` is UNDEFINED). E.g. `0E850EE1`
//!     (fARM64 `fmls v1.4h,…`) → UNDEFINED; the real form is `0EC50EE1`.
//!
//!  2. **NEON `.2d`/`d` by-element FP** — for the double-precision (`size==11`)
//!     FMLA/FMLS/FMUL/FMULX by-element forms the index is `H` alone, so the `L`
//!     bit (`word<21>`) is a fixed `0`; `word<21>==1` is reserved. E.g.
//!     `4FE7996A` (fARM64 `fmul v10.2d,…,v7.d[1]`) → UNDEFINED; real `4FC7996A`.
//!
//!  3. **NEON FCMLA by-element** — the complex index addresses pairs, so its
//!     width tracks the lane count: for `.4h` (size==01, Q==0) the top index bit
//!     `H = word<11>` is fixed `0`, and for `.4s` (size==10, Q==1) the `L` bit
//!     `word<21>` is fixed `0`. E.g. `2F623A9B` (`fcmla v27.4h,…,v2.h[3],#0x5a`)
//!     and `6FA2329B` (`.4s` with `L==1`) → UNDEFINED.
//!
//!  4. **Advanced-SIMD scalar three-same FP / FP16** (`0x5E`/`0x7E`) — the scalar
//!     class allocates only a subset of the vector FP opcodes; the add/sub/min/
//!     max/pairwise/FMLA-style ops are vector-only. Valid scalar `(U,a/o1,opcode)`
//!     ∈ FMULX/FCMEQ/FRECPS/FRSQRTS/FCMGE/FACGE/FABD/FCMGT/FACGT only. E.g.
//!     `5E560750` (fARM64 `fmaxnm h16,…`), `5E2CD6DA` (`fadd s22,…`),
//!     `7E560750` (`fmaxnmp h…`) → UNDEFINED. The real scalar `fmaxnm h` lives in
//!     the FP-data-processing class (`1EF66B50`) and is unaffected.

#![cfg(feature = "std")]

use fARM64::decode::decode;
use fARM64::format::{format_to_string, FmtFormatter};
use fARM64::{encode, FeatureSet};

fn text(word: u32) -> String {
    let insn = decode(word, 0, FeatureSet::ALL);
    format_to_string(&FmtFormatter::new(), &insn)
}

fn norm(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Decode, assert disasm == `expected`, and prove a bit-exact encode round-trip.
#[track_caller]
fn check(word: u32, expected: &str) {
    let insn = decode(word, 0, FeatureSet::ALL);
    assert!(!insn.is_invalid(), "{:08X} decoded Invalid (want `{}`)", word, expected);
    assert_eq!(norm(&text(word)), norm(expected), "{:08X} disasm mismatch", word);
    let enc = encode(&insn)
        .unwrap_or_else(|e| panic!("{:08X} ({}) encode error {:?}", word, insn.mnemonic().name(), e));
    assert_eq!(enc, word, "{:08X} round-trip produced {:08X}", word, enc);
}

/// Assert a word is rejected (reserved / UNDEFINED).
#[track_caller]
fn reserved(word: u32) {
    assert!(decode(word, 0, FeatureSet::ALL).is_invalid(), "{word:08X} should be reserved (Invalid)");
}

// ---------------------------------------------------------------------------
// Valid forms — must still decode + round-trip exactly (regression guard).
// ---------------------------------------------------------------------------

#[test]
fn valid_forms_still_decode_and_roundtrip() {
    // (1) NEON three-same FP16 with `word<22>==1` (the real forms).
    check(0x0EC50EE1, "fmls v1.4h, v23.4h, v5.4h");
    check(0x4E561750, "fadd v16.8h, v26.8h, v22.8h");
    check(0x6E563750, "fmaxp v16.8h, v26.8h, v22.8h");

    // (2) NEON `.2d`/`d` by-element FP with `L==0` (the real forms).
    check(0x4FC7996A, "fmul v10.2d, v11.2d, v7.d[1]");
    check(0x4FC713EE, "fmla v14.2d, v31.2d, v7.d[0]");
    check(0x6FC7996A, "fmulx v10.2d, v11.2d, v7.d[1]");

    // (3) FCMLA by-element — valid `.4h` (Q==0, H==0), `.8h` (Q==1), `.4s`
    // (Q==1, L==0). fARM64 renders the rotate in hex (`#0x5a` = 90°), an
    // intentional radix difference from LLVM's `#90`.
    check(0x2F42329B, "fcmla v27.4h, v20.4h, v2.h[0], #0x5a");
    check(0x6F62329B, "fcmla v27.8h, v20.8h, v2.h[1], #0x5a");
    check(0x6F523A9B, "fcmla v27.8h, v20.8h, v18.h[2], #0x5a");

    // (4) Advanced-SIMD scalar three-same FP / FP16 — the allocated subset.
    check(0x5E2CDED6, "fmulx s22, s22, s12");
    check(0x5E2CE6D6, "fcmeq s22, s22, s12");
    check(0x7EECD6D6, "fabd d22, d22, d12");
    check(0x5EACFED6, "frsqrts s22, s22, s12");
    check(0x5E561F50, "fmulx h16, h26, h22"); // scalar FP16 FMULX
    check(0x5E562750, "fcmeq h16, h26, h22"); // scalar FP16 FCMEQ
    check(0x7ED62750, "fcmgt h16, h26, h22"); // scalar FP16 FCMGT
    // The real scalar `fmaxnm h` lives in the FP-data-processing class and must
    // keep decoding (it is *not* in the `0x5E`/`0x7E` scalar-three-same class).
    check(0x1EF66B50, "fmaxnm h16, h26, h22");
}

// ---------------------------------------------------------------------------
// (1) NEON three-same FP16: `word<22>` must be 1.
// ---------------------------------------------------------------------------

#[test]
fn fp16_three_same_bit22_reserved() {
    // The over-decode examples from the task (all `a==1`, `size==10`).
    reserved(0x0E850EE1); // fARM64 `fmls v1.4h,…`
    reserved(0x0E921796); // fARM64 `fsub`
    reserved(0x0E903734); // fARM64 `fmin`
    reserved(0x0E8E067D); // fARM64 `fminnm`
    reserved(0x2E8E067D); // fARM64 `fminnmp`
    reserved(0x2E903734); // fARM64 `fminp`

    // Exhaustive: every reachable `word<22>==0` FP16-three-same word (`a==1`,
    // `size==10`, so `<23:22>=10`) is reserved, across U/Q/opcode.
    for q in [0u32, 1] {
        for u in [0u32, 1] {
            for op in 0u32..8 {
                // size<0> (`word<22>`) is left `0` (reserved); `a = word<23>` = 1.
                let w = (q << 30)
                    | (u << 29)
                    | (0b01110 << 24)
                    | (1 << 23) // a
                    | (5 << 16)
                    | (op << 11)
                    | (1 << 10)
                    | (23 << 5)
                    | 1;
                reserved(w);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// (2) `.2d`/`d` by-element FP: `word<21>` (L) must be 0 for size==11.
// ---------------------------------------------------------------------------

#[test]
fn by_element_d_l_bit_reserved() {
    reserved(0x4FE7996A); // fARM64 `fmul v10.2d,…,v7.d[1]`
    reserved(0x4FE713EE); // fARM64 `fmla`
    reserved(0x6FE7996A); // fARM64 `fmulx`
    reserved(0x5FE7996A); // scalar `fmul d,…` with L==1

    // Sweep L==1 across FMLA/FMLS/FMUL/FMULX, scalar+vector, both H values.
    for (top, op) in [
        (0x4Fu32, 0b0001u32), // fmla vector
        (0x4F, 0b0101),       // fmls vector
        (0x4F, 0b1001),       // fmul vector
        (0x6F, 0b1001),       // fmulx vector
        (0x5F, 0b1001),       // fmul scalar
        (0x7F, 0b1001),       // fmulx scalar
    ] {
        for h in [0u32, 1] {
            let w = (top << 24)
                | (0b11 << 22) // size == .d
                | (1 << 21) // L == 1 → reserved
                | (0b00111 << 16) // M:Vm
                | (op << 12)
                | (h << 11)
                | (11 << 5)
                | 10;
            reserved(w);
        }
    }

    // The `.s` (size==10) by-element forms legitimately use L as an index bit
    // (here index 3 = H:L = 11) and must stay valid.
    check(0x4FA799EE, "fmul v14.4s, v15.4s, v7.s[3]");
}

// ---------------------------------------------------------------------------
// (3) FCMLA by-element index/lane reserved bits.
// ---------------------------------------------------------------------------

#[test]
fn fcmla_by_element_reserved() {
    // `.4h` (size==01, Q==0): the top index bit `H = word<11>` is fixed 0.
    reserved(0x2F623A9B); // fARM64 `fcmla v27.4h,…,v2.h[3],#0x5a`
    for l in [0u32, 1] {
        for m in [0u32, 1] {
            for rot in 0u32..4 {
                let op = (rot << 1) | 1; // opcode<15:12> = 0:rot:1
                // Q == 0 (`word<30>` left 0); H == 1 (`word<11>`) is reserved.
                let w = (1 << 29) // U
                    | (0b01111 << 24)
                    | (0b01 << 22) // size == .h
                    | (l << 21)
                    | (m << 20)
                    | (2 << 16)
                    | (op << 12)
                    | (1 << 11) // H == 1 → reserved for .4h
                    | (20 << 5)
                    | 27;
                reserved(w);
            }
        }
    }

    // `.4s` (size==10, Q==1): the `L = word<21>` bit is fixed 0.
    reserved(0x6FA2329B);
    for h in [0u32, 1] {
        for rot in 0u32..4 {
            let op = (rot << 1) | 1;
            let w = (1 << 30) // Q == 1
                | (1 << 29) // U
                | (0b01111 << 24)
                | (0b10 << 22) // size == .s
                | (1 << 21) // L == 1 → reserved for .s
                | (2 << 16)
                | (op << 12)
                | (h << 11)
                | (20 << 5)
                | 27;
            reserved(w);
        }
    }

    // `.2s` (size==10, Q==0) FCMLA by element is entirely unallocated.
    reserved(0x2F82329B);
}

// ---------------------------------------------------------------------------
// (4) Advanced-SIMD scalar three-same FP / FP16: only a subset is allocated.
// ---------------------------------------------------------------------------

#[test]
fn scalar_three_same_fp_reserved_opcodes() {
    // Scalar FP16 (`0x5E`/`0x7E`, `<15:14>==00`, opcode `<13:11>`).
    reserved(0x5E560750); // fARM64 `fmaxnm h16,…`
    reserved(0x5E5F347A); // fARM64 `fmax h`
    reserved(0x7E560750); // fARM64 `fmaxnmp h`

    // Scalar single/double (`0x5E`/`0x7E`, opcode `<15:11>`).
    reserved(0x5E2CD6DA); // fARM64 `fadd s`

    // The allocated scalar-FP16 set is exactly these 9 (U,a,opcode); everything
    // else in the reachable opcode space is reserved.
    let valid: &[(u32, u32, u32)] = &[
        (0, 0, 0b011), // FMULX
        (0, 0, 0b100), // FCMEQ
        (0, 0, 0b111), // FRECPS
        (0, 1, 0b111), // FRSQRTS
        (1, 0, 0b100), // FCMGE
        (1, 0, 0b101), // FACGE
        (1, 1, 0b010), // FABD
        (1, 1, 0b100), // FCMGT
        (1, 1, 0b101), // FACGT
    ];
    for u in [0u32, 1] {
        for a in [0u32, 1] {
            for op in 0u32..8 {
                let w = (1 << 30)
                    | (u << 29)
                    | (0b11110 << 24)
                    | (a << 23)
                    | (1 << 22)
                    | (22 << 16)
                    | (op << 11)
                    | (1 << 10)
                    | (26 << 5)
                    | 16;
                if valid.contains(&(u, a, op)) {
                    assert!(!decode(w, 0, FeatureSet::ALL).is_invalid(), "{w:08X} should be a valid scalar FP16 op");
                } else {
                    reserved(w);
                }
            }
        }
    }

    // The single/double scalar class has the same allocation. Sweep the FP
    // opcode block `<15:11>` ∈ 11xxx with `<21>==1`.
    let valid_sd: &[(u32, u32, u32)] = &[
        (0, 0, 0b11011), // FMULX
        (0, 0, 0b11100), // FCMEQ
        (0, 0, 0b11111), // FRECPS
        (0, 1, 0b11111), // FRSQRTS
        (1, 0, 0b11100), // FCMGE
        (1, 0, 0b11101), // FACGE
        (1, 1, 0b11010), // FABD
        (1, 1, 0b11100), // FCMGT
        (1, 1, 0b11101), // FACGT
    ];
    for u in [0u32, 1] {
        for o1 in [0u32, 1] {
            for sz in [0u32, 1] {
                for op in 0b11000u32..=0b11111 {
                    let w = (1 << 30)
                        | (u << 29)
                        | (0b11110 << 24)
                        | (o1 << 23)
                        | (sz << 22)
                        | (1 << 21)
                        | (12 << 16)
                        | (op << 11)
                        | (1 << 10)
                        | (22 << 5)
                        | 22;
                    if valid_sd.contains(&(u, o1, op)) {
                        assert!(
                            !decode(w, 0, FeatureSet::ALL).is_invalid(),
                            "{w:08X} should be a valid scalar S/D FP op"
                        );
                    } else {
                        reserved(w);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Feature gating: FP16 forms require FEAT_FP16.
// ---------------------------------------------------------------------------

#[test]
fn fp16_feature_gating() {
    use fARM64::features::{Feature, FeatureSet};
    // Build `FeatureSet::ALL` with the FP16 bit cleared.
    let mask = !(1u64 << (Feature::Fp16 as u32));
    let no_fp16 = FeatureSet { features0: FeatureSet::ALL.features0 & mask, features1: FeatureSet::ALL.features1 & mask };
    // A valid FP16 scalar three-same op is gated off without FEAT_FP16.
    assert!(decode(0x5E561F50, 0, no_fp16).is_invalid());
    // A valid FP16 vector three-same op likewise.
    assert!(decode(0x4E561750, 0, no_fp16).is_invalid());
}
