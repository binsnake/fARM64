//! Batch J: SME/SVE decode correctness + over-decode hardening.
//!
//! Each item was found via an LLVM (`clang` + `llvm-objdump --mattr=+all`) oracle
//! field sweep and proven 0-regression both directions (every newly-rejected word
//! is `<unknown>` in LLVM; no LLVM-valid word is newly rejected). The canonical
//! example words below are LLVM oracle encodings.
//!
//! 1. **SME single-vector `MOVA` → `MOVAZ`** (correctness + mnemonic). The
//!    zeroing tile→vector readout `MOVAZ` shares the single-vector `MOVA`
//!    tile→vector shell but sets `word<9> == 1` and has no governing predicate;
//!    fARM64 used to mis-decode it as the predicated `MOVA`. Also hardens the
//!    region: `Q` (`word<16>`) RES0 for sizes `!= .Q`, `MOVAZ` `Pg` (`word<12:10>`)
//!    RES0, vector→tile `word<4>` RES0.
//! 2. **SVE 64-bit gather `uxtw`/`sxtw` over-decode**: a `.d` vector-offset gather
//!    cannot use the 32-bit-unpacked `uxtw`/`sxtw` modifier.
//! 3. **SVE predicated `CPY`/`MOV`-immediate reserved**: the shift field has a
//!    reserved value.
//! 4. **SVE `EXT` immediate reserved**: the 2-register `EXT` position immediate
//!    is bounded.

#![cfg(feature = "std")]

use fARM64::decode::decode;
use fARM64::format::{format_to_string, FmtFormatter};
use fARM64::{encode, Feature, FeatureSet};

/// Render `word` with the default UAL formatter (all features accepted).
fn text(word: u32) -> String {
    let insn = decode(word, 0, FeatureSet::ALL);
    format_to_string(&FmtFormatter::new(), &insn)
}

/// Decode `word`, re-encode, require the identical word back.
#[track_caller]
fn assert_roundtrip(word: u32) {
    let insn = decode(word, 0, FeatureSet::ALL);
    assert!(!insn.is_invalid(), "{word:08X} decoded Invalid");
    let enc = encode(&insn).unwrap_or_else(|e| panic!("{word:08X} re-encode failed: {e:?}"));
    assert_eq!(enc, word, "{word:08X} round-trip mismatch -> {enc:08X}");
}

#[track_caller]
fn assert_invalid(word: u32) {
    assert!(
        decode(word, 0, FeatureSet::ALL).is_invalid(),
        "{word:08X} expected Invalid, got `{}`",
        text(word)
    );
}

// ---------------------------------------------------------------------------
// Item 1: SME single-vector MOVAZ.
// ---------------------------------------------------------------------------

#[test]
fn j1_movaz_single_vector() {
    // The zeroing tile→vector readout (no predicate). fARM64 renders the tile
    // slice in binja house style (`z`-prefix, hex immediate); LLVM prints
    // `movaz z8.h, za0h.h[w12, 1]` — same instruction, intentional radix/prefix.
    assert_eq!(text(0xC0420228), "movaz   z8.h, z0h.h[w12, #0x1]");
    assert_eq!(text(0xC0020200), "movaz   z0.b, z0h.b[w12, #0x0]");
    // Vertical slice, `.s` element, `Ws == w13`.
    assert_eq!(text(0xC082A200), "movaz   z0.s, z0v.s[w13, #0x0]");
    assert_roundtrip(0xC0420228);
    assert_roundtrip(0xC0020200);
    assert_roundtrip(0xC082A200);
}

#[test]
fn j1_predicated_mova_preserved() {
    // `word<9> == 0` is still the predicated single-vector MOVA, both directions.
    assert_eq!(text(0xC0420028), "mova    z8.h, p0/m, z0h.h[w12, #0x1]");
    assert_eq!(text(0xC002D4B0), "mova    z16.b, p5/m, z0v.b[w14, #0x5]");
    // vector→tile direction (word<17> == 0) is unchanged.
    assert_eq!(text(0xC0000000), "mova    z0h.b[w12, #0x0], p0/m, z0.b");
    assert_eq!(text(0xC0002F06), "mova    z0h.b[w13, #0x6], p3/m, z24.b");
    assert_roundtrip(0xC0420028);
    assert_roundtrip(0xC002D4B0);
    assert_roundtrip(0xC0000000);
}

#[test]
fn j1_movaz_reserved() {
    // `MOVAZ` has no predicate → `Pg` field `word<12:10>` is RES0.
    assert_invalid(0xC0020600); // pg == 001
    assert_invalid(0xC0020A00); // pg == 010
    assert_invalid(0xC0021200); // pg == 100
    // `Q` (`word<16>`) RES0 for non-`.Q` sizes (`.b`/`.h`/`.s`/`.d`).
    assert_invalid(0xC0010000); // size .b, Q set
    assert_invalid(0xC0410000); // size .h, Q set
    assert_invalid(0xC0810000); // size .s, Q set
    // vector→tile `word<4>` RES0 (lies between the field and `Zn`).
    assert_invalid(0xC00000FF); // valid sibling is C00000EF
}

#[test]
fn j1_movaz_feature_gated_on_sme2() {
    // MOVAZ is FEAT_SME2; without it the encoding must stay Invalid even though
    // base SME is accepted.
    let mut feats = FeatureSet::BASE;
    feats = feats.with(Feature::Sme); // base SME on, SME2 off
    let insn = decode(0xC0420228, 0, feats);
    assert!(insn.is_invalid(), "MOVAZ decoded without FEAT_SME2");
    // The predicated MOVA (base SME) still decodes under base SME.
    let insn2 = decode(0xC0420028, 0, feats);
    assert!(!insn2.is_invalid(), "predicated MOVA needs only base SME");
}

// ---------------------------------------------------------------------------
// Item 2: SVE 64-bit gather signed-dword (`uxtw`/`sxtw`/`lsl`) over-decode.
// ---------------------------------------------------------------------------

#[test]
fn j2_gather64_signed_dword_reserved() {
    // The signed `dword` 64-bit-element gather does not exist (`LD1SD`/`LDFF1SD`
    // /`LDNT1SD`): an 8-byte fetch already fills the 64-bit element.
    assert_invalid(0xC5AF14C1); // ld1d-shaped, but op==0 (signed), uxtw #3
    assert_invalid(0xC5D12DFD); // ldff1d-shaped, op==1 (signed+ff), sxtw
    // Also the vector+immediate and packed-offset signed-dword slots.
    assert_invalid(0xC5A084C1); // vector+imm, op4 (signed)
    assert_invalid(0xC5E084C1); // packed lsl, op4 (signed)
    assert_invalid(0xC5A004C1); // LDNT1 (region 00), op4 -> would be ldnt1sd
}

#[test]
fn j2_gather64_valid_forms_preserved() {
    // The unsigned dword gather (op2/op3 unpacked, op6/op7 packed) stays valid.
    assert_eq!(
        text(0xC5AF54C1),
        "ld1d    {z1.d}, p5/z, [x6, z15.d, uxtw #0x3]"
    );
    assert_eq!(text(0xC5CFD4C1), "ld1d    {z1.d}, p5/z, [x6, z15.d]");
    assert_roundtrip(0xC5AF54C1);
    assert_roundtrip(0xC5CFD4C1);
    // A signed *word* gather (msz==2) is unaffected (sign-extends 4→8 bytes).
    assert_eq!(
        text(0xC52F14C1),
        "ld1sw   {z1.d}, p5/z, [x6, z15.d, uxtw #0x2]"
    );
    assert_roundtrip(0xC52F14C1);
}

// ---------------------------------------------------------------------------
// Item 3: SVE predicated CPY-immediate (`MOV`) `.b` shift reserved.
// ---------------------------------------------------------------------------

#[test]
fn j3_cpy_imm_byte_shift_reserved() {
    // `LSL #8` (`sh == 1`) cannot apply to a `.b` element → UNDEFINED.
    assert_invalid(0x05156075); // mov z21.b, p5/m, #imm, lsl #8 — reserved
    assert_invalid(0x05152075); // /z variant, .b + shift
    // The non-shifted `.b` MOV-imm stays valid.
    assert_eq!(text(0x05154075), "mov     z21.b, p5/m, #0x3");
    assert_roundtrip(0x05154075);
}

#[test]
fn j3_cpy_imm_shifted_other_sizes_preserved() {
    // `.h`/`.s`/`.d` accept the `LSL #8` shift (imm rendered already-shifted).
    assert_eq!(text(0x05556075), "mov     z21.h, p5/m, #0x300");
    assert_eq!(text(0x05D52055), "mov     z21.d, p5/z, #0x200");
    assert_roundtrip(0x05556075);
    assert_roundtrip(0x05D52055);
}

#[test]
fn j3_zip_requires_bit21() {
    // The ZIP/UZP/TRN permute leaf fixes `word<21> == 1`; the `<21> == 0` slot
    // (a reserved CPY-imm) must not be mis-claimed as `ZIP1`.
    assert_invalid(0x05156075); // was mis-decoding to `zip1 z21.b, z3.b, z21.b`
    // A genuine ZIP1 (`word<21> == 1`) still decodes.
    assert_eq!(text(0x053E617B), "zip1    z27.b, z11.b, z30.b");
    assert_roundtrip(0x053E617B);
}

// ---------------------------------------------------------------------------
// Item 4: SVE EXT immediate — `word<23:21>` must be `001`/`011`.
// ---------------------------------------------------------------------------

#[test]
fn j4_ext_reserved_slots() {
    // `word<23> == 1` is reserved (was mis-decoding to `ext …, #0x7d`).
    assert_invalid(0x05AF14C1);
    // `word<21> == 0` is reserved (was mis-decoding to `ext …, #0x1`).
    assert_invalid(0x050007E1);
}

#[test]
fn j4_ext_valid_forms_preserved() {
    // Destructive EXT (`word<23:21> == 001`).
    assert_eq!(text(0x052F14C1), "ext     z1.b, z1.b, z6.b, #0x7d");
    assert_eq!(text(0x052007E1), "ext     z1.b, z1.b, z31.b, #0x1");
    assert_roundtrip(0x052F14C1);
    assert_roundtrip(0x052007E1);
    // Constructive EXT (`word<23:21> == 011`); fARM64 follows the binja corpus,
    // which renders a spurious `z0.b` between the list and the immediate.
    assert_eq!(text(0x056F14C1), "ext     z1.b, {z6.b, z7.b}, z0.b, #0x7d");
    assert_roundtrip(0x056F14C1);
}
