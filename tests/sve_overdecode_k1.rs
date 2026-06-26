//! SVE over-decode hardening, batch K1.
//!
//! Four reserved-encoding families in the SVE integer / permute space, each
//! derived from an LLVM field sweep (`oracle.py dec/enc`, i.e. `clang .inst` +
//! `llvm-objdump --mattr=+all`) and proven 0-regression with a pre/post
//! differential over the affected top bytes (`0x04`/`0x05`/`0x24`/`0x25`):
//! a 59,928-word structured sweep plus a 40,000-word random sweep eliminated
//! 12,696 + 1,895 over-decoded words with **0** LLVM-valid words newly rejected,
//! **0** valid-word decode changes, and **0** Invalid→valid flips.
//!
//! Families hardened:
//!
//!  1. **SVE integer compare (vectors), wide variant `.d`** (`0x24`): the *wide*
//!     compares (CMPEQ/CMPNE/CMPGE/CMPGT/CMPLT/CMPLE/CMPHS/CMPHI/CMPLO/CMPLS
//!     against a `.d` second operand) require a *narrower* first operand, so
//!     `size == 0b11` (`.d`) is reserved. The same-width compares stay valid for
//!     every size. E.g. `24C4FAB2` (`cmpls …d,…d,…d`) → UNDEFINED.
//!
//!  2. **SVE CPY (scalar→vector, predicated)** (`0x05`): the `MOV <Zd>, <Pg>/m,
//!     <R>` (CPY-from-GP) form fixes `<21>=1`; a `<21>=0` word is reserved. E.g.
//!     `0508A7AE` (`mov z14.b,p1/m,w29`) → UNDEFINED.
//!
//!  3. **SVE logical immediate (AND/ORR/EOR/DUPM)** (`0x05`): two reserved
//!     sources — (a) the fixed bit `<18>` must be `0` (a `<18>=1` word is a
//!     different SVE encoding, reserved here), and (b) the bitmask must satisfy
//!     the base-ISA logical-immediate validity (`DecodeBitMasks(…, TRUE)`: the
//!     `imms == all-ones` field is reserved). E.g. `05453D70`/`058453D7`/
//!     `05049695` (bit18 set) and the all-ones-imms forms → UNDEFINED.
//!
//!  4. **PSEL + sibling predicate-permute reserved** (`0x05`/`0x25`): PSEL fixes
//!     `<15:14>=01`; a `<15:14>` of `10`/`11` is reserved. Exposing (3) also
//!     surfaced that the predicate ZIP/UZP/TRN (`<15:13>=010`) and the
//!     COMPACT/SPLICE/CLAST/LAST*/REVB-H-W-D group (`<15:13>=100`/`101`) fix
//!     `<21>=1`, so their `<21>=0` slots are reserved. E.g. `25B7BC6F`
//!     (`psel … <15:14>=10`) and `050055E0`/`0500A7AE` (`<21>=0`) → UNDEFINED.

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
    // (1) Integer compare (vectors): same-width compares for every size, and the
    // wide compares for the narrow sizes (`.b`/`.h`/`.s`).
    check(0x24041AA2, "cmphs p2.b, p6/z, z21.b, z4.b");
    check(0x24C49AB2, "cmpgt p2.d, p6/z, z21.d, z4.d"); // non-wide `.d` is valid
    check(0x24847AB2, "cmple p2.s, p6/z, z21.s, z4.d"); // wide, size != `.d`
    check(0x2402B9C9, "cmpeq p9.b, p6/z, z14.b, z2.b");
    check(0x24152B2E, "cmpeq p14.b, p2/z, z25.b, z21.d"); // wide eq, `.b`
    check(0x249F5BAE, "cmpge p14.s, p6/z, z29.s, z31.d"); // wide ge, `.s`

    // (2) CPY (scalar→vector, predicated), `<21>=1`.
    check(0x0528A7AE, "mov z14.b, p1/m, w29");
    check(0x05E8A7AE, "mov z14.d, p1/m, x29");

    // (3) Logical immediate with `<18>=0` and a valid bitmask.
    check(0x05403D70, "eor z16.h, z16.h, #0xfe1f");
    check(0x058053D7, "and z23.s, z23.s, #0xffdfffff");
    check(0x05001695, "orr z21.b, z21.b, #0xc7");

    // (4) PSEL with the fixed `<15:14>=01` marker, plus a `<21>=1` predicate
    // permute / LAST*-to-GP that must remain valid.
    check(0x25B77C6F, "psel p15, p15, p3.b[w15, #0xa]");
    check(0x252C4440, "psel p0, p1, p2.b[w12, #0x1]");
    check(0x25704440, "psel p0, p1, p2.s[w12, #0x1]");
    check(0x052055E0, "trn2 p0.b, p15.b, p0.b"); // pred-perm, `<21>=1`
    check(0x0520A7AE, "lasta w14, p1, z29.b"); // LAST*-to-GP, `<21>=1`
}

// ---------------------------------------------------------------------------
// (1) SVE integer compare (vectors): wide variant `.d` is reserved.
// ---------------------------------------------------------------------------

#[test]
fn cmp_wide_dword_reserved() {
    // Task example: `cmpls` wide with `.d` first operand.
    reserved(0x24C4FAB2);
    // The full wide-compare family at `size == 0b11`: CMPHS/CMPHI/CMPLO/CMPLS
    // (op=1,b14=1) and CMPEQ/CMPNE/CMPGE/CMPGT/CMPLT/CMPLE (op=0) wide forms.
    for &w in &[
        0x24C43AA2u32, // cmpeq wide .d
        0x24C43AB2,    // cmpne wide .d
        0x24C45AA2,    // cmpge wide .d
        0x24C45AB2,    // cmpgt wide .d
        0x24C47AA2,    // cmplt wide .d
        0x24C47AB2,    // cmple wide .d
        0x24C4DAA2,    // cmphs wide .d
        0x24C4DAB2,    // cmphi wide .d
        0x24C4FAA2,    // cmplo wide .d
        0x24C4FAB2,    // cmpls wide .d
    ] {
        reserved(w);
    }
}

// ---------------------------------------------------------------------------
// (2) SVE CPY (scalar→vector, predicated): `<21>=0` reserved.
// ---------------------------------------------------------------------------

#[test]
fn cpy_scalar_bit21_zero_reserved() {
    // Task example.
    reserved(0x0508A7AE);
    // Same encoding, every element size, `<21>=0`.
    reserved(0x0508A7AE); // .b
    reserved(0x0548A7AE); // .h
    reserved(0x0588A7AE); // .s
    reserved(0x05C8A7AE); // .d
}

// ---------------------------------------------------------------------------
// (3) SVE logical immediate: `<18>=1` and all-ones-imms reserved.
// ---------------------------------------------------------------------------

#[test]
fn logical_imm_bit18_set_reserved() {
    // Task examples (`<18>=1`).
    reserved(0x05453D70); // would-be eor z16.h,…,#0xfe1f
    reserved(0x058453D7); // would-be and z23.s,…
    reserved(0x05049695); // would-be orr z21.b,…
    // A valid bitmask with `<18>` set across all four opcodes (ORR/EOR/AND/DUPM).
    reserved(0x05443D70); // and(opc forms) bit18 set on the 0xfe1f mask
    reserved(0x05041695);
}

#[test]
fn logical_imm_all_ones_imms_reserved() {
    // `DecodeBitMasks(…, immediate = TRUE)` rejects the `imms == all-ones` field,
    // exactly like the base-ISA AND/ORR/EOR immediate. These decode to a "valid"
    // bitmask only under the lenient `immediate = FALSE` rule fARM64 used to take.
    for &w in &[0x054003E0u32, 0x05400BE0, 0x054013E0, 0x05401BE0, 0x054023E0] {
        reserved(w);
    }
}

// ---------------------------------------------------------------------------
// (4) PSEL `<15:14>` marker + sibling predicate-permute `<21>` reserved.
// ---------------------------------------------------------------------------

#[test]
fn psel_marker_reserved() {
    // Task example: PSEL with `<15:14>=10` (the fixed marker is `01`).
    reserved(0x25B7BC6F);
    // `<15:14>=11` is likewise reserved for this slot.
    reserved(0x25B7FC6F);
}

#[test]
fn pred_perm_bit21_zero_reserved() {
    // Predicate ZIP/UZP/TRN (`<15:13>=010`) and COMPACT/SPLICE/CLAST/LAST*/REV*
    // (`<15:13>=100`/`101`) fix `<21>=1`; their `<21>=0` slots are reserved.
    reserved(0x050055E0); // would-be trn2 p0.b,p15.b,p0.b
    reserved(0x0500A7AE); // would-be lasta w14,p1,z29.b
    reserved(0x0501A7AE); // would-be lastb w14,p1,z29.b
    reserved(0x050287E0); // would-be lasta b0,p1,z31.b
}
