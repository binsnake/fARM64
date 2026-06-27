//! Apple IMPLEMENTATION-DEFINED instructions: AMX matrix coprocessor + GXF.
//!
//! These are **not** Arm-architectural and are **not** decodable by LLVM, so
//! there is no oracle: the example words are derived from the reverse-engineered
//! encodings (corsix/amx for AMX; Apple-silicon research for `GENTER`/`GEXIT`).
//!
//! * AMX: `0x0020_1000 | (op << 5) | operand`, `op` in word<9:5> (0..=22),
//!   `operand` in word<4:0> (an `Xn`, or the set/clr selector for op 17).
//! * GXF: `GEXIT` = `0x0020_1400`; `GENTER #imm5` = `0x0020_1420 | imm5`.
//!
//! Both are gated by a runtime feature (`AppleAmx` / `Gxf`); with neither
//! enabled the words stay invalid so Arm-only decoding is never perturbed.

#![cfg(feature = "std")]

use fARM64::decode::decode;
use fARM64::format::{BufSink, FmtFormatter, Formatter};
use fARM64::{encode, Feature, FeatureSet, Mnemonic};

const ADDR: u64 = 0x8000_0000_0000_0004;

/// Render `word` (with everything enabled) to its disassembly text.
fn text(word: u32) -> String {
    let insn = decode(word, ADDR, FeatureSet::ALL);
    let mut buf = [0u8; 160];
    let mut sink = BufSink::new(&mut buf);
    FmtFormatter::new().format(&insn, &mut sink);
    sink.as_str().to_string()
}

#[track_caller]
fn assert_dis(word: u32, expected: &str) {
    assert_eq!(text(word), expected, "word={word:#010x}");
}

/// Decode (all features), re-encode, require the identical word, and re-decode
/// to confirm mnemonic/op-count stability.
#[track_caller]
fn rt(word: u32) {
    let insn = decode(word, ADDR, FeatureSet::ALL);
    assert!(!insn.is_invalid(), "{word:08X} decoded Invalid");
    let enc = encode(&insn)
        .unwrap_or_else(|e| panic!("{word:08X} ({}) encode {e:?}", insn.mnemonic().name()));
    assert_eq!(enc, word, "{word:08X} ({}) re-encoded {enc:08X}", insn.mnemonic().name());
    let insn2 = decode(enc, ADDR, FeatureSet::ALL);
    assert_eq!(insn.mnemonic(), insn2.mnemonic(), "{word:08X} mnemonic drift");
    assert_eq!(insn.op_count(), insn2.op_count(), "{word:08X} op-count drift");
}

#[test]
fn amx_rendering() {
    // op 0..=16, 18..=22 take an Xn; op 17 is the operand-less set/clr.
    assert_dis(0x0020_1000, "ldx     x0"); // op 0, x0
    assert_dis(0x0020_1003, "ldx     x3"); // op 0, x3
    assert_dis(0x0020_1020, "ldy     x0"); // op 1
    assert_dis(0x0020_1043, "stx     x3"); // op 2, x3
    assert_dis(0x0020_1060, "sty     x0"); // op 3
    assert_dis(0x0020_1080, "ldz     x0"); // op 4
    assert_dis(0x0020_10a0, "stz     x0"); // op 5
    assert_dis(0x0020_10c5, "ldzi    x5"); // op 6, x5
    assert_dis(0x0020_10e0, "stzi    x0"); // op 7
    assert_dis(0x0020_1100, "extrx   x0"); // op 8
    assert_dis(0x0020_1120, "extry   x0"); // op 9
    assert_dis(0x0020_1140, "fma64   x0"); // op 10
    assert_dis(0x0020_1160, "fms64   x0"); // op 11
    assert_dis(0x0020_1180, "fma32   x0"); // op 12
    assert_dis(0x0020_11a0, "fms32   x0"); // op 13
    assert_dis(0x0020_11c0, "mac16   x0"); // op 14
    assert_dis(0x0020_11e0, "fma16   x0"); // op 15
    assert_dis(0x0020_1200, "fms16   x0"); // op 16
    assert_dis(0x0020_1220, "set"); //        op 17, operand 0
    assert_dis(0x0020_1221, "clr"); //        op 17, operand 1
    assert_dis(0x0020_1240, "vecint  x0"); // op 18
    assert_dis(0x0020_1260, "vecfp   x0"); // op 19
    assert_dis(0x0020_1280, "matint  x0"); // op 20
    assert_dis(0x0020_12a0, "matfp   x0"); // op 21
    assert_dis(0x0020_12c0, "genlut  x0"); // op 22
    // Register 31 renders as xzr.
    assert_dis(0x0020_101f, "ldx     xzr");
}

#[test]
fn amx_mnemonics() {
    let f = FeatureSet::BASE.with(Feature::AppleAmx);
    assert_eq!(decode(0x0020_1003, ADDR, f).mnemonic(), Mnemonic::AmxLdx);
    assert_eq!(decode(0x0020_11c0, ADDR, f).mnemonic(), Mnemonic::AmxMac16);
    assert_eq!(decode(0x0020_1220, ADDR, f).mnemonic(), Mnemonic::AmxSet);
    assert_eq!(decode(0x0020_1221, ADDR, f).mnemonic(), Mnemonic::AmxClr);
    assert_eq!(decode(0x0020_12c0, ADDR, f).mnemonic(), Mnemonic::AmxGenlut);
}

#[test]
fn amx_roundtrip() {
    for op in 0u32..=22 {
        if op == 17 {
            rt(0x0020_1000 | (17 << 5)); // set
            rt(0x0020_1000 | (17 << 5) | 1); // clr
            continue;
        }
        // A couple of operand registers per op.
        rt(0x0020_1000 | (op << 5));
        rt(0x0020_1000 | (op << 5) | 7);
        rt(0x0020_1000 | (op << 5) | 31);
    }
}

#[test]
fn gxf_rendering_and_roundtrip() {
    assert_dis(0x0020_1400, "gexit");
    assert_dis(0x0020_1420, "genter  #0x0");
    assert_dis(0x0020_1425, "genter  #0x5");
    assert_dis(0x0020_143f, "genter  #0x1f");
    rt(0x0020_1400);
    rt(0x0020_1420);
    rt(0x0020_1425);
    rt(0x0020_143f);
}

#[test]
fn gated_off_by_default() {
    // Neither feature in the base set: the whole 0x0020_10xx/14xx region is
    // invalid, exactly as before these were added.
    assert!(decode(0x0020_1003, ADDR, FeatureSet::BASE).is_invalid());
    assert!(decode(0x0020_1220, ADDR, FeatureSet::BASE).is_invalid());
    assert!(decode(0x0020_1400, ADDR, FeatureSet::BASE).is_invalid());
    assert!(decode(0x0020_1420, ADDR, FeatureSet::BASE).is_invalid());

    // AMX and GXF are independently gated: enabling one must not surface the
    // other.
    let amx = FeatureSet::BASE.with(Feature::AppleAmx);
    assert_eq!(decode(0x0020_1003, ADDR, amx).mnemonic(), Mnemonic::AmxLdx);
    assert!(decode(0x0020_1400, ADDR, amx).is_invalid()); // GXF needs Feature::Gxf
    assert!(decode(0x0020_1420, ADDR, amx).is_invalid());

    let gxf = FeatureSet::BASE.with(Feature::Gxf);
    assert_eq!(decode(0x0020_1400, ADDR, gxf).mnemonic(), Mnemonic::Gexit);
    assert!(decode(0x0020_1003, ADDR, gxf).is_invalid()); // AMX needs Feature::AppleAmx
}

#[test]
fn reserved_forms_stay_invalid() {
    // op 23..=31 are unallocated even with AMX enabled.
    for op in 23u32..=31 {
        let w = 0x0020_1000 | (op << 5);
        assert!(decode(w, ADDR, FeatureSet::ALL).is_invalid(), "op {op} should be invalid ({w:#x})");
    }
    // op 17 with an operand other than 0/1 is not architected.
    assert!(decode(0x0020_1222, ADDR, FeatureSet::ALL).is_invalid());
    assert!(decode(0x0020_123f, ADDR, FeatureSet::ALL).is_invalid());
    // GXF: gexit must have a zero operand; sub-opcodes >= 2 are unallocated.
    assert!(decode(0x0020_1401, ADDR, FeatureSet::ALL).is_invalid());
    assert!(decode(0x0020_1440, ADDR, FeatureSet::ALL).is_invalid()); // sub 2
    assert!(decode(0x0020_1460, ADDR, FeatureSet::ALL).is_invalid()); // sub 3
    // bit<11> set leaves the Apple cluster entirely (0x0020_1800+).
    assert!(decode(0x0020_1800, ADDR, FeatureSet::ALL).is_invalid());
}
