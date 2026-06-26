//! O batch — SVE2.2 multi-vector FP-to-int convert-narrow (`FCVTZSN`/`FCVTZUN`).
//!
//! A single half-width-element destination `Zd.<Tn>` from a consecutive
//! 2-register source group `{ Zn.<T>, Zn+1.<T> }` whose first register is even.
//! Top byte 0x65, `<21>=0`, `<15:13>=001`, `<20:16>=01101`, `<12:11>=10`;
//! `<10>` selects signed (`FCVTZSN`, 0) / unsigned (`FCVTZUN`, 1). `<23:22>`
//! size: `01`=.b<-{.h}, `10`=.h<-{.s}, `11`=.s<-{.d} (`00` reserved).
//!
//! All canonical example words are LLVM (`clang`/`llvm-objdump --mattr=+all`)
//! oracle encodings; the reserved words below are `<unknown>` in LLVM.

#![cfg(feature = "std")]

use fARM64::decode::decode;
use fARM64::format::{BufSink, FmtFormatter, Formatter};
use fARM64::{encode, Feature, FeatureSet};

/// Render `word` to its textual disassembly.
fn text(word: u32) -> String {
    let insn = decode(word, 0x1000, FeatureSet::ALL);
    let mut buf = [0u8; 160];
    let mut sink = BufSink::new(&mut buf);
    FmtFormatter::new().format(&insn, &mut sink);
    sink.as_str().to_string()
}

/// Decode `word`, re-encode, require an identical word; then re-decode and
/// require mnemonic + operand stability.
fn assert_roundtrip(word: u32) {
    let insn = decode(word, 0x1000, FeatureSet::ALL);
    assert!(!insn.is_invalid(), "{word:08X} decoded Invalid");
    let enc = encode(&insn)
        .unwrap_or_else(|e| panic!("{word:08X} ({}) encode error {e:?}", insn.mnemonic().name()));
    assert_eq!(enc, word, "{word:08X} ({}) re-encoded to {enc:08X}", insn.mnemonic().name());
    let insn2 = decode(enc, 0x1000, FeatureSet::ALL);
    assert_eq!(insn.mnemonic(), insn2.mnemonic(), "{word:08X} mnemonic drift");
    assert_eq!(insn.op_count(), insn2.op_count(), "{word:08X} operand-count drift");
}

/// `(word, expected disassembly)` pairs — the LLVM oracle renderings.
const CASES: &[(u32, &str)] = &[
    // .h <- { .s, .s } (size==10), the canonical task examples.
    (0x658D3113, "fcvtzsn z19.h, { z8.s, z9.s }"),
    (0x658D3513, "fcvtzun z19.h, { z8.s, z9.s }"),
    // .b <- { .h, .h } (size==01).
    (0x654D3113, "fcvtzsn z19.b, { z8.h, z9.h }"),
    (0x654D3513, "fcvtzun z19.b, { z8.h, z9.h }"),
    // .s <- { .d, .d } (size==11).
    (0x65CD3113, "fcvtzsn z19.s, { z8.d, z9.d }"),
    (0x65CD3513, "fcvtzun z19.s, { z8.d, z9.d }"),
    // Source group base z0 / z16, alternate destination register.
    (0x658D3013, "fcvtzsn z19.h, { z0.s, z1.s }"),
    (0x658D3505, "fcvtzun z5.h, { z8.s, z9.s }"),
    (0x658D3213, "fcvtzsn z19.h, { z16.s, z17.s }"),
];

#[test]
fn examples_decode_and_render() {
    for &(w, expected) in CASES {
        assert_eq!(text(w), expected, "{w:08X} rendering");
    }
}

#[test]
fn examples_roundtrip() {
    for &(w, _) in CASES {
        assert_roundtrip(w);
    }
}

/// Exhaustive round-trip over the whole family: all three sizes, both signs,
/// every even source-group base and every destination register.
#[test]
fn family_roundtrip_exhaustive() {
    for size in [0b01u32, 0b10, 0b11] {
        for sign in [0u32, 1] {
            for zn in (0u32..32).step_by(2) {
                for zd in 0u32..32 {
                    let w = (0b01100101u32 << 24)
                        | (size << 22)
                        | (0b01101 << 16)
                        | (0b001 << 13)
                        | (0b10 << 11)
                        | (sign << 10)
                        | (zn << 5)
                        | zd;
                    assert_roundtrip(w);
                }
            }
        }
    }
}

/// Reserved neighbours must stay `Invalid` (all `<unknown>` in LLVM).
#[test]
fn reserved_neighbours_invalid() {
    let reserved = [
        // size==00 (the `.b`-below element does not exist) for both signs.
        0x650D3113u32,
        0x650D3513,
        // odd source-group base (z9) across all three sizes.
        0x654D3133,
        0x658D3133,
        0x65CD3133,
        // <12:11> != 10 (the fixed opcode tail): 00 / 01 / 11.
        0x658D2113,
        0x658D2913,
        0x658D3913,
    ];
    for &w in &reserved {
        assert!(decode(w, 0, FeatureSet::ALL).is_invalid(), "{w:08X} must be Invalid");
    }
}

/// FEAT_SVE2p2 gating: every family word needs `Sve2p2` and is otherwise
/// `Invalid` (with the rest of SVE enabled).
#[test]
fn feature_gated_on_sve2p2() {
    let no = FeatureSet::BASE
        .with(Feature::Sve)
        .with(Feature::Sve2p1)
        .with(Feature::Fp16);
    let yes = no.with(Feature::Sve2p2);
    for &(w, _) in CASES {
        assert!(decode(w, 0, no).is_invalid(), "{w:08X} must need Sve2p2");
        assert!(!decode(w, 0, yes).is_invalid(), "{w:08X} should decode with Sve2p2");
    }
}
