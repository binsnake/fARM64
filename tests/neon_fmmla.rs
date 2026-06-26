//! NEON FP/BF16/FP8 matrix multiply-accumulate (FMMLA / BFMMLA): decode + round-trip.
//!
//! The Advanced SIMD three-register matrix products in the `word<15:10>==111011`,
//! `Q==1` slot: FMMLA (FEAT_F16F32MM `.4s,.8h,.8h`; FEAT_F16MM `.8h,.8h,.8h`;
//! FEAT_F8F16MM `.8h,.16b,.16b`; FEAT_F8F32MM `.4s,.16b,.16b`) and BFMMLA
//! (FEAT_BF16 `.4s,.8h,.8h`). Canonical words are LLVM (`--mattr=+all`) encodings.

#![cfg(feature = "std")]

use fARM64::decode::decode;
use fARM64::{encode, Feature, FeatureSet};

fn rt(word: u32) {
    let insn = decode(word, 0, FeatureSet::ALL);
    assert!(!insn.is_invalid(), "{:08X} decoded Invalid", word);
    let enc = encode(&insn).unwrap_or_else(|e| panic!("{:08X} ({}) encode err {:?}", word, insn.mnemonic().name(), e));
    assert_eq!(enc, word, "{:08X} ({}) re-encoded {:08X}", word, insn.mnemonic().name(), enc);
}

#[test]
fn examples_decode_and_roundtrip() {
    let cases: &[(u32, &str)] = &[
        (0x4E42EC20, "fmmla"),  // fmmla v0.4s, v1.8h, v2.8h   (F16F32MM)
        (0x4EC2EC20, "fmmla"),  // fmmla v0.8h, v1.8h, v2.8h   (F16MM)
        (0x6E02EC20, "fmmla"),  // fmmla v0.8h, v1.16b, v2.16b (F8F16MM)
        (0x6E82EC20, "fmmla"),  // fmmla v0.4s, v1.16b, v2.16b (F8F32MM)
        (0x6E42EC20, "bfmmla"), // bfmmla v0.4s, v1.8h, v2.8h  (BF16)
    ];
    for &(w, m) in cases {
        let insn = decode(w, 0, FeatureSet::ALL);
        assert_eq!(insn.mnemonic().name(), m, "{:08X}", w);
        assert_eq!(insn.op_count(), 3, "{:08X}", w);
        rt(w);
    }
}

#[test]
fn unallocated_slots_are_invalid() {
    // (u,size) combos NOT in {(0,01),(0,11),(1,00),(1,01),(1,10)} are UNDEFINED.
    // Base word: lo=111011, Q=1, Rm=2, Rn=1, Rd=0.
    let mk = |u: u32, size: u32, q: u32| {
        (q << 30) | (u << 29) | (0b0_1110 << 24) | (size << 22) | (2 << 16) | (0b111011 << 10) | (1 << 5)
    };
    let undef = [(0u32, 0b00u32), (0, 0b10), (1, 0b11)];
    for (u, size) in undef {
        let w = mk(u, size, 1);
        assert!(decode(w, 0, FeatureSet::ALL).is_invalid(), "{:08X} (u={},size={:02b}) should be Invalid", w, u, size);
    }
    // Q==0 is UNDEFINED for every (u,size).
    for u in 0..2 {
        for size in 0..4 {
            let w = mk(u, size, 0);
            assert!(decode(w, 0, FeatureSet::ALL).is_invalid(), "{:08X} Q=0 should be Invalid", w);
        }
    }
}

#[test]
fn feature_gated() {
    let cases: &[(u32, Feature)] = &[
        (0x4E42EC20, Feature::F16f32mm),
        (0x4EC2EC20, Feature::F16mm),
        (0x6E02EC20, Feature::F8f16mm),
        (0x6E82EC20, Feature::F8f32mm),
        (0x6E42EC20, Feature::Bf16),
    ];
    for &(w, feat) in cases {
        let bit = feat as u32;
        let without = FeatureSet { features0: FeatureSet::ALL.features0 & !(1u64 << bit), features1: FeatureSet::ALL.features1 & !(1u64 << bit) };
        assert!(decode(w, 0, without).is_invalid(), "{:08X} should require {:?}", w, feat);
        assert!(!decode(w, 0, FeatureSet::ALL).is_invalid(), "{:08X} should decode with {:?}", w, feat);
    }
}
