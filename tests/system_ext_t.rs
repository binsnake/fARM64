//! T: AArch64 system-region named-instruction completion — decode + encode.
//!
//! Covers the `SYS`/`SYSL` alias families that fARM64 previously rendered as the
//! generic `sys`/`sysl`: the newer `TLBI` ops (outer-shareable `*OS`, non-XS
//! `*NXS`, range `RVA*`, GPT `RPA*`/`PAALL`), the newer `DC` ops (FEAT_OCCMO
//! `CVAOC`, FEAT_MEC `CIPAPA`, the GPT `*PAPS`/`GVA`/`GZVA`...), the FEAT_ATS1A
//! `AT S1E*A`, and the brand-new `PLBI`/`GIC`/`GICR`/`MLBI`/`APAS`/`TRCIT`/
//! `COSP`/`GSB`/`BRB` families. Plus the FEAT_GCS stack ops
//! (`GCSPUSHM`/`GCSPOPM`/`GCSSS1`/`GCSSS2`/`GCSPUSHX`/`GCSPOPX`/`GCSPOPCX`), the
//! `GCSB DSYNC`/`SHUH`/`STSHH`/`STCPH`/`CHKFEAT`/`DGH`/`CLRBHB`/`PACM` hints, the
//! `XAFLAG`/`AXFLAG` PSTATE ops, the `TENTER` guarded entry, and the
//! `GCSSTR`/`GCSSTTR` stores.
//!
//! The canonical example words and operand spellings are LLVM
//! (`clang`/`llvm-objdump --mattr=+all`) oracle output.

#![cfg(feature = "std")]

use fARM64::decode::decode;
use fARM64::format::{BufSink, FmtFormatter, Formatter};
use fARM64::{encode, Feature, FeatureSet};

const ADDR: u64 = 0x8000_0000_0000_0004;

/// Render `word` to its disassembly text under the full feature set.
fn text(word: u32) -> String {
    let insn = decode(word, ADDR, FeatureSet::ALL);
    let mut buf = [0u8; 160];
    let mut sink = BufSink::new(&mut buf);
    FmtFormatter::new().format(&insn, &mut sink);
    sink.as_str().to_string()
}

/// Assert the disassembly text of `word` equals `expected`.
#[track_caller]
fn assert_dis(word: u32, expected: &str) {
    assert_eq!(text(word), expected, "word={word:#010x}");
}

/// Decode `word`, re-encode, require the identical word; re-decode for stability.
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
fn named_examples() {
    // TLBI newer ops.
    assert_dis(0xD50C8407, "tlbi    ipas2e1os, x7");
    assert_dis(0xD5088220, "tlbi    rvae1is, x0"); // range
    assert_dis(0xD508911F, "tlbi    vmalle1osnxs"); // CRn==9 NXS, whole-TLB
    assert_dis(0xD50E819F, "tlbi    paallos"); // GPT whole-structure, no Xt
    // DC newer ops (OCCMO/MEC).
    assert_dis(0xD50B7B12, "dc      cvaoc, x18");
    assert_dis(0xD50B7FE0, "dc      cigdvaoc, x0");
    assert_dis(0xD50C7E00, "dc      cipae, x0");
    // AT FEAT_ATS1A.
    assert_dis(0xD508795E, "at      s1e1a, x30");
    // New families.
    assert_dis(0xD508A94A, "plbi    aside1osnxs, x10");
    assert_dis(0xD50CC13A, "gic     vden, x26");
    assert_dis(0xD528C31E, "gicr    x30, cdia");
    assert_dis(0xD50C70E5, "mlbi    vpmge1, x5");
    assert_dis(0xD50E7014, "apas    x20");
    assert_dis(0xD50B72E0, "trcit   x0");
    assert_dis(0xD50B73C0, "cosp    rctx, x0");
    assert_dis(0xD508C01F, "gsb     sys");
    assert_dis(0xD509729F, "brb     iall");
}

#[test]
fn gcs_and_hint_and_singletons() {
    // GCS stack ops.
    assert_dis(0xD50B7700, "gcspushm x0");
    assert_dis(0xD52B773F, "gcspopm"); // Rt==31 elided
    assert_dis(0xD50B7740, "gcsss1  x0");
    assert_dis(0xD52B7760, "gcsss2  x0");
    assert_dis(0xD508779F, "gcspushx");
    assert_dis(0xD50877DF, "gcspopx");
    assert_dis(0xD50877BF, "gcspopcx");
    // Hints.
    assert_dis(0xD50320DF, "dgh");
    assert_dis(0xD503227F, "gcsb    dsync");
    assert_dis(0xD50322DF, "clrbhb");
    assert_dis(0xD50324FF, "pacm");
    assert_dis(0xD503251F, "chkfeat x16");
    assert_dis(0xD503265F, "shuh");
    assert_dis(0xD503267F, "shuh    ph");
    assert_dis(0xD503261F, "stshh   keep");
    assert_dis(0xD503263F, "stshh   strm");
    assert_dis(0xD503269F, "stcph");
    // PSTATE XAFLAG/AXFLAG (CRm is a don't-care, matching LLVM).
    assert_dis(0xD500403F, "xaflag");
    assert_dis(0xD500405F, "axflag");
    assert_dis(0xD500453F, "xaflag"); // non-zero CRm still names it
    // TENTER (0xD4 exception group).
    assert_dis(0xD4E004C0, "tenter  #0x26");
    // GCS stores.
    assert_dis(0xD91F0C20, "gcsstr  x0, [x1]");
    assert_dis(0xD91F1C20, "gcssttr x0, [x1]");
}

#[test]
fn round_trip_named_table() {
    // Every named SYS/SYSL alias (canonical Rt) re-encodes bit-exactly.
    for &w in NAMED_WORDS {
        rt(w);
    }
}

#[test]
fn round_trip_gcs_hint_singletons() {
    for &w in &[
        // GCS stack ops (Rt==0 for the *M/SS forms; *X forms have no Rt).
        0xD50B7700u32, 0xD50B7740, 0xD52B7760, 0xD508779F, 0xD50877DF, 0xD50877BF, 0xD52B7720,
        // Hints.
        0xD50320DF, 0xD503227F, 0xD50322DF, 0xD50324FF, 0xD503251F, 0xD503265F, 0xD503267F,
        0xD503261F, 0xD503263F, 0xD503269F,
        // PSTATE bare ops (canonical CRm==0).
        0xD500403F, 0xD500405F,
        // TENTER.
        0xD4E004C0,
        // GCS stores.
        0xD91F0C20, 0xD91F1C20,
    ] {
        rt(w);
    }
}

#[test]
fn feature_gating() {
    // GCS stack ops, the GCSB hint, TENTER, and the GCS stores are gated on
    // FEAT_GCS — without it they are not the named instruction.
    assert!(decode(0xD50B7700, ADDR, FeatureSet::BASE).mnemonic() != fARM64::Mnemonic::Gcspushm);
    assert!(decode(0xD91F0C20, ADDR, FeatureSet::BASE).is_invalid());
    assert!(decode(0xD4E004C0, ADDR, FeatureSet::BASE).is_invalid());
    // With FEAT_GCS they decode.
    let g = FeatureSet::BASE.with(Feature::Gcs);
    assert_eq!(decode(0xD50B7700, ADDR, g).mnemonic(), fARM64::Mnemonic::Gcspushm);
    assert_eq!(decode(0xD91F0C20, ADDR, g).mnemonic(), fARM64::Mnemonic::Gcsstr);
    // TENTER is gated on FEAT_TEV (not GCS).
    assert!(decode(0xD4E004C0, ADDR, g).is_invalid());
    let tev = FeatureSet::BASE.with(Feature::Tev);
    assert_eq!(decode(0xD4E004C0, ADDR, tev).mnemonic(), fARM64::Mnemonic::Tenter);
    // CHKFEAT is gated on FEAT_CHK, SHUH on FEAT_PCDPHINT.
    assert_eq!(decode(0xD503251F, ADDR, FeatureSet::BASE).mnemonic(), fARM64::Mnemonic::Hint);
    assert_eq!(
        decode(0xD503251F, ADDR, FeatureSet::BASE.with(Feature::Chk)).mnemonic(),
        fARM64::Mnemonic::Chkfeat
    );
    // The SYS/SYSL alias families are not feature-gated (they are renamings of the
    // base SYS encoding, matching LLVM and the existing IC/DC/AT/TLBI behaviour).
    assert_eq!(decode(0xD508A94A, ADDR, FeatureSet::BASE).mnemonic(), fARM64::Mnemonic::Plbi);
}

#[test]
fn never_panics_system_space() {
    // The whole D4xx/D5xx space must stay total and never desync operands.
    for hi in [0xD4u32, 0xD5] {
        for lo in 0..=0xffffu32 {
            let w = (hi << 24) | lo;
            let _ = decode(w, ADDR, FeatureSet::ALL);
            let _ = decode(w, ADDR, FeatureSet::BASE);
        }
    }
}

/// Canonical words for every named SYS/SYSL alias in the directory.
const NAMED_WORDS: &[u32] = &[
        0xD508711F, 0xD508751F, 0xD5087620, 0xD5087640, 0xD5087660, 0xD5087680, 0xD50876A0, 0xD50876C0,
        0xD5087800, 0xD5087820, 0xD5087840, 0xD5087860, 0xD5087900, 0xD5087920, 0xD5087940, 0xD5087A40,
        0xD5087A80, 0xD5087AC0, 0xD5087E40, 0xD5087E80, 0xD5087EC0, 0xD5087F20, 0xD5087FA0, 0xD508811F,
        0xD5088120, 0xD5088140, 0xD5088160, 0xD50881A0, 0xD50881E0, 0xD5088220, 0xD5088260, 0xD50882A0,
        0xD50882E0, 0xD508831F, 0xD5088320, 0xD5088340, 0xD5088360, 0xD50883A0, 0xD50883E0, 0xD5088520,
        0xD5088560, 0xD50885A0, 0xD50885E0, 0xD5088620, 0xD5088660, 0xD50886A0, 0xD50886E0, 0xD508871F,
        0xD5088720, 0xD5088740, 0xD5088760, 0xD50887A0, 0xD50887E0, 0xD508911F, 0xD5089120, 0xD5089140,
        0xD5089160, 0xD50891A0, 0xD50891E0, 0xD5089220, 0xD5089260, 0xD50892A0, 0xD50892E0, 0xD508931F,
        0xD5089320, 0xD5089340, 0xD5089360, 0xD50893A0, 0xD50893E0, 0xD5089520, 0xD5089560, 0xD50895A0,
        0xD50895E0, 0xD5089620, 0xD5089660, 0xD50896A0, 0xD50896E0, 0xD508971F, 0xD5089720, 0xD5089740,
        0xD5089760, 0xD50897A0, 0xD50897E0, 0xD508A11F, 0xD508A120, 0xD508A140, 0xD508A160, 0xD508A31F,
        0xD508A320, 0xD508A340, 0xD508A360, 0xD508A71F, 0xD508A720, 0xD508A740, 0xD508A760, 0xD508A91F,
        0xD508A920, 0xD508A940, 0xD508A960, 0xD508AB1F, 0xD508AB20, 0xD508AB40, 0xD508AB60, 0xD508AF1F,
        0xD508AF20, 0xD508AF40, 0xD508AF60, 0xD508C100, 0xD508C120, 0xD508C140, 0xD508C160, 0xD508C180,
        0xD508C1A0, 0xD508C1FF, 0xD508C200, 0xD508C220, 0xD528C300, 0xD528C320, 0xD50B72E0, 0xD50B7380,
        0xD50B73A0, 0xD50B73C0, 0xD50B73E0, 0xD50B7420, 0xD50B7460, 0xD50B7480, 0xD50B74A0, 0xD50B74E0,
        0xD50B7520, 0xD50B7A20, 0xD50B7A60, 0xD50B7AA0, 0xD50B7B00, 0xD50B7B20, 0xD50B7BE0, 0xD50B7C20,
        0xD50B7C60, 0xD50B7CA0, 0xD50B7D20, 0xD50B7D60, 0xD50B7DA0, 0xD50B7E20, 0xD50B7E60, 0xD50B7EA0,
        0xD50B7F00, 0xD50B7FE0, 0xD50C709F, 0xD50C70BF, 0xD50C70C0, 0xD50C70E0, 0xD50C7800, 0xD50C7820,
        0xD50C7880, 0xD50C78A0, 0xD50C78C0, 0xD50C78E0, 0xD50C7940, 0xD50C7E00, 0xD50C7EE0, 0xD50C8020,
        0xD50C8040, 0xD50C80A0, 0xD50C80C0, 0xD50C811F, 0xD50C8120, 0xD50C819F, 0xD50C81A0, 0xD50C81DF,
        0xD50C8220, 0xD50C825F, 0xD50C82A0, 0xD50C831F, 0xD50C8320, 0xD50C839F, 0xD50C83A0, 0xD50C83DF,
        0xD50C8400, 0xD50C8420, 0xD50C8440, 0xD50C8460, 0xD50C8480, 0xD50C84A0, 0xD50C84C0, 0xD50C84E0,
        0xD50C8520, 0xD50C855F, 0xD50C85A0, 0xD50C8620, 0xD50C865F, 0xD50C86A0, 0xD50C871F, 0xD50C8720,
        0xD50C879F, 0xD50C87A0, 0xD50C87DF, 0xD50C9020, 0xD50C9040, 0xD50C90A0, 0xD50C90C0, 0xD50C911F,
        0xD50C9120, 0xD50C919F, 0xD50C91A0, 0xD50C91DF, 0xD50C9220, 0xD50C925F, 0xD50C92A0, 0xD50C931F,
        0xD50C9320, 0xD50C939F, 0xD50C93A0, 0xD50C93DF, 0xD50C9400, 0xD50C9420, 0xD50C9440, 0xD50C9460,
        0xD50C9480, 0xD50C94A0, 0xD50C94C0, 0xD50C94E0, 0xD50C9520, 0xD50C955F, 0xD50C95A0, 0xD50C9620,
        0xD50C965F, 0xD50C96A0, 0xD50C971F, 0xD50C9720, 0xD50C979F, 0xD50C97A0, 0xD50C97DF, 0xD50CA11F,
        0xD50CA120, 0xD50CA19F, 0xD50CA31F, 0xD50CA320, 0xD50CA39F, 0xD50CA71F, 0xD50CA720, 0xD50CA79F,
        0xD50CA91F, 0xD50CA920, 0xD50CA99F, 0xD50CAB1F, 0xD50CAB20, 0xD50CAB9F, 0xD50CAF1F, 0xD50CAF20,
        0xD50CAF9F, 0xD50CC100, 0xD50CC120, 0xD50CC140, 0xD50CC160, 0xD50CC180, 0xD50CC1A0, 0xD50CC200,
        0xD50CC220, 0xD50E7000, 0xD50E7800, 0xD50E7820, 0xD50E7940, 0xD50E7E20, 0xD50E7EA0, 0xD50E811F,
        0xD50E8120, 0xD50E819F, 0xD50E81A0, 0xD50E8220, 0xD50E82A0, 0xD50E831F, 0xD50E8320, 0xD50E83A0,
        0xD50E8460, 0xD50E84E0, 0xD50E8520, 0xD50E85A0, 0xD50E8620, 0xD50E86A0, 0xD50E871F, 0xD50E8720,
        0xD50E879F, 0xD50E87A0, 0xD50E911F, 0xD50E9120, 0xD50E919F, 0xD50E91A0, 0xD50E9220, 0xD50E92A0,
        0xD50E931F, 0xD50E9320, 0xD50E93A0, 0xD50E9460, 0xD50E94E0, 0xD50E9520, 0xD50E95A0, 0xD50E9620,
        0xD50E96A0, 0xD50E971F, 0xD50E9720, 0xD50E979F, 0xD50E97A0, 0xD50EA11F, 0xD50EA120, 0xD50EA31F,
        0xD50EA320, 0xD50EA71F, 0xD50EA720, 0xD50EA91F, 0xD50EA920, 0xD50EAB1F, 0xD50EAB20, 0xD50EAF1F,
        0xD50EAF20, 0xD50EC100, 0xD50EC120, 0xD50EC140, 0xD50EC160, 0xD50EC180, 0xD50EC1A0, 0xD50EC200,
        0xD50EC220, 0xD508C01F, 0xD508C03F, 0xD509729F, 0xD50972BF,
];
