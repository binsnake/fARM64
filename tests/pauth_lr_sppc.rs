//! FEAT_PAuth_LR: the no-offset `PACI*SPPC`/`PACNBI*SPPC` and the register-
//! modifier `AUTI*SPPCR`/`RETA*SPPCR <Xm>` forms.
//!
//! These eight encodings assemble in LLVM (oracle: `clang`/`llvm-objdump
//! --mattr=+all`) but were previously decoded `Invalid` by fARM64 — the sibling
//! PC-relative `*SPPC <label>` forms were present, these were the gap. Words and
//! spellings are the LLVM oracle output:
//!
//! ```text
//! paciasppc   = 0xDAC1A3FE   autiasppcr x2 = 0xDAC1905E   retaasppcr x2 = 0xD65F0BE2
//! pacibsppc   = 0xDAC1A7FE   autibsppcr x2 = 0xDAC1945E   retabsppcr x2 = 0xD65F0FE2
//! pacnbiasppc = 0xDAC183FE
//! pacnbibsppc = 0xDAC187FE
//! ```

#![cfg(feature = "std")]

use fARM64::decode::decode;
use fARM64::format::{BufSink, FmtFormatter, Formatter};
use fARM64::{encode, Feature, FeatureSet, Mnemonic};

const ADDR: u64 = 0x8000_0000_0000_0004;

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

#[track_caller]
fn rt(word: u32) {
    let insn = decode(word, ADDR, FeatureSet::ALL);
    assert!(!insn.is_invalid(), "{word:08X} decoded Invalid");
    let enc = encode(&insn)
        .unwrap_or_else(|e| panic!("{word:08X} ({}) encode {e:?}", insn.mnemonic().name()));
    assert_eq!(enc, word, "{word:08X} ({}) re-encoded {enc:08X}", insn.mnemonic().name());
}

#[test]
fn rendering() {
    assert_dis(0xDAC1A3FE, "paciasppc");
    assert_dis(0xDAC1A7FE, "pacibsppc");
    assert_dis(0xDAC183FE, "pacnbiasppc");
    assert_dis(0xDAC187FE, "pacnbibsppc");
    assert_dis(0xDAC1905E, "autiasppcr x2");
    assert_dis(0xDAC1945E, "autibsppcr x2");
    assert_dis(0xD65F0BE2, "retaasppcr x2");
    assert_dis(0xD65F0FE2, "retabsppcr x2");
}

#[test]
fn mnemonics() {
    assert_eq!(decode(0xDAC1A3FE, ADDR, FeatureSet::ALL).mnemonic(), Mnemonic::Paciasppc);
    assert_eq!(decode(0xDAC183FE, ADDR, FeatureSet::ALL).mnemonic(), Mnemonic::Pacnbiasppc);
    assert_eq!(decode(0xDAC1905E, ADDR, FeatureSet::ALL).mnemonic(), Mnemonic::Autiasppcr);
    assert_eq!(decode(0xD65F0BE2, ADDR, FeatureSet::ALL).mnemonic(), Mnemonic::Retaasppcr);
}

#[test]
fn roundtrip() {
    for w in [0xDAC1A3FEu32, 0xDAC1A7FE, 0xDAC183FE, 0xDAC187FE, 0xDAC1905E, 0xDAC1945E, 0xD65F0BE2, 0xD65F0FE2] {
        rt(w);
    }
    // The register-modifier forms across the whole Xm range round-trip.
    for xm in 0u32..=30 {
        rt(0xDAC1901E | (xm << 5)); // autiasppcr Xm
        rt(0xDAC1941E | (xm << 5)); // autibsppcr Xm
        rt(0xD65F0BE0 | xm); // retaasppcr Xm
        rt(0xD65F0FE0 | xm); // retabsppcr Xm
    }
}

#[test]
fn gated_on_pauth_lr() {
    // The base ISA (no FEAT_PAuth_LR) leaves all eight invalid.
    for w in [0xDAC1A3FEu32, 0xDAC183FE, 0xDAC1905E, 0xD65F0BE2, 0xD65F0FE2] {
        assert!(decode(w, ADDR, FeatureSet::BASE).is_invalid(), "{w:08X} should need FEAT_PAuth_LR");
    }
    // FEAT_PAuth_LR alone (no FEAT_PAuth) is enough — the new forms live in the
    // same encoding slots but are gated independently.
    let lr = FeatureSet::BASE.with(Feature::PauthLr);
    assert_eq!(decode(0xDAC1A3FE, ADDR, lr).mnemonic(), Mnemonic::Paciasppc);
    assert_eq!(decode(0xDAC1905E, ADDR, lr).mnemonic(), Mnemonic::Autiasppcr);
    assert_eq!(decode(0xD65F0BE2, ADDR, lr).mnemonic(), Mnemonic::Retaasppcr);
}

#[test]
fn reserved_neighbors_stay_invalid() {
    // PACI*SPPC require the implicit LR dest (Rd==30) and SP source (Rn==31).
    assert!(decode(0xDAC1A3FF, ADDR, FeatureSet::ALL).is_invalid()); // Rd==31, not 30
    assert!(decode(0xDAC1A35E, ADDR, FeatureSet::ALL).is_invalid()); // Rn==2, not 31 (paciasppc)
    // AUTI*SPPCR require the implicit LR dest (Rd==30).
    assert!(decode(0xDAC1905F, ADDR, FeatureSet::ALL).is_invalid()); // Rd==31, not 30
    // An unallocated opcode in the same opcode2==00001 slot stays invalid.
    assert!(decode(0xDAC1885E, ADDR, FeatureSet::ALL).is_invalid()); // opcode 100010
    // RETA*SPPCR require Rn==11111; the op4==11111 case is plain RETAA (not the R form).
    assert_eq!(decode(0xD65F0BFF, ADDR, FeatureSet::ALL).mnemonic(), Mnemonic::Retaa);
    assert_eq!(decode(0xD65F0FFF, ADDR, FeatureSet::ALL).mnemonic(), Mnemonic::Retab);
}
