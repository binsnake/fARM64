//! FEAT_LUT lookup-table reads — SVE `LUTI2` / `LUTI4`.
//!
//! These live in the SVE2 quadrant (top byte `0x45`, `word<31:29> = 0b010`) and
//! share the `<21>=1`, `<15:13>=0b101` skeleton. The family is selected by
//! `<10>` (0 = `LUTI2`, 1 = `LUTI4`) together with `<12>`/`<11>`; the element
//! arrangement (`.B`/`.H`) and the number of table registers fall out of the
//! same bits. The third operand is a *vector-element selector* `<Zm>[<index>]`
//! (a `Z` register with no arrangement suffix and a bracketed lane index); the
//! index field is interleaved across `<23:22>` and — for the `LUTI2 .H` form —
//! `<12>`, so its width grows with the element size:
//!
//! | Form | `<12>` | `<11>` | `<10>` | index bits (msb..lsb) |
//! |-|-|-|-|-|
//! | `LUTI2 .B`        | `1` | `0` | `0` | `<23>:<22>`            |
//! | `LUTI2 .H`        | idx | `1` | `0` | `<23>:<22>:<12>`       |
//! | `LUTI4 .B`        | `0` | `0` | `1` | `<23>` ( `<22>`=1 )    |
//! | `LUTI4 .H`        | `1` | `1` | `1` | `<23>:<22>`            |
//! | `LUTI4 .H` 2-table| `1` | `0` | `1` | `<23>:<22>`            |
//!
//! Transcribed from the *ARM Architecture Reference Manual* (FEAT_LUT) and
//! cross-checked against `llvm-mc --mattr=+all`. Total and panic-free; only the
//! exact valid encodings above are recognized, everything else is left
//! [`crate::mnemonic::Code::Invalid`] for the caller.

use crate::decode::bits::{bit, bits};
use crate::enums::VectorArrangement as VA;
use crate::features::{Feature, FeatureSet};
use crate::instruction::Instruction;
use crate::mnemonic::{Code, Mnemonic};
use crate::operand::Operand;
use crate::register::Register;

const Z: [Register; 32] = [
    Register::Z0, Register::Z1, Register::Z2, Register::Z3, Register::Z4, Register::Z5, Register::Z6, Register::Z7,
    Register::Z8, Register::Z9, Register::Z10, Register::Z11, Register::Z12, Register::Z13, Register::Z14, Register::Z15,
    Register::Z16, Register::Z17, Register::Z18, Register::Z19, Register::Z20, Register::Z21, Register::Z22, Register::Z23,
    Register::Z24, Register::Z25, Register::Z26, Register::Z27, Register::Z28, Register::Z29, Register::Z30, Register::Z31,
];

/// A scalable `Z{n}` operand with arrangement `a` (the destination).
#[inline]
fn zreg(n: u32, a: VA) -> Operand {
    Operand::Reg { reg: Z[(n & 0x1f) as usize], arr: Some(a), lane: None, shift: None, extend: None, pred: None }
}

/// A single-register Z list `{Z{n}.<T>}` (one table register).
#[inline]
fn zlist1(n: u32, a: VA) -> Operand {
    Operand::MultiReg { regs: [Z[(n & 0x1f) as usize], Register::None, Register::None, Register::None], count: 1, arr: Some(a), lane: None }
}

/// A two-register Z list `{Z{n}.<T>, Z{n+1}.<T>}` (two table registers).
#[inline]
fn zlist2(n: u32, a: VA) -> Operand {
    let n0 = (n & 0x1f) as usize;
    let n1 = ((n + 1) & 0x1f) as usize;
    Operand::MultiReg { regs: [Z[n0], Z[n1], Register::None, Register::None], count: 2, arr: Some(a), lane: None }
}

/// The vector-element selector `Z{m}[index]` (no arrangement suffix; the lane
/// index renders as `[index]`).
#[inline]
fn zidx(m: u32, index: u32) -> Operand {
    Operand::Reg { reg: Z[(m & 0x1f) as usize], arr: None, lane: Some(index as u8), shift: None, extend: None, pred: None }
}

/// Decode an SVE `LUTI2`/`LUTI4` (FEAT_LUT) lookup-table read into `out`.
///
/// Called as a fallback from [`super::decode`] for the `word<31:29> = 0b010`
/// quadrant. Recognizes only the FEAT_LUT skeleton (top byte `0x45`, `<21>=1`,
/// `<15:13>=0b101`); anything else is left untouched so the caller's prior
/// result (or `Invalid`) stands.
#[inline]
pub fn decode(word: u32, features: FeatureSet, out: &mut Instruction) {
    // Skeleton: top byte 0x45, <21>=1, <15:13>=0b101. (Note <15:13>=0b101 with
    // <12:10>=000 is HISTSEG, owned by the integer decoder — never reached here.)
    if bits(word, 24, 8) != 0b0100_0101 || bit(word, 21) != 1 || bits(word, 13, 3) != 0b101 {
        return;
    }
    if !features.has(Feature::Lut) {
        return;
    }

    let zm = bits(word, 16, 5);
    let zn = bits(word, 5, 5);
    let zd = bits(word, 0, 5);
    let b23 = bit(word, 23);
    let b22 = bit(word, 22);
    let b12 = bit(word, 12);

    if bit(word, 10) == 0 {
        // LUTI2: <11> selects element size (.B = 0, .H = 1).
        if bit(word, 11) == 0 {
            // LUTI2 .B: 2-bit index at <23:22>; <12> is a fixed 1.
            if b12 != 1 {
                return;
            }
            let index = (b23 << 1) | b22;
            out.set(Code::SveLuti2);
            out.set_mnemonic(Mnemonic::Luti2);
            out.push_operand(zreg(zd, VA::Sb));
            out.push_operand(zlist1(zn, VA::Sb));
            out.push_operand(zidx(zm, index));
        } else {
            // LUTI2 .H: 3-bit index = <23>:<22>:<12>.
            let index = (b23 << 2) | (b22 << 1) | b12;
            out.set(Code::SveLuti2);
            out.set_mnemonic(Mnemonic::Luti2);
            out.push_operand(zreg(zd, VA::Sh));
            out.push_operand(zlist1(zn, VA::Sh));
            out.push_operand(zidx(zm, index));
        }
    } else {
        // LUTI4: (<12>, <11>) select the element size / table count.
        match (b12, bit(word, 11)) {
            // .B single table: 1-bit index at <23>; <22> is a fixed 1.
            (0, 0) => {
                if b22 != 1 {
                    return;
                }
                let index = b23;
                out.set(Code::SveLuti4);
                out.set_mnemonic(Mnemonic::Luti4);
                out.push_operand(zreg(zd, VA::Sb));
                out.push_operand(zlist1(zn, VA::Sb));
                out.push_operand(zidx(zm, index));
            }
            // .H single table: 2-bit index at <23:22>.
            (1, 1) => {
                let index = (b23 << 1) | b22;
                out.set(Code::SveLuti4);
                out.set_mnemonic(Mnemonic::Luti4);
                out.push_operand(zreg(zd, VA::Sh));
                out.push_operand(zlist1(zn, VA::Sh));
                out.push_operand(zidx(zm, index));
            }
            // .H two table registers: 2-bit index at <23:22>.
            (1, 0) => {
                let index = (b23 << 1) | b22;
                out.set(Code::SveLuti4Two);
                out.set_mnemonic(Mnemonic::Luti4);
                out.push_operand(zreg(zd, VA::Sh));
                out.push_operand(zlist2(zn, VA::Sh));
                out.push_operand(zidx(zm, index));
            }
            // (0, 1) is unallocated.
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::format::{BufSink, FmtFormatter, Formatter};
    use crate::mnemonic::{Code, Mnemonic};
    use crate::{Decoder, DecoderOptions};

    /// Decode `word` and render with the default UAL formatter.
    fn render(word: u32, buf: &mut [u8]) -> &str {
        let bytes = word.to_le_bytes();
        let mut dec = Decoder::new(&bytes, 0x1000, DecoderOptions::default());
        let insn = dec.decode();
        let n = {
            let mut sink = BufSink::new(buf);
            FmtFormatter::new().format(&insn, &mut sink);
            sink.len()
        };
        core::str::from_utf8(&buf[..n]).unwrap_or("")
    }

    #[track_caller]
    fn check(word: u32, expected: &str) {
        let mut buf = [0u8; 128];
        assert_eq!(render(word, &mut buf), expected, "word={word:#010x}");
    }

    /// Decode `word`, re-encode, and require the encoder to reproduce it exactly.
    #[track_caller]
    fn roundtrip(word: u32) {
        let bytes = word.to_le_bytes();
        let mut dec = Decoder::new(&bytes, 0, DecoderOptions::default());
        let insn = dec.decode();
        assert!(!insn.is_invalid(), "decoded Invalid: word={word:#010x}");
        let enc = insn.encode().expect("encode failed");
        assert_eq!(enc, word, "round-trip mismatch: word={word:#010x} got={enc:#010x}");
    }

    #[test]
    fn luti_decode_matches_llvm() {
        // Cross-checked against `llvm-mc --mattr=+all`.
        check(0x4520a9ca, "luti2   z10.h, {z14.h}, z0[0]");
        check(0x45e2b020, "luti2   z0.b, {z1.b}, z2[3]"); // .B, 2-bit index max
        check(0x45e5b883, "luti2   z3.h, {z4.h}, z5[7]"); // .H, 3-bit index max
        check(0x4522bd0d, "luti4   z13.h, {z8.h}, z2[0]");
        check(0x45e8a4e6, "luti4   z6.b, {z7.b}, z8[1]"); // .B, 1-bit index
        check(0x45ebbd49, "luti4   z9.h, {z10.h}, z11[3]"); // .H single, 2-bit index
        check(0x4522b50d, "luti4   z13.h, {z8.h, z9.h}, z2[0]"); // two-table
        check(0x45fdb7fe, "luti4   z30.h, {z31.h, z0.h}, z29[3]"); // two-table, wrap
    }

    #[test]
    fn luti_encode_roundtrips() {
        for w in [
            0x4520a9ca_u32,
            0x45e2b020,
            0x45e5b883,
            0x4522bd0d,
            0x45e8a4e6,
            0x45ebbd49,
            0x4522b50d,
            0x45fdb7fe,
        ] {
            roundtrip(w);
        }
    }

    #[test]
    fn luti_code_and_mnemonic() {
        let bytes = 0x4522b50d_u32.to_le_bytes();
        let mut dec = Decoder::new(&bytes, 0, DecoderOptions::default());
        let insn = dec.decode();
        assert_eq!(insn.code(), Code::SveLuti4Two);
        assert_eq!(insn.mnemonic(), Mnemonic::Luti4);
        assert_eq!(insn.op_count(), 3);
    }
}
