//! Advanced SIMD cryptographic instructions — hand-written from the ARM ARM.
//!
//! These are the NEON scalar/vector cryptographic encodings that live in the
//! "Data Processing -- Scalar FP and Advanced SIMD" top-level group (ARM ARM
//! C4.1.97), distinct from the SVE2 crypto forms in [`crate::decode::sve`]. They
//! are all gated behind the `crypto` cargo feature *and* the runtime
//! [`Feature::Crypto`] gate (the ARM ARM splits these across `FEAT_AES`,
//! `FEAT_SHA1`/`FEAT_SHA256`, `FEAT_SHA512`, `FEAT_SHA3`, `FEAT_SM3` and
//! `FEAT_SM4`; fARM64 collapses them under a single `Crypto` feature for the
//! lookup-table gate, matching the corpus oracle which accepts the whole crypto
//! block together).
//!
//! Eight encoding classes are dispatched from [`decode`] using the SIMD&FP
//! discriminators `op0=word<31:28>`, `op1=word<24:23>`, `op2=word<22:19>`,
//! `op3=word<18:10>` (the same fields [`super`] computes). The class → mnemonic
//! tables follow the ARM ARM exactly:
//!
//! | class | `op0`/`op1`/… | members |
//! |-|-|-|
//! | `cryptoaes` | `op0=4`, `(op2&7)==5`, `(op3&0x183)==2` | AESE/AESD/AESMC/AESIMC |
//! | `cryptosha3` | `op0=5`, `!(op2&4)`, `!(op3&0x23)` | SHA1C/P/M/SU0, SHA256H/H2/SU1 |
//! | `cryptosha2` | `op0=5`, `(op2&7)==5`, `(op3&0x183)==2` | SHA1H/SU1, SHA256SU0 |
//! | `cryptosha512_3` | `op0=12`, `op1=0`, `(op2&12)==12`, `(op3&0x2c)==0x20` | SHA512H/H2/SU1, RAX1, SM3PARTW1/2, SM4EKEY |
//! | `cryptosha512_2` | `op0=12`, `op1=1`, `op2=8`, `(op3&0x1fc)==0x20` | SHA512SU0, SM4E |
//! | `crypto4` | `op0=12`, `op1=0`, `!(op3&0x20)` | EOR3, BCAX, SM3SS1 |
//! | `crypto3_imm2` | `op0=12`, `op1=0`, `(op2&12)==8`, `(op3&0x30)==0x20` | SM3TT1A/1B/2A/2B |
//! | `crypto3_imm6` | `op0=12`, `op1=1`, `!(op2&12)` | XAR |
//!
//! Every leaf builds operands directly with the [`super::simd_arith`]-style
//! constructors re-implemented locally; all paths are total and panic-free,
//! leaving [`Code::Invalid`] for unallocated rows.

use crate::decode::bits::{bit, bits};
use crate::enums::VectorArrangement as VA;
use crate::features::{Feature, FeatureSet};
use crate::instruction::Instruction;
use crate::mnemonic::Code;
use crate::operand::Operand;
use crate::register::Register;

// ---------------------------------------------------------------------------
// Register-bank tables (local; mirroring `simd_arith`).
// ---------------------------------------------------------------------------

const V: [Register; 32] = [
    Register::V0, Register::V1, Register::V2, Register::V3, Register::V4, Register::V5, Register::V6, Register::V7,
    Register::V8, Register::V9, Register::V10, Register::V11, Register::V12, Register::V13, Register::V14, Register::V15,
    Register::V16, Register::V17, Register::V18, Register::V19, Register::V20, Register::V21, Register::V22, Register::V23,
    Register::V24, Register::V25, Register::V26, Register::V27, Register::V28, Register::V29, Register::V30, Register::V31,
];
const SR: [Register; 32] = [
    Register::S0, Register::S1, Register::S2, Register::S3, Register::S4, Register::S5, Register::S6, Register::S7,
    Register::S8, Register::S9, Register::S10, Register::S11, Register::S12, Register::S13, Register::S14, Register::S15,
    Register::S16, Register::S17, Register::S18, Register::S19, Register::S20, Register::S21, Register::S22, Register::S23,
    Register::S24, Register::S25, Register::S26, Register::S27, Register::S28, Register::S29, Register::S30, Register::S31,
];
const QR: [Register; 32] = [
    Register::Q0, Register::Q1, Register::Q2, Register::Q3, Register::Q4, Register::Q5, Register::Q6, Register::Q7,
    Register::Q8, Register::Q9, Register::Q10, Register::Q11, Register::Q12, Register::Q13, Register::Q14, Register::Q15,
    Register::Q16, Register::Q17, Register::Q18, Register::Q19, Register::Q20, Register::Q21, Register::Q22, Register::Q23,
    Register::Q24, Register::Q25, Register::Q26, Register::Q27, Register::Q28, Register::Q29, Register::Q30, Register::Q31,
];

/// A bare register operand.
#[inline]
fn plain(reg: Register) -> Operand {
    Operand::Reg { reg, arr: None, lane: None, shift: None, extend: None, pred: None }
}

/// A vector register operand `V{n}.<arr>`.
#[inline]
fn vreg(n: u32, arr: VA) -> Operand {
    Operand::Reg { reg: V[(n & 0x1f) as usize], arr: Some(arr), lane: None, shift: None, extend: None, pred: None }
}

/// An indexed vector-element operand `V{n}.<Ts>[index]`.
#[inline]
fn vreg_idx(n: u32, arr: VA, index: u8) -> Operand {
    Operand::Reg { reg: V[(n & 0x1f) as usize], arr: Some(arr), lane: Some(index), shift: None, extend: None, pred: None }
}

/// A scalar `Q{n}` operand.
#[inline]
fn qreg(n: u32) -> Operand {
    plain(QR[(n & 0x1f) as usize])
}

/// A scalar `S{n}` operand.
#[inline]
fn sreg(n: u32) -> Operand {
    plain(SR[(n & 0x1f) as usize])
}

// ---------------------------------------------------------------------------
// Dispatch.
// ---------------------------------------------------------------------------

/// Try to decode a word as one of the AdvSIMD cryptographic encodings.
///
/// `op0`/`op1`/`op2`/`op3` are the SIMD&FP discriminator fields already extracted
/// by [`super::decode`]. Returns `true` (and fills `out`) when the word matched a
/// crypto class — even if the *specific* opcode within that class is unallocated
/// (in which case `out` is left invalid but the word is still "claimed", exactly
/// as the ARM ARM classification tree behaves). Returns `false` when the word is
/// not in the crypto space at all, so the caller can continue routing.
///
/// Runtime-gated on [`Feature::Crypto`]: if the feature is not accepted, the
/// function returns `false` and the word falls through to the rest of the SIMD
/// decode (which will leave it invalid).
#[inline]
pub fn decode(
    word: u32,
    op0: u32,
    op1: u32,
    op2: u32,
    op3: u32,
    features: FeatureSet,
    out: &mut Instruction,
) -> bool {
    if !features.has(Feature::Crypto) {
        return false;
    }

    // cryptoaes: op0==4, !(op1&2), (op2&7)==5, (op3&0x183)==2.
    if op0 == 4 && (op1 & 2) == 0 && (op2 & 7) == 5 && (op3 & 0x183) == 2 {
        decode_aes(word, out);
        return true;
    }
    // cryptosha3: op0==5, !(op1&2), !(op2&4), !(op3&0x23).
    if op0 == 5 && (op1 & 2) == 0 && (op2 & 4) == 0 && (op3 & 0x23) == 0 {
        decode_sha3(word, out);
        return true;
    }
    // cryptosha2: op0==5, !(op1&2), (op2&7)==5, (op3&0x183)==2.
    if op0 == 5 && (op1 & 2) == 0 && (op2 & 7) == 5 && (op3 & 0x183) == 2 {
        decode_sha2(word, out);
        return true;
    }
    // The `op0==12` (0b1100) crypto block.
    if op0 == 12 {
        // crypto3_imm2: op1==0, (op2&12)==8, (op3&0x30)==0x20.
        if op1 == 0 && (op2 & 12) == 8 && (op3 & 0x30) == 0x20 {
            decode_sm3tt(word, out);
            return true;
        }
        // cryptosha512_3: op1==0, (op2&12)==12, (op3&0x2c)==0x20.
        if op1 == 0 && (op2 & 12) == 12 && (op3 & 0x2c) == 0x20 {
            decode_sha512_3(word, out);
            return true;
        }
        // crypto4: op1==0, !(op3&0x20).
        if op1 == 0 && (op3 & 0x20) == 0 {
            decode_crypto4(word, out);
            return true;
        }
        // crypto3_imm6 (XAR): op1==1, !(op2&12).
        if op1 == 1 && (op2 & 12) == 0 {
            decode_xar(word, out);
            return true;
        }
        // cryptosha512_2: op1==1, op2==8, (op3&0x1fc)==0x20.
        if op1 == 1 && op2 == 8 && (op3 & 0x1fc) == 0x20 {
            decode_sha512_2(word, out);
            return true;
        }
    }

    false
}

// ---------------------------------------------------------------------------
// AES (two-register): `cryptoaes`.
// ---------------------------------------------------------------------------

/// `AESE`/`AESD`/`AESMC`/`AESIMC` — `<Vd>.16B, <Vn>.16B`. `size=word<23:22>`
/// must be `00`; `opcode=word<16:12>` selects the operation (4/5/6/7).
fn decode_aes(word: u32, out: &mut Instruction) {
    let size = bits(word, 22, 2);
    let opcode = bits(word, 12, 5);
    if size != 0 {
        return;
    }
    let code = match opcode {
        4 => Code::AdvAese,
        5 => Code::AdvAesd,
        6 => Code::AdvAesmc,
        7 => Code::AdvAesimc,
        _ => return,
    };
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);
    out.set(code);
    out.push_operand(vreg(rd, VA::V16B));
    out.push_operand(vreg(rn, VA::V16B));
}

// ---------------------------------------------------------------------------
// SHA1/SHA256 three-register: `cryptosha3`.
// ---------------------------------------------------------------------------

/// `SHA1C`/`SHA1P`/`SHA1M` (`<Qd>, <Sn>, <Vm>.4S`), `SHA1SU0` / `SHA256SU1`
/// (`<Vd>.4S, <Vn>.4S, <Vm>.4S`), `SHA256H`/`SHA256H2` (`<Qd>, <Qn>, <Vm>.4S`).
/// `size=word<23:22>` must be `00`; `opcode=word<14:12>` selects the row.
fn decode_sha3(word: u32, out: &mut Instruction) {
    let size = bits(word, 22, 2);
    let opcode = bits(word, 12, 3);
    if size != 0 {
        return;
    }
    let rm = bits(word, 16, 5);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);

    match opcode {
        // SHA1C/P/M: Qd, Sn, Vm.4S.
        0..=2 => {
            let code = match opcode {
                0 => Code::AdvSha1c,
                1 => Code::AdvSha1p,
                _ => Code::AdvSha1m,
            };
            out.set(code);
            out.push_operand(qreg(rd));
            out.push_operand(sreg(rn));
            out.push_operand(vreg(rm, VA::V4S));
        }
        // SHA1SU0: Vd.4S, Vn.4S, Vm.4S.
        3 => {
            out.set(Code::AdvSha1su0);
            out.push_operand(vreg(rd, VA::V4S));
            out.push_operand(vreg(rn, VA::V4S));
            out.push_operand(vreg(rm, VA::V4S));
        }
        // SHA256H/H2: Qd, Qn, Vm.4S.
        4 | 5 => {
            out.set(if opcode == 4 { Code::AdvSha256h } else { Code::AdvSha256h2 });
            out.push_operand(qreg(rd));
            out.push_operand(qreg(rn));
            out.push_operand(vreg(rm, VA::V4S));
        }
        // SHA256SU1: Vd.4S, Vn.4S, Vm.4S.
        6 => {
            out.set(Code::AdvSha256su1);
            out.push_operand(vreg(rd, VA::V4S));
            out.push_operand(vreg(rn, VA::V4S));
            out.push_operand(vreg(rm, VA::V4S));
        }
        // opcode==7 is unallocated.
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// SHA1/SHA256 two-register: `cryptosha2`.
// ---------------------------------------------------------------------------

/// `SHA1H` (`<Sd>, <Sn>`), `SHA1SU1` / `SHA256SU0` (`<Vd>.4S, <Vn>.4S`).
/// `size=word<23:22>` must be `00`; `opcode=word<16:12>` selects the row.
fn decode_sha2(word: u32, out: &mut Instruction) {
    let size = bits(word, 22, 2);
    let opcode = bits(word, 12, 5);
    if size != 0 {
        return;
    }
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);

    match opcode {
        // SHA1H: Sd, Sn.
        0 => {
            out.set(Code::AdvSha1h);
            out.push_operand(sreg(rd));
            out.push_operand(sreg(rn));
        }
        // SHA1SU1: Vd.4S, Vn.4S.
        1 => {
            out.set(Code::AdvSha1su1);
            out.push_operand(vreg(rd, VA::V4S));
            out.push_operand(vreg(rn, VA::V4S));
        }
        // SHA256SU0: Vd.4S, Vn.4S.
        2 => {
            out.set(Code::AdvSha256su0);
            out.push_operand(vreg(rd, VA::V4S));
            out.push_operand(vreg(rn, VA::V4S));
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// SHA512 / RAX1 / SM3PARTW / SM4EKEY three-register: `cryptosha512_3`.
// ---------------------------------------------------------------------------

/// `cryptosha512_3`: `O=word<14>`, `opcode=word<11:10>`.
/// `O==0`: SHA512H(0)/SHA512H2(1)/SHA512SU1(2)/RAX1(3).
/// `O==1`: SM3PARTW1(0)/SM3PARTW2(1)/SM4EKEY(2)/unallocated(3).
fn decode_sha512_3(word: u32, out: &mut Instruction) {
    let o = bit(word, 14);
    let opcode = bits(word, 10, 2);
    let rm = bits(word, 16, 5);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);

    if o == 0 {
        match opcode {
            // SHA512H/H2: Qd, Qn, Vm.2D.
            0 | 1 => {
                out.set(if opcode == 0 { Code::AdvSha512h } else { Code::AdvSha512h2 });
                out.push_operand(qreg(rd));
                out.push_operand(qreg(rn));
                out.push_operand(vreg(rm, VA::V2D));
            }
            // SHA512SU1: Vd.2D, Vn.2D, Vm.2D.
            2 => {
                out.set(Code::AdvSha512su1);
                out.push_operand(vreg(rd, VA::V2D));
                out.push_operand(vreg(rn, VA::V2D));
                out.push_operand(vreg(rm, VA::V2D));
            }
            // RAX1: Vd.2D, Vn.2D, Vm.2D.
            _ => {
                out.set(Code::AdvRax1);
                out.push_operand(vreg(rd, VA::V2D));
                out.push_operand(vreg(rn, VA::V2D));
                out.push_operand(vreg(rm, VA::V2D));
            }
        }
    } else {
        match opcode {
            // SM3PARTW1/2: Vd.4S, Vn.4S, Vm.4S.
            0 | 1 => {
                out.set(if opcode == 0 { Code::AdvSm3partw1 } else { Code::AdvSm3partw2 });
                out.push_operand(vreg(rd, VA::V4S));
                out.push_operand(vreg(rn, VA::V4S));
                out.push_operand(vreg(rm, VA::V4S));
            }
            // SM4EKEY: Vd.4S, Vn.4S, Vm.4S.
            2 => {
                out.set(Code::AdvSm4ekey);
                out.push_operand(vreg(rd, VA::V4S));
                out.push_operand(vreg(rn, VA::V4S));
                out.push_operand(vreg(rm, VA::V4S));
            }
            // opcode==3 unallocated.
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// SHA512SU0 / SM4E two-register: `cryptosha512_2`.
// ---------------------------------------------------------------------------

/// `cryptosha512_2`: `opcode=word<11:10>` selects SHA512SU0(0) / SM4E(1).
/// SHA512SU0 is `<Vd>.2D, <Vn>.2D`; SM4E is `<Vd>.4S, <Vn>.4S`.
fn decode_sha512_2(word: u32, out: &mut Instruction) {
    let opcode = bits(word, 10, 2);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);
    match opcode {
        0 => {
            out.set(Code::AdvSha512su0);
            out.push_operand(vreg(rd, VA::V2D));
            out.push_operand(vreg(rn, VA::V2D));
        }
        1 => {
            out.set(Code::AdvSm4e);
            out.push_operand(vreg(rd, VA::V4S));
            out.push_operand(vreg(rn, VA::V4S));
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// EOR3 / BCAX / SM3SS1 four-register: `crypto4`.
// ---------------------------------------------------------------------------

/// `crypto4`: `Op0=word<22:21>`. EOR3(0)/BCAX(1): `<Vd>.16B, <Vn>.16B, <Vm>.16B,
/// <Va>.16B`. SM3SS1(2): `<Vd>.4S, <Vn>.4S, <Vm>.4S, <Va>.4S`. `Op0==3`
/// unallocated. `Ra=word<14:10>`.
fn decode_crypto4(word: u32, out: &mut Instruction) {
    let op0 = bits(word, 21, 2);
    let rm = bits(word, 16, 5);
    let ra = bits(word, 10, 5);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);

    let (code, arr) = match op0 {
        0 => (Code::AdvEor3, VA::V16B),
        1 => (Code::AdvBcax, VA::V16B),
        2 => (Code::AdvSm3ss1, VA::V4S),
        _ => return,
    };
    out.set(code);
    out.push_operand(vreg(rd, arr));
    out.push_operand(vreg(rn, arr));
    out.push_operand(vreg(rm, arr));
    out.push_operand(vreg(ra, arr));
}

// ---------------------------------------------------------------------------
// SM3TT1A/1B/2A/2B three-register, 2-bit index: `crypto3_imm2`.
// ---------------------------------------------------------------------------

/// `crypto3_imm2`: `imm2=word<13:12>`, `opcode=word<11:10>` selects the variant.
/// All are `<Vd>.4S, <Vn>.4S, <Vm>.S[<imm2>]`.
fn decode_sm3tt(word: u32, out: &mut Instruction) {
    let opcode = bits(word, 10, 2);
    let imm2 = bits(word, 12, 2) as u8;
    let rm = bits(word, 16, 5);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);

    let code = match opcode {
        0 => Code::AdvSm3tt1a,
        1 => Code::AdvSm3tt1b,
        2 => Code::AdvSm3tt2a,
        _ => Code::AdvSm3tt2b,
    };
    out.set(code);
    out.push_operand(vreg(rd, VA::V4S));
    out.push_operand(vreg(rn, VA::V4S));
    out.push_operand(vreg_idx(rm, VA::Ss, imm2));
}

// ---------------------------------------------------------------------------
// XAR three-register, 6-bit rotate: `crypto3_imm6`.
// ---------------------------------------------------------------------------

/// `XAR <Vd>.2D, <Vn>.2D, <Vm>.2D, #<imm6>`. `imm6=word<15:10>` is the rotate.
fn decode_xar(word: u32, out: &mut Instruction) {
    let rm = bits(word, 16, 5);
    let imm6 = bits(word, 10, 6);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);
    out.set(Code::AdvXar);
    out.push_operand(vreg(rd, VA::V2D));
    out.push_operand(vreg(rn, VA::V2D));
    out.push_operand(vreg(rm, VA::V2D));
    out.push_operand(Operand::ImmUnsigned(imm6 as u64));
}

#[cfg(test)]
mod tests {
    use crate::format::{BufSink, FmtFormatter, Formatter};
    use crate::{Decoder, DecoderOptions};

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

    #[test]
    fn aes() {
        check(0x4E284BE7, "aese    v7.16b, v31.16b");
        check(0x4E285802, "aesd    v2.16b, v0.16b");
        check(0x4E286A06, "aesmc   v6.16b, v16.16b");
        check(0x4E287A67, "aesimc  v7.16b, v19.16b");
    }

    #[test]
    fn sha3_three_reg() {
        check(0x5E1B00ED, "sha1c   q13, s7, v27.4s");
        check(0x5E1311EF, "sha1p   q15, s15, v19.4s");
        check(0x5E0A23E3, "sha1m   q3, s31, v10.4s");
        check(0x5E0533DF, "sha1su0 v31.4s, v30.4s, v5.4s");
        check(0x5E1E4007, "sha256h q7, q0, v30.4s");
        check(0x5E1253B5, "sha256h2 q21, q29, v18.4s");
        check(0x5E0C622D, "sha256su1 v13.4s, v17.4s, v12.4s");
    }

    #[test]
    fn sha2_two_reg() {
        check(0x5E280ABB, "sha1h   s27, s21");
        check(0x5E281A6D, "sha1su1 v13.4s, v19.4s");
        check(0x5E282969, "sha256su0 v9.4s, v11.4s");
    }

    #[test]
    fn sha512_and_sm() {
        check(0xCE6A81DE, "sha512h q30, q14, v10.2d");
        check(0xCE6F841E, "sha512h2 q30, q0, v15.2d");
        check(0xCE6588CD, "sha512su1 v13.2d, v6.2d, v5.2d");
        check(0xCE648DF2, "rax1    v18.2d, v15.2d, v4.2d");
        check(0xCE7AC007, "sm3partw1 v7.4s, v0.4s, v26.4s");
        check(0xCE64C528, "sm3partw2 v8.4s, v9.4s, v4.4s");
        check(0xCE7ECA16, "sm4ekey v22.4s, v16.4s, v30.4s");
        check(0xCEC080CA, "sha512su0 v10.2d, v6.2d");
        check(0xCEC084D8, "sm4e    v24.4s, v6.4s");
    }

    #[test]
    fn crypto4_and_imm() {
        check(0xCE0B75E2, "eor3    v2.16b, v15.16b, v11.16b, v29.16b");
        check(0xCE382E9A, "bcax    v26.16b, v20.16b, v24.16b, v11.16b");
        check(0xCE4E66E1, "sm3ss1  v1.4s, v23.4s, v14.4s, v25.4s");
        check(0xCE5FB3F5, "sm3tt1a v21.4s, v31.4s, v31.s[3]");
        check(0xCE43A604, "sm3tt1b v4.4s, v16.4s, v3.s[2]");
        check(0xCE539AC3, "sm3tt2a v3.4s, v22.4s, v19.s[1]");
        check(0xCE5AACE4, "sm3tt2b v4.4s, v7.4s, v26.s[2]");
        check(0xCE8FCBA8, "xar     v8.2d, v29.2d, v15.2d, #0x32");
    }

    #[test]
    fn feature_gate_off_leaves_invalid() {
        use crate::features::FeatureSet;
        let opts = DecoderOptions { features: FeatureSet::BASE };
        let bytes = 0x4E284BE7u32.to_le_bytes(); // aese
        let mut dec = Decoder::new(&bytes, 0x1000, opts);
        assert!(dec.decode().is_invalid());
    }
}
