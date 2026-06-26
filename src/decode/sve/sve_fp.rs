//! SVE / SVE2 floating-point encodings — hand-written.
//!
//! Transcribed from the *ARM Architecture Reference Manual* SVE encoding index.
//! This module owns the SVE floating-point families:
//!
//! * FP arithmetic, predicated & unpredicated — `FADD`/`FSUB`/`FSUBR`/`FMUL`/
//!   `FDIV`/`FDIVR`/`FABD`/`FSCALE`/`FMAX`/`FMIN`/`FMAXNM`/`FMINNM`/`FMULX`, the
//!   `FADD`/`FSUB`/`FMUL`/`FMAX`/.. immediate forms, the FP pairwise (SVE2)
//!   `FADDP`/`FMAXP`/.., and predicated unary `FNEG`/`FABS`/`FRECPX`/`FSQRT`/
//!   `FLOGB`;
//! * FP multiply-add — `FMLA`/`FMLS`/`FNMLA`/`FNMLS`/`FMAD`/`FMSB`/`FNMAD`/
//!   `FNMSB`, `FMLA`/`FMLS` and `FMUL` by indexed element, and the bf16 / half
//!   widening multiply-add long family;
//! * FP complex — `FCADD`, `FCMLA` (vector and by indexed element);
//! * FP reductions — `FADDV`/`FMAXV`/`FMINV`/`FMAXNMV`/`FMINNMV` and `FADDA`;
//! * FP compare — `FCMEQ`/`FCMNE`/`FCMGT`/`FCMGE`/`FCMUO`/`FCMLT`/`FCMLE` (vector
//!   and with `#0.0`), `FACGT`/`FACGE`;
//! * FP convert / round — `FCVT`, `FCVTZS`/`FCVTZU`, `SCVTF`/`UCVTF`, the
//!   `FRINT*` family, `FCVTLT`/`FCVTNT`/`FCVTX`/`FCVTXNT` (SVE2), `BFCVT`/
//!   `BFCVTNT` (bf16);
//! * FP misc — `FEXPA`, `FTMAD`, `FTSMUL`, `FTSSEL`, `FRECPE`, `FRSQRTE`,
//!   `FRECPS`, `FRSQRTS`, `FMMLA`/`BFMMLA`, and the FP-immediate `FCPY`/`FMOV`/
//!   `FDUP`.
//!
//! Code identity follows the module convention: one [`Code`] per ARM ARM
//! encoding class, the preferred-disassembly alias installed via
//! [`Instruction::set_mnemonic`] where the corpus uses one (`FMOV` for `FCPY`/
//! `FDUP`), and arrangement / predicate / lane decoration carried in the
//! operands. Every path is total and panic-free; unallocated encodings are left
//! [`Code::Invalid`].
//!
//! Most encodings live in the SVE floating-point quadrant (`word<31:29> = 011`,
//! top bytes `0x64`/`0x65`); a handful of unary / select forms (`FABS`/`FNEG`/
//! `FEXPA`/`FTSSEL`) live in the `000` quadrant (`0x04`) and the FP-immediate
//! `FCPY`/`FDUP` in `0x05`/`0x25`. The latter are dispatched here from
//! [`super::decode`] via [`decode_fp_misc_04`] / [`decode_fcpy_05`] /
//! [`decode_fdup_25`].

use crate::decode::bits::{bit, bits, vfp_expand_imm};
use crate::enums::VectorArrangement as VA;
use crate::features::{Feature, FeatureSet};
use crate::instruction::Instruction;
use crate::mnemonic::{Code, Mnemonic};
use crate::operand::{Operand, PredQual};
use crate::register::Register;

// ---------------------------------------------------------------------------
// Register-bank tables (local mirrors, as in the sibling SVE modules).
// ---------------------------------------------------------------------------

const Z: [Register; 32] = [
    Register::Z0, Register::Z1, Register::Z2, Register::Z3, Register::Z4, Register::Z5, Register::Z6, Register::Z7,
    Register::Z8, Register::Z9, Register::Z10, Register::Z11, Register::Z12, Register::Z13, Register::Z14, Register::Z15,
    Register::Z16, Register::Z17, Register::Z18, Register::Z19, Register::Z20, Register::Z21, Register::Z22, Register::Z23,
    Register::Z24, Register::Z25, Register::Z26, Register::Z27, Register::Z28, Register::Z29, Register::Z30, Register::Z31,
];
const P: [Register; 16] = [
    Register::P0, Register::P1, Register::P2, Register::P3, Register::P4, Register::P5, Register::P6, Register::P7,
    Register::P8, Register::P9, Register::P10, Register::P11, Register::P12, Register::P13, Register::P14, Register::P15,
];
const HR: [Register; 32] = [
    Register::H0, Register::H1, Register::H2, Register::H3, Register::H4, Register::H5, Register::H6, Register::H7,
    Register::H8, Register::H9, Register::H10, Register::H11, Register::H12, Register::H13, Register::H14, Register::H15,
    Register::H16, Register::H17, Register::H18, Register::H19, Register::H20, Register::H21, Register::H22, Register::H23,
    Register::H24, Register::H25, Register::H26, Register::H27, Register::H28, Register::H29, Register::H30, Register::H31,
];
const SR: [Register; 32] = [
    Register::S0, Register::S1, Register::S2, Register::S3, Register::S4, Register::S5, Register::S6, Register::S7,
    Register::S8, Register::S9, Register::S10, Register::S11, Register::S12, Register::S13, Register::S14, Register::S15,
    Register::S16, Register::S17, Register::S18, Register::S19, Register::S20, Register::S21, Register::S22, Register::S23,
    Register::S24, Register::S25, Register::S26, Register::S27, Register::S28, Register::S29, Register::S30, Register::S31,
];
const DR: [Register; 32] = [
    Register::D0, Register::D1, Register::D2, Register::D3, Register::D4, Register::D5, Register::D6, Register::D7,
    Register::D8, Register::D9, Register::D10, Register::D11, Register::D12, Register::D13, Register::D14, Register::D15,
    Register::D16, Register::D17, Register::D18, Register::D19, Register::D20, Register::D21, Register::D22, Register::D23,
    Register::D24, Register::D25, Register::D26, Register::D27, Register::D28, Register::D29, Register::D30, Register::D31,
];

// ---------------------------------------------------------------------------
// Small operand constructors.
// ---------------------------------------------------------------------------

/// Element-size arrangement (`.b`/`.h`/`.s`/`.d`) from a 2-bit `size`.
#[inline]
fn arr(size: u32) -> VA {
    match size & 3 {
        0 => VA::Sb,
        1 => VA::Sh,
        2 => VA::Ss,
        _ => VA::Sd,
    }
}

/// A scalable `Z{n}` operand with arrangement `a`.
#[inline]
fn zreg(n: u32, a: VA) -> Operand {
    Operand::Reg { reg: Z[(n & 0x1f) as usize], arr: Some(a), lane: None, shift: None, extend: None, pred: None }
}

/// A scalable `Z{n}` operand with arrangement `a` and an element-index lane.
#[inline]
fn zreg_idx(n: u32, a: VA, lane: u8) -> Operand {
    Operand::Reg { reg: Z[(n & 0x1f) as usize], arr: Some(a), lane: Some(lane), shift: None, extend: None, pred: None }
}

/// A governing predicate `P{n}` with a `/z` or `/m` qualifier.
#[inline]
fn preg_q(n: u32, q: PredQual) -> Operand {
    Operand::Reg { reg: P[(n & 0xf) as usize], arr: None, lane: None, shift: None, extend: None, pred: Some(q) }
}

/// A bare predicate `P{n}` (no qualifier, no size) — the `<Pg>` of reductions.
#[inline]
fn preg(n: u32) -> Operand {
    Operand::Reg { reg: P[(n & 0xf) as usize], arr: None, lane: None, shift: None, extend: None, pred: None }
}

/// A sized predicate `P{n}.<T>` destination (compare results).
#[inline]
fn preg_sz(n: u32, a: VA) -> Operand {
    Operand::Reg { reg: P[(n & 0xf) as usize], arr: Some(a), lane: None, shift: None, extend: None, pred: None }
}

/// A NEON `V{n}` operand with a full-128-bit arrangement (`v0.8h`/`.4s`/`.2d`),
/// the destination of the SVE2.1 quadword FP reductions.
#[inline]
fn vreg(n: u32, a: VA) -> Operand {
    Operand::Reg {
        reg: crate::register::v_numbered((n & 0x1f) as u8),
        arr: Some(a),
        lane: None,
        shift: None,
        extend: None,
        pred: None,
    }
}

/// The full-128-bit NEON arrangement matching FP element `size`
/// (`1`=`.8h`, `2`=`.4s`, `3`=`.2d`).
#[inline]
fn va_neon(size: u32) -> VA {
    match size & 3 {
        1 => VA::V8H,
        2 => VA::V4S,
        _ => VA::V2D,
    }
}

/// A scalar SIMD `B/H/S/D` operand of the element width for `size`.
#[inline]
fn scalar_fp(n: u32, size: u32) -> Operand {
    let n = (n & 0x1f) as usize;
    let reg = match size & 3 {
        0 => Register::B0, // byte FP reductions do not exist; placeholder, callers reject size 0
        1 => HR[n],
        2 => SR[n],
        _ => DR[n],
    };
    let reg = if size & 3 == 0 { HR[n] } else { reg };
    Operand::Reg { reg, arr: None, lane: None, shift: None, extend: None, pred: None }
}

/// An FP immediate operand (rendered shortest-decimal by the formatter).
#[inline]
fn fpimm_bits(bits: u64) -> Operand {
    // Bit-cast the low 32 bits to f32 for the formatter (no FP arithmetic here).
    Operand::FpImm(f32::from_bits(bits as u32))
}

// ---------------------------------------------------------------------------
// Entry point.
// ---------------------------------------------------------------------------

/// Decode an SVE/SVE2 floating-point instruction (`word<31:29> == 011`, top
/// bytes `0x64`/`0x65`/`0x66`/`0x67`) into `out`.
///
/// Called from [`super::decode`]. Total and panic-free; leaves `out` invalid for
/// unallocated encodings.
#[inline]
pub fn decode(word: u32, ip: u64, features: FeatureSet, out: &mut Instruction) {
    let _ = ip;
    if !features.has(Feature::Sve) {
        return;
    }
    match bits(word, 24, 8) {
        0x64 => decode_64(word, features, out),
        0x65 => decode_65(word, features, out),
        // 0x66 / 0x67 are SVE memory in the 011-quadrant overlap; not FP. Leave
        // invalid (the SVE FP space is entirely within 0x64/0x65).
        _ => {}
    }
}

// ===========================================================================
// Top byte 0x65 — the bulk of SVE FP (arithmetic / convert / compare / etc).
// ===========================================================================

#[inline]
fn decode_65(word: u32, features: FeatureSet, out: &mut Instruction) {
    let b21 = bit(word, 21);
    let b20 = bit(word, 20);
    let sel = bits(word, 13, 3); // word<15:13>
    if b21 == 0 {
        // <21>=0: the unpredicated 3-source / 2-source family lives here when
        // <15:13>==0xx, otherwise predicated unary/binary/compare.
        match sel {
            0b000 => decode_65_unpred_arith(word, out),
            0b001 => decode_65_unary_misc(word, out),
            0b010 | 0b011 => decode_65_compare(word, out),
            0b100 => decode_65_pred_binary(word, out),
            0b101 => decode_65_pred_unary(word, out),
            0b110 | 0b111 => decode_65_compare(word, out),
            _ => {}
        }
    } else {
        // <21>=1: FP multiply-add (4-operand predicated), or FP compare with the
        // <20> bit. The 3-way add family uses <15:13> as the opcode.
        let _ = b20;
        decode_65_fma(word, out);
    }
    let _ = features;
}

/// Unpredicated FP 3-register and 2-register (`<21>=0`, `<15:13>=000`):
/// `FADD`/`FSUB`/`FMUL`/`FTSMUL`/`FRECPS`/`FRSQRTS` `<Zd>.<T>, <Zn>.<T>, <Zm>.<T>`.
/// The op is in `word<12:10>` (the low `opc` of the `..._z_zz_` rows).
#[inline]
fn decode_65_unpred_arith(word: u32, out: &mut Instruction) {
    let size = bits(word, 22, 2);
    if size == 0 {
        return; // FP needs H/S/D.
    }
    let zm = bits(word, 16, 5);
    let opc = bits(word, 10, 3); // word<12:10>
    let zn = bits(word, 5, 5);
    let zd = bits(word, 0, 5);
    let a = arr(size);
    let code = match opc {
        0b000 => Code::SveFaddZzz,
        0b001 => Code::SveFsubZzz,
        0b010 => Code::SveFmulZzz,
        0b011 => Code::SveFtsmulZzz,
        0b110 => Code::SveFrecpsZzz,
        0b111 => Code::SveFrsqrtsZzz,
        _ => return,
    };
    out.set(code);
    out.push_operand(zreg(zd, a));
    out.push_operand(zreg(zn, a));
    out.push_operand(zreg(zm, a));
}

/// Unpredicated FP unary / reduction (`<21>=0`, `<15:13>=001`), plus the FP
/// compare-with-zero forms that share this slot.
///
/// This slot is shared by:
/// * the FP compare-with-zero `FCM{GE,GT,LT,LE,EQ,NE}` (`<20>=1`, result
///   `<Pd>.<T>`);
/// * the FP reductions `FADDV`/`FMAXNMV`/`FMINNMV`/`FMAXV`/`FMINV` (`<20:19>=00`,
///   `<12:10>=000`) and the strictly-ordered `FADDA` (`<20:16>=11000`);
/// * `FRECPE`/`FRSQRTE` (`<20:16>=01110/01111`, `<12:10>=100`).
#[inline]
fn decode_65_unary_misc(word: u32, out: &mut Instruction) {
    let size = bits(word, 22, 2);
    if size == 0 {
        return;
    }
    let opc = bits(word, 16, 5); // word<20:16>
    let pg = bits(word, 10, 3);
    let zn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);
    let a = arr(size);

    // FP compare-with-zero: 01100101 size 01 0 ooo 001 Pg Zn b4 Pd, i.e. <20>=1,
    // <19>=0, <18:16>=op, with the predicate result in <3:0>.
    if bit(word, 20) == 1 && bit(word, 19) == 0 {
        let op = bits(word, 16, 3); // word<18:16>
        let b4 = bit(word, 4);
        let pd = bits(word, 0, 4);
        let code = match (op, b4) {
            (0b000, 0) => Code::SveFcmgeZ0,
            (0b000, 1) => Code::SveFcmgtZ0,
            (0b001, 0) => Code::SveFcmltZ0,
            (0b001, 1) => Code::SveFcmleZ0,
            (0b010, 0) => Code::SveFcmeqZ0,
            (0b011, 0) => Code::SveFcmneZ0,
            _ => return,
        };
        out.set(code);
        out.push_operand(preg_sz(pd, a));
        out.push_operand(preg_q(pg, PredQual::Zeroing));
        out.push_operand(zreg(zn, a));
        out.push_operand(Operand::FpImm(0.0));
        return;
    }

    // FRECPE / FRSQRTE: <20:16> = 0_1110 / 0_1111 with <12:10>=100 (unpredicated).
    if bits(word, 10, 6) == 0b001100 {
        match opc {
            0b01110 => {
                out.set(Code::SveFrecpe);
                out.push_operand(zreg(rd, a));
                out.push_operand(zreg(zn, a));
                return;
            }
            0b01111 => {
                out.set(Code::SveFrsqrte);
                out.push_operand(zreg(rd, a));
                out.push_operand(zreg(zn, a));
                return;
            }
            _ => {}
        }
    }

    // FP recursive reductions: 01100101 size 000 opc 001 Pg Zn Vd, with
    // <20:19>=00 (the `001` is <15:13>, this slot; <12:10> is Pg). The scalar
    // destination is the element-width SIMD reg; the source is the vector `Zn`.
    if bits(word, 19, 2) == 0b00 {
        let code = match bits(word, 16, 3) {
            0b000 => Code::SveFaddv,
            0b100 => Code::SveFmaxnmv,
            0b101 => Code::SveFminnmv,
            0b110 => Code::SveFmaxv,
            0b111 => Code::SveFminv,
            _ => return,
        };
        out.set(code);
        out.push_operand(scalar_fp(rd, size));
        out.push_operand(preg(pg));
        out.push_operand(zreg(zn, a));
        return;
    }

    // FADDA (strictly-ordered add reduction): 01100101 size 011000 001 Pg Zn Vdn,
    // with <20:16>=11000 (the `001` is <15:13>, already this slot; <12:10> is Pg).
    // `Vdn` is both operand 0 and 2.
    if bits(word, 16, 5) == 0b11000 {
        out.set(Code::SveFadda);
        out.push_operand(scalar_fp(rd, size));
        out.push_operand(preg(pg));
        out.push_operand(scalar_fp(rd, size));
        out.push_operand(zreg(zn, a));
    }
}

/// Predicated FP binary destructive (`<21>=0`, `<15:13>=100`):
/// `op <Zdn>.<T>, <Pg>/M, <Zdn>.<T>, <Zm>.<T>` and the immediate forms, plus the
/// SVE2 pairwise group. The op is selected by `word<20:16>`.
#[inline]
fn decode_65_pred_binary(word: u32, out: &mut Instruction) {
    let size = bits(word, 22, 2);
    if size == 0 {
        return;
    }
    let opc = bits(word, 16, 5); // word<20:16>
    let pg = bits(word, 10, 3);
    let zm = bits(word, 5, 5);
    let zdn = bits(word, 0, 5);
    let a = arr(size);

    // The <20:19>=11 sub-block is the FP "arithmetic with immediate" (FADD/FSUB/
    // FMUL/.. with #const). <18:16> selects which. The plain binary group uses
    // <20:19> in {00,01} (so `<20>=0` distinguishes the two destructive halves).
    if bits(word, 19, 2) == 0b11 {
        // immediate forms: 01100101 size 011 opc<2:0> 100 i1 0000 Zdn  (the i1
        // is word<5>, with the rest of the source field zero).
        let opc3 = bits(word, 16, 3); // word<18:16>
        let i1 = bit(word, 5);
        let (code, imm) = match opc3 {
            0b000 => (Code::SveFaddZpzi, if i1 == 0 { HALF } else { ONE }),
            0b001 => (Code::SveFsubZpzi, if i1 == 0 { HALF } else { ONE }),
            0b010 => (Code::SveFmulZpzi, if i1 == 0 { HALF } else { TWO }),
            0b011 => (Code::SveFsubrZpzi, if i1 == 0 { HALF } else { ONE }),
            0b100 => (Code::SveFmaxnmZpzi, if i1 == 0 { ZERO } else { ONE }),
            0b101 => (Code::SveFminnmZpzi, if i1 == 0 { ZERO } else { ONE }),
            0b110 => (Code::SveFmaxZpzi, if i1 == 0 { ZERO } else { ONE }),
            0b111 => (Code::SveFminZpzi, if i1 == 0 { ZERO } else { ONE }),
            _ => return,
        };
        out.set(code);
        out.push_operand(zreg(zdn, a));
        out.push_operand(preg_q(pg, PredQual::Merging));
        out.push_operand(zreg(zdn, a));
        out.push_operand(fpimm_bits(imm));
        return;
    }

    // FTMAD: 01100101 size 010 imm3 100000 Zm Zdn (<20:19>=10, <15:10>=100000).
    // It shares this sel=100 slot with the plain binary group but is identified
    // by the fixed <15:10>=100000 tail and the trailing 3-bit immediate.
    if bits(word, 19, 2) == 0b10 && bits(word, 10, 6) == 0b100000 {
        let imm3 = bits(word, 16, 3);
        out.set(Code::SveFtmad);
        out.push_operand(zreg(zdn, a));
        out.push_operand(zreg(zdn, a));
        out.push_operand(zreg(zm, a));
        out.push_operand(Operand::ImmUnsigned(imm3 as u64));
        return;
    }

    // <19>=0, <20>=0: the plain predicated binary group.
    let code = match opc {
        0b00000 => Code::SveFaddZpzz,
        0b00001 => Code::SveFsubZpzz,
        0b00010 => Code::SveFmulZpzz,
        0b00011 => Code::SveFsubrZpzz,
        0b00100 => Code::SveFmaxnmZpzz,
        0b00101 => Code::SveFminnmZpzz,
        0b00110 => Code::SveFmaxZpzz,
        0b00111 => Code::SveFminZpzz,
        0b01000 => Code::SveFabdZpzz,
        0b01001 => Code::SveFscaleZpzz,
        0b01010 => Code::SveFmulxZpzz,
        0b01100 => Code::SveFdivrZpzz,
        0b01101 => Code::SveFdivZpzz,
        _ => return,
    };
    out.set(code);
    push_pred_binary(out, zdn, pg, zm, a);
}

/// Push the standard `Zdn.T, Pg/M, Zdn.T, Zm.T` quad for a predicated binary
/// destructive FP instruction.
#[inline]
fn push_pred_binary(out: &mut Instruction, zdn: u32, pg: u32, zm: u32, a: VA) {
    out.push_operand(zreg(zdn, a));
    out.push_operand(preg_q(pg, PredQual::Merging));
    out.push_operand(zreg(zdn, a));
    out.push_operand(zreg(zm, a));
}

/// Predicated FP unary (`<21>=0`, `<15:13>=101`):
/// `FRINT*`/`FRECPX`/`FSQRT`/`FCVT*`/`FCVTZ*`/`SCVTF`/`UCVTF`/`FLOGB`/`FNEG`/
/// `FABS`/`BFCVT`. Distinguished by `word<20:16>` (`opc`) and `<23:22>` (`size`).
#[inline]
fn decode_65_pred_unary(word: u32, out: &mut Instruction) {
    let size = bits(word, 22, 2);
    let opc = bits(word, 16, 5); // word<20:16>
    let pg = bits(word, 10, 3);
    let zn = bits(word, 5, 5);
    let zd = bits(word, 0, 5);

    // The round-to-integral / reciprocal-exp / sqrt group keeps the element size
    // in <23:22> and is selected purely by <20:16> (`opc`): FRINTN/P/M/Z/A/X/I,
    // FRECPX, FSQRT. Everything else in this sel=101 slot is a precision/integer
    // convert (FCVT*/FCVTZ*/SCVTF/UCVTF/FLOGB/BFCVT/FCVTX), where the element
    // sizes are derived from the full <23:16> opcode.
    if matches!(
        opc,
        0b00000 | 0b00001 | 0b00010 | 0b00011 | 0b00100 | 0b00110 | 0b00111 | 0b01100 | 0b01101
    ) {
        if size == 0 {
            return;
        }
        let a = arr(size);
        let code = match opc {
            0b00000 => Code::SveFrintnZpz,
            0b00001 => Code::SveFrintpZpz,
            0b00010 => Code::SveFrintmZpz,
            0b00011 => Code::SveFrintzZpz,
            0b00100 => Code::SveFrintaZpz,
            0b00110 => Code::SveFrintxZpz,
            0b00111 => Code::SveFrintiZpz,
            0b01100 => Code::SveFrecpxZpz,
            0b01101 => Code::SveFsqrtZpz,
            _ => return,
        };
        out.set(code);
        out.push_operand(zreg(zd, a));
        out.push_operand(preg_q(pg, PredQual::Merging));
        out.push_operand(zreg(zn, a));
        return;
    }

    // The convert sub-block: pick the (src,dst) precision pair from <23:16>.
    decode_65_convert(word, pg, zn, zd, out);
}

/// FP convert sub-block (`0x65`, `<21>=0`, `<15:13>=101`, `<19>=1`). Selects on
/// the full `word<23:16>` opcode to set the right element arrangements.
#[inline]
fn decode_65_convert(word: u32, pg: u32, zn: u32, zd: u32, out: &mut Instruction) {
    // FLOGB has its own layout: <23:22>=00, <21:19>=011, <18:17>=size (1/2/3),
    // <16>=0. Handle it before the precision-pair table below.
    if bits(word, 22, 2) == 0b00 && bits(word, 19, 3) == 0b011 && bit(word, 16) == 0 {
        let sz = bits(word, 17, 2);
        if sz == 0 {
            return;
        }
        let a = arr(sz);
        out.set(Code::SveFlogbZpz);
        out.push_operand(zreg(zd, a));
        out.push_operand(preg_q(pg, PredQual::Merging));
        out.push_operand(zreg(zn, a));
        return;
    }

    // The selector is bits<23:16> (8 bits): size(23:22) : opc ...
    // We match the literal encodings observed in the ARM ARM index.
    let sel = bits(word, 16, 8); // word<23:16>
    // Helper to finish a convert with dst-arr / src-arr.
    macro_rules! conv {
        ($code:expr, $da:expr, $sa:expr) => {{
            out.set($code);
            out.push_operand(zreg(zd, $da));
            out.push_operand(preg_q(pg, PredQual::Merging));
            out.push_operand(zreg(zn, $sa));
            return;
        }};
    }
    match sel {
        // FCVT precision pairs (size bits select direction):
        0b10_001_001 => conv!(Code::SveFcvt, VA::Ss, VA::Sh), // h->s
        0b10_001_000 => conv!(Code::SveFcvt, VA::Sh, VA::Ss), // s->h
        0b11_001_001 => conv!(Code::SveFcvt, VA::Sd, VA::Sh), // h->d
        0b11_001_000 => conv!(Code::SveFcvt, VA::Sh, VA::Sd), // d->h
        0b11_001_011 => conv!(Code::SveFcvt, VA::Sd, VA::Ss), // s->d
        0b11_001_010 => conv!(Code::SveFcvt, VA::Ss, VA::Sd), // d->s
        // BFCVT s->bf16(h): 01100101 10 0010 10 101 -> sel = 10_001_010.
        0b10_001_010 => conv!(Code::SveBfcvt, VA::Sh, VA::Ss),
        // FCVTX d->s rounding to odd (SVE2): 01100101 00 0010 10 101 -> 00_001_010.
        0b00_001_010 => conv!(Code::SveFcvtx, VA::Ss, VA::Sd),
        // FCVTZS:
        0b01_011_010 => conv!(Code::SveFcvtzs, VA::Sh, VA::Sh), // fp16 -> h
        0b01_011_100 => conv!(Code::SveFcvtzs, VA::Ss, VA::Sh), // fp16 -> w
        0b01_011_110 => conv!(Code::SveFcvtzs, VA::Sd, VA::Sh), // fp16 -> x
        0b10_011_100 => conv!(Code::SveFcvtzs, VA::Ss, VA::Ss), // s -> w
        0b11_011_100 => conv!(Code::SveFcvtzs, VA::Sd, VA::Ss), // s -> x
        0b11_011_000 => conv!(Code::SveFcvtzs, VA::Ss, VA::Sd), // d -> w
        0b11_011_110 => conv!(Code::SveFcvtzs, VA::Sd, VA::Sd), // d -> x
        // FCVTZU:
        0b01_011_011 => conv!(Code::SveFcvtzu, VA::Sh, VA::Sh), // fp16 -> h
        0b01_011_101 => conv!(Code::SveFcvtzu, VA::Ss, VA::Sh), // fp16 -> w
        0b01_011_111 => conv!(Code::SveFcvtzu, VA::Sd, VA::Sh), // fp16 -> x
        0b10_011_101 => conv!(Code::SveFcvtzu, VA::Ss, VA::Ss), // s -> w
        0b11_011_101 => conv!(Code::SveFcvtzu, VA::Sd, VA::Ss), // s -> x
        0b11_011_001 => conv!(Code::SveFcvtzu, VA::Ss, VA::Sd), // d -> w
        0b11_011_111 => conv!(Code::SveFcvtzu, VA::Sd, VA::Sd), // d -> x
        // SCVTF:
        0b01_010_010 => conv!(Code::SveScvtf, VA::Sh, VA::Sh), // h -> fp16
        0b01_010_100 => conv!(Code::SveScvtf, VA::Sh, VA::Ss), // w -> fp16
        0b01_010_110 => conv!(Code::SveScvtf, VA::Sh, VA::Sd), // x -> fp16
        0b10_010_100 => conv!(Code::SveScvtf, VA::Ss, VA::Ss), // w -> s
        0b11_010_000 => conv!(Code::SveScvtf, VA::Sd, VA::Ss), // w -> d
        0b11_010_100 => conv!(Code::SveScvtf, VA::Ss, VA::Sd), // x -> s
        0b11_010_110 => conv!(Code::SveScvtf, VA::Sd, VA::Sd), // x -> d
        // UCVTF:
        0b01_010_011 => conv!(Code::SveUcvtf, VA::Sh, VA::Sh), // h -> fp16
        0b01_010_101 => conv!(Code::SveUcvtf, VA::Sh, VA::Ss), // w -> fp16
        0b01_010_111 => conv!(Code::SveUcvtf, VA::Sh, VA::Sd), // x -> fp16
        0b10_010_101 => conv!(Code::SveUcvtf, VA::Ss, VA::Ss), // w -> s
        0b11_010_001 => conv!(Code::SveUcvtf, VA::Sd, VA::Ss), // w -> d
        0b11_010_101 => conv!(Code::SveUcvtf, VA::Ss, VA::Sd), // x -> s
        0b11_010_111 => conv!(Code::SveUcvtf, VA::Sd, VA::Sd), // x -> d
        _ => {}
    }
}

/// FP vector compare (`0x65`, `<21>=0`, `<15:13>` in {010,011,110,111}):
/// `FCMEQ`/`FCMGE`/`FCMGT`/`FCMNE`/`FCMUO`/`FACGE`/`FACGT` (result `<Pd>.<T>`).
/// The compare-with-zero forms live in [`decode_65_unary_misc`] (sel=001).
#[inline]
fn decode_65_compare(word: u32, out: &mut Instruction) {
    let size = bits(word, 22, 2);
    if size == 0 {
        return;
    }
    let a = arr(size);
    let pg = bits(word, 10, 3);
    let zn = bits(word, 5, 5);
    let pd = bits(word, 0, 4);
    let sel = bits(word, 13, 3); // word<15:13>
    let b4 = bit(word, 4); // discriminator within several rows

    // Vector compares: 01100101 size 0 Zm <15:13> Pg Zn b4 Pd.
    let zm = bits(word, 16, 5);
    let code = match (sel, b4) {
        (0b010, 0) => Code::SveFcmgeZz,
        (0b010, 1) => Code::SveFcmgtZz,
        (0b011, 0) => Code::SveFcmeqZz,
        (0b011, 1) => Code::SveFcmneZz,
        (0b110, 0) => Code::SveFcmuoZz,
        (0b110, 1) => Code::SveFacgeZz,
        (0b111, 1) => Code::SveFacgtZz,
        _ => return,
    };
    out.set(code);
    out.push_operand(preg_sz(pd, a));
    out.push_operand(preg_q(pg, PredQual::Zeroing));
    out.push_operand(zreg(zn, a));
    out.push_operand(zreg(zm, a));
}

/// FP multiply-add 4-operand predicated (`0x65`, `<21>=1`):
/// `FMLA`/`FMLS`/`FNMLA`/`FNMLS` (accumulator destructive) and `FMAD`/`FMSB`/
/// `FNMAD`/`FNMSB` (multiplicand destructive). The op is `word<15:13>`.
#[inline]
fn decode_65_fma(word: u32, out: &mut Instruction) {
    let size = bits(word, 22, 2);
    if size == 0 {
        return;
    }
    let a = arr(size);
    let zm = bits(word, 16, 5);
    let pg = bits(word, 10, 3);
    let zn = bits(word, 5, 5);
    let zd = bits(word, 0, 5);
    let op = bits(word, 13, 3); // word<15:13>
    match op {
        // Accumulator-destructive: FMLA/FMLS/FNMLA/FNMLS.
        0b000 => {
            out.set(Code::SveFmlaZpzzz);
            push_fma_acc(out, zd, pg, zn, zm, a);
        }
        0b001 => {
            out.set(Code::SveFmlsZpzzz);
            push_fma_acc(out, zd, pg, zn, zm, a);
        }
        0b010 => {
            out.set(Code::SveFnmlaZpzzz);
            push_fma_acc(out, zd, pg, zn, zm, a);
        }
        0b011 => {
            out.set(Code::SveFnmlsZpzzz);
            push_fma_acc(out, zd, pg, zn, zm, a);
        }
        // Multiplicand-destructive: FMAD/FMSB/FNMAD/FNMSB.
        0b100 => {
            out.set(Code::SveFmadZpzzz);
            push_fma_mul(out, zd, pg, zn, zm, a);
        }
        0b101 => {
            out.set(Code::SveFmsbZpzzz);
            push_fma_mul(out, zd, pg, zn, zm, a);
        }
        0b110 => {
            out.set(Code::SveFnmadZpzzz);
            push_fma_mul(out, zd, pg, zn, zm, a);
        }
        0b111 => {
            out.set(Code::SveFnmsbZpzzz);
            push_fma_mul(out, zd, pg, zn, zm, a);
        }
        _ => {}
    }
}

/// `<Zda>.<T>, <Pg>/M, <Zn>.<T>, <Zm>.<T>` for the accumulator-destructive FMA.
#[inline]
fn push_fma_acc(out: &mut Instruction, zda: u32, pg: u32, zn: u32, zm: u32, a: VA) {
    out.push_operand(zreg(zda, a));
    out.push_operand(preg_q(pg, PredQual::Merging));
    out.push_operand(zreg(zn, a));
    out.push_operand(zreg(zm, a));
}

/// `<Zdn>.<T>, <Pg>/M, <Zm>.<T>, <Za>.<T>` for the multiplicand-destructive FMA.
/// In the `*_z_p_zzz_` multiplicand encodings `Zm` is `<9:5>` and `Za` is
/// `<20:16>` (the addend); the destructive `Zdn` is `<4:0>`.
#[inline]
fn push_fma_mul(out: &mut Instruction, zdn: u32, pg: u32, zm: u32, za: u32, a: VA) {
    out.push_operand(zreg(zdn, a));
    out.push_operand(preg_q(pg, PredQual::Merging));
    out.push_operand(zreg(zm, a));
    out.push_operand(zreg(za, a));
}

// ===========================================================================
// Top byte 0x64 — SVE2 FP: multiply-add indexed, complex, MMLA, bf16/fhm long.
// ===========================================================================

#[inline]
fn decode_64(word: u32, features: FeatureSet, out: &mut Instruction) {
    let b21 = bit(word, 21);
    let b15 = bit(word, 15);

    if b21 == 0 {
        // FCMLA (vector, predicated): 01100100 size 0 Zm 0 rot Pg Zn Zda, with
        // <15>=0.
        if b15 == 0 {
            decode_64_fcmla_vec(word, out);
            return;
        }
        // SVE2.1 FP unary predicated convert/round, ZEROING (`/z`): the marker
        // `<20:19>=11` distinguishes them from the narrow/long converts (`00`) and
        // the qv-reductions / pairwise (`10`). Element width in `<23:22>`.
        if features.has(Feature::Sve2p1) && bits(word, 19, 2) == 0b11 {
            decode_64_pred_unary_z(word, out);
            if !out.is_invalid() {
                return;
            }
        }
        // <15>=1 region: FCADD and the SVE2 FP pairwise group (both
        // <15:13>=100), and the SVE2 narrow/long converts (<15:13>=101).
        if bits(word, 13, 3) == 0b100 {
            // FCADD: 01100100 size 00000 rot 100 Pg Zn Zdn (<20:17>=0000).
            if bits(word, 17, 4) == 0b0000 {
                decode_64_fcadd(word, out);
                return;
            }
            // Pairwise: 01100100 size 10 opc 100 Pg Zm Zdn (<20:19>=10).
            if bits(word, 19, 2) == 0b10 {
                decode_64_pairwise(word, out);
                return;
            }
        }
        if bits(word, 13, 3) == 0b101 {
            // SVE2.1 quadword FP reductions to a NEON `V` register come first;
            // the narrow/long converts share this sel=101 slot.
            decode_64_fp_qv_reduction(word, out);
            if out.is_invalid() {
                decode_64_narrow_convert(word, features, out);
            }
        }
        return;
    }

    // <21>=1 region: indexed multiply-add / multiply, complex indexed, MMLA,
    // bf16 / fhm widening. Dispatch on the structural sub-fields.
    decode_64_indexed_and_long(word, features, out);
}

/// SVE2 FP pairwise (`0x64`, `<21>=0`, `<20:19>=10`, `<15:13>=100`):
/// `op <Zdn>.<T>, <Pg>/M, <Zdn>.<T>, <Zm>.<T>`.
#[inline]
fn decode_64_pairwise(word: u32, out: &mut Instruction) {
    let size = bits(word, 22, 2);
    if size == 0 {
        return;
    }
    let a = arr(size);
    let pg = bits(word, 10, 3);
    let zm = bits(word, 5, 5);
    let zdn = bits(word, 0, 5);
    let code = match bits(word, 16, 3) {
        0b000 => Code::SveFaddpZpzz,
        0b100 => Code::SveFmaxnmpZpzz,
        0b101 => Code::SveFminnmpZpzz,
        0b110 => Code::SveFmaxpZpzz,
        0b111 => Code::SveFminpZpzz,
        _ => return,
    };
    out.set(code);
    push_pred_binary(out, zdn, pg, zm, a);
}

/// SVE2.1 quadword FP reductions (`0x64`, `<21>=0`, `<15:13>=101`,
/// `<20:16>=10000/10100/10101/10110/10111`): `op <Vd>.<T>, <Pg>, <Zn>.<T>`,
/// reducing each 128-bit segment into a NEON `V` register. `.h`/`.s`/`.d` only.
#[inline]
fn decode_64_fp_qv_reduction(word: u32, out: &mut Instruction) {
    let size = bits(word, 22, 2);
    if size == 0 {
        return; // no `.b` FP form.
    }
    let code = match bits(word, 16, 5) {
        0b10000 => Code::SveFaddqv,
        0b10100 => Code::SveFmaxnmqv,
        0b10101 => Code::SveFminnmqv,
        0b10110 => Code::SveFmaxqv,
        0b10111 => Code::SveFminqv,
        _ => return,
    };
    let pg = bits(word, 10, 3);
    let zn = bits(word, 5, 5);
    let vd = bits(word, 0, 5);
    out.set(code);
    out.push_operand(vreg(vd, va_neon(size)));
    out.push_operand(preg(pg));
    out.push_operand(zreg(zn, arr(size)));
}

/// SVE2 narrow / long FP converts (`0x64`, `<21>=0`, `<15:13>=101`):
/// `FCVTNT`/`FCVTLT`/`FCVTXNT`/`BFCVTNT` `<Zd>.<T>, <Pg>/M, <Zn>.<T>`. Selected
/// by the `word<23:16>` opcode (predicated, like the 0x65 converts).
// The 8-bit opcode literals are grouped `size(2)_opc(4)_size2(2)` to mirror the
// ARM ARM `word<23:16>` field layout, not into uniform nibbles.
#[allow(clippy::unusual_byte_groupings)]
#[inline]
fn decode_64_narrow_convert(word: u32, features: FeatureSet, out: &mut Instruction) {
    let pg = bits(word, 10, 3);
    let zn = bits(word, 5, 5);
    let zd = bits(word, 0, 5);
    macro_rules! conv {
        ($code:expr, $da:expr, $sa:expr) => {{
            out.set($code);
            out.push_operand(zreg(zd, $da));
            out.push_operand(preg_q(pg, PredQual::Merging));
            out.push_operand(zreg(zn, $sa));
            return;
        }};
    }
    match bits(word, 16, 8) {
        // FCVTNT (narrow, top of pairs): s->h and d->s.
        0b10_0010_00 => conv!(Code::SveFcvtnt, VA::Sh, VA::Ss), // s -> h
        0b11_0010_10 => conv!(Code::SveFcvtnt, VA::Ss, VA::Sd), // d -> s
        // FCVTLT (long, top of pairs): h->s and s->d.
        0b10_0010_01 => conv!(Code::SveFcvtlt, VA::Ss, VA::Sh), // h -> s
        0b11_0010_11 => conv!(Code::SveFcvtlt, VA::Sd, VA::Ss), // s -> d
        // FCVTXNT (narrow rounding-to-odd, top of pairs): d->s.
        0b00_0010_10 => conv!(Code::SveFcvtxnt, VA::Ss, VA::Sd),
        // BFCVTNT (convert to BFloat16, top of pairs): s->bf16(h).
        0b10_0010_10 => {
            if !features.has(Feature::Bf16) {
                return;
            }
            conv!(Code::SveBfcvtnt, VA::Sh, VA::Ss);
        }
        _ => {}
    }
}

/// SVE2.1 FP unary predicated convert/round, ZEROING `/z` (`0x64`, `<21>=0`,
/// `<20:19>=11`, `<15>=1`). These are the `/z` analogues of the existing `0x65`
/// merging (`/m`) forms but with a wholly different opcode layout: the operation
/// is selected by `size(<23:22>) : opc(<18:16>) : sel(<15:13>)`, and the
/// governing predicate is zeroing. The round-to-integral / reciprocal-exp / sqrt
/// group keeps the element size in `<23:22>` (selected by `opc:sel` with
/// `opc` in {0,1,3}); everything else is a precision/integer convert keyed on the
/// full 8-bit `<23:22>:<18:16>:<15:13>` opcode.
#[inline]
fn decode_64_pred_unary_z(word: u32, out: &mut Instruction) {
    let size = bits(word, 22, 2);
    let opc = bits(word, 16, 3); // <18:16>
    let sel = bits(word, 13, 3); // <15:13>
    let pg = bits(word, 10, 3);
    let zn = bits(word, 5, 5);
    let zd = bits(word, 0, 5);

    // Round-to-integral / FRECPX / FSQRT: opc in {0,1,3}, element size in <23:22>.
    if matches!(opc, 0b000 | 0b001 | 0b011) {
        let code = match (opc, sel) {
            (0b000, 0b100) => Code::SveFrintnZ,
            (0b000, 0b101) => Code::SveFrintpZ,
            (0b000, 0b110) => Code::SveFrintmZ,
            (0b000, 0b111) => Code::SveFrintzZ,
            (0b001, 0b100) => Code::SveFrintaZ,
            (0b001, 0b110) => Code::SveFrintxZ,
            (0b001, 0b111) => Code::SveFrintiZ,
            (0b011, 0b100) => Code::SveFrecpxZ,
            (0b011, 0b101) => Code::SveFsqrtZ,
            _ => return,
        };
        if size == 0 {
            return; // no `.b` FP form.
        }
        let a = arr(size);
        out.set(code);
        out.push_operand(zreg(zd, a));
        out.push_operand(preg_q(pg, PredQual::Zeroing));
        out.push_operand(zreg(zn, a));
        return;
    }

    // Convert sub-block: select on the full 8-bit opcode `size:opc:sel`.
    let full = (size << 6) | (opc << 3) | sel;
    macro_rules! conv {
        ($code:expr, $da:expr, $sa:expr) => {{
            out.set($code);
            out.push_operand(zreg(zd, $da));
            out.push_operand(preg_q(pg, PredQual::Zeroing));
            out.push_operand(zreg(zn, $sa));
            return;
        }};
    }
    match full {
        // FCVT precision pairs.
        0b10_010_100 => conv!(Code::SveFcvtZ, VA::Sh, VA::Ss),
        0b10_010_101 => conv!(Code::SveFcvtZ, VA::Ss, VA::Sh),
        0b11_010_100 => conv!(Code::SveFcvtZ, VA::Sh, VA::Sd),
        0b11_010_101 => conv!(Code::SveFcvtZ, VA::Sd, VA::Sh),
        0b11_010_110 => conv!(Code::SveFcvtZ, VA::Ss, VA::Sd),
        0b11_010_111 => conv!(Code::SveFcvtZ, VA::Sd, VA::Ss),
        // FCVTX (round-to-odd, d->s).
        0b00_010_110 => conv!(Code::SveFcvtxZ, VA::Ss, VA::Sd),
        // BFCVT (s->bf16(h)).
        0b10_010_110 => conv!(Code::SveBfcvtZ, VA::Sh, VA::Ss),
        // FLOGB (element size in `sel`: h/s/d).
        0b00_110_101 => conv!(Code::SveFlogbZ, VA::Sh, VA::Sh),
        0b00_110_110 => conv!(Code::SveFlogbZ, VA::Ss, VA::Ss),
        0b00_110_111 => conv!(Code::SveFlogbZ, VA::Sd, VA::Sd),
        // SCVTF.
        0b01_100_110 => conv!(Code::SveScvtfZ, VA::Sh, VA::Sh),
        0b01_101_100 => conv!(Code::SveScvtfZ, VA::Sh, VA::Ss),
        0b01_101_110 => conv!(Code::SveScvtfZ, VA::Sh, VA::Sd),
        0b10_101_100 => conv!(Code::SveScvtfZ, VA::Ss, VA::Ss),
        0b11_100_100 => conv!(Code::SveScvtfZ, VA::Sd, VA::Ss),
        0b11_101_100 => conv!(Code::SveScvtfZ, VA::Ss, VA::Sd),
        0b11_101_110 => conv!(Code::SveScvtfZ, VA::Sd, VA::Sd),
        // UCVTF.
        0b01_100_111 => conv!(Code::SveUcvtfZ, VA::Sh, VA::Sh),
        0b01_101_101 => conv!(Code::SveUcvtfZ, VA::Sh, VA::Ss),
        0b01_101_111 => conv!(Code::SveUcvtfZ, VA::Sh, VA::Sd),
        0b10_101_101 => conv!(Code::SveUcvtfZ, VA::Ss, VA::Ss),
        0b11_100_101 => conv!(Code::SveUcvtfZ, VA::Sd, VA::Ss),
        0b11_101_101 => conv!(Code::SveUcvtfZ, VA::Ss, VA::Sd),
        0b11_101_111 => conv!(Code::SveUcvtfZ, VA::Sd, VA::Sd),
        // FCVTZS.
        0b01_110_110 => conv!(Code::SveFcvtzsZ, VA::Sh, VA::Sh),
        0b01_111_100 => conv!(Code::SveFcvtzsZ, VA::Ss, VA::Sh),
        0b01_111_110 => conv!(Code::SveFcvtzsZ, VA::Sd, VA::Sh),
        0b10_111_100 => conv!(Code::SveFcvtzsZ, VA::Ss, VA::Ss),
        0b11_111_100 => conv!(Code::SveFcvtzsZ, VA::Sd, VA::Ss),
        0b11_110_100 => conv!(Code::SveFcvtzsZ, VA::Ss, VA::Sd),
        0b11_111_110 => conv!(Code::SveFcvtzsZ, VA::Sd, VA::Sd),
        // FCVTZU.
        0b01_110_111 => conv!(Code::SveFcvtzuZ, VA::Sh, VA::Sh),
        0b01_111_101 => conv!(Code::SveFcvtzuZ, VA::Ss, VA::Sh),
        0b01_111_111 => conv!(Code::SveFcvtzuZ, VA::Sd, VA::Sh),
        0b10_111_101 => conv!(Code::SveFcvtzuZ, VA::Ss, VA::Ss),
        0b11_111_101 => conv!(Code::SveFcvtzuZ, VA::Sd, VA::Ss),
        0b11_110_101 => conv!(Code::SveFcvtzuZ, VA::Ss, VA::Sd),
        0b11_111_111 => conv!(Code::SveFcvtzuZ, VA::Sd, VA::Sd),
        _ => {}
    }
}

/// FCADD (`0x64`, predicated complex add):
/// `FCADD <Zdn>.<T>, <Pg>/M, <Zdn>.<T>, <Zm>.<T>, #rot` (rot = 90 or 270).
#[inline]
fn decode_64_fcadd(word: u32, out: &mut Instruction) {
    let size = bits(word, 22, 2);
    if size == 0 {
        return;
    }
    let a = arr(size);
    let rot = bit(word, 16); // 0 => 90, 1 => 270
    let pg = bits(word, 10, 3);
    let zm = bits(word, 5, 5);
    let zdn = bits(word, 0, 5);
    out.set(Code::SveFcadd);
    out.push_operand(zreg(zdn, a));
    out.push_operand(preg_q(pg, PredQual::Merging));
    out.push_operand(zreg(zdn, a));
    out.push_operand(zreg(zm, a));
    out.push_operand(Operand::ImmUnsigned(if rot == 0 { 90 } else { 270 }));
}

/// FCMLA (`0x64`, predicated complex multiply-add):
/// `FCMLA <Zda>.<T>, <Pg>/M, <Zn>.<T>, <Zm>.<T>, #rot` (rot in {0,90,180,270}).
#[inline]
fn decode_64_fcmla_vec(word: u32, out: &mut Instruction) {
    let size = bits(word, 22, 2);
    if size == 0 {
        return;
    }
    let a = arr(size);
    let zm = bits(word, 16, 5);
    let rot = bits(word, 13, 2); // word<14:13>
    let pg = bits(word, 10, 3);
    let zn = bits(word, 5, 5);
    let zda = bits(word, 0, 5);
    out.set(Code::SveFcmlaZpzzz);
    out.push_operand(zreg(zda, a));
    out.push_operand(preg_q(pg, PredQual::Merging));
    out.push_operand(zreg(zn, a));
    out.push_operand(zreg(zm, a));
    out.push_operand(Operand::ImmUnsigned((rot * 90) as u64));
}

/// The `0x64`, `<21>=1` region: FMLA/FMLS/FMUL by indexed element, FCMLA by
/// index, FMMLA/BFMMLA, BFDOT, and the bf16 / half multiply-add-long family.
#[inline]
fn decode_64_indexed_and_long(word: u32, features: FeatureSet, out: &mut Instruction) {
    let opc2322 = bits(word, 22, 2); // size / opc<1:0>
    let sub = bits(word, 10, 6); // word<15:10>

    // FMMLA / BFMMLA: <15:10> = 111001. opc<23:22> selects: 01 = BFMMLA (bf16),
    // 10 = FMMLA .s, 11 = FMMLA .d.
    if sub == 0b111001 {
        let zm = bits(word, 16, 5);
        let zn = bits(word, 5, 5);
        let zda = bits(word, 0, 5);
        match opc2322 {
            0b01 => {
                if !features.has(Feature::Bf16) {
                    return;
                }
                out.set(Code::SveBfmmla);
                out.push_operand(zreg(zda, VA::Ss));
                out.push_operand(zreg(zn, VA::Sh));
                out.push_operand(zreg(zm, VA::Sh));
            }
            0b10 => {
                out.set(Code::SveFmmla);
                out.push_operand(zreg(zda, VA::Ss));
                out.push_operand(zreg(zn, VA::Ss));
                out.push_operand(zreg(zm, VA::Ss));
            }
            0b11 => {
                out.set(Code::SveFmmla);
                out.push_operand(zreg(zda, VA::Sd));
                out.push_operand(zreg(zn, VA::Sd));
                out.push_operand(zreg(zm, VA::Sd));
            }
            _ => {}
        }
        return;
    }

    // FMLA/FMLS by indexed element: <15:11>=00000, op=<10> (0=FMLA, 1=FMLS).
    // FMUL by indexed element: <15:10>=001000.
    if bits(word, 11, 5) == 0b00000 {
        decode_64_fmla_indexed(word, bit(word, 10) == 1, out);
        return;
    }
    if bits(word, 10, 6) == 0b001000 {
        decode_64_fmul_indexed(word, out);
        return;
    }

    // FCMLA by indexed element: 01100100 1x1 Zm 0001 rot Zn Zda, <15:12>=0001.
    if bits(word, 12, 4) == 0b0001 {
        decode_64_fcmla_indexed(word, out);
        return;
    }

    // BFDOT (vector / indexed) and bf16 / half multiply-add-long.
    decode_64_dot_and_mlal(word, features, out);
}

/// FMLA/FMLS by indexed element (`0x64`, `<21>=1`, `<15:13>=000`).
///
/// The element size and index are encoded differently per width:
/// * `.h` (`<23:22>=0x`): `Zm` is `<18:16>` (3 bits), index = `i3h:i3l` =
///   `<22>:<20:19>` style — here index = `<22>` concat `<20:19>`? We follow the
///   ARM ARM `fmla_z_zzzi_h`: index `i3h(<22>)`:`i3l? `. In practice the corpus
///   uses `Zm<3 bits>` and a 3-bit index.
/// * `.s` (`<23:22>=10`): `Zm` is `<18:16>`, index = `<20:19>` (2 bits).
/// * `.d` (`<23:22>=11`): `Zm` is `<19:16>`, index = `<20>` (1 bit).
#[inline]
fn decode_64_fmla_indexed(word: u32, is_fmls: bool, out: &mut Instruction) {
    let zn = bits(word, 5, 5);
    let zda = bits(word, 0, 5);
    let sz = bits(word, 22, 2);
    let (a, zm, idx) = match sz {
        0b11 => {
            // .d : Zm<19:16>, index = i1<20>.
            (VA::Sd, bits(word, 16, 4), bit(word, 20))
        }
        0b10 => {
            // .s : Zm<18:16>, index = i2<20:19>.
            (VA::Ss, bits(word, 16, 3), bits(word, 19, 2))
        }
        _ => {
            // .h : Zm<18:16>, index = i3h:i3l = <22>:<20:19>.
            (VA::Sh, bits(word, 16, 3), (bit(word, 22) << 2) | bits(word, 19, 2))
        }
    };
    out.set(if is_fmls { Code::SveFmlsIdx } else { Code::SveFmlaIdx });
    out.push_operand(zreg(zda, a));
    out.push_operand(zreg(zn, a));
    out.push_operand(zreg_idx(zm, a, idx as u8));
}

/// FMUL by indexed element (`0x64`, `<21>=1`, `<15:10>=001000`).
#[inline]
fn decode_64_fmul_indexed(word: u32, out: &mut Instruction) {
    let zn = bits(word, 5, 5);
    let zd = bits(word, 0, 5);
    let sz = bits(word, 22, 2);
    let (a, zm, idx) = match sz {
        0b11 => (VA::Sd, bits(word, 16, 4), bit(word, 20)),
        0b10 => (VA::Ss, bits(word, 16, 3), bits(word, 19, 2)),
        _ => (VA::Sh, bits(word, 16, 3), (bit(word, 22) << 2) | bits(word, 19, 2)),
    };
    out.set(Code::SveFmulIdx);
    out.push_operand(zreg(zd, a));
    out.push_operand(zreg(zn, a));
    out.push_operand(zreg_idx(zm, a, idx as u8));
}

/// FCMLA by indexed element (`0x64`, `<15:12>=0001`):
/// `FCMLA <Zda>.<T>, <Zn>.<T>, <Zm>.<T>[<imm>], #rot`. Only `.h` (`<23:22>=10`)
/// and `.s` (`<23:22>=11`) exist; the index/Zm widths differ accordingly.
#[inline]
fn decode_64_fcmla_indexed(word: u32, out: &mut Instruction) {
    let rot = bits(word, 10, 2); // word<11:10>
    let zn = bits(word, 5, 5);
    let zda = bits(word, 0, 5);
    let sz = bits(word, 22, 2);
    let (a, zm, idx) = match sz {
        0b10 => {
            // .h : Zm<18:16>, index = i2<20:19>.
            (VA::Sh, bits(word, 16, 3), bits(word, 19, 2))
        }
        0b11 => {
            // .s : Zm<19:16>, index = i1<20>.
            (VA::Ss, bits(word, 16, 4), bit(word, 20))
        }
        _ => return,
    };
    out.set(Code::SveFcmlaIdx);
    out.push_operand(zreg(zda, a));
    out.push_operand(zreg(zn, a));
    out.push_operand(zreg_idx(zm, a, idx as u8));
    out.push_operand(Operand::ImmUnsigned((rot * 90) as u64));
}

/// BFDOT, BFMLALB/T, FMLALB/T, FMLSLB/T (vector and indexed), all in the
/// `0x64`, `<21>=1` widening region. Element types are fixed (`.s` <- `.h`).
#[inline]
fn decode_64_dot_and_mlal(word: u32, features: FeatureSet, out: &mut Instruction) {
    let zda = bits(word, 0, 5);
    let zn = bits(word, 5, 5);

    // BFDOT: <23:16>? per index `011001000 op=1 1 Zm 100000` vector and
    //   `011001000 op=1 1 i2 Zm 010000` indexed. Detect via <15:10>.
    let sub = bits(word, 10, 6);
    let opc = bits(word, 22, 2); // <23:22>

    // ---- FEAT_SVE_B16B16 BF16 multiply-add / multiply, indexed ----
    // `<Zda>.h, <Zn>.h, <Zm>.h[i]`. Zm restricted to z0..z7 (<18:16>), index =
    // i3h(<22>):i2l(<20:19>) (3-bit), `<23>==0`, `<21>==1`. The mnemonic is
    // chosen by `<15:10>`: 000010 BFMLA, 000011 BFMLS, 001010 BFMUL.
    if bit(word, 23) == 0 {
        let bf_code = match sub {
            0b000010 => Some(Code::SveBfmlaIdx),
            0b000011 => Some(Code::SveBfmlsIdx),
            0b001010 => Some(Code::SveBfmulIdx),
            _ => None,
        };
        if let Some(code) = bf_code {
            if !features.has(Feature::Bf16) {
                return;
            }
            let zm = bits(word, 16, 3);
            let idx = (bit(word, 22) << 2) | bits(word, 19, 2);
            out.set(code);
            out.push_operand(zreg(zda, VA::Sh));
            out.push_operand(zreg(zn, VA::Sh));
            out.push_operand(zreg_idx(zm, VA::Sh, idx as u8));
            return;
        }
    }

    // ---- FEAT_SSVE_FP8FMA FP8 widening multiply-add long, indexed z-form ----
    // The two index bits il live in <11:10>, so key on the opcode field <15:12>
    // (not the full <15:10>, which carries il).
    // FMLALB/FMLALT (to `.h`): `<15:12>==0101`, `<22>==0`, T=<23>, Zm z0..z7
    //   (<18:16>), index = ih(<20:19>):il(<11:10>) (4-bit).
    // FMLALLBB/BT/TB/TT (to `.s`): `<15:12>==1100`, B/T pair in <23:22>, same
    //   Zm/index layout.
    let op1512 = bits(word, 12, 4);
    if op1512 == 0b0101 && bit(word, 22) == 0 {
        if !features.has(Feature::Fp8) {
            return;
        }
        let zm = bits(word, 16, 3);
        let idx = (bits(word, 19, 2) << 2) | bits(word, 10, 2);
        let code = if bit(word, 23) == 0 { Code::SveFmlalbFp8Idx } else { Code::SveFmlaltFp8Idx };
        out.set(code);
        out.push_operand(zreg(zda, VA::Sh));
        out.push_operand(zreg(zn, VA::Sb));
        out.push_operand(zreg_idx(zm, VA::Sb, idx as u8));
        return;
    }
    if op1512 == 0b1100 {
        if !features.has(Feature::Fp8) {
            return;
        }
        let zm = bits(word, 16, 3);
        let idx = (bits(word, 19, 2) << 2) | bits(word, 10, 2);
        let code = match bits(word, 22, 2) {
            0b00 => Code::SveFmlallbbFp8Idx,
            0b01 => Code::SveFmlallbtFp8Idx,
            0b10 => Code::SveFmlalltbFp8Idx,
            _ => Code::SveFmlallttFp8Idx,
        };
        out.set(code);
        out.push_operand(zreg(zda, VA::Ss));
        out.push_operand(zreg(zn, VA::Sb));
        out.push_operand(zreg_idx(zm, VA::Sb, idx as u8));
        return;
    }

    // BFDOT vector: <23:22>=01, <15:10>=100000.
    if opc == 0b01 && sub == 0b100000 {
        if !features.has(Feature::Bf16) {
            return;
        }
        let zm = bits(word, 16, 5);
        out.set(Code::SveBfdot);
        out.push_operand(zreg(zda, VA::Ss));
        out.push_operand(zreg(zn, VA::Sh));
        out.push_operand(zreg(zm, VA::Sh));
        return;
    }
    // BFDOT indexed: <23:22>=01, <15:12>=0100 (then i2 in <20:19>, Zm<18:16>).
    if opc == 0b01 && bits(word, 12, 4) == 0b0100 {
        if !features.has(Feature::Bf16) {
            return;
        }
        let zm = bits(word, 16, 3);
        let idx = bits(word, 19, 2);
        out.set(Code::SveBfdotIdx);
        out.push_operand(zreg(zda, VA::Ss));
        out.push_operand(zreg(zn, VA::Sh));
        out.push_operand(zreg_idx(zm, VA::Sh, idx as u8));
        return;
    }

    // The half / BFloat16 multiply-add-long family (all `.s <- .h, .h`):
    //   vector : 011001001 o2 1 Zm     10 op 00 T Zn Zda  (<15:14>=10, <12:11>=00)
    //   indexed: 011001001 o2 1 i3h Zm 01 op 0 i3l T Zn Zda (<15:14>=01, <12>=0)
    // with o2=<22> (0 => F16 FMLAL/FMLSL, 1 => BFMLAL), op=<13>, T=<10>.
    let o2 = bit(word, 22);
    let bf16 = o2 == 1;
    if bf16 && !features.has(Feature::Bf16) {
        return;
    }
    let op = bit(word, 13);

    // Vector long-MLA: <15:14>=10, <12:11>=00, T=<10>.
    if bits(word, 14, 2) == 0b10 && bits(word, 11, 2) == 0b00 {
        let t = bit(word, 10);
        let zm = bits(word, 16, 5);
        let Some(code) = mlal_code(bf16, op, t, false) else { return };
        out.set(code);
        out.push_operand(zreg(zda, VA::Ss));
        out.push_operand(zreg(zn, VA::Sh));
        out.push_operand(zreg(zm, VA::Sh));
        return;
    }

    // Indexed long-MLA: <15:14>=01, <12>=0, index = i3h(<20:19>):i3l(<11>),
    // T=<10>, Zm=<18:16> (3-bit).
    if bits(word, 14, 2) == 0b01 && bit(word, 12) == 0 {
        let t = bit(word, 10);
        let i3l = bit(word, 11);
        let zm = bits(word, 16, 3);
        let i3h = bits(word, 19, 2);
        let idx = (i3h << 1) | i3l;
        let Some(code) = mlal_code(bf16, op, t, true) else { return };
        out.set(code);
        out.push_operand(zreg(zda, VA::Ss));
        out.push_operand(zreg(zn, VA::Sh));
        out.push_operand(zreg_idx(zm, VA::Sh, idx as u8));
    }
}

/// Map `(bf16, op, T, indexed)` to the MLAL-long [`Code`]. `op==0` is the
/// add (`*MLAL*`), `op==1` the subtract (`*MLSL*`, F16 only); `T` selects
/// bottom(0)/top(1).
#[inline]
fn mlal_code(bf16: bool, op: u32, t: u32, indexed: bool) -> Option<Code> {
    Some(match (bf16, op, t, indexed) {
        // BFloat16: only MLAL (op must be 0).
        (true, 0, 0, false) => Code::SveBfmlalb,
        (true, 0, 1, false) => Code::SveBfmlalt,
        (true, 0, 0, true) => Code::SveBfmlalbIdx,
        (true, 0, 1, true) => Code::SveBfmlaltIdx,
        // Half-precision FMLAL / FMLSL.
        (false, 0, 0, false) => Code::SveFmlalb,
        (false, 0, 1, false) => Code::SveFmlalt,
        (false, 1, 0, false) => Code::SveFmlslb,
        (false, 1, 1, false) => Code::SveFmlslt,
        (false, 0, 0, true) => Code::SveFmlalbIdx,
        (false, 0, 1, true) => Code::SveFmlaltIdx,
        (false, 1, 0, true) => Code::SveFmlslbIdx,
        (false, 1, 1, true) => Code::SveFmlsltIdx,
        _ => return None,
    })
}

// ===========================================================================
// FP leaves that live outside the 011 quadrant (called from super::decode).
// ===========================================================================

/// FP unary / select forms in the `000` quadrant (top byte `0x04`):
/// `FABS`/`FNEG` (predicated unary), `FEXPA`, `FTSSEL`. Returns having set `out`
/// only when the word matches one of these; otherwise leaves it untouched so the
/// caller can fall through to the integer / permute decoders.
#[inline]
pub fn decode_fp_misc_04(word: u32, features: FeatureSet, out: &mut Instruction) {
    if !features.has(Feature::Sve) {
        return;
    }
    let size = bits(word, 22, 2);
    if size == 0 {
        return;
    }
    let a = arr(size);
    let zd = bits(word, 0, 5);
    let zn = bits(word, 5, 5);

    // FABS/FNEG predicated (`<15:13>=101`): operation in `<19:16>` (1100/1101),
    // with `<20>` selecting merging (`/m`, SVE) vs zeroing (`/z`, FEAT_SVE2p1).
    if bit(word, 21) == 0 && bits(word, 13, 3) == 0b101 {
        let op4 = bits(word, 16, 4);
        let merging = bit(word, 20) == 1;
        let pg = bits(word, 10, 3);
        let code = match op4 {
            0b1100 => Code::SveFabsZpz,
            0b1101 => Code::SveFnegZpz,
            _ => return,
        };
        out.set(code);
        out.push_operand(zreg(zd, a));
        out.push_operand(preg_q(pg, if merging { PredQual::Merging } else { PredQual::Zeroing }));
        out.push_operand(zreg(zn, a));
        return;
    }

    // FEXPA: 00000100 size 100000 101110 Zn Zd  (<21>=1, <20:16>=00000,
    //   <15:10>=101110).
    if bit(word, 21) == 1 && bits(word, 16, 5) == 0b00000 && bits(word, 10, 6) == 0b101110 {
        out.set(Code::SveFexpa);
        out.push_operand(zreg(zd, a));
        out.push_operand(zreg(zn, a));
        return;
    }

    // FTSSEL: 00000100 size 1 Zm 101100 Zn Zd  (<21>=1, <15:10>=101100).
    if bit(word, 21) == 1 && bits(word, 10, 6) == 0b101100 {
        let zm = bits(word, 16, 5);
        out.set(Code::SveFtssel);
        out.push_operand(zreg(zd, a));
        out.push_operand(zreg(zn, a));
        out.push_operand(zreg(zm, a));
    }
}

/// FCPY (FP copy immediate, predicated) in the `0x05` top byte:
/// `FCPY <Zd>.<T>, <Pg>/M, #const` (rendered as `FMOV` per the corpus).
/// Encoding `00000101 size 01 Pg 110 imm8 Zd`: `<21:20>=01`, `Pg=<19:16>`,
/// `<15:13>=110`.
#[inline]
pub fn decode_fcpy_05(word: u32, features: FeatureSet, out: &mut Instruction) {
    if !features.has(Feature::Sve) {
        return;
    }
    let size = bits(word, 22, 2);
    if size == 0 {
        return;
    }
    // <21:20>=01 and <15:13>=110.
    if bits(word, 20, 2) != 0b01 || bits(word, 13, 3) != 0b110 {
        return;
    }
    let a = arr(size);
    let pg = bits(word, 16, 4); // word<19:16> (governing predicate, /M)
    let imm8 = bits(word, 5, 8);
    let n = match size {
        1 => 16,
        2 => 32,
        _ => 64,
    };
    out.set(Code::SveFcpy);
    out.set_mnemonic(Mnemonic::Fmov);
    out.push_operand(zreg(zd_of(word), a));
    out.push_operand(preg_q(pg, PredQual::Merging));
    out.push_operand(fpimm_for(imm8, n));
}

/// FDUP (broadcast FP immediate) in the `0x25` top byte:
/// `FDUP <Zd>.<T>, #const` (rendered as `FMOV`). Encoding
/// `00100101 size 111001 110 imm8 Zd`.
#[inline]
pub fn decode_fdup_25(word: u32, features: FeatureSet, out: &mut Instruction) {
    if !features.has(Feature::Sve) {
        return;
    }
    let size = bits(word, 22, 2);
    if size == 0 {
        return;
    }
    if bits(word, 16, 6) != 0b111001 || bits(word, 13, 3) != 0b110 {
        return;
    }
    let a = arr(size);
    let imm8 = bits(word, 5, 8);
    let n = match size {
        1 => 16,
        2 => 32,
        _ => 64,
    };
    out.set(Code::SveFdup);
    out.set_mnemonic(Mnemonic::Fmov);
    out.push_operand(zreg(zd_of(word), a));
    out.push_operand(fpimm_for(imm8, n));
}

/// The destination `Zd` field (`word<4:0>`), shared by the immediate-move forms.
#[inline]
fn zd_of(word: u32) -> u32 {
    bits(word, 0, 5)
}

/// Build an FP-immediate operand from an 8-bit `VFPExpandImm` encoding at element
/// width `n` (16/32/64). The 16/64-bit bit patterns are widened/narrowed to the
/// `f32` carried by [`Operand::FpImm`] without FP arithmetic (the formatter
/// renders the shortest exact decimal, matching Binary Ninja).
#[inline]
fn fpimm_for(imm8: u32, n: u32) -> Operand {
    let raw = vfp_expand_imm(imm8, n);
    let val = match n {
        16 => f16_bits_to_f32(raw as u16),
        32 => f32::from_bits(raw as u32),
        // 64-bit: the VFP 8-bit immediate set is a small exact set; convert the
        // f64 bit pattern to f32 (lossless for this immediate set's magnitudes).
        _ => f64::from_bits(raw) as f32,
    };
    Operand::FpImm(val)
}

/// Expand an IEEE half-precision bit pattern to `f32` (no `std`, exact). Only the
/// VFP 8-bit immediate subset is needed, but this handles the full normal /
/// subnormal / inf / nan range so it is total.
#[inline]
fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // Subnormal: normalize.
            let mut e = -1i32;
            let mut m = mant;
            loop {
                e += 1;
                m <<= 1;
                if m & 0x400 != 0 {
                    break;
                }
            }
            let exp32 = (127 - 15 - e) as u32;
            (sign << 31) | (exp32 << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        // Inf / NaN.
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        let exp32 = exp + (127 - 15);
        (sign << 31) | (exp32 << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

// ---------------------------------------------------------------------------
// FP immediate constants used by the FADD/FSUB/FMUL/FMAX/.. #const forms.
// These are raw f32 bit patterns (the formatter bit-casts and renders shortest
// decimal). Values: 0.0, 0.5, 1.0, 2.0.
// ---------------------------------------------------------------------------

const ZERO: u64 = 0x0000_0000; // 0.0f
const HALF: u64 = 0x3f00_0000; // 0.5f
const ONE: u64 = 0x3f80_0000; // 1.0f
const TWO: u64 = 0x4000_0000; // 2.0f

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::FeatureSet;
    use crate::format::{FmtFormatter, Formatter};

    fn feats() -> FeatureSet {
        // Accept everything (SVE + BF16 + FP16 included).
        FeatureSet::ALL
    }

    /// Decode `word` and assert it renders to `expected` (zero-alloc, no_std).
    #[track_caller]
    fn check(word: u32, expected: &str) {
        let mut insn = Instruction::default();
        crate::decode::decode_into(word, 0x1000, feats(), &mut insn);
        let mut buf = [0u8; 128];
        let mut sink = crate::format::BufSink::new(&mut buf);
        FmtFormatter::new().format(&insn, &mut sink);
        assert_eq!(sink.as_str(), expected, "word={word:#010x}");
    }

    #[test]
    fn fadd_unpred() {
        // 65900036 fadd z22.s, z1.s, z16.s
        check(0x65900036, "fadd    z22.s, z1.s, z16.s");
    }

    #[test]
    fn fadd_pred() {
        // 65C08830 fadd z16.d, p2/m, z16.d, z1.d
        check(0x65C08830, "fadd    z16.d, p2/m, z16.d, z1.d");
    }

    #[test]
    fn fadd_imm() {
        // 65D89021 fadd z1.d, p4/m, z1.d, #1.0
        check(0x65D89021, "fadd    z1.d, p4/m, z1.d, #1.0");
    }

    #[test]
    fn fmul_imm_two() {
        // 655A9414 fmul z20.h, p5/m, z20.h, #0.5
        check(0x655A9414, "fmul    z20.h, p5/m, z20.h, #0.5");
    }

    #[test]
    fn fmla_pred() {
        // 65AD10BC fmla z28.s, p4/m, z5.s, z13.s
        check(0x65AD10BC, "fmla    z28.s, p4/m, z5.s, z13.s");
    }

    #[test]
    fn fmad_pred() {
        // 65F29337 fmad z23.d, p4/m, z25.d, z18.d
        check(0x65F29337, "fmad    z23.d, p4/m, z25.d, z18.d");
    }

    #[test]
    fn fcvtzs_d2x() {
        // 65DEB3F7 fcvtzs z23.d, p4/m, z31.d
        check(0x65DEB3F7, "fcvtzs  z23.d, p4/m, z31.d");
    }

    #[test]
    fn scvtf_w2s() {
        // 6594B1AF scvtf z15.s, p4/m, z13.s
        check(0x6594B1AF, "scvtf   z15.s, p4/m, z13.s");
    }

    #[test]
    fn fcmeq_zz() {
        // 659B7E0C fcmeq p12.s, p7/z, z16.s, z27.s
        check(0x659B7E0C, "fcmeq   p12.s, p7/z, z16.s, z27.s");
    }

    #[test]
    fn fcmeq_zero() {
        // 65D23DC8 fcmeq p8.d, p7/z, z14.d, #0.0
        check(0x65D23DC8, "fcmeq   p8.d, p7/z, z14.d, #0.0");
    }

    #[test]
    fn faddv_reduction() {
        // 65402324 faddv h4, p0, z25.h
        check(0x65402324, "faddv   h4, p0, z25.h");
    }

    #[test]
    fn fadda_reduction() {
        // 655832AB fadda h11, p4, h11, z21.h
        check(0x655832AB, "fadda   h11, p4, h11, z21.h");
    }

    #[test]
    fn frecpe_unary() {
        // 654E3147 frecpe z7.h, z10.h
        check(0x654E3147, "frecpe  z7.h, z10.h");
    }

    #[test]
    fn fsqrt_pred() {
        // 658DAB21 fsqrt z1.s, p2/m, z25.s
        check(0x658DAB21, "fsqrt   z1.s, p2/m, z25.s");
    }

    #[test]
    fn fabs_pred() {
        // 045CBAF7 fabs z23.h, p6/m, z23.h
        check(0x045CBAF7, "fabs    z23.h, p6/m, z23.h");
    }

    #[track_caller]
    fn rt(word: u32) {
        let bytes = word.to_le_bytes();
        let mut dec = crate::Decoder::new(&bytes, 0x1000, crate::DecoderOptions::default());
        let insn = dec.decode();
        assert!(!insn.is_invalid(), "decode left Invalid: word={word:#010x}");
        let enc = crate::encode::encode(&insn).expect("encode failed");
        assert_eq!(enc, word, "round-trip mismatch word={word:#010x}");
    }

    #[test]
    fn fabs_fneg_zeroing_sve2p1() {
        check(0x044CA180, "fabs    z0.h, p0/z, z12.h");
        check(0x044DA00C, "fneg    z12.h, p0/z, z0.h");
        rt(0x044CA180);
        rt(0x044DA00C);
        rt(0x045CBAF7); // merging still round-trips
    }

    #[test]
    fn quadword_reductions_fp() {
        check(0x6450ADE5, "faddqv  v5.8h, p3, z15.h");
        check(0x6457ABCA, "fminqv  v10.8h, p2, z30.h");
        check(0x6454A000, "fmaxnmqv v0.8h, p0, z0.h");
        check(0x6455A000, "fminnmqv v0.8h, p0, z0.h");
        check(0x6456A000, "fmaxqv  v0.8h, p0, z0.h");
        check(0x6490A000, "faddqv  v0.4s, p0, z0.s");
        check(0x64D0A000, "faddqv  v0.2d, p0, z0.d");
        for w in [0x6450ADE5, 0x6457ABCA, 0x6454A000, 0x6455A000, 0x6456A000, 0x6490A000, 0x64D0A000] {
            rt(w);
        }
    }

    #[test]
    fn fexpa_unary() {
        // 0460BAFF fexpa z31.h, z23.h
        check(0x0460BAFF, "fexpa   z31.h, z23.h");
    }

    #[test]
    fn ftssel_vec() {
        // 04A2B140 ftssel z0.s, z10.s, z2.s
        check(0x04A2B140, "ftssel  z0.s, z10.s, z2.s");
    }

    #[test]
    fn fmul_indexed() {
        // 64F323C5 fmul z5.d, z30.d, z3.d[1]
        check(0x64F323C5, "fmul    z5.d, z30.d, z3.d[1]");
    }

    #[test]
    fn fmla_indexed_s() {
        // 64AD01C2 fmla z2.s, z14.s, z5.s[1]
        check(0x64AD01C2, "fmla    z2.s, z14.s, z5.s[1]");
    }

    #[test]
    fn fcadd_complex() {
        // 64418669 fcadd z9.h, p1/m, z9.h, z19.h, #0x10e
        check(0x64418669, "fcadd   z9.h, p1/m, z9.h, z19.h, #0x10e");
    }

    #[test]
    fn ftmad_imm() {
        // 659483CC ftmad z12.s, z12.s, z30.s, #0x4
        check(0x659483CC, "ftmad   z12.s, z12.s, z30.s, #0x4");
    }

    #[test]
    fn fmmla_s() {
        // 64A7E5E7 fmmla z7.s, z15.s, z7.s
        check(0x64A7E5E7, "fmmla   z7.s, z15.s, z7.s");
    }

    #[test]
    fn fmsb_pred() {
        // 657BAD5C fmsb z28.h, p3/m, z10.h, z27.h
        check(0x657BAD5C, "fmsb    z28.h, p3/m, z10.h, z27.h");
    }

    #[test]
    fn fnmla_pred() {
        // 65A84A16 fnmla z22.s, p2/m, z16.s, z8.s
        check(0x65A84A16, "fnmla   z22.s, p2/m, z16.s, z8.s");
    }

    #[test]
    fn fnmls_pred() {
        // 65626A37 fnmls z23.h, p2/m, z17.h, z2.h
        check(0x65626A37, "fnmls   z23.h, p2/m, z17.h, z2.h");
    }

    #[test]
    fn fnmsb_pred() {
        // 6573F235 fnmsb z21.h, p4/m, z17.h, z19.h
        check(0x6573F235, "fnmsb   z21.h, p4/m, z17.h, z19.h");
    }

    #[test]
    fn fcmuo_vec() {
        // 6542D98A fcmuo p10.h, p6/z, z12.h, z2.h
        check(0x6542D98A, "fcmuo   p10.h, p6/z, z12.h, z2.h");
    }

    #[test]
    fn fcmne_zero() {
        // 65532D0E fcmne p14.h, p3/z, z8.h, #0.0
        check(0x65532D0E, "fcmne   p14.h, p3/z, z8.h, #0.0");
    }

    #[test]
    fn fabd_pred() {
        // 65889F72 fabd z18.s, p7/m, z18.s, z27.s
        check(0x65889F72, "fabd    z18.s, p7/m, z18.s, z27.s");
    }

    #[test]
    fn fscale_pred() {
        // 6549917A fscale z26.h, p4/m, z26.h, z11.h
        check(0x6549917A, "fscale  z26.h, p4/m, z26.h, z11.h");
    }

    #[test]
    fn fdivr_pred() {
        // 65CC9134 fdivr z20.d, p4/m, z20.d, z9.d
        check(0x65CC9134, "fdivr   z20.d, p4/m, z20.d, z9.d");
    }

    #[test]
    fn fcmle_zero_vs_fcmlt() {
        // 65D133D7 fcmle p7.d, p4/z, z30.d, #0.0
        check(0x65D133D7, "fcmle   p7.d, p4/z, z30.d, #0.0");
        // 65D13CED fcmlt p13.d, p7/z, z7.d, #0.0
        check(0x65D13CED, "fcmlt   p13.d, p7/z, z7.d, #0.0");
    }

    #[test]
    fn fmov_fcpy_imm() {
        // 05D0CDDB fmov z27.d, p0/m, #0.9375  (FCPY rendered as FMOV)
        check(0x05D0CDDB, "fmov    z27.d, p0/m, #0.9375");
        // 05D2C535 fmov z21.d, p2/m, #12.5
        check(0x05D2C535, "fmov    z21.d, p2/m, #12.5");
    }

    #[test]
    fn fmov_fdup_imm() {
        // 25F9D7A8 fmov z8.d, #-29.0  (FDUP rendered as FMOV)
        check(0x25F9D7A8, "fmov    z8.d, #-29.0");
    }

    #[test]
    fn faddp_pairwise() {
        // 645094C5 faddp z5.h, p5/m, z5.h, z6.h
        check(0x645094C5, "faddp   z5.h, p5/m, z5.h, z6.h");
    }

    #[test]
    fn fcvtlt_h2s() {
        // 6489B691 fcvtlt z17.s, p5/m, z20.h
        check(0x6489B691, "fcvtlt  z17.s, p5/m, z20.h");
    }

    #[test]
    fn fcvtxnt_narrow() {
        // 640AA4C9 fcvtxnt z9.s, p1/m, z6.d
        check(0x640AA4C9, "fcvtxnt z9.s, p1/m, z6.d");
    }

    #[test]
    fn bfcvtnt_narrow() {
        // 648AABAB bfcvtnt z11.h, p2/m, z29.s
        check(0x648AABAB, "bfcvtnt z11.h, p2/m, z29.s");
    }

    #[test]
    fn bfcvt_pred() {
        // 658AB9BB bfcvt z27.h, p6/m, z13.s
        check(0x658AB9BB, "bfcvt   z27.h, p6/m, z13.s");
    }

    #[test]
    fn bfdot_vec() {
        // 64698385 bfdot z5.s, z28.h, z9.h
        check(0x64698385, "bfdot   z5.s, z28.h, z9.h");
    }

    #[test]
    fn fmlalb_long() {
        // 64B083C3 fmlalb z3.s, z30.h, z16.h
        check(0x64B083C3, "fmlalb  z3.s, z30.h, z16.h");
    }

    #[test]
    fn frecps_unpred() {
        // 65D61BFB frecps z27.d, z31.d, z22.d
        check(0x65D61BFB, "frecps  z27.d, z31.d, z22.d");
    }

    #[test]
    fn fcmla_indexed() {
        // 64A81101 fcmla z1.h, z8.h, z0.h[1], #0x0
        check(0x64A81101, "fcmla   z1.h, z8.h, z0.h[1], #0x0");
    }
}
