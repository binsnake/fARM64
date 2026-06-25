//! Scalar floating-point data-processing (ARM ARM C4.1.97, the `(op0&5)==1`
//! rows): conversions to/from integer and fixed-point, FP data-processing
//! (1/2/3 source), FP compare, conditional compare and select, and the FP
//! immediate move.
//!
//! `ftype = word<23:22>` selects the precision: `00`=single (`S`), `01`=double
//! (`D`), `11`=half (`H`, gated on [`Feature::Fp16`]), `10` reserved. Half-
//! precision and BF16 forms are gated by [`Feature`] so a base-only feature set
//! omits them.
//!
//! Preferred-alias policy matches the rest of the crate: `code` is the canonical
//! per-encoding identity while `mnemonic` carries the spelling. The scalar FP
//! group has no UAL aliases beyond the `#0.0` compare spelling (handled as an
//! operand), so `code.mnemonic()` is used directly throughout.

use crate::decode::bits::{bit, bits, vfp_expand_imm};
use crate::enums::Condition;
use crate::features::{Feature, FeatureSet};
use crate::instruction::Instruction;
use crate::mnemonic::Code;
use crate::operand::Operand;
use crate::register::{gp_register, Register, RegWidth};

// ---------------------------------------------------------------------------
// Small operand/precision helpers.
// ---------------------------------------------------------------------------

/// The scalar-FP precision selected by `ftype` (`word<23:22>`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Prec {
    /// Single precision (`S`, 32-bit), `ftype == 00`.
    S,
    /// Double precision (`D`, 64-bit), `ftype == 01`.
    D,
    /// Half precision (`H`, 16-bit), `ftype == 11` (FEAT_FP16).
    H,
}

impl Prec {
    /// Decode `ftype`. Returns `None` for the reserved `10` encoding.
    #[inline]
    fn from_ftype(ftype: u32) -> Option<Prec> {
        match ftype {
            0b00 => Some(Prec::S),
            0b01 => Some(Prec::D),
            0b11 => Some(Prec::H),
            _ => None,
        }
    }

    /// `true` when this precision additionally requires FEAT_FP16.
    #[inline]
    fn needs_fp16(self) -> bool {
        matches!(self, Prec::H)
    }
}

/// Build a scalar FP register operand of the given precision and number.
#[inline]
fn fp_reg(p: Prec, n: u32) -> Operand {
    let n = (n & 0x1f) as u8;
    let reg = match p {
        Prec::S => s_reg(n),
        Prec::D => d_reg(n),
        Prec::H => h_reg(n),
    };
    plain(reg)
}

/// Build a plain register operand (no decorations).
#[inline]
fn plain(reg: Register) -> Operand {
    Operand::Reg {
        reg,
        arr: None,
        lane: None,
        shift: None,
        extend: None,
        pred: None,
    }
}

/// Build a plain GP register operand of width `w`.
#[inline]
fn gpr(w: RegWidth, n: u32) -> Operand {
    plain(gp_register(false, w, (n & 0x1f) as u8))
}

/// The GP width for an `sf` bit (`0` => W, `1` => X).
#[inline]
fn width_of(sf: u32) -> RegWidth {
    if sf & 1 == 1 {
        RegWidth::X64
    } else {
        RegWidth::W32
    }
}

// ---------------------------------------------------------------------------
// Conversion between floating-point and fixed-point (decode_float2fix).
// ---------------------------------------------------------------------------

/// `SCVTF`/`UCVTF`/`FCVTZS`/`FCVTZU` (scalar, fixed-point). `scale` is encoded in
/// `word<15:10>`; the rendered `#fbits` is `64 - scale`. For 32-bit forms
/// (`sf==0`) `scale<5>` must be `1` (else UNALLOCATED).
#[inline]
pub fn decode_float2fix(word: u32, features: FeatureSet, out: &mut Instruction) {
    let sf = bit(word, 31);
    let s = bit(word, 29); // must be 0
    let ftype = bits(word, 22, 2);
    let rmode = bits(word, 19, 2);
    let opcode = bits(word, 16, 3);
    let scale = bits(word, 10, 6);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);

    if s != 0 {
        return;
    }
    let p = match Prec::from_ftype(ftype) {
        Some(p) => p,
        None => return,
    };
    if p.needs_fp16() && !features.has(Feature::Fp16) {
        return;
    }
    // 32-bit forms require scale<5> == 1 (fbits in 1..=32).
    if sf == 0 && bit(scale, 5) == 0 {
        return;
    }

    // (rmode, opcode) select the operation:
    //   rmode==00, opcode==010 -> SCVTF  (int->fp)
    //   rmode==00, opcode==011 -> UCVTF  (int->fp)
    //   rmode==11, opcode==000 -> FCVTZS (fp->int)
    //   rmode==11, opcode==001 -> FCVTZU (fp->int)
    let w = width_of(sf);
    let fbits = 64 - scale; // #fbits operand value.

    let (code, int_to_fp) = match (rmode, opcode) {
        (0b00, 0b010) => (scvtf_fixed_code(p, sf), true),
        (0b00, 0b011) => (ucvtf_fixed_code(p, sf), true),
        (0b11, 0b000) => (fcvtzs_fixed_code(p, sf), false),
        (0b11, 0b001) => (fcvtzu_fixed_code(p, sf), false),
        _ => return,
    };
    out.set(code);

    if int_to_fp {
        // SCVTF/UCVTF <Sd|Dd|Hd>, <Wn|Xn>, #fbits.
        out.push_operand(fp_reg(p, rd));
        out.push_operand(gpr(w, rn));
        out.push_operand(Operand::ImmUnsigned(fbits as u64));
    } else {
        // FCVTZS/FCVTZU <Wd|Xd>, <Sn|Dn|Hn>, #fbits.
        out.push_operand(gpr(w, rd));
        out.push_operand(fp_reg(p, rn));
        out.push_operand(Operand::ImmUnsigned(fbits as u64));
    }
}

// ---------------------------------------------------------------------------
// Conversion between floating-point and integer (decode_float2int).
// ---------------------------------------------------------------------------

/// `FCVT{N,P,M,Z,A}{S,U}`, `SCVTF`/`UCVTF`, `FMOV` (general<->FP, incl. the
/// `Vn.D[1]` top-half forms) and `FJCVTZS`. `rmode`/`opcode` (`word<20:16>`)
/// select the operation; `word<15:10>` must be `000000` (guaranteed by the
/// dispatcher).
#[inline]
pub fn decode_float2int(word: u32, features: FeatureSet, out: &mut Instruction) {
    let sf = bit(word, 31);
    let s = bit(word, 29); // must be 0
    let ftype = bits(word, 22, 2);
    let rmode = bits(word, 19, 2);
    let opcode = bits(word, 16, 3);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);

    if s != 0 {
        return;
    }

    // FMOV between GP and the high (bits<127:64>) half of a 128-bit vector, and
    // FJCVTZS, are special ftype/rmode combinations handled first.
    // FMOV top-half: sf==1, ftype==10, rmode==01, opcode 6/7.
    if sf == 1 && ftype == 0b10 && rmode == 0b01 {
        match opcode {
            0b110 => {
                // FMOV <Xd>, <Vn>.D[1]  (read high half to GP).
                out.set(Code::FmovTopToGp);
                out.push_operand(gpr(RegWidth::X64, rd));
                out.push_operand(v_d1(rn));
                return;
            }
            0b111 => {
                // FMOV <Vd>.D[1], <Xn>  (write GP into high half).
                out.set(Code::FmovTopFromGp);
                out.push_operand(v_d1(rd));
                out.push_operand(gpr(RegWidth::X64, rn));
                return;
            }
            _ => return,
        }
    }

    // FJCVTZS: sf==0, ftype==01 (double), rmode==11, opcode==110 (FEAT_JSCVT).
    if sf == 0 && ftype == 0b01 && rmode == 0b11 && opcode == 0b110 {
        // FEAT_JSCVT is part of the base catalog here (no dedicated Feature);
        // decode unconditionally as the corpus enables it.
        out.set(Code::Fjcvtzs);
        out.push_operand(gpr(RegWidth::W32, rd));
        out.push_operand(fp_reg(Prec::D, rn));
        return;
    }

    let p = match Prec::from_ftype(ftype) {
        Some(p) => p,
        None => return,
    };
    if p.needs_fp16() && !features.has(Feature::Fp16) {
        return;
    }
    let w = width_of(sf);

    // FMOV general<->FP: opcode 6/7 with rmode==00.
    if rmode == 0b00 && (opcode == 0b110 || opcode == 0b111) {
        return fmov_general(word, p, sf, w, opcode, rn, rd, out);
    }

    // The FCVT*/SCVTF/UCVTF integer conversions. (rmode, opcode):
    //   rmode==00: opcode 0=FCVTNS 1=FCVTNU 2=SCVTF 3=UCVTF 4=FCVTAS 5=FCVTAU
    //   rmode==01: opcode 0=FCVTPS 1=FCVTPU
    //   rmode==10: opcode 0=FCVTMS 1=FCVTMU
    //   rmode==11: opcode 0=FCVTZS 1=FCVTZU
    let (code, int_to_fp) = match (rmode, opcode) {
        (0b00, 0b000) => (Code::FcvtnsScalar, false),
        (0b00, 0b001) => (Code::FcvtnuScalar, false),
        (0b00, 0b010) => (scvtf_int_code(p, sf), true),
        (0b00, 0b011) => (ucvtf_int_code(p, sf), true),
        (0b00, 0b100) => (Code::FcvtasScalar, false),
        (0b00, 0b101) => (Code::FcvtauScalar, false),
        (0b01, 0b000) => (Code::FcvtpsScalar, false),
        (0b01, 0b001) => (Code::FcvtpuScalar, false),
        (0b10, 0b000) => (Code::FcvtmsScalar, false),
        (0b10, 0b001) => (Code::FcvtmuScalar, false),
        (0b11, 0b000) => (fcvtzs_int_code(p, sf), false),
        (0b11, 0b001) => (fcvtzu_int_code(p, sf), false),
        _ => return,
    };
    out.set(code);

    if int_to_fp {
        // SCVTF/UCVTF <Sd|Dd|Hd>, <Wn|Xn>.
        out.push_operand(fp_reg(p, rd));
        out.push_operand(gpr(w, rn));
    } else {
        // FCVT* <Wd|Xd>, <Sn|Dn|Hn>.
        out.push_operand(gpr(w, rd));
        out.push_operand(fp_reg(p, rn));
    }
}

/// FMOV between a general-purpose register and a scalar FP register (opcode 6/7,
/// rmode 00). `opcode==6` reads FP->GP, `opcode==7` writes GP->FP. The GP width
/// must structurally match the FP precision (W for S/H, X for D); the mismatched
/// combinations are UNALLOCATED.
#[inline]
#[allow(clippy::too_many_arguments)]
fn fmov_general(
    _word: u32,
    p: Prec,
    sf: u32,
    w: RegWidth,
    opcode: u32,
    rn: u32,
    rd: u32,
    out: &mut Instruction,
) {
    // Allowed (sf, ftype) pairings for the GP<->FP FMOV:
    //   sf==0 & S  : FMOV Wd,Sn / Sd,Wn
    //   sf==1 & D  : FMOV Xd,Dn / Dd,Xn
    //   sf==0 & H  : FMOV Wd,Hn / Hd,Wn   (FEAT_FP16)
    //   sf==1 & H  : FMOV Xd,Hn / Hd,Xn   (FEAT_FP16)
    let ok = matches!((sf, p), (0, Prec::S) | (1, Prec::D) | (_, Prec::H));
    if !ok {
        return;
    }

    let from_fp = opcode == 0b110; // 6 -> FP to GP, 7 -> GP to FP.
    let code = match (p, sf, from_fp) {
        (Prec::S, _, true) => Code::FmovToGp32,
        (Prec::S, _, false) => Code::FmovFromGp32,
        (Prec::D, _, true) => Code::FmovToGp64,
        (Prec::D, _, false) => Code::FmovFromGp64,
        (Prec::H, 0, true) => Code::FmovToGpH32,
        (Prec::H, 0, false) => Code::FmovFromGpH32,
        (Prec::H, _, true) => Code::FmovToGpH64,
        (Prec::H, _, false) => Code::FmovFromGpH64,
    };
    out.set(code);
    if from_fp {
        out.push_operand(gpr(w, rd));
        out.push_operand(fp_reg(p, rn));
    } else {
        out.push_operand(fp_reg(p, rd));
        out.push_operand(gpr(w, rn));
    }
}

// ---------------------------------------------------------------------------
// Floating-point data-processing (1 source) (decode_floatdp1).
// ---------------------------------------------------------------------------

/// `FMOV`/`FABS`/`FNEG`/`FSQRT`, `FCVT` (between precisions), `FRINT{N,P,M,Z,A,
/// X,I}` and the v8.5 `FRINT{32,64}{X,Z}`. `opcode = word<20:15>` selects the
/// operation; `ftype` the source precision.
#[inline]
pub fn decode_floatdp1(word: u32, features: FeatureSet, out: &mut Instruction) {
    let m = bit(word, 31);
    let s = bit(word, 29);
    let ftype = bits(word, 22, 2);
    let opcode = bits(word, 15, 6);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);

    if m != 0 || s != 0 {
        return;
    }
    let p = match Prec::from_ftype(ftype) {
        Some(p) => p,
        None => return,
    };
    if p.needs_fp16() && !features.has(Feature::Fp16) {
        return;
    }

    // BFCVT (BF16): `BFCVT <Hd>, <Sn>`, converting single precision to BFloat16.
    // It is encoded in the `ftype == 01` slot with `opcode == 0b000110`, but the
    // *source* operand is single-precision (`Sn`) and the destination a half-
    // width `Hd`. It shares the `0b0001xx` opcode region with FCVT, so test it
    // first.
    if opcode == 0b000110 {
        if p != Prec::D || !features.has(Feature::Bf16) {
            return;
        }
        out.set(Code::Bfcvt);
        out.push_operand(plain(h_reg(rd as u8 & 0x1f)));
        out.push_operand(fp_reg(Prec::S, rn));
        return;
    }

    // FCVT changes precision: opcode 0b0001xx selects the destination type from
    // opcode<1:0> (00=S, 01=D, 11=H), and the source is `ftype`. Handle FCVT
    // next so the same-precision ops below stay simple.
    if (opcode & 0b111100) == 0b000100 {
        let dst = match opcode & 0b11 {
            0b00 => Prec::S,
            0b01 => Prec::D,
            0b11 => Prec::H,
            _ => return, // opcode<1:0>==10 is UNALLOCATED.
        };
        if dst == p {
            return; // same-precision FCVT is not allocated.
        }
        // FCVT to/from H is always allowed (it is half-precision *arithmetic*
        // that gates on FEAT_FP16, not the convert), matching LLVM/binja.
        let code = match fcvt_code(p, dst) {
            Some(c) => c,
            None => return,
        };
        out.set(code);
        out.push_operand(fp_reg(dst, rd));
        out.push_operand(fp_reg(p, rn));
        return;
    }

    // Same-precision 1-source ops.
    let code = match opcode {
        0b000000 => fmov_dp1_code(p),
        0b000001 => fabs_code(p),
        0b000010 => fneg_code(p),
        0b000011 => fsqrt_code(p),
        0b001000 => frintn_code(p),
        0b001001 => frintp_code(p),
        0b001010 => frintm_code(p),
        0b001011 => frintz_code(p),
        0b001100 => frinta_code(p),
        0b001110 => frintx_code(p),
        0b001111 => frinti_code(p),
        // v8.5 FRINT32/64 (FEAT_FRINTTS); S/D only.
        0b010000 => match frint32z_code(p, features) {
            Some(c) => c,
            None => return,
        },
        0b010001 => match frint32x_code(p, features) {
            Some(c) => c,
            None => return,
        },
        0b010010 => match frint64z_code(p, features) {
            Some(c) => c,
            None => return,
        },
        0b010011 => match frint64x_code(p, features) {
            Some(c) => c,
            None => return,
        },
        _ => return,
    };
    out.set(code);
    out.push_operand(fp_reg(p, rd));
    out.push_operand(fp_reg(p, rn));
}

// ---------------------------------------------------------------------------
// Floating-point data-processing (2 source) (decode_floatdp2).
// ---------------------------------------------------------------------------

/// `FMUL`/`FDIV`/`FADD`/`FSUB`/`FMAX`/`FMIN`/`FMAXNM`/`FMINNM`/`FNMUL`.
/// `opcode = word<15:12>` selects the operation.
#[inline]
pub fn decode_floatdp2(word: u32, features: FeatureSet, out: &mut Instruction) {
    let m = bit(word, 31);
    let s = bit(word, 29);
    let ftype = bits(word, 22, 2);
    let opcode = bits(word, 12, 4);
    let rm = bits(word, 16, 5);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);

    if m != 0 || s != 0 {
        return;
    }
    let p = match Prec::from_ftype(ftype) {
        Some(p) => p,
        None => return,
    };
    if p.needs_fp16() && !features.has(Feature::Fp16) {
        return;
    }

    let code = match opcode {
        0b0000 => fmul_code(p),
        0b0001 => fdiv_code(p),
        0b0010 => fadd_code(p),
        0b0011 => fsub_code(p),
        0b0100 => fmax_code(p),
        0b0101 => fmin_code(p),
        0b0110 => fmaxnm_code(p),
        0b0111 => fminnm_code(p),
        0b1000 => fnmul_code(p),
        _ => return,
    };
    out.set(code);
    out.push_operand(fp_reg(p, rd));
    out.push_operand(fp_reg(p, rn));
    out.push_operand(fp_reg(p, rm));
}

// ---------------------------------------------------------------------------
// Floating-point data-processing (3 source) (decode_floatdp3).
// ---------------------------------------------------------------------------

/// `FMADD`/`FMSUB`/`FNMADD`/`FNMSUB`. `o1 = word<21>`, `o0 = word<15>` select
/// the operation; the four operands are `Rd, Rn, Rm, Ra`.
#[inline]
pub fn decode_floatdp3(word: u32, features: FeatureSet, out: &mut Instruction) {
    let m = bit(word, 31);
    let s = bit(word, 29);
    let ftype = bits(word, 22, 2);
    let o1 = bit(word, 21);
    let o0 = bit(word, 15);
    let rm = bits(word, 16, 5);
    let ra = bits(word, 10, 5);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);

    if m != 0 || s != 0 {
        return;
    }
    let p = match Prec::from_ftype(ftype) {
        Some(p) => p,
        None => return,
    };
    if p.needs_fp16() && !features.has(Feature::Fp16) {
        return;
    }

    let code = match (o1, o0) {
        (0, 0) => fmadd_code(p),
        (0, 1) => fmsub_code(p),
        (1, 0) => fnmadd_code(p),
        (_, _) => fnmsub_code(p),
    };
    out.set(code);
    out.push_operand(fp_reg(p, rd));
    out.push_operand(fp_reg(p, rn));
    out.push_operand(fp_reg(p, rm));
    out.push_operand(fp_reg(p, ra));
}

// ---------------------------------------------------------------------------
// Floating-point compare (decode_floatcmp).
// ---------------------------------------------------------------------------

/// `FCMP`/`FCMPE` (register and the `#0.0` forms). `op = word<15:14>` must be
/// `00`; `opcode2 = word<4:0>` selects register-vs-zero and E-vs-non-E.
#[inline]
pub fn decode_floatcmp(word: u32, features: FeatureSet, out: &mut Instruction) {
    let m = bit(word, 31);
    let s = bit(word, 29);
    let ftype = bits(word, 22, 2);
    let op = bits(word, 14, 2);
    let opcode2 = bits(word, 0, 5);
    let rm = bits(word, 16, 5);
    let rn = bits(word, 5, 5);

    if m != 0 || s != 0 || op != 0 {
        return;
    }
    let p = match Prec::from_ftype(ftype) {
        Some(p) => p,
        None => return,
    };
    if p.needs_fp16() && !features.has(Feature::Fp16) {
        return;
    }

    // opcode2: 00000=FCMP, 01000=FCMP #0.0, 10000=FCMPE, 11000=FCMPE #0.0.
    // (bits<2:0> must be 0; bit<3> selects the #0.0 form; bit<4> selects E.)
    if (opcode2 & 0b00111) != 0 {
        return;
    }
    let is_zero = bit(opcode2, 3) == 1;
    let is_e = bit(opcode2, 4) == 1;
    // For the #0.0 forms the Rm field (word<20:16>) is "should be zero" but is
    // architecturally unused: the decode does not depend on it (matches binja,
    // which accepts non-zero Rm bits and still renders the `#0.0` compare).

    let code = if is_e { fcmpe_code(p) } else { fcmp_code(p) };
    out.set(code);
    out.push_operand(fp_reg(p, rn));
    if is_zero {
        // Render the immediate `#0.0` zero comparison.
        out.push_operand(Operand::FpImm(0.0));
    } else {
        out.push_operand(fp_reg(p, rm));
    }
}

// ---------------------------------------------------------------------------
// Floating-point conditional compare (decode_floatccmp).
// ---------------------------------------------------------------------------

/// `FCCMP`/`FCCMPE`. `op = word<4>` selects the E variant; the operands are
/// `Rn, Rm, #nzcv, <cond>`.
#[inline]
pub fn decode_floatccmp(word: u32, features: FeatureSet, out: &mut Instruction) {
    let m = bit(word, 31);
    let s = bit(word, 29);
    let ftype = bits(word, 22, 2);
    let op = bit(word, 4);
    let rm = bits(word, 16, 5);
    let cond = bits(word, 12, 4);
    let rn = bits(word, 5, 5);
    let nzcv = bits(word, 0, 4);

    if m != 0 || s != 0 {
        return;
    }
    let p = match Prec::from_ftype(ftype) {
        Some(p) => p,
        None => return,
    };
    if p.needs_fp16() && !features.has(Feature::Fp16) {
        return;
    }

    let code = if op == 1 { fccmpe_code(p) } else { fccmp_code(p) };
    out.set(code);
    out.push_operand(fp_reg(p, rn));
    out.push_operand(fp_reg(p, rm));
    out.push_operand(Operand::ImmUnsigned(nzcv as u64));
    out.push_operand(Operand::Cond(Condition::from_u4(cond as u8)));
}

// ---------------------------------------------------------------------------
// Floating-point conditional select (decode_floatsel).
// ---------------------------------------------------------------------------

/// `FCSEL <Sd|Dd|Hd>, <Sn..>, <Sm..>, <cond>`.
#[inline]
pub fn decode_floatsel(word: u32, features: FeatureSet, out: &mut Instruction) {
    let m = bit(word, 31);
    let s = bit(word, 29);
    let ftype = bits(word, 22, 2);
    let rm = bits(word, 16, 5);
    let cond = bits(word, 12, 4);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);

    if m != 0 || s != 0 {
        return;
    }
    let p = match Prec::from_ftype(ftype) {
        Some(p) => p,
        None => return,
    };
    if p.needs_fp16() && !features.has(Feature::Fp16) {
        return;
    }

    out.set(fcsel_code(p));
    out.push_operand(fp_reg(p, rd));
    out.push_operand(fp_reg(p, rn));
    out.push_operand(fp_reg(p, rm));
    out.push_operand(Operand::Cond(Condition::from_u4(cond as u8)));
}

// ---------------------------------------------------------------------------
// Floating-point immediate (decode_floatimm).
// ---------------------------------------------------------------------------

/// `FMOV <Sd|Dd|Hd>, #imm`. The 8-bit `imm8 = word<20:13>` is expanded by
/// [`vfp_expand_imm`]; `imm5 = word<9:5>` must be zero (guaranteed by the
/// dispatcher precondition).
#[inline]
pub fn decode_floatimm(word: u32, features: FeatureSet, out: &mut Instruction) {
    let m = bit(word, 31);
    let s = bit(word, 29);
    let ftype = bits(word, 22, 2);
    let imm8 = bits(word, 13, 8);
    let imm5 = bits(word, 5, 5);
    let rd = bits(word, 0, 5);

    if m != 0 || s != 0 || imm5 != 0 {
        return;
    }
    let p = match Prec::from_ftype(ftype) {
        Some(p) => p,
        None => return,
    };
    if p.needs_fp16() && !features.has(Feature::Fp16) {
        return;
    }

    // Expand to the element bit-pattern, then bit-cast to the f32 the formatter
    // renders. Half/double values are representable exactly as f32 here because
    // VFPExpandImm only encodes the 8-bit (sign:exp3:frac4) immediate set.
    let bits_val = match p {
        Prec::S => vfp_expand_imm(imm8, 32),
        Prec::D => vfp_expand_imm(imm8, 64),
        Prec::H => vfp_expand_imm(imm8, 16),
    };
    let f = fp_value_as_f32(p, bits_val);

    let code = match p {
        Prec::S => Code::FmovImmS,
        Prec::D => Code::FmovImmD,
        Prec::H => Code::FmovImmH,
    };
    out.set(code);
    out.push_operand(fp_reg(p, rd));
    out.push_operand(Operand::FpImm(f));
}

/// Reinterpret the raw element bit-pattern of a VFP immediate as the `f32` the
/// formatter renders. The 8-bit FP immediate set is exactly representable in
/// `f32` for all three precisions, so the half/double bit patterns are converted
/// by value (not by bit-cast).
#[inline]
fn fp_value_as_f32(p: Prec, bits_val: u64) -> f32 {
    match p {
        Prec::S => f32::from_bits(bits_val as u32),
        Prec::D => f64::from_bits(bits_val) as f32,
        Prec::H => f16_bits_to_f32(bits_val as u16),
    }
}

/// Convert an IEEE-754 half-precision bit pattern to `f32`. Only the normal /
/// signed-zero values produced by [`vfp_expand_imm`] occur here, but the full
/// conversion (incl. subnormal/inf/nan) is implemented for totality.
///
/// Shared with the Advanced-SIMD modified-immediate `FMOV.half` decoder.
#[inline]
pub(crate) fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if frac == 0 {
            // Signed zero.
            sign << 31
        } else {
            // Subnormal: normalize.
            let mut e = -1i32;
            let mut f = frac;
            while (f & 0x400) == 0 {
                f <<= 1;
                e -= 1;
            }
            f &= 0x3ff;
            let exp32 = (e + 127 - 15) as u32;
            (sign << 31) | (exp32 << 23) | (f << 13)
        }
    } else if exp == 0x1f {
        // Inf / NaN.
        (sign << 31) | (0xff << 23) | (frac << 13)
    } else {
        // Normal.
        let exp32 = exp + (127 - 15);
        (sign << 31) | (exp32 << 23) | (frac << 13)
    };
    f32::from_bits(bits)
}

// ---------------------------------------------------------------------------
// Scalar FP register-view constructors.
// ---------------------------------------------------------------------------

/// The `H{n}` scalar view (16-bit).
#[inline]
fn h_reg(n: u8) -> Register {
    fp_view(Register::H0, n)
}
/// The `S{n}` scalar view (32-bit).
#[inline]
fn s_reg(n: u8) -> Register {
    fp_view(Register::S0, n)
}
/// The `D{n}` scalar view (64-bit).
#[inline]
fn d_reg(n: u8) -> Register {
    fp_view(Register::D0, n)
}

/// Offset a contiguous register-bank base by `n` (`0..=31`). The FP banks are
/// laid out as 32 contiguous discriminants each (see [`crate::register`]).
#[inline]
fn fp_view(base: Register, n: u8) -> Register {
    // SAFETY-free: build from the discriminant numerically via a match on the
    // base bank so we never transmute. Each bank is contiguous in declaration
    // order, so we index a small lookup.
    let n = (n & 0x1f) as usize;
    match base {
        Register::H0 => H_BANK[n],
        Register::S0 => S_BANK[n],
        Register::D0 => D_BANK[n],
        _ => Register::None,
    }
}

/// Build a `Vn.D[1]` operand: the `V{n}` 128-bit view, `.d` arrangement, lane 1.
#[inline]
fn v_d1(n: u32) -> Operand {
    Operand::Reg {
        reg: V_BANK[(n & 0x1f) as usize],
        arr: Some(crate::enums::VectorArrangement::V2D),
        lane: Some(1),
        shift: None,
        extend: None,
        pred: None,
    }
}

// Contiguous register banks (indexed 0..=31). These mirror the append-only
// discriminant order in `crate::register::Register`.
const H_BANK: [Register; 32] = [
    Register::H0, Register::H1, Register::H2, Register::H3, Register::H4, Register::H5, Register::H6, Register::H7,
    Register::H8, Register::H9, Register::H10, Register::H11, Register::H12, Register::H13, Register::H14, Register::H15,
    Register::H16, Register::H17, Register::H18, Register::H19, Register::H20, Register::H21, Register::H22, Register::H23,
    Register::H24, Register::H25, Register::H26, Register::H27, Register::H28, Register::H29, Register::H30, Register::H31,
];
const S_BANK: [Register; 32] = [
    Register::S0, Register::S1, Register::S2, Register::S3, Register::S4, Register::S5, Register::S6, Register::S7,
    Register::S8, Register::S9, Register::S10, Register::S11, Register::S12, Register::S13, Register::S14, Register::S15,
    Register::S16, Register::S17, Register::S18, Register::S19, Register::S20, Register::S21, Register::S22, Register::S23,
    Register::S24, Register::S25, Register::S26, Register::S27, Register::S28, Register::S29, Register::S30, Register::S31,
];
const D_BANK: [Register; 32] = [
    Register::D0, Register::D1, Register::D2, Register::D3, Register::D4, Register::D5, Register::D6, Register::D7,
    Register::D8, Register::D9, Register::D10, Register::D11, Register::D12, Register::D13, Register::D14, Register::D15,
    Register::D16, Register::D17, Register::D18, Register::D19, Register::D20, Register::D21, Register::D22, Register::D23,
    Register::D24, Register::D25, Register::D26, Register::D27, Register::D28, Register::D29, Register::D30, Register::D31,
];
const V_BANK: [Register; 32] = [
    Register::V0, Register::V1, Register::V2, Register::V3, Register::V4, Register::V5, Register::V6, Register::V7,
    Register::V8, Register::V9, Register::V10, Register::V11, Register::V12, Register::V13, Register::V14, Register::V15,
    Register::V16, Register::V17, Register::V18, Register::V19, Register::V20, Register::V21, Register::V22, Register::V23,
    Register::V24, Register::V25, Register::V26, Register::V27, Register::V28, Register::V29, Register::V30, Register::V31,
];

// ---------------------------------------------------------------------------
// Per-precision Code selectors.
// ---------------------------------------------------------------------------

macro_rules! by_prec {
    ($name:ident, $s:ident, $d:ident, $h:ident) => {
        #[inline]
        fn $name(p: Prec) -> Code {
            match p {
                Prec::S => Code::$s,
                Prec::D => Code::$d,
                Prec::H => Code::$h,
            }
        }
    };
}

by_prec!(fmov_dp1_code, FmovS, FmovD, FmovH);
by_prec!(fabs_code, FabsS, FabsD, FabsH);
by_prec!(fneg_code, FnegS, FnegD, FnegH);
by_prec!(fsqrt_code, FsqrtS, FsqrtD, FsqrtH);
by_prec!(frintn_code, FrintnS, FrintnD, FrintnH);
by_prec!(frintp_code, FrintpS, FrintpD, FrintpH);
by_prec!(frintm_code, FrintmS, FrintmD, FrintmH);
by_prec!(frintz_code, FrintzS, FrintzD, FrintzH);
by_prec!(frinta_code, FrintaS, FrintaD, FrintaH);
by_prec!(frintx_code, FrintxS, FrintxD, FrintxH);
by_prec!(frinti_code, FrintiS, FrintiD, FrintiH);
by_prec!(fmul_code, FmulS, FmulD, FmulH);
by_prec!(fdiv_code, FdivS, FdivD, FdivH);
by_prec!(fadd_code, FaddS, FaddD, FaddH);
by_prec!(fsub_code, FsubS, FsubD, FsubH);
by_prec!(fmax_code, FmaxS, FmaxD, FmaxH);
by_prec!(fmin_code, FminS, FminD, FminH);
by_prec!(fmaxnm_code, FmaxnmS, FmaxnmD, FmaxnmH);
by_prec!(fminnm_code, FminnmS, FminnmD, FminnmH);
by_prec!(fnmul_code, FnmulS, FnmulD, FnmulH);
by_prec!(fmadd_code, FmaddS, FmaddD, FmaddH);
by_prec!(fmsub_code, FmsubS, FmsubD, FmsubH);
by_prec!(fnmadd_code, FnmaddS, FnmaddD, FnmaddH);
by_prec!(fnmsub_code, FnmsubS, FnmsubD, FnmsubH);
by_prec!(fcmp_code, FcmpS, FcmpD, FcmpH);
by_prec!(fcmpe_code, FcmpeS, FcmpeD, FcmpeH);
by_prec!(fccmp_code, FccmpS, FccmpD, FccmpH);
by_prec!(fccmpe_code, FccmpeS, FccmpeD, FccmpeH);
by_prec!(fcsel_code, FcselS, FcselD, FcselH);

/// FRINT32Z/X and FRINT64Z/X exist only for S/D and require FEAT_FRINTTS
/// (folded into FEAT_FP16-independent gating; gated by `Feature::Fp16`-style is
/// wrong, so they are gated explicitly by `Feature::Frintts`).
#[inline]
fn frint32z_code(p: Prec, features: FeatureSet) -> Option<Code> {
    frintts(p, features, Code::Frint32zS, Code::Frint32zD)
}
#[inline]
fn frint32x_code(p: Prec, features: FeatureSet) -> Option<Code> {
    frintts(p, features, Code::Frint32xS, Code::Frint32xD)
}
#[inline]
fn frint64z_code(p: Prec, features: FeatureSet) -> Option<Code> {
    frintts(p, features, Code::Frint64zS, Code::Frint64zD)
}
#[inline]
fn frint64x_code(p: Prec, features: FeatureSet) -> Option<Code> {
    frintts(p, features, Code::Frint64xS, Code::Frint64xD)
}
#[inline]
fn frintts(p: Prec, features: FeatureSet, s: Code, d: Code) -> Option<Code> {
    if !features.has(Feature::Frintts) {
        return None;
    }
    match p {
        Prec::S => Some(s),
        Prec::D => Some(d),
        Prec::H => None,
    }
}

/// FCVT between two distinct precisions (`src` -> `dst`). Returns `None` for the
/// (unreachable) same-precision pair.
#[inline]
fn fcvt_code(src: Prec, dst: Prec) -> Option<Code> {
    let c = match (src, dst) {
        (Prec::S, Prec::D) => Code::FcvtSD,
        (Prec::S, Prec::H) => Code::FcvtSH,
        (Prec::D, Prec::S) => Code::FcvtDS,
        (Prec::D, Prec::H) => Code::FcvtDH,
        (Prec::H, Prec::S) => Code::FcvtHS,
        (Prec::H, Prec::D) => Code::FcvtHD,
        _ => return None,
    };
    Some(c)
}

/// SCVTF (int->fp), per precision and GP width.
#[inline]
fn scvtf_int_code(p: Prec, sf: u32) -> Code {
    match (p, sf) {
        (Prec::S, 0) => Code::ScvtfS32,
        (Prec::S, _) => Code::ScvtfS64,
        (Prec::D, 0) => Code::ScvtfD32,
        (Prec::D, _) => Code::ScvtfD64,
        (Prec::H, 0) => Code::ScvtfH32,
        (Prec::H, _) => Code::ScvtfH64,
    }
}
#[inline]
fn ucvtf_int_code(p: Prec, sf: u32) -> Code {
    match (p, sf) {
        (Prec::S, 0) => Code::UcvtfS32,
        (Prec::S, _) => Code::UcvtfS64,
        (Prec::D, 0) => Code::UcvtfD32,
        (Prec::D, _) => Code::UcvtfD64,
        (Prec::H, 0) => Code::UcvtfH32,
        (Prec::H, _) => Code::UcvtfH64,
    }
}
#[inline]
fn fcvtzs_int_code(p: Prec, sf: u32) -> Code {
    match (p, sf) {
        (Prec::S, 0) => Code::FcvtzsScalarS32,
        (Prec::S, _) => Code::FcvtzsScalarS64,
        (Prec::D, 0) => Code::FcvtzsScalarD32,
        (Prec::D, _) => Code::FcvtzsScalarD64,
        (Prec::H, 0) => Code::FcvtzsScalarH32,
        (Prec::H, _) => Code::FcvtzsScalarH64,
    }
}
#[inline]
fn fcvtzu_int_code(p: Prec, sf: u32) -> Code {
    match (p, sf) {
        (Prec::S, 0) => Code::FcvtzuScalarS32,
        (Prec::S, _) => Code::FcvtzuScalarS64,
        (Prec::D, 0) => Code::FcvtzuScalarD32,
        (Prec::D, _) => Code::FcvtzuScalarD64,
        (Prec::H, 0) => Code::FcvtzuScalarH32,
        (Prec::H, _) => Code::FcvtzuScalarH64,
    }
}

// Fixed-point variants reuse the same per-(precision,width) split but with the
// dedicated fixed-point Code rows.
#[inline]
fn scvtf_fixed_code(p: Prec, sf: u32) -> Code {
    match (p, sf) {
        (Prec::S, 0) => Code::ScvtfFixedS32,
        (Prec::S, _) => Code::ScvtfFixedS64,
        (Prec::D, 0) => Code::ScvtfFixedD32,
        (Prec::D, _) => Code::ScvtfFixedD64,
        (Prec::H, 0) => Code::ScvtfFixedH32,
        (Prec::H, _) => Code::ScvtfFixedH64,
    }
}
#[inline]
fn ucvtf_fixed_code(p: Prec, sf: u32) -> Code {
    match (p, sf) {
        (Prec::S, 0) => Code::UcvtfFixedS32,
        (Prec::S, _) => Code::UcvtfFixedS64,
        (Prec::D, 0) => Code::UcvtfFixedD32,
        (Prec::D, _) => Code::UcvtfFixedD64,
        (Prec::H, 0) => Code::UcvtfFixedH32,
        (Prec::H, _) => Code::UcvtfFixedH64,
    }
}
#[inline]
fn fcvtzs_fixed_code(p: Prec, sf: u32) -> Code {
    match (p, sf) {
        (Prec::S, 0) => Code::FcvtzsFixedS32,
        (Prec::S, _) => Code::FcvtzsFixedS64,
        (Prec::D, 0) => Code::FcvtzsFixedD32,
        (Prec::D, _) => Code::FcvtzsFixedD64,
        (Prec::H, 0) => Code::FcvtzsFixedH32,
        (Prec::H, _) => Code::FcvtzsFixedH64,
    }
}
#[inline]
fn fcvtzu_fixed_code(p: Prec, sf: u32) -> Code {
    match (p, sf) {
        (Prec::S, 0) => Code::FcvtzuFixedS32,
        (Prec::S, _) => Code::FcvtzuFixedS64,
        (Prec::D, 0) => Code::FcvtzuFixedD32,
        (Prec::D, _) => Code::FcvtzuFixedD64,
        (Prec::H, 0) => Code::FcvtzuFixedH32,
        (Prec::H, _) => Code::FcvtzuFixedH64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{BufSink, FmtFormatter, Formatter};

    /// Decode `word` at ip 0 with all features and assert the rendered text.
    #[track_caller]
    fn render(word: u32, expected: &str) {
        let mut insn = Instruction::default();
        crate::decode::simd_fp::decode(word, 0, FeatureSet::ALL, &mut insn);
        let mut buf = [0u8; 128];
        let mut sink = BufSink::new(&mut buf);
        FmtFormatter::new().format(&insn, &mut sink);
        assert!(!sink.overflowed(), "overflow rendering {expected:?}");
        assert_eq!(sink.as_str(), expected, "word={word:#010x}");
    }

    /// Assert `word` decodes to the invalid sentinel.
    #[track_caller]
    fn invalid(word: u32) {
        let mut insn = Instruction::default();
        crate::decode::simd_fp::decode(word, 0, FeatureSet::ALL, &mut insn);
        assert!(insn.is_invalid(), "expected invalid for {word:#010x}");
    }

    #[test]
    fn dp1_basic() {
        render(0x1E604101, "fmov    d1, d8");
        render(0x1E20C2CF, "fabs    s15, s22");
        render(0x1EE0C35A, "fabs    h26, h26");
        render(0x1E214041, "fneg    s1, s2");
        render(0x1E61C041, "fsqrt   d1, d2");
    }

    #[test]
    fn dp1_fcvt() {
        render(0x1E22C0E0, "fcvt    d0, s7");
        render(0x1EE2C3AB, "fcvt    d11, h29");
        render(0x1E23C041, "fcvt    h1, s2"); // S->H
    }

    #[test]
    fn dp1_frint() {
        render(0x1E644041, "frintn  d1, d2");
        render(0x1E264041, "frinta  s1, s2");
        // FRINT32Z (FEAT_FRINTTS), single.
        render(0x1E284041, "frint32z s1, s2");
    }

    #[test]
    fn dp2() {
        render(0x1E602800, "fadd    d0, d0, d0");
        render(0x1E200801, "fmul    s1, s0, s0");
        render(0x1E601800, "fdiv    d0, d0, d0");
    }

    #[test]
    fn dp3() {
        render(0x1F4E3209, "fmadd   d9, d16, d14, d12");
        render(0x1F5994E5, "fmsub   d5, d7, d25, d5");
        render(0x1F79746A, "fnmadd  d10, d3, d25, d29");
        render(0x1F75A799, "fnmsub  d25, d28, d21, d9");
    }

    #[test]
    fn compare() {
        render(0x1E202108, "fcmp    s8, #0.0");
        render(0x1E212090, "fcmpe   s4, s1");
        render(0x1E2123A0, "fcmp    s29, s1");
        // Half-precision (FEAT_FP16) forms. The `#0.0` encoding ignores the Rm
        // field (word<20:16>), which may be non-zero in the wild.
        render(0x1EEC2118, "fcmpe   h8, #0.0");
        render(0x1EE52120, "fcmp    h9, h5");
        // Double `#0.0` with a non-zero (ignored) Rm field still renders `#0.0`.
        render(0x1E7F22F8, "fcmpe   d23, #0.0");
    }

    #[test]
    fn ccmp_and_sel() {
        render(0x1E743417, "fccmpe  d0, d20, #0x7, lo");
        render(0x1E746C6A, "fcsel   d10, d3, d20, vs");
        render(0x1E75ED1E, "fcsel   d30, d8, d21, al");
    }

    #[test]
    fn fmov_imm() {
        render(0x1E66700B, "fmov    d11, #19.0");
        render(0x1E79900C, "fmov    d12, #-0.21875");
        render(0x1E6D900D, "fmov    d13, #0.875");
        render(0x1E24B000, "fmov    s0, #10.5");
    }

    #[test]
    fn fmov_gp() {
        render(0x1E2602EA, "fmov    w10, s23");
        render(0x1E270041, "fmov    s1, w2");
        render(0x9E660041, "fmov    x1, d2"); // opcode 6: FP -> GP
        render(0x9E670041, "fmov    d1, x2"); // opcode 7: GP -> FP
    }

    #[test]
    fn half_precision_and_bf16() {
        // Half-precision data-processing (FEAT_FP16).
        render(0x1EE0C35A, "fabs    h26, h26");
        render(0x1EED2AF8, "fadd    h24, h23, h13");
        render(0x1EE09000, "fmov    h0, #2.5");
        // BFCVT (FEAT_BF16): Hd <- Sn.
        render(0x1E634041, "bfcvt   h1, s2");
        // FMOV between GP and the top half of a vector.
        render(0x9EAE0041, "fmov    x1, v2.d[1]");
        render(0x9EAF0041, "fmov    v1.d[1], x2");
    }

    #[test]
    fn conversions() {
        render(0x1E7802E5, "fcvtzs  w5, d23");
        render(0x1E620041, "scvtf   d1, w2");
        render(0x1E6300E0, "ucvtf   d0, w7");
        // Fixed-point.
        render(0x1E58ABAC, "fcvtzs  w12, d29, #0x16");
        render(0x1E438660, "ucvtf   d0, w19, #0x1f");
    }

    #[test]
    fn never_panics_sample() {
        for w in (0x1E00_0000u32..0x1E00_0000u32.wrapping_add(8192)).step_by(13) {
            let mut insn = Instruction::default();
            crate::decode::simd_fp::decode(w, 0, FeatureSet::ALL, &mut insn);
        }
        // A reserved ftype==10 dp1 form must be invalid.
        invalid(0x1EC0C041);
    }
}
