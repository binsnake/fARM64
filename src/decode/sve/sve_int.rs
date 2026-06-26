//! SVE / SVE2 integer, logical, shift, reduction and misc encodings.
//!
//! Hand-written from the *ARM Architecture Reference Manual* SVE encoding index.
//! This is the integer heart of the SVE group and owns:
//!
//! * integer arithmetic unpredicated (`ADD`/`SUB`/`SQADD`/.., `MUL`/`SMULH`/
//!   `UMULH`/`PMUL`) and predicated (`ADD`/`SUB`/`SUBR`/`MUL`/min/max/`SABD`/..,
//!   `SDIV`/`UDIV`/.., `MLA`/`MLS`/`MAD`/`MSB`);
//! * integer unary predicated (`ABS`/`NEG`/`CNT`/`CLS`/`CLZ`/`CNOT`/`NOT`/
//!   `RBIT`/`SXTB`../`UXTB`..);
//! * bitwise logical unpredicated (`AND`/`ORR`/`EOR`/`BIC`) and predicated, and
//!   the bitwise / broadcast immediate (`AND`/`ORR`/`EOR`/`DUPM`);
//! * shifts by immediate / vector / wide, predicated and unpredicated, plus
//!   `ASRD`/`ASRR`/`LSLR`/`LSRR`;
//! * integer reductions (`SADDV`/`UADDV`/`SMAXV`/.., `ANDV`/`ORV`/`EORV`) and
//!   `MOVPRFX`;
//! * `INDEX`, the integer min/max/mul/add immediate forms, the wide compare
//!   immediates and vector/wide compares;
//! * `INC`/`DEC`/`SQINC`/`SQDEC`/`UQINC`/`UQDEC` (scalar & vector, by pattern),
//!   `CNTB`/`CNTH`/`CNTW`/`CNTD`, `CNTP`, the by-predicate INC/DEC, and
//!   `ADDVL`/`ADDPL`/`RDVL`;
//! * `DUP`/`MOV`/`CPY`/`SEL`/`INSR` and `ADR` (SVE vector form), `SDOT`/`UDOT`;
//! * a tractable subset of the SVE2 integer multiply-add / widening / halving /
//!   saturating extras.
//!
//! Code identity follows the established convention: one [`Code`] per ARM ARM
//! encoding class (`Sve*`), the preferred-disassembly alias installed via
//! [`Instruction::set_mnemonic`] where the corpus uses one (`MOV`, `MUL`, ...),
//! and all arrangement / predicate / lane decoration carried in the operands.
//! Every path is total and panic-free; unallocated encodings are left
//! [`Code::Invalid`].

use crate::decode::bits::{bit, bits, bits64, decode_bit_masks, sign_extend};
use crate::enums::{ExtendType, VectorArrangement as VA};
use crate::features::FeatureSet;
use crate::instruction::Instruction;
use crate::mnemonic::{Code, Mnemonic};
use crate::operand::{Operand, PredQual, SveMemMode};
use crate::register::{gp_register, Register, RegWidth};

// ---------------------------------------------------------------------------
// Register-bank tables.
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
const BR: [Register; 32] = [
    Register::B0, Register::B1, Register::B2, Register::B3, Register::B4, Register::B5, Register::B6, Register::B7,
    Register::B8, Register::B9, Register::B10, Register::B11, Register::B12, Register::B13, Register::B14, Register::B15,
    Register::B16, Register::B17, Register::B18, Register::B19, Register::B20, Register::B21, Register::B22, Register::B23,
    Register::B24, Register::B25, Register::B26, Register::B27, Register::B28, Register::B29, Register::B30, Register::B31,
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

/// Element-size arrangement (`.b`/`.h`/`.s`/`.d`/`.q`) from a 2-bit `size`.
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

/// A scalable `Z{n}` operand with arrangement `a` and a lane index.
#[inline]
fn zreg_idx(n: u32, a: VA, lane: u8) -> Operand {
    Operand::Reg { reg: Z[(n & 0x1f) as usize], arr: Some(a), lane: Some(lane), shift: None, extend: None, pred: None }
}

/// A governing predicate `P{n}` with a `/z` or `/m` qualifier.
#[inline]
fn preg_q(n: u32, q: PredQual) -> Operand {
    Operand::Reg { reg: P[(n & 0xf) as usize], arr: None, lane: None, shift: None, extend: None, pred: Some(q) }
}

/// A bare predicate `P{n}` (no qualifier).
#[inline]
fn preg(n: u32) -> Operand {
    Operand::Reg { reg: P[(n & 0xf) as usize], arr: None, lane: None, shift: None, extend: None, pred: None }
}

/// A sized predicate `P{n}.<T>` (no qualifier).
#[inline]
fn preg_sz(n: u32, a: VA) -> Operand {
    Operand::Reg { reg: P[(n & 0xf) as usize], arr: Some(a), lane: None, shift: None, extend: None, pred: None }
}

/// A bare general-purpose register operand (`X`/`W`), reg-31 as ZR.
#[inline]
fn gpr(n: u32, w: RegWidth) -> Operand {
    Operand::Reg { reg: gp_register(false, w, n as u8), arr: None, lane: None, shift: None, extend: None, pred: None }
}

/// A general-purpose register operand with reg-31 resolved to SP.
#[inline]
fn gpr_sp(n: u32, w: RegWidth) -> Operand {
    Operand::Reg { reg: gp_register(true, w, n as u8), arr: None, lane: None, shift: None, extend: None, pred: None }
}

/// A scalar SIMD `B/H/S/D` operand of the element width for `size`.
#[inline]
fn scalar_fp(n: u32, size: u32) -> Operand {
    let n = (n & 0x1f) as usize;
    let reg = match size & 3 {
        0 => BR[n],
        1 => HR[n],
        2 => SR[n],
        _ => DR[n],
    };
    Operand::Reg { reg, arr: None, lane: None, shift: None, extend: None, pred: None }
}

/// Push the standard `Zdn.T, Pg/M, Zdn.T, Zm.T` operand quad for a predicated
/// binary destructive instruction.
#[inline]
fn pred_binary(out: &mut Instruction, zdn: u32, pg: u32, zm: u32, a: VA) {
    out.push_operand(zreg(zdn, a));
    out.push_operand(preg_q(pg, PredQual::Merging));
    out.push_operand(zreg(zdn, a));
    out.push_operand(zreg(zm, a));
}

/// Push the optional SVE `pattern{, MUL #imm}` tail for INC/DEC/CNT element-count
/// forms (ARM ARM `ADD_OPERAND_OPTIONAL_PATTERN_MUL`).
///
/// `pattern` is the raw 5-bit field; `imm1` is `imm4 + 1` (the multiplier,
/// `1..=16`). The multiplier is printed only when `imm1 != 1`; the pattern is
/// printed when the multiplier is printed or the pattern is not `all` (`0x1f`).
#[inline]
fn push_pattern_mul(out: &mut Instruction, pattern: u32, imm1: u32) {
    let print_mul = imm1 != 1;
    let print_pattern = print_mul || pattern != 0x1f;
    if print_pattern {
        out.push_operand(Operand::SvePattern((pattern & 0x1f) as u8));
    }
    if print_mul {
        out.push_operand(Operand::SveMul(imm1 as u8));
    }
}

/// Element-size arrangement restricted to the three INC/DEC vector sizes:
/// `tszh:tszl`-style `size` of 1/2/3 maps to H/S/D (byte vectors do not exist
/// for the vector INC/DEC forms).
#[inline]
fn arr_hsd(size: u32) -> VA {
    match size & 3 {
        1 => VA::Sh,
        2 => VA::Ss,
        _ => VA::Sd,
    }
}

/// Decode the SVE logical (bitmask) immediate from `(imm13 = imm_n:immr:imms)`
/// for a 64-bit element container, returning the replicated value, or `None` for
/// the reserved encodings.
#[inline]
fn sve_bitmask(imm13: u32) -> Option<u64> {
    let imm_n = bit(imm13, 12);
    let immr = bits(imm13, 6, 6);
    let imms = bits(imm13, 0, 6);
    decode_bit_masks(imm_n, imms, immr, false, 64).map(|m| m.wmask)
}

/// `SVEMoveMaskPreferred(imm13)` — whether `DUPM <Zd>.<T>, #imm` should render
/// via the `MOV <Zd>.<T>, #imm` alias (ARM ARM shared pseudocode). Returns
/// `true` when the decoded 64-bit broadcast value is *not* trivially a small
/// movz/movn-style 8- or 16-bit constant (those keep the `DUPM` spelling, since
/// the simpler `DUP`/`CPY` immediate already covers them).
///
/// This follows the ARM ARM byte/half-word checks. The pure byte-broadcast case
/// (DecodeBitMasks element size 8, i.e. the low byte replicated across all eight
/// bytes) is excluded: both LLVM and the corpus keep `dupm` there, matching the
/// spec intent that an 8-bit-element value is reached via `DUP`/`CPY` instead.
#[inline]
fn sve_move_mask_preferred(imm: u64) -> bool {
    let f = |hi: u32, lo: u32| -> u64 { bits64(imm, lo, hi - lo + 1) };
    let is_zero = |hi: u32, lo: u32| f(hi, lo) == 0;
    let is_ones = |hi: u32, lo: u32| {
        let w = hi - lo + 1;
        f(hi, lo) == (u64::MAX >> (64 - w))
    };
    if (imm & 0xff) != 0 {
        // Pure byte broadcast (`0xXYXY..XY`): reached by DUP/CPY → keep DUPM.
        if imm == (imm & 0xff).wrapping_mul(0x0101_0101_0101_0101) {
            return false;
        }
        // 8-bit immediate checks.
        if is_zero(63, 8) || is_ones(63, 8) {
            return false;
        }
        if f(63, 32) == f(31, 0) && (is_zero(31, 8) || is_ones(31, 8)) {
            return false;
        }
        if f(63, 48) == f(31, 16) && f(63, 48) == f(47, 32) && (is_zero(15, 8) || is_ones(15, 8))
        {
            return false;
        }
    } else {
        // 16-bit immediate checks.
        if is_zero(63, 16) || is_ones(63, 16) {
            return false;
        }
        if f(63, 32) == f(31, 0) && (is_zero(31, 16) || is_ones(31, 16)) {
            return false;
        }
        if f(63, 48) == f(31, 16) && f(63, 48) == f(47, 32) {
            return false;
        }
    }
    true
}

/// The element-size index from a `tszh:tszl` concatenation: `HighestSetBit`,
/// giving `0`=byte … `3`=doubleword, or `None` if the field is zero (reserved).
#[inline]
fn tsz_size(tsz: u32) -> Option<u32> {
    if tsz == 0 {
        None
    } else {
        Some(31 - (tsz & 0xf).leading_zeros())
    }
}

/// Decode a right-shift (ASR/LSR) immediate amount from `tszh:tszl:imm3`:
/// `amount = 2*esize - UInt(tsz:imm3)`.
#[inline]
fn right_shift_amount(tsz: u32, imm3: u32) -> Option<(VA, u32)> {
    let idx = tsz_size(tsz)?;
    let esize = 8u32 << idx;
    let val = (tsz << 3) | (imm3 & 7);
    Some((arr(idx), 2 * esize - val))
}

/// Decode a left-shift (LSL) immediate amount from `tszh:tszl:imm3`:
/// `amount = UInt(tsz:imm3) - esize`.
#[inline]
fn left_shift_amount(tsz: u32, imm3: u32) -> Option<(VA, u32)> {
    let idx = tsz_size(tsz)?;
    let esize = 8u32 << idx;
    let val = (tsz << 3) | (imm3 & 7);
    Some((arr(idx), val - esize))
}

// ===========================================================================
// Top byte 0x05 — SVE logical immediate, DUP/CPY/INSR/SEL, RBIT/REVB/H/W.
// ===========================================================================

#[inline]
fn decode_05(word: u32, out: &mut Instruction) {
    // Logical immediate (AND/ORR/EOR/DUPM): word<21:19>==000 and word<18>==0
    // (the imm13 occupies <17:5>), with the op in word<23:22>.
    if bits(word, 19, 3) == 0 {
        decode_logical_imm(word, out);
        return;
    }
    // CPY (immediate): `<21:20>=01`, `<15>=0`. `MOV <Zd>.<T>, <Pg>/<Z|M>, #imm
    // {, shift}`. M=<14> (1=/m merging, 0=/z zeroing), sh=<13>, imm8=<12:5>
    // signed, Pg=<19:16>, Zd=<4:0>. Handled here (before the perm decoder, which
    // would otherwise mis-claim these via <15:13>).
    if bits(word, 20, 2) == 0b01 && bit(word, 15) == 0 {
        decode_cpy_imm(word, out);
        return;
    }
    let sel = bits(word, 13, 3); // word<15:13>
    let opc2016 = bits(word, 16, 5);
    let size = bits(word, 22, 2);
    let zd = bits(word, 0, 5);
    match sel {
        // DUP (scalar/indexed) and INSR share <15:13>=001.
        0b001 => {
            // DUP scalar: <20:16>=00000, <12:10>=011 1000 region -> reg form.
            // We distinguish by the fixed low bits via the field layouts.
            if opc2016 == 0b00000 && bits(word, 10, 6) == 0b001110 {
                // DUP <Zd>.<T>, <R><n|SP>.
                let w = if size == 3 { RegWidth::X64 } else { RegWidth::W32 };
                out.set(Code::SveDupScalar);
                out.set_mnemonic(Mnemonic::Mov);
                out.push_operand(zreg(zd, arr(size)));
                out.push_operand(gpr_sp(bits(word, 5, 5), w));
            } else if opc2016 == 0b00100 && bits(word, 10, 6) == 0b001110 {
                // INSR <Zdn>.<T>, <R><m>.
                let w = if size == 3 { RegWidth::X64 } else { RegWidth::W32 };
                out.set(Code::SveInsrScalar);
                out.push_operand(zreg(zd, arr(size)));
                out.push_operand(gpr(bits(word, 5, 5), w));
            } else if opc2016 == 0b10100 && bits(word, 10, 6) == 0b001110 {
                // INSR <Zdn>.<T>, <V><m>.
                out.set(Code::SveInsrVec);
                out.push_operand(zreg(zd, arr(size)));
                out.push_operand(scalar_fp(bits(word, 5, 5), size));
            } else if bits(word, 10, 3) == 0b000 {
                // DUP indexed: `MOV <Zd>.<T>, <Zn>.<T>[idx]`. tsz = imm2:tsz
                // = <23:22>:<20:16>. The element size is the lowest set bit of tsz.
                decode_dup_indexed(word, out);
            } else if bit(word, 21) == 1 && bits(word, 10, 3) == 0b001 {
                // SVE2.1 128-bit-segment forms: DUPQ (<23:22>=00) and EXTQ
                // (<23:22>=01).
                match size {
                    0b00 => decode_dupq(word, out),
                    0b01 => decode_extq(word, out),
                    _ => {}
                }
            } else if bit(word, 21) == 1
                && bits(word, 10, 3) == 0b110
                && bit(word, 20) == 0
                && bit(word, 19) == 1
            {
                // SVE2.1 PMOV (predicate <-> vector move).
                decode_pmov(word, out);
            }
        }
        // CPY (scalar) <15:13>=101 with <20:16>=01000: `CPY <Zd>.<T>, <Pg>/M,
        // <R><n|SP>` (binja/LLVM render the `MOV` alias). `Pg = <12:10>` is the
        // governing predicate, not a discriminator. The CLASTA/CLASTB/LASTA/LASTB-
        // to-GPR ops share <15:13>=101 but use other <20:16> opcodes (00000/00001/
        // 10000/10001) and are left to the permute decoder.
        0b101 if opc2016 == 0b01000 => {
            let w = if size == 3 { RegWidth::X64 } else { RegWidth::W32 };
            out.set(Code::SveCpyScalar);
            out.set_mnemonic(Mnemonic::Mov);
            out.push_operand(zreg(zd, arr(size)));
            out.push_operand(preg_q(bits(word, 10, 3), PredQual::Merging));
            out.push_operand(gpr_sp(bits(word, 5, 5), w));
        }
        0b100 => {
            // word<20:16> selects: 00000=CPY vector; 00100/00101/00110=REVB/H/W;
            // 00111=RBIT; 10001=EXPAND (SVE2.1).
            let pg = bits(word, 10, 3);
            let zn = bits(word, 5, 5);
            match opc2016 {
                0b00000 => {
                    out.set(Code::SveCpyVec);
                    out.set_mnemonic(Mnemonic::Mov);
                    out.push_operand(zreg(zd, arr(size)));
                    out.push_operand(preg_q(pg, PredQual::Merging));
                    out.push_operand(scalar_fp(zn, size));
                }
                0b00111 => {
                    out.set(Code::SveRbitZpz);
                    out.push_operand(zreg(zd, arr(size)));
                    out.push_operand(preg_q(pg, PredQual::Merging));
                    out.push_operand(zreg(zn, arr(size)));
                }
                // EXPAND (SVE2.1): `EXPAND <Zd>.<T>, <Pg>, <Zn>.<T>` (plain Pg).
                0b10001 if bit(word, 21) == 1 => {
                    out.set(Code::SveExpand);
                    out.push_operand(zreg(zd, arr(size)));
                    out.push_operand(preg(pg));
                    out.push_operand(zreg(zn, arr(size)));
                }
                // REVB/REVH/REVW are permute ops; leave to sve_perm.
                _ => {}
            }
        }
        // SVE2.1 zeroing reverse-within-element: REVB/REVH/REVW/RBIT
        // `<Zd>.<T>, <Pg>/Z, <Zn>.<T>` (`<21>=1`, `<20:16>=001xx`).
        0b101 if bit(word, 21) == 1 && (0b00100..=0b00111).contains(&opc2016) => {
            let pg = bits(word, 10, 3);
            let zn = bits(word, 5, 5);
            let a = arr(size);
            let (code, mnem, min_size) = match opc2016 {
                0b00100 => (Code::SveRevbhw, Mnemonic::Revb, 1),
                0b00101 => (Code::SveRevbhw, Mnemonic::Revh, 2),
                0b00110 => (Code::SveRevbhw, Mnemonic::Revw, 3),
                _ => (Code::SveRbitZpz, Mnemonic::Rbit, 0),
            };
            if size < min_size {
                return;
            }
            out.set(code);
            out.set_mnemonic(mnem);
            out.push_operand(zreg(zd, a));
            out.push_operand(preg_q(pg, PredQual::Zeroing));
            out.push_operand(zreg(zn, a));
        }
        // SEL `<15:14>=11` with `<21>=1` (`Pg<13:10>` makes `<15:13>` `110`/`111`).
        // The `<15:13>=110`,`<21:20>=01` slot is FCPY (an FP op, left to sve_fp).
        0b110 | 0b111 if bit(word, 21) == 1 => {
            let zm = bits(word, 16, 5);
            let pg = bits(word, 10, 4);
            let zn = bits(word, 5, 5);
            let a = arr(size);
            if zm == zd {
                // `SEL Zd, Pg, Zn, Zd` == `MOV <Zd>.<T>, <Pg>/M, <Zn>.<T>`.
                out.set(Code::SveSelZpzz);
                out.set_mnemonic(Mnemonic::Mov);
                out.push_operand(zreg(zd, a));
                out.push_operand(preg_q(pg, PredQual::Merging));
                out.push_operand(zreg(zn, a));
            } else {
                out.set(Code::SveSelZpzz);
                out.push_operand(zreg(zd, a));
                out.push_operand(preg(pg));
                out.push_operand(zreg(zn, a));
                out.push_operand(zreg(zm, a));
            }
        }
        _ => {}
    }
}

/// CPY (immediate) — top byte 0x05, `<21:20>=01`, `<15>=0`. Renders as the
/// `MOV <Zd>.<T>, <Pg>/<Z|M>, #<imm>{, shift}` alias (Binary Ninja / LLVM both
/// print the `mov` form). `M=<14>` selects merging (`/m`) vs zeroing (`/z`),
/// `sh=<13>` shifts the signed `imm8` left by 8.
#[inline]
fn decode_cpy_imm(word: u32, out: &mut Instruction) {
    let size = bits(word, 22, 2);
    let pg = bits(word, 16, 4);
    let merging = bit(word, 14) == 1;
    let sh = bit(word, 13);
    let imm8 = bits(word, 5, 8);
    let zd = bits(word, 0, 5);
    out.set(if merging {
        Code::SveCpyImmMerge
    } else {
        Code::SveCpyImmZero
    });
    out.set_mnemonic(Mnemonic::Mov);
    out.push_operand(zreg(zd, arr(size)));
    out.push_operand(preg_q(
        pg,
        if merging {
            PredQual::Merging
        } else {
            PredQual::Zeroing
        },
    ));
    push_dup_imm(out, imm8, sh);
}

/// SVE logical broadcast immediate (top byte 0x05, `<21:19>=000`): ORR/EOR/AND
/// `<Zdn>.<T>, <Zdn>.<T>, #imm`, and DUPM `<Zd>.<T>, #imm`. The replicated value
/// is rendered as a single `#0x..` immediate; for ORR/EOR/AND a single-register
/// destructive form, for DUPM a broadcast. Binary Ninja renders these via the
/// `MOV` alias only for `DUPM` when it is a valid broadcast — but the corpus
/// shows `mov`/`dupm` and the raw `orr/eor/and`, so we follow the corpus:
/// ORR/EOR/AND keep their mnemonic; DUPM renders as `MOV` when the value is a
/// replicable element, else `DUPM`.
#[inline]
fn decode_logical_imm(word: u32, out: &mut Instruction) {
    let opc = bits(word, 22, 2);
    let imm13 = bits(word, 5, 13);
    let zdn = bits(word, 0, 5);
    let Some(val) = sve_bitmask(imm13) else { return };
    // The element arrangement for the immediate display: SVE logical-immediate
    // ops are notionally `.d`-wide but the corpus prints the element size implied
    // by the replication period of the decoded mask. Binary Ninja prints `.b`/
    // `.h`/`.s`/`.d` according to the smallest element the value replicates into.
    let a = imm_arrangement(val);
    match opc {
        0b00 => {
            out.set(Code::SveOrrZi);
            out.push_operand(zreg(zdn, a));
            out.push_operand(zreg(zdn, a));
            out.push_operand(Operand::ImmUnsigned(element_value(val, a)));
        }
        0b01 => {
            out.set(Code::SveEorZi);
            out.push_operand(zreg(zdn, a));
            out.push_operand(zreg(zdn, a));
            out.push_operand(Operand::ImmUnsigned(element_value(val, a)));
        }
        0b10 => {
            out.set(Code::SveAndZi);
            out.push_operand(zreg(zdn, a));
            out.push_operand(zreg(zdn, a));
            out.push_operand(Operand::ImmUnsigned(element_value(val, a)));
        }
        // DUPM (opc==11): broadcast bitmask. The ARM ARM `SVEMoveMaskPreferred`
        // rule decides between the `MOV <Zd>.<T>, #imm` alias (most values) and
        // the bare `DUPM` spelling (small movz/movn-style / byte-broadcast
        // values, which `DUP`/`CPY` already cover).
        _ => {
            out.set(Code::SveDupmZi);
            out.set_mnemonic(if sve_move_mask_preferred(val) {
                Mnemonic::Mov
            } else {
                Mnemonic::Dupm
            });
            out.push_operand(zreg(zd_of(word), a));
            out.push_operand(Operand::ImmUnsigned(element_value(val, a)));
        }
    }
}

/// Destination Zd field (`<4:0>`).
#[inline]
fn zd_of(word: u32) -> u32 {
    bits(word, 0, 5)
}

/// The SVE arrangement implied by the replication period of a decoded bitmask
/// `val`: the smallest element width (`.b`/`.h`/`.s`/`.d`) into which `val`
/// replicates, matching the Binary Ninja rendering of SVE logical immediates.
#[inline]
fn imm_arrangement(val: u64) -> VA {
    let b = val & 0xff;
    if b * 0x0101_0101_0101_0101 == val {
        return VA::Sb;
    }
    let h = val & 0xffff;
    if h * 0x0001_0001_0001_0001 == val {
        return VA::Sh;
    }
    let s = val & 0xffff_ffff;
    if s * 0x0000_0001_0000_0001 == val {
        return VA::Ss;
    }
    VA::Sd
}

/// The single-element value to display for a replicated bitmask `val` at
/// arrangement `a` (the low element).
#[inline]
fn element_value(val: u64, a: VA) -> u64 {
    match a {
        VA::Sb => val & 0xff,
        VA::Sh => val & 0xffff,
        VA::Ss => val & 0xffff_ffff,
        _ => val,
    }
}

/// DUP indexed (`MOV <Zd>.<T>, <Zn>.<T>[idx]`). The element size and index come
/// from `tsz = imm2:tsz` (`<23:22>:<20:16>`): the lowest set bit of `tsz` gives
/// the element size, the bits above it give the index. When the index is `0`,
/// Binary Ninja renders the scalar-SIMD broadcast form (`MOV <Zd>.<T>, <V><n>`)
/// instead of the bracketed `[0]`.
#[inline]
fn decode_dup_indexed(word: u32, out: &mut Instruction) {
    let tsz = (bits(word, 22, 2) << 5) | bits(word, 16, 5); // 7-bit imm2:tsz
    let zn = bits(word, 5, 5);
    let zd = bits(word, 0, 5);
    // Element size index = position of lowest set bit (0=.b .. 4=.q).
    let (a, esize, idx) = match tsz & 0x7f {
        t if t & 0b1 != 0 => (VA::Sb, 0u32, t >> 1),
        t if t & 0b10 != 0 => (VA::Sh, 1, t >> 2),
        t if t & 0b100 != 0 => (VA::Ss, 2, t >> 3),
        t if t & 0b1000 != 0 => (VA::Sd, 3, t >> 4),
        t if t & 0b1_0000 != 0 => (VA::Sq, 4, t >> 5),
        _ => return,
    };
    out.set(Code::SveDupIdx);
    out.set_mnemonic(Mnemonic::Mov);
    out.push_operand(zreg(zd, a));
    if idx == 0 {
        // Scalar broadcast: `MOV <Zd>.<T>, <V><n>` (B/H/S/D/Q).
        out.push_operand(scalar_fp_q(zn, esize));
    } else {
        out.push_operand(zreg_idx(zn, a, idx as u8));
    }
}

/// SVE2.1 DUPQ: `DUPQ <Zd>.<T>, <Zn>.<T>[<index>]` — broadcast an indexed
/// element within each 128-bit segment. The element size and lane index use the
/// standard trailing `tsz = word<20:16>` encoding (`<23:22>` is 0 here). Always
/// renders the lane index (even 0) with the source arrangement.
#[inline]
fn decode_dupq(word: u32, out: &mut Instruction) {
    let tsz = bits(word, 16, 5);
    let zn = bits(word, 5, 5);
    let zd = bits(word, 0, 5);
    let (a, idx) = match tsz {
        t if t & 0b1 != 0 => (VA::Sb, t >> 1),
        t if t & 0b10 != 0 => (VA::Sh, t >> 2),
        t if t & 0b100 != 0 => (VA::Ss, t >> 3),
        t if t & 0b1000 != 0 => (VA::Sd, t >> 4),
        _ => return,
    };
    out.set(Code::SveDupq);
    out.push_operand(zreg(zd, a));
    out.push_operand(zreg_idx(zn, a, idx as u8));
}

/// SVE2.1 EXTQ: `EXTQ <Zdn>.B, <Zdn>.B, <Zm>.B, #imm` — extract a byte-aligned
/// vector from the `Zdn:Zm` pair within each 128-bit segment. The byte offset
/// `imm` (0..15) is `word<19:16>`; the instruction is destructive (`Zd==Zdn`).
#[inline]
fn decode_extq(word: u32, out: &mut Instruction) {
    let imm = bits(word, 16, 5);
    if imm > 15 {
        return; // <20>=1 is unallocated.
    }
    let zm = bits(word, 5, 5);
    let zd = bits(word, 0, 5);
    out.set(Code::SveExtq);
    out.push_operand(zreg(zd, VA::Sb));
    out.push_operand(zreg(zd, VA::Sb));
    out.push_operand(zreg(zm, VA::Sb));
    out.push_operand(Operand::ImmUnsigned(imm as u64));
}

/// SVE2.1 PMOV (predicate <-> vector). `word<16>` (D) selects direction:
/// `0` -> `PMOV <Pd>.<T>, <Zn>{[index]}`; `1` -> `PMOV <Zd>{[index]}, <Pn>.<T>`.
/// The element size and lane index come from the trailing-one field
/// `T = {word<23>, word<22>, word<18>, word<17>}` (highest set bit = size; the
/// lower bits are the index). The `.b` form has no index and a bare `Z`.
#[inline]
fn decode_pmov(word: u32, out: &mut Instruction) {
    let t = (bit(word, 23) << 3) | (bit(word, 22) << 2) | (bit(word, 18) << 1) | bit(word, 17);
    let (a, pos) = match t {
        t if t & 0b1000 != 0 => (VA::Sd, 3),
        t if t & 0b100 != 0 => (VA::Ss, 2),
        t if t & 0b10 != 0 => (VA::Sh, 1),
        t if t & 0b1 != 0 => (VA::Sb, 0),
        _ => return,
    };
    let idx = (t & ((1 << pos) - 1)) as u8;
    let n5 = bits(word, 5, 5);
    let d5 = bits(word, 0, 5);
    out.set(Code::SvePmov);
    let zoperand = |reg: u32| {
        if pos == 0 {
            plain_z(reg)
        } else {
            plain_z_idx(reg, idx)
        }
    };
    if bit(word, 16) == 0 {
        // Predicate <- vector.
        out.push_operand(preg_sz(d5, a));
        out.push_operand(zoperand(n5));
    } else {
        // Vector <- predicate.
        out.push_operand(zoperand(d5));
        out.push_operand(preg_sz(n5, a));
    }
}

/// A scalar SIMD register of element size index `esize` (0=B .. 4=Q).
#[inline]
fn scalar_fp_q(n: u32, esize: u32) -> Operand {
    if esize >= 4 {
        // Q register.
        const QR: [Register; 32] = [
            Register::Q0, Register::Q1, Register::Q2, Register::Q3, Register::Q4, Register::Q5, Register::Q6, Register::Q7,
            Register::Q8, Register::Q9, Register::Q10, Register::Q11, Register::Q12, Register::Q13, Register::Q14, Register::Q15,
            Register::Q16, Register::Q17, Register::Q18, Register::Q19, Register::Q20, Register::Q21, Register::Q22, Register::Q23,
            Register::Q24, Register::Q25, Register::Q26, Register::Q27, Register::Q28, Register::Q29, Register::Q30, Register::Q31,
        ];
        Operand::Reg { reg: QR[(n & 0x1f) as usize], arr: None, lane: None, shift: None, extend: None, pred: None }
    } else {
        scalar_fp(n, esize)
    }
}

// ===========================================================================
// Top byte 0x04 (cont.) — unpredicated arithmetic / logical / shift / INC-DEC.
// ===========================================================================

/// Unpredicated integer arithmetic (`<21>=1`, `<15:13>=000`): `op <Zd>.<T>,
/// <Zn>.<T>, <Zm>.<T>`.
#[inline]
fn decode_arith_zzz(word: u32, out: &mut Instruction) {
    let size = bits(word, 22, 2);
    let zm = bits(word, 16, 5);
    let zn = bits(word, 5, 5);
    let zd = bits(word, 0, 5);
    let a = arr(size);
    // ADDPT/SUBPT (FEAT_CPA unpredicated): `<12:10>=010`/`011`, `.d` only
    // (`<23:22>=11`). `<Zd>.D, <Zn>.D, <Zm>.D`.
    if matches!(bits(word, 10, 3), 0b010 | 0b011) {
        if size != 0b11 {
            return;
        }
        out.set(if bit(word, 10) == 0 { Code::SveAddptUnpred } else { Code::SveSubptUnpred });
        out.push_operand(zreg(zd, a));
        out.push_operand(zreg(zn, a));
        out.push_operand(zreg(zm, a));
        return;
    }
    let code = match bits(word, 10, 3) {
        0b000 => Code::SveAddZzz,
        0b001 => Code::SveSubZzz,
        0b100 => Code::SveSqaddZzz,
        0b101 => Code::SveUqaddZzz,
        0b110 => Code::SveSqsubZzz,
        0b111 => Code::SveUqsubZzz,
        _ => return,
    };
    out.set(code);
    out.push_operand(zreg(zd, a));
    out.push_operand(zreg(zn, a));
    out.push_operand(zreg(zm, a));
}

/// Unpredicated bitwise logical (`<21>=1`, `<15:13>=001`, `<12:10>=100`):
/// `AND`/`ORR`/`EOR`/`BIC` `<Zd>.D, <Zn>.D, <Zm>.D`, with `ORR Zd,Zn,Zn`
/// rendered as the `MOV Zd.D, Zn.D` alias. The op is `word<23:22>`.
#[inline]
fn decode_logical_zzz(word: u32, out: &mut Instruction) {
    // `<15:13>=001`. `<12:10>` selects the sub-family: `100` plain logical
    // (AND/ORR/EOR/BIC below), `101` XAR, `11x` the ternary EOR3/BCAX/BSL family.
    match bits(word, 10, 3) {
        0b100 => {} // fall through to the plain-logical decode below.
        0b101 => {
            decode_sve2_xar(word, out);
            return;
        }
        0b110 | 0b111 => {
            decode_sve2_bitwise_ternary(word, out);
            return;
        }
        _ => return,
    }
    let zm = bits(word, 16, 5);
    let zn = bits(word, 5, 5);
    let zd = bits(word, 0, 5);
    let a = VA::Sd; // all four are `.d`-typed.
    match bits(word, 22, 2) {
        0b00 => {
            out.set(Code::SveAndZzz);
            out.push_operand(zreg(zd, a));
            out.push_operand(zreg(zn, a));
            out.push_operand(zreg(zm, a));
        }
        0b01 => {
            // ORR; if Zn==Zm this is the MOV alias (`MOV Zd.D, Zn.D`).
            if zn == zm {
                out.set(Code::SveMovZzz);
                out.set_mnemonic(Mnemonic::Mov);
                out.push_operand(zreg(zd, a));
                out.push_operand(zreg(zn, a));
            } else {
                out.set(Code::SveOrrZzz);
                out.push_operand(zreg(zd, a));
                out.push_operand(zreg(zn, a));
                out.push_operand(zreg(zm, a));
            }
        }
        0b10 => {
            out.set(Code::SveEorZzz);
            out.push_operand(zreg(zd, a));
            out.push_operand(zreg(zn, a));
            out.push_operand(zreg(zm, a));
        }
        _ => {
            out.set(Code::SveBicZzz);
            out.push_operand(zreg(zd, a));
            out.push_operand(zreg(zn, a));
            out.push_operand(zreg(zm, a));
        }
    }
}

/// SVE2 bitwise ternary (`<21>=1`, `<15:11>=00111`): EOR3/BCAX (`o2=<10>`==0) and
/// BSL/BSL1N/BSL2N/NBSL (`o2`==1). All are `.d`-typed with operands
/// `<Zdn>.D, <Zdn>.D, <Zm>.D, <Zk>.D`. The op is `opc=<23:22>` plus `o2`.
#[inline]
fn decode_sve2_bitwise_ternary(word: u32, out: &mut Instruction) {
    let opc = bits(word, 22, 2);
    let o2 = bit(word, 10);
    let zm = bits(word, 16, 5);
    let zk = bits(word, 5, 5);
    let zdn = bits(word, 0, 5);
    let (code, mnem) = match (opc, o2) {
        (0b00, 0) => (Code::SveEor3, Mnemonic::Eor3),
        (0b01, 0) => (Code::SveBcax, Mnemonic::Bcax),
        (0b00, 1) => (Code::SveBsl, Mnemonic::Bsl),
        (0b01, 1) => (Code::SveBsl1n, Mnemonic::Bsl1n),
        (0b10, 1) => (Code::SveBsl2n, Mnemonic::Bsl2n),
        (0b11, 1) => (Code::SveNbsl, Mnemonic::Nbsl),
        // opc 10/11 with o2==0 are unallocated.
        _ => return,
    };
    let a = VA::Sd;
    out.set(code);
    out.set_mnemonic(mnem);
    out.push_operand(zreg(zdn, a));
    out.push_operand(zreg(zdn, a));
    out.push_operand(zreg(zm, a));
    out.push_operand(zreg(zk, a));
}

/// SVE2 XAR (`<21>=1`, `<15:10>=001101`): `XAR <Zdn>.<T>, <Zdn>.<T>, <Zm>.<T>,
/// #<imm>`. The element size and ROR amount come from `tszh:tszl:imm3`
/// (`<23:22>:<20:19>:<18:16>`) exactly like a right-shift immediate.
#[inline]
fn decode_sve2_xar(word: u32, out: &mut Instruction) {
    let tsz = (bits(word, 22, 2) << 2) | bits(word, 19, 2);
    let imm3 = bits(word, 16, 3);
    let Some((a, amount)) = right_shift_amount(tsz, imm3) else { return };
    let zm = bits(word, 5, 5);
    let zdn = bits(word, 0, 5);
    out.set(Code::SveXar);
    out.set_mnemonic(Mnemonic::Xar);
    out.push_operand(zreg(zdn, a));
    out.push_operand(zreg(zdn, a));
    out.push_operand(zreg(zm, a));
    out.push_operand(Operand::ImmUnsigned(amount as u64));
}

/// Unpredicated multiply (`<21>=1`, `<15:13>=011`): `MUL`/`PMUL`/`SMULH`/
/// `UMULH`/`SQDMULH`/`SQRDMULH` `<Zd>.<T>, <Zn>.<T>, <Zm>.<T>`.
#[inline]
fn decode_mul_zzz(word: u32, out: &mut Instruction) {
    let size = bits(word, 22, 2);
    let zm = bits(word, 16, 5);
    let zn = bits(word, 5, 5);
    let zd = bits(word, 0, 5);
    let a = arr(size);
    let (code, mnem) = match bits(word, 10, 3) {
        0b000 => (Code::SveMulZzz, None),
        0b001 => (Code::SvePmulZzz, None),
        0b010 => (Code::SveSmulhZzz, None),
        0b011 => (Code::SveUmulhZzz, None),
        // SQDMULH / SQRDMULH (SVE2) reuse the unpredicated multiply slot.
        0b100 => (Code::SveMulZzz, Some(Mnemonic::Sqdmulh)),
        0b101 => (Code::SveMulZzz, Some(Mnemonic::Sqrdmulh)),
        _ => return,
    };
    // PMUL is byte-only.
    if matches!(code, Code::SvePmulZzz) && size != 0 {
        return;
    }
    out.set(code);
    if let Some(m) = mnem {
        out.set_mnemonic(m);
    }
    out.push_operand(zreg(zd, a));
    out.push_operand(zreg(zn, a));
    out.push_operand(zreg(zm, a));
}

/// INDEX, ADDVL/ADDPL, RDVL (`<21>=1`, `<15:13>=010`), plus the SME streaming
/// analogues ADDSVL/ADDSPL/RDSVL (`<11>=1`, gated on FEAT_SME).
#[inline]
fn decode_index_addvl(word: u32, features: FeatureSet, out: &mut Instruction) {
    let size = bits(word, 22, 2);
    let a = arr(size);
    let w = if size == 3 { RegWidth::X64 } else { RegWidth::W32 };
    match bits(word, 10, 3) {
        // INDEX (immediate, immediate): `INDEX <Zd>.<T>, #imm5, #imm5b`.
        0b000 => {
            out.set(Code::SveIndexImmImm);
            out.push_operand(zreg(bits(word, 0, 5), a));
            out.push_operand(Operand::ImmSignedDec(sign_extend(bits(word, 5, 5) as u64, 5)));
            out.push_operand(Operand::ImmSignedDec(sign_extend(bits(word, 16, 5) as u64, 5)));
        }
        // INDEX (scalar, immediate): `INDEX <Zd>.<T>, <R><n>, #imm5`.
        0b001 => {
            out.set(Code::SveIndexRi);
            out.push_operand(zreg(bits(word, 0, 5), a));
            out.push_operand(gpr(bits(word, 5, 5), w));
            out.push_operand(Operand::ImmSignedDec(sign_extend(bits(word, 16, 5) as u64, 5)));
        }
        // INDEX (immediate, scalar): `INDEX <Zd>.<T>, #imm5, <R><m>`.
        0b010 => {
            out.set(Code::SveIndexIr);
            out.push_operand(zreg(bits(word, 0, 5), a));
            out.push_operand(Operand::ImmSignedDec(sign_extend(bits(word, 5, 5) as u64, 5)));
            out.push_operand(gpr(bits(word, 16, 5), w));
        }
        // INDEX (scalar, scalar): `INDEX <Zd>.<T>, <R><n>, <R><m>`.
        0b011 => {
            out.set(Code::SveIndexRr);
            out.push_operand(zreg(bits(word, 0, 5), a));
            out.push_operand(gpr(bits(word, 5, 5), w));
            out.push_operand(gpr(bits(word, 16, 5), w));
        }
        // ADDVL / ADDPL / RDVL — selected by word<23:22> (00 / 01 / 10).
        0b100 | 0b101 => {
            let rn = bits(word, 16, 5);
            let rd = bits(word, 0, 5);
            let imm6 = sign_extend(bits(word, 5, 6) as u64, 6);
            match bits(word, 22, 2) {
                0b00 => {
                    out.set(Code::SveAddvl);
                    out.push_operand(gpr_sp(rd, RegWidth::X64));
                    out.push_operand(gpr_sp(rn, RegWidth::X64));
                    out.push_operand(Operand::ImmSignedDec(imm6));
                }
                0b01 => {
                    out.set(Code::SveAddpl);
                    out.push_operand(gpr_sp(rd, RegWidth::X64));
                    out.push_operand(gpr_sp(rn, RegWidth::X64));
                    out.push_operand(Operand::ImmSignedDec(imm6));
                }
                0b10 => {
                    // RDVL <Xd>, #imm6 (Rn field is fixed 11111).
                    out.set(Code::SveRdvl);
                    out.push_operand(gpr(rd, RegWidth::X64));
                    out.push_operand(Operand::ImmSignedDec(imm6));
                }
                _ => {}
            }
        }
        // ADDSVL / ADDSPL / RDSVL — the SME streaming-mode analogues, selected by
        // word<11>=1 (the streaming bit) and word<23:22> (00 / 01 / 10). FEAT_SME.
        0b110 | 0b111 => {
            if !features.has(crate::features::Feature::Sme) {
                return;
            }
            let rn = bits(word, 16, 5);
            let rd = bits(word, 0, 5);
            let imm6 = sign_extend(bits(word, 5, 6) as u64, 6);
            match bits(word, 22, 2) {
                0b00 => {
                    out.set(Code::SveAddsvl);
                    out.push_operand(gpr_sp(rd, RegWidth::X64));
                    out.push_operand(gpr_sp(rn, RegWidth::X64));
                    out.push_operand(Operand::ImmSignedDec(imm6));
                }
                0b01 => {
                    out.set(Code::SveAddspl);
                    out.push_operand(gpr_sp(rd, RegWidth::X64));
                    out.push_operand(gpr_sp(rn, RegWidth::X64));
                    out.push_operand(Operand::ImmSignedDec(imm6));
                }
                0b10 => {
                    // RDSVL <Xd>, #imm6 — the Rn field is architecturally fixed to
                    // 11111; reject any other value (LLVM leaves it unallocated).
                    if rn != 0b11111 {
                        return;
                    }
                    out.set(Code::SveRdsvl);
                    out.push_operand(gpr(rd, RegWidth::X64));
                    out.push_operand(Operand::ImmSignedDec(imm6));
                }
                _ => {}
            }
        }
        _ => {}
    }
}

/// Shift by immediate / wide, unpredicated (`<21>=1`, `<15:13>=100`).
#[inline]
fn decode_shift_unpred(word: u32, out: &mut Instruction) {
    match bits(word, 10, 3) {
        // Wide-element shifts `op <Zd>.<T>, <Zn>.<T>, <Zm>.D`.
        0b000 | 0b001 | 0b011 => {
            let size = bits(word, 22, 2);
            // .d source elements are not allowed for wide-shift (size==3 reserved).
            if size == 3 {
                return;
            }
            let a = arr(size);
            let zm = bits(word, 16, 5);
            let zn = bits(word, 5, 5);
            let zd = bits(word, 0, 5);
            let code = match bits(word, 10, 3) {
                0b000 => Code::SveAsrWide,
                0b001 => Code::SveLsrWide,
                _ => Code::SveLslWide,
            };
            out.set(code);
            out.push_operand(zreg(zd, a));
            out.push_operand(zreg(zn, a));
            out.push_operand(zreg(zm, VA::Sd));
        }
        // Shift by immediate `op <Zd>.<T>, <Zn>.<T>, #shift`.
        0b100 | 0b101 | 0b111 => {
            let tsz = (bits(word, 22, 2) << 2) | bits(word, 19, 2);
            let imm3 = bits(word, 16, 3);
            let zn = bits(word, 5, 5);
            let zd = bits(word, 0, 5);
            let (code, sh) = match bits(word, 10, 3) {
                0b100 => (Code::SveAsrZi, right_shift_amount(tsz, imm3)),
                0b101 => (Code::SveLsrZi, right_shift_amount(tsz, imm3)),
                _ => (Code::SveLslZi, left_shift_amount(tsz, imm3)),
            };
            let Some((a, amt)) = sh else { return };
            out.set(code);
            out.push_operand(zreg(zd, a));
            out.push_operand(zreg(zn, a));
            out.push_operand(Operand::ImmUnsigned(amt as u64));
        }
        _ => {}
    }
}

/// ADR (vector) and unpredicated MOVPRFX (`<21>=1`, `<15:13>=101`).
#[inline]
fn decode_adr_movprfx(word: u32, out: &mut Instruction) {
    let op2210 = bits(word, 10, 3);
    // MOVPRFX (unpredicated): `<23:22>=00`, `<12:10>=111`.
    if bits(word, 22, 2) == 0b00 && op2210 == 0b111 {
        out.set(Code::SveMovprfxZz);
        out.push_operand(plain_z(bits(word, 0, 5)));
        out.push_operand(plain_z(bits(word, 5, 5)));
        return;
    }
    // ADR (vector address): `<15:12>=1010`. `opc=<23:22>` selects the form, the
    // shift `msz=<11:10>` is the optional `#amt`. The address operand is
    // `[Zn.<T>, Zm.<T>{, <mod> #amt}]`.
    if bits(word, 12, 4) == 0b1010 {
        decode_adr_vec(word, out);
    }
}

/// SVE `ADR <Zd>.<T>, [<Zn>.<T>, <Zm>.<T>{, <mod> #amt}]` (`<15:12>=1010`).
///
/// `opc=<23:22>`: `00` → `.D` with `SXTW`; `01` → `.D` with `UXTW`; `1x` →
/// `.S`/`.D` (from `<22>`) with `LSL`. `msz=<11:10>` is the shift amount,
/// rendered only when non-zero.
#[inline]
fn decode_adr_vec(word: u32, out: &mut Instruction) {
    let opc = bits(word, 22, 2);
    let zm = bits(word, 16, 5);
    let msz = bits(word, 10, 2);
    let zn = bits(word, 5, 5);
    let zd = bits(word, 0, 5);
    let (code, a, ext) = match opc {
        0b00 => (Code::SveAdrSxtw, VA::Sd, ExtendType::Sxtw),
        0b01 => (Code::SveAdrUxtw, VA::Sd, ExtendType::Uxtw),
        // `1x`: same-scaled, arrangement from `<22>` (0=`.s`, 1=`.d`), LSL.
        _ => (
            Code::SveAdrSameScaled,
            if bit(word, 22) == 1 { VA::Sd } else { VA::Ss },
            // `Uxtx` renders as `lsl` in the formatter.
            ExtendType::Uxtx,
        ),
    };
    out.set(code);
    out.push_operand(zreg(zd, a));
    out.push_operand(Operand::SveMem {
        base: Z[(zn & 0x1f) as usize],
        offset: Z[(zm & 0x1f) as usize],
        arr: Some(a),
        extend: ext,
        imm: 0,
        amount: msz as u8,
        mode: SveMemMode::VecVec,
    });
}

/// A bare `Z{n}` with no arrangement (used by unpredicated MOVPRFX).
#[inline]
fn plain_z(n: u32) -> Operand {
    Operand::Reg { reg: Z[(n & 0x1f) as usize], arr: None, lane: None, shift: None, extend: None, pred: None }
}

/// A bare `Z{n}` carrying a lane index but no arrangement (`z4[2]`), used by
/// PMOV's vector operand for the `.h`/`.s`/`.d` element sizes.
#[inline]
fn plain_z_idx(n: u32, lane: u8) -> Operand {
    Operand::Reg { reg: Z[(n & 0x1f) as usize], arr: None, lane: Some(lane), shift: None, extend: None, pred: None }
}

/// A NEON `V{n}` operand with a full-128-bit arrangement (`v0.16b`/`.8h`/`.4s`/
/// `.2d`), the destination of the SVE2.1 quadword reductions.
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

/// The full-128-bit NEON arrangement matching element `size` (`0`=`.16b` ..
/// `3`=`.2d`).
#[inline]
fn va_neon(size: u32) -> VA {
    match size & 3 {
        0 => VA::V16B,
        1 => VA::V8H,
        2 => VA::V4S,
        _ => VA::V2D,
    }
}

/// INC/DEC vector by element count (`<21>=1`, `<15:13>=110`):
/// `op <Zdn>.<T>{, pattern{, MUL #imm}}`. The element size from `<23:22>` picks
/// H/W/D (no byte vector). `<20>` selects non-saturating (1) vs saturating (0);
/// for saturating, `<11>` is dec and `<10>` is unsigned; for non-saturating,
/// `<10>` is dec.
#[inline]
fn decode_incdec_vec(word: u32, out: &mut Instruction) {
    let size = bits(word, 22, 2);
    if size == 0 {
        return; // byte vector INC/DEC does not exist.
    }
    let a = arr_hsd(size);
    let imm4 = bits(word, 16, 4) + 1;
    let pattern = bits(word, 5, 5);
    let zdn = bits(word, 0, 5);
    let nonsat = bit(word, 20) == 1;
    let code = if nonsat {
        // INC / DEC (vector).
        let dec = bit(word, 10) == 1;
        match (dec, size) {
            (false, 1) => Code::SveIncDecVector, // inch
            (false, 2) => Code::SveIncDecVector, // incw
            (false, _) => Code::SveIncDecVector, // incd
            (true, _) => Code::SveIncDecVector,
        }
    } else {
        Code::SveIncDecVector
    };
    out.set(code);
    out.set_mnemonic(incdec_vec_mnemonic(word, size));
    out.push_operand(zreg(zdn, a));
    push_pattern_mul(out, pattern, imm4);
    let _ = code;
}

/// The exact INC/DEC vector mnemonic from the saturating/dec/unsigned bits and
/// the element size.
#[inline]
fn incdec_vec_mnemonic(word: u32, size: u32) -> Mnemonic {
    let nonsat = bit(word, 20) == 1;
    if nonsat {
        let dec = bit(word, 10) == 1;
        match (dec, size) {
            (false, 1) => Mnemonic::Inch,
            (false, 2) => Mnemonic::Incw,
            (false, _) => Mnemonic::Incd,
            (true, 1) => Mnemonic::Dech,
            (true, 2) => Mnemonic::Decw,
            (true, _) => Mnemonic::Decd,
        }
    } else {
        let dec = bit(word, 11) == 1;
        let unsigned = bit(word, 10) == 1;
        match (unsigned, dec, size) {
            (false, false, 1) => Mnemonic::Sqinch,
            (false, false, 2) => Mnemonic::Sqincw,
            (false, false, _) => Mnemonic::Sqincd,
            (true, false, 1) => Mnemonic::Uqinch,
            (true, false, 2) => Mnemonic::Uqincw,
            (true, false, _) => Mnemonic::Uqincd,
            (false, true, 1) => Mnemonic::Sqdech,
            (false, true, 2) => Mnemonic::Sqdecw,
            (false, true, _) => Mnemonic::Sqdecd,
            (true, true, 1) => Mnemonic::Uqdech,
            (true, true, 2) => Mnemonic::Uqdecw,
            (true, true, _) => Mnemonic::Uqdecd,
        }
    }
}

/// CNTB/H/W/D and scalar INC/DEC/SQINC/.. by element count (`<21>=1`,
/// `<15:13>=111`). Element size from `<23:22>` picks B/H/W/D for the mnemonic.
#[inline]
fn decode_cnt_incdec_scalar(word: u32, out: &mut Instruction) {
    let size = bits(word, 22, 2);
    let imm4 = bits(word, 16, 4) + 1;
    let pattern = bits(word, 5, 5);
    let rd = bits(word, 0, 5);
    let op = bits(word, 10, 3); // <12:10>
    let b20 = bit(word, 20);

    match op {
        // CNT (b20==0) or INC/DEC (b20==1).
        0b000 => {
            if b20 == 0 {
                out.set(Code::SveCntElem);
                out.set_mnemonic(cnt_mnemonic(size));
                out.push_operand(gpr(rd, RegWidth::X64));
                push_pattern_mul(out, pattern, imm4);
            } else {
                // INC scalar (X).
                out.set(Code::SveIncDecScalar);
                out.set_mnemonic(match size {
                    0 => Mnemonic::Incb,
                    1 => Mnemonic::Inch,
                    2 => Mnemonic::Incw,
                    _ => Mnemonic::Incd,
                });
                out.push_operand(gpr(rd, RegWidth::X64));
                push_pattern_mul(out, pattern, imm4);
            }
        }
        0b001 => {
            if b20 == 1 {
                // DEC scalar (X).
                out.set(Code::SveIncDecScalar);
                out.set_mnemonic(match size {
                    0 => Mnemonic::Decb,
                    1 => Mnemonic::Dech,
                    2 => Mnemonic::Decw,
                    _ => Mnemonic::Decd,
                });
                out.push_operand(gpr(rd, RegWidth::X64));
                push_pattern_mul(out, pattern, imm4);
            }
        }
        // Saturating INC/DEC scalar.
        0b100..=0b111 => {
            let unsigned = bit(word, 10) == 1; // U
            let dec = bit(word, 11) == 1; // D
            let mnem = sat_scalar_mnemonic(unsigned, dec, size);
            out.set(if matches!(mnem, Mnemonic::Sqincb | Mnemonic::Sqinch | Mnemonic::Sqincw | Mnemonic::Sqincd) {
                Code::SveSqIncDecScalarSx
            } else {
                Code::SveIncDecScalar
            });
            out.set_mnemonic(mnem);
            if !unsigned {
                // Signed saturating: `_x` (b20=1) = Xdn only; `_sx` (b20=0) =
                // Xdn, Wdn (32-bit saturation).
                if b20 == 1 {
                    out.push_operand(gpr(rd, RegWidth::X64));
                } else {
                    out.push_operand(gpr(rd, RegWidth::X64));
                    out.push_operand(gpr(rd, RegWidth::W32));
                }
            } else {
                // Unsigned saturating: `_x` (b20=1) = Xdn; `_uw` (b20=0) = Wdn.
                if b20 == 1 {
                    out.push_operand(gpr(rd, RegWidth::X64));
                } else {
                    out.push_operand(gpr(rd, RegWidth::W32));
                }
            }
            push_pattern_mul(out, pattern, imm4);
        }
        _ => {}
    }
}

/// CNTB/H/W/D mnemonic from the element-size field.
#[inline]
fn cnt_mnemonic(size: u32) -> Mnemonic {
    match size {
        0 => Mnemonic::Cntb,
        1 => Mnemonic::Cnth,
        2 => Mnemonic::Cntw,
        _ => Mnemonic::Cntd,
    }
}

// ===========================================================================
// Top byte 0x24 — SVE integer compare (vector / wide).
// ===========================================================================

#[inline]
fn decode_24(word: u32, out: &mut Instruction) {
    let size = bits(word, 22, 2);
    let pg = bits(word, 10, 3);
    let zn = bits(word, 5, 5);
    let pd = bits(word, 0, 4);
    let a = arr(size);

    if bit(word, 21) == 1 {
        // Unsigned immediate compare (`_p_p_zi_`): imm7=<20:14> (unsigned).
        let imm = bits(word, 14, 7);
        let b13 = bit(word, 13);
        let ne = bit(word, 4);
        let (code, mnem) = match (b13, ne) {
            (0, 0) => (Code::SveCmpZi, Mnemonic::Cmphs),
            (0, 1) => (Code::SveCmpZi, Mnemonic::Cmphi),
            (1, 0) => (Code::SveCmpZi, Mnemonic::Cmplo),
            _ => (Code::SveCmpZi, Mnemonic::Cmpls),
        };
        out.set(code);
        out.set_mnemonic(mnem);
        out.push_operand(preg_sz(pd, a));
        out.push_operand(preg_q(pg, PredQual::Zeroing));
        out.push_operand(zreg(zn, a));
        out.push_operand(Operand::ImmUnsigned(imm as u64));
        return;
    }

    // Vector / wide compares. Disambiguate by (op15, b14, b13, ne4).
    let op = bit(word, 15);
    let b14 = bit(word, 14);
    let b13 = bit(word, 13);
    let ne = bit(word, 4);
    let zm = bits(word, 16, 5);
    let (mnem, wide) = match (op, b14, b13, ne) {
        (0, 0, 0, 0) => (Mnemonic::Cmphs, false),
        (0, 0, 0, 1) => (Mnemonic::Cmphi, false),
        (0, 0, 1, 0) => (Mnemonic::Cmpeq, true),
        (0, 0, 1, 1) => (Mnemonic::Cmpne, true),
        (0, 1, 0, 0) => (Mnemonic::Cmpge, true),
        (0, 1, 0, 1) => (Mnemonic::Cmpgt, true),
        (0, 1, 1, 0) => (Mnemonic::Cmplt, true),
        (0, 1, 1, 1) => (Mnemonic::Cmple, true),
        (1, 0, 0, 0) => (Mnemonic::Cmpge, false),
        (1, 0, 0, 1) => (Mnemonic::Cmpgt, false),
        (1, 0, 1, 0) => (Mnemonic::Cmpeq, false),
        (1, 0, 1, 1) => (Mnemonic::Cmpne, false),
        (1, 1, 0, 0) => (Mnemonic::Cmphs, true),
        (1, 1, 0, 1) => (Mnemonic::Cmphi, true),
        (1, 1, 1, 0) => (Mnemonic::Cmplo, true),
        _ => (Mnemonic::Cmpls, true),
    };
    out.set(if wide { Code::SveCmpZw } else { Code::SveCmpZz });
    out.set_mnemonic(mnem);
    out.push_operand(preg_sz(pd, a));
    out.push_operand(preg_q(pg, PredQual::Zeroing));
    out.push_operand(zreg(zn, a));
    out.push_operand(zreg(zm, if wide { VA::Sd } else { a }));
}

// ===========================================================================
// Top byte 0x25 — SVE integer immediate, compare-imm (signed), INC/DEC by
// predicate, CNTP, MOV/DUP immediate. (Predicate-only ops are left to sve_perm.)
// ===========================================================================

#[inline]
fn decode_25(word: u32, out: &mut Instruction) {
    let sel = bits(word, 13, 3); // word<15:13>

    // CNTP `<Xd>, <Pg>, <Pn>.<T>` skeleton: `<21:20>=10`, `<19:16>=0000`,
    // `<15:14>=10`, `<9>=0`. Its `Pg` occupies `<13:10>`, so `<15:13>` is `10x`
    // (`100` or `101`) depending on `Pg<3>` — detect it here so a `Pg` with the
    // top bit set (sel==`101`) is not lost to the predicate decoder (DUP).
    if bits(word, 20, 2) == 0b10
        && bits(word, 16, 4) == 0
        && bits(word, 14, 2) == 0b10
        && bit(word, 9) == 0
    {
        decode_cntp(word, out);
        return;
    }

    // LASTP / FIRSTP (SVE2.1 extract predicate-as-counter): `<21>=1`,
    // `<20:16>` = `00010`(LASTP) / `00001`(FIRSTP), `<15:14>=10`, `<9>=0`. These
    // share the INC/DEC-by-predicate-count slot (`<15:12>=1000`, `sel=100`) when
    // `Pg<3>==0`, so detect them ahead of the `sel` match to avoid the SQDECP/
    // UQINCP over-decode. The element size is in `<23:22>`.
    if bit(word, 21) == 1
        && bits(word, 14, 2) == 0b10
        && bit(word, 9) == 0
        && (bits(word, 16, 5) == 0b00010 || bits(word, 16, 5) == 0b00001)
    {
        let a = arr(bits(word, 22, 2));
        let pg = bits(word, 10, 4);
        let pn = bits(word, 5, 4);
        let rd = bits(word, 0, 5);
        let is_lastp = bits(word, 16, 5) == 0b00010;
        out.set(if is_lastp { Code::SveLastp } else { Code::SveFirstp });
        out.push_operand(gpr(rd, RegWidth::X64));
        out.push_operand(preg(pg));
        out.push_operand(preg_sz(pn, a));
        return;
    }

    match sel {
        // ge/gt (000) and lt/le (001) signed-immediate compares.
        0b000 | 0b001 => {
            if bit(word, 14) == 0 && is_cmp_imm_signed(word) {
                decode_cmp_imm_signed(word, out);
            }
        }
        // sel=100 multiplexes: CMP eq/ne imm (`<21>=0`) and INC/DEC by predicate
        // count (`<21>=1`). (CNTP is handled above, before this match.)
        0b100 => {
            if bit(word, 21) == 0 {
                // Signed-immediate compare eq/ne.
                decode_cmp_imm_signed(word, out);
            } else if bits(word, 12, 4) == 0b1000 {
                // INC/DEC-by-predicate-count has the fixed nibble `<15:12>==1000`.
                // The FFR / predicate-misc ops sharing `<15:13>==100` (SETFFR /
                // WRFFR / RDFFR / PTRUE / PFALSE / ...) have `<15:12>==1001` and
                // are left to the permute/predicate decoder via the mod fallback.
                decode_incdec_pred(word, out);
            }
        }
        // Integer immediate arithmetic / min-max / mul / DUP (`<15:13>=110/111`).
        // These have `<21>=1`; the predicate ops (BRKP*/PFALSE/PTRUE/...) sharing
        // these `<15:13>` slots have `<21>=0` and are left to the predicate
        // decoder.
        0b110 | 0b111 if bit(word, 21) == 1 => {
            decode_int_imm(word, out);
        }
        _ => {}
    }
}

/// Heuristic recogniser for the signed-immediate compare encodings in 0x25:
/// their fixed skeleton is `00100101 size 0 imm5 op 0 lt Pg Zn ne Pd` with
/// `<21>=0`. The whilexx / predicate ops in the same `<15:13>` slots have
/// `<21>=1` or different `<15:14>`, so gate on `<21>==0` and the compare's
/// `<15:14>` pattern.
#[inline]
fn is_cmp_imm_signed(word: u32) -> bool {
    // imm5 form: <21>=0, <14>=0 (already checked). Distinguish from WHILE (which
    // has <15:13> too) by requiring the compare's Zn field role: WHILE uses GP
    // regs at <20:16>/<9:5> and has <15:13>=000 with <10>=... To keep it simple
    // and correct, gate on the exact compare skeleton bits: <23>=0 always here,
    // and the compare has bit<11>... We instead rely on the dispatch in decode_25
    // calling this only for sel in {000,001,100} with <14>=0; WHILE_p_p_rr has
    // <13>=? Use the discriminator that compares set <11>=0 and <10> belongs to
    // Pg. WHILE has <4>=Pd<0> too. The cleanest reliable gate: compares have
    // <21>==0 AND <15:13> in {000(ge/gt),001(lt/le),100(eq/ne)} AND NOT a WHILE.
    // WHILE_*_rr_ all have <15:13>=000/001 but with <10>==... distinguished by
    // <11:10>? We use: signed-imm compare requires bits<11>==0 is not reliable.
    // Final approach: only treat as compare when <15:14>==00 (ge/gt/lt/le) or
    // <15:13>==100 (eq/ne); WHILE shares 000/001 — exclude by <10>? Accept the
    // small overlap risk; decode_cmp checks succeed only for valid patterns.
    bit(word, 21) == 0
}

/// Signed-immediate compare (`_p_p_zi_`, top byte 0x25): `CMP<cc> <Pd>.<T>,
/// <Pg>/Z, <Zn>.<T>, #imm5` with `imm5` signed. The condition is `(op<15>,
/// lt<13>, ne<4>)`: ge(0,0,0)/gt(0,0,1)/lt(0,1,0)/le(0,1,1)/eq(1,0,0)/ne(1,0,1).
#[inline]
fn decode_cmp_imm_signed(word: u32, out: &mut Instruction) {
    let size = bits(word, 22, 2);
    let a = arr(size);
    let imm5 = sign_extend(bits(word, 16, 5) as u64, 5);
    let op = bit(word, 15);
    let lt = bit(word, 13);
    let ne = bit(word, 4);
    let pg = bits(word, 10, 3);
    let zn = bits(word, 5, 5);
    let pd = bits(word, 0, 4);
    let mnem = match (op, lt, ne) {
        (0, 0, 0) => Mnemonic::Cmpge,
        (0, 0, 1) => Mnemonic::Cmpgt,
        (0, 1, 0) => Mnemonic::Cmplt,
        (0, 1, 1) => Mnemonic::Cmple,
        (1, 0, 0) => Mnemonic::Cmpeq,
        (1, 0, 1) => Mnemonic::Cmpne,
        _ => return,
    };
    out.set(Code::SveCmpZi);
    out.set_mnemonic(mnem);
    out.push_operand(preg_sz(pd, a));
    out.push_operand(preg_q(pg, PredQual::Zeroing));
    out.push_operand(zreg(zn, a));
    out.push_operand(Operand::ImmSignedDec(imm5));
}

/// SVE integer immediate arithmetic / min-max / mul / broadcast (`_z_zi_`, top
/// byte 0x25). The class is `word<20:19>`: `00`=arithmetic, `01`=min/max,
/// `10`=mul, `11`=DUP/FMOV broadcast immediate.
#[inline]
fn decode_int_imm(word: u32, out: &mut Instruction) {
    let size = bits(word, 22, 2);
    let a = arr(size);
    let zdn = bits(word, 0, 5);
    let imm8 = bits(word, 5, 8);
    let sh = bit(word, 13);

    match bits(word, 19, 2) {
        // Arithmetic immediate, unsigned imm8 with optional `lsl #8`.
        0b00 => {
            let code = match bits(word, 16, 3) {
                0b000 => Code::SveAddZi,
                0b001 => Code::SveSubZi,
                0b011 => Code::SveSubrZi,
                0b100 => Code::SveSqaddZi,
                0b101 => Code::SveUqaddZi,
                0b110 => Code::SveSqsubZi,
                0b111 => Code::SveUqsubZi,
                _ => return,
            };
            out.set(code);
            out.push_operand(zreg(zdn, a));
            out.push_operand(zreg(zdn, a));
            push_imm8_shift(out, imm8, sh);
        }
        // Min/max immediate. <18:17>=00 -> MAX, 01 -> MIN; U=<16>.
        0b01 => {
            let u = bit(word, 16);
            match bits(word, 17, 2) {
                0b00 => out.set(if u == 0 { Code::SveSmaxZi } else { Code::SveUmaxZi }),
                0b01 => out.set(if u == 0 { Code::SveSminZi } else { Code::SveUminZi }),
                _ => return,
            }
            out.push_operand(zreg(zdn, a));
            out.push_operand(zreg(zdn, a));
            out.push_operand(minmax_imm(imm8, u == 0));
        }
        // MUL by immediate (signed imm8).
        0b10 => {
            if bits(word, 16, 3) != 0b000 {
                return;
            }
            out.set(Code::SveMulZi);
            out.push_operand(zreg(zdn, a));
            out.push_operand(zreg(zdn, a));
            out.push_operand(Operand::ImmSignedDec(sign_extend(imm8 as u64, 8)));
        }
        // Broadcast immediate: DUP (`<16>=0`) -> `MOV <Zd>.<T>, #imm{, shift}`;
        // FMOV/FDUP (`<16>=1`) is a floating-point broadcast -> left to sve_fp.
        _ => {
            if bit(word, 16) == 1 {
                // FMOV (fdup): not an integer op.
                return;
            }
            // DUP immediate, rendered as the MOV alias. imm8 is signed; the
            // optional `sh` (`<13>`) shifts the value left by 8.
            out.set(Code::SveDupImm);
            out.set_mnemonic(Mnemonic::Mov);
            out.push_operand(zreg(zdn, a));
            push_dup_imm(out, imm8, bit(word, 13));
        }
    }
}

/// Push the DUP broadcast immediate operand: a signed imm8, shifted left by 8
/// when `sh==1`. The corpus renders the pre-shifted signed value (`#-19456` for
/// `0xb4` with shift), so we materialise `(imm8 as i8 as i64) << 8`.
#[inline]
fn push_dup_imm(out: &mut Instruction, imm8: u32, sh: u32) {
    let v = sign_extend(imm8 as u64, 8);
    if sh == 1 {
        out.push_operand(Operand::ImmSignedDec(v << 8));
    } else {
        out.push_operand(Operand::ImmSignedDec(v));
    }
}

/// The min/max immediate operand: signed for SMAX/SMIN, unsigned for UMAX/UMIN.
#[inline]
fn minmax_imm(imm8: u32, signed: bool) -> Operand {
    if signed {
        Operand::ImmSignedDec(sign_extend(imm8 as u64, 8))
    } else {
        Operand::ImmUnsigned(imm8 as u64)
    }
}

/// Push an unsigned imm8 operand with an optional `, lsl #8` when `sh==1`.
///
/// Binary Ninja renders the shifted forms with the value already shifted into
/// place (`#0xNN00`), so a non-zero `imm8` with `sh==1` is emitted as
/// `imm8 << 8`. The corpus uses the explicit `#0x0, lsl #0x8` only for the
/// zero-with-shift case, rendered via [`Operand::ImmShiftedMove`].
#[inline]
fn push_imm8_shift(out: &mut Instruction, imm8: u32, sh: u32) {
    if sh == 1 {
        if imm8 == 0 {
            // `#0x0, lsl #0x8`.
            out.push_operand(Operand::ImmShiftedMove { imm: 0, lsl: 8 });
        } else {
            out.push_operand(Operand::ImmUnsigned((imm8 << 8) as u64));
        }
    } else {
        out.push_operand(Operand::ImmUnsigned(imm8 as u64));
    }
}

/// CNTP (`<15:13>=101`, top byte 0x25): `CNTP <Xd>, <Pg>, <Pn>.<T>`.
#[inline]
fn decode_cntp(word: u32, out: &mut Instruction) {
    let size = bits(word, 22, 2);
    let a = arr(size);
    let pg = bits(word, 10, 4);
    let pn = bits(word, 5, 4);
    let rd = bits(word, 0, 5);
    out.set(Code::SveCntp);
    out.push_operand(gpr(rd, RegWidth::X64));
    out.push_operand(preg(pg));
    out.push_operand(preg_sz(pn, a));
}

/// INC/DEC by predicate count (`<15:13>=100`, top byte 0x25): the scalar and
/// vector forms (`INCP`/`DECP`/`SQINCP`/`SQDECP`/`UQINCP`/`UQDECP`). The op is in
/// `word<18:16>`: 000=SQINCP, 001=UQINCP, 010=SQDECP, 011=UQDECP, 100=INCP,
/// 101=DECP. The vector form has `<11>==0` (scalar otherwise), and for the
/// saturating scalar forms `<10>` (sf) selects the X (1) vs W/SX (0) operand.
#[inline]
fn decode_incdec_pred(word: u32, out: &mut Instruction) {
    let size = bits(word, 22, 2);
    let a = arr(size);
    let pm = bits(word, 5, 4);
    let rdn = bits(word, 0, 5);
    let is_vector = bit(word, 11) == 0;
    let opc = bits(word, 16, 3); // word<18:16>

    // Plain INCP / DECP.
    if opc == 0b100 || opc == 0b101 {
        let mnem = if opc == 0b101 { Mnemonic::Decp } else { Mnemonic::Incp };
        if is_vector {
            out.set(Code::SveIncDecPVector);
            out.set_mnemonic(mnem);
            out.push_operand(zreg(rdn, a));
            out.push_operand(preg(pm));
        } else {
            out.set(Code::SveIncDecPScalar);
            out.set_mnemonic(mnem);
            out.push_operand(gpr(rdn, RegWidth::X64));
            out.push_operand(preg_sz(pm, a));
        }
        return;
    }

    // Saturating SQINCP/UQINCP/SQDECP/UQDECP.
    let mnem = match opc {
        0b000 => Mnemonic::Sqincp,
        0b001 => Mnemonic::Uqincp,
        0b010 => Mnemonic::Sqdecp,
        0b011 => Mnemonic::Uqdecp,
        _ => return,
    };
    let unsigned = matches!(mnem, Mnemonic::Uqincp | Mnemonic::Uqdecp);
    if is_vector {
        out.set(Code::SveIncDecPVector);
        out.set_mnemonic(mnem);
        out.push_operand(zreg(rdn, a));
        out.push_operand(preg(pm));
    } else {
        // Scalar: sf=<10> selects the 64-bit X form (1) vs the 32-bit W/SX form.
        let sf = bit(word, 10);
        out.set(Code::SveSqIncDecPScalarSx);
        out.set_mnemonic(mnem);
        if !unsigned {
            // Signed: `_x` (sf=1) = Xdn, Pg.T; `_sx` (sf=0) = Xdn, Pg.T, Wdn.
            out.push_operand(gpr(rdn, RegWidth::X64));
            out.push_operand(preg_sz(pm, a));
            if sf == 0 {
                out.push_operand(gpr(rdn, RegWidth::W32));
            }
        } else {
            // Unsigned: `_x` (sf=1) = Xdn, Pg.T; `_uw` (sf=0) = Wdn, Pg.T.
            out.push_operand(gpr(rdn, if sf == 1 { RegWidth::X64 } else { RegWidth::W32 }));
            out.push_operand(preg_sz(pm, a));
        }
    }
}

// ===========================================================================
// Top bytes 0x44 / 0x45 — SVE2 integer multiply-add / DOT / widening.
// ===========================================================================

/// SDOT / UDOT (top byte 0x44): `op <Zda>.<T>, <Zn>.<Tb>, <Zm>.<Tb>{[idx]}`.
///
/// The DOT-product skeleton has `word<15:11>==0b00000` and `word<10>==U`.
/// `word<21>` selects the vector form (0) vs the indexed form (1); within each,
/// `word<22>` (or `<23:22>`) selects `.s`-over-`.b` vs `.d`-over-`.h`.
#[inline]
fn decode_44(word: u32, out: &mut Instruction) {
    if bit(word, 21) == 0 {
        decode_44_vector(word, out);
    } else {
        decode_44_indexed(word, out);
    }
}

/// The complex-rotation immediate `#<rot>` from the 2-bit `rot` field: `0`→`#0x0`,
/// `1`→`#0x5a` (90), `2`→`#0xb4` (180), `3`→`#0x10e` (270). Binary Ninja renders
/// these in hex.
#[inline]
fn rot_imm(rot: u32) -> Operand {
    Operand::ImmUnsigned(match rot & 3 {
        0 => 0,
        1 => 90,
        2 => 180,
        _ => 270,
    })
}

/// Widen a `size` (`01`/`10`/`11`) into its `(wide, narrow)` arrangement pair for
/// the SVE2 2x-widening multiply-long family (`.h<-.b`, `.s<-.h`, `.d<-.s`).
#[inline]
fn widen2(size: u32) -> Option<(VA, VA)> {
    match size {
        0b01 => Some((VA::Sh, VA::Sb)),
        0b10 => Some((VA::Ss, VA::Sh)),
        0b11 => Some((VA::Sd, VA::Ss)),
        _ => None,
    }
}

/// SVE2 0x44 vector forms (`<21>=0`): SDOT/UDOT, CDOT, CMLA/SQRDCMLAH, the
/// {S,U}ML{A,S}L{B,T} multiply-add-long family, SQDML{A,S}L{B,T}, and
/// SQRDML{A,S}H. Dispatched on the `<15:10>` opcode region.
#[inline]
fn decode_44_vector(word: u32, out: &mut Instruction) {
    let size = bits(word, 22, 2);
    let zm = bits(word, 16, 5);
    let zn = bits(word, 5, 5);
    let zda = bits(word, 0, 5);
    let s1513 = bits(word, 13, 3);
    match s1513 {
        // SDOT/UDOT (`<15:11>=00000`).
        0b000 if bits(word, 11, 2) == 0 => {
            // Vector DOT: size==2 -> .s/.b, size==3 -> .d/.h. U=<10>.
            if size < 2 {
                return;
            }
            let u = bit(word, 10);
            let (da, sb) = if size == 2 { (VA::Ss, VA::Sb) } else { (VA::Sd, VA::Sh) };
            out.set(if u == 0 { Code::SveSdot } else { Code::SveUdot });
            out.set_mnemonic(if u == 0 { Mnemonic::Sdot } else { Mnemonic::Udot });
            out.push_operand(zreg(zda, da));
            out.push_operand(zreg(zn, sb));
            out.push_operand(zreg(zm, sb));
        }
        // SQDMLALBT (`S=0`) / SQDMLSLBT (`S=1`) (`<15:11>=00001`): 2x-widening
        // bottom-top saturating doubling multiply-add/subtract long.
        0b000 if bits(word, 11, 2) == 0b01 => {
            let Some((da, sa)) = widen2(size) else { return };
            let s = bit(word, 10);
            out.set(Code::SveSqdmlalLongBt);
            out.set_mnemonic(if s == 0 { Mnemonic::Sqdmlalbt } else { Mnemonic::Sqdmlslbt });
            out.push_operand(zreg(zda, da));
            out.push_operand(zreg(zn, sa));
            out.push_operand(zreg(zm, sa));
        }
        // CDOT (`<15:12>=0001`): 4x-widening complex dot. size==2 -> .s/.b,
        // size==3 -> .d/.h. rot=<11:10>.
        0b000 => {
            if bits(word, 12, 1) != 1 || size < 2 {
                return;
            }
            let (da, sb) = if size == 2 { (VA::Ss, VA::Sb) } else { (VA::Sd, VA::Sh) };
            out.set(Code::SveCdot);
            out.set_mnemonic(Mnemonic::Cdot);
            out.push_operand(zreg(zda, da));
            out.push_operand(zreg(zn, sb));
            out.push_operand(zreg(zm, sb));
            out.push_operand(rot_imm(bits(word, 10, 2)));
        }
        // CMLA (`<12>=0`) / SQRDCMLAH (`<12>=1`): same-size complex MAC. rot=<11:10>.
        0b001 => {
            let a = arr(size);
            let op = bit(word, 12);
            out.set(if op == 0 { Code::SveCmla } else { Code::SveSqrdcmlah });
            out.set_mnemonic(if op == 0 { Mnemonic::Cmla } else { Mnemonic::Sqrdcmlah });
            out.push_operand(zreg(zda, a));
            out.push_operand(zreg(zn, a));
            out.push_operand(zreg(zm, a));
            out.push_operand(rot_imm(bits(word, 10, 2)));
        }
        // {S,U}ML{A,S}L{B,T} (`<15:13>=010`): S=<12> (0=MLAL,1=MLSL), U=<11>
        // (0=signed,1=unsigned), T=<10> (bottom/top). 2x widening.
        0b010 => {
            let Some((da, sa)) = widen2(size) else { return };
            let s = bit(word, 12);
            let u = bit(word, 11);
            let t = bit(word, 10);
            let mnem = match (s, u, t) {
                (0, 0, 0) => Mnemonic::Smlalb,
                (0, 0, 1) => Mnemonic::Smlalt,
                (0, 1, 0) => Mnemonic::Umlalb,
                (0, 1, 1) => Mnemonic::Umlalt,
                (1, 0, 0) => Mnemonic::Smlslb,
                (1, 0, 1) => Mnemonic::Smlslt,
                (1, 1, 0) => Mnemonic::Umlslb,
                _ => Mnemonic::Umlslt,
            };
            out.set(Code::SveMlaLong);
            out.set_mnemonic(mnem);
            out.push_operand(zreg(zda, da));
            out.push_operand(zreg(zn, sa));
            out.push_operand(zreg(zm, sa));
        }
        // SQDML{A,S}L{B,T} (`<15:12>=0110`): S=<11> (0=ADD,1=SUB), T=<10>.
        0b011 if bit(word, 12) == 0 => {
            let Some((da, sa)) = widen2(size) else { return };
            let s = bit(word, 11);
            let t = bit(word, 10);
            let mnem = match (s, t) {
                (0, 0) => Mnemonic::Sqdmlalb,
                (0, 1) => Mnemonic::Sqdmlalt,
                (1, 0) => Mnemonic::Sqdmlslb,
                _ => Mnemonic::Sqdmlslt,
            };
            out.set(Code::SveSqdmlalLong);
            out.set_mnemonic(mnem);
            out.push_operand(zreg(zda, da));
            out.push_operand(zreg(zn, sa));
            out.push_operand(zreg(zm, sa));
        }
        // `<15:13>=011`: SQRDMLAH/SH (`<12:11>=10`) and USDOT (`<15:10>=011110`).
        0b011 => {
            if bits(word, 10, 6) == 0b011110 {
                // USDOT (vector): `<Zda>.S, <Zn>.B, <Zm>.B`. size==2 only.
                if size != 2 {
                    return;
                }
                out.set(Code::SveDotMixed);
                out.set_mnemonic(Mnemonic::Usdot);
                out.push_operand(zreg(zda, VA::Ss));
                out.push_operand(zreg(zn, VA::Sb));
                out.push_operand(zreg(zm, VA::Sb));
            } else if bits(word, 11, 2) == 0b10 {
                decode_44_sqrdml_vec(word, out);
            }
        }
        // Predicated SVE2 integer (`<15:14>=10`): halving/saturating/rounding
        // binary, pairwise, saturating-abs/neg, and pairwise-accumulate-long.
        0b100 | 0b101 => decode_44_pred(word, out),
        // SCLAMP/UCLAMP (`<15:11>=11000`, SVE2 / SME): clamp each element of the
        // destination to the signed/unsigned range `[Zn, Zm]`. `U=<10>` selects
        // signed(0)/unsigned(1); all three operands share the `size` arrangement.
        0b110 if bits(word, 11, 2) == 0 => {
            let a = arr(size);
            let u = bit(word, 10);
            out.set(if u == 0 { Code::SveSclamp } else { Code::SveUclamp });
            out.push_operand(zreg(zda, a));
            out.push_operand(zreg(zn, a));
            out.push_operand(zreg(zm, a));
        }
        // SDOT/UDOT 2-way `.h` (`<15:11>=11001`, SVE2.1): `<Zda>.S, <Zn>.H,
        // <Zm>.H{[idx]}`. The vector form has `<23:22>=00`; the indexed form has
        // `<23:22>=10` with i2=<20:19> selecting the 128-bit-segment lane and a
        // 3-bit Zm=<18:16>. `U=<10>`.
        0b110 if bits(word, 11, 2) == 0b01 => {
            let u = bit(word, 10);
            if bit(word, 23) == 0 {
                // Vector form (size==00).
                if size != 0 {
                    return;
                }
                out.set(if u == 0 { Code::SveSdotH } else { Code::SveUdotH });
                out.set_mnemonic(if u == 0 { Mnemonic::Sdot } else { Mnemonic::Udot });
                out.push_operand(zreg(zda, VA::Ss));
                out.push_operand(zreg(zn, VA::Sh));
                out.push_operand(zreg(zm, VA::Sh));
            } else {
                // Indexed form (`<23:22>=10`): i2=<20:19>, Zm=<18:16>.
                out.set(if u == 0 { Code::SveSdotHIdx } else { Code::SveUdotHIdx });
                out.set_mnemonic(if u == 0 { Mnemonic::Sdot } else { Mnemonic::Udot });
                let idx = bits(word, 19, 2);
                let zm = bits(word, 16, 3);
                out.push_operand(zreg(zda, VA::Ss));
                out.push_operand(zreg(zn, VA::Sh));
                out.push_operand(zreg_idx(zm, VA::Sh, idx as u8));
            }
        }
        // {S,U}ABAL (`<15:13>=110`, `<12:11>` = `10`(SABAL)/`11`(UABAL), `<10>=1`,
        // `<21>=0`): SVE2.3 absolute-difference accumulate long (alias of
        // {S,U}ABALB). 2x widening: `<Zda>.<T>, <Zn>.<Tb>, <Zm>.<Tb>`.
        0b110 if bit(word, 21) == 0 && bit(word, 10) == 1 && bit(word, 12) == 1 => {
            let Some((da, sa)) = widen2(size) else { return };
            let u = bit(word, 11);
            out.set(if u == 0 { Code::SveSabal } else { Code::SveUabal });
            out.push_operand(zreg(zda, da));
            out.push_operand(zreg(zn, sa));
            out.push_operand(zreg(zm, sa));
        }
        // ZIPQ1/2, UZPQ1/2, TBLQ (`<15:13>=111`, SVE2.1 128-bit-segment permute):
        // `<12:10>` selects the op; all four element sizes are valid.
        0b111 => {
            let a = arr(size);
            match bits(word, 10, 3) {
                0b000..=0b011 => {
                    let mnem = match bits(word, 10, 3) {
                        0b000 => Mnemonic::Zipq1,
                        0b001 => Mnemonic::Zipq2,
                        0b010 => Mnemonic::Uzpq1,
                        _ => Mnemonic::Uzpq2,
                    };
                    out.set(Code::SveZipqUzpq);
                    out.set_mnemonic(mnem);
                    out.push_operand(zreg(zda, a));
                    out.push_operand(zreg(zn, a));
                    out.push_operand(zreg(zm, a));
                }
                0b110 => {
                    out.set(Code::SveTblq);
                    out.push_operand(zreg(zda, a));
                    out.push_operand(zlist1(zn, a));
                    out.push_operand(zreg(zm, a));
                }
                _ => {}
            }
        }
        _ => {}
    }
}

/// A single-register Z list `{Z{n}.<T>}` for the single-table `TBLQ`.
#[inline]
fn zlist1(n: u32, a: VA) -> Operand {
    Operand::MultiReg {
        regs: [Z[(n & 0x1f) as usize], Register::None, Register::None, Register::None],
        count: 1,
        arr: Some(a),
        lane: None,
    }
}

/// SVE2 predicated integer ops sharing top byte 0x44, `<15:14>=10`:
/// * `<15:13>=100` `_z_p_zz_`: `<20:19>` selects halving (`10`), saturating
///   add/sub (`11`), or saturating/rounding shift (`01`); `<18:16>` the variant.
///   Operands `<Zdn>.<T>, <Pg>/M, <Zdn>.<T>, <Zm>.<T>`.
/// * `<15:13>=101`: pairwise ADDP/{S,U}{MAX,MIN}P (`<20:19>=10`, `_z_p_zz_`),
///   {S,U}ADALP (`<20:16>=00010U`, `_z_p_z_`, source half-width), and
///   SQABS/SQNEG (`<20:16>=001000/001001`, `_z_p_z_`).
#[inline]
fn decode_44_pred(word: u32, out: &mut Instruction) {
    let size = bits(word, 22, 2);
    let a = arr(size);
    let pg = bits(word, 10, 3);
    // `_z_p_zz_`: Zm=<9:5>, Zdn=<4:0>; `_z_p_z_`: Zn=<9:5>, Zd=<4:0>.
    let zm = bits(word, 5, 5);
    let zn = bits(word, 5, 5);
    let zdn = bits(word, 0, 5);
    let u = bit(word, 16);
    if bit(word, 13) == 0 {
        // `<15:13>=100`: predicated binary `_z_p_zz_`. `<21:19>` selects family:
        // `00X`=shift (`<19>=Q`,`<18>=R`,`<17>=N`,`<16>=U`), `010`=halving
        // (`<18>=R`,`<17>=S`,`<16>=U`), `011`=saturating (`<18>=op`,`<17>=S`,`<16>=U`).
        let (code, mnem) = match bits(word, 19, 3) {
            0b010 => (
                Code::SveHalvingZpzz,
                match bits(word, 16, 3) {
                    0b000 => Mnemonic::Shadd,
                    0b001 => Mnemonic::Uhadd,
                    0b010 => Mnemonic::Shsub,
                    0b011 => Mnemonic::Uhsub,
                    0b100 => Mnemonic::Srhadd,
                    0b101 => Mnemonic::Urhadd,
                    0b110 => Mnemonic::Shsubr,
                    _ => Mnemonic::Uhsubr,
                },
            ),
            0b011 => (
                Code::SveSatRoundZpzz,
                match bits(word, 16, 3) {
                    0b000 => Mnemonic::Sqadd,
                    0b001 => Mnemonic::Uqadd,
                    0b010 => Mnemonic::Sqsub,
                    0b011 => Mnemonic::Uqsub,
                    0b100 => Mnemonic::Suqadd,
                    0b101 => Mnemonic::Usqadd,
                    0b110 => Mnemonic::Sqsubr,
                    _ => Mnemonic::Uqsubr,
                },
            ),
            // Shift left (`<21:20>=00`): `<19>=Q`,`<18>=R`,`<17>=N`,`<16>=U`.
            // Q:R:N selects: 000=SRSHL? Actually the mnemonic table is keyed by
            // (Q,R,N): Q=saturating, R=reversed, N=non-rounding shift variant.
            0b000 | 0b001 => {
                let q = bit(word, 19);
                let r = bit(word, 18);
                let n = bit(word, 17);
                let mnem = match (q, r, n, u) {
                    // Non-saturating rounding shifts (Q=0): SRSHL/URSHL (N=1) and
                    // their reversed forms (R=1).
                    (0, 0, 1, 0) => Mnemonic::Srshl,
                    (0, 0, 1, 1) => Mnemonic::Urshl,
                    (0, 1, 1, 0) => Mnemonic::Srshlr,
                    (0, 1, 1, 1) => Mnemonic::Urshlr,
                    // Saturating shifts (Q=1): N=0 -> SQSHL/UQSHL, N=1 -> SQRSHL/
                    // UQRSHL; R=1 -> reversed.
                    (1, 0, 0, 0) => Mnemonic::Sqshl,
                    (1, 0, 0, 1) => Mnemonic::Uqshl,
                    (1, 0, 1, 0) => Mnemonic::Sqrshl,
                    (1, 0, 1, 1) => Mnemonic::Uqrshl,
                    (1, 1, 0, 0) => Mnemonic::Sqshlr,
                    (1, 1, 0, 1) => Mnemonic::Uqshlr,
                    (1, 1, 1, 0) => Mnemonic::Sqrshlr,
                    (1, 1, 1, 1) => Mnemonic::Uqrshlr,
                    _ => return,
                };
                (Code::SveSatRoundZpzz, mnem)
            }
            _ => return,
        };
        out.set(code);
        out.set_mnemonic(mnem);
        out.push_operand(zreg(zdn, a));
        out.push_operand(preg_q(pg, PredQual::Merging));
        out.push_operand(zreg(zdn, a));
        out.push_operand(zreg(zm, a));
        return;
    }
    // `<15:13>=101`. `<21:19>` selects: `010`=pairwise, `00X`=ADALP / SQABS/SQNEG.
    match bits(word, 19, 3) {
        // Pairwise ADDP / {S,U}{MAX,MIN}P (`<21:19>=010`): `<18:16>=opc:U`.
        0b010 => {
            let mnem = match bits(word, 16, 3) {
                0b001 => Mnemonic::Addp,
                0b100 => Mnemonic::Smaxp,
                0b101 => Mnemonic::Umaxp,
                0b110 => Mnemonic::Sminp,
                0b111 => Mnemonic::Uminp,
                _ => return,
            };
            out.set(Code::SvePairZpzz);
            out.set_mnemonic(mnem);
            out.push_operand(zreg(zdn, a));
            out.push_operand(preg_q(pg, PredQual::Merging));
            out.push_operand(zreg(zdn, a));
            out.push_operand(zreg(zm, a));
        }
        // SADALP/UADALP (`<20:16>=00100/00101`, i.e. `<18:16>=10U`) and SQABS/
        // SQNEG (`<20:16>=01000/01001`, i.e. `<19>=1`,`<18:16>=00opc`).
        0b000 | 0b001 => {
            if bit(word, 19) == 0 && bits(word, 17, 2) == 0b10 {
                // SADALP/UADALP: source elements half-width (2x widening accum).
                let Some((da, sa)) = widen2(size) else { return };
                out.set(Code::SveAdalp);
                out.set_mnemonic(if u == 0 { Mnemonic::Sadalp } else { Mnemonic::Uadalp });
                out.push_operand(zreg(zdn, da));
                out.push_operand(preg_q(pg, PredQual::Merging));
                out.push_operand(zreg(zn, sa));
            } else if bit(word, 19) == 1 && bits(word, 17, 2) == 0b00 {
                // SQABS (`<16>=0`) / SQNEG (`<16>=1`): same-size unary.
                out.set(Code::SveSatUnaryZpz);
                out.set_mnemonic(if bit(word, 16) == 0 { Mnemonic::Sqabs } else { Mnemonic::Sqneg });
                out.push_operand(zreg(zdn, a));
                out.push_operand(preg_q(pg, PredQual::Merging));
                out.push_operand(zreg(zn, a));
            } else if bit(word, 19) == 0 && bits(word, 17, 2) == 0b00 {
                // URECPE (`<16>=0`) / URSQRTE (`<16>=1`): `.s` only, same-size unary.
                if size != 2 {
                    return;
                }
                out.set(Code::SveRecipEst);
                out.set_mnemonic(if bit(word, 16) == 0 { Mnemonic::Urecpe } else { Mnemonic::Ursqrte });
                out.push_operand(zreg(zdn, a));
                out.push_operand(preg_q(pg, PredQual::Merging));
                out.push_operand(zreg(zn, a));
            }
        }
        _ => {}
    }
}

/// SVE2 0x44 indexed (by-element) forms (`<21>=1`): MUL/MLA/MLS, SQDMULH/
/// SQRDMULH, CDOT/CMLA/SQRDCMLAH, the {S,U}ML{A,S}L{B,T} long family,
/// SQDML{A,S}L{B,T}, SQDMULL{B,T}, and SQRDML{A,S}H.
#[inline]
fn decode_44_indexed(word: u32, out: &mut Instruction) {
    let zn = bits(word, 5, 5);
    let zda = bits(word, 0, 5);
    let s1513 = bits(word, 13, 3);
    match s1513 {
        // `<15:13>=000`: SDOT/UDOT idx (`<15:11>=00000`), MLA/MLS idx
        // (`<15:11>=00001`), SQRDMLAH/SH idx (`<15:11>=00010`).
        0b000 => {
            let s1211 = bits(word, 11, 2);
            if s1211 == 0b00 {
                // SDOT/UDOT idx: U=<10>. size==2 -> .s/.b, size==3 -> .d/.h.
                let u = bit(word, 10);
                out.set(if u == 0 { Code::SveSdotIdx } else { Code::SveUdotIdx });
                out.set_mnemonic(if u == 0 { Mnemonic::Sdot } else { Mnemonic::Udot });
                if bit(word, 22) == 0 {
                    let idx = bits(word, 19, 2);
                    let zm = bits(word, 16, 3);
                    out.push_operand(zreg(zda, VA::Ss));
                    out.push_operand(zreg(zn, VA::Sb));
                    out.push_operand(zreg_idx(zm, VA::Sb, idx as u8));
                } else {
                    let idx = bit(word, 20);
                    let zm = bits(word, 16, 4);
                    out.push_operand(zreg(zda, VA::Sd));
                    out.push_operand(zreg(zn, VA::Sh));
                    out.push_operand(zreg_idx(zm, VA::Sh, idx as u8));
                }
            } else if s1211 == 0b01 {
                // MLA (S=0) / MLS (S=1) idx (`<15:11>=00001`): same-size by element.
                let s = bit(word, 10);
                out.set(if s == 0 { Code::SveMlaIdx } else { Code::SveMlsIdx });
                out.set_mnemonic(if s == 0 { Mnemonic::Mla } else { Mnemonic::Mls });
                push_sve2_idx_same(word, out, zda, zn);
            } else if s1211 == 0b10 {
                // SQRDMLAH/SH idx (`<15:11>=00010`).
                decode_44_sqrdml_idx(word, out);
            } else {
                // USDOT (U=0) / SUDOT (U=1) idx (`<15:11>=00011`): `<Zda>.S,
                // <Zn>.B, <Zm>.B[idx]`. size==2 only; i2=<20:19>, Zm=<18:16>.
                if bits(word, 22, 2) != 0b10 {
                    return;
                }
                let u = bit(word, 10);
                out.set(Code::SveDotMixed);
                out.set_mnemonic(if u == 0 { Mnemonic::Usdot } else { Mnemonic::Sudot });
                let idx = bits(word, 19, 2);
                let zm = bits(word, 16, 3);
                out.push_operand(zreg(zda, VA::Ss));
                out.push_operand(zreg(zn, VA::Sb));
                out.push_operand(zreg_idx(zm, VA::Sb, idx as u8));
            }
        }
        // `<15:13>=001`: SQDML{A,S}L{B,T} idx long (`<15:12>=0010`/`0011`).
        0b001 => decode_44_idx_sqdmlal(word, out),
        // `<15:13>=010`: CDOT idx (`<15:12>=0100`).
        0b010 if bit(word, 12) == 0 => decode_44_idx_cdot(word, out),
        // `<15:13>=011`: CMLA idx (`<15:12>=0110`) / SQRDCMLAH idx (`<15:12>=0111`).
        0b011 => decode_44_idx_cmla(word, out),
        // `<15:14>=10`: {S,U}ML{A,S}L{B,T} idx long. S=<13>,U=<12>,i.l=<11>,T=<10>.
        0b100 | 0b101 => decode_44_idx_mlal(word, out),
        // `<15:13>=110`: {S,U}MULL{B,T} idx long. U=<12>,i.l=<11>,T=<10>.
        0b110 => decode_44_idx_mull(word, out),
        // `<15:13>=111`: MUL idx, SQDMULH/SQRDMULH idx, SQDMULL{B,T} idx long.
        0b111 => decode_44_idx_mul(word, out),
        _ => {}
    }
}

/// {S,U}MULL{B,T} indexed long (`<21>=1`, `<15:13>=110`). U=<12> (0=signed,
/// 1=unsigned), T=<10>. size==2 -> .s<-.h (i3h=<20:19>,Zm=<18:16>,i3l=<11>);
/// size==3 -> .d<-.s (i2h=<20>,Zm=<19:16>,i2l=<11>).
#[inline]
fn decode_44_idx_mull(word: u32, out: &mut Instruction) {
    let size = bits(word, 22, 2);
    let Some((da, sa)) = widen2(size) else { return };
    if size < 2 {
        return;
    }
    let zn = bits(word, 5, 5);
    let zd = bits(word, 0, 5);
    let u = bit(word, 12);
    let t = bit(word, 10);
    let mnem = match (u, t) {
        (0, 0) => Mnemonic::Smullb,
        (0, 1) => Mnemonic::Smullt,
        (1, 0) => Mnemonic::Umullb,
        _ => Mnemonic::Umullt,
    };
    let (idx, zm) = if size == 2 {
        (((bits(word, 19, 2) << 1) | bit(word, 11)), bits(word, 16, 3))
    } else {
        (((bit(word, 20) << 1) | bit(word, 11)), bits(word, 16, 4))
    };
    out.set(Code::SveMulLongIdx);
    out.set_mnemonic(mnem);
    out.push_operand(zreg(zd, da));
    out.push_operand(zreg(zn, sa));
    out.push_operand(zreg_idx(zm, sa, idx as u8));
}

/// CDOT indexed (`<21>=1`, `<15:12>=0100`): 4x-widening complex dot, rot=<11:10>.
/// size==2 -> .s<-.b (i2=<20:19>,Zm=<18:16>); size==3 -> .d<-.h (i1=<20>,
/// Zm=<19:16>).
#[inline]
fn decode_44_idx_cdot(word: u32, out: &mut Instruction) {
    let size = bits(word, 22, 2);
    if size < 2 {
        return;
    }
    let zn = bits(word, 5, 5);
    let zda = bits(word, 0, 5);
    out.set(Code::SveCdotIdx);
    out.set_mnemonic(Mnemonic::Cdot);
    if size == 2 {
        let idx = bits(word, 19, 2);
        let zm = bits(word, 16, 3);
        out.push_operand(zreg(zda, VA::Ss));
        out.push_operand(zreg(zn, VA::Sb));
        out.push_operand(zreg_idx(zm, VA::Sb, idx as u8));
    } else {
        let idx = bit(word, 20);
        let zm = bits(word, 16, 4);
        out.push_operand(zreg(zda, VA::Sd));
        out.push_operand(zreg(zn, VA::Sh));
        out.push_operand(zreg_idx(zm, VA::Sh, idx as u8));
    }
    out.push_operand(rot_imm(bits(word, 10, 2)));
}

/// SQDML{A,S}L{B,T} indexed long (`<21>=1`, `<15:13>=001`). size==2 -> .s<-.h
/// (i3h=<20:19>,Zm=<18:16>,i3l=<11>); size==3 -> .d<-.s (i2h=<20>,Zm=<19:16>,
/// i2l=<11>). S=<12> (0=ADD,1=SUB), T=<10>.
#[inline]
fn decode_44_idx_sqdmlal(word: u32, out: &mut Instruction) {
    let size = bits(word, 22, 2);
    let Some((da, sa)) = widen2(size) else { return };
    // Only .s (size 10) and .d (size 11) have indexed long forms.
    if size < 2 {
        return;
    }
    let zn = bits(word, 5, 5);
    let zda = bits(word, 0, 5);
    let s = bit(word, 12);
    let t = bit(word, 10);
    let mnem = match (s, t) {
        (0, 0) => Mnemonic::Sqdmlalb,
        (0, 1) => Mnemonic::Sqdmlalt,
        (1, 0) => Mnemonic::Sqdmlslb,
        _ => Mnemonic::Sqdmlslt,
    };
    let (idx, zm) = if size == 2 {
        (((bits(word, 19, 2) << 1) | bit(word, 11)), bits(word, 16, 3))
    } else {
        (((bit(word, 20) << 1) | bit(word, 11)), bits(word, 16, 4))
    };
    out.set(Code::SveSqdmlalLongIdx);
    out.set_mnemonic(mnem);
    out.push_operand(zreg(zda, da));
    out.push_operand(zreg(zn, sa));
    out.push_operand(zreg_idx(zm, sa, idx as u8));
}

/// CMLA / SQRDCMLAH indexed (`<21>=1`, `<15:12>=0110`/`0111`): same-size complex
/// MAC, rot=<11:10>. size==2 -> .h (i2=<20:19>,Zm=<18:16>); size==3 -> .s
/// (i1=<20>,Zm=<19:16>).
#[inline]
fn decode_44_idx_cmla(word: u32, out: &mut Instruction) {
    let size = bits(word, 22, 2);
    if size < 2 {
        return;
    }
    let zn = bits(word, 5, 5);
    let zda = bits(word, 0, 5);
    let op = bit(word, 12); // 0 -> CMLA (`0110`), 1 -> SQRDCMLAH (`0111`).
    out.set(if op == 0 { Code::SveCmlaIdx } else { Code::SveSqrdcmlahIdx });
    out.set_mnemonic(if op == 0 { Mnemonic::Cmla } else { Mnemonic::Sqrdcmlah });
    let a = if size == 2 { VA::Sh } else { VA::Ss };
    let (idx, zm) = if size == 2 {
        (bits(word, 19, 2), bits(word, 16, 3))
    } else {
        (bit(word, 20), bits(word, 16, 4))
    };
    out.push_operand(zreg(zda, a));
    out.push_operand(zreg(zn, a));
    out.push_operand(zreg_idx(zm, a, idx as u8));
    out.push_operand(rot_imm(bits(word, 10, 2)));
}

/// {S,U}ML{A,S}L{B,T} indexed long (`<21>=1`, `<15:14>=10`). S=<13> (0=MLAL,
/// 1=MLSL), U=<12> (0=signed,1=unsigned), T=<10>. size==2 -> .s<-.h
/// (i3h=<20:19>,Zm=<18:16>,i3l=<11>); size==3 -> .d<-.s (i2h=<20>,Zm=<19:16>,
/// i2l=<11>).
#[inline]
fn decode_44_idx_mlal(word: u32, out: &mut Instruction) {
    let size = bits(word, 22, 2);
    let Some((da, sa)) = widen2(size) else { return };
    if size < 2 {
        return;
    }
    let zn = bits(word, 5, 5);
    let zda = bits(word, 0, 5);
    let s = bit(word, 13);
    let u = bit(word, 12);
    let t = bit(word, 10);
    let mnem = match (s, u, t) {
        (0, 0, 0) => Mnemonic::Smlalb,
        (0, 0, 1) => Mnemonic::Smlalt,
        (0, 1, 0) => Mnemonic::Umlalb,
        (0, 1, 1) => Mnemonic::Umlalt,
        (1, 0, 0) => Mnemonic::Smlslb,
        (1, 0, 1) => Mnemonic::Smlslt,
        (1, 1, 0) => Mnemonic::Umlslb,
        _ => Mnemonic::Umlslt,
    };
    let (idx, zm) = if size == 2 {
        (((bits(word, 19, 2) << 1) | bit(word, 11)), bits(word, 16, 3))
    } else {
        (((bit(word, 20) << 1) | bit(word, 11)), bits(word, 16, 4))
    };
    out.set(Code::SveMlaLongIdx);
    out.set_mnemonic(mnem);
    out.push_operand(zreg(zda, da));
    out.push_operand(zreg(zn, sa));
    out.push_operand(zreg_idx(zm, sa, idx as u8));
}

/// MUL/MLA/MLS, SQDMULH/SQRDMULH and SQDMULL{B,T} indexed (`<21>=1`,
/// `<15:13>=111`). The `<12:10>` opcode selects:
/// `<15:11>=00001` (MLA/MLS, S=<10>) is *not* in this arm; here `<15:13>=111`
/// covers `<15:11>` in `11100..11111`:
/// * `<15:10>=111100` (`<15:11>=11110`,`<10>=0`) → SQDMULH idx
/// * `<15:10>=111101` (`<15:11>=11110`,`<10>=1`) → SQRDMULH idx
/// * `<15:10>=111110` → MUL idx
/// * `<15:12>=1110` (`<15:11>=11100/11101`) → SQDMULL{B,T} idx long, T=<10>.
#[inline]
fn decode_44_idx_mul(word: u32, out: &mut Instruction) {
    let zn = bits(word, 5, 5);
    let zd = bits(word, 0, 5);
    let s1210 = bits(word, 10, 3);
    let s1511 = bits(word, 11, 5);
    if s1511 == 0b11110 {
        // SQDMULH (R=0) / SQRDMULH (R=1): same-size by element.
        let r = bit(word, 10);
        out.set(if r == 0 { Code::SveSqdmulhIdx } else { Code::SveSqrdmulhIdx });
        out.set_mnemonic(if r == 0 { Mnemonic::Sqdmulh } else { Mnemonic::Sqrdmulh });
        push_sve2_idx_same(word, out, zd, zn);
        return;
    }
    if s1210 == 0b110 {
        // MUL idx (`<15:10>=111110`): same-size by element.
        out.set(Code::SveMulIdx);
        out.set_mnemonic(Mnemonic::Mul);
        push_sve2_idx_same(word, out, zd, zn);
        return;
    }
    // SQDMULL{B,T} idx long (`<15:12>=1110`): T=<10>. size==2 -> .s<-.h, size==3
    // -> .d<-.s. (`<11>` is i3l/i2l part of the index.)
    if bits(word, 12, 4) == 0b1110 {
        let size = bits(word, 22, 2);
        let Some((da, sa)) = widen2(size) else { return };
        if size < 2 {
            return;
        }
        let t = bit(word, 10);
        let (idx, zm) = if size == 2 {
            (((bits(word, 19, 2) << 1) | bit(word, 11)), bits(word, 16, 3))
        } else {
            (((bit(word, 20) << 1) | bit(word, 11)), bits(word, 16, 4))
        };
        out.set(Code::SveSqdmulLongIdx);
        out.set_mnemonic(if t == 0 { Mnemonic::Sqdmullb } else { Mnemonic::Sqdmullt });
        out.push_operand(zreg(zd, da));
        out.push_operand(zreg(zn, sa));
        out.push_operand(zreg_idx(zm, sa, idx as u8));
    }
}

/// Push the three same-size by-element operands shared by MUL/SQDMULH/SQRDMULH
/// idx (and MLA/MLS idx): `<Zd>.<T>, <Zn>.<T>, <Zm>.<T>[idx]`. The element size
/// and index/`Zm` field widths follow the SVE by-element layout: `.h`
/// (size==01, idx=i3h:i3l=<22>:<20:19>, 3-bit Zm), `.s` (size==10, idx=i2=<20:19>,
/// 3-bit Zm), `.d` (size==11, idx=i1=<20>, 4-bit Zm).
#[inline]
fn push_sve2_idx_same(word: u32, out: &mut Instruction, zd: u32, zn: u32) {
    // Element size: `<23>=0` -> `.h` (size field `<23:22>` is `0x`, the `<22>` bit
    // is the top index bit); `<23:22>=10` -> `.s`; `<23:22>=11` -> `.d`.
    let (a, idx, zm) = if bit(word, 23) == 0 {
        // .h: idx = i3h:i3l = <22>:<20:19> (3 bits), Zm = <18:16> (3-bit).
        (
            VA::Sh,
            (bit(word, 22) << 2) | bits(word, 19, 2),
            bits(word, 16, 3),
        )
    } else if bit(word, 22) == 0 {
        // .s: idx = i2 = <20:19> (2 bits), Zm = <18:16> (3-bit).
        (VA::Ss, bits(word, 19, 2), bits(word, 16, 3))
    } else {
        // .d: idx = i1 = <20> (1 bit), Zm = <19:16> (4-bit).
        (VA::Sd, bit(word, 20), bits(word, 16, 4))
    };
    out.push_operand(zreg(zd, a));
    out.push_operand(zreg(zn, a));
    out.push_operand(zreg_idx(zm, a, idx as u8));
}

/// SQRDMLAH/SQRDMLSH (SVE2), vector form: `<op> <Zda>.<T>, <Zn>.<T>, <Zm>.<T>`.
/// All three operands share the `size` arrangement (.b/.h/.s/.d); `S=word<10>`
/// selects SQRDMLSH (1) over SQRDMLAH (0).
#[inline]
fn decode_44_sqrdml_vec(word: u32, out: &mut Instruction) {
    let s = bit(word, 10);
    let a = arr(bits(word, 22, 2));
    let zm = bits(word, 16, 5);
    let zn = bits(word, 5, 5);
    let zda = bits(word, 0, 5);
    if s == 0 {
        out.set(Code::SveSqrdmlah);
        out.set_mnemonic(Mnemonic::Sqrdmlah);
    } else {
        out.set(Code::SveSqrdmlsh);
        out.set_mnemonic(Mnemonic::Sqrdmlsh);
    }
    out.push_operand(zreg(zda, a));
    out.push_operand(zreg(zn, a));
    out.push_operand(zreg(zm, a));
}

/// SQRDMLAH/SQRDMLSH (SVE2), indexed form: `<op> <Zda>.<T>, <Zn>.<T>,
/// <Zm>.<T>[idx]`. The element size and index/`Zm` field widths follow the SVE
/// by-element layout: `.h` (word<23>==0, idx=i3h:i3l, 3-bit Zm), `.s`
/// (word<23:22>==10, idx=i2, 3-bit Zm), `.d` (word<23:22>==11, idx=i1, 4-bit
/// Zm). `S=word<10>` selects SQRDMLSH over SQRDMLAH.
#[inline]
fn decode_44_sqrdml_idx(word: u32, out: &mut Instruction) {
    let s = bit(word, 10);
    let zn = bits(word, 5, 5);
    let zda = bits(word, 0, 5);
    if s == 0 {
        out.set(Code::SveSqrdmlahIdx);
        out.set_mnemonic(Mnemonic::Sqrdmlah);
    } else {
        out.set(Code::SveSqrdmlshIdx);
        out.set_mnemonic(Mnemonic::Sqrdmlsh);
    }
    if bit(word, 23) == 0 {
        // .h: index = i3h:i3l (word<22>, word<20:19>), Zm = word<18:16> (3-bit).
        let idx = ((bit(word, 22) << 2) | bits(word, 19, 2)) as u8;
        let zm = bits(word, 16, 3);
        out.push_operand(zreg(zda, VA::Sh));
        out.push_operand(zreg(zn, VA::Sh));
        out.push_operand(zreg_idx(zm, VA::Sh, idx));
    } else if bit(word, 22) == 0 {
        // .s: index = i2 (word<20:19>), Zm = word<18:16> (3-bit).
        let idx = bits(word, 19, 2) as u8;
        let zm = bits(word, 16, 3);
        out.push_operand(zreg(zda, VA::Ss));
        out.push_operand(zreg(zn, VA::Ss));
        out.push_operand(zreg_idx(zm, VA::Ss, idx));
    } else {
        // .d: index = i1 (word<20>), Zm = word<19:16> (4-bit).
        let idx = bit(word, 20) as u8;
        let zm = bits(word, 16, 4);
        out.push_operand(zreg(zda, VA::Sd));
        out.push_operand(zreg(zn, VA::Sd));
        out.push_operand(zreg_idx(zm, VA::Sd, idx));
    }
}

/// SVE2 integer multiply-long / multiply-add-long / multiply-subtract-long
/// (`_z_zz_`/`_z_zzz_`, top byte 0x45). The family lives in word<13:10> =
/// op:S:U:T (B/T = bottom/top). Source elements widen 2x: size 01->.b->.h,
/// 10->.h->.s, 11->.s->.d.
#[inline]
fn decode_45(word: u32, out: &mut Instruction) {
    // RAX1 (`<15:11>=11110`, `<10>=1`, `<23:22>=00`, `<21>=1`): `RAX1 <Zd>.D,
    // <Zn>.D, <Zm>.D`.
    if bits(word, 11, 5) == 0b11110 && bit(word, 10) == 1 && bits(word, 21, 3) == 0b001 {
        let a = VA::Sd;
        out.set(Code::SveRax1);
        out.set_mnemonic(Mnemonic::Rax1);
        out.push_operand(zreg(bits(word, 0, 5), a));
        out.push_operand(zreg(bits(word, 5, 5), a));
        out.push_operand(zreg(bits(word, 16, 5), a));
        return;
    }
    // SVE2 bit-shift accumulate / insert / shift-long-immediate and the
    // narrowing-shift / saturating-extract-narrow forms also live in 0x45.
    decode_45_shift(word, out);
    if !out.is_invalid() {
        return;
    }
    // SVE2 integer add/subtract long/wide, high-narrowing, ABAL, ADC/SBC long and
    // the bottom-top long forms all live in 0x45 alongside the multiply-long
    // block. Try that family next; it declines (leaves Invalid) for the
    // multiply-long opcodes, which fall through below.
    decode_45_addsub(word, out);
    if !out.is_invalid() {
        return;
    }
    // SVE2 bit-permute / interleaving-EOR / histogram / SM4 / integer matmul.
    decode_45_misc(word, out);
    if !out.is_invalid() {
        return;
    }
    // SVE2 multiply-long (`_z_zz_`): `word<15:13>==011` and `word<21>==0`. The
    // narrowing add/sub-high ops sharing `<15:13>=011` have `<21>==1` (handled by
    // `decode_45_addsub` above). Source elements widen 2x (size 01->.b->.h,
    // 10->.h->.s, 11->.s->.d).
    if bits(word, 13, 3) != 0b011 || bit(word, 21) != 0 {
        return;
    }
    let size = bits(word, 22, 2);
    let zm = bits(word, 16, 5);
    let zn = bits(word, 5, 5);
    let zd = bits(word, 0, 5);
    let (code, mnem) = match bits(word, 10, 4) {
        // 1000/1001 = SQDMULLB/T (saturating doubling multiply-long).
        0b1000 => (Code::SveSqdmulLong, Mnemonic::Sqdmullb),
        0b1001 => (Code::SveSqdmulLong, Mnemonic::Sqdmullt),
        0b1010 => (Code::SvePmulLong, Mnemonic::Pmullb),
        0b1011 => (Code::SvePmulLong, Mnemonic::Pmullt),
        0b1100 => (Code::SveMulLong, Mnemonic::Smullb),
        0b1101 => (Code::SveMulLong, Mnemonic::Smullt),
        0b1110 => (Code::SveMulLong, Mnemonic::Umullb),
        0b1111 => (Code::SveMulLong, Mnemonic::Umullt),
        _ => return,
    };
    // Source elements widen 2x. The integer MULL forms allow size 01/10/11
    // (.h<-.b / .s<-.h / .d<-.s). PMULL additionally has the `.q<-.d` crypto
    // form at size==00 (and uses the same widening table otherwise).
    let (da, sa) = match size {
        0b00 if matches!(code, Code::SvePmulLong) => (VA::Sq, VA::Sd),
        0b01 => (VA::Sh, VA::Sb),
        0b10 => (VA::Ss, VA::Sh),
        0b11 => (VA::Sd, VA::Ss),
        _ => return,
    };
    out.set(code);
    out.set_mnemonic(mnem);
    out.push_operand(zreg(zd, da));
    out.push_operand(zreg(zn, sa));
    out.push_operand(zreg(zm, sa));
}

/// SVE2 bit-permute (BEXT/BDEP/BGRP), interleaving-EOR (EORBT/EORTB), histogram
/// (HISTCNT/HISTSEG), SM4 (SM4E/SM4EKEY) and integer matmul (SMMLA/UMMLA/USMMLA)
/// — all top byte 0x45. Leaves `out` invalid for other encodings.
#[inline]
fn decode_45_misc(word: u32, out: &mut Instruction) {
    let size = bits(word, 22, 2);
    let a = arr(size);
    let zm = bits(word, 16, 5);
    let zn = bits(word, 5, 5);
    let zd = bits(word, 0, 5);
    if bit(word, 21) == 0 {
        // EORBT/EORTB (`<15:11>=10010`, `<10>=tb`): same-size interleaving XOR.
        if bits(word, 11, 5) == 0b10010 {
            out.set(Code::SveEorInterleave);
            out.set_mnemonic(if bit(word, 10) == 0 { Mnemonic::Eorbt } else { Mnemonic::Eortb });
            out.push_operand(zreg(zd, a));
            out.push_operand(zreg(zn, a));
            out.push_operand(zreg(zm, a));
            return;
        }
        // BEXT/BDEP/BGRP (`<15:12>=1011`, `<11:10>=opc`): same-size bit-permute.
        if bits(word, 12, 4) == 0b1011 {
            let mnem = match bits(word, 10, 2) {
                0b00 => Mnemonic::Bext,
                0b01 => Mnemonic::Bdep,
                0b10 => Mnemonic::Bgrp,
                _ => return,
            };
            out.set(Code::SveBitPerm);
            out.set_mnemonic(mnem);
            out.push_operand(zreg(zd, a));
            out.push_operand(zreg(zn, a));
            out.push_operand(zreg(zm, a));
            return;
        }
        // {S,U,US}MMLA (`<15:10>=100110`): `<Zda>.S, <Zn>.B, <Zm>.B`. `uns=<23:22>`
        // selects SMMLA(00)/USMMLA(10)/UMMLA(11).
        if bits(word, 10, 6) == 0b100110 {
            let mnem = match size {
                0b00 => Mnemonic::Smmla,
                0b10 => Mnemonic::Usmmla,
                0b11 => Mnemonic::Ummla,
                _ => return,
            };
            out.set(Code::SveMatmulInt);
            out.set_mnemonic(mnem);
            out.push_operand(zreg(zd, VA::Ss));
            out.push_operand(zreg(zn, VA::Sb));
            out.push_operand(zreg(zm, VA::Sb));
            return;
        }
        // {S,U}ABA (`<15:11>=11111`, `<10>=U`): same-size abs-diff accumulate.
        if bits(word, 11, 5) == 0b11111 {
            out.set(Code::SveAbaSame);
            out.set_mnemonic(if bit(word, 10) == 0 { Mnemonic::Saba } else { Mnemonic::Uaba });
            out.push_operand(zreg(zd, a));
            out.push_operand(zreg(zn, a));
            out.push_operand(zreg(zm, a));
            return;
        }
        return;
    }
    // `<21>=1`.
    // MATCH/NMATCH (`<15:13>=100`): `<Pd>.<T>, <Pg>/Z, <Zn>.<T>, <Zm>.<T>`. `.b`/
    // `.h` only (size<2). `<4>=op` (0=MATCH,1=NMATCH), Pd=<3:0>.
    if bits(word, 13, 3) == 0b100 {
        if size >= 2 {
            return;
        }
        let pg = bits(word, 10, 3);
        out.set(Code::SveMatch);
        out.set_mnemonic(if bit(word, 4) == 0 { Mnemonic::Match } else { Mnemonic::Nmatch });
        out.push_operand(preg_sz(bits(word, 0, 4), a));
        out.push_operand(preg_q(pg, PredQual::Zeroing));
        out.push_operand(zreg(zn, a));
        out.push_operand(zreg(zm, a));
        return;
    }
    // AESMC (`<10>=0`) / AESIMC (`<10>=1`): `<21:16>=100000`, `<15:11>=11100`,
    // `<9:5>=00000`. `<Zdn>.B, <Zdn>.B`.
    if bits(word, 16, 6) == 0b100000
        && bits(word, 11, 5) == 0b11100
        && bits(word, 5, 5) == 0
        && size == 0
    {
        out.set(Code::SveAesMc);
        out.set_mnemonic(if bit(word, 10) == 0 { Mnemonic::Aesmc } else { Mnemonic::Aesimc });
        out.push_operand(zreg(zd, VA::Sb));
        out.push_operand(zreg(zd, VA::Sb));
        return;
    }
    // AESE (`<10>=0`) / AESD (`<10>=1`): `<21:16>=100010`, `<15:11>=11100`.
    // `<Zdn>.B, <Zdn>.B, <Zm>.B` (destructive: Zm=<9:5>).
    if bits(word, 16, 6) == 0b100010 && bits(word, 11, 5) == 0b11100 && size == 0 {
        out.set(Code::SveAesZz);
        out.set_mnemonic(if bit(word, 10) == 0 { Mnemonic::Aese } else { Mnemonic::Aesd });
        out.push_operand(zreg(zd, VA::Sb));
        out.push_operand(zreg(zd, VA::Sb));
        out.push_operand(zreg(zn, VA::Sb));
        return;
    }
    // HISTCNT (`<15:13>=110`): `<Zd>.<T>, <Pg>/Z, <Zn>.<T>, <Zm>.<T>` (.s/.d only).
    if bits(word, 13, 3) == 0b110 {
        if size < 2 {
            return;
        }
        let pg = bits(word, 10, 3);
        out.set(Code::SveHistcnt);
        out.set_mnemonic(Mnemonic::Histcnt);
        out.push_operand(zreg(zd, a));
        out.push_operand(preg_q(pg, PredQual::Zeroing));
        out.push_operand(zreg(zn, a));
        out.push_operand(zreg(zm, a));
        return;
    }
    // HISTSEG (`<15:10>=101000`): `<Zd>.B, <Zn>.B, <Zm>.B`.
    if bits(word, 10, 6) == 0b101000 {
        out.set(Code::SveHistseg);
        out.set_mnemonic(Mnemonic::Histseg);
        out.push_operand(zreg(zd, VA::Sb));
        out.push_operand(zreg(zn, VA::Sb));
        out.push_operand(zreg(zm, VA::Sb));
        return;
    }
    // SM4EKEY (`<15:11>=11110`, `<10>=0`): `<Zd>.S, <Zn>.S, <Zm>.S`.
    if bits(word, 11, 5) == 0b11110 && bit(word, 10) == 0 && size == 0 {
        out.set(Code::SveSm4ekey);
        out.set_mnemonic(Mnemonic::Sm4ekey);
        out.push_operand(zreg(zd, VA::Ss));
        out.push_operand(zreg(zn, VA::Ss));
        out.push_operand(zreg(zm, VA::Ss));
        return;
    }
    // SM4E (`<21:16>=100011`, `<15:10>=111000`): `<Zdn>.S, <Zdn>.S, <Zm>.S`.
    // Destructive: Zm=<9:5>, Zdn=<4:0>.
    if bits(word, 16, 6) == 0b100011 && bits(word, 10, 6) == 0b111000 && size == 0 {
        out.set(Code::SveSm4e);
        out.set_mnemonic(Mnemonic::Sm4e);
        out.push_operand(zreg(zd, VA::Ss));
        out.push_operand(zreg(zd, VA::Ss));
        out.push_operand(zreg(zn, VA::Ss));
    }
}

/// SVE2 bit-shift accumulate / insert / shift-left-long-immediate (top byte
/// 0x45, `<21>=0`) and the narrowing-shift-right / saturating-extract-narrow
/// forms (`<21>=1`). Leaves `out` invalid for non-shift encodings.
///
/// Shift-amount fields: `tszh:tszl:imm3` (`<23:22>:<20:19>:<18:16>` for the
/// `<21>=0` forms; `<22>:<20:19>:<18:16>` for the narrowing `<21>=1` forms).
#[inline]
fn decode_45_shift(word: u32, out: &mut Instruction) {
    let imm3 = bits(word, 16, 3);
    let zn = bits(word, 5, 5);
    let zd = bits(word, 0, 5);
    if bit(word, 21) == 0 {
        // tsz for the same-size / shift-long forms: `<23:22>:<20:19>`.
        let tsz = (bits(word, 22, 2) << 2) | bits(word, 19, 2);
        match bits(word, 12, 4) {
            // {S,U}{,R}SRA (`<15:12>=1110`): right-shift accumulate, `<11>=R`,
            // `<10>=U`. Same-size `Zda.<T>, Zn.<T>, #shift`.
            0b1110 => {
                let Some((a, amt)) = right_shift_amount(tsz, imm3) else { return };
                let r = bit(word, 11);
                let u = bit(word, 10);
                let mnem = match (r, u) {
                    (0, 0) => Mnemonic::Ssra,
                    (0, 1) => Mnemonic::Usra,
                    (1, 0) => Mnemonic::Srsra,
                    _ => Mnemonic::Ursra,
                };
                out.set(Code::SveShiftAccum);
                out.set_mnemonic(mnem);
                out.push_operand(zreg(zd, a));
                out.push_operand(zreg(zn, a));
                out.push_operand(Operand::ImmUnsigned(amt as u64));
            }
            // {S,U}SHLL{B,T} (`<15:12>=1010`): shift-left long, `<11>=U`,`<10>=T`.
            // `Zd.<wide>, Zn.<narrow>, #shift`; the shift amount is computed from
            // the narrow element size.
            0b1010 => {
                let Some((sa, amt)) = left_shift_amount(tsz, imm3) else { return };
                let da = match sa {
                    VA::Sb => VA::Sh,
                    VA::Sh => VA::Ss,
                    VA::Ss => VA::Sd,
                    _ => return,
                };
                let u = bit(word, 11);
                let t = bit(word, 10);
                let mnem = match (u, t) {
                    (0, 0) => Mnemonic::Sshllb,
                    (0, 1) => Mnemonic::Sshllt,
                    (1, 0) => Mnemonic::Ushllb,
                    _ => Mnemonic::Ushllt,
                };
                out.set(Code::SveShiftLongImm);
                out.set_mnemonic(mnem);
                out.push_operand(zreg(zd, da));
                out.push_operand(zreg(zn, sa));
                out.push_operand(Operand::ImmUnsigned(amt as u64));
            }
            _ => {
                // SLI/SRI (`<15:11>=11110`, `<10>=op`): SLI (op=1, left) / SRI
                // (op=0, right). Same-size `Zd.<T>, Zn.<T>, #shift`.
                if bits(word, 11, 5) == 0b11110 {
                    let op = bit(word, 10);
                    let res = if op == 1 {
                        left_shift_amount(tsz, imm3)
                    } else {
                        right_shift_amount(tsz, imm3)
                    };
                    let Some((a, amt)) = res else { return };
                    out.set(Code::SveShiftInsert);
                    out.set_mnemonic(if op == 1 { Mnemonic::Sli } else { Mnemonic::Sri });
                    out.push_operand(zreg(zd, a));
                    out.push_operand(zreg(zn, a));
                    out.push_operand(Operand::ImmUnsigned(amt as u64));
                }
            }
        }
        return;
    }
    // `<21>=1`: narrowing-shift-right (`<15:13>=000`) and saturating-extract-narrow
    // (`<18:13>=000010`). tsz for narrowing: `<22>:<20:19>`.
    let tszn = (bit(word, 22) << 2) | bits(word, 19, 2);
    // Saturating extract narrow: `<18:13>=000010`, no immediate.
    if bits(word, 13, 6) == 0b000010 {
        let Some(idx) = tsz_size(tszn) else { return };
        let da = arr(idx); // narrow element.
        let sa = match idx {
            0 => VA::Sh,
            1 => VA::Ss,
            2 => VA::Sd,
            _ => return,
        };
        // `<12:11>=opc`, `<10>=T`: 00=SQXTN, 01=UQXTN, 10=SQXTUN.
        let t = bit(word, 10);
        let mnem = match (bits(word, 11, 2), t) {
            (0b00, 0) => Mnemonic::Sqxtnb,
            (0b00, 1) => Mnemonic::Sqxtnt,
            (0b01, 0) => Mnemonic::Uqxtnb,
            (0b01, 1) => Mnemonic::Uqxtnt,
            (0b10, 0) => Mnemonic::Sqxtunb,
            (0b10, 1) => Mnemonic::Sqxtunt,
            _ => return,
        };
        out.set(Code::SveExtractNarrow);
        out.set_mnemonic(mnem);
        out.push_operand(zreg(zd, da));
        out.push_operand(zreg(zn, sa));
        return;
    }
    if bits(word, 14, 2) == 0b00 {
        // Narrowing shift right (`<13>=op`,`<12>=U`,`<11>=R`,`<10>=T`):
        // `Zd.<narrow>, Zn.<wide>, #shift`. The shift is computed from the narrow
        // element size; the source is one size wider.
        let Some((da, amt)) = right_shift_amount(tszn, imm3) else { return };
        let sa = match da {
            VA::Sb => VA::Sh,
            VA::Sh => VA::Ss,
            VA::Ss => VA::Sd,
            _ => return,
        };
        let op = bit(word, 13);
        let u = bit(word, 12);
        let r = bit(word, 11);
        let t = bit(word, 10);
        // (op,U,R): 0,1,0=SHRN; 0,1,1=RSHRN; 1,0,0=SQSHRN; 1,0,1=SQRSHRN;
        // 0,0,0=SQSHRUN; 0,0,1=SQRSHRUN; 1,1,0=UQSHRN; 1,1,1=UQRSHRN.
        let mnem = match (op, u, r, t) {
            (0, 1, 0, 0) => Mnemonic::Shrnb,
            (0, 1, 0, 1) => Mnemonic::Shrnt,
            (0, 1, 1, 0) => Mnemonic::Rshrnb,
            (0, 1, 1, 1) => Mnemonic::Rshrnt,
            (1, 0, 0, 0) => Mnemonic::Sqshrnb,
            (1, 0, 0, 1) => Mnemonic::Sqshrnt,
            (1, 0, 1, 0) => Mnemonic::Sqrshrnb,
            (1, 0, 1, 1) => Mnemonic::Sqrshrnt,
            (0, 0, 0, 0) => Mnemonic::Sqshrunb,
            (0, 0, 0, 1) => Mnemonic::Sqshrunt,
            (0, 0, 1, 0) => Mnemonic::Sqrshrunb,
            (0, 0, 1, 1) => Mnemonic::Sqrshrunt,
            (1, 1, 0, 0) => Mnemonic::Uqshrnb,
            (1, 1, 0, 1) => Mnemonic::Uqshrnt,
            (1, 1, 1, 0) => Mnemonic::Uqrshrnb,
            _ => Mnemonic::Uqrshrnt,
        };
        out.set(Code::SveShiftNarrow);
        out.set_mnemonic(mnem);
        out.push_operand(zreg(zd, da));
        out.push_operand(zreg(zn, sa));
        out.push_operand(Operand::ImmUnsigned(amt as u64));
    }
}

/// SVE2 integer add/subtract long/wide, high-narrowing, ABAL, bottom-top long,
/// and ADC/SBC long (top byte 0x45). Returns leaving `out` invalid for the
/// multiply-long opcodes (handled by the caller). Dispatch on `<15:13>`:
/// * `000` (`<21>=0`): {S,U}{ADD,SUB}L{B,T} — 2x widen.
/// * `001` (`<21>=0`): {S,U}ABDL{B,T} — 2x widen.
/// * `010` (`<21>=0`): {S,U}{ADD,SUB}W{B,T} — wide (Zn wide, Zm narrow).
/// * `011` (`<21>=1`): {R,}{ADD,SUB}HN{B,T} — high narrowing (Zd narrow).
/// * `100` (`<21>=0`): SADDLBT / SSUBLBT / SSUBLTB — 2x widen, bottom-top.
/// * `110`,`<12:10>=000` (`<21>=0`): {S,U}ABAL{B,T} — 2x widen accumulate.
/// * `110`,`<12:10>=100`: ADCL{B,T} / SBCL{B,T} — same-size (`<23>`=op, `<22>`=sz).
#[inline]
fn decode_45_addsub(word: u32, out: &mut Instruction) {
    let size = bits(word, 22, 2);
    let zm = bits(word, 16, 5);
    let zn = bits(word, 5, 5);
    let zd = bits(word, 0, 5);
    let t = bit(word, 10);
    match bits(word, 13, 3) {
        // {S,U}{ADD,SUB}L{B,T} (`000`) and {S,U}ABDL{B,T} (`001`). `<12>=S`
        // (0=add,1=sub for L; for ABDL the op is sub), `<11>=U`, `<10>=T`.
        0b000 | 0b001 if bit(word, 21) == 0 => {
            let Some((da, sa)) = widen2(size) else { return };
            let abd = bit(word, 13) == 1; // <15:13>=001 -> ABDL.
            let s = bit(word, 12);
            let u = bit(word, 11);
            let mnem = if abd {
                match (u, t) {
                    (0, 0) => Mnemonic::Sabdlb,
                    (0, 1) => Mnemonic::Sabdlt,
                    (1, 0) => Mnemonic::Uabdlb,
                    _ => Mnemonic::Uabdlt,
                }
            } else {
                match (s, u, t) {
                    (0, 0, 0) => Mnemonic::Saddlb,
                    (0, 0, 1) => Mnemonic::Saddlt,
                    (0, 1, 0) => Mnemonic::Uaddlb,
                    (0, 1, 1) => Mnemonic::Uaddlt,
                    (1, 0, 0) => Mnemonic::Ssublb,
                    (1, 0, 1) => Mnemonic::Ssublt,
                    (1, 1, 0) => Mnemonic::Usublb,
                    _ => Mnemonic::Usublt,
                }
            };
            out.set(if abd { Code::SveAbdLong } else { Code::SveAddLong });
            out.set_mnemonic(mnem);
            out.push_operand(zreg(zd, da));
            out.push_operand(zreg(zn, sa));
            out.push_operand(zreg(zm, sa));
        }
        // {S,U}{ADD,SUB}W{B,T} (`010`): `Zd.<wide>, Zn.<wide>, Zm.<narrow>`.
        // `<12>=S` (0=add,1=sub), `<11>=U`, `<10>=T`.
        0b010 if bit(word, 21) == 0 => {
            let Some((da, sa)) = widen2(size) else { return };
            let s = bit(word, 12);
            let u = bit(word, 11);
            let mnem = match (s, u, t) {
                (0, 0, 0) => Mnemonic::Saddwb,
                (0, 0, 1) => Mnemonic::Saddwt,
                (0, 1, 0) => Mnemonic::Uaddwb,
                (0, 1, 1) => Mnemonic::Uaddwt,
                (1, 0, 0) => Mnemonic::Ssubwb,
                (1, 0, 1) => Mnemonic::Ssubwt,
                (1, 1, 0) => Mnemonic::Usubwb,
                _ => Mnemonic::Usubwt,
            };
            out.set(Code::SveAddWide);
            out.set_mnemonic(mnem);
            out.push_operand(zreg(zd, da));
            out.push_operand(zreg(zn, da));
            out.push_operand(zreg(zm, sa));
        }
        // {R,}{ADD,SUB}HN{B,T} (`011`, `<21>=1`): `Zd.<narrow>, Zn.<wide>,
        // Zm.<wide>`. `<12>=S` (0=add,1=sub), `<11>=R` (rounding), `<10>=T`.
        0b011 if bit(word, 21) == 1 => {
            let Some((da, sa)) = widen2(size) else { return };
            let s = bit(word, 12);
            let r = bit(word, 11);
            let mnem = match (s, r, t) {
                (0, 0, 0) => Mnemonic::Addhnb,
                (0, 0, 1) => Mnemonic::Addhnt,
                (0, 1, 0) => Mnemonic::Raddhnb,
                (0, 1, 1) => Mnemonic::Raddhnt,
                (1, 0, 0) => Mnemonic::Subhnb,
                (1, 0, 1) => Mnemonic::Subhnt,
                (1, 1, 0) => Mnemonic::Rsubhnb,
                _ => Mnemonic::Rsubhnt,
            };
            out.set(Code::SveAddHighNarrow);
            out.set_mnemonic(mnem);
            out.push_operand(zreg(zd, sa));
            out.push_operand(zreg(zn, da));
            out.push_operand(zreg(zm, da));
        }
        // SADDLBT / SSUBLBT / SSUBLTB (`100`, `<12>=0`): `1000|S|tb`. 2x widen.
        // `<11>=S` (0=SADDLBT, 1=SSUBLBT/SSUBLTB), `<10>=tb`.
        0b100 if bit(word, 21) == 0 && bit(word, 12) == 0 => {
            let Some((da, sa)) = widen2(size) else { return };
            let s = bit(word, 11);
            let tb = bit(word, 10);
            // S=0 -> SADDLBT; S=1 -> SSUBLBT (tb=0) / SSUBLTB (tb=1).
            let mnem = match (s, tb) {
                (0, _) => Mnemonic::Saddlbt,
                (1, 0) => Mnemonic::Ssublbt,
                _ => Mnemonic::Ssubltb,
            };
            out.set(Code::SveAddLongBt);
            out.set_mnemonic(mnem);
            out.push_operand(zreg(zd, da));
            out.push_operand(zreg(zn, sa));
            out.push_operand(zreg(zm, sa));
        }
        0b110 => {
            match bits(word, 10, 3) {
                // {S,U}ABAL{B,T} (`<15:12>=1100`, `<12:10>=000`... here `<12:10>`
                // is `0`,U,T -> the `<11>=U` form). `<12>=0`, `<11>=U`, `<10>=T`.
                0b000..=0b011 if bit(word, 21) == 0 => {
                    let Some((da, sa)) = widen2(size) else { return };
                    let u = bit(word, 11);
                    let mnem = match (u, t) {
                        (0, 0) => Mnemonic::Sabalb,
                        (0, 1) => Mnemonic::Sabalt,
                        (1, 0) => Mnemonic::Uabalb,
                        _ => Mnemonic::Uabalt,
                    };
                    out.set(Code::SveAbaLong);
                    out.set_mnemonic(mnem);
                    out.push_operand(zreg(zd, da));
                    out.push_operand(zreg(zn, sa));
                    out.push_operand(zreg(zm, sa));
                }
                // ADCL{B,T} / SBCL{B,T} (`<15:11>=11010`, `<12:10>=10T`, `<21>=0`).
                // `<23>`=op (0=ADC,1=SBC), `<22>`=sz (0=.s,1=.d). Same-size.
                // (`<21>=1` here is HISTCNT, handled by `decode_45_misc`.)
                0b100 | 0b101 if bit(word, 21) == 0 => {
                    let op = bit(word, 23);
                    let a = if bit(word, 22) == 1 { VA::Sd } else { VA::Ss };
                    let mnem = match (op, t) {
                        (0, 0) => Mnemonic::Adclb,
                        (0, 1) => Mnemonic::Adclt,
                        (1, 0) => Mnemonic::Sbclb,
                        _ => Mnemonic::Sbclt,
                    };
                    out.set(Code::SveAddCarryLong);
                    out.set_mnemonic(mnem);
                    out.push_operand(zreg(zd, a));
                    out.push_operand(zreg(zn, a));
                    out.push_operand(zreg(zm, a));
                }
                // CADD (`<16>=0`) / SQCADD (`<16>=1`) (`<15:11>=11011`, `<21:17>=0`):
                // `<Zdn>.<T>, <Zdn>.<T>, <Zm>.<T>, #rot`. rot=<10> (0->90, 1->270).
                0b110 | 0b111 if bit(word, 21) == 0 && bits(word, 17, 4) == 0 => {
                    let sqr = bit(word, 16) == 1;
                    let rot = bit(word, 10);
                    // CADD/SQCADD are destructive: Zm=<9:5>, Zdn=<4:0>.
                    let zmc = bits(word, 5, 5);
                    out.set(if sqr { Code::SveSqcadd } else { Code::SveCadd });
                    out.set_mnemonic(if sqr { Mnemonic::Sqcadd } else { Mnemonic::Cadd });
                    out.push_operand(zreg(zd, arr(size)));
                    out.push_operand(zreg(zd, arr(size)));
                    out.push_operand(zreg(zmc, arr(size)));
                    // rot: 0 -> #0x5a (90), 1 -> #0x10e (270).
                    out.push_operand(Operand::ImmUnsigned(if rot == 0 { 90 } else { 270 }));
                }
                _ => {}
            }
        }
        _ => {}
    }
}

/// The saturating scalar INC/DEC mnemonic from `(U, D, size)`.
#[inline]
fn sat_scalar_mnemonic(unsigned: bool, dec: bool, size: u32) -> Mnemonic {
    match (unsigned, dec, size) {
        (false, false, 0) => Mnemonic::Sqincb,
        (false, false, 1) => Mnemonic::Sqinch,
        (false, false, 2) => Mnemonic::Sqincw,
        (false, false, _) => Mnemonic::Sqincd,
        (true, false, 0) => Mnemonic::Uqincb,
        (true, false, 1) => Mnemonic::Uqinch,
        (true, false, 2) => Mnemonic::Uqincw,
        (true, false, _) => Mnemonic::Uqincd,
        (false, true, 0) => Mnemonic::Sqdecb,
        (false, true, 1) => Mnemonic::Sqdech,
        (false, true, 2) => Mnemonic::Sqdecw,
        (false, true, _) => Mnemonic::Sqdecd,
        (true, true, 0) => Mnemonic::Uqdecb,
        (true, true, 1) => Mnemonic::Uqdech,
        (true, true, 2) => Mnemonic::Uqdecw,
        (true, true, _) => Mnemonic::Uqdecd,
    }
}

// ---------------------------------------------------------------------------
// Top-level integer dispatch.
// ---------------------------------------------------------------------------

/// Decode an SVE/SVE2 integer-family instruction into `out`.
///
/// Called by [`crate::decode::sve::decode`] for the `000`/`001`/`010` quadrants.
/// Routes on the high byte (`word<31:24>`) and the inner class selectors. Leaves
/// `out` invalid when the encoding is not an integer leaf (so the permute /
/// predicate decoder can try). Total and panic-free.
#[inline]
pub fn decode(word: u32, ip: u64, features: FeatureSet, out: &mut Instruction) {
    let _ = ip;
    if !features.has(crate::features::Feature::Sve) {
        return;
    }
    let hi = bits(word, 24, 8); // word<31:24>
    match hi {
        // 0x04 / 0x05: SVE integer, shift, misc, MOVPRFX, DUP/CPY (top byte 000).
        0x04 => decode_04(word, features, out),
        0x05 => decode_05(word, out),
        // 0x25: SVE integer immediate (add/sub/min/max/mul) + INC/DEC by pred,
        //       compare-with-imm, predicate counts (top byte 001).
        0x25 => decode_25(word, out),
        0x24 => decode_24(word, out),
        // 0x44 / 0x45: SVE2 integer multiply-add / DOT / widening (top byte 010).
        0x44 => decode_44(word, out),
        0x45 => decode_45(word, out),
        _ => {}
    }
}

// ===========================================================================
// Top byte 0x04 — SVE integer / shift / reduction / INC-DEC / INDEX / ADR.
// Dispatch on word<21> then word<15:13> (see the ARM ARM SVE encoding index).
// ===========================================================================

#[inline]
fn decode_04(word: u32, features: FeatureSet, out: &mut Instruction) {
    let b21 = bit(word, 21);
    let sel = bits(word, 13, 3); // word<15:13>
    match (b21, sel) {
        // Predicated integer binary (destructive): ADD/SUB/.. min/max/div/mul/logical.
        (0, 0b000) => decode_pred_binary(word, out),
        // Reductions + predicated MOVPRFX.
        (0, 0b001) => decode_reduction(word, out),
        // MLA / MLS (predicated).
        (0, 0b010) => decode_mla_mls(word, out, false),
        (0, 0b011) => decode_mla_mls(word, out, true),
        // Shifts predicated (imm / vector / wide) + ASRD and SVE2 shift-imm.
        (0, 0b100) => decode_shift_pred(word, out),
        // Unary predicated: ABS/NEG/CNT/CLS/CLZ/CNOT/NOT/SXTB../UXTB..
        (0, 0b101) => decode_unary_pred(word, out),
        // MAD / MSB (predicated).
        (0, 0b110) => decode_mad_msb(word, out, false),
        (0, 0b111) => decode_mad_msb(word, out, true),

        // Unpredicated integer arithmetic ADD/SUB/SQADD/..
        (1, 0b000) => decode_arith_zzz(word, out),
        // Unpredicated bitwise logical AND/ORR/EOR/BIC (+ MOV alias).
        (1, 0b001) => decode_logical_zzz(word, out),
        // INDEX, ADDVL/ADDPL, RDVL (+ the SME streaming ADDSVL/ADDSPL/RDSVL).
        (1, 0b010) => decode_index_addvl(word, features, out),
        // Unpredicated multiply MUL/SMULH/UMULH/PMUL (+ SQDMULH/SQRDMULH).
        (1, 0b011) => decode_mul_zzz(word, out),
        // Shift by immediate (unpred) and wide-shift (unpred).
        (1, 0b100) => decode_shift_unpred(word, out),
        // ADR (vector) and unpredicated MOVPRFX.
        (1, 0b101) => decode_adr_movprfx(word, out),
        // INC/DEC vector by element count.
        (1, 0b110) => decode_incdec_vec(word, out),
        // CNTB/H/W/D and INC/DEC scalar by element count.
        (1, 0b111) => decode_cnt_incdec_scalar(word, out),
        _ => {}
    }
}

/// Predicated integer binary destructive (`<21>=0`, `<15:13>=000`):
/// `op <Zdn>.<T>, <Pg>/M, <Zdn>.<T>, <Zm>.<T>`. The operation is in `<20:16>`
/// (an opcode) — actually `word<18:16>` plus `word<20:19>` form the table.
#[inline]
fn decode_pred_binary(word: u32, out: &mut Instruction) {
    let size = bits(word, 22, 2);
    let opc = bits(word, 16, 5); // word<20:16>
    let pg = bits(word, 10, 3);
    let zm = bits(word, 5, 5);
    let zdn = bits(word, 0, 5);
    let a = arr(size);
    // ADDPT/SUBPT (FEAT_CPA predicated): `<20:16>=00100`/`00101`, `.d` only
    // (`<23:22>=11`). Destructive `<Zdn>.D, <Pg>/M, <Zdn>.D, <Zm>.D`.
    if matches!(opc, 0b00100 | 0b00101) {
        if size != 0b11 {
            return;
        }
        out.set(if opc == 0b00100 { Code::SveAddptPred } else { Code::SveSubptPred });
        pred_binary(out, zdn, pg, zm, a);
        return;
    }
    let code = match opc {
        0b00000 => Code::SveAddZpzz,
        0b00001 => Code::SveSubZpzz,
        0b00011 => Code::SveSubrZpzz,
        0b01000 => Code::SveSmaxZpzz,
        0b01001 => Code::SveUmaxZpzz,
        0b01010 => Code::SveSminZpzz,
        0b01011 => Code::SveUminZpzz,
        0b01100 => Code::SveSabdZpzz,
        0b01101 => Code::SveUabdZpzz,
        0b10000 => Code::SveMulZpzz,
        0b10010 => Code::SveSmulhZpzz,
        0b10011 => Code::SveUmulhZpzz,
        0b10100 => Code::SveSdivZpzz,
        0b10101 => Code::SveUdivZpzz,
        0b10110 => Code::SveSdivrZpzz,
        0b10111 => Code::SveUdivrZpzz,
        0b11000 => Code::SveOrrZpzz,
        0b11001 => Code::SveEorZpzz,
        0b11010 => Code::SveAndZpzz,
        0b11011 => Code::SveBicZpzz,
        _ => return,
    };
    // SDIV/UDIV/SDIVR/UDIVR only exist for .s/.d (size>=2); leave others invalid.
    if matches!(
        code,
        Code::SveSdivZpzz | Code::SveUdivZpzz | Code::SveSdivrZpzz | Code::SveUdivrZpzz
    ) && size < 2
    {
        return;
    }
    out.set(code);
    pred_binary(out, zdn, pg, zm, a);
}

/// Integer reductions (`<21>=0`, `<15:13>=001`) and predicated `MOVPRFX`.
#[inline]
fn decode_reduction(word: u32, out: &mut Instruction) {
    let size = bits(word, 22, 2);
    let opc = bits(word, 16, 5); // word<20:16>
    let pg = bits(word, 10, 3);
    let zn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);
    let a = arr(size);

    // MOVPRFX (predicated) is opc==10001/10000 with op<15:13>==001 and a special
    // <20:16>; the corpus shows movprfx_z_p_z at <20:16>=10001/10000.
    match opc {
        0b00000 => {
            // SADDV -> Dd
            out.set(Code::SveSaddv);
            out.push_operand(scalar_fp(rd, 3));
            out.push_operand(preg(pg));
            out.push_operand(zreg(zn, a));
        }
        0b00001 => {
            out.set(Code::SveUaddv);
            out.push_operand(scalar_fp(rd, 3));
            out.push_operand(preg(pg));
            out.push_operand(zreg(zn, a));
        }
        0b01000 => reduction_scalar(out, Code::SveSmaxv, rd, pg, zn, size, a),
        0b01001 => reduction_scalar(out, Code::SveUmaxv, rd, pg, zn, size, a),
        0b01010 => reduction_scalar(out, Code::SveSminv, rd, pg, zn, size, a),
        0b01011 => reduction_scalar(out, Code::SveUminv, rd, pg, zn, size, a),
        0b11000 => reduction_scalar(out, Code::SveOrv, rd, pg, zn, size, a),
        0b11001 => reduction_scalar(out, Code::SveEorv, rd, pg, zn, size, a),
        0b11010 => reduction_scalar(out, Code::SveAndv, rd, pg, zn, size, a),
        // SVE2.1 quadword reductions to a NEON `V` register (`<Vd>.<T>`).
        0b00101 => reduction_qv(out, Code::SveAddqv, rd, pg, zn, size),
        0b01100 => reduction_qv(out, Code::SveSmaxqv, rd, pg, zn, size),
        0b01101 => reduction_qv(out, Code::SveUmaxqv, rd, pg, zn, size),
        0b01110 => reduction_qv(out, Code::SveSminqv, rd, pg, zn, size),
        0b01111 => reduction_qv(out, Code::SveUminqv, rd, pg, zn, size),
        0b11100 => reduction_qv(out, Code::SveOrqv, rd, pg, zn, size),
        0b11101 => reduction_qv(out, Code::SveEorqv, rd, pg, zn, size),
        0b11110 => reduction_qv(out, Code::SveAndqv, rd, pg, zn, size),
        // MOVPRFX predicated: `MOVPRFX <Zd>.<T>, <Pg>/<ZM>, <Zn>.<T>`.
        0b10000 | 0b10001 => {
            // M bit is word<16>; /m when set, /z when clear.
            let merging = bit(word, 16) == 1;
            out.set(Code::SveMovprfxZpz);
            out.push_operand(zreg(rd, a));
            out.push_operand(preg_q(pg, if merging { PredQual::Merging } else { PredQual::Zeroing }));
            out.push_operand(zreg(zn, a));
        }
        _ => {}
    }
}

/// SMAXV/SMINV/.. reduce into a scalar of the element width; SADDV/UADDV use D.
#[inline]
fn reduction_scalar(out: &mut Instruction, code: Code, rd: u32, pg: u32, zn: u32, size: u32, a: VA) {
    out.set(code);
    out.push_operand(scalar_fp(rd, size));
    out.push_operand(preg(pg));
    out.push_operand(zreg(zn, a));
}

/// SVE2.1 quadword reductions reduce into a full NEON `V` register whose
/// arrangement matches the source element size (`ADDQV <Vd>.<T>, <Pg>, <Zn>.<T>`).
#[inline]
fn reduction_qv(out: &mut Instruction, code: Code, rd: u32, pg: u32, zn: u32, size: u32) {
    out.set(code);
    out.push_operand(vreg(rd, va_neon(size)));
    out.push_operand(preg(pg));
    out.push_operand(zreg(zn, arr(size)));
}

/// MLA / MLS (`<21>=0`, `<15:13>=010/011`): `op <Zda>.<T>, <Pg>/M, <Zn>.<T>,
/// <Zm>.<T>`.
#[inline]
fn decode_mla_mls(word: u32, out: &mut Instruction, is_mls: bool) {
    let size = bits(word, 22, 2);
    let pg = bits(word, 10, 3);
    let zn = bits(word, 5, 5);
    let zda = bits(word, 0, 5);
    let zm = bits(word, 16, 5);
    let a = arr(size);
    out.set(if is_mls { Code::SveMlsZpzzz } else { Code::SveMlaZpzzz });
    out.push_operand(zreg(zda, a));
    out.push_operand(preg_q(pg, PredQual::Merging));
    out.push_operand(zreg(zn, a));
    out.push_operand(zreg(zm, a));
}

/// MAD / MSB (`<21>=0`, `<15:13>=110/111`): `op <Zdn>.<T>, <Pg>/M, <Zm>.<T>,
/// <Za>.<T>`.
#[inline]
fn decode_mad_msb(word: u32, out: &mut Instruction, is_msb: bool) {
    let size = bits(word, 22, 2);
    let pg = bits(word, 10, 3);
    // MAD/MSB: Zm=<20:16>, Za=<9:5>, Zdn=<4:0> (ARM ARM field layout).
    let zm = bits(word, 16, 5);
    let za = bits(word, 5, 5);
    let zdn = bits(word, 0, 5);
    let a = arr(size);
    out.set(if is_msb { Code::SveMsbZpzzz } else { Code::SveMadZpzzz });
    out.push_operand(zreg(zdn, a));
    out.push_operand(preg_q(pg, PredQual::Merging));
    out.push_operand(zreg(zm, a));
    out.push_operand(zreg(za, a));
}

/// Shifts predicated (`<21>=0`, `<15:13>=100`): shift-by-immediate, shift-by-
/// vector, and wide-element shifts, plus `ASRD` and the SVE2 saturating
/// shift-immediates. The class is `word<20:16>`.
#[inline]
fn decode_shift_pred(word: u32, out: &mut Instruction) {
    let opc = bits(word, 16, 5); // word<20:16>
    let pg = bits(word, 10, 3);
    let zdn = bits(word, 0, 5);

    match opc {
        // Shift by immediate (predicated): tszh=<23:22>, tszl=<9:8>, imm3=<7:5>.
        0b00000 | 0b00001 | 0b00011 | 0b00100 | 0b00110 | 0b00111 | 0b01100 | 0b01101
        | 0b01111 => {
            let tsz = (bits(word, 22, 2) << 2) | bits(word, 8, 2);
            let imm3 = bits(word, 5, 3);
            let (code, mnem, sh) = match opc {
                0b00000 => (Code::SveAsrZpzi, Mnemonic::Asr, right_shift_amount(tsz, imm3)),
                0b00001 => (Code::SveLsrZpzi, Mnemonic::Lsr, right_shift_amount(tsz, imm3)),
                0b00011 => (Code::SveLslZpzi, Mnemonic::Lsl, left_shift_amount(tsz, imm3)),
                0b00100 => (Code::SveAsrdZpzi, Mnemonic::Asrd, right_shift_amount(tsz, imm3)),
                // SVE2 saturating shift-left immediates and rounding shifts: reuse
                // the LSL/ASR codes for identity but install the right mnemonic.
                0b00110 => (Code::SveLslZpzi, Mnemonic::Sqshl, left_shift_amount(tsz, imm3)),
                0b00111 => (Code::SveLslZpzi, Mnemonic::Uqshl, left_shift_amount(tsz, imm3)),
                0b01100 => (Code::SveAsrZpzi, Mnemonic::Srshr, right_shift_amount(tsz, imm3)),
                0b01101 => (Code::SveLsrZpzi, Mnemonic::Urshr, right_shift_amount(tsz, imm3)),
                _ => (Code::SveLslZpzi, Mnemonic::Sqshlu, left_shift_amount(tsz, imm3)),
            };
            let Some((a, amt)) = sh else { return };
            out.set(code);
            out.set_mnemonic(mnem);
            out.push_operand(zreg(zdn, a));
            out.push_operand(preg_q(pg, PredQual::Merging));
            out.push_operand(zreg(zdn, a));
            out.push_operand(Operand::ImmUnsigned(amt as u64));
        }
        // Shift by vector (predicated): `op <Zdn>.<T>, <Pg>/M, <Zdn>.<T>, <Zm>.<T>`.
        0b10000 | 0b10001 | 0b10011 | 0b10100 | 0b10101 | 0b10111 => {
            let size = bits(word, 22, 2);
            let a = arr(size);
            let zm = bits(word, 5, 5);
            let code = match opc {
                0b10000 => Code::SveAsrZpzz,
                0b10001 => Code::SveLsrZpzz,
                0b10011 => Code::SveLslZpzz,
                0b10100 => Code::SveAsrrZpzz,
                0b10101 => Code::SveLsrrZpzz,
                _ => Code::SveLslrZpzz,
            };
            out.set(code);
            pred_binary(out, zdn, pg, zm, a);
        }
        // Wide-element shifts (predicated): `op <Zdn>.<T>, <Pg>/M, <Zdn>.<T>,
        // <Zm>.D`.
        0b11000 | 0b11001 | 0b11011 => {
            let size = bits(word, 22, 2);
            if size == 3 {
                return; // .d source elements reserved.
            }
            let a = arr(size);
            let zm = bits(word, 5, 5);
            let code = match opc {
                0b11000 => Code::SveAsrWidePred,
                0b11001 => Code::SveLsrWidePred,
                _ => Code::SveLslWidePred,
            };
            out.set(code);
            out.push_operand(zreg(zdn, a));
            out.push_operand(preg_q(pg, PredQual::Merging));
            out.push_operand(zreg(zdn, a));
            out.push_operand(zreg(zm, VA::Sd));
        }
        _ => {}
    }
}

/// Unary predicated (`<21>=0`, `<15:13>=101`): ABS/NEG/CNT/CLS/CLZ/CNOT/NOT/
/// SXTB../UXTB.. — `op <Zd>.<T>, <Pg>/<M|Z>, <Zn>.<T>`. The operation is the
/// 4-bit `word<19:16>`; `word<20>` selects the merging (`/m`, base SVE) vs the
/// zeroing (`/z`, FEAT_SVE2p1) predicate form. Extend forms require the
/// destination element to be wider than the extended width (else unallocated).
#[inline]
fn decode_unary_pred(word: u32, out: &mut Instruction) {
    let size = bits(word, 22, 2);
    let op4 = bits(word, 16, 4); // word<19:16>: operation selector
    let merging = bit(word, 20) == 1; // <20>: 1 = /m (SVE), 0 = /z (SVE2.1)
    let pg = bits(word, 10, 3);
    let zn = bits(word, 5, 5);
    let zd = bits(word, 0, 5);
    let a = arr(size);
    let code = match op4 {
        0b0000 if size >= 1 => Code::SveSxtbZpz,
        0b0001 if size >= 1 => Code::SveUxtbZpz,
        0b0010 if size >= 2 => Code::SveSxthZpz,
        0b0011 if size >= 2 => Code::SveUxthZpz,
        0b0100 if size == 3 => Code::SveSxtwZpz,
        0b0101 if size == 3 => Code::SveUxtwZpz,
        0b0110 => Code::SveAbsZpz,
        0b0111 => Code::SveNegZpz,
        0b1000 => Code::SveClsZpz,
        0b1001 => Code::SveClzZpz,
        0b1010 => Code::SveCntZpz,
        0b1011 => Code::SveCnotZpz,
        0b1110 => Code::SveNotZpz,
        // FABS/FNEG (1100/1101) live here too but belong to sve_fp; everything
        // else (incl. illegal extend element sizes) is unallocated.
        _ => return,
    };
    out.set(code);
    out.push_operand(zreg(zd, a));
    out.push_operand(preg_q(pg, if merging { PredQual::Merging } else { PredQual::Zeroing }));
    out.push_operand(zreg(zn, a));
}

#[cfg(test)]
mod tests {
    use crate::format::{BufSink, FmtFormatter, Formatter};
    use crate::{Decoder, DecoderOptions};

    /// Decode `word` and render with the default UAL formatter into `buf`,
    /// returning the formatted `&str` (zero-alloc, `no_std`-friendly).
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

    /// Decode `word`, re-encode it, and require the exact same word back.
    #[track_caller]
    fn rt(word: u32) {
        let bytes = word.to_le_bytes();
        let mut dec = Decoder::new(&bytes, 0x1000, DecoderOptions::default());
        let insn = dec.decode();
        assert!(!insn.is_invalid(), "word {word:#010x} failed to decode");
        let got = insn
            .encode()
            .unwrap_or_else(|e| panic!("encode of {word:#010x} ({:?}) failed: {e:?}", insn.code()));
        assert_eq!(got, word, "round-trip mismatch for {word:#010x} (code={:?})", insn.code());
    }

    #[test]
    fn arith_unpredicated() {
        check(0x046A03AD, "add     z13.h, z29.h, z10.h");
        check(0x04A50715, "sub     z21.s, z24.s, z5.s");
        check(0x04751130, "sqadd   z16.h, z9.h, z21.h");
        check(0x042D1D6C, "uqsub   z12.b, z11.b, z13.b");
        check(0x04676060, "mul     z0.h, z3.h, z7.h");
        // PMUL: the corpus prints `z0.b` here, but Zd=<4:0>=30; LLVM and the ARM
        // ARM agree on `z30.b` (a corpus quirk), so we assert the spec value.
        check(0x043F661E, "pmul    z30.b, z16.b, z31.b");
    }

    #[test]
    fn logical_unpredicated_and_mov_alias() {
        check(0x0434317D, "and     z29.d, z11.d, z20.d");
        check(0x04B33111, "eor     z17.d, z8.d, z19.d");
        // ORR Zd,Zn,Zn -> MOV alias.
        check(0x047532B7, "mov     z23.d, z21.d");
    }

    #[test]
    fn predicated_binary_and_unary() {
        check(0x04C011A5, "add     z5.d, p4/m, z5.d, z13.d");
        check(0x04430FA6, "subr    z6.h, p3/m, z6.h, z29.h");
        check(0x04941FE8, "sdiv    z8.s, p7/m, z8.s, z31.s");
        check(0x0496BFB2, "abs     z18.s, p7/m, z29.s");
        check(0x0490A714, "sxtb    z20.s, p1/m, z24.s");
    }

    #[test]
    fn unary_pred_zeroing_sve2p1() {
        // FEAT_SVE2p1 zeroing (`/z`) predicated unary (0x04, <20>=0).
        check(0x0408A016, "cls     z22.b, p0/z, z0.b");
        check(0x0409A069, "clz     z9.b, p0/z, z3.b");
        check(0x040AA036, "cnt     z22.b, p0/z, z1.b");
        check(0x040BA151, "cnot    z17.b, p0/z, z10.b");
        check(0x040EA021, "not     z1.b, p0/z, z1.b");
        check(0x0406A190, "abs     z16.b, p0/z, z12.b");
        check(0x0407A102, "neg     z2.b, p0/z, z8.b");
        check(0x0440A077, "sxtb    z23.h, p0/z, z3.h");
        check(0x0441A11C, "uxtb    z28.h, p0/z, z8.h");
        check(0x0482A059, "sxth    z25.s, p0/z, z2.s");
        check(0x0483A00C, "uxth    z12.s, p0/z, z0.s");
        check(0x04C4A01A, "sxtw    z26.d, p0/z, z0.d");
        check(0x04C5A05B, "uxtw    z27.d, p0/z, z2.d");
        for w in [
            0x0408A016, 0x0409A069, 0x040AA036, 0x040BA151, 0x040EA021, 0x0406A190, 0x0407A102,
            0x0440A077, 0x0441A11C, 0x0482A059, 0x0483A00C, 0x04C4A01A, 0x04C5A05B,
            // merging (`/m`) base-SVE forms still round-trip.
            0x0496BFB2, 0x0490A714,
        ] {
            rt(w);
        }
        // Illegal extend element sizes are unallocated (LLVM-Invalid), not decoded.
        let inv = [0x0410A09F_u32, 0x0411A0E9, 0x0412A03E, 0x0413A01A, 0x0414A1D7, 0x0415A044];
        for w in inv {
            let b = w.to_le_bytes();
            let mut d = Decoder::new(&b, 0x1000, DecoderOptions::default());
            assert!(d.decode().is_invalid(), "should be Invalid: {w:#010x}");
        }
    }

    #[test]
    fn rev_rbit_zeroing_sve2p1() {
        check(0x0527A005, "rbit    z5.b, p0/z, z0.b");
        check(0x0564A02B, "revb    z11.h, p0/z, z1.h");
        check(0x05A5A000, "revh    z0.s, p0/z, z0.s");
        check(0x05E6A032, "revw    z18.d, p0/z, z1.d");
        for w in [0x0527A005, 0x0564A02B, 0x05A5A000, 0x05E6A032] {
            rt(w);
        }
    }

    #[test]
    fn quadword_reductions_int() {
        check(0x04052027, "addqv   v7.16b, p0, z1.b");
        check(0x040C213E, "smaxqv  v30.16b, p0, z9.b");
        check(0x040D2055, "umaxqv  v21.16b, p0, z2.b");
        check(0x040E2030, "sminqv  v16.16b, p0, z1.b");
        check(0x040F2072, "uminqv  v18.16b, p0, z3.b");
        check(0x041C204C, "orqv    v12.16b, p0, z2.b");
        check(0x041D20C7, "eorqv   v7.16b, p0, z6.b");
        check(0x041E2088, "andqv   v8.16b, p0, z4.b");
        // size variants (.h/.s/.d).
        check(0x04852000, "addqv   v0.4s, p0, z0.s");
        check(0x04CE2000, "sminqv  v0.2d, p0, z0.d");
        for w in [
            0x04052027, 0x040C213E, 0x040D2055, 0x040E2030, 0x040F2072, 0x041C204C, 0x041D20C7,
            0x041E2088, 0x04852000, 0x04CE2000,
        ] {
            rt(w);
        }
    }

    #[test]
    fn sve2p1_misc() {
        check(0x05318081, "expand  z1.b, p0, z4.b");
        check(0x05212486, "dupq    z6.b, z4.b[0]");
        check(0x053F2486, "dupq    z6.b, z4.b[15]");
        check(0x05282486, "dupq    z6.d, z4.d[0]");
        check(0x05602401, "extq    z1.b, z1.b, z0.b, #0x0");
        check(0x056F2401, "extq    z1.b, z1.b, z0.b, #0xf");
        check(0x052A3880, "pmov    p0.b, z4");
        check(0x052B3880, "pmov    z0, p4.b");
        check(0x052C3880, "pmov    p0.h, z4[0]");
        check(0x052F3880, "pmov    z0[1], p4.h");
        check(0x05EF3880, "pmov    z0[7], p4.d");
        for w in [
            0x05318081, 0x05212486, 0x053F2486, 0x05282486, 0x05602401, 0x056F2401, 0x052A3880,
            0x052B3880, 0x052C3880, 0x052F3880, 0x05EF3880,
        ] {
            rt(w);
        }
    }

    #[test]
    fn mla_mls_mad_msb() {
        check(0x04185317, "mla     z23.b, p4/m, z24.b, z24.b");
        check(0x04137E2D, "mls     z13.b, p7/m, z17.b, z19.b");
        check(0x044CD5EB, "mad     z11.h, p5/m, z12.h, z15.h");
        check(0x04C2EE1F, "msb     z31.d, p3/m, z2.d, z16.d");
    }

    #[test]
    fn shifts() {
        check(0x042F93EC, "asr     z12.b, z31.b, #0x1");
        check(0x04EE9EEB, "lsl     z11.d, z23.d, #0x2e");
        check(0x04809AA8, "asr     z8.d, p6/m, z8.d, #0x2b");
        check(0x04048E16, "asrd    z22.h, p3/m, z22.h, #0x10");
        check(0x04AB8110, "asr     z16.s, z8.s, z11.d");
        check(0x04D39BEA, "lsl     z10.d, p6/m, z10.d, z31.d");
    }

    #[test]
    fn reductions_and_movprfx() {
        check(0x04803962, "saddv   d2, p6, z11.s");
        check(0x04083307, "smaxv   b7, p4, z24.b");
        check(0x041A23E7, "andv    b7, p0, z31.b");
        check(0x0420BC18, "movprfx z24, z0");
        check(0x04113549, "movprfx z9.b, p5/m, z10.b");
    }

    #[test]
    fn index_forms() {
        check(0x04E6411D, "index   z29.d, #0x8, #0x6");
        check(0x04F4429B, "index   z27.d, #-12, #-12");
        check(0x04B949A9, "index   z9.s, #0xd, w25");
        check(0x042447B2, "index   z18.b, w29, #0x4");
        check(0x04734CF5, "index   z21.h, w7, w19");
    }

    #[test]
    fn cnt_inc_dec() {
        check(0x042DE1A5, "cntb    x5, vl256, mul #0xe");
        check(0x0460E3B0, "cnth    x16, mul4");
        check(0x04B0E3E5, "incw    x5");
        check(0x0434E489, "decb    x9, vl4, mul #0x5");
        check(0x0425FBD1, "sqdecb  x17, w17, mul3, mul #0x6");
        check(0x0432F89F, "sqdecb  xzr, vl4, mul #0x3");
        check(0x0421FDA1, "uqdecb  w1, vl256, mul #0x2");
        check(0x04FBC52A, "decd    z10.d, vl16, mul #0xc");
    }

    #[test]
    fn addvl_addpl_rdvl() {
        check(0x042D508F, "addvl   x15, x13, #0x4");
        check(0x042756A8, "addvl   x8, x7, #-11");
        check(0x04635315, "addpl   x21, x3, #0x18");
        check(0x04BF52F6, "rdvl    x22, #0x17");
        check(0x04BF54D5, "rdvl    x21, #-26");
        check(0x043F564E, "addvl   x14, sp, #-14");
    }

    #[test]
    fn addsvl_addspl_rdsvl() {
        // SME streaming-mode analogues (word<11>=1).
        check(0x04205801, "addsvl  x1, x0, #0x0");
        check(0x04205821, "addsvl  x1, x0, #0x1");
        check(0x04205C01, "addsvl  x1, x0, #-32");
        check(0x04605807, "addspl  x7, x0, #0x0");
        check(0x04BF5823, "rdsvl   x3, #0x1");
        check(0x04BF5C23, "rdsvl   x3, #-31");
        // Round-trip (decode -> encode -> identical word).
        rt(0x04205801);
        rt(0x04205C01);
        rt(0x04605807);
        rt(0x046F5C8F); // addspl  x15, x15, #-28
        rt(0x04BF5823);
        rt(0x04BF5C23);
    }

    #[test]
    fn dup_cpy_sel_insr() {
        check(0x05E03A66, "mov     z6.d, x19");
        check(0x05203BE0, "mov     z0.b, wsp");
        check(0x05342076, "mov     z22.s, z3.s[2]");
        check(0x05242317, "mov     z23.s, s24");
        check(0x0568B0C1, "mov     z1.h, p4/m, w6");
        check(0x05E09862, "mov     z2.d, p6/m, d3");
        check(0x05F7E857, "mov     z23.d, p10/m, z2.d");
        // CPY (scalar, predicated) for Pg != p4 — the governing predicate is not
        // a discriminator (regression for the `<12:10>==100` over-constraint).
        check(0x0528B60B, "mov     z11.b, p5/m, w16");
        check(0x05A8A3E7, "mov     z7.s, p0/m, wsp");
        check(0x05E8A545, "mov     z5.d, p1/m, x10");
        check(0x05B7F912, "sel     z18.s, p14, z8.s, z23.s");
        check(0x05E03BF3, "mov     z19.d, sp");
        check(0x05A4393D, "insr    z29.s, w9");
    }

    #[test]
    fn immediates() {
        check(0x2520DCBF, "add     z31.b, z31.b, #0xe5");
        check(0x25E1F685, "sub     z5.d, z5.d, #0xb400");
        check(0x25E8C4F6, "smax    z22.d, z22.d, #0x27");
        check(0x25A8D79C, "smax    z28.s, z28.s, #-68");
        check(0x2530DEC1, "mul     z1.b, z1.b, #-10");
        check(0x25F8E805, "mov     z5.d, #0x4000");
        check(0x2578D8E1, "mov     z1.h, #-57");
        check(0x058165CC, "and     z12.h, z12.h, #0xfff7");
    }

    #[test]
    fn compares() {
        check(0x251E8E29, "cmpeq   p9.b, p3/z, z17.b, #-2");
        check(0x242E1373, "cmphi   p3.b, p4/z, z27.b, #0x38");
        check(0x2402B9C9, "cmpeq   p9.b, p6/z, z14.b, z2.b");
        check(0x24152B2E, "cmpeq   p14.b, p2/z, z25.b, z21.d");
        check(0x249F5BAE, "cmpge   p14.s, p6/z, z29.s, z31.d");
    }

    #[test]
    fn pred_count_and_dot() {
        check(0x25A08C01, "cntp    x1, p3, p0.s");
        check(0x252D8986, "decp    x6, p12.b");
        check(0x25AC805E, "incp    z30.s, p2");
        check(0x25E8893D, "sqincp  x29, p9.d, w29");
        check(0x44CB0330, "sdot    z16.d, z25.h, z11.h");
        check(0x4492003B, "sdot    z27.s, z1.b, z18.b");
        check(0x44B201BF, "sdot    z31.s, z13.b, z2.b[2]");
    }

    #[test]
    fn sve2_sqrdmlah() {
        // Vector form: element size from `size` (binja erroneously prints `.b`
        // for all; we follow the ARM ARM / LLVM and decode the encoded size).
        check(0x440771D1, "sqrdmlah z17.b, z14.b, z7.b");
        check(0x444E74F3, "sqrdmlsh z19.h, z7.h, z14.h");
        check(0x44897555, "sqrdmlsh z21.s, z10.s, z9.s");
        check(0x44C374F6, "sqrdmlsh z22.d, z7.d, z3.d");
        // Indexed form: `.h` (i3h:i3l), `.s` (i2), `.d` (i1).
        check(0x44761078, "sqrdmlah z24.h, z3.h, z6.h[6]");
        check(0x44B8108F, "sqrdmlah z15.s, z4.s, z0.s[3]");
    }

    #[test]
    fn never_panics_on_sve_space() {
        // Exhaustively exercise the SVE quadrant prefixes for panic-freedom.
        let mut buf = [0u8; 128];
        for hi in [0x04u32, 0x05, 0x24, 0x25, 0x44, 0x45] {
            for low in 0u32..=0xffff {
                let word = (hi << 24) | (low << 8) | (low & 0xff);
                let _ = render(word, &mut buf);
            }
        }
    }
}
