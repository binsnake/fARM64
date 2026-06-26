//! FEAT_MOPS — Memory Copy and Memory Set instructions.
//!
//! The Memory Copy (`CPYF*` forward-only, `CPY*` overlapping) and Memory Set
//! (`SET*`, `SETG*` tag-setting) families live in the Loads-and-Stores group and
//! are reached from [`crate::decode::ldst`] (they share `word<29:24> == 0b011001`
//! with the RCpc-unscaled `LDAPUR`/`STLUR` and the memory-tagging forms, and are
//! distinguished by `word<11:10> == 0b01`).
//!
//! Encoding (ARM ARM C4.1.66, "Memory Copy and Memory Set"):
//!
//! ```text
//!  31      27 26 25 24 23 21 20 16 15 12 11 10 9  5 4  0
//! [0 0 0 1 1][o][0  1][ op1 ][  Rn ][ op2 ][0  1][ Rs ][ Rd ]
//! ```
//!
//! * `o` = `word<26>` selects the family class: `0` = `CPYF*`/`SET*`,
//!   `1` = `CPY*`/`SETG*`.
//! * `op1` = `word<23:21>` selects copy stage (`0` prologue, `2` main,
//!   `4` epilogue) or, when `6`, the Set family (stage is then encoded in `op2`).
//! * `op2` = `word<15:12>` selects the read/write ordering and non-temporal hint
//!   variant (copy) or the stage + ordering hint (set).
//!
//! Operands (already SP/ZR-resolved; register `31` is `xzr`):
//!
//! * Copy: `<mn> [Xd]!, [Xs]!, Xn!` — `Rd`=dst, `Rn`=src, `Rs`=size.
//! * Set:  `<mn> [Xd]!, Xn!, Xm`    — `Rd`=dst, `Rs`=size (with `!`),
//!   `Rn`=value (no `!`).
//!
//! Constrained-UNPREDICTABLE encodings (which LLVM rejects, so we leave
//! [`Code::Invalid`]): `Rd == 31`, any two of `{Rd, Rs, Rn}` equal, and — for the
//! copy family only — `Rn == 31`.

use crate::decode::bits::{bit, bits};
use crate::instruction::Instruction;
use crate::mnemonic::Code;
use crate::operand::{MemIndexMode, Operand};
use crate::register::{gp_register, RegWidth};

/// `[Xn]!` — a writeback address operand with no displacement (`Rn != 31`
/// guaranteed by the caller; base is rendered as `Xn`, never SP).
#[inline]
fn mem_bang(n: u32) -> Operand {
    Operand::MemImm {
        base: gp_register(false, RegWidth::X64, (n & 0x1f) as u8),
        imm: 0,
        mode: MemIndexMode::PreNoOffset,
    }
}

/// `Xn!` — a 64-bit GP register with a writeback `!` suffix. Register 31 is the
/// zero register `xzr` (the MOPS size/count slot permits it).
#[inline]
fn reg_bang(n: u32) -> Operand {
    Operand::RegBang(gp_register(false, RegWidth::X64, (n & 0x1f) as u8))
}

/// A plain 64-bit GP register `Xn` (no writeback). Register 31 is `xzr`.
#[inline]
fn x_reg(n: u32) -> Operand {
    Operand::Reg {
        reg: gp_register(false, RegWidth::X64, (n & 0x1f) as u8),
        arr: None,
        lane: None,
        shift: None,
        extend: None,
        pred: None,
    }
}

/// Decode a Memory Copy / Memory Set (`FEAT_MOPS`) encoding.
///
/// Called from [`crate::decode::ldst`] for `word<29:24> == 0b011001`,
/// `word<21> == 0`, `word<11:10> == 0b01`. Leaves `out` invalid for any field
/// combination LLVM/the ARM ARM does not allocate or that violates the register
/// constraints.
#[inline]
pub fn decode(word: u32, out: &mut Instruction) {
    // Fixed opcode bits: `word<31:30> == 0b00`, `word<25:24> == 0b01`,
    // `word<11:10> == 0b01`. The size field `word<31:30>` is `0` for every MOPS
    // encoding (a non-zero value here is a different — unallocated — form), so
    // reject it up front to avoid over-decoding the `sz != 0` neighbours.
    if bits(word, 30, 2) != 0b00 || bits(word, 24, 2) != 0b01 || bits(word, 10, 2) != 0b01 {
        return;
    }

    let family = bit(word, 26); // 0 = CPYF/SET, 1 = CPY/SETG
    let op1 = bits(word, 21, 3);
    let op2 = bits(word, 12, 4);
    let rn = bits(word, 16, 5); // src (copy) / value (set)
    let rs = bits(word, 5, 5); //  size / count
    let rd = bits(word, 0, 5); //  dst

    let code = match (family, op1, op2) {
        (0, 0, 0) => Code::Cpyfp,
        (0, 0, 1) => Code::Cpyfpwt,
        (0, 0, 2) => Code::Cpyfprt,
        (0, 0, 3) => Code::Cpyfpt,
        (0, 0, 4) => Code::Cpyfpwn,
        (0, 0, 5) => Code::Cpyfpwtwn,
        (0, 0, 6) => Code::Cpyfprtwn,
        (0, 0, 7) => Code::Cpyfptwn,
        (0, 0, 8) => Code::Cpyfprn,
        (0, 0, 9) => Code::Cpyfpwtrn,
        (0, 0, 10) => Code::Cpyfprtrn,
        (0, 0, 11) => Code::Cpyfptrn,
        (0, 0, 12) => Code::Cpyfpn,
        (0, 0, 13) => Code::Cpyfpwtn,
        (0, 0, 14) => Code::Cpyfprtn,
        (0, 0, 15) => Code::Cpyfptn,
        (0, 2, 0) => Code::Cpyfm,
        (0, 2, 1) => Code::Cpyfmwt,
        (0, 2, 2) => Code::Cpyfmrt,
        (0, 2, 3) => Code::Cpyfmt,
        (0, 2, 4) => Code::Cpyfmwn,
        (0, 2, 5) => Code::Cpyfmwtwn,
        (0, 2, 6) => Code::Cpyfmrtwn,
        (0, 2, 7) => Code::Cpyfmtwn,
        (0, 2, 8) => Code::Cpyfmrn,
        (0, 2, 9) => Code::Cpyfmwtrn,
        (0, 2, 10) => Code::Cpyfmrtrn,
        (0, 2, 11) => Code::Cpyfmtrn,
        (0, 2, 12) => Code::Cpyfmn,
        (0, 2, 13) => Code::Cpyfmwtn,
        (0, 2, 14) => Code::Cpyfmrtn,
        (0, 2, 15) => Code::Cpyfmtn,
        (0, 4, 0) => Code::Cpyfe,
        (0, 4, 1) => Code::Cpyfewt,
        (0, 4, 2) => Code::Cpyfert,
        (0, 4, 3) => Code::Cpyfet,
        (0, 4, 4) => Code::Cpyfewn,
        (0, 4, 5) => Code::Cpyfewtwn,
        (0, 4, 6) => Code::Cpyfertwn,
        (0, 4, 7) => Code::Cpyfetwn,
        (0, 4, 8) => Code::Cpyfern,
        (0, 4, 9) => Code::Cpyfewtrn,
        (0, 4, 10) => Code::Cpyfertrn,
        (0, 4, 11) => Code::Cpyfetrn,
        (0, 4, 12) => Code::Cpyfen,
        (0, 4, 13) => Code::Cpyfewtn,
        (0, 4, 14) => Code::Cpyfertn,
        (0, 4, 15) => Code::Cpyfetn,
        (0, 6, 0) => Code::Setp,
        (0, 6, 1) => Code::Setpt,
        (0, 6, 2) => Code::Setpn,
        (0, 6, 3) => Code::Setptn,
        (0, 6, 4) => Code::Setm,
        (0, 6, 5) => Code::Setmt,
        (0, 6, 6) => Code::Setmn,
        (0, 6, 7) => Code::Setmtn,
        (0, 6, 8) => Code::Sete,
        (0, 6, 9) => Code::Setet,
        (0, 6, 10) => Code::Seten,
        (0, 6, 11) => Code::Setetn,
        (1, 0, 0) => Code::Cpyp,
        (1, 0, 1) => Code::Cpypwt,
        (1, 0, 2) => Code::Cpyprt,
        (1, 0, 3) => Code::Cpypt,
        (1, 0, 4) => Code::Cpypwn,
        (1, 0, 5) => Code::Cpypwtwn,
        (1, 0, 6) => Code::Cpyprtwn,
        (1, 0, 7) => Code::Cpyptwn,
        (1, 0, 8) => Code::Cpyprn,
        (1, 0, 9) => Code::Cpypwtrn,
        (1, 0, 10) => Code::Cpyprtrn,
        (1, 0, 11) => Code::Cpyptrn,
        (1, 0, 12) => Code::Cpypn,
        (1, 0, 13) => Code::Cpypwtn,
        (1, 0, 14) => Code::Cpyprtn,
        (1, 0, 15) => Code::Cpyptn,
        (1, 2, 0) => Code::Cpym,
        (1, 2, 1) => Code::Cpymwt,
        (1, 2, 2) => Code::Cpymrt,
        (1, 2, 3) => Code::Cpymt,
        (1, 2, 4) => Code::Cpymwn,
        (1, 2, 5) => Code::Cpymwtwn,
        (1, 2, 6) => Code::Cpymrtwn,
        (1, 2, 7) => Code::Cpymtwn,
        (1, 2, 8) => Code::Cpymrn,
        (1, 2, 9) => Code::Cpymwtrn,
        (1, 2, 10) => Code::Cpymrtrn,
        (1, 2, 11) => Code::Cpymtrn,
        (1, 2, 12) => Code::Cpymn,
        (1, 2, 13) => Code::Cpymwtn,
        (1, 2, 14) => Code::Cpymrtn,
        (1, 2, 15) => Code::Cpymtn,
        (1, 4, 0) => Code::Cpye,
        (1, 4, 1) => Code::Cpyewt,
        (1, 4, 2) => Code::Cpyert,
        (1, 4, 3) => Code::Cpyet,
        (1, 4, 4) => Code::Cpyewn,
        (1, 4, 5) => Code::Cpyewtwn,
        (1, 4, 6) => Code::Cpyertwn,
        (1, 4, 7) => Code::Cpyetwn,
        (1, 4, 8) => Code::Cpyern,
        (1, 4, 9) => Code::Cpyewtrn,
        (1, 4, 10) => Code::Cpyertrn,
        (1, 4, 11) => Code::Cpyetrn,
        (1, 4, 12) => Code::Cpyen,
        (1, 4, 13) => Code::Cpyewtn,
        (1, 4, 14) => Code::Cpyertn,
        (1, 4, 15) => Code::Cpyetn,
        (1, 6, 0) => Code::Setgp,
        (1, 6, 1) => Code::Setgpt,
        (1, 6, 2) => Code::Setgpn,
        (1, 6, 3) => Code::Setgptn,
        (1, 6, 4) => Code::Setgm,
        (1, 6, 5) => Code::Setgmt,
        (1, 6, 6) => Code::Setgmn,
        (1, 6, 7) => Code::Setgmtn,
        (1, 6, 8) => Code::Setge,
        (1, 6, 9) => Code::Setget,
        (1, 6, 10) => Code::Setgen,
        (1, 6, 11) => Code::Setgetn,
        _ => return,
    };

    // Register-distinctness constraints (Constrained UNPREDICTABLE otherwise):
    // dst, size and the src/value register must all differ, and the destination
    // register must not be `31`.
    if rd == rs || rd == rn || rs == rn || rd == 31 {
        return;
    }

    let is_set = op1 == 6;
    if is_set {
        // Set family: `[Xd]!, Xn!, Xm` (dst, size!, value). The value register
        // may be `xzr`; the size register may be `xzr`.
        out.set(code);
        out.push_operand(mem_bang(rd));
        out.push_operand(reg_bang(rs));
        out.push_operand(x_reg(rn));
    } else {
        // Copy family: `[Xd]!, [Xs]!, Xn!` (dst, src, size!). None of the three
        // may be `xzr` except the size register.
        if rn == 31 {
            return;
        }
        out.set(code);
        out.push_operand(mem_bang(rd));
        out.push_operand(mem_bang(rn));
        out.push_operand(reg_bang(rs));
    }
}

/// Decode a FEAT_MOPS memory-set-with-tag *option* form (`SETGO*`).
///
/// These share the SETG major (`word<26> == 1`, `op1 (word<23:21>) == 6`) but use
/// `word<11:10> == 0b00` instead of `0b01`: the value-source register is replaced
/// by an implementation-defined option, so only `[Xd]!, Xn!` are operands
/// (`Rd`=dst at `word<4:0>`, `Rn`=size at `word<9:5>`, both with writeback). The
/// value field `word<20:16>` must be `11111` (`xzr`); other values are
/// UNALLOCATED. The stage/hint is selected by `op2 (word<15:12>)`, exactly as the
/// SETG value-register forms.
///
/// Called from [`crate::decode::ldst`] for the matching signature. Leaves `out`
/// invalid for unallocated `op2`, a non-`xzr` value field, or a register-
/// distinctness violation (`Rd == 31`, `Rd == Rs`).
#[inline]
pub fn decode_setg_option(word: u32, out: &mut Instruction) {
    // The value-source field must be `xzr` (the option replaces it).
    if bits(word, 16, 5) != 0b11111 {
        return;
    }
    let op2 = bits(word, 12, 4);
    let rs = bits(word, 5, 5); // size / count
    let rd = bits(word, 0, 5); // dst
    let code = match op2 {
        0 => Code::SetgopMops,
        1 => Code::SetgoptMops,
        2 => Code::SetgopnMops,
        3 => Code::SetgoptnMops,
        4 => Code::SetgomMops,
        5 => Code::SetgomtMops,
        6 => Code::SetgomnMops,
        7 => Code::SetgomtnMops,
        8 => Code::SetgoeMops,
        9 => Code::SetgoetMops,
        10 => Code::SetgoenMops,
        11 => Code::SetgoetnMops,
        _ => return,
    };
    // Distinctness: dst must not be `31` and must differ from the size register.
    if rd == 31 || rd == rs {
        return;
    }
    out.set(code);
    out.push_operand(mem_bang(rd));
    out.push_operand(reg_bang(rs));
}
