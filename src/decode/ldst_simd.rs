//! Advanced SIMD load/store *structure* instructions (ARM ARM C4.1.96).
//!
//! These live in the Loads-and-Stores region (`op0 = word<28:25> = x1x0`) and are
//! dispatched here from [`crate::decode::ldst::decode`]. They are distinguished
//! from the rest of the load/store family by `word<29:23>`:
//!
//! * `0011000` / `0011001` — load/store **multiple** structures
//!   (`LD1`/`LD2`/`LD3`/`LD4` and `ST1`..`ST4`); the post-index form sets bit 23.
//! * `0011010` / `0011011` — load/store **single** structure
//!   (`LD1`..`LD4` single-lane, `ST1`..`ST4` single-lane) and the load-replicate
//!   forms (`LD1R`/`LD2R`/`LD3R`/`LD4R`); the post-index form sets bit 23.
//!
//! Three addressing modes are rendered exactly as Binary Ninja does:
//! `[Xn]` (no offset), `[Xn], #imm` (post-index by the structure's byte size,
//! selected by `Rm == 0b11111`), and `[Xn], Xm` (post-index by register).
//!
//! All forms here are base-ISA Advanced SIMD ([`Feature::Base`]); none are gated.

use crate::decode::bits::{bit, bits};
use crate::enums::VectorArrangement;
use crate::instruction::Instruction;
use crate::mnemonic::{Code, Mnemonic};
use crate::operand::{MemIndexMode, Operand};
use crate::register::{gp_register, RegWidth, Register};

// Numbered SIMD&FP `V` register table (the enum lays these out contiguously, but
// listing them avoids any discriminant arithmetic / transmute).
#[rustfmt::skip]
const V: [Register; 32] = [
    Register::V0, Register::V1, Register::V2, Register::V3, Register::V4, Register::V5, Register::V6, Register::V7,
    Register::V8, Register::V9, Register::V10, Register::V11, Register::V12, Register::V13, Register::V14, Register::V15,
    Register::V16, Register::V17, Register::V18, Register::V19, Register::V20, Register::V21, Register::V22, Register::V23,
    Register::V24, Register::V25, Register::V26, Register::V27, Register::V28, Register::V29, Register::V30, Register::V31,
];

/// Build a register-list operand of `count` consecutive `V` registers starting at
/// `first` (wrapping mod 32), all sharing `arr` and the optional `lane` index.
#[inline]
fn vlist(first: u32, count: u8, arr: VectorArrangement, lane: Option<u8>) -> Operand {
    let count = count.clamp(1, 4);
    let mut regs = [Register::None; 4];
    let mut i = 0u32;
    while i < count as u32 {
        regs[i as usize] = V[((first + i) & 0x1f) as usize];
        i += 1;
    }
    Operand::MultiReg {
        regs,
        count,
        arr: Some(arr),
        lane,
    }
}

/// `[Xn|SP]` — base only (the no-offset form). Base is SP-capable.
#[inline]
fn mem_base(rn: u32) -> Operand {
    Operand::MemImm {
        base: gp_register(true, RegWidth::X64, (rn & 0x1f) as u8),
        imm: 0,
        mode: MemIndexMode::Offset,
    }
}

/// `[Xn|SP], #imm` — post-index by the structure's transferred byte count.
#[inline]
fn mem_post_imm(rn: u32, imm: i64) -> Operand {
    Operand::MemImm {
        base: gp_register(true, RegWidth::X64, (rn & 0x1f) as u8),
        imm,
        mode: MemIndexMode::PostImm,
    }
}

/// `[Xn|SP], <Xm>` — post-index by register. The trailing `Xm` is pushed as a
/// separate GP operand after the (bracket-closing) `MemImm`/`PostReg` operand.
#[inline]
fn mem_post_reg(rn: u32) -> Operand {
    Operand::MemImm {
        base: gp_register(true, RegWidth::X64, (rn & 0x1f) as u8),
        imm: 0,
        mode: MemIndexMode::PostReg,
    }
}

/// A plain `Xm` GP register operand (used as the post-index register).
#[inline]
fn xm(rm: u32) -> Operand {
    Operand::Reg {
        reg: gp_register(false, RegWidth::X64, (rm & 0x1f) as u8),
        arr: None,
        lane: None,
        shift: None,
        extend: None,
        pred: None,
    }
}

/// The multiple-structures arrangement for `(size, Q)`.
#[inline]
fn arr_mult(size: u32, q: u32) -> VectorArrangement {
    match (size, q) {
        (0, 0) => VectorArrangement::V8B,
        (0, _) => VectorArrangement::V16B,
        (1, 0) => VectorArrangement::V4H,
        (1, _) => VectorArrangement::V8H,
        (2, 0) => VectorArrangement::V2S,
        (2, _) => VectorArrangement::V4S,
        (3, 0) => VectorArrangement::V1D,
        _ => VectorArrangement::V2D,
    }
}

/// Element width in bytes for an arrangement's element size (`8/16/32/64` bits).
#[inline]
fn elem_bytes(arr: VectorArrangement) -> i64 {
    (arr.element_bits() / 8) as i64
}

/// Append the addressing operand(s) for a structure load/store.
///
/// `post` selects the post-index form (bit 23); when post-indexing, `Rm` selects
/// the immediate form (`Rm == 0b11111`, byte count = `total_bytes`) or the
/// register form (`[Xn], Xm`). The pre-/non-indexed form renders as `[Xn]`.
#[inline]
fn push_addr(out: &mut Instruction, post: bool, rm: u32, rn: u32, total_bytes: i64) {
    if !post {
        out.push_operand(mem_base(rn));
    } else if rm == 0b11111 {
        out.push_operand(mem_post_imm(rn, total_bytes));
    } else {
        out.push_operand(mem_post_reg(rn));
        out.push_operand(xm(rm));
    }
}

// ---------------------------------------------------------------------------
// Entry point.
// ---------------------------------------------------------------------------

/// Decode an Advanced SIMD load/store *structure* instruction, or leave `out`
/// invalid for unallocated encodings. Called from [`crate::decode::ldst::decode`]
/// for `word<29:24> == 0b001100` / `0b001101` (with `word<31>==0`, `word<23>` the
/// post-index selector).
#[inline]
pub(super) fn decode(word: u32, out: &mut Instruction) {
    // Shared layout: 0 Q 0011 0 {0:mult,1:single} {idx} L ... size Rn Rt.
    // word<31> must be 0; word<23> is the post-index bit (handled per sub-form).
    if bit(word, 31) != 0 {
        return;
    }
    // word<29:24> == 0b001100 -> multiple, 0b001101 -> single/replicate.
    match bits(word, 24, 6) {
        0b001100 => decode_multiple(word, out),
        0b001101 => decode_single(word, out),
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Load/store multiple structures.
// ---------------------------------------------------------------------------

/// `LD1`/`LD2`/`LD3`/`LD4` and `ST1`..`ST4` (multiple structures).
///
/// Encoding: `0 Q 0011000 L 0 00000 opcode<3:0> size<1:0> Rn Rt` (no offset) and
/// `0 Q 0011001 L 0 Rm opcode size Rn Rt` (post-index). `opcode<3:0>` selects
/// both the structure (`LD1`..`LD4`) and the register count.
#[inline]
fn decode_multiple(word: u32, out: &mut Instruction) {
    let q = bit(word, 30);
    let post = bit(word, 23) == 1;
    let l = bit(word, 22);
    let rm = bits(word, 16, 5);
    let opcode = bits(word, 12, 4);
    let size = bits(word, 10, 2);
    let rn = bits(word, 5, 5);
    let rt = bits(word, 0, 5);

    // For the non-post-indexed form, bits<20:16> (the `Rm` slot) must be zero.
    if !post && rm != 0 {
        return;
    }

    // opcode<3:0> -> (Code carrier, mnemonic, register count). The structure
    // count comes straight from the ARM ARM C4.1.96 "opcode" column.
    let load = l == 1;
    let (code, mnem, nregs) = match opcode {
        0b0000 => mk_mult(load, 4, Mnemonic::Ld4, Mnemonic::St4, Code::Ld4Multiple, Code::St4Multiple),
        0b0010 => mk_mult(load, 4, Mnemonic::Ld1, Mnemonic::St1, Code::Ld1Multiple, Code::St1Multiple),
        0b0100 => mk_mult(load, 3, Mnemonic::Ld3, Mnemonic::St3, Code::Ld3Multiple, Code::St3Multiple),
        0b0110 => mk_mult(load, 3, Mnemonic::Ld1, Mnemonic::St1, Code::Ld1Multiple, Code::St1Multiple),
        0b0111 => mk_mult(load, 1, Mnemonic::Ld1, Mnemonic::St1, Code::Ld1Multiple, Code::St1Multiple),
        0b1000 => mk_mult(load, 2, Mnemonic::Ld2, Mnemonic::St2, Code::Ld2Multiple, Code::St2Multiple),
        0b1010 => mk_mult(load, 2, Mnemonic::Ld1, Mnemonic::St1, Code::Ld1Multiple, Code::St1Multiple),
        _ => return, // unallocated opcode
    };

    let arr = arr_mult(size, q);
    // Transferred bytes = registers * (16 if Q else 8).
    let reg_bytes: i64 = if q == 1 { 16 } else { 8 };
    let total = nregs as i64 * reg_bytes;

    out.set(code);
    out.set_mnemonic(mnem);
    out.push_operand(vlist(rt, nregs, arr, None));
    push_addr(out, post, rm, rn, total);
}

/// Helper selecting the `(Code, Mnemonic, nregs)` triple for a multiple-structures
/// form by load/store direction.
#[inline]
fn mk_mult(
    load: bool,
    nregs: u8,
    ld_mn: Mnemonic,
    st_mn: Mnemonic,
    ld_code: Code,
    st_code: Code,
) -> (Code, Mnemonic, u8) {
    if load {
        (ld_code, ld_mn, nregs)
    } else {
        (st_code, st_mn, nregs)
    }
}

// ---------------------------------------------------------------------------
// Load/store single structure (and load-replicate).
// ---------------------------------------------------------------------------

/// `LD1`..`LD4` / `ST1`..`ST4` (single structure) and `LD1R`..`LD4R` (replicate).
///
/// Encoding: `0 Q 0011010 L R 00000 opcode<2:0> S size<1:0> Rn Rt` (no offset)
/// and `0 Q 0011011 L R Rm opcode S size Rn Rt` (post-index). `opcode<0>` and
/// `R` give the register count (`nregs = (opcode<0> << 1) + R + 1`); `opcode<2:1>`
/// selects the element size (`00`=B, `01`=H, `10`=S/D by `size<0>`, `11`=replicate).
#[inline]
fn decode_single(word: u32, out: &mut Instruction) {
    let q = bit(word, 30);
    let post = bit(word, 23) == 1;
    let l = bit(word, 22);
    let r = bit(word, 21);
    let rm = bits(word, 16, 5);
    let opcode = bits(word, 13, 3);
    let s = bit(word, 12);
    let size = bits(word, 10, 2);
    let rn = bits(word, 5, 5);
    let rt = bits(word, 0, 5);

    let nregs = (((opcode & 1) << 1) + r + 1) as u8;
    let scale = opcode >> 1; // opcode<2:1>
    let load = l == 1;

    if scale == 0b11 {
        // Replicate: LD1R/LD2R/LD3R/LD4R (load only; store is unallocated).
        if !load {
            return;
        }
        decode_replicate(word, post, rm, rn, rt, size, q, nregs, out);
        return;
    }

    // Single structure. Element size and lane index per ARM ARM C4.1.96.
    // `index = Q:S:size` truncated by the element scale; the unused low `size`
    // bits act as encoding checks (must be zero) for the wider elements.
    let (arr, index, ebytes) = match scale {
        0b00 => (VectorArrangement::V8B, (q << 3) | (s << 2) | size, 1i64),
        0b01 => {
            // H: size<0> must be 0.
            if size & 1 != 0 {
                return;
            }
            (VectorArrangement::V4H, (q << 2) | (s << 1) | (size >> 1), 2)
        }
        _ => {
            // scale == 0b10: S when size<0>==0, D when size<0>==1.
            if size & 1 == 0 {
                (VectorArrangement::V2S, (q << 1) | s, 4)
            } else {
                // D: S must be 0.
                if s != 0 {
                    return;
                }
                (VectorArrangement::V1D, q, 8)
            }
        }
    };

    let (code, mnem) = single_code(load, nregs);
    let total = nregs as i64 * ebytes;

    out.set(code);
    out.set_mnemonic(mnem);
    // The list uses the truncated element-size suffix (`.b`/`.h`/`.s`/`.d`); the
    // formatter selects that automatically when a lane index is present.
    out.push_operand(vlist(rt, nregs, arr, Some(index as u8)));
    push_addr(out, post, rm, rn, total);
}

/// Replicate form (`LD1R`/`LD2R`/`LD3R`/`LD4R`): the full arrangement from
/// `(size, Q)`, no lane index; transferred bytes = `nregs * element_bytes`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn decode_replicate(
    _word: u32,
    post: bool,
    rm: u32,
    rn: u32,
    rt: u32,
    size: u32,
    q: u32,
    nregs: u8,
    out: &mut Instruction,
) {
    let arr = arr_mult(size, q);
    let (code, mnem) = rep_code(nregs);
    let total = nregs as i64 * elem_bytes(arr);

    out.set(code);
    out.set_mnemonic(mnem);
    out.push_operand(vlist(rt, nregs, arr, None));
    push_addr(out, post, rm, rn, total);
}

/// `(Code, Mnemonic)` for a single-structure form by direction and register count.
#[inline]
fn single_code(load: bool, nregs: u8) -> (Code, Mnemonic) {
    match (load, nregs) {
        (true, 1) => (Code::Ld1Single, Mnemonic::Ld1),
        (true, 2) => (Code::Ld2Single, Mnemonic::Ld2),
        (true, 3) => (Code::Ld3Single, Mnemonic::Ld3),
        (true, _) => (Code::Ld4Single, Mnemonic::Ld4),
        (false, 1) => (Code::St1Single, Mnemonic::St1),
        (false, 2) => (Code::St2Single, Mnemonic::St2),
        (false, 3) => (Code::St3Single, Mnemonic::St3),
        (false, _) => (Code::St4Single, Mnemonic::St4),
    }
}

/// `(Code, Mnemonic)` for a replicate form by register count.
#[inline]
fn rep_code(nregs: u8) -> (Code, Mnemonic) {
    match nregs {
        1 => (Code::Ld1SingleRep, Mnemonic::Ld1r),
        2 => (Code::Ld2Rep, Mnemonic::Ld2r),
        3 => (Code::Ld3Rep, Mnemonic::Ld3r),
        _ => (Code::Ld4Rep, Mnemonic::Ld4r),
    }
}
