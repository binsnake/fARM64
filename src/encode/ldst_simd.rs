//! Encoder for Advanced SIMD load/store *structure* instructions — the exact
//! inverse of [`crate::decode::ldst_simd`].
//!
//! Covers the `LD1`/`LD2`/`LD3`/`LD4` and `ST1`..`ST4` multiple-structure forms,
//! the single-structure (single-lane) forms, and the `LD1R`..`LD4R` replicate
//! loads. Recovers the register list, arrangement, lane index and the three
//! addressing modes (`[Xn]`, `[Xn], #bytes`, `[Xn], Xm`) back into the raw
//! fields. Reconstructs the word purely from semantics — never reads
//! [`Instruction::word`].

use crate::encode::EncodeError;
use crate::enums::VectorArrangement;
use crate::instruction::Instruction;
use crate::mnemonic::Code;
use crate::operand::{MemIndexMode, Operand};

type R = Result<u32, EncodeError>;

/// `true` for every [`Code`] produced by [`crate::decode::ldst_simd`] — the
/// Advanced SIMD load/store structure forms the encoder handles here.
#[inline]
pub fn is_ldst_simd(code: Code) -> bool {
    use Code::*;
    matches!(
        code,
        Ld1Multiple
            | Ld2Multiple
            | Ld3Multiple
            | Ld4Multiple
            | St1Multiple
            | St2Multiple
            | St3Multiple
            | St4Multiple
            | Ld1Single
            | Ld2Single
            | Ld3Single
            | Ld4Single
            | St1Single
            | St2Single
            | St3Single
            | St4Single
            | Ld1SingleRep
            | Ld2Rep
            | Ld3Rep
            | Ld4Rep
    )
}

/// Encode an Advanced SIMD load/store structure instruction.
#[inline]
pub fn encode(insn: &Instruction) -> R {
    use Code::*;
    match insn.code() {
        Ld1Multiple | Ld2Multiple | Ld3Multiple | Ld4Multiple | St1Multiple | St2Multiple
        | St3Multiple | St4Multiple => enc_multiple(insn),
        Ld1Single | Ld2Single | Ld3Single | Ld4Single | St1Single | St2Single | St3Single
        | St4Single => enc_single(insn),
        Ld1SingleRep | Ld2Rep | Ld3Rep | Ld4Rep => enc_replicate(insn),
        _ => Err(EncodeError::Unsupported),
    }
}

// ---------------------------------------------------------------------------
// Register-list / addressing helpers.
// ---------------------------------------------------------------------------

/// Unpack a [`Operand::MultiReg`] into `(first_reg, count, arr, lane)`.
fn multireg(
    insn: &Instruction,
    n: usize,
) -> Result<(u32, u8, VectorArrangement, Option<u8>), EncodeError> {
    match insn.op(n) {
        Operand::MultiReg {
            regs,
            count,
            arr,
            lane,
        } => {
            let first = regs[0].number() as u32;
            let arr = arr.ok_or(EncodeError::InvalidOperand)?;
            Ok((first, count, arr, lane))
        }
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// Recover the addressing fields `(post, rm, rn)` from the memory operand(s).
///
/// * `[Xn]` (offset, imm 0)          -> `post=false`, `rm=0`.
/// * `[Xn], #bytes` (post-imm)       -> `post=true`,  `rm=0b11111`.
/// * `[Xn], Xm` (post-reg)           -> `post=true`,  `rm` from the trailing reg.
///
/// `expected_bytes` is the transferred byte count the post-imm form must carry.
fn recover_addr(
    insn: &Instruction,
    mem_idx: usize,
    expected_bytes: i64,
) -> Result<(bool, u32, u32), EncodeError> {
    match insn.op(mem_idx) {
        Operand::MemImm { base, imm, mode } => {
            let rn = base.number() as u32;
            match mode {
                MemIndexMode::Offset => {
                    if imm != 0 {
                        return Err(EncodeError::InvalidImmediate);
                    }
                    Ok((false, 0, rn))
                }
                MemIndexMode::PostImm => {
                    if imm != expected_bytes {
                        return Err(EncodeError::InvalidImmediate);
                    }
                    Ok((true, 0b11111, rn))
                }
                MemIndexMode::PostReg => {
                    // The trailing Xm register is the next operand.
                    let rm = match insn.op(mem_idx + 1) {
                        Operand::Reg { reg, .. } => reg.number() as u32,
                        _ => return Err(EncodeError::InvalidOperand),
                    };
                    // Xm == 0b11111 (SP/ZR) would alias the post-imm form; the
                    // post-reg encoding uses a real Xm != 31.
                    if rm == 0b11111 {
                        return Err(EncodeError::InvalidOperand);
                    }
                    Ok((true, rm, rn))
                }
                MemIndexMode::PreIndex | MemIndexMode::PreNoOffset => {
                    Err(EncodeError::InvalidOperand)
                }
            }
        }
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// The `(size, Q)` for a multiple-structures / replicate arrangement.
fn size_q_of(arr: VectorArrangement) -> Option<(u32, u32)> {
    use VectorArrangement as VA;
    Some(match arr {
        VA::V8B => (0, 0),
        VA::V16B => (0, 1),
        VA::V4H => (1, 0),
        VA::V8H => (1, 1),
        VA::V2S => (2, 0),
        VA::V4S => (2, 1),
        VA::V1D => (3, 0),
        VA::V2D => (3, 1),
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Load/store multiple structures.
// ---------------------------------------------------------------------------

/// `(opcode<3:0>, nregs)` for a multiple-structures form, keyed on `(Code,
/// list-length)`. The `LD1`/`ST1` family has three opcodes (1/2/3/4 regs map to
/// 0b0111/0b1010/0b0110/0b0010); the others have a single opcode.
fn multiple_opcode(code: Code, count: u8) -> Option<u32> {
    use Code::*;
    Some(match (code, count) {
        (Ld4Multiple, 4) | (St4Multiple, 4) => 0b0000,
        (Ld3Multiple, 3) | (St3Multiple, 3) => 0b0100,
        (Ld2Multiple, 2) | (St2Multiple, 2) => 0b1000,
        // LD1/ST1 by register count.
        (Ld1Multiple, 1) | (St1Multiple, 1) => 0b0111,
        (Ld1Multiple, 2) | (St1Multiple, 2) => 0b1010,
        (Ld1Multiple, 3) | (St1Multiple, 3) => 0b0110,
        (Ld1Multiple, 4) | (St1Multiple, 4) => 0b0010,
        _ => return None,
    })
}

/// `true` if `code` is a store (`ST*`) structure form.
fn is_store_multiple(code: Code) -> bool {
    use Code::*;
    matches!(code, St1Multiple | St2Multiple | St3Multiple | St4Multiple)
}

/// `LD1`..`LD4` / `ST1`..`ST4` (multiple structures). Encoding (post-index form):
/// `0 Q 0011001 L 0 Rm opcode<3:0> size<1:0> Rn Rt`; no-offset form clears bit23
/// and has `Rm == 0`.
fn enc_multiple(insn: &Instruction) -> R {
    let code = insn.code();
    let (first, count, arr, lane) = multireg(insn, 0)?;
    if lane.is_some() {
        return Err(EncodeError::InvalidOperand);
    }
    let opcode = multiple_opcode(code, count).ok_or(EncodeError::InvalidOperand)?;
    let (size, q) = size_q_of(arr).ok_or(EncodeError::InvalidOperand)?;
    let l = u32::from(!is_store_multiple(code));

    let reg_bytes: i64 = if q == 1 { 16 } else { 8 };
    let total = count as i64 * reg_bytes;
    let (post, rm, rn) = recover_addr(insn, 1, total)?;

    let word = (q << 30)
        | (0b0011 << 26)
        | ((post as u32) << 23)
        | (l << 22)
        | (rm << 16)
        | (opcode << 12)
        | (size << 10)
        | (rn << 5)
        | first;
    Ok(word)
}

// ---------------------------------------------------------------------------
// Load/store single structure.
// ---------------------------------------------------------------------------

/// `true` if `code` is a single-structure store.
fn is_store_single(code: Code) -> bool {
    use Code::*;
    matches!(code, St1Single | St2Single | St3Single | St4Single)
}

/// The register count for a single-structure code.
fn single_count(code: Code) -> u8 {
    use Code::*;
    match code {
        Ld1Single | St1Single => 1,
        Ld2Single | St2Single => 2,
        Ld3Single | St3Single => 3,
        _ => 4,
    }
}

/// `LD1`..`LD4` / `ST1`..`ST4` (single structure). Encoding (post-index form):
/// `0 Q 0011011 L R Rm opcode<2:0> S size<1:0> Rn Rt`.
///
/// `nregs = (opcode<0> << 1) + R + 1`; `opcode<2:1>` is the element-size scale
/// (`00`=B, `01`=H, `10`=S/D); the lane index packs into `Q:S:size` per element.
fn enc_single(insn: &Instruction) -> R {
    let code = insn.code();
    let (first, count, arr, lane) = multireg(insn, 0)?;
    let lane = lane.ok_or(EncodeError::InvalidOperand)? as u32;
    if count != single_count(code) {
        return Err(EncodeError::InvalidOperand);
    }
    let nregs = count as u32;
    let l = u32::from(!is_store_single(code));

    // nregs = (opcode<0> << 1) + R + 1  ->  (R, opcode<0>) from nregs-1.
    let nm1 = nregs - 1; // 0..3
    let r = nm1 & 1;
    let opc0 = (nm1 >> 1) & 1;

    // Element scale (opcode<2:1>) + lane packing from the arrangement.
    let ebytes_log2 = match arr {
        VectorArrangement::V8B => 0u32,
        VectorArrangement::V4H => 1,
        VectorArrangement::V2S => 2,
        VectorArrangement::V1D => 3,
        _ => return Err(EncodeError::InvalidOperand),
    };
    let (scale, q, s, size) = match ebytes_log2 {
        0 => {
            // B: index = Q:S:size (4 bits over a 16-byte register Q==1).
            if lane > 15 {
                return Err(EncodeError::InvalidImmediate);
            }
            let q = (lane >> 3) & 1;
            let s = (lane >> 2) & 1;
            let size = lane & 0b11;
            (0b00u32, q, s, size)
        }
        1 => {
            // H: index = Q:S:size<1>; size<0> == 0.
            if lane > 7 {
                return Err(EncodeError::InvalidImmediate);
            }
            let q = (lane >> 2) & 1;
            let s = (lane >> 1) & 1;
            let size_hi = lane & 1;
            (0b01u32, q, s, size_hi << 1)
        }
        2 => {
            // S: index = Q:S; size == 00.
            if lane > 3 {
                return Err(EncodeError::InvalidImmediate);
            }
            let q = (lane >> 1) & 1;
            let s = lane & 1;
            (0b10u32, q, s, 0b00)
        }
        _ => {
            // D: index = Q; S == 0; size == 01.
            if lane > 1 {
                return Err(EncodeError::InvalidImmediate);
            }
            let q = lane & 1;
            (0b10u32, q, 0, 0b01)
        }
    };

    let opcode = (scale << 1) | opc0;
    let ebytes = 1i64 << ebytes_log2;
    let total = nregs as i64 * ebytes;
    let (post, rm, rn) = recover_addr(insn, 1, total)?;

    let word = (q << 30)
        | (0b0011 << 26)
        | (0b01 << 24)
        | ((post as u32) << 23)
        | (l << 22)
        | (r << 21)
        | (rm << 16)
        | (opcode << 13)
        | (s << 12)
        | (size << 10)
        | (rn << 5)
        | first;
    Ok(word)
}

// ---------------------------------------------------------------------------
// Load replicate (LD1R/LD2R/LD3R/LD4R).
// ---------------------------------------------------------------------------

/// The register count for a replicate code.
fn rep_count(code: Code) -> u8 {
    use Code::*;
    match code {
        Ld1SingleRep => 1,
        Ld2Rep => 2,
        Ld3Rep => 3,
        _ => 4,
    }
}

/// `LD1R`..`LD4R` (replicate). Encoding (post-index form):
/// `0 Q 0011011 L=1 R Rm opcode<2:0>=11x S=0 size<1:0> Rn Rt`, scale `opcode<2:1>
/// == 0b11`. `nregs = (opcode<0> << 1) + R + 1`.
fn enc_replicate(insn: &Instruction) -> R {
    let code = insn.code();
    let (first, count, arr, lane) = multireg(insn, 0)?;
    if lane.is_some() {
        return Err(EncodeError::InvalidOperand);
    }
    if count != rep_count(code) {
        return Err(EncodeError::InvalidOperand);
    }
    let nregs = count as u32;
    let (size, q) = size_q_of(arr).ok_or(EncodeError::InvalidOperand)?;

    let nm1 = nregs - 1;
    let r = nm1 & 1;
    let opc0 = (nm1 >> 1) & 1;
    let opcode = (0b11 << 1) | opc0; // scale == 0b11
    let s = 0u32;

    let ebytes = (arr.element_bits() / 8) as i64;
    let total = nregs as i64 * ebytes;
    let (post, rm, rn) = recover_addr(insn, 1, total)?;

    let word = (q << 30)
        | (0b0011 << 26)
        | (0b01 << 24)
        | ((post as u32) << 23)
        | (1 << 22) // L == 1 (load only)
        | (r << 21)
        | (rm << 16)
        | (opcode << 13)
        | (s << 12)
        | (size << 10)
        | (rn << 5)
        | first;
    Ok(word)
}

#[cfg(test)]
mod tests {
    use crate::features::FeatureSet;
    use crate::instruction::Instruction;

    #[track_caller]
    fn rt(word: u32) {
        let mut insn = Instruction::default();
        crate::decode::decode_into(word, 0x1000, FeatureSet::ALL, &mut insn);
        assert!(!insn.is_invalid(), "word {word:#010x} failed to decode");
        let got = insn
            .encode()
            .unwrap_or_else(|e| panic!("encode of {word:#010x} ({:?}) failed: {e:?}", insn.code()));
        assert_eq!(
            got, word,
            "round-trip mismatch for {word:#010x}: re-encoded {got:#010x} (code={:?}, mnem={:?})",
            insn.code(),
            insn.mnemonic()
        );
    }

    #[test]
    fn simd_ldst_known_words() {
        // Multiple structures.
        rt(0x0C407FEE); // ld1 {v14.1d}, [sp]
        rt(0x4C4073AC); // ld1 {v12.16b}, [x29]
        rt(0x4CDF70DD); // ld1 {v29.16b}, [x6], #0x10
        rt(0x4CDFA129); // ld1 {v9.16b, v10.16b}, [x9], #0x20
        rt(0x0CDF22DE); // ld1 {v30.8b, v31.8b, v0.8b, v1.8b}, [x22], #0x20
        rt(0x4C000080); // st4 {v0.16b..v3.16b}, [x4]
        // Single structure.
        rt(0x4D9F1CBC); // st1 {v28.b}[15], [x5], #0x1
        rt(0x4DCD8712); // ld1 {v18.d}[1], [x24], x13
        rt(0x0DD7A0B2); // ld3 {v18.s, v19.s, v20.s}[0], [x5], x23
        rt(0x0D80A4C5); // st3 {v5.d, v6.d, v7.d}[0], [x6], x0
        rt(0x0DBFB139); // st4 {v25.s..v28.s}[1], [x9], #0x10
        // Replicate.
        rt(0x0DDFC4A5); // ld1r {v5.4h}, [x5], #0x2
        rt(0x0DFFE507); // ld4r {v7.4h..v10.4h}, [x8], #0x8
        rt(0x4DF8EED7); // ld4r {v23.2d..v26.2d}, [x22], x24
    }
}
