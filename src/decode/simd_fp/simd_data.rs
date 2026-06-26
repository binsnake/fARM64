//! Advanced SIMD data-movement encodings (ARM ARM C4.1.97).
//!
//! Owns the copy / permute / extract / table-lookup / modified-immediate and
//! shift-by-immediate leaves of the Data-Processing SIMD&FP group. Pure, total
//! and panic-free for every 32-bit input; unallocated encodings leave `out` as
//! the invalid default ([`crate::mnemonic::Code::Invalid`]).
//!
//! Sub-classification follows the ARM ARM C4.1.97 table (the bit layouts are
//! reproduced inline in each decoder). Preferred-disassembly aliases match the
//! Binary Ninja differential corpus: `MOV` for INS(element)/INS(general) and for
//! the `.S`/`.D` UMOV forms, and `SXTL`/`UXTL` for the `SSHLL`/`USHLL #0` forms.

use crate::decode::bits::{adv_simd_expand_imm, bit, bits, vfp_expand_imm};
use crate::enums::VectorArrangement;
use crate::features::{Feature, FeatureSet};
use crate::instruction::Instruction;
use crate::mnemonic::{Code, Mnemonic};
use crate::operand::Operand;
use crate::register::{gp_register, Register, RegWidth};

// ---------------------------------------------------------------------------
// Register-bank tables (contiguous discriminants, mirror `crate::register`).
// ---------------------------------------------------------------------------

const V_BANK: [Register; 32] = [
    Register::V0, Register::V1, Register::V2, Register::V3, Register::V4, Register::V5, Register::V6, Register::V7,
    Register::V8, Register::V9, Register::V10, Register::V11, Register::V12, Register::V13, Register::V14, Register::V15,
    Register::V16, Register::V17, Register::V18, Register::V19, Register::V20, Register::V21, Register::V22, Register::V23,
    Register::V24, Register::V25, Register::V26, Register::V27, Register::V28, Register::V29, Register::V30, Register::V31,
];
const B_BANK: [Register; 32] = [
    Register::B0, Register::B1, Register::B2, Register::B3, Register::B4, Register::B5, Register::B6, Register::B7,
    Register::B8, Register::B9, Register::B10, Register::B11, Register::B12, Register::B13, Register::B14, Register::B15,
    Register::B16, Register::B17, Register::B18, Register::B19, Register::B20, Register::B21, Register::B22, Register::B23,
    Register::B24, Register::B25, Register::B26, Register::B27, Register::B28, Register::B29, Register::B30, Register::B31,
];
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
const Q_BANK: [Register; 32] = [
    Register::Q0, Register::Q1, Register::Q2, Register::Q3, Register::Q4, Register::Q5, Register::Q6, Register::Q7,
    Register::Q8, Register::Q9, Register::Q10, Register::Q11, Register::Q12, Register::Q13, Register::Q14, Register::Q15,
    Register::Q16, Register::Q17, Register::Q18, Register::Q19, Register::Q20, Register::Q21, Register::Q22, Register::Q23,
    Register::Q24, Register::Q25, Register::Q26, Register::Q27, Register::Q28, Register::Q29, Register::Q30, Register::Q31,
];

// ---------------------------------------------------------------------------
// Operand builders.
// ---------------------------------------------------------------------------

/// A `V{n}` register carrying a full arrangement (`v3.4s`).
#[inline]
fn vreg(n: u32, arr: VectorArrangement) -> Operand {
    Operand::Reg {
        reg: V_BANK[(n & 0x1f) as usize],
        arr: Some(arr),
        lane: None,
        shift: None,
        extend: None,
        pred: None,
    }
}

/// A `V{n}` register carrying an indexed element (`v3.s[2]`).
#[inline]
fn vreg_lane(n: u32, arr: VectorArrangement, lane: u8) -> Operand {
    Operand::Reg {
        reg: V_BANK[(n & 0x1f) as usize],
        arr: Some(arr),
        lane: Some(lane),
        shift: None,
        extend: None,
        pred: None,
    }
}

/// A scalar FP register of element size `esize` (8/16/32/64/128 bits).
#[inline]
fn scalar_reg(n: u32, esize: u32) -> Operand {
    let n = (n & 0x1f) as usize;
    let reg = match esize {
        8 => B_BANK[n],
        16 => H_BANK[n],
        32 => S_BANK[n],
        64 => D_BANK[n],
        _ => Q_BANK[n],
    };
    plain(reg)
}

/// A plain register operand (no decorations).
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

/// A plain GP register operand of width `w`.
#[inline]
fn gpr(w: RegWidth, n: u32) -> Operand {
    plain(gp_register(false, w, (n & 0x1f) as u8))
}

/// Build a list operand of `count` consecutive `V{n}` registers, all `.16b`
/// (the table-lookup register-list arrangement).
#[inline]
fn vlist_16b(first: u32, count: u8) -> Operand {
    let mut regs = [Register::None; 4];
    let count = count.clamp(1, 4);
    let mut i = 0u32;
    while i < count as u32 {
        regs[i as usize] = V_BANK[((first + i) & 0x1f) as usize];
        i += 1;
    }
    Operand::MultiReg {
        regs,
        count,
        arr: Some(VectorArrangement::V16B),
        lane: None,
    }
}

/// A single-register `V{n}` list `{v{n}.<T>}` (the LUTI single-table form).
#[inline]
fn vlist1(first: u32, arr: VectorArrangement) -> Operand {
    Operand::MultiReg {
        regs: [
            V_BANK[(first & 0x1f) as usize],
            Register::None,
            Register::None,
            Register::None,
        ],
        count: 1,
        arr: Some(arr),
        lane: None,
    }
}

/// A two-register `V{n}` list `{v{n}.<T>, v{n+1}.<T>}` (consecutive, wrapping).
#[inline]
fn vlist2(first: u32, arr: VectorArrangement) -> Operand {
    Operand::MultiReg {
        regs: [
            V_BANK[(first & 0x1f) as usize],
            V_BANK[((first + 1) & 0x1f) as usize],
            Register::None,
            Register::None,
        ],
        count: 2,
        arr: Some(arr),
        lane: None,
    }
}

/// A `V{m}` vector-element selector `v{m}[index]` (no arrangement suffix; the
/// LUTI table index operand).
#[inline]
fn vidx(m: u32, index: u32) -> Operand {
    Operand::Reg {
        reg: V_BANK[(m & 0x1f) as usize],
        arr: None,
        lane: Some(index as u8),
        shift: None,
        extend: None,
        pred: None,
    }
}

// ---------------------------------------------------------------------------
// Top-level dispatch.
// ---------------------------------------------------------------------------

/// Decode an Advanced-SIMD data-movement instruction into `out`.
#[inline]
pub fn decode(word: u32, ip: u64, features: FeatureSet, out: &mut Instruction) {
    let _ = ip;

    // Every Advanced-SIMD data-movement encoding has word<31> == 0. Words with
    // word<31> == 1 in this region are the 4-register crypto forms (`EOR3`,
    // `SM3SS1`, ...) and similar — owned elsewhere — so reject them up front to
    // avoid a false copy/permute/table decode.
    if bit(word, 31) == 1 {
        return;
    }

    // word<28:24> == 01110 covers copy/permute/ext/tbl (three-register-ish data
    // movement); word<31>==0 && word<27:23> patterns cover imm/shift.
    let b28_24 = bits(word, 24, 5);

    // Shift-by-immediate (vector `asimdshf` word<28:23>==011110, and scalar
    // `asisdshf` word<28:23>==111110) and modified-immediate (vector,
    // word<28:19>==0111100000) share word<27:23>==11110. The immh field
    // (word<22:19>) is zero only for modified-immediate; non-zero selects a
    // shift-by-immediate. word<10> must be 1.
    if bits(word, 23, 5) == 0b11110 && bit(word, 10) == 1 {
        let immh = bits(word, 19, 4);
        if immh == 0 {
            // Modified immediate is a vector-only form (word<28>==0).
            if bit(word, 28) == 0 {
                decode_modified_immediate(word, features, out);
            }
            return;
        }
        decode_shift_by_immediate(word, features, out);
        return;
    }

    // Advanced SIMD scalar copy (`asisdone`): `01 op 1 11110000 imm5 0 imm4 1 Rn
    // Rd`, i.e. word<31:30>==01, word<28:21>==11110000, word<15>==0, word<10>==1.
    // The only allocated form is DUP (element) scalar (op==0, imm4==0000), which
    // binja/LLVM render via the `MOV <V><d>, <Vn>.<Ts>[index]` alias.
    if bits(word, 30, 2) == 0b01
        && bits(word, 21, 8) == 0b11110000
        && bit(word, 15) == 0
        && bit(word, 10) == 1
    {
        if bit(word, 29) == 0 && bits(word, 11, 4) == 0 {
            decode_dup_element_scalar(word, out);
        }
        return;
    }

    // Copy / permute / extract / table-lookup: word<28:24> == 01110, word<15>==0.
    // (The op bit word<29> is part of the copy/ext encoding — do NOT branch on it
    // alone; classify by the fixed signature bits instead.)
    if b28_24 == 0b01110 && bit(word, 15) == 0 {
        // Copy (asimdins): word<23:21>==000, word<10>==1. `op` is word<29>.
        if bit(word, 10) == 1 {
            if bits(word, 21, 3) == 0b000 {
                decode_copy(word, out);
            }
            return;
        }
        // word<10>==0 below.
        // EXT (asimdext): word<29>==1, word<23:21>==000.
        if bit(word, 29) == 1 {
            if bits(word, 21, 3) == 0b000 {
                decode_ext(word, out);
            }
            return;
        }
        // word<29>==0 below: table (word<11:10>==00) or permute (word<11:10>==10).
        match bits(word, 10, 2) {
            // Table (asimdtbl): word<23:21>==000.
            0b00 if bits(word, 21, 3) == 0b000 => decode_table(word, out),
            // LUTI2/LUTI4 (FEAT_LUT, NEON): word<11:10>==00, word<21>==0 with a
            // non-zero size field (size==00 is the TBL/TBX form matched above).
            0b00 if bit(word, 21) == 0 => decode_luti_neon(word, features, out),
            // Permute (asimdperm): word<21>==0 (size in word<23:22>).
            0b10 if bit(word, 21) == 0 => decode_permute(word, out),
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// LUTI2 / LUTI4 (FEAT_LUT, Advanced SIMD / NEON).
// ---------------------------------------------------------------------------

/// Decode the Advanced SIMD (`V`-register) `LUTI2`/`LUTI4` lookup-table reads
/// (FEAT_LUT) into `out`. Reached from the copy/permute region for
/// `word<11:10>==00`, `word<21>==0` with a non-zero `size` (`word<23:22>`); the
/// destination is always 128-bit (`Q==1`).
///
/// The `size` field selects the family and element size; the table index is
/// spread across `word<14:12>` (its width growing with the element size). Cross-
/// checked against `llvm-mc --mattr=+all`:
///
/// | `size` | `<12>` | form | index bits (msb..lsb) |
/// |-|-|-|-|
/// | `01` | `1` | `LUTI4 .8h` two-table | `<14>:<13>`      |
/// | `01` | `0` | `LUTI4 .16b` (needs `<13>==1`) | `<14>` |
/// | `10` | `1` | `LUTI2 .16b`          | `<14>:<13>`      |
/// | `11` | -   | `LUTI2 .8h`           | `<14>:<13>:<12>` |
#[inline]
fn decode_luti_neon(word: u32, features: FeatureSet, out: &mut Instruction) {
    if !features.has(Feature::Lut) {
        return;
    }
    // 128-bit destination only.
    if bit(word, 30) != 1 {
        return;
    }
    let b14 = bit(word, 14);
    let b13 = bit(word, 13);
    let b12 = bit(word, 12);
    let rm = bits(word, 16, 5);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);

    match bits(word, 22, 2) {
        0b01 => {
            if b12 == 1 {
                // LUTI4 .8h, two-register table; 2-bit index.
                let index = (b14 << 1) | b13;
                out.set(Code::Luti4TwoVec);
                out.set_mnemonic(Mnemonic::Luti4);
                out.push_operand(vreg(rd, VectorArrangement::V8H));
                out.push_operand(vlist2(rn, VectorArrangement::V8H));
                out.push_operand(vidx(rm, index));
            } else if b13 == 1 {
                // LUTI4 .16b, single-register table; 1-bit index.
                out.set(Code::Luti4Vec);
                out.set_mnemonic(Mnemonic::Luti4);
                out.push_operand(vreg(rd, VectorArrangement::V16B));
                out.push_operand(vlist1(rn, VectorArrangement::V16B));
                out.push_operand(vidx(rm, b14));
            }
        }
        0b10 => {
            if b12 == 1 {
                // LUTI2 .16b, single-register table; 2-bit index.
                let index = (b14 << 1) | b13;
                out.set(Code::Luti2Vec);
                out.set_mnemonic(Mnemonic::Luti2);
                out.push_operand(vreg(rd, VectorArrangement::V16B));
                out.push_operand(vlist1(rn, VectorArrangement::V16B));
                out.push_operand(vidx(rm, index));
            }
        }
        0b11 => {
            // LUTI2 .8h, single-register table; 3-bit index.
            let index = (b14 << 2) | (b13 << 1) | b12;
            out.set(Code::Luti2Vec);
            out.set_mnemonic(Mnemonic::Luti2);
            out.push_operand(vreg(rd, VectorArrangement::V8H));
            out.push_operand(vlist1(rn, VectorArrangement::V8H));
            out.push_operand(vidx(rm, index));
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Copy: DUP(element/general), INS(element/general), SMOV, UMOV.
// ---------------------------------------------------------------------------

/// Advanced SIMD copy (`asimdins`):
/// `0 Q 0 01110000 imm5 0 imm4 1 Rn Rd`, with `op = word<29>` (here 0 for all
/// but INS(element) which is `Q1` with `op==1`). `imm5` selects the element size
/// and index; `imm4` selects the operation.
#[inline]
fn decode_copy(word: u32, out: &mut Instruction) {
    let q = bit(word, 30);
    let op = bit(word, 29);
    let imm5 = bits(word, 16, 5);
    let imm4 = bits(word, 11, 4);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);

    // `op == 1` is INS (element) — the only `op==1` copy form (requires Q==1).
    if op == 1 {
        decode_ins_element(q, imm5, imm4, rn, rd, out);
        return;
    }

    match imm4 {
        0b0000 => decode_dup_element(q, imm5, rn, rd, out),
        0b0001 => decode_dup_general(q, imm5, rn, rd, out),
        0b0011 => decode_ins_general(q, imm5, rn, rd, out),
        0b0101 => decode_smov(q, imm5, rn, rd, out),
        0b0111 => decode_umov(q, imm5, rn, rd, out),
        _ => {}
    }
}

/// Decode the `imm5` element-size selector into `(esize_bits, index)`.
///
/// The lowest set bit of `imm5` selects the element size; the bits above it form
/// the index. Returns `None` for the reserved all-zero-low encoding.
#[inline]
fn imm5_size_index(imm5: u32) -> Option<(u32, u8)> {
    if imm5 & 0b00001 != 0 {
        Some((8, (imm5 >> 1) as u8 & 0b1111))
    } else if imm5 & 0b00010 != 0 {
        Some((16, (imm5 >> 2) as u8 & 0b111))
    } else if imm5 & 0b00100 != 0 {
        Some((32, (imm5 >> 3) as u8 & 0b11))
    } else if imm5 & 0b01000 != 0 {
        Some((64, (imm5 >> 4) as u8 & 0b1))
    } else {
        None
    }
}

/// Arrangement for a vector destination of element size `esize` and `Q` bit.
#[inline]
fn vec_arr(esize: u32, q: u32) -> Option<VectorArrangement> {
    Some(match (esize, q) {
        (8, 0) => VectorArrangement::V8B,
        (8, _) => VectorArrangement::V16B,
        (16, 0) => VectorArrangement::V4H,
        (16, _) => VectorArrangement::V8H,
        (32, 0) => VectorArrangement::V2S,
        (32, _) => VectorArrangement::V4S,
        (64, 0) => return None, // .1d is not a valid SIMD data arrangement here
        (64, _) => VectorArrangement::V2D,
        _ => return None,
    })
}

/// The element-size arrangement (`.b`/`.h`/`.s`/`.d`) for an indexed element.
#[inline]
fn elem_arr(esize: u32) -> VectorArrangement {
    match esize {
        8 => VectorArrangement::V16B,
        16 => VectorArrangement::V8H,
        32 => VectorArrangement::V4S,
        _ => VectorArrangement::V2D,
    }
}

/// `DUP <Vd>.<T>, <Vn>.<Ts>[index]` (element).
#[inline]
fn decode_dup_element(q: u32, imm5: u32, rn: u32, rd: u32, out: &mut Instruction) {
    let (esize, index) = match imm5_size_index(imm5) {
        Some(v) => v,
        None => return,
    };
    let arr = match vec_arr(esize, q) {
        Some(a) => a,
        None => return,
    };
    out.set(Code::DupElement);
    out.push_operand(vreg(rd, arr));
    out.push_operand(vreg_lane(rn, elem_arr(esize), index));
}

/// `DUP <V><d>, <Vn>.<Ts>[index]` (scalar). Rendered as the `MOV` alias to match
/// the corpus (`mov h24, v13.h[0]`). The scalar destination element size and the
/// source element index both derive from `imm5` (lowest set bit = size).
#[inline]
fn decode_dup_element_scalar(word: u32, out: &mut Instruction) {
    let imm5 = bits(word, 16, 5);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);
    let (esize, index) = match imm5_size_index(imm5) {
        Some(v) => v,
        None => return,
    };
    out.set(Code::DupElementScalar);
    out.set_mnemonic(Mnemonic::Mov);
    out.push_operand(scalar_reg(rd, esize));
    out.push_operand(vreg_lane(rn, elem_arr(esize), index));
}

/// `DUP <Vd>.<T>, <R><n>` (general register).
#[inline]
fn decode_dup_general(q: u32, imm5: u32, rn: u32, rd: u32, out: &mut Instruction) {
    let (esize, _index) = match imm5_size_index(imm5) {
        Some(v) => v,
        None => return,
    };
    let arr = match vec_arr(esize, q) {
        Some(a) => a,
        None => return,
    };
    // The source GP width is X for the 64-bit element, W otherwise.
    let w = if esize == 64 { RegWidth::X64 } else { RegWidth::W32 };
    out.set(Code::DupGeneral);
    out.push_operand(vreg(rd, arr));
    out.push_operand(gpr(w, rn));
}

/// `INS <Vd>.<Ts>[index], <R><n>` (general register). Rendered as the `MOV`
/// alias to match the corpus.
#[inline]
fn decode_ins_general(_q: u32, imm5: u32, rn: u32, rd: u32, out: &mut Instruction) {
    let (esize, index) = match imm5_size_index(imm5) {
        Some(v) => v,
        None => return,
    };
    let w = if esize == 64 { RegWidth::X64 } else { RegWidth::W32 };
    out.set(Code::InsGeneral);
    out.set_mnemonic(Mnemonic::Mov);
    out.push_operand(vreg_lane(rd, elem_arr(esize), index));
    out.push_operand(gpr(w, rn));
}

/// `INS <Vd>.<Ts>[index1], <Vn>.<Ts>[index2]` (element). Rendered as `MOV`.
#[inline]
fn decode_ins_element(_q: u32, imm5: u32, imm4: u32, rn: u32, rd: u32, out: &mut Instruction) {
    let (esize, dst_index) = match imm5_size_index(imm5) {
        Some(v) => v,
        None => return,
    };
    // The source index `index2` is `imm4` shifted right by log2(esize/8).
    let shift = match esize {
        8 => 0,
        16 => 1,
        32 => 2,
        _ => 3,
    };
    let src_index = (imm4 >> shift) as u8;
    let arr = elem_arr(esize);
    out.set(Code::InsElement);
    out.set_mnemonic(Mnemonic::Mov);
    out.push_operand(vreg_lane(rd, arr, dst_index));
    out.push_operand(vreg_lane(rn, arr, src_index));
}

/// `SMOV <Wd|Xd>, <Vn>.<Ts>[index]`. The GP width is `Q` (X for `Q==1`).
#[inline]
fn decode_smov(q: u32, imm5: u32, rn: u32, rd: u32, out: &mut Instruction) {
    // SMOV sizes: imm5 low bit set -> B, next -> H, next -> S (only for Q==1).
    // The destination width is X when Q==1, W when Q==0; S source requires Q==1.
    let (esize, index) = match imm5_size_index(imm5) {
        Some(v) => v,
        None => return,
    };
    // Valid: B/H for both widths; S only for the X (Q==1) form. 64-bit element is
    // not valid for SMOV.
    let w = if q == 1 { RegWidth::X64 } else { RegWidth::W32 };
    match (esize, q) {
        (8, _) | (16, _) => {}
        (32, 1) => {}
        _ => return,
    }
    out.set(Code::Smov);
    out.push_operand(gpr(w, rd));
    out.push_operand(vreg_lane(rn, elem_arr(esize), index));
}

/// `UMOV <Wd|Xd>, <Vn>.<Ts>[index]`. The `.S`/`.D` forms render as the `MOV`
/// alias (matching the corpus); `.B`/`.H` keep `UMOV`.
#[inline]
fn decode_umov(q: u32, imm5: u32, rn: u32, rd: u32, out: &mut Instruction) {
    let (esize, index) = match imm5_size_index(imm5) {
        Some(v) => v,
        None => return,
    };
    // Valid: B/H/S for the W (Q==0) form; only D (with Q==1) for the X form.
    let w = if esize == 64 { RegWidth::X64 } else { RegWidth::W32 };
    match (esize, q) {
        (8, 0) | (16, 0) | (32, 0) => {}
        (64, 1) => {}
        _ => return,
    }
    out.set(Code::Umov);
    // The full-width element transfers (`.S` for W, `.D` for X) are spelled `mov`.
    if esize == 32 || esize == 64 {
        out.set_mnemonic(Mnemonic::Mov);
    }
    out.push_operand(gpr(w, rd));
    out.push_operand(vreg_lane(rn, elem_arr(esize), index));
}

// ---------------------------------------------------------------------------
// Permute: ZIP1/ZIP2/UZP1/UZP2/TRN1/TRN2.
// ---------------------------------------------------------------------------

/// Advanced SIMD permute (`asimdperm`):
/// `0 Q 001110 size 0 Rm 0 opcode 10 Rn Rd`. `opcode` selects the operation;
/// `size`+`Q` select the arrangement.
#[inline]
fn decode_permute(word: u32, out: &mut Instruction) {
    let q = bit(word, 30);
    let size = bits(word, 22, 2);
    let rm = bits(word, 16, 5);
    let opcode = bits(word, 12, 3);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);

    let code = match opcode {
        0b001 => Code::Uzp1,
        0b010 => Code::Trn1,
        0b011 => Code::Zip1,
        0b101 => Code::Uzp2,
        0b110 => Code::Trn2,
        0b111 => Code::Zip2,
        _ => return,
    };
    let arr = match three_same_arr(size, q) {
        Some(a) => a,
        None => return,
    };
    out.set(code);
    out.push_operand(vreg(rd, arr));
    out.push_operand(vreg(rn, arr));
    out.push_operand(vreg(rm, arr));
}

/// Arrangement for a `size:Q` three-same vector form. `size==11` with `Q==0`
/// (`.1d`) is reserved.
#[inline]
fn three_same_arr(size: u32, q: u32) -> Option<VectorArrangement> {
    Some(match (size, q) {
        (0b00, 0) => VectorArrangement::V8B,
        (0b00, _) => VectorArrangement::V16B,
        (0b01, 0) => VectorArrangement::V4H,
        (0b01, _) => VectorArrangement::V8H,
        (0b10, 0) => VectorArrangement::V2S,
        (0b10, _) => VectorArrangement::V4S,
        (0b11, 0) => return None,
        (0b11, _) => VectorArrangement::V2D,
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Extract: EXT.
// ---------------------------------------------------------------------------

/// Advanced SIMD extract (`asimdext`):
/// `0 Q 101110 00 0 Rm 0 imm4 0 Rn Rd`. The arrangement is `.16b` (`Q==1`) or
/// `.8b` (`Q==0`); for `.8b` only `imm4<2:0>` is valid (`imm4<3>` must be 0).
#[inline]
fn decode_ext(word: u32, out: &mut Instruction) {
    let q = bit(word, 30);
    let rm = bits(word, 16, 5);
    let imm4 = bits(word, 11, 4);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);

    // For the 64-bit (`.8b`) form, imm4<3> must be zero.
    if q == 0 && bit(imm4, 3) == 1 {
        return;
    }
    let arr = if q == 1 {
        VectorArrangement::V16B
    } else {
        VectorArrangement::V8B
    };
    out.set(Code::Ext);
    out.push_operand(vreg(rd, arr));
    out.push_operand(vreg(rn, arr));
    out.push_operand(vreg(rm, arr));
    out.push_operand(Operand::ImmUnsigned(imm4 as u64));
}

// ---------------------------------------------------------------------------
// Table lookup: TBL/TBX.
// ---------------------------------------------------------------------------

/// Advanced SIMD table lookup (`asimdtbl`):
/// `0 Q 001110 000 Rm 0 len op 00 Rn Rd`. `len` (`word<14:13>`) gives the
/// register-list length minus one; `op` (`word<12>`) selects TBX.
#[inline]
fn decode_table(word: u32, out: &mut Instruction) {
    let q = bit(word, 30);
    let rm = bits(word, 16, 5);
    let len = bits(word, 13, 2);
    let op = bit(word, 12);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);

    let arr = if q == 1 {
        VectorArrangement::V16B
    } else {
        VectorArrangement::V8B
    };
    out.set(if op == 1 { Code::Tbx } else { Code::Tbl });
    out.push_operand(vreg(rd, arr));
    out.push_operand(vlist_16b(rn, (len + 1) as u8));
    out.push_operand(vreg(rm, arr));
}

// ---------------------------------------------------------------------------
// Modified immediate: MOVI/MVNI/ORR/BIC/FMOV (vector).
// ---------------------------------------------------------------------------

/// Advanced SIMD modified immediate (`asimdimm`):
/// `0 Q op 0111100000 abc cmode o2 1 defgh Rn Rd`. `imm8 = abc:defgh`;
/// `(op, cmode)` select the operation and the rendered immediate form.
#[inline]
fn decode_modified_immediate(word: u32, features: FeatureSet, out: &mut Instruction) {
    let q = bit(word, 30);
    let op = bit(word, 29);
    let cmode = bits(word, 12, 4);
    let o2 = bit(word, 11);
    let rd = bits(word, 0, 5);
    let abc = bits(word, 16, 3);
    let defgh = bits(word, 5, 5);
    let imm8 = (abc << 5) | defgh;

    // `o2` must be 0 for every allocated modified-immediate encoding (the single
    // exception is FMOV.half, cmode==1111 op==0 o2==1, handled below).
    let cmode_hi = bits(cmode, 1, 3); // cmode<3:1>
    let cmode0 = cmode & 1;

    // cmode<3:1> == 111x region: byte / 64-bit / FMOV forms.
    if cmode == 0b1110 {
        // op==0: MOVI byte (per-lane imm8). op==1: MOVI 64-bit (Dd / .2d).
        if o2 != 0 {
            return;
        }
        if op == 0 {
            // MOVI <Vd>.<8B|16B>, #imm8.
            let arr = if q == 1 {
                VectorArrangement::V16B
            } else {
                VectorArrangement::V8B
            };
            out.set(Code::MoviVector);
            out.push_operand(vreg(rd, arr));
            out.push_operand(Operand::ImmUnsigned(imm8 as u64));
            return;
        }
        // op==1: 64-bit per-element MOVI. Q==0 -> scalar Dd; Q==1 -> Vd.2D.
        let val = adv_simd_expand_imm(op, cmode, imm8);
        if q == 0 {
            out.set(Code::MoviScalarD);
            out.push_operand(scalar_reg(rd, 64));
            out.push_operand(Operand::ImmUnsigned(val));
        } else {
            out.set(Code::MoviVec2D);
            out.push_operand(vreg(rd, VectorArrangement::V2D));
            out.push_operand(Operand::ImmUnsigned(val));
        }
        return;
    }

    if cmode == 0b1111 {
        // FMOV (vector, immediate). op==0,o2==0 -> single; op==0,o2==1 -> half
        // (FEAT_FP16); op==1,o2==0 -> .2D double; op==1,o2==1 -> UNALLOCATED.
        if op == 0 && o2 == 0 {
            let arr = if q == 1 {
                VectorArrangement::V4S
            } else {
                VectorArrangement::V2S
            };
            let f = f32::from_bits(vfp_expand_imm(imm8, 32) as u32);
            out.set(Code::FmovVecImmS);
            out.push_operand(vreg(rd, arr));
            out.push_operand(Operand::FpImm(f));
            return;
        }
        if op == 0 && o2 == 1 {
            if !features.has(Feature::Fp16) {
                return;
            }
            let arr = if q == 1 {
                VectorArrangement::V8H
            } else {
                VectorArrangement::V4H
            };
            let f = crate::decode::simd_fp::scalar_fp::f16_bits_to_f32(vfp_expand_imm(imm8, 16) as u16);
            out.set(Code::FmovVecImmH);
            out.push_operand(vreg(rd, arr));
            out.push_operand(Operand::FpImm(f));
            return;
        }
        if op == 1 && o2 == 0 {
            // FMOV <Vd>.2D, #imm (Q must be 1).
            if q == 0 {
                return;
            }
            let f = f64::from_bits(vfp_expand_imm(imm8, 64)) as f32;
            out.set(Code::FmovVecImmD2);
            out.push_operand(vreg(rd, VectorArrangement::V2D));
            out.push_operand(Operand::FpImm(f));
            return;
        }
        return;
    }

    // From here `o2` is part of `cmode` and must structurally be the data form.
    if o2 != 0 {
        return;
    }

    // cmode<3:1> == 110: MOVI/MVNI with the MSL (shift-ones) form (32-bit). The
    // MSL amount is cmode<0>==0 -> #8, cmode<0>==1 -> #16 (always shown).
    if cmode_hi == 0b110 {
        let amt: u8 = if cmode0 == 0 { 8 } else { 16 };
        let arr = if q == 1 {
            VectorArrangement::V4S
        } else {
            VectorArrangement::V2S
        };
        let code = if op == 1 { Code::MvniVector } else { Code::MoviVector };
        out.set(code);
        out.push_operand(vreg(rd, arr));
        out.push_operand(Operand::ImmShiftedMsl {
            imm: imm8 as u16,
            msl: amt,
        });
        return;
    }

    // cmode<3:2> == 10: 16-bit family. cmode<1> is the shift bit (LSL #0/#8);
    // cmode<0> selects MOVI/MVNI (0) vs ORR/BIC (1).
    if (cmode >> 2) == 0b10 {
        let amt: u8 = if bit(cmode, 1) == 0 { 0 } else { 8 };
        let arr = if q == 1 {
            VectorArrangement::V8H
        } else {
            VectorArrangement::V4H
        };
        emit_movi_or_logical(cmode0, op, arr, rd, imm8, amt, out);
        return;
    }

    // cmode<3> == 0 (cmode `0xxx`): 32-bit family. cmode<2:1> selects the LSL
    // amount (0/8/16/24); cmode<0> selects MOVI/MVNI (0) vs ORR/BIC (1). The
    // `10xx`/`110x`/`111x` cmodes were already consumed above, so reaching here
    // with cmode<3>==0 is unambiguous.
    if bit(cmode, 3) == 0 {
        let amt: u8 = (((cmode >> 1) & 0b11) * 8) as u8;
        let arr = if q == 1 {
            VectorArrangement::V4S
        } else {
            VectorArrangement::V2S
        };
        emit_movi_or_logical(cmode0, op, arr, rd, imm8, amt, out);
    }
}

/// Emit a MOVI/MVNI (`cmode0 == 0`) or ORR/BIC (`cmode0 == 1`) vector immediate
/// `Vd.<T>, #imm8{, lsl #amt}`. `op` (`word<29>`) selects the inverted variant
/// (MVNI / BIC); the `, lsl #amt` is elided when `amt == 0`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn emit_movi_or_logical(
    cmode0: u32,
    op: u32,
    arr: VectorArrangement,
    rd: u32,
    imm8: u32,
    amt: u8,
    out: &mut Instruction,
) {
    let code = if cmode0 == 0 {
        if op == 1 { Code::MvniVector } else { Code::MoviVector }
    } else if op == 1 {
        Code::BicVecImm
    } else {
        Code::OrrVecImm
    };
    out.set(code);
    out.push_operand(vreg(rd, arr));
    out.push_operand(Operand::ImmShiftedMove {
        imm: imm8 as u16,
        lsl: amt,
    });
}

// ---------------------------------------------------------------------------
// Shift by immediate (vector + scalar).
// ---------------------------------------------------------------------------

/// Advanced SIMD shift by immediate (`asimdshf`/`asisdshf`):
/// `0 Q U 011110 immh immb opcode 1 Rn Rd`. `immh:immb` encode `(esize, shift)`;
/// `U`+`opcode` select the operation. `Q` selects vector width (scalar forms use
/// the scalar register view, `word<23:22>` reserved-checked by immh).
#[inline]
fn decode_shift_by_immediate(word: u32, features: FeatureSet, out: &mut Instruction) {
    let _ = features;
    // Scalar (`asisdshf`) vs vector (`asimdshf`): the two share `word<28:23> ==
    // 011110`, differing only in `word<28>` (vector 0 / scalar 1). The scalar
    // form has top byte `0x5f`/`0x7f` (word<31>==0, word<30>==1, word<28:24>
    // == 11111); the vector form has `word<28:24> == 01110` with `Q` in <30>.
    if bit(word, 28) == 1 {
        decode_shift_scalar(word, out);
    } else {
        decode_shift_vector(word, out);
    }
}

/// Decode `immh:immb` for the *left*-shift forms (SHL/SLI/SQSHL/SSHLL/...):
/// `shift = (immh:immb) - esize`, with `esize` chosen by the highest set bit of
/// `immh`. Returns `(esize_bits, shift)` or `None` if `immh == 0`.
#[inline]
fn left_shift_size(immh: u32, immb: u32) -> Option<(u32, u32)> {
    let val = (immh << 3) | immb; // 7-bit immh:immb
    let esize = match immh {
        0b0001 => 8,
        0b0010..=0b0011 => 16,
        0b0100..=0b0111 => 32,
        0b1000..=0b1111 => 64,
        _ => return None,
    };
    Some((esize, val - esize))
}

/// Decode `immh:immb` for the *right*-shift forms (SSHR/USHR/SRSHR/...):
/// `shift = 2*esize - (immh:immb)`.
#[inline]
fn right_shift_size(immh: u32, immb: u32) -> Option<(u32, u32)> {
    let val = (immh << 3) | immb;
    let esize = match immh {
        0b0001 => 8,
        0b0010..=0b0011 => 16,
        0b0100..=0b0111 => 32,
        0b1000..=0b1111 => 64,
        _ => return None,
    };
    Some((esize, 2 * esize - val))
}

/// Decode `immh:immb` for the *narrowing* forms (SHRN/SQSHRN/...): the source is
/// twice the destination element size; `shift = 2*esize - (immh:immb)` where
/// `esize` is the *destination* element size selected by immh (immh<3> must be 0,
/// else UNALLOCATED). Returns `(dst_esize_bits, shift)`.
#[inline]
fn narrow_shift_size(immh: u32, immb: u32) -> Option<(u32, u32)> {
    if immh == 0 || bit(immh, 3) == 1 {
        return None;
    }
    let val = (immh << 3) | immb;
    let esize = match immh {
        0b0001 => 8,
        0b0010..=0b0011 => 16,
        0b0100..=0b0111 => 32,
        _ => return None,
    };
    Some((esize, 2 * esize - val))
}

/// Vector shift-by-immediate. The `opcode` (`word<15:11>`) table follows the
/// ARM ARM C4.1.97 `asimdshf` rows; `U` (`word<29>`) selects the signed/unsigned
/// or insert variant.
#[inline]
fn decode_shift_vector(word: u32, out: &mut Instruction) {
    let q = bit(word, 30);
    let u = bit(word, 29);
    let immh = bits(word, 19, 4);
    let immb = bits(word, 16, 3);
    let opcode = bits(word, 11, 5);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);

    match opcode {
        // --- Right-shift, same size in/out. ---
        0b00000 => right_shift(if u == 0 { Code::SshrVec } else { Code::UshrVec }, q, immh, immb, rn, rd, out),
        0b00010 => right_shift(if u == 0 { Code::SsraVec } else { Code::UsraVec }, q, immh, immb, rn, rd, out),
        0b00100 => right_shift(if u == 0 { Code::SrshrVec } else { Code::UrshrVec }, q, immh, immb, rn, rd, out),
        0b00110 => right_shift(if u == 0 { Code::SrsraVec } else { Code::UrsraVec }, q, immh, immb, rn, rd, out),
        0b01000 if u == 1 => right_shift(Code::SriVec, q, immh, immb, rn, rd, out),
        // --- Left-shift, same size in/out. ---
        0b01010 => left_shift(if u == 0 { Code::ShlVec } else { Code::SliVec }, q, immh, immb, rn, rd, out),
        0b01100 if u == 1 => left_shift(Code::SqshluImmVec, q, immh, immb, rn, rd, out),
        0b01110 => left_shift(if u == 0 { Code::SqshlImmVec } else { Code::UqshlImmVec }, q, immh, immb, rn, rd, out),
        // --- Narrowing shift-right. ---
        0b10000..=0b10011 => decode_narrow(q, u, opcode, immh, immb, rn, rd, out),
        // --- Shift-left long (+ SXTL/UXTL aliases when shift==0). ---
        0b10100 => decode_shll(q, u, immh, immb, rn, rd, out),
        // --- Fixed-point convert. ---
        0b11100 => fixed_cvt(if u == 0 { Code::ScvtfFixedVec } else { Code::UcvtfFixedVec }, q, immh, immb, rn, rd, out),
        0b11111 => fixed_cvt(if u == 0 { Code::FcvtzsFixedVec } else { Code::FcvtzuFixedVec }, q, immh, immb, rn, rd, out),
        _ => {}
    }
}

/// A right-shift same-size vector form (`Vd.T, Vn.T, #shift`).
#[inline]
fn right_shift(code: Code, q: u32, immh: u32, immb: u32, rn: u32, rd: u32, out: &mut Instruction) {
    let (esize, shift) = match right_shift_size(immh, immb) {
        Some(v) => v,
        None => return,
    };
    let arr = match shift_arr(esize, q) {
        Some(a) => a,
        None => return,
    };
    out.set(code);
    push_shift3(out, rd, rn, arr, shift);
}

/// A left-shift same-size vector form (`Vd.T, Vn.T, #shift`).
#[inline]
fn left_shift(code: Code, q: u32, immh: u32, immb: u32, rn: u32, rd: u32, out: &mut Instruction) {
    let (esize, shift) = match left_shift_size(immh, immb) {
        Some(v) => v,
        None => return,
    };
    let arr = match shift_arr(esize, q) {
        Some(a) => a,
        None => return,
    };
    out.set(code);
    push_shift3(out, rd, rn, arr, shift);
}

/// Arrangement for a same-size shift form (`esize:Q`); `.1d` (64-bit, Q==0) is
/// reserved.
#[inline]
fn shift_arr(esize: u32, q: u32) -> Option<VectorArrangement> {
    Some(match (esize, q) {
        (8, 0) => VectorArrangement::V8B,
        (8, _) => VectorArrangement::V16B,
        (16, 0) => VectorArrangement::V4H,
        (16, _) => VectorArrangement::V8H,
        (32, 0) => VectorArrangement::V2S,
        (32, _) => VectorArrangement::V4S,
        (64, 0) => return None,
        (64, _) => VectorArrangement::V2D,
        _ => return None,
    })
}

/// Push `Vd, Vn, #shift` for a same-arrangement shift form.
#[inline]
fn push_shift3(out: &mut Instruction, rd: u32, rn: u32, arr: VectorArrangement, shift: u32) {
    out.push_operand(vreg(rd, arr));
    out.push_operand(vreg(rn, arr));
    out.push_operand(Operand::ImmUnsigned(shift as u64));
}

/// Narrowing shift-right (SHRN/RSHRN/SQSHRN/SQRSHRN/UQSHRN/UQRSHRN/SQSHRUN/
/// SQRSHRUN). The destination arrangement is `Tb` (half-width, `2`-form on
/// `Q==1`); the source is `Ta` (full-width double element).
#[inline]
#[allow(clippy::too_many_arguments)]
fn decode_narrow(q: u32, u: u32, opcode: u32, immh: u32, immb: u32, rn: u32, rd: u32, out: &mut Instruction) {
    let (dst_esize, shift) = match narrow_shift_size(immh, immb) {
        Some(v) => v,
        None => return,
    };
    let (dst_arr, src_arr) = match narrow_arrs(dst_esize, q) {
        Some(v) => v,
        None => return,
    };
    let upper = q == 1;
    let code = match (u, opcode, upper) {
        (0, 0b10000, false) => Code::ShrnVec,
        (0, 0b10000, true) => Code::Shrn2Vec,
        (0, 0b10001, false) => Code::RshrnVec,
        (0, 0b10001, true) => Code::Rshrn2Vec,
        (0, 0b10010, false) => Code::SqshrnVec,
        (0, 0b10010, true) => Code::Sqshrn2Vec,
        (0, 0b10011, false) => Code::SqrshrnVec,
        (0, 0b10011, true) => Code::Sqrshrn2Vec,
        (1, 0b10000, false) => Code::SqshrunVec,
        (1, 0b10000, true) => Code::Sqshrun2Vec,
        (1, 0b10001, false) => Code::SqrshrunVec,
        (1, 0b10001, true) => Code::Sqrshrun2Vec,
        (1, 0b10010, false) => Code::UqshrnVec,
        (1, 0b10010, true) => Code::Uqshrn2Vec,
        (1, 0b10011, false) => Code::UqrshrnVec,
        (1, 0b10011, true) => Code::Uqrshrn2Vec,
        _ => return,
    };
    out.set(code);
    out.push_operand(vreg(rd, dst_arr));
    out.push_operand(vreg(rn, src_arr));
    out.push_operand(Operand::ImmUnsigned(shift as u64));
}

/// `(dst_arr, src_arr)` for a narrowing form, given the *destination* element
/// size and `Q`. The `2`-forms (`Q==1`) write the upper half (`.16b`/`.8h`/
/// `.4s`) and the non-`2` forms the lower half (`.8b`/`.4h`/`.2s`); the source
/// is always the full 128-bit double-width arrangement.
#[inline]
fn narrow_arrs(dst_esize: u32, q: u32) -> Option<(VectorArrangement, VectorArrangement)> {
    Some(match dst_esize {
        8 => (
            if q == 1 { VectorArrangement::V16B } else { VectorArrangement::V8B },
            VectorArrangement::V8H,
        ),
        16 => (
            if q == 1 { VectorArrangement::V8H } else { VectorArrangement::V4H },
            VectorArrangement::V4S,
        ),
        32 => (
            if q == 1 { VectorArrangement::V4S } else { VectorArrangement::V2S },
            VectorArrangement::V2D,
        ),
        _ => return None,
    })
}

/// SSHLL/USHLL (and the SXTL/UXTL `#0` aliases). The destination is the
/// long (double-width) arrangement; the source is `Tb`. `Q` selects the source
/// half (`2`-forms read the upper half).
#[inline]
fn decode_shll(q: u32, u: u32, immh: u32, immb: u32, rn: u32, rd: u32, out: &mut Instruction) {
    // The element size selected by immh is the *source* element size; the
    // shift is `(immh:immb) - esize`.
    let (src_esize, shift) = match left_shift_size(immh, immb) {
        Some(v) => v,
        None => return,
    };
    if src_esize == 64 {
        return; // no long form past 32->64.
    }
    let (dst_arr, src_arr) = match long_arrs(src_esize, q) {
        Some(v) => v,
        None => return,
    };
    let upper = q == 1;
    if shift == 0 {
        // SXTL / UXTL alias (no immediate operand).
        let code = match (u, upper) {
            (0, false) => Code::SxtlVec,
            (0, true) => Code::Sxtl2Vec,
            (1, false) => Code::UxtlVec,
            (_, _) => Code::Uxtl2Vec,
        };
        out.set(code);
        out.push_operand(vreg(rd, dst_arr));
        out.push_operand(vreg(rn, src_arr));
        return;
    }
    let code = match (u, upper) {
        (0, false) => Code::SshllVec,
        (0, true) => Code::Sshll2Vec,
        (1, false) => Code::UshllVec,
        (_, _) => Code::Ushll2Vec,
    };
    out.set(code);
    out.push_operand(vreg(rd, dst_arr));
    out.push_operand(vreg(rn, src_arr));
    out.push_operand(Operand::ImmUnsigned(shift as u64));
}

/// `(dst_arr, src_arr)` for a lengthening (long) form, given the *source*
/// element size and `Q`. Source `2`-forms read the upper half.
#[inline]
fn long_arrs(src_esize: u32, q: u32) -> Option<(VectorArrangement, VectorArrangement)> {
    Some(match src_esize {
        8 => (
            VectorArrangement::V8H,
            if q == 1 { VectorArrangement::V16B } else { VectorArrangement::V8B },
        ),
        16 => (
            VectorArrangement::V4S,
            if q == 1 { VectorArrangement::V8H } else { VectorArrangement::V4H },
        ),
        32 => (
            VectorArrangement::V2D,
            if q == 1 { VectorArrangement::V4S } else { VectorArrangement::V2S },
        ),
        _ => return None,
    })
}

/// Fixed-point convert (SCVTF/UCVTF/FCVTZS/FCVTZU). `#fbits = 2*esize -
/// (immh:immb)`; the element size is selected by immh (`immh==0001` reserved for
/// these — 16-bit needs FEAT_FP16, gated by the dispatcher's feature set already
/// being permissive here). The arrangement is same-size in/out.
#[inline]
fn fixed_cvt(code: Code, q: u32, immh: u32, immb: u32, rn: u32, rd: u32, out: &mut Instruction) {
    // immh==0001 -> 16-bit (half), 001x/01xx -> 32-bit, 1xxx -> 64-bit. The
    // value field gives fbits = 2*esize - (immh:immb).
    let (esize, shift) = match right_shift_size(immh, immb) {
        Some(v) => v,
        None => return,
    };
    // 8-bit element (immh==0001 would be esize 8 from the right-shift table) is
    // not valid for fixed-point convert; the smallest is 16-bit.
    if esize == 8 {
        return;
    }
    let arr = match shift_arr(esize, q) {
        Some(a) => a,
        None => return,
    };
    out.set(code);
    push_shift3(out, rd, rn, arr, shift);
}

// ---------------------------------------------------------------------------
// Scalar shift by immediate.
// ---------------------------------------------------------------------------

/// Scalar shift-by-immediate (`asisdshf`):
/// `0 1 U 111110 immh immb opcode 1 Rn Rd`. The same-size forms operate on the
/// element-size scalar register view; the narrowing forms transfer between the
/// half-width and full-width scalar views.
#[inline]
fn decode_shift_scalar(word: u32, out: &mut Instruction) {
    let u = bit(word, 29);
    let immh = bits(word, 19, 4);
    let immb = bits(word, 16, 3);
    let opcode = bits(word, 11, 5);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);

    match opcode {
        // Right-shift same-size (only 64-bit `D` is architecturally valid).
        0b00000 | 0b00010 | 0b00100 | 0b00110 | 0b01000 => {
            let (esize, shift) = match right_shift_size(immh, immb) {
                Some(v) => v,
                None => return,
            };
            if esize != 64 {
                return;
            }
            let code = match (u, opcode) {
                (0, 0b00000) => Code::SshrScalar,
                (1, 0b00000) => Code::UshrScalar,
                (0, 0b00010) => Code::SsraScalar,
                (1, 0b00010) => Code::UsraScalar,
                (0, 0b00100) => Code::SrshrScalar,
                (1, 0b00100) => Code::UrshrScalar,
                (0, 0b00110) => Code::SrsraScalar,
                (1, 0b00110) => Code::UrsraScalar,
                (1, 0b01000) => Code::SriScalar,
                _ => return,
            };
            out.set(code);
            out.push_operand(scalar_reg(rd, esize));
            out.push_operand(scalar_reg(rn, esize));
            out.push_operand(Operand::ImmUnsigned(shift as u64));
        }
        // SHL (U==0) / SLI (U==1): left, same-size, 64-bit `D` only.
        0b01010 => scalar_left(if u == 0 { Code::ShlScalar } else { Code::SliScalar }, immh, immb, rn, rd, out),
        // Saturating left shift (any size): SQSHL/UQSHL/SQSHLU.
        0b01100 if u == 1 => scalar_left_any(Code::SqshluImmScalar, immh, immb, rn, rd, out),
        0b01110 => {
            let code = if u == 0 { Code::SqshlImmScalar } else { Code::UqshlImmScalar };
            scalar_left_any(code, immh, immb, rn, rd, out);
        }
        // Narrowing saturating shift-right (B/H/S destination).
        0b10000..=0b10011 => scalar_narrow(u, opcode, immh, immb, rn, rd, out),
        // Fixed-point convert (any size).
        0b11100 => {
            let code = if u == 0 { Code::ScvtfFixedScalar } else { Code::UcvtfFixedScalar };
            scalar_fixed_cvt(code, immh, immb, rn, rd, out);
        }
        0b11111 => {
            let code = if u == 0 { Code::FcvtzsFixedScalar } else { Code::FcvtzuFixedScalar };
            scalar_fixed_cvt(code, immh, immb, rn, rd, out);
        }
        _ => {}
    }
}

/// Scalar left-shift restricted to the 64-bit `D` view (SHL/SLI).
#[inline]
fn scalar_left(code: Code, immh: u32, immb: u32, rn: u32, rd: u32, out: &mut Instruction) {
    let (esize, shift) = match left_shift_size(immh, immb) {
        Some(v) => v,
        None => return,
    };
    if esize != 64 {
        return;
    }
    out.set(code);
    out.push_operand(scalar_reg(rd, esize));
    out.push_operand(scalar_reg(rn, esize));
    out.push_operand(Operand::ImmUnsigned(shift as u64));
}

/// Scalar saturating left-shift for any element size (SQSHL/UQSHL/SQSHLU).
#[inline]
fn scalar_left_any(code: Code, immh: u32, immb: u32, rn: u32, rd: u32, out: &mut Instruction) {
    let (esize, shift) = match left_shift_size(immh, immb) {
        Some(v) => v,
        None => return,
    };
    out.set(code);
    out.push_operand(scalar_reg(rd, esize));
    out.push_operand(scalar_reg(rn, esize));
    out.push_operand(Operand::ImmUnsigned(shift as u64));
}

/// Scalar narrowing saturating shift-right (SQSHRN/SQRSHRN/UQSHRN/UQRSHRN/
/// SQSHRUN/SQRSHRUN). The destination is the half-width scalar view, the source
/// the full-width view.
#[inline]
#[allow(clippy::too_many_arguments)]
fn scalar_narrow(u: u32, opcode: u32, immh: u32, immb: u32, rn: u32, rd: u32, out: &mut Instruction) {
    let (dst_esize, shift) = match narrow_shift_size(immh, immb) {
        Some(v) => v,
        None => return,
    };
    let src_esize = dst_esize * 2;
    let code = match (u, opcode) {
        (0, 0b10010) => Code::SqshrnScalar,
        (0, 0b10011) => Code::SqrshrnScalar,
        (1, 0b10000) => Code::SqshrunScalar,
        (1, 0b10001) => Code::SqrshrunScalar,
        (1, 0b10010) => Code::UqshrnScalar,
        (1, 0b10011) => Code::UqrshrnScalar,
        _ => return,
    };
    out.set(code);
    out.push_operand(scalar_reg(rd, dst_esize));
    out.push_operand(scalar_reg(rn, src_esize));
    out.push_operand(Operand::ImmUnsigned(shift as u64));
}

/// Scalar fixed-point convert (any size, smallest 16-bit).
#[inline]
fn scalar_fixed_cvt(code: Code, immh: u32, immb: u32, rn: u32, rd: u32, out: &mut Instruction) {
    let (esize, shift) = match right_shift_size(immh, immb) {
        Some(v) => v,
        None => return,
    };
    if esize == 8 {
        return;
    }
    out.set(code);
    out.push_operand(scalar_reg(rd, esize));
    out.push_operand(scalar_reg(rn, esize));
    out.push_operand(Operand::ImmUnsigned(shift as u64));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{BufSink, FmtFormatter, Formatter};

    /// Decode `word` at ip 0 with all features, render with the default
    /// formatter, and assert the text equals `expected`.
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

    /// Decode then re-encode and require the exact same word back.
    #[track_caller]
    fn rt(word: u32) {
        let mut insn = Instruction::default();
        crate::decode::decode_into(word, 0x1000, FeatureSet::ALL, &mut insn);
        assert!(!insn.is_invalid(), "word {word:#010x} failed to decode");
        let got = insn.encode().expect("encode");
        assert_eq!(got, word, "round-trip mismatch for {word:#010x}: got {got:#010x}");
    }

    #[test]
    fn luti_neon() {
        // LUTI2 .16b (2-bit index) / .8h (3-bit index).
        render(0x4E801081, "luti2   v1.16b, {v4.16b}, v0[0]");
        render(0x4E807081, "luti2   v1.16b, {v4.16b}, v0[3]");
        render(0x4EC00081, "luti2   v1.8h, {v4.8h}, v0[0]");
        render(0x4EC07081, "luti2   v1.8h, {v4.8h}, v0[7]");
        // LUTI4 .16b single-table (1-bit index).
        render(0x4E402081, "luti4   v1.16b, {v4.16b}, v0[0]");
        render(0x4E406081, "luti4   v1.16b, {v4.16b}, v0[1]");
        // LUTI4 .8h two-table (2-bit index).
        render(0x4E401015, "luti4   v21.8h, {v0.8h, v1.8h}, v0[0]");
        render(0x4E407081, "luti4   v1.8h, {v4.8h, v5.8h}, v0[3]");
        for w in [
            0x4E801081, 0x4E803081, 0x4E805081, 0x4E807081, 0x4EC00081, 0x4EC07081, 0x4E402081,
            0x4E406081, 0x4E401015, 0x4E407081,
        ] {
            rt(w);
        }
        // Gated off without FEAT_LUT; the slot is otherwise unallocated.
        let mut insn = Instruction::default();
        crate::decode::simd_fp::decode(0x4E801081, 0, FeatureSet::BASE, &mut insn);
        assert!(insn.is_invalid());
        // TBL (size==00) in the adjacent slot is unaffected.
        render(0x4E1101A3, "tbl     v3.16b, {v13.16b}, v17.16b");
    }

    #[test]
    fn copy_dup_smov_umov_ins() {
        render(0x4E010E3D, "dup     v29.16b, w17"); // DUP (general)
        render(0x4E0506E3, "dup     v3.16b, v23.b[2]"); // DUP (element)
        render(0x4E18060D, "dup     v13.2d, v16.d[1]");
        render(0x0E062EE2, "smov    w2, v23.h[1]");
        render(0x4E1B2FD9, "smov    x25, v30.b[13]");
        render(0x0E1A3ECB, "umov    w11, v22.h[6]");
        // UMOV .S/.D render as the `mov` alias.
        render(0x0E143C1E, "mov     w30, v0.s[2]");
        // INS (general) and INS (element) render as `mov`.
        render(0x4E1A1E38, "mov     v24.h[6], w17");
        render(0x6E060CE4, "mov     v4.h[1], v7.h[0]");
        render(0x6E046698, "mov     v24.s[0], v20.s[3]");
        // DUP (element, scalar) — `asisdone`, rendered as the `mov` alias.
        render(0x5E0205B8, "mov     h24, v13.h[0]");
        render(0x5E03045B, "mov     b27, v2.b[1]");
        render(0x5E040506, "mov     s6, v8.s[0]");
        render(0x5E1F0723, "mov     b3, v25.b[15]");
        render(0x5E180400, "mov     d0, v0.d[1]");
    }

    #[test]
    fn permute() {
        render(0x0E4B3AF4, "zip1    v20.4h, v23.4h, v11.4h");
        render(0x4ED33AC9, "zip1    v9.2d, v22.2d, v19.2d");
        render(0x4EC81B6C, "uzp1    v12.2d, v27.2d, v8.2d");
        render(0x0E132B06, "trn1    v6.8b, v24.8b, v19.8b");
        render(0x4E4829BD, "trn1    v29.8h, v13.8h, v8.8h");
    }

    #[test]
    fn ext_and_table() {
        render(0x2E1221F9, "ext     v25.8b, v15.8b, v18.8b, #0x4");
        render(0x6E080B29, "ext     v9.16b, v25.16b, v8.16b, #0x1");
        render(0x4E1101A3, "tbl     v3.16b, {v13.16b}, v17.16b");
        render(0x0E022065, "tbl     v5.8b, {v3.16b, v4.16b}, v2.8b");
        render(0x0E061044, "tbx     v4.8b, {v2.16b}, v6.8b");
    }

    #[test]
    fn modified_immediate() {
        // MOVI byte, shifted-LSL, MSL, 64-bit scalar and .2d.
        render(0x4F04E52A, "movi    v10.16b, #0x89");
        render(0x4F02249E, "movi    v30.4s, #0x44, lsl #0x8");
        render(0x0F0104C1, "movi    v1.2s, #0x26");
        render(0x0F04D6D1, "movi    v17.2s, #0x96, msl #0x10");
        render(0x2F05E64B, "movi    d11, #0xff00ffff0000ff00");
        render(0x6F03E664, "movi    v4.2d, #0xffffff0000ffff");
        // MVNI shifted / MSL.
        render(0x2F0286D3, "mvni    v19.4h, #0x56");
        render(0x6F02C517, "mvni    v23.4s, #0x48, msl #0x8");
        // ORR / BIC vector immediate.
        render(0x4F00150A, "orr     v10.4s, #0x8");
        render(0x0F0457DD, "orr     v29.2s, #0x9e, lsl #0x10");
        render(0x2F0415BC, "bic     v28.2s, #0x8d");
        // FMOV vector immediate (single / .2d).
        render(0x0F04F40D, "fmov    v13.2s, #-2.0");
        render(0x4F03F6C0, "fmov    v0.4s, #1.375");
        render(0x6F04F50D, "fmov    v13.2d, #-3.0");
    }

    #[test]
    fn shift_by_immediate_vector() {
        render(0x4F7D06E0, "sshr    v0.2d, v23.2d, #0x3");
        render(0x2F1C0579, "ushr    v25.4h, v11.4h, #0x4");
        render(0x4F1317DB, "ssra    v27.8h, v30.8h, #0xd");
        render(0x0F28266D, "srshr   v13.2s, v19.2s, #0x18");
        render(0x2F1A3413, "ursra   v19.4h, v0.4h, #0x6");
        render(0x4F4256B3, "shl     v19.2d, v21.2d, #0x2");
        render(0x0F3A75B0, "sqshl   v16.2s, v13.2s, #0x1a");
        render(0x2F1A469C, "sri     v28.4h, v20.4h, #0x6");
        render(0x6F6C54AA, "sli     v10.2d, v5.2d, #0x2c");
    }

    #[test]
    fn shift_narrow_long_and_aliases() {
        render(0x0F1897FC, "sqshrn  v28.4h, v31.4s, #0x8");
        render(0x4F2796A9, "sqshrn2 v9.4s, v21.2d, #0x19");
        render(0x2F299448, "uqshrn  v8.2s, v2.2d, #0x17");
        render(0x2F2A85EC, "sqshrun v12.2s, v15.2d, #0x16");
        render(0x4F3E8559, "shrn2   v25.4s, v10.2d, #0x2");
        render(0x0F29A72B, "sshll   v11.2d, v25.2s, #0x9");
        render(0x2F30A561, "ushll   v1.2d, v11.2s, #0x10");
        // SXTL/UXTL aliases (SSHLL/USHLL #0).
        render(0x0F08A790, "sxtl    v16.8h, v28.8b");
        render(0x4F20A47A, "sxtl2   v26.2d, v3.4s");
    }

    #[test]
    fn shift_fixed_convert() {
        render(0x0F11E77E, "scvtf   v30.4h, v27.4h, #0xf");
        render(0x6F51FF23, "fcvtzu  v3.2d, v25.2d, #0x2f");
        render(0x4F10FFBF, "fcvtzs  v31.8h, v29.8h, #0x10");
    }

    #[test]
    fn shift_by_immediate_scalar() {
        render(0x5F450550, "sshr    d16, d10, #0x3b");
        render(0x5F7E54D2, "shl     d18, d6, #0x3e");
        render(0x7F7A656A, "sqshlu  d10, d11, #0x3a");
        render(0x7F6D5475, "sli     d21, d3, #0x2d");
        render(0x5F2197AC, "sqshrn  s12, d29, #0x1f");
        render(0x5F119431, "sqshrn  h17, s1, #0xf");
        render(0x5F11E574, "scvtf   h20, h11, #0xf");
        render(0x5F7CFC95, "fcvtzs  d21, d4, #0x4");
    }

    #[test]
    fn never_panics_sample() {
        // Sweep broad slices of the SIMD data-movement space; must never panic.
        for w in (0x0E00_0000u32..0x0E00_0000u32.wrapping_add(0x0100_0000)).step_by(2731) {
            let mut insn = Instruction::default();
            crate::decode::simd_fp::decode(w, 0, FeatureSet::ALL, &mut insn);
        }
        for w in (0x2F00_0000u32..0x2F00_0000u32.wrapping_add(0x0100_0000)).step_by(2731) {
            let mut insn = Instruction::default();
            crate::decode::simd_fp::decode(w, 0, FeatureSet::ALL, &mut insn);
        }
        // word<31>==1 in this region must not produce a *data-movement* decode:
        // `0xCE0A1EE1` is a valid `EOR3` (AdvSIMD crypto4), which is claimed by
        // the crypto sub-decoder in the parent dispatch — never by `simd_data`.
        {
            let mut insn = Instruction::default();
            decode(0xCE0A1EE1, 0, FeatureSet::ALL, &mut insn);
            assert!(insn.is_invalid(), "simd_data must not claim crypto4 EOR3");
        }
    }
}
