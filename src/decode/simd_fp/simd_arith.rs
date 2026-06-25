//! Advanced SIMD arithmetic encodings (ARM ARM C4.1.97).
//!
//! Hand-written from the ARM ARM. Owns the arithmetic leaves of the Advanced
//! SIMD group, in both vector (`asimd`, `word<30>=Q` selecting 64/128-bit) and
//! scalar (`asisd`) forms:
//!
//! * Three same (`+ FP16 + the "extra" SQRDMLAH/SQRDMLSH/FCADD/FCMLA`)
//! * Three different (widening / narrowing)
//! * Two-register miscellaneous (`+ FP / FP16`)
//! * Across lanes
//! * By element (vector × indexed element `Vm.T[index]`)
//!
//! Pairwise lives inside the three-same families (`ADDP`/`FADDP`/...), plus the
//! scalar-pairwise reductions (`ADDP (scalar)`, `FADDP`/`FMAXP`/...).
//!
//! Discrimination of the families is by the standard SIMD field layout
//! (`word<31>=0`, `word<30>=Q`, `word<29>=U`, `word<28:24>` = `01110`/`11110`
//! [vector] or `01111`/`11111` [by-element], `word<23:22>=size`, then the
//! `word<21>` / `word<11:10>` / `word<21:17>` selectors). Half-precision and
//! BF16 forms gate on [`Feature::Fp16`] / [`Feature::Bf16`] (decoding under the
//! default `FeatureSet::ALL`).
//!
//! Code identity: one [`Code`] per Advanced-SIMD mnemonic (suffixed `…Vec`); the
//! arrangement / lane / scalar-width live entirely in the operands, matching the
//! Binary Ninja differential corpus rendering.

use crate::decode::bits::{bit, bits};
use crate::enums::VectorArrangement as VA;
use crate::features::{Feature, FeatureSet};
use crate::instruction::Instruction;
use crate::mnemonic::Code;
use crate::operand::Operand;
use crate::register::Register;

// ---------------------------------------------------------------------------
// Register-bank tables + small operand constructors.
// ---------------------------------------------------------------------------

const V: [Register; 32] = [
    Register::V0, Register::V1, Register::V2, Register::V3, Register::V4, Register::V5, Register::V6, Register::V7,
    Register::V8, Register::V9, Register::V10, Register::V11, Register::V12, Register::V13, Register::V14, Register::V15,
    Register::V16, Register::V17, Register::V18, Register::V19, Register::V20, Register::V21, Register::V22, Register::V23,
    Register::V24, Register::V25, Register::V26, Register::V27, Register::V28, Register::V29, Register::V30, Register::V31,
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
const QR: [Register; 32] = [
    Register::Q0, Register::Q1, Register::Q2, Register::Q3, Register::Q4, Register::Q5, Register::Q6, Register::Q7,
    Register::Q8, Register::Q9, Register::Q10, Register::Q11, Register::Q12, Register::Q13, Register::Q14, Register::Q15,
    Register::Q16, Register::Q17, Register::Q18, Register::Q19, Register::Q20, Register::Q21, Register::Q22, Register::Q23,
    Register::Q24, Register::Q25, Register::Q26, Register::Q27, Register::Q28, Register::Q29, Register::Q30, Register::Q31,
];

/// A bare register operand (no decorations).
#[inline]
fn plain(reg: Register) -> Operand {
    Operand::Reg { reg, arr: None, lane: None, shift: None, extend: None, pred: None }
}

/// A vector register operand `V{n}` with arrangement `arr`.
#[inline]
fn vreg(n: u32, arr: VA) -> Operand {
    Operand::Reg { reg: V[(n & 0x1f) as usize], arr: Some(arr), lane: None, shift: None, extend: None, pred: None }
}

/// An indexed vector-element operand `V{n}.<Ts>[index]`.
#[inline]
fn vreg_idx(n: u32, arr: VA, index: u8) -> Operand {
    Operand::Reg { reg: V[(n & 0x1f) as usize], arr: Some(arr), lane: Some(index), shift: None, extend: None, pred: None }
}

/// A scalar SIMD register operand of the element width `eb` (8/16/32/64/128).
#[inline]
fn sca(n: u32, eb: u16) -> Operand {
    let n = (n & 0x1f) as usize;
    let reg = match eb {
        8 => BR[n],
        16 => HR[n],
        32 => SR[n],
        64 => DR[n],
        _ => QR[n],
    };
    plain(reg)
}

// ---------------------------------------------------------------------------
// Arrangement helpers.
// ---------------------------------------------------------------------------

/// The integer arrangement for `(size, Q)`: `size` selects the element width
/// (`00`=B `01`=H `10`=S `11`=D) and `Q` the 64-vs-128-bit width. `D` with
/// `Q==0` (`.1d`) is structurally invalid for most three-same integer ops; the
/// caller decides whether `.1d` is allowed.
#[inline]
fn arr_sizeq(size: u32, q: u32) -> VA {
    let q = q & 1 == 1;
    match size & 3 {
        0 => if q { VA::V16B } else { VA::V8B },
        1 => if q { VA::V8H } else { VA::V4H },
        2 => if q { VA::V4S } else { VA::V2S },
        _ => if q { VA::V2D } else { VA::V1D },
    }
}

/// The FP arrangement for `(sz, Q)` in the single/double FP families: `sz<0>`
/// picks S (`0`) vs D (`1`); `Q` the width. `.1d` (D with `Q==0`) is invalid.
#[inline]
fn arr_fp(sz: u32, q: u32) -> Option<VA> {
    let q = q & 1 == 1;
    if sz & 1 == 0 {
        Some(if q { VA::V4S } else { VA::V2S })
    } else if q {
        Some(VA::V2D)
    } else {
        None // .1d not allowed
    }
}

/// The half-precision FP arrangement for `Q`: `.4h`/`.8h`.
#[inline]
fn arr_fp16(q: u32) -> VA {
    if q & 1 == 1 { VA::V8H } else { VA::V4H }
}

/// Element width in bits for an integer `size` field.
#[inline]
fn esize(size: u32) -> u16 {
    match size & 3 {
        0 => 8,
        1 => 16,
        2 => 32,
        _ => 64,
    }
}

// ---------------------------------------------------------------------------
// Top-level entry.
// ---------------------------------------------------------------------------

/// Decode an Advanced-SIMD arithmetic instruction into `out`.
///
/// Handles only the arithmetic families (three-same / three-different /
/// two-reg-misc / across-lanes / by-element, vector and scalar). Anything not
/// in those families is left invalid so the SIMD data-movement decoder can try.
/// Pure, total and panic-free for every input.
#[inline]
pub fn decode(word: u32, ip: u64, features: FeatureSet, out: &mut Instruction) {
    let _ = ip;
    if bit(word, 31) != 0 {
        return; // bit31 must be 0 for this whole arithmetic family.
    }
    let top = bits(word, 24, 5); // word<28:24>
    match top {
        // Vector (asimd) and scalar (asisd) three-X / two-reg-misc / across.
        0b01110 => decode_vector(word, false, features, out),
        0b11110 => decode_vector(word, true, features, out),
        // By element: vector (01111) and scalar (11111).
        0b01111 => decode_by_element(word, false, features, out),
        0b11111 => decode_by_element(word, true, features, out),
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Vector / scalar (non by-element) dispatch.
// ---------------------------------------------------------------------------

/// Dispatch the `01110`/`11110` (vector / scalar) encodings to the right family
/// by the `word<21>` / `word<11:10>` / `word<21:17>` selectors.
#[inline]
fn decode_vector(word: u32, scalar: bool, features: FeatureSet, out: &mut Instruction) {
    let b21 = bit(word, 21);
    let b11_10 = bits(word, 10, 2);
    let bits21_17 = bits(word, 17, 5);

    if b21 == 1 {
        // Three same has `word<10>==1` (with `word<11>` being the opcode LSB, so
        // `word<11:10>` is `01` *or* `11`). Three different has `word<11:10>==00`.
        // Two-reg-misc / across-lanes / scalar-pairwise share `word<11:10>==10`,
        // split on `word<21:17>`.
        if bit(word, 10) == 1 {
            three_same(word, scalar, features, out);
        } else if b11_10 == 0b00 {
            // Three different: opcode = word<15:12>. Vector for all forms;
            // scalar only for SQDMULL/SQDMLAL/SQDMLSL.
            three_different(word, scalar, features, out);
        } else if b11_10 == 0b10 {
            if bits21_17 == 0b10000 {
                two_reg_misc(word, scalar, features, out);
            } else if bits21_17 == 0b11000 {
                across_lanes(word, scalar, features, out);
            } else if bits21_17 == 0b11100 {
                fp16_two_reg_misc(word, scalar, features, out);
            }
        }
        return;
    }

    // word<21> == 0: the "three-same extra" rows (SQRDMLAH/SQRDMLSH and the
    // complex FCMLA/FCADD), plus the FP16 three-same family.
    three_same_extra_or_fp16(word, scalar, features, out);
}

// ---------------------------------------------------------------------------
// Three same (integer + single/double FP).
// ---------------------------------------------------------------------------

/// Advanced SIMD three-same (vector and scalar). `U = word<29>`, `size =
/// word<23:22>`, `opcode = word<15:11>`. Covers the integer ops, the
/// single/double FP ops (the `0b11xxx` opcode region with `size<1>` picking
/// S/D), and the pairwise forms.
#[inline]
fn three_same(word: u32, scalar: bool, features: FeatureSet, out: &mut Instruction) {
    let q = bit(word, 30);
    let u = bit(word, 29);
    let size = bits(word, 22, 2);
    let opcode = bits(word, 11, 5);
    let rm = bits(word, 16, 5);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);

    // The FP three-same opcodes live in `0b11000..=0b11111`; `size<1>` is the
    // S/D select and `size<0>` distinguishes some ops. Handle FP first.
    if (opcode & 0b11000) == 0b11000 {
        fp_three_same(word, scalar, q, u, size, opcode, rm, rn, rd, features, out);
        return;
    }

    // Opcode 0b00011 is the bitwise logical family (AND/BIC/ORR/ORN [U=0],
    // EOR/BSL/BIT/BIF [U=1]). The `size` field is reused as the op selector and
    // the arrangement is always `.8b`/`.16b`; handle it specially (no scalar
    // form exists).
    if opcode == 0b00011 {
        if !scalar {
            three_same_logical(q, u, size, rm, rn, rd, out);
        }
        return;
    }

    // Integer three-same.
    let code = match int_three_same_code(u, opcode) {
        Some(c) => c,
        None => return,
    };
    let is_cmp = matches!(opcode, 0b00110 | 0b00111 | 0b10001);

    if scalar {
        // Scalar three-same is defined for a subset of ops at the encoded
        // element width; the rest are vector-only.
        let eb = match scalar_three_same_width(opcode, size) {
            Some(e) => e,
            None => return,
        };
        out.set(code);
        out.push_operand(sca(rd, eb));
        out.push_operand(sca(rn, eb));
        out.push_operand(sca(rm, eb));
        return;
    }

    // Vector.
    if !int_three_same_size_ok(opcode, u, size, q) {
        return;
    }
    let arr = arr_sizeq(size, q);
    out.set(code);
    out.push_operand(vreg(rd, arr));
    out.push_operand(vreg(rn, arr));
    out.push_operand(vreg(rm, arr));
    let _ = is_cmp;
}

/// The authoritative `(U, opcode)` -> integer three-same [`Code`] table (ARM
/// ARM C4.1.97 "Advanced SIMD three same"). Returns `None` for the logical
/// family (`opcode==00011`, owned by `simd_data`) and unallocated opcodes.
#[inline]
fn int_three_same_code(u: u32, opcode: u32) -> Option<Code> {
    let c = match (u, opcode) {
        (0, 0b00000) => Code::ShaddVec,
        (0, 0b00001) => Code::SqaddVec,
        (0, 0b00010) => Code::SrhaddVec,
        (0, 0b00100) => Code::ShsubVec,
        (0, 0b00101) => Code::SqsubVec,
        (0, 0b00110) => Code::CmgtVec,
        (0, 0b00111) => Code::CmgeVec,
        (0, 0b01000) => Code::SshlVec,
        (0, 0b01001) => Code::SqshlVec,
        (0, 0b01010) => Code::SrshlVec,
        (0, 0b01011) => Code::SqrshlVec,
        (0, 0b01100) => Code::SmaxVec,
        (0, 0b01101) => Code::SminVec,
        (0, 0b01110) => Code::SabdVec,
        (0, 0b01111) => Code::SabaVec,
        (0, 0b10000) => Code::AddVec,
        (0, 0b10001) => Code::CmtstVec,
        (0, 0b10010) => Code::MlaVec,
        (0, 0b10011) => Code::MulVec,
        (0, 0b10100) => Code::SmaxpVec,
        (0, 0b10101) => Code::SminpVec,
        (0, 0b10110) => Code::SqdmulhVec,
        (0, 0b10111) => Code::AddpVec,
        (1, 0b00000) => Code::UhaddVec,
        (1, 0b00001) => Code::UqaddVec,
        (1, 0b00010) => Code::UrhaddVec,
        (1, 0b00100) => Code::UhsubVec,
        (1, 0b00101) => Code::UqsubVec,
        (1, 0b00110) => Code::CmhiVec,
        (1, 0b00111) => Code::CmhsVec,
        (1, 0b01000) => Code::UshlVec,
        (1, 0b01001) => Code::UqshlVec,
        (1, 0b01010) => Code::UrshlVec,
        (1, 0b01011) => Code::UqrshlVec,
        (1, 0b01100) => Code::UmaxVec,
        (1, 0b01101) => Code::UminVec,
        (1, 0b01110) => Code::UabdVec,
        (1, 0b01111) => Code::UabaVec,
        (1, 0b10000) => Code::SubVec,
        (1, 0b10001) => Code::CmeqVec,
        (1, 0b10010) => Code::MlsVec,
        (1, 0b10011) => Code::PmulVec,
        (1, 0b10100) => Code::UmaxpVec,
        (1, 0b10101) => Code::UminpVec,
        (1, 0b10110) => Code::SqrdmulhVec,
        _ => return None,
    };
    Some(c)
}

/// Advanced SIMD three-same *logical* family (opcode `0b00011`, vector only).
/// `U` and the `size` field (reused as op) select the operation; the arrangement
/// is always `.8b` (Q==0) or `.16b` (Q==1):
///
/// * U=0: AND(00) / BIC(01) / ORR(10) / ORN(11)
/// * U=1: EOR(00) / BSL(01) / BIT(10) / BIF(11)
///
/// `ORR` with `Vn == Vm` is the preferred `MOV <Vd>.<T>, <Vn>.<T>` alias (two
/// operands), matching the Binary Ninja corpus.
#[inline]
fn three_same_logical(q: u32, u: u32, size: u32, rm: u32, rn: u32, rd: u32, out: &mut Instruction) {
    let arr = if q == 1 { VA::V16B } else { VA::V8B };
    let code = match (u, size & 3) {
        (0, 0b00) => Code::AndVec,
        (0, 0b01) => Code::BicVec,
        (0, 0b10) => Code::OrrVec,
        (0, 0b11) => Code::OrnVec,
        (1, 0b00) => Code::EorVec,
        (1, 0b01) => Code::BslVec,
        (1, 0b10) => Code::BitVec,
        _ => Code::BifVec, // (1, 0b11)
    };
    // ORR with identical source registers is the preferred `MOV` two-operand
    // alias (binja renders `mov Vd.T, Vn.T`).
    if code == Code::OrrVec && rn == rm {
        out.set(Code::OrrVec);
        out.set_mnemonic(crate::mnemonic::Mnemonic::Mov);
        out.push_operand(vreg(rd, arr));
        out.push_operand(vreg(rn, arr));
        return;
    }
    out.set(code);
    out.push_operand(vreg(rd, arr));
    out.push_operand(vreg(rn, arr));
    out.push_operand(vreg(rm, arr));
}

/// Whether an integer three-same `(U, opcode, size)` vector form is allocated
/// for `(size, Q)`. PMUL is byte-only; MUL/MLA/MLS/max/min/abd/aba/pairwise/
/// SQDMUL[H]/CMHI/CMHS are not defined for D; the elementwise add/sub/cmp/shift
/// and saturating ops accept D (but never `.1d`, i.e. D with `Q==0`).
#[inline]
fn int_three_same_size_ok(opcode: u32, u: u32, size: u32, q: u32) -> bool {
    // PMUL (U=1, opcode 10011) is byte-only.
    if u == 1 && opcode == 0b10011 {
        return size == 0b00;
    }
    if size == 0b11 {
        if q == 0 {
            return false; // `.1d` never valid.
        }
        // For `.2d`, the D-capable ops are: SQADD/UQADD, SQSUB/UQSUB, the four
        // shifts (SSHL/USHL, SQSHL/UQSHL, SRSHL/URSHL, SQRSHL/UQRSHL), ADD/SUB,
        // CMGT/CMGE/CMHI/CMHS, CMTST/CMEQ, ABS/NEG (misc), and ADDP.
        return matches!(
            opcode,
            0b00001 // SQADD/UQADD
                | 0b00101 // SQSUB/UQSUB
                | 0b00110 // CMGT/CMHI
                | 0b00111 // CMGE/CMHS
                | 0b01000 // SSHL/USHL
                | 0b01001 // SQSHL/UQSHL
                | 0b01010 // SRSHL/URSHL
                | 0b01011 // SQRSHL/UQRSHL
                | 0b10000 // ADD/SUB
                | 0b10001 // CMTST/CMEQ
                | 0b10111 // ADDP
        );
    }
    true
}

/// Width (bits) of a scalar three-same form, or `None` if not a scalar form.
/// Scalar three-same is defined for the saturating add/sub, the four shifts,
/// SQDMULH/SQRDMULH (H/S only), and ADD/SUB/CMP/CMTST (doubleword only).
#[inline]
fn scalar_three_same_width(opcode: u32, size: u32) -> Option<u16> {
    match opcode {
        // ADD/SUB and the compare family (CMGT/CMGE/CMHI/CMHS/CMTST/CMEQ):
        // doubleword only.
        0b10000 | 0b10001 | 0b00110 | 0b00111 => {
            if size == 0b11 {
                Some(64)
            } else {
                None
            }
        }
        // Saturating add/sub and the four shifts: any element size.
        0b00001 // SQADD/UQADD
        | 0b00101 // SQSUB/UQSUB
        | 0b01000 // SSHL/USHL
        | 0b01001 // SQSHL/UQSHL
        | 0b01010 // SRSHL/URSHL
        | 0b01011 // SQRSHL/UQRSHL
            => Some(esize(size)),
        // SQDMULH/SQRDMULH: H or S only.
        0b10110 => {
            if size == 0b01 || size == 0b10 {
                Some(esize(size))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// FP three-same (vector and scalar), single/double precision.
#[allow(clippy::too_many_arguments)]
#[inline]
fn fp_three_same(
    word: u32,
    scalar: bool,
    q: u32,
    u: u32,
    size: u32,
    opcode: u32,
    rm: u32,
    rn: u32,
    rd: u32,
    _features: FeatureSet,
    out: &mut Instruction,
) {
    let _ = word;
    // FMLAL/FMLSL/FMLAL2/FMLSL2 (vector, three-same) are widening FP ops that
    // share the FP three-same opcode space: U==0 op 11101 (FMLAL a=0 / FMLSL
    // a=1), U==1 op 11001 (FMLAL2 a=0 / FMLSL2 a=1). They widen `.2h`/`.4h`
    // sources into `.2s`/`.4s` and so are not scalar.
    let a = bit(size, 1);
    let fmlal_code = match (u, opcode) {
        (0, 0b11101) => Some(if a == 0 { Code::FmlalVec } else { Code::FmlslVec }),
        (1, 0b11001) => Some(if a == 0 { Code::Fmlal2Vec } else { Code::Fmlsl2Vec }),
        _ => None,
    };
    if let Some(code) = fmlal_code {
        if scalar {
            return;
        }
        let ta = if q == 1 { VA::V4S } else { VA::V2S };
        let tb = if q == 1 { VA::V4H } else { VA::V2H };
        out.set(code);
        out.push_operand(vreg(rd, ta));
        out.push_operand(vreg(rn, tb));
        out.push_operand(vreg(rm, tb));
        return;
    }

    // For FP three-same the precision is `size<0>` (`0`=S, `1`=D) and the
    // opcode-group bit is `size<1>` (the add/sub vs min/max-style selector).
    let o1 = bit(size, 1);
    let code = match (u, o1, opcode) {
        (0, 0, 0b11000) => Code::FmaxnmVec,
        (0, 0, 0b11001) => Code::FmlaVec,
        (0, 0, 0b11010) => Code::FaddVec,
        (0, 0, 0b11011) => Code::FmulxVec,
        (0, 0, 0b11100) => Code::FcmeqVec,
        (0, 0, 0b11110) => Code::FmaxVec,
        (0, 0, 0b11111) => Code::FrecpsVec,
        (0, 1, 0b11000) => Code::FminnmVec,
        (0, 1, 0b11001) => Code::FmlsVec,
        (0, 1, 0b11010) => Code::FsubVec,
        (0, 1, 0b11110) => Code::FminVec,
        (0, 1, 0b11111) => Code::FrsqrtsVec,
        (1, 0, 0b11000) => Code::FmaxnmpVec,
        (1, 0, 0b11010) => Code::FaddpVec,
        (1, 0, 0b11011) => Code::FmulVec,
        (1, 0, 0b11100) => Code::FcmgeVec,
        (1, 0, 0b11101) => Code::FacgeVec,
        (1, 0, 0b11110) => Code::FmaxpVec,
        (1, 0, 0b11111) => Code::FdivVec,
        (1, 1, 0b11000) => Code::FminnmpVec,
        (1, 1, 0b11010) => Code::FabdVec,
        (1, 1, 0b11100) => Code::FcmgtVec,
        (1, 1, 0b11101) => Code::FacgtVec,
        (1, 1, 0b11110) => Code::FminpVec,
        _ => return,
    };

    if scalar {
        let eb: u16 = if bit(size, 0) == 1 { 64 } else { 32 };
        out.set(code);
        out.push_operand(sca(rd, eb));
        out.push_operand(sca(rn, eb));
        out.push_operand(sca(rm, eb));
        return;
    }
    // Precision from size<0>; `.2d` requires Q==1, `.1d` invalid.
    let arr = match arr_fp(bit(size, 0), q) {
        Some(a) => a,
        None => return,
    };
    out.set(code);
    out.push_operand(vreg(rd, arr));
    out.push_operand(vreg(rn, arr));
    out.push_operand(vreg(rm, arr));
}

// ---------------------------------------------------------------------------
// Three-same "extra" (SQRDMLAH/SQRDMLSH, FCMLA/FCADD) and FP16 three-same.
// ---------------------------------------------------------------------------

/// The `word<21>==0` rows of the `01110`/`11110` space: the integer "extra"
/// three-same (`SQRDMLAH`/`SQRDMLSH`, U=1, `word<11:10>=01`), the complex FP
/// (`FCMLA`/`FCADD`, U=1, `word<11:10>` = `01`/`11` with rotate), and the
/// half-precision three-same family (`word<11:10>=01`, opcode in `word<13:11>`).
#[inline]
fn three_same_extra_or_fp16(word: u32, scalar: bool, features: FeatureSet, out: &mut Instruction) {
    // The Advanced SIMD copy family (DUP/INS/SMOV/UMOV/MOV) lives in the same
    // `word<21>==0` region with `word<15>==0 && word<10>==1`, distinguished by
    // `word<23:22>==00`. The three-same "extra" (SQRDMLAH/SQRDMLSH), complex
    // (FCMLA/FCADD), and FP16-three-same families all carry a non-zero size
    // field here (`word<23:22> != 00`); bail out otherwise so `simd_data` owns
    // the copy family.
    if bits(word, 22, 2) == 0 {
        return;
    }
    let u = bit(word, 29);
    // All families in this region fix `word<10>==1`.
    if bit(word, 10) != 1 {
        return;
    }
    // Route on `word<15:13>`:
    //   000/001 (word<15:14>==00) -> FP16 three-same (3-bit opcode word<13:11>);
    //   100     -> SQRDMLAH/SQRDMLSH (the v8.1 integer "extra", U==1);
    //   110     -> FCMLA (complex, U==1, rotate word<12:11>);
    //   111     -> FCADD (complex, U==1, rotate word<12>).
    if bits(word, 14, 2) == 0b00 {
        fp16_three_same(word, scalar, features, out);
        return;
    }
    // SDOT/UDOT (Advanced SIMD dot product, vector form): `word<15:10>==100101`,
    // `size==10`, `U` selecting signed (0) vs unsigned (1). Only the vector form
    // exists (no scalar). It sits in this `word<21>==0` region but, unlike the
    // SQRDML/complex families below, is defined for both U values.
    if bits(word, 10, 6) == 0b100101 && bits(word, 22, 2) == 0b10 && !scalar {
        let q = bit(word, 30);
        let rm = bits(word, 16, 5);
        let rn = bits(word, 5, 5);
        let rd = bits(word, 0, 5);
        let code = if u == 0 { Code::SdotVec } else { Code::UdotVec };
        let ta = arr_sizeq(0b10, q); // .2s / .4s accumulator
        let tb = arr_sizeq(0b00, q); // .8b / .16b sources
        out.set(code);
        out.push_operand(vreg(rd, ta));
        out.push_operand(vreg(rn, tb));
        out.push_operand(vreg(rm, tb));
        return;
    }
    // The remaining families (SQRDMLAH/SQRDMLSH and the complex FCMLA/FCADD)
    // all require U==1.
    if u != 1 {
        return;
    }
    match bits(word, 13, 3) {
        0b100 => sqrdml_extra(word, scalar, out),
        0b110 | 0b111 => complex_fp(word, scalar, features, out),
        _ => {}
    }
}

/// `SQRDMLAH`/`SQRDMLSH` (the v8.1 "extra" three-same), vector and scalar.
/// `U==1`, `word<11:10>==01`, opcode bit at `word<11>` already consumed; the
/// op is selected by `word<11>` within the 5-bit `word<15:11>` = `10000`
/// (SQRDMLAH) / `10001` (SQRDMLSH).
#[inline]
fn sqrdml_extra(word: u32, scalar: bool, out: &mut Instruction) {
    let q = bit(word, 30);
    let size = bits(word, 22, 2);
    let opcode = bits(word, 11, 5);
    let rm = bits(word, 16, 5);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);

    let code = match opcode {
        0b10000 => Code::SqrdmlahVec,
        0b10001 => Code::SqrdmlshVec,
        _ => return,
    };
    // Defined for H/S only.
    if size != 0b01 && size != 0b10 {
        return;
    }
    if scalar {
        let eb = esize(size);
        out.set(code);
        out.push_operand(sca(rd, eb));
        out.push_operand(sca(rn, eb));
        out.push_operand(sca(rm, eb));
        return;
    }
    if size == 0b11 && q == 0 {
        return;
    }
    let arr = arr_sizeq(size, q);
    out.set(code);
    out.push_operand(vreg(rd, arr));
    out.push_operand(vreg(rn, arr));
    out.push_operand(vreg(rm, arr));
}

/// Half-precision (`FEAT_FP16`) three-same, vector and scalar. `a = word<23>`
/// and the 3-bit opcode `word<13:11>` select the operation; the element size is
/// fixed to half (`.4h`/`.8h` vector, `h` scalar).
#[inline]
fn fp16_three_same(word: u32, scalar: bool, features: FeatureSet, out: &mut Instruction) {
    if !features.has(Feature::Fp16) {
        return;
    }
    let q = bit(word, 30);
    let u = bit(word, 29);
    let a = bit(word, 23); // size<1>
    let opcode = bits(word, 11, 3);
    let rm = bits(word, 16, 5);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);

    // FP16 three-same opcode = `word<13:11>`, with `a = word<23>` and `U`
    // distinguishing the variant (ARM ARM C4.1.97, pinned to the corpus).
    let code = match (u, a, opcode) {
        (0, 0, 0b000) => Code::FmaxnmVec,
        (0, 0, 0b001) => Code::FmlaVec,
        (0, 0, 0b010) => Code::FaddVec,
        (0, 0, 0b011) => Code::FmulxVec,
        (0, 0, 0b100) => Code::FcmeqVec,
        (0, 0, 0b110) => Code::FmaxVec,
        (0, 0, 0b111) => Code::FrecpsVec,
        (0, 1, 0b000) => Code::FminnmVec,
        (0, 1, 0b001) => Code::FmlsVec,
        (0, 1, 0b010) => Code::FsubVec,
        (0, 1, 0b110) => Code::FminVec,
        (0, 1, 0b111) => Code::FrsqrtsVec,
        (1, 0, 0b000) => Code::FmaxnmpVec,
        (1, 0, 0b010) => Code::FaddpVec,
        (1, 0, 0b011) => Code::FmulVec,
        (1, 0, 0b100) => Code::FcmgeVec,
        (1, 0, 0b101) => Code::FacgeVec,
        (1, 0, 0b110) => Code::FmaxpVec,
        (1, 0, 0b111) => Code::FdivVec,
        (1, 1, 0b000) => Code::FminnmpVec,
        (1, 1, 0b010) => Code::FabdVec,
        (1, 1, 0b100) => Code::FcmgtVec,
        (1, 1, 0b101) => Code::FacgtVec,
        (1, 1, 0b110) => Code::FminpVec,
        _ => return,
    };
    if scalar {
        out.set(code);
        out.push_operand(sca(rd, 16));
        out.push_operand(sca(rn, 16));
        out.push_operand(sca(rm, 16));
        return;
    }
    let arr = arr_fp16(q);
    out.set(code);
    out.push_operand(vreg(rd, arr));
    out.push_operand(vreg(rn, arr));
    out.push_operand(vreg(rm, arr));
}

/// Complex-number FP: `FCMLA` (with rotate `0/90/180/270`) and `FCADD` (with
/// rotate `90/270`), vector forms (`FEAT_FCMA`). Both add a trailing `#rot`
/// immediate operand in degrees.
#[inline]
fn complex_fp(word: u32, scalar: bool, features: FeatureSet, out: &mut Instruction) {
    if scalar {
        return; // no scalar complex forms in this group.
    }
    let q = bit(word, 30);
    let size = bits(word, 22, 2);
    let rm = bits(word, 16, 5);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);
    // FCADD: word<15:12> == 1110x with word<11:10>==01? Actual layout:
    //   FCMLA: word<15:11> = 110 rot(2), word<11:10> = 01
    //   FCADD: word<15:12> = 1110,  rot1 = word<12>, word<11:10> = 01
    // Distinguish by word<15:13>.
    let hi = bits(word, 13, 3); // word<15:13>
    if hi == 0b110 {
        // FCMLA: rot = word<12:11>, degrees = rot*90.
        let rot = bits(word, 11, 2);
        let deg = (rot * 90) as u64;
        // size: 01=.4h/.8h(FP16), 10=.2s/.4s, 11=.2d
        let (arr, fp16) = match size {
            0b01 => (arr_fp16(q), true),
            0b10 => (if q == 1 { VA::V4S } else { VA::V2S }, false),
            0b11 => {
                if q == 0 {
                    return;
                }
                (VA::V2D, false)
            }
            _ => return,
        };
        if fp16 && !features.has(Feature::Fp16) {
            return;
        }
        out.set(Code::FcmlaVec);
        out.push_operand(vreg(rd, arr));
        out.push_operand(vreg(rn, arr));
        out.push_operand(vreg(rm, arr));
        out.push_operand(Operand::ImmUnsigned(deg));
        return;
    }
    if hi == 0b111 {
        // FCADD: rot bit = word<12>, degrees = 90 if 0 else 270.
        let rot1 = bit(word, 12);
        let deg = if rot1 == 0 { 90u64 } else { 270u64 };
        let (arr, fp16) = match size {
            0b01 => (arr_fp16(q), true),
            0b10 => (if q == 1 { VA::V4S } else { VA::V2S }, false),
            0b11 => {
                if q == 0 {
                    return;
                }
                (VA::V2D, false)
            }
            _ => return,
        };
        if fp16 && !features.has(Feature::Fp16) {
            return;
        }
        out.set(Code::FcaddVec);
        out.push_operand(vreg(rd, arr));
        out.push_operand(vreg(rn, arr));
        out.push_operand(vreg(rm, arr));
        out.push_operand(Operand::ImmUnsigned(deg));
    }
}

// ---------------------------------------------------------------------------
// Three different (widening / narrowing).
// ---------------------------------------------------------------------------

/// Advanced SIMD three-different (vector only). `U = word<29>`, `size =
/// word<23:22>`, `opcode = word<15:12>`. The `2`-suffixed forms use `Q==1` to
/// select the high half of the source for the widening/narrowing.
#[inline]
fn three_different(word: u32, scalar: bool, _features: FeatureSet, out: &mut Instruction) {
    let q = bit(word, 30);
    let u = bit(word, 29);
    let size = bits(word, 22, 2);
    let opcode = bits(word, 12, 4);
    let rm = bits(word, 16, 5);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);

    // Scalar three-different is only SQDMULL/SQDMLAL/SQDMLSL (U==0, opcode
    // 1101/1001/1011), with H/S element source and a doubled-width result.
    if scalar {
        let code = match (u, opcode) {
            (0, 0b1101) => Code::SqdmullVec,
            (0, 0b1001) => Code::SqdmlalVec,
            (0, 0b1011) => Code::SqdmlslVec,
            _ => return,
        };
        if size != 0b01 && size != 0b10 {
            return;
        }
        let eb = esize(size);
        out.set(code);
        out.push_operand(sca(rd, eb * 2));
        out.push_operand(sca(rn, eb));
        out.push_operand(sca(rm, eb));
        return;
    }

    // Wide element arrangement `Ta` (2× the `size` width) and narrow `Tb`.
    // The element size in `size` is the *narrow* operand size for the L/W/abal
    // long ops; for the narrowing ops (ADDHN/SUBHN/RADDHN/RSUBHN) it is the
    // narrow *result* size, while the sources are wide.
    let (code, shape) = match (u, opcode) {
        (0, 0b0000) => (Code::Saddl2Vec, Shape3D::LongL),
        (0, 0b0001) => (Code::Saddw2Vec, Shape3D::WideW),
        (0, 0b0010) => (Code::Ssubl2Vec, Shape3D::LongL),
        (0, 0b0011) => (Code::Ssubw2Vec, Shape3D::WideW),
        (0, 0b0100) => (Code::Addhn2Vec, Shape3D::NarrowHN),
        (0, 0b0101) => (Code::Sabal2Vec, Shape3D::LongL),
        (0, 0b0110) => (Code::Subhn2Vec, Shape3D::NarrowHN),
        (0, 0b0111) => (Code::Sabdl2Vec, Shape3D::LongL),
        (0, 0b1000) => (Code::Smlal2Vec, Shape3D::LongL),
        (0, 0b1001) => (Code::Sqdmlal2Vec, Shape3D::LongLSat),
        (0, 0b1010) => (Code::Smlsl2Vec, Shape3D::LongL),
        (0, 0b1011) => (Code::Sqdmlsl2Vec, Shape3D::LongLSat),
        (0, 0b1100) => (Code::Smull2Vec, Shape3D::LongL),
        (0, 0b1101) => (Code::Sqdmull2Vec, Shape3D::LongLSat),
        (0, 0b1110) => (Code::Pmull2Vec, Shape3D::Pmull),
        (1, 0b0000) => (Code::Uaddl2Vec, Shape3D::LongL),
        (1, 0b0001) => (Code::Uaddw2Vec, Shape3D::WideW),
        (1, 0b0010) => (Code::Usubl2Vec, Shape3D::LongL),
        (1, 0b0011) => (Code::Usubw2Vec, Shape3D::WideW),
        (1, 0b0100) => (Code::Raddhn2Vec, Shape3D::NarrowHN),
        (1, 0b0101) => (Code::Uabal2Vec, Shape3D::LongL),
        (1, 0b0110) => (Code::Rsubhn2Vec, Shape3D::NarrowHN),
        (1, 0b0111) => (Code::Uabdl2Vec, Shape3D::LongL),
        (1, 0b1000) => (Code::Umlal2Vec, Shape3D::LongL),
        (1, 0b1010) => (Code::Umlsl2Vec, Shape3D::LongL),
        (1, 0b1100) => (Code::Umull2Vec, Shape3D::LongL),
        _ => return,
    };

    // The `2`/non-`2` spelling is selected by Q; substitute the base mnemonic
    // when Q==0 by overriding the Code with its non-2 sibling.
    let code = if q == 0 { three_diff_base(code) } else { code };

    match shape {
        Shape3D::LongL | Shape3D::LongLSat => {
            // Long: Vd.<Ta>, Vn.<Tb>, Vm.<Tb>, where Ta is 2× the size width.
            // SQDMULL/SQDMLAL/SQDMLSL are defined for size 01/10 only.
            if matches!(shape, Shape3D::LongLSat) && !(size == 0b01 || size == 0b10) {
                return;
            }
            if size == 0b11 {
                return; // long source size of D would overflow to .1q (only PMULL).
            }
            let ta = wide_arr(size);
            let tb = arr_sizeq(size, q);
            out.set(code);
            out.push_operand(vreg(rd, ta));
            out.push_operand(vreg(rn, tb));
            out.push_operand(vreg(rm, tb));
        }
        Shape3D::WideW => {
            // Wide: Vd.<Ta>, Vn.<Ta>, Vm.<Tb>.
            if size == 0b11 {
                return;
            }
            let ta = wide_arr(size);
            let tb = arr_sizeq(size, q);
            out.set(code);
            out.push_operand(vreg(rd, ta));
            out.push_operand(vreg(rn, ta));
            out.push_operand(vreg(rm, tb));
        }
        Shape3D::NarrowHN => {
            // Narrowing: Vd.<Tb>, Vn.<Ta>, Vm.<Ta>; result narrow (size), sources
            // wide (2× size). The narrow result arrangement uses Q for hi-half.
            if size == 0b11 {
                return;
            }
            let ta = wide_arr(size);
            let tb = arr_sizeq(size, q);
            out.set(code);
            out.push_operand(vreg(rd, tb));
            out.push_operand(vreg(rn, ta));
            out.push_operand(vreg(rm, ta));
        }
        Shape3D::Pmull => {
            // PMULL{2}: Vd.<Ta>, Vn.<Tb>, Vm.<Tb>. Defined for size 00 (.8b/.16b
            // -> .8h) and size 11 (.1d/.2d -> .1q).
            let (ta, tb) = match size {
                0b00 => (VA::V8H, if q == 1 { VA::V16B } else { VA::V8B }),
                0b11 => (VA::V1Q, if q == 1 { VA::V2D } else { VA::V1D }),
                _ => return,
            };
            out.set(code);
            out.push_operand(vreg(rd, ta));
            out.push_operand(vreg(rn, tb));
            out.push_operand(vreg(rm, tb));
        }
    }
}

/// Operand shape of a three-different form.
#[derive(Clone, Copy)]
enum Shape3D {
    /// Long: `Vd.<2×>, Vn.<n>, Vm.<n>`.
    LongL,
    /// Long, saturating (SQDMULL/SQDMLAL/SQDMLSL): size 01/10 only.
    LongLSat,
    /// Wide: `Vd.<2×>, Vn.<2×>, Vm.<n>`.
    WideW,
    /// Narrowing high-half: `Vd.<n>, Vn.<2×>, Vm.<2×>`.
    NarrowHN,
    /// Polynomial long (PMULL/PMULL2).
    Pmull,
}

/// The full-width (`2×` element, 128-bit) arrangement for a narrow `size`:
/// B->8H, H->4S, S->2D. Used by the three-different long/narrow shapes whose
/// wide side is always the full 128-bit vector.
#[inline]
fn wide_arr(size: u32) -> VA {
    match size & 3 {
        0 => VA::V8H,
        1 => VA::V4S,
        _ => VA::V2D,
    }
}

/// The wide-element arrangement with the *same* register width as a narrow
/// `(size, Q)` source — i.e. the element doubles and the lane count halves
/// (`.8b`->`.4h`, `.16b`->`.8h`, `.2s`->`.1d`, `.4s`->`.2d`). Used by the
/// pairwise-long two-reg-misc ops (SADDLP/UADDLP/SADALP/UADALP).
#[inline]
fn wide_arr_q(size: u32, q: u32) -> VA {
    arr_sizeq((size & 3) + 1, q)
}

/// Map a `…2Vec` three-different [`Code`] to its non-`2` sibling (used when
/// `Q==0`). Codes without a `2` form are returned unchanged.
#[inline]
fn three_diff_base(code: Code) -> Code {
    match code {
        Code::Saddl2Vec => Code::SaddlVec,
        Code::Saddw2Vec => Code::SaddwVec,
        Code::Ssubl2Vec => Code::SsublVec,
        Code::Ssubw2Vec => Code::SsubwVec,
        Code::Addhn2Vec => Code::AddhnVec,
        Code::Sabal2Vec => Code::SabalVec,
        Code::Subhn2Vec => Code::SubhnVec,
        Code::Sabdl2Vec => Code::SabdlVec,
        Code::Smlal2Vec => Code::SmlalVec,
        Code::Sqdmlal2Vec => Code::SqdmlalVec,
        Code::Smlsl2Vec => Code::SmlslVec,
        Code::Sqdmlsl2Vec => Code::SqdmlslVec,
        Code::Smull2Vec => Code::SmullVec,
        Code::Sqdmull2Vec => Code::SqdmullVec,
        Code::Pmull2Vec => Code::PmullVec,
        Code::Uaddl2Vec => Code::UaddlVec,
        Code::Uaddw2Vec => Code::UaddwVec,
        Code::Usubl2Vec => Code::UsublVec,
        Code::Usubw2Vec => Code::UsubwVec,
        Code::Raddhn2Vec => Code::RaddhnVec,
        Code::Uabal2Vec => Code::UabalVec,
        Code::Rsubhn2Vec => Code::RsubhnVec,
        Code::Uabdl2Vec => Code::UabdlVec,
        Code::Umlal2Vec => Code::UmlalVec,
        Code::Umlsl2Vec => Code::UmlslVec,
        Code::Umull2Vec => Code::UmullVec,
        other => other,
    }
}

// ---------------------------------------------------------------------------
// Two-register miscellaneous.
// ---------------------------------------------------------------------------

/// Advanced SIMD two-register miscellaneous (vector and scalar). `U =
/// word<29>`, `size = word<23:22>`, `opcode = word<16:12>`.
#[inline]
fn two_reg_misc(word: u32, scalar: bool, features: FeatureSet, out: &mut Instruction) {
    let q = bit(word, 30);
    let u = bit(word, 29);
    let size = bits(word, 22, 2);
    let opcode = bits(word, 12, 5);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);

    // FP-vs-integer split for two-reg-misc (ARM ARM C4.1.97). The FP forms are:
    //   * the compare-against-zero / FABS / FNEG block `0b011xx` (with a==1),
    //   * the FCVTN/FCVTL/FCVTXN narrow/widen converts `0b1011x`,
    //   * the FRINT*/FCVT*/SCVTF/UCVTF/URECPE/FRECPE/URSQRTE/FRSQRTE/FSQRT block
    //     `0b11xxx`.
    // Everything else (REV*/SADDLP/SUQADD/CLS/CNT/SADALP/SQABS/ABS/XTN/SQXTN/
    // SQXTUN/UQXTN/NEG/CLZ/SQNEG/...) is the integer family.
    let is_fp = (opcode & 0b11100) == 0b01100
        || (opcode & 0b11110) == 0b10110
        || (opcode & 0b11000) == 0b11000;
    if is_fp {
        fp_two_reg_misc(word, scalar, q, u, size, opcode, rn, rd, features, out);
        return;
    }

    // Integer two-reg-misc.
    int_two_reg_misc(word, scalar, q, u, size, opcode, rn, rd, out);
}

/// Integer two-register miscellaneous (opcode `0b0xxxx`).
#[allow(clippy::too_many_arguments)]
#[inline]
fn int_two_reg_misc(
    word: u32,
    scalar: bool,
    q: u32,
    u: u32,
    size: u32,
    opcode: u32,
    rn: u32,
    rd: u32,
    out: &mut Instruction,
) {
    let _ = word;
    // Compare-against-zero forms carry a trailing `#0` immediate.
    // (CMGT/CMEQ/CMLT for U=0; CMGE/CMLE for U=1, plus ABS/NEG.)
    enum Misc {
        Same(Code),      // Vd.T, Vn.T
        CmpZero(Code),   // Vd.T, Vn.T, #0
        Narrow(Code),    // Vd.<Tb>, Vn.<Ta>  (XTN/SQXTN/...)
        Long(Code),      // SADDLP/UADDLP/SADALP/UADALP: Vd.<Ta>, Vn.<Tb>
        Rev(Code, u8),   // REV with element grouping (containers)
        ShllOp,          // SHLL/SHLL2 (special shift)
        Bitwise(Code),   // MVN/RBIT: Vd.T, Vn.T with T always .8b/.16b
    }

    let m = match (u, opcode) {
        (0, 0b00000) => Misc::Rev(Code::Rev64Vec, 64),
        (0, 0b00001) => Misc::Rev(Code::Rev16Vec, 16),
        (0, 0b00010) => Misc::Long(Code::SaddlpVec),
        (0, 0b00011) => Misc::Same(Code::SuqaddVec),
        (0, 0b00100) => Misc::Same(Code::ClsVec),
        (0, 0b00101) => Misc::Same(Code::CntVec),
        (0, 0b00110) => Misc::Long(Code::SadalpVec),
        (0, 0b00111) => Misc::Same(Code::SqabsVec),
        (0, 0b01000) => Misc::CmpZero(Code::CmgtVec),
        (0, 0b01001) => Misc::CmpZero(Code::CmeqVec),
        (0, 0b01010) => Misc::CmpZero(Code::CmltVec),
        (0, 0b01011) => Misc::Same(Code::AbsVec),
        (0, 0b10010) => Misc::Narrow(Code::XtnVec),
        (0, 0b10100) => Misc::Narrow(Code::SqxtnVec),
        (1, 0b00000) => Misc::Rev(Code::Rev32Vec, 32),
        (1, 0b00010) => Misc::Long(Code::UaddlpVec),
        (1, 0b00011) => Misc::Same(Code::UsqaddVec),
        (1, 0b00100) => Misc::Same(Code::ClzVec),
        // NOT (preferred MVN) for size==00, RBIT for size==01 (U=1, opcode 00101).
        (1, 0b00101) => match size {
            0b00 => Misc::Bitwise(Code::MvnVec),
            0b01 => Misc::Bitwise(Code::RbitVec),
            _ => return,
        },
        (1, 0b00110) => Misc::Long(Code::UadalpVec),
        (1, 0b00111) => Misc::Same(Code::SqnegVec),
        (1, 0b01000) => Misc::CmpZero(Code::CmgeVec),
        (1, 0b01001) => Misc::CmpZero(Code::CmleVec),
        (1, 0b01011) => Misc::Same(Code::NegVec),
        (1, 0b10010) => Misc::Narrow(Code::SqxtunVec),
        (1, 0b10011) => Misc::ShllOp, // SHLL/SHLL2 (U=1, opcode 10011)
        (1, 0b10100) => Misc::Narrow(Code::UqxtnVec),
        _ => return,
    };

    match m {
        Misc::Same(code) => {
            // CLS/CLZ/CNT/REV are byte/half/word only; SUQADD/USQADD/SQABS/SQNEG
            // accept any size; ABS/NEG accept any size (incl. D).
            if scalar {
                // Scalar SUQADD/USQADD/SQABS/SQNEG (any size), ABS/NEG (D only).
                let eb = match code {
                    Code::AbsVec | Code::NegVec => {
                        if size == 0b11 {
                            64
                        } else {
                            return;
                        }
                    }
                    _ => esize(size),
                };
                out.set(code);
                out.push_operand(sca(rd, eb));
                out.push_operand(sca(rn, eb));
                return;
            }
            // CNT is byte-only; CLS/CLZ are B/H/S; ABS/NEG/SQABS/SQNEG/SUQADD/
            // USQADD are B/H/S/D.
            if !int_misc_size_ok(code, size, q) {
                return;
            }
            let arr = arr_sizeq(size, q);
            out.set(code);
            out.push_operand(vreg(rd, arr));
            out.push_operand(vreg(rn, arr));
        }
        Misc::CmpZero(code) => {
            if scalar {
                // Scalar compare-zero is doubleword only.
                if size != 0b11 {
                    return;
                }
                out.set(code);
                out.push_operand(sca(rd, 64));
                out.push_operand(sca(rn, 64));
                out.push_operand(Operand::ImmUnsigned(0));
                return;
            }
            if size == 0b11 && q == 0 {
                return;
            }
            let arr = arr_sizeq(size, q);
            out.set(code);
            out.push_operand(vreg(rd, arr));
            out.push_operand(vreg(rn, arr));
            out.push_operand(Operand::ImmUnsigned(0));
        }
        Misc::Narrow(code) => {
            // XTN/SQXTN/UQXTN/SQXTUN: Vd.<Tb>, Vn.<Ta>. size is the *result*
            // (narrow) size; source is 2× wide. size==11 invalid (would narrow
            // from .1q).
            if size == 0b11 {
                return;
            }
            if scalar {
                // Scalar SQXTN/UQXTN/SQXTUN: result eb, source 2× eb.
                let eb = esize(size);
                let src_eb = eb * 2;
                out.set(code);
                out.push_operand(sca(rd, eb));
                out.push_operand(sca(rn, src_eb));
                return;
            }
            // Q==1 selects the high-half `…2` form (e.g. `xtn2`/`sqxtn2`).
            let code = if q == 1 { narrow_to_2(code) } else { code };
            let tb = arr_sizeq(size, q); // narrow result (Q -> hi half / "2")
            let ta = wide_arr(size); // wide source
            out.set(code);
            out.push_operand(vreg(rd, tb));
            out.push_operand(vreg(rn, ta));
        }
        Misc::Long(code) => {
            // SADDLP/UADDLP/SADALP/UADALP: Vd.<Ta>, Vn.<Tb>; the destination has
            // the doubled element width but the *same* register width (so half
            // the lane count): `.8b`->`.4h`, `.16b`->`.8h`, etc. `size` is the
            // source (narrow) size; size==11 invalid.
            if size == 0b11 {
                return;
            }
            let ta = wide_arr_q(size, q);
            let tb = arr_sizeq(size, q);
            out.set(code);
            out.push_operand(vreg(rd, ta));
            out.push_operand(vreg(rn, tb));
        }
        Misc::Rev(code, container) => {
            // REV64/REV32/REV16: byte-ish elements grouped into containers; the
            // element arrangement is the byte/half/word size, the container size
            // restricts validity. REV64 valid for B/H/S; REV32 for B/H; REV16 B.
            let ok = match container {
                64 => size <= 0b10,
                32 => size <= 0b01,
                _ => size == 0b00,
            };
            if !ok {
                return;
            }
            let arr = arr_sizeq(size, q);
            out.set(code);
            out.push_operand(vreg(rd, arr));
            out.push_operand(vreg(rn, arr));
        }
        Misc::ShllOp => {
            // SHLL/SHLL2 <Vd>.<Ta>, <Vn>.<Tb>, #shift  (shift = element width).
            if size == 0b11 {
                return;
            }
            let ta = wide_arr(size);
            let tb = arr_sizeq(size, q);
            let shift = esize(size) as u64; // 8/16/32
            out.set(if q == 1 { Code::Shll2Vec } else { Code::ShllVec });
            out.push_operand(vreg(rd, ta));
            out.push_operand(vreg(rn, tb));
            out.push_operand(Operand::ImmUnsigned(shift));
        }
        Misc::Bitwise(code) => {
            // MVN (NOT) / RBIT: Vd.<T>, Vn.<T> with T always .8b/.16b. No scalar
            // form. The size-field validity was checked when selecting the code.
            if scalar {
                return;
            }
            let arr = if q == 1 { VA::V16B } else { VA::V8B };
            out.set(code);
            out.push_operand(vreg(rd, arr));
            out.push_operand(vreg(rn, arr));
        }
    }
}

/// Whether an integer two-reg-misc `code` is allocated for `(size, Q)`.
#[inline]
fn int_misc_size_ok(code: Code, size: u32, q: u32) -> bool {
    match code {
        // CNT: byte-only.
        Code::CntVec => size == 0b00,
        // CLS/CLZ: B/H/S.
        Code::ClsVec | Code::ClzVec => size <= 0b10,
        // ABS/NEG/SQABS/SQNEG/SUQADD/USQADD: B/H/S/D, but `.1d` invalid.
        Code::AbsVec | Code::NegVec | Code::SqabsVec | Code::SqnegVec | Code::SuqaddVec | Code::UsqaddVec => {
            !(size == 0b11 && q == 0)
        }
        _ => !(size == 0b11 && q == 0),
    }
}

/// FP two-register miscellaneous (opcode `0b1xxxx`), vector and scalar,
/// single/double and (FEAT_FP16) half precision.
#[allow(clippy::too_many_arguments)]
#[inline]
fn fp_two_reg_misc(
    word: u32,
    scalar: bool,
    q: u32,
    u: u32,
    size: u32,
    opcode: u32,
    rn: u32,
    rd: u32,
    features: FeatureSet,
    out: &mut Instruction,
) {
    let _ = size;
    // FP two-reg-misc: the opcode-group bit is `a = size<1>` (word<23>) while
    // the *precision* is `sz = size<0>` (word<22>): `0` -> single (`.2s`/`.4s`),
    // `1` -> double (`.2d`). The corpus carries no half-precision members of
    // this family (FP16 misc is a separate encoding), so only S/D is produced.
    let a = bit(word, 23);
    let sz = bit(word, 22);

    // Scalar-only FP two-reg-misc forms:
    //   FRECPX <V><d>, <V><n>   (op 11111, a==1, U==0; same width S/D),
    //   FCVTXN <Sd>, <Dn>       (op 10110, U==1; narrows double to single).
    if scalar {
        if u == 0 && a == 1 && opcode == 0b11111 {
            let eb: u16 = if sz == 1 { 64 } else { 32 };
            out.set(Code::FrecpxVec);
            out.push_operand(sca(rd, eb));
            out.push_operand(sca(rn, eb));
            return;
        }
        if u == 1 && opcode == 0b10110 {
            // FCVTXN scalar: source D (64), dest S (32).
            out.set(Code::FcvtxnVec);
            out.push_operand(sca(rd, 32));
            out.push_operand(sca(rn, 64));
            return;
        }
    }

    // FCVTL/FCVTN/FCVTXN have bespoke widen/narrow shapes (vector only).
    if let Some(special) = fp_misc_special(u, a, opcode) {
        if !scalar {
            // sz here selects half<->single (0) vs single<->double (1).
            fp_misc_widenarrow(special, q, sz, rd, rn, features, out);
        }
        return;
    }

    let code = match fp_misc_code(u, a, opcode) {
        Some(c) => c,
        None => return,
    };

    // Compare-against-zero FP forms carry a trailing `#0.0`.
    let is_cmp_zero = matches!(
        code,
        Code::FcmgtVec | Code::FcmgeVec | Code::FcmeqVec | Code::FcmleVec | Code::FcmltVec
    );

    if scalar {
        let eb: u16 = if sz == 1 { 64 } else { 32 };
        out.set(code);
        out.push_operand(sca(rd, eb));
        out.push_operand(sca(rn, eb));
        if is_cmp_zero {
            out.push_operand(Operand::FpImm(0.0));
        }
        return;
    }
    // Vector: precision from sz (S/D); `.2d` requires Q==1, `.1d` invalid.
    let arr = match arr_fp(sz, q) {
        Some(x) => x,
        None => return,
    };
    out.set(code);
    out.push_operand(vreg(rd, arr));
    out.push_operand(vreg(rn, arr));
    if is_cmp_zero {
        out.push_operand(Operand::FpImm(0.0));
    }
}

/// Half-precision (`FEAT_FP16`) two-register miscellaneous, the
/// `bits<21:17>==11100` slot. Reuses the SP/DP opcode table (the opcode->op
/// mapping is identical) but renders with a half element width (`.4h`/`.8h`
/// vector, `h` scalar). Compare-against-zero forms carry a trailing `#0.0`.
#[inline]
fn fp16_two_reg_misc(word: u32, scalar: bool, features: FeatureSet, out: &mut Instruction) {
    if !features.has(Feature::Fp16) {
        return;
    }
    let q = bit(word, 30);
    let u = bit(word, 29);
    let a = bit(word, 23);
    let opcode = bits(word, 12, 5);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);

    // Scalar-only half-precision FRECPX (op 11111, a==1, U==0).
    if scalar && u == 0 && a == 1 && opcode == 0b11111 {
        out.set(Code::FrecpxVec);
        out.push_operand(sca(rd, 16));
        out.push_operand(sca(rn, 16));
        return;
    }

    let code = match fp_misc_code(u, a, opcode) {
        Some(c) => c,
        None => return,
    };
    let is_cmp_zero = matches!(
        code,
        Code::FcmgtVec | Code::FcmgeVec | Code::FcmeqVec | Code::FcmleVec | Code::FcmltVec
    );
    if scalar {
        out.set(code);
        out.push_operand(sca(rd, 16));
        out.push_operand(sca(rn, 16));
        if is_cmp_zero {
            out.push_operand(Operand::FpImm(0.0));
        }
        return;
    }
    let arr = arr_fp16(q);
    out.set(code);
    out.push_operand(vreg(rd, arr));
    out.push_operand(vreg(rn, arr));
    if is_cmp_zero {
        out.push_operand(Operand::FpImm(0.0));
    }
}

/// FP two-reg-misc `(U, a=word<23>, opcode=word<16:12>)` -> [`Code`] for the
/// single/double family (precision is carried by `a`). Pinned to the corpus.
#[inline]
fn fp_misc_code(u: u32, a: u32, opcode: u32) -> Option<Code> {
    let c = match (u, a, opcode) {
        // a==0 (single) region.
        (0, 0, 0b11000) => Code::FrintnVec,
        (0, 0, 0b11001) => Code::FrintmVec,
        (0, 0, 0b11010) => Code::FcvtnsVec,
        (0, 0, 0b11011) => Code::FcvtmsVec,
        (0, 0, 0b11100) => Code::FcvtasVec,
        (0, 0, 0b11101) => Code::ScvtfVec,
        (0, 0, 0b11110) => Code::Frint32zVec,
        (0, 0, 0b11111) => Code::Frint64zVec,
        // a==1 (double) region.
        (0, 1, 0b01100) => Code::FcmgtVec, // FCMGT #0
        (0, 1, 0b01101) => Code::FcmeqVec, // FCMEQ #0
        (0, 1, 0b01110) => Code::FcmltVec, // FCMLT #0
        (0, 1, 0b01111) => Code::FabsVec,
        (0, 1, 0b11000) => Code::FrintpVec,
        (0, 1, 0b11001) => Code::FrintzVec,
        (0, 1, 0b11010) => Code::FcvtpsVec,
        (0, 1, 0b11011) => Code::FcvtzsVec,
        (0, 1, 0b11100) => Code::UrecpeVec,
        (0, 1, 0b11101) => Code::FrecpeVec,
        // U==1, a==0 region.
        (1, 0, 0b11000) => Code::FrintaVec,
        (1, 0, 0b11001) => Code::FrintxVec,
        (1, 0, 0b11010) => Code::FcvtnuVec,
        (1, 0, 0b11011) => Code::FcvtmuVec,
        (1, 0, 0b11100) => Code::FcvtauVec,
        (1, 0, 0b11101) => Code::UcvtfVec,
        (1, 0, 0b11110) => Code::Frint32xVec,
        (1, 0, 0b11111) => Code::Frint64xVec,
        // U==1, a==1 region.
        (1, 1, 0b01100) => Code::FcmgeVec, // FCMGE #0
        (1, 1, 0b01101) => Code::FcmleVec, // FCMLE #0
        (1, 1, 0b01111) => Code::FnegVec,
        (1, 1, 0b11001) => Code::FrintiVec,
        (1, 1, 0b11010) => Code::FcvtpuVec,
        (1, 1, 0b11011) => Code::FcvtzuVec,
        (1, 1, 0b11100) => Code::UrsqrteVec,
        (1, 1, 0b11101) => Code::FrsqrteVec,
        (1, 1, 0b11111) => Code::FsqrtVec,
        _ => return None,
    };
    Some(c)
}

/// Identify the widen/narrow FP-convert specials by `(U,a,opcode)`.
#[inline]
fn fp_misc_special(u: u32, a: u32, opcode: u32) -> Option<FpSpecial> {
    match (u, a, opcode) {
        (0, 0, 0b10111) => Some(FpSpecial::Fcvtl),
        (0, 0, 0b10110) => Some(FpSpecial::Fcvtn),
        (1, 0, 0b10110) => Some(FpSpecial::Fcvtxn),
        _ => None,
    }
}

/// Widen/narrow FP convert family tag.
#[derive(Clone, Copy)]
enum FpSpecial {
    Fcvtl,
    Fcvtn,
    Fcvtxn,
}

/// Render the FCVTL/FCVTN/FCVTXN widen/narrow forms (with their `2` hi-half
/// variants selected by `Q`).
#[inline]
fn fp_misc_widenarrow(
    special: FpSpecial,
    q: u32,
    sz: u32,
    rd: u32,
    rn: u32,
    _features: FeatureSet,
    out: &mut Instruction,
) {
    // sz==0: half<->single (.4h/.8h <-> .4s); sz==1: single<->double (.2s/.4s
    // <-> .2d).
    match special {
        FpSpecial::Fcvtl => {
            // FCVTL{2} <Vd>.<Ta>, <Vn>.<Tb> : widen. Tb is the narrow source.
            let (ta, tb) = if sz == 0 {
                (VA::V4S, if q == 1 { VA::V8H } else { VA::V4H })
            } else {
                (VA::V2D, if q == 1 { VA::V4S } else { VA::V2S })
            };
            out.set(if q == 1 { Code::Fcvtl2Vec } else { Code::FcvtlVec });
            out.push_operand(vreg(rd, ta));
            out.push_operand(vreg(rn, tb));
        }
        FpSpecial::Fcvtn => {
            // FCVTN{2} <Vd>.<Tb>, <Vn>.<Ta> : narrow.
            let (ta, tb) = if sz == 0 {
                (VA::V4S, if q == 1 { VA::V8H } else { VA::V4H })
            } else {
                (VA::V2D, if q == 1 { VA::V4S } else { VA::V2S })
            };
            out.set(if q == 1 { Code::Fcvtn2Vec } else { Code::FcvtnVec });
            out.push_operand(vreg(rd, tb));
            out.push_operand(vreg(rn, ta));
        }
        FpSpecial::Fcvtxn => {
            // FCVTXN{2} <Vd>.<Tb>, <Vn>.2d : narrow double->single round-to-odd.
            let ta = VA::V2D;
            let tb = if q == 1 { VA::V4S } else { VA::V2S };
            out.set(if q == 1 { Code::Fcvtxn2Vec } else { Code::FcvtxnVec });
            out.push_operand(vreg(rd, tb));
            out.push_operand(vreg(rn, ta));
        }
    }
}

// ---------------------------------------------------------------------------
// Scalar pairwise.
// ---------------------------------------------------------------------------

/// Advanced SIMD scalar pairwise reductions (`asisd`, `bits<21:17>==11000`):
/// `ADDP <V><d>, <Vn>.<T>` and the FP `FADDP`/`FMAXP`/`FMINP`/`FMAXNMP`/
/// `FMINNMP`, each reducing a 2-element vector to one scalar. `U = word<29>`,
/// `size = word<23:22>`, `opcode = word<16:12>`.
#[inline]
fn scalar_pairwise(word: u32, features: FeatureSet, out: &mut Instruction) {
    let u = bit(word, 29);
    let size = bits(word, 22, 2);
    let opcode = bits(word, 12, 5);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);

    // Integer ADDP (scalar): U==0, opcode 11011, doubleword only — reduces a
    // `.2d` pair to a `D` scalar.
    if u == 0 && opcode == 0b11011 {
        if size != 0b11 {
            return;
        }
        out.set(Code::AddpVec);
        out.push_operand(sca(rd, 64));
        out.push_operand(vreg(rn, VA::V2D));
        return;
    }

    // FP scalar pairwise. The precision is `sz = size<0>` for FADDP/FMAXP/FMINP/
    // FMAXNMP/FMINNMP in the single/double rows (`U==1`), and half precision in
    // the `U==0` rows (FEAT_FP16). `size<1>` selects max vs min for the
    // min/max-style ops.
    let half = u == 0; // U==0 rows are the FP16 family here.
    let is_min = bit(size, 1) == 1;
    let code = match opcode {
        0b01101 => Code::FaddpVec,
        0b01100 => {
            if is_min {
                Code::FminnmpVec
            } else {
                Code::FmaxnmpVec
            }
        }
        0b01111 => {
            if is_min {
                Code::FminpVec
            } else {
                Code::FmaxpVec
            }
        }
        _ => return,
    };

    let (eb, arr) = if half {
        if !features.has(Feature::Fp16) {
            return;
        }
        (16u16, VA::V2H)
    } else if bit(size, 0) == 0 {
        (32, VA::V2S)
    } else {
        (64, VA::V2D)
    };
    out.set(code);
    out.push_operand(sca(rd, eb));
    out.push_operand(vreg(rn, arr));
}

// ---------------------------------------------------------------------------
// Across lanes.
// ---------------------------------------------------------------------------

/// Advanced SIMD across-lanes reductions (vector source, scalar dest). `U =
/// word<29>`, `size = word<23:22>`, `opcode = word<16:12>`.
#[inline]
fn across_lanes(word: u32, scalar: bool, features: FeatureSet, out: &mut Instruction) {
    if scalar {
        // The scalar (asisd) `bits<21:17>==11000` slot is the scalar-pairwise
        // reduction family (ADDP / FADDP / FMAXP / FMINP / FMAXNMP / FMINNMP),
        // not across-lanes.
        scalar_pairwise(word, features, out);
        return;
    }
    let q = bit(word, 30);
    let u = bit(word, 29);
    let size = bits(word, 22, 2);
    let opcode = bits(word, 12, 5);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);

    // FP across-lanes (FMAXNMV/FMAXV/FMINNMV/FMINV): opcode 01100/01111. `U`
    // selects half-precision (`0`, dest H, source `.4h`/`.8h`, FEAT_FP16) vs
    // single-precision (`1`, dest S, source `.4s`, Q must be 1); `a=word<23>`
    // selects max (`0`) vs min (`1`); opcode 01100 is the NM variant.
    if opcode == 0b01100 || opcode == 0b01111 {
        let a = bit(word, 23);
        let is_nm = opcode == 0b01100;
        let code = match (a, is_nm) {
            (0, true) => Code::FmaxnmvVec,
            (0, false) => Code::FmaxvVec,
            (1, true) => Code::FminnmvVec,
            (_, _) => Code::FminvVec,
        };
        if u == 0 {
            // Half precision.
            if !features.has(Feature::Fp16) {
                return;
            }
            let arr = arr_fp16(q);
            out.set(code);
            out.push_operand(sca(rd, 16));
            out.push_operand(vreg(rn, arr));
        } else {
            // Single precision: source `.4s` (Q must be 1), dest S.
            if q == 0 {
                return;
            }
            out.set(code);
            out.push_operand(sca(rd, 32));
            out.push_operand(vreg(rn, VA::V4S));
        }
        return;
    }

    // Integer across-lanes.
    let code = match (u, opcode) {
        (0, 0b00011) => Code::SaddlvVec,
        (0, 0b01010) => Code::SmaxvVec,
        (0, 0b11010) => Code::SminvVec,
        (0, 0b11011) => Code::AddvVec,
        (1, 0b00011) => Code::UaddlvVec,
        (1, 0b01010) => Code::UmaxvVec,
        (1, 0b11010) => Code::UminvVec,
        _ => return,
    };

    // ADDV/SMAXV/SMINV/UMAXV/UMINV: dest scalar of element width `size`; source
    // arrangement of element width `size`. Valid sizes: B/H with Q either,
    // S with Q==1 (8B/16B/4H/8H/4S). `.2s`/`.1d`/`.2d` invalid.
    // SADDLV/UADDLV: dest is *double* the source element width.
    let is_long = matches!(code, Code::SaddlvVec | Code::UaddlvVec);

    // Validate (size, Q): size==11 invalid; size==10 requires Q==1.
    if size == 0b11 {
        return;
    }
    if size == 0b10 && q == 0 {
        return;
    }
    let src = arr_sizeq(size, q);
    let dst_eb = if is_long { esize(size) * 2 } else { esize(size) };
    out.set(code);
    out.push_operand(sca(rd, dst_eb));
    out.push_operand(vreg(rn, src));
}

// ---------------------------------------------------------------------------
// By element (vector × indexed element).
// ---------------------------------------------------------------------------

/// Advanced SIMD "by element" (`01111`/`11111`). `U = word<29>`, `size =
/// word<23:22>`, `opcode = word<15:12>`, with the indexed element `Vm.<Ts>[i]`
/// built from `H:L:M` and the size.
#[inline]
fn decode_by_element(word: u32, scalar: bool, features: FeatureSet, out: &mut Instruction) {
    // By-element is distinguished from the shift-by-immediate and modified-
    // immediate encodings (which share the `x1111` top space) by `word<10>==0`.
    if bit(word, 10) != 0 {
        return;
    }
    let q = bit(word, 30);
    let u = bit(word, 29);
    let size = bits(word, 22, 2);
    let opcode = bits(word, 12, 4);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);

    // Decode the indexed Vm + index per element size.
    let (vm, index, idx_arr) = decode_index(word, size);

    // The FP by-element ops (FMLA/FMLS/FMUL/FMULX) and FMLAL/FMLSL families use
    // the FP arrangement; the integer ones (MLA/MLS/MUL/SMULL/SQDMULL/...) use
    // the integer arrangement.
    #[derive(Clone, Copy)]
    enum ByEl {
        SameInt(Code),     // MUL/MLA/MLS: Vd.T, Vn.T, Vm.Ts[i]
        SameFp(Code),      // FMUL/FMLA/...: Vd.T(fp), Vn.T(fp), Vm.Ts[i]
        LongInt(Code),     // SMULL/UMULL/SMLAL/...: Vd.<2×>, Vn.<n>, Vm.Ts[i]
        LongSat(Code),     // SQDMULL/SQDMLAL/SQDMLSL
        SatSame(Code),     // SQDMULH/SQRDMULH/SQRDMLAH/SQRDMLSH
        Fmlal(Code),       // FMLAL/FMLSL (widening .2s/.4s <- .2h/.4h)
        Dot(Code),         // SDOT/UDOT by element: Vd.<2s/4s>, Vn.<8b/16b>, Vm.4b[i]
    }

    // FCMLA by element (U=1, opcode `RR01` with rotate `RR=word<14:13>`): a
    // bespoke shape with a trailing `#rot`. Handle it before the main table.
    if u == 1 && (opcode & 0b1001) == 0b0001 {
        by_element_fcmla(word, scalar, features, out);
        return;
    }

    let kind = match (u, opcode) {
        (0, 0b0000) => Some(ByEl::Fmlal(Code::FmlalVec)),
        (0, 0b0001) => Some(ByEl::SameFp(Code::FmlaVec)),
        (0, 0b0010) => Some(ByEl::LongInt(Code::SmlalVec)),
        (0, 0b0011) => Some(ByEl::LongSat(Code::SqdmlalVec)),
        (0, 0b0100) => Some(ByEl::Fmlal(Code::FmlslVec)),
        (0, 0b0101) => Some(ByEl::SameFp(Code::FmlsVec)),
        (0, 0b0110) => Some(ByEl::LongInt(Code::SmlslVec)),
        (0, 0b0111) => Some(ByEl::LongSat(Code::SqdmlslVec)),
        (0, 0b1000) => Some(ByEl::SameInt(Code::MulVec)),
        (0, 0b1001) => Some(ByEl::SameFp(Code::FmulVec)),
        (0, 0b1010) => Some(ByEl::LongInt(Code::SmullVec)),
        (0, 0b1011) => Some(ByEl::LongSat(Code::SqdmullVec)),
        (0, 0b1100) => Some(ByEl::SatSame(Code::SqdmulhVec)),
        (0, 0b1101) => Some(ByEl::SatSame(Code::SqrdmulhVec)),
        (0, 0b1110) => Some(ByEl::Dot(Code::SdotIdx)),
        (1, 0b1110) => Some(ByEl::Dot(Code::UdotIdx)),
        (1, 0b0000) => Some(ByEl::SameInt(Code::MlaVec)),
        (1, 0b0010) => Some(ByEl::LongInt(Code::UmlalVec)),
        (1, 0b0100) => Some(ByEl::SameInt(Code::MlsVec)),
        (1, 0b0110) => Some(ByEl::LongInt(Code::UmlslVec)),
        (1, 0b1000) => Some(ByEl::Fmlal(Code::Fmlal2Vec)),
        (1, 0b1001) => Some(ByEl::SameFp(Code::FmulxVec)),
        (1, 0b1010) => Some(ByEl::LongInt(Code::UmullVec)),
        (1, 0b1100) => Some(ByEl::Fmlal(Code::Fmlsl2Vec)),
        (1, 0b1101) => Some(ByEl::SatSame(Code::SqrdmlahVec)),
        (1, 0b1111) => Some(ByEl::SatSame(Code::SqrdmlshVec)),
        _ => None,
    };
    let kind = match kind {
        Some(k) => k,
        None => return,
    };

    match kind {
        ByEl::SameInt(code) => {
            // MUL/MLA/MLS by element: vector Vd.T, Vn.T, Vm.Ts[i]; element H/S
            // only (size 01/10).
            if size != 0b01 && size != 0b10 {
                return;
            }
            if scalar {
                return; // no scalar MUL-by-element.
            }
            let arr = arr_sizeq(size, q);
            out.set(code);
            out.push_operand(vreg(rd, arr));
            out.push_operand(vreg(rn, arr));
            out.push_operand(vreg_idx(vm, idx_arr, index));
        }
        ByEl::SameFp(code) => {
            // FMLA/FMLS/FMUL/FMULX by element. The FP precision is the `size`
            // field: `00` -> half (FEAT_FP16), `10` -> single, `11` -> double.
            if size == 0b00 {
                // FP16 by element: index is H:L:M with a 4-bit Vm.
                if !features.has(Feature::Fp16) {
                    return;
                }
                let (vm_h, idx_h) = decode_index_h(word);
                if scalar {
                    out.set(code);
                    out.push_operand(sca(rd, 16));
                    out.push_operand(sca(rn, 16));
                    out.push_operand(vreg_idx(vm_h, VA::Sh, idx_h));
                    return;
                }
                let arr = arr_fp16(q);
                out.set(code);
                out.push_operand(vreg(rd, arr));
                out.push_operand(vreg(rn, arr));
                out.push_operand(vreg_idx(vm_h, VA::Sh, idx_h));
                return;
            }
            if size == 0b01 {
                return; // unallocated FP-by-element precision.
            }
            // single/double.
            if scalar {
                let eb: u16 = if size == 0b11 { 64 } else { 32 };
                out.set(code);
                out.push_operand(sca(rd, eb));
                out.push_operand(sca(rn, eb));
                out.push_operand(vreg_idx(vm, idx_arr, index));
                return;
            }
            let arr = match arr_fp(bit(size, 0), q) {
                Some(a) => a,
                None => return,
            };
            out.set(code);
            out.push_operand(vreg(rd, arr));
            out.push_operand(vreg(rn, arr));
            out.push_operand(vreg_idx(vm, idx_arr, index));
        }
        ByEl::LongInt(code) | ByEl::LongSat(code) => {
            // Long by element: Vd.<2×>, Vn.<n>, Vm.Ts[i]; element H/S only.
            if size != 0b01 && size != 0b10 {
                return;
            }
            let is_sat = matches!(kind, ByEl::LongSat(_));
            if scalar {
                if !is_sat {
                    return; // only SQDMULL/SQDMLAL/SQDMLSL have scalar forms.
                }
                // Scalar long: result 2× element, source element width.
                let eb = esize(size);
                out.set(code);
                out.push_operand(sca(rd, eb * 2));
                out.push_operand(sca(rn, eb));
                out.push_operand(vreg_idx(vm, idx_arr, index));
                return;
            }
            let code = if q == 1 { long_to_2(code) } else { code };
            let ta = wide_arr(size);
            let tb = arr_sizeq(size, q);
            out.set(code);
            out.push_operand(vreg(rd, ta));
            out.push_operand(vreg(rn, tb));
            out.push_operand(vreg_idx(vm, idx_arr, index));
        }
        ByEl::SatSame(code) => {
            // SQDMULH/SQRDMULH/SQRDMLAH/SQRDMLSH by element: same shape; H/S.
            if size != 0b01 && size != 0b10 {
                return;
            }
            if scalar {
                let eb = esize(size);
                out.set(code);
                out.push_operand(sca(rd, eb));
                out.push_operand(sca(rn, eb));
                out.push_operand(vreg_idx(vm, idx_arr, index));
                return;
            }
            let arr = arr_sizeq(size, q);
            out.set(code);
            out.push_operand(vreg(rd, arr));
            out.push_operand(vreg(rn, arr));
            out.push_operand(vreg_idx(vm, idx_arr, index));
        }
        ByEl::Fmlal(code) => {
            // FMLAL/FMLSL/FMLAL2/FMLSL2 by element: Vd.<2s/4s>, Vn.<2h/4h>,
            // Vm.h[i]; the index uses the H:L:M half-precision form.
            if scalar {
                return;
            }
            let (ta, tb) = if q == 1 { (VA::V4S, VA::V4H) } else { (VA::V2S, VA::V2H) };
            // For FMLAL the index is always a half-element index.
            let (vm_h, idx_h) = decode_index_h(word);
            out.set(code);
            out.push_operand(vreg(rd, ta));
            out.push_operand(vreg(rn, tb));
            out.push_operand(vreg_idx(vm_h, VA::Sh, idx_h));
        }
        ByEl::Dot(code) => {
            // SDOT/UDOT by element: Vd.<2s/4s>, Vn.<8b/16b>, Vm.4b[index].
            // Fixed `size==10`; the index is H:L (2-bit) with a full 5-bit Vm,
            // and the indexed element is always the `.4b` group.
            if size != 0b10 || scalar {
                return;
            }
            let ta = arr_sizeq(0b10, q); // .2s / .4s
            let tb = arr_sizeq(0b00, q); // .8b / .16b
            out.set(code);
            out.push_operand(vreg(rd, ta));
            out.push_operand(vreg(rn, tb));
            out.push_operand(vreg_idx(vm, VA::V4B, index));
        }
    }
}

/// FCMLA by element: `FCMLA <Vd>.<T>, <Vn>.<T>, <Vm>.<Ts>[index], #rot`. Defined
/// for `.4h`/`.8h` (size=01, FEAT_FP16) and `.2s`/`.4s` (size=10). The index
/// addresses complex pairs (`.h`: H:L / 4-bit Vm; `.s`: H / 5-bit Vm), and the
/// rotate `word<14:13>` renders as `0`/`90`/`180`/`270`.
#[inline]
fn by_element_fcmla(word: u32, scalar: bool, features: FeatureSet, out: &mut Instruction) {
    if scalar {
        return;
    }
    let q = bit(word, 30);
    let size = bits(word, 22, 2);
    let rot = bits(word, 13, 2);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);
    let deg = (rot * 90) as u64;
    let h = bit(word, 11);
    let l = bit(word, 21);

    let (arr, ts, vm, index) = match size {
        0b01 => {
            if !features.has(Feature::Fp16) {
                return;
            }
            // FCMLA `.h`: index = H:L (complex pairs), Vm is 5-bit (bits<20:16>).
            let vm = bits(word, 16, 5);
            (arr_fp16(q), VA::Sh, vm, ((h << 1) | l) as u8)
        }
        0b10 => {
            if q == 0 {
                return; // FCMLA .2s by element is not allocated.
            }
            // FCMLA `.s`: index = H, Vm is 5-bit.
            let vm = bits(word, 16, 5);
            (VA::V4S, VA::Ss, vm, h as u8)
        }
        _ => return,
    };
    out.set(Code::FcmlaVec);
    out.push_operand(vreg(rd, arr));
    out.push_operand(vreg(rn, arr));
    out.push_operand(vreg_idx(vm, ts, index));
    out.push_operand(Operand::ImmUnsigned(deg));
}

/// Decode the indexed `Vm` register, element index, and the indexed-element
/// arrangement (`.h`/`.s`/`.d`) for a by-element encoding of `size`.
#[inline]
fn decode_index(word: u32, size: u32) -> (u32, u8, VA) {
    let l = bit(word, 21);
    let m = bit(word, 20);
    let h = bit(word, 11);
    match size {
        0b01 => {
            // .h: index = H:L:M, Vm = bits<19:16> (4-bit).
            let vm = bits(word, 16, 4);
            let idx = ((h << 2) | (l << 1) | m) as u8;
            (vm, idx, VA::Sh)
        }
        0b10 => {
            // .s: index = H:L, Vm = bits<20:16> (5-bit).
            let vm = bits(word, 16, 5);
            let idx = ((h << 1) | l) as u8;
            (vm, idx, VA::Ss)
        }
        _ => {
            // .d: index = H, Vm = bits<20:16> (5-bit).
            let vm = bits(word, 16, 5);
            let idx = h as u8;
            (vm, idx, VA::Sd)
        }
    }
}

/// Decode the half-precision indexed element (used by FMLAL/FMLSL): index =
/// H:L:M (3-bit), Vm = bits<19:16> (4-bit).
#[inline]
fn decode_index_h(word: u32) -> (u32, u8) {
    let l = bit(word, 21);
    let m = bit(word, 20);
    let h = bit(word, 11);
    let vm = bits(word, 16, 4);
    let idx = ((h << 2) | (l << 1) | m) as u8;
    (vm, idx)
}

/// Map a narrowing two-reg-misc [`Code`] to its `2` hi-half sibling (XTN2/
/// SQXTN2/UQXTN2/SQXTUN2), used when `Q==1`.
#[inline]
fn narrow_to_2(code: Code) -> Code {
    match code {
        Code::XtnVec => Code::Xtn2Vec,
        Code::SqxtnVec => Code::Sqxtn2Vec,
        Code::UqxtnVec => Code::Uqxtn2Vec,
        Code::SqxtunVec => Code::Sqxtun2Vec,
        other => other,
    }
}

/// Map a "long" by-element [`Code`] to its `2` hi-half sibling.
#[inline]
fn long_to_2(code: Code) -> Code {
    match code {
        Code::SmlalVec => Code::Smlal2Vec,
        Code::SmlslVec => Code::Smlsl2Vec,
        Code::SmullVec => Code::Smull2Vec,
        Code::UmlalVec => Code::Umlal2Vec,
        Code::UmlslVec => Code::Umlsl2Vec,
        Code::UmullVec => Code::Umull2Vec,
        Code::SqdmlalVec => Code::Sqdmlal2Vec,
        Code::SqdmlslVec => Code::Sqdmlsl2Vec,
        Code::SqdmullVec => Code::Sqdmull2Vec,
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{BufSink, FmtFormatter, Formatter};

    /// Decode `word` at ip 0 with all features (via the SIMD&FP group entry so
    /// the dispatcher is exercised), render, and assert the text.
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

    /// Assert `word` decodes to the invalid sentinel within this group.
    #[track_caller]
    fn invalid(word: u32) {
        let mut insn = Instruction::default();
        crate::decode::simd_fp::decode(word, 0, FeatureSet::ALL, &mut insn);
        assert!(insn.is_invalid(), "expected invalid for {word:#010x}");
    }

    #[test]
    fn three_same_integer() {
        render(0x4EAE86AF, "add     v15.4s, v21.4s, v14.4s");
        render(0x4E29BD6C, "addp    v12.16b, v11.16b, v9.16b");
        render(0x6E268FBF, "cmeq    v31.16b, v29.16b, v6.16b");
        // Scalar three-same (doubleword add / shift).
        render(0x5EE687A0, "add     d0, d29, d6");
        render(0x5EE344DD, "sshl    d29, d6, d3");
    }

    #[test]
    fn three_same_logical() {
        // U=0: AND / BIC / ORR / ORN.
        render(0x4E391F6A, "and     v10.16b, v27.16b, v25.16b");
        render(0x0E391D48, "and     v8.8b, v10.8b, v25.8b");
        render(0x4E6B1F5B, "bic     v27.16b, v26.16b, v11.16b");
        render(0x4EAC1CC9, "orr     v9.16b, v6.16b, v12.16b");
        render(0x4EF61D8B, "orn     v11.16b, v12.16b, v22.16b");
        // ORR with Vn==Vm renders as the `mov` two-operand alias.
        render(0x0EAB1D68, "mov     v8.8b, v11.8b");
        render(0x4EAA1D52, "mov     v18.16b, v10.16b");
        // U=1: EOR / BSL / BIT / BIF.
        render(0x6E271F90, "eor     v16.16b, v28.16b, v7.16b");
        render(0x2E681D2E, "bsl     v14.8b, v9.8b, v8.8b");
        render(0x6EAC1CF1, "bit     v17.16b, v7.16b, v12.16b");
        render(0x2EE71E78, "bif     v24.8b, v19.8b, v7.8b");
    }

    #[test]
    fn three_same_fp_and_fp16() {
        render(0x6EF7E4B8, "fcmgt   v24.2d, v5.2d, v23.2d");
        // FP16 three-same gates on FEAT_FP16 (decoded under ALL).
        render(0x6ED21776, "fabd    v22.8h, v27.8h, v18.8h");
        render(0x6E5B1769, "faddp   v9.8h, v27.8h, v27.8h");
        // Widening FMLAL/FMLSL three-same (`.4s` <- `.4h`).
        render(0x4E30EF32, "fmlal   v18.4s, v25.4h, v16.4h");
    }

    #[test]
    fn three_same_extra_and_complex() {
        render(0x6F6BDB3D, "sqrdmlah v29.8h, v25.8h, v11.h[6]");
        render(0x6E47E437, "fcadd   v23.8h, v1.8h, v7.8h, #0x5a");
        render(0x6F52302B, "fcmla   v11.8h, v1.8h, v18.h[0], #0x5a");
    }

    #[test]
    fn three_different() {
        render(0x0E2343B2, "addhn   v18.8b, v29.8h, v3.8h");
        render(0x0E2900A7, "saddl   v7.8h, v5.8b, v9.8b");
        render(0x4E78C1A1, "smull2  v1.4s, v13.8h, v24.8h");
        // Scalar three-different (SQDMULL).
        render(0x5E6CD0A6, "sqdmull s6, h5, h12");
    }

    #[test]
    fn two_reg_misc() {
        // Compare-against-zero carries `#0x0`.
        render(0x0E2098FE, "cmeq    v30.8b, v7.8b, #0x0");
        render(0x4E208987, "cmgt    v7.16b, v12.16b, #0x0");
        // Narrowing with the `2` hi-half form.
        render(0x0E614964, "sqxtn   v4.4h, v11.4s");
        render(0x4EA14A31, "sqxtn2  v17.4s, v17.2d");
        // Pairwise-long (dest half the lane count).
        render(0x0EA02AD5, "saddlp  v21.1d, v22.2s");
        render(0x0E303992, "saddlv  h18, v12.8b");
        render(0x4E31B9A0, "addv    b0, v13.16b");
        // MVN (NOT), RBIT — byte-arrangement bitwise misc ops.
        render(0x6E205A52, "mvn     v18.16b, v18.16b");
        render(0x2E2059E4, "mvn     v4.8b, v15.8b");
        render(0x6E605938, "rbit    v24.16b, v9.16b");
        render(0x2E605AB4, "rbit    v20.8b, v21.8b");
        // SHLL / SHLL2 (shift = element width).
        render(0x2E213961, "shll    v1.8h, v11.8b, #0x8");
        render(0x2EA13A8E, "shll    v14.2d, v20.2s, #0x20");
        render(0x6E213B5A, "shll2   v26.8h, v26.16b, #0x8");
        render(0x6EA1389B, "shll2   v27.2d, v4.4s, #0x20");
    }

    #[test]
    fn two_reg_misc_fp() {
        render(0x4EE1D95A, "frecpe  v26.2d, v10.2d");
        render(0x0E21781E, "fcvtl   v30.4s, v0.4h");
        render(0x0E6169A2, "fcvtn   v2.2s, v13.2d");
        // FP compare-against-zero carries `#0.0`.
        render(0x4EA0F87E, "fabs    v30.4s, v3.4s");
        // Scalar FP misc.
        render(0x7E6168A4, "fcvtxn  s4, d5");
        render(0x5EA1FAE1, "frecpx  s1, s23");
    }

    #[test]
    fn scalar_pairwise() {
        render(0x5EF1B94B, "addp    d11, v10.2d");
        render(0x7E70DB3D, "faddp   d29, v25.2d");
        render(0x5E30F943, "fmaxp   h3, v10.2h");
    }

    #[test]
    fn by_element() {
        render(0x4FA39897, "fmul    v23.4s, v4.4s, v3.s[3]");
        render(0x4F2913F6, "fmla    v22.8h, v31.8h, v9.h[2]");
        // FMLAL by element (widening, half-element index).
        render(0x4F8E00D4, "fmlal   v20.4s, v6.4h, v14.h[0]");
        // Long by element, hi-half `2` form.
        render(0x4F999929, "fmul    v9.4s, v9.4s, v25.s[2]");
    }

    #[test]
    fn dot_product() {
        // SDOT/UDOT vector form: Vd.<2s/4s>, Vn.<8b/16b>, Vm.<8b/16b>.
        render(0x0E9F97ED, "sdot    v13.2s, v31.8b, v31.8b");
        render(0x4E8A94AC, "sdot    v12.4s, v5.16b, v10.16b");
        render(0x6E9B97DB, "udot    v27.4s, v30.16b, v27.16b");
        // SDOT/UDOT by element: the indexed operand is always `Vm.4b[index]`.
        render(0x0F9BE85D, "sdot    v29.2s, v2.8b, v27.4b[2]");
        render(0x4FB3E310, "sdot    v16.4s, v24.16b, v19.4b[1]");
        render(0x6FA7E0FA, "udot    v26.4s, v7.16b, v7.4b[1]");
    }

    #[test]
    fn reserved_and_panic_free() {
        // `.1d` (size==11, Q==0) is invalid for a three-same byte/half/word op.
        invalid(0x0EE08400); // add v0.1d, ... -> reserved
        // Sweep a slice of the SIMD-arith space for panic-freedom.
        for w in (0x0E00_0000u32..0x0E00_0000u32.wrapping_add(8192)).step_by(11) {
            let mut insn = Instruction::default();
            crate::decode::simd_fp::decode(w, 0, FeatureSet::ALL, &mut insn);
            let mut insn2 = Instruction::default();
            crate::decode::simd_fp::decode(w | 0x4000_0000, 0, FeatureSet::ALL, &mut insn2);
        }
    }
}
