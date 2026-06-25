//! SME (Scalable Matrix Extension) encodings — hand-written from the ARM ARM.
//!
//! SME instructions live in the **reserved** top-level group
//! (`op0 = word<28:25> = 0b0000`) with `word<31> == 1`; they are dispatched here
//! from [`crate::decode::decode_reserved`] (only when the `sme` cargo feature is
//! compiled in and the runtime [`Feature::Sme`] is accepted). The single
//! exception is `SMSTART`/`SMSTOP`, which are `MSR (immediate)` PSTATE encodings
//! and are handled in [`crate::decode::branch_sys`].
//!
//! Dispatch key — the four SME sub-areas are selected by `word<31:29>`:
//!
//! | `word<31:29>` | Area |
//! |-|-|
//! | `100` | Outer-product (FP / BF16): `FMOPA`/`FMOPS`/`BFMOPA`/`BFMOPS` |
//! | `101` | Outer-product (integer): `[US]MOPA`/`[US]MOPS` and the mixed forms |
//! | `110` | `MOVA` (ZA tile slice ↔ Z) and `ADDHA`/`ADDVA` |
//! | `111` | ZA-array load/store: `LD1*`/`ST1*` (tile slice) and `LDR`/`STR` ZA |
//!
//! Binary Ninja renders the SME ZA tile-slice operands with a `z` prefix (e.g.
//! `z0v.b[w14, #0x5]`, not `za0v...`) and the outer-product `ZAda` accumulator
//! as a plain vector register `z<n>.<T>`. We follow the corpus exactly here; see
//! [`crate::operand::Operand::SmeTileSlice`]. Every path is total and panic-free,
//! leaving [`Code::Invalid`] for unallocated encodings.

use crate::decode::bits::{bit, bits};
use crate::enums::{ExtendType, VectorArrangement as VA};
use crate::features::{Feature, FeatureSet};
use crate::instruction::Instruction;
use crate::mnemonic::Code;
use crate::operand::{Operand, PredQual, SliceIndicator, SveMemMode};
use crate::register::{gp_register, Register, RegWidth};

pub(crate) mod sme2;

// ---------------------------------------------------------------------------
// Register-bank tables (local, mirroring the SVE decoders).
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

/// Decode a single SME instruction `word` at `ip` into `out`.
///
/// Called from [`crate::decode::decode_reserved`] under `#[cfg(feature = "sme")]`
/// once the reserved group has selected a `word<31> == 1` encoding. Runtime-gated
/// on [`Feature::Sme`]; dispatches on `word<31:29>` to the family decoders. Total
/// and panic-free for all inputs.
#[inline]
pub fn decode(word: u32, ip: u64, features: FeatureSet, out: &mut Instruction) {
    let _ = ip;
    if !features.has(Feature::Sme) {
        return;
    }
    // SME2 multi-vector forms (FEAT_SME2) carve unallocated sub-regions out of
    // the FMOPA quadrant (`100`, outer-product size `word<23:22> == 01`) and the
    // MOVA/ADDHA quadrant (`110`, `word<24> == 1`). Route them first when the
    // runtime accepts FEAT_SME2; they never overlap the base SME encodings.
    let sme2 = features.has(Feature::Sme2);
    match bits(word, 29, 3) {
        0b100 => {
            if sme2 && bits(word, 22, 2) == 0b01 {
                sme2::decode_tmopa(word, out);
            } else {
                decode_mopa_fp(word, out);
            }
        }
        0b101 => decode_mopa_int(word, out),
        0b110 => {
            if sme2 && bit(word, 24) == 1 {
                sme2::decode_mul(word, out);
            } else {
                decode_mova_add(word, out);
            }
        }
        0b111 => decode_za_ldst(word, out),
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Small operand helpers (mirroring the SVE decoders' conventions).
// ---------------------------------------------------------------------------

/// `Z<n>` with an arrangement (`z3.s`), used for the outer-product accumulator
/// `ZAda` and the `Zn`/`Zm` source vectors (binja renders `ZAda` as a `z` reg).
#[inline]
fn zreg(n: u32, a: VA) -> Operand {
    Operand::Reg {
        reg: Z[(n & 0x1f) as usize],
        arr: Some(a),
        lane: None,
        shift: None,
        extend: None,
        pred: None,
    }
}

/// `P<n>/M` (merging) governing predicate.
#[inline]
fn preg_m(n: u32) -> Operand {
    Operand::Reg {
        reg: P[(n & 0x7) as usize],
        arr: None,
        lane: None,
        shift: None,
        extend: None,
        pred: Some(PredQual::Merging),
    }
}

/// `P<n>` with the given qualifier (`/z` for loads, none for stores).
#[inline]
fn preg_q(n: u32, q: PredQual) -> Operand {
    Operand::Reg {
        reg: P[(n & 0x7) as usize],
        arr: None,
        lane: None,
        shift: None,
        extend: None,
        pred: Some(q),
    }
}

/// The `Ws`/`Wv` slice-select register (`w12..w15`) from a 2-bit `Rs`/`Rv`.
#[inline]
fn wsel(rs: u32) -> Register {
    gp_register(false, RegWidth::W32, 12 + (rs & 3) as u8)
}

/// Build a ZA tile-slice operand `z<tile><h|v>.<T>[<Ws>{, #<imm>}]`.
#[inline]
fn tile_slice(tile: u32, vertical: bool, arr: VA, rs: u32, imm: Option<i16>) -> Operand {
    Operand::SmeTileSlice {
        reg: Z[(tile & 0x1f) as usize],
        slice: if vertical {
            SliceIndicator::Vertical
        } else {
            SliceIndicator::Horizontal
        },
        arr: Some(arr),
        sel: wsel(rs),
        imm: imm.unwrap_or(0),
        has_imm: imm.is_some(),
    }
}

// ---------------------------------------------------------------------------
// Outer products (FP / BF16): word<31:29> == 100.
// ---------------------------------------------------------------------------

/// `FMOPA`/`FMOPS`/`BFMOPA`/`BFMOPS` — floating-point and BFloat16 outer
/// product, accumulate (`S==0`) or subtract (`S==1`) into a ZA tile.
///
/// Common fields: `ZAda` (low bits), `Zn = word<9:5>`, `Pn = word<12:10>`,
/// `Pm = word<15:13>`, `Zm = word<20:16>`, `S = word<4>`. The variant is
/// selected by `word<24>` and the size `word<23:22>`:
/// `op<24>=0, sz<23:22>=10` → FP32; `=11` → FP64; `op<24>=1, sz=10` →
/// 16-bit-input (`word<21>` then picks `FMOPA`(1)/`BFMOPA`(0)).
fn decode_mopa_fp(word: u32, out: &mut Instruction) {
    // Reserved bits of the outer-product frame must be zero (`word<3>`,
    // `word<2>`? — only `word<4>` is `S`). Keep total: just decode the cases the
    // architecture allocates and leave anything else invalid.
    let s = bit(word, 4);
    let zn = bits(word, 5, 5);
    let pn = bits(word, 10, 3);
    let pm = bits(word, 13, 3);
    let zm = bits(word, 16, 5);
    let op24 = bit(word, 24);
    let sz = bits(word, 22, 2);

    // Determine (code, dst-arr, src-arr).
    let (code, dst, src) = if op24 == 0 {
        match sz {
            0b10 => {
                // FP32: ZAda.S, Zn.S, Zm.S. (ZAda = word<1:0>.)
                (if s == 0 { Code::SmeFmopaS } else { Code::SmeFmopsS }, VA::Ss, VA::Ss)
            }
            0b11 => {
                // FP64: ZAda.D, Zn.D, Zm.D. (ZAda = word<2:0>.)
                (if s == 0 { Code::SmeFmopaD } else { Code::SmeFmopsD }, VA::Sd, VA::Sd)
            }
            _ => return,
        }
    } else {
        // op24 == 1: 16-bit-input forms, ZAda.S accumulator, src .H.
        if sz != 0b10 {
            return;
        }
        if bit(word, 21) == 1 {
            // FMOPA/FMOPS (FP16 → FP32).
            (if s == 0 { Code::SmeFmopaH } else { Code::SmeFmopsH }, VA::Ss, VA::Sh)
        } else {
            // BFMOPA/BFMOPS (BF16 → FP32).
            (if s == 0 { Code::SmeBfmopa } else { Code::SmeBfmops }, VA::Ss, VA::Sh)
        }
    };

    let zada = if matches!(dst, VA::Sd) {
        bits(word, 0, 3)
    } else {
        bits(word, 0, 2)
    };

    out.set(code);
    out.push_operand(zreg(zada, dst));
    out.push_operand(preg_m(pn));
    out.push_operand(preg_m(pm));
    out.push_operand(zreg(zn, src));
    out.push_operand(zreg(zm, src));
}

// ---------------------------------------------------------------------------
// Outer products (integer): word<31:29> == 101.
// ---------------------------------------------------------------------------

/// `SMOPA`/`UMOPA`/`SUMOPA`/`USMOPA` and their `*OPS` (subtract) counterparts —
/// integer outer product into a ZA tile.
///
/// Signedness is the pair `(u0 = word<24>, u1 = word<21>)`: `(0,0)`→signed,
/// `(1,1)`→unsigned, `(0,1)`→signed×unsigned, `(1,0)`→unsigned×signed.
/// `S = word<4>` selects accumulate(0)/subtract(1). The size `word<23:22>` is
/// `10`→32-bit (`ZAda.S`, `Zn.B`/`Zm.B`) or `11`→64-bit (`ZAda.D`, `Zn.H`/`Zm.H`,
/// the `FEAT_SME_I16I64` form).
fn decode_mopa_int(word: u32, out: &mut Instruction) {
    let s = bit(word, 4);
    let zn = bits(word, 5, 5);
    let pn = bits(word, 10, 3);
    let pm = bits(word, 13, 3);
    let zm = bits(word, 16, 5);
    let u0 = bit(word, 24);
    let u1 = bit(word, 21);
    let sz = bits(word, 22, 2);

    // (dst-arr, src-arr, ZAda width).
    let (dst, src, is64) = match sz {
        0b10 => (VA::Ss, VA::Sb, false),
        0b11 => (VA::Sd, VA::Sh, true),
        _ => return,
    };

    // Map (u0, u1, S) to the encoding row.
    let code = match (u0, u1, s, is64) {
        (0, 0, 0, false) => Code::SmeSmopaS,
        (0, 0, 1, false) => Code::SmeSmopsS,
        (1, 1, 0, false) => Code::SmeUmopaS,
        (1, 1, 1, false) => Code::SmeUmopsS,
        (0, 1, 0, false) => Code::SmeSumopaS,
        (0, 1, 1, false) => Code::SmeSumopsS,
        (1, 0, 0, false) => Code::SmeUsmopaS,
        (1, 0, 1, false) => Code::SmeUsmopsS,
        (0, 0, 0, true) => Code::SmeSmopaD,
        (0, 0, 1, true) => Code::SmeSmopsD,
        (1, 1, 0, true) => Code::SmeUmopaD,
        (1, 1, 1, true) => Code::SmeUmopsD,
        (0, 1, 0, true) => Code::SmeSumopaD,
        (0, 1, 1, true) => Code::SmeSumopsD,
        (1, 0, 0, true) => Code::SmeUsmopaD,
        (1, 0, 1, true) => Code::SmeUsmopsD,
        _ => return,
    };

    let zada = if is64 { bits(word, 0, 3) } else { bits(word, 0, 2) };

    out.set(code);
    out.push_operand(zreg(zada, dst));
    out.push_operand(preg_m(pn));
    out.push_operand(preg_m(pm));
    out.push_operand(zreg(zn, src));
    out.push_operand(zreg(zm, src));
}

// ---------------------------------------------------------------------------
// MOVA / ADDHA / ADDVA: word<31:29> == 110.
// ---------------------------------------------------------------------------

/// Dispatch the `110` SME quadrant. Both `MOVA` and `ADDHA`/`ADDVA` carry
/// `word<24> == 0` (the `11000000`/`11000001` size-dependent rows) and use
/// `word<23:22>` as the element size; they are distinguished by the opcode field
/// `word<21:17>`: `MOVA` is `0000x` (`word<20> == 0`, direction in `word<17>`)
/// while `ADDHA`/`ADDVA` is `01000` (`word<20> == 1`). `word<24> == 1` is
/// unallocated here.
fn decode_mova_add(word: u32, out: &mut Instruction) {
    if bit(word, 24) != 0 {
        return;
    }
    // Opcode `word<21:17>`: ADDHA/ADDVA are `01000`; MOVA is `0000x`.
    if bit(word, 21) == 0 && bit(word, 20) == 1 && bits(word, 17, 3) == 0 {
        decode_addha_addva(word, out);
    } else if bits(word, 18, 4) == 0 {
        // MOVA: word<21:18> == 0000 (direction in word<17>).
        decode_mova(word, out);
    }
}

/// `ADDHA`/`ADDVA` — add horizontally / vertically the elements of a vector to
/// a 32- or 64-bit-element ZA tile.
///
/// `V = word<16>` selects `ADDHA`(0)/`ADDVA`(1). The element size is
/// `word<22>` (`0`→`.S`, ZAda 2-bit; `1`→`.D`, ZAda 3-bit). Operand order is
/// `ZAda.<T>, Pn/M, Pm/M, Zn.<T>` with `Zn = word<9:5>`, `Pn = word<12:10>`,
/// `Pm = word<15:13>`.
fn decode_addha_addva(word: u32, out: &mut Instruction) {
    let v = bit(word, 16);
    let pm = bits(word, 13, 3);
    let pn = bits(word, 10, 3);
    let zn = bits(word, 5, 5);
    let is64 = bit(word, 22) == 1;
    let arr = if is64 { VA::Sd } else { VA::Ss };
    let zada = if is64 { bits(word, 0, 3) } else { bits(word, 0, 2) };

    out.set(if v == 0 { Code::SmeAddha } else { Code::SmeAddva });
    out.push_operand(zreg(zada, arr));
    out.push_operand(preg_m(pn));
    out.push_operand(preg_m(pm));
    out.push_operand(zreg(zn, arr));
}

/// `MOVA` — move a vector to/from a ZA tile slice.
///
/// `word<17>` selects the direction: `0` → tile ← vector (`MOVA <tile-slice>,
/// <Pg>/M, <Zn>`); `1` → vector ← tile (`MOVA <Zd>, <Pg>/M, <tile-slice>`). The
/// element size is `word<23:22>` for `B`/`H`/`S`/`D`, with `Q` selected by
/// `word<16> == 1` (`Q`) when size is `D` (`11`). `V = word<15>` is the slice
/// direction (`h`/`v`); `Rs = word<14:13>` the `Ws` slice select; `Pg =
/// word<12:10>`.
///
/// The tile number and slice index share a 4-bit field, with the tile in the
/// high bits and the index in the low bits; the split point is the element size
/// (`B`: 0 tile bits / 4 index; `H`: 1/3; `S`: 2/2; `D`: 3/1; `Q`: 4/0). For the
/// tile→vector direction this field is `word<8:5>` (with `Zd = word<4:0>`); for
/// the vector→tile direction it is `word<3:0>` (with `Zn = word<9:5>`).
fn decode_mova(word: u32, out: &mut Instruction) {
    let size = bits(word, 22, 2);
    let q = bit(word, 16);
    let vertical = bit(word, 15) == 1;
    let rs = bits(word, 13, 2);
    let pg = bits(word, 10, 3);

    // Element kind and the number of index immediate bits.
    // (arr, index-bit-width). Q is a special case of size==11 with Q==1.
    let (arr, imm_bits) = match size {
        0b00 => (VA::Sb, 4u32),
        0b01 => (VA::Sh, 3),
        0b10 => (VA::Ss, 2),
        0b11 => {
            if q == 1 {
                (VA::Sq, 0)
            } else {
                (VA::Sd, 1)
            }
        }
        _ => return,
    };

    let to_vector = bit(word, 17) == 1;

    if to_vector {
        // Z ← ZA tile slice: tile-slice field at word<8:5>, Zd = word<4:0>.
        let zd = bits(word, 0, 5);
        let field = bits(word, 5, 4);
        let (tile, imm) = split_tile_field(field, imm_bits);
        out.set(Code::SmeMovaTileToZ);
        out.push_operand(zreg(zd, arr));
        out.push_operand(preg_m(pg));
        out.push_operand(tile_slice(tile, vertical, arr, rs, imm));
    } else {
        // ZA tile slice ← Z: tile-slice field at word<3:0>, Zn = word<9:5>.
        let zn = bits(word, 5, 5);
        let field = bits(word, 0, 4);
        let (tile, imm) = split_tile_field(field, imm_bits);
        out.set(Code::SmeMovaZToTile);
        out.push_operand(tile_slice(tile, vertical, arr, rs, imm));
        out.push_operand(preg_m(pg));
        out.push_operand(zreg(zn, arr));
    }
}

/// Split a 4-bit `ZAd:imm` field into `(tile_number, slice_index)`.
///
/// The slice index occupies the low `imm_bits`; the tile number the high
/// `4 - imm_bits`. When `imm_bits == 0` (the `.Q` forms) there is no index.
#[inline]
fn split_tile_field(field: u32, imm_bits: u32) -> (u32, Option<i16>) {
    let imm_mask = (1u32 << imm_bits) - 1;
    let imm = field & imm_mask;
    let tile = field >> imm_bits;
    if imm_bits == 0 {
        (tile, None)
    } else {
        (tile, Some(imm as i16))
    }
}

// ---------------------------------------------------------------------------
// ZA-array load/store: word<31:29> == 111.
// ---------------------------------------------------------------------------

/// Dispatch the `111` SME quadrant: the whole-array `LDR`/`STR` ZA forms
/// (`word<24> == 1` with size `word<23:22> == 00`) versus the per-slice
/// `LD1*`/`ST1*` ZA-array vector loads/stores (everything else).
fn decode_za_ldst(word: u32, out: &mut Instruction) {
    // LDR/STR ZA: `1110000100|op|000000|Rv|000|Rn|0|imm4` — bit24==1 and the
    // size field word<23:22>==00 (LD1Q/ST1Q also set bit24 but have size==11).
    if bit(word, 24) == 1 && bits(word, 22, 2) == 0 {
        decode_ldr_str_za(word, out);
    } else {
        decode_ld1_st1_za(word, out);
    }
}

/// `LD1B/H/W/D/Q` and `ST1B/H/W/D/Q` (ZA array vector) — contiguous load/store
/// of a single ZA tile slice, predicated.
///
/// The single ZA tile is implicitly `0`. `V = word<15>` selects the slice
/// direction (`h`/`v`); `Rs = word<14:13>` the `Ws` slice select; `Pg =
/// word<12:10>`; `Rn = word<9:5>` the base; `Rm = word<20:16>` the index. The
/// element/byte size is `word<23:22>` (`Q` when `word<24> == 1`). The slice
/// index immediate occupies the low bits of `word<3:0>` (width per element size;
/// none for `.Q`). The index register is scaled by `LSL #log2(bytes)`. Loads
/// take a `/Z` predicate; stores take a bare predicate.
fn decode_ld1_st1_za(word: u32, out: &mut Instruction) {
    let is_q = bit(word, 24) == 1;
    let size = bits(word, 22, 2);
    let is_store = bit(word, 21) == 1;
    let rm = bits(word, 16, 5);
    let vertical = bit(word, 15) == 1;
    let rs = bits(word, 13, 2);
    let pg = bits(word, 10, 3);
    let rn = bits(word, 5, 5);

    // (code, arr, log2(bytes), index-imm-bits).
    let (code_ld, code_st, arr, log2, imm_bits): (Code, Code, VA, u8, u32) = if is_q {
        (Code::SmeLd1qZa, Code::SmeSt1qZa, VA::Sq, 4, 0)
    } else {
        match size {
            0b00 => (Code::SmeLd1bZa, Code::SmeSt1bZa, VA::Sb, 0, 4),
            0b01 => (Code::SmeLd1hZa, Code::SmeSt1hZa, VA::Sh, 1, 3),
            0b10 => (Code::SmeLd1wZa, Code::SmeSt1wZa, VA::Ss, 2, 2),
            0b11 => (Code::SmeLd1dZa, Code::SmeSt1dZa, VA::Sd, 3, 1),
            _ => return,
        }
    };

    let imm = if imm_bits == 0 {
        None
    } else {
        Some((bits(word, 0, imm_bits) & ((1 << imm_bits) - 1)) as i16)
    };

    let code = if is_store { code_st } else { code_ld };
    let pred = if is_store {
        preg_q(pg, PredQual::None)
    } else {
        preg_q(pg, PredQual::Zeroing)
    };

    // Memory: `[Xn{, Xm, LSL #log2}]`. The index uses `Uxtx` (rendered as `lsl`),
    // with the scale shown for the scaled element sizes (H/W/D/Q) and elided for
    // the byte form (`log2 == 0`).
    let base = gp_register(true, RegWidth::X64, rn as u8);
    let index = gp_register(false, RegWidth::X64, rm as u8);
    // `shift` carries the amount in the low 7 bits and "show amount" in bit7.
    let shift = if log2 == 0 { 0 } else { 0x80 | log2 };
    let mem = Operand::MemExt {
        base,
        index,
        extend: ExtendType::Uxtx,
        shift,
    };

    out.set(code);
    out.push_operand(tile_slice(0, vertical, arr, rs, imm));
    out.push_operand(pred);
    out.push_operand(mem);
}

/// `LDR`/`STR` (ZA array vector) — load/store one whole ZA array vector to/from
/// memory, addressed by `[Xn{, #imm, MUL VL}]`.
///
/// `op = word<21>` selects `LDR`(0)/`STR`(1). The vector select is
/// `ZA[Wv, #imm4]` with `Wv = w12 + word<14:13>` and `imm4 = word<3:0>`; the
/// same `imm4` is the `MUL VL` multiple of the memory operand. `Rn = word<9:5>`
/// is the base (`SP`-resolved). When `imm4 == 0` both the `, #imm` and the
/// `, MUL VL` are elided, matching binja (`za[w15, #0x0], [x21]`).
fn decode_ldr_str_za(word: u32, out: &mut Instruction) {
    let is_store = bit(word, 21) == 1;
    let rv = bits(word, 13, 2);
    let rn = bits(word, 5, 5);
    let imm4 = bits(word, 0, 4) as i32;

    // Whole-array vector select `za[Wv, #imm4]` (no tile, no slice direction).
    let select = Operand::SmeTileSlice {
        reg: Register::None,
        slice: SliceIndicator::None,
        arr: None,
        sel: wsel(rv),
        imm: imm4 as i16,
        has_imm: true,
    };

    // `[Xn{, #imm4, MUL VL}]` — SVE scalar-plus-VL-scaled-immediate addressing.
    let mem = Operand::SveMem {
        base: gp_register(true, RegWidth::X64, rn as u8),
        offset: Register::None,
        arr: None,
        extend: ExtendType::Uxtx,
        imm: imm4,
        amount: 0,
        mode: SveMemMode::ScalarImmMulVl,
    };

    out.set(if is_store { Code::SmeStrZa } else { Code::SmeLdrZa });
    out.push_operand(select);
    out.push_operand(mem);
}

#[cfg(test)]
mod tests {
    use crate::format::{BufSink, FmtFormatter, Formatter};
    use crate::{Decoder, DecoderOptions};

    /// Decode `word` and render with the default UAL formatter into `buf`.
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
    fn outer_product_fp_bf() {
        // FMOPA/FMOPS — FP32, FP64, and FP16→FP32; BFMOPA/BFMOPS.
        check(0x809B4941, "fmopa   z1.s, p2/m, p2/m, z10.s, z27.s");
        check(0x80D69446, "fmopa   z6.d, p5/m, p4/m, z2.d, z22.d");
        check(0x81BB7F43, "fmopa   z3.s, p7/m, p3/m, z26.h, z27.h");
        check(0x80851312, "fmops   z2.s, p4/m, p0/m, z24.s, z5.s");
        check(0x8184B942, "bfmopa  z2.s, p6/m, p5/m, z10.h, z4.h");
        check(0x819EC990, "bfmops  z0.s, p2/m, p6/m, z12.h, z30.h");
    }

    #[test]
    fn outer_product_int() {
        // The four signedness pairings, accumulate and subtract, 32- and 64-bit.
        check(0xA0822DA1, "smopa   z1.s, p3/m, p1/m, z13.b, z2.b");
        check(0xA1A98383, "umopa   z3.s, p0/m, p4/m, z28.b, z9.b");
        check(0xA0A0AC03, "sumopa  z3.s, p3/m, p5/m, z0.b, z0.b");
        check(0xA19670C1, "usmopa  z1.s, p4/m, p3/m, z6.b, z22.b");
        check(0xA094D912, "smops   z2.s, p6/m, p6/m, z8.b, z20.b");
        check(0xA0CFCB26, "smopa   z6.d, p2/m, p6/m, z25.h, z15.h");
        check(0xA1E301C4, "umopa   z4.d, p0/m, p0/m, z14.h, z3.h");
    }

    #[test]
    fn addha_addva() {
        check(0xC0909662, "addha   z2.s, p5/m, p4/m, z19.s");
        check(0xC091BA23, "addva   z3.s, p6/m, p5/m, z17.s");
        check(0xC0D053E5, "addha   z5.d, p4/m, p2/m, z31.d");
        check(0xC0D16EE5, "addva   z5.d, p3/m, p3/m, z23.d");
    }

    #[test]
    fn mova_tile_to_vector() {
        // Z ← ZA tile slice (binja renders the tile with a `z` prefix).
        check(0xC002D4B0, "mova    z16.b, p5/m, z0v.b[w14, #0x5]");
        check(0xC0825448, "mova    z8.s, p5/m, z0h.s[w14, #0x2]");
        check(0xC042BD71, "mova    z17.h, p7/m, z1v.h[w13, #0x3]");
        // The .Q form has no slice index immediate.
        check(0xC0C350F0, "mova    z16.q, p4/m, z7h.q[w14]");
    }

    #[test]
    fn mova_vector_to_tile() {
        check(0xC0002F06, "mova    z0h.b[w13, #0x6], p3/m, z24.b");
        check(0xC0803C6A, "mova    z2h.s[w13, #0x2], p7/m, z3.s");
        check(0xC0C0F4E9, "mova    z4v.d[w15, #0x1], p5/m, z7.d");
        check(0xC0C1D2E5, "mova    z5v.q[w14], p4/m, z23.q");
    }

    #[test]
    fn za_array_load_store() {
        // LD1*/ST1* ZA-array vector: tile always 0, `/z` for loads, bare for
        // stores, index scaled by the element size (elided for byte).
        check(0xE011E5A3, "ld1b    z0v.b[w15, #0x3], p1/z, [x13, x17]");
        check(0xE059A9C2, "ld1h    z0v.h[w13, #0x2], p2/z, [x14, x25, lsl #0x1]");
        check(0xE0D76421, "ld1d    z0h.d[w15, #0x1], p1/z, [x1, x23, lsl #0x3]");
        check(0xE1DE4A4D, "ld1q    z0h.q[w14], p2/z, [x18, x30, lsl #0x4]");
        check(0xE024806B, "st1b    z0v.b[w12, #0xb], p0, [x3, x4]");
        check(0xE1F50B6D, "st1q    z0h.q[w12], p2, [x27, x21, lsl #0x4]");
        // SP base resolves to `sp`.
        check(0xE0B24FE7, "st1w    z0h.s[w14, #0x3], p3, [sp, x18, lsl #0x2]");
    }

    #[test]
    fn ldr_str_za_whole_array() {
        check(0xE100004D, "ldr     za[w12, #0xd], [x2, #0xd, mul vl]");
        check(0xE1200106, "str     za[w12, #0x6], [x8, #0x6, mul vl]");
        // imm4 == 0 elides both the slice `#imm` and the `, mul vl`.
        check(0xE10062A0, "ldr     za[w15, #0x0], [x21]");
        // SP base.
        check(0xE10043EB, "ldr     za[w14, #0xb], [sp, #0xb, mul vl]");
    }

    #[test]
    fn smstart_smstop() {
        // SMSTART/SMSTOP are MSR PSTATE encodings (decoded in branch_sys); verify
        // the SME spellings end-to-end here.
        check(0xD503437F, "smstart sm");
        check(0xD503457F, "smstart za");
        check(0xD503477F, "smstart");
        check(0xD503467F, "smstop");
        check(0xD503447F, "smstop  za");
    }

    #[test]
    fn feature_gate_off_leaves_invalid() {
        // With FEAT_SME not accepted, the SME reserved-region encodings must not
        // decode (stay Invalid) — but SMSTART/SMSTOP, being PSTATE MSRs in the
        // system group, are unaffected by the SME structural gate.
        use crate::features::FeatureSet;
        let opts = DecoderOptions {
            features: FeatureSet::BASE, // no SME accepted
        };
        let bytes = 0x809B4941u32.to_le_bytes(); // fmopa
        let mut dec = Decoder::new(&bytes, 0x1000, opts);
        assert!(dec.decode().is_invalid());
    }

    #[test]
    fn never_panics_on_sme_space() {
        // Exhaustively exercise the SME reserved-region prefixes (word<31> == 1,
        // op0 == 0000) for panic-freedom across the full low 16 bits.
        let mut buf = [0u8; 128];
        for hi in [0x80u32, 0x81, 0xa0, 0xa1, 0xc0, 0xc1, 0xe0, 0xe1] {
            for low in 0u32..=0xffff {
                let word = (hi << 24) | (low << 8) | (low & 0xff);
                let _ = render(word, &mut buf);
            }
        }
    }
}
