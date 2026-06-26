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
pub(crate) mod sme_lut;
pub(crate) mod sme_za_move;

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
            } else if is_mop4(word) {
                // FEAT_SME_MOP4 4-source forms carve `sz == 00` and the
                // `sz == 11, word<3> == 1` sub-regions out of the predicated
                // FP outer-product quadrant.
                decode_mop4(word, features, out);
            } else {
                decode_mopa_fp(word, features, out);
            }
        }
        0b101 => {
            // SME2/SVE2.1 contiguous multi-vector load/store carve the
            // `word<23> == 0` sub-region (the integer outer products set
            // `word<23>`); gated on FEAT_SME2.
            if sme2 && bit(word, 23) == 0 {
                sme2::decode_mem(word, out);
            } else if is_mop4(word) {
                // Integer MOP4 lives only in `sz == 11, word<3> == 1`.
                decode_mop4(word, features, out);
            } else {
                decode_mopa_int(word, features, out);
            }
        }
        0b110 => {
            if sme2 && bit(word, 24) == 1 {
                sme2::decode_mul(word, out);
            } else {
                decode_mova_add(word, features, out);
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

/// ZAda field width and reserved-bit check for a destination element size.
///
/// Returns `Some(zada)` when the tile-number field is in range and the
/// element-size-dependent reserved high bits of `word<2:0>` are zero, else
/// `None` (the encoding is then unallocated). The accumulator tile counts are
/// `.D`→8 (`word<2:0>`), `.S`→4 (`word<1:0>`, `word<2>` reserved), `.H`→2
/// (`word<0>`, `word<2:1>` reserved).
#[inline]
fn zada_field(word: u32, dst: VA) -> Option<u32> {
    match dst {
        VA::Sd => Some(bits(word, 0, 3)),
        VA::Ss => {
            if bit(word, 2) != 0 {
                return None;
            }
            Some(bits(word, 0, 2))
        }
        VA::Sh => {
            if bits(word, 1, 2) != 0 {
                return None;
            }
            Some(bits(word, 0, 1))
        }
        _ => None,
    }
}

/// Emit a predicated outer product: `ZAda.<dst>, Pn/M, Pm/M, Zn.<src>, Zm.<src>`.
#[inline]
fn emit_mopa(word: u32, code: Code, dst: VA, src: VA, out: &mut Instruction) {
    let zada = match zada_field(word, dst) {
        Some(z) => z,
        None => return,
    };
    let pn = bits(word, 10, 3);
    let pm = bits(word, 13, 3);
    let zn = bits(word, 5, 5);
    let zm = bits(word, 16, 5);
    out.set(code);
    out.push_operand(zreg(zada, dst));
    out.push_operand(preg_m(pn));
    out.push_operand(preg_m(pm));
    out.push_operand(zreg(zn, src));
    out.push_operand(zreg(zm, src));
}

/// `FMOPA`/`FMOPS`/`BFMOPA`/`BFMOPS`/`BMOPA`/`BMOPS` — floating-point, BFloat16
/// and the non-widening b16b16 outer products, accumulate (`S==0`) or subtract
/// (`S==1`) into a ZA tile.
///
/// Common fields: `ZAda` (low bits), `Zn = word<9:5>`, `Pn = word<12:10>`,
/// `Pm = word<15:13>`, `Zm = word<20:16>`, `S = word<4>`. The variant is a
/// function of `op = word<24>`, the size `word<23:22>`, `b21 = word<21>` and
/// `b3 = word<3>` (validated by oracle sweep — see `tests/sme_outerproduct_h2`).
fn decode_mopa_fp(word: u32, features: FeatureSet, out: &mut Instruction) {
    let s = bit(word, 4);
    let op24 = bit(word, 24);
    let sz = bits(word, 22, 2);
    let b21 = bit(word, 21);
    let b3 = bit(word, 3);

    // (code, dst-arr, src-arr, feature). `None` for any unallocated slot.
    let sel: Option<(Code, VA, VA, Feature)> = if op24 == 0 {
        match (sz, b21, b3) {
            // FP32: ZAda.S, Zn.S, Zm.S.
            (0b10, 0, 0) => Some((
                if s == 0 { Code::SmeFmopaS } else { Code::SmeFmopsS },
                VA::Ss, VA::Ss, Feature::Sme,
            )),
            // b16b16 BMOPA/BMOPS: ZAda.S, Zn.S, Zm.S.
            (0b10, 0, 1) => Some((
                if s == 0 { Code::SmeBmopaS } else { Code::SmeBmopsS },
                VA::Ss, VA::Ss, Feature::SmeB16b16,
            )),
            // FP8 → FP32 (FMOPA only, no subtract): ZAda.S, Zn.B, Zm.B.
            (0b10, 1, 0) if s == 0 => Some((Code::SmeFmopaB, VA::Ss, VA::Sb, Feature::SmeF8f32)),
            // FP8 → FP16 (FMOPA only): ZAda.H, Zn.B, Zm.B.
            (0b10, 1, 1) if s == 0 => Some((Code::SmeFmopaBh, VA::Sh, VA::Sb, Feature::SmeF8f16)),
            // FP64: ZAda.D, Zn.D, Zm.D.
            (0b11, 0, 0) => Some((
                if s == 0 { Code::SmeFmopaD } else { Code::SmeFmopsD },
                VA::Sd, VA::Sd, Feature::Sme,
            )),
            _ => None,
        }
    } else {
        // op24 == 1, sz == 10: 16-bit-input forms, src .H. (b21, b3) selects
        // BF16→FP32 / FP16→FP16 / FP16→FP32 / BF16→BF16.
        match (sz, b21, b3) {
            (0b10, 0, 0) => Some((
                if s == 0 { Code::SmeBfmopa } else { Code::SmeBfmops },
                VA::Ss, VA::Sh, Feature::Sme,
            )),
            (0b10, 0, 1) => Some((
                if s == 0 { Code::SmeFmopaHh } else { Code::SmeFmopsHh },
                VA::Sh, VA::Sh, Feature::SmeF16f16,
            )),
            (0b10, 1, 0) => Some((
                if s == 0 { Code::SmeFmopaH } else { Code::SmeFmopsH },
                VA::Ss, VA::Sh, Feature::Sme,
            )),
            (0b10, 1, 1) => Some((
                if s == 0 { Code::SmeBfmopaH } else { Code::SmeBfmopsH },
                VA::Sh, VA::Sh, Feature::SmeB16b16,
            )),
            _ => None,
        }
    };

    if let Some((code, dst, src, feat)) = sel {
        if !features.has(feat) {
            return;
        }
        emit_mopa(word, code, dst, src, out);
    }
}

// ---------------------------------------------------------------------------
// Outer products (integer): word<31:29> == 101.
// ---------------------------------------------------------------------------

/// `SMOPA`/`UMOPA`/`SUMOPA`/`USMOPA` and their `*OPS` (subtract) counterparts —
/// integer outer product into a ZA tile.
///
/// Signedness is the pair `(u0 = word<24>, u1 = word<21>)`. `S = word<4>`
/// selects accumulate(0)/subtract(1). The size `word<23:22>` and `b3 = word<3>`
/// select the source/destination element widths:
/// * `sz==10, b3==0` → 32-bit accumulator, `.B` sources (FEAT_SME); the
///   signedness for `.B` is `(0,0)`→S, `(0,1)`→SU, `(1,0)`→US, `(1,1)`→U.
/// * `sz==10, b3==1` → 32-bit accumulator, `.H` sources (FEAT_SME2); only the
///   symmetric `SMOPA`(`u0==0,u1==0`) / `UMOPA`(`u0==1,u1==0`) forms exist.
/// * `sz==11, b3==0` → 64-bit accumulator, `.H` sources (FEAT_SME, I16I64).
fn decode_mopa_int(word: u32, features: FeatureSet, out: &mut Instruction) {
    let u0 = bit(word, 24);
    let u1 = bit(word, 21);
    let sz = bits(word, 22, 2);
    let b3 = bit(word, 3);

    // (code, dst-arr, src-arr, feature) for the (sz, b3, u0, u1) slot.
    let sel: Option<(Code, VA, VA, Feature)> = match (sz, b3) {
        // 32-bit accumulator, byte sources (FEAT_SME).
        (0b10, 0) => {
            let code = mopa_int_code((u0, u1), bit(word, 4), IntForm::SByte);
            code.map(|c| (c, VA::Ss, VA::Sb, Feature::Sme))
        }
        // 32-bit accumulator, halfword sources (FEAT_SME2): SMOPA / UMOPA only.
        (0b10, 1) => {
            let code = mopa_int_code((u0, u1), bit(word, 4), IntForm::SHalf);
            code.map(|c| (c, VA::Ss, VA::Sh, Feature::Sme2))
        }
        // 64-bit accumulator, halfword sources (FEAT_SME, I16I64).
        (0b11, 0) => {
            let code = mopa_int_code((u0, u1), bit(word, 4), IntForm::DHalf);
            code.map(|c| (c, VA::Sd, VA::Sh, Feature::Sme))
        }
        _ => None,
    };

    if let Some((code, dst, src, feat)) = sel {
        if !features.has(feat) {
            return;
        }
        emit_mopa(word, code, dst, src, out);
    }
}

/// Which integer outer-product row family (selects the `(u0,u1)` → mnemonic map).
#[derive(Clone, Copy)]
enum IntForm {
    /// 32-bit accumulator, `.B` sources — all four signedness pairings.
    SByte,
    /// 32-bit accumulator, `.H` sources — only `SMOPA`/`UMOPA`.
    SHalf,
    /// 64-bit accumulator, `.H` sources — all four signedness pairings.
    DHalf,
}

/// Map `(u0, u1)` signedness and `S` to the integer outer-product `Code`.
#[inline]
fn mopa_int_code(uu: (u32, u32), s: u32, form: IntForm) -> Option<Code> {
    use Code::*;
    let acc_sub = |a: Code, sub: Code| if s == 0 { a } else { sub };
    Some(match form {
        IntForm::SByte => match uu {
            (0, 0) => acc_sub(SmeSmopaS, SmeSmopsS),
            (0, 1) => acc_sub(SmeSumopaS, SmeSumopsS),
            (1, 0) => acc_sub(SmeUsmopaS, SmeUsmopsS),
            (1, 1) => acc_sub(SmeUmopaS, SmeUmopsS),
            _ => return None,
        },
        IntForm::SHalf => match uu {
            (0, 0) => acc_sub(SmeSmopaHs, SmeSmopsHs),
            (1, 0) => acc_sub(SmeUmopaHs, SmeUmopsHs),
            _ => return None,
        },
        IntForm::DHalf => match uu {
            (0, 0) => acc_sub(SmeSmopaD, SmeSmopsD),
            (0, 1) => acc_sub(SmeSumopaD, SmeSumopsD),
            (1, 0) => acc_sub(SmeUsmopaD, SmeUsmopsD),
            (1, 1) => acc_sub(SmeUmopaD, SmeUmopsD),
            _ => return None,
        },
    })
}

// ---------------------------------------------------------------------------
// 4-source outer products (FEAT_SME_MOP4): the predicate-free quarter-tile
// `FMOP4A`/`SMOP4A`/... forms that replace `Pn`/`Pm` with `{Zm, Zm+1}` (and
// optionally `{Zn, Zn+1}`) register-pair sources.
// ---------------------------------------------------------------------------

/// `true` when `word` sits in the MOP4 sub-region of the outer-product quadrants
/// (`word<31:29>` is `100` or `101`). MOP4 carves `sz == 00` and, within the
/// `.d`/I16I64 `sz == 11` region, the `word<3> == 1` slot (`word<3>` is reserved
/// for the predicated `sz == 11` forms). The `sz == 01` slot is `*TMOPA` and the
/// predicated `sz == 10` forms keep `word<3>` for their own sub-typing, so MOP4
/// never overlaps them.
#[inline]
fn is_mop4(word: u32) -> bool {
    match bits(word, 22, 2) {
        0b00 => true,
        0b11 => bit(word, 3) == 1,
        _ => false,
    }
}

/// Map the integer MOP4 of a 32-bit accumulator (`sz == 00`, re-typed via
/// `word<15> == 1`) to `(Code, src_arr)`. `b3 = word<3>` selects byte (`0`) vs
/// halfword (`1`) sources; for the halfword sources only the symmetric
/// `SMOP4`/`UMOP4` rows exist. Returns `None` for the unallocated combinations.
#[inline]
fn mop4_int_s_code(uu: (u32, u32), b3: u32, s: u32) -> Option<(Code, VA)> {
    use Code::*;
    let acc_sub = |a: Code, sub: Code| if s == 0 { a } else { sub };
    Some(match (uu, b3) {
        // Byte sources.
        ((0, 0), 0) => (acc_sub(SmeSmop4aS, SmeSmop4sS), VA::Sb),
        ((0, 1), 0) => (acc_sub(SmeSumop4aS, SmeSumop4sS), VA::Sb),
        ((1, 0), 0) => (acc_sub(SmeUsmop4aS, SmeUsmop4sS), VA::Sb),
        ((1, 1), 0) => (acc_sub(SmeUmop4aS, SmeUmop4sS), VA::Sb),
        // Halfword sources (16-bit): only SMOPA / UMOPA.
        ((0, 0), 1) => (acc_sub(SmeSmop4aHs, SmeSmop4sHs), VA::Sh),
        ((1, 0), 1) => (acc_sub(SmeUmop4aHs, SmeUmop4sHs), VA::Sh),
        _ => return None,
    })
}

/// Map the integer MOP4 of a 64-bit accumulator (`sz == 11`) to a `Code`. All
/// four `(u0, u1)` signedness pairings are valid; the source is always `.h`.
#[inline]
fn mop4_int_d_code(uu: (u32, u32), s: u32) -> Option<Code> {
    use Code::*;
    let acc_sub = |a: Code, sub: Code| if s == 0 { a } else { sub };
    Some(match uu {
        (0, 0) => acc_sub(SmeSmop4aD, SmeSmop4sD),
        (0, 1) => acc_sub(SmeSumop4aD, SmeSumop4sD),
        (1, 0) => acc_sub(SmeUsmop4aD, SmeUsmop4sD),
        (1, 1) => acc_sub(SmeUmop4aD, SmeUmop4sD),
        _ => return None,
    })
}

/// Read a MOP4 `Zn` source: `word<5>` must be 0; `word<9>` selects a `{Zn,
/// Zn+1}` pair; the base register is `word<9:5> & 0x0E`. Returns `None` when the
/// low bit is set (reserved).
#[inline]
fn mop4_zn(word: u32, arr: VA) -> Option<Operand> {
    if bit(word, 5) != 0 {
        return None;
    }
    let base = bits(word, 5, 5) & 0x0E;
    Some(if bit(word, 9) == 1 {
        vec_pair(base, arr)
    } else {
        zreg(base, arr)
    })
}

/// Read a MOP4 `Zm` source: `word<16>` must be 0; `word<20>` selects a `{Zm,
/// Zm+1}` pair; the base register is `16 + (word<20:16> & 0x0E)`. Returns `None`
/// when the low bit is set (reserved).
#[inline]
fn mop4_zm(word: u32, arr: VA) -> Option<Operand> {
    if bit(word, 16) != 0 {
        return None;
    }
    let base = 16 + (bits(word, 16, 5) & 0x0E);
    Some(if bit(word, 20) == 1 {
        vec_pair(base, arr)
    } else {
        zreg(base, arr)
    })
}

/// A two-register consecutive vector group `{ z<n>.<T>, z<n+1>.<T> }`.
#[inline]
fn vec_pair(first: u32, arr: VA) -> Operand {
    Operand::SveVecGroup {
        first: Z[(first & 0x1f) as usize],
        count: 2,
        arr: Some(arr),
        range: false,
        stride: 1,
    }
}

/// `FMOP4A`/`FMOP4S`/`BFMOP4*`/`SMOP4*`/... — the 4-source outer products.
///
/// Operands are `ZAda.<dst>, Zn.<src>, Zm.<src>` where the `Zn`/`Zm` may be
/// single registers or `{Z, Z+1}` pairs (selected by `word<9>` / `word<20>`).
/// There are no governing predicates: `word<16>` (`Zm<0>`) and `word<14:10>`
/// are fixed at zero. The type is a function of `op29 = word<29>`, the size
/// `word<23:22>`, `op24 = word<24>`, `u1 = word<21>`, the integer-select
/// `b15 = word<15>` (only meaningful at `sz == 00`), `b3 = word<3>` and
/// `S = word<4>` — derived by oracle sweep (see `tests/sme_outerproduct_h2`).
fn decode_mop4(word: u32, features: FeatureSet, out: &mut Instruction) {
    // Structural MOP4 frame: `word<14:10>` (the low predicate-position bits) and
    // `Zm<0> = word<16>` are zero. `word<15>` is the int/FP type-select at
    // `sz == 00`, so it is excluded from this guard.
    if bits(word, 10, 5) != 0 || bit(word, 16) != 0 {
        return;
    }
    if !features.has(Feature::SmeMop4) {
        return;
    }

    let op29 = bit(word, 29);
    let op24 = bit(word, 24);
    let sz = bits(word, 22, 2);
    let u1 = bit(word, 21);
    let s = bit(word, 4);
    let b15 = bit(word, 15);
    let b3 = bit(word, 3);
    let acc_sub = |a: Code, sub: Code| if s == 0 { a } else { sub };

    // (code, dst-arr, src-arr).
    let sel: Option<(Code, VA, VA)> = match (op29, sz) {
        // Floating-point MOP4 quadrant, `sz == 00`. `word<15> == 1` re-types the
        // slot as an *integer* (`.s` accumulator, `.b` sources).
        (0, 0b00) if b15 == 0 => match (op24, u1, b3, s) {
            (0, 0, 0, _) => Some((acc_sub(Code::SmeFmop4aS, Code::SmeFmop4sS), VA::Ss, VA::Ss)),
            (0, 1, 0, 0) => Some((Code::SmeFmop4aB, VA::Ss, VA::Sb)),
            (0, 1, 1, 0) => Some((Code::SmeFmop4aBh, VA::Sh, VA::Sb)),
            (1, 0, 0, _) => Some((acc_sub(Code::SmeBfmop4aS, Code::SmeBfmop4sS), VA::Ss, VA::Sh)),
            (1, 0, 1, _) => Some((acc_sub(Code::SmeFmop4aHh, Code::SmeFmop4sHh), VA::Sh, VA::Sh)),
            (1, 1, 0, _) => Some((acc_sub(Code::SmeFmop4aHs, Code::SmeFmop4sHs), VA::Ss, VA::Sh)),
            (1, 1, 1, _) => Some((acc_sub(Code::SmeBfmop4aHh, Code::SmeBfmop4sHh), VA::Sh, VA::Sh)),
            _ => None,
        },
        // Integer MOP4 re-typed out of the FP `sz == 00` quadrant (`word<15>`==1):
        // `.s` accumulator, `.b`/`.h` sources. `(u0=op24, u1, b3)` select
        // signedness and source size.
        (0, 0b00) /* b15 == 1 */ => {
            mop4_int_s_code((op24, u1), b3, s).map(|(c, src)| (c, VA::Ss, src))
        }
        // FP64 MOP4: `sz == 11`, `word<3> == 1`, `word<15> == 0`.
        (0, 0b11) if b3 == 1 && b15 == 0 && op24 == 0 && u1 == 0 => {
            Some((acc_sub(Code::SmeFmop4aD, Code::SmeFmop4sD), VA::Sd, VA::Sd))
        }
        // Integer MOP4 (`word<29> == 1`): `.d` accumulator, `.h` sources.
        (1, 0b11) if b3 == 1 && b15 == 0 => {
            mop4_int_d_code((op24, u1), s).map(|c| (c, VA::Sd, VA::Sh))
        }
        _ => None,
    };

    let (code, dst, src) = match sel {
        Some(v) => v,
        None => return,
    };

    let zada = match zada_field(word, dst) {
        Some(z) => z,
        None => return,
    };
    let zn = match mop4_zn(word, src) {
        Some(z) => z,
        None => return,
    };
    let zm = match mop4_zm(word, src) {
        Some(z) => z,
        None => return,
    };

    out.set(code);
    out.push_operand(zreg(zada, dst));
    out.push_operand(zn);
    out.push_operand(zm);
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
fn decode_mova_add(word: u32, features: FeatureSet, out: &mut Instruction) {
    if bit(word, 24) != 0 {
        return;
    }
    // SME2 LUTI2/LUTI4 (ZT0) carve the `word<23> == 1, word<21:20> == 00,
    // word<19> == 1` sub-region (FEAT_LUT); the SME2 ZA tile-slice `MOV`/`MOVAZ`
    // (move multi-vectors) carve `word<21:19> == 000, word<18> == 1,
    // word<16> == 0` (FEAT_SME2). Route them first; their shells never overlap
    // the base MOVA / ADDHA / ADDVA encodings (which have word<18:17> == 00 with
    // word<21:20> selecting the family).
    if features.has(Feature::Lut)
        && bit(word, 23) == 1
        && bit(word, 21) == 0
        && bit(word, 19) == 1
    {
        // word<20> is the LUTI strided/consecutive selector (free here); word<21>
        // must be 0 and word<19> == 1 pins the LUTI ZT0 family.
        sme_lut::decode(word, out);
        return;
    }
    if features.has(Feature::Sme2)
        && bits(word, 19, 3) == 0b000
        && bit(word, 18) == 1
        && bit(word, 16) == 0
        && bits(word, 11, 2) == 0
    {
        sme_za_move::decode(word, out);
        return;
    }
    // Opcode `word<21:17>`: ADDHA/ADDVA are `01000`; MOVA is `0000x`.
    if bit(word, 21) == 0 && bit(word, 20) == 1 && bits(word, 17, 3) == 0 {
        decode_addha_addva(word, out);
    } else if bits(word, 18, 4) == 0 {
        // MOVA: word<21:18> == 0000 (direction in word<17>).
        decode_mova(word, features, out);
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
///
/// For the tile→vector direction only, `word<9>` (which lies above the 5-bit
/// `Zd`) selects the *zeroing* readout `MOVAZ` (`word<9> == 1`, FEAT_SME2) versus
/// the non-zeroing predicated `MOVA` (`word<9> == 0`). `MOVAZ` has no governing
/// predicate. In the vector→tile direction `word<9>` is part of `Zn`, so there is
/// no `MOVAZ` there.
fn decode_mova(word: u32, features: FeatureSet, out: &mut Instruction) {
    let size = bits(word, 22, 2);
    let q = bit(word, 16);
    let vertical = bit(word, 15) == 1;
    let rs = bits(word, 13, 2);
    let pg = bits(word, 10, 3);

    // `word<16>` (`Q`) is the `.Q`/`.D` selector, but it is only meaningful for
    // the 64-bit size `word<23:22> == 11` (`Q == 1` → `.Q`, `Q == 0` → `.D`). For
    // every smaller element size `Q` is RES0 — a set bit is UNDEFINED in LLVM
    // (e.g. `C0010000`, `C0410000`, `C0810000`).
    if q == 1 && size != 0b11 {
        return;
    }

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
        // `word<9>` selects the zeroing readout `MOVAZ` (1, FEAT_SME2, no
        // predicate) vs the predicated `MOVA` (0).
        let zd = bits(word, 0, 5);
        let field = bits(word, 5, 4);
        let (tile, imm) = split_tile_field(field, imm_bits);
        if bit(word, 9) == 1 {
            // MOVAZ — requires FEAT_SME2; renders without a governing predicate.
            // The `Pg` field `word<12:10>` is therefore RES0 (a set bit is
            // UNDEFINED, e.g. `C0020600`/`C0020A00`/`C0021200`).
            if !features.has(Feature::Sme2) || pg != 0 {
                return;
            }
            out.set(Code::SmeMovazTileToZ);
            out.push_operand(zreg(zd, arr));
            out.push_operand(tile_slice(tile, vertical, arr, rs, imm));
            return;
        }
        out.set(Code::SmeMovaTileToZ);
        out.push_operand(zreg(zd, arr));
        out.push_operand(preg_m(pg));
        out.push_operand(tile_slice(tile, vertical, arr, rs, imm));
    } else {
        // ZA tile slice ← Z: tile-slice field at word<3:0>, Zn = word<9:5>.
        // `word<4>` lies between the 4-bit `ZAd:imm` field and `Zn`; it is RES0,
        // so a set bit is UNDEFINED (e.g. `C00000FF` vs the valid `C00000EF`).
        if bit(word, 4) != 0 {
            return;
        }
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

    // `word<4>` is a fixed-zero bit for *every* contiguous ZA tile-slice
    // load/store (B/H/W/D and Q): the slice-select group `word<3:0>` packs the
    // tile number (high bits) and the slice index (low bits, width per element
    // size), and bit 4 lies above it. LLVM leaves `word<4> == 1` UNDEFINED
    // across all sizes (e.g. `E03779B1`, `E0623A9B`, `E0A66D13`, `E0F34739`).
    if bit(word, 4) != 0 {
        return;
    }

    // (code, arr, log2(bytes), index-imm-bits).
    let (code_ld, code_st, arr, log2, imm_bits): (Code, Code, VA, u8, u32) = if is_q {
        // The `.Q` form (`word<24> == 1`) is allocated only when the size field
        // `word<23:22> == 11`; `01`/`10` are reserved (`00` was already routed to
        // `LDR`/`STR` ZA by the dispatcher). Reject `E14CDA26`, `E16EF362`.
        if size != 0b11 {
            return;
        }
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
    // `LDR`/`STR` (ZA array vector) is `1110 0001 00 0 op 0 00000 0 Rv 000 Rn 0
    // imm4`. Only `op` (`word<21>`), `Rv` (`word<14:13>`), `Rn` (`word<9:5>`) and
    // `imm4` (`word<3:0>`) vary; the intervening fields are fixed zero. Reject
    // any encoding that sets `word<20:16>` (the `Rm` slot), `word<15>`,
    // `word<12:10>` or `word<4>` (e.g. the over-decode `E13779B1`).
    if bits(word, 16, 5) != 0 || bit(word, 15) != 0 || bits(word, 10, 3) != 0 || bit(word, 4) != 0 {
        return;
    }

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
    fn sme2_multivector_alu() {
        // SEL (predicate-as-counter): vgx2 comma-list, vgx4 range, plus sizes.
        check(0xC1208452, "sel     { z18.b, z19.b }, pn9, { z2.b, z3.b }, { z0.b, z1.b }");
        check(0xC1258010, "sel     { z16.b - z19.b }, pn8, { z0.b - z3.b }, { z4.b - z7.b }");
        check(0xC1608452, "sel     { z18.h, z19.h }, pn9, { z2.h, z3.h }, { z0.h, z1.h }");
        // S/U/F/BF clamp.
        check(0xC120C40F, "uclamp  { z14.b, z15.b }, z0.b, z0.b");
        check(0xC1A0CC0D, "uclamp  { z12.s - z15.s }, z0.s, z0.s");
        check(0xC120C40E, "sclamp  { z14.b, z15.b }, z0.b, z0.b");
        check(0xC160C00E, "fclamp  { z14.h, z15.h }, z0.h, z0.h");
        check(0xC1A0C80C, "fclamp  { z12.s - z15.s }, z0.s, z0.s");
        check(0xC120C000, "bfclamp { z0.h, z1.h }, z0.h, z0.h");
        check(0xC120C800, "bfclamp { z0.h - z3.h }, z0.h, z0.h");
        // ZIP/UZP: vgx2 (incl .q) and vgx4 (incl .q).
        check(0xC120D000, "zip     { z0.b, z1.b }, z0.b, z0.b");
        check(0xC120D400, "zip     { z0.q, z1.q }, z0.q, z0.q");
        check(0xC120D001, "uzp     { z0.b, z1.b }, z0.b, z0.b");
        check(0xC136E000, "zip     { z0.b - z3.b }, { z0.b - z3.b }");
        check(0xC136E002, "uzp     { z0.b - z3.b }, { z0.b - z3.b }");
        check(0xC137E000, "zip     { z0.q - z3.q }, { z0.q - z3.q }");
    }

    #[test]
    fn sme2_multivector_mem() {
        // Contiguous multi-vector loads/stores with a predicate-as-counter.
        check(0xA0004014, "ld1w    { z20.s, z21.s }, pn8/z, [x0, x0, lsl #0x2]");
        check(0xA000E814, "ld1d    { z20.d - z23.d }, pn10/z, [x0, x0, lsl #0x3]");
        check(0xA0000000, "ld1b    { z0.b, z1.b }, pn8/z, [x0, x0]");
        check(0xA0414000, "ld1w    { z0.s, z1.s }, pn8/z, [x0, #0x2, mul vl]");
        check(0xA041E000, "ld1d    { z0.d - z3.d }, pn8/z, [x0, #0x4, mul vl]");
        check(0xA0480000, "ld1b    { z0.b, z1.b }, pn8/z, [x0, #-16, mul vl]");
        check(0xA0404000, "ld1w    { z0.s, z1.s }, pn8/z, [x0]");
        check(0xA0004015, "ldnt1w  { z20.s, z21.s }, pn8/z, [x0, x0, lsl #0x2]");
        check(0xA0200001, "stnt1b  { z0.b, z1.b }, pn8, [x0, x0]");
        check(0xA0204014, "st1w    { z20.s, z21.s }, pn8, [x0, x0, lsl #0x2]");
    }

    #[test]
    fn sme2_multivector_strided() {
        // word<24> == 1: the strided (non-consecutive) register lists. 2-register
        // groups step by 8 (`z16, z24`); 4-register groups step by 4.
        check(0xA1206710, "st1d    { z16.d, z24.d }, pn9, [x24, x0, lsl #0x3]");
        check(0xA1204983, "st1w    { z3.s, z11.s }, pn10, [x12, x0, lsl #0x2]");
        check(0xA1004DB1, "ld1w    { z17.s, z25.s }, pn11/z, [x13, x0, lsl #0x2]");
        check(0xA120A541, "st1h    { z1.h, z5.h, z9.h, z13.h }, pn9, [x10, x0, lsl #0x1]");
        // Strided nontemporal (NT = word<3>), and the byte form (no LSL shown).
        check(0xA1206718, "stnt1d  { z16.d, z24.d }, pn9, [x24, x0, lsl #0x3]");
        check(0xA1004000, "ld1w    { z0.s, z8.s }, pn8/z, [x0, x0, lsl #0x2]");
        // Strided scalar+immediate (`MUL VL`): the offset is `imm4 * count`.
        check(0xA1606710, "st1d    { z16.d, z24.d }, pn9, [x24]");
        check(0xA1414000, "ld1w    { z0.s, z8.s }, pn8/z, [x0, #0x2, mul vl]");
        // vgx4 strided leaves word<2> reserved (must be zero) -> Invalid.
        let bytes = 0xA120E714u32.to_le_bytes();
        let mut dec = Decoder::new(&bytes, 0x1000, DecoderOptions::default());
        assert!(dec.decode().is_invalid());
    }

    #[test]
    fn sme2_multivector_shift_narrow() {
        // 4-vector -> 1-vector saturating rounding shift right narrow; the dest
        // element (.b/.h), source element (.s/.d) and shift range come from the
        // `tsz` size field. `#shift` renders as a plain decimal.
        check(0xC161D920, "uqrshr  z0.b, { z8.s - z11.s }, #31");
        check(0xC161D998, "sqrshr  z24.b, { z12.s - z15.s }, #31");
        check(0xC160DC9A, "sqrshrn z26.b, { z4.s - z7.s }, #32");
        check(0xC161DE2B, "uqrshrn z11.b, { z16.s - z19.s }, #31");
        check(0xC164DB5D, "sqrshru z29.b, { z24.s - z27.s }, #28");
        check(0xC162DFD7, "sqrshrun z23.b, { z28.s - z31.s }, #30");
        // .h dest / .d source, both shift sub-ranges (`#64`..`#33` and `#32`..`#1`).
        check(0xC1A0D900, "sqrshr  z0.h, { z8.d - z11.d }, #64");
        check(0xC1E0D900, "sqrshr  z0.h, { z8.d - z11.d }, #32");
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
