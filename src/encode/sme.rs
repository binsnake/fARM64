//! Encoder for the SME (Scalable Matrix Extension) group — the inverse of
//! [`crate::decode::sme`].
//!
//! Gated behind `#[cfg(feature = "sme")]`. Without the feature the [`encode`]
//! stub returns [`EncodeError::Unsupported`] and `is_sme` reports `false`, so the
//! default build still compiles. With it, every `Sme*` [`Code`] the decoder
//! produces is inverted: dispatch on [`Instruction::code`] to a family encoder,
//! read the operands the decoder pushed (the binja `z`-prefixed tile-slice
//! spellings included), and pack the exact bitfields in reverse. It reconstructs
//! the word purely from the instruction's semantics — it never reads
//! [`Instruction::word`]. Total and panic-free.
//!
//! `SMSTART`/`SMSTOP` are *not* handled here: they are `MSR (immediate)` PSTATE
//! encodings owned by [`crate::encode::branch_sys`] (mirroring the decoder).
//!
//! ## Field layout recap (inverted from the decoder)
//!
//! All SME reserved-region words have `word<31> == 1`, `op0 == 0b0000`, and are
//! split by `word<31:29>`:
//!
//! | `word<31:29>` | base | family |
//! |-|-|-|
//! | `100` | `0x8000_0000` | outer-product FP/BF16 |
//! | `101` | `0xA000_0000` | outer-product integer |
//! | `110` | `0xC000_0000` | `MOVA` / `ADDHA` / `ADDVA` |
//! | `111` | `0xE000_0000` | ZA load/store |

#[cfg(not(feature = "sme"))]
mod stub {
    use crate::encode::EncodeError;
    use crate::instruction::Instruction;
    use crate::mnemonic::Code;

    /// Without the `sme` feature the encoder declines the whole group.
    #[inline]
    pub fn encode(_insn: &Instruction) -> Result<u32, EncodeError> {
        Err(EncodeError::Unsupported)
    }

    /// Without the `sme` feature, report that no code is an SME code so the
    /// dispatcher falls through to its `Unsupported` arm.
    #[inline]
    pub fn is_sme(_code: Code) -> bool {
        false
    }
}

#[cfg(not(feature = "sme"))]
pub use stub::{encode, is_sme};

#[cfg(feature = "sme")]
pub use imp::{encode, is_sme};

#[cfg(feature = "sme")]
mod imp {
    use crate::encode::EncodeError;
    use crate::enums::{ExtendType, VectorArrangement as VA};
    use crate::instruction::Instruction;
    use crate::mnemonic::Code;
    use crate::operand::{Operand, SliceIndicator, SveMemMode};
    use crate::register::{Register, RegClass};

    type R = Result<u32, EncodeError>;

    /// `true` for every [`Code`] produced by the SME decoder. (The `Sme*` enum
    /// variants exist regardless of the `sme` feature, so this is a plain match.)
    #[inline]
    pub fn is_sme(code: Code) -> bool {
        use Code::*;
        if matches!(
            code,
            // Outer products (FP / BF16).
            SmeFmopaH | SmeFmopaS | SmeFmopaD | SmeFmopsH | SmeFmopsS | SmeFmopsD
                | SmeBfmopa | SmeBfmops
            // Outer products (integer).
                | SmeSmopaS | SmeSmopaD | SmeSmopsS | SmeSmopsD
                | SmeUmopaS | SmeUmopaD | SmeUmopsS | SmeUmopsD
                | SmeSumopaS | SmeSumopaD | SmeSumopsS | SmeSumopsD
                | SmeUsmopaS | SmeUsmopaD | SmeUsmopsS | SmeUsmopsD
            // MOVA / ADDHA / ADDVA.
                | SmeMovaZToTile | SmeMovaTileToZ | SmeAddha | SmeAddva
            // ZA load/store.
                | SmeLd1bZa | SmeLd1hZa | SmeLd1wZa | SmeLd1dZa | SmeLd1qZa
                | SmeSt1bZa | SmeSt1hZa | SmeSt1wZa | SmeSt1dZa | SmeSt1qZa
                | SmeLdrZa | SmeStrZa
            // SME2/SVE2.1 contiguous/strided multi-vector load/store.
                | SmeLd1bMV | SmeLd1hMV | SmeLd1wMV | SmeLd1dMV
                | SmeLdnt1bMV | SmeLdnt1hMV | SmeLdnt1wMV | SmeLdnt1dMV
                | SmeSt1bMV | SmeSt1hMV | SmeSt1wMV | SmeSt1dMV
                | SmeStnt1bMV | SmeStnt1hMV | SmeStnt1wMV | SmeStnt1dMV
            // SME2 multi-vector shift-right-narrow.
                | SmeSqrshr | SmeUqrshr | SmeSqrshrn | SmeUqrshrn | SmeSqrshru | SmeSqrshrun
        ) {
            return true;
        }
        // SME2 multi-vector + *TMOPA forms are table-driven (one Code per row),
        // as are the multi-vector ALU forms (SEL/CLAMP/ZIP/UZP).
        crate::decode::sme::sme2::form_for_code(code).is_some()
            || crate::decode::sme::sme2::alu_form_for_code(code).is_some()
    }

    /// Encode an SME instruction by inverting its decoder.
    #[inline]
    pub fn encode(insn: &Instruction) -> R {
        use Code::*;
        match insn.code() {
            SmeFmopaH | SmeFmopaS | SmeFmopaD | SmeFmopsH | SmeFmopsS | SmeFmopsD | SmeBfmopa
            | SmeBfmops => enc_mopa_fp(insn),
            SmeSmopaS | SmeSmopaD | SmeSmopsS | SmeSmopsD | SmeUmopaS | SmeUmopaD | SmeUmopsS
            | SmeUmopsD | SmeSumopaS | SmeSumopaD | SmeSumopsS | SmeSumopsD | SmeUsmopaS
            | SmeUsmopaD | SmeUsmopsS | SmeUsmopsD => enc_mopa_int(insn),
            SmeAddha | SmeAddva => enc_addha_addva(insn),
            SmeMovaZToTile | SmeMovaTileToZ => enc_mova(insn),
            SmeLd1bZa | SmeLd1hZa | SmeLd1wZa | SmeLd1dZa | SmeLd1qZa | SmeSt1bZa | SmeSt1hZa
            | SmeSt1wZa | SmeSt1dZa | SmeSt1qZa => enc_ld1_st1_za(insn),
            SmeLdrZa | SmeStrZa => enc_ldr_str_za(insn),
            SmeLd1bMV | SmeLd1hMV | SmeLd1wMV | SmeLd1dMV | SmeLdnt1bMV | SmeLdnt1hMV
            | SmeLdnt1wMV | SmeLdnt1dMV | SmeSt1bMV | SmeSt1hMV | SmeSt1wMV | SmeSt1dMV
            | SmeStnt1bMV | SmeStnt1hMV | SmeStnt1wMV | SmeStnt1dMV => enc_mem(insn),
            SmeSqrshr | SmeUqrshr | SmeSqrshrn | SmeUqrshrn | SmeSqrshru | SmeSqrshrun => {
                enc_narrow_shift(insn)
            }
            other => {
                // Multi-vector ALU (SEL/CLAMP/ZIP/UZP) is table-driven; fall back
                // to the multiply-into-ZA / *TMOPA table otherwise.
                if crate::decode::sme::sme2::alu_form_for_code(other).is_some() {
                    enc_alu(insn, other)
                } else {
                    enc_sme2(insn, other)
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Shared operand readers.
    // -----------------------------------------------------------------------

    /// A `Z` (scalable-vector) register number at operand `n` (the decoder uses
    /// `Z` registers for `ZAda`/`Zn`/`Zm`).
    #[inline]
    fn z(insn: &Instruction, n: usize) -> Result<u32, EncodeError> {
        match insn.op(n) {
            Operand::Reg { reg, .. } if reg.class() == RegClass::Sve => Ok(reg.number() as u32),
            _ => Err(EncodeError::InvalidOperand),
        }
    }

    /// A predicate `P` register number at operand `n` (3-bit field `0..=7`).
    #[inline]
    fn p3(insn: &Instruction, n: usize) -> Result<u32, EncodeError> {
        match insn.op(n) {
            Operand::Reg { reg, .. } if reg.class() == RegClass::Predicate => {
                let v = reg.number() as u32;
                if v > 7 {
                    return Err(EncodeError::InvalidOperand);
                }
                Ok(v)
            }
            _ => Err(EncodeError::InvalidOperand),
        }
    }

    /// The 2-bit `Rs`/`Rv` slice-select field from a `W12..W15` register.
    #[inline]
    fn wsel_field(sel: Register) -> Result<u32, EncodeError> {
        if sel.class() != RegClass::Gp {
            return Err(EncodeError::InvalidOperand);
        }
        let n = sel.number() as u32;
        if !(12..=15).contains(&n) {
            return Err(EncodeError::InvalidOperand);
        }
        Ok(n - 12)
    }

    // -----------------------------------------------------------------------
    // Outer products (FP / BF16): word<31:29> == 100, base 0x8000_0000.
    // -----------------------------------------------------------------------

    /// `FMOPA`/`FMOPS`/`BFMOPA`/`BFMOPS`. Operands (as the decoder pushed them):
    /// `[ZAda, Pn/M, Pm/M, Zn, Zm]`.
    fn enc_mopa_fp(insn: &Instruction) -> R {
        use Code::*;
        let code = insn.code();
        // S = accumulate(0)/subtract(1); op24 + sz + (b21 for 16-bit) select form.
        let (s, op24, sz, b21, zada_is_d) = match code {
            SmeFmopaS => (0u32, 0u32, 0b10u32, 0u32, false),
            SmeFmopsS => (1, 0, 0b10, 0, false),
            SmeFmopaD => (0, 0, 0b11, 0, true),
            SmeFmopsD => (1, 0, 0b11, 0, true),
            // op24 == 1, sz == 10: word<21> picks FMOPA(1)/BFMOPA(0).
            SmeFmopaH => (0, 1, 0b10, 1, false),
            SmeFmopsH => (1, 1, 0b10, 1, false),
            SmeBfmopa => (0, 1, 0b10, 0, false),
            _ /* SmeBfmops */ => (1, 1, 0b10, 0, false),
        };

        let zada = z(insn, 0)?;
        let pn = p3(insn, 1)?;
        let pm = p3(insn, 2)?;
        let zn = z(insn, 3)?;
        let zm = z(insn, 4)?;

        // ZAda field width: 3 bits for `.D` (0..7), 2 bits for `.S` (0..3).
        if zada_is_d {
            if zada > 7 {
                return Err(EncodeError::InvalidOperand);
            }
        } else if zada > 3 {
            return Err(EncodeError::InvalidOperand);
        }

        let word = 0x8000_0000
            | (op24 << 24)
            | (sz << 22)
            | (b21 << 21)
            | (zm << 16)
            | (pm << 13)
            | (pn << 10)
            | (zn << 5)
            | (s << 4)
            | zada;
        Ok(word)
    }

    // -----------------------------------------------------------------------
    // Outer products (integer): word<31:29> == 101, base 0xA000_0000.
    // -----------------------------------------------------------------------

    /// `[US]MOPA`/`[US]MOPS` and the mixed-sign forms. Operands:
    /// `[ZAda, Pn/M, Pm/M, Zn, Zm]`. Signedness is `(u0=word<24>, u1=word<21>)`;
    /// `S = word<4>`; size `word<23:22>` is `10` (32-bit) or `11` (64-bit).
    fn enc_mopa_int(insn: &Instruction) -> R {
        use Code::*;
        let code = insn.code();
        // (u0, u1, S, is64).
        let (u0, u1, s, is64) = match code {
            SmeSmopaS => (0u32, 0u32, 0u32, false),
            SmeSmopsS => (0, 0, 1, false),
            SmeUmopaS => (1, 1, 0, false),
            SmeUmopsS => (1, 1, 1, false),
            SmeSumopaS => (0, 1, 0, false),
            SmeSumopsS => (0, 1, 1, false),
            SmeUsmopaS => (1, 0, 0, false),
            SmeUsmopsS => (1, 0, 1, false),
            SmeSmopaD => (0, 0, 0, true),
            SmeSmopsD => (0, 0, 1, true),
            SmeUmopaD => (1, 1, 0, true),
            SmeUmopsD => (1, 1, 1, true),
            SmeSumopaD => (0, 1, 0, true),
            SmeSumopsD => (0, 1, 1, true),
            SmeUsmopaD => (1, 0, 0, true),
            _ /* SmeUsmopsD */ => (1, 0, 1, true),
        };
        let sz = if is64 { 0b11u32 } else { 0b10 };

        let zada = z(insn, 0)?;
        let pn = p3(insn, 1)?;
        let pm = p3(insn, 2)?;
        let zn = z(insn, 3)?;
        let zm = z(insn, 4)?;

        if is64 {
            if zada > 7 {
                return Err(EncodeError::InvalidOperand);
            }
        } else if zada > 3 {
            return Err(EncodeError::InvalidOperand);
        }

        let word = 0xA000_0000
            | (u0 << 24)
            | (sz << 22)
            | (u1 << 21)
            | (zm << 16)
            | (pm << 13)
            | (pn << 10)
            | (zn << 5)
            | (s << 4)
            | zada;
        Ok(word)
    }

    // -----------------------------------------------------------------------
    // ADDHA / ADDVA: word<31:29> == 110, opcode word<21:17> == 01000.
    // Fixed bits: b23 == 1, b20 == 1 (=> base 0xC090_0000).
    // -----------------------------------------------------------------------

    /// `ADDHA`/`ADDVA`. Operands: `[ZAda, Pn/M, Pm/M, Zn]`. `V = word<16>` picks
    /// ADDHA(0)/ADDVA(1); `word<22>` is the element size (`.S` 2-bit / `.D`
    /// 3-bit ZAda).
    fn enc_addha_addva(insn: &Instruction) -> R {
        let v = if insn.code() == Code::SmeAddva { 1u32 } else { 0 };

        let zada = z(insn, 0)?;
        let pn = p3(insn, 1)?;
        let pm = p3(insn, 2)?;
        let zn = z(insn, 3)?;

        // Element size from the ZAda arrangement (`.S` or `.D`).
        let is64 = match insn.op(0) {
            Operand::Reg { arr: Some(VA::Sd), .. } => true,
            Operand::Reg { arr: Some(VA::Ss), .. } => false,
            _ => return Err(EncodeError::InvalidOperand),
        };
        let sz22 = if is64 { 1u32 } else { 0 };
        if is64 {
            if zada > 7 {
                return Err(EncodeError::InvalidOperand);
            }
        } else if zada > 3 {
            return Err(EncodeError::InvalidOperand);
        }

        let word = 0xC090_0000
            | (sz22 << 22)
            | (pm << 13)
            | (pn << 10)
            | (zn << 5)
            | (v << 16)
            | zada;
        Ok(word)
    }

    // -----------------------------------------------------------------------
    // MOVA: word<31:29> == 110, word<24> == 0, opcode word<21:18> == 0000.
    // Direction in word<17>: 0 => ZA tile <- Z, 1 => Z <- ZA tile.
    // -----------------------------------------------------------------------

    /// `MOVA` (both directions). The element size determines how the shared
    /// 4-bit `ZAd:imm` field splits into `(tile, slice-index)`: `.B` 0/4, `.H`
    /// 1/3, `.S` 2/2, `.D` 3/1, `.Q` 4/0. `Q` (size `11`, `word<16> == 1`) has no
    /// index.
    fn enc_mova(insn: &Instruction) -> R {
        let to_vector = insn.code() == Code::SmeMovaTileToZ;

        // The tile-slice operand is operand 2 (TileToZ) or operand 0 (ZToTile).
        let (zd_or_zn_idx, slice_idx, pg_reg, vec_idx) = if to_vector {
            // MOVA <Zd>, <Pg>/M, <tile-slice>.
            (0usize, 2usize, 1usize, 0usize)
        } else {
            // MOVA <tile-slice>, <Pg>/M, <Zn>.
            (2usize, 0usize, 1usize, 2usize)
        };
        let _ = zd_or_zn_idx;

        // Read the tile-slice operand fields.
        let (tile_reg, slice, arr, sel, imm, has_imm) = match insn.op(slice_idx) {
            Operand::SmeTileSlice {
                reg,
                slice,
                arr,
                sel,
                imm,
                has_imm,
            } => (reg, slice, arr, sel, imm, has_imm),
            _ => return Err(EncodeError::InvalidOperand),
        };

        let arr = arr.ok_or(EncodeError::InvalidOperand)?;
        // (size<23:22>, q<16>, index-bit-width).
        let (size, q, imm_bits) = match arr {
            VA::Sb => (0b00u32, 0u32, 4u32),
            VA::Sh => (0b01, 0, 3),
            VA::Ss => (0b10, 0, 2),
            VA::Sd => (0b11, 0, 1),
            VA::Sq => (0b11, 1, 0),
            _ => return Err(EncodeError::InvalidOperand),
        };

        let vertical = match slice {
            SliceIndicator::Vertical => 1u32,
            SliceIndicator::Horizontal => 0,
            SliceIndicator::None => return Err(EncodeError::InvalidOperand),
        };
        let rs = wsel_field(sel)?;
        let pg = p3(insn, pg_reg)?;
        let tile = tile_reg.number() as u32;

        // Recombine the 4-bit ZAd:imm field: tile in the high (4 - imm_bits) bits,
        // index in the low imm_bits.
        if imm_bits == 0 {
            // `.Q`: no index; the whole 4-bit field is the tile (0..15).
            if has_imm {
                return Err(EncodeError::InvalidOperand);
            }
            if tile > 0xf {
                return Err(EncodeError::InvalidOperand);
            }
        } else {
            if !has_imm {
                return Err(EncodeError::InvalidOperand);
            }
            let imm_max = (1i32 << imm_bits) - 1;
            if !(0..=imm_max).contains(&(imm as i32)) {
                return Err(EncodeError::InvalidImmediate);
            }
            let tile_max = (1u32 << (4 - imm_bits)) - 1;
            if tile > tile_max {
                return Err(EncodeError::InvalidOperand);
            }
        }
        let imm_u = (imm as u32) & ((1u32 << imm_bits).wrapping_sub(1));
        let field = (tile << imm_bits) | imm_u;

        let base = 0xC000_0000 | (size << 22) | (q << 16) | (vertical << 15) | (rs << 13) | (pg << 10);

        let word = if to_vector {
            // word<17> == 1; field at word<8:5>, Zd at word<4:0>.
            let zd = z(insn, vec_idx)?;
            base | (1u32 << 17) | (field << 5) | zd
        } else {
            // word<17> == 0; field at word<3:0>, Zn at word<9:5>.
            let zn = z(insn, vec_idx)?;
            base | (zn << 5) | field
        };
        Ok(word)
    }

    // -----------------------------------------------------------------------
    // ZA-array LD1*/ST1*: word<31:29> == 111, base 0xE000_0000.
    // -----------------------------------------------------------------------

    /// `LD1B/H/W/D/Q` and `ST1B/H/W/D/Q` (ZA array vector). Operands:
    /// `[tile-slice(tile=0), Pg, mem]`. `is_q = word<24>`, size `word<23:22>`,
    /// `is_store = word<21>`, `Rm = word<20:16>`, `V = word<15>`,
    /// `Rs = word<14:13>`, `Pg = word<12:10>`, `Rn = word<9:5>`, index imm in the
    /// low bits of `word<3:0>`.
    fn enc_ld1_st1_za(insn: &Instruction) -> R {
        use Code::*;
        let code = insn.code();
        // (is_q, size, is_store, imm_bits).
        let (is_q, size, is_store, imm_bits) = match code {
            SmeLd1bZa => (0u32, 0b00u32, 0u32, 4u32),
            SmeLd1hZa => (0, 0b01, 0, 3),
            SmeLd1wZa => (0, 0b10, 0, 2),
            SmeLd1dZa => (0, 0b11, 0, 1),
            SmeLd1qZa => (1, 0b11, 0, 0),
            SmeSt1bZa => (0, 0b00, 1, 4),
            SmeSt1hZa => (0, 0b01, 1, 3),
            SmeSt1wZa => (0, 0b10, 1, 2),
            SmeSt1dZa => (0, 0b11, 1, 1),
            _ /* SmeSt1qZa */ => (1, 0b11, 1, 0),
        };

        // Operand 0: the tile-slice (tile is implicitly 0).
        let (slice, sel, imm, has_imm) = match insn.op(0) {
            Operand::SmeTileSlice {
                reg,
                slice,
                sel,
                imm,
                has_imm,
                ..
            } => {
                // Tile is always 0 for these forms.
                if reg.number() != 0 {
                    return Err(EncodeError::InvalidOperand);
                }
                (slice, sel, imm, has_imm)
            }
            _ => return Err(EncodeError::InvalidOperand),
        };

        let vertical = match slice {
            SliceIndicator::Vertical => 1u32,
            SliceIndicator::Horizontal => 0,
            SliceIndicator::None => return Err(EncodeError::InvalidOperand),
        };
        let rs = wsel_field(sel)?;

        // Operand 1: the governing predicate (3-bit).
        let pg = p3(insn, 1)?;

        // Operand 2: `[Xn{, Xm, LSL #log2}]`.
        let (rn, rm) = match insn.op(2) {
            Operand::MemExt { base, index, .. } => {
                if base.class() != RegClass::Gp || index.class() != RegClass::Gp {
                    return Err(EncodeError::InvalidOperand);
                }
                (base.number() as u32, index.number() as u32)
            }
            _ => return Err(EncodeError::InvalidOperand),
        };

        // Slice index immediate.
        let imm_field = if imm_bits == 0 {
            if has_imm {
                return Err(EncodeError::InvalidOperand);
            }
            0u32
        } else {
            if !has_imm {
                return Err(EncodeError::InvalidOperand);
            }
            let imm_max = (1i32 << imm_bits) - 1;
            if !(0..=imm_max).contains(&(imm as i32)) {
                return Err(EncodeError::InvalidImmediate);
            }
            (imm as u32) & ((1u32 << imm_bits) - 1)
        };

        let word = 0xE000_0000
            | (is_q << 24)
            | (size << 22)
            | (is_store << 21)
            | (rm << 16)
            | (vertical << 15)
            | (rs << 13)
            | (pg << 10)
            | (rn << 5)
            | imm_field;
        Ok(word)
    }

    // -----------------------------------------------------------------------
    // LDR/STR ZA (whole array vector): word<31:29> == 111, word<24> == 1,
    // size word<23:22> == 00 (=> base 0xE100_0000).
    // -----------------------------------------------------------------------

    /// `LDR`/`STR` (ZA array vector). Operands: `[za[Wv, #imm4], SveMem]`.
    /// `op = word<21>` picks LDR(0)/STR(1); `Wv = w12 + word<14:13>`;
    /// `imm4 = word<3:0>` (also the `MUL VL` multiple); `Rn = word<9:5>`.
    fn enc_ldr_str_za(insn: &Instruction) -> R {
        let is_store = if insn.code() == Code::SmeStrZa { 1u32 } else { 0 };

        // Operand 0: the whole-array select `za[Wv, #imm4]`.
        let (sel, imm4) = match insn.op(0) {
            Operand::SmeTileSlice {
                reg,
                slice,
                arr,
                sel,
                imm,
                ..
            } => {
                // Whole-array form: no tile, no slice direction, no arrangement.
                if reg != Register::None
                    || slice != SliceIndicator::None
                    || arr.is_some()
                {
                    return Err(EncodeError::InvalidOperand);
                }
                (sel, imm as i32)
            }
            _ => return Err(EncodeError::InvalidOperand),
        };
        let rv = wsel_field(sel)?;
        if !(0..=0xf).contains(&imm4) {
            return Err(EncodeError::InvalidImmediate);
        }
        let imm_field = imm4 as u32;

        // Operand 1: `[Xn{, #imm4, MUL VL}]` — must agree with the select imm4.
        let rn = match insn.op(1) {
            Operand::SveMem { base, imm, .. } => {
                if base.class() != RegClass::Gp {
                    return Err(EncodeError::InvalidOperand);
                }
                if imm != imm4 {
                    return Err(EncodeError::InvalidImmediate);
                }
                base.number() as u32
            }
            _ => return Err(EncodeError::InvalidOperand),
        };

        let word = 0xE100_0000 | (is_store << 21) | (rv << 13) | (rn << 5) | imm_field;
        Ok(word)
    }

    // -----------------------------------------------------------------------
    // SME2 multi-vector + *TMOPA (table-driven, inverse of `decode::sme::sme2`).
    // -----------------------------------------------------------------------

    use crate::decode::sme::sme2::{form_for_code, pdep, Form, Sh};

    /// Encode an SME2 multi-vector / `*TMOPA` form by scattering its operand
    /// fields back into the matched [`Form`]'s opcode template `val`. The inverse
    /// of [`crate::decode::sme::sme2`]: read the structured operands the decoder
    /// produced, recover each field's value, and [`pdep`] it into place.
    fn enc_sme2(insn: &Instruction, code: Code) -> R {
        let f = form_for_code(code).ok_or(EncodeError::Unsupported)?;
        if f.shape == Sh::Tmopa {
            return enc_sme2_tmopa(insn, f);
        }
        // Operand 0: the `za.<T>[Ws, off{:..}{, vgxN}]` destination.
        let (arr, sel, off, span, vg) = match insn.op(0) {
            Operand::SmeZaSlice { arr, sel, off, span, vg } => (arr, sel, off, span, vg),
            _ => return Err(EncodeError::InvalidOperand),
        };
        if arr != Some(f.acc) || span != f.span || vg != f.vg {
            return Err(EncodeError::InvalidOperand);
        }
        let mut word = f.val;
        word |= pdep(ws_field(sel)?, f.ws);
        word |= pdep(off_field(off, f.span)?, f.off);

        // Sources (operands 1, 2) per shape.
        match f.shape {
            Sh::SingleSingle => {
                word |= pdep(z_single(insn, 1, f.src)?, f.zn);
                word |= pdep(z_single(insn, 2, f.src)?, f.zm);
            }
            Sh::SingleIdx => {
                word |= pdep(z_single(insn, 1, f.src)?, f.zn);
                let (zm, idx) = z_single_idx(insn, 2, f.src)?;
                word |= pdep(zm, f.zm);
                word |= pdep(idx, f.idx);
            }
            Sh::GroupSingle => {
                word |= pdep(group_field(insn, 1, f.vg, f.src, f.zn)?, f.zn);
                word |= pdep(z_single(insn, 2, f.src)?, f.zm);
            }
            Sh::GroupIdx => {
                word |= pdep(group_field(insn, 1, f.vg, f.src, f.zn)?, f.zn);
                let (zm, idx) = z_single_idx(insn, 2, f.src)?;
                word |= pdep(zm, f.zm);
                word |= pdep(idx, f.idx);
            }
            Sh::GroupGroup => {
                word |= pdep(group_field(insn, 1, f.vg, f.src, f.zn)?, f.zn);
                word |= pdep(group_field(insn, 2, f.vg, f.src, f.zm)?, f.zm);
            }
            Sh::GroupOnly => {
                word |= pdep(group_field(insn, 1, f.vg, f.src, f.zn)?, f.zn);
            }
            Sh::GroupIdxB => {
                // FP8 FVDOTB/FVDOTT: a two-register group with a `vgx4` ZA dest.
                word |= pdep(group_field(insn, 1, 2, f.src, f.zn)?, f.zn);
                let (zm, idx) = z_single_idx(insn, 2, f.src)?;
                word |= pdep(zm, f.zm);
                word |= pdep(idx, f.idx);
            }
            Sh::Tmopa => unreachable!(),
        }
        Ok(word)
    }

    /// Encode a `*TMOPA` form: `ZAda, { Zn, Zn+1 }.<T>, Zm.<T>, Zk[idx]`.
    fn enc_sme2_tmopa(insn: &Instruction, f: &Form) -> R {
        // Operand 0: ZAda tile (SmeTile, slice none).
        let zada = match insn.op(0) {
            Operand::SmeTile { tile, .. } => (tile & 0x0f) as u32,
            _ => return Err(EncodeError::InvalidOperand),
        };
        // Operand 1: the consecutive even pair { Zn, Zn+1 }.
        let znp = match insn.op(1) {
            Operand::SveVecGroup { first, count, arr, .. } if count == 2 && arr == Some(f.src) => {
                let n = first.number() as u32;
                if n & 1 != 0 {
                    return Err(EncodeError::InvalidOperand);
                }
                n >> 1
            }
            _ => return Err(EncodeError::InvalidOperand),
        };
        // Operand 2: the single Zm.<src>.
        let zm = z_single(insn, 2, f.src)?;
        // Operand 3: the restricted Zk[idx].
        let (zk, idx) = match insn.op(3) {
            Operand::Reg { reg, arr: None, lane: Some(l), .. } if reg.class() == RegClass::Sve => {
                (reg.number() as u32, l as u32)
            }
            _ => return Err(EncodeError::InvalidOperand),
        };
        // Restricted Zk: z20..z23 → 0..3, z28..z31 → 4..7.
        let zkf = match zk {
            20..=23 => zk - 20,
            28..=31 => zk - 28 + 4,
            _ => return Err(EncodeError::InvalidOperand),
        };
        let mut word = f.val;
        word |= pdep(zada, f.za);
        word |= pdep(znp, f.zn);
        word |= pdep(zm, f.zm);
        word |= pdep(zkf, f.zk);
        word |= pdep(idx, f.idx);
        Ok(word)
    }

    /// The slice-select field value (`Ws - 8`) from a `W8..W11`/`W12..W15` reg.
    #[inline]
    fn ws_field(sel: Register) -> Result<u32, EncodeError> {
        if sel.class() != RegClass::Gp {
            return Err(EncodeError::InvalidOperand);
        }
        let n = sel.number() as u32;
        // SME2 multi-vector slice selects use W8..W11; the field is `Ws - 8`.
        if !(8..=11).contains(&n) {
            return Err(EncodeError::InvalidOperand);
        }
        Ok(n - 8)
    }

    /// The slice-offset field value (`off / span`), validating divisibility.
    #[inline]
    fn off_field(off: u8, span: u8) -> Result<u32, EncodeError> {
        let off = off as u32;
        let span = span as u32;
        if span == 0 || off % span != 0 {
            return Err(EncodeError::InvalidImmediate);
        }
        Ok(off / span)
    }

    /// A single `Z<n>.<arr>` source register number at operand `n`.
    #[inline]
    fn z_single(insn: &Instruction, n: usize, arr: VA) -> Result<u32, EncodeError> {
        match insn.op(n) {
            Operand::Reg { reg, arr: a, lane: None, .. }
                if reg.class() == RegClass::Sve && a == Some(arr) =>
            {
                Ok(reg.number() as u32)
            }
            _ => Err(EncodeError::InvalidOperand),
        }
    }

    /// An indexed `Z<n>.<arr>[idx]` source at operand `n` → `(reg, idx)`.
    #[inline]
    fn z_single_idx(insn: &Instruction, n: usize, arr: VA) -> Result<(u32, u32), EncodeError> {
        match insn.op(n) {
            Operand::Reg { reg, arr: a, lane: Some(l), .. }
                if reg.class() == RegClass::Sve && a == Some(arr) =>
            {
                Ok((reg.number() as u32, l as u32))
            }
            _ => Err(EncodeError::InvalidOperand),
        }
    }

    /// The group-base field value at operand `n`: `base / scale`, where
    /// `scale = 2^(5 - popcount(mask))` (inverse of the decoder's `group_base`).
    /// Validates the group count and that the base is a multiple of the stride.
    #[inline]
    fn group_field(
        insn: &Instruction,
        n: usize,
        vg: u8,
        arr: VA,
        mask: u32,
    ) -> Result<u32, EncodeError> {
        match insn.op(n) {
            Operand::SveVecGroup { first, count, arr: a, .. }
                if count == vg && a == Some(arr) && first.class() == RegClass::Sve =>
            {
                let base = first.number() as u32;
                let scale = 1u32 << (5 - mask.count_ones());
                if base % scale != 0 {
                    return Err(EncodeError::InvalidOperand);
                }
                Ok(base / scale)
            }
            _ => Err(EncodeError::InvalidOperand),
        }
    }

    // -----------------------------------------------------------------------
    // SME2 / SVE2.1 multi-vector ALU (SEL / CLAMP / ZIP / UZP), inverse of
    // `decode::sme::sme2::build_alu`.
    // -----------------------------------------------------------------------

    use crate::decode::sme::sme2::{alu_form_for_code, AluArr, AluForm, AluSh};

    /// Encode a multi-vector ALU form by scattering its operand fields back into
    /// the matched [`AluForm`]'s opcode template.
    fn enc_alu(insn: &Instruction, code: Code) -> R {
        let f = alu_form_for_code(code).ok_or(EncodeError::Unsupported)?;
        // Operand 0 is always the destination vector group; its arrangement is
        // the shared element size of the whole instruction.
        let arr = match insn.op(0) {
            Operand::SveVecGroup { arr: Some(a), .. } => a,
            _ => return Err(EncodeError::InvalidOperand),
        };
        let mut word = f.val;
        word |= enc_alu_arr(f, arr)?;
        word |= pdep(group_field(insn, 0, f.vg, arr, f.zd)?, f.zd);
        match f.shape {
            AluSh::SelGroup => {
                word |= pdep(pn_field(insn, 1, false)?, f.pn);
                word |= pdep(group_field(insn, 2, f.vg, arr, f.zn)?, f.zn);
                word |= pdep(group_field(insn, 3, f.vg, arr, f.zm)?, f.zm);
            }
            AluSh::TwoSingle => {
                word |= pdep(z_single(insn, 1, arr)?, f.zn);
                word |= pdep(z_single(insn, 2, arr)?, f.zm);
            }
            AluSh::ZipGroup => {
                word |= pdep(group_field(insn, 1, f.vg, arr, f.zn)?, f.zn);
            }
        }
        Ok(word)
    }

    /// The element-size opcode bits (`size<23:22>`, or the `.q` bit) for an ALU
    /// form's arrangement.
    fn enc_alu_arr(f: &AluForm, arr: VA) -> Result<u32, EncodeError> {
        let size = |a: VA| -> Result<u32, EncodeError> {
            match a {
                VA::Sb => Ok(0),
                VA::Sh => Ok(1),
                VA::Ss => Ok(2),
                VA::Sd => Ok(3),
                _ => Err(EncodeError::InvalidOperand),
            }
        };
        match f.arr {
            AluArr::Bhsd => Ok(size(arr)? << 22),
            AluArr::Fp => {
                let s = size(arr)?;
                if s == 0 {
                    return Err(EncodeError::InvalidOperand); // floating-point `.b` is invalid
                }
                Ok(s << 22)
            }
            AluArr::BfH => {
                // BFloat16 clamp is `.h` only; the size field stays `00`.
                if arr != VA::Sh {
                    return Err(EncodeError::InvalidOperand);
                }
                Ok(0)
            }
            AluArr::Zip2 => {
                if arr == VA::Sq {
                    Ok(1 << 10)
                } else {
                    Ok(size(arr)? << 22)
                }
            }
            AluArr::Zip4 => {
                if arr == VA::Sq {
                    Ok(1 << 16)
                } else {
                    Ok(size(arr)? << 22)
                }
            }
        }
    }

    /// The 3-bit `PNg` field (`PNg - 8`) from a predicate-as-counter operand,
    /// requiring the `/z` qualifier to match `expect_zeroing`.
    fn pn_field(insn: &Instruction, n: usize, expect_zeroing: bool) -> Result<u32, EncodeError> {
        match insn.op(n) {
            Operand::PredCounter { reg, zeroing }
                if reg.class() == RegClass::Predicate && zeroing == expect_zeroing =>
            {
                let v = reg.number() as u32;
                if !(8..=15).contains(&v) {
                    return Err(EncodeError::InvalidOperand);
                }
                Ok(v - 8)
            }
            _ => Err(EncodeError::InvalidOperand),
        }
    }

    // -----------------------------------------------------------------------
    // SME2 / SVE2.1 contiguous multi-vector load/store, inverse of
    // `decode::sme::sme2::decode_mem`.
    // -----------------------------------------------------------------------

    /// `msz<14:13>` → element arrangement.
    #[inline]
    fn msz_to_va(msz: u32) -> VA {
        match msz & 3 {
            0 => VA::Sb,
            1 => VA::Sh,
            2 => VA::Ss,
            _ => VA::Sd,
        }
    }

    /// Encode an SME2 contiguous multi-vector load/store. Operands:
    /// `[{ Zt.. }, PNg{/z}, mem]`, with `mem` a `[Xn, Xm{, LSL #msz}]` register
    /// index (scalar+scalar) or a `[Xn{, #imm, MUL VL}]` immediate (scalar+imm).
    fn enc_mem(insn: &Instruction) -> R {
        use Code::*;
        let (msz, is_store, is_nt) = match insn.code() {
            SmeLd1bMV => (0u32, 0u32, 0u32),
            SmeLd1hMV => (1, 0, 0),
            SmeLd1wMV => (2, 0, 0),
            SmeLd1dMV => (3, 0, 0),
            SmeLdnt1bMV => (0, 0, 1),
            SmeLdnt1hMV => (1, 0, 1),
            SmeLdnt1wMV => (2, 0, 1),
            SmeLdnt1dMV => (3, 0, 1),
            SmeSt1bMV => (0, 1, 0),
            SmeSt1hMV => (1, 1, 0),
            SmeSt1wMV => (2, 1, 0),
            SmeSt1dMV => (3, 1, 0),
            SmeStnt1bMV => (0, 1, 1),
            SmeStnt1hMV => (1, 1, 1),
            SmeStnt1wMV => (2, 1, 1),
            _ /* SmeStnt1dMV */ => (3, 1, 1),
        };
        let arr = msz_to_va(msz);

        // Operand 0: the data vector group. `count` selects vgx2/vgx4; `stride`
        // selects the consecutive (`1`) or strided (`8`/`4`) family.
        let (first, count, group_arr, stride) = match insn.op(0) {
            Operand::SveVecGroup { first, count, arr: Some(a), stride, .. } => (first, count, a, stride),
            _ => return Err(EncodeError::InvalidOperand),
        };
        if first.class() != RegClass::Sve || group_arr != arr {
            return Err(EncodeError::InvalidOperand);
        }
        let num_bit = match count {
            2 => 0u32,
            4 => 1u32 << 15,
            _ => return Err(EncodeError::InvalidOperand),
        };

        // Operand 1: the predicate-as-counter (loads `/z`, stores bare).
        let pn = pn_field(insn, 1, is_store == 0)?;

        let mut word = 0xA000_0000 | (is_store << 21) | num_bit | (msz << 13) | (pn << 10);

        // Data register group + nontemporal flag — packed differently per family.
        if stride == 1 {
            // Consecutive: base packed with stride 2 (vgx2) / 4 (vgx4); `NT` at
            // bit0.
            let zt_mask = if count == 4 { 0x1cu32 } else { 0x1eu32 };
            let zt = group_field(insn, 0, count, arr, zt_mask)?;
            word |= is_nt | pdep(zt, zt_mask);
        } else {
            // Strided: base = `word<4>:word<2:0>` (vgx2, step 8) or
            // `word<4>:word<1:0>` (vgx4, step 4); `NT` at bit3. The base lives in a
            // `{z0..7,z16..23}` / `{z0..3,z16..19}` window (bit3 always clear, so it
            // never collides with the `NT` bit).
            let base = first.number() as u32;
            let want_stride = if count == 4 { 4u8 } else { 8u8 };
            let allowed = if count == 4 { 0x13u32 } else { 0x17u32 };
            if stride != want_stride || base & !allowed != 0 {
                return Err(EncodeError::InvalidOperand);
            }
            word |= (1 << 24) | (is_nt << 3) | base;
        }

        // Operand 2: the addressing mode.
        match insn.op(2) {
            Operand::MemExt { base, index, extend, shift } => {
                if base.class() != RegClass::Gp || index.class() != RegClass::Gp {
                    return Err(EncodeError::InvalidOperand);
                }
                if extend != ExtendType::Uxtx {
                    return Err(EncodeError::InvalidOperand);
                }
                let expected = if msz == 0 { 0 } else { 0x80 | msz as u8 };
                if shift != expected {
                    return Err(EncodeError::InvalidOperand);
                }
                word |= (index.number() as u32) << 16; // bit22 == 0: scalar+scalar
                word |= (base.number() as u32) << 5;
            }
            Operand::SveMem { base, imm, mode: SveMemMode::ScalarImmMulVl, .. } => {
                if base.class() != RegClass::Gp {
                    return Err(EncodeError::InvalidOperand);
                }
                // The displayed offset is `imm4 * count`; recover the 4-bit signed
                // field and validate the range and divisibility.
                let c = count as i32;
                if imm % c != 0 {
                    return Err(EncodeError::InvalidImmediate);
                }
                let imm4 = imm / c;
                if !(-8..=7).contains(&imm4) {
                    return Err(EncodeError::InvalidImmediate);
                }
                word |= 1 << 22; // scalar+imm
                word |= ((imm4 as u32) & 0xf) << 16;
                word |= (base.number() as u32) << 5;
            }
            _ => return Err(EncodeError::InvalidOperand),
        }
        Ok(word)
    }

    // -----------------------------------------------------------------------
    // SME2 multi-vector saturating-rounding shift-right-narrow, inverse of
    // `decode::sme::sme2::decode_narrow_shift`.
    // -----------------------------------------------------------------------

    /// Encode an SME2 multi-vector shift-right-narrow. Operands: a single
    /// destination `Zd.<b|h>`, a 4-register source group `{ Zn.s/d - .. }`, and a
    /// `#shift` immediate. The destination element + shift range pick the
    /// `tsz`-style size field (`word<23:21>`).
    fn enc_narrow_shift(insn: &Instruction) -> R {
        use Code::*;
        let (uresult, uinput, n) = match insn.code() {
            SmeSqrshr => (0u32, 0u32, 0u32),
            SmeUqrshr => (0, 1, 0),
            SmeSqrshrn => (0, 0, 1),
            SmeUqrshrn => (0, 1, 1),
            SmeSqrshru => (1, 0, 0),
            _ /* SmeSqrshrun */ => (1, 0, 1),
        };
        // Operand 0: the single destination vector `Zd.<b|h>`.
        let (zd, dst) = match insn.op(0) {
            Operand::Reg { reg, arr: Some(a), lane: None, .. } if reg.class() == RegClass::Sve => {
                (reg.number() as u32, a)
            }
            _ => return Err(EncodeError::InvalidOperand),
        };
        // Operand 2: the `#shift` immediate.
        let shift = match insn.op(2) {
            Operand::ShiftAmount(s) => s as u32,
            _ => return Err(EncodeError::InvalidOperand),
        };
        // Destination element + shift range select the source element and the
        // `tsz` size field / 5-bit immediate.
        let (src, size3, imm5) = match dst {
            VA::Sb => {
                if !(1..=32).contains(&shift) {
                    return Err(EncodeError::InvalidImmediate);
                }
                (VA::Ss, 0b011u32, 32 - shift)
            }
            VA::Sh if (1..=32).contains(&shift) => (VA::Sd, 0b111u32, 32 - shift),
            VA::Sh if (33..=64).contains(&shift) => (VA::Sd, 0b101u32, 64 - shift),
            VA::Sh => return Err(EncodeError::InvalidImmediate),
            _ => return Err(EncodeError::InvalidOperand),
        };
        // Operand 1: the 4-register consecutive source group, base in `word<9:7>`.
        let zn = group_field(insn, 1, 4, src, 0x380)?;
        let word = 0xc100_d800
            | (size3 << 21)
            | (imm5 << 16)
            | (n << 10)
            | (zn << 7)
            | (uresult << 6)
            | (uinput << 5)
            | zd;
        Ok(word)
    }

    #[cfg(test)]
    mod tests {
        use crate::features::FeatureSet;
        use crate::instruction::Instruction;

        /// Decode `word` to an instruction.
        fn dec(word: u32) -> Instruction {
            let mut insn = Instruction::default();
            crate::decode::decode_into(word, 0x1000, FeatureSet::ALL, &mut insn);
            insn
        }

        /// Decode `word` then re-encode it; require the exact same word back.
        #[track_caller]
        fn rt(word: u32) {
            let insn = dec(word);
            assert!(!insn.is_invalid(), "word {word:#010x} failed to decode");
            let got = insn.encode().unwrap_or_else(|e| {
                panic!("encode of {word:#010x} ({:?}) failed: {e:?}", insn.code())
            });
            assert_eq!(
                got, word,
                "round-trip mismatch for {word:#010x}: re-encoded {got:#010x} (code={:?}, mnem={:?})",
                insn.code(),
                insn.mnemonic()
            );
        }

        /// Like [`rt`] but only requires a *semantic* round-trip: the re-encoded
        /// word must decode back to an equal instruction. Used for the `.Q` ZA
        /// load/store forms, whose `word<3:0>` carry no slice index and are
        /// discarded by the decoder — so the canonical re-encoding zeroes them.
        #[track_caller]
        fn rt_sem(word: u32) {
            let insn = dec(word);
            assert!(!insn.is_invalid(), "word {word:#010x} failed to decode");
            let got = insn.encode().unwrap_or_else(|e| {
                panic!("encode of {word:#010x} ({:?}) failed: {e:?}", insn.code())
            });
            let re = dec(got);
            assert!(
                !re.is_invalid()
                    && re.code() == insn.code()
                    && re.mnemonic() == insn.mnemonic()
                    && re.op_count() == insn.op_count()
                    && (0..re.op_count()).all(|i| re.op(i) == insn.op(i)),
                "semantic round-trip failed for {word:#010x}: re-encoded {got:#010x} \
                 decoded back to a different instruction (code={:?})",
                insn.code()
            );
        }

        #[test]
        fn sme_known_words() {
            // Outer products (FP / BF16).
            rt(0x809B4941); // fmopa  z1.s, ...
            rt(0x80D69446); // fmopa  z6.d, ...
            rt(0x81BB7F43); // fmopa  z3.s (FP16 -> FP32)
            rt(0x80851312); // fmops  z2.s
            rt(0x8184B942); // bfmopa z2.s
            rt(0x819EC990); // bfmops z0.s
            // Outer products (integer).
            rt(0xA0822DA1); // smopa  z1.s
            rt(0xA1A98383); // umopa  z3.s
            rt(0xA0A0AC03); // sumopa z3.s
            rt(0xA19670C1); // usmopa z1.s
            rt(0xA094D912); // smops  z2.s
            rt(0xA0CFCB26); // smopa  z6.d
            rt(0xA1E301C4); // umopa  z4.d
            // ADDHA / ADDVA.
            rt(0xC0909662); // addha  z2.s
            rt(0xC091BA23); // addva  z3.s
            rt(0xC0D053E5); // addha  z5.d
            rt(0xC0D16EE5); // addva  z5.d
            // MOVA tile -> vector.
            rt(0xC002D4B0); // mova z16.b, ..., z0v.b[w14,#5]
            rt(0xC0825448); // mova z8.s
            rt(0xC042BD71); // mova z17.h
            rt(0xC0C350F0); // mova z16.q (no index)
            // MOVA vector -> tile.
            rt(0xC0002F06);
            rt(0xC0803C6A);
            rt(0xC0C0F4E9);
            rt(0xC0C1D2E5); // .q vertical (no index)
            // ZA array load/store.
            rt(0xE011E5A3); // ld1b
            rt(0xE059A9C2); // ld1h
            rt(0xE0D76421); // ld1d
            rt_sem(0xE1DE4A4D); // ld1q (word<3:0> discarded; semantic round-trip)
            rt(0xE024806B); // st1b
            rt_sem(0xE1F50B6D); // st1q (word<3:0> discarded; semantic round-trip)
            rt_sem(0xE0B24FE7); // st1w sp base (bits<3:2> above index discarded)
            // LDR / STR ZA whole array.
            rt(0xE100004D); // ldr za[w12,#d]
            rt(0xE1200106); // str za[w12,#6]
            rt(0xE10062A0); // ldr za[w15,#0] (imm4==0)
            rt(0xE10043EB); // ldr sp base
        }

        #[test]
        fn sme2_multivector_alu_known_words() {
            // SEL (predicate-as-counter), vgx2 and vgx4.
            rt(0xC1208452); // sel { z18.b, z19.b }, pn9, { z2.b, z3.b }, { z0.b, z1.b }
            rt(0xC1258010); // sel { z16.b - z19.b }, pn8, { z0.b - z3.b }, { z4.b - z7.b }
            rt(0xC1608452); // sel .h
            // CLAMP (S/U/F/BF), vgx2 and vgx4.
            rt(0xC120C40F); // uclamp { z14.b, z15.b }, z0.b, z0.b
            rt(0xC1A0CC0D); // uclamp { z12.s - z15.s }, z0.s, z0.s
            rt(0xC120C40E); // sclamp { z14.b, z15.b }, z0.b, z0.b
            rt(0xC160C00E); // fclamp { z14.h, z15.h }, z0.h, z0.h
            rt(0xC1A0C80C); // fclamp { z12.s - z15.s }, z0.s, z0.s
            rt(0xC120C000); // bfclamp { z0.h, z1.h }, z0.h, z0.h
            rt(0xC120C800); // bfclamp { z0.h - z3.h }, z0.h, z0.h
            // ZIP/UZP, vgx2 (incl .q) and vgx4 (incl .q).
            rt(0xC120D000); // zip { z0.b, z1.b }, z0.b, z0.b
            rt(0xC120D400); // zip { z0.q, z1.q }, z0.q, z0.q
            rt(0xC120D001); // uzp { z0.b, z1.b }, z0.b, z0.b
            rt(0xC136E000); // zip { z0.b - z3.b }, { z0.b - z3.b }
            rt(0xC136E002); // uzp { z0.b - z3.b }, { z0.b - z3.b }
            rt(0xC137E000); // zip { z0.q - z3.q }, { z0.q - z3.q }
        }

        #[test]
        fn sme2_multivector_mem_known_words() {
            // LD1/LDNT1/ST1/STNT1, vgx2 and vgx4, scalar+scalar and scalar+imm.
            rt(0xA0004014); // ld1w { z20.s, z21.s }, pn8/z, [x0, x0, lsl #2]
            rt(0xA000E814); // ld1d { z20.d - z23.d }, pn10/z, [x0, x0, lsl #3]
            rt(0xA0000000); // ld1b { z0.b, z1.b }, pn8/z, [x0, x0]
            rt(0xA0002000); // ld1h { z0.h, z1.h }, pn8/z, [x0, x0, lsl #1]
            rt(0xA0404000); // ld1w { z0.s, z1.s }, pn8/z, [x0]  (imm == 0)
            rt(0xA0414000); // ld1w { z0.s, z1.s }, pn8/z, [x0, #2, mul vl]
            rt(0xA041E000); // ld1d { z0.d - z3.d }, pn8/z, [x0, #4, mul vl]
            rt(0xA0480000); // ld1b { z0.b, z1.b }, pn8/z, [x0, #-16, mul vl]
            rt(0xA0004015); // ldnt1w { z20.s, z21.s }, pn8/z, [x0, x0, lsl #2]
            rt(0xA0000001); // ldnt1b { z0.b, z1.b }, pn8/z, [x0, x0]
            rt(0xA0204014); // st1w { z20.s, z21.s }, pn8, [x0, x0, lsl #2]
            rt(0xA0200000); // st1b { z0.b, z1.b }, pn8, [x0, x0]
            rt(0xA0604000); // st1w { z0.s, z1.s }, pn8, [x0]
            rt(0xA0204015); // stnt1w { z20.s, z21.s }, pn8, [x0, x0, lsl #2]
            rt(0xA0200001); // stnt1b { z0.b, z1.b }, pn8, [x0, x0]
            // SP base resolves and round-trips.
            rt(0xA00043E0 | (31 << 5)); // ld1d ... [sp, ...] style base
        }

        #[test]
        fn sme2_multivector_strided_known_words() {
            // Strided (word<24>==1) register lists: 2-reg step 8, 4-reg step 4.
            rt(0xA1206710); // st1d { z16.d, z24.d }, pn9, [x24, x0, lsl #3]
            rt(0xA1204983); // st1w { z3.s, z11.s }, pn10, [x12, x0, lsl #2]
            rt(0xA1004DB1); // ld1w { z17.s, z25.s }, pn11/z, [x13, x0, lsl #2]
            rt(0xA120A541); // st1h { z1.h, z5.h, z9.h, z13.h }, pn9, [x10, x0, lsl #1]
            rt(0xA1206718); // stnt1d { z16.d, z24.d }, pn9, ... (NT = word<3>)
            rt(0xA1004000); // ld1w { z0.s, z8.s }, pn8/z, [x0, x0, lsl #2]
            rt(0xA1606710); // st1d { z16.d, z24.d }, pn9, [x24]  (imm == 0)
            rt(0xA1414000); // ld1w { z0.s, z8.s }, pn8/z, [x0, #2, mul vl]
        }

        #[test]
        fn sme2_shift_narrow_known_words() {
            // 4-vector -> 1-vector saturating rounding shift right narrow.
            rt(0xC161D920); // uqrshr z0.b, { z8.s - z11.s }, #31
            rt(0xC161D998); // sqrshr z24.b, { z12.s - z15.s }, #31
            rt(0xC160DC9A); // sqrshrn z26.b, { z4.s - z7.s }, #32
            rt(0xC161DE2B); // uqrshrn z11.b, { z16.s - z19.s }, #31
            rt(0xC164DB5D); // sqrshru z29.b, { z24.s - z27.s }, #28
            rt(0xC162DFD7); // sqrshrun z23.b, { z28.s - z31.s }, #30
            rt(0xC1A0D900); // sqrshr z0.h, { z8.d - z11.d }, #64
            rt(0xC1E0D900); // sqrshr z0.h, { z8.d - z11.d }, #32
            rt(0xC1FFDD3F); // uqrshrn z31.h, { z24.d - z27.d }, #1 (max fields)
        }

        /// Exhaustive structural round-trip sweep of the multi-vector ALU and
        /// load/store carve: every word the decoder accepts must re-encode to the
        /// exact same word.
        #[test]
        fn sme2_multivector_roundtrip_sweep() {
            let mut checked = 0u64;
            // Memory carve: quadrant 101, word<23> == 0, both the consecutive
            // (`word<24> == 0`) and strided (`word<24> == 1`) families. Iterate
            // every structural bit (strided/imm/store/reserved/num/msz/reserved/N)
            // and the predicate, striding the register/offset fields.
            for b24 in 0..2u32 {
            for b22 in 0..2u32 {
                for b21 in 0..2u32 {
                    for b20 in 0..2u32 {
                        for b15 in 0..2u32 {
                            for msz in 0..4u32 {
                                for b1 in 0..2u32 {
                                    for b0 in 0..2u32 {
                                        for b3 in 0..2u32 {
                                        for pn in [0u32, 3, 7] {
                                            for hi16 in [0u32, 1, 7, 17, 31] {
                                                for rn in [0u32, 5, 31] {
                                                    for zt in [0u32, 2, 12, 28] {
                                                        let word = 0xA000_0000
                                                            | (b24 << 24)
                                                            | (b22 << 22)
                                                            | (b21 << 21)
                                                            | (b20 << 20)
                                                            | (hi16 << 16)
                                                            | (b15 << 15)
                                                            | (msz << 13)
                                                            | (pn << 10)
                                                            | (rn << 5)
                                                            | zt
                                                            | (b3 << 3)
                                                            | (b1 << 1)
                                                            | b0;
                                                        let insn = dec(word);
                                                        if insn.is_invalid() {
                                                            continue;
                                                        }
                                                        let got =
                                                            insn.encode().unwrap_or_else(|e| {
                                                                panic!(
                                                                    "encode {word:#010x} ({:?}) failed: {e:?}",
                                                                    insn.code()
                                                                )
                                                            });
                                                        assert_eq!(
                                                            got, word,
                                                            "mem round-trip {word:#010x} ({:?})",
                                                            insn.code()
                                                        );
                                                        checked += 1;
                                                    }
                                                }
                                            }
                                        }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            }
            assert!(checked > 0, "swept no memory encodings");

            // ALU carve: drive every form's operand fields (Zd/Zn/Zm/PNg) and
            // every legal element size directly through the table masks, so the
            // round-trip is exercised across all forms and all field positions.
            use crate::decode::sme::sme2::{pdep, AluArr, SME2_ALU_FORMS};
            let mut alu_checked = 0u64;
            for f in SME2_ALU_FORMS {
                let sizes: &[u32] = match f.arr {
                    AluArr::Bhsd | AluArr::Zip2 | AluArr::Zip4 => &[0, 1, 2, 3],
                    AluArr::Fp => &[1, 2, 3],
                    AluArr::BfH => &[0],
                };
                let qbits: &[u32] = match f.arr {
                    AluArr::Zip2 => &[0, 1 << 10],
                    AluArr::Zip4 => &[0, 1 << 16],
                    _ => &[0],
                };
                for &sz in sizes {
                    for &q in qbits {
                        if q != 0 && sz != 0 {
                            continue; // `.q` is valid only with size == 00
                        }
                        for zd in [0u32, 1, 3, 7, 0xf] {
                            for zn in [0u32, 1, 2, 7] {
                                for zm in [0u32, 1, 3] {
                                    for pn in [0u32, 5, 7] {
                                        let mut w = f.val | (sz << 22) | q;
                                        w |= pdep(zd, f.zd);
                                        w |= pdep(zn, f.zn);
                                        if f.zm != 0 {
                                            w |= pdep(zm, f.zm);
                                        }
                                        if f.pn != 0 {
                                            w |= pdep(pn, f.pn);
                                        }
                                        let insn = dec(w);
                                        if insn.is_invalid() {
                                            continue;
                                        }
                                        assert_eq!(
                                            insn.code(),
                                            f.code,
                                            "decode {w:#010x} -> {:?}, expected {:?}",
                                            insn.code(),
                                            f.code
                                        );
                                        let got = insn.encode().unwrap_or_else(|e| {
                                            panic!("encode {w:#010x} ({:?}) failed: {e:?}", insn.code())
                                        });
                                        assert_eq!(
                                            got, w,
                                            "alu round-trip {w:#010x} ({:?})",
                                            insn.code()
                                        );
                                        alu_checked += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            assert!(alu_checked > 0, "swept no ALU encodings");
        }
    }
}
