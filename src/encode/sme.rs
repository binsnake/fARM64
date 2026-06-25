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
    use crate::enums::VectorArrangement as VA;
    use crate::instruction::Instruction;
    use crate::mnemonic::Code;
    use crate::operand::{Operand, SliceIndicator};
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
        ) {
            return true;
        }
        // SME2 multi-vector + *TMOPA forms are table-driven (one Code per row).
        crate::decode::sme::sme2::form_for_code(code).is_some()
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
            other => enc_sme2(insn, other),
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
    }
}
