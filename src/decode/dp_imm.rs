//! Data Processing -- Immediate (ARM ARM C4.1.93).
//!
//! Hand-written decoder for the immediate data-processing group, dispatched here
//! from [`crate::decode::decode_into`] when `op0 = word<28:25>` selects `100x`.
//! Sub-class on `op1 = word<25:23>`:
//!
//! | `op1` (`word<25:23>`) | class |
//! |-|-|
//! | `00x` | PC-relative addressing (`ADR`, `ADRP`) |
//! | `010` | Add/subtract (immediate) |
//! | `011` | Add/subtract (immediate, with tags) |
//! | `100` | Logical (immediate) — via [`crate::decode::bits::decode_bit_masks`] |
//! | `101` | Move wide (immediate) |
//! | `110` | Bitfield |
//! | `111` | Extract |
//!
//! Preferred-disassembly aliases (ARM ARM alias conditions) are emitted on the
//! `mnemonic` while `code` stays the canonical encoding identity, matching the
//! Binary Ninja differential corpus: `MOV` (ADD-to/from-SP, ORR-bitmask,
//! MOVZ/MOVN), `CMP`/`CMN`/`TST`, the bitfield family
//! (`LSL`/`LSR`/`ASR`/`UBFX`/`SBFX`/`UBFIZ`/`SBFIZ`/`BFI`/`BFXIL`/`BFC` and the
//! `SXT*`/`UXT*` sign/zero-extends), and `ROR` from `EXTR`.

use crate::decode::bits::{bit, bits, decode_bit_masks, move_wide_preferred, sign_extend};
use crate::features::{Feature, FeatureSet};
use crate::instruction::Instruction;
use crate::mnemonic::{Code, Mnemonic};
use crate::operand::Operand;
use crate::register::{gp_register, RegWidth};

/// Build a plain (undecorated) GP register operand.
#[inline]
fn reg(use_sp: bool, width: RegWidth, n: u32) -> Operand {
    Operand::Reg {
        reg: gp_register(use_sp, width, (n & 0x1f) as u8),
        arr: None,
        lane: None,
        shift: None,
        extend: None,
        pred: None,
    }
}

/// The GP register width for an `sf` bit (`0` => `W`/32-bit, `1` => `X`/64-bit).
#[inline]
fn width_of(sf: u32) -> RegWidth {
    if sf & 1 == 1 {
        RegWidth::X64
    } else {
        RegWidth::W32
    }
}

/// Decode a Data Processing -- Immediate instruction into `out`.
///
/// `word` is the raw little-endian instruction; `ip` is its address (needed for
/// the PC-relative `ADR`/`ADRP` forms). On an unallocated encoding, leaves `out`
/// as the invalid default. Pure, total and panic-free for every input.
#[inline]
pub fn decode(word: u32, ip: u64, features: FeatureSet, out: &mut Instruction) {
    // FEAT_PAuth_LR AUTIASPPC/AUTIBSPPC (PC-relative authenticate of LR) overlap
    // the Extract (`op1 == 111`) sub-class space but use a distinct fixed pattern
    // `word<31:22> == 1111001110` with `word<4:0> == 11111` (the `word<21>` bit
    // is the A/B key, `word<20:5>` the imm16). They render a PC-relative label,
    // so they belong with the branch forms — delegate to `branch_sys`. Gated on
    // FEAT_PAuth_LR; an unrecognized neighbour falls through to the normal
    // dp-immediate dispatch (and is rejected by `decode_extract`).
    if (word & 0xFFC0_001F) == 0xF380_001F {
        if features.has(Feature::PauthLr) {
            crate::decode::branch_sys::decode_auti_sppc(word, out);
        }
        return;
    }

    // op1 = word<25:23> selects the seven sub-classes.
    let op1 = bits(word, 23, 3);
    match op1 {
        0b000 | 0b001 => decode_pc_rel(word, ip, out),
        0b010 => decode_addsub_imm(word, out),
        0b011 => {
            // Within op1==011, word<22> (`o2`) splits two classes: `o2==0` is
            // Add/subtract (immediate, with tags) — ADDG/SUBG (FEAT_MTE);
            // `o2==1` is Min/max (immediate) — SMAX/SMIN/UMAX/UMIN (FEAT_CSSC).
            if bit(word, 22) == 1 {
                decode_minmax_imm(word, features, out);
            } else {
                decode_addsub_imm_tags(word, features, out);
            }
        }
        0b100 => decode_logical_imm(word, out),
        0b101 => decode_move_wide(word, out),
        0b110 => decode_bitfield(word, out),
        0b111 => decode_extract(word, out),
        // op1 is 3 bits and fully covered above; stay total without a panic.
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// 00x : PC-relative addressing (ADR / ADRP).
// ---------------------------------------------------------------------------

/// `ADR`/`ADRP` (ARM ARM C4.1.93 PC-rel.). `op` (bit 31) selects ADRP; the
/// 21-bit immediate is `immhi:immlo`, sign-extended. ADRP shifts left by 12 and
/// bases off the 4 KiB page of `ip`.
#[inline]
fn decode_pc_rel(word: u32, ip: u64, out: &mut Instruction) {
    let op = bit(word, 31);
    let rd = bits(word, 0, 5);
    let immlo = bits(word, 29, 2);
    let immhi = bits(word, 5, 19);
    let imm21 = (immhi << 2) | immlo; // immhi:immlo

    let target = if op == 1 {
        // ADRP: imm = SignExtend(immhi:immlo:Zeros(12), 64); base = page of ip.
        let imm = sign_extend((imm21 as u64) << 12, 33);
        let base = ip & !0xFFF;
        base.wrapping_add(imm as u64)
    } else {
        // ADR: imm = SignExtend(immhi:immlo, 21); base = ip.
        let imm = sign_extend(imm21 as u64, 21);
        ip.wrapping_add(imm as u64)
    };

    out.set(if op == 1 { Code::Adrp } else { Code::Adr });
    out.push_operand(reg(false, RegWidth::X64, rd));
    out.push_operand(Operand::Label(target));
}

// ---------------------------------------------------------------------------
// 010 : Add/subtract (immediate).
// ---------------------------------------------------------------------------

/// `ADD`/`ADDS`/`SUB`/`SUBS` (immediate). `sh` (bit 22) shifts the 12-bit
/// immediate left by 12; `sh==1x` reserved beyond the single bit is impossible
/// (it is a 1-bit field). Emits the `MOV`/`CMP`/`CMN` preferred aliases.
#[inline]
fn decode_addsub_imm(word: u32, out: &mut Instruction) {
    let sf = bit(word, 31);
    let op = bit(word, 30); // 0 = ADD, 1 = SUB
    let s = bit(word, 29); // flag-setting
    let sh = bit(word, 22); // shift the imm12 left by 12
    let imm12 = bits(word, 10, 12);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);
    let w = width_of(sf);

    // Binary Ninja renders the add/sub-immediate shift as a trailing
    // `, lsl #0xc` while showing the *unshifted* imm12: this is exactly the
    // `ImmShiftedMove{imm, lsl}` rendering (imm12 always fits in 16 bits, and a
    // zero shift is elided). Reuse it so the LSL decoration is correct.
    let imm_op = Operand::ImmShiftedMove {
        imm: imm12 as u16,
        lsl: if sh == 1 { 12 } else { 0 },
    };

    let code = match (sf, op, s) {
        (0, 0, 0) => Code::AddImm32,
        (1, 0, 0) => Code::AddImm64,
        (0, 0, 1) => Code::AddsImm32,
        (1, 0, 1) => Code::AddsImm64,
        (0, 1, 0) => Code::SubImm32,
        (1, 1, 0) => Code::SubImm64,
        (0, 1, 1) => Code::SubsImm32,
        _ => Code::SubsImm64,
    };
    out.set(code);

    // Preferred aliases:
    //   ADD <Rd|SP>, <Rn|SP>, #0  with (Rd==SP || Rn==SP) -> MOV <Rd|SP>, <Rn|SP>
    //   ADDS Rd==ZR -> CMN ; SUBS Rd==ZR -> CMP
    let flag_setting = s == 1;
    let rd_is_31 = rd == 31;
    let rn_is_31 = rn == 31;

    if !flag_setting
        && op == 0
        && sh == 0
        && imm12 == 0
        && (rd_is_31 || rn_is_31)
    {
        // MOV (to/from SP): both registers are SP-capable.
        out.set_mnemonic(Mnemonic::Mov);
        out.push_operand(reg(true, w, rd));
        out.push_operand(reg(true, w, rn));
        return;
    }

    if flag_setting && rd_is_31 {
        // CMP (SUBS) / CMN (ADDS): drop Rd, Rn is SP-capable.
        out.set_mnemonic(if op == 1 { Mnemonic::Cmp } else { Mnemonic::Cmn });
        out.push_operand(reg(true, w, rn));
        out.push_operand(imm_op);
        return;
    }

    // Canonical ADD/ADDS/SUB/SUBS.
    //   ADD/SUB:  Rd, Rn both SP-capable.
    //   ADDS/SUBS: Rn SP-capable, Rd is ZR.
    out.push_operand(reg(!flag_setting, w, rd));
    out.push_operand(reg(true, w, rn));
    out.push_operand(imm_op);
}

// ---------------------------------------------------------------------------
// 011 : Add/subtract (immediate, with tags) — FEAT_MTE.
// ---------------------------------------------------------------------------

/// `ADDG`/`SUBG` (immediate, with tags). Feature-gated on `FEAT_MTE`; left
/// Invalid when MTE is not accepted. `uimm6` is scaled by 16 (the tag-granule
/// log size), `uimm4` is the tag offset.
#[inline]
fn decode_addsub_imm_tags(word: u32, features: FeatureSet, out: &mut Instruction) {
    if !features.has(Feature::Mte) {
        return; // leave Invalid
    }
    // Fixed pattern bits: sf=1, S=0, o2=0 (word<22>==0). Reject otherwise.
    let sf = bit(word, 31);
    let s = bit(word, 29);
    let o2 = bit(word, 22);
    if sf != 1 || s != 0 || o2 != 0 {
        return;
    }
    let op = bit(word, 30); // 0 = ADDG, 1 = SUBG
    let uimm6 = bits(word, 16, 6);
    let uimm4 = bits(word, 10, 4);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);

    out.set(if op == 1 { Code::SubgImm } else { Code::AddgImm });
    // Rd, Rn are both SP-capable (64-bit).
    out.push_operand(reg(true, RegWidth::X64, rd));
    out.push_operand(reg(true, RegWidth::X64, rn));
    out.push_operand(Operand::ImmUnsigned((uimm6 as u64) << 4));
    out.push_operand(Operand::ImmUnsigned(uimm4 as u64));
}

/// `SMAX`/`SMIN`/`UMAX`/`UMIN` (immediate, FEAT_CSSC). Reached for `op1==011`
/// with `word<22>==1`. Fixed bits: `op (word<30>)==0`, `S (word<29>)==0`,
/// `word<21:20>==00`. `opc (word<19:18>)` selects the operation: `word<19>`
/// chooses min (1) vs max (0), `word<18>` chooses unsigned (1) vs signed (0).
/// The 8-bit immediate (`word<17:10>`) is signed for SMAX/SMIN and unsigned for
/// UMAX/UMIN. `Rd`/`Rn` are plain (ZR-form) GP registers of the operation width.
#[inline]
fn decode_minmax_imm(word: u32, features: FeatureSet, out: &mut Instruction) {
    if !features.has(Feature::Cssc) {
        return; // leave Invalid
    }
    // op (word<30>) and S (word<29>) must be 0; word<21:20> must be 0.
    if bit(word, 30) != 0 || bit(word, 29) != 0 || bits(word, 20, 2) != 0 {
        return;
    }
    let sf = bit(word, 31);
    let opc = bits(word, 18, 2);
    let imm8 = bits(word, 10, 8);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);
    let w = width_of(sf);

    let code = match (sf, opc) {
        (0, 0b00) => Code::SmaxImm32,
        (1, 0b00) => Code::SmaxImm64,
        (0, 0b01) => Code::UmaxImm32,
        (1, 0b01) => Code::UmaxImm64,
        (0, 0b10) => Code::SminImm32,
        (1, 0b10) => Code::SminImm64,
        (0, 0b11) => Code::UminImm32,
        _ => Code::UminImm64,
    };
    out.set(code);
    out.push_operand(reg(false, w, rd));
    out.push_operand(reg(false, w, rn));
    // SMAX/SMIN take a signed imm8; UMAX/UMIN take an unsigned imm8.
    let is_signed = bit(opc, 0) == 0;
    if is_signed {
        out.push_operand(Operand::ImmSigned(sign_extend(imm8 as u64, 8)));
    } else {
        out.push_operand(Operand::ImmUnsigned(imm8 as u64));
    }
}

// ---------------------------------------------------------------------------
// 100 : Logical (immediate).
// ---------------------------------------------------------------------------

/// `AND`/`ORR`/`EOR`/`ANDS` (logical immediate). The bitmask immediate is
/// decoded via [`decode_bit_masks`]; an invalid mask (or `sf==0 && N==1`) is
/// UNALLOCATED. Emits `MOV` (from ORR when not move-wide-preferred) and `TST`
/// (ANDS Rd==ZR) aliases.
#[inline]
fn decode_logical_imm(word: u32, out: &mut Instruction) {
    let sf = bit(word, 31);
    let opc = bits(word, 29, 2);
    let n = bit(word, 22);
    let immr = bits(word, 16, 6);
    let imms = bits(word, 10, 6);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);

    // sf==0 with N==1 is UNALLOCATED.
    if sf == 0 && n == 1 {
        return;
    }
    let datasize = if sf == 1 { 64 } else { 32 };
    let masks = match decode_bit_masks(n, imms, immr, true, datasize) {
        Some(m) => m,
        None => return, // reserved encoding
    };
    let imm = if datasize == 32 {
        (masks.wmask as u32) as u64
    } else {
        masks.wmask
    };
    let w = width_of(sf);

    let code = match (sf, opc) {
        (0, 0b00) => Code::AndImm32,
        (1, 0b00) => Code::AndImm64,
        (0, 0b01) => Code::OrrImm32,
        (1, 0b01) => Code::OrrImm64,
        (0, 0b10) => Code::EorImm32,
        (1, 0b10) => Code::EorImm64,
        (0, _) => Code::AndsImm32,
        (_, _) => Code::AndsImm64,
    };
    out.set(code);

    // ANDS Rd==ZR -> TST Rn, #imm.
    if opc == 0b11 && rd == 31 {
        out.set_mnemonic(Mnemonic::Tst);
        out.push_operand(reg(false, w, rn));
        out.push_operand(Operand::ImmLogical(imm));
        return;
    }

    // ORR Rn==ZR with !MoveWidePreferred -> MOV Rd, #imm (signed display).
    if opc == 0b01 && rn == 31 && !move_wide_preferred(sf, n, imms, immr) {
        out.set_mnemonic(Mnemonic::Mov);
        // MOV (bitmask) is SP-capable on Rd per ARM ARM, but binja prints the
        // immediate as a signed value.
        out.push_operand(reg(true, w, rd));
        out.push_operand(Operand::ImmSigned(signed_imm(imm, datasize)));
        return;
    }

    // Canonical AND/ORR/EOR (Rd SP-capable) ; ANDS (Rd ZR). Rn is always ZR.
    let rd_use_sp = opc != 0b11;
    out.push_operand(reg(rd_use_sp, w, rd));
    out.push_operand(reg(false, w, rn));
    out.push_operand(Operand::ImmLogical(imm));
}

/// Sign-interpret a `datasize`-bit logical immediate for the `MOV (bitmask)`
/// alias, matching Binary Ninja's signed rendering of those immediates.
#[inline]
fn signed_imm(imm: u64, datasize: u32) -> i64 {
    sign_extend(imm, datasize)
}

// ---------------------------------------------------------------------------
// 101 : Move wide (immediate).
// ---------------------------------------------------------------------------

/// `MOVN`/`MOVZ`/`MOVK` (move wide). `hw` selects the 16-bit lane shift
/// (`hw*16`); for 32-bit forms `hw<1>` must be `0` (else UNALLOCATED). Emits the
/// `MOV` alias for MOVZ/MOVN when `MoveWidePreferred`.
#[inline]
fn decode_move_wide(word: u32, out: &mut Instruction) {
    let sf = bit(word, 31);
    let opc = bits(word, 29, 2);
    let hw = bits(word, 21, 2);
    let imm16 = bits(word, 5, 16);
    let rd = bits(word, 0, 5);

    // opc==01 is UNALLOCATED for move-wide.
    if opc == 0b01 {
        return;
    }
    // 32-bit forms must have hw<1> == 0.
    if sf == 0 && (hw & 0b10) != 0 {
        return;
    }
    let w = width_of(sf);
    let shift = (hw * 16) as u8;
    let datasize = if sf == 1 { 64 } else { 32 };

    let code = match (sf, opc) {
        (0, 0b00) => Code::Movn32,
        (1, 0b00) => Code::Movn64,
        (0, 0b10) => Code::Movz32,
        (1, 0b10) => Code::Movz64,
        (0, _) => Code::Movk32,
        _ => Code::Movk64,
    };
    out.set(code);

    let is_movn = opc == 0b00;
    let is_movz = opc == 0b10;

    // MOV alias for MOVN/MOVZ when MoveWidePreferred is *not* satisfied? No:
    // the ARM ARM uses MOVZ/MOVN -> MOV when the wide form is the preferred
    // representation of the constant, gated by !MoveWidePreferred for ORR and by
    // the dedicated MOVN/MOVZ alias conditions here. Binja emits MOV when:
    //   MOVZ -> MOV : !(IsZero(imm16) && hw != 0)
    //   MOVN -> MOV : !(IsZero(imm16) && hw != 0) && !(sf==0 && imm16==0xffff)
    if is_movz {
        let imm_is_zero_shifted = imm16 == 0 && hw != 0;
        if !imm_is_zero_shifted {
            let val = (imm16 as u64) << shift;
            out.set_mnemonic(Mnemonic::Mov);
            out.push_operand(reg(false, w, rd));
            out.push_operand(Operand::ImmSigned(signed_imm(val, datasize)));
            return;
        }
    } else if is_movn {
        let imm_is_zero_shifted = imm16 == 0 && hw != 0;
        let is_32_all_ones = sf == 0 && imm16 == 0xffff;
        if !imm_is_zero_shifted && !is_32_all_ones {
            // value = NOT(imm16 << shift), truncated to datasize.
            let raw = !((imm16 as u64) << shift);
            let val = if datasize == 32 { raw & 0xffff_ffff } else { raw };
            out.set_mnemonic(Mnemonic::Mov);
            out.push_operand(reg(false, w, rd));
            out.push_operand(Operand::ImmSigned(signed_imm(val, datasize)));
            return;
        }
    }

    // Canonical MOVN/MOVZ/MOVK: Rd is ZR, immediate is the raw imm16 shifted.
    out.push_operand(reg(false, w, rd));
    out.push_operand(Operand::ImmShiftedMove {
        imm: imm16 as u16,
        lsl: shift,
    });
}

// ---------------------------------------------------------------------------
// 110 : Bitfield (SBFM / BFM / UBFM) and aliases.
// ---------------------------------------------------------------------------

/// `SBFM`/`BFM`/`UBFM` (bitfield) plus the rich alias family
/// (`LSL`/`LSR`/`ASR`/`UBFX`/`SBFX`/`UBFIZ`/`SBFIZ`/`BFI`/`BFXIL`/`BFC`/`SXT*`/
/// `UXT*`). `N` must equal `sf` (else UNALLOCATED); 32-bit forms additionally
/// require `immr<5>==0 && imms<5>==0`.
#[inline]
fn decode_bitfield(word: u32, out: &mut Instruction) {
    let sf = bit(word, 31);
    let opc = bits(word, 29, 2);
    let n = bit(word, 22);
    let immr = bits(word, 16, 6);
    let imms = bits(word, 10, 6);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);

    // opc==11 is UNALLOCATED for bitfield.
    if opc == 0b11 {
        return;
    }
    // N must match sf.
    if n != sf {
        return;
    }
    // 32-bit forms: immr<5> and imms<5> must be zero.
    if sf == 0 && (bit(immr, 5) == 1 || bit(imms, 5) == 1) {
        return;
    }

    let w = width_of(sf);
    let datasize = if sf == 1 { 64u32 } else { 32u32 };
    let r = immr;
    let s = imms;

    let code = match (sf, opc) {
        (0, 0b00) => Code::Sbfm32,
        (1, 0b00) => Code::Sbfm64,
        (0, 0b01) => Code::Bfm32,
        (1, 0b01) => Code::Bfm64,
        (0, _) => Code::Ubfm32,
        _ => Code::Ubfm64,
    };
    out.set(code);

    match opc {
        0b00 => emit_sbfm_alias(out, w, datasize, r, s, rn, rd),
        0b01 => emit_bfm_alias(out, w, datasize, r, s, rn, rd),
        _ => emit_ubfm_alias(out, w, datasize, r, s, rn, rd),
    }
}

/// SBFM aliases: ASR / SBFIZ / SBFX / SXTB / SXTH / SXTW, else canonical SBFM.
#[inline]
fn emit_sbfm_alias(
    out: &mut Instruction,
    w: RegWidth,
    datasize: u32,
    immr: u32,
    imms: u32,
    rn: u32,
    rd: u32,
) {
    let max = datasize - 1;

    // SXTB/SXTH/SXTW: immr==0 and imms in {7,15,31}. (32-bit excludes SXTW.)
    if immr == 0 {
        let ext = match imms {
            7 => Some(Mnemonic::Sxtb),
            15 => Some(Mnemonic::Sxth),
            31 if datasize == 64 => Some(Mnemonic::Sxtw),
            _ => None,
        };
        if let Some(m) = ext {
            out.set_mnemonic(m);
            out.push_operand(reg(false, w, rd));
            // Source is always the 32-bit (W) view for the sign-extends.
            out.push_operand(reg(false, RegWidth::W32, rn));
            return;
        }
    }

    // ASR: imms == datasize-1.
    if imms == max {
        out.set_mnemonic(Mnemonic::Asr);
        out.push_operand(reg(false, w, rd));
        out.push_operand(reg(false, w, rn));
        out.push_operand(Operand::ImmUnsigned(immr as u64));
        return;
    }

    // SBFIZ: imms < immr -> SBFIZ Rd, Rn, #(-immr MOD datasize), #(imms+1).
    if imms < immr {
        out.set_mnemonic(Mnemonic::Sbfiz);
        out.push_operand(reg(false, w, rd));
        out.push_operand(reg(false, w, rn));
        out.push_operand(Operand::ImmUnsigned(((datasize - immr) % datasize) as u64));
        out.push_operand(Operand::ImmUnsigned((imms + 1) as u64));
        return;
    }

    // SBFX: imms >= immr -> SBFX Rd, Rn, #immr, #(imms-immr+1).
    out.set_mnemonic(Mnemonic::Sbfx);
    out.push_operand(reg(false, w, rd));
    out.push_operand(reg(false, w, rn));
    out.push_operand(Operand::ImmUnsigned(immr as u64));
    out.push_operand(Operand::ImmUnsigned((imms - immr + 1) as u64));
}

/// UBFM aliases: LSL / LSR / UBFIZ / UBFX / UXTB / UXTH, else canonical UBFM.
#[inline]
fn emit_ubfm_alias(
    out: &mut Instruction,
    w: RegWidth,
    datasize: u32,
    immr: u32,
    imms: u32,
    rn: u32,
    rd: u32,
) {
    let max = datasize - 1;

    // UXTB/UXTH: immr==0 and imms in {7,15} (both only on 32-bit per binja/UAL).
    if immr == 0 && datasize == 32 {
        let ext = match imms {
            7 => Some(Mnemonic::Uxtb),
            15 => Some(Mnemonic::Uxth),
            _ => None,
        };
        if let Some(m) = ext {
            out.set_mnemonic(m);
            out.push_operand(reg(false, w, rd));
            out.push_operand(reg(false, RegWidth::W32, rn));
            return;
        }
    }

    // LSL: imms + 1 == immr  (i.e. imms != max && imms+1 == immr).
    if imms != max && imms + 1 == immr {
        out.set_mnemonic(Mnemonic::Lsl);
        out.push_operand(reg(false, w, rd));
        out.push_operand(reg(false, w, rn));
        out.push_operand(Operand::ImmUnsigned((max - imms) as u64));
        return;
    }

    // LSR: imms == datasize-1.
    if imms == max {
        out.set_mnemonic(Mnemonic::Lsr);
        out.push_operand(reg(false, w, rd));
        out.push_operand(reg(false, w, rn));
        out.push_operand(Operand::ImmUnsigned(immr as u64));
        return;
    }

    // UBFIZ: imms < immr -> UBFIZ Rd, Rn, #(-immr MOD datasize), #(imms+1).
    if imms < immr {
        out.set_mnemonic(Mnemonic::Ubfiz);
        out.push_operand(reg(false, w, rd));
        out.push_operand(reg(false, w, rn));
        out.push_operand(Operand::ImmUnsigned(((datasize - immr) % datasize) as u64));
        out.push_operand(Operand::ImmUnsigned((imms + 1) as u64));
        return;
    }

    // UBFX: imms >= immr -> UBFX Rd, Rn, #immr, #(imms-immr+1).
    out.set_mnemonic(Mnemonic::Ubfx);
    out.push_operand(reg(false, w, rd));
    out.push_operand(reg(false, w, rn));
    out.push_operand(Operand::ImmUnsigned(immr as u64));
    out.push_operand(Operand::ImmUnsigned((imms - immr + 1) as u64));
}

/// BFM aliases: BFC (Rn==ZR, imms<immr) / BFI (imms<immr) / BFXIL (imms>=immr).
#[inline]
fn emit_bfm_alias(
    out: &mut Instruction,
    w: RegWidth,
    datasize: u32,
    immr: u32,
    imms: u32,
    rn: u32,
    rd: u32,
) {
    if imms < immr {
        // BFI / BFC : lsb = -immr MOD datasize, width = imms+1.
        let lsb = (datasize - immr) % datasize;
        let width = imms + 1;
        if rn == 31 {
            // BFC Rd, #lsb, #width.
            out.set_mnemonic(Mnemonic::Bfc);
            out.push_operand(reg(false, w, rd));
            out.push_operand(Operand::ImmUnsigned(lsb as u64));
            out.push_operand(Operand::ImmUnsigned(width as u64));
        } else {
            // BFI Rd, Rn, #lsb, #width.
            out.set_mnemonic(Mnemonic::Bfi);
            out.push_operand(reg(false, w, rd));
            out.push_operand(reg(false, w, rn));
            out.push_operand(Operand::ImmUnsigned(lsb as u64));
            out.push_operand(Operand::ImmUnsigned(width as u64));
        }
        return;
    }

    // BFXIL Rd, Rn, #immr, #(imms-immr+1).
    out.set_mnemonic(Mnemonic::Bfxil);
    out.push_operand(reg(false, w, rd));
    out.push_operand(reg(false, w, rn));
    out.push_operand(Operand::ImmUnsigned(immr as u64));
    out.push_operand(Operand::ImmUnsigned((imms - immr + 1) as u64));
}

// ---------------------------------------------------------------------------
// 111 : Extract (EXTR) and the ROR alias.
// ---------------------------------------------------------------------------

/// `EXTR` (extract). `N` must equal `sf`, `imms<5>` must be `0` for 32-bit, and
/// the `o0`/`op21` fixed bits must be `0` (else UNALLOCATED). When `Rn==Rm` the
/// preferred alias is `ROR Rd, Rn, #lsb`.
#[inline]
fn decode_extract(word: u32, out: &mut Instruction) {
    let sf = bit(word, 31);
    let op21 = bits(word, 29, 2); // must be 00
    let n = bit(word, 22);
    let o0 = bit(word, 21); // must be 0
    let rm = bits(word, 16, 5);
    let imms = bits(word, 10, 6);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);

    if op21 != 0 || o0 != 0 {
        return;
    }
    // N must equal sf.
    if n != sf {
        return;
    }
    // 32-bit: imms<5> must be 0.
    if sf == 0 && bit(imms, 5) == 1 {
        return;
    }

    let w = width_of(sf);
    out.set(if sf == 1 { Code::Extr64 } else { Code::Extr32 });

    // ROR alias when Rn == Rm.
    if rn == rm {
        out.set_mnemonic(Mnemonic::Ror);
        out.push_operand(reg(false, w, rd));
        out.push_operand(reg(false, w, rn));
        out.push_operand(Operand::ImmUnsigned(imms as u64));
        return;
    }

    out.push_operand(reg(false, w, rd));
    out.push_operand(reg(false, w, rn));
    out.push_operand(reg(false, w, rm));
    out.push_operand(Operand::ImmUnsigned(imms as u64));
}
