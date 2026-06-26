//! Branches, Exception generating and System instructions (ARM ARM C4.1.3).
//!
//! Hand-written decoder dispatched here from [`crate::decode::decode_into`] when
//! `op0 = word<28:25>` selects `101x`. Covers:
//!
//! * Unconditional branch (immediate): `B`, `BL`.
//! * Compare-and-branch / test-and-branch: `CBZ`, `CBNZ`, `TBZ`, `TBNZ`.
//! * Conditional branch (immediate): `B.cond` (and `BC.cond`, FEAT_HBC).
//! * Unconditional branch (register): `BR`, `BLR`, `RET`, `ERET`, `DRPS` plus the
//!   pointer-authenticated forms (`BRAA`/`BRAAZ`/`BLRAA`/`RETAA`/...).
//! * Exception generation: `SVC`, `HVC`, `SMC`, `BRK`, `HLT`, `DCPS{1,2,3}`,
//!   `TCANCEL`.
//! * System: the `HINT` space (`NOP`/`YIELD`/`WFE`/`WFI`/`SEV`/`SEVL`/`ESB`/
//!   `PSB CSYNC`/`TSB CSYNC`/`BTI`/PAuth hints/...), `WFET`/`WFIT`, barriers
//!   (`CLREX`/`DSB`/`DMB`/`ISB`/`SB`/`SSBB`/`PSSBB`/`TCOMMIT`), `MSR (immediate)`
//!   PSTATE, `MSR`/`MRS` (register), and `SYS`/`SYSL`/`SYSP` with the
//!   `IC`/`DC`/`AT`/`TLBI`/`CFP`/`CPP`/`DVP` aliases.
//!
//! Rendering targets the Binary Ninja differential corpus: `Code` stays the
//! canonical encoding while `mnemonic` carries the preferred alias, exactly like
//! [`crate::decode::dp_imm`].

use crate::decode::bits::{bit, bits, sign_extend};
use crate::enums::Condition;
use crate::features::{Feature, FeatureSet};
use crate::instruction::Instruction;
use crate::mnemonic::{Code, Mnemonic};
use crate::operand::Operand;
use crate::register::{gp_register, RegWidth};
use crate::sysop::SysToken;
use crate::sysreg::SystemReg;

/// Build a [`Operand::SysOp`] keyword operand from its name.
#[inline]
fn sysop(name: &str) -> Operand {
    Operand::SysOp(SysToken::of(name))
}

/// Build a plain 64-bit X register operand (SP/ZR resolved via `use_sp`).
#[inline]
fn xreg(use_sp: bool, n: u32) -> Operand {
    Operand::Reg {
        reg: gp_register(use_sp, RegWidth::X64, (n & 0x1f) as u8),
        arr: None,
        lane: None,
        shift: None,
        extend: None,
        pred: None,
    }
}

/// Build a GP register operand of the given width (SP/ZR resolved via `use_sp`).
#[inline]
fn reg(use_sp: bool, w: RegWidth, n: u32) -> Operand {
    Operand::Reg {
        reg: gp_register(use_sp, w, (n & 0x1f) as u8),
        arr: None,
        lane: None,
        shift: None,
        extend: None,
        pred: None,
    }
}

/// Decode a Branch / Exception / System instruction into `out`.
///
/// `ip` resolves the PC-relative branch targets. Unallocated encodings leave
/// `out` at the invalid default; the decode is total and panic-free.
#[inline]
pub fn decode(word: u32, ip: u64, features: FeatureSet, out: &mut Instruction) {
    // Top-level split for the 101x group (ARM ARM C4.1.3). We key off the high
    // byte / sub-fields directly rather than re-deriving the manual's op0/op1.
    let op = bits(word, 29, 3); // word<31:29>
    let b25_31 = bits(word, 25, 7); // word<31:25>

    // Unconditional branch (register): word<31:25> == 1101011.
    if b25_31 == 0b1101011 {
        decode_uncond_branch_reg(word, features, out);
        return;
    }
    // Exception generation + System: word<31:24> == 11010100 / system block.
    if bits(word, 24, 8) == 0b1101_0100 {
        decode_exception(word, features, out);
        return;
    }
    if bits(word, 22, 10) == 0b11_0101_0100 {
        decode_system(word, features, out);
        return;
    }
    // FEAT_D128 system pair block: identical to the system block but with
    // `word<22> == 1` (so the 10-bit prefix is `11_0101_0101`). Holds `MRRS`,
    // `MSRR` and `SYSP`/`TLBIP`.
    if bits(word, 22, 10) == 0b11_0101_0101 {
        decode_system_pair(word, features, out);
        return;
    }
    // TCHANGE translation-table change block: `word<31:22> == 11_0101_0110`.
    if bits(word, 22, 10) == 0b11_0101_0110 {
        decode_tchange(word, features, out);
        return;
    }

    // Conditional branch (immediate): word<31:25> == 0101010.
    if b25_31 == 0b0101010 {
        decode_cond_branch(word, features, out);
        return;
    }

    // Compare-and-branch (immediate): word<30:25> == 011010.
    if bits(word, 25, 6) == 0b011010 {
        decode_compare_branch(word, ip, out);
        return;
    }
    // Test-and-branch (immediate): word<30:25> == 011011.
    if bits(word, 25, 6) == 0b011011 {
        decode_test_branch(word, ip, out);
        return;
    }
    // Compare-and-branch (register/immediate), FEAT_CMPBR: word<30:25> == 111010.
    if bits(word, 25, 6) == 0b111010 {
        decode_cmpbr(word, ip, features, out);
        return;
    }

    // Unconditional branch (immediate): word<30:26> == 00101.
    if bits(word, 26, 5) == 0b00101 {
        out.set(if op & 0b100 != 0 { Code::BlImm } else { Code::BUncond });
        let imm26 = bits(word, 0, 26);
        let off = sign_extend((imm26 as u64) << 2, 28);
        out.push_operand(Operand::Label(ip.wrapping_add(off as u64)));
    }

    // Anything else in this group is unallocated: leave invalid.
}

// ---------------------------------------------------------------------------
// Conditional branch (immediate): B.cond / BC.cond.
// ---------------------------------------------------------------------------

/// `B.<cond>`, the FEAT_HBC `BC.<cond>`, and the FEAT_PAuth_LR PC-relative
/// returns `RETAASPPC`/`RETABSPPC` — the encodings whose top seven bits are
/// `0101010` (this is the `word<31:25> == 0101010` arm of [`decode`]).
///
/// * `o1` (`word<24>`) == 0 — conditional branch. Target is
///   `ip + SignExtend(imm19:00, 21)`; the condition is carried as an
///   [`Operand::Cond`] which the formatter fuses into the mnemonic. `o0`
///   (`word<4>`) selects the hinted `BC.<cond>` (1, FEAT_HBC) over the ordinary
///   `B.<cond>` (0).
/// * `o1` (`word<24>`) == 1 — the FEAT_PAuth_LR `RETAASPPC`/`RETABSPPC`
///   PC-relative returns (`word<23:22> == 00`, `word<4:0> == 11111`,
///   `word<21>` the A/B key); see [`decode_ret_sppc`].
#[inline]
fn decode_cond_branch(word: u32, features: FeatureSet, out: &mut Instruction) {
    if bit(word, 24) != 0 {
        // FEAT_PAuth_LR RETAASPPC/RETABSPPC live in this `0101010 1 ...` slot.
        decode_ret_sppc(word, features, out);
        return;
    }
    let imm19 = bits(word, 5, 19);
    let off = sign_extend((imm19 as u64) << 2, 21);
    let cond = Condition::from_u4(bits(word, 0, 4) as u8);

    // o0 (word<4>) selects the FEAT_HBC hinted branch `BC.<cond>` (1) over the
    // ordinary `B.<cond>` (0). The two encodings disassemble to distinct
    // mnemonics (`b.<cond>` vs `bc.<cond>`); `BC.<cond>` is gated on FEAT_HBC.
    if bit(word, 4) != 0 {
        if !features.has(Feature::Hbc) {
            return;
        }
        out.set(Code::BcCond);
    } else {
        out.set(Code::BCond);
    }
    out.push_operand(Operand::Cond(cond));
    out.push_operand(Operand::Label(out.ip().wrapping_add(off as u64)));
}

/// `RETAASPPC`/`RETABSPPC <label>` (FEAT_PAuth_LR PC-relative return) —
/// `0101010 1 00 M imm16 11111`. The `word<21>` `M` bit selects the signing key
/// (0 = key A / `RETAASPPC`, 1 = key B / `RETABSPPC`); the 16-bit `imm16`
/// (`word<20:5>`) gives a *backward* PC-relative target `ip - (imm16:00)`. The
/// fixed structural bits `word<23:22> == 00` and `word<4:0> == 11111` must hold;
/// every other selector here is UNALLOCATED. Gated on [`Feature::PauthLr`].
#[inline]
fn decode_ret_sppc(word: u32, features: FeatureSet, out: &mut Instruction) {
    if !features.has(Feature::PauthLr) {
        return;
    }
    // Fixed: word<23:22> == 00, word<4:0> == 11111.
    if bits(word, 22, 2) != 0 || bits(word, 0, 5) != 0b11111 {
        return;
    }
    out.set(if bit(word, 21) == 1 {
        Code::Retabsppc
    } else {
        Code::Retaasppc
    });
    push_sppc_label(word, out);
}

/// Push the backward PC-relative [`Operand::Label`] shared by the FEAT_PAuth_LR
/// `*SPPC` forms. The `imm16` at `word<20:5>` is the *negated* offset:
/// `target = ip - (imm16 << 2)`.
#[inline]
fn push_sppc_label(word: u32, out: &mut Instruction) {
    let imm16 = bits(word, 5, 16);
    let off = (imm16 as u64) << 2;
    out.push_operand(Operand::Label(out.ip().wrapping_sub(off)));
}

/// `AUTIASPPC`/`AUTIBSPPC <label>` (FEAT_PAuth_LR, PC-relative authenticate of
/// the link register against a signing instruction) — `1111001110 M imm16
/// 11111`. These live in the Data-Processing (Immediate) top-level group
/// (`op0 == 100x`), so [`crate::decode::dp_imm`] routes here once it recognizes
/// the `word<31:22> == 1111001110` / `word<4:0> == 11111` pattern.
///
/// `word<21>` `M` selects the key (0 = key A / `AUTIASPPC`, 1 = key B /
/// `AUTIBSPPC`); the backward PC-relative target is `ip - (imm16:00)`, exactly
/// like [`decode_ret_sppc`]. The caller has already matched the fixed mask and
/// checked [`Feature::PauthLr`]; this routine only sets the code + label.
#[inline]
pub(crate) fn decode_auti_sppc(word: u32, out: &mut Instruction) {
    out.set(if bit(word, 21) == 1 {
        Code::Autibsppc
    } else {
        Code::Autiasppc
    });
    push_sppc_label(word, out);
}

// ---------------------------------------------------------------------------
// Compare and branch / Test and branch (immediate).
// ---------------------------------------------------------------------------

/// `CBZ`/`CBNZ <Wt|Xt>, <label>`. `sf` selects the register width; `op` selects
/// the non-zero variant. Target is `ip + SignExtend(imm19:00, 21)`.
#[inline]
fn decode_compare_branch(word: u32, ip: u64, out: &mut Instruction) {
    let sf = bit(word, 31);
    let op = bit(word, 24); // 0 = CBZ, 1 = CBNZ
    let imm19 = bits(word, 5, 19);
    let rt = bits(word, 0, 5);
    let w = if sf == 1 { RegWidth::X64 } else { RegWidth::W32 };

    out.set(match (op, sf) {
        (0, 0) => Code::Cbz32,
        (0, _) => Code::Cbz64,
        (1, 0) => Code::Cbnz32,
        (_, _) => Code::Cbnz64,
    });
    out.push_operand(reg(false, w, rt));
    let off = sign_extend((imm19 as u64) << 2, 21);
    out.push_operand(Operand::Label(ip.wrapping_add(off as u64)));
}

/// `TBZ`/`TBNZ <R><t>, #imm, <label>`. The bit position is `b5:b40`
/// (`word<31>` is the high bit, `word<23:19>` the low five); the register is `X`
/// when `b5 == 1`. Target is `ip + SignExtend(imm14:00, 16)`.
#[inline]
fn decode_test_branch(word: u32, ip: u64, out: &mut Instruction) {
    let b5 = bit(word, 31);
    let op = bit(word, 24); // 0 = TBZ, 1 = TBNZ
    let b40 = bits(word, 19, 5);
    let imm14 = bits(word, 5, 14);
    let rt = bits(word, 0, 5);
    let bitpos = (b5 << 5) | b40;
    // Binary Ninja prints the register width following b5 (X when b5==1).
    let w = if b5 == 1 { RegWidth::X64 } else { RegWidth::W32 };

    out.set(if op == 1 { Code::Tbnz } else { Code::Tbz });
    out.push_operand(reg(false, w, rt));
    out.push_operand(Operand::ImmUnsigned(bitpos as u64));
    let off = sign_extend((imm14 as u64) << 2, 16);
    out.push_operand(Operand::Label(ip.wrapping_add(off as u64)));
}

// ---------------------------------------------------------------------------
// Compare-and-branch (register / immediate), FEAT_CMPBR.
// ---------------------------------------------------------------------------

/// FEAT_CMPBR compare-and-branch (word<30:25> == 111010). `bit24` selects the
/// register form (0) from the immediate form (1); the common branch target is
/// `ip + SignExtend(imm9:00, 11)` from `word<13:5>`.
///
/// * Register form: `CB<cc> <Rm>, <Rn>, <label>` — `sf` (word<31>) and the
///   size field `word<15:14>` (00 = word, 10 = byte `CBB`, 11 = halfword `CBH`)
///   select the register width / mnemonic family; `Rm = word<20:16>`,
///   `Rn = word<4:0>`. Byte/halfword require `sf == 0`; size `01` is unallocated.
///   The condition `word<23:21>` is `gt/ge/hi/hs/--/--/eq/ne` (100/101 reserved).
/// * Immediate form: `CB<cc> <Rn>, #imm6, <label>` — `sf` selects the width, the
///   size field must be `00`, the 6-bit compare immediate is `word<20:15>`, and
///   the condition `word<23:21>` is `gt/lt/hi/lo/--/--/eq/ne`.
#[inline]
fn decode_cmpbr(word: u32, ip: u64, features: FeatureSet, out: &mut Instruction) {
    if !features.has(Feature::Cmpbr) {
        return;
    }
    let sf = bit(word, 31);
    let imm = bit(word, 24); // 0 = register form, 1 = immediate form
    let cc = bits(word, 21, 3);
    let size = bits(word, 14, 2);
    let rt = bits(word, 0, 5);
    // Common PC-relative target: imm9 at word<13:5>, *4, sign-extended to 11 bits.
    let off = sign_extend((bits(word, 5, 9) as u64) << 2, 11);
    let label = Operand::Label(ip.wrapping_add(off as u64));

    if imm == 0 {
        // Register form. Map (size, sf) -> width + condition family.
        let code = match (size, cc) {
            // Word (size==00): sf selects W/X.
            (0b00, 0b000) => Code::Cbgt,
            (0b00, 0b001) => Code::Cbge,
            (0b00, 0b010) => Code::Cbhi,
            (0b00, 0b011) => Code::Cbhs,
            (0b00, 0b110) => Code::Cbeq,
            (0b00, 0b111) => Code::Cbne,
            // Byte (size==10): sf must be 0.
            (0b10, 0b000) if sf == 0 => Code::Cbbgt,
            (0b10, 0b001) if sf == 0 => Code::Cbbge,
            (0b10, 0b010) if sf == 0 => Code::Cbbhi,
            (0b10, 0b011) if sf == 0 => Code::Cbbhs,
            (0b10, 0b110) if sf == 0 => Code::Cbbeq,
            (0b10, 0b111) if sf == 0 => Code::Cbbne,
            // Halfword (size==11): sf must be 0.
            (0b11, 0b000) if sf == 0 => Code::Cbhgt,
            (0b11, 0b001) if sf == 0 => Code::Cbhge,
            (0b11, 0b010) if sf == 0 => Code::Cbhhi,
            (0b11, 0b011) if sf == 0 => Code::Cbhhs,
            (0b11, 0b110) if sf == 0 => Code::Cbheq,
            (0b11, 0b111) if sf == 0 => Code::Cbhne,
            // size==01, the reserved conditions (100/101), or a width-illegal
            // byte/halfword: unallocated.
            _ => return,
        };
        // Byte/halfword forms always use W registers; the word form follows sf.
        let w = if size == 0b00 && sf == 1 {
            RegWidth::X64
        } else {
            RegWidth::W32
        };
        let rm = bits(word, 16, 5);
        // LLVM orders the operands `<Rt>, <Rm>, <label>` (the compared register
        // `Rt` at word<4:0> first, then `Rm` at word<20:16>).
        out.set(code);
        out.push_operand(reg(false, w, rt));
        out.push_operand(reg(false, w, rm));
        out.push_operand(label);
        return;
    }

    // Immediate form: the 6-bit compare immediate occupies word<20:15>, so the
    // only structural bit below it (word<14>) must be 0; everything else here is
    // unallocated. (The register form's `size` was word<15:14>, but word<15> is
    // the imm6 LSB in this form, so we check word<14> directly.)
    if bit(word, 14) != 0 {
        return;
    }
    let code = match cc {
        0b000 => Code::Cbgt,
        0b001 => Code::Cblt,
        0b010 => Code::Cbhi,
        0b011 => Code::Cblo,
        0b110 => Code::Cbeq,
        0b111 => Code::Cbne,
        _ => return,
    };
    let w = if sf == 1 { RegWidth::X64 } else { RegWidth::W32 };
    let imm6 = bits(word, 15, 6);
    out.set(code);
    out.push_operand(reg(false, w, rt));
    out.push_operand(Operand::ImmUnsigned(imm6 as u64));
    out.push_operand(label);
}

// ---------------------------------------------------------------------------
// Unconditional branch (register): BR/BLR/RET/ERET/DRPS + PAuth variants.
// ---------------------------------------------------------------------------

/// Unconditional branch-register class (ARM ARM C4.1.3). Dispatches on
/// `opc:op2:op3:op4` (`word<24:21>:<20:16>:<15:10>:<4:0>`). Covers the plain
/// `BR`/`BLR`/`RET`/`ERET`/`DRPS` and the FEAT_PAuth `BRAA`.. family.
#[inline]
fn decode_uncond_branch_reg(word: u32, features: FeatureSet, out: &mut Instruction) {
    let opc = bits(word, 21, 4);
    let op2 = bits(word, 16, 5);
    let op3 = bits(word, 10, 6);
    let rn = bits(word, 5, 5);
    let op4 = bits(word, 0, 5);

    // op2 is always 11111 for the allocated branch-register encodings.
    if op2 != 0b11111 {
        return;
    }

    let pauth = features.has(Feature::PAuth);

    match (opc, op3, op4) {
        // BR <Xn>
        (0b0000, 0b000000, 0b00000) => {
            out.set(Code::Br);
            out.push_operand(xreg(false, rn));
        }
        // BRAAZ/BRABZ <Xn> (Z form, op4==11111), op3 = 00001x
        (0b0000, 0b000010, 0b11111) if pauth => {
            out.set(Code::Braaz);
            out.push_operand(xreg(false, rn));
        }
        (0b0000, 0b000011, 0b11111) if pauth => {
            out.set(Code::Brabz);
            out.push_operand(xreg(false, rn));
        }
        // BLR <Xn>
        (0b0001, 0b000000, 0b00000) => {
            out.set(Code::Blr);
            out.push_operand(xreg(false, rn));
        }
        (0b0001, 0b000010, 0b11111) if pauth => {
            out.set(Code::Blraaz);
            out.push_operand(xreg(false, rn));
        }
        (0b0001, 0b000011, 0b11111) if pauth => {
            out.set(Code::Blrabz);
            out.push_operand(xreg(false, rn));
        }
        // RET {<Xn>} — default x30 elided.
        (0b0010, 0b000000, 0b00000) => {
            out.set(Code::Ret);
            if rn != 30 {
                out.push_operand(xreg(false, rn));
            }
        }
        // RETAA/RETAB (Rn==11111, Rm==11111).
        (0b0010, 0b000010, 0b11111) if pauth && rn == 0b11111 => {
            out.set(Code::Retaa);
        }
        (0b0010, 0b000011, 0b11111) if pauth && rn == 0b11111 => {
            out.set(Code::Retab);
        }
        // ERET / ERETAA / ERETAB (Rn==11111).
        (0b0100, 0b000000, 0b00000) if rn == 0b11111 => {
            out.set(Code::Eret);
        }
        (0b0100, 0b000010, 0b11111) if pauth && rn == 0b11111 => {
            out.set(Code::Eretaa);
        }
        (0b0100, 0b000011, 0b11111) if pauth && rn == 0b11111 => {
            out.set(Code::Eretab);
        }
        // DRPS (Rn==11111).
        (0b0101, 0b000000, 0b00000) if rn == 0b11111 => {
            out.set(Code::Drps);
        }
        // BRAA/BRAB <Xn>, <Xm|SP> (opc==1000, op3==00001x).
        (0b1000, 0b000010, _) if pauth => {
            out.set(Code::Braa);
            out.push_operand(xreg(false, rn));
            out.push_operand(xreg(true, op4));
        }
        (0b1000, 0b000011, _) if pauth => {
            out.set(Code::Brab);
            out.push_operand(xreg(false, rn));
            out.push_operand(xreg(true, op4));
        }
        // BLRAA/BLRAB <Xn>, <Xm|SP> (opc==1001).
        (0b1001, 0b000010, _) if pauth => {
            out.set(Code::Blraa);
            out.push_operand(xreg(false, rn));
            out.push_operand(xreg(true, op4));
        }
        (0b1001, 0b000011, _) if pauth => {
            out.set(Code::Blrab);
            out.push_operand(xreg(false, rn));
            out.push_operand(xreg(true, op4));
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Exception generation: SVC/HVC/SMC/BRK/HLT/DCPS{1,2,3}/TCANCEL.
// ---------------------------------------------------------------------------

/// Exception-generation class (ARM ARM C4.1.3): `word<31:24> == 11010100`.
/// Dispatches on `opc:op2:LL` (`word<23:21>:<4:2>:<1:0>`); the payload is the
/// 16-bit `imm16` at `word<20:5>`.
#[inline]
fn decode_exception(word: u32, features: FeatureSet, out: &mut Instruction) {
    let opc = bits(word, 21, 3);
    let imm16 = bits(word, 5, 16);
    let op2 = bits(word, 2, 3);
    let ll = bits(word, 0, 2);

    // op2 must be 000 for every allocated exception-generation encoding.
    if op2 != 0 {
        return;
    }

    let imm = Operand::ImmUnsigned(imm16 as u64);
    match (opc, ll) {
        (0b000, 0b01) => {
            out.set(Code::Svc);
            out.push_operand(imm);
        }
        (0b000, 0b10) => {
            out.set(Code::Hvc);
            out.push_operand(imm);
        }
        (0b000, 0b11) => {
            out.set(Code::Smc);
            out.push_operand(imm);
        }
        (0b001, 0b00) => {
            out.set(Code::Brk);
            out.push_operand(imm);
        }
        (0b010, 0b00) => {
            out.set(Code::Hlt);
            out.push_operand(imm);
        }
        // TCANCEL #imm16 (FEAT_TME): opc==011, LL==00.
        (0b011, 0b00) if features.has(Feature::Tme) => {
            out.set(Code::Tcancel);
            out.push_operand(imm);
        }
        (0b101, 0b01) => {
            out.set(Code::Dcps1);
            push_optional_imm(out, imm16);
        }
        (0b101, 0b10) => {
            out.set(Code::Dcps2);
            push_optional_imm(out, imm16);
        }
        (0b101, 0b11) => {
            out.set(Code::Dcps3);
            push_optional_imm(out, imm16);
        }
        _ => {}
    }
}

/// `DCPS{1,2,3}` take an optional `#imm`; a zero immediate is elided.
#[inline]
fn push_optional_imm(out: &mut Instruction, imm16: u32) {
    if imm16 != 0 {
        out.push_operand(Operand::ImmUnsigned(imm16 as u64));
    }
}

// ---------------------------------------------------------------------------
// System: hints, barriers, MSR(imm)/PSTATE, MSR/MRS(reg), SYS/SYSL.
// ---------------------------------------------------------------------------

/// TCHANGE translation-table change block (`word<31:22> == 1101010110`).
///
/// `TCHANGE{F,B} <Xt>, <Xn>{, nb}` (register) and `TCHANGE{F,B} <Xt>, #<imm>{,
/// nb}` (immediate). Layout: `L (word<21>) == 0`, `op0 (word<20:19>)` selects
/// register (`00`, `Xn = word<9:5>`) vs immediate (`10`, `imm7 = word<11:5>`);
/// `word<18>` selects forward (0) / backward (1); `word<17>` selects the `nb`
/// (no-barrier) modifier; `word<16> == 0` and `CRn (word<15:12>) == 0`. For the
/// register form the high `CRm` bits (`word<11:10>`) must be 0. `Xt = word<4:0>`.
#[inline]
fn decode_tchange(word: u32, features: FeatureSet, out: &mut Instruction) {
    if !features.has(Feature::Tchange) {
        return;
    }
    // Fixed structural bits: L==0, op1 LSB (word<16>)==0, CRn==0.
    if bit(word, 21) != 0 || bit(word, 16) != 0 || bits(word, 12, 4) != 0 {
        return;
    }
    let backward = bit(word, 18) == 1;
    let nb = bit(word, 17) == 1; // no-barrier modifier (trailing `, nb`).
    let rt = bits(word, 0, 5);
    match bits(word, 19, 2) {
        0b00 => {
            // Register form: `Xn = word<9:5>`; the upper `CRm` bits must be 0.
            if bits(word, 10, 2) != 0 {
                return;
            }
            let xn = bits(word, 5, 5);
            out.set(if backward { Code::TchangebReg } else { Code::TchangefReg });
            out.push_operand(xreg(false, rt));
            out.push_operand(xreg(false, xn));
        }
        0b10 => {
            // Immediate form: `imm7 = word<11:5>`.
            let imm = bits(word, 5, 7);
            out.set(if backward { Code::TchangebImm } else { Code::TchangefImm });
            out.push_operand(xreg(false, rt));
            out.push_operand(Operand::ImmUnsigned(imm as u64));
        }
        _ => return,
    }
    if nb {
        out.push_operand(Operand::SysOp(crate::sysop::SysToken::of("nb")));
    }
}

/// System-instruction block (ARM ARM C4.1.3): `word<31:22> == 1101010100`.
/// Fields: `L` (`word<21>`), `op0` (`word<20:19>`), `op1` (`word<18:16>`),
/// `CRn` (`word<15:12>`), `CRm` (`word<11:8>`), `op2` (`word<7:5>`),
/// `Rt` (`word<4:0>`).
#[inline]
fn decode_system(word: u32, features: FeatureSet, out: &mut Instruction) {
    let l = bit(word, 21);
    let op0 = bits(word, 19, 2);
    let op1 = bits(word, 16, 3);
    let crn = bits(word, 12, 4);
    let crm = bits(word, 8, 4);
    let op2 = bits(word, 5, 3);
    let rt = bits(word, 0, 5);

    if l == 0 {
        // Write side.
        match op0 {
            0b00 => {
                // PSTATE / hints / barriers / WFxT (Rt == 11111, except WFxT which
                // carries Rt). These named encodings occupy only a subset of
                // `(CRn, CRm, op2)`; every selector that is *architecturally
                // unallocated* is a generic `MSR (register)` per the ARM ARM (and
                // LLVM): `msr S0_<op1>_c<CRn>_c<CRm>_<op2>, <Xt>`.
                decode_op0_00_named(out, features, false, op1, crn, crm, op2, rt);
                if out.is_invalid() && op0_00_unallocated(false, op1, crn, crm, op2, rt) {
                    decode_sysreg_move(out, false, op0, op1, crn, crm, op2, rt);
                }
            }
            0b01 => {
                // SYS / IC / DC / AT / TLBI / CFP / CPP / DVP.
                decode_sys(out, op1, crn, crm, op2, rt, false);
            }
            // op0 == 10/11 with L==0 is MSR (register).
            _ => {
                decode_sysreg_move(out, false, op0, op1, crn, crm, op2, rt);
            }
        }
    } else {
        // Read side (L == 1).
        match op0 {
            0b00 => {
                // System instructions with result (FEAT_TME): TSTART/TTEST. Every
                // other op0==00 read selector that is architecturally unallocated
                // is a generic `MRS` from system register `S0_<op1>_...`.
                decode_op0_00_named(out, features, true, op1, crn, crm, op2, rt);
                if out.is_invalid() && op0_00_unallocated(true, op1, crn, crm, op2, rt) {
                    decode_sysreg_move(out, true, op0, op1, crn, crm, op2, rt);
                }
            }
            0b01 => {
                // SYSL <Xt>, #op1, Cn, Cm, #op2.
                decode_sys(out, op1, crn, crm, op2, rt, true);
            }
            0b10 | 0b11 => {
                // MRS <Xt>, <systemreg>.
                decode_sysreg_move(out, true, op0, op1, crn, crm, op2, rt);
            }
            _ => {}
        }
    }
}

/// Run the op0==00 named-encoding dispatch (PSTATE / hint / barrier / WFxT for
/// the write side; the system-result `TSTART`/`TTEST` for the read side). Leaves
/// `out` invalid if nothing matched (either unallocated or feature-gated-off).
#[inline]
#[allow(clippy::too_many_arguments)]
fn decode_op0_00_named(
    out: &mut Instruction,
    features: FeatureSet,
    read: bool,
    op1: u32,
    crn: u32,
    crm: u32,
    op2: u32,
    rt: u32,
) {
    if read {
        // System instructions with result (FEAT_TME): TSTART/TTEST.
        decode_systemresult(out, features, op1, crn, crm, op2, rt);
        return;
    }
    if crn == 0b0100 {
        // MSR (immediate) — PSTATE field, op1:op2 select it, CRm = imm.
        decode_msr_imm(out, op1, op2, crm, rt);
    } else if crn == 0b0010 {
        // HINT space (CRn==2): CRm:op2 select the hint.
        decode_hint(out, features, crm, op2, rt);
    } else if crn == 0b0011 {
        // Barriers (CRn==3).
        decode_barrier(out, features, crm, op2, rt);
    } else if crn == 0b0001 {
        // WFET/WFIT (CRn==1, CRm==0, op2 selects), with Rt.
        decode_wfxt(out, features, crm, op2, rt);
    }
}

/// Whether the op0==00 selector is *genuinely unallocated* as a named system
/// encoding — i.e. it would decode to nothing even with every feature enabled.
/// Such a selector falls back to the generic `MSR`/`MRS (register)` form (what
/// the ARM ARM and LLVM emit). A selector that decodes under [`FeatureSet::ALL`]
/// but not under the current `features` is a *feature-gated* encoding (e.g. a
/// `DSB <option>nXS` without `FEAT_XS`); it stays `Invalid` rather than masking
/// the real instruction with a generic move.
///
/// Implemented by replaying the named dispatch into a scratch instruction with
/// all features on; this needs no hand-maintained allocation table and tracks
/// the named decoders automatically.
#[inline]
fn op0_00_unallocated(read: bool, op1: u32, crn: u32, crm: u32, op2: u32, rt: u32) -> bool {
    let mut scratch = Instruction::default();
    decode_op0_00_named(&mut scratch, FeatureSet::ALL, read, op1, crn, crm, op2, rt);
    scratch.is_invalid()
}

/// `MSR <pstatefield>, #imm` (immediate, ARM ARM C6) plus the no-operand PSTATE
/// ops `CFINV`/`XAFLAG`/`AXFLAG`. Named fields render `msr <field>, #CRm`;
/// selectors Binary Ninja does not name (e.g. `PAN`, or an out-of-range `SSBS`
/// immediate) fall back to the generic `MSR (register)` form it emits
/// (`msr s0_0_c4_cN_M, xzr`).
#[inline]
fn decode_msr_imm(out: &mut Instruction, op1: u32, op2: u32, crm: u32, rt: u32) {
    // The PSTATE encodings require Rt == 11111.
    if rt != 0b11111 {
        return;
    }

    // SME SMSTART / SMSTOP (SVCR PSTATE, FEAT_SME): op1==011, op2==011, CRn==0100
    // (already established by the caller). The 4-bit CRm field is `0:ZA:SM:start`
    // where bit0 selects SMSTART(1)/SMSTOP(0), bit1 the streaming-mode (SM) state
    // and bit2 the ZA state. Binary Ninja renders the SM-only / ZA-only forms with
    // a bare `sm` / `za` option and the combined `SM+ZA` form with no option.
    //
    // This special-case is deliberately scoped to op1==3 && op2==3 so it cannot
    // perturb any other `MSR (immediate)` PSTATE selector; it is *not* feature
    // gated for rendering (the golden corpus and binja always show these), but a
    // standalone `Code` (`Smstart`/`Smstop`) carries the `Feature::Sme` gate.
    if op1 == 0b011 && op2 == 0b011 {
        let is_start = (crm & 0b0001) != 0;
        let sm = (crm & 0b0010) != 0;
        let za = (crm & 0b0100) != 0;
        out.set(if is_start { Code::Smstart } else { Code::Smstop });
        // Option keyword: `sm` (SM only), `za` (ZA only), or none (SM and ZA).
        match (sm, za) {
            (true, false) => out.push_operand(sysop("sm")),
            (false, true) => out.push_operand(sysop("za")),
            // Both set (the canonical bare `smstart`/`smstop`) or neither
            // (reserved CRm — binja still prints the bare mnemonic): no option.
            _ => {}
        }
        return;
    }

    // CFINV / XAFLAG / AXFLAG: op1==000, op2 in {0,1,2}, only when CRm==0 (the
    // immediate field is reserved). Other CRm values fall through to generic MSR.
    if op1 == 0b000 && crm == 0 {
        let code = match op2 {
            0b000 => Some(Code::Cfinv),
            0b001 => Some(Code::Xaflag),
            0b010 => Some(Code::Axflag),
            _ => None,
        };
        if let Some(c) = code {
            out.set(c);
            return;
        }
    }

    if let Some(field) = pstate_field_name(op1, op2, crm) {
        out.set(Code::MsrImm);
        out.set_mnemonic(Mnemonic::Msr);
        out.push_operand(sysop(field));
        out.push_operand(Operand::ImmUnsigned(crm as u64));
        return;
    }
    // Unknown PSTATE selector: Binary Ninja renders the generic MSR(register)
    // form `msr s0_0_c4_cCRm_op2, xzr`. Match that rather than inventing a name.
    out.set(Code::MsrReg);
    let sr = SystemReg::from_encoding(0, op1 as u8, 0b0100, crm as u8, op2 as u8);
    out.push_operand(Operand::SysReg(sr));
    out.push_operand(xreg(false, 0b11111));
}

/// The named PSTATE field for `MSR (immediate)` selected by `(op1, op2)`, or
/// `None` for a selector Binary Ninja renders generically.
///
/// Divergence note: this mirrors Binary Ninja's directory rather than the full
/// ARM ARM `MSR (immediate)` table — it omits `PAN` (binja never names it) and
/// validates the `SSBS` immediate (`CRm <= 1`), both binja-specific behaviours
/// the differential corpus pins down. `SVCR*` (op1=3,op2=3 / `SMSTART`/`SMSTOP`)
/// is left generic because the SME mnemonics are out of this group's scope.
#[inline]
const fn pstate_field_name(op1: u32, op2: u32, crm: u32) -> Option<&'static str> {
    let name = match (op1, op2) {
        (0b000, 0b011) => "uao",
        (0b000, 0b101) => "spsel",
        // SSBS takes a 1-bit immediate; binja only names a valid CRm (0/1).
        (0b011, 0b001) if crm <= 1 => "ssbs",
        (0b011, 0b010) => "dit",
        (0b011, 0b100) => "tco",
        (0b011, 0b110) => "daifset",
        (0b011, 0b111) => "daifclr",
        (0b001, 0b000) => "allint", // FEAT_NMI
        _ => return None,
    };
    Some(name)
}

/// The `HINT` space (`CRn==2`): `CRm:op2` form a 7-bit selector. Renders the
/// NOP-family names, the PAuth hints (gated), and falls back to `HINT #imm` for
/// unallocated selectors. All require `Rt == 11111`.
#[inline]
fn decode_hint(out: &mut Instruction, features: FeatureSet, crm: u32, op2: u32, rt: u32) {
    if rt != 0b11111 {
        return;
    }
    let sel = (crm << 3) | op2; // CRm:op2, the HINT immediate.
    let pauth = features.has(Feature::PAuth);

    // PAuth hint codes live in the HINT space; gate them on FEAT_PAuth and fall
    // back to the generic `HINT #imm` when the feature is not accepted.
    let code = match sel {
        0 => Code::Nop,
        1 => Code::Yield,
        2 => Code::Wfe,
        3 => Code::Wfi,
        4 => Code::Sev,
        5 => Code::Sevl,
        // 6 == DGH (FEAT_DGH) — Binary Ninja renders it as the generic HINT.
        7 if pauth => {
            // XPACLRI (HINT #7).
            out.set(Code::HintGeneric);
            out.set_mnemonic(Mnemonic::Xpaclri);
            return;
        }
        8 if pauth => return set_pauth_hint(out, Mnemonic::Pacia1716),
        10 if pauth => return set_pauth_hint(out, Mnemonic::Pacib1716),
        12 if pauth => return set_pauth_hint(out, Mnemonic::Autia1716),
        14 if pauth => return set_pauth_hint(out, Mnemonic::Autib1716),
        16 => Code::Esb,
        17 => Code::Psb,  // PSB CSYNC
        18 => {
            // TSB CSYNC (FEAT_TRF).
            if features.has(Feature::Trf) {
                out.set(Code::Tsb);
                out.push_operand(sysop("csync"));
                return;
            }
            return set_generic_hint(out, sel);
        }
        20 => Code::Csdb,
        24 if pauth => return set_pauth_hint(out, Mnemonic::Paciaz),
        25 if pauth => return set_pauth_hint(out, Mnemonic::Paciasp),
        26 if pauth => return set_pauth_hint(out, Mnemonic::Pacibz),
        27 if pauth => return set_pauth_hint(out, Mnemonic::Pacibsp),
        28 if pauth => return set_pauth_hint(out, Mnemonic::Autiaz),
        29 if pauth => return set_pauth_hint(out, Mnemonic::Autiasp),
        30 if pauth => return set_pauth_hint(out, Mnemonic::Autibz),
        31 if pauth => return set_pauth_hint(out, Mnemonic::Autibsp),
        // BTI {<targets>} (FEAT_BTI): CRm==4, op2 selects the targets.
        _ if crm == 0b0100 && (op2 & 0b001) == 0 => {
            out.set(Code::Bti);
            // op2: 000 -> (none), 010 -> c, 100 -> j, 110 -> jc.
            let t = match op2 {
                0b010 => Some("c"),
                0b100 => Some("j"),
                0b110 => Some("jc"),
                _ => None,
            };
            if let Some(s) = t {
                out.push_operand(sysop(s));
            }
            return;
        }
        _ => return set_generic_hint(out, sel),
    };
    out.set(code);
    // PSB and ESB take no operand; PSB renders "psb csync".
    if matches!(code, Code::Psb) {
        out.push_operand(sysop("csync"));
    }
}

/// Set a PAuth hint mnemonic (no operands), keeping the canonical HINT code.
#[inline]
fn set_pauth_hint(out: &mut Instruction, m: Mnemonic) {
    out.set(Code::HintGeneric);
    out.set_mnemonic(m);
}

/// Set the generic `HINT #imm` for an unallocated selector.
#[inline]
fn set_generic_hint(out: &mut Instruction, sel: u32) {
    out.set(Code::HintGeneric);
    out.push_operand(Operand::ImmUnsigned(sel as u64));
}

/// `WFET`/`WFIT <Xt>` (FEAT_WFxT): `CRn==1, CRm==0`, `op2` selects.
#[inline]
fn decode_wfxt(out: &mut Instruction, features: FeatureSet, crm: u32, op2: u32, rt: u32) {
    if crm != 0 || !features.has(Feature::Wfxt) {
        return;
    }
    match op2 {
        0b000 => {
            out.set(Code::Wfet);
            out.push_operand(xreg(false, rt));
        }
        0b001 => {
            out.set(Code::Wfit);
            out.push_operand(xreg(false, rt));
        }
        _ => {}
    }
}

/// Barriers (`CRn==3`): `CLREX`/`DSB`/`DMB`/`ISB`/`SB`/`SSBB`/`PSSBB`/`TCOMMIT`,
/// keyed on `op2` with the option in `CRm`. `Rt` must be `11111`.
#[inline]
fn decode_barrier(out: &mut Instruction, features: FeatureSet, crm: u32, op2: u32, rt: u32) {
    if rt != 0b11111 {
        return;
    }
    match op2 {
        0b001 => {
            // DSB <option>nXS (FEAT_XS): op2==001, CRm<1:0>==10, CRm<3:2>==imm2
            // selecting osh/nsh/ish/sy. Canonical `Code::Dsb` with an `nXS`
            // option keyword. Other CRm<1:0> in this op2 are unallocated (binja
            // `decode_iclass_barriers`).
            if (crm & 0b0011) == 0b0010 && features.has(Feature::Xs) {
                out.set(Code::Dsb);
                let name = match (crm >> 2) & 0b11 {
                    0b00 => "oshnxs",
                    0b01 => "nshnxs",
                    0b10 => "ishnxs",
                    _ => "synxs",
                };
                out.push_operand(sysop(name));
            }
        }
        0b010 => {
            // CLREX {#imm} — the #imm (CRm) is elided when it is 0xf (the default).
            out.set(Code::Clrex);
            if crm != 0b1111 {
                out.push_operand(Operand::ImmUnsigned(crm as u64));
            }
        }
        0b011 if crm == 0 => {
            // TCOMMIT: CRm==0, op2==011 (FEAT_TME). Other CRm is unallocated.
            if features.has(Feature::Tme) {
                out.set(Code::Tcommit);
            }
        }
        0b100 => {
            // DSB <option>|#imm and the special SSBB/PSSBB (CRm 0/4). SSBB and
            // PSSBB are DSB-encoding aliases (canonical `Code::Dsb`).
            out.set(Code::Dsb);
            if crm == 0b0000 {
                out.set_mnemonic(Mnemonic::Ssbb);
            } else if crm == 0b0100 {
                out.set_mnemonic(Mnemonic::Pssbb);
            } else {
                push_barrier_option(out, crm);
            }
        }
        0b101 => {
            // DMB <option>|#imm.
            out.set(Code::Dmb);
            push_barrier_option(out, crm);
        }
        0b110 => {
            // ISB {<option>|#imm}: the only defined option is sy (CRm==1111),
            // which is elided. Any other CRm renders as the numeric `#imm`.
            out.set(Code::Isb);
            if crm != 0b1111 {
                out.push_operand(Operand::ImmUnsigned(crm as u64));
            }
        }
        0b111 => {
            // SB (FEAT_SB): op2==111, Rt==11111. The CRm field is SHOULD-BE-ZERO;
            // CRm!=0 is a reserved SB encoding. LLVM (and binja's current
            // `decode_iclass_barriers`) decode the whole CRm 0..15 range to `SB`
            // (LLVM with a "potentially undefined encoding" warning for CRm!=0),
            // so we follow the spec/LLVM reading and decode all of them as SB.
            //
            // DIVERGENCE: the (stale) binja differential corpus renders CRm 1..15
            // as the generic `msr s0_3_c3_cCRm_7, xzr` (a non-spec MSR(register)
            // fallback). We do NOT reproduce that — SB is the defined instruction
            // both LLVM and the current binja source emit. The decoder drops the
            // SBZ CRm bits; the encoder re-emits the canonical CRm==0, a
            // documented exact-word loss (the re-encode is the same SB).
            out.set(Code::Sb);
        }
        _ => {}
    }
}

/// Push the DSB/DMB barrier option as a named token (`sy`/`ld`/`st`/`ish`/...),
/// or the numeric `#imm` for the unnamed encodings. Includes the FEAT_XS `nXS`
/// variants Binary Ninja prints (`nshnxs`, `synxs`, ...).
#[inline]
fn push_barrier_option(out: &mut Instruction, crm: u32) {
    let name = match crm {
        0b0001 => "oshld",
        0b0010 => "oshst",
        0b0011 => "osh",
        0b0101 => "nshld",
        0b0110 => "nshst",
        0b0111 => "nsh",
        0b1001 => "ishld",
        0b1010 => "ishst",
        0b1011 => "ish",
        0b1101 => "ld",
        0b1110 => "st",
        0b1111 => "sy",
        _ => {
            out.push_operand(Operand::ImmUnsigned(crm as u64));
            return;
        }
    };
    out.push_operand(sysop(name));
}

/// `SYS`/`SYSL` and the `IC`/`DC`/`AT`/`TLBI`/`CFP`/`CPP`/`DVP` aliases.
///
/// `read` selects `SYSL` (`L==1`) vs `SYS` (`L==0`). For the write side we try
/// to resolve a named system-instruction alias from `(op1,CRn,CRm,op2)`; an
/// unrecognized tuple renders the canonical `SYS #op1, Cn, Cm, #op2{, Xt}`.
#[inline]
fn decode_sys(out: &mut Instruction, op1: u32, crn: u32, crm: u32, op2: u32, rt: u32, read: bool) {
    if read {
        // SYSL <Xt>, #op1, Cn, Cm, #op2.
        out.set(Code::Sysl);
        out.push_operand(xreg(false, rt));
        out.push_operand(Operand::ImmUnsigned(op1 as u64));
        out.push_operand(Operand::SysOp(SysToken::cr(crn)));
        out.push_operand(Operand::SysOp(SysToken::cr(crm)));
        out.push_operand(Operand::ImmUnsigned(op2 as u64));
        return;
    }

    // Try the named alias table first.
    if let Some((m, needs_rt)) = sys_alias(op1, crn, crm, op2) {
        out.set(Code::Sys);
        out.set_mnemonic(m);
        // The alias operation name (e.g. "ivau", "zva", "rctx").
        out.push_operand(sysop(alias_op_name(op1, crn, crm, op2)));
        // The optional Xt: only the register-qualified ops print it. The
        // whole-TLB/IALLU* forms never print a register even when the Rt field
        // holds a value other than XZR (matching Binary Ninja).
        if needs_rt {
            out.push_operand(xreg(false, rt));
        }
        return;
    }

    // Canonical SYS #op1, Cn, Cm, #op2{, Xt}.
    out.set(Code::Sys);
    out.push_operand(Operand::ImmUnsigned(op1 as u64));
    out.push_operand(Operand::SysOp(SysToken::cr(crn)));
    out.push_operand(Operand::SysOp(SysToken::cr(crm)));
    out.push_operand(Operand::ImmUnsigned(op2 as u64));
    // Binary Ninja always prints the Xt for the canonical SYS form.
    out.push_operand(xreg(false, rt));
}

/// `TSTART <Xt>` / `TTEST <Xt>` — the FEAT_TME "System instructions with
/// result" (read side, `op0==00`, `op1==011`, `CRn==0011`, `op2==011`). `CRm`
/// selects `TSTART`(0) vs `TTEST`(1); `Rt` is the destination `Xt`. Runtime-gated
/// on [`Feature::Tme`]; all other tuples in this class are UNALLOCATED.
#[inline]
fn decode_systemresult(
    out: &mut Instruction,
    features: FeatureSet,
    op1: u32,
    crn: u32,
    crm: u32,
    op2: u32,
    rt: u32,
) {
    if !features.has(Feature::Tme) {
        return;
    }
    if op1 != 0b011 || crn != 0b0011 || op2 != 0b011 {
        return;
    }
    match crm {
        0 => out.set(Code::Tstart),
        1 => out.set(Code::Ttest),
        _ => return,
    }
    out.push_operand(xreg(false, rt));
}

/// `MSR`/`MRS` (register): move to/from a system register. `read` selects `MRS`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn decode_sysreg_move(
    out: &mut Instruction,
    read: bool,
    op0: u32,
    op1: u32,
    crn: u32,
    crm: u32,
    op2: u32,
    rt: u32,
) {
    let sr = SystemReg::from_encoding(op0 as u8, op1 as u8, crn as u8, crm as u8, op2 as u8);
    if read {
        out.set(Code::Mrs);
        out.push_operand(xreg(false, rt));
        out.push_operand(Operand::SysReg(sr));
    } else {
        out.set(Code::MsrReg);
        out.push_operand(Operand::SysReg(sr));
        out.push_operand(xreg(false, rt));
    }
}

// ---------------------------------------------------------------------------
// FEAT_D128 system-pair block (word<22> == 1): MRRS / MSRR / SYSP / TLBIP.
// ---------------------------------------------------------------------------

/// The `FEAT_D128` system-pair block — the system instructions whose 10-bit
/// prefix is `1101010101` (`word<22> == 1`). Same field layout as the normal
/// system block but transferring a 128-bit value through a consecutive `X`
/// register pair `<Xt>:<Xt+1>` (`Rt` must be even):
///
/// * `MRRS <Xt>, <Xt+1>, <systemreg>` — `L == 1`, read a system-register pair.
/// * `MSRR <systemreg>, <Xt>, <Xt+1>` — `L == 0`, `op0 != 01`, write a pair.
/// * `SYSP #op1, Cn, Cm, #op2{, <Xt>, <Xt+1>}` — `L == 0`, `op0 == 01`; with the
///   `TLBIP` (CRn==8) alias.
///
/// All gated on [`Feature::D128`]; everything else here is UNALLOCATED.
#[inline]
fn decode_system_pair(word: u32, features: FeatureSet, out: &mut Instruction) {
    if !features.has(Feature::D128) {
        return;
    }
    let l = bit(word, 21);
    let op0 = bits(word, 19, 2);
    let op1 = bits(word, 16, 3);
    let crn = bits(word, 12, 4);
    let crm = bits(word, 8, 4);
    let op2 = bits(word, 5, 3);
    let rt = bits(word, 0, 5);

    if l == 0 && op0 == 0b01 {
        // SYSP / TLBIP (system pair). Rt==11111 means "no transfer register"
        // (both halves elided) for the generic form; the TLBIP alias always
        // prints its register pair.
        decode_sysp(out, op1, crn, crm, op2, rt);
        return;
    }

    // MRRS (L==1) / MSRR (L==0): 128-bit system-register pair move. `Rt` must be
    // an even base; an odd `Rt` (including 31) is UNALLOCATED.
    if rt & 1 != 0 {
        return;
    }
    let pair = match reg_pair(rt) {
        Some(p) => p,
        None => return,
    };
    let sr = SystemReg::from_encoding(op0 as u8, op1 as u8, crn as u8, crm as u8, op2 as u8);
    if l == 1 {
        out.set(Code::Mrrs);
        out.push_operand(pair);
        out.push_operand(Operand::SysReg(sr));
    } else {
        out.set(Code::Msrr);
        out.push_operand(Operand::SysReg(sr));
        out.push_operand(pair);
    }
}

/// `SYSP`/`TLBIP` (FEAT_D128 system pair). Mirrors [`decode_sys`] but with a
/// 128-bit `<Xt>:<Xt+1>` transfer pair and only the `TLBIP` (CRn==8) named
/// alias. The generic form elides the pair when `Rt == 11111`; the `TLBIP`
/// alias always prints it.
#[inline]
fn decode_sysp(out: &mut Instruction, op1: u32, crn: u32, crm: u32, op2: u32, rt: u32) {
    // An odd transfer base (other than the no-transfer XZR sentinel 31) is
    // UNALLOCATED for the pair forms.
    if rt & 1 != 0 && rt != 0b11111 {
        return;
    }

    // TLBIP alias: the TLB-maintenance pair lives at CRn==8, reusing the TLBI
    // operation-name directory. It always prints the <Xt>:<Xt+1> pair.
    if crn == 8 && is_tlbi_op(op1, crm, op2) {
        if let Some(pair) = reg_pair_or_zz(rt) {
            out.set(Code::Sysp);
            out.set_mnemonic(Mnemonic::Tlbip);
            out.push_operand(sysop(tlbi_op_name(op1, crm, op2)));
            out.push_operand(pair);
            return;
        }
        return;
    }

    // Canonical SYSP #op1, Cn, Cm, #op2{, <Xt>, <Xt+1>}.
    out.set(Code::Sysp);
    out.push_operand(Operand::ImmUnsigned(op1 as u64));
    out.push_operand(Operand::SysOp(SysToken::cr(crn)));
    out.push_operand(Operand::SysOp(SysToken::cr(crm)));
    out.push_operand(Operand::ImmUnsigned(op2 as u64));
    // The transfer pair is elided when Rt==11111 (no-transfer); otherwise the
    // even base yields `<Xt>, <Xt+1>` (the odd half is XZR when the base is 30).
    if rt != 0b11111 {
        if let Some(pair) = reg_pair(rt) {
            out.push_operand(pair);
        }
    }
}

/// Build an [`Operand::RegPair`] for an even 64-bit base `rt` (`<Xt>:<Xt+1>`),
/// or `None` if `rt` is odd. The odd half resolves to `xzr` when the base is 30.
#[inline]
fn reg_pair(rt: u32) -> Option<Operand> {
    if rt & 1 != 0 {
        return None;
    }
    Some(Operand::RegPair {
        first: gp_register(false, RegWidth::X64, (rt & 0x1f) as u8),
        second: gp_register(false, RegWidth::X64, ((rt + 1) & 0x1f) as u8),
    })
}

/// Like [`reg_pair`] but maps the `Rt == 11111` no-transfer sentinel to the
/// `xzr, xzr` pair (used by `TLBIP`, which always prints the register pair).
#[inline]
fn reg_pair_or_zz(rt: u32) -> Option<Operand> {
    if rt == 0b11111 {
        return Some(Operand::RegPair {
            first: gp_register(false, RegWidth::X64, 31),
            second: gp_register(false, RegWidth::X64, 31),
        });
    }
    reg_pair(rt)
}

// ---------------------------------------------------------------------------
// System-instruction alias tables (ARM ARM C5.1 "System instructions").
// ---------------------------------------------------------------------------

/// Resolve a named `IC`/`DC`/`AT`/`TLBI`/`CFP`/`CPP`/`DVP` alias from the
/// `(op1, CRn, CRm, op2)` system-instruction tuple. Returns the alias mnemonic
/// and whether the operation takes an `<Xt>` operand. `None` means no alias
/// applies and the canonical `SYS` form should be used.
#[inline]
fn sys_alias(op1: u32, crn: u32, crm: u32, op2: u32) -> Option<(Mnemonic, bool)> {
    // CFP/CPP/DVP RCTX (CRn==7, CRm==3): op1==3, op2 in {4,5,7}. Take Xt.
    if crn == 7 && crm == 3 && op1 == 3 {
        return match op2 {
            0b100 => Some((Mnemonic::Cfp, true)),
            0b101 => Some((Mnemonic::Dvp, true)),
            0b111 => Some((Mnemonic::Cpp, true)),
            _ => None,
        };
    }
    // IC (CRn==7, CRm in {1,5}): instruction cache.
    if crn == 7 && (crm == 1 || crm == 5) {
        // IALLUIS (op1=0,CRm=1,op2=0), IALLU (op1=0,CRm=5,op2=0): no Xt.
        // IVAU    (op1=3,CRm=5,op2=1): takes Xt.
        return match (op1, crm, op2) {
            (0, 1, 0) => Some((Mnemonic::Ic, false)),
            (0, 5, 0) => Some((Mnemonic::Ic, false)),
            (3, 5, 1) => Some((Mnemonic::Ic, true)),
            _ => None,
        };
    }
    // DC (CRn==7): data cache, all take Xt.
    if crn == 7 && is_dc_op(op1, crm, op2) {
        return Some((Mnemonic::Dc, true));
    }
    // AT (CRn==7, CRm in {8,9}): address translation, all take Xt.
    if crn == 7 && (crm == 8 || crm == 9) && is_at_op(op1, crm, op2) {
        return Some((Mnemonic::At, true));
    }
    // TLBI (CRn==8): TLB maintenance, all take Xt except the *ALL* forms.
    if crn == 8 && is_tlbi_op(op1, crm, op2) {
        let needs_rt = tlbi_needs_xt(crm, op2);
        return Some((Mnemonic::Tlbi, needs_rt));
    }
    None
}

/// `true` if `(op1, CRm, op2)` is a defined `DC` (data-cache) operation.
#[inline]
const fn is_dc_op(op1: u32, crm: u32, op2: u32) -> bool {
    matches!(
        (op1, crm, op2),
        (0, 6, 1)   // IVAC
        | (0, 6, 2) // ISW
        | (0, 10, 2) // CSW
        | (0, 14, 2) // CISW
        | (3, 4, 1) // ZVA
        | (3, 10, 1) // CVAC
        | (3, 11, 1) // CVAU
        | (3, 12, 1) // CVAP  (FEAT_DPB)
        | (3, 13, 1) // CVADP (FEAT_DPB2)
        | (3, 14, 1) // CIVAC
        // FEAT_MTE tag-cache ops.
        | (0, 6, 3) // IGVAC
        | (0, 6, 4) // IGSW
        | (0, 6, 5) // IGDVAC
        | (0, 6, 6) // IGDSW
        | (0, 10, 4) // CGSW
        | (0, 10, 6) // CGDSW
        | (0, 14, 4) // CIGSW
        | (0, 14, 6) // CIGDSW
        | (3, 10, 3) // CGVAC
        | (3, 10, 5) // CGDVAC
        | (3, 12, 3) // CGVAP
        | (3, 12, 5) // CGDVAP
        | (3, 13, 3) // CGVADP
        | (3, 13, 5) // CGDVADP
        | (3, 14, 3) // CIGVAC
        | (3, 14, 5) // CIGDVAC
    )
}

/// `true` if `(op1, CRm, op2)` is a defined `AT` (address-translation) operation.
#[inline]
const fn is_at_op(op1: u32, crm: u32, op2: u32) -> bool {
    match crm {
        // S1E1R/W, S1E0R/W (op1=0), plus S1E1RP/WP (FEAT_PAN2).
        8 => matches!((op1, op2), (0, 0) | (0, 1) | (0, 2) | (0, 3)
            | (4, 0) | (4, 1) | (4, 2) | (4, 3) | (4, 4) | (4, 5) | (4, 6) | (4, 7)
            | (6, 0) | (6, 1)),
        9 => matches!((op1, op2), (0, 0) | (0, 1) | (4, 0) | (4, 1)),
        _ => false,
    }
}

/// `true` if `(op1, CRm, op2)` is a defined `TLBI` operation (coarse: the
/// architectural TLBI block is `CRn==8`, `CRm` in `{0..7}` for the range/IPAS
/// ops and `{3,7}` for the broadcast/inner-shareable families). We accept the
/// standard non-range encodings and let the rest fall through to `SYS`.
#[inline]
const fn is_tlbi_op(op1: u32, crm: u32, op2: u32) -> bool {
    // The widely-used TLBI ops live at CRm in {3 (IS), 7 (no-IS)} with op2
    // selecting the variant, plus the EL2/EL3 IPA forms at CRm in {0,4}.
    let _ = op1;
    match crm {
        0 | 4 => matches!(op2, 1 | 5),          // IPAS2E1IS/L, etc.
        3 | 7 => op2 <= 7,                       // VMALLE1{IS}, ASIDE1, VAE1, ...
        1 | 5 => matches!(op2, 0 | 1 | 2 | 3 | 5), // range/auxiliary forms
        _ => false,
    }
}

/// Whether a `TLBI` operation takes an `<Xt>` operand. The `*ALL*` / `VMALL*`
/// forms (op2 in {0,4} at CRm in {3,7}) operate on the whole TLB and take no
/// register; everything else is address/ASID-qualified and takes `<Xt>`.
#[inline]
const fn tlbi_needs_xt(crm: u32, op2: u32) -> bool {
    !((crm == 3 || crm == 7) && matches!(op2, 0 | 4))
}

/// The lowercase operation-name token for a named system-instruction alias
/// (`"ialluis"`, `"zva"`, `"rctx"`, `"alle3"`, ...). Keyed on the full
/// `(op1,CRn,CRm,op2)` tuple; defaults to a generic spelling if unmapped (which
/// only happens for tuples `sys_alias` does not recognize, so it is unreachable
/// in practice).
#[inline]
fn alias_op_name(op1: u32, crn: u32, crm: u32, op2: u32) -> &'static str {
    // CFP/CPP/DVP all use the RCTX operand spelling.
    if crn == 7 && crm == 3 {
        return "rctx";
    }
    if crn == 7 && (crm == 1 || crm == 5) {
        return match (op1, crm, op2) {
            (0, 1, 0) => "ialluis",
            (0, 5, 0) => "iallu",
            (3, 5, 1) => "ivau",
            _ => "iallu",
        };
    }
    if crn == 7
        && (crm == 4 || crm == 6 || crm == 10 || crm == 11 || crm == 12 || crm == 13 || crm == 14)
    {
        return dc_op_name(op1, crm, op2);
    }
    if crn == 7 && (crm == 8 || crm == 9) {
        return at_op_name(op1, crm, op2);
    }
    if crn == 8 {
        return tlbi_op_name(op1, crm, op2);
    }
    "rctx"
}

/// Operation-name token for a `DC` alias.
#[inline]
fn dc_op_name(op1: u32, crm: u32, op2: u32) -> &'static str {
    match (op1, crm, op2) {
        (0, 6, 1) => "ivac",
        (0, 6, 2) => "isw",
        (0, 6, 3) => "igvac",
        (0, 6, 4) => "igsw",
        (0, 6, 5) => "igdvac",
        (0, 6, 6) => "igdsw",
        (0, 10, 2) => "csw",
        (0, 10, 4) => "cgsw",
        (0, 10, 6) => "cgdsw",
        (0, 14, 2) => "cisw",
        (0, 14, 4) => "cigsw",
        (0, 14, 6) => "cigdsw",
        (3, 4, 1) => "zva",
        (3, 10, 1) => "cvac",
        (3, 10, 3) => "cgvac",
        (3, 10, 5) => "cgdvac",
        (3, 11, 1) => "cvau",
        (3, 12, 1) => "cvap",
        (3, 12, 3) => "cgvap",
        (3, 12, 5) => "cgdvap",
        (3, 13, 1) => "cvadp",
        (3, 13, 3) => "cgvadp",
        (3, 13, 5) => "cgdvadp",
        (3, 14, 1) => "civac",
        (3, 14, 3) => "cigvac",
        (3, 14, 5) => "cigdvac",
        _ => "zva",
    }
}

/// Operation-name token for an `AT` alias.
#[inline]
fn at_op_name(op1: u32, crm: u32, op2: u32) -> &'static str {
    match (crm, op1, op2) {
        (8, 0, 0) => "s1e1r",
        (8, 0, 1) => "s1e1w",
        (8, 0, 2) => "s1e0r",
        (8, 0, 3) => "s1e0w",
        (9, 0, 0) => "s1e1rp",
        (9, 0, 1) => "s1e1wp",
        (8, 4, 0) => "s1e2r",
        (8, 4, 1) => "s1e2w",
        (8, 4, 4) => "s12e1r",
        (8, 4, 5) => "s12e1w",
        (8, 4, 6) => "s12e0r",
        (8, 4, 7) => "s12e0w",
        (8, 6, 0) => "s1e3r",
        (8, 6, 1) => "s1e3w",
        (9, 4, 0) => "s1e2rp",
        (9, 4, 1) => "s1e2wp",
        _ => "s1e1r",
    }
}

/// Operation-name token for a (non-range) `TLBI` alias.
#[inline]
fn tlbi_op_name(op1: u32, crm: u32, op2: u32) -> &'static str {
    match (op1, crm, op2) {
        // EL1 inner-shareable (CRm==3).
        (0, 3, 0) => "vmalle1is",
        (0, 3, 1) => "vae1is",
        (0, 3, 2) => "aside1is",
        (0, 3, 3) => "vaae1is",
        (0, 3, 5) => "vale1is",
        (0, 3, 7) => "vaale1is",
        // EL1 (CRm==7).
        (0, 7, 0) => "vmalle1",
        (0, 7, 1) => "vae1",
        (0, 7, 2) => "aside1",
        (0, 7, 3) => "vaae1",
        (0, 7, 5) => "vale1",
        (0, 7, 7) => "vaale1",
        // EL2 (op1==4).
        (4, 0, 1) => "ipas2e1is",
        (4, 0, 5) => "ipas2le1is",
        (4, 3, 0) => "alle2is",
        (4, 3, 1) => "vae2is",
        (4, 3, 4) => "alle1is",
        (4, 3, 5) => "vale2is",
        (4, 3, 6) => "vmalls12e1is",
        (4, 4, 1) => "ipas2e1",
        (4, 4, 5) => "ipas2le1",
        (4, 7, 0) => "alle2",
        (4, 7, 1) => "vae2",
        (4, 7, 4) => "alle1",
        (4, 7, 5) => "vale2",
        (4, 7, 6) => "vmalls12e1",
        // EL3 (op1==6).
        (6, 3, 0) => "alle3is",
        (6, 3, 1) => "vae3is",
        (6, 3, 5) => "vale3is",
        (6, 7, 0) => "alle3",
        (6, 7, 1) => "vae3",
        (6, 7, 5) => "vale3",
        _ => "alle1",
    }
}

#[cfg(test)]
mod tests {
    use crate::decode::decode;
    use crate::features::{Feature, FeatureSet};
    use crate::format::{BufSink, FmtFormatter, Formatter};
    use crate::mnemonic::Code;
    use crate::operand::Operand;

    // The harness anchor address the differential corpus uses for PC-relative
    // targets, so the expected labels below match the corpus verbatim.
    const ADDRESS: u64 = 0x8000_0000_0000_0004;

    /// Decode `word` at the corpus anchor, render with the default formatter into
    /// a fixed buffer, and assert the text equals `expected`. Allocation-free so
    /// it builds on the default no-`alloc` test tier.
    #[track_caller]
    fn assert_dis(word: u32, expected: &str) {
        let insn = decode(word, ADDRESS, FeatureSet::ALL);
        let mut buf = [0u8; 128];
        let mut sink = BufSink::new(&mut buf);
        FmtFormatter::new().format(&insn, &mut sink);
        assert!(!sink.overflowed(), "BufSink overflowed rendering {expected:?}");
        assert_eq!(sink.as_str(), expected, "word={word:#010x}");
    }

    #[test]
    fn branch_immediate() {
        // B / BL with the corpus anchor.
        assert_dis(0x14F5E1BA, "b       0x8000000003d786ec");
        assert_dis(0x94A68B21, "bl      0x80000000029a2c88");
    }

    #[test]
    fn conditional_branch_fuses_cond() {
        // B.<cond> renders as a single `b.<cond>` token, then the label.
        assert_dis(0x54156881, "b.ne    0x800000000002ad14");
        assert_dis(0x54FEE2A8, "b.hi    0x7fffffffffffdc58");
        let insn = decode(0x54156881, ADDRESS, FeatureSet::ALL);
        assert_eq!(insn.code(), Code::BCond);
    }

    #[test]
    fn compare_and_test_branch() {
        assert_dis(0x342F64AB, "cbz     w11, 0x800000000005ec98");
        assert_dis(0x358FD614, "cbnz    w20, 0x7ffffffffff1fac4");
        assert_dis(0x36EE4A53, "tbz     w19, #0x1d, 0x7fffffffffffc94c");
        assert_dis(0xB72DA58C, "tbnz    x12, #0x25, 0x7fffffffffffb4b4");
    }

    #[test]
    fn branch_register_and_ret_elision() {
        assert_dis(0xD61F00E0, "br      x7");
        assert_dis(0xD63F0080, "blr     x4");
        // RET x30 elides the register; other Xn is shown.
        assert_dis(0xD65F03C0, "ret");
        assert_dis(0xD65F0220, "ret     x17");
        assert_dis(0xD69F03E0, "eret");
        assert_dis(0xD6BF03E0, "drps");
    }

    #[test]
    fn pauth_branch_register() {
        // BRAA/BLRAA take two registers; BRAAZ/RETAA are the Z/implicit forms.
        assert_dis(0xD71F091C, "braa    x8, x28");
        assert_dis(0xD73F0A0B, "blraa   x16, x11");
        assert_dis(0xD61F0BBF, "braaz   x29");
        assert_dis(0xD65F0BFF, "retaa");
    }

    #[test]
    fn pauth_branch_gated_off() {
        // Without FEAT_PAuth the PAuth branch forms are unallocated.
        let insn = decode(0xD71F091C, ADDRESS, FeatureSet::BASE);
        assert!(insn.is_invalid());
        // Plain BR remains valid under BASE.
        let br = decode(0xD61F00E0, ADDRESS, FeatureSet::BASE);
        assert_eq!(br.code(), Code::Br);
    }

    #[test]
    fn cmpbr_register_and_immediate() {
        // Register form `CB<cc> <Rt>, <Rm>, <label>` (Rt at word<4:0>, Rm at
        // word<20:16>), the word width following sf.
        assert_dis(0x740005A7, "cbgt    w7, w0, 0x80000000000000b8");
        assert_dis(0x74201313, "cbge    w19, w0, 0x8000000000000264");
        assert_dis(0x744007CE, "cbhi    w14, w0, 0x80000000000000fc");
        assert_dis(0x746003FB, "cbhs    w27, w0, 0x8000000000000080");
        assert_dis(0x74C008F8, "cbeq    w24, w0, 0x8000000000000120");
        assert_dis(0x74E00459, "cbne    w25, w0, 0x800000000000008c");
        assert_dis(0xF40005A7, "cbgt    x7, x0, 0x80000000000000b8");
        assert_dis(0xF4E939C5, "cbne    x5, x9, 0x7fffffffffffff3c");
        // Byte (`CBB<cc>`) and halfword (`CBH<cc>`) register forms — W only.
        assert_dis(0x74E0841D, "cbbne   w29, w0, 0x8000000000000084");
        assert_dis(0x74409072, "cbbhi   w18, w0, 0x8000000000000210");
        assert_dis(0x74E0C6BA, "cbhne   w26, w0, 0x80000000000000d8");
        assert_dis(0x74C0C240, "cbheq   w0, w0, 0x800000000000004c");
        assert_dis(0x7420E35D, "cbhge   w29, w0, 0x7ffffffffffffc6c");
        // Immediate-compare form `CB<cc> <Rt>, #imm6, <label>`.
        assert_dis(0x752002C9, "cblt    w9, #0x0, 0x800000000000005c");
        assert_dis(0x75600E4B, "cblo    w11, #0x0, 0x80000000000001cc");
        assert_dis(0x75050672, "cbgt    w18, #0xa, 0x80000000000000d0");
        assert_dis(0xF53F8BFC, "cblt    x28, #0x3f, 0x8000000000000180");
        assert_dis(0xF5E387B7, "cbne    x23, #0x7, 0x80000000000000f8");
    }

    #[test]
    fn cmpbr_code_identity_and_forms() {
        // Shared spellings: the word register form and the immediate form both
        // map onto the same `Code` but carry different operand shapes.
        let reg = decode(0x740005A7, ADDRESS, FeatureSet::ALL);
        assert_eq!(reg.code(), Code::Cbgt);
        assert!(matches!(reg.op(1), Operand::Reg { .. }));
        let imm = decode(0x75050672, ADDRESS, FeatureSet::ALL);
        assert_eq!(imm.code(), Code::Cbgt);
        assert!(matches!(imm.op(1), Operand::ImmUnsigned(10)));
        // Byte/halfword codes are distinct.
        assert_eq!(decode(0x74E0841D, ADDRESS, FeatureSet::ALL).code(), Code::Cbbne);
        assert_eq!(decode(0x74E0C6BA, ADDRESS, FeatureSet::ALL).code(), Code::Cbhne);
        // Immediate-only spellings.
        assert_eq!(decode(0x752002C9, ADDRESS, FeatureSet::ALL).code(), Code::Cblt);
        assert_eq!(decode(0x75600E4B, ADDRESS, FeatureSet::ALL).code(), Code::Cblo);
    }

    #[test]
    fn cmpbr_unallocated() {
        // Register form with the reserved size 01 is unallocated.
        assert!(decode(0x74005000, ADDRESS, FeatureSet::ALL).is_invalid());
        // Reserved conditions 100/101 (register and immediate forms).
        assert!(decode(0x74800000, ADDRESS, FeatureSet::ALL).is_invalid());
        assert!(decode(0x75A00000, ADDRESS, FeatureSet::ALL).is_invalid());
        // Byte/halfword with sf == 1 is illegal.
        assert!(decode(0xF4008000, ADDRESS, FeatureSet::ALL).is_invalid());
        assert!(decode(0xF400C000, ADDRESS, FeatureSet::ALL).is_invalid());
        // Immediate form with word<14> set is unallocated.
        assert!(decode(0x75004000, ADDRESS, FeatureSet::ALL).is_invalid());
    }

    #[test]
    fn cmpbr_gated_off() {
        // Without FEAT_CMPBR every compare-and-branch form is unallocated.
        assert!(decode(0x740005A7, ADDRESS, FeatureSet::BASE).is_invalid());
        assert!(decode(0x752002C9, ADDRESS, FeatureSet::BASE).is_invalid());
        assert!(decode(0x74E0841D, ADDRESS, FeatureSet::BASE).is_invalid());
        // It decodes once the feature is enabled.
        let on = decode(0x740005A7, ADDRESS, FeatureSet::BASE.with(Feature::Cmpbr));
        assert_eq!(on.code(), Code::Cbgt);
    }

    #[test]
    fn exceptions() {
        assert_dis(0xD4016F21, "svc     #0xb79");
        assert_dis(0xD40DF462, "hvc     #0x6fa3");
        assert_dis(0xD419FA83, "smc     #0xcfd4");
        assert_dis(0xD422A5A0, "brk     #0x152d");
        assert_dis(0xD4424C60, "hlt     #0x1263");
        assert_dis(0xD4A9E481, "dcps1   #0x4f24");
    }

    #[test]
    fn hints() {
        assert_dis(0xD503201F, "nop");
        assert_dis(0xD503203F, "yield");
        assert_dis(0xD503205F, "wfe");
        assert_dis(0xD503221F, "esb");
        assert_dis(0xD503223F, "psb     csync");
        assert_dis(0xD503245F, "bti     c");
        assert_dis(0xD503241F, "bti");
        // DGH is not named by the oracle -> generic HINT.
        assert_dis(0xD50320DF, "hint    #0x6");
    }

    #[test]
    fn barriers() {
        assert_dis(0xD5033F9F, "dsb     sy");
        assert_dis(0xD503369F, "dsb     nshst");
        assert_dis(0xD50335BF, "dmb     nshld");
        assert_dis(0xD5033FDF, "isb"); // sy is elided
        assert_dis(0xD50331DF, "isb     #0x1");
        assert_dis(0xD5033E5F, "clrex   #0xe");
        assert_dis(0xD503309F, "ssbb");
    }

    #[test]
    fn dsb_nxs_variants() {
        // FEAT_XS DSB <option>nXS: op2==001, CRm==imm2:10.
        assert_dis(0xD5033E3F, "dsb     synxs");
        assert_dis(0xD503363F, "dsb     nshnxs");
        assert_dis(0xD503323F, "dsb     oshnxs");
        assert_dis(0xD5033A3F, "dsb     ishnxs");
        // Without FEAT_XS the nXS DSB is unallocated.
        let insn = decode(0xD5033E3F, ADDRESS, FeatureSet::BASE);
        assert!(insn.is_invalid());
    }

    #[test]
    fn sb_reserved_crm() {
        // SB: CRm==0 is the architectural barrier; CRm!=0 are SBZ-reserved but
        // LLVM/binja decode the whole range to `sb`.
        assert_dis(0xD50330FF, "sb");
        assert_dis(0xD50331FF, "sb");
        assert_dis(0xD5033FFF, "sb");
        assert_eq!(decode(0xD5033FFF, ADDRESS, FeatureSet::ALL).code(), Code::Sb);
    }

    #[test]
    fn msr_immediate_pstate() {
        assert_dis(0xD50049BF, "msr     spsel, #0x9");
        assert_dis(0xD5034EDF, "msr     daifset, #0xe");
        assert_dis(0xD503455F, "msr     dit, #0x5");
        // CFINV/XAFLAG/AXFLAG are bare; PAN falls back to generic MSR.
        assert_dis(0xD500401F, "cfinv");
        assert_dis(0xD500459F, "msr     s0_0_c4_c5_4, xzr");
    }

    #[test]
    fn msr_mrs_register() {
        assert_dis(0xD51B192B, "msr     s3_3_c1_c9_1, x11");
        assert_dis(0xD539533E, "mrs     x30, s3_1_c5_c3_1");
        // A known sysreg name resolves.
        assert_dis(0xD53B4200, "mrs     x0, nzcv");
    }

    #[test]
    fn msr_mrs_generic_op0_zero() {
        // op0==00 selectors that are not a PSTATE/hint/barrier/WFxT/result
        // encoding decode as a generic MSR/MRS (register) — the large gap LLVM
        // accepts but a curated sysreg list rejected. Rt is the real transfer reg.
        assert_dis(0xD5000064, "msr     s0_0_c0_c0_3, x4");
        assert_dis(0xD52000C2, "mrs     x2, s0_0_c0_c0_6");
        // CRn==5/6/7 etc. (write) and any read selector outside TSTART/TTEST.
        assert_dis(0xD5005007, "msr     s0_0_c5_c0_0, x7");
        assert_dis(0xD5203002, "mrs     x2, s0_0_c3_c0_0");
        // A PSTATE-CRn slot with Rt != XZR is a generic MSR, not MSR(immediate).
        assert_dis(0xD50045A5, "msr     s0_0_c4_c5_5, x5");
        // The op0==00 barrier op2==0 hole is a generic MSR (xzr transfer shown).
        assert_dis(0xD500301F, "msr     s0_0_c3_c0_0, xzr");
    }

    #[test]
    fn d128_mrrs_msrr() {
        // FEAT_D128 128-bit system-register pair move; Rt is an even base, the
        // odd half rendered (xzr when the base is x30). Generic + named sysreg.
        assert_dis(0xD56001CC, "mrrs    x12, x13, s0_0_c0_c1_6");
        assert_dis(0xD540005E, "msrr    s0_0_c0_c0_2, x30, xzr");
        assert_dis(0xD578200C, "mrrs    x12, x13, ttbr0_el1");
        assert_dis(0xD558200C, "msrr    ttbr0_el1, x12, x13");
        // An odd transfer base (incl. 31) is UNALLOCATED.
        assert!(decode(0xD56001CD, ADDRESS, FeatureSet::ALL).is_invalid());
        assert!(decode(0xD540005F, ADDRESS, FeatureSet::ALL).is_invalid());
        // Gated off without FEAT_D128.
        assert!(decode(0xD56001CC, ADDRESS, FeatureSet::BASE).is_invalid());
    }

    #[test]
    fn d128_sysp_tlbip() {
        // Generic SYSP; the pair is elided when Rt==11111 (no-transfer).
        assert_dis(0xD5480036, "sysp    #0x0, c0, c0, #0x1, x22, x23");
        assert_dis(0xD548003F, "sysp    #0x0, c0, c0, #0x1");
        assert_dis(0xD5480020, "sysp    #0x0, c0, c0, #0x1, x0, x1");
        // TLBIP alias (CRn==8): always prints the <Xt>:<Xt+1> pair, incl. xzr,xzr.
        assert_dis(0xD54C8020, "tlbip   ipas2e1is, x0, x1");
        assert_dis(0xD54C8300, "tlbip   alle2is, x0, x1");
        assert_dis(0xD54C803F, "tlbip   ipas2e1is, xzr, xzr");
        // An odd transfer base (other than the XZR sentinel) is UNALLOCATED.
        assert!(decode(0xD548003D, ADDRESS, FeatureSet::ALL).is_invalid());
        // Gated off without FEAT_D128.
        assert!(decode(0xD5480036, ADDRESS, FeatureSet::BASE).is_invalid());
    }

    #[test]
    fn sys_and_aliases() {
        assert_dis(0xD50D3428, "sys     #0x5, c3, c4, #0x1, x8");
        assert_dis(0xD52F54FF, "sysl    xzr, #0x7, c5, c4, #0x7");
        assert_dis(0xD5087115, "ic      ialluis");
        assert_dis(0xD508750E, "ic      iallu");
        assert_dis(0xD50B752A, "ic      ivau, x10");
        assert_dis(0xD50B7438, "dc      zva, x24");
        assert_dis(0xD50B7B24, "dc      cvau, x4");
        assert_dis(0xD508792A, "at      s1e1wp, x10");
        assert_dis(0xD50E871F, "tlbi    alle3");
        assert_dis(0xD50B738C, "cfp     rctx, x12");
        assert_dis(0xD50B73F0, "cpp     rctx, x16");
        assert_dis(0xD50B73A6, "dvp     rctx, x6");
    }

    #[test]
    fn wfxt_and_tme_and_tsb() {
        assert_dis(0xD503100F, "wfet    x15");
        assert_dis(0xD503103E, "wfit    x30");
        assert_dis(0xD503307F, "tcommit");
        assert_dis(0xD503225F, "tsb     csync");
        assert_dis(0xD478DA60, "tcancel #0xc6d3");
        // TSTART/TTEST: FEAT_TME "system instruction with result" (read side).
        assert_dis(0xD5233070, "tstart  x16");
        assert_dis(0xD5233178, "ttest   x24");
    }

    #[test]
    fn udf_reserved() {
        // UDF lives in the op0==0000 reserved group, handled by decode_reserved.
        assert_dis(0x00004ABD, "udf     #0x4abd");
        // A non-UDF reserved word stays invalid.
        let insn = decode(0x0001_0000, ADDRESS, FeatureSet::ALL);
        assert!(insn.is_invalid());
    }

    #[test]
    fn never_panics_on_system_space() {
        // Sweep the whole D5xx / D4xx / D6xx system-and-branch space; the decoder
        // must be total (no panic) and never desync the operand count.
        for hi in [0xD4u32, 0xD5, 0xD6] {
            for lo in 0..=0xffffu32 {
                let word = (hi << 24) | lo;
                let _ = decode(word, ADDRESS, FeatureSet::ALL);
                let _ = decode(word, ADDRESS, FeatureSet::BASE);
            }
        }
    }
}
