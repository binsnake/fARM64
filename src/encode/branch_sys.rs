//! Encoder for Branches, Exception generating and System instructions — the
//! exact inverse of [`crate::decode::branch_sys`] (plus the `UDF` reserved path
//! from [`crate::decode`]).
//!
//! Dispatches on [`Instruction::code`] (the canonical encoding identity), then
//! branches on [`Instruction::mnemonic`] / the operand list to recover the
//! fields the decoder produced (the preferred-disassembly alias, if any) and
//! packs the raw bitfields in reverse. It reconstructs the word purely from the
//! instruction's semantics — it never reads [`Instruction::word`].
//!
//! ## Sub-classes covered
//!
//! Every sub-class of the decoder is inverted here: conditional branch
//! (`B.cond`/`BC.cond`, recovering `imm19` from the absolute [`Operand::Label`]
//! and [`Instruction::ip`]); unconditional branch immediate (`B`/`BL`,
//! `imm26`); compare-and-branch (`CBZ`/`CBNZ`) and test-and-branch
//! (`TBZ`/`TBNZ`, the split `b5:b40` bit position); unconditional branch
//! register (`BR`/`BLR`/`RET`+default x30/`ERET`/`DRPS` and the FEAT_PAuth
//! `BRAA`.. family); exception generation (`SVC`/`HVC`/`SMC`/`BRK`/`HLT`/
//! `DCPS{1,2,3}`/`TCANCEL`); the system block — the `HINT` space
//! (`NOP`/`YIELD`/.../`BTI`/PAuth hints), `WFET`/`WFIT`, barriers
//! (`CLREX`/`DSB`/`DMB`/`ISB`/`SB`/`SSBB`/`PSSBB`/`TCOMMIT`/`TSB`), `MSR`
//! (immediate) PSTATE (incl. `CFINV`/`XAFLAG`/`AXFLAG` and `SMSTART`/`SMSTOP`),
//! `MSR`/`MRS` (register), `SYS`/`SYSL` with the `IC`/`DC`/`AT`/`TLBI`/`CFP`/
//! `CPP`/`DVP` aliases, and `TSTART`/`TTEST`; and the reserved `UDF`.
//!
//! ## Documented exact-word losses
//!
//! A handful of decoded forms discard raw bits the architecture treats as
//! don't-care, so the re-encoded word can differ while the *semantics* are
//! preserved (the re-encode decodes back to the identical [`Instruction`]):
//!
//! * `TLBI`/`IC` whole-TLB/`IALLU*` forms: the decoder drops the `Rt` field
//!   (they operate on the whole structure), so we re-emit the canonical
//!   `Rt == 0b11111`. A corpus word with a different `Rt` is irrecoverable from
//!   semantics, but the re-encode is the same instruction.
//!
//! Everything else is a bit-exact round-trip.

use crate::encode::EncodeError;
use crate::instruction::Instruction;
use crate::mnemonic::{Code, Mnemonic};
use crate::operand::Operand;

type R = Result<u32, EncodeError>;

/// Encode a Branch / Exception / System instruction (or `UDF`).
#[inline]
pub fn encode(insn: &Instruction) -> R {
    use Code::*;
    match insn.code() {
        // Conditional branch (immediate): B.cond and FEAT_HBC BC.cond.
        BCond | BcCond => enc_cond_branch(insn),
        // FEAT_PAuth_LR PC-relative authenticate/return branches.
        Retaasppc | Retabsppc | Autiasppc | Autibsppc => enc_pauth_lr_sppc(insn),
        // Unconditional branch (immediate).
        BUncond | BlImm => enc_uncond_branch_imm(insn),
        // Compare-and-branch.
        Cbz32 | Cbz64 | Cbnz32 | Cbnz64 => enc_compare_branch(insn),
        // FEAT_CMPBR compare-and-branch (register / immediate).
        Cbgt | Cbge | Cbhi | Cbhs | Cbeq | Cbne | Cblt | Cblo | Cbbgt | Cbbge | Cbbhi | Cbbhs
        | Cbbeq | Cbbne | Cbhgt | Cbhge | Cbhhi | Cbhhs | Cbheq | Cbhne => enc_cmpbr(insn),
        // Test-and-branch.
        Tbz | Tbnz => enc_test_branch(insn),
        // Unconditional branch (register).
        Br | Blr | Ret | Eret | Drps | Braaz | Brabz | Blraaz | Blrabz | Braa | Brab | Blraa
        | Blrab | Retaa | Retab | Eretaa | Eretab | Retaasppcr | Retabsppcr => enc_branch_reg(insn),
        // Exception generation (incl. the FEAT_GCS TENTER).
        Svc | Hvc | Smc | Brk | Hlt | Tcancel | Tenter | Dcps1 | Dcps2 | Dcps3 => {
            enc_exception(insn)
        }
        // System: hints (incl. GCSB/SHUH/STSHH/STCPH/CHKFEAT/DGH/CLRBHB/PACM).
        Nop | Yield | Wfe | Wfi | Sev | Sevl | Esb | Psb | Csdb | Bti | Tsb | HintGeneric
        | Gcsb | Shuh | Stshh | Stcph | Chkfeat | Dgh | Clrbhb | Pacm => enc_hint(insn),
        // System: GCS stack-maintenance ops (SYS/SYSL forms).
        Gcspushm | Gcspopm | Gcsss1 | Gcsss2 | Gcspushx | Gcspopx | Gcspopcx => enc_gcs_sys(insn),
        // System: WFET/WFIT.
        Wfet | Wfit => enc_wfxt(insn),
        // System: barriers.
        Clrex | Dmb | Dsb | Isb | Sb | Tcommit => enc_barrier(insn),
        // System: MSR (immediate) PSTATE and the bare PSTATE ops.
        MsrImm | Cfinv | Xaflag | Axflag | Smstart | Smstop => enc_msr_imm(insn),
        // System: MSR/MRS (register).
        MsrReg | Mrs => enc_sysreg_move(insn),
        // System: SYS/SYSL and the IC/DC/AT/TLBI/CFP/CPP/DVP aliases.
        Sys | Sysl => enc_sys(insn),
        // System: TSTART/TTEST (system instruction with result).
        Tstart | Ttest => enc_systemresult(insn),
        // System: FEAT_D128 MRRS/MSRR (128-bit system-register pair move).
        Mrrs | Msrr => enc_sysreg_pair(insn),
        // System: FEAT_D128 SYSP / TLBIP (system pair).
        Sysp => enc_sysp(insn),
        // K4: TCHANGE translation-table change (register / immediate).
        TchangefReg | TchangebReg | TchangefImm | TchangebImm => enc_tchange(insn),
        // Reserved: UDF.
        Udf => enc_udf(insn),
        _ => Err(EncodeError::Unsupported),
    }
}

/// `TCHANGE{F,B} <Xt>, <Xn>` / `TCHANGE{F,B} <Xt>, #<imm>`. Inverse of
/// `decode_tchange`. Base `word<31:22> == 1101010110`, `word<16> == 0`,
/// `CRn == 0`; `word<18>` = forward(0)/backward(1); `word<20:19>` = register(00)
/// / immediate(10).
fn enc_tchange(insn: &Instruction) -> R {
    use Code::*;
    let base = 0b11_0101_0110_u32 << 22; // word<31:22> = 1101010110
    let rt = reg_num(insn, 0)?;
    let backward = matches!(insn.code(), TchangebReg | TchangebImm);
    let b18 = if backward { 1u32 } else { 0 };
    let is_reg = matches!(insn.code(), TchangefReg | TchangebReg);
    let mut word = if is_reg {
        let xn = reg_num(insn, 1)?;
        // op0 (word<20:19>) == 00 for the register form.
        base | (b18 << 18) | (xn << 5) | rt
    } else {
        // immediate form: imm7 = word<11:5>.
        let imm = imm_u(insn, 1)?;
        if imm > 0x7f {
            return Err(EncodeError::InvalidImmediate);
        }
        base | (0b10 << 19) | (b18 << 18) | ((imm as u32) << 5) | rt
    };
    // Optional trailing `, nb` (no-barrier) modifier sets word<17>.
    if let Operand::SysOp(tok) = insn.op(2) {
        if tok.name() == "nb" {
            word |= 1 << 17;
        } else {
            return Err(EncodeError::InvalidOperand);
        }
    }
    Ok(word)
}

// ---------------------------------------------------------------------------
// Small field/operand helpers.
// ---------------------------------------------------------------------------

/// The 5-bit register number of operand `n`, or an error if it is not a plain
/// register. SP-vs-ZR is irrelevant for the *number* (both are 31).
#[inline]
fn reg_num(insn: &Instruction, n: usize) -> Result<u32, EncodeError> {
    match insn.op(n) {
        Operand::Reg { reg, .. } => Ok(reg.number() as u32),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// The unsigned immediate value of an immediate operand `n`.
#[inline]
fn imm_u(insn: &Instruction, n: usize) -> Result<u64, EncodeError> {
    match insn.op(n) {
        Operand::ImmUnsigned(v) | Operand::ImmLogical(v) => Ok(v),
        Operand::ImmSigned(v) => Ok(v as u64),
        Operand::ShiftAmount(v) => Ok(v as u64),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// The absolute target of a [`Operand::Label`] operand `n`.
#[inline]
fn label(insn: &Instruction, n: usize) -> Result<u64, EncodeError> {
    match insn.op(n) {
        Operand::Label(v) => Ok(v),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// Recover a signed PC-relative offset from an absolute label `target` and the
/// instruction's `ip`, requiring `offset = imm << shift` to fit a signed
/// `bits`-wide field and be `2^shift`-aligned. Returns the raw `imm` field.
#[inline]
fn rel_imm(target: u64, ip: u64, bits: u32, shift: u32) -> Result<u32, EncodeError> {
    let off = target.wrapping_sub(ip) as i64;
    // Must be aligned to the instruction granule.
    if off & ((1i64 << shift) - 1) != 0 {
        return Err(EncodeError::InvalidImmediate);
    }
    let imm = off >> shift;
    // Must fit a signed `bits`-wide field.
    let hi = imm >> (bits - 1);
    if hi != 0 && hi != -1 {
        return Err(EncodeError::InvalidImmediate);
    }
    Ok((imm as u32) & ((1u32 << bits) - 1))
}

/// The name of a [`Operand::SysOp`] keyword operand `n`, or an error.
#[inline]
fn sysop_name(insn: &Instruction, n: usize) -> Result<&'static str, EncodeError> {
    match insn.op(n) {
        Operand::SysOp(tok) => Ok(tok.name()),
        _ => Err(EncodeError::InvalidOperand),
    }
}

// ---------------------------------------------------------------------------
// Conditional branch (immediate): B.cond / BC.cond.
// ---------------------------------------------------------------------------

/// `B.<cond>` / `BC.<cond>` — `0101_0100 imm19 o0 cond`. Operands: `[Cond,
/// Label]`. `o0` (word<4>) selects the FEAT_HBC hinted `BC.<cond>` ([`Code::BcCond`],
/// o0 == 1) over the ordinary `B.<cond>` ([`Code::BCond`], o0 == 0); each round-trips
/// to its own encoding bit-exactly.
fn enc_cond_branch(insn: &Instruction) -> R {
    let cond = match insn.op(0) {
        Operand::Cond(c) => c,
        _ => return Err(EncodeError::InvalidOperand),
    };
    let imm19 = rel_imm(label(insn, 1)?, insn.ip(), 19, 2)?;
    let o0 = if insn.code() == Code::BcCond { 1u32 } else { 0 };
    let word = (0b0101_0100u32 << 24) | (imm19 << 5) | (o0 << 4) | (cond.as_u4() as u32);
    Ok(word)
}

/// `RETAASPPC`/`RETABSPPC`/`AUTIASPPC`/`AUTIBSPPC <label>` (FEAT_PAuth_LR) — the
/// PC-relative authenticate/return branch forms. Operand 0 is the
/// [`Operand::Label`]; the 16-bit `imm16` (word<20:5>) is the *negated* offset
/// (`target = ip - (imm16 << 2)`, so the target must lie at or before `ip`).
/// The base word and `M` (word<21>) key bit come from the [`Code`]:
///
/// * `RETAASPPC`: `0101010 1 00 0 imm16 11111`
/// * `RETABSPPC`: `0101010 1 00 1 imm16 11111`
/// * `AUTIASPPC`: `1111001110 0 imm16 11111`
/// * `AUTIBSPPC`: `1111001110 1 imm16 11111`
fn enc_pauth_lr_sppc(insn: &Instruction) -> R {
    use Code::*;
    // (base word with M==0, Rd/Rn fixed at 11111; M is OR-ed in below.) The base
    // is `word<31:22> << 22` | Rt(11111); `word<23:22> == 00` for the RET forms.
    const RET_BASE: u32 = (0b01_0101_0100u32 << 22) | 0b11111; // 0101010 1 00 ...
    const AUTI_BASE: u32 = (0b11_1100_1110u32 << 22) | 0b11111; // 1111001110 ...
    let (base, m): (u32, u32) = match insn.code() {
        Retaasppc => (RET_BASE, 0),
        Retabsppc => (RET_BASE, 1),
        Autiasppc => (AUTI_BASE, 0),
        Autibsppc => (AUTI_BASE, 1),
        _ => return Err(EncodeError::Unsupported),
    };
    // The target is `ip - (imm16 << 2)`, so the *backward distance* `ip - target`
    // must equal `imm16 << 2`: 4-byte aligned and within `2^16 * 4`. Work on the
    // unsigned distance to stay panic-free (no signed negation overflow).
    let target = label(insn, 0)?;
    let dist = insn.ip().wrapping_sub(target);
    if dist & 0b11 != 0 || (dist >> 2) > 0xFFFF {
        return Err(EncodeError::InvalidImmediate);
    }
    let imm16 = (dist >> 2) as u32;
    Ok(base | (m << 21) | (imm16 << 5))
}

// ---------------------------------------------------------------------------
// Unconditional branch (immediate): B / BL.
// ---------------------------------------------------------------------------

/// `B`/`BL <label>` — `op 00101 imm26` with `op == 0` for `B`, `1` for `BL`.
fn enc_uncond_branch_imm(insn: &Instruction) -> R {
    let op = if insn.code() == Code::BlImm { 1u32 } else { 0 };
    let imm26 = rel_imm(label(insn, 0)?, insn.ip(), 26, 2)?;
    let word = (op << 31) | (0b00101 << 26) | imm26;
    Ok(word)
}

// ---------------------------------------------------------------------------
// Compare and branch / Test and branch (immediate).
// ---------------------------------------------------------------------------

/// `CBZ`/`CBNZ <Wt|Xt>, <label>` — `sf 011010 op imm19 Rt`.
fn enc_compare_branch(insn: &Instruction) -> R {
    use Code::*;
    let (sf, op) = match insn.code() {
        Cbz32 => (0u32, 0u32),
        Cbz64 => (1, 0),
        Cbnz32 => (0, 1),
        _ => (1, 1), // Cbnz64
    };
    let rt = reg_num(insn, 0)?;
    let imm19 = rel_imm(label(insn, 1)?, insn.ip(), 19, 2)?;
    let word = (sf << 31) | (0b011010 << 25) | (op << 24) | (imm19 << 5) | rt;
    Ok(word)
}

/// `TBZ`/`TBNZ <R><t>, #imm, <label>` — `b5 011011 op b40 imm14 Rt`, the bit
/// position split across `b5` (word<31>) and `b40` (word<23:19>).
fn enc_test_branch(insn: &Instruction) -> R {
    let op = if insn.code() == Code::Tbnz { 1u32 } else { 0 };
    let rt = reg_num(insn, 0)?;
    let bitpos = imm_u(insn, 1)? as u32;
    if bitpos > 63 {
        return Err(EncodeError::InvalidImmediate);
    }
    let b5 = (bitpos >> 5) & 0x1;
    let b40 = bitpos & 0x1f;
    let imm14 = rel_imm(label(insn, 2)?, insn.ip(), 14, 2)?;
    let word = (b5 << 31) | (0b011011 << 25) | (op << 24) | (b40 << 19) | (imm14 << 5) | rt;
    Ok(word)
}

// ---------------------------------------------------------------------------
// FEAT_CMPBR compare-and-branch (register / immediate).
// ---------------------------------------------------------------------------

/// `true` if the register operand at slot `n` is a 64-bit `X` register.
#[inline]
fn is_x_reg(insn: &Instruction, n: usize) -> Result<bool, EncodeError> {
    match insn.op(n) {
        Operand::Reg { reg, .. } => Ok(reg.width_bits() == 64),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// FEAT_CMPBR compare-and-branch — the inverse of [`crate::decode::branch_sys`]'s
/// `decode_cmpbr`. Packs `sf 1110 10 op2 imm/Rm size imm9 Rt` where the form is
/// recovered from operand 1: a register (`CB<cc> <Rm>, <Rn>, <label>`,
/// `bit24 == 0`) or an immediate (`CB<cc> <Rn>, #imm6, <label>`, `bit24 == 1`).
///
/// `Code` carries the condition and the size family (word / `CBB` byte / `CBH`
/// halfword); `sf` (the W/X width) comes from the register operands, and the
/// branch `imm9` from the [`Operand::Label`].
fn enc_cmpbr(insn: &Instruction) -> R {
    use Code::*;

    // Per-Code facets: the size field, the register-form condition (`None` if the
    // spelling has no register form), and the immediate-form condition (`None` if
    // no immediate form). Word spellings share a condition across both forms.
    let (size, reg_cc, imm_cc): (u32, Option<u32>, Option<u32>) = match insn.code() {
        // Word family (size == 00): byte/half are size 10/11.
        Cbgt => (0b00, Some(0b000), Some(0b000)),
        Cbge => (0b00, Some(0b001), None),
        Cbhi => (0b00, Some(0b010), Some(0b010)),
        Cbhs => (0b00, Some(0b011), None),
        Cbeq => (0b00, Some(0b110), Some(0b110)),
        Cbne => (0b00, Some(0b111), Some(0b111)),
        Cblt => (0b00, None, Some(0b001)),
        Cblo => (0b00, None, Some(0b011)),
        // Byte register family (size == 10).
        Cbbgt => (0b10, Some(0b000), None),
        Cbbge => (0b10, Some(0b001), None),
        Cbbhi => (0b10, Some(0b010), None),
        Cbbhs => (0b10, Some(0b011), None),
        Cbbeq => (0b10, Some(0b110), None),
        Cbbne => (0b10, Some(0b111), None),
        // Halfword register family (size == 11).
        Cbhgt => (0b11, Some(0b000), None),
        Cbhge => (0b11, Some(0b001), None),
        Cbhhi => (0b11, Some(0b010), None),
        Cbhhs => (0b11, Some(0b011), None),
        Cbheq => (0b11, Some(0b110), None),
        Cbhne => (0b11, Some(0b111), None),
        _ => return Err(EncodeError::Unsupported),
    };

    let imm9 = rel_imm(label(insn, 2)?, insn.ip(), 9, 2)?;

    // Operand 1 selects the form: a register -> register form; an immediate ->
    // immediate-compare form.
    match insn.op(1) {
        Operand::Reg { .. } => {
            let cc = reg_cc.ok_or(EncodeError::InvalidOperand)?;
            // Byte/half forms are W-only (sf == 0); the word form follows the
            // register width. Both register operands must agree in width.
            let sf = if size == 0b00 {
                u32::from(is_x_reg(insn, 0)?)
            } else {
                if is_x_reg(insn, 0)? || is_x_reg(insn, 1)? {
                    return Err(EncodeError::InvalidOperand);
                }
                0
            };
            // Operand order is `<Rt>, <Rm>, <label>`: Rt at word<4:0>, Rm at
            // word<20:16>.
            let rt = reg_num(insn, 0)?;
            let rm = reg_num(insn, 1)?;
            let word = (sf << 31)
                | (0b111010 << 25)
                | (cc << 21)
                | (rm << 16)
                | (size << 14)
                | (imm9 << 5)
                | rt;
            Ok(word)
        }
        Operand::ImmUnsigned(_) | Operand::ImmSigned(_) | Operand::ImmLogical(_) => {
            // Immediate form: size field is always 00.
            let cc = imm_cc.ok_or(EncodeError::InvalidOperand)?;
            let sf = u32::from(is_x_reg(insn, 0)?);
            let rt = reg_num(insn, 0)?;
            let imm6 = imm_u(insn, 1)?;
            if imm6 > 0x3f {
                return Err(EncodeError::InvalidImmediate);
            }
            let word = (sf << 31)
                | (0b111010 << 25)
                | (1 << 24)
                | (cc << 21)
                | ((imm6 as u32) << 15)
                | (imm9 << 5)
                | rt;
            Ok(word)
        }
        _ => Err(EncodeError::InvalidOperand),
    }
}

// ---------------------------------------------------------------------------
// Unconditional branch (register).
// ---------------------------------------------------------------------------

/// `BR`/`BLR`/`RET`/`ERET`/`DRPS` and the FEAT_PAuth `BRAA`.. family —
/// `1101011 opc 11111 op3 Rn op4`.
fn enc_branch_reg(insn: &Instruction) -> R {
    use Code::*;
    let code = insn.code();

    // (opc, op3, rn, op4) per encoding.
    let (opc, op3, rn, op4): (u32, u32, u32, u32) = match code {
        // Plain forms: op3 == 000000, op4 == 00000.
        Br => (0b0000, 0b000000, reg_num(insn, 0)?, 0),
        Blr => (0b0001, 0b000000, reg_num(insn, 0)?, 0),
        // RET {<Xn>}: default x30 when the register is elided.
        Ret => {
            let rn = if insn.op_count() >= 1 {
                reg_num(insn, 0)?
            } else {
                30
            };
            (0b0010, 0b000000, rn, 0)
        }
        Eret => (0b0100, 0b000000, 0b11111, 0),
        Drps => (0b0101, 0b000000, 0b11111, 0),
        // Z forms: op3 == 00001x, op4 == 11111, Rn from operand 0.
        Braaz => (0b0000, 0b000010, reg_num(insn, 0)?, 0b11111),
        Brabz => (0b0000, 0b000011, reg_num(insn, 0)?, 0b11111),
        Blraaz => (0b0001, 0b000010, reg_num(insn, 0)?, 0b11111),
        Blrabz => (0b0001, 0b000011, reg_num(insn, 0)?, 0b11111),
        // RETAA/RETAB/ERETAA/ERETAB: Rn == 11111, Rm(op4) == 11111.
        Retaa => (0b0010, 0b000010, 0b11111, 0b11111),
        Retab => (0b0010, 0b000011, 0b11111, 0b11111),
        // RETAASPPCR/RETABSPPCR <Xm> (FEAT_PAuth_LR): op4 holds the modifier Xm.
        Retaasppcr => (0b0010, 0b000010, 0b11111, reg_num(insn, 0)?),
        Retabsppcr => (0b0010, 0b000011, 0b11111, reg_num(insn, 0)?),
        Eretaa => (0b0100, 0b000010, 0b11111, 0b11111),
        Eretab => (0b0100, 0b000011, 0b11111, 0b11111),
        // BRAA/BRAB/BLRAA/BLRAB <Xn>, <Xm|SP>: op4 holds Xm.
        Braa => (0b1000, 0b000010, reg_num(insn, 0)?, reg_num(insn, 1)?),
        Brab => (0b1000, 0b000011, reg_num(insn, 0)?, reg_num(insn, 1)?),
        Blraa => (0b1001, 0b000010, reg_num(insn, 0)?, reg_num(insn, 1)?),
        Blrab => (0b1001, 0b000011, reg_num(insn, 0)?, reg_num(insn, 1)?),
        _ => return Err(EncodeError::Unsupported),
    };

    let word =
        (0b1101011u32 << 25) | (opc << 21) | (0b11111 << 16) | (op3 << 10) | (rn << 5) | op4;
    Ok(word)
}

// ---------------------------------------------------------------------------
// Exception generation.
// ---------------------------------------------------------------------------

/// `SVC`/`HVC`/`SMC`/`BRK`/`HLT`/`TCANCEL`/`DCPS{1,2,3}` —
/// `11010100 opc imm16 000 LL`. `DCPS*` carry an optional (elided-when-zero)
/// immediate.
fn enc_exception(insn: &Instruction) -> R {
    use Code::*;
    let code = insn.code();
    let (opc, ll): (u32, u32) = match code {
        Svc => (0b000, 0b01),
        Hvc => (0b000, 0b10),
        Smc => (0b000, 0b11),
        Brk => (0b001, 0b00),
        Hlt => (0b010, 0b00),
        Tcancel => (0b011, 0b00),
        Tenter => (0b111, 0b00),
        Dcps1 => (0b101, 0b01),
        Dcps2 => (0b101, 0b10),
        _ => (0b101, 0b11), // Dcps3
    };

    // DCPS* take an optional immediate (elided when zero); the rest require one.
    let imm16 = if matches!(code, Dcps1 | Dcps2 | Dcps3) && insn.op_count() == 0 {
        0
    } else if code == Tenter {
        // TENTER carries imm16<6:0> as its immediate and imm16<12> as the optional
        // `nb` modifier (a trailing SysOp operand); the other bits are RES0.
        (imm_u(insn, 0)? & 0x7f) | if insn.op_count() > 1 { 0x1000 } else { 0 }
    } else {
        imm_u(insn, 0)?
    };
    if imm16 > 0xffff {
        return Err(EncodeError::InvalidImmediate);
    }

    let word = (0b1101_0100u32 << 24) | (opc << 21) | ((imm16 as u32) << 5) | ll;
    Ok(word)
}

// ---------------------------------------------------------------------------
// System: HINT space.
// ---------------------------------------------------------------------------

/// The `HINT` space (`CRn==2`): rebuild the 7-bit `CRm:op2` selector from the
/// (named) code/mnemonic, then pack `1101010100 0 00 011 0010 CRm op2 11111`.
fn enc_hint(insn: &Instruction) -> R {
    use Code::*;
    let sel: u32 = match insn.code() {
        Nop => 0,
        Yield => 1,
        Wfe => 2,
        Wfi => 3,
        Sev => 4,
        Sevl => 5,
        Dgh => 6,
        Esb => 16,
        Psb => 17,  // PSB CSYNC
        Tsb => 18,  // TSB CSYNC
        Gcsb => 19, // GCSB DSYNC
        Csdb => 20,
        Clrbhb => 22,
        Pacm => 39,
        Chkfeat => 40, // CHKFEAT X16
        Shuh => {
            // SHUH (CRm==6): op2==2 (bare) or op2==3 (PH).
            match insn.op(0) {
                Operand::None => (0b0110 << 3) | 2,
                Operand::SysOp(tok) if tok.name() == "ph" => (0b0110 << 3) | 3,
                _ => return Err(EncodeError::InvalidOperand),
            }
        }
        Stcph => (0b0110 << 3) | 4, // STCPH (CRm==6, op2==4)
        Stshh => {
            // STSHH (CRm==6): keep/strm (op2 0/1) or a numeric #imm (op2 5..7).
            let op2 = match insn.op(0) {
                Operand::SysOp(tok) => match tok.name() {
                    "keep" => 0u32,
                    "strm" => 1,
                    _ => return Err(EncodeError::InvalidOperand),
                },
                Operand::ImmUnsigned(v) => v as u32 & 0x7,
                _ => return Err(EncodeError::InvalidOperand),
            };
            (0b0110 << 3) | op2
        }
        Bti => {
            // BTI {<targets>}: CRm==4, op2 in {0,2,4,6}.
            let op2 = match insn.op(0) {
                Operand::None => 0u32,
                Operand::SysOp(tok) => match tok.name() {
                    "c" => 0b010,
                    "j" => 0b100,
                    "jc" => 0b110,
                    _ => return Err(EncodeError::InvalidOperand),
                },
                _ => return Err(EncodeError::InvalidOperand),
            };
            (0b0100 << 3) | op2
        }
        // HintGeneric carries either a PAuth hint mnemonic (no operand) or a
        // generic `HINT #imm` whose immediate is the selector.
        HintGeneric => match insn.mnemonic() {
            Mnemonic::Xpaclri => 7,
            Mnemonic::Pacia1716 => 8,
            Mnemonic::Pacib1716 => 10,
            Mnemonic::Autia1716 => 12,
            Mnemonic::Autib1716 => 14,
            Mnemonic::Paciaz => 24,
            Mnemonic::Paciasp => 25,
            Mnemonic::Pacibz => 26,
            Mnemonic::Pacibsp => 27,
            Mnemonic::Autiaz => 28,
            Mnemonic::Autiasp => 29,
            Mnemonic::Autibz => 30,
            Mnemonic::Autibsp => 31,
            // PACM (HINT #39).
            Mnemonic::Pacm => 39,
            // Generic `HINT #imm`.
            Mnemonic::Hint => imm_u(insn, 0)? as u32,
            _ => return Err(EncodeError::InvalidOperand),
        },
        _ => return Err(EncodeError::Unsupported),
    };
    if sel > 0x7f {
        return Err(EncodeError::InvalidImmediate);
    }
    let crm = (sel >> 3) & 0xf;
    let op2 = sel & 0x7;
    Ok(system_word(0, 0b00, 0b011, 0b0010, crm, op2, 0b11111))
}

// ---------------------------------------------------------------------------
// System: WFET / WFIT.
// ---------------------------------------------------------------------------

/// `WFET`/`WFIT <Xt>` — `CRn==1, CRm==0, op1==011`, `op2` selects, `Rt` is Xt.
fn enc_wfxt(insn: &Instruction) -> R {
    let op2 = if insn.code() == Code::Wfit { 0b001 } else { 0b000 };
    let rt = reg_num(insn, 0)?;
    Ok(system_word(0, 0b00, 0b011, 0b0001, 0b0000, op2, rt))
}

// ---------------------------------------------------------------------------
// System: barriers (CRn==3).
// ---------------------------------------------------------------------------

/// Barriers (`CRn==3`): `CLREX`/`DSB`/`DMB`/`ISB`/`SB`/`SSBB`/`PSSBB`/`TCOMMIT`.
fn enc_barrier(insn: &Instruction) -> R {
    use Code::*;
    let (crm, op2): (u32, u32) = match insn.code() {
        Clrex => {
            // CLREX {#imm}; elided imm is the default 0xf.
            let crm = if insn.op_count() == 0 {
                0b1111
            } else {
                imm_u(insn, 0)? as u32
            };
            (crm & 0xf, 0b010)
        }
        Tcommit => (0b0000, 0b011),
        Dsb => {
            // DSB has the SSBB/PSSBB alias spellings (CRm 0/4), the FEAT_XS
            // `<option>nXS` variants (op2==001, CRm==imm2:10), plus the normal
            // option/imm form (op2==100).
            if let Operand::SysOp(tok) = insn.op(0) {
                if let Some(imm2) = dsb_nxs_imm2(tok.name()) {
                    // CRm = imm2:10.
                    return Ok(system_word(0, 0b00, 0b011, 0b0011, (imm2 << 2) | 0b10, 0b001, 0b11111));
                }
            }
            let crm = match insn.mnemonic() {
                Mnemonic::Ssbb => 0b0000,
                Mnemonic::Pssbb => 0b0100,
                _ => barrier_crm(insn)?,
            };
            (crm, 0b100)
        }
        Dmb => (barrier_crm(insn)?, 0b101),
        Isb => {
            // ISB {<option>|#imm}: sy (CRm==1111) is elided.
            let crm = if insn.op_count() == 0 {
                0b1111
            } else {
                imm_u(insn, 0)? as u32
            };
            (crm & 0xf, 0b110)
        }
        Sb => (0b0000, 0b111),
        _ => return Err(EncodeError::Unsupported),
    };
    Ok(system_word(0, 0b00, 0b011, 0b0011, crm, op2, 0b11111))
}

/// Recover the `CRm` of a `DSB`/`DMB` from its option keyword or numeric `#imm`.
fn barrier_crm(insn: &Instruction) -> Result<u32, EncodeError> {
    match insn.op(0) {
        Operand::SysOp(tok) => barrier_option_crm(tok.name()).ok_or(EncodeError::InvalidOperand),
        Operand::ImmUnsigned(v) | Operand::ImmLogical(v) => Ok((v as u32) & 0xf),
        Operand::ImmSigned(v) => Ok((v as u32) & 0xf),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// The `imm2` (CRm<3:2>) for a `DSB <option>nXS` keyword (FEAT_XS), or `None`.
/// Inverse of the decoder's nXS option table.
#[inline]
fn dsb_nxs_imm2(name: &str) -> Option<u32> {
    let imm2 = match name {
        "oshnxs" => 0b00,
        "nshnxs" => 0b01,
        "ishnxs" => 0b10,
        "synxs" => 0b11,
        _ => return None,
    };
    Some(imm2)
}

/// Inverse of [`crate::decode::branch_sys`]'s barrier-option table (`sy`/`ld`/
/// `st`/`ish`/...). Returns the `CRm` value for a named option.
#[inline]
fn barrier_option_crm(name: &str) -> Option<u32> {
    let crm = match name {
        "oshld" => 0b0001,
        "oshst" => 0b0010,
        "osh" => 0b0011,
        "nshld" => 0b0101,
        "nshst" => 0b0110,
        "nsh" => 0b0111,
        "ishld" => 0b1001,
        "ishst" => 0b1010,
        "ish" => 0b1011,
        "ld" => 0b1101,
        "st" => 0b1110,
        "sy" => 0b1111,
        _ => return None,
    };
    Some(crm)
}

// ---------------------------------------------------------------------------
// System: MSR (immediate) PSTATE, CFINV/XAFLAG/AXFLAG, SMSTART/SMSTOP.
// ---------------------------------------------------------------------------

/// `MSR <pstatefield>, #imm` and the bare PSTATE ops. `CRn` is fixed at `0100`.
fn enc_msr_imm(insn: &Instruction) -> R {
    use Code::*;
    let (op1, op2, crm): (u32, u32, u32) = match insn.code() {
        // Bare PSTATE ops (op1==000, CRm==0). op2: CFINV=0, XAFLAG=1, AXFLAG=2.
        Cfinv => (0b000, 0b000, 0),
        Xaflag => (0b000, 0b001, 0),
        Axflag => (0b000, 0b010, 0),
        // SMSTART/SMSTOP (SVCR PSTATE): op1==011, op2==011. CRm packs
        // `0:ZA:SM:start` where bit0 selects start(1)/stop(0).
        Smstart | Smstop => {
            let start = if insn.code() == Smstart { 1u32 } else { 0 };
            // Optional `sm`/`za` option; bare form sets both SM and ZA.
            let (sm, za) = match insn.op(0) {
                Operand::None => (1u32, 1u32),
                Operand::SysOp(tok) => match tok.name() {
                    "sm" => (1, 0),
                    "za" => (0, 1),
                    _ => return Err(EncodeError::InvalidOperand),
                },
                _ => return Err(EncodeError::InvalidOperand),
            };
            let crm = (za << 2) | (sm << 1) | start;
            (0b011, 0b011, crm)
        }
        // MSR (immediate) named PSTATE field: SysOp name -> (op1, op2), CRm = imm.
        MsrImm => {
            let field = sysop_name(insn, 0)?;
            let (op1, op2) = pstate_field_fields(field).ok_or(EncodeError::InvalidOperand)?;
            let crm = imm_u(insn, 1)? as u32;
            (op1, op2, crm)
        }
        _ => return Err(EncodeError::Unsupported),
    };
    if crm > 0xf {
        return Err(EncodeError::InvalidImmediate);
    }
    Ok(system_word(0, 0b00, op1, 0b0100, crm, op2, 0b11111))
}

/// Inverse of [`crate::decode::branch_sys`]'s `pstate_field_name`: map a named
/// PSTATE field to its `(op1, op2)` selector.
#[inline]
fn pstate_field_fields(name: &str) -> Option<(u32, u32)> {
    let pair = match name {
        "uao" => (0b000, 0b011),
        "spsel" => (0b000, 0b101),
        "ssbs" => (0b011, 0b001),
        "dit" => (0b011, 0b010),
        "tco" => (0b011, 0b100),
        "daifset" => (0b011, 0b110),
        "daifclr" => (0b011, 0b111),
        "allint" => (0b001, 0b000),
        _ => return None,
    };
    Some(pair)
}

// ---------------------------------------------------------------------------
// System: MSR / MRS (register).
// ---------------------------------------------------------------------------

/// `MSR <systemreg>, <Xt>` / `MRS <Xt>, <systemreg>` — move to/from a system
/// register. The `SystemReg` carries the full `op0/op1/CRn/CRm/op2` directory.
fn enc_sysreg_move(insn: &Instruction) -> R {
    let read = insn.code() == Code::Mrs;
    let (sr, rt) = if read {
        // MRS <Xt>, <sysreg>.
        let rt = reg_num(insn, 0)?;
        let sr = match insn.op(1) {
            Operand::SysReg(sr) => sr,
            _ => return Err(EncodeError::InvalidOperand),
        };
        (sr, rt)
    } else {
        // MSR <sysreg>, <Xt>.
        let sr = match insn.op(0) {
            Operand::SysReg(sr) => sr,
            _ => return Err(EncodeError::InvalidOperand),
        };
        let rt = reg_num(insn, 1)?;
        (sr, rt)
    };
    let l = if read { 1u32 } else { 0 };
    Ok(system_word(
        l,
        sr.op0() as u32,
        sr.op1() as u32,
        sr.crn() as u32,
        sr.crm() as u32,
        sr.op2() as u32,
        rt,
    ))
}

// ---------------------------------------------------------------------------
// System: SYS / SYSL and the IC/DC/AT/TLBI/CFP/CPP/DVP aliases.
// ---------------------------------------------------------------------------

/// `SYS`/`SYSL` and the named system-instruction aliases. For `SYSL` and the
/// canonical `SYS` we read the fields directly off the operands; for an alias
/// (`IC`/`DC`/`AT`/`TLBI`/`CFP`/`CPP`/`DVP`) we recover `(op1,CRn,CRm,op2)` from
/// the operation-name keyword.
fn enc_sys(insn: &Instruction) -> R {
    // The *canonical* SYSL (mnemonic `Sysl`) reads its raw fields off the
    // operands. A `Sysl`-coded *alias* (e.g. `GICR`) falls through to the named
    // path below.
    if insn.code() == Code::Sysl && insn.mnemonic() == Mnemonic::Sysl {
        // SYSL <Xt>, #op1, Cn, Cm, #op2.
        let rt = reg_num(insn, 0)?;
        let op1 = imm_u(insn, 1)? as u32;
        let crn = cr_num(insn, 2)?;
        let crm = cr_num(insn, 3)?;
        let op2 = imm_u(insn, 4)? as u32;
        return Ok(system_word(1, 0b01, op1, crn, crm, op2, rt));
    }

    let read = insn.code() == Code::Sysl;

    // SYS/SYSL alias forms carry a named operation keyword; the canonical form
    // carries the raw #op1 immediate.
    match insn.mnemonic() {
        // Canonical SYS #op1, Cn, Cm, #op2{, Xt}.
        Mnemonic::Sys => {
            let op1 = imm_u(insn, 0)? as u32;
            let crn = cr_num(insn, 1)?;
            let crm = cr_num(insn, 2)?;
            let op2 = imm_u(insn, 3)? as u32;
            // The canonical form always carries the Xt; default to XZR if absent.
            let rt = if insn.op_count() >= 5 {
                reg_num(insn, 4)?
            } else {
                0b11111
            };
            Ok(system_word(0, 0b01, op1, crn, crm, op2, rt))
        }
        // CFP/CPP/DVP RCTX: handled with a fixed tuple (their keyword is "rctx").
        Mnemonic::Cfp | Mnemonic::Cpp | Mnemonic::Dvp => {
            let (op1, crn, crm, op2) = match insn.mnemonic() {
                Mnemonic::Cfp => (3, 7, 3, 0b100),
                Mnemonic::Dvp => (3, 7, 3, 0b101),
                _ => (3, 7, 3, 0b111),
            };
            let rt = reg_num(insn, 1)?;
            Ok(system_word(0, 0b01, op1, crn, crm, op2, rt))
        }
        // Named alias family (IC/DC/AT/TLBI/PLBI/GIC/GICR/MLBI/APAS/TRCIT/COSP):
        // recover the tuple from the shared directory, keyed on `(mnem, op-name)`.
        m => {
            // Read-side register-first forms (`gicr <Xt>, <op>`) carry the keyword
            // last; all the others carry it first.
            let kw_slot = match m {
                Mnemonic::Gicr => 1,
                _ => 0,
            };
            // APAS/TRCIT take no keyword (their entry's `name` is "").
            let name = if matches!(m, Mnemonic::Apas | Mnemonic::Trcit) {
                ""
            } else {
                sysop_name(insn, kw_slot)?
            };
            let t = crate::tables::sysins::lookup_by_name(m, name)
                .ok_or(EncodeError::InvalidOperand)?;
            // The Xt slot: a keyword-first form with a keyword puts the register
            // after it (slot 1); the register-first read forms (`gicr`) and the
            // keyword-less forms (`apas`/`trcit`) put it at slot 0. Whole-structure
            // forms (no Xt) default to canonical XZR.
            let rt = if t.needs_rt {
                let rt_slot = if t.kw_first && !t.name.is_empty() { 1 } else { 0 };
                reg_num(insn, rt_slot)?
            } else {
                0b11111
            };
            Ok(system_word(
                u32::from(read),
                0b01,
                t.op1 as u32,
                t.crn as u32,
                t.crm as u32,
                t.op2 as u32,
                rt,
            ))
        }
    }
}

/// The GCS stack-maintenance ops (`GCSPUSHM`/`GCSPOPM`/`GCSSS1`/`GCSSS2`/
/// `GCSPUSHX`/`GCSPOPX`/`GCSPOPCX`) — the inverse of `decode_gcs_sys`. All live
/// at `CRn==7, CRm==7`; the `*M`/`SS1`/`SS2` forms carry `<Xt>` (`GCSPOPM` elides
/// `Rt==11111`), the `*X`/`*CX` forms take none.
fn enc_gcs_sys(insn: &Instruction) -> R {
    use Code::*;
    // (read/L, op1, op2, takes_rt, elide_rt_31).
    let (read, op1, op2, takes_rt, elide): (u32, u32, u32, bool, bool) = match insn.code() {
        Gcspushx => (0, 0, 4, false, false),
        Gcspopcx => (0, 0, 5, false, false),
        Gcspopx => (0, 0, 6, false, false),
        Gcspushm => (0, 3, 0, true, false),
        Gcsss1 => (0, 3, 2, true, false),
        Gcspopm => (1, 3, 1, true, true),
        Gcsss2 => (1, 3, 3, true, false),
        _ => return Err(EncodeError::Unsupported),
    };
    let rt = if takes_rt {
        if elide && insn.op_count() == 0 {
            0b11111
        } else {
            reg_num(insn, 0)?
        }
    } else {
        0b11111
    };
    Ok(system_word(read, 0b01, op1, 0b0111, 0b0111, op2, rt))
}

/// The `CRn`/`CRm` 4-bit value of a `cN` [`Operand::SysOp`] token at slot `n`.
fn cr_num(insn: &Instruction, n: usize) -> Result<u32, EncodeError> {
    let name = sysop_name(insn, n)?;
    // `cN` tokens are "c0".."c15".
    let digits = name.strip_prefix('c').ok_or(EncodeError::InvalidOperand)?;
    digits.parse::<u32>().map_err(|_| EncodeError::InvalidOperand)
}

// ---------------------------------------------------------------------------
// System: TSTART / TTEST (system instruction with result).
// ---------------------------------------------------------------------------

/// `TSTART`/`TTEST <Xt>` — read side, `op1==011, CRn==0011, op2==011`, `CRm`
/// selects (0 = TSTART, 1 = TTEST), `Rt` is the destination.
fn enc_systemresult(insn: &Instruction) -> R {
    let crm = if insn.code() == Code::Ttest { 1u32 } else { 0 };
    let rt = reg_num(insn, 0)?;
    Ok(system_word(1, 0b00, 0b011, 0b0011, crm, 0b011, rt))
}

// ---------------------------------------------------------------------------
// System: FEAT_D128 MRRS / MSRR / SYSP / TLBIP (system-pair block, word<22>==1).
// ---------------------------------------------------------------------------

/// `MRRS <Xt>, <Xt+1>, <sysreg>` / `MSRR <sysreg>, <Xt>, <Xt+1>` — the 128-bit
/// system-register pair move. The `SystemReg` carries the full directory; the
/// even transfer base comes from the [`Operand::RegPair`].
fn enc_sysreg_pair(insn: &Instruction) -> R {
    let read = insn.code() == Code::Mrrs;
    let (sr, rt) = if read {
        // MRRS <Xt>, <Xt+1>, <sysreg>.
        let rt = reg_pair_base(insn, 0)?;
        let sr = match insn.op(1) {
            Operand::SysReg(sr) => sr,
            _ => return Err(EncodeError::InvalidOperand),
        };
        (sr, rt)
    } else {
        // MSRR <sysreg>, <Xt>, <Xt+1>.
        let sr = match insn.op(0) {
            Operand::SysReg(sr) => sr,
            _ => return Err(EncodeError::InvalidOperand),
        };
        let rt = reg_pair_base(insn, 1)?;
        (sr, rt)
    };
    let l = if read { 1u32 } else { 0 };
    Ok(system_pair_word(
        l,
        sr.op0() as u32,
        sr.op1() as u32,
        sr.crn() as u32,
        sr.crm() as u32,
        sr.op2() as u32,
        rt,
    ))
}

/// `SYSP #op1, Cn, Cm, #op2{, <Xt>, <Xt+1>}` and the `TLBIP` alias. The generic
/// form reads its fields off the operands (with the pair elided -> `Rt==11111`);
/// the `TLBIP` alias recovers `(op1, CRm, op2)` from the operation-name keyword
/// (CRn is always 8) and always carries the pair.
fn enc_sysp(insn: &Instruction) -> R {
    match insn.mnemonic() {
        // TLBIP <op>, <Xt>, <Xt+1>: tuple from the shared TLBI op-name directory.
        Mnemonic::Tlbip => {
            let name = sysop_name(insn, 0)?;
            let t = crate::tables::sysins::lookup_by_name(Mnemonic::Tlbi, name)
                .ok_or(EncodeError::InvalidOperand)?;
            let rt = reg_pair_base(insn, 1)?;
            Ok(system_pair_word(0, 0b01, t.op1 as u32, t.crn as u32, t.crm as u32, t.op2 as u32, rt))
        }
        // Canonical SYSP #op1, Cn, Cm, #op2{, <Xt>, <Xt+1>}.
        _ => {
            let op1 = imm_u(insn, 0)? as u32;
            let crn = cr_num(insn, 1)?;
            let crm = cr_num(insn, 2)?;
            let op2 = imm_u(insn, 3)? as u32;
            // The pair is present iff a 5th operand exists; absent -> no-transfer
            // sentinel Rt==11111.
            let rt = if insn.op_count() >= 5 {
                reg_pair_base(insn, 4)?
            } else {
                0b11111
            };
            Ok(system_pair_word(0, 0b01, op1, crn, crm, op2, rt))
        }
    }
}

/// Recover the even 64-bit transfer base `Rt` from an [`Operand::RegPair`] at
/// slot `n`. The pair must be `(<Xt>, <Xt+1>)` with `Xt` even; a `(xzr, xzr)`
/// pair (the `TLBIP` no-transfer spelling) maps to `Rt == 11111`.
fn reg_pair_base(insn: &Instruction, n: usize) -> Result<u32, EncodeError> {
    let (first, second) = match insn.op(n) {
        Operand::RegPair { first, second } => (first, second),
        _ => return Err(EncodeError::InvalidOperand),
    };
    let lo = first.number() as u32;
    let hi = second.number() as u32;
    // `xzr, xzr` is the no-transfer pair -> Rt==11111.
    if lo == 31 && hi == 31 {
        return Ok(0b11111);
    }
    // Otherwise the base must be even and the high half its successor.
    if lo & 1 != 0 || hi != (lo + 1) & 0x1f {
        return Err(EncodeError::InvalidOperand);
    }
    Ok(lo)
}

// ---------------------------------------------------------------------------
// Reserved: UDF.
// ---------------------------------------------------------------------------

/// `UDF #imm16` — the permanently-undefined encoding (`word<31:16> == 0`).
fn enc_udf(insn: &Instruction) -> R {
    let imm16 = imm_u(insn, 0)?;
    if imm16 > 0xffff {
        return Err(EncodeError::InvalidImmediate);
    }
    Ok(imm16 as u32)
}

// ---------------------------------------------------------------------------
// Shared system-word packer.
// ---------------------------------------------------------------------------

/// Pack a System-block instruction word: `1101010100 L op0 op1 CRn CRm op2 Rt`.
#[inline]
fn system_word(l: u32, op0: u32, op1: u32, crn: u32, crm: u32, op2: u32, rt: u32) -> u32 {
    (0b1101010100u32 << 22)
        | ((l & 0x1) << 21)
        | ((op0 & 0x3) << 19)
        | ((op1 & 0x7) << 16)
        | ((crn & 0xf) << 12)
        | ((crm & 0xf) << 8)
        | ((op2 & 0x7) << 5)
        | (rt & 0x1f)
}

/// Pack a FEAT_D128 system-pair word: identical to [`system_word`] but with the
/// `word<22> == 1` prefix (`1101010101`).
#[inline]
fn system_pair_word(l: u32, op0: u32, op1: u32, crn: u32, crm: u32, op2: u32, rt: u32) -> u32 {
    system_word(l, op0, op1, crn, crm, op2, rt) | (1u32 << 22)
}

#[cfg(test)]
mod tests {
    use crate::features::FeatureSet;
    use crate::instruction::Instruction;

    /// Decode a word then re-encode it and require the exact same word back.
    #[track_caller]
    fn rt(word: u32) {
        let mut insn = Instruction::default();
        crate::decode::decode_into(word, 0x8000_0000_0000_0004, FeatureSet::ALL, &mut insn);
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
    fn branch_sys_known_words() {
        // B / BL.
        rt(0x14F5E1BA);
        rt(0x94A68B21);
        // B.cond.
        rt(0x54156881);
        rt(0x54FEE2A8);
        // FEAT_HBC BC.cond (bit4 == 1).
        rt(0x54156891); // bc.<cond>, forward offset
        rt(0x54FEE2B8); // bc.<cond>, backward offset
        rt(0x543779B1); // bc.ne (oracle example)
        // FEAT_PAuth_LR RETAASPPC/RETABSPPC (PC-relative return, backward imm16).
        rt(0x551E8D9F); // retaasppc
        rt(0x552F577F); // retabsppc
        rt(0x5500001F); // retaasppc, offset 0
        // FEAT_PAuth_LR AUTIASPPC/AUTIBSPPC (decoded in dp_imm, encoded here).
        rt(0xF3983E9F); // autiasppc
        rt(0xF3B9D25F); // autibsppc
        rt(0xF3A0001F); // autibsppc, offset 0
        // CBZ/CBNZ, TBZ/TBNZ.
        rt(0x342F64AB);
        rt(0x358FD614);
        rt(0x36EE4A53);
        rt(0xB72DA58C);
        // FEAT_CMPBR compare-and-branch (register, word) — gt/ge/hi/hs/eq/ne.
        rt(0x740005A7); // cbgt w7, w0, <label>
        rt(0x74201313); // cbge w19, w0, <label>
        rt(0x744007CE); // cbhi w14, w0, <label>
        rt(0x746003FB); // cbhs w27, w0, <label>
        rt(0x74C008F8); // cbeq w24, w0, <label>
        rt(0x74E00459); // cbne w25, w0, <label>
        rt(0xF40005A7); // cbgt x7, x0, <label> (64-bit)
        rt(0xF4E939C5); // cbne x5, x9, <label> (64-bit, negative offset)
        // FEAT_CMPBR byte register form (CBB<cc>).
        rt(0x74E0841D); // cbbne w29, w0, <label>
        rt(0x74409072); // cbbhi w18, w0, <label>
        // FEAT_CMPBR halfword register form (CBH<cc>).
        rt(0x74E0C6BA); // cbhne w26, w0, <label>
        rt(0x74C0C240); // cbheq w0, w0, <label>
        rt(0x7420E35D); // cbhge w29, w0, <label> (negative offset)
        // FEAT_CMPBR immediate-compare form — gt/lt/hi/lo/eq/ne, W and X.
        rt(0x752002C9); // cblt w9, #0, <label>
        rt(0x75600E4B); // cblo w11, #0, <label>
        rt(0x75050672); // cbgt w18, #10, <label>
        rt(0x75CA8BC0); // cbeq w0, #21, <label>  (imm6 != 0)
        rt(0xF53F8BFC); // cblt x28, #63, <label> (64-bit, imm6 == 63)
        rt(0xF5E387B7); // cbne x23, #7, <label>
        // Branch register + RET elision.
        rt(0xD61F00E0);
        rt(0xD63F0080);
        rt(0xD65F03C0); // ret (x30 elided)
        rt(0xD65F0220); // ret x17
        rt(0xD69F03E0); // eret
        rt(0xD6BF03E0); // drps
        // PAuth branch register.
        rt(0xD71F091C);
        rt(0xD73F0A0B);
        rt(0xD61F0BBF);
        rt(0xD65F0BFF);
        // Exceptions.
        rt(0xD4016F21);
        rt(0xD40DF462);
        rt(0xD419FA83);
        rt(0xD422A5A0);
        rt(0xD4424C60);
        rt(0xD4A9E481); // dcps1 #imm
        rt(0xD478DA60); // tcancel
        // Hints.
        rt(0xD503201F); // nop
        rt(0xD503203F); // yield
        rt(0xD503221F); // esb
        rt(0xD503223F); // psb csync
        rt(0xD503245F); // bti c
        rt(0xD503241F); // bti
        rt(0xD50320DF); // dgh (HINT #6)
        rt(0xD503225F); // tsb csync
        // T: newer named hints.
        rt(0xD503227F); // gcsb dsync
        rt(0xD50322DF); // clrbhb
        rt(0xD50324FF); // pacm
        rt(0xD503251F); // chkfeat x16
        rt(0xD503267F); // shuh ph
        rt(0xD503265F); // shuh
        rt(0xD503261F); // stshh keep
        rt(0xD503263F); // stshh strm
        rt(0xD503269F); // stcph
        // Barriers.
        rt(0xD5033F9F); // dsb sy
        rt(0xD503369F); // dsb nshst
        rt(0xD50335BF); // dmb nshld
        rt(0xD5033FDF); // isb
        rt(0xD50331DF); // isb #1
        rt(0xD5033E5F); // clrex #0xe
        rt(0xD503309F); // ssbb
        rt(0xD503307F); // tcommit
        // DSB nXS (FEAT_XS).
        rt(0xD5033E3F); // dsb synxs
        rt(0xD503363F); // dsb nshnxs
        rt(0xD503323F); // dsb oshnxs
        rt(0xD5033A3F); // dsb ishnxs
        // SB (canonical CRm==0).
        rt(0xD50330FF); // sb
        // MSR (immediate) PSTATE + bare ops.
        rt(0xD50049BF); // msr spsel, #9
        rt(0xD5034EDF); // msr daifset, #0xe
        rt(0xD503455F); // msr dit, #5
        rt(0xD500401F); // cfinv
        rt(0xD500459F); // msr s0_0_c4_c5_4, xzr (generic fallback)
        // MSR/MRS register.
        rt(0xD51B192B);
        rt(0xD539533E);
        rt(0xD53B4200); // mrs x0, nzcv
        // Generic MSR/MRS (register) for the op0==00 holes LLVM accepts.
        rt(0xD5000064); // msr s0_0_c0_c0_3, x4
        rt(0xD52000C2); // mrs x2, s0_0_c0_c0_6
        rt(0xD5005007); // msr s0_0_c5_c0_0, x7
        rt(0xD50045A5); // msr s0_0_c4_c5_5, x5 (PSTATE CRn, Rt!=xzr)
        rt(0xD500301F); // msr s0_0_c3_c0_0, xzr (barrier op2==0 hole)
        // FEAT_D128 MRRS/MSRR/SYSP/TLBIP.
        rt(0xD56001CC); // mrrs x12, x13, s0_0_c0_c1_6
        rt(0xD540005E); // msrr s0_0_c0_c0_2, x30, xzr
        rt(0xD578200C); // mrrs x12, x13, ttbr0_el1
        rt(0xD558200C); // msrr ttbr0_el1, x12, x13
        rt(0xD5480036); // sysp #0, c0, c0, #1, x22, x23
        rt(0xD548003F); // sysp #0, c0, c0, #1 (no-transfer)
        rt(0xD54C8020); // tlbip ipas2e1is, x0, x1
        rt(0xD54C803F); // tlbip ipas2e1is, xzr, xzr
        // SYS / SYSL / aliases.
        rt(0xD50D3428); // sys
        rt(0xD52F54FF); // sysl
        rt(0xD50B752A); // ic ivau, x10
        rt(0xD50B7438); // dc zva, x24
        rt(0xD50B7B24); // dc cvau, x4
        rt(0xD508792A); // at s1e1wp, x10
        rt(0xD50B738C); // cfp rctx, x12
        rt(0xD50B73F0); // cpp rctx, x16
        rt(0xD50B73A6); // dvp rctx, x6
        // WFET/WFIT.
        rt(0xD503100F);
        rt(0xD503103E);
        // TSTART/TTEST.
        rt(0xD5233070);
        rt(0xD5233178);
        // UDF.
        rt(0x00004ABD);
    }
}
