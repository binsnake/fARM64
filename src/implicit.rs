//! Implicit register reads and writes — the architectural state an A64
//! instruction touches **without naming it in an operand**.
//!
//! A64 hides a surprising amount of dataflow behind the mnemonic:
//!
//! * `BL`/`BLR` write the link register `X30`; `RET` reads it.
//! * The `PAC*`/`AUT*`/`XPACLRI` hint forms sign or authenticate `X30`
//!   in place, some of them using `SP` (and, for FEAT_PAuth_LR, `PC`) as the
//!   modifier — none of which appear in the disassembly text.
//! * `PACIA1716` and friends read `X16` and read-modify `X17`.
//! * The FEAT_LS64 `LD64B`/`ST64B`/`ST64BV`/`ST64BV0` transfer the *eight*
//!   consecutive registers `Xt..Xt+7` while spelling only `Xt`.
//! * The SVE non-faulting / first-faulting loads and `RDFFR`/`WRFFR`/`SETFFR`
//!   read and write the first-fault register `FFR`.
//! * `ADR`/`ADRP` and the PC-relative literal loads read `PC` as a genuine data
//!   input.
//! * Every flag-setting and flag-consuming form touches `NZCV`.
//!
//! This module reports all of that as a small, fixed-capacity list of
//! [`UsedRegister`]s, using the [`Register`] pseudo-register values
//! [`Register::Nzcv`], [`Register::Ffr`], [`Register::Za`] and [`Register::Pc`]
//! for the architectural state that has no numbered register.
//!
//! The list is **only** the implicit part. [`crate::info::instruction_info`]
//! folds it into the full [`crate::info::InstructionInfo::used_registers`] set
//! alongside the explicit operands, merging accesses where an instruction both
//! names a register and touches it implicitly (`RET X30`).
//!
//! Like the rest of the crate this is `no_std`, zero-alloc and total: it never
//! panics and never allocates.
//!
//! ```
//! use fARM64::{Decoder, DecoderOptions, OpAccess, Register};
//!
//! // `bl #4` — writes the link register, which is nowhere in the text.
//! let bytes = 0x9400_0001u32.to_le_bytes();
//! let mut dec = Decoder::new(&bytes, 0, DecoderOptions::NONE);
//! let insn = dec.decode();
//! let imp = insn.implicit_registers();
//! assert_eq!(imp.access_of(Register::X30), OpAccess::Write);
//! ```

use crate::enums::FlowControl;
use crate::info::{OpAccess, UsedRegister};
use crate::instruction::Instruction;
use crate::mnemonic::Mnemonic;
use crate::operand::Operand;
use crate::register::Register;

/// Maximum number of implicit register accesses recorded for one instruction.
///
/// The worst case is the FEAT_LS64 `ST64BV` family: seven implicit data
/// registers (`Xt+1..Xt+7`) plus `NZCV`. Sized with headroom so
/// [`implicit_registers`] never has to drop an entry.
pub const MAX_IMPLICIT_REGS: usize = 10;

/// The implicit register accesses of one instruction (no_std, no alloc).
///
/// A `Copy` value holding a fixed-capacity inline list; obtain it from
/// [`implicit_registers`] or [`Instruction::implicit_registers`].
#[derive(Debug, Clone, Copy)]
pub struct ImplicitRegisters {
    regs: [UsedRegister; MAX_IMPLICIT_REGS],
    count: u8,
}

impl ImplicitRegisters {
    /// The empty list.
    #[inline]
    fn new() -> Self {
        ImplicitRegisters {
            regs: [UsedRegister {
                register: Register::None,
                access: OpAccess::None,
            }; MAX_IMPLICIT_REGS],
            count: 0,
        }
    }

    /// Add (or merge) one implicit access. [`Register::None`] /
    /// [`OpAccess::None`] are ignored; a register already present has its
    /// access merged (a read plus a write becomes [`OpAccess::ReadWrite`]).
    #[inline]
    fn add(&mut self, register: Register, access: OpAccess) {
        if register == Register::None || access == OpAccess::None {
            return;
        }
        for entry in self.regs[..self.count as usize].iter_mut() {
            if entry.register == register {
                entry.access = entry.access.merge(access);
                return;
            }
        }
        if (self.count as usize) < MAX_IMPLICIT_REGS {
            self.regs[self.count as usize] = UsedRegister { register, access };
            self.count += 1;
        }
    }

    /// The implicit accesses, in a stable order.
    #[inline]
    pub fn as_slice(&self) -> &[UsedRegister] {
        &self.regs[..self.count as usize]
    }

    /// Number of implicit accesses.
    #[inline]
    pub fn len(&self) -> usize {
        self.count as usize
    }

    /// `true` if the instruction touches no register implicitly.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Iterate the implicit accesses.
    #[inline]
    pub fn iter(&self) -> core::slice::Iter<'_, UsedRegister> {
        self.as_slice().iter()
    }

    /// How `register` is accessed implicitly, or [`OpAccess::None`] if it is
    /// not touched implicitly at all.
    #[inline]
    pub fn access_of(&self, register: Register) -> OpAccess {
        for entry in self.as_slice() {
            if entry.register == register {
                return entry.access;
            }
        }
        OpAccess::None
    }

    /// `true` if `register` is implicitly read (including read-modify-write).
    #[inline]
    pub fn reads(&self, register: Register) -> bool {
        matches!(
            self.access_of(register),
            OpAccess::Read | OpAccess::ReadWrite | OpAccess::CondRead
        )
    }

    /// `true` if `register` is implicitly written (including read-modify-write).
    #[inline]
    pub fn writes(&self, register: Register) -> bool {
        matches!(
            self.access_of(register),
            OpAccess::Write | OpAccess::ReadWrite | OpAccess::CondWrite
        )
    }
}

impl<'a> IntoIterator for &'a ImplicitRegisters {
    type Item = &'a UsedRegister;
    type IntoIter = core::slice::Iter<'a, UsedRegister>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl Default for ImplicitRegisters {
    #[inline]
    fn default() -> Self {
        ImplicitRegisters::new()
    }
}

/// `true` if the mnemonic reads NZCV without carrying an explicit condition
/// operand: the carry-consuming (`ADC`/`SBC`/`NGC`) and conditional-select /
/// conditional-compare families, plus the flag-manipulation forms that preserve
/// the flags they do not write (`RMIF`, `SETF8`/`SETF16`, `CFINV`,
/// `AXFLAG`/`XAFLAG`).
///
/// The conditional forms that *do* carry a condition code are detected
/// separately through [`Instruction::condition`].
pub(crate) fn mnemonic_reads_flags(m: Mnemonic) -> bool {
    use Mnemonic::*;
    matches!(
        m,
        Adc | Adcs
            | Sbc
            | Sbcs
            | Ngc
            | Ngcs
            | Csel
            | Csinc
            | Csinv
            | Csneg
            | Cinc
            | Cinv
            | Cneg
            | Cset
            | Csetm
            | Ccmp
            | Ccmn
            | Fcsel
            // Flag manipulation: each writes only a subset of NZCV and leaves
            // the rest intact, so the flags are read-modified rather than
            // wholly rewritten.
            | Rmif
            | Setf8
            | Setf16
            | Cfinv
            | Axflag
            | Xaflag
    )
}

/// `true` if the instruction reads any of the NZCV condition flags.
#[inline]
pub(crate) fn reads_nzcv(insn: &Instruction) -> bool {
    insn.condition().is_some() || mnemonic_reads_flags(insn.mnemonic())
}

/// The implicit register accesses of `insn`.
///
/// Returns only the registers the instruction touches *without* naming them in
/// an operand (plus the [`Register::Nzcv`] / [`Register::Ffr`] /
/// [`Register::Pc`] / [`Register::Za`] pseudo-registers for architectural state
/// that has no numbered register). Explicit operands are covered by
/// [`crate::info::instruction_info`], which folds this list in.
///
/// Zero-alloc and total.
pub fn implicit_registers(insn: &Instruction) -> ImplicitRegisters {
    let mut out = ImplicitRegisters::new();
    let m = insn.mnemonic();

    add_nzcv(&mut out, insn);
    add_link_register(&mut out, insn, m);
    add_pointer_auth(&mut out, m);
    add_ls64_group(&mut out, insn, m);
    add_ffr(&mut out, m);
    add_sme_state(&mut out, insn, m);
    add_pc(&mut out, insn, m);

    out
}

/// The NZCV pseudo-register, from the instruction's flag read/write behaviour.
fn add_nzcv(out: &mut ImplicitRegisters, insn: &Instruction) {
    let written = insn.set_flags().writes_flags();
    let read = reads_nzcv(insn);
    let access = match (read, written) {
        (true, true) => OpAccess::ReadWrite,
        (true, false) => OpAccess::Read,
        (false, true) => OpAccess::Write,
        (false, false) => return,
    };
    out.add(Register::Nzcv, access);
}

/// The link register `X30` for calls and returns.
///
/// Calls (`BL`, `BLR`, and the pointer-authenticated `BLRA*`) write the return
/// address into `X30`. `RET` without an explicit register returns through
/// `X30`; `RET Xn` names its register and gets nothing implicit here. The
/// pointer-authenticated returns (`RETAA`/`RETAB` and the FEAT_PAuth_LR
/// `RETA*SPPC`/`RETA*SPPCR`) always use `X30`.
fn add_link_register(out: &mut ImplicitRegisters, insn: &Instruction, m: Mnemonic) {
    use Mnemonic::*;
    match insn.flow_control() {
        // Every call form writes the return address to X30.
        FlowControl::Call | FlowControl::IndirectCall => out.add(Register::X30, OpAccess::Write),
        FlowControl::Return => match m {
            // `RET` defaults to X30; `RET Xn` states its register explicitly.
            Ret if insn.op_count() == 0 => out.add(Register::X30, OpAccess::Read),
            // The authenticated returns are hard-wired to X30 and never spell it.
            Retaa | Retab | Retaasppc | Retabsppc | Retaasppcr | Retabsppcr => {
                out.add(Register::X30, OpAccess::Read)
            }
            _ => {}
        },
        _ => {}
    }
}

/// The pointer-authentication hint forms, which sign or authenticate a register
/// in place using an implicit modifier.
///
/// * `PACIASP`/`AUTIASP`/... read-modify `X30` with `SP` as the modifier.
/// * `PACIAZ`/`AUTIAZ`/... read-modify `X30` with a zero modifier.
/// * `PACIA1716`/`AUTIA1716`/... read-modify `X17` with `X16` as the modifier.
/// * `XPACLRI` strips the PAC from `X30` in place.
/// * The FEAT_PAuth_LR `PAC*SPPC` / `AUT*SPPC` forms read-modify `X30` with
///   `SP` (and `PC`, added by [`add_pc`]) as the modifier; the `RETA*SPPC*`
///   returns read `X30` (added by [`add_link_register`]) and `SP`.
fn add_pointer_auth(out: &mut ImplicitRegisters, m: Mnemonic) {
    use Mnemonic::*;
    match m {
        // Sign / authenticate LR, SP as modifier.
        Paciasp | Pacibsp | Autiasp | Autibsp => {
            out.add(Register::X30, OpAccess::ReadWrite);
            out.add(Register::Sp, OpAccess::Read);
        }
        // Sign / authenticate LR, zero modifier.
        Paciaz | Pacibz | Autiaz | Autibz => out.add(Register::X30, OpAccess::ReadWrite),
        // Strip the PAC from LR.
        Xpaclri => out.add(Register::X30, OpAccess::ReadWrite),
        // Sign / authenticate X17 using X16 as the modifier.
        Pacia1716 | Pacib1716 | Autia1716 | Autib1716 => {
            out.add(Register::X17, OpAccess::ReadWrite);
            out.add(Register::X16, OpAccess::Read);
        }
        // FEAT_PAuth_LR: sign / authenticate LR using SP (+ PC) as the modifier.
        // The `AUT*SPPCR` register forms carry their extra modifier explicitly.
        Paciasppc | Pacibsppc | Pacnbiasppc | Pacnbibsppc | Autiasppc | Autibsppc | Autiasppcr
        | Autibsppcr => {
            out.add(Register::X30, OpAccess::ReadWrite);
            out.add(Register::Sp, OpAccess::Read);
        }
        // FEAT_PAuth_LR returns: X30 comes from `add_link_register`; SP is the
        // implicit modifier here.
        Retaasppc | Retabsppc | Retaasppcr | Retabsppcr => out.add(Register::Sp, OpAccess::Read),
        // `RETAA`/`RETAB` authenticate LR using SP as the modifier.
        Retaa | Retab => out.add(Register::Sp, OpAccess::Read),
        // `ERETAA`/`ERETAB` authenticate ELR_ELx using SP as the modifier.
        Eretaa | Eretab => out.add(Register::Sp, OpAccess::Read),
        _ => {}
    }
}

/// The FEAT_LS64 64-byte transfers, which move `Xt..Xt+7` while naming only
/// `Xt`.
///
/// `LD64B <Xt>, [Xn]` writes eight consecutive registers; `ST64B <Xt>, [Xn]`
/// and `ST64BV{0} <Xs>, <Xt>, [Xn]` read eight. `Xt` itself is explicit, so only
/// `Xt+1..Xt+7` are recorded here. The architecture requires `Xt` to be even and
/// at most `x22`, so the group never runs past `x29`.
fn add_ls64_group(out: &mut ImplicitRegisters, insn: &Instruction, m: Mnemonic) {
    use Mnemonic::*;
    // Slot of the 64-byte data register `Xt`, and how the group is accessed.
    let (slot, access) = match m {
        Ld64b => (0, OpAccess::Write),
        St64b => (0, OpAccess::Read),
        // `ST64BV`/`ST64BV0` put the status register `Xs` first.
        St64bv | St64bv0 => (1, OpAccess::Read),
        _ => return,
    };
    let base = match insn.op(slot) {
        Operand::Reg { reg, .. } => reg,
        _ => return,
    };
    // Only a 64-bit GP view names a valid LS64 group.
    if base.class() != crate::register::RegClass::Gp || base.width_bits() != 64 {
        return;
    }
    let n = base.number();
    // `Xt` must be even and `<= 22`; anything else is a reserved encoding the
    // decoder rejects, so bail rather than invent registers.
    if n & 1 != 0 || n > 22 {
        return;
    }
    for i in 1..8u8 {
        out.add(
            crate::register::gp_register(false, crate::register::RegWidth::X64, n + i),
            access,
        );
    }
}

/// The SVE first-fault register `FFR`.
///
/// The first-faulting loads (`LDFF1*`) read `FFR` to find the active prefix and
/// write it back on a fault; the non-faulting loads (`LDNF1*`) read it.
/// `SETFFR`/`WRFFR` write it, `RDFFR`/`RDFFRS` read it (and `RDFFRS` also writes
/// NZCV, via [`Instruction::set_flags`]).
fn add_ffr(out: &mut ImplicitRegisters, m: Mnemonic) {
    use Mnemonic::*;
    match m {
        // First-faulting loads read-modify FFR.
        Ldff1b | Ldff1h | Ldff1w | Ldff1d | Ldff1sb | Ldff1sh | Ldff1sw => {
            out.add(Register::Ffr, OpAccess::ReadWrite)
        }
        // Non-faulting loads read FFR (they never update it).
        Ldnf1b | Ldnf1h | Ldnf1w | Ldnf1d | Ldnf1sb | Ldnf1sh | Ldnf1sw => {
            out.add(Register::Ffr, OpAccess::Read)
        }
        // Explicit FFR moves: `SETFFR` sets every bit, `WRFFR <Pn>.B` copies a
        // predicate into it, and `RDFFR`/`RDFFRS` copy it back out.
        Setffr | Wrffr => out.add(Register::Ffr, OpAccess::Write),
        Rdffr | Rdffrs => out.add(Register::Ffr, OpAccess::Read),
        _ => {}
    }
}

/// The SME `ZA` array where it is touched without a ZA operand.
///
/// `SMSTART`/`SMSTOP` select which `PSTATE` bits they change with an optional
/// keyword operand: `SMSTART ZA` / `SMSTOP ZA` touch `PSTATE.ZA`, the bare
/// `SMSTART` / `SMSTOP` touch both `SM` and `ZA`, and `SMSTART SM` / `SMSTOP SM`
/// touch **only** streaming mode and leave `ZA` alone. Enabling or disabling
/// `PSTATE.ZA` makes the previous array contents architecturally unusable, so
/// the two `ZA`-affecting spellings report a `ZA` write and the `SM`-only
/// spelling reports nothing.
///
/// Entering or leaving streaming mode additionally zeroes every `Z` and `P`
/// register. That is 48 registers — far past the inline capacity here — so it
/// is deliberately **not** reported; treat a streaming-mode transition as a
/// barrier for SVE register state rather than relying on this list.
///
/// Instructions that *name* a ZA tile, tile slice, or ZA-array vector group
/// contribute `ZA` through their explicit operand instead, in [`crate::info`].
fn add_sme_state(out: &mut ImplicitRegisters, insn: &Instruction, m: Mnemonic) {
    use Mnemonic::*;
    if !matches!(m, Smstart | Smstop) {
        return;
    }
    // `SMSTART SM` / `SMSTOP SM` name streaming mode only and never touch ZA.
    if let Operand::SysOp(tok) = insn.op(0) {
        if tok.name() == "sm" {
            return;
        }
    }
    out.add(Register::Za, OpAccess::Write);
}

/// `PC` where it is a genuine data input.
///
/// `ADR`/`ADRP` and the PC-relative literal loads compute from `PC`: they carry
/// a pre-resolved [`Operand::Label`] but do not transfer control. The
/// FEAT_PAuth_LR `PAC*SPPC` forms fold `PC` into the pointer-authentication
/// modifier without carrying any operand at all.
///
/// Sequential fetch and branch-target formation are deliberately *not*
/// reported: every instruction would otherwise read `PC`.
fn add_pc(out: &mut ImplicitRegisters, insn: &Instruction, m: Mnemonic) {
    use Mnemonic::*;
    // FEAT_PAuth_LR folds the PC into the modifier with no operand to show it.
    if matches!(
        m,
        Paciasppc | Pacibsppc | Pacnbiasppc | Pacnbibsppc | Autiasppc | Autibsppc
    ) {
        out.add(Register::Pc, OpAccess::Read);
        return;
    }
    // A resolved label on an instruction that does *not* transfer control means
    // PC-relative address generation or a literal load.
    if insn.flow_control().is_control_transfer() {
        return;
    }
    for i in 0..insn.op_count() {
        if matches!(insn.op(i), Operand::Label(_)) {
            out.add(Register::Pc, OpAccess::Read);
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Decoder, DecoderOptions};

    /// Decode one word at `ip` with every runtime feature accepted.
    fn decode(word: u32) -> Instruction {
        let bytes = word.to_le_bytes();
        let mut dec = Decoder::new(&bytes, 0x1000, DecoderOptions::NONE);
        dec.decode()
    }

    /// Assert the implicit set is exactly `expected` (order-independent).
    #[track_caller]
    fn assert_implicit(word: u32, expected: &[(Register, OpAccess)]) {
        let insn = decode(word);
        let imp = implicit_registers(&insn);
        assert_eq!(
            imp.len(),
            expected.len(),
            "implicit count mismatch for {:#010x} ({:?}): got {:?}, expected {:?}",
            word,
            insn.mnemonic(),
            imp.as_slice(),
            expected
        );
        for &(reg, access) in expected {
            assert_eq!(
                imp.access_of(reg),
                access,
                "implicit access mismatch for {reg:?} in {:#010x} ({:?}): got {:?}",
                word,
                insn.mnemonic(),
                imp.as_slice()
            );
        }
    }

    #[test]
    fn call_writes_link_register() {
        // `bl #4`
        assert_implicit(0x9400_0001, &[(Register::X30, OpAccess::Write)]);
        // `blr x1`
        assert_implicit(0xD63F_0020, &[(Register::X30, OpAccess::Write)]);
    }

    #[test]
    fn ret_reads_link_register_only_when_implicit() {
        // `ret` (assembles as `ret x30`, but the decoder drops the default reg).
        let insn = decode(0xD65F_03C0);
        assert_eq!(insn.mnemonic(), Mnemonic::Ret);
        let imp = implicit_registers(&insn);
        // `ret` without an operand returns through X30 implicitly.
        if insn.op_count() == 0 {
            assert_eq!(imp.access_of(Register::X30), OpAccess::Read);
        }
        // `ret x5` names its register: nothing implicit.
        let explicit = decode(0xD65F_00A0);
        assert_eq!(explicit.mnemonic(), Mnemonic::Ret);
        assert!(implicit_registers(&explicit).is_empty());
    }

    #[test]
    fn paciasp_touches_lr_and_sp() {
        // `paciasp` (HINT #25).
        assert_implicit(
            0xD503_233F,
            &[
                (Register::X30, OpAccess::ReadWrite),
                (Register::Sp, OpAccess::Read),
            ],
        );
    }

    #[test]
    fn pacia1716_touches_x17_and_x16() {
        // `pacia1716` (HINT #8).
        assert_implicit(
            0xD503_211F,
            &[
                (Register::X17, OpAccess::ReadWrite),
                (Register::X16, OpAccess::Read),
            ],
        );
    }

    #[test]
    fn adr_reads_pc() {
        // `adr x0, #0` — PC-relative address generation.
        assert_implicit(0x1000_0000, &[(Register::Pc, OpAccess::Read)]);
    }

    #[test]
    fn plain_alu_has_no_implicit_state() {
        // `add x0, x1, x2`
        assert_implicit(0x8B02_0020, &[]);
    }

    #[test]
    fn flag_setter_writes_nzcv() {
        // `adds x0, x1, x2`
        assert_implicit(0xAB02_0020, &[(Register::Nzcv, OpAccess::Write)]);
        // `adcs x0, x1, x2` reads the carry and writes the flags.
        assert_implicit(0xBA02_0020, &[(Register::Nzcv, OpAccess::ReadWrite)]);
        // `csel x0, x1, x2, ne` reads the flags only.
        assert_implicit(0x9A82_1020, &[(Register::Nzcv, OpAccess::Read)]);
    }
}
