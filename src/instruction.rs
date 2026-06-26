//! The `Copy` value-type [`Instruction`] — the public projection of a decoded
//! A64 instruction.
//!
//! A `Copy` value type holding an inline `[Operand; MAX_OPERANDS]` with no heap
//! allocation and no internal pointers. Its size is dominated by that inline
//! operand array (`5 * 16 = 80` bytes) plus a small header; the realized ceiling
//! is asserted at `<= 112` bytes in `lib.rs`'s `static_asserts`. The fat
//! internal decode representation never reaches this type.

use crate::enums::{Condition, FlagEffect, FlowControl};
use crate::mnemonic::{Code, Mnemonic};
use crate::operand::{OpKind, Operand};
use crate::register::Register;
use crate::{INSN_LEN, MAX_OPERANDS};

/// A fully decoded AArch64 instruction.
///
/// Construct via [`crate::Decoder::decode`] / [`crate::Decoder::decode_into`].
/// All accessors are cheap; the type is `Copy` and safe to pass by value.
///
/// Derives `PartialEq` but not `Eq`/`Hash`, because its inline
/// `[Operand; MAX_OPERANDS]` contains a floating-point ([`Operand::FpImm`])
/// payload. Compare by [`Instruction::word`] + [`Instruction::ip`] if a total
/// key is needed.
///
/// [`Operand::FpImm`]: crate::operand::Operand::FpImm
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Instruction {
    /// The raw little-endian instruction word as decoded.
    pub(crate) word: u32,
    /// Address of this instruction (the `ip` it was decoded at).
    pub(crate) ip: u64,
    /// Encoding-level identity.
    pub(crate) code: Code,
    /// Resolved mnemonic (may be an alias when alias resolution is enabled).
    pub(crate) mnemonic: Mnemonic,
    /// Number of valid entries in `operands`.
    pub(crate) op_count: u8,
    /// Packed instruction flags (flow-control class, flag effect, alias-applied,
    /// ...), laid out by codegen. Kept as a `u8` to hold `Instruction` small.
    pub(crate) flags: u8,
    /// Inline operand storage; only `op_count` entries are meaningful.
    pub(crate) operands: [Operand; MAX_OPERANDS],
}

impl Instruction {
    /// The encoding identity.
    #[inline]
    pub const fn code(&self) -> Code {
        self.code
    }

    /// The (possibly alias-resolved) mnemonic.
    #[inline]
    pub const fn mnemonic(&self) -> Mnemonic {
        self.mnemonic
    }

    /// Number of explicit operands (`0..=MAX_OPERANDS`).
    #[inline]
    pub const fn op_count(&self) -> usize {
        self.op_count as usize
    }

    /// The [`OpKind`] discriminant of operand `n`. Out-of-range `n` yields
    /// [`OpKind::None`].
    #[inline]
    pub fn op_kind(&self, n: usize) -> OpKind {
        if n < self.op_count as usize {
            self.operands[n].kind()
        } else {
            OpKind::None
        }
    }

    /// The full rich [`Operand`] at slot `n`. Out-of-range `n` yields
    /// [`Operand::None`].
    #[inline]
    pub fn op(&self, n: usize) -> Operand {
        if n < self.op_count as usize {
            self.operands[n]
        } else {
            Operand::None
        }
    }

    /// Fast indexed accessor: the register of operand `n`, or [`Register::None`]
    /// if it is not a plain register operand.
    #[inline]
    pub fn op_register(&self, n: usize) -> Register {
        match self.op(n) {
            Operand::Reg { reg, .. } => reg,
            _ => Register::None,
        }
    }

    /// Fast indexed accessor: the immediate value of operand `n` as `u64`
    /// (signed immediates are reinterpreted via `as u64`). `0` if operand `n`
    /// is not an immediate.
    #[inline]
    pub fn op_immediate(&self, n: usize) -> u64 {
        match self.op(n) {
            Operand::ImmUnsigned(v) | Operand::ImmLogical(v) => v,
            Operand::ImmSigned(v) => v as u64,
            Operand::Label(v) => v,
            _ => 0,
        }
    }

    /// Length of this instruction in bytes. A64 is fixed-width: always
    /// [`INSN_LEN`] (4).
    #[inline]
    pub const fn len(&self) -> usize {
        INSN_LEN
    }

    /// Always `false` — an A64 instruction is never zero-length. Present so
    /// Clippy does not demand it alongside [`Instruction::len`].
    #[inline]
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// The address this instruction was decoded at.
    #[inline]
    pub const fn ip(&self) -> u64 {
        self.ip
    }

    /// The address of the following instruction (`ip + 4`).
    #[inline]
    pub const fn next_ip(&self) -> u64 {
        self.ip.wrapping_add(INSN_LEN as u64)
    }

    /// The raw little-endian instruction word.
    #[inline]
    pub const fn word(&self) -> u32 {
        self.word
    }

    /// Control-flow classification (derived from [`Code`]/[`Mnemonic`]).
    ///
    /// Classifies from the (alias-resolved) [`Mnemonic`], with the one
    /// disambiguation that needs the encoding: `B` is a
    /// [`FlowControl::ConditionalBranch`] only for the `B.<cond>` encoding
    /// ([`Code::BCond`]) and otherwise a [`FlowControl::UnconditionalBranch`].
    #[inline]
    pub fn flow_control(&self) -> FlowControl {
        use Mnemonic::*;
        match self.mnemonic {
            // Direct unconditional / conditional branches.
            B => {
                if matches!(self.code, Code::BCond) {
                    FlowControl::ConditionalBranch
                } else {
                    FlowControl::UnconditionalBranch
                }
            }
            // Compare/test-and-branch are conditional direct branches.
            Cbz | Cbnz | Tbz | Tbnz => FlowControl::ConditionalBranch,
            // FEAT_CMPBR compare-and-branch (register / immediate) — also
            // conditional direct branches.
            Cbgt | Cbge | Cbhi | Cbhs | Cbeq | Cbne | Cblt | Cblo | Cbbgt | Cbbge | Cbbhi
            | Cbbhs | Cbbeq | Cbbne | Cbhgt | Cbhge | Cbhhi | Cbhhs | Cbheq | Cbhne => {
                FlowControl::ConditionalBranch
            }
            // Direct call (writes the link register).
            Bl => FlowControl::Call,
            // Indirect call via register (incl. pointer-authenticated forms).
            Blr | Blraa | Blraaz | Blrab | Blrabz => FlowControl::IndirectCall,
            // Indirect branch via register (incl. pointer-authenticated forms).
            Br | Braa | Braaz | Brab | Brabz => FlowControl::IndirectBranch,
            // Returns from subroutine / exception.
            Ret | Retaa | Retab | Eret | Eretaa | Eretab | Drps => FlowControl::Return,
            // Exception generation / system calls / debug-state entry.
            Svc | Hvc | Smc | Brk | Hlt | Dcps1 | Dcps2 | Dcps3 => FlowControl::Exception,
            // Everything else falls through linearly.
            _ => FlowControl::Next,
        }
    }

    /// NZCV write behaviour.
    ///
    /// Integer flag-setters (the `S`-suffixed ALU forms plus `CMP`/`CMN`/`TST`
    /// and the conditional-compare `CCMP`/`CCMN`) report
    /// [`FlagEffect::SetsNormal`]; the floating-point compares
    /// (`FCMP`/`FCMPE`/`FCCMP`/`FCCMPE`) report [`FlagEffect::SetsFloat`]. All
    /// other instructions report [`FlagEffect::None`].
    #[inline]
    pub fn set_flags(&self) -> FlagEffect {
        use Mnemonic::*;
        match self.mnemonic {
            // Integer ALU forms that write NZCV.
            Adds | Subs | Adcs | Sbcs | Ands | Bics | Negs | Ngcs | Cmp | Cmn | Tst | Ccmp
            | Ccmn => FlagEffect::SetsNormal,
            // Floating-point comparisons write NZCV from the FP compare path.
            Fcmp | Fcmpe | Fccmp | Fccmpe => FlagEffect::SetsFloat,
            // ADC/SBC/CSEL/... read flags but do not write them.
            _ => FlagEffect::None,
        }
    }

    /// The resolved absolute target of a direct (PC-relative) branch.
    ///
    /// Returns the pre-resolved [`Operand::Label`] target (`ip + offset`) for the
    /// direct branches — `B`, `BL`, `B.<cond>`/`BC.<cond>`, `CBZ`/`CBNZ`,
    /// `TBZ`/`TBNZ`, and the FEAT_CMPBR `CB<cc>` compare-and-branch family — and
    /// `0` for every other instruction. The indirect register branches
    /// (`BR`/`BLR`/`RET`/...) encode no target and return `0`, as do non-branch
    /// instructions that merely carry a label (`ADR`/`ADRP`, literal loads).
    /// Mirrors iced-x86's `near_branch_target`.
    #[inline]
    pub fn near_branch_target(&self) -> u64 {
        // Direct branches are exactly the conditional / unconditional / direct-call
        // flow classes; each carries one resolved `Label` operand.
        match self.flow_control() {
            FlowControl::ConditionalBranch
            | FlowControl::UnconditionalBranch
            | FlowControl::Call => {
                for op in &self.operands[..self.op_count as usize] {
                    if let Operand::Label(target) = *op {
                        return target;
                    }
                }
                0
            }
            _ => 0,
        }
    }

    /// The condition code governing this instruction, if it has one.
    ///
    /// Covers every conditional form that carries the condition as an explicit
    /// [`Operand::Cond`] — the integer conditional-select / conditional-compare
    /// family (`CSEL`/`CSINC`/`CSINV`/`CSNEG` and the `CSET*`/`CINC`/`CINV`/`CNEG`
    /// aliases, `CCMP`/`CCMN`), the floating-point conditional forms
    /// (`FCSEL`/`FCCMP`/`FCCMPE`), and the direct conditional branches
    /// `B.<cond>`/`BC.<cond>` — and additionally recovers the condition fused into
    /// the mnemonic of the FEAT_CMPBR `CB<cc>` compare-and-branch family (whose
    /// `cc` is encoded in the [`Code`], not as an operand). Returns `None` for
    /// instructions that have no condition.
    #[inline]
    pub fn condition(&self) -> Option<Condition> {
        // The common case: the condition is an explicit operand.
        for op in &self.operands[..self.op_count as usize] {
            if let Operand::Cond(c) = *op {
                return Some(c);
            }
        }
        // The FEAT_CMPBR `CB<cc>` family fuses the condition into the mnemonic.
        use Mnemonic::*;
        let c = match self.mnemonic {
            Cbgt | Cbbgt | Cbhgt => Condition::Gt,
            Cbge | Cbbge | Cbhge => Condition::Ge,
            Cbhi | Cbbhi | Cbhhi => Condition::Hi,
            Cbhs | Cbbhs | Cbhhs => Condition::Cs,
            Cbeq | Cbbeq | Cbheq => Condition::Eq,
            Cbne | Cbbne | Cbhne => Condition::Ne,
            Cblt => Condition::Lt,
            Cblo => Condition::Cc,
            _ => return None,
        };
        Some(c)
    }

    /// The base register of this instruction's memory operand, or
    /// [`Register::None`] if it has no [`Operand::MemImm`] / [`Operand::MemExt`]
    /// memory operand. Mirrors iced-x86's `memory_base`.
    #[inline]
    pub fn memory_base(&self) -> Register {
        for op in &self.operands[..self.op_count as usize] {
            if let Operand::MemImm { base, .. } | Operand::MemExt { base, .. } = *op {
                return base;
            }
        }
        Register::None
    }

    /// The index register of a register-offset memory operand
    /// ([`Operand::MemExt`]), or [`Register::None`] otherwise. Immediate-offset
    /// ([`Operand::MemImm`]) forms have no index. Mirrors iced-x86's
    /// `memory_index`.
    #[inline]
    pub fn memory_index(&self) -> Register {
        for op in &self.operands[..self.op_count as usize] {
            if let Operand::MemExt { index, .. } = *op {
                return index;
            }
        }
        Register::None
    }

    /// The left-shift amount applied to the index register of a register-offset
    /// memory operand ([`Operand::MemExt`]); the effective multiplier is
    /// `1 << memory_index_scale()`. `0` when there is no register index.
    ///
    /// AArch64 encodes index scaling as a shift, so this returns the shift amount
    /// rather than the multiplier (the name keeps the iced-x86 `_scale` lineage).
    #[inline]
    pub fn memory_index_scale(&self) -> u32 {
        for op in &self.operands[..self.op_count as usize] {
            if let Operand::MemExt { shift, .. } = *op {
                // The decoder packs a formatter "show amount" flag into bit 7 of
                // `shift`; the actual left-shift amount is the low 7 bits.
                return (shift & 0x7f) as u32;
            }
        }
        0
    }

    /// The immediate displacement of this instruction's memory operand, as a
    /// signed byte offset. Returns the [`Operand::MemImm`] displacement; `0` for
    /// register-offset ([`Operand::MemExt`]) forms and for instructions with no
    /// memory operand. Mirrors iced-x86's `memory_displacement64`.
    #[inline]
    pub fn memory_displacement64(&self) -> i64 {
        for op in &self.operands[..self.op_count as usize] {
            if let Operand::MemImm { imm, .. } = *op {
                return imm;
            }
        }
        0
    }

    /// `true` if this is the invalid sentinel ([`Code::Invalid`]); check
    /// [`crate::Decoder::last_error`] for the reason.
    #[inline]
    pub fn is_invalid(&self) -> bool {
        matches!(self.code, Code::Invalid)
    }

    /// Crate-internal: an invalid instruction that nonetheless remembers the
    /// `word` and `ip` it was decoded from.
    ///
    /// Used by the hand-written decode tree to seed `out` before a group decoder
    /// fills it in, so that reserved / unallocated encodings still report the
    /// correct address and raw word via [`Instruction::word`] /
    /// [`Instruction::ip`] while remaining [`Code::Invalid`].
    #[inline]
    pub(crate) const fn new_invalid(word: u32, ip: u64) -> Self {
        Instruction {
            word,
            ip,
            code: Code::Invalid,
            mnemonic: Mnemonic::Invalid,
            op_count: 0,
            flags: 0,
            operands: [Operand::None; MAX_OPERANDS],
        }
    }

    /// Crate-internal: set the encoding `code` and its default [`Mnemonic`]
    /// (`code.mnemonic()`), and reset the operand list to empty.
    ///
    /// Group decoders call this first, then [`Instruction::push_operand`] for
    /// each operand, and optionally [`Instruction::set_mnemonic`] to install a
    /// preferred-disassembly alias while keeping `code` canonical.
    #[inline]
    pub(crate) fn set(&mut self, code: Code) {
        self.code = code;
        self.mnemonic = code.mnemonic();
        self.op_count = 0;
        self.operands = [Operand::None; MAX_OPERANDS];
    }

    /// Crate-internal: override the (alias-resolved) mnemonic while leaving
    /// [`Instruction::code`] as the canonical encoding identity.
    #[inline]
    pub(crate) fn set_mnemonic(&mut self, mnemonic: Mnemonic) {
        self.mnemonic = mnemonic;
    }

    /// Crate-internal: append `op` to the operand list (saturating at
    /// [`MAX_OPERANDS`]; excess operands are dropped, never panicking).
    #[inline]
    pub(crate) fn push_operand(&mut self, op: Operand) {
        let i = self.op_count as usize;
        if i < MAX_OPERANDS {
            self.operands[i] = op;
            self.op_count = (i + 1) as u8;
        }
    }
}

impl Default for Instruction {
    /// The invalid/empty instruction: [`Code::Invalid`], no operands, `ip == 0`.
    #[inline]
    fn default() -> Self {
        Instruction {
            word: 0,
            ip: 0,
            code: Code::Invalid,
            mnemonic: Mnemonic::Invalid,
            op_count: 0,
            flags: 0,
            operands: [Operand::None; MAX_OPERANDS],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal instruction with a chosen code + mnemonic for
    /// classification tests (operands are irrelevant here).
    fn insn(code: Code, mnemonic: Mnemonic) -> Instruction {
        let mut i = Instruction::new_invalid(0, 0);
        i.code = code;
        i.mnemonic = mnemonic;
        i
    }

    #[test]
    fn flow_control_branches() {
        // B.<cond> is conditional; plain B is unconditional.
        assert_eq!(
            insn(Code::BCond, Mnemonic::B).flow_control(),
            FlowControl::ConditionalBranch
        );
        assert_eq!(
            insn(Code::BUncond, Mnemonic::B).flow_control(),
            FlowControl::UnconditionalBranch
        );
        assert_eq!(
            insn(Code::Cbz64, Mnemonic::Cbz).flow_control(),
            FlowControl::ConditionalBranch
        );
        assert_eq!(
            insn(Code::Tbnz, Mnemonic::Tbnz).flow_control(),
            FlowControl::ConditionalBranch
        );
    }

    #[test]
    fn flow_control_calls_and_returns() {
        assert_eq!(
            insn(Code::BlImm, Mnemonic::Bl).flow_control(),
            FlowControl::Call
        );
        assert_eq!(
            insn(Code::Blr, Mnemonic::Blr).flow_control(),
            FlowControl::IndirectCall
        );
        assert_eq!(
            insn(Code::Br, Mnemonic::Br).flow_control(),
            FlowControl::IndirectBranch
        );
        assert_eq!(
            insn(Code::Ret, Mnemonic::Ret).flow_control(),
            FlowControl::Return
        );
    }

    #[test]
    fn flow_control_exceptions_and_next() {
        assert_eq!(
            insn(Code::Invalid, Mnemonic::Svc).flow_control(),
            FlowControl::Exception
        );
        assert_eq!(
            insn(Code::Invalid, Mnemonic::Brk).flow_control(),
            FlowControl::Exception
        );
        // A plain ALU op falls through.
        assert_eq!(
            insn(Code::Invalid, Mnemonic::Add).flow_control(),
            FlowControl::Next
        );
    }

    #[test]
    fn set_flags_classification() {
        // S-suffixed and compare/test forms set NZCV via the integer path.
        for m in [
            Mnemonic::Adds,
            Mnemonic::Subs,
            Mnemonic::Adcs,
            Mnemonic::Sbcs,
            Mnemonic::Ands,
            Mnemonic::Bics,
            Mnemonic::Cmp,
            Mnemonic::Cmn,
            Mnemonic::Tst,
            Mnemonic::Ccmp,
            Mnemonic::Ccmn,
        ] {
            assert_eq!(
                insn(Code::Invalid, m).set_flags(),
                FlagEffect::SetsNormal,
                "{m:?} should set NZCV"
            );
        }
        // FP compares set NZCV via the FP path.
        for m in [
            Mnemonic::Fcmp,
            Mnemonic::Fcmpe,
            Mnemonic::Fccmp,
            Mnemonic::Fccmpe,
        ] {
            assert_eq!(insn(Code::Invalid, m).set_flags(), FlagEffect::SetsFloat);
        }
        // ADC reads flags but does not write them; ADD writes none.
        assert_eq!(
            insn(Code::Invalid, Mnemonic::Adc).set_flags(),
            FlagEffect::None
        );
        assert_eq!(
            insn(Code::Invalid, Mnemonic::Add).set_flags(),
            FlagEffect::None
        );
    }

    /// Decode a single 32-bit word at `ip` through the public decoder (all
    /// features on). Encodings below are cross-checked against `llvm-mc`.
    fn decode(word: u32, ip: u64) -> Instruction {
        let bytes = word.to_le_bytes();
        let mut dec = crate::Decoder::new(&bytes, ip, crate::DecoderOptions::NONE);
        dec.decode()
    }

    #[test]
    fn bcond_near_branch_and_condition() {
        // `b.eq #8` @ 0x1000 -> target 0x1008, condition EQ.
        let i = decode(0x5400_0040, 0x1000);
        assert_eq!(i.code(), Code::BCond);
        assert_eq!(i.mnemonic(), Mnemonic::B);
        assert_eq!(i.condition(), Some(Condition::Eq));
        assert_eq!(i.near_branch_target(), 0x1008);
        assert_eq!(i.flow_control(), FlowControl::ConditionalBranch);
    }

    #[test]
    fn uncond_branch_targets() {
        // `b #4` @ 0.
        let b = decode(0x1400_0001, 0);
        assert_eq!(b.code(), Code::BUncond);
        assert_eq!(b.near_branch_target(), 4);
        assert_eq!(b.condition(), None);
        // `bl #4` @ 0 (direct call still has a near-branch target).
        let bl = decode(0x9400_0001, 0);
        assert_eq!(bl.mnemonic(), Mnemonic::Bl);
        assert_eq!(bl.near_branch_target(), 4);
        assert_eq!(bl.condition(), None);
    }

    #[test]
    fn cbz_near_branch_no_condition() {
        // `cbz x0, #8` @ 0x2000.
        let i = decode(0xB400_0040, 0x2000);
        assert_eq!(i.mnemonic(), Mnemonic::Cbz);
        assert_eq!(i.near_branch_target(), 0x2008);
        // CBZ/CBNZ test against zero — they carry no condition code.
        assert_eq!(i.condition(), None);
    }

    #[test]
    fn cmpbr_condition_recovered_from_mnemonic() {
        // `cbgt w2, w1, #4` @ 0 (FEAT_CMPBR register form): the cc is fused into
        // the Code/mnemonic, not an operand.
        let i = decode(0x7401_0022, 0);
        assert_eq!(i.mnemonic(), Mnemonic::Cbgt);
        assert_eq!(i.condition(), Some(Condition::Gt));
        assert_eq!(i.near_branch_target(), 4);
    }

    #[test]
    fn csel_condition_operand() {
        // `csel x0, x1, x2, ne` — cc carried as an Operand::Cond.
        let i = decode(0x9A82_1020, 0);
        assert_eq!(i.mnemonic(), Mnemonic::Csel);
        assert_eq!(i.condition(), Some(Condition::Ne));
        assert_eq!(i.near_branch_target(), 0);
    }

    #[test]
    fn memory_imm_offset_projection() {
        // `ldr x0, [x1, #8]`.
        let i = decode(0xF940_0420, 0);
        assert_eq!(i.mnemonic(), Mnemonic::Ldr);
        assert_eq!(i.memory_base(), Register::X1);
        assert_eq!(i.memory_displacement64(), 8);
        assert_eq!(i.memory_index(), Register::None);
        assert_eq!(i.memory_index_scale(), 0);
    }

    #[test]
    fn memory_reg_offset_projection() {
        // `ldr x0, [x1, x2, lsl #3]`.
        let i = decode(0xF862_7820, 0);
        assert_eq!(i.mnemonic(), Mnemonic::Ldr);
        assert_eq!(i.memory_base(), Register::X1);
        assert_eq!(i.memory_index(), Register::X2);
        assert_eq!(i.memory_index_scale(), 3);
        // Register-offset forms have no immediate displacement.
        assert_eq!(i.memory_displacement64(), 0);
    }

    #[test]
    fn non_memory_non_branch_projections_are_inert() {
        // `add x0, x1, x2` — no memory operand, no condition, no branch target.
        let i = decode(0x8B02_0020, 0);
        assert_eq!(i.memory_base(), Register::None);
        assert_eq!(i.memory_index(), Register::None);
        assert_eq!(i.memory_index_scale(), 0);
        assert_eq!(i.memory_displacement64(), 0);
        assert_eq!(i.condition(), None);
        assert_eq!(i.near_branch_target(), 0);
    }
}
