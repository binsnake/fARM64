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
    /// Packed instruction flags. Bit 0 ([`Instruction::FLAG_MODIFIED`]) records
    /// that an operand/`ip`/`code` mutator has run, so [`Instruction::word`] no
    /// longer describes this value. Kept as a `u8` to hold `Instruction` small.
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

    /// The raw little-endian instruction word this instruction was **decoded
    /// from**.
    ///
    /// After any of the [editing](#editing-and-re-encoding) mutators has run,
    /// this still returns the *original* word and [`Instruction::is_modified`]
    /// returns `true`; call [`Instruction::encode`] for the word the edited
    /// semantics encode to, or [`Instruction::re_encode`] to encode and store it
    /// back here.
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
            // The FEAT_HBC hinted conditional branch `BC.<cond>` is the
            // `bit4 == 1` sibling of `B.<cond>`; it carries its own mnemonic
            // rather than reusing `B`, so it needs its own arm.
            Bc => FlowControl::ConditionalBranch,
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
            // FEAT_PAuth_LR authenticated returns (PC-relative `<label>` and
            // register-modifier `<Xm>` forms) also return from a subroutine.
            Retaasppc | Retabsppc | Retaasppcr | Retabsppcr => FlowControl::Return,
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
            // Flag manipulation: each writes a *subset* of NZCV directly rather
            // than through the ALU or FP compare path, so it reports the generic
            // `Sets`. `RMIF` inserts a rotated bitfield of `Xn` into the mask-
            // selected flags; `SETF8`/`SETF16` set N/Z/V from the low bits of
            // `Wn` (C is preserved); `CFINV` inverts C; `AXFLAG`/`XAFLAG`
            // convert between the Arm and the alternate FP condition-flag
            // encodings. The SVE `RDFFRS` writes NZCV from the FFR predicate.
            Rmif | Setf8 | Setf16 | Cfinv | Axflag | Xaflag | Rdffrs => FlagEffect::Sets,
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

    /// The registers this instruction touches **without naming them in an
    /// operand** — the link register of a call/return, the `X30`/`SP`/`X16`/
    /// `X17` of the pointer-authentication hints, the `Xt+1..Xt+7` of the
    /// FEAT_LS64 64-byte transfers, the SVE `FFR`, the `PC` of PC-relative
    /// address generation, and the `NZCV` flags.
    ///
    /// See [`crate::implicit`] for the full rule set. The *explicit* operands
    /// are not included here; [`crate::info::instruction_info`] merges both into
    /// one [`crate::info::InstructionInfo::used_registers`] list.
    #[inline]
    pub fn implicit_registers(&self) -> crate::implicit::ImplicitRegisters {
        crate::implicit::implicit_registers(self)
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
    /// each operand, and optionally [`Instruction::set_alias`] to install a
    /// preferred-disassembly alias while keeping `code` canonical.
    #[inline]
    pub(crate) fn set(&mut self, code: Code) {
        self.code = code;
        self.mnemonic = code.mnemonic();
        self.op_count = 0;
        self.operands = [Operand::None; MAX_OPERANDS];
    }

    /// Crate-internal: install a preferred-disassembly alias mnemonic while
    /// leaving [`Instruction::code`] as the canonical encoding identity.
    ///
    /// Distinct from the public [`Instruction::set_mnemonic`] editor: this is
    /// part of *decoding* and does not mark the instruction modified.
    #[inline]
    pub(crate) fn set_alias(&mut self, mnemonic: Mnemonic) {
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

// ---------------------------------------------------------------------------
// Editing and re-encoding.
// ---------------------------------------------------------------------------

/// `true` if `reg` is legal for the operand *shape* `op`, beyond the register
/// class and width that [`Instruction::set_op_register`] already checks.
///
/// Most shapes name a whole register file and accept anything of the right
/// class. [`Operand::PredCounter`] is the exception: the predicate-as-counter
/// `PNg` field is three bits wide, so it can only name `P8`..`P15` even though
/// `P0`..`P7` are the same class and width.
#[inline]
fn operand_accepts_register(op: &Operand, reg: Register) -> bool {
    match op {
        Operand::PredCounter { .. } => reg.number() >= 8,
        _ => true,
    }
}

/// # Editing and re-encoding
///
/// Every mutator below edits the instruction's *semantics* in place and marks
/// it [modified](Instruction::is_modified). Because
/// [`Instruction::encode`](crate::encode::encode) rebuilds the 32-bit word from
/// semantics alone — it never reads [`Instruction::word`] — an edit followed by
/// `encode()` yields the word for the *edited* instruction:
///
/// ```
/// use fARM64::{Decoder, DecoderOptions, Register};
///
/// // `add x0, x1, x2`
/// let bytes = 0x8B02_0020u32.to_le_bytes();
/// let mut insn = Decoder::new(&bytes, 0, DecoderOptions::NONE).decode();
///
/// // Retarget the destination to x5 and the second source to x7.
/// assert!(insn.set_op_register(0, Register::X5));
/// assert!(insn.set_op_register(2, Register::X7));
/// assert!(insn.is_modified());
///
/// // `add x5, x1, x7`
/// assert_eq!(insn.encode(), Ok(0x8B07_0025));
/// ```
///
/// ## Rules
///
/// * Every mutator is **total**: it returns `false` (or leaves the value
///   untouched) instead of panicking when the edit does not apply — an
///   out-of-range slot, a wrong operand shape, or a value that does not fit the
///   operand variant.
/// * Register replacement is **class- and width-checked** by default:
///   [`set_op_register`](Instruction::set_op_register) refuses to put a `W`
///   register where an `X` register was, or a `P` register where a `Z` register
///   was, because [`Instruction::code`] — not the operand — carries the operand
///   size. Use
///   [`set_op_register_unchecked`](Instruction::set_op_register_unchecked) to
///   override, or [`set_code`](Instruction::set_code) to move to the encoding
///   that matches.
/// * Operand **decorations are preserved**: replacing the register of
///   `v0.4s[1]` keeps the arrangement and lane; replacing an immediate keeps
///   the immediate's variant (and therefore how the encoder packs it).
/// * The edit is not validated against the encoding. A value with no
///   representation in the instruction's fields surfaces as an
///   [`EncodeError`](crate::EncodeError) from `encode()`, not here.
/// * [`Instruction::word`] keeps returning the word the instruction was decoded
///   from. Use [`re_encode`](Instruction::re_encode) to encode and store the new
///   word in its place.
impl Instruction {
    /// `flags` bit 0: an editing mutator has run, so [`Instruction::word`] no
    /// longer describes this value.
    pub(crate) const FLAG_MODIFIED: u8 = 1 << 0;

    /// `true` if any editing mutator has run since this instruction was
    /// decoded (or since the last [`re_encode`](Instruction::re_encode)).
    ///
    /// When this is `true`, [`Instruction::word`] is still the *original*
    /// decoded word and can no longer be assumed to encode this instruction.
    ///
    /// It is an upper bound, not an exact answer: an operand edit sets it even
    /// in the rare case where the encoding cannot tell the difference (swapping
    /// `SP` for `XZR`, which both encode as register 31). The one case that is
    /// exact is the address — [`set_ip`](Instruction::set_ip) and
    /// [`relocate`](Instruction::relocate) set it only for an instruction whose
    /// encoding actually depends on `ip`, i.e. one carrying an
    /// [`Operand::Label`].
    #[inline]
    pub const fn is_modified(&self) -> bool {
        self.flags & Self::FLAG_MODIFIED != 0
    }

    /// Mark the instruction edited.
    #[inline]
    fn mark_modified(&mut self) {
        self.flags |= Self::FLAG_MODIFIED;
    }

    /// `true` if any operand carries a resolved PC-relative target, which is
    /// exactly when the encoding depends on [`Instruction::ip`].
    #[inline]
    fn has_label(&self) -> bool {
        self.operands[..self.op_count as usize]
            .iter()
            .any(|op| matches!(op, Operand::Label(_)))
    }

    // --- Identity -----------------------------------------------------------

    /// Set the address this instruction sits at.
    ///
    /// Operands that carry a **resolved absolute** target ([`Operand::Label`])
    /// are left alone, so moving an instruction keeps it pointing at the same
    /// address and the encoder re-derives the PC-relative displacement. To move
    /// an instruction and keep its *relative* displacement instead, use
    /// [`relocate`](Instruction::relocate).
    ///
    /// Only a PC-relative instruction's *encoding* depends on `ip`, so this
    /// marks the instruction [modified](Instruction::is_modified) only when it
    /// carries a label. Moving an `ADD` leaves its word — and `is_modified()` —
    /// untouched.
    #[inline]
    pub fn set_ip(&mut self, ip: u64) {
        self.ip = ip;
        if self.has_label() {
            self.mark_modified();
        }
    }

    /// Move this instruction to `ip`, keeping every PC-relative displacement
    /// the same.
    ///
    /// Each [`Operand::Label`] is shifted by `ip - self.ip()`, so a branch that
    /// jumped `+8` still jumps `+8` from the new address. This is the
    /// "copy an instruction elsewhere and keep it self-relative" move;
    /// [`set_ip`](Instruction::set_ip) is the "keep pointing at the same
    /// absolute address" move.
    ///
    /// `ADRP` resolves to a 4 KiB **page**, so only a page-aligned `delta`
    /// keeps it encodable; a finer one leaves a label `ADRP` cannot name and
    /// `encode()` reports
    /// [`EncodeError::InvalidImmediate`](crate::EncodeError::InvalidImmediate).
    /// See [`set_label`](Instruction::set_label).
    #[inline]
    pub fn relocate(&mut self, ip: u64) {
        let delta = ip.wrapping_sub(self.ip);
        let mut moved = false;
        for op in self.operands[..self.op_count as usize].iter_mut() {
            if let Operand::Label(target) = op {
                *target = target.wrapping_add(delta);
                moved = true;
            }
        }
        self.ip = ip;
        // A label-less instruction encodes identically wherever it sits, so
        // moving it does not make `word()` stale.
        if moved {
            self.mark_modified();
        }
    }

    /// Set the encoding identity and reset the mnemonic to that encoding's
    /// default ([`Code::mnemonic`]).
    ///
    /// The operand list is **kept**: this is how you move an instruction to a
    /// sibling encoding that takes the same operand shape (`ADD` to `SUB`, the
    /// 32-bit form to the 64-bit form). The encoder dispatches on
    /// [`Instruction::code`], so the new code decides how the operands are
    /// packed — and rejects them if they do not fit.
    ///
    /// ```
    /// use fARM64::{Code, Decoder, DecoderOptions};
    ///
    /// // `add x0, x1, x2` -> `sub x0, x1, x2`
    /// let bytes = 0x8B02_0020u32.to_le_bytes();
    /// let mut insn = Decoder::new(&bytes, 0, DecoderOptions::NONE).decode();
    /// insn.set_code(Code::SubShifted64);
    /// assert_eq!(insn.encode(), Ok(0xCB02_0020));
    /// ```
    #[inline]
    pub fn set_code(&mut self, code: Code) {
        self.code = code;
        self.mnemonic = code.mnemonic();
        self.mark_modified();
    }

    /// Override the displayed mnemonic, leaving [`Instruction::code`] — the
    /// canonical encoding identity the encoder dispatches on — unchanged.
    ///
    /// This selects between the aliases of one encoding (`SUBS XZR, Xn, Xm`
    /// spelled as `CMP`, `ORR Xd, XZR, Xm` spelled as `MOV`). The encoder
    /// inverts the alias it is given, so the mnemonic must match the operand
    /// list actually present; to change the *encoding*, use
    /// [`set_code`](Instruction::set_code).
    #[inline]
    pub fn set_mnemonic(&mut self, mnemonic: Mnemonic) {
        self.mnemonic = mnemonic;
        self.mark_modified();
    }

    // --- Operands -----------------------------------------------------------

    /// Replace operand `n` wholesale.
    ///
    /// `n` may be any slot below [`MAX_OPERANDS`]; writing past the current
    /// [`op_count`](Instruction::op_count) extends the operand list (the slots
    /// in between become [`Operand::None`]). Returns `false` — changing
    /// nothing — if `n >= MAX_OPERANDS`.
    #[inline]
    pub fn set_op(&mut self, n: usize, op: Operand) -> bool {
        if n >= MAX_OPERANDS {
            return false;
        }
        self.operands[n] = op;
        if n >= self.op_count as usize {
            self.op_count = (n + 1) as u8;
        }
        self.mark_modified();
        true
    }

    /// Set the number of meaningful operands.
    ///
    /// Truncating clears the dropped slots to [`Operand::None`]; growing
    /// exposes [`Operand::None`] slots for [`set_op`](Instruction::set_op) to
    /// fill. Returns `false` if `count > MAX_OPERANDS`.
    #[inline]
    pub fn set_op_count(&mut self, count: usize) -> bool {
        if count > MAX_OPERANDS {
            return false;
        }
        for op in self.operands[count..].iter_mut() {
            *op = Operand::None;
        }
        self.op_count = count as u8;
        self.mark_modified();
        true
    }

    /// Append `op` after the last operand.
    ///
    /// Returns `false` — changing nothing — if the operand list is already
    /// [`MAX_OPERANDS`] long.
    #[inline]
    pub fn push_op(&mut self, op: Operand) -> bool {
        let i = self.op_count as usize;
        if i >= MAX_OPERANDS {
            return false;
        }
        self.operands[i] = op;
        self.op_count = (i + 1) as u8;
        self.mark_modified();
        true
    }

    /// Replace the register of operand `n`, keeping every decoration
    /// (arrangement, lane, shift, extend, predicate qualifier).
    ///
    /// `reg` must be the same [register class](crate::RegClass) and
    /// [width](Register::width_bits) as the register it replaces — `X5` for
    /// `X1`, `V7` for `V0`, `P3` for `P0` — because the operand *size* lives in
    /// [`Instruction::code`], not in the operand. Operand shapes that can only
    /// name part of a register file are checked against that range too: the
    /// predicate-as-counter [`Operand::PredCounter`] has a 3-bit `PNg` field and
    /// accepts only `P8`..`P15`. A mismatch returns `false` and changes nothing;
    /// use
    /// [`set_op_register_unchecked`](Instruction::set_op_register_unchecked) if
    /// you are deliberately changing class or width alongside
    /// [`set_code`](Instruction::set_code).
    ///
    /// Covers the single-register operand shapes ([`Operand::Reg`],
    /// [`Operand::RegBang`], [`Operand::IndexedElement`],
    /// [`Operand::PredCounter`]). Register *lists* and pairs
    /// ([`Operand::MultiReg`], [`Operand::RegPair`], [`Operand::SveVecGroup`])
    /// are ambiguous and return `false` — rebuild them with
    /// [`set_op`](Instruction::set_op). Memory bases and indices have their own
    /// setters ([`set_memory_base`](Instruction::set_memory_base) /
    /// [`set_memory_index`](Instruction::set_memory_index)).
    #[inline]
    pub fn set_op_register(&mut self, n: usize, reg: Register) -> bool {
        let op = self.op(n);
        let old = match self.op_register_component(n) {
            Some(r) => r,
            None => return false,
        };
        if old.class() != reg.class() || old.width_bits() != reg.width_bits() {
            return false;
        }
        if !operand_accepts_register(&op, reg) {
            return false;
        }
        self.set_op_register_unchecked(n, reg)
    }

    /// Replace the register of operand `n` **without** the class/width check of
    /// [`set_op_register`](Instruction::set_op_register).
    ///
    /// Use this when the operand size is changing too — pair it with
    /// [`set_code`](Instruction::set_code) so the encoding agrees. Still returns
    /// `false` for operand shapes that carry no single replaceable register.
    #[inline]
    pub fn set_op_register_unchecked(&mut self, n: usize, reg: Register) -> bool {
        if n >= self.op_count as usize {
            return false;
        }
        let ok = match &mut self.operands[n] {
            Operand::Reg { reg: r, .. } => {
                *r = reg;
                true
            }
            Operand::RegBang(r) => {
                *r = reg;
                true
            }
            Operand::IndexedElement { reg: r, .. } => {
                *r = reg;
                true
            }
            Operand::PredCounter { reg: r, .. } => {
                *r = reg;
                true
            }
            _ => false,
        };
        if ok {
            self.mark_modified();
        }
        ok
    }

    /// The single register carried by operand `n`, for the shapes
    /// [`set_op_register`](Instruction::set_op_register) can edit.
    ///
    /// See also [`operand_accepts_register`], which decides whether a *new*
    /// register is legal for that shape.
    #[inline]
    fn op_register_component(&self, n: usize) -> Option<Register> {
        match self.op(n) {
            Operand::Reg { reg, .. }
            | Operand::RegBang(reg)
            | Operand::IndexedElement { reg, .. }
            | Operand::PredCounter { reg, .. } => Some(reg),
            _ => None,
        }
    }

    /// Replace the value of the immediate at operand `n`, keeping its
    /// [`Operand`] variant.
    ///
    /// The variant decides how the encoder packs the value (a bitmask immediate
    /// is not packed like an unsigned one), so it is deliberately preserved:
    /// [`Operand::ImmUnsigned`], [`Operand::ImmSigned`],
    /// [`Operand::ImmLogical`], [`Operand::ImmSignedDec`],
    /// [`Operand::ShiftAmount`], [`Operand::Label`] and the shifted-move
    /// immediates all keep their kind. Signed variants reinterpret `value` via
    /// `as i64`.
    ///
    /// Returns `false` — changing nothing — if operand `n` is not an immediate,
    /// or if `value` does not fit the variant's payload (an
    /// [`Operand::ShiftAmount`] above `255`, an
    /// [`Operand::ImmShiftedMove`] above `0xFFFF`). To change the *kind* of an
    /// operand, or to set an [`Operand::FpImm`], use
    /// [`set_op`](Instruction::set_op).
    #[inline]
    pub fn set_op_immediate(&mut self, n: usize, value: u64) -> bool {
        if n >= self.op_count as usize {
            return false;
        }
        let ok = match &mut self.operands[n] {
            Operand::ImmUnsigned(v) | Operand::ImmLogical(v) | Operand::Label(v) => {
                *v = value;
                true
            }
            Operand::ImmSigned(v) | Operand::ImmSignedDec(v) => {
                *v = value as i64;
                true
            }
            Operand::ShiftAmount(v) => match u8::try_from(value) {
                Ok(x) => {
                    *v = x;
                    true
                }
                Err(_) => false,
            },
            Operand::ImmShiftedMove { imm, .. } | Operand::ImmShiftedMsl { imm, .. } => {
                match u16::try_from(value) {
                    Ok(x) => {
                        *imm = x;
                        true
                    }
                    Err(_) => false,
                }
            }
            _ => false,
        };
        if ok {
            self.mark_modified();
        }
        ok
    }

    /// Replace the condition code of the [`Operand::Cond`] operand.
    ///
    /// Returns `false` if this instruction carries no condition operand — which
    /// includes the families that fuse the condition into the [`Code`] instead
    /// (the FEAT_CMPBR `CB<cc>` compare-and-branch forms); change those with
    /// [`set_code`](Instruction::set_code).
    #[inline]
    pub fn set_condition(&mut self, cond: Condition) -> bool {
        for op in self.operands[..self.op_count as usize].iter_mut() {
            if let Operand::Cond(c) = op {
                *c = cond;
                self.mark_modified();
                return true;
            }
        }
        false
    }

    /// Retarget a direct branch, as an **absolute** address.
    ///
    /// The setter counterpart of
    /// [`near_branch_target`](Instruction::near_branch_target): it applies only
    /// to the direct branch / direct call forms that carry a resolved target
    /// (`B`, `BL`, `B.<cond>`, `CBZ`/`CBNZ`, `TBZ`/`TBNZ`, the FEAT_CMPBR
    /// `CB<cc>` family), and returns `false` for everything else. The encoder
    /// re-derives the PC-relative displacement from
    /// [`ip`](Instruction::ip), and reports
    /// [`EncodeError::InvalidImmediate`](crate::EncodeError::InvalidImmediate)
    /// if the new target is out of range or misaligned for the encoding.
    ///
    /// Use [`set_label`](Instruction::set_label) for the non-branch labels of
    /// `ADR`/`ADRP` and the PC-relative literal loads.
    #[inline]
    pub fn set_near_branch_target(&mut self, target: u64) -> bool {
        match self.flow_control() {
            FlowControl::ConditionalBranch
            | FlowControl::UnconditionalBranch
            | FlowControl::Call => self.set_label(target),
            _ => false,
        }
    }

    /// Retarget the [`Operand::Label`] of this instruction, as an **absolute**
    /// address.
    ///
    /// Covers every label-bearing form, branches included: `ADR`/`ADRP`, the
    /// PC-relative literal loads, and the direct branches. Returns `false` if
    /// there is no label operand.
    ///
    /// The target must be representable at the encoding's granularity — `ADRP`
    /// addresses a 4 KiB **page**, the branches are 4-byte aligned — and a
    /// target that is not reports
    /// [`EncodeError::InvalidImmediate`](crate::EncodeError::InvalidImmediate)
    /// from [`encode`](Instruction::encode). A label that encodes at all always
    /// encodes exactly; no form silently rounds the target it was given.
    #[inline]
    pub fn set_label(&mut self, target: u64) -> bool {
        for op in self.operands[..self.op_count as usize].iter_mut() {
            if let Operand::Label(t) = op {
                *t = target;
                self.mark_modified();
                return true;
            }
        }
        false
    }

    /// Replace the base register of this instruction's memory operand.
    ///
    /// Applies to the first [`Operand::MemImm`] / [`Operand::MemExt`] /
    /// [`Operand::SveMem`] operand — the counterpart of
    /// [`memory_base`](Instruction::memory_base) — keeping the addressing mode,
    /// displacement, index and extend. Returns `false` if there is no memory
    /// operand.
    ///
    /// Not class-checked: an A64 memory base is always `Xn|SP` (or, for the
    /// SVE vector-base modes, a `Zn`), so pass the register the mode expects.
    #[inline]
    pub fn set_memory_base(&mut self, reg: Register) -> bool {
        for op in self.operands[..self.op_count as usize].iter_mut() {
            match op {
                Operand::MemImm { base, .. }
                | Operand::MemExt { base, .. }
                | Operand::SveMem { base, .. } => {
                    *base = reg;
                    self.mark_modified();
                    return true;
                }
                _ => {}
            }
        }
        false
    }

    /// Replace the index register of a register-offset memory operand.
    ///
    /// Applies to the first [`Operand::MemExt`] (or the offset register of an
    /// [`Operand::SveMem`]) — the counterpart of
    /// [`memory_index`](Instruction::memory_index) — keeping the extend and
    /// shift. Returns `false` for immediate-offset forms, which have no index.
    #[inline]
    pub fn set_memory_index(&mut self, reg: Register) -> bool {
        for op in self.operands[..self.op_count as usize].iter_mut() {
            match op {
                Operand::MemExt { index, .. } | Operand::SveMem { offset: index, .. } => {
                    *index = reg;
                    self.mark_modified();
                    return true;
                }
                _ => {}
            }
        }
        false
    }

    /// Replace the immediate displacement of this instruction's memory operand.
    ///
    /// Applies to the first [`Operand::MemImm`] (or [`Operand::SveMem`], whose
    /// displacement is 32-bit) — the counterpart of
    /// [`memory_displacement64`](Instruction::memory_displacement64) — keeping
    /// the addressing mode. Returns `false` if there is no immediate-offset
    /// memory operand, or if `disp` does not fit an [`Operand::SveMem`]'s
    /// 32-bit displacement.
    ///
    /// Scaling is the encoder's job: pass the **byte** displacement exactly as
    /// [`memory_displacement64`](Instruction::memory_displacement64) reports
    /// it. A displacement the encoding cannot represent surfaces as
    /// [`EncodeError::InvalidImmediate`](crate::EncodeError::InvalidImmediate).
    #[inline]
    pub fn set_memory_displacement64(&mut self, disp: i64) -> bool {
        for op in self.operands[..self.op_count as usize].iter_mut() {
            match op {
                Operand::MemImm { imm, .. } => {
                    *imm = disp;
                    self.mark_modified();
                    return true;
                }
                Operand::SveMem { imm, .. } => match i32::try_from(disp) {
                    Ok(v) => {
                        *imm = v;
                        self.mark_modified();
                        return true;
                    }
                    Err(_) => return false,
                },
                _ => {}
            }
        }
        false
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
