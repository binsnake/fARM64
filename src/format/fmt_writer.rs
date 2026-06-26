//! The default `no_std`, zero-alloc UAL formatter ([`FmtFormatter`]) and the
//! fixed `&mut [u8]` sink ([`BufSink`]).
//!
//! [`FmtFormatter`] writes through a [`FormatterOutput`]; with [`BufSink`] the
//! entire format path touches no heap. Integer/hex emission is done without
//! allocation. This default output is what the differential oracles compare
//! against during development.
//!
//! # Rendering policy
//!
//! Output targets the Binary Ninja / ARM UAL spelling so the differential
//! corpus can compare directly:
//!
//! * mnemonic, lower-case, padded to [`FormatterOptions::first_operand_char_index`];
//! * operands separated by `", "` (configurable via
//!   [`FormatterOptions::space_after_operand_separator`]);
//! * unsigned/logical/wide-move immediates render `#0x…` with the `0x` prefix
//!   always present, including zero (`#0x0`) — Binary Ninja prints the prefix
//!   unconditionally and the corpus never uses a bare `#0`;
//! * signed immediates render with an explicit `-` and the magnitude in hex when
//!   [`FormatterOptions::signed_immediates`] is set;
//! * shifts/extends render `", lsl #0x…"` / `", sxtw #0x…"` (the `#amt` is
//!   elided for the no-amount extend forms and, unless
//!   [`FormatterOptions::show_lsl_zero`], for an `LSL #0`);
//! * memory operands render `[base{, #imm}]`, `[base, #imm]!`, `[base], #imm`
//!   and `[base, index{, extend #amt}]`;
//! * floating-point immediates render `#<f>` with eight fractional digits, like
//!   Binary Ninja's `%.08f`.

use super::{Formatter, FormatterOptions, FormatterOutput, TokenKind};
use crate::instruction::Instruction;
use crate::operand::{MemIndexMode, Operand, PredQual, SliceIndicator, SveMemMode};
use crate::register::Register;

/// A fixed-capacity output sink wrapping a borrowed `&mut [u8]`.
///
/// Writes are UTF-8 bytes; on overflow the excess is dropped and
/// [`BufSink::overflowed`] returns `true` (the buffer is **never** written past
/// its end). [`BufSink::as_str`] returns the bytes written so far.
#[derive(Debug)]
pub struct BufSink<'a> {
    buf: &'a mut [u8],
    len: usize,
    overflow: bool,
}

impl<'a> BufSink<'a> {
    /// Wrap a byte buffer as an output sink.
    #[inline]
    pub fn new(buf: &'a mut [u8]) -> BufSink<'a> {
        BufSink {
            buf,
            len: 0,
            overflow: false,
        }
    }

    /// The bytes written so far, as `&str` (always valid UTF-8 because only
    /// whole `&str` chunks are appended, truncated only at chunk granularity).
    #[inline]
    pub fn as_str(&self) -> &str {
        // Safe: we only ever append complete `&str` slices and never split a
        // chunk, so `buf[..len]` is valid UTF-8. Done without `unsafe` via the
        // checked converter (errors are impossible by construction but handled).
        core::str::from_utf8(&self.buf[..self.len]).unwrap_or("")
    }

    /// Number of bytes written.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` if nothing has been written.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// `true` if any write was truncated due to insufficient capacity.
    #[inline]
    pub fn overflowed(&self) -> bool {
        self.overflow
    }
}

impl<'a> core::fmt::Write for BufSink<'a> {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        let bytes = s.as_bytes();
        let remaining = self.buf.len() - self.len;
        if bytes.len() > remaining {
            // Record overflow but only append a whole chunk if it fits, to keep
            // `as_str` valid UTF-8. The partial chunk is dropped.
            self.overflow = true;
            return Err(core::fmt::Error);
        }
        self.buf[self.len..self.len + bytes.len()].copy_from_slice(bytes);
        self.len += bytes.len();
        Ok(())
    }
}

/// The default ARM UAL formatter (zero-alloc).
///
/// Construct with [`FmtFormatter::new`] (UAL defaults) or
/// [`FmtFormatter::with_options`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FmtFormatter {
    opts: FormatterOptions,
}

impl FmtFormatter {
    /// A formatter with the default ARM UAL options.
    #[inline]
    pub fn new() -> Self {
        FmtFormatter {
            opts: FormatterOptions::default(),
        }
    }

    /// A formatter with explicit options.
    #[inline]
    pub fn with_options(opts: FormatterOptions) -> Self {
        FmtFormatter { opts }
    }
}

impl Default for FmtFormatter {
    #[inline]
    fn default() -> Self {
        FmtFormatter::new()
    }
}

impl Formatter for FmtFormatter {
    fn format(&self, insn: &Instruction, out: &mut dyn FormatterOutput) {
        // Single dispatch path: mnemonic (padded) then each operand separated by
        // the operand separator. Memory-bracket fusion (`]`, `]!`, `], `),
        // predicate qualifiers (`/z` `/m`), sysreg names and signed-magnitude
        // immediates are all handled inside `format_operand`.
        self.format_mnemonic(insn, out);
        let n = insn.op_count();
        // `B.<cond>` fuses its condition into the mnemonic (see
        // `format_mnemonic`); skip operand 0 (the `Cond`) so only the label
        // remains in the operand stream.
        let start = if is_bcond(insn) { 1 } else { 0 };
        for i in start..n {
            self.write_separator(i - start, out);
            self.format_operand(insn, i, out);
        }
    }

    fn format_mnemonic(&self, insn: &Instruction, out: &mut dyn FormatterOutput) {
        // Emit the (alias-resolved) mnemonic name, applying case + padding policy,
        // as a `TokenKind::Mnemonic` token. `Instruction::mnemonic` already holds
        // the alias-resolved spelling when alias resolution was enabled at decode
        // time, so honouring `opts.aliases` here is a no-op beyond name selection.
        let name = insn.mnemonic().name();
        self.emit_cased(name, self.opts.uppercase_mnemonics, TokenKind::Mnemonic, out);

        // `B.<cond>` renders the condition as a `.cond` suffix fused onto the `b`
        // mnemonic (`b.ne`), matching the corpus. The condition is carried as
        // operand 0 and skipped from the operand stream in `format`.
        let mut have = name.len();
        if is_bcond(insn) {
            if let Operand::Cond(c) = insn.op(0) {
                out.write(".", TokenKind::Mnemonic);
                let cn = c.name();
                self.emit_cased(cn, self.opts.uppercase_mnemonics, TokenKind::Mnemonic, out);
                have += 1 + cn.len();
            }
        }

        // Pad with spaces up to `first_operand_char_index` so the first operand
        // lines up. Only pad when there is at least one operand to follow.
        let operands_follow = if is_bcond(insn) {
            insn.op_count() > 1
        } else {
            insn.op_count() > 0
        };
        if operands_follow {
            let want = self.opts.first_operand_char_index as usize;
            // At least one space always separates mnemonic and operands.
            let pad = if want > have { want - have } else { 1 };
            self.emit_spaces(pad, out);
        }
    }

    fn format_operand(&self, insn: &Instruction, n: usize, out: &mut dyn FormatterOutput) {
        let op = insn.op(n);
        match op {
            Operand::None => {}

            Operand::Reg {
                reg,
                arr,
                lane,
                shift,
                extend,
                pred,
            } => {
                self.emit_register(reg, out);
                // Arrangement suffix (`.4s`, `.b`, ...). Use the full form for a
                // bare register; the truncated element-size form is used only
                // with a lane index.
                if let Some(a) = arr {
                    let full = lane.is_none();
                    let suf = a.suffix(full);
                    if !suf.is_empty() {
                        self.emit_cased(suf, self.opts.uppercase_registers, TokenKind::Register, out);
                    }
                }
                // Lane index (`[2]`).
                if let Some(l) = lane {
                    out.write("[", TokenKind::BeginMemory);
                    self.emit_dec(l as u64, out);
                    out.write("]", TokenKind::EndMemory);
                }
                // SVE predicate qualifier (`/z`, `/m`).
                if let Some(p) = pred {
                    self.emit_pred(p, out);
                }
                // Register-extension (`, uxtw`, `, sxtx #2`).
                if let Some(e) = extend {
                    self.emit_extend(e, shift.map(|(_, a)| a), out);
                } else if let Some((st, amt)) = shift {
                    // Shift modifier (`, lsl #2`).
                    self.emit_shift(st, amt, out);
                }
            }

            Operand::ImmUnsigned(v) | Operand::ImmLogical(v) => {
                out.write("#", TokenKind::Number);
                self.emit_hex(v, out);
            }

            Operand::ImmSigned(v) => {
                out.write("#", TokenKind::Number);
                self.emit_signed(v, out);
            }

            Operand::ImmShiftedMove { imm, lsl } => {
                out.write("#", TokenKind::Number);
                self.emit_hex(imm as u64, out);
                // Fold the shift away when it is zero unless asked to show it.
                if lsl != 0 || self.opts.show_lsl_zero {
                    self.emit_shift(crate::enums::ShiftType::Lsl, lsl, out);
                }
            }

            Operand::ImmShiftedMsl { imm, msl } => {
                out.write("#", TokenKind::Number);
                self.emit_hex(imm as u64, out);
                // The MSL amount (8 or 16) is always shown.
                self.emit_shift(crate::enums::ShiftType::Msl, msl, out);
            }

            Operand::FpImm(f) => {
                out.write("#", TokenKind::Float);
                self.emit_float(f, out);
            }

            Operand::ShiftAmount(a) => {
                out.write("#", TokenKind::Number);
                self.emit_dec(a as u64, out);
            }

            Operand::Label(addr) => {
                // Absolute target, bare `0x…` (no `#`), like Binary Ninja.
                self.emit_addr(addr, out);
            }

            Operand::Cond(c) => {
                self.emit_cased(c.name(), self.opts.uppercase_mnemonics, TokenKind::Decorator, out);
            }

            Operand::SysReg(sr) => {
                self.emit_sysreg(sr, out);
            }

            Operand::SysOp(tok) => {
                // A fixed system keyword (barrier option, PSTATE field, `csync`,
                // `cN`, BTI target, or an IC/DC/AT/TLBI/CFP/CPP/DVP op name).
                self.emit_cased(tok.name(), self.opts.uppercase_mnemonics, TokenKind::Decorator, out);
            }

            Operand::MemImm { base, imm, mode } => {
                self.emit_mem_imm(base, imm, mode, out);
            }

            Operand::MemExt {
                base,
                index,
                extend,
                shift,
            } => {
                self.emit_mem_ext(base, index, extend, shift, out);
            }

            Operand::MultiReg {
                regs,
                count,
                arr,
                lane,
            } => {
                self.emit_multireg(&regs, count, arr, lane, out);
            }

            Operand::IndexedElement {
                reg,
                arr,
                index,
                imm,
            } => {
                // `reg[, arr][index{, #imm}]`-style: the data register with its
                // arrangement, then a bracketed index register and offset.
                self.emit_register(reg, out);
                if let Some(a) = arr {
                    let suf = a.suffix(true);
                    if !suf.is_empty() {
                        self.emit_cased(suf, self.opts.uppercase_registers, TokenKind::Register, out);
                    }
                }
                out.write("[", TokenKind::BeginMemory);
                self.emit_register(index, out);
                if imm != 0 {
                    self.write_raw_separator(out);
                    out.write("#", TokenKind::Number);
                    self.emit_signed(imm, out);
                }
                out.write("]", TokenKind::EndMemory);
            }

            Operand::SmeTile { tile, slice } => {
                self.emit_sme_tile(tile, slice, out);
            }

            Operand::ImplSpec(bytes) => {
                // Opaque payload: render the five raw bytes as `{0x..,...}` so it
                // is at least lossless and self-describing.
                out.write("{", TokenKind::Punctuation);
                for (i, b) in bytes.iter().enumerate() {
                    if i != 0 {
                        self.write_raw_separator(out);
                    }
                    out.write("0x", TokenKind::Number);
                    self.emit_hex_byte(*b, out);
                }
                out.write("}", TokenKind::Punctuation);
            }

            Operand::SvePattern(pat) => {
                // SVE element-count pattern: a keyword (`pow2`, `vl1`..`vl256`,
                // `mul3`/`mul4`, `all`) or, for the unnamed `0b01110..=0b11100`
                // range, the raw value as `#0xN` — matching the corpus.
                match sve_pattern_name(pat) {
                    Some(name) => self.emit_cased(
                        name,
                        self.opts.uppercase_mnemonics,
                        TokenKind::Decorator,
                        out,
                    ),
                    None => {
                        out.write("#", TokenKind::Number);
                        self.emit_hex(pat as u64, out);
                    }
                }
            }

            Operand::SveMul(imm) => {
                // The trailing `MUL #<imm>` multiplier of INC/DEC/CNT forms.
                self.emit_cased("mul", self.opts.uppercase_mnemonics, TokenKind::Decorator, out);
                out.write(" ", TokenKind::Decorator);
                out.write("#", TokenKind::Number);
                self.emit_hex(imm as u64, out);
            }

            Operand::ImmSignedDec(v) => {
                // SVE radix convention: non-negative in hex, negative in decimal.
                out.write("#", TokenKind::Number);
                if v < 0 {
                    out.write("-", TokenKind::Number);
                    self.emit_dec((v as i128).unsigned_abs() as u64, out);
                } else {
                    self.emit_hex(v as u64, out);
                }
            }

            Operand::SveMem {
                base,
                offset,
                arr,
                extend,
                imm,
                amount,
                mode,
            } => {
                self.emit_sve_mem(base, offset, arr, extend, imm, amount, mode, out);
            }

            Operand::SmeTileSlice {
                reg,
                slice,
                arr,
                sel,
                imm,
                has_imm,
            } => {
                self.emit_sme_tile_slice(reg, slice, arr, sel, imm, has_imm, out);
            }

            Operand::RegBang(reg) => {
                // A GP register with a writeback `!` suffix (MOPS size/count
                // operand): `x2!`.
                self.emit_register(reg, out);
                out.write("!", TokenKind::Punctuation);
            }

            Operand::RegPair { first, second } => {
                // A consecutive 64-bit register pair (FEAT_D128 MRRS/MSRR/SYSP):
                // two comma-separated registers `x12, x13`.
                self.emit_register(first, out);
                self.write_raw_separator(out);
                self.emit_register(second, out);
            }

            Operand::SmeZaSlice {
                arr,
                sel,
                off,
                span,
                vg,
                tile,
                slice,
            } => {
                self.emit_sme_za_slice(arr, sel, off, span, vg, tile, slice, out);
            }

            Operand::SveVecGroup {
                first,
                count,
                arr,
                range,
                stride,
            } => {
                self.emit_sve_vec_group(first, count, arr, range, stride, out);
            }

            Operand::PredCounter { reg, zeroing, arr } => {
                self.emit_pred_counter(reg, zeroing, arr, out);
            }

            Operand::VlMul(n) => {
                self.emit_cased("vlx", self.opts.uppercase_mnemonics, TokenKind::Decorator, out);
                self.emit_dec_kind(n as u64, TokenKind::Decorator, out);
            }
        }
    }

    #[inline]
    fn options(&self) -> &FormatterOptions {
        &self.opts
    }

    #[inline]
    fn options_mut(&mut self) -> &mut FormatterOptions {
        &mut self.opts
    }
}

impl FmtFormatter {
    // --- token helpers -----------------------------------------------------

    /// The operand separator before operand `i` (`", "` by default). Operand 0
    /// follows the mnemonic padding and gets no leading separator.
    #[inline]
    fn write_separator(&self, i: usize, out: &mut dyn FormatterOutput) {
        if i == 0 {
            return;
        }
        self.write_raw_separator(out);
    }

    /// Emit a `", "`/`","` separator unconditionally (used inside brackets too).
    #[inline]
    fn write_raw_separator(&self, out: &mut dyn FormatterOutput) {
        if self.opts.space_after_operand_separator {
            out.write(", ", TokenKind::OperandSeparator);
        } else {
            out.write(",", TokenKind::OperandSeparator);
        }
    }

    /// Emit a register name, honouring the SP-vs-ZR display policy and case.
    #[inline]
    fn emit_register(&self, reg: Register, out: &mut dyn FormatterOutput) {
        let reg = self.sp_zr_remap(reg);
        self.emit_cased(reg.name(), self.opts.uppercase_registers, TokenKind::Register, out);
    }

    /// Apply [`FormatterOptions::use_sp_not_xzr`]. The decoder already resolves
    /// reg-31 to the role-correct `SP`/`WSP` or `XZR`/`WZR`; when the caller asks
    /// for the zero-register spelling (`use_sp_not_xzr == false`) we rewrite the
    /// stack-pointer view to the zero view so both spellings are reachable.
    #[inline]
    fn sp_zr_remap(&self, reg: Register) -> Register {
        if self.opts.use_sp_not_xzr {
            reg
        } else {
            match reg {
                Register::Sp => Register::Xzr,
                Register::Wsp => Register::Wzr,
                other => other,
            }
        }
    }

    /// Emit the SVE predicate qualifier (`/z` or `/m`).
    #[inline]
    fn emit_pred(&self, p: PredQual, out: &mut dyn FormatterOutput) {
        let s = match p {
            PredQual::None => return,
            PredQual::Zeroing => "/z",
            PredQual::Merging => "/m",
        };
        self.emit_cased(s, self.opts.uppercase_registers, TokenKind::Decorator, out);
    }

    /// Emit a shift modifier (`, lsl #2`). `None`/empty shift names are dropped.
    fn emit_shift(&self, st: crate::enums::ShiftType, amt: u8, out: &mut dyn FormatterOutput) {
        let name = st.name();
        if name.is_empty() {
            return;
        }
        self.write_raw_separator(out);
        self.emit_cased(name, self.opts.uppercase_mnemonics, TokenKind::Decorator, out);
        out.write(" ", TokenKind::Decorator);
        out.write("#", TokenKind::Number);
        self.emit_hex(amt as u64, out);
    }

    /// Emit a register-extension modifier (`, uxtw`, `, sxtx #2`). The shift
    /// amount is shown only when present and non-zero (UAL elides `#0`).
    fn emit_extend(&self, e: crate::enums::ExtendType, amt: Option<u8>, out: &mut dyn FormatterOutput) {
        self.write_raw_separator(out);
        self.emit_cased(e.name(), self.opts.uppercase_mnemonics, TokenKind::Decorator, out);
        match amt {
            Some(a) if a != 0 || self.opts.show_lsl_zero => {
                out.write(" ", TokenKind::Decorator);
                out.write("#", TokenKind::Number);
                self.emit_hex(a as u64, out);
            }
            _ => {}
        }
    }

    /// Emit a memory operand with an immediate displacement.
    fn emit_mem_imm(
        &self,
        base: Register,
        imm: i64,
        mode: MemIndexMode,
        out: &mut dyn FormatterOutput,
    ) {
        out.write("[", TokenKind::BeginMemory);
        self.emit_register(base, out);
        match mode {
            MemIndexMode::Offset => {
                if imm != 0 {
                    self.write_raw_separator(out);
                    out.write("#", TokenKind::Number);
                    self.emit_signed(imm, out);
                }
                out.write("]", TokenKind::EndMemory);
            }
            MemIndexMode::PreIndex => {
                self.write_raw_separator(out);
                out.write("#", TokenKind::Number);
                self.emit_signed(imm, out);
                out.write("]", TokenKind::EndMemory);
                out.write("!", TokenKind::Punctuation);
            }
            MemIndexMode::PreNoOffset => {
                // `[base]!` — writeback, no displacement (MOPS address operands).
                out.write("]", TokenKind::EndMemory);
                out.write("!", TokenKind::Punctuation);
            }
            MemIndexMode::PostImm => {
                out.write("]", TokenKind::EndMemory);
                self.write_raw_separator(out);
                out.write("#", TokenKind::Number);
                self.emit_signed(imm, out);
            }
            MemIndexMode::PostReg => {
                // `[base], <Xm>` — the index register is carried in `imm` as a
                // register discriminant by the few SIMD post-index forms that use
                // it; render the bracket and let the trailing register operand be
                // emitted separately. We close the bracket here.
                out.write("]", TokenKind::EndMemory);
            }
        }
    }

    /// Emit a memory operand with a register index and optional extend/shift.
    ///
    /// The `shift` byte is packed by the load/store decoder as `(S << 7) | amt`,
    /// where `S` is the encoding's scale-select bit and `amt` the shift amount.
    /// Binary Ninja's rendering rule for register-offset addressing is driven by
    /// `S`, not by whether `amt` is zero:
    ///
    /// * the `uxtx` (LSL-equivalent) extend with `S == 0` prints **no** decoration
    ///   at all (`[x16, xzr]`);
    /// * every other case prints the extend keyword, and appends `#amt` (even
    ///   `#0x0`) exactly when `S == 1`.
    ///
    /// `uxtx` with `S == 1` is rendered as `lsl #amt` to match the corpus.
    fn emit_mem_ext(
        &self,
        base: Register,
        index: Register,
        extend: crate::enums::ExtendType,
        shift: u8,
        out: &mut dyn FormatterOutput,
    ) {
        let show_amt = (shift >> 7) == 1;
        let amt = shift & 0x7f;
        out.write("[", TokenKind::BeginMemory);
        self.emit_register(base, out);
        self.write_raw_separator(out);
        self.emit_register(index, out);

        let is_lsl = matches!(extend, crate::enums::ExtendType::Uxtx);
        // The LSL-equivalent extend with no scale bit prints nothing extra.
        if !(is_lsl && !show_amt && !self.opts.show_lsl_zero) {
            self.write_raw_separator(out);
            // `uxtx` is spelled `lsl` in register-offset addressing.
            let name = if is_lsl {
                crate::enums::ShiftType::Lsl.name()
            } else {
                extend.name()
            };
            self.emit_cased(name, self.opts.uppercase_mnemonics, TokenKind::Decorator, out);
            if show_amt || self.opts.show_lsl_zero {
                out.write(" ", TokenKind::Decorator);
                out.write("#", TokenKind::Number);
                self.emit_hex(amt as u64, out);
            }
        }
        out.write("]", TokenKind::EndMemory);
    }

    /// Emit an SVE memory addressing operand.
    ///
    /// Renders the five SVE addressing shapes exactly as Binary Ninja / the ARM
    /// UAL spell them:
    ///
    /// * [`SveMemMode::ScalarImmMulVl`]: `[base{, #imm, mul vl}]` — the `#imm`
    ///   and the `mul vl` suffix are elided when the immediate is zero (`[x0]`);
    /// * [`SveMemMode::ScalarImm`]: `[base{, #imm}]` — plain byte offset
    ///   (`LD1RQ`/`LD1RO`/`LD1R*`), elided when zero;
    /// * [`SveMemMode::VecImm`]: `[Zn.<T>{, #imm}]` — vector base + immediate;
    /// * [`SveMemMode::VecScalar`]: `[Zn.<T>, Xm]` — vector base + scalar offset;
    /// * [`SveMemMode::ScalarVec`]: `[Xn, Zm.<T>{, <mod> #amt}]` — scalar base +
    ///   vector index with an optional `sxtw`/`uxtw`/`lsl` modifier (`amount ==
    ///   0xFF` suppresses the `#amt`).
    #[allow(clippy::too_many_arguments)]
    fn emit_sve_mem(
        &self,
        base: Register,
        offset: Register,
        arr: Option<crate::enums::VectorArrangement>,
        extend: crate::enums::ExtendType,
        imm: i32,
        amount: u8,
        mode: SveMemMode,
        out: &mut dyn FormatterOutput,
    ) {
        out.write("[", TokenKind::BeginMemory);
        self.emit_register(base, out);
        // Arrangement suffix on the base for the vector-base modes.
        let base_arr = matches!(
            mode,
            SveMemMode::VecImm | SveMemMode::VecScalar | SveMemMode::VecVec
        );
        if base_arr {
            if let Some(a) = arr {
                let suf = a.suffix(true);
                if !suf.is_empty() {
                    self.emit_cased(suf, self.opts.uppercase_registers, TokenKind::Register, out);
                }
            }
        }
        match mode {
            SveMemMode::ScalarImmMulVl => {
                if imm != 0 {
                    self.write_raw_separator(out);
                    self.emit_sve_signed_dec(imm as i64, out);
                    self.write_raw_separator(out);
                    self.emit_cased("mul vl", self.opts.uppercase_mnemonics, TokenKind::Decorator, out);
                }
            }
            SveMemMode::ScalarImmDec => {
                // SVE radix: negative in decimal, non-negative in hex (no MUL VL).
                if imm != 0 {
                    self.write_raw_separator(out);
                    self.emit_sve_signed_dec(imm as i64, out);
                }
            }
            SveMemMode::ScalarImm | SveMemMode::VecImm => {
                // Signed hex (`#-0x1`/`#0x40`); VecImm offsets are always >= 0.
                if imm != 0 {
                    self.write_raw_separator(out);
                    out.write("#", TokenKind::Number);
                    self.emit_signed(imm as i64, out);
                }
            }
            SveMemMode::VecScalar => {
                self.write_raw_separator(out);
                self.emit_register(offset, out);
            }
            SveMemMode::ScalarVec => {
                self.write_raw_separator(out);
                self.emit_register(offset, out);
                if let Some(a) = arr {
                    let suf = a.suffix(true);
                    if !suf.is_empty() {
                        self.emit_cased(suf, self.opts.uppercase_registers, TokenKind::Register, out);
                    }
                }
                // Modifier: `uxtx` renders as `lsl`. The `#amt` is shown unless
                // `amount == 0xFF` (the unscaled gather/scatter forms, which omit
                // both `lsl` and `#amt` but keep `sxtw`/`uxtw` bare).
                let is_lsl = matches!(extend, crate::enums::ExtendType::Uxtx);
                let show_amt = amount != 0xFF;
                if !is_lsl || show_amt {
                    self.write_raw_separator(out);
                    let name = if is_lsl {
                        crate::enums::ShiftType::Lsl.name()
                    } else {
                        extend.name()
                    };
                    self.emit_cased(name, self.opts.uppercase_mnemonics, TokenKind::Decorator, out);
                    if show_amt {
                        out.write(" ", TokenKind::Decorator);
                        out.write("#", TokenKind::Number);
                        self.emit_hex(amount as u64, out);
                    }
                }
            }
            SveMemMode::VecVec => {
                // `[Zn.<T>, Zm.<T>{, <mod> #<amt>}]` (SVE ADR). Offset is a
                // scalable-vector with the shared arrangement; the modifier
                // (`sxtw`/`uxtw`, or `lsl` for `Uxtx`) and `#amt` are shown only
                // when `amount != 0`.
                self.write_raw_separator(out);
                self.emit_register(offset, out);
                if let Some(a) = arr {
                    let suf = a.suffix(true);
                    if !suf.is_empty() {
                        self.emit_cased(suf, self.opts.uppercase_registers, TokenKind::Register, out);
                    }
                }
                if amount != 0 {
                    let is_lsl = matches!(extend, crate::enums::ExtendType::Uxtx);
                    self.write_raw_separator(out);
                    let name = if is_lsl {
                        crate::enums::ShiftType::Lsl.name()
                    } else {
                        extend.name()
                    };
                    self.emit_cased(name, self.opts.uppercase_mnemonics, TokenKind::Decorator, out);
                    out.write(" ", TokenKind::Decorator);
                    out.write("#", TokenKind::Number);
                    self.emit_hex(amount as u64, out);
                }
            }
        }
        out.write("]", TokenKind::EndMemory);
    }

    /// Emit a vector register list (`{v0.4s, v1.4s}`), with an optional shared
    /// lane index after the closing brace (`{v0.s, v1.s}[2]`).
    fn emit_multireg(
        &self,
        regs: &[Register; 4],
        count: u8,
        arr: Option<crate::enums::VectorArrangement>,
        lane: Option<u8>,
        out: &mut dyn FormatterOutput,
    ) {
        let count = (count as usize).min(4);
        out.write("{", TokenKind::Punctuation);
        for (i, &r) in regs.iter().take(count).enumerate() {
            if i != 0 {
                self.write_raw_separator(out);
            }
            self.emit_register(r, out);
            if let Some(a) = arr {
                // With a lane index the truncated element-size suffix is used.
                let suf = a.suffix(lane.is_none());
                if !suf.is_empty() {
                    self.emit_cased(suf, self.opts.uppercase_registers, TokenKind::Register, out);
                }
            }
        }
        out.write("}", TokenKind::Punctuation);
        if let Some(l) = lane {
            out.write("[", TokenKind::BeginMemory);
            self.emit_dec(l as u64, out);
            out.write("]", TokenKind::EndMemory);
        }
    }

    /// Emit an SME ZA tile reference (`za`, `za0.s`, `za1h.b`, ...).
    fn emit_sme_tile(&self, tile: u16, slice: SliceIndicator, out: &mut dyn FormatterOutput) {
        // Packed tile id: low nibble is the tile number, the next two bits encode
        // the element size (0=>none/byte, 1=>h, 2=>s, 3=>d, 4=>q). This mirrors
        // the decode-side packing; render conservatively.
        let num = (tile & 0x0f) as u64;
        let size_code = (tile >> 4) & 0x07;
        out.write("za", TokenKind::Register);
        // Tile number is omitted for the whole-array form (`za`).
        if (tile & 0x8000) == 0 {
            self.emit_dec(num, out);
        }
        // Slice direction suffix (`h`/`v`).
        match slice {
            SliceIndicator::Horizontal => out.write("h", TokenKind::Register),
            SliceIndicator::Vertical => out.write("v", TokenKind::Register),
            SliceIndicator::None => {}
        }
        // Element-size suffix.
        let suf = match size_code {
            1 => ".h",
            2 => ".s",
            3 => ".d",
            4 => ".q",
            5 => ".b",
            _ => "",
        };
        if !suf.is_empty() {
            out.write(suf, TokenKind::Register);
        }
    }

    /// Emit an SME ZA-array tile *slice* operand
    /// ([`Operand::SmeTileSlice`](crate::operand::Operand::SmeTileSlice)).
    ///
    /// Renders either the per-tile slice `z<n><h|v>.<T>[<Ws>{, #<imm>}]` (Binary
    /// Ninja prints the tile with a `z` prefix) when `reg` is a `Z` register, or
    /// the whole-array select `za[<Ws>{, #<imm>}]` (`LDR`/`STR` ZA) when `reg`
    /// is [`Register::None`].
    #[allow(clippy::too_many_arguments)]
    fn emit_sme_tile_slice(
        &self,
        reg: Register,
        slice: SliceIndicator,
        arr: Option<crate::enums::VectorArrangement>,
        sel: Register,
        imm: i16,
        has_imm: bool,
        out: &mut dyn FormatterOutput,
    ) {
        if matches!(reg, Register::None) {
            // Whole-array `za[...]` (LDR/STR ZA).
            out.write("za", TokenKind::Register);
        } else {
            // Per-tile `z<n>` with the binja `z` prefix, then the slice
            // direction (`h`/`v`) and the element-size suffix.
            self.emit_register(reg, out);
            match slice {
                SliceIndicator::Horizontal => out.write("h", TokenKind::Register),
                SliceIndicator::Vertical => out.write("v", TokenKind::Register),
                SliceIndicator::None => {}
            }
            if let Some(a) = arr {
                let suf = a.suffix(true);
                if !suf.is_empty() {
                    self.emit_cased(suf, self.opts.uppercase_registers, TokenKind::Register, out);
                }
            }
        }
        // `[<Ws>{, #<imm>}]`.
        out.write("[", TokenKind::BeginMemory);
        self.emit_register(sel, out);
        if has_imm {
            self.write_raw_separator(out);
            out.write("#", TokenKind::Number);
            self.emit_signed(imm as i64, out);
        }
        out.write("]", TokenKind::EndMemory);
    }

    /// Emit an SME2 ZA-array vector slice-group destination
    /// ([`Operand::SmeZaSlice`](crate::operand::Operand::SmeZaSlice)):
    /// `za.<T>[<Ws>, <off>{:<off+span-1>}{, vgx2|vgx4}]`.
    ///
    /// LLVM renders the slice offset/range and the `vgxN` qualifier in decimal,
    /// e.g. `za.s[w8, 0:3]`, `za.h[w8, 6, vgx2]`, `za.s[w9, 4:7, vgx4]`.
    ///
    /// When `slice` selects a tile direction the *tile*-slice spelling
    /// `za<tile><h|v>.<T>[<Ws>, <off>:<off+span-1>]` is emitted instead (the SME2
    /// `MOV`/`MOVAZ` move-multi-vectors-to/from-ZA-tile form), with no `vgxN`.
    #[allow(clippy::too_many_arguments)]
    fn emit_sme_za_slice(
        &self,
        arr: Option<crate::enums::VectorArrangement>,
        sel: Register,
        off: u8,
        span: u8,
        vg: u8,
        tile: u8,
        slice: SliceIndicator,
        out: &mut dyn FormatterOutput,
    ) {
        out.write("za", TokenKind::Register);
        // Tile-slice form: `za<tile><h|v>` before the element-size suffix.
        let is_tile = !matches!(slice, SliceIndicator::None);
        if is_tile {
            self.emit_dec_kind(tile as u64, TokenKind::Register, out);
            match slice {
                SliceIndicator::Horizontal => out.write("h", TokenKind::Register),
                SliceIndicator::Vertical => out.write("v", TokenKind::Register),
                SliceIndicator::None => {}
            }
        }
        if let Some(a) = arr {
            let suf = a.suffix(true);
            if !suf.is_empty() {
                self.emit_cased(suf, self.opts.uppercase_registers, TokenKind::Register, out);
            }
        }
        out.write("[", TokenKind::BeginMemory);
        self.emit_register(sel, out);
        self.write_raw_separator(out);
        // `<off>` or, for a multi-slice span, the `<off>:<off+span-1>` range.
        self.emit_dec(off as u64, out);
        if span > 1 {
            out.write(":", TokenKind::Punctuation);
            self.emit_dec((off + span - 1) as u64, out);
        }
        // The optional `, vgx2`/`, vgx4` multi-vector qualifier (group form only;
        // the tile-slice form never carries it).
        if !is_tile && (vg == 2 || vg == 4) {
            self.write_raw_separator(out);
            self.emit_cased("vgx", self.opts.uppercase_registers, TokenKind::Decorator, out);
            self.emit_dec(vg as u64, out);
        }
        out.write("]", TokenKind::EndMemory);
    }

    /// Emit an SME2/SVE2 multi-vector register group
    /// ([`Operand::SveVecGroup`](crate::operand::Operand::SveVecGroup)):
    /// `{ z0.b, z1.b }` (comma list) or `{ z0.b - z3.b }` (range). LLVM puts a
    /// space just inside each brace. `stride` is the register-number step between
    /// members (`1` consecutive; `8`/`4` for the strided multi-vector lists, which
    /// always render as a comma list).
    fn emit_sve_vec_group(
        &self,
        first: Register,
        count: u8,
        arr: Option<crate::enums::VectorArrangement>,
        range: bool,
        stride: u8,
        out: &mut dyn FormatterOutput,
    ) {
        let count = count.clamp(1, 4);
        let stride = stride.max(1);
        let suf = arr.map(|a| a.suffix(true)).unwrap_or("");
        let first_n = first.number();
        let emit_one = |this: &Self, n: u8, out: &mut dyn FormatterOutput| {
            this.emit_register(crate::register::sve_register(n), out);
            if !suf.is_empty() {
                this.emit_cased(suf, this.opts.uppercase_registers, TokenKind::Register, out);
            }
        };
        out.write("{ ", TokenKind::Punctuation);
        if range && count > 1 {
            // `{ z0.b - z3.b }` — only the first and last registers are shown
            // (consecutive range; the strided lists never take this path).
            emit_one(self, first_n, out);
            out.write(" - ", TokenKind::Punctuation);
            emit_one(self, first_n.wrapping_add((count - 1) * stride) & 0x1f, out);
        } else {
            // `{ z0.b, z1.b }` — each register at `first + i*stride`, comma-separated.
            for i in 0..count {
                if i != 0 {
                    self.write_raw_separator(out);
                }
                emit_one(self, first_n.wrapping_add(i * stride) & 0x1f, out);
            }
        }
        out.write(" }", TokenKind::Punctuation);
    }

    /// Emit an SME2/SVE2.1 predicate-as-counter
    /// ([`Operand::PredCounter`](crate::operand::Operand::PredCounter)):
    /// `pn8`..`pn15`, optionally with a `/z` qualifier. The underlying register
    /// is `P8`..`P15`; the `p` name is rewritten to `pn`.
    fn emit_pred_counter(
        &self,
        reg: Register,
        zeroing: bool,
        arr: Option<crate::enums::VectorArrangement>,
        out: &mut dyn FormatterOutput,
    ) {
        self.emit_cased("pn", self.opts.uppercase_registers, TokenKind::Register, out);
        // The architectural number (8..=15); written as a register-kind token so
        // it groups with the `pn` prefix rather than reading as an immediate.
        self.emit_dec_kind(reg.number() as u64, TokenKind::Register, out);
        // Element-size suffix (SVE2.1 `WHILE<cc>` predicate-as-counter result).
        if let Some(a) = arr {
            let suf = a.suffix(true);
            if !suf.is_empty() {
                self.emit_cased(suf, self.opts.uppercase_registers, TokenKind::Register, out);
            }
        }
        if zeroing {
            self.emit_cased("/z", self.opts.uppercase_registers, TokenKind::Decorator, out);
        }
    }

    /// Emit a system-register reference, falling back to the generic
    /// `S<op0>_<op1>_c<CRn>_c<CRm>_<op2>` form for unknown encodings.
    fn emit_sysreg(&self, sr: crate::sysreg::SystemReg, out: &mut dyn FormatterOutput) {
        if let Some(name) = sr.name() {
            self.emit_cased(name, self.opts.uppercase_registers, TokenKind::SysReg, out);
        } else {
            // Generic form. The leading `s` plus digit-by-digit body are written
            // through a small adapter so this stays zero-alloc.
            out.write("s", TokenKind::SysReg);
            let mut adapter = SinkAdapter { out };
            let _ = sr.render(&mut adapter);
        }
    }

    // --- numeric emission (zero-alloc) -------------------------------------

    /// Emit `value` as hex using the configured prefix. Binary Ninja always
    /// prints the `0x` prefix, including for zero (`#0x0`); the corpus contains
    /// `#0x0` and never a bare `#0`, so zero is rendered as `0x0` rather than
    /// C's `%#x` bare `0`.
    fn emit_hex(&self, value: u64, out: &mut dyn FormatterOutput) {
        out.write(self.opts.hex_prefix, TokenKind::Number);
        if value == 0 {
            out.write("0", TokenKind::Number);
            return;
        }
        self.emit_hex_digits(value, out);
    }

    /// Emit a signed immediate: an explicit `-` (when negative and
    /// [`FormatterOptions::signed_immediates`]) followed by the hex magnitude.
    fn emit_signed(&self, value: i64, out: &mut dyn FormatterOutput) {
        if self.opts.signed_immediates && value < 0 {
            out.write("-", TokenKind::Number);
            self.emit_hex((value as i128).unsigned_abs() as u64, out);
        } else {
            self.emit_hex(value as u64, out);
        }
    }

    /// Emit an immediate with the SVE radix convention: a leading `#`, then a
    /// negative value in **decimal** (`#-12`) and a non-negative value in hex
    /// (`#0x4`). Matches [`Operand::ImmSignedDec`] and the SVE `MUL VL` /
    /// `LD1RQ` offsets.
    fn emit_sve_signed_dec(&self, v: i64, out: &mut dyn FormatterOutput) {
        out.write("#", TokenKind::Number);
        if v < 0 {
            out.write("-", TokenKind::Number);
            self.emit_dec((v as i128).unsigned_abs() as u64, out);
        } else {
            self.emit_hex(v as u64, out);
        }
    }

    /// Emit a bare absolute address (`0x…`), always prefixed and lower-case hex.
    fn emit_addr(&self, addr: u64, out: &mut dyn FormatterOutput) {
        out.write(self.opts.hex_prefix, TokenKind::Address);
        if addr == 0 {
            out.write("0", TokenKind::Address);
        } else {
            self.emit_hex_digits_kind(addr, TokenKind::Address, out);
        }
    }

    /// Emit the hex digits of a non-zero value (no prefix), lower-case.
    #[inline]
    fn emit_hex_digits(&self, value: u64, out: &mut dyn FormatterOutput) {
        self.emit_hex_digits_kind(value, TokenKind::Number, out);
    }

    fn emit_hex_digits_kind(&self, mut value: u64, kind: TokenKind, out: &mut dyn FormatterOutput) {
        // u64 -> at most 16 hex digits. Build into a fixed scratch buffer and
        // emit as a single chunk so the sink sees whole UTF-8.
        let mut tmp = [0u8; 16];
        let mut i = tmp.len();
        while value > 0 {
            i -= 1;
            let nib = (value & 0xf) as u8;
            tmp[i] = HEX[nib as usize];
            value >>= 4;
        }
        // SAFETY-free: bytes are ASCII hex by construction.
        let s = core::str::from_utf8(&tmp[i..]).unwrap_or("");
        out.write(s, kind);
    }

    /// Emit a single byte as exactly two lower-case hex digits.
    fn emit_hex_byte(&self, b: u8, out: &mut dyn FormatterOutput) {
        let tmp = [HEX[(b >> 4) as usize], HEX[(b & 0xf) as usize]];
        let s = core::str::from_utf8(&tmp).unwrap_or("");
        out.write(s, TokenKind::Number);
    }

    /// Emit a decimal integer (used for lane indices and tile numbers).
    fn emit_dec(&self, value: u64, out: &mut dyn FormatterOutput) {
        self.emit_dec_kind(value, TokenKind::Number, out);
    }

    /// Like [`emit_dec`](Self::emit_dec) but tags the digits with an explicit
    /// [`TokenKind`] (used for the `pn8`..`pn15` predicate-as-counter spelling,
    /// whose number is part of the register name rather than an immediate).
    fn emit_dec_kind(&self, value: u64, kind: TokenKind, out: &mut dyn FormatterOutput) {
        if value == 0 {
            out.write("0", kind);
            return;
        }
        let mut tmp = [0u8; 20]; // u64::MAX is 20 digits.
        let mut i = tmp.len();
        let mut v = value;
        while v > 0 {
            i -= 1;
            tmp[i] = b'0' + (v % 10) as u8;
            v /= 10;
        }
        let s = core::str::from_utf8(&tmp[i..]).unwrap_or("");
        out.write(s, kind);
    }

    /// Emit a 32-bit float as the shortest exact decimal that round-trips,
    /// always keeping at least one fractional digit (`19.0`, `-0.21875`,
    /// `0.0`), matching Binary Ninja's rendering of the A64 FP-immediate set.
    ///
    /// The A64 8-bit FP immediate (`VFPExpandImm`) is always an exact dyadic
    /// rational with at most seven fractional digits, so seven digits captured
    /// at `10^7` and then stripped of trailing zeros reproduces the value
    /// exactly. Implemented without `core::fmt` float formatting so the path is
    /// allocation-free and stable across targets.
    fn emit_float(&self, f: f32, out: &mut dyn FormatterOutput) {
        // Handle the non-finite cases explicitly (they are rare in real FP-imm
        // encodings but must not panic).
        if f.is_nan() {
            out.write("nan", TokenKind::Float);
            return;
        }
        if f.is_infinite() {
            out.write(if f < 0.0 { "-inf" } else { "inf" }, TokenKind::Float);
            return;
        }

        let neg = f.is_sign_negative();
        let x = if neg { -(f as f64) } else { f as f64 };

        // Split into integer and fractional parts without `f64::floor` (a
        // `std`-only method): the cast to `u64` truncates toward zero, and `x`
        // is already non-negative here.
        const FRAC: f64 = 10_000_000.0; // 10^7 -> seven fractional digits.
        // Guard the integer cast: magnitudes this large carry no meaningful
        // fractional digits, so clamp and print `.0` after the point.
        let (int_part, frac_part): (u64, u64) = if x >= 18_000_000_000_000_000_000.0 {
            (u64::MAX, 0)
        } else {
            let mut ip = x as u64;
            let frac = x - ip as f64;
            // Round half-away-from-zero to seven digits.
            let mut fp = (frac * FRAC + 0.5) as u64;
            if fp >= 10_000_000 {
                // Rounding carried into the integer part.
                fp -= 10_000_000;
                ip = ip.wrapping_add(1);
            }
            (ip, fp)
        };

        if neg {
            out.write("-", TokenKind::Float);
        }
        self.emit_dec(int_part, out);
        out.write(".", TokenKind::Float);
        // Seven digits, zero-padded, then strip trailing zeros but keep at
        // least one fractional digit so integers render as `N.0`.
        let mut tmp = [b'0'; 7];
        let mut v = frac_part;
        let mut idx = tmp.len();
        while v > 0 && idx > 0 {
            idx -= 1;
            tmp[idx] = b'0' + (v % 10) as u8;
            v /= 10;
        }
        // Determine the length after stripping trailing zeros (minimum 1).
        let mut end = tmp.len();
        while end > 1 && tmp[end - 1] == b'0' {
            end -= 1;
        }
        let s = core::str::from_utf8(&tmp[..end]).unwrap_or("0");
        out.write(s, TokenKind::Float);
    }

    // --- case-aware text emission ------------------------------------------

    /// Emit `text`, upper-casing ASCII when `upper` is set. Lower-case emission
    /// is a single chunk; the upper-case path emits ASCII byte windows without
    /// allocation (it only special-cases `a..=z`).
    fn emit_cased(&self, text: &str, upper: bool, kind: TokenKind, out: &mut dyn FormatterOutput) {
        if !upper {
            out.write(text, kind);
            return;
        }
        // Upper-case ASCII letters in place using a small rolling buffer so we
        // never allocate. Non-letters (digits, `.`, `[`, `/`, `_`) pass through.
        let bytes = text.as_bytes();
        let mut buf = [0u8; 16];
        let mut len = 0usize;
        for &b in bytes {
            let c = if b.is_ascii_lowercase() { b - 32 } else { b };
            if len == buf.len() {
                let s = core::str::from_utf8(&buf[..len]).unwrap_or("");
                out.write(s, kind);
                len = 0;
            }
            buf[len] = c;
            len += 1;
        }
        if len > 0 {
            let s = core::str::from_utf8(&buf[..len]).unwrap_or("");
            out.write(s, kind);
        }
    }

    /// Emit `n` spaces for mnemonic padding (chunked, zero-alloc).
    fn emit_spaces(&self, n: usize, out: &mut dyn FormatterOutput) {
        const SPACES: &str = "                                "; // 32 spaces
        let mut left = n;
        while left > 0 {
            let take = left.min(SPACES.len());
            out.write(&SPACES[..take], TokenKind::Mnemonic);
            left -= take;
        }
    }
}

/// `true` when `insn` is a conditional-branch encoding whose condition is fused
/// into the mnemonic (`b.ne` / `bc.ne`) rather than printed as a separate
/// operand. Covers `B.<cond>` ([`Code::BCond`]) and the FEAT_HBC `BC.<cond>`
/// ([`Code::BcCond`]).
#[inline]
fn is_bcond(insn: &Instruction) -> bool {
    matches!(
        insn.code(),
        crate::mnemonic::Code::BCond | crate::mnemonic::Code::BcCond
    )
}

/// The SVE element-count pattern keyword for a 5-bit `pattern` field, or `None`
/// for the unnamed `0b01110..=0b11100` range (which the formatter renders as a
/// raw `#0xN`). Matches the ARM ARM / Binary Ninja `pattern_lookup` mapping.
#[inline]
fn sve_pattern_name(pat: u8) -> Option<&'static str> {
    Some(match pat & 0x1f {
        0b00000 => "pow2",
        0b00001 => "vl1",
        0b00010 => "vl2",
        0b00011 => "vl3",
        0b00100 => "vl4",
        0b00101 => "vl5",
        0b00110 => "vl6",
        0b00111 => "vl7",
        0b01000 => "vl8",
        0b01001 => "vl16",
        0b01010 => "vl32",
        0b01011 => "vl64",
        0b01100 => "vl128",
        0b01101 => "vl256",
        0b11101 => "mul4",
        0b11110 => "mul3",
        0b11111 => "all",
        // 0b01110..=0b11100 are unnamed: rendered as a raw immediate.
        _ => return None,
    })
}

/// Lower-case hexadecimal digit table.
const HEX: &[u8; 16] = b"0123456789abcdef";

/// Adapts a [`FormatterOutput`] to a [`core::fmt::Write`] for [`SystemReg::render`].
///
/// Chunks are forwarded verbatim as [`TokenKind::SysReg`]; this is only used for
/// the generic system-register fallback, which produces ASCII.
///
/// [`SystemReg::render`]: crate::sysreg::SystemReg::render
struct SinkAdapter<'a> {
    out: &'a mut dyn FormatterOutput,
}

impl<'a> core::fmt::Write for SinkAdapter<'a> {
    #[inline]
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        self.out.write(s, TokenKind::SysReg);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enums::{Condition, ExtendType, ShiftType, VectorArrangement};
    use crate::mnemonic::{Code, Mnemonic};
    use crate::operand::Operand;
    use crate::register::Register;
    use crate::MAX_OPERANDS;

    /// Build an instruction from a mnemonic + a slice of operands.
    fn make(mnem: Mnemonic, ops: &[Operand]) -> Instruction {
        let mut i = Instruction {
            code: Code::Invalid,
            mnemonic: mnem,
            ..Default::default()
        };
        let n = ops.len().min(MAX_OPERANDS);
        for (slot, op) in ops.iter().take(n).enumerate() {
            i.operands[slot] = *op;
        }
        i.op_count = n as u8;
        i
    }

    /// Assert that `insn` renders to `expected` through a stack `BufSink`. The
    /// whole path is allocation-free (no `String`), so these tests build even on
    /// the default no-`alloc` tier.
    #[track_caller]
    fn assert_render(insn: &Instruction, expected: &str) {
        assert_render_with(&FmtFormatter::new(), insn, expected);
    }

    #[track_caller]
    fn assert_render_with(fmt: &FmtFormatter, insn: &Instruction, expected: &str) {
        let mut buf = [0u8; 256];
        let mut sink = BufSink::new(&mut buf);
        fmt.format(insn, &mut sink);
        assert!(!sink.overflowed(), "BufSink overflowed rendering {expected:?}");
        assert_eq!(sink.as_str(), expected);
    }

    fn reg(r: Register) -> Operand {
        Operand::Reg {
            reg: r,
            arr: None,
            lane: None,
            shift: None,
            extend: None,
            pred: None,
        }
    }

    #[test]
    fn add_imm() {
        // ADD x0, x1, #1
        let insn = make(
            Mnemonic::Add,
            &[reg(Register::X0), reg(Register::X1), Operand::ImmUnsigned(1)],
        );
        assert_render(&insn, "add     x0, x1, #0x1");
    }

    #[test]
    fn add_imm_zero_is_bare() {
        // ADD x0, x1, #0  -> Binary Ninja always prints the 0x prefix (`#0x0`),
        // matching the corpus (which has 458 `#0x0` and no bare `#0`).
        let insn = make(
            Mnemonic::Add,
            &[reg(Register::X0), reg(Register::X1), Operand::ImmUnsigned(0)],
        );
        assert_render(&insn, "add     x0, x1, #0x0");
    }

    #[test]
    fn mnemonic_padding_width() {
        // Mnemonic padded to column 8; a short mnemonic gets spaces, a long one
        // gets a single space.
        let insn = make(Mnemonic::Add, &[reg(Register::X0)]);
        assert_render(&insn, "add     x0"); // "add" + 5 spaces == 8
    }

    #[test]
    fn no_operands_no_padding() {
        let insn = make(Mnemonic::Nop, &[]);
        assert_render(&insn, "nop");
    }

    #[test]
    fn bcond_label() {
        // B.eq 0x1000 — condition is folded into the operand stream here; the
        // label renders as a bare absolute address.
        let insn = make(Mnemonic::B, &[Operand::Cond(Condition::Eq), Operand::Label(0x1000)]);
        assert_render(&insn, "b       eq, 0x1000");
    }

    #[test]
    fn ldr_preindex() {
        // LDR x0, [x1, #8]!
        let insn = make(
            Mnemonic::Ldr,
            &[
                reg(Register::X0),
                Operand::MemImm {
                    base: Register::X1,
                    imm: 8,
                    mode: MemIndexMode::PreIndex,
                },
            ],
        );
        assert_render(&insn, "ldr     x0, [x1, #0x8]!");
    }

    #[test]
    fn ldr_offset_zero_elides_imm() {
        // LDR x0, [x1]
        let insn = make(
            Mnemonic::Ldr,
            &[
                reg(Register::X0),
                Operand::MemImm {
                    base: Register::X1,
                    imm: 0,
                    mode: MemIndexMode::Offset,
                },
            ],
        );
        assert_render(&insn, "ldr     x0, [x1]");
    }

    #[test]
    fn ldr_postindex() {
        // LDR x0, [x1], #8
        let insn = make(
            Mnemonic::Ldr,
            &[
                reg(Register::X0),
                Operand::MemImm {
                    base: Register::X1,
                    imm: 8,
                    mode: MemIndexMode::PostImm,
                },
            ],
        );
        assert_render(&insn, "ldr     x0, [x1], #0x8");
    }

    #[test]
    fn ldr_offset_negative_signed() {
        // LDUR x0, [x1, #-8]
        let insn = make(
            Mnemonic::Ldur,
            &[
                reg(Register::X0),
                Operand::MemImm {
                    base: Register::X1,
                    imm: -8,
                    mode: MemIndexMode::Offset,
                },
            ],
        );
        assert_render(&insn, "ldur    x0, [x1, #-0x8]");
    }

    #[test]
    fn mem_ext_register_index() {
        // LDR x0, [x1, w2, sxtw #2] -- the load/store decoder packs (S<<7)|amt,
        // so a shown amount of 2 is encoded as 0x82.
        let insn = make(
            Mnemonic::Ldr,
            &[
                reg(Register::X0),
                Operand::MemExt {
                    base: Register::X1,
                    index: Register::W2,
                    extend: ExtendType::Sxtw,
                    shift: 0x82,
                },
            ],
        );
        assert_render(&insn, "ldr     x0, [x1, w2, sxtw #0x2]");
    }

    #[test]
    fn mem_ext_sxtx_amount_zero_shown() {
        // LDR x0, [x1, x2, sxtx #0x0] -- S==1 (bit7) forces the `#0x0` to show.
        let insn = make(
            Mnemonic::Ldr,
            &[
                reg(Register::X0),
                Operand::MemExt {
                    base: Register::X1,
                    index: Register::X2,
                    extend: ExtendType::Sxtx,
                    shift: 0x80,
                },
            ],
        );
        assert_render(&insn, "ldr     x0, [x1, x2, sxtx #0x0]");
    }

    #[test]
    fn mem_ext_lsl_zero_elided() {
        // LDR x0, [x1, x2]  -> uxtx (LSL-equiv) with S==0 prints no decoration.
        let insn = make(
            Mnemonic::Ldr,
            &[
                reg(Register::X0),
                Operand::MemExt {
                    base: Register::X1,
                    index: Register::X2,
                    extend: ExtendType::Uxtx,
                    shift: 0,
                },
            ],
        );
        assert_render(&insn, "ldr     x0, [x1, x2]");
    }

    #[test]
    fn mem_ext_lsl_amount_shown() {
        // LDR x0, [x1, x2, lsl #0x3]  -> uxtx with S==1 renders as `lsl #amt`.
        let insn = make(
            Mnemonic::Ldr,
            &[
                reg(Register::X0),
                Operand::MemExt {
                    base: Register::X1,
                    index: Register::X2,
                    extend: ExtendType::Uxtx,
                    shift: 0x83,
                },
            ],
        );
        assert_render(&insn, "ldr     x0, [x1, x2, lsl #0x3]");
    }

    #[test]
    fn shifted_register() {
        // ADD x0, x1, x2, lsl #3
        let insn = make(
            Mnemonic::Add,
            &[
                reg(Register::X0),
                reg(Register::X1),
                Operand::Reg {
                    reg: Register::X2,
                    arr: None,
                    lane: None,
                    shift: Some((ShiftType::Lsl, 3)),
                    extend: None,
                    pred: None,
                },
            ],
        );
        assert_render(&insn, "add     x0, x1, x2, lsl #0x3");
    }

    #[test]
    fn extended_register() {
        // ADD x0, x1, w2, uxtw #2
        let insn = make(
            Mnemonic::Add,
            &[
                reg(Register::X0),
                reg(Register::X1),
                Operand::Reg {
                    reg: Register::W2,
                    arr: None,
                    lane: None,
                    shift: Some((ShiftType::Lsl, 2)),
                    extend: Some(ExtendType::Uxtw),
                    pred: None,
                },
            ],
        );
        assert_render(&insn, "add     x0, x1, w2, uxtw #0x2");
    }

    #[test]
    fn wide_move_with_shift() {
        // MOVZ x0, #0x1234, lsl #16
        let insn = make(
            Mnemonic::Movz,
            &[
                reg(Register::X0),
                Operand::ImmShiftedMove {
                    imm: 0x1234,
                    lsl: 16,
                },
            ],
        );
        assert_render(&insn, "movz    x0, #0x1234, lsl #0x10");
    }

    #[test]
    fn wide_move_no_shift() {
        // MOVZ x0, #0x1234
        let insn = make(
            Mnemonic::Movz,
            &[
                reg(Register::X0),
                Operand::ImmShiftedMove { imm: 0x1234, lsl: 0 },
            ],
        );
        assert_render(&insn, "movz    x0, #0x1234");
    }

    #[test]
    fn vector_arrangement_and_lane() {
        // arrangement on a bare register and a lane on another.
        let bare = Operand::Reg {
            reg: Register::V0,
            arr: Some(VectorArrangement::V4S),
            lane: None,
            shift: None,
            extend: None,
            pred: None,
        };
        let laned = Operand::Reg {
            reg: Register::V1,
            arr: Some(VectorArrangement::V4S),
            lane: Some(2),
            shift: None,
            extend: None,
            pred: None,
        };
        let insn = make(Mnemonic::Add, &[bare, laned]);
        assert_render(&insn, "add     v0.4s, v1.s[2]");
    }

    #[test]
    fn multireg_list() {
        // {v0.4s, v1.4s}
        let regs = [Register::V0, Register::V1, Register::None, Register::None];
        let op = Operand::MultiReg {
            regs,
            count: 2,
            arr: Some(VectorArrangement::V4S),
            lane: None,
        };
        let insn = make(Mnemonic::Ld1, &[op]);
        assert_render(&insn, "ld1     {v0.4s, v1.4s}");
    }

    #[test]
    fn sve_predicate_qualifier() {
        // z0.d with /z and another with /m
        let z = Operand::Reg {
            reg: Register::Z0,
            arr: Some(VectorArrangement::Sd),
            lane: None,
            shift: None,
            extend: None,
            pred: Some(PredQual::Zeroing),
        };
        let insn = make(Mnemonic::Mov, &[z]);
        assert_render(&insn, "mov     z0.d/z");
    }

    #[test]
    fn fp_immediate() {
        // FMOV s0, #1.0 — shortest round-trip, one fractional digit kept.
        let insn = make(Mnemonic::Fmov, &[reg(Register::S0), Operand::FpImm(1.0)]);
        assert_render(&insn, "fmov    s0, #1.0");
    }

    #[test]
    fn fp_immediate_negative_fraction() {
        let insn = make(Mnemonic::Fmov, &[reg(Register::D0), Operand::FpImm(-2.5)]);
        assert_render(&insn, "fmov    d0, #-2.5");
    }

    #[test]
    fn fp_immediate_zero_and_dyadic() {
        // The `#0.0` compare spelling and a multi-digit dyadic fraction.
        let z = make(Mnemonic::Fcmp, &[reg(Register::S0), Operand::FpImm(0.0)]);
        assert_render(&z, "fcmp    s0, #0.0");
        let frac = make(Mnemonic::Fmov, &[reg(Register::D0), Operand::FpImm(-0.21875)]);
        assert_render(&frac, "fmov    d0, #-0.21875");
    }

    #[test]
    fn signed_immediate_negative() {
        let insn = make(Mnemonic::Movn, &[reg(Register::X0), Operand::ImmSigned(-5)]);
        assert_render(&insn, "movn    x0, #-0x5");
    }

    #[test]
    fn shift_amount_operand() {
        let insn = make(
            Mnemonic::Lsl,
            &[reg(Register::X0), reg(Register::X1), Operand::ShiftAmount(7)],
        );
        assert_render(&insn, "lsl     x0, x1, #7");
    }

    #[test]
    fn label_absolute() {
        let insn = make(Mnemonic::Bl, &[Operand::Label(0xdead_beef)]);
        assert_render(&insn, "bl      0xdeadbeef");
    }

    #[test]
    fn uppercase_option() {
        let mut fmt = FmtFormatter::new();
        fmt.options_mut().uppercase_mnemonics = true;
        fmt.options_mut().uppercase_registers = true;
        let insn = make(
            Mnemonic::Add,
            &[reg(Register::X0), reg(Register::X1), Operand::ImmUnsigned(1)],
        );
        assert_render_with(&fmt, &insn, "ADD     X0, X1, #0x1");
    }

    #[test]
    fn use_xzr_not_sp_option() {
        // With use_sp_not_xzr = false, the SP view is rewritten to the zero view.
        let mut fmt = FmtFormatter::new();
        fmt.options_mut().use_sp_not_xzr = false;
        let insn = make(Mnemonic::Add, &[reg(Register::Sp), reg(Register::Xzr)]);
        assert_render_with(&fmt, &insn, "add     xzr, xzr");
    }

    #[test]
    fn sp_kept_by_default() {
        let insn = make(
            Mnemonic::Add,
            &[reg(Register::Sp), reg(Register::Sp), Operand::ImmUnsigned(16)],
        );
        assert_render(&insn, "add     sp, sp, #0x10");
    }

    #[test]
    fn no_space_separator_option() {
        let mut fmt = FmtFormatter::new();
        fmt.options_mut().space_after_operand_separator = false;
        let insn = make(
            Mnemonic::Add,
            &[reg(Register::X0), reg(Register::X1), Operand::ImmUnsigned(1)],
        );
        assert_render_with(&fmt, &insn, "add     x0,x1,#0x1");
    }

    #[test]
    fn show_lsl_zero_option() {
        let mut fmt = FmtFormatter::new();
        fmt.options_mut().show_lsl_zero = true;
        let insn = make(
            Mnemonic::Movz,
            &[
                reg(Register::X0),
                Operand::ImmShiftedMove { imm: 0x10, lsl: 0 },
            ],
        );
        assert_render_with(&fmt, &insn, "movz    x0, #0x10, lsl #0x0");
    }

    #[test]
    fn condition_operand_name() {
        // CCMP x0, x1, #0, eq
        let insn = make(
            Mnemonic::Ccmp,
            &[
                reg(Register::X0),
                reg(Register::X1),
                Operand::ImmUnsigned(0),
                Operand::Cond(Condition::Eq),
            ],
        );
        assert_render(&insn, "ccmp    x0, x1, #0x0, eq");
    }

    #[test]
    fn bufsink_overflow_is_tracked() {
        // A tiny buffer overflows; the formatter must not write past the end and
        // must mark overflow.
        let insn = make(
            Mnemonic::Add,
            &[reg(Register::X0), reg(Register::X1), Operand::ImmUnsigned(1)],
        );
        let mut buf = [0u8; 4];
        let mut sink = BufSink::new(&mut buf);
        FmtFormatter::new().format(&insn, &mut sink);
        assert!(sink.overflowed());
        // Whatever made it in is valid UTF-8 and within bounds.
        assert!(sink.len() <= 4);
        let _ = sink.as_str();
    }
}
