//! Formatting: turn an [`Instruction`] into text, **zero-alloc by default**.
//!
//! The [`Formatter`] trait writes through a [`FormatterOutput`] sink. A blanket
//! impl makes **any** [`core::fmt::Write`] a valid sink (so `no_std` callers can
//! use a stack `&mut [u8]` via [`BufSink`], or a `&mut String` under `alloc`).
//! There is a single operand-dispatch path: no parallel string/token code.
//!
//! The default [`FmtFormatter`] renders ARM UAL (Unified Assembler Language)
//! syntax; an optional GNU/objdump dialect lives behind `feature = "fmt-gnu"`.

mod fmt_writer;
pub use fmt_writer::{BufSink, FmtFormatter};

#[cfg(feature = "fmt-gnu")]
#[cfg_attr(docsrs, doc(cfg(feature = "fmt-gnu")))]
mod gnu;
#[cfg(feature = "fmt-gnu")]
#[cfg_attr(docsrs, doc(cfg(feature = "fmt-gnu")))]
pub use gnu::GnuFormatter;

use crate::instruction::Instruction;

/// Classification of an emitted output token (mnemonic, register, immediate,
/// punctuation, decorator, system register, ...), for callers that syntax-color
/// or post-process the rendered text.
///
/// Sinks that only care about the text may ignore the kind (the
/// [`core::fmt::Write`] blanket impl does exactly that).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenKind {
    /// The instruction mnemonic.
    Mnemonic,
    /// A register name.
    Register,
    /// An integer literal.
    Number,
    /// A floating-point literal.
    Float,
    /// An absolute address / PC-relative target.
    Address,
    /// Punctuation (commas handled separately via `OperandSeparator`).
    Punctuation,
    /// The opening `[` of a memory operand.
    BeginMemory,
    /// The closing `]` of a memory operand.
    EndMemory,
    /// The separator between operands (`", "`).
    OperandSeparator,
    /// A decoration such as a shift/extend keyword or predicate qualifier.
    Decorator,
    /// A system-register name.
    SysReg,
}

/// A text/token output sink.
///
/// Implementors receive each chunk of formatted output together with its
/// [`TokenKind`]. A blanket impl is provided for every [`core::fmt::Write`]
/// (ignoring `kind`), so any such writer works in `no_std`; [`BufSink`] gives a
/// fixed `&mut [u8]` target with overflow tracking; and `String` gains a
/// token-collecting impl under `feature = "alloc"`.
pub trait FormatterOutput {
    /// Append `text`, classified as `kind`.
    fn write(&mut self, text: &str, kind: TokenKind);
}

/// Blanket impl: any `core::fmt::Write` is a (text-only) [`FormatterOutput`].
/// Write failures are swallowed (the formatter is infallible by contract);
/// fixed-buffer overflow is observable via [`BufSink::overflowed`].
impl<W: core::fmt::Write> FormatterOutput for W {
    #[inline]
    fn write(&mut self, text: &str, _kind: TokenKind) {
        let _ = core::fmt::Write::write_str(self, text);
    }
}

/// Rendering options. [`Default`] produces ARM UAL output; `aliases` defaults to
/// `true` (preferred disassembly: `MOV`/`CMP`/`MUL`/`LSL`/`NOP`/...).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FormatterOptions {
    /// Emit preferred aliases instead of canonical forms.
    pub aliases: bool,
    /// Upper-case mnemonics.
    pub uppercase_mnemonics: bool,
    /// Upper-case register names.
    pub uppercase_registers: bool,
    /// Render reg-31 as `sp`/`wsp` rather than `xzr`/`wzr` where ambiguous.
    pub use_sp_not_xzr: bool,
    /// Hex prefix for numeric literals (UAL uses `"0x"`).
    pub hex_prefix: &'static str,
    /// Render immediates as signed where the encoding is signed.
    pub signed_immediates: bool,
    /// Show `LSL #0` explicitly instead of eliding it.
    pub show_lsl_zero: bool,
    /// Put a space after the operand separator (`", "` vs `","`).
    pub space_after_operand_separator: bool,
    /// Column at which the first operand starts (mnemonic field width).
    pub first_operand_char_index: u8,
}

impl Default for FormatterOptions {
    #[inline]
    fn default() -> Self {
        FormatterOptions {
            aliases: true,
            uppercase_mnemonics: false,
            uppercase_registers: false,
            use_sp_not_xzr: true,
            hex_prefix: "0x",
            signed_immediates: true,
            show_lsl_zero: false,
            space_after_operand_separator: true,
            first_operand_char_index: 8,
        }
    }
}

/// Renders an [`Instruction`] to text through a [`FormatterOutput`] sink.
///
/// Object-safe (`&dyn Formatter` is usable). All output goes through the sink,
/// so a `Formatter` never allocates on its own.
pub trait Formatter {
    /// Format the whole instruction (mnemonic + all operands).
    fn format(&self, insn: &Instruction, out: &mut dyn FormatterOutput);

    /// Format only the mnemonic (with padding policy applied).
    fn format_mnemonic(&self, insn: &Instruction, out: &mut dyn FormatterOutput);

    /// Format only operand `n`.
    fn format_operand(&self, insn: &Instruction, n: usize, out: &mut dyn FormatterOutput);

    /// Read the active options.
    fn options(&self) -> &FormatterOptions;

    /// Mutate the active options.
    fn options_mut(&mut self) -> &mut FormatterOptions;
}

/// Optional symbolization hook: map an address to a human-readable symbol.
///
/// No allocation is required — a [`SymbolResult`] borrows a `&'static str` or a
/// caller-owned buffer.
pub trait SymbolResolver {
    /// Resolve the address referenced by operand `operand` of `insn`, if known.
    fn symbol(
        &mut self,
        insn: &Instruction,
        operand: usize,
        address: u64,
    ) -> Option<SymbolResult<'_>>;
}

/// A resolved symbol for an address (borrowed name + optional addend).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SymbolResult<'a> {
    /// The symbol name (borrowed; no allocation).
    pub name: &'a str,
    /// Offset from the symbol base, if the address is not exactly on it.
    pub offset: i64,
}

/// Convenience: format an instruction to an owned `String`. Strictly additive;
/// the default `no_std` path never calls this.
#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub fn format_to_string(fmt: &dyn Formatter, insn: &Instruction) -> alloc::string::String {
    let mut s = alloc::string::String::new();
    fmt.format(insn, &mut s);
    s
}
