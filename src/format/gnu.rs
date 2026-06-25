//! Optional GNU / objdump-style formatter dialect (`feature = "fmt-gnu"`).
//!
//! Same [`Formatter`] trait, alternate rendering policy. Pure `no_std`, zero
//! alloc — like [`super::FmtFormatter`] it writes through the
//! [`FormatterOutput`] sink.

use super::{FmtFormatter, Formatter, FormatterOptions, FormatterOutput};
use crate::instruction::Instruction;

/// A GNU/objdump-flavoured formatter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GnuFormatter {
    opts: FormatterOptions,
}

impl GnuFormatter {
    /// A GNU-dialect formatter with default options.
    #[inline]
    pub fn new() -> Self {
        GnuFormatter {
            opts: FormatterOptions::default(),
        }
    }

    /// A GNU-dialect formatter with explicit options.
    #[inline]
    pub fn with_options(opts: FormatterOptions) -> Self {
        GnuFormatter { opts }
    }
}

impl Default for GnuFormatter {
    #[inline]
    fn default() -> Self {
        GnuFormatter::new()
    }
}

impl Formatter for GnuFormatter {
    // The GNU/objdump dialect currently shares the UAL rendering engine; it is a
    // thin wrapper around [`FmtFormatter`] carrying its own options so the two
    // dialects can diverge later (e.g. objdump's lower-case hex with no padding)
    // without touching call sites. Delegating keeps a single operand-dispatch
    // path and avoids a second `todo!()`-shaped stub.
    #[inline]
    fn format(&self, insn: &Instruction, out: &mut dyn FormatterOutput) {
        FmtFormatter::with_options(self.opts).format(insn, out);
    }

    #[inline]
    fn format_mnemonic(&self, insn: &Instruction, out: &mut dyn FormatterOutput) {
        FmtFormatter::with_options(self.opts).format_mnemonic(insn, out);
    }

    #[inline]
    fn format_operand(&self, insn: &Instruction, n: usize, out: &mut dyn FormatterOutput) {
        FmtFormatter::with_options(self.opts).format_operand(insn, n, out);
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
