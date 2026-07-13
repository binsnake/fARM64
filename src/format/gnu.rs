//! Optional GNU-formatter compatibility adapter (`feature = "fmt-gnu"`).
//!
//! This type currently delegates to [`super::FmtFormatter`] and therefore emits
//! the same Arm UAL text. Keeping a distinct type provides an API boundary for
//! future GNU/objdump-specific rendering policy. It is pure `no_std` and
//! zero-alloc, writing through the [`FormatterOutput`] sink.

use super::{FmtFormatter, Formatter, FormatterOptions, FormatterOutput};
use crate::instruction::Instruction;

/// A GNU compatibility formatter that currently emits Arm UAL text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GnuFormatter {
    opts: FormatterOptions,
}

impl GnuFormatter {
    /// A compatibility formatter with default UAL options.
    #[inline]
    pub fn new() -> Self {
        GnuFormatter {
            opts: FormatterOptions::default(),
        }
    }

    /// A compatibility formatter with explicit UAL options.
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
    // Keep a separate options value and API type while sharing the UAL rendering
    // engine. GNU-specific policy can be introduced without changing callers.
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
