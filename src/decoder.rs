//! The borrowing, zero-alloc [`Decoder`] — the primary entry point.
//!
//! A `Decoder` borrows a byte slice and walks it 4 bytes at a time, reading each
//! little-endian word and handing it to the hand-written decode tree
//! ([`crate::decode::decode_into`]). It is also an [`Iterator`] (both consuming
//! and by-`&mut`). It never allocates and never panics on malformed input —
//! failures surface as an [`Instruction`] with [`Code::Invalid`] plus a recorded
//! [`last_error`].
//!
//! [`last_error`]: Decoder::last_error

use crate::error::DecodeError;
use crate::features::FeatureSet;
use crate::instruction::Instruction;
use crate::mnemonic::Code;
use crate::INSN_LEN;

/// Decode-time options.
///
/// Distinct from formatter options: these affect *what* is decoded (e.g. which
/// extensions are accepted), not how it is rendered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DecoderOptions {
    /// Architecture extensions to accept. Defaults to [`FeatureSet::ALL`] so
    /// every encoding decodes out of the box.
    pub features: FeatureSet,
}

impl DecoderOptions {
    /// The "no special options" preset, mirroring iced-x86's
    /// [`DecoderOptions::NONE`]. Equivalent to [`DecoderOptions::default`]: it
    /// accepts every architecture extension ([`FeatureSet::ALL`]).
    ///
    /// Unlike iced-x86 there is **no invalid-instruction toggle** to set here:
    /// fARM64 always surfaces an unrecognized or genuinely-undefined word as an
    /// [`Instruction`] with [`Code::Invalid`] (and a recorded
    /// [`Decoder::last_error`]). There is no option that changes that behaviour.
    pub const NONE: DecoderOptions = DecoderOptions {
        features: FeatureSet::ALL,
    };

    /// `const` constructor for the default options ([`FeatureSet::ALL`]) — the
    /// `const fn` analog of [`DecoderOptions::default`], usable in `const`
    /// contexts. Returns the same value as [`DecoderOptions::NONE`].
    #[inline]
    #[must_use]
    pub const fn new() -> DecoderOptions {
        DecoderOptions::NONE
    }
}

impl Default for DecoderOptions {
    #[inline]
    fn default() -> Self {
        DecoderOptions::NONE
    }
}

/// A streaming A64 instruction decoder over a borrowed byte slice.
///
/// There is no bitness parameter — A64 is always 64-bit. `ip` is the address of
/// `data[0]`; PC-relative operands are resolved against the running `ip`.
#[derive(Debug, Clone)]
pub struct Decoder<'a> {
    /// The code bytes being decoded.
    data: &'a [u8],
    /// Byte cursor within `data`.
    pos: usize,
    /// Address corresponding to `data[pos]`.
    ip: u64,
    /// Decode-time options.
    options: DecoderOptions,
    /// Status of the most recent [`Decoder::decode`] call.
    last_error: DecodeError,
}

impl<'a> Decoder<'a> {
    /// Create a decoder over `data`, treating `data[0]` as address `ip`. Never
    /// panics.
    #[inline]
    pub fn new(data: &'a [u8], ip: u64, options: DecoderOptions) -> Decoder<'a> {
        Decoder {
            data,
            pos: 0,
            ip,
            options,
            last_error: DecodeError::None,
        }
    }

    /// Fallible constructor for symmetry; validates option/feature consistency.
    #[inline]
    pub fn try_new(
        data: &'a [u8],
        ip: u64,
        options: DecoderOptions,
    ) -> Result<Decoder<'a>, DecodeError> {
        Ok(Decoder::new(data, ip, options))
    }

    /// Decode one instruction, advance the cursor by 4, and return a `Copy`
    /// [`Instruction`]. On failure returns an instruction with
    /// [`Code::Invalid`] and records [`Decoder::last_error`].
    #[inline]
    pub fn decode(&mut self) -> Instruction {
        let mut out = Instruction::default();
        self.decode_into(&mut out);
        out
    }

    /// Primary zero-copy decode method; preferred in tight loops.
    ///
    /// Reads 4 little-endian bytes at the cursor into a `u32`, captures `ip`,
    /// hands the word to the hand-written decode tree
    /// ([`crate::decode::decode_into`]), writes the result into `out`, and
    /// advances the cursor and `ip` by [`INSN_LEN`]. [`Decoder::decode`] is a
    /// thin wrapper over this.
    ///
    /// On a short tail (fewer than 4 bytes remaining) it records
    /// [`DecodeError::EndOfInstruction`], leaves `out` as the [`Code::Invalid`]
    /// default, and does not advance.
    pub fn decode_into(&mut self, out: &mut Instruction) {
        // Bounds check: need a full 4-byte word at the cursor.
        let Some(window) = self.data.get(self.pos..self.pos + INSN_LEN) else {
            *out = Instruction::default();
            self.last_error = DecodeError::EndOfInstruction;
            return;
        };

        // A64 is little-endian; assemble the instruction word.
        let word = u32::from_le_bytes([window[0], window[1], window[2], window[3]]);
        let ip = self.ip;

        // Hand-written decode tree builds `out` directly.
        crate::decode::decode_into(word, ip, self.options.features, out);

        // Reflect success/failure for `last_error()`.
        self.last_error = if out.is_invalid() {
            DecodeError::Unmatched
        } else {
            DecodeError::None
        };

        // Fixed-width ISA: always advance by one word.
        self.pos += INSN_LEN;
        self.ip = self.ip.wrapping_add(INSN_LEN as u64);
    }

    /// iced-x86-style alias for [`Decoder::decode_into`]: decode one instruction
    /// into the caller-owned `out`, advancing the cursor and `ip` by
    /// [`INSN_LEN`]. Provided for source compatibility with code written against
    /// iced-x86's `decode_out`.
    #[inline]
    pub fn decode_out(&mut self, out: &mut Instruction) {
        self.decode_into(out);
    }

    /// `true` if at least one full instruction (4 bytes) remains.
    #[inline]
    pub fn can_decode(&self) -> bool {
        self.pos + INSN_LEN <= self.data.len()
    }

    /// Current byte cursor within the slice.
    #[inline]
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Seek the byte cursor; keeps `ip` consistent with the new position
    /// relative to the original base.
    #[inline]
    pub fn set_position(&mut self, pos: usize) {
        let delta = pos as i64 - self.pos as i64;
        self.ip = self.ip.wrapping_add(delta as u64);
        self.pos = pos;
    }

    /// The current decode address.
    #[inline]
    pub fn ip(&self) -> u64 {
        self.ip
    }

    /// Set the current decode address (does not move the byte cursor).
    #[inline]
    pub fn set_ip(&mut self, ip: u64) {
        self.ip = ip;
    }

    /// Status of the most recent decode ([`DecodeError::None`] on success).
    #[inline]
    pub fn last_error(&self) -> DecodeError {
        self.last_error
    }

    /// The options this decoder was created with.
    #[inline]
    pub fn options(&self) -> &DecoderOptions {
        &self.options
    }
}

/// Consuming iterator over a [`Decoder`] (`for insn in decoder`). Yields until
/// fewer than 4 bytes remain.
#[derive(Debug)]
pub struct DecoderIntoIter<'a> {
    dec: Decoder<'a>,
}

impl<'a> Iterator for DecoderIntoIter<'a> {
    type Item = Instruction;

    #[inline]
    fn next(&mut self) -> Option<Instruction> {
        if self.dec.can_decode() {
            Some(self.dec.decode())
        } else {
            None
        }
    }
}

impl<'a> IntoIterator for Decoder<'a> {
    type Item = Instruction;
    type IntoIter = DecoderIntoIter<'a>;

    #[inline]
    fn into_iter(self) -> DecoderIntoIter<'a> {
        DecoderIntoIter { dec: self }
    }
}

/// Borrowing iterator over a [`Decoder`] (`for insn in &mut decoder`); after the
/// loop, inspect [`Decoder::last_error`].
#[derive(Debug)]
pub struct DecoderIter<'a, 'b> {
    dec: &'b mut Decoder<'a>,
}

impl<'a, 'b> Iterator for DecoderIter<'a, 'b> {
    type Item = Instruction;

    #[inline]
    fn next(&mut self) -> Option<Instruction> {
        if self.dec.can_decode() {
            Some(self.dec.decode())
        } else {
            None
        }
    }
}

impl<'a, 'b> IntoIterator for &'b mut Decoder<'a> {
    type Item = Instruction;
    type IntoIter = DecoderIter<'a, 'b>;

    #[inline]
    fn into_iter(self) -> DecoderIter<'a, 'b> {
        DecoderIter { dec: self }
    }
}

// Keep `Code` referenced in this module's documented surface.
const _: fn() -> Code = || Code::Invalid;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mnemonic::Mnemonic;

    #[test]
    fn options_none_and_new_equal_default() {
        assert_eq!(DecoderOptions::NONE, DecoderOptions::default());
        assert_eq!(DecoderOptions::new(), DecoderOptions::default());
        assert_eq!(DecoderOptions::NONE.features, FeatureSet::ALL);
        // `new()` is usable in a `const` context.
        const OPTS: DecoderOptions = DecoderOptions::new();
        assert_eq!(OPTS, DecoderOptions::NONE);
    }

    #[test]
    fn decode_out_aliases_decode_into() {
        // `nop` = 0xD503201F.
        let bytes = 0xD503_201Fu32.to_le_bytes();
        let mut a = Decoder::new(&bytes, 0, DecoderOptions::new());
        let mut b = Decoder::new(&bytes, 0, DecoderOptions::NONE);
        let mut via_out = Instruction::default();
        let mut via_into = Instruction::default();
        a.decode_out(&mut via_out);
        b.decode_into(&mut via_into);
        assert_eq!(via_out, via_into);
        assert_eq!(via_out.mnemonic(), Mnemonic::Nop);
        // The cursor advanced exactly one word.
        assert_eq!(a.position(), INSN_LEN);
    }
}
