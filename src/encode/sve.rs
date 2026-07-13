//! Encoder for the SVE / SVE2 group — the inverse of [`crate::decode::sve`].
//!
//! Gated behind `#[cfg(feature = "sve")]`. Without the feature the
//! [`crate::encode::sve::encode`] stub returns
//! [`crate::encode::EncodeError::Unsupported`] and the default build still
//! compiles. With it, every `Sve*` [`crate::mnemonic::Code`] the decoder produces
//! is inverted: dispatch on [`crate::instruction::Instruction::code`], branch on
//! [`crate::instruction::Instruction::mnemonic`] to
//! recover the alias operand layout, then pack the exact bitfields (the inverse
//! of the decoder's field + alias math). Reconstructs the word purely from the
//! instruction's semantics — never reads [`crate::instruction::Instruction::word`]. Total and
//! panic-free.

/// Encode an SVE/SVE2 instruction. Without the `sve` feature this is the
/// compiling stub that declines everything.
#[cfg(not(feature = "sve"))]
#[inline]
pub fn encode(_insn: &crate::instruction::Instruction) -> Result<u32, crate::encode::EncodeError> {
    Err(crate::encode::EncodeError::Unsupported)
}

/// Without the `sve` feature, no code is an SVE code (the tables are compiled
/// out), so the encoder reports `Unsupported` for the whole group.
#[cfg(not(feature = "sve"))]
#[inline]
pub fn is_sve(_code: crate::mnemonic::Code) -> bool {
    false
}

#[cfg(feature = "sve")]
pub use imp::{encode, is_sve};

#[cfg(feature = "sve")]
mod imp;
