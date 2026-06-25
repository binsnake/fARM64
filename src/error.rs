//! The core decode-error enum.
//!
//! A small set of failure reasons, each carrying a stable negative status code
//! (`-9..=0`) for C/FFI interop, plus a [`DecodeError::FeatureRequired`] case.
//!
//! This is a plain enum implementing [`core::fmt::Display`]. It implements
//! [`std::error::Error`] **only** under `feature = "std"` so the default
//! `no_std` core stays heap- and std-free.

use crate::features::Feature;

/// Why a decode did not yield a valid instruction (or [`DecodeError::None`] on
/// success).
///
/// | Variant | status |
/// |-|-|
/// | [`DecodeError::None`] | `0` |
/// | [`DecodeError::Reserved`] | `-1` |
/// | [`DecodeError::Unmatched`] | `-2` |
/// | [`DecodeError::Unallocated`] | `-3` |
/// | [`DecodeError::Undefined`] | `-4` |
/// | [`DecodeError::EndOfInstruction`] | `-5` |
/// | [`DecodeError::Lost`] | `-6` |
/// | [`DecodeError::Unreachable`] | `-7` |
/// | [`DecodeError::AssertFailed`] | `-8` |
/// | [`DecodeError::ErrorOperands`] | `-9` |
/// | [`DecodeError::FeatureRequired`] | `-4` |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DecodeError {
    /// Success — the instruction decoded cleanly. (`DECODE_STATUS_OK`)
    None,
    /// The encoding lands in spec-reserved space. (`DECODE_STATUS_RESERVED`)
    Reserved,
    /// Decoding fell through all of the spec's checks. (`DECODE_STATUS_UNMATCHED`)
    Unmatched,
    /// The bit pattern is unallocated. (`DECODE_STATUS_UNALLOCATED`)
    Unallocated,
    /// The encoding is explicitly UNDEFINED. (`DECODE_STATUS_UNDEFINED`)
    Undefined,
    /// Reached the end of the input before a full 4-byte instruction word was
    /// available, or a `HINT`-style sentinel meaning the instruction ended.
    EndOfInstruction,
    /// Descended past valid checks ("SEE encoding higher up").
    Lost,
    /// Hit a pseudocode `Unreachable()`.
    Unreachable,
    /// Failed an internal assertion.
    AssertFailed,
    /// Operand construction failed.
    ErrorOperands,
    /// A required architecture extension was not enabled in the
    /// [`crate::FeatureSet`]; carries the missing [`Feature`].
    FeatureRequired(Feature),
}

impl DecodeError {
    /// The stable integer status code (`-9..=0`) for C/FFI interop; the
    /// [`DecodeError::FeatureRequired`] case maps to the "undefined" status `-4`.
    #[inline]
    pub const fn status(self) -> i32 {
        match self {
            DecodeError::None => 0,
            DecodeError::Reserved => -1,
            DecodeError::Unmatched => -2,
            DecodeError::Unallocated => -3,
            DecodeError::Undefined => -4,
            DecodeError::EndOfInstruction => -5,
            DecodeError::Lost => -6,
            DecodeError::Unreachable => -7,
            DecodeError::AssertFailed => -8,
            DecodeError::ErrorOperands => -9,
            DecodeError::FeatureRequired(_) => -4,
        }
    }

    /// `true` only for [`DecodeError::None`].
    #[inline]
    pub const fn is_ok(self) -> bool {
        matches!(self, DecodeError::None)
    }
}

impl Default for DecodeError {
    #[inline]
    fn default() -> Self {
        DecodeError::None
    }
}

impl core::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let s = match self {
            DecodeError::None => "ok",
            DecodeError::Reserved => "reserved encoding",
            DecodeError::Unmatched => "unmatched encoding",
            DecodeError::Unallocated => "unallocated encoding",
            DecodeError::Undefined => "undefined encoding",
            DecodeError::EndOfInstruction => "end of instruction",
            DecodeError::Lost => "decode lost (see-higher)",
            DecodeError::Unreachable => "unreachable pcode",
            DecodeError::AssertFailed => "assertion failed",
            DecodeError::ErrorOperands => "operand build error",
            DecodeError::FeatureRequired(_) => "required feature not enabled",
        };
        f.write_str(s)
    }
}

// `std::error::Error` is opt-in (Tier C) so the default `no_std` core never
// references `std`.
#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl std::error::Error for DecodeError {}
