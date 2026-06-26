//! Architecture-extension model.
//!
//! [`FeatureSet`] carries **two** `u64` bitfields, kept deliberately
//! **separate**: `features0` gates *decode-time* structural admission
//! (`FEAT_*` "is this encoding present") and `features1` gates *pseudocode-time*
//! behaviour (`FEAT_*` "is this behaviour available"). They are split because
//! the ARM ARM treats those two questions independently. This is a runtime
//! accept/reject layer that is independent of the cargo compile-out features.

/// A single architecture extension identity, for per-encoding gating and
/// [`crate::DecodeError::FeatureRequired`].
///
/// One bit position per `FEAT_*` extension. Used by [`crate::Code::feature`]
/// and [`FeatureSet`]. The spine below is representative; the full set is
/// completed by codegen.
#[repr(u16)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Feature {
    /// Base ISA — always available, never gated.
    Base = 0,
    /// Half-precision floating point (`FEAT_FP16`).
    Fp16,
    /// BFloat16 (`FEAT_BF16`).
    Bf16,
    /// Large System Extensions / atomics (`FEAT_LSE`).
    Lse,
    /// Pointer authentication (`FEAT_PAuth`).
    PAuth,
    /// Memory tagging (`FEAT_MTE`).
    Mte,
    /// Scalable Vector Extension (`FEAT_SVE`).
    Sve,
    /// Scalable Matrix Extension (`FEAT_SME`).
    Sme,
    /// Cryptographic extensions (AES/SHA/SM3/SM4 family).
    Crypto,
    /// Transactional Memory Extension (`FEAT_TME`).
    Tme,
    /// Trace synchronization barrier (`FEAT_TRF`).
    Trf,
    /// Wait-for-event/interrupt with timeout (`FEAT_WFxT`).
    Wfxt,
    /// FP round-to-integer with rounding to 32/64-bit integers (`FEAT_FRINTTS`).
    Frintts,
    /// Single-copy atomic 64-byte load/store (`FEAT_LS64`).
    Ls64,
    /// Memory system non-XS qualifier for barriers (`FEAT_XS`): the `DSB`
    /// `<option>nXS` variants.
    Xs,
    /// Memory Copy and Memory Set instructions (`FEAT_MOPS`): the `CPYF*`/`CPY*`
    /// and `SET*`/`SETG*` families.
    Mops,
    /// Common Short Sequence Compression (`FEAT_CSSC`): the `ABS`/`CNT`/`CTZ`
    /// 1-source forms and the `SMAX`/`SMIN`/`UMAX`/`UMIN` register- and
    /// immediate-operand min/max forms.
    Cssc,
    /// Release Consistent processor consistent, version 3 (`FEAT_LRCPC3`): the
    /// SIMD&FP `LDAPUR`/`STLUR` (unscaled release/acquire) forms and the
    /// `LDIAPP`/`STILP` ordered load/store-pair forms.
    Rcpc3,
    /// 128-bit system-register and translation support (`FEAT_D128`): the
    /// `MRRS`/`MSRR` 128-bit system-register pair moves and the `SYSP`/`TLBIP`
    /// system-pair operations.
    D128,
    /// Translation Hardening Extension (`FEAT_THE`): the unprivileged
    /// translation-enhanced load/store-pair forms `LDTP`/`STTP` (post/offset/pre)
    /// and the non-temporal `LDTNP`/`STTNP`.
    The,
    /// 128-bit Large System Extension atomics (`FEAT_LSE128`): the
    /// `LDCLRP`/`LDSETP`/`SWPP` 128-bit atomic load-op-store-pair forms (with
    /// their acquire/release ordering variants).
    Lse128,
    /// SME2 multi-vector extension (`FEAT_SME2`): the multi-vector ZA-array
    /// accumulate / multiply-into-ZA family (`SMLALL`/`FMLAL`/`FMLA`/... with
    /// `za.<T>[w8, <slice>{, vgx2|vgx4}]` destinations) and the `FTMOPA`/`STMOPA`
    /// outer-product forms.
    Sme2,
    /// Unprivileged Load Store (`FEAT_LSUI`): the quadword unprivileged
    /// translation-enhanced load/store-pair forms `LDTP`/`STTP` (post/offset/pre)
    /// and the non-temporal `LDTNP`/`STTNP` with `Q` data registers, plus the
    /// unprivileged atomics — load/store-exclusive (`LDTXR`/`LDATXR`/`STTXR`/
    /// `STLTXR`) and compare-and-swap (`CAST`/`CASAT`/`CASLT`/`CASALT` and the
    /// pair `CASPT`/`CASPAT`/`CASPLT`/`CASPALT`).
    Lsui,
    /// SVE2.1 / SME2.1 (`FEAT_SVE2p1`): the 128-bit-segment quadword permutes
    /// (`ZIPQ1/2`, `UZPQ1/2`, `TBLQ`, `TBXQ`) and the 2-way `SDOT`/`UDOT` `.h`
    /// (`<Zda>.s, <Zn>.h, <Zm>.h{[idx]}`) dot-product forms.
    Sve2p1,
    /// Checked Pointer Arithmetic (`FEAT_CPA`): the pointer-arithmetic
    /// `ADDPT`/`SUBPT` forms — the SVE predicated/unpredicated `.d` vector forms
    /// and the scalar `ADDPT`/`SUBPT` (and `MADDPT`/`MSUBPT`) base forms.
    Cpa,
    /// Compare and Branch (`FEAT_CMPBR`): the register/register and
    /// register/immediate compare-and-branch forms `CB<cc>` (word/doubleword)
    /// plus their byte (`CBB<cc>`) and halfword (`CBH<cc>`) register variants.
    Cmpbr,
    /// Int8 matrix multiply / mixed-sign dot product (`FEAT_I8MM`): the
    /// Advanced SIMD `USDOT`/`SUDOT` byte dot products (vector and by-element).
    I8mm,
    /// 8-bit floating-point (`FEAT_FP8` and the `FP8DOT2`/`FP8DOT4`/`FP8FMA`
    /// sub-features): the Advanced SIMD `FDOT` (to single/half), `FMLALB`/
    /// `FMLALT` and `FMLALLBB`/`FMLALLBT`/`FMLALLTB`/`FMLALLTT` FP8 widening
    /// multiply-accumulate forms (vector and by-element).
    Fp8,
    /// Lookup table (`FEAT_LUT`): the SVE `LUTI2`/`LUTI4` vector lookup-table
    /// reads — single- and two-register table forms with `.b`/`.h` element
    /// variants, indexed by a vector-element selector (`<Zm>[<index>]`).
    Lut,
    // codegen/expand: the remaining ARCH_FEATURE_* extensions.
}

/// The set of architecture extensions the decoder should *accept*.
///
/// Two `u64` words: `features0` gates decode-time structural admission;
/// `features1` gates pcode-time helper behaviour. Passed by value into the
/// [`crate::Decoder`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FeatureSet {
    /// Decode-time guard bits (structural admission of an encoding).
    pub features0: u64,
    /// Pseudocode-time guard bits (availability of a behaviour).
    pub features1: u64,
}

impl FeatureSet {
    /// Base ISA only — no extensions accepted.
    pub const BASE: Self = FeatureSet {
        features0: 0,
        features1: 0,
    };

    /// Every extension accepted (both words all-ones) — a "decode everything"
    /// posture useful for differential testing.
    pub const ALL: Self = FeatureSet {
        features0: u64::MAX,
        features1: u64::MAX,
    };

    /// Nothing accepted (alias of [`FeatureSet::BASE`] for symmetry with the
    /// public-API naming).
    pub const NONE: Self = FeatureSet::BASE;

    /// `true` if extension `f` is present (checks the structural-admission
    /// word). [`Feature::Base`] is always present.
    ///
    /// Each [`Feature`] maps to one bit position given by its discriminant; the
    /// `features0` word carries decode-time structural admission. `Base` (bit 0)
    /// is treated as always-on regardless of the bitfield.
    #[inline]
    pub fn has(self, f: Feature) -> bool {
        let bit = f as u32;
        if bit == 0 {
            // Base ISA is always available.
            return true;
        }
        if bit >= 64 {
            return false;
        }
        (self.features0 & (1u64 << bit)) != 0
    }

    /// Return a copy with extension `f` enabled in both words.
    #[inline]
    pub fn with(self, f: Feature) -> Self {
        let bit = f as u32;
        if bit == 0 || bit >= 64 {
            return self;
        }
        let mask = 1u64 << bit;
        FeatureSet {
            features0: self.features0 | mask,
            features1: self.features1 | mask,
        }
    }
}

impl Default for FeatureSet {
    /// The default accepts everything ([`FeatureSet::ALL`]), so a decoder
    /// constructed with [`Default`] decodes every encoding out of the box.
    #[inline]
    fn default() -> Self {
        FeatureSet::ALL
    }
}
