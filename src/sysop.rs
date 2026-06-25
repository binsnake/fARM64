//! Fixed system-instruction *operation* and *option* name tokens.
//!
//! A [`SysToken`] is a compact `u16` index into a `const` table of the
//! `&'static str` keyword operands used by the System encoding group: barrier
//! options (`sy`/`ish`/...), the `csync` of `PSB`/`TSB`, `MSR (immediate)`
//! PSTATE field names (`daifset`/`pan`/...), `BTI` targets (`c`/`j`/`jc`), the
//! `cN` register tokens of the canonical `SYS`/`SYSL`, and the
//! `IC`/`DC`/`AT`/`TLBI`/`CFP`/`CPP`/`DVP` operation names (`ialluis`/`zva`/
//! `rctx`/`alle3`/...).
//!
//! Storing an index (rather than a `&'static str`) keeps
//! [`crate::operand::Operand`] within its 16-byte budget while still rendering a
//! real keyword with zero allocation. The decoder builds tokens by *name* via
//! [`SysToken::of`], which resolves to the table index; an unknown name yields
//! the empty token (index 0), so the path is total and never panics.

/// A compact handle to a fixed system keyword operand (see module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SysToken(pub(crate) u16);

impl SysToken {
    /// The empty/sentinel token (renders `""`).
    pub(crate) const EMPTY: SysToken = SysToken(0);

    /// Resolve a keyword to its [`SysToken`] by linear lookup in the name table.
    ///
    /// Intended for the decoder, which passes string literals from this module's
    /// own table, so the lookup always succeeds; an unknown name maps to
    /// [`SysToken::EMPTY`] to stay total.
    #[inline]
    pub(crate) fn of(name: &str) -> SysToken {
        let mut i = 0usize;
        while i < SYSOP_NAMES.len() {
            if str_eq(SYSOP_NAMES[i], name) {
                return SysToken(i as u16);
            }
            i += 1;
        }
        SysToken::EMPTY
    }

    /// The `cN` token for a 4-bit `CRn`/`CRm` field (`c0`..`c15`). These occupy
    /// table slots `1..=16` by construction.
    #[inline]
    pub(crate) fn cr(n: u32) -> SysToken {
        SysToken(1 + (n & 0xf) as u16)
    }

    /// The keyword text for this token, or `""` if the index is out of range.
    #[inline]
    pub const fn name(self) -> &'static str {
        let i = self.0 as usize;
        if i < SYSOP_NAMES.len() {
            SYSOP_NAMES[i]
        } else {
            ""
        }
    }
}

/// Byte-wise `&str` equality (avoids pulling in any non-const machinery).
#[inline]
fn str_eq(a: &str, b: &str) -> bool {
    a.as_bytes() == b.as_bytes()
}

/// The flat keyword table. Slot `0` is the empty sentinel and slots `1..=16` are
/// `c0..c15` (relied on by [`SysToken::cr`]); the remaining order is arbitrary.
pub(crate) static SYSOP_NAMES: &[&str] = &[
    // 0: sentinel / empty.
    "",
    // 1..=16: cN register tokens c0..c15.
    "c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10", "c11", "c12", "c13", "c14",
    "c15",
    // barrier options.
    "sy", "st", "ld", "ish", "ishst", "ishld", "nsh", "nshst", "nshld", "osh", "oshst", "oshld",
    // nXS barrier options (FEAT_XS).
    "synxs", "nshnxs", "nshstnxs", "nshldnxs", "ishnxs", "ishstnxs", "ishldnxs", "oshnxs",
    "oshstnxs", "oshldnxs",
    // misc keyword operands.
    "csync", "c", "j", "jc", "rctx",
    // SME SMSTART/SMSTOP option keywords (`smstart sm`, `smstart za`).
    "sm", "za",
    // PRFM/PRFUM prefetch operations (<type>l<target><policy>).
    "pldl1keep", "pldl1strm", "pldl2keep", "pldl2strm", "pldl3keep", "pldl3strm", "plil1keep",
    "plil1strm", "plil2keep", "plil2strm", "plil3keep", "plil3strm", "pstl1keep", "pstl1strm",
    "pstl2keep", "pstl2strm", "pstl3keep", "pstl3strm",
    // PSTATE fields.
    "spsel", "daifset", "daifclr", "uao", "pan", "dit", "ssbs", "tco", "allint", "svcrsm", "svcrza",
    "svcrsmza",
    // IC ops.
    "ialluis", "iallu", "ivau",
    // DC ops.
    "ivac", "isw", "igvac", "igsw", "igdvac", "igdsw", "csw", "cgsw", "cgdsw", "cisw", "cigsw",
    "cigdsw", "zva", "cvac", "cgvac", "cgdvac", "cvau", "cvap", "cgvap", "cgdvap", "cvadp",
    "cgvadp", "cgdvadp", "civac", "cigvac", "cigdvac",
    // AT ops.
    "s1e1r", "s1e1w", "s1e0r", "s1e0w", "s1e1rp", "s1e1wp", "s1e2r", "s1e2w", "s12e1r", "s12e1w",
    "s12e0r", "s12e0w", "s1e3r", "s1e3w", "s1e2rp", "s1e2wp",
    // TLBI ops.
    "vmalle1is", "vae1is", "aside1is", "vaae1is", "vale1is", "vaale1is", "vmalle1", "vae1",
    "aside1", "vaae1", "vale1", "vaale1", "ipas2e1is", "ipas2le1is", "alle2is", "vae2is", "alle1is",
    "vale2is", "vmalls12e1is", "ipas2e1", "ipas2le1", "alle2", "vae2", "alle1", "vale2",
    "vmalls12e1", "alle3is", "vae3is", "vale3is", "alle3", "vae3", "vale3",
];
