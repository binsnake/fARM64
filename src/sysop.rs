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
    // RPRFM range-prefetch operations (<type><policy>, no level component).
    "pldkeep", "pstkeep", "pldstrm", "pststrm",
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
    // K4: TCHANGE `nb` (no-barrier) modifier keyword.
    "nb",
    // T: GCS / hint operand keywords (`gcsb dsync`, `shuh ph`, `stshh keep|strm`).
    "dsync", "ph", "keep", "strm",
    // T: newer TLBI/DC/AT ops (OS/NXS/range/GPT), PLBI, GIC/GICR, MLBI, COSP, and
    // the GICR read-side op-names. Swept from the LLVM oracle.
    "alle1isnxs", "alle1nxs", "alle1os", "alle1osnxs", "alle2isnxs", "alle2nxs", "alle2os", "alle2osnxs",
    "alle3isnxs", "alle3nxs", "alle3os", "alle3osnxs", "aside1isnxs", "aside1nxs", "aside1os", "aside1osnxs",
    "cdaff", "cddi", "cddis", "cden", "cdeoi", "cdhm", "cdia", "cdnmia",
    "cdpend", "cdpri", "cdrcfg", "cgdvaoc", "cigdpae", "cigdpapa", "cigdvaoc", "cigdvaps",
    "cipae", "cipapa", "civaoc", "civaps", "cvaoc", "gbva", "gva",
    "gzva", "ipas2e1isnxs", "ipas2e1nxs", "ipas2e1os", "ipas2e1osnxs", "ipas2le1isnxs", "ipas2le1nxs", "ipas2le1os",
    "ipas2le1osnxs", "ldaff", "lddi", "lddis", "lden", "ldhm", "ldpend",
    "ldpri", "ldrcfg", "paall", "paallnxs", "paallos", "paallosnxs", "permae1", "permae1is",
    "permae1isnxs", "permae1nxs", "permae1os", "permae1osnxs", "perme1", "perme1is", "perme1isnxs", "perme1nxs",
    "perme1os", "perme1osnxs", "perme2", "perme2is", "perme2isnxs", "perme2nxs", "perme2os", "perme2osnxs",
    "perme3", "perme3is", "perme3isnxs", "perme3nxs", "perme3os", "perme3osnxs", "ripas2e1",
    "ripas2e1is", "ripas2e1isnxs", "ripas2e1nxs", "ripas2e1os", "ripas2e1osnxs", "ripas2le1", "ripas2le1is", "ripas2le1isnxs",
    "ripas2le1nxs", "ripas2le1os", "ripas2le1osnxs", "rpalos", "rpalosnxs", "rpaos", "rpaosnxs", "rvaae1",
    "rvaae1is", "rvaae1isnxs", "rvaae1nxs", "rvaae1os", "rvaae1osnxs", "rvaale1", "rvaale1is", "rvaale1isnxs",
    "rvaale1nxs", "rvaale1os", "rvaale1osnxs", "rvae1", "rvae1is", "rvae1isnxs", "rvae1nxs", "rvae1os",
    "rvae1osnxs", "rvae2", "rvae2is", "rvae2isnxs", "rvae2nxs", "rvae2os", "rvae2osnxs", "rvae3",
    "rvae3is", "rvae3isnxs", "rvae3nxs", "rvae3os", "rvae3osnxs", "rvale1", "rvale1is", "rvale1isnxs",
    "rvale1nxs", "rvale1os", "rvale1osnxs", "rvale2", "rvale2is", "rvale2isnxs", "rvale2nxs", "rvale2os",
    "rvale2osnxs", "rvale3", "rvale3is", "rvale3isnxs", "rvale3nxs", "rvale3os", "rvale3osnxs", "s1e1a",
    "s1e2a", "s1e3a", "vaae1isnxs", "vaae1nxs", "vaae1os", "vaae1osnxs", "vaale1isnxs",
    "vaale1nxs", "vaale1os", "vaale1osnxs", "vae1isnxs", "vae1nxs", "vae1os", "vae1osnxs", "vae2isnxs",
    "vae2nxs", "vae2os", "vae2osnxs", "vae3isnxs", "vae3nxs", "vae3os", "vae3osnxs", "vale1isnxs",
    "vale1nxs", "vale1os", "vale1osnxs", "vale2isnxs", "vale2nxs", "vale2os", "vale2osnxs", "vale3isnxs",
    "vale3nxs", "vale3os", "vale3osnxs", "vdaff", "vddi", "vddis", "vden", "vdhm",
    "vdpend", "vdpri", "vdrcfg", "vmalle1isnxs", "vmalle1nxs", "vmalle1os", "vmalle1osnxs", "vmalls12e1isnxs",
    "vmalls12e1nxs", "vmalls12e1os", "vmalls12e1osnxs", "vmallws2e1", "vmallws2e1is", "vmallws2e1isnxs", "vmallws2e1nxs", "vmallws2e1os",
    "vmallws2e1osnxs", "vpide1", "vpmge1", "zgbva",
    // T: GSB (GICv5 stream barrier) and BRB (branch-record buffer) op-names.
    "sys", "ack", "iall", "inj",
];
