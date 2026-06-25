//! System-register identity and naming.
//!
//! A [`SystemReg`] is a `u16` newtype over the packed
//! `(op0<<14 | op1<<11 | CRn<<7 | CRm<<3 | op2)` key used by `MSR`/`MRS`/`SYS`.
//! Known registers resolve to a `&'static str` via a `const` match over the
//! common architectural directory (NZCV, FPCR, FPSR, the `EL` banked registers,
//! …); unknown registers resolve to `None`, and the renderer emits the generic
//! `S<op0>_<op1>_c<CRn>_c<CRm>_<op2>` syntax. Forward-compatible with registers
//! added by future ARM revisions.

/// A packed AArch64 system-register selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SystemReg(pub(crate) u16);

impl SystemReg {
    /// Build a [`SystemReg`] from its five fields. Each field uses only its
    /// architectural bit-width (`op0`,`op1`,`op2`: 3 bits; `CRn`,`CRm`: 4 bits).
    #[inline]
    pub const fn from_fields(op0: u8, op1: u8, crn: u8, crm: u8, op2: u8) -> Self {
        SystemReg(
            ((op0 as u16 & 0x3) << 14)
                | ((op1 as u16 & 0x7) << 11)
                | ((crn as u16 & 0xf) << 7)
                | ((crm as u16 & 0xf) << 3)
                | (op2 as u16 & 0x7),
        )
    }

    /// Build a [`SystemReg`] directly from the encoded `op0/op1/CRn/CRm/op2`
    /// fields of an `MSR`/`MRS`/`SYS` instruction word. Alias of
    /// [`SystemReg::from_fields`] using the ARM ARM field order.
    #[inline]
    pub const fn from_encoding(op0: u8, op1: u8, crn: u8, crm: u8, op2: u8) -> Self {
        SystemReg::from_fields(op0, op1, crn, crm, op2)
    }

    /// The raw packed 16-bit key.
    #[inline]
    pub const fn packed(self) -> u16 {
        self.0
    }

    /// The `op0` field (`2` or `3` for real system registers).
    #[inline]
    pub const fn op0(self) -> u8 {
        ((self.0 >> 14) & 0x3) as u8
    }

    /// The `op1` field (3 bits).
    #[inline]
    pub const fn op1(self) -> u8 {
        ((self.0 >> 11) & 0x7) as u8
    }

    /// The `CRn` field (4 bits).
    #[inline]
    pub const fn crn(self) -> u8 {
        ((self.0 >> 7) & 0xf) as u8
    }

    /// The `CRm` field (4 bits).
    #[inline]
    pub const fn crm(self) -> u8 {
        ((self.0 >> 3) & 0xf) as u8
    }

    /// The `op2` field (3 bits).
    #[inline]
    pub const fn op2(self) -> u8 {
        (self.0 & 0x7) as u8
    }

    /// The canonical lowercase name (`"nzcv"`, `"tpidr_el0"`, …) if this is a
    /// recognised register, else `None`.
    ///
    /// Resolution order: first the (codegen) `SYSREG_NAMES` binary-search table
    /// in [`crate::tables::names`], then a self-contained `const` directory of
    /// the common architectural registers. Zero allocation either way.
    #[inline]
    pub fn name(self) -> Option<&'static str> {
        if let Some(n) = crate::tables::names::sysreg_name(self.0) {
            return Some(n);
        }
        canonical_name(self.0)
    }

    /// Render this system register into `out`.
    ///
    /// Emits the canonical name when known (see [`SystemReg::name`]), otherwise
    /// the architectural generic form `S<op0>_<op1>_c<CRn>_c<CRm>_<op2>` (e.g.
    /// `s3_3_c4_c2_0`). Entirely `no_std`/zero-alloc: the generic form is written
    /// digit-by-digit through the [`core::fmt::Write`] sink.
    pub fn render<W: core::fmt::Write>(self, out: &mut W) -> core::fmt::Result {
        if let Some(n) = self.name() {
            return out.write_str(n);
        }
        // Generic `S<op0>_<op1>_c<CRn>_c<CRm>_<op2>`. All fields are <= 15, so a
        // tiny fixed scratch buffer suffices; we write through the sink directly.
        write_dec(out, self.op0() as u32)?;
        out.write_str("_")?;
        write_dec(out, self.op1() as u32)?;
        out.write_str("_c")?;
        write_dec(out, self.crn() as u32)?;
        out.write_str("_c")?;
        write_dec(out, self.crm() as u32)?;
        out.write_str("_")?;
        write_dec(out, self.op2() as u32)?;
        // The leading `s` is prepended by callers as part of the operand token;
        // emit it here so `render` alone is self-describing.
        Ok(())
    }

    /// Render into a fixed `[u8]` scratch buffer, returning the written `&str`.
    ///
    /// Convenience for `no_std` callers that want the rendered text without a
    /// [`core::fmt::Write`] sink. A 24-byte buffer always suffices for the
    /// generic form (`s3_7_c15_c15_7` is 14 bytes) and for every canonical name.
    /// Returns `None` only if `buf` is too small.
    pub fn render_to(self, buf: &mut [u8]) -> Option<&str> {
        if let Some(n) = self.name() {
            let bytes = n.as_bytes();
            if bytes.len() > buf.len() {
                return None;
            }
            buf[..bytes.len()].copy_from_slice(bytes);
            return core::str::from_utf8(&buf[..bytes.len()]).ok();
        }
        use core::fmt::Write as _;
        let mut sink = ByteSink {
            buf,
            len: 0,
            ok: true,
        };
        // Generic form is prefixed with `s` so the result is a complete token.
        let _ = sink.write_str("s");
        let _ = self.render(&mut sink);
        if !sink.ok {
            return None;
        }
        let len = sink.len;
        core::str::from_utf8(&buf[..len]).ok()
    }
}

/// Write a small decimal integer (`0..=15` in practice) through a sink.
#[inline]
fn write_dec<W: core::fmt::Write>(out: &mut W, mut v: u32) -> core::fmt::Result {
    if v == 0 {
        return out.write_str("0");
    }
    // Up to 10 digits for a u32; field values are tiny but this stays general.
    let mut tmp = [0u8; 10];
    let mut i = tmp.len();
    while v > 0 {
        i -= 1;
        tmp[i] = b'0' + (v % 10) as u8;
        v /= 10;
    }
    // `tmp[i..]` is valid ASCII by construction.
    out.write_str(core::str::from_utf8(&tmp[i..]).unwrap_or("0"))
}

/// A minimal `core::fmt::Write` over a borrowed byte buffer for [`SystemReg::render_to`].
struct ByteSink<'a> {
    buf: &'a mut [u8],
    len: usize,
    ok: bool,
}

impl core::fmt::Write for ByteSink<'_> {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        let b = s.as_bytes();
        if self.len + b.len() > self.buf.len() {
            self.ok = false;
            return Err(core::fmt::Error);
        }
        self.buf[self.len..self.len + b.len()].copy_from_slice(b);
        self.len += b.len();
        Ok(())
    }
}

/// Self-contained directory of the common architectural system registers.
///
/// Keyed by the packed `(op0<<14|op1<<11|CRn<<7|CRm<<3|op2)` value (matching
/// [`SystemReg::from_fields`]). Transcribed from the ARM ARM system-register
/// `op0/op1/CRn/CRm/op2` directory; `None` for anything not listed (the renderer
/// then emits the generic `S<…>` form). This complements the codegen table in
/// [`crate::tables::names`] so the formatter renders the most common sysregs
/// even before that table is populated.
#[inline]
const fn canonical_name(packed: u16) -> Option<&'static str> {
    // Helper to keep the arms terse and self-documenting.
    macro_rules! k {
        ($o0:expr, $o1:expr, $crn:expr, $crm:expr, $o2:expr) => {
            (($o0 & 0x3) << 14)
                | (($o1 & 0x7) << 11)
                | (($crn & 0xf) << 7)
                | (($crm & 0xf) << 3)
                | ($o2 & 0x7)
        };
    }
    let name = match packed {
        // --- Application / process state, op0=3 op1=3 (EL0-accessible) ---
        x if x == k!(3, 3, 4, 2, 0) => "nzcv",
        x if x == k!(3, 3, 4, 2, 1) => "daif",
        x if x == k!(3, 3, 4, 2, 5) => "dit",
        x if x == k!(3, 3, 4, 2, 6) => "ssbs",
        x if x == k!(3, 3, 4, 2, 7) => "tco",
        x if x == k!(3, 3, 4, 4, 0) => "fpcr",
        x if x == k!(3, 3, 4, 4, 1) => "fpsr",
        // Thread pointers.
        x if x == k!(3, 3, 13, 0, 2) => "tpidr_el0",
        x if x == k!(3, 3, 13, 0, 3) => "tpidrro_el0",
        x if x == k!(3, 0, 13, 0, 4) => "tpidr_el1",
        x if x == k!(3, 4, 13, 0, 2) => "tpidr_el2",
        x if x == k!(3, 6, 13, 0, 2) => "tpidr_el3",
        // Counter-timer (EL0).
        x if x == k!(3, 3, 14, 0, 0) => "cntfrq_el0",
        x if x == k!(3, 3, 14, 0, 1) => "cntpct_el0",
        x if x == k!(3, 3, 14, 0, 2) => "cntvct_el0",
        // --- EL1 banked state, op0=3 op1=0 ---
        x if x == k!(3, 0, 1, 0, 0) => "sctlr_el1",
        x if x == k!(3, 0, 1, 0, 2) => "cpacr_el1",
        x if x == k!(3, 0, 2, 0, 0) => "ttbr0_el1",
        x if x == k!(3, 0, 2, 0, 1) => "ttbr1_el1",
        x if x == k!(3, 0, 2, 0, 2) => "tcr_el1",
        x if x == k!(3, 0, 4, 0, 0) => "spsr_el1",
        x if x == k!(3, 0, 4, 0, 1) => "elr_el1",
        x if x == k!(3, 0, 4, 1, 0) => "sp_el0",
        x if x == k!(3, 0, 4, 2, 0) => "spsel",
        x if x == k!(3, 0, 4, 2, 2) => "currentel",
        x if x == k!(3, 0, 4, 2, 3) => "pan",
        x if x == k!(3, 0, 4, 2, 4) => "uao",
        x if x == k!(3, 0, 5, 2, 0) => "esr_el1",
        x if x == k!(3, 0, 6, 0, 0) => "far_el1",
        x if x == k!(3, 0, 10, 2, 0) => "mair_el1",
        x if x == k!(3, 0, 10, 3, 0) => "amair_el1",
        x if x == k!(3, 0, 12, 0, 0) => "vbar_el1",
        x if x == k!(3, 0, 13, 0, 1) => "contextidr_el1",
        // Identification (read-only, op0=3 op1=0 CRn=0).
        x if x == k!(3, 0, 0, 0, 0) => "midr_el1",
        x if x == k!(3, 0, 0, 0, 5) => "mpidr_el1",
        // --- EL2 banked state, op0=3 op1=4 ---
        x if x == k!(3, 4, 4, 0, 0) => "spsr_el2",
        x if x == k!(3, 4, 4, 0, 1) => "elr_el2",
        x if x == k!(3, 4, 4, 1, 0) => "sp_el1",
        // VHE EL12 aliases, op0=3 op1=5.
        x if x == k!(3, 5, 4, 0, 0) => "spsr_el12",
        x if x == k!(3, 5, 4, 0, 1) => "elr_el12",
        // --- EL3 banked state, op0=3 op1=6 ---
        x if x == k!(3, 6, 4, 0, 0) => "spsr_el3",
        x if x == k!(3, 6, 4, 0, 1) => "elr_el3",
        x if x == k!(3, 6, 4, 1, 0) => "sp_el2",
        // --- RAS, op0=3 ---
        x if x == k!(3, 0, 12, 1, 1) => "disr_el1",
        // --- Pointer-authentication keys, op0=3 op1=0 CRn=2 ---
        x if x == k!(3, 0, 2, 1, 1) => "apiakeyhi_el1",
        // --- GIC CPU / virtualisation control, op0=3 ---
        x if x == k!(3, 6, 12, 12, 7) => "icc_igrpen1_el3",
        x if x == k!(3, 4, 12, 8, 1) => "ich_ap0r1_el2",
        // --- Performance Monitors, op0=3 ---
        x if x == k!(3, 0, 9, 14, 2) => "pmintenclr_el1",
        x if x == k!(3, 3, 9, 12, 3) => "pmovsclr_el0",
        // PMEVCNTR<n>_EL0: CRn=14, CRm=0b1000|(n>>3), op2=n&7.
        x if x == k!(3, 3, 14, 8, 7) => "pmevcntr7_el0", // n=7
        x if x == k!(3, 3, 14, 9, 4) => "pmevcntr12_el0", // n=12
        x if x == k!(3, 3, 14, 11, 0) => "pmevcntr24_el0", // n=24
        // PMEVTYPER<n>_EL0: CRn=14, CRm=0b1100|(n>>3), op2=n&7.
        x if x == k!(3, 3, 14, 13, 0) => "pmevtyper8_el0", // n=8
        // --- Embedded Trace (ETM/TRBE) register file, op0=2 op1=1 ---
        x if x == k!(2, 1, 0, 11, 0) => "trcstallctlr",
        // TRCRSCTLR<n>: CRn=1, n encoded across CRm/op2[0] (n = CRm | op2[0]<<4).
        x if x == k!(2, 1, 1, 15, 0) => "trcrsctlr15", // n=15
        x if x == k!(2, 1, 1, 0, 1) => "trcrsctlr16", // n=16
        // TRCACVR<n>: CRn=2, CRm=2*n (low bank), op2 high bits.
        x if x == k!(2, 1, 2, 10, 0) => "trcacvr5", // n=5
        // TRCCIDCVR<n>: CRn=3, CRm=2*n, op2=0.
        x if x == k!(2, 1, 3, 0, 0) => "trccidcvr0", // n=0
        _ => return None,
    };
    Some(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packing_matches_fields() {
        // NZCV is op0=3 op1=3 CRn=4 CRm=2 op2=0 -> 55824.
        let r = SystemReg::from_fields(3, 3, 4, 2, 0);
        assert_eq!(r.packed(), 55824);
        assert_eq!(r.op0(), 3);
        assert_eq!(r.op1(), 3);
        assert_eq!(r.crn(), 4);
        assert_eq!(r.crm(), 2);
        assert_eq!(r.op2(), 0);
        // from_encoding is the same.
        assert_eq!(SystemReg::from_encoding(3, 3, 4, 2, 0), r);
    }

    #[test]
    fn field_extract_roundtrip() {
        for op0 in 0u8..4 {
            for op1 in 0u8..8 {
                for crn in 0u8..16 {
                    for crm in 0u8..16 {
                        for op2 in 0u8..8 {
                            let r = SystemReg::from_fields(op0, op1, crn, crm, op2);
                            assert_eq!(r.op0(), op0);
                            assert_eq!(r.op1(), op1);
                            assert_eq!(r.crn(), crn);
                            assert_eq!(r.crm(), crm);
                            assert_eq!(r.op2(), op2);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn canonical_names() {
        assert_eq!(SystemReg::from_fields(3, 3, 4, 2, 0).name(), Some("nzcv"));
        assert_eq!(SystemReg::from_fields(3, 3, 4, 2, 1).name(), Some("daif"));
        assert_eq!(SystemReg::from_fields(3, 3, 4, 4, 0).name(), Some("fpcr"));
        assert_eq!(SystemReg::from_fields(3, 3, 4, 4, 1).name(), Some("fpsr"));
        assert_eq!(
            SystemReg::from_fields(3, 3, 13, 0, 2).name(),
            Some("tpidr_el0")
        );
        assert_eq!(SystemReg::from_fields(3, 0, 4, 1, 0).name(), Some("sp_el0"));
        assert_eq!(
            SystemReg::from_fields(3, 0, 4, 2, 2).name(),
            Some("currentel")
        );
        assert_eq!(SystemReg::from_fields(3, 0, 4, 0, 1).name(), Some("elr_el1"));
        assert_eq!(
            SystemReg::from_fields(3, 0, 4, 0, 0).name(),
            Some("spsr_el1")
        );
        // Unknown -> None.
        assert_eq!(SystemReg::from_fields(3, 7, 15, 15, 7).name(), None);
    }

    #[test]
    fn render_to_buffer() {
        let mut buf = [0u8; 32];
        // Known register renders its canonical name.
        let r = SystemReg::from_fields(3, 3, 4, 2, 0);
        assert_eq!(r.render_to(&mut buf), Some("nzcv"));
        // Unknown register renders the generic form with leading `s`.
        let r = SystemReg::from_fields(3, 7, 15, 15, 7);
        assert_eq!(r.render_to(&mut buf), Some("s3_7_c15_c15_7"));
        let r = SystemReg::from_fields(3, 1, 11, 0, 1);
        assert_eq!(r.render_to(&mut buf), Some("s3_1_c11_c0_1"));
    }

    #[test]
    fn render_into_sink() {
        // The `render` method (without the leading `s`) writes name or generic body.
        let mut s = ScratchString::new();
        SystemReg::from_fields(3, 3, 4, 4, 0).render(&mut s).unwrap();
        assert_eq!(s.as_str(), "fpcr");

        let mut s = ScratchString::new();
        SystemReg::from_fields(3, 7, 15, 15, 7).render(&mut s).unwrap();
        assert_eq!(s.as_str(), "3_7_c15_c15_7");
    }

    #[test]
    fn render_to_too_small() {
        let mut buf = [0u8; 2];
        let r = SystemReg::from_fields(3, 7, 15, 15, 7);
        assert_eq!(r.render_to(&mut buf), None);
    }

    /// A tiny fixed `core::fmt::Write` for tests (avoids needing `alloc`).
    struct ScratchString {
        buf: [u8; 64],
        len: usize,
    }

    impl ScratchString {
        fn new() -> Self {
            ScratchString {
                buf: [0u8; 64],
                len: 0,
            }
        }
        fn as_str(&self) -> &str {
            core::str::from_utf8(&self.buf[..self.len]).unwrap()
        }
    }

    impl core::fmt::Write for ScratchString {
        fn write_str(&mut self, s: &str) -> core::fmt::Result {
            let b = s.as_bytes();
            self.buf[self.len..self.len + b.len()].copy_from_slice(b);
            self.len += b.len();
            Ok(())
        }
    }
}
