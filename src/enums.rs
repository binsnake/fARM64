//! Small public enums shared across the API: condition codes, shift/extend
//! modifiers, vector arrangements, and flow/flag classifications.
//!
//! `Code` and `Mnemonic` (the two big `#[repr(u16)]` enums) live in
//! [`crate::mnemonic`]. All name tables are `const`/`static` `&'static str`.

/// AArch64 4-bit condition field (`B.<cond>`, `CSEL`, `CCMP`, ...).
///
/// `name()` emits the ARM *preferred* spelling (`cs`/`cc`); differential
/// comparison additionally accepts the `hs`/`lo` synonyms.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Condition {
    /// Equal (`Z == 1`).
    Eq = 0b0000,
    /// Not equal (`Z == 0`).
    Ne = 0b0001,
    /// Carry set / unsigned higher-or-same (`hs`).
    Cs = 0b0010,
    /// Carry clear / unsigned lower (`lo`).
    Cc = 0b0011,
    /// Negative.
    Mi = 0b0100,
    /// Positive or zero.
    Pl = 0b0101,
    /// Overflow set.
    Vs = 0b0110,
    /// Overflow clear.
    Vc = 0b0111,
    /// Unsigned higher.
    Hi = 0b1000,
    /// Unsigned lower or same.
    Ls = 0b1001,
    /// Signed greater than or equal.
    Ge = 0b1010,
    /// Signed less than.
    Lt = 0b1011,
    /// Signed greater than.
    Gt = 0b1100,
    /// Signed less than or equal.
    Le = 0b1101,
    /// Always.
    Al = 0b1110,
    /// Always (never as a real condition; `B.NV` decodes as `B.AL`).
    Nv = 0b1111,
}

impl Condition {
    /// The condition encoded in a 4-bit field (only the low 4 bits are used).
    ///
    /// This is total: every value `0..=15` maps to a variant, and any high bits
    /// of `bits` are masked off first.
    #[inline]
    pub const fn from_bits(bits: u8) -> Condition {
        match bits & 0b1111 {
            0b0000 => Condition::Eq,
            0b0001 => Condition::Ne,
            0b0010 => Condition::Cs,
            0b0011 => Condition::Cc,
            0b0100 => Condition::Mi,
            0b0101 => Condition::Pl,
            0b0110 => Condition::Vs,
            0b0111 => Condition::Vc,
            0b1000 => Condition::Hi,
            0b1001 => Condition::Ls,
            0b1010 => Condition::Ge,
            0b1011 => Condition::Lt,
            0b1100 => Condition::Gt,
            0b1101 => Condition::Le,
            0b1110 => Condition::Al,
            _ => Condition::Nv,
        }
    }

    /// Decode the 4-bit condition field. Alias of [`Condition::from_bits`] using
    /// the ARM ARM's `cond<3:0>` naming.
    #[inline]
    pub const fn from_u4(bits: u8) -> Condition {
        Condition::from_bits(bits)
    }

    /// The raw 4-bit encoding of this condition (`0b0000` for `EQ` …
    /// `0b1111` for `NV`).
    #[inline]
    pub const fn as_u4(self) -> u8 {
        self as u8
    }

    /// The ARM-preferred lowercase spelling (`"eq"`, `"cs"`, ...), a
    /// `&'static str`.
    #[inline]
    pub const fn name(self) -> &'static str {
        crate::tables::names::condition_name(self)
    }

    /// The inverted condition, with its low bit flipped (`EQ`<->`NE`,
    /// `CS`<->`CC`, `GE`<->`LT`, ...). `AL`<->`NV`.
    ///
    /// This matches the ARM ARM `InvertCond` used by alias selection
    /// (`CSET`/`CSETM`, `CSINC`/`CINC`, ...): the encoded value is XORed with 1.
    #[inline]
    pub const fn invert(self) -> Condition {
        Condition::from_bits((self as u8) ^ 1)
    }

    /// Compatibility alias of [`Condition::invert`].
    #[inline]
    pub const fn inverted(self) -> Condition {
        self.invert()
    }
}

/// True shift modifiers on register / immediate operands.
///
/// Register-*extension* types live in [`ExtendType`] for type clarity (kept
/// separate rather than combined into one enum). `Msl` is the "mask shift left"
/// used by some `MOVI` forms.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShiftType {
    /// No shift.
    None = 0,
    /// Logical shift left (`00`).
    Lsl,
    /// Logical shift right (`01`).
    Lsr,
    /// Arithmetic shift right (`10`).
    Asr,
    /// Rotate right (`11`).
    Ror,
    /// Mask shift left (ones shifted in), used by `MOVI`/`MVNI`.
    Msl,
}

impl ShiftType {
    /// Decode the 2-bit `shift` field of shifted-register forms:
    /// `00 -> Lsl`, `01 -> Lsr`, `10 -> Asr`, `11 -> Ror` (only the low two
    /// bits are used). `Msl`/`None` are never produced by this decode.
    #[inline]
    pub const fn from_bits(bits: u8) -> ShiftType {
        match bits & 0b11 {
            0b00 => ShiftType::Lsl,
            0b01 => ShiftType::Lsr,
            0b10 => ShiftType::Asr,
            _ => ShiftType::Ror,
        }
    }

    /// Lowercase mnemonic (`"lsl"`, `"lsr"`, `"asr"`, `"ror"`, `"msl"`); empty
    /// for [`ShiftType::None`].
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            ShiftType::None => "",
            ShiftType::Lsl => "lsl",
            ShiftType::Lsr => "lsr",
            ShiftType::Asr => "asr",
            ShiftType::Ror => "ror",
            ShiftType::Msl => "msl",
        }
    }
}

/// Register-extension type for extended-register addressing / arithmetic
/// (`UXTB`..`SXTX`).
///
/// Kept separate from [`ShiftType`] so `MemExt` / extended-register operands are
/// type-safe.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExtendType {
    /// Unsigned extend byte (`000`).
    Uxtb = 0b000,
    /// Unsigned extend halfword (`001`).
    Uxth = 0b001,
    /// Unsigned extend word (`010`); also the `LSL`-equivalent for W-base.
    Uxtw = 0b010,
    /// Unsigned extend doubleword (`011`); the `LSL`-equivalent for X-base.
    Uxtx = 0b011,
    /// Signed extend byte (`100`).
    Sxtb = 0b100,
    /// Signed extend halfword (`101`).
    Sxth = 0b101,
    /// Signed extend word (`110`).
    Sxtw = 0b110,
    /// Signed extend doubleword (`111`).
    Sxtx = 0b111,
}

impl ExtendType {
    /// Decode the 3-bit `option` field of extended-register forms
    /// (`000 -> UXTB` … `111 -> SXTX`). Total over `0..=7`; high bits of `bits`
    /// are masked off.
    #[inline]
    pub const fn from_bits(bits: u8) -> ExtendType {
        match bits & 0b111 {
            0b000 => ExtendType::Uxtb,
            0b001 => ExtendType::Uxth,
            0b010 => ExtendType::Uxtw,
            0b011 => ExtendType::Uxtx,
            0b100 => ExtendType::Sxtb,
            0b101 => ExtendType::Sxth,
            0b110 => ExtendType::Sxtw,
            _ => ExtendType::Sxtx,
        }
    }

    /// The raw 3-bit `option` encoding (`0b000` for `UXTB` … `0b111` for
    /// `SXTX`).
    #[inline]
    pub const fn as_bits(self) -> u8 {
        self as u8
    }

    /// Lowercase mnemonic (`"uxtb"`..`"sxtx"`).
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            ExtendType::Uxtb => "uxtb",
            ExtendType::Uxth => "uxth",
            ExtendType::Uxtw => "uxtw",
            ExtendType::Uxtx => "uxtx",
            ExtendType::Sxtb => "sxtb",
            ExtendType::Sxth => "sxth",
            ExtendType::Sxtw => "sxtw",
            ExtendType::Sxtx => "sxtx",
        }
    }
}

/// SIMD / SVE arrangement specifier, carried orthogonally on register operands.
///
/// The variant spine below is representative; the full set (`.8B`/`.16B`/`.4H`/
/// `.8H`/`.2S`/`.4S`/`.1D`/`.2D` plus SVE element widths) is completed by
/// codegen.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorArrangement {
    /// No arrangement.
    None = 0,
    /// `.8B` — eight bytes.
    V8B,
    /// `.16B` — sixteen bytes.
    V16B,
    /// `.4H` — four halfwords.
    V4H,
    /// `.8H` — eight halfwords.
    V8H,
    /// `.2S` — two single words.
    V2S,
    /// `.4S` — four single words.
    V4S,
    /// `.1D` — one doubleword.
    V1D,
    /// `.2D` — two doublewords.
    V2D,
    /// `.2H` — two halfwords (FMLAL/FMLSL widening source).
    V2H,
    /// `.4B` — four bytes (the indexed-element arrangement of the SDOT/UDOT/
    /// USDOT/SUDOT dot-product by-element forms: `Vm.4B[index]`). Unlike the
    /// other indexed arrangements this keeps its `4b` lane-group prefix even
    /// with a lane index, so its truncated suffix is also `.4b`.
    V4B,
    /// `.1Q` — one quadword (PMULL/PMULL2 polynomial-long result).
    V1Q,
    // --- SVE element-width forms (size only; count is VL-dependent) ---
    /// SVE byte elements (`.B`).
    Sb,
    /// SVE halfword elements (`.H`).
    Sh,
    /// SVE word elements (`.S`).
    Ss,
    /// SVE doubleword elements (`.D`).
    Sd,
    /// SVE quadword elements (`.Q`).
    Sq,
    // codegen/expand: any remaining arrangement variants.
}

impl VectorArrangement {
    /// Element width in bits (`8`/`16`/`32`/`64`/`128`), or `0` for
    /// [`VectorArrangement::None`].
    #[inline]
    pub const fn element_bits(self) -> u16 {
        match self {
            VectorArrangement::None => 0,
            VectorArrangement::V8B
            | VectorArrangement::V16B
            | VectorArrangement::V4B
            | VectorArrangement::Sb => 8,
            VectorArrangement::V4H | VectorArrangement::V8H | VectorArrangement::V2H | VectorArrangement::Sh => 16,
            VectorArrangement::V2S | VectorArrangement::V4S | VectorArrangement::Ss => 32,
            VectorArrangement::V1D | VectorArrangement::V2D | VectorArrangement::Sd => 64,
            VectorArrangement::V1Q | VectorArrangement::Sq => 128,
        }
    }

    /// Number of elements for fixed-width SIMD arrangements; `0` for SVE
    /// (VL-dependent) and [`VectorArrangement::None`].
    #[inline]
    pub const fn element_count(self) -> u8 {
        match self {
            VectorArrangement::V8B => 8,
            VectorArrangement::V16B => 16,
            VectorArrangement::V4B => 4,
            VectorArrangement::V4H => 4,
            VectorArrangement::V8H => 8,
            VectorArrangement::V2S => 2,
            VectorArrangement::V4S => 4,
            VectorArrangement::V1D => 1,
            VectorArrangement::V2D => 2,
            VectorArrangement::V2H => 2,
            VectorArrangement::V1Q => 1,
            // SVE element specifiers and `None` carry no fixed lane count.
            VectorArrangement::None
            | VectorArrangement::Sb
            | VectorArrangement::Sh
            | VectorArrangement::Ss
            | VectorArrangement::Sd
            | VectorArrangement::Sq => 0,
        }
    }

    /// `true` for the SVE/SME element-width specifiers (`.b`/`.h`/`.s`/`.d`/`.q`,
    /// which carry an element size but no fixed lane count).
    #[inline]
    pub const fn is_scalable(self) -> bool {
        matches!(
            self,
            VectorArrangement::Sb
                | VectorArrangement::Sh
                | VectorArrangement::Ss
                | VectorArrangement::Sd
                | VectorArrangement::Sq
        )
    }

    /// The arrangement suffix string, including the leading dot.
    ///
    /// `full == true` yields the count+size form used by Advanced SIMD
    /// (`.4s`, `.16b`, ...). `full == false` yields the truncated element-size
    /// form (`.s`, `.b`, ...) used by indexed-element operands and SVE/SME
    /// renderings. SVE specifiers and [`VectorArrangement::None`] are unaffected
    /// by `full`.
    #[inline]
    pub const fn suffix(self, full: bool) -> &'static str {
        // The dot-product indexed arrangement keeps its `.4b` lane-group prefix
        // even with a lane index, so it ignores the truncated (`full == false`)
        // element-size rendering used by every other indexed operand.
        if let VectorArrangement::V4B = self {
            return ".4b";
        }
        if !full {
            return match self.element_bits() {
                8 => ".b",
                16 => ".h",
                32 => ".s",
                64 => ".d",
                128 => ".q",
                _ => "",
            };
        }
        match self {
            VectorArrangement::None => "",
            VectorArrangement::V8B => ".8b",
            VectorArrangement::V16B => ".16b",
            VectorArrangement::V4H => ".4h",
            VectorArrangement::V8H => ".8h",
            VectorArrangement::V2S => ".2s",
            VectorArrangement::V4S => ".4s",
            VectorArrangement::V1D => ".1d",
            VectorArrangement::V2D => ".2d",
            VectorArrangement::V2H => ".2h",
            VectorArrangement::V4B => ".4b",
            VectorArrangement::V1Q => ".1q",
            VectorArrangement::Sb => ".b",
            VectorArrangement::Sh => ".h",
            VectorArrangement::Ss => ".s",
            VectorArrangement::Sd => ".d",
            VectorArrangement::Sq => ".q",
        }
    }
}

/// NZCV write behaviour of an instruction.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FlagEffect {
    /// Does not write the condition flags.
    None = 0,
    /// Writes flags (generic).
    Sets,
    /// Writes flags via the normal integer ALU path.
    SetsNormal,
    /// Writes flags via a floating-point comparison.
    SetsFloat,
}

impl FlagEffect {
    /// `true` if the instruction writes any of the NZCV condition flags.
    #[inline]
    pub const fn writes_flags(self) -> bool {
        !matches!(self, FlagEffect::None)
    }

    /// A short lowercase label (`""`, `"sets"`, `"sets-nzcv"`, `"sets-fp"`),
    /// for diagnostics.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            FlagEffect::None => "",
            FlagEffect::Sets => "sets",
            FlagEffect::SetsNormal => "sets-nzcv",
            FlagEffect::SetsFloat => "sets-fp",
        }
    }
}

impl Default for FlagEffect {
    #[inline]
    fn default() -> Self {
        FlagEffect::None
    }
}

/// Control-flow classification used by the [`crate::info`] facility (iced
/// `FlowControl` analog).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FlowControl {
    /// Falls through to the next instruction.
    Next = 0,
    /// Unconditional direct branch.
    UnconditionalBranch,
    /// Conditional direct branch.
    ConditionalBranch,
    /// Branch through a register / computed target.
    IndirectBranch,
    /// Direct call (writes a link register).
    Call,
    /// Indirect call.
    IndirectCall,
    /// Return from subroutine.
    Return,
    /// Exception generation / system call (`SVC`, `BRK`, ...).
    Exception,
}

impl FlowControl {
    /// A short lowercase label for the flow class, for diagnostics.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            FlowControl::Next => "next",
            FlowControl::UnconditionalBranch => "branch",
            FlowControl::ConditionalBranch => "cond-branch",
            FlowControl::IndirectBranch => "indirect-branch",
            FlowControl::Call => "call",
            FlowControl::IndirectCall => "indirect-call",
            FlowControl::Return => "return",
            FlowControl::Exception => "exception",
        }
    }

    /// `true` if control may leave the linear instruction stream (anything other
    /// than [`FlowControl::Next`]).
    #[inline]
    pub const fn is_control_transfer(self) -> bool {
        !matches!(self, FlowControl::Next)
    }

    /// `true` for any branch or call that writes a link register
    /// ([`FlowControl::Call`] / [`FlowControl::IndirectCall`]).
    #[inline]
    pub const fn is_call(self) -> bool {
        matches!(self, FlowControl::Call | FlowControl::IndirectCall)
    }

    /// `true` for branches whose target is taken from a register
    /// (indirect branch, indirect call, return).
    #[inline]
    pub const fn is_indirect(self) -> bool {
        matches!(
            self,
            FlowControl::IndirectBranch | FlowControl::IndirectCall | FlowControl::Return
        )
    }
}

impl Default for FlowControl {
    #[inline]
    fn default() -> Self {
        FlowControl::Next
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn condition_roundtrip() {
        for raw in 0u8..16 {
            let c = Condition::from_u4(raw);
            assert_eq!(c.as_u4(), raw, "as_u4 must round-trip {raw:#06b}");
            assert_eq!(Condition::from_bits(raw), c);
            // High bits are masked off.
            assert_eq!(Condition::from_u4(raw | 0xF0), c);
        }
    }

    #[test]
    fn condition_invert_pairs() {
        // The low bit is flipped; double invert is identity.
        for raw in 0u8..16 {
            let c = Condition::from_u4(raw);
            assert_eq!(c.invert().as_u4(), raw ^ 1);
            assert_eq!(c.invert().invert(), c);
            assert_eq!(c.inverted(), c.invert());
        }
        assert_eq!(Condition::Eq.invert(), Condition::Ne);
        assert_eq!(Condition::Ne.invert(), Condition::Eq);
        assert_eq!(Condition::Cs.invert(), Condition::Cc);
        assert_eq!(Condition::Ge.invert(), Condition::Lt);
        assert_eq!(Condition::Gt.invert(), Condition::Le);
        assert_eq!(Condition::Al.invert(), Condition::Nv);
        assert_eq!(Condition::Hi.invert(), Condition::Ls);
    }

    #[test]
    fn condition_names() {
        assert_eq!(Condition::Eq.name(), "eq");
        // The carry conditions use the unsigned-comparison synonyms hs/lo (the
        // spelling Binary Ninja emits; same 4-bit encoding as cs/cc).
        assert_eq!(Condition::Cs.name(), "hs");
        assert_eq!(Condition::Cc.name(), "lo");
        assert_eq!(Condition::Nv.name(), "nv");
        assert_eq!(Condition::Al.name(), "al");
    }

    #[test]
    fn shift_decode_and_name() {
        assert_eq!(ShiftType::from_bits(0b00), ShiftType::Lsl);
        assert_eq!(ShiftType::from_bits(0b01), ShiftType::Lsr);
        assert_eq!(ShiftType::from_bits(0b10), ShiftType::Asr);
        assert_eq!(ShiftType::from_bits(0b11), ShiftType::Ror);
        // High bits ignored.
        assert_eq!(ShiftType::from_bits(0b1100), ShiftType::Lsl);

        assert_eq!(ShiftType::None.name(), "");
        assert_eq!(ShiftType::Lsl.name(), "lsl");
        assert_eq!(ShiftType::Lsr.name(), "lsr");
        assert_eq!(ShiftType::Asr.name(), "asr");
        assert_eq!(ShiftType::Ror.name(), "ror");
        assert_eq!(ShiftType::Msl.name(), "msl");
    }

    #[test]
    fn extend_decode_and_name() {
        let expect = [
            (0b000u8, ExtendType::Uxtb, "uxtb"),
            (0b001, ExtendType::Uxth, "uxth"),
            (0b010, ExtendType::Uxtw, "uxtw"),
            (0b011, ExtendType::Uxtx, "uxtx"),
            (0b100, ExtendType::Sxtb, "sxtb"),
            (0b101, ExtendType::Sxth, "sxth"),
            (0b110, ExtendType::Sxtw, "sxtw"),
            (0b111, ExtendType::Sxtx, "sxtx"),
        ];
        for (bits, ty, nm) in expect {
            assert_eq!(ExtendType::from_bits(bits), ty);
            assert_eq!(ty.as_bits(), bits);
            assert_eq!(ty.name(), nm);
            // High bits ignored.
            assert_eq!(ExtendType::from_bits(bits | 0b1000), ty);
        }
    }

    #[test]
    fn arrangement_suffix_full() {
        assert_eq!(VectorArrangement::None.suffix(true), "");
        assert_eq!(VectorArrangement::V8B.suffix(true), ".8b");
        assert_eq!(VectorArrangement::V16B.suffix(true), ".16b");
        assert_eq!(VectorArrangement::V4H.suffix(true), ".4h");
        assert_eq!(VectorArrangement::V8H.suffix(true), ".8h");
        assert_eq!(VectorArrangement::V2S.suffix(true), ".2s");
        assert_eq!(VectorArrangement::V4S.suffix(true), ".4s");
        assert_eq!(VectorArrangement::V1D.suffix(true), ".1d");
        assert_eq!(VectorArrangement::V2D.suffix(true), ".2d");
        // SVE element specifiers.
        assert_eq!(VectorArrangement::Sb.suffix(true), ".b");
        assert_eq!(VectorArrangement::Sh.suffix(true), ".h");
        assert_eq!(VectorArrangement::Ss.suffix(true), ".s");
        assert_eq!(VectorArrangement::Sd.suffix(true), ".d");
        assert_eq!(VectorArrangement::Sq.suffix(true), ".q");
    }

    #[test]
    fn arrangement_suffix_truncated() {
        assert_eq!(VectorArrangement::V16B.suffix(false), ".b");
        assert_eq!(VectorArrangement::V8H.suffix(false), ".h");
        assert_eq!(VectorArrangement::V4S.suffix(false), ".s");
        assert_eq!(VectorArrangement::V2D.suffix(false), ".d");
        assert_eq!(VectorArrangement::Sq.suffix(false), ".q");
        assert_eq!(VectorArrangement::None.suffix(false), "");
    }

    #[test]
    fn arrangement_bits_and_count() {
        assert_eq!(VectorArrangement::V8B.element_bits(), 8);
        assert_eq!(VectorArrangement::V8B.element_count(), 8);
        assert_eq!(VectorArrangement::V16B.element_count(), 16);
        assert_eq!(VectorArrangement::V4S.element_bits(), 32);
        assert_eq!(VectorArrangement::V4S.element_count(), 4);
        assert_eq!(VectorArrangement::V2D.element_bits(), 64);
        assert_eq!(VectorArrangement::Sq.element_bits(), 128);
        // Scalable forms carry size but no fixed lane count.
        assert_eq!(VectorArrangement::Sd.element_count(), 0);
        assert!(VectorArrangement::Sd.is_scalable());
        assert!(!VectorArrangement::V2D.is_scalable());
        assert_eq!(VectorArrangement::None.element_bits(), 0);
        assert_eq!(VectorArrangement::None.element_count(), 0);
    }

    #[test]
    fn flow_and_flag_helpers() {
        assert_eq!(FlowControl::default(), FlowControl::Next);
        assert!(!FlowControl::Next.is_control_transfer());
        assert!(FlowControl::Call.is_call());
        assert!(FlowControl::IndirectCall.is_call());
        assert!(!FlowControl::UnconditionalBranch.is_call());
        assert!(FlowControl::Return.is_indirect());
        assert!(FlowControl::IndirectBranch.is_indirect());
        assert!(!FlowControl::ConditionalBranch.is_indirect());
        assert_eq!(FlowControl::Exception.name(), "exception");

        assert_eq!(FlagEffect::default(), FlagEffect::None);
        assert!(!FlagEffect::None.writes_flags());
        assert!(FlagEffect::SetsNormal.writes_flags());
        assert_eq!(FlagEffect::SetsFloat.name(), "sets-fp");
    }
}
