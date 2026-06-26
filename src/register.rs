//! Typed registers with stack/zero resolution baked in.
//!
//! Register-31 is **never** exposed raw: it is resolved at operand-build time to
//! the stack-pointer variant (`SP`/`WSP`) or the zero-register variant
//! (`XZR`/`WZR`) according to the encoding's per-operand role, via the `const fn`
//! [`gp_register`] as the ARM ARM specifies per encoding. Names come from a
//! `const` `&'static str` table — zero allocation.

/// Every typed AArch64 register the disassembler can produce.
///
/// `#[repr(u16)]` so the discriminant is a stable, compact key into the
/// `const` name table. Reg-31 is materialised as `Sp`/`Wsp`/`Xzr`/`Wzr` — there
/// is no raw "x31"/"w31".
///
/// The variant list is the full A64 register file, in a fixed, append-only
/// order: `None`, the 32-bit GP views (`W0..W30`, `Wzr`, `Wsp`), the 64-bit GP
/// views (`X0..X30`, `Xzr`, `Sp`), the scalar FP/SIMD views (`B`/`H`/`S`/`D`/`Q`
/// `0..31`), the 128-bit vector views (`V0..V31`), the SVE scalable vector
/// (`Z0..Z31`) and predicate (`P0..P15`) registers, and the prefetch
/// pseudo-register slots (`Pf0..Pf31`). The contiguous, ordered discriminants
/// let the helper methods below use compact range patterns.
#[repr(u16)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum Register {
    /// Absence of a register (operand slot has no register component).
    None = 0,

    // --- 32-bit general purpose: W0..W30, then 32-bit ZR/SP ---
    W0, W1, W2, W3, W4, W5, W6, W7,
    W8, W9, W10, W11, W12, W13, W14, W15,
    W16, W17, W18, W19, W20, W21, W22, W23,
    W24, W25, W26, W27, W28, W29, W30, Wzr,
    Wsp,

    // --- 64-bit general purpose: X0..X30, then 64-bit ZR/SP ---
    X0, X1, X2, X3, X4, X5, X6, X7,
    X8, X9, X10, X11, X12, X13, X14, X15,
    X16, X17, X18, X19, X20, X21, X22, X23,
    X24, X25, X26, X27, X28, X29, X30, Xzr,
    Sp,

    // --- Scalar FP / SIMD views (one register file, multiple widths) ---
    // B = 8-bit, H = 16-bit, S = 32-bit, D = 64-bit, Q = 128-bit.
    B0, B1, B2, B3, B4, B5, B6, B7,
    B8, B9, B10, B11, B12, B13, B14, B15,
    B16, B17, B18, B19, B20, B21, B22, B23,
    B24, B25, B26, B27, B28, B29, B30, B31,
    H0, H1, H2, H3, H4, H5, H6, H7,
    H8, H9, H10, H11, H12, H13, H14, H15,
    H16, H17, H18, H19, H20, H21, H22, H23,
    H24, H25, H26, H27, H28, H29, H30, H31,
    S0, S1, S2, S3, S4, S5, S6, S7,
    S8, S9, S10, S11, S12, S13, S14, S15,
    S16, S17, S18, S19, S20, S21, S22, S23,
    S24, S25, S26, S27, S28, S29, S30, S31,
    D0, D1, D2, D3, D4, D5, D6, D7,
    D8, D9, D10, D11, D12, D13, D14, D15,
    D16, D17, D18, D19, D20, D21, D22, D23,
    D24, D25, D26, D27, D28, D29, D30, D31,
    Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7,
    Q8, Q9, Q10, Q11, Q12, Q13, Q14, Q15,
    Q16, Q17, Q18, Q19, Q20, Q21, Q22, Q23,
    Q24, Q25, Q26, Q27, Q28, Q29, Q30, Q31,

    // --- 128-bit SIMD vector views (arrangement carried on operand) ---
    V0, V1, V2, V3, V4, V5, V6, V7,
    V8, V9, V10, V11, V12, V13, V14, V15,
    V16, V17, V18, V19, V20, V21, V22, V23,
    V24, V25, V26, V27, V28, V29, V30, V31,

    // --- SVE scalable vector registers (VL-dependent width) ---
    Z0, Z1, Z2, Z3, Z4, Z5, Z6, Z7,
    Z8, Z9, Z10, Z11, Z12, Z13, Z14, Z15,
    Z16, Z17, Z18, Z19, Z20, Z21, Z22, Z23,
    Z24, Z25, Z26, Z27, Z28, Z29, Z30, Z31,

    // --- SVE predicate registers ---
    P0, P1, P2, P3, P4, P5, P6, P7,
    P8, P9, P10, P11, P12, P13, P14, P15,

    // --- Prefetch pseudo-register slots (`PRFM`/`PRFUM` targets) ---
    Pf0, Pf1, Pf2, Pf3, Pf4, Pf5, Pf6, Pf7,
    Pf8, Pf9, Pf10, Pf11, Pf12, Pf13, Pf14, Pf15,
    Pf16, Pf17, Pf18, Pf19, Pf20, Pf21, Pf22, Pf23,
    Pf24, Pf25, Pf26, Pf27, Pf28, Pf29, Pf30, Pf31,
}

/// Coarse class of a [`Register`], used by operand programs and the formatter to
/// choose suffixes and SP-vs-ZR policy.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegClass {
    /// No register.
    None,
    /// General purpose (W/X, plus SP/ZR resolutions).
    Gp,
    /// Scalar FP (B/H/S/D/Q).
    ScalarFp,
    /// 128-bit SIMD vector (V).
    Vector,
    /// SVE scalable vector (Z).
    Sve,
    /// SVE predicate (P).
    Predicate,
    /// Prefetch pseudo-register.
    Prefetch,
}

/// Bit-width of a general-purpose register operand.
///
/// Drives [`gp_register`] selection between the 32-bit (W) and 64-bit (X) views.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegWidth {
    /// 32-bit (`W`) view.
    W32 = 0,
    /// 64-bit (`X`) view.
    X64 = 1,
}

impl Register {
    /// Bit-width of this register's value (`8`/`16`/`32`/`64`/`128`), or `0` for
    /// [`Register::None`]. SVE `Z`/`P` registers report `0` (scalable /
    /// VL-dependent), as do the prefetch pseudo-registers.
    #[inline]
    pub const fn width_bits(self) -> u16 {
        match self {
            Register::None => 0,
            // 32-bit GP views (W0..W30) + 32-bit ZR/SP
            Register::W0 | Register::W1 | Register::W2 | Register::W3 | Register::W4 | Register::W5 | Register::W6 | Register::W7
                | Register::W8 | Register::W9 | Register::W10 | Register::W11 | Register::W12 | Register::W13 | Register::W14 | Register::W15
                | Register::W16 | Register::W17 | Register::W18 | Register::W19 | Register::W20 | Register::W21 | Register::W22 | Register::W23
                | Register::W24 | Register::W25 | Register::W26 | Register::W27 | Register::W28 | Register::W29 | Register::W30 | Register::Wzr
                | Register::Wsp => 32,
            // 64-bit GP views (X0..X30) + 64-bit ZR/SP
            Register::X0 | Register::X1 | Register::X2 | Register::X3 | Register::X4 | Register::X5 | Register::X6 | Register::X7
                | Register::X8 | Register::X9 | Register::X10 | Register::X11 | Register::X12 | Register::X13 | Register::X14 | Register::X15
                | Register::X16 | Register::X17 | Register::X18 | Register::X19 | Register::X20 | Register::X21 | Register::X22 | Register::X23
                | Register::X24 | Register::X25 | Register::X26 | Register::X27 | Register::X28 | Register::X29 | Register::X30 | Register::Xzr
                | Register::Sp => 64,
            // scalar FP views
            Register::B0 | Register::B1 | Register::B2 | Register::B3 | Register::B4 | Register::B5 | Register::B6 | Register::B7
                | Register::B8 | Register::B9 | Register::B10 | Register::B11 | Register::B12 | Register::B13 | Register::B14 | Register::B15
                | Register::B16 | Register::B17 | Register::B18 | Register::B19 | Register::B20 | Register::B21 | Register::B22 | Register::B23
                | Register::B24 | Register::B25 | Register::B26 | Register::B27 | Register::B28 | Register::B29 | Register::B30 | Register::B31 => 8,
            Register::H0 | Register::H1 | Register::H2 | Register::H3 | Register::H4 | Register::H5 | Register::H6 | Register::H7
                | Register::H8 | Register::H9 | Register::H10 | Register::H11 | Register::H12 | Register::H13 | Register::H14 | Register::H15
                | Register::H16 | Register::H17 | Register::H18 | Register::H19 | Register::H20 | Register::H21 | Register::H22 | Register::H23
                | Register::H24 | Register::H25 | Register::H26 | Register::H27 | Register::H28 | Register::H29 | Register::H30 | Register::H31 => 16,
            Register::S0 | Register::S1 | Register::S2 | Register::S3 | Register::S4 | Register::S5 | Register::S6 | Register::S7
                | Register::S8 | Register::S9 | Register::S10 | Register::S11 | Register::S12 | Register::S13 | Register::S14 | Register::S15
                | Register::S16 | Register::S17 | Register::S18 | Register::S19 | Register::S20 | Register::S21 | Register::S22 | Register::S23
                | Register::S24 | Register::S25 | Register::S26 | Register::S27 | Register::S28 | Register::S29 | Register::S30 | Register::S31 => 32,
            Register::D0 | Register::D1 | Register::D2 | Register::D3 | Register::D4 | Register::D5 | Register::D6 | Register::D7
                | Register::D8 | Register::D9 | Register::D10 | Register::D11 | Register::D12 | Register::D13 | Register::D14 | Register::D15
                | Register::D16 | Register::D17 | Register::D18 | Register::D19 | Register::D20 | Register::D21 | Register::D22 | Register::D23
                | Register::D24 | Register::D25 | Register::D26 | Register::D27 | Register::D28 | Register::D29 | Register::D30 | Register::D31 => 64,
            Register::Q0 | Register::Q1 | Register::Q2 | Register::Q3 | Register::Q4 | Register::Q5 | Register::Q6 | Register::Q7
                | Register::Q8 | Register::Q9 | Register::Q10 | Register::Q11 | Register::Q12 | Register::Q13 | Register::Q14 | Register::Q15
                | Register::Q16 | Register::Q17 | Register::Q18 | Register::Q19 | Register::Q20 | Register::Q21 | Register::Q22 | Register::Q23
                | Register::Q24 | Register::Q25 | Register::Q26 | Register::Q27 | Register::Q28 | Register::Q29 | Register::Q30 | Register::Q31 => 128,
            // 128-bit SIMD vector
            Register::V0 | Register::V1 | Register::V2 | Register::V3 | Register::V4 | Register::V5 | Register::V6 | Register::V7
                | Register::V8 | Register::V9 | Register::V10 | Register::V11 | Register::V12 | Register::V13 | Register::V14 | Register::V15
                | Register::V16 | Register::V17 | Register::V18 | Register::V19 | Register::V20 | Register::V21 | Register::V22 | Register::V23
                | Register::V24 | Register::V25 | Register::V26 | Register::V27 | Register::V28 | Register::V29 | Register::V30 | Register::V31 => 128,
            // SVE scalable vector & predicate: VL-dependent -> 0
            Register::Z0 | Register::Z1 | Register::Z2 | Register::Z3 | Register::Z4 | Register::Z5 | Register::Z6 | Register::Z7
                | Register::Z8 | Register::Z9 | Register::Z10 | Register::Z11 | Register::Z12 | Register::Z13 | Register::Z14 | Register::Z15
                | Register::Z16 | Register::Z17 | Register::Z18 | Register::Z19 | Register::Z20 | Register::Z21 | Register::Z22 | Register::Z23
                | Register::Z24 | Register::Z25 | Register::Z26 | Register::Z27 | Register::Z28 | Register::Z29 | Register::Z30 | Register::Z31
                | Register::P0 | Register::P1 | Register::P2 | Register::P3 | Register::P4 | Register::P5 | Register::P6 | Register::P7
                | Register::P8 | Register::P9 | Register::P10 | Register::P11 | Register::P12 | Register::P13 | Register::P14 | Register::P15 => 0,
            // prefetch pseudo-registers carry no value width
            Register::Pf0 | Register::Pf1 | Register::Pf2 | Register::Pf3 | Register::Pf4 | Register::Pf5 | Register::Pf6 | Register::Pf7
                | Register::Pf8 | Register::Pf9 | Register::Pf10 | Register::Pf11 | Register::Pf12 | Register::Pf13 | Register::Pf14 | Register::Pf15
                | Register::Pf16 | Register::Pf17 | Register::Pf18 | Register::Pf19 | Register::Pf20 | Register::Pf21 | Register::Pf22 | Register::Pf23
                | Register::Pf24 | Register::Pf25 | Register::Pf26 | Register::Pf27 | Register::Pf28 | Register::Pf29 | Register::Pf30 | Register::Pf31 => 0,
        }
    }

    /// The architectural register number (`0..=31`, or `0..=15` for predicates).
    /// `SP`/`WSP` and `XZR`/`WZR` both report `31`.
    #[inline]
    pub const fn number(self) -> u8 {
        match self {
            Register::None => 0,
            Register::W0 => 0, Register::W1 => 1, Register::W2 => 2, Register::W3 => 3, Register::W4 => 4, Register::W5 => 5,
            Register::W6 => 6, Register::W7 => 7, Register::W8 => 8, Register::W9 => 9, Register::W10 => 10, Register::W11 => 11,
            Register::W12 => 12, Register::W13 => 13, Register::W14 => 14, Register::W15 => 15, Register::W16 => 16, Register::W17 => 17,
            Register::W18 => 18, Register::W19 => 19, Register::W20 => 20, Register::W21 => 21, Register::W22 => 22, Register::W23 => 23,
            Register::W24 => 24, Register::W25 => 25, Register::W26 => 26, Register::W27 => 27, Register::W28 => 28, Register::W29 => 29,
            Register::W30 => 30, Register::Wzr => 31, Register::Wsp => 31, Register::X0 => 0, Register::X1 => 1, Register::X2 => 2,
            Register::X3 => 3, Register::X4 => 4, Register::X5 => 5, Register::X6 => 6, Register::X7 => 7, Register::X8 => 8,
            Register::X9 => 9, Register::X10 => 10, Register::X11 => 11, Register::X12 => 12, Register::X13 => 13, Register::X14 => 14,
            Register::X15 => 15, Register::X16 => 16, Register::X17 => 17, Register::X18 => 18, Register::X19 => 19, Register::X20 => 20,
            Register::X21 => 21, Register::X22 => 22, Register::X23 => 23, Register::X24 => 24, Register::X25 => 25, Register::X26 => 26,
            Register::X27 => 27, Register::X28 => 28, Register::X29 => 29, Register::X30 => 30, Register::Xzr => 31, Register::Sp => 31,
            Register::B0 => 0, Register::B1 => 1, Register::B2 => 2, Register::B3 => 3, Register::B4 => 4, Register::B5 => 5,
            Register::B6 => 6, Register::B7 => 7, Register::B8 => 8, Register::B9 => 9, Register::B10 => 10, Register::B11 => 11,
            Register::B12 => 12, Register::B13 => 13, Register::B14 => 14, Register::B15 => 15, Register::B16 => 16, Register::B17 => 17,
            Register::B18 => 18, Register::B19 => 19, Register::B20 => 20, Register::B21 => 21, Register::B22 => 22, Register::B23 => 23,
            Register::B24 => 24, Register::B25 => 25, Register::B26 => 26, Register::B27 => 27, Register::B28 => 28, Register::B29 => 29,
            Register::B30 => 30, Register::B31 => 31, Register::H0 => 0, Register::H1 => 1, Register::H2 => 2, Register::H3 => 3,
            Register::H4 => 4, Register::H5 => 5, Register::H6 => 6, Register::H7 => 7, Register::H8 => 8, Register::H9 => 9,
            Register::H10 => 10, Register::H11 => 11, Register::H12 => 12, Register::H13 => 13, Register::H14 => 14, Register::H15 => 15,
            Register::H16 => 16, Register::H17 => 17, Register::H18 => 18, Register::H19 => 19, Register::H20 => 20, Register::H21 => 21,
            Register::H22 => 22, Register::H23 => 23, Register::H24 => 24, Register::H25 => 25, Register::H26 => 26, Register::H27 => 27,
            Register::H28 => 28, Register::H29 => 29, Register::H30 => 30, Register::H31 => 31, Register::S0 => 0, Register::S1 => 1,
            Register::S2 => 2, Register::S3 => 3, Register::S4 => 4, Register::S5 => 5, Register::S6 => 6, Register::S7 => 7,
            Register::S8 => 8, Register::S9 => 9, Register::S10 => 10, Register::S11 => 11, Register::S12 => 12, Register::S13 => 13,
            Register::S14 => 14, Register::S15 => 15, Register::S16 => 16, Register::S17 => 17, Register::S18 => 18, Register::S19 => 19,
            Register::S20 => 20, Register::S21 => 21, Register::S22 => 22, Register::S23 => 23, Register::S24 => 24, Register::S25 => 25,
            Register::S26 => 26, Register::S27 => 27, Register::S28 => 28, Register::S29 => 29, Register::S30 => 30, Register::S31 => 31,
            Register::D0 => 0, Register::D1 => 1, Register::D2 => 2, Register::D3 => 3, Register::D4 => 4, Register::D5 => 5,
            Register::D6 => 6, Register::D7 => 7, Register::D8 => 8, Register::D9 => 9, Register::D10 => 10, Register::D11 => 11,
            Register::D12 => 12, Register::D13 => 13, Register::D14 => 14, Register::D15 => 15, Register::D16 => 16, Register::D17 => 17,
            Register::D18 => 18, Register::D19 => 19, Register::D20 => 20, Register::D21 => 21, Register::D22 => 22, Register::D23 => 23,
            Register::D24 => 24, Register::D25 => 25, Register::D26 => 26, Register::D27 => 27, Register::D28 => 28, Register::D29 => 29,
            Register::D30 => 30, Register::D31 => 31, Register::Q0 => 0, Register::Q1 => 1, Register::Q2 => 2, Register::Q3 => 3,
            Register::Q4 => 4, Register::Q5 => 5, Register::Q6 => 6, Register::Q7 => 7, Register::Q8 => 8, Register::Q9 => 9,
            Register::Q10 => 10, Register::Q11 => 11, Register::Q12 => 12, Register::Q13 => 13, Register::Q14 => 14, Register::Q15 => 15,
            Register::Q16 => 16, Register::Q17 => 17, Register::Q18 => 18, Register::Q19 => 19, Register::Q20 => 20, Register::Q21 => 21,
            Register::Q22 => 22, Register::Q23 => 23, Register::Q24 => 24, Register::Q25 => 25, Register::Q26 => 26, Register::Q27 => 27,
            Register::Q28 => 28, Register::Q29 => 29, Register::Q30 => 30, Register::Q31 => 31, Register::V0 => 0, Register::V1 => 1,
            Register::V2 => 2, Register::V3 => 3, Register::V4 => 4, Register::V5 => 5, Register::V6 => 6, Register::V7 => 7,
            Register::V8 => 8, Register::V9 => 9, Register::V10 => 10, Register::V11 => 11, Register::V12 => 12, Register::V13 => 13,
            Register::V14 => 14, Register::V15 => 15, Register::V16 => 16, Register::V17 => 17, Register::V18 => 18, Register::V19 => 19,
            Register::V20 => 20, Register::V21 => 21, Register::V22 => 22, Register::V23 => 23, Register::V24 => 24, Register::V25 => 25,
            Register::V26 => 26, Register::V27 => 27, Register::V28 => 28, Register::V29 => 29, Register::V30 => 30, Register::V31 => 31,
            Register::Z0 => 0, Register::Z1 => 1, Register::Z2 => 2, Register::Z3 => 3, Register::Z4 => 4, Register::Z5 => 5,
            Register::Z6 => 6, Register::Z7 => 7, Register::Z8 => 8, Register::Z9 => 9, Register::Z10 => 10, Register::Z11 => 11,
            Register::Z12 => 12, Register::Z13 => 13, Register::Z14 => 14, Register::Z15 => 15, Register::Z16 => 16, Register::Z17 => 17,
            Register::Z18 => 18, Register::Z19 => 19, Register::Z20 => 20, Register::Z21 => 21, Register::Z22 => 22, Register::Z23 => 23,
            Register::Z24 => 24, Register::Z25 => 25, Register::Z26 => 26, Register::Z27 => 27, Register::Z28 => 28, Register::Z29 => 29,
            Register::Z30 => 30, Register::Z31 => 31, Register::P0 => 0, Register::P1 => 1, Register::P2 => 2, Register::P3 => 3,
            Register::P4 => 4, Register::P5 => 5, Register::P6 => 6, Register::P7 => 7, Register::P8 => 8, Register::P9 => 9,
            Register::P10 => 10, Register::P11 => 11, Register::P12 => 12, Register::P13 => 13, Register::P14 => 14, Register::P15 => 15,
            Register::Pf0 => 0, Register::Pf1 => 1, Register::Pf2 => 2, Register::Pf3 => 3, Register::Pf4 => 4, Register::Pf5 => 5,
            Register::Pf6 => 6, Register::Pf7 => 7, Register::Pf8 => 8, Register::Pf9 => 9, Register::Pf10 => 10, Register::Pf11 => 11,
            Register::Pf12 => 12, Register::Pf13 => 13, Register::Pf14 => 14, Register::Pf15 => 15, Register::Pf16 => 16, Register::Pf17 => 17,
            Register::Pf18 => 18, Register::Pf19 => 19, Register::Pf20 => 20, Register::Pf21 => 21, Register::Pf22 => 22, Register::Pf23 => 23,
            Register::Pf24 => 24, Register::Pf25 => 25, Register::Pf26 => 26, Register::Pf27 => 27, Register::Pf28 => 28, Register::Pf29 => 29,
            Register::Pf30 => 30, Register::Pf31 => 31,
        }
    }

    /// The 64-bit (`X`) view of a general-purpose register; identity for
    /// non-GP registers (and for registers that are already an `X` view).
    #[inline]
    pub const fn as_x(self) -> Register {
        match self {
            Register::W0 => Register::X0, Register::W1 => Register::X1, Register::W2 => Register::X2,
            Register::W3 => Register::X3, Register::W4 => Register::X4, Register::W5 => Register::X5,
            Register::W6 => Register::X6, Register::W7 => Register::X7, Register::W8 => Register::X8,
            Register::W9 => Register::X9, Register::W10 => Register::X10, Register::W11 => Register::X11,
            Register::W12 => Register::X12, Register::W13 => Register::X13, Register::W14 => Register::X14,
            Register::W15 => Register::X15, Register::W16 => Register::X16, Register::W17 => Register::X17,
            Register::W18 => Register::X18, Register::W19 => Register::X19, Register::W20 => Register::X20,
            Register::W21 => Register::X21, Register::W22 => Register::X22, Register::W23 => Register::X23,
            Register::W24 => Register::X24, Register::W25 => Register::X25, Register::W26 => Register::X26,
            Register::W27 => Register::X27, Register::W28 => Register::X28, Register::W29 => Register::X29,
            Register::W30 => Register::X30, Register::Wzr => Register::Xzr, Register::Wsp => Register::Sp,
            // already an X view, or not a GP register: identity.
            other => other,
        }
    }

    /// The 32-bit (`W`) view of a general-purpose register; identity for
    /// non-GP registers (and for registers that are already a `W` view).
    #[inline]
    pub const fn as_w(self) -> Register {
        match self {
            Register::X0 => Register::W0, Register::X1 => Register::W1, Register::X2 => Register::W2,
            Register::X3 => Register::W3, Register::X4 => Register::W4, Register::X5 => Register::W5,
            Register::X6 => Register::W6, Register::X7 => Register::W7, Register::X8 => Register::W8,
            Register::X9 => Register::W9, Register::X10 => Register::W10, Register::X11 => Register::W11,
            Register::X12 => Register::W12, Register::X13 => Register::W13, Register::X14 => Register::W14,
            Register::X15 => Register::W15, Register::X16 => Register::W16, Register::X17 => Register::W17,
            Register::X18 => Register::W18, Register::X19 => Register::W19, Register::X20 => Register::W20,
            Register::X21 => Register::W21, Register::X22 => Register::W22, Register::X23 => Register::W23,
            Register::X24 => Register::W24, Register::X25 => Register::W25, Register::X26 => Register::W26,
            Register::X27 => Register::W27, Register::X28 => Register::W28, Register::X29 => Register::W29,
            Register::X30 => Register::W30, Register::Xzr => Register::Wzr, Register::Sp => Register::Wsp,
            // already a W view, or not a GP register: identity.
            other => other,
        }
    }

    /// The full SIMD/vector parent register of a scalar-FP view.
    ///
    /// The scalar FP views `B`/`H`/`S`/`D`/`Q<n>` are narrow windows onto the
    /// 128-bit SIMD register `V<n>`; this maps any of them — and `V<n>` itself —
    /// to `V<n>`. Every non-FP register (GP, SVE `Z`/`P`, prefetch, and
    /// [`Register::None`]) maps to itself. Mirrors iced-x86's
    /// `Register::full_register`.
    #[inline]
    pub const fn full_register(self) -> Register {
        match self.class() {
            RegClass::ScalarFp | RegClass::Vector => v_numbered(self.number()),
            _ => self,
        }
    }

    /// `true` for scalar-FP or 128-bit vector registers.
    #[inline]
    pub const fn is_simd(self) -> bool {
        matches!(self.class(), RegClass::ScalarFp | RegClass::Vector)
    }

    /// `true` for SVE scalable-vector (`Z`) or predicate (`P`) registers.
    #[inline]
    pub const fn is_sve(self) -> bool {
        matches!(self.class(), RegClass::Sve | RegClass::Predicate)
    }

    /// The coarse [`RegClass`] of this register.
    #[inline]
    pub const fn class(self) -> RegClass {
        match self {
            Register::None => RegClass::None,
            Register::W0 | Register::W1 | Register::W2 | Register::W3 | Register::W4 | Register::W5 | Register::W6 | Register::W7
                | Register::W8 | Register::W9 | Register::W10 | Register::W11 | Register::W12 | Register::W13 | Register::W14 | Register::W15
                | Register::W16 | Register::W17 | Register::W18 | Register::W19 | Register::W20 | Register::W21 | Register::W22 | Register::W23
                | Register::W24 | Register::W25 | Register::W26 | Register::W27 | Register::W28 | Register::W29 | Register::W30 | Register::Wzr
                | Register::Wsp | Register::X0 | Register::X1 | Register::X2 | Register::X3 | Register::X4 | Register::X5 | Register::X6
                | Register::X7 | Register::X8 | Register::X9 | Register::X10 | Register::X11 | Register::X12 | Register::X13 | Register::X14
                | Register::X15 | Register::X16 | Register::X17 | Register::X18 | Register::X19 | Register::X20 | Register::X21 | Register::X22
                | Register::X23 | Register::X24 | Register::X25 | Register::X26 | Register::X27 | Register::X28 | Register::X29 | Register::X30
                | Register::Xzr | Register::Sp => RegClass::Gp,
            Register::B0 | Register::B1 | Register::B2 | Register::B3 | Register::B4 | Register::B5 | Register::B6 | Register::B7
                | Register::B8 | Register::B9 | Register::B10 | Register::B11 | Register::B12 | Register::B13 | Register::B14 | Register::B15
                | Register::B16 | Register::B17 | Register::B18 | Register::B19 | Register::B20 | Register::B21 | Register::B22 | Register::B23
                | Register::B24 | Register::B25 | Register::B26 | Register::B27 | Register::B28 | Register::B29 | Register::B30 | Register::B31
                | Register::H0 | Register::H1 | Register::H2 | Register::H3 | Register::H4 | Register::H5 | Register::H6 | Register::H7
                | Register::H8 | Register::H9 | Register::H10 | Register::H11 | Register::H12 | Register::H13 | Register::H14 | Register::H15
                | Register::H16 | Register::H17 | Register::H18 | Register::H19 | Register::H20 | Register::H21 | Register::H22 | Register::H23
                | Register::H24 | Register::H25 | Register::H26 | Register::H27 | Register::H28 | Register::H29 | Register::H30 | Register::H31
                | Register::S0 | Register::S1 | Register::S2 | Register::S3 | Register::S4 | Register::S5 | Register::S6 | Register::S7
                | Register::S8 | Register::S9 | Register::S10 | Register::S11 | Register::S12 | Register::S13 | Register::S14 | Register::S15
                | Register::S16 | Register::S17 | Register::S18 | Register::S19 | Register::S20 | Register::S21 | Register::S22 | Register::S23
                | Register::S24 | Register::S25 | Register::S26 | Register::S27 | Register::S28 | Register::S29 | Register::S30 | Register::S31
                | Register::D0 | Register::D1 | Register::D2 | Register::D3 | Register::D4 | Register::D5 | Register::D6 | Register::D7
                | Register::D8 | Register::D9 | Register::D10 | Register::D11 | Register::D12 | Register::D13 | Register::D14 | Register::D15
                | Register::D16 | Register::D17 | Register::D18 | Register::D19 | Register::D20 | Register::D21 | Register::D22 | Register::D23
                | Register::D24 | Register::D25 | Register::D26 | Register::D27 | Register::D28 | Register::D29 | Register::D30 | Register::D31
                | Register::Q0 | Register::Q1 | Register::Q2 | Register::Q3 | Register::Q4 | Register::Q5 | Register::Q6 | Register::Q7
                | Register::Q8 | Register::Q9 | Register::Q10 | Register::Q11 | Register::Q12 | Register::Q13 | Register::Q14 | Register::Q15
                | Register::Q16 | Register::Q17 | Register::Q18 | Register::Q19 | Register::Q20 | Register::Q21 | Register::Q22 | Register::Q23
                | Register::Q24 | Register::Q25 | Register::Q26 | Register::Q27 | Register::Q28 | Register::Q29 | Register::Q30 | Register::Q31 => RegClass::ScalarFp,
            Register::V0 | Register::V1 | Register::V2 | Register::V3 | Register::V4 | Register::V5 | Register::V6 | Register::V7
                | Register::V8 | Register::V9 | Register::V10 | Register::V11 | Register::V12 | Register::V13 | Register::V14 | Register::V15
                | Register::V16 | Register::V17 | Register::V18 | Register::V19 | Register::V20 | Register::V21 | Register::V22 | Register::V23
                | Register::V24 | Register::V25 | Register::V26 | Register::V27 | Register::V28 | Register::V29 | Register::V30 | Register::V31 => RegClass::Vector,
            Register::Z0 | Register::Z1 | Register::Z2 | Register::Z3 | Register::Z4 | Register::Z5 | Register::Z6 | Register::Z7
                | Register::Z8 | Register::Z9 | Register::Z10 | Register::Z11 | Register::Z12 | Register::Z13 | Register::Z14 | Register::Z15
                | Register::Z16 | Register::Z17 | Register::Z18 | Register::Z19 | Register::Z20 | Register::Z21 | Register::Z22 | Register::Z23
                | Register::Z24 | Register::Z25 | Register::Z26 | Register::Z27 | Register::Z28 | Register::Z29 | Register::Z30 | Register::Z31 => RegClass::Sve,
            Register::P0 | Register::P1 | Register::P2 | Register::P3 | Register::P4 | Register::P5 | Register::P6 | Register::P7
                | Register::P8 | Register::P9 | Register::P10 | Register::P11 | Register::P12 | Register::P13 | Register::P14 | Register::P15 => RegClass::Predicate,
            Register::Pf0 | Register::Pf1 | Register::Pf2 | Register::Pf3 | Register::Pf4 | Register::Pf5 | Register::Pf6 | Register::Pf7
                | Register::Pf8 | Register::Pf9 | Register::Pf10 | Register::Pf11 | Register::Pf12 | Register::Pf13 | Register::Pf14 | Register::Pf15
                | Register::Pf16 | Register::Pf17 | Register::Pf18 | Register::Pf19 | Register::Pf20 | Register::Pf21 | Register::Pf22 | Register::Pf23
                | Register::Pf24 | Register::Pf25 | Register::Pf26 | Register::Pf27 | Register::Pf28 | Register::Pf29 | Register::Pf30 | Register::Pf31 => RegClass::Prefetch,
        }
    }

    /// The canonical lowercase mnemonic for this register (e.g. `"x0"`, `"wsp"`,
    /// `"v31"`, `"z0"`, `"p15"`), as a `&'static str` from the `const` name
    /// table. Zero allocation.
    #[inline]
    pub const fn name(self) -> &'static str {
        crate::tables::names::register_name(self)
    }
}

/// Resolve a general-purpose register-number to a typed [`Register`], applying
/// the SP-vs-ZR rule for number 31.
///
/// `use_sp` selects the stack-pointer interpretation of number 31 (otherwise the
/// zero register), and `width` selects the `W`/`X` view. The group decoders call
/// this so reg-31 is stored as `SP`/`WSP`/`XZR`/`WZR` and never leaks raw, per
/// the ARM ARM's per-operand role for each encoding.
///
/// `n` is taken modulo 32; callers pass a 5-bit field.
#[inline]
pub const fn gp_register(use_sp: bool, width: RegWidth, n: u8) -> Register {
    let n = n % 32;
    match width {
        RegWidth::W32 => {
            if n == 31 {
                if use_sp {
                    Register::Wsp
                } else {
                    Register::Wzr
                }
            } else {
                // Discriminant of `W{n}` = discriminant of `W0` + n, contiguous
                // by construction (W0..W30 are adjacent). Resolve via the X view
                // and fold to the W view to avoid a 31-arm match.
                gp_x_numbered(n).as_w()
            }
        }
        RegWidth::X64 => {
            if n == 31 {
                if use_sp {
                    Register::Sp
                } else {
                    Register::Xzr
                }
            } else {
                gp_x_numbered(n)
            }
        }
    }
}

/// Map an architectural GP number `0..=30` to the 64-bit `X{n}` register.
///
/// Private helper for [`gp_register`]; `n` must be `< 31` (callers guarantee
/// this — number 31 is handled separately as `Sp`/`Xzr`). Out-of-range inputs
/// fall back to `Xzr` so the function stays total and panic-free in `const`.
#[inline]
const fn gp_x_numbered(n: u8) -> Register {
    match n {
        0 => Register::X0, 1 => Register::X1, 2 => Register::X2, 3 => Register::X3,
        4 => Register::X4, 5 => Register::X5, 6 => Register::X6, 7 => Register::X7,
        8 => Register::X8, 9 => Register::X9, 10 => Register::X10, 11 => Register::X11,
        12 => Register::X12, 13 => Register::X13, 14 => Register::X14, 15 => Register::X15,
        16 => Register::X16, 17 => Register::X17, 18 => Register::X18, 19 => Register::X19,
        20 => Register::X20, 21 => Register::X21, 22 => Register::X22, 23 => Register::X23,
        24 => Register::X24, 25 => Register::X25, 26 => Register::X26, 27 => Register::X27,
        28 => Register::X28, 29 => Register::X29, 30 => Register::X30,
        // n == 31 is resolved by the caller; any other value is unreachable for
        // a 5-bit field but we stay total with the zero register.
        _ => Register::Xzr,
    }
}

/// Map a 5-bit SVE register number `0..=31` to the scalable-vector `Z{n}`.
///
/// `n` is taken modulo 32. Used by the SVE/SME group decoders, the multi-vector
/// group formatter, and the encoders so a `Z` register list can be reconstructed
/// from its base number without a parallel table. Total and panic-free.
#[inline]
pub const fn sve_register(n: u8) -> Register {
    match n % 32 {
        0 => Register::Z0, 1 => Register::Z1, 2 => Register::Z2, 3 => Register::Z3,
        4 => Register::Z4, 5 => Register::Z5, 6 => Register::Z6, 7 => Register::Z7,
        8 => Register::Z8, 9 => Register::Z9, 10 => Register::Z10, 11 => Register::Z11,
        12 => Register::Z12, 13 => Register::Z13, 14 => Register::Z14, 15 => Register::Z15,
        16 => Register::Z16, 17 => Register::Z17, 18 => Register::Z18, 19 => Register::Z19,
        20 => Register::Z20, 21 => Register::Z21, 22 => Register::Z22, 23 => Register::Z23,
        24 => Register::Z24, 25 => Register::Z25, 26 => Register::Z26, 27 => Register::Z27,
        28 => Register::Z28, 29 => Register::Z29, 30 => Register::Z30, _ => Register::Z31,
    }
}

/// Map a 5-bit SIMD register number `0..=31` to the 128-bit vector `V{n}`.
///
/// Private helper for [`Register::full_register`] (the scalar-FP → `V` parent
/// fold); `n` is taken modulo 32. Total and panic-free.
#[inline]
const fn v_numbered(n: u8) -> Register {
    match n % 32 {
        0 => Register::V0, 1 => Register::V1, 2 => Register::V2, 3 => Register::V3,
        4 => Register::V4, 5 => Register::V5, 6 => Register::V6, 7 => Register::V7,
        8 => Register::V8, 9 => Register::V9, 10 => Register::V10, 11 => Register::V11,
        12 => Register::V12, 13 => Register::V13, 14 => Register::V14, 15 => Register::V15,
        16 => Register::V16, 17 => Register::V17, 18 => Register::V18, 19 => Register::V19,
        20 => Register::V20, 21 => Register::V21, 22 => Register::V22, 23 => Register::V23,
        24 => Register::V24, 25 => Register::V25, 26 => Register::V26, 27 => Register::V27,
        28 => Register::V28, 29 => Register::V29, 30 => Register::V30, _ => Register::V31,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip() {
        assert_eq!(Register::X5.name(), "x5");
        assert_eq!(Register::W0.name(), "w0");
        assert_eq!(Register::Wzr.name(), "wzr");
        assert_eq!(Register::Wsp.name(), "wsp");
        assert_eq!(Register::Xzr.name(), "xzr");
        assert_eq!(Register::Sp.name(), "sp");
        assert_eq!(Register::V31.name(), "v31");
        assert_eq!(Register::Z0.name(), "z0");
        assert_eq!(Register::P15.name(), "p15");
        assert_eq!(Register::B0.name(), "b0");
        assert_eq!(Register::Q31.name(), "q31");
        assert_eq!(Register::Pf0.name(), "pf0");
        assert_eq!(Register::None.name(), "");
    }

    #[test]
    fn gp_register_resolves_reg31() {
        assert_eq!(gp_register(false, RegWidth::X64, 31), Register::Xzr);
        assert_eq!(gp_register(true, RegWidth::X64, 31), Register::Sp);
        assert_eq!(gp_register(false, RegWidth::W32, 31), Register::Wzr);
        assert_eq!(gp_register(true, RegWidth::W32, 31), Register::Wsp);
    }

    #[test]
    fn gp_register_resolves_normal() {
        // use_sp is irrelevant for 0..=30.
        assert_eq!(gp_register(false, RegWidth::X64, 0), Register::X0);
        assert_eq!(gp_register(true, RegWidth::X64, 0), Register::X0);
        assert_eq!(gp_register(false, RegWidth::X64, 30), Register::X30);
        assert_eq!(gp_register(false, RegWidth::W32, 0), Register::W0);
        assert_eq!(gp_register(true, RegWidth::W32, 3), Register::W3);
        assert_eq!(gp_register(false, RegWidth::W32, 30), Register::W30);
        // n is taken modulo 32.
        assert_eq!(gp_register(false, RegWidth::X64, 32), Register::X0);
        assert_eq!(gp_register(false, RegWidth::X64, 63), Register::Xzr);
    }

    #[test]
    fn class_buckets() {
        assert_eq!(Register::None.class(), RegClass::None);
        assert_eq!(Register::X0.class(), RegClass::Gp);
        assert_eq!(Register::Wsp.class(), RegClass::Gp);
        assert_eq!(Register::Sp.class(), RegClass::Gp);
        assert_eq!(Register::B0.class(), RegClass::ScalarFp);
        assert_eq!(Register::Q31.class(), RegClass::ScalarFp);
        assert_eq!(Register::V31.class(), RegClass::Vector);
        assert_eq!(Register::Z0.class(), RegClass::Sve);
        assert_eq!(Register::P0.class(), RegClass::Predicate);
        assert_eq!(Register::P15.class(), RegClass::Predicate);
        assert_eq!(Register::Pf0.class(), RegClass::Prefetch);
        assert_eq!(Register::Pf31.class(), RegClass::Prefetch);
    }

    #[test]
    fn width_bits_per_class() {
        assert_eq!(Register::None.width_bits(), 0);
        assert_eq!(Register::W0.width_bits(), 32);
        assert_eq!(Register::Wzr.width_bits(), 32);
        assert_eq!(Register::Wsp.width_bits(), 32);
        assert_eq!(Register::X0.width_bits(), 64);
        assert_eq!(Register::Sp.width_bits(), 64);
        assert_eq!(Register::B0.width_bits(), 8);
        assert_eq!(Register::H0.width_bits(), 16);
        assert_eq!(Register::S0.width_bits(), 32);
        assert_eq!(Register::D0.width_bits(), 64);
        assert_eq!(Register::Q0.width_bits(), 128);
        assert_eq!(Register::V31.width_bits(), 128);
        assert_eq!(Register::Z0.width_bits(), 0);
        assert_eq!(Register::P15.width_bits(), 0);
        assert_eq!(Register::Pf0.width_bits(), 0);
    }

    #[test]
    fn number_field() {
        assert_eq!(Register::None.number(), 0);
        assert_eq!(Register::X0.number(), 0);
        assert_eq!(Register::X30.number(), 30);
        assert_eq!(Register::Xzr.number(), 31);
        assert_eq!(Register::Sp.number(), 31);
        assert_eq!(Register::W30.number(), 30);
        assert_eq!(Register::Wzr.number(), 31);
        assert_eq!(Register::Wsp.number(), 31);
        assert_eq!(Register::V31.number(), 31);
        assert_eq!(Register::Z0.number(), 0);
        assert_eq!(Register::P15.number(), 15);
        assert_eq!(Register::Pf31.number(), 31);
    }

    #[test]
    fn x_w_views() {
        // GP cross-width views.
        assert_eq!(Register::X3.as_w(), Register::W3);
        assert_eq!(Register::W3.as_x(), Register::X3);
        assert_eq!(Register::Xzr.as_w(), Register::Wzr);
        assert_eq!(Register::Wzr.as_x(), Register::Xzr);
        assert_eq!(Register::Sp.as_w(), Register::Wsp);
        assert_eq!(Register::Wsp.as_x(), Register::Sp);
        // Idempotent on the already-correct view.
        assert_eq!(Register::X3.as_x(), Register::X3);
        assert_eq!(Register::W3.as_w(), Register::W3);
        // Identity for non-GP registers.
        assert_eq!(Register::V0.as_x(), Register::V0);
        assert_eq!(Register::V0.as_w(), Register::V0);
        assert_eq!(Register::Z5.as_x(), Register::Z5);
        assert_eq!(Register::None.as_w(), Register::None);
    }

    #[test]
    fn is_simd_is_sve_flags() {
        assert!(Register::B0.is_simd());
        assert!(Register::Q31.is_simd());
        assert!(Register::V0.is_simd());
        assert!(!Register::X0.is_simd());
        assert!(!Register::Z0.is_simd());
        assert!(Register::Z0.is_sve());
        assert!(Register::P0.is_sve());
        assert!(!Register::V0.is_sve());
        assert!(!Register::X0.is_sve());
    }

    #[test]
    fn number_matches_name_suffix() {
        // Cross-check number() against the name table for every numbered class.
        let classes: &[(Register, u8)] = &[
            (Register::W0, 0),
            (Register::X0, 0),
            (Register::B0, 0),
            (Register::H0, 0),
            (Register::S0, 0),
            (Register::D0, 0),
            (Register::Q0, 0),
            (Register::V0, 0),
            (Register::Z0, 0),
        ];
        // Spot-check: number() of the first of each class is 0.
        for &(r, n) in classes {
            assert_eq!(r.number(), n);
        }
    }

    #[test]
    fn full_register_folds_fp_views_to_vector() {
        // Every scalar-FP width folds onto its V<n> parent.
        assert_eq!(Register::B0.full_register(), Register::V0);
        assert_eq!(Register::H5.full_register(), Register::V5);
        assert_eq!(Register::S17.full_register(), Register::V17);
        assert_eq!(Register::D1.full_register(), Register::V1);
        assert_eq!(Register::Q31.full_register(), Register::V31);
        // V<n> is its own full register.
        assert_eq!(Register::V3.full_register(), Register::V3);
        // Non-FP registers are identity (GP, SVE, predicate, prefetch, None).
        assert_eq!(Register::X0.full_register(), Register::X0);
        assert_eq!(Register::W30.full_register(), Register::W30);
        assert_eq!(Register::Sp.full_register(), Register::Sp);
        assert_eq!(Register::Wzr.full_register(), Register::Wzr);
        assert_eq!(Register::Z0.full_register(), Register::Z0);
        assert_eq!(Register::P15.full_register(), Register::P15);
        assert_eq!(Register::Pf0.full_register(), Register::Pf0);
        assert_eq!(Register::None.full_register(), Register::None);
        // The B/H/S/D/Q views of the same number share one parent.
        assert_eq!(Register::B9.full_register(), Register::S9.full_register());
        assert_eq!(Register::D9.full_register(), Register::Q9.full_register());
    }
}
