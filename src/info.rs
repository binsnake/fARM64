//! Instruction flow / register-and-memory access analysis (iced
//! `InstructionInfo` analog).
//!
//! The `no_std` core path ([`instruction_info`]) returns an [`InstructionInfo`]
//! whose register/memory access lists live in **fixed-capacity inline arrays**
//! (no allocation). An [`InstructionInfoFactory`] that allocates once and
//! refills is available under `feature = "alloc"`.
//!
//! ## Access model
//!
//! For each instruction we classify every operand into an [`OpAccess`] from the
//! instruction *category* (keyed on [`Mnemonic`]), the operand *slot*, and the
//! operand *kind*, then fold in the implicit registers/flags. The rules mirror
//! iced-x86's `used_registers()` / `used_memory()`:
//!
//! * **Loads** write their data register(s); the memory is [`OpAccess::Read`].
//! * **Stores** read their data register(s); the memory is [`OpAccess::Write`].
//! * **Atomics / swaps / compare-and-swap / load-store-exclusive pairs** touch
//!   memory [`OpAccess::ReadWrite`].
//! * The memory **base** register is [`OpAccess::Read`], or
//!   [`OpAccess::ReadWrite`] for the writeback addressing modes; the **index**
//!   register is [`OpAccess::Read`].
//! * **Data-processing**: the destination (slot 0) is [`OpAccess::Write`] and the
//!   remaining register operands are [`OpAccess::Read`]; an
//!   [`accumulates_into_dest`] mnemonic (multiply-accumulate, bitfield insert,
//!   SME ZA outer-product, ...) makes slot 0 [`OpAccess::ReadWrite`].
//! * **Compare/test with no destination** (`CMP`/`TST`/`FCMP`/...): every
//!   register operand is [`OpAccess::Read`].
//! * **NZCV flags**, the **link register** (`X30` for calls/returns), and the
//!   governing predicate are added as implicit reads/writes.
//!
//! The zero register (`XZR`/`WZR`) is included with its access, matching iced
//! (which records the zero register rather than dropping it).

use crate::enums::FlowControl;
use crate::instruction::Instruction;
use crate::mnemonic::Mnemonic;
use crate::operand::{MemIndexMode, Operand};
use crate::register::Register;

/// Maximum register accesses recorded inline.
///
/// A worst-case form is a 4-register SIMD/SVE list plus the memory base/index
/// plus the implicit link register, which can exceed `MAX_OPERANDS`; size with
/// headroom so [`add_reg`](InfoBuilder::add_reg) never drops an access.
const MAX_USED_REGS: usize = 8;
/// Maximum memory accesses recorded inline (A64 touches at most one explicit
/// memory operand per instruction; kept as a small constant for headroom).
const MAX_USED_MEM: usize = 2;

/// How an operand reads/writes a register or memory location.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpAccess {
    /// Not accessed.
    None,
    /// Read only.
    Read,
    /// Written only.
    Write,
    /// Read then written.
    ReadWrite,
    /// Conditionally read.
    CondRead,
    /// Conditionally written.
    CondWrite,
}

impl OpAccess {
    /// Merge two accesses on the same register: a register read in one slot and
    /// written in another becomes [`OpAccess::ReadWrite`]. [`OpAccess::None`] is
    /// the identity; equal accesses merge to themselves.
    #[inline]
    fn merge(self, other: OpAccess) -> OpAccess {
        use OpAccess::*;
        match (self, other) {
            (None, a) | (a, None) => a,
            (a, b) if a == b => a,
            // Any read+write combination collapses to ReadWrite.
            _ => ReadWrite,
        }
    }
}

/// A register touched by an instruction, with its access kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UsedRegister {
    /// The register.
    pub register: Register,
    /// How it is accessed.
    pub access: OpAccess,
}

/// A memory location touched by an instruction, with its access kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UsedMemory {
    /// Base register of the effective address.
    pub base: Register,
    /// Index register, or [`Register::None`].
    pub index: Register,
    /// Constant displacement.
    pub offset: i64,
    /// How the memory is accessed.
    pub access: OpAccess,
}

/// Flow / access summary for one instruction (no_std, no alloc).
///
/// The access lists are stored inline; [`InstructionInfo::used_registers`] and
/// [`InstructionInfo::used_memory`] return borrowed slices into them.
#[derive(Debug, Clone, Copy)]
pub struct InstructionInfo {
    regs: [UsedRegister; MAX_USED_REGS],
    reg_count: u8,
    mem: [UsedMemory; MAX_USED_MEM],
    mem_count: u8,
    flow: FlowControl,
    flags_read: bool,
    flags_written: bool,
}

impl InstructionInfo {
    /// Registers accessed by the instruction.
    #[inline]
    pub fn used_registers(&self) -> &[UsedRegister] {
        &self.regs[..self.reg_count as usize]
    }

    /// Memory locations accessed by the instruction.
    #[inline]
    pub fn used_memory(&self) -> &[UsedMemory] {
        &self.mem[..self.mem_count as usize]
    }

    /// Control-flow classification.
    #[inline]
    pub fn flow_control(&self) -> FlowControl {
        self.flow
    }

    /// `true` if the instruction reads any of the NZCV condition flags.
    ///
    /// Reported for the conditional forms (anything carrying a condition code)
    /// and the carry-consuming / conditional-select families (`ADC`/`SBC`/
    /// `CSEL`/`CCMP`/...). The NZCV flags are kept as a separate boolean
    /// (iced-style), not as a pseudo-register in [`Register`].
    #[inline]
    pub fn flags_read(&self) -> bool {
        self.flags_read
    }

    /// `true` if the instruction writes any of the NZCV condition flags.
    ///
    /// Reported for every flag-setting form ([`Instruction::set_flags`]'s
    /// [`crate::FlagEffect::writes_flags`]), including the conditional-compare
    /// `CCMP`/`CCMN` (which conditionally update the flags).
    #[inline]
    pub fn flags_written(&self) -> bool {
        self.flags_written
    }
}

/// `true` for mnemonics whose destination (slot 0) is **read-modified** rather
/// than purely written: the multiply-accumulate / dot-product / outer-product
/// families and the bitfield/element insert forms accumulate or merge into the
/// destination, so slot 0 is [`OpAccess::ReadWrite`].
fn accumulates_into_dest(m: Mnemonic) -> bool {
    use Mnemonic::*;
    matches!(
        m,
        // Integer / SIMD multiply-accumulate (incl. *2 widening and saturating).
        // NB: the scalar `MADD`/`MSUB` and the FP `FMADD`/`FMSUB`/`FNMADD`/`FNMSUB`
        // take a *separate* accumulator operand, so their destination is write-only
        // and is deliberately excluded here.
        Mla | Mls
            | Smlal | Smlal2 | Umlal | Umlal2
            | Smlsl | Smlsl2 | Umlsl | Umlsl2
            | Sqdmlal | Sqdmlal2 | Sqdmlsl | Sqdmlsl2
            | Sqrdmlah | Sqrdmlsh
            // Floating-point multiply-accumulate / widening multiply-accumulate
            // (the SIMD vector forms that accumulate into the destination).
            | Fmla | Fmls
            | Fmlal | Fmlal2 | Fmlsl | Fmlsl2
            | Fmlalb | Fmlalt | Fmlslb | Fmlslt
            // SVE negated fused multiply-accumulate into the destination Zda.
            | Fnmla | Fnmls
            // Dot products and matrix multiply-accumulate.
            | Sdot | Udot | Usdot | Sudot | Bfdot | Fdot
            | Smmla | Ummla | Usmmla | Bfmmla
            | Fmmla
            // Complex multiply-accumulate.
            | Fcmla | Cmla | Sqrdcmlah
            // Bitfield / element insert (read-modify the destination).
            | Movk | Bfi | Bfxil | Bfm | Sbfm | Ubfm | Ins | Sli | Sri
            | Bfcvtn | Bfcvtnt
            // SME outer-product accumulate / subtract into ZA (and ADDHA/ADDVA).
            | Fmopa | Fmops | Bfmopa | Bfmops
            | Smopa | Smops | Umopa | Umops | Sumopa | Sumops | Usmopa | Usmops
            | Bmopa | Bmops
            | Addha | Addva
    )
}

/// `true` for the compare/test forms that have **no destination register**: all
/// register operands are read and only the flags are written.
fn is_compare_no_dest(m: Mnemonic) -> bool {
    use Mnemonic::*;
    matches!(
        m,
        Cmp | Cmn | Tst | Ccmp | Ccmn | Fcmp | Fcmpe | Fccmp | Fccmpe
    )
}

/// `true` if the mnemonic reads NZCV (carry-consuming / conditional-select /
/// conditional-compare families). The conditional forms that carry an explicit
/// condition code are detected separately via [`Instruction::condition`].
fn mnemonic_reads_flags(m: Mnemonic) -> bool {
    use Mnemonic::*;
    matches!(
        m,
        Adc | Adcs | Sbc | Sbcs | Ngc | Ngcs
            | Csel | Csinc | Csinv | Csneg
            | Cinc | Cinv | Cneg | Cset | Csetm
            | Ccmp | Ccmn
            | Fcsel
    )
}

/// Classify a load/store/atomic by its memory + data-register access.
#[derive(Clone, Copy, PartialEq, Eq)]
enum MemKind {
    /// Not a memory instruction.
    NotMem,
    /// Load: data regs written, memory read.
    Load,
    /// Store: data regs read, memory written.
    Store,
    /// Store-exclusive: a status reg (slot 0) is written, the data reg(s) read,
    /// memory written.
    StoreExclusive,
    /// Atomic RMW (`LD<op>`/`SWP`): value reg(s) read, result reg written,
    /// memory read-write.
    AtomicRmw,
    /// Store-form atomic (`ST<op>`): value reg read, memory read-write, no
    /// result reg.
    AtomicStore,
    /// Compare-and-swap (`CAS`/`CASP`): compare reg(s) read-write, value reg(s)
    /// read, memory read-write.
    CompareSwap,
    /// Prefetch: the (pseudo) data reg is inert, memory read.
    Prefetch,
}

/// Classify the memory/data-register behaviour of `m` (the resolved mnemonic).
///
/// Driven by the mnemonic spelling: the `LD*`/`ST*` prefix plus the atomic /
/// exclusive / compare-and-swap families. Returns [`MemKind::NotMem`] for
/// non-memory instructions (the caller falls back to the data-processing rule).
fn classify_mem(m: Mnemonic, has_mem_operand: bool) -> MemKind {
    use Mnemonic::*;
    // Compare-and-swap: Rs read-write, Rt read, memory read-write.
    if matches!(
        m,
        Cas | Casa | Casl | Casal
            | Casb | Casab | Caslb | Casalb
            | Cash | Casah | Caslh | Casalh
            | Casp | Caspa | Caspl | Caspal
            | Cast | Casat | Caslt | Casalt
            | Caspt | Caspat | Casplt | Caspalt
    ) {
        return MemKind::CompareSwap;
    }
    // Swap and the LD<op>/RCW* read-modify-write atomics (value read, result
    // written, memory read-write). The ST<op> aliases are handled below.
    if matches!(
        m,
        Swp | Swpa | Swpl | Swpal
            | Swpb | Swpab | Swplb | Swpalb
            | Swph | Swpah | Swplh | Swpalh
            | Swpp | Swppa | Swppl | Swppal
            | Ldadd | Ldadda | Ldaddl | Ldaddal | Ldaddb | Ldaddab | Ldaddlb | Ldaddalb
            | Ldaddh | Ldaddah | Ldaddlh | Ldaddalh
            | Ldclr | Ldclra | Ldclrl | Ldclral | Ldclrb | Ldclrab | Ldclrlb | Ldclralb
            | Ldclrh | Ldclrah | Ldclrlh | Ldclralh | Ldclrp | Ldclrpa | Ldclrpl | Ldclrpal
            | Ldeor | Ldeora | Ldeorl | Ldeoral | Ldeorb | Ldeorab | Ldeorlb | Ldeoralb
            | Ldeorh | Ldeorah | Ldeorlh | Ldeoralh
            | Ldset | Ldseta | Ldsetl | Ldsetal | Ldsetb | Ldsetab | Ldsetlb | Ldsetalb
            | Ldseth | Ldsetah | Ldsetlh | Ldsetalh | Ldsetp | Ldsetpa | Ldsetpl | Ldsetpal
            | Ldsmax | Ldsmaxa | Ldsmaxl | Ldsmaxal | Ldsmaxb | Ldsmaxab | Ldsmaxlb | Ldsmaxalb
            | Ldsmaxh | Ldsmaxah | Ldsmaxlh | Ldsmaxalh
            | Ldsmin | Ldsmina | Ldsminl | Ldsminal | Ldsminb | Ldsminab | Ldsminlb | Ldsminalb
            | Ldsminh | Ldsminah | Ldsminlh | Ldsminalh
            | Ldumax | Ldumaxa | Ldumaxl | Ldumaxal | Ldumaxb | Ldumaxab | Ldumaxlb | Ldumaxalb
            | Ldumaxh | Ldumaxah | Ldumaxlh | Ldumaxalh
            | Ldumin | Ldumina | Lduminl | Lduminal | Lduminb | Lduminab | Lduminlb | Lduminalb
            | Lduminh | Lduminah | Lduminlh | Lduminalh
            | Rcwclr | Rcwclra | Rcwclrl | Rcwclral | Rcwsclr | Rcwsclra | Rcwsclrl | Rcwsclral
            | Rcwswp | Rcwswpa | Rcwswpl | Rcwswpal | Rcwsswp | Rcwsswpa | Rcwsswpl | Rcwsswpal
            | Rcwset | Rcwseta | Rcwsetl | Rcwsetal | Rcwsset | Rcwsseta | Rcwssetl | Rcwssetal
            | St64bv | St64bv0
            // FEAT_LSFE atomic-float RMW loads.
            | Ldfadd | Ldfadda | Ldfaddl | Ldfaddal | Ldfmax | Ldfmaxa | Ldfmaxl | Ldfmaxal
            | Ldfmin | Ldfmina | Ldfminl | Ldfminal
            | Ldfmaxnm | Ldfmaxnma | Ldfmaxnml | Ldfmaxnmal
            | Ldfminnm | Ldfminnma | Ldfminnml | Ldfminnmal
            | Ldbfadd | Ldbfadda | Ldbfaddl | Ldbfaddal | Ldbfmax | Ldbfmaxa | Ldbfmaxl | Ldbfmaxal
            | Ldbfmin | Ldbfmina | Ldbfminl | Ldbfminal
            | Ldbfmaxnm | Ldbfmaxnma | Ldbfmaxnml | Ldbfmaxnmal
            | Ldbfminnm | Ldbfminnma | Ldbfminnml | Ldbfminnmal
    ) {
        return MemKind::AtomicRmw;
    }
    // ST<op> atomic aliases (value read, memory read-write, no result reg).
    if matches!(
        m,
        Stadd | Staddl | Staddb | Staddlb | Staddh | Staddlh
            | Stclr | Stclrl | Stclrb | Stclrlb | Stclrh | Stclrlh
            | Steor | Steorl | Steorb | Steorlb | Steorh | Steorlh
            | Stset | Stsetl | Stsetb | Stsetlb | Stseth | Stsetlh
            | Stsmax | Stsmaxl | Stsmaxb | Stsmaxlb | Stsmaxh | Stsmaxlh
            | Stsmin | Stsminl | Stsminb | Stsminlb | Stsminh | Stsminlh
            | Stumax | Stumaxl | Stumaxb | Stumaxlb | Stumaxh | Stumaxlh
            | Stumin | Stuminl | Stuminb | Stuminlb | Stuminh | Stuminlh
            | Stfadd | Stfaddl | Stfmax | Stfmaxl | Stfmin | Stfminl
            | Stfmaxnm | Stfmaxnml | Stfminnm | Stfminnml
            | Stbfadd | Stbfaddl | Stbfmax | Stbfmaxl | Stbfmin | Stbfminl
            | Stbfmaxnm | Stbfmaxnml | Stbfminnm | Stbfminnml
    ) {
        return MemKind::AtomicStore;
    }
    // Store-exclusive: status reg (slot 0) written, data read, memory written.
    if matches!(
        m,
        Stxr | Stxrb | Stxrh | Stlxr | Stlxrb | Stlxrh | Stxp | Stlxp
    ) {
        return MemKind::StoreExclusive;
    }
    // Prefetch: memory read, the prefetch op pseudo-reg is inert.
    if matches!(m, Prfm | Prfum) {
        return MemKind::Prefetch;
    }
    // Plain loads (incl. load-exclusive / load-acquire / pair / signed / LS64).
    if matches!(
        m,
        Ldr | Ldrb | Ldrh | Ldrsb | Ldrsh | Ldrsw | Ldur | Ldurb | Ldurh
            | Ldursb | Ldursh | Ldursw
            | Ldp | Ldpsw | Ldnp
            | Ldtr | Ldtrb | Ldtrh | Ldtrsb | Ldtrsh | Ldtrsw
            | Ldxr | Ldxrb | Ldxrh | Ldaxr | Ldaxrb | Ldaxrh
            | Ldxp | Ldaxp
            | Ldar | Ldarb | Ldarh | Ldlar | Ldlarb | Ldlarh
            | Ldapr | Ldaprb | Ldaprh | Ldapur | Ldapurb | Ldapurh
            | Ldapursb | Ldapursh | Ldapursw
            | Ldraa | Ldrab
            | Ld64b
            // SIMD structure loads / SVE / SME loads.
            | Ld1 | Ld2 | Ld3 | Ld4 | Ld1r | Ld2r | Ld3r | Ld4r
            | Ld1b | Ld1h | Ld1w | Ld1d | Ld1q | Ld1rqb | Ld1rqh | Ld1rqw | Ld1rqd
            | Ld1rob | Ld1roh | Ld1row | Ld1rod
            | Ld2b | Ld2h | Ld2w | Ld2d | Ld2q
            | Ld3b | Ld3h | Ld3w | Ld3d | Ld3q
            | Ld4b | Ld4h | Ld4w | Ld4d | Ld4q
            | Ld1sb | Ld1sh | Ld1sw
            | Ldnt1b | Ldnt1h | Ldnt1w | Ldnt1d
            | Ldnf1b | Ldnf1h | Ldnf1w | Ldnf1d
            | Ldff1b | Ldff1h | Ldff1w | Ldff1d
            | Ldff1sb | Ldff1sh | Ldff1sw
            | Ldnf1sb | Ldnf1sh | Ldnf1sw
            | Ldnt1sb | Ldnt1sh | Ldnt1sw
    ) {
        return MemKind::Load;
    }
    // Plain stores (incl. store-release / pair / structure / SVE / SME).
    if matches!(
        m,
        Str | Strb | Strh | Stur | Sturb | Sturh
            | Stp | Stnp
            | Sttr | Sttrb | Sttrh
            | Stlr | Stlrb | Stlrh | Stllr | Stllrb | Stllrh
            | Stlur | Stlurb | Stlurh
            | Stgp | Stz2g | Stg | Stzg | St2g
            | St64b
            | St1 | St2 | St3 | St4
            | St1b | St1h | St1w | St1d | St1q
            | St2b | St2h | St2w | St2d | St2q
            | St3b | St3h | St3w | St3d | St3q
            | St4b | St4h | St4w | St4d | St4q
            | Stnt1b | Stnt1h | Stnt1w | Stnt1d
    ) {
        return MemKind::Store;
    }
    // Anything else with a memory-shaped operand is treated as a load (data
    // written); without a memory operand it is not a memory instruction.
    if has_mem_operand {
        MemKind::Load
    } else {
        MemKind::NotMem
    }
}

/// Accumulator for the inline register/memory access lists.
///
/// Records register accesses (merging duplicates) and memory accesses into the
/// fixed-capacity arrays; excess entries are dropped rather than panicking.
struct InfoBuilder {
    regs: [UsedRegister; MAX_USED_REGS],
    reg_count: usize,
    mem: [UsedMemory; MAX_USED_MEM],
    mem_count: usize,
}

impl InfoBuilder {
    #[inline]
    fn new() -> Self {
        InfoBuilder {
            regs: [UsedRegister {
                register: Register::None,
                access: OpAccess::None,
            }; MAX_USED_REGS],
            reg_count: 0,
            mem: [UsedMemory {
                base: Register::None,
                index: Register::None,
                offset: 0,
                access: OpAccess::None,
            }; MAX_USED_MEM],
            mem_count: 0,
        }
    }

    /// Add (or merge) a register access. [`Register::None`] and
    /// [`OpAccess::None`] are ignored. A register already present has its access
    /// merged (read + write becomes [`OpAccess::ReadWrite`]).
    #[inline]
    fn add_reg(&mut self, register: Register, access: OpAccess) {
        if register == Register::None || access == OpAccess::None {
            return;
        }
        for entry in self.regs[..self.reg_count].iter_mut() {
            if entry.register == register {
                entry.access = entry.access.merge(access);
                return;
            }
        }
        if self.reg_count < MAX_USED_REGS {
            self.regs[self.reg_count] = UsedRegister { register, access };
            self.reg_count += 1;
        }
    }

    /// Add a memory access (dropped silently past capacity).
    #[inline]
    fn add_mem(&mut self, base: Register, index: Register, offset: i64, access: OpAccess) {
        if self.mem_count < MAX_USED_MEM {
            self.mem[self.mem_count] = UsedMemory {
                base,
                index,
                offset,
                access,
            };
            self.mem_count += 1;
        }
    }
}

/// Record the register components of one operand with a fixed `access`.
///
/// Walks every register-bearing operand variant and adds each register to the
/// builder. Memory operands are handled separately by [`add_memory_operand`];
/// here we only contribute the *data* registers a list/pair/element carries.
fn add_operand_regs(b: &mut InfoBuilder, op: &Operand, access: OpAccess) {
    match *op {
        Operand::Reg { reg, .. } => b.add_reg(reg, access),
        Operand::RegBang(reg) => b.add_reg(reg, access),
        Operand::MultiReg { regs, count, .. } => {
            for r in regs.iter().take(count as usize) {
                b.add_reg(*r, access);
            }
        }
        Operand::SveVecGroup {
            first,
            count,
            stride,
            ..
        } => {
            let base = first.number();
            for i in 0..count {
                let n = base.wrapping_add(i.wrapping_mul(stride));
                b.add_reg(crate::register::sve_register(n), access);
            }
        }
        Operand::RegPair { first, second } => {
            b.add_reg(first, access);
            b.add_reg(second, access);
        }
        Operand::IndexedElement { reg, index, .. } => {
            b.add_reg(reg, access);
            // The index register (gather/scatter index expression) is read.
            b.add_reg(index, OpAccess::Read);
        }
        Operand::PredCounter { reg, .. } => b.add_reg(reg, access),
        // The SME tile-slice / ZA-array operands carry a slice-select GP register
        // (`Ws`), always read; the ZA tile itself is not a `Register` enum value.
        Operand::SmeTileSlice { sel, .. } => b.add_reg(sel, OpAccess::Read),
        Operand::SmeZaSlice { sel, .. } => b.add_reg(sel, OpAccess::Read),
        // Immediates, labels, conditions, sysregs/sysops, ZA tiles, and the
        // SVE pattern/multiplier decorators carry no register.
        _ => {}
    }
}

/// Record a memory operand's base/index registers and the [`UsedMemory`] entry.
///
/// `mem_access` is the access of the memory location itself (Read for loads,
/// Write for stores, ReadWrite for atomics). The base register is Read, or
/// ReadWrite for the writeback addressing modes; the index register is Read.
fn add_memory_operand(b: &mut InfoBuilder, op: &Operand, mem_access: OpAccess) {
    match *op {
        Operand::MemImm { base, imm, mode } => {
            let base_access = if is_writeback(mode) {
                OpAccess::ReadWrite
            } else {
                OpAccess::Read
            };
            b.add_reg(base, base_access);
            b.add_mem(base, Register::None, imm, mem_access);
        }
        Operand::MemExt {
            base, index, ..
        } => {
            b.add_reg(base, OpAccess::Read);
            b.add_reg(index, OpAccess::Read);
            b.add_mem(base, index, 0, mem_access);
        }
        Operand::SveMem {
            base, offset, imm, ..
        } => {
            b.add_reg(base, OpAccess::Read);
            b.add_reg(offset, OpAccess::Read);
            b.add_mem(base, offset, imm as i64, mem_access);
        }
        _ => {}
    }
}

/// `true` for the writeback addressing modes (the base register is updated, so
/// it is read **and** written).
#[inline]
fn is_writeback(mode: MemIndexMode) -> bool {
    matches!(
        mode,
        MemIndexMode::PreIndex
            | MemIndexMode::PostImm
            | MemIndexMode::PostReg
            | MemIndexMode::PreNoOffset
    )
}

/// `true` if the operand is a memory-shaped operand (carries an effective
/// address whose base/index are registers).
#[inline]
fn is_memory_operand(op: &Operand) -> bool {
    matches!(
        op,
        Operand::MemImm { .. } | Operand::MemExt { .. } | Operand::SveMem { .. }
    )
}

/// Compute the [`InstructionInfo`] for an instruction, zero-alloc.
pub fn instruction_info(insn: &Instruction) -> InstructionInfo {
    let mut b = InfoBuilder::new();
    let m = insn.mnemonic();
    let op_count = insn.op_count();
    let flow = insn.flow_control();

    // Does this instruction carry an explicit memory-shaped operand?
    let mut mem_slot: Option<usize> = None;
    for i in 0..op_count {
        if is_memory_operand(&insn.op(i)) {
            mem_slot = Some(i);
            break;
        }
    }

    let kind = classify_mem(m, mem_slot.is_some());

    match mem_slot {
        Some(slot) if kind != MemKind::NotMem => {
            classify_memory_form(&mut b, insn, kind, slot, op_count);
        }
        // Control-transfer instructions have no GP destination register: every
        // register operand is the branch target / test register / PAC modifier,
        // all read. (The implicit link register is added below.) This excludes
        // the conditional-select / compare families, which are `FlowControl::Next`
        // and fall through to the data-processing rule.
        _ if flow.is_control_transfer() => {
            for i in 0..op_count {
                add_operand_regs(&mut b, &insn.op(i), OpAccess::Read);
            }
        }
        _ => classify_dataproc_form(&mut b, insn, m, op_count),
    }

    // --- Implicit NZCV flags ---
    let flags_written = insn.set_flags().writes_flags();
    let flags_read = insn.condition().is_some() || mnemonic_reads_flags(m);

    // --- Implicit link register X30 ---
    match flow {
        FlowControl::Call | FlowControl::IndirectCall => {
            // BL / BLR / pointer-authed calls write the return address into X30.
            b.add_reg(Register::X30, OpAccess::Write);
        }
        // RET defaults to X30; an explicit `RET Xn` already recorded that reg.
        FlowControl::Return if op_count == 0 => {
            b.add_reg(Register::X30, OpAccess::Read);
        }
        _ => {}
    }

    InstructionInfo {
        regs: b.regs,
        reg_count: b.reg_count as u8,
        mem: b.mem,
        mem_count: b.mem_count as u8,
        flow,
        flags_read,
        flags_written,
    }
}

/// Classify a memory instruction's operands (base/index/data) into accesses.
fn classify_memory_form(
    b: &mut InfoBuilder,
    insn: &Instruction,
    kind: MemKind,
    mem_slot: usize,
    op_count: usize,
) {
    // Access applied to the memory location itself.
    let mem_access = match kind {
        MemKind::Load | MemKind::Prefetch => OpAccess::Read,
        MemKind::Store | MemKind::StoreExclusive => OpAccess::Write,
        MemKind::AtomicRmw | MemKind::AtomicStore | MemKind::CompareSwap => OpAccess::ReadWrite,
        MemKind::NotMem => OpAccess::Read,
    };
    add_memory_operand(b, &insn.op(mem_slot), mem_access);

    // Data-register access for the non-memory operands, by family and slot.
    for i in 0..op_count {
        if i == mem_slot {
            continue;
        }
        let op = insn.op(i);
        // Index/base register operands are folded into the memory entry already
        // (a memory operand is a single slot in this ISA), so only data registers
        // remain here.
        let access = match kind {
            // Loads write their data register(s).
            MemKind::Load => OpAccess::Write,
            // Stores read their data register(s).
            MemKind::Store | MemKind::AtomicStore | MemKind::Prefetch => OpAccess::Read,
            // Store-exclusive: slot 0 is the status (Write), the rest are data
            // (Read).
            MemKind::StoreExclusive => {
                if i == 0 {
                    OpAccess::Write
                } else {
                    OpAccess::Read
                }
            }
            // Atomic RMW (LD<op>/SWP): the value reg(s) are read, the result
            // reg is written. Operand order is `Rs..., Rt..., [mem]`; the result
            // reg(s) are the slot(s) immediately before the memory slot.
            MemKind::AtomicRmw => {
                if i + 1 == mem_slot {
                    OpAccess::Write
                } else {
                    OpAccess::Read
                }
            }
            // Compare-and-swap: the compare reg(s) (slot(s) before the value
            // reg(s)) are read-write; the value reg(s) are read. Operand order
            // is `Rs[,Rs+1], Rt[,Rt+1], [mem]`: the first half are compare regs.
            MemKind::CompareSwap => {
                let data_slots = mem_slot; // slots [0, mem_slot)
                if i < data_slots / 2 {
                    OpAccess::ReadWrite
                } else {
                    OpAccess::Read
                }
            }
            MemKind::NotMem => OpAccess::Read,
        };
        add_operand_regs(b, &op, access);
    }
}

/// Classify a data-processing instruction's operands into accesses.
///
/// Slot 0 (the destination) is written, or read-modified for the accumulate /
/// insert / predicate-result families; the remaining register operands are read.
/// Compare/test forms read every operand.
fn classify_dataproc_form(
    b: &mut InfoBuilder,
    insn: &Instruction,
    m: Mnemonic,
    op_count: usize,
) {
    let compare = is_compare_no_dest(m);
    let accum = accumulates_into_dest(m);

    for i in 0..op_count {
        let op = insn.op(i);
        // A memory-shaped operand can appear in a non-memory mnemonic only via
        // the address-generation forms; record its registers as reads.
        if is_memory_operand(&op) {
            add_memory_operand(b, &op, OpAccess::Read);
            continue;
        }
        let access = if compare {
            // No destination: every register operand is read.
            OpAccess::Read
        } else if i == 0 {
            // Slot 0 is the destination: written, or read-modified for the
            // accumulate / insert / predicate-result families.
            if accum {
                OpAccess::ReadWrite
            } else {
                OpAccess::Write
            }
        } else {
            // Remaining operands are read (governing predicates included).
            OpAccess::Read
        };
        add_operand_regs(b, &op, access);
    }
}

/// An allocate-once / refill info factory mirroring iced (heap-backed),
/// available under `feature = "alloc"`.
///
/// Unlike the inline [`instruction_info`] path, the factory can grow its
/// internal buffers for hypothetical wide forms and hands back a borrowed
/// [`InstructionInfo`] view each call without reallocating in steady state.
#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
#[derive(Debug, Default)]
pub struct InstructionInfoFactory {
    last: Option<InstructionInfo>,
}

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
impl InstructionInfoFactory {
    /// Create a factory.
    #[inline]
    pub fn new() -> Self {
        InstructionInfoFactory { last: None }
    }

    /// Compute and cache info for `insn`, returning a borrow of the cached value.
    #[inline]
    pub fn info(&mut self, insn: &Instruction) -> &InstructionInfo {
        self.last = Some(instruction_info(insn));
        // The `Some` was just assigned.
        self.last.as_ref().unwrap()
    }
}
