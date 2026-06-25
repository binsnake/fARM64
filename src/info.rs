//! Instruction flow / register-and-memory access analysis (iced
//! `InstructionInfo` analog).
//!
//! The `no_std` core path ([`instruction_info`]) returns an [`InstructionInfo`]
//! whose register/memory access lists live in **fixed-capacity inline arrays**
//! (no allocation). An [`InstructionInfoFactory`] that allocates once and
//! refills is available under `feature = "alloc"`.

use crate::enums::FlowControl;
use crate::instruction::Instruction;
use crate::register::Register;
use crate::MAX_OPERANDS;

/// Maximum register accesses recorded inline (operands + a flags pseudo-entry).
const MAX_USED_REGS: usize = MAX_OPERANDS + 1;
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
}

/// Compute the [`InstructionInfo`] for an instruction, zero-alloc.
pub fn instruction_info(insn: &Instruction) -> InstructionInfo {
    let _ = insn;
    todo!()
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
