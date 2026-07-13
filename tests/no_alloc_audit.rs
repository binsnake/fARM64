#![cfg(feature = "no-alloc-audit")]

//! Allocation audit for the guaranteed zero-heap decode and format path.
//!
//! The allocator counts calls instead of panicking inside `GlobalAlloc`:
//! panicking while servicing an allocation can abort the process or recurse
//! through panic formatting. The audit window is disabled by a guard during
//! unwinding, and this integration-test binary contains only this one test so
//! unrelated tests cannot contribute allocator calls while the gate is open.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use fARM64::decode::decode_into;
use fARM64::format::{BufSink, FmtFormatter, Formatter};
use fARM64::{FeatureSet, Instruction};

static AUDITING: AtomicBool = AtomicBool::new(false);
static ALLOCATION_CALLS: AtomicUsize = AtomicUsize::new(0);

struct CountingAllocator;

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record_allocation();
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[inline]
fn record_allocation() {
    if AUDITING.load(Ordering::Relaxed) {
        ALLOCATION_CALLS.fetch_add(1, Ordering::Relaxed);
    }
}

struct AuditGuard;

impl AuditGuard {
    fn start() -> Self {
        AUDITING.store(false, Ordering::SeqCst);
        ALLOCATION_CALLS.store(0, Ordering::SeqCst);
        AUDITING.store(true, Ordering::SeqCst);
        AuditGuard
    }
}

impl Drop for AuditGuard {
    fn drop(&mut self) {
        AUDITING.store(false, Ordering::SeqCst);
    }
}

#[test]
fn decode_into_and_fixed_buffer_format_do_not_allocate() {
    // Exercise several independent base-ISA decoder groups. Keep all test
    // harness setup and assertions outside the audit window.
    let words = [
        0x9100_0420, // add x0, x1, #1
        0xD65F_03C0, // ret
        0x1400_0000, // b <pc>
        0xF940_0020, // ldr x0, [x1]
        0x4E22_8420, // add v0.16b, v1.16b, v2.16b
        0xD503_201F, // nop
    ];
    let mut rendered_lengths = [0usize; 6];
    let mut overflowed = false;

    let audit = AuditGuard::start();
    for ((word, rendered_len), index) in words.iter().zip(rendered_lengths.iter_mut()).zip(0u64..) {
        let mut instruction = Instruction::default();
        decode_into(*word, 0x1000 + index * 4, FeatureSet::ALL, &mut instruction);

        let formatter = FmtFormatter::new();
        let mut buffer = [0u8; 128];
        let mut sink = BufSink::new(&mut buffer);
        formatter.format(&instruction, &mut sink);
        *rendered_len = sink.len();
        overflowed |= sink.overflowed();
    }
    drop(audit);

    let allocation_calls = ALLOCATION_CALLS.load(Ordering::SeqCst);
    assert_eq!(
        allocation_calls, 0,
        "decode_into + FmtFormatter + BufSink performed {allocation_calls} allocation(s)"
    );
    assert!(
        !overflowed,
        "fixed formatting buffer unexpectedly overflowed"
    );
    assert!(
        rendered_lengths.iter().all(|&len| len > 0),
        "every audited instruction should render output: {rendered_lengths:?}"
    );
}
