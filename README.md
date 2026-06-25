# fARM64

**fARM64** is a pure-Rust, `#![no_std]`, zero-heap **AArch64 (A64) disassembler *and* encoder**. It decodes 64-bit ARM machine code into a rich, `Copy` value-type `Instruction`, renders it through a pluggable `Formatter`, and can re-encode an `Instruction` back into its 32-bit word. The public API is deliberately iced-x86-shaped (a borrowing `Decoder`, a value-type `Instruction`, typed `OpKind`/`Operand` accessors, a token-emitting `Formatter`). The decode path is an **original, hand-written recursive decode tree built directly from the ARM Architecture Reference Manual** (the "ARM ARM") — not a port or transpile of any other decoder. "Correct" is defined as *matching the ARM ARM*: 100% of the Binary Ninja golden corpus decodes (99.75% text parity, with the residual being documented Binary Ninja rendering bugs rather than fARM64 errors), the extension surface is validated differentially against LLVM 21 (`llvm-mc`), and the encoder round-trips every supported encoding 100% semantically.

---

## Highlights

- **Freestanding by default.** `#![no_std]` *unconditionally*, with **no `alloc` and no `std`** in the default build. No-CRT / bare-metal / wasm friendly; builds for `wasm32-unknown-unknown` and `aarch64-unknown-none`.
- **Zero heap on the core path.** `Decoder::decode_into` writes into a caller-owned `Copy` `Instruction` (no `Vec`, no `Box`, no internal pointers); the default formatter writes into a fixed `&mut [u8]` (`BufSink`) or any `core::fmt::Write`.
- **`Copy` value-type `Instruction`.** Pass it by value; inline `[Operand; MAX_OPERANDS]` storage, `<= 112` bytes, asserted at compile time. Never panics on malformed input — bad words decode to `Code::Invalid` with a recorded `last_error`.
- **Ergonomic iteration.** `Decoder` is an `Iterator` (both `for insn in &mut dec` and consuming `for insn in dec`), plus a `decode_into` fast path for tight loops.
- **Broad ISA coverage.** Full base A64 plus Advanced SIMD / FP, SVE / SVE2, SME / SME2, the crypto extensions, and a long tail of recent additions: MOPS, CSSC, RCPC3, D128, THE, LSE128, SVE2p1, CMPBR, CPA, and more.
- **Encoder included.** `Instruction::encode()` reconstructs the 32-bit word from instruction *semantics* (never from the stored raw word), proving the decode is invertible.
- **Pluggable formatting.** Default ARM UAL `FmtFormatter`, an optional GNU/objdump dialect behind `fmt-gnu`, a token-classifying `FormatterOutput` sink, and a `SymbolResolver` hook.
- **Two independent feature layers.** Cargo features control what is *compiled in*; a runtime `FeatureSet` controls what is *accepted at decode time*.

---

## Supported targets

| Target | Notes |
|-|-|
| `x86_64-*`, `aarch64-*` (hosted) | development and `std` testing |
| `wasm32-unknown-unknown` | default features (`no_std`, no `alloc`) |
| `aarch64-unknown-none` | bare-metal, no-CRT; build with `-Zbuild-std=core` |
| any target providing `core` | the default tier is `core`-only |

## Feature matrix

Cargo features decide what is **compiled**; the runtime `FeatureSet` decides what is **accepted** at decode time. The two are independent layers.

| Cargo feature | Tier | Effect |
|-|-|-|
| *(none / default)* | A | `no_std`, **no `alloc`**, freestanding. Decoder + `FmtFormatter` + all enums + encoder. Always builds. |
| `alloc` | B | Adds `String`/`Vec` conveniences (`format_to_string`, the allocate-once `InstructionInfoFactory`, a token-collecting `String` sink). |
| `std` | C | Implies `alloc`; adds `std::error::Error` for `DecodeError`/`EncodeError` and std-only test helpers. |
| `fmt-gnu` | A | Optional GNU/objdump formatter dialect (`GnuFormatter`). Pure `no_std`. |
| `fp16` `bf16` `lse` `pauth` `mte` `sme` `sve` `crypto` | A | Compile-in per-extension table slices and enum variants. |
| `full` | A | All per-extension features at once. |

The default build links neither `alloc` nor `std`. `std` implies `alloc`. The runtime `FeatureSet` (`FeatureSet::ALL`, `FeatureSet::BASE`, `.with(Feature::Sve)`, `.has(..)`) is orthogonal to all of the above.

---

## Install

```toml
[dependencies]
# Default: no_std, no alloc, zero-heap decoder + formatter + encoder.
fARM64 = "0.0.1"
```

Opt into more as needed:

```toml
# Owned-String conveniences (format_to_string, info factory).
fARM64 = { version = "0.0.1", features = ["alloc"] }

# Everything: std + every architecture extension + the GNU dialect.
fARM64 = { version = "0.0.1", features = ["std", "full", "fmt-gnu"] }
```

The import path uses the stylized crate name: `use fARM64::...`.

---

## Quick start

Zero-allocation decode-and-print into a fixed stack buffer — no heap, no `std`:

```rust
use fARM64::{Decoder, DecoderOptions};
use fARM64::format::{Formatter, FmtFormatter, BufSink};

fn main() {
    // `ADD W0, W1, #1`, little-endian; decode at address 0x1000.
    let code = [0x20, 0x04, 0x00, 0x11];
    let mut dec = Decoder::new(&code, 0x1000, DecoderOptions::default());
    let insn = dec.decode();

    // Format into a fixed [u8; N] — the whole path touches no heap.
    let mut buf = [0u8; 64];
    let mut sink = BufSink::new(&mut buf);
    FmtFormatter::new().format(&insn, &mut sink);

    let text: &str = sink.as_str();
    let _ = text; // e.g. "add     w0, w1, #0x1"
}
```

`Decoder::new(data, ip, options)` borrows the byte slice; `ip` is the address of `data[0]` and PC-relative operands resolve against it. `decode()` returns a `Copy` `Instruction` and advances the cursor by 4. The `FmtFormatter::format` call goes through the `Formatter` trait, which is why that trait is imported.

---

## Decoding

```rust
use fARM64::{Decoder, DecoderOptions, Code};

fn main() {
    let code: &[u8] = &[
        0x20, 0x04, 0x00, 0x11, // add w0, w1, #1
        0x1f, 0x20, 0x03, 0xd5, // nop
    ];

    // Construct over the slice; `ip` is the address of code[0].
    let mut dec = Decoder::new(code, 0x1000, DecoderOptions::default());

    // Iterate by &mut: yields instructions until fewer than 4 bytes remain.
    for insn in &mut dec {
        if insn.is_invalid() {
            // Bad/unallocated word: inspect dec.last_error() for why.
            continue;
        }
        let _ = (insn.code(), insn.ip());
    }

    // After the loop, the most recent decode status is available:
    let _ = dec.last_error();
    let _ = Code::Invalid; // the sentinel a failed decode carries
}
```

Key points:

- **`Decoder::new(data, ip, options)`** never panics. There is also a `try_new` returning `Result<Decoder, DecodeError>` for API symmetry. A64 has no bitness parameter — it is always 64-bit, always 4-byte fixed-width.
- **Iteration forms.** `for insn in &mut dec` borrows the decoder (you can inspect `dec.last_error()` afterward); `for insn in dec` consumes it. Both yield until fewer than 4 bytes remain.
- **`decode()` vs `decode_into()`.** `decode()` returns a fresh `Instruction`. `decode_into(&mut out)` writes into a caller-owned `Instruction`, which is the preferred zero-allocation form in a tight loop because nothing is constructed or moved per iteration:

```rust
use fARM64::{Decoder, DecoderOptions, Instruction};

fn main() {
    let code: &[u8] = &[0x20, 0x04, 0x00, 0x11, 0x1f, 0x20, 0x03, 0xd5];
    let mut dec = Decoder::new(code, 0x1000, DecoderOptions::default());

    // Reuse one Instruction across the whole loop — no per-iteration alloc/move.
    let mut insn = Instruction::default();
    while dec.can_decode() {
        dec.decode_into(&mut insn);
        let _ = insn.code();
    }
}
```

- **Position and address.** `position()`/`set_position(pos)` move the byte cursor (and keep `ip` consistent relative to the original base); `ip()`/`set_ip(ip)` read/set the current decode address directly.
- **Invalid handling.** A malformed or unallocated word never panics — it decodes to an `Instruction` with `Code::Invalid` (check `insn.is_invalid()`), and `dec.last_error()` returns the reason (`DecodeError::Unmatched`, `DecodeError::EndOfInstruction` on a short tail, or `DecodeError::None` on success).

### Restricting accepted extensions

`DecoderOptions` carries a `FeatureSet`. The default accepts everything (`FeatureSet::ALL`); narrow it to reject encodings outside the extensions you target:

```rust
use fARM64::{Decoder, DecoderOptions, FeatureSet, Feature};

fn main() {
    // Accept only the base ISA plus FEAT_LSE atomics; reject everything else.
    let features = FeatureSet::BASE.with(Feature::Lse);
    let options = DecoderOptions { features };

    let code = [0x20, 0x04, 0x00, 0x11]; // add w0, w1, #1 (base ISA)
    let mut dec = Decoder::new(&code, 0x1000, options);
    let insn = dec.decode();
    let _ = insn.code();
}
```

---

## Inspecting an `Instruction`

```rust
use fARM64::{Decoder, DecoderOptions};
use fARM64::{OpKind, Operand, Register};

fn main() {
    let code = [0x20, 0x04, 0x00, 0x11]; // add w0, w1, #1
    let mut dec = Decoder::new(&code, 0x1000, DecoderOptions::default());
    let insn = dec.decode();

    // `code()` is the precise encoding identity (e.g. Code::AddImm32);
    // `mnemonic()` is the preferred/alias display spelling; `.name()` is the text.
    let _enc = insn.code();
    let _mnem = insn.mnemonic();
    let _name: &str = insn.mnemonic().name();

    // Address helpers (A64 is fixed 4-byte wide).
    let _ = (insn.ip(), insn.next_ip(), insn.len(), insn.word());

    // Control-flow class and NZCV write behaviour.
    let _ = (insn.flow_control(), insn.set_flags());

    // Walk operands. `op_kind(n)` is the cheap discriminant; `op(n)` is the rich value.
    for i in 0..insn.op_count() {
        match insn.op_kind(i) {
            // Fast typed accessors that skip the match for the common cases:
            OpKind::Register => {
                let r: Register = insn.op_register(i);
                let _ = (r.name(), r.class(), r.number());
            }
            OpKind::ImmUnsigned | OpKind::ImmSigned | OpKind::ImmLogical => {
                let _v: u64 = insn.op_immediate(i);
            }
            _ => {}
        }

        // Or match the full rich Operand for everything:
        match insn.op(i) {
            Operand::Reg { reg, arr, shift, extend, .. } => {
                let _ = (reg, arr, shift, extend);
            }
            Operand::ImmUnsigned(v) | Operand::ImmLogical(v) => { let _ = v; }
            Operand::ImmSigned(v) => { let _ = v; }
            Operand::MemImm { base, imm, mode } => { let _ = (base, imm, mode); }
            Operand::MemExt { base, index, extend, shift } => { let _ = (base, index, extend, shift); }
            Operand::Label(target) => { let _ = target; }
            Operand::Cond(c) => { let _ = c; }
            _ => {}
        }
    }
}
```

What each accessor means:

- **`code()`** — the encoding-level identity (`Code`), one variant per distinct ARM ARM encoding row. Use this when you need the exact encoding (e.g. for re-encoding, or to distinguish `B.cond` from `B`).
- **`mnemonic()` / `mnemonic().name()`** — the width/encoding-independent `Mnemonic` (alias-resolved for preferred disassembly such as `MOV`/`CMP`/`LSL`), and its `&'static str` spelling.
- **`op_count()` / `op_kind(n)` / `op(n)`** — operand count, the `OpKind` discriminant of slot `n` (out-of-range yields `OpKind::None`), and the full rich `Operand` (out-of-range yields `Operand::None`).
- **`op_register(n)` / `op_immediate(n)`** — fast indexed accessors. `op_register` returns `Register::None` if slot `n` is not a plain register; `op_immediate` returns the unsigned/logical/signed-as-`u64`/label value, or `0` otherwise.
- **`len()` / `ip()` / `next_ip()` / `word()`** — fixed length (always 4), decode address, following address (`ip + 4`), and the raw little-endian word.
- **`flow_control()`** — `FlowControl` classification (branch / call / return / exception / next). **`set_flags()`** — `FlagEffect` NZCV behaviour (`SetsNormal`, `SetsFloat`, or `None`).

---

## Formatting

The default `FmtFormatter` renders ARM UAL syntax. It writes through the `Formatter` trait into any `FormatterOutput` sink. There is a single operand-dispatch path; nothing allocates inside the formatter itself.

### Zero-alloc into a fixed buffer or any `core::fmt::Write`

```rust
use core::fmt::Write;
use fARM64::{Decoder, DecoderOptions};
use fARM64::format::{Formatter, FmtFormatter, BufSink};

fn main() {
    let code = [0x20, 0x04, 0x00, 0x11];
    let mut dec = Decoder::new(&code, 0x1000, DecoderOptions::default());
    let insn = dec.decode();
    let fmt = FmtFormatter::new();

    // (a) Into a fixed [u8; N] via BufSink (no heap). Overflow is observable.
    let mut buf = [0u8; 64];
    let mut sink = BufSink::new(&mut buf);
    fmt.format(&insn, &mut sink);
    assert!(!sink.overflowed());
    let _text: &str = sink.as_str();

    // (b) Into any core::fmt::Write — the blanket impl makes it a sink.
    struct Counter(usize);
    impl Write for Counter {
        fn write_str(&mut self, s: &str) -> core::fmt::Result { self.0 += s.len(); Ok(()) }
    }
    let mut c = Counter(0);
    fmt.format(&insn, &mut c);
    let _ = c.0;
}
```

### Owned `String` (requires `alloc`)

```rust
use fARM64::{Decoder, DecoderOptions};
use fARM64::format::{FmtFormatter, format_to_string};

fn main() {
    let code = [0x20, 0x04, 0x00, 0x11];
    let mut dec = Decoder::new(&code, 0x1000, DecoderOptions::default());
    let insn = dec.decode();

    let fmt = FmtFormatter::new();
    let s: String = format_to_string(&fmt, &insn);
    let _ = s;
}
```

### `FormatterOptions`

`FmtFormatter::with_options(opts)` overrides the defaults. Fields and their defaults:

| Field | Default | Meaning |
|-|-|-|
| `aliases` | `true` | Emit preferred aliases (`MOV`/`CMP`/`MUL`/`LSL`/`NOP`/...) instead of canonical forms. |
| `uppercase_mnemonics` | `false` | Upper-case mnemonics. |
| `uppercase_registers` | `false` | Upper-case register names. |
| `use_sp_not_xzr` | `true` | Render reg-31 as `sp`/`wsp` rather than `xzr`/`wzr` where the role is ambiguous. |
| `hex_prefix` | `"0x"` | Prefix for hex literals. |
| `signed_immediates` | `true` | Render signed immediates with an explicit `-` and hex magnitude. |
| `show_lsl_zero` | `false` | Show `LSL #0` explicitly instead of eliding it. |
| `space_after_operand_separator` | `true` | `", "` vs `","` between operands. |
| `first_operand_char_index` | `8` | Column at which the first operand starts (mnemonic field width). |

```rust
use fARM64::format::{FmtFormatter, FormatterOptions};

fn main() {
    let opts = FormatterOptions { uppercase_mnemonics: true, ..FormatterOptions::default() };
    let _fmt = FmtFormatter::with_options(opts);
}
```

A GNU/objdump dialect (`GnuFormatter`) is available behind `feature = "fmt-gnu"`.

### A token sink (`FormatterOutput` + `TokenKind`)

For syntax coloring or post-processing, implement `FormatterOutput` and receive every chunk together with its `TokenKind`:

```rust
use fARM64::{Decoder, DecoderOptions};
use fARM64::format::{Formatter, FmtFormatter, FormatterOutput, TokenKind};

struct TokenSink {
    mnemonics: usize,
    registers: usize,
}

impl FormatterOutput for TokenSink {
    fn write(&mut self, _text: &str, kind: TokenKind) {
        match kind {
            TokenKind::Mnemonic => self.mnemonics += 1,
            TokenKind::Register => self.registers += 1,
            _ => {}
        }
    }
}

fn main() {
    let code = [0x20, 0x04, 0x00, 0x11];
    let mut dec = Decoder::new(&code, 0x1000, DecoderOptions::default());
    let insn = dec.decode();

    let mut sink = TokenSink { mnemonics: 0, registers: 0 };
    FmtFormatter::new().format(&insn, &mut sink);
    let _ = (sink.mnemonics, sink.registers);
}
```

### Resolving branch targets (`SymbolResolver`)

`SymbolResolver` maps an address to a borrowed name (no allocation required):

```rust
use fARM64::Instruction;
use fARM64::format::{SymbolResolver, SymbolResult};

struct MyResolver;

impl SymbolResolver for MyResolver {
    fn symbol(
        &mut self,
        _insn: &Instruction,
        _operand: usize,
        address: u64,
    ) -> Option<SymbolResult<'_>> {
        if address == 0x2000 {
            Some(SymbolResult { name: "my_func", offset: 0 })
        } else {
            None
        }
    }
}

fn main() {
    let mut r = MyResolver;
    // A formatter integration can call `r.symbol(insn, n, target)` for each
    // Label/Address operand to substitute a name for the bare 0x... target.
    let _ = &mut r;
}
```

---

## Encoding

`Instruction::encode()` (or the free function `fARM64::encode(&insn)`) reconstructs the 32-bit little-endian word from the instruction's **semantics** — its `Code`, `Mnemonic`, operands, and `ip`. It deliberately never reads `Instruction::word()`, so a successful round-trip proves the decode is invertible. The encoder is `no_std`, zero-alloc, and total: it returns `EncodeError` rather than panicking.

```rust
use fARM64::{Decoder, DecoderOptions, EncodeError};

fn main() {
    let code = [0x20, 0x04, 0x00, 0x11]; // add w0, w1, #1
    let mut dec = Decoder::new(&code, 0x1000, DecoderOptions::default());
    let insn = dec.decode();

    // Decode -> (optionally inspect/modify) -> re-encode from semantics.
    let word: Result<u32, EncodeError> = insn.encode();
    match word {
        Ok(w) => {
            // Semantic round-trip: re-decoding `w` yields an equivalent instruction.
            let _ = w;
        }
        // Encodings the encoder does not yet cover return EncodeError::Unsupported.
        Err(e) => { let _ = e; }
    }
}
```

`EncodeError` variants: `Unsupported` (this `Code`/group is not implemented yet), `InvalidOperand` (operand missing or of the wrong kind), `InvalidImmediate` (an immediate, shift, or PC-relative target with no valid field encoding), and `Invalid` (the `Code::Invalid` sentinel has no encoding).

Because the encoder rebuilds from the canonical `Code`, the guarantee is a **semantic** round-trip (the re-encoded word decodes to an equivalent instruction), not necessarily a byte-identical one for encodings that have multiple equivalent spellings.

---

## Feature gating explained

There are two independent layers:

1. **Cargo features** decide what is **compiled into the binary**. A base-only wasm build can omit the SVE/SME tables entirely (smaller code) by not enabling `sve`/`sme`.
2. **The runtime `FeatureSet`** decides what the decoder will **accept** at decode time. Even with everything compiled in, you can refuse encodings outside a chosen extension set.

```rust
use fARM64::{Decoder, DecoderOptions, FeatureSet};

fn main() {
    // Restrict the decoder to the base ISA only: an SVE/SME/extension word will
    // be rejected (decoded as Code::Invalid) even though its tables are compiled in.
    let options = DecoderOptions { features: FeatureSet::BASE };

    let code = [0x20, 0x04, 0x00, 0x11]; // a base-ISA ADD: still accepted
    let mut dec = Decoder::new(&code, 0x1000, options);
    let insn = dec.decode();
    let _ = insn.is_invalid();
}
```

`FeatureSet::ALL` (the default) accepts everything; `FeatureSet::BASE`/`NONE` accept only the base ISA; `.with(Feature::X)` enables one extension; `.has(Feature::X)` queries one. `Feature::Base` is always present.

---

## `no_std`, embedded, and wasm

The default build is `#![no_std]` with **no `alloc`**: it links neither an allocator nor `std`. The core decode path (`Decoder` + `Instruction`) and the default formatter (`FmtFormatter` + `BufSink`) never allocate — all names are `&'static str` from `const` tables, there are no thread-locals, no I/O, no time, and no panics-as-control-flow. This makes fARM64 usable directly in kernels, bootloaders, hypervisors, and `wasm32-unknown-unknown`. The `alloc` and `std` features are strictly additive conveniences; enabling them never changes the zero-heap behaviour of the core path.

---

## Validation and testing

fARM64 is validated against multiple independent oracles. The big sweeps are `#[ignore]`d (report-only) and run explicitly:

- **Binary Ninja golden corpus** — `tests/golden.rs`. Decodes the full corpus and compares rendered text; 100% decode, 99.75% text parity (residual = documented Binary Ninja rendering bugs).
  ```
  cargo test --features "std full" --test golden -- --ignored --nocapture
  ```
- **LLVM 21 differential** — `tests/llvm_diff.rs`. A discovery sweep that diffs fARM64 against `llvm-mc` (needs LLVM 21 installed).
  ```
  cargo test --features "std full" --test llvm_diff -- --ignored --nocapture
  ```
- **Encoder round-trip** — `tests/roundtrip.rs`. Decodes the corpus, re-encodes from semantics, and checks the semantic round-trip.
  ```
  cargo test --features "std full" --test roundtrip -- --ignored --nocapture
  ```
- **Example CLI** — `examples/disasm.rs`. Decodes 8-hex-digit words from args or stdin:
  ```
  cargo run --example disasm 11000420 d503201f
  ```

The fast (non-ignored) unit and integration tests run with a plain `cargo test --features "std full"`. The documented spec-vs-Binary-Ninja divergences (where fARM64 follows the ARM ARM and Binary Ninja does not) are recorded in `docs/VALIDATION.md`.

---

## Project layout and architecture

```
src/
  lib.rs          crate docs, public re-exports, MAX_OPERANDS / INSN_LEN, static asserts
  decoder.rs      Decoder, DecoderOptions, iterators, position/ip, last_error
  decode/         hand-written recursive A64 decode tree (+ shared ARM pseudocode)
  encode/         hand-written A64 encoder (the inverse of decode)
  instruction.rs  the Copy value-type Instruction and its accessors
  operand.rs      Operand enum + OpKind discriminant
  register.rs     Register, RegClass, RegWidth, gp_register
  enums.rs        Condition / ShiftType / ExtendType / VectorArrangement / FlowControl / FlagEffect
  mnemonic.rs     Code (encoding identity) + Mnemonic (display) enums
  features.rs     Feature + FeatureSet (runtime accept/reject)
  format/         Formatter trait, FmtFormatter, BufSink, options, token sink
  info/ sysop/ sysreg/ tables/   info factory, system ops/registers, name tables
tests/            golden.rs, llvm_diff.rs, roundtrip.rs, the_atomics.rs
examples/         disasm.rs
docs/             DESIGN.md, API.md, ENCODING.md, ROADMAP.md, VALIDATION.md
```

Design and reference docs: [`docs/DESIGN.md`](docs/DESIGN.md), [`docs/API.md`](docs/API.md), [`docs/ENCODING.md`](docs/ENCODING.md), [`docs/ROADMAP.md`](docs/ROADMAP.md), [`docs/VALIDATION.md`](docs/VALIDATION.md).

---

## Status and coverage

- 100% of the Binary Ninja golden corpus decoded; 99.75% text parity (residual = documented Binary Ninja rendering bugs).
- Encoder round-trips 100% semantically over the implemented groups.
- Extension surface validated differentially against LLVM 21 (`llvm-mc`).
- ~308 tests (fast unit/integration plus the report-only corpus sweeps).
- Version `0.0.1`; the `Code`/`Mnemonic`/`Register`/`Feature` enums are `#[non_exhaustive]` with an append-only discriminant policy, so new ARM revisions add variants without breaking downstream `match`es.

## License

Licensed under either of **MIT** or **Apache-2.0**, at your option. Original work derived from the publicly documented ARM ARM instruction encodings; it is not a derivative of any other disassembler. See `NOTICE` for details.
