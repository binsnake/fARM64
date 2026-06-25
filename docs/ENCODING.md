# ENCODING.md — AArch64 Encoding Reference & Decoder Notes (fARM64)

This is the encoding-level reference for the `fARM64` AArch64 disassembler. It
documents how a 32-bit little-endian A64 instruction word is classified, how each
base-ISA group lays out its bit fields, the operand shapes those fields produce,
and the tricky derivations (logical-immediate masks, PC-relative `ADR`/`ADRP`,
move-wide `hw`, load/store addressing modes, SP-vs-ZR resolution at register 31,
condition codes, and the alias / preferred-disassembly rules). It closes with the
differential-testing plan and the oracle corpora it compares against.

The correctness contract is **conformance with the *Arm Architecture Reference
Manual* (the "ARM ARM")**. `fARM64`'s decoder is an **original, hand-written
implementation derived directly from the ARM ARM** — a recursive decode tree plus
plain hand-written transcriptions of the ARM pseudocode (`DecodeBitMasks`,
`AdvSIMDExpandImm`, `VFPExpandImm`, `MoveWidePreferred`, `Replicate`, …). It is
**not** a derivative of any other disassembler. "Correct" means *matches the ARM
ARM*, cross-checked against multiple independent oracles (LLVM `llvm-mc` /
`llvm-objdump`, GNU binutils `objdump`, and Binary Ninja's `arch-arm64`
`test_cases.txt` corpus used **only as one development guide**).

Where another tool deliberately diverges from the spec, `fARM64` **follows the
spec** and records the divergence on an allow-list. The load-bearing example is
`DecodeBitMasks`: Binary Ninja sets `tmask == wmask` under a `// TODO: do this
right` comment at `refs/arch-arm64-master/disassembler/pcode.c:88`; `fARM64`
instead computes `tmask` per the ARM ARM and notes the intentional difference. See
[DecodeBitMasks](#decodebitmasks-logical-immediates) below.

All structure described here is consumed at decode time with **zero heap
allocation**: the decode tree is hand-written `match`/`if` Rust, fields are
extracted with small inline bit helpers, ARM pseudocode runs as plain integer
functions, register/mnemonic/sysreg names are `&'static str` from `const` tables,
and the default formatter writes into a caller-supplied `&mut dyn core::fmt::Write`
or a fixed `&mut [u8]` buffer (`BufSink`). `alloc`/`std` are strictly additive.
Nothing in this document requires an allocator.

---

## Conventions

- Bit numbering: bit 31 is the most-significant bit of the 32-bit word; bit 0 is
  least-significant. A field written `imm12[21:10]` occupies bits 21 down to 10
  inclusive (12 bits). Extraction is `(word >> 10) & 0xFFF`.
- A64 is **fixed-width**: every instruction is exactly 4 bytes, 4-byte aligned,
  little-endian. `next_ip == ip + 4` always. There is no lookahead.
- `sf` (bit 31 in most data-processing forms) selects operand width: `sf==0` → 32-bit
  (`Wn`), `sf==1` → 64-bit (`Xn`).
- Register field meaning of value 31 is context-dependent and resolved during
  operand build, never leaked raw. See [SP-vs-ZR](#sp-vs-zr-register-31).
- ASCII bit-field tables below show each field's bit span and width; `(0)`/`(1)`
  denote fixed bits that participate in the accept mask/pattern.

---

## Decode pipeline overview

The decoder is a **hand-written recursive descent over the A64 encoding tree**,
transcribed directly from the ARM ARM. A word becomes `(Code, [Operand; up to 5])`
without any table interpreter, `EncId` index, or operand bytecode — every step is
ordinary Rust code in the `src/decode/*` modules.

1. **Top-level dispatch** (`decode::decode_into`, `src/decode/mod.rs`): a single
   `match (word >> 25) & 0xf` on `op0 = bits[28:25]` selects one of the eight A64
   encoding groups and calls the matching hand-written group decoder. The entry
   function also handles the reserved/UDF space and the `HINT` special case
   (`EndOfInstruction` accepted, not an error).
2. **Per-group decoders** (one file per group — see the module map below): each
   decoder matches the group's sub-fields with nested `match`/`if`, validates the
   fixed bits and constraints inline, and builds the `Instruction` **directly**.
   Field extraction uses small inline bit helpers in `src/decode/bits.rs`
   (e.g. `bits(word, hi, lo)`, `bit(word, n)`, `sext`) — the readable form of
   `let Rm = (word >> 16) & 0x1f;`.
3. **ARM pseudocode helpers** (`src/decode/bits.rs`, hand-written): the genuinely
   derived values are plain functions transcribed from the ARM ARM —
   `decode_bit_masks`, `move_wide_preferred`, `bfx_preferred`, `adv_simd_expand_imm`,
   `vfp_expand_imm`, `decode_shift`, `replicate`, `highest_set_bit`, sign/zero
   extend. All integer math, no heap, no FP arithmetic.
4. **Operand + SP/ZR resolution**: operands are constructed inline in each group
   decoder as it matches. Register-31 ambiguity is resolved with
   `gp_register(use_sp, width, n)` at the point of construction; raw `31` never
   leaks. (See [SP-vs-ZR](#sp-vs-zr-register-31).)

Aliasing is applied **in code**, per the ARM ARM "alias conditions" for each
encoding (e.g. `MOV` ← `ORR`, `CMP` ← `SUBS`, `LSL`/`LSR`/`ASR` ← `UBFM`/`SBFM`,
`NOP` ← `HINT`). Decode always yields a canonical `Code`; the alias condition sets
the resolved `Mnemonic`, gated by `FormatterOptions::aliases` (default on). The
`Code`/`Mnemonic` enums and the `&'static str` name tables are the *only*
mechanically generated artefacts (emitted by the optional host-only `cargo xtask
gen` from a curated, ARM-spec-derived dataset); they contain no decode **logic**.

### DecodeError outcomes

The decode tree returns a typed `DecodeError` on every fall-through. The variants
correspond to the ARM ARM's decode outcomes (reserved / unallocated / UNDEFINED
encodings, `SEE`-elsewhere redirections, and constraint violations) plus a Rust-side
`FeatureRequired`. The discriminant values are stable and chosen for ergonomics.

| `DecodeError` | meaning |
|-|-|
| `None` | success; the named encoding is accurate |
| `Reserved` | the spec marks this encoding space reserved |
| `Unmatched` | the word fell through the spec's structural checks |
| `Unallocated` | the encoding space is unallocated in the ARM ARM |
| `Undefined` | a decode constraint made this encoding UNDEFINED |
| `EndOfInstruction` | accepted (not an error) only for a HINT encoding |
| `Lost` | descended past checks where the spec redirects (`SEE` up-higher) |
| `Unreachable` | reached an `Unreachable()` point in the pseudocode |
| `AssertFailed` | a decode-time assertion failed |
| `ErrorOperands` | operand construction failed |
| `FeatureRequired(Feature)` | encoding gated off by the active `FeatureSet` |

The `HINT`/`EndOfInstruction` special case lives in the entry function
(`decode::decode_into`): `EndOfInstruction` is accepted (not an error) when the
matched encoding is a HINT.

---

## Top-level decode tree (root key: bits[28:25] = op0)

This is the root `match` in `decode::decode_into` (`src/decode/mod.rs`), keyed on
the ARM ARM's top-level `op0` field:

```rust
let op0 = (word >> 25) & 0xf;   // bits[28:25]
match op0 {
    0b0000 => decode_reserved(...),                  // UDF + SME (word<31>==1, gated)
    0b0001 | 0b0011 => { /* unallocated: leave Invalid */ }
    0b0010 => decode_sve(...),                        // SVE/SVE2 (gated)
    0b1000 | 0b1001 => dp_imm::decode(...),           // 100x
    0b1010 | 0b1011 => branch_sys::decode(...),       // 101x
    0b0100 | 0b0110 | 0b1100 | 0b1110 => ldst::decode(...),  // x1x0
    0b0101 | 0b1101 => dp_reg::decode(...),           // x101
    0b0111 | 0b1111 => simd_fp::decode(...),          // x111
    _ => {}                                           // total without panic
}
```

Each row maps to a hand-written module under `src/decode/`. `x` is a don't-care
bit of `op0`. (`x1x0` matches `{0100,0110,1100,1110}`; `x101` matches
`{0101,1101}`; `x111` matches `{0111,1111}`.)

| op0 (bits[28:25]) | group | `src/decode/` module | feature gate |
|-|-|-|-|
| `0000` | Reserved (`UDF`) + SME (`word<31>==1`) | entry `decode_reserved` → `sme/` | `Feature::Sme` for the SME sub-region |
| `0001` / `0011` | Unallocated | — (left `Invalid`) | — |
| `0010` | SVE / SVE2 | `sve/` | `Feature::Sve` (else `Invalid`/`FeatureRequired`) |
| `100x` | Data Processing — Immediate | `dp_imm.rs` | base |
| `101x` | Branch / Exception / System | `branch_sys.rs` | base |
| `x1x0` | Loads and Stores | `ldst.rs` (+ `ldst_simd.rs`) | base |
| `x101` | Data Processing — Register | `dp_reg.rs` | base |
| `x111` | DP — Scalar FP & Advanced SIMD | `simd_fp/` | FP/NEON (+ `crypto` sub-gate) |

Notes:

- Bit 31 (`sf` / `op` in various forms) further splits some spaces *inside* a group
  decoder (e.g. `ADR` vs `ADRP`, where it is the page selector). It does not change
  the top-level group selection.
- The reserved space (`op0==0000` with `word<31:16>==0`) decodes to `UDF`, the
  permanently-undefined encoding (`udf #imm16`). Within the same `op0==0000`
  region, `word<31>==1` is the SME sub-tree (routed to `sme/` only when the `sme`
  cargo feature is compiled and `Feature::Sme` is accepted; otherwise the word is
  left `Invalid`).
- The SME and SVE sub-trees are large and structurally distinct; they are routed
  only when both compiled (cargo feature) and accepted (runtime `FeatureSet`).
  Otherwise the word is left `Code::Invalid` rather than being silently
  misdecoded as base ISA. `SMSTART`/`SMSTOP` are the exception: they are
  `MSR (immediate)` PSTATE encodings handled in `branch_sys.rs`.
- Feature gating happens at two independent layers: a **cargo feature** decides
  what code/enum variants are *compiled* (size control for wasm/embedded), and the
  runtime **`FeatureSet`** decides what is *accepted*. A base-only build omits the
  SVE/SME decoders entirely.

The base-ISA groups are detailed below in `op0` order.

---

## Group: Data Processing — Immediate (op0 = `100x`, `src/decode/dp_imm.rs`)

Sub-decoded by `bits[25:23]`.

| bits[25:23] | sub-class | representative encodings |
|-|-|-|
| `000` / `001` | PC-relative addressing | `ADR`, `ADRP` |
| `010` | Add/subtract (immediate) | `ADD`, `ADDS`, `SUB`, `SUBS` (+ `CMP`/`CMN`/`MOV` aliases) |
| `011` | Add/subtract (immediate, tags) | `ADDG`, `SUBG` (MTE) |
| `100` | Logical (immediate) | `AND`, `ORR`, `EOR`, `ANDS` (+ `MOV`/`TST` aliases) |
| `101` | Move wide (immediate) | `MOVN`, `MOVZ`, `MOVK` |
| `110` | Bitfield | `SBFM`, `BFM`, `UBFM` (+ many aliases) |
| `111` | Extract | `EXTR` (+ `ROR` alias) |

### PC-relative: ADR / ADRP

```
 31  30 29   28          24 23                                5 4       0
+---+-----+----------------+-----------------------------------+---------+
|op | immlo |  1  0  0  0  0 |              immhi               |    Rd   |
+---+-----+----------------+-----------------------------------+---------+
 op=bit31  immlo=[30:29]                immhi=[23:5] (19 bits)   Rd=[4:0]
```

- `imm21 = immhi:immlo` (21 bits), sign-extended.
- `op==0` → **ADR**: `Rd = ip + SignExtend(imm21)`.
- `op==1` → **ADRP**: `imm = SignExtend(imm21) << 12`, `Rd = (ip & ~0xFFF) + imm`
  (page-aligned base). The corpus prints the *resolved absolute address* as an
  `Address` token, not the raw immediate.
- `Rd` is always a general-purpose X register (use-ZR form; value 31 → `XZR`).

### Add/subtract (immediate)

```
 31  30 29  28      24 23 22 21          10 9       5 4       0
+--+--+--+-------------+--+----------------+---------+---------+
|sf|op| S| 1  0  0  0  1|sh|     imm12      |    Rn   |    Rd   |
+--+--+--+-------------+--+----------------+---------+---------+
```

- `sf`: 0=32-bit, 1=64-bit. `op`: 0=ADD, 1=SUB. `S`: set flags (`ADDS`/`SUBS`).
- `sh` (bit 22): `0` → `imm12` used as-is; `1` → `imm12 << 12` (rendered `lsl #0xc`).
- `Rn`/`Rd` are **use-SP** forms here (value 31 → `SP`/`WSP`), *except* when `S==1`
  where `Rd` is use-ZR.
- Aliases: `CMN <Xn>, #imm` ← `ADDS Rd==ZR`; `CMP <Xn>, #imm` ← `SUBS Rd==ZR`;
  `MOV <Xd|SP>, <Xn|SP>` ← `ADD #0` with `sh==0` when `Rd` or `Rn` is SP.

### Logical (immediate)

```
 31  30 29  28      23 22 21        16 15        10 9       5 4       0
+--+-----+-------------+--+------------+------------+---------+---------+
|sf| opc | 1  0  0  1  0| N|    immr    |    imms    |    Rn   |    Rd   |
+--+-----+-------------+--+------------+------------+---------+---------+
```

- `opc`: `00`=AND, `01`=ORR, `10`=EOR, `11`=ANDS.
- The actual 64/32-bit immediate is `DecodeBitMasks(N, imms, immr).wmask`
  (see below). Some `(N,immr,imms)` triples are reserved/UNDEFINED.
- For `sf==0` (32-bit), `N` must be 0 or the encoding is UNDEFINED.
- `Rd` is use-SP for `AND/ORR/EOR` (writes can target SP), use-ZR for `ANDS`.
  `Rn` is use-ZR.
- Aliases: `MOV <Xd|SP>, #imm` ← `ORR` with `Rn==ZR`; `TST <Xn>, #imm` ← `ANDS`
  with `Rd==ZR`.

#### DecodeBitMasks (logical immediates)

This is the helper most worth getting right. `fARM64::decode::bits::decode_bit_masks`
is a **faithful, hand-written transcription of the ARM ARM `DecodeBitMasks`
pseudocode** — it computes both `wmask` and `tmask` per the spec.

Algorithm (integer-only, no heap, no FP), from the ARM ARM:

1. Determine element size from `(immN, imms)` via `HighestSetBit(immN:NOT(imms))`:
   if `immN==1`, the element is 64 bits; otherwise the highest set bit of the
   6-bit field `NOT(imms)` selects element widths 32/16/8/4/2. An all-ones run with
   no valid length is **UNDEFINED** (return `DecodeError::Undefined`).
2. Compute `levels` (a mask of `len` ones), `S = imms & levels`, `R = immr & levels`.
   `S == levels` is reserved for non-`tmask` consumers and rejected where the spec
   requires (`immediate` forms).
3. `welem = Ones(S + 1)`, then `wmask = Replicate(ROR(welem, R), esize)`.
4. `telem = Ones(diff + 1)` (with `diff = (S - R) & levels`), then
   `tmask = Replicate(telem, esize)` — `Replicate` is the hand-written helper in
   `bits.rs`.
5. Return both `wmask` and `tmask`.

**Intentional divergence (allow-listed).** Binary Ninja's `arch-arm64` sets
`tmask = wmask` under a `// TODO: do this right` comment at
`refs/arch-arm64-master/disassembler/pcode.c:88`. `fARM64` follows the ARM ARM and
computes the correct `tmask`. For logical-**immediate** disassembly the two agree
(only `wmask` is printed), so this never affects base-ISA output; where `tmask`
matters, the binja corpus entry is recorded on the
[intentional-divergence allow-list](#oracle-corpora-and-the-divergence-allow-list).

### Move wide (immediate): MOVN / MOVZ / MOVK

```
 31  30 29  28      23 22 21 20                       5 4       0
+--+-----+-------------+-----+---------------------------+---------+
|sf| opc | 1  0  0  1  0|  hw |          imm16            |    Rd   |
+--+-----+-------------+-----+---------------------------+---------+
```

- `opc`: `00`=MOVN, `10`=MOVZ, `11`=MOVK (`01` unallocated).
- `hw` (bits[22:21]) is the shift selector: `lsl #(hw*16)`. `Rd` is use-ZR.
- **`hw` undefined rule**: when `sf==0` (32-bit), `hw[1]` (bit 22) must be 0; a
  value of `hw ∈ {10,11}` with `sf==0` is **UNDEFINED** (only `lsl #0`/`#16` are
  legal for 32-bit). For `sf==1`, all four `hw` values are legal (`lsl #0/16/32/48`).
- Operand shape: `Operand::ImmShiftedMove { imm: imm16, lsl: hw*16 }`.
- Alias: `MOV <Xd>, #imm` is preferred for `MOVZ`/`MOVN` (and the bitmask `ORR`)
  when `move_wide_preferred` (the ARM ARM `MoveWidePreferred` pseudocode, in
  `decode/bits.rs`) selects it. `MoveWidePreferred` requires the value fit
  MOVZ/MOVN's "≤16 bits not crossing a halfword boundary" test and checks
  `sf`-dependent splat constraints.

### Bitfield: SBFM / BFM / UBFM

```
 31  30 29  28      23 22 21        16 15        10 9       5 4       0
+--+-----+-------------+--+------------+------------+---------+---------+
|sf| opc | 1  0  0  1  1| N|    immr    |    imms    |    Rn   |    Rd   |
+--+-----+-------------+--+------------+------------+---------+---------+
```

- `opc`: `00`=SBFM, `01`=BFM, `10`=UBFM. `N` must equal `sf`.
- This is the alias-richest base group. Selection follows the ARM ARM alias
  conditions, applied in `decode/dp_imm.rs` (with the `bfx_preferred` helper —
  the ARM ARM `BFXPreferred` pseudocode — in `decode/bits.rs`):
  - `LSL <Xd>, <Xn>, #sh` ← `UBFM` when `imms+1 == immr` (i.e. `imms != 63/31`).
  - `LSR <Xd>, <Xn>, #sh` ← `UBFM` when `imms == 63` (or 31 for 32-bit).
  - `ASR <Xd>, <Xn>, #sh` ← `SBFM` when `imms == 63`/`31`.
  - `UBFIZ`/`SBFIZ` ← `U/SBFM` when `imms < immr`.
  - `UBFX`/`SBFX` ← `U/SBFM` when `BFXPreferred` (imms ≥ immr and not an
    LSR/ASR/LSL/UXT/SXT alias).
  - `UXTB/UXTH` ← `UBFM`, `SXTB/SXTH/SXTW` ← `SBFM` with `immr==0` and
    `imms ∈ {7,15,31}`.
  - `BFI`/`BFXIL`/`BFC` ← `BFM`.

### Extract: EXTR

```
 31  30 29  28      23 22 21 20      16 15        10 9       5 4       0
+--+-----+-------------+--+--+----------+------------+---------+---------+
|sf| 0 0 | 1  0  0  1  1| N| 0|    Rm    |    imms    |    Rn   |    Rd   |
+--+-----+-------------+--+--+----------+------------+---------+---------+
```

- `imms` is the rotate amount (`lsb`). `N` must equal `sf`.
- Alias: `ROR <Xd>, <Xs>, #imm` ← `EXTR` when `Rn == Rm`.

---

## Group: Branches, Exception generation & System (op0 = `101x`, `src/decode/branch_sys.rs`)

Sub-decoded by `bits[31:29]`.

| bits[31:29] | sub-class | examples |
|-|-|-|
| `010` | Conditional branch (immediate) | `B.<cond>`, `BC.<cond>` |
| `110` | Exception generation | `SVC`, `HVC`, `SMC`, `BRK`, `HLT`, `DCPS1-3` |
| `110` (System sub-space) | System instructions | `NOP`/`YIELD`/`WFE`/`WFI`/`SEV`/`SEVL` (HINT), `CLREX`, `DSB`, `DMB`, `ISB`, `SYS`/`SYSL`, `MSR`/`MRS` |
| `100` | Unconditional branch (register) | `BR`, `BLR`, `RET`, `ERET`, `DRPS` (+ `BRAA`/`BLRAA` PAuth) |
| `000`/`100` | Unconditional branch (immediate) | `B`, `BL` |
| `0xx` | Compare and branch (immediate) | `CBZ`, `CBNZ` |
| `0xx` | Test and branch (immediate) | `TBZ`, `TBNZ` |

### Conditional branch (immediate): B.<cond>

```
 31           24 23                              5 4   3       0
+---------------+---------------------------------+--+----------+
| 0 1 0 1 0 1 0 0|            imm19               |o0|   cond   |
+---------------+---------------------------------+--+----------+
```

- `imm19` (bits[23:5]) is a word offset: `target = ip + SignExtend(imm19:0b00)`,
  i.e. `<< 2`. Rendered as an `Address` token.
- `cond` (bits[3:0]) selects the 16-way condition (see [Condition codes](#condition-codes)).
  `o0` (bit 4) distinguishes `B.<cond>` (0) from `BC.<cond>` (1, consistent-branch).
- Operand shape: `Operand::Cond` baked into the mnemonic suffix plus an
  `Operand::Label`.

### Unconditional branch (immediate): B / BL

```
 31  30           26 25                                          0
+--+----------------+----------------------------------------------+
|op| 0  0  1  0  1   |                  imm26                       |
+--+----------------+----------------------------------------------+
```

- `op`: 0=B, 1=BL. `target = ip + SignExtend(imm26:0b00)` (`<< 2`).
- `BL` additionally writes the return address to `X30` (modeled in `InstructionInfo`).

### Compare-and-branch / Test-and-branch

```
CBZ/CBNZ:
 31 30        25 24 23                              5 4       0
+--+-----------+--+----------------------------------+---------+
|sf| 0 1 1 0 1 0|op|             imm19               |    Rt   |
+--+-----------+--+----------------------------------+---------+

TBZ/TBNZ:
 31 30        25 24 23     19 18                  5 4       0
+--+-----------+--+----------+----------------------+---------+
|b5| 0 1 1 0 1 1|op|   b40    |        imm14         |    Rt   |
+--+-----------+--+----------+----------------------+---------+
```

- CBZ/CBNZ: `op` 0=CBZ, 1=CBNZ; offset is `imm19 << 2`. `Rt` is use-ZR.
- TBZ/TBNZ: bit number is `(b5 << 5) | b40` (so b5 also doubles as `sf`); offset is
  `imm14 << 2`.

### Unconditional branch (register): BR / BLR / RET / ...

```
 31              25 24    21 20  16 15      10 9       5 4       0
+------------------+--------+------+----------+---------+---------+
| 1 1 0 1 0 1 1     |  opc   |  op2 |   op3    |    Rn   |   op4   |
+------------------+--------+------+----------+---------+---------+
```

- `opc` selects `BR`(0000)/`BLR`(0001)/`RET`(0010)/`ERET`(0100)/`DRPS`(0101).
- `Rn` is the branch target register (use-ZR). PAuth variants (`BRAA`, `BRAB`,
  `BLRAA`, `BLRAB`, `RETAA`, `RETAB`) live here under `Feature::PAuth`.

### System: HINT and the alias-by-immediate space

The HINT space (`bits` select `CRm:op2`) decodes named hints:

| CRm:op2 | mnemonic |
|-|-|
| `0000:000` | `NOP` |
| `0000:001` | `YIELD` |
| `0000:010` | `WFE` |
| `0000:011` | `WFI` |
| `0000:100` | `SEV` |
| `0000:101` | `SEVL` |
| `0000:110` | `DGH` |

- `NOP <- HINT #0` etc. are the *named-hint* preferred forms; unknown CRm:op2
  values format as `HINT #<imm>`. Where the binja corpus prints `hint ...` instead
  of a named hint (`dgh`) or `msr ...` instead of `cfinv`, those entries are on the
  [intentional-divergence allow-list](#oracle-corpora-and-the-divergence-allow-list).
- `MSR`/`MRS` carry a 15-bit system-register key
  `op0:op1:CRn:CRm:op2`; see [System registers](#system-registers).
- Barriers `DSB`/`DMB`/`ISB`/`CLREX` take a 4-bit `CRm` "option" immediate
  (e.g. `dsb sy`, `dmb ish`).

---

## Group: Loads and Stores (op0 = `x1x0`, `src/decode/ldst.rs`)

Further subdivided by `bits[31:30]` (`size`), `bits[29:28]`, and `bits[26:24]`.
The six addressing modes are captured by `Operand::MemImm { base, imm, mode }`
(`mode ∈ {Offset, PreIndex, PostImm}`) and `Operand::MemExt { base, index, extend, shift }`.

| sub-class | addressing | example |
|-|-|-|
| Load register (literal) | PC-relative `imm19` word offset | `LDR <Xt>, <label>`, `LDRSW`, `PRFM` |
| Load/store exclusive | base-only `[Xn|SP]` | `LDXR`, `STXR`, `LDAXR`, `STLXR`, `LDXP`, `STXP` |
| Load/store pair | `imm7` scaled, offset/pre/post | `LDP`, `STP`, `LDNP`, `STNP` |
| Load/store unsigned imm | `imm12` scaled by access size | `LDR [Xn,#imm]`, `STRB` |
| Load/store unscaled | `imm9` (`LDUR`/`STUR`) | `LDUR <Xt>,[Xn,#imm]` |
| Load/store pre-index | `imm9`, write-back, `]!` | `LDR <Xt>,[Xn,#imm]!` |
| Load/store post-index | `imm9`, write-back, `],#imm` | `LDR <Xt>,[Xn],#imm` |
| Load/store register-offset | extended/shifted index reg | `LDR <Xt>,[Xn,Xm,LSL #s]` |
| Atomic memory ops (LSE) | base-only `[Xn|SP]` | `LDADD`, `SWP`, `LDCLR`, `LDSET`, `CAS` |
| Adv. SIMD load/store | structure forms | `LD1`..`LD4`, `ST1`..`ST4` |

### Register (unsigned immediate) — the canonical scaled form

```
 31 30 29   27 26 25 24 23 22 21          10 9       5 4       0
+-----+-------+--+-----+-----+--------------+---------+---------+
|size | 1 1 1 |V | 0 1 | opc |    imm12     |    Rn   |    Rt   |
+-----+-------+--+-----+-----+--------------+---------+---------+
```

- `size` (bits[31:30]): access size 8/16/32/64-bit (`B/H/W/X`); `V` selects the
  SIMD&FP register file. `opc` distinguishes load/store and sign-extension.
- `imm12` is **scaled by the access size**: effective byte offset = `imm12 << size`.
- `Rn` is **use-SP** (the base register; value 31 → `SP`). `Rt` is use-ZR (or a
  SIMD register when `V==1`).
- Mode is `MemImm{ mode: Offset }`.

### Unscaled / pre / post (imm9)

```
 31 30 29   27 26 25 24 23 22 21 20      12 11 10 9    5 4    0
+-----+-------+--+-----+-----+--+----------+-----+------+------+
|size | 1 1 1 |V | 0 0 | opc | 0|   imm9   | mode|  Rn  |  Rt  |
+-----+-------+--+-----+-----+--+----------+-----+------+------+
```

- `imm9` (bits[20:12]) is **signed**, **not scaled**.
- The `mode` bits[11:10] select: `00`→`LDUR`/`STUR` (`Offset`, unscaled);
  `11`→pre-index (`PreIndex`, `]!`); `01`→post-index (`PostImm`, `],#imm`).
- Base `Rn` is use-SP; pre/post variants write the new base back into `Rn`.

### Pair (imm7)

```
 31 30 29   27 26 25 23 22 21       15 14    10 9    5 4    0
+-----+-------+--+--------+--+----------+------+------+------+
|opc  | 1 0 1 |V | mode   | L|   imm7   |  Rt2 |  Rn  |  Rt  |
+-----+-------+--+--------+--+----------+------+------+------+
```

- `imm7` (bits[21:15]) is signed, **scaled by the pair's element size**. `L`:
  0=store, 1=load. The `mode` field selects offset / pre-index / post-index /
  no-allocate (`LDNP`/`STNP`).
- Two transfer registers `Rt`, `Rt2` (use-ZR or SIMD). `Rn` is the use-SP base.

### Register-offset (extended index)

The index register carries an `ExtendType` (`UXTW`/`LSL`/`SXTW`/`SXTX`) and an
optional scale: `Operand::MemExt { base, index, extend, shift }`, e.g.
`[x0, w1, uxtw #2]` or `[x0, x1, lsl #3]`. `option==011` with `S` set renders as
`lsl` (the canonical no-extend case for X index).

---

## Group: Data Processing — Register (op0 = `x101`, `src/decode/dp_reg.rs`)

Decoded by `bits[28:24]` and friends.

| sub-class | examples |
|-|-|
| Logical (shifted register) | `AND`, `ORR`, `EOR`, `ANDS`, `BIC`, `ORN`, `EON` (+ `MOV`/`MVN`/`TST`) |
| Add/sub (shifted register) | `ADD`, `SUB`, `ADDS`, `SUBS` (+ `NEG`/`NEGS`/`CMP`/`CMN`) |
| Add/sub (extended register) | `ADD`/`SUB` with `UXTB`..`SXTX` (+ `CMP`/`CMN`) |
| Add/sub with carry | `ADC`, `SBC`, `ADCS`, `SBCS` (+ `NGC`/`NGCS`) |
| Rotate/flag manipulation | `RMIF`, `SETF8`, `SETF16` |
| Conditional compare | `CCMN`, `CCMP` (register & immediate) |
| Conditional select | `CSEL`, `CSINC`, `CSINV`, `CSNEG` (+ `CSET`/`CSETM`/`CINC`/`CINV`/`CNEG`) |
| Data-processing (2 source) | `UDIV`, `SDIV`, `LSLV`, `LSRV`, `ASRV`, `RORV`, `CRC32*`, `PACGA` |
| Data-processing (1 source) | `RBIT`, `REV16`, `REV32`, `REV`, `CLZ`, `CLS`, `PACIA`, `AUTIA`, `XPAC` |
| Data-processing (3 source) | `MADD`, `MSUB`, `SMADDL`, `UMADDL`, `SMULH`, `UMULH` (+ `MUL`/`MNEG`/`SMULL`/`UMULL`) |

### Logical (shifted register)

```
 31 30 29 28    24 23 22 21 20  16 15        10 9    5 4    0
+--+-----+--------+-----+--+------+------------+------+------+
|sf| opc | 0 1 0 1 0|shift|N |  Rm  |    imm6    |  Rn  |  Rd  |
+--+-----+--------+-----+--+------+------------+------+------+
```

- `opc`+`N` select AND/BIC/ORR/ORN/EOR/EON/ANDS/BICS.
- `shift` (bits[23:22]): `00`=LSL, `01`=LSR, `10`=ASR, `11`=ROR; amount `imm6`.
  All operands are use-ZR (no SP). For `sf==0`, `imm6[5]` must be 0.
- The `Operand::Reg` carries `shift: Option<(ShiftType, u8)>`.
- Aliases: `MOV <Xd>, <Xm>` ← `ORR` with `Rn==ZR, imm6==0, shift==LSL`;
  `MVN` ← `ORN Rn==ZR`; `TST` ← `ANDS Rd==ZR`.

### Add/sub (shifted register)

Same field layout with `shift ∈ {LSL,LSR,ASR}` (ROR illegal). Aliases:
`CMN` ← `ADDS Rd==ZR`, `CMP` ← `SUBS Rd==ZR`, `NEG` ← `SUB Rn==ZR`,
`NEGS` ← `SUBS Rn==ZR`.

### Add/sub (extended register)

```
 31 30 29 28    24 23 22 21 20  16 15  13 12 10 9    5 4    0
+--+--+--+--------+-----+--+------+------+------+------+------+
|sf|op| S| 0 1 0 1 1| 0 0|1 |  Rm  |option| imm3 |  Rn  |  Rd  |
+--+--+--+--------+-----+--+------+------+------+------+------+
```

- `option` (bits[15:13]): `000`=UXTB ... `111`=SXTX (see [Extend types](#extend-and-shift-types)).
- `imm3` (bits[12:10]) is the left-shift amount 0–4. `Rn`/`Rd` are **use-SP**
  (this is the canonical place SP appears as a source/dest in arithmetic; the
  preferred display drops `LSL #0` and `UXTX`/`UXTW` when redundant per
  `FormatterOptions::show_lsl_zero`).

### Conditional select & DP 1/2/3-source

- CSEL family: `CSEL Rd, Rn, Rm, cond`. Aliases: `CSET`/`CSETM` (`Rn==Rm==ZR`,
  inverted cond), `CINC`/`CINV`/`CNEG` (`Rn==Rm`, inverted cond).
- DP 3-source `MADD Rd, Rn, Rm, Ra`: alias `MUL` ← `MADD Ra==ZR`,
  `MNEG` ← `MSUB Ra==ZR`, `SMULL`/`UMULL` ← `S/UMADDL Ra==ZR`.

---

## Group: Data Processing — Scalar FP & Advanced SIMD (op0 = `x111`, `src/decode/simd_fp/`)

Decoded by `bit[28]` + `bits[27:24]`. This is the largest base surface; the
hand-written `simd_fp/mod.rs` sub-classifier (ARM ARM C4.1.97) dispatches to
`scalar_fp.rs` (scalar FP), `simd_arith.rs` (Adv-SIMD arithmetic), `simd_data.rs`
(Adv-SIMD data-movement / modified-immediate) and `crypto.rs` (gated by the
`crypto` feature), all leaning on the NEON/FP pseudocode helpers in
`decode/bits.rs`. SVE and SME are *not* here — they are their own gated sub-trees
(`sve/`, `sme/`).

| sub-class | examples |
|-|-|
| Scalar FP compare/convert/DP | `FCMP`, `FCVT`, `FMOV`, `FADD`, `FMUL`, `FMADD`, `FCSEL` |
| Adv. SIMD three same / three different | `ADD <Vd>.T`, `FMLA`, `SQDMULL` |
| Adv. SIMD two-reg misc / across lanes | `ABS`, `CNT`, `ADDV`, `SADDLV` |
| Adv. SIMD copy / permute / table / extract | `DUP`, `INS`, `TRN1/2`, `UZP1/2`, `ZIP1/2`, `TBL`, `TBX`, `EXT` |
| Adv. SIMD modified immediate | `MOVI`, `MVNI`, `FMOV` (via `AdvSIMDExpandImm`) |
| Indexed-element forms | `MUL <Vd>.T, <Vn>.T, <Vm>.S[idx]` |
| Crypto (gated) | `AES*`, `SHA1*`, `SHA256*`, `SHA512*`, `SM3*`, `SM4*` |

Key derivations (hand-written pseudocode helpers in `decode/bits.rs`, transcribed
from the ARM ARM):

- **`adv_simd_expand_imm`** (ARM ARM `AdvSIMDExpandImm`): expands the
  `abc:defgh` + `cmode` modified-immediate into the 64-bit `MOVI`/`MVNI` value;
  `MSL` shift (`ShiftType::Msl`) appears here for the `cmode==110x` forms.
- **`vfp_expand_imm`** (ARM ARM `VFPExpandImm`): the 8-bit `FMOV #imm` to the
  IEEE-754 value used by `Operand::FpImm(f32)`. FP is *only* a bit-cast +
  formatting concern (`f32::from_bits`); there is **no FP arithmetic** in decode,
  preserving soft-float/wasm/bare-metal portability.
- Vector arrangement (`VectorArrangement`, e.g. `.16b`, `.4s`, `.2d`) is carried
  orthogonally on `Operand::Reg { arr }`; `size:Q` selects element size and lane
  count. The group decoder derives it inline from the `size`/`Q` fields.
- Indexed elements (`V0.s[2]`) use `Operand::IndexedElement { reg, arr, index, imm }`.

---

## SP-vs-ZR (register 31)

A register field value of `31` means **either** the stack pointer **or** the zero
register, depending on the encoding's role for that operand (per the ARM ARM's
per-operand SP/ZR choice). `fARM64` resolves this as each group decoder builds its
operands and **never leaks raw 31** to callers. The decoder picks use-SP vs use-ZR
per the ARM ARM for that operand, combines it with the operand width, and calls:

```rust
pub const fn gp_register(use_sp: bool, width: RegWidth, n: u8) -> Register
```

Resolution:

| n | width | use_sp | resulting `Register` |
|-|-|-|-|
| 0–30 | 32 | either | `W0`..`W30` |
| 0–30 | 64 | either | `X0`..`X30` |
| 31 | 32 | false | `Wzr` |
| 31 | 32 | true | `Wsp` |
| 31 | 64 | false | `Xzr` |
| 31 | 64 | true | `Sp` |

Rule of thumb for base ISA: the **base register** of a memory operand and the
`Rn`/`Rd` of `ADD/SUB (imm)` and `ADD/SUB (extended reg)` are use-SP; almost
everything else (and any `S`-flag-setting variant's `Rd`) is use-ZR. The
differential comparator treats `sp.d == sp` and `xN.d == xN` as equal (the `.d`
arrangement on an X register is dropped); see
[Token-equality comparator](#token-equality-comparator).

---

## Condition codes

`Condition` is the 4-bit `cond` field (`#[repr(u8)]`, 16 variants):

| cond | name | name | cond | name |
|-|-|-|-|-|
| `0000` | EQ | NE | `0001` | (Z=1 / Z=0) |
| `0010` | CS/HS | CC/LO | `0011` | (C=1 / C=0) |
| `0100` | MI | PL | `0101` | (N=1 / N=0) |
| `0110` | VS | VC | `0111` | (V=1 / V=0) |
| `1000` | HI | LS | `1001` | (unsigned >, ≤) |
| `1010` | GE | LT | `1011` | (signed ≥, <) |
| `1100` | GT | LE | `1101` | (signed >, ≤) |
| `1110` | AL | NV | `1111` | (always; NV == AL) |

- `Condition::name()` emits the ARM-preferred spelling. The differential comparator
  accepts `cs == hs` and `cc == lo` as equivalent; both spellings are treated as
  correct by the harness.
- `NV` (`1111`) behaves as `AL` for conditional branches.
- Conditional instructions whose mnemonic encodes the *inverted* condition
  (`CSET`, `CINC`, etc.) invert the field during alias selection.

---

## Extend and shift types

`ShiftType` (true shifts only — kept separate from extends for type safety):

| value | `ShiftType` |
|-|-|
| `00` | `Lsl` |
| `01` | `Lsr` |
| `10` | `Asr` |
| `11` | `Ror` |
| — | `Msl` (MOVI modified-immediate only) |

`ExtendType` (register-extension family; the `option` field, per the ARM ARM
`DecodeRegExtend` pseudocode in `decode/bits.rs`):

| option | `ExtendType` |
|-|-|
| `000` | `Uxtb` |
| `001` | `Uxth` |
| `010` | `Uxtw` |
| `011` | `Uxtx` (renders `lsl` for X index when no extend needed) |
| `100` | `Sxtb` |
| `101` | `Sxth` |
| `110` | `Sxtw` |
| `111` | `Sxtx` |

---

## Alias / preferred-disassembly rules

Aliases are applied **in code**, per the ARM ARM "alias conditions" for each
encoding (in the relevant `src/decode/*` group decoder). Decode always produces a
canonical `Code`; the alias condition sets the displayed `Mnemonic`, gated by
`FormatterOptions::aliases` (default **on**).

Representative rules (each implemented in its group decoder):

| alias | canonical encoding | condition |
|-|-|-|
| `MOV` | `ORR` (shifted reg) | `Rn==ZR`, `imm6==0`, `shift==LSL` |
| `MOV` | `ORR`/`MOVZ`/`ADD #0` (imm) | per `MoveWidePreferred` / SP form |
| `MVN` | `ORN` | `Rn==ZR` |
| `TST` | `ANDS` | `Rd==ZR` |
| `CMP` | `SUBS` | `Rd==ZR` |
| `CMN` | `ADDS` | `Rd==ZR` |
| `NEG` / `NEGS` | `SUB` / `SUBS` | `Rn==ZR` |
| `MUL` / `MNEG` | `MADD` / `MSUB` | `Ra==ZR` |
| `SMULL`/`UMULL` | `SMADDL`/`UMADDL` | `Ra==ZR` |
| `LSL` / `LSR` | `UBFM` | `imms+1==immr` / `imms==63/31` |
| `ASR` | `SBFM` | `imms==63/31` |
| `UBFX`/`SBFX` | `UBFM`/`SBFM` | `bfx_preferred` (ARM ARM `BFXPreferred`) |
| `UXTB/H`, `SXTB/H/W` | `UBFM`/`SBFM` | `immr==0`, `imms∈{7,15,31}` |
| `ROR` | `EXTR` | `Rn==Rm` |
| `NOP`/`YIELD`/`WFE`/... | `HINT` | named `CRm:op2` |
| `CSET`/`CSETM` | `CSINC`/`CSINV` | `Rn==Rm==ZR`, inverted cond |

The `EndOfInstruction` status is *not* an error for HINT encodings; the entry
function (`decode::decode_into`) accepts it as the ARM ARM's HINT special case.

---

## Where the extension groups slot into the tree

Extensions are **gated, not separately authored**. The hand-written decode tree
contains the extension encodings inline; cargo features decide what is compiled and
the runtime `FeatureSet` decides what is accepted.

- **SVE / SVE2** (`op0 == 0b0010`): a structurally distinct scalable-vector
  sub-tree under `src/decode/sve/` (dispatched by `word<31:29>` into
  `sve_int`/`sve_perm`/`sve_fp`/`sve_mem`), compiled under the `sve` cargo feature
  and routed only when `Feature::Sve` is accepted. Predicated forms carry
  `pred: Option<PredQual>` (`/z`, `/m`) and use `Z`/`P` registers with truncated
  arrangement suffixes (e.g. `z29.q`, `p4/m`). SVE addressing uses
  `Operand::SveMem` (with `SveMemMode` and the `MUL VL` decorator).
- **SME** (`op0 == 0b0000`, `word<31>==1`): the Scalable Matrix Extension shares
  the reserved group with `UDF`; routed from `decode_reserved` to
  `src/decode/sme/` under the `sme` cargo feature when `Feature::Sme` is accepted.
  ZA operands use `Operand::SmeTile`/`Operand::SmeTileSlice`. Example:
  `str za[w12, #0x6], [x8, #0x6, mul vl]`. `SMSTART`/`SMSTOP` are handled in
  `branch_sys.rs` (they are `MSR (immediate)` PSTATE encodings).
- **FP/NEON, FP16, BF16, LSE, PAuth, MTE, crypto** live *within* the base groups
  (mostly the `x111` and load/store spaces; crypto in `simd_fp/crypto.rs`) and are
  gated by an innermost `Code::feature()` test after the structural match.
- `FeatureSet` is **two u64 words** (`features0` for decode-time admission,
  `features1` for pcode-time behaviour), kept separate because the ARM ARM treats
  those questions independently.

---

## Differential-testing plan and oracles

Correctness is defined against the **ARM ARM**. The differential harness
cross-checks `fARM64`'s default-formatter output against **multiple independent
oracles**, with no single tool treated as ground truth:

- **LLVM** — `llvm-mc -disassemble` / `llvm-objdump`.
- **GNU binutils** — `objdump -d`.
- **Binary Ninja `arch-arm64`** — the `test_cases.txt` corpus, used **only as a
  development guide**, read locally and never shipped, with a documented
  [allow-list](#oracle-corpora-and-the-divergence-allow-list) for the places where
  binja intentionally diverges from the spec.

Disagreement among oracles is resolved by the ARM ARM. The corpus formats below
describe how `fARM64` ingests and compares against the binja corpus specifically
(it is the richest single file, so it is convenient as a coverage guide).

### Binja corpus format (development oracle)

The development corpus is `refs/arch-arm64-master/disassembler/test_cases.txt`
(~42k cases, SVE/SME-dominated) — Binary Ninja's own `arch-arm64` generated output.
It is read **locally only**, is **not authoritative**, and is **never shipped**.

File grammar:

- Lines beginning `// ` are comments. They come in two flavors used by the harness:
  an **encoding-group label + bit template**, e.g.
  `// REVD_Z_P_Z_ 00000101|size=00|101110100|Pg=xxx|Zn=xxxxx|Zd=xxxxx`, and a
  **syntax template** `// REVD <Zd>.Q,<Pg>/M,<Zn>.Q`. The group label is tracked so
  failures can be bucketed (e.g. `REVD_Z_P_Z_: 3/8 failed`).
- Data lines: 8 hex digits, a single space (`line[8] == ' '`), then the expected
  normalized disassembly. The 8 hex digits are the instruction word
  **big-endian-as-text** but the bytes are interpreted **little-endian** when
  decoding: `insword = u32::from_str_radix(&line[0..8], 16)`, then packed `<I`.
  Example: `052E93FD revd z29.q, p4/m, z29.q`.
- All cases are decoded at a fixed `ADDRESS_TEST = 0x8000000000000004` so
  PC-relative targets (`ADR`/`ADRP`/`B`/`B.cond`) line up with the corpus's
  pre-resolved addresses.

### Normalization

Both actual and expected strings are normalized before comparison:

1. `strip()`, then collapse runs of whitespace to a single space.
2. Strip a trailing ` //...` comment.
3. Expand SVE register ranges `{z14.s-z17.s}` → `{z14.s, z15.s, z16.s, z17.s}`
   (3- and 4-register runs, mod-32 wrap).
4. Remove spaces inside `{ ... }` lists.
5. Strip leading hex zeros: `0x00000000071eb000` → `0x71eb000`.
6. Convert decimal immediates to hex: `#6` → `#0x6` (both `#\d+[,\]]` and trailing
   `#\d+$` forms).
7. Normalize float immediates `#-3.375000000000000000e+00` → `#-3.375`, and
   `0.000000`/`0.000` → `0.0`.
8. Lowercase everything.

### Token-equality comparator

After normalization, an exact string match is the fast path. On a miss, compare
token-by-token (`split()` on whitespace; equal token count + equal mnemonic
required), with these equivalences:

- Strip leading/trailing "trash" chars `#{}[]!,` symmetrically.
- `xN.d == xN` and `sp.d == sp` (the `.d` arrangement on an X register is dropped).
- `cs == hs`, `cc == lo` (condition spellings).
- Numeric equality across bases: `0xff == 255`; and signed/unsigned hex
  equivalence at the byte/word/dword widths — `0xbc == -68` (len-4 hex),
  `0xfffffffe == -2` (len-10 hex), `0xffff...fffe == -2` (len-18 hex), via the
  `<b>/<i>/<q>` width-aware reinterpretation.

### Oracle corpora and the divergence allow-list

When the binja corpus disagrees with `fARM64`'s spec-faithful output, the entry is
recorded on an explicit allow-list rather than counted as a failure. These are
cases where binja (or its source) diverges from the ARM ARM; `fARM64` follows the
spec and documents the difference. The allow-list includes:

- `DecodeBitMasks` `tmask` (binja `tmask == wmask` at `pcode.c:88`; `fARM64`
  computes the spec value — see [DecodeBitMasks](#decodebitmasks-logical-immediates)).
- `dgh` vs `hint ...` (named hint vs raw `HINT`).
- `cfinv` / `sb` / `xaflag` / `msr ssbs` / `msr pan` vs `msr ...`.
- `mov ...` vs `dupm ...`.
- `at `/`dc `/`cfp ` vs `sys ...`; `tlbi...` vs `sys ...`.
- `cmpp ...` vs `subps ...`.
- any `axflag...`.

Each allow-listed entry should be cross-confirmed against LLVM/binutils so the
`fARM64` side is verified spec-correct, not merely "different".

### Test harness & oracles

- **Spec unit tests** (the primary correctness source): per-encoding cases
  transcribed directly from the ARM ARM (ADR/ADRP reconstruction, MOVZ/MOVN/MOVK
  `hw`-undef for `sf==0`, bitmask-immediate edge/reserved cases, all addressing
  modes, all 16 conditions, sysreg packing). The pseudocode helpers
  (`decode_bit_masks` — computing the **spec-correct `tmask`** — `move_wide_preferred`,
  `bfx_preferred`, `adv_simd_expand_imm`, `vfp_expand_imm`) get direct value tests
  pinned to ARM ARM expected results.
- **Differential harness** (`tests/golden.rs`): iterates the binja corpus, decodes
  with the default `FmtFormatter` (default `FormatterOptions`, aliases on),
  normalizes (`tests/common/mod.rs`), and compares, bucketing by encoding group.
  It is `#[ignore]`d and currently reports a coverage/parity summary rather than
  gating; mismatches are dumped to `target/golden-mismatches.txt`. The measured
  result is **99.35% coverage at 99.78% parity**; the residual mismatches are the
  documented spec-vs-binja divergences. See [VALIDATION.md](./VALIDATION.md) for
  the full results and per-group tables.
- **LLVM cross-check** (`tests/llvm_diff.rs`): re-disassembles sampled words with
  `llvm-mc --disassemble` to confirm `fARM64` is spec-correct where it diverges
  from the binja corpus; skips cleanly if `llvm-mc` is absent.
- **Static assertions** (`lib.rs`): `size_of::<Operand>() <= 16`,
  `size_of::<Instruction>() <= 112`, and the `Copy` witnesses for both. Decode is
  total and panic-free for all 2^32 words, always advancing exactly 4 bytes; the
  default decode + `FmtFormatter` + `BufSink` path is zero-alloc by construction
  (no `alloc` dependency).

---

## System registers

`MSR`/`MRS` carry a 15-bit key packed as
`op0<<14 | op1<<11 | CRn<<7 | CRm<<3 | op2`. `SystemReg(u16)` wraps this key:

```rust
impl SystemReg {
    pub const fn from_fields(op0: u8, op1: u8, crn: u8, crm: u8, op2: u8) -> Self;
    pub fn name(self) -> Option<&'static str>;   // sorted-slice binary search
    pub const fn packed(self) -> u16;
}
```

`name()` does a binary search over a sorted `&'static (u16, &'static str)` table.
Unknown registers return `None`; the formatter then emits the generic
`S<op0>_<op1>_c<CRn>_c<CRm>_<op2>` syntax (forward-compatible with new sysregs).
The binja corpus's `msr ssbs`/`msr s0_...` discrepancies are on the
[divergence allow-list](#oracle-corpora-and-the-divergence-allow-list).

---

## Provenance and licensing

`fARM64`'s decoder is an **original implementation** written by hand from the *Arm
Architecture Reference Manual*. It is **not** a derivative of Binary Ninja's
`arch-arm64` or any other disassembler. The crate is licensed `MIT`
(the Rust default); there is no required-attribution obligation to any third party.

Binary Ninja's `test_cases.txt` corpus is used **only as one differential-testing
oracle during development** — read locally, never shipped, and not authoritative.
Where binja diverges from the spec (e.g. the `DecodeBitMasks` `tmask == wmask`
shortcut at `pcode.c:88`), `fARM64` follows the ARM ARM and records the intentional
divergence on the
[allow-list](#oracle-corpora-and-the-divergence-allow-list).
