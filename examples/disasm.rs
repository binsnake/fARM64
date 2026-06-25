//! Tiny CLI for manual spot-checks of fARM64 decode output.
//!
//! Reads 8-hex-digit instruction WORD values (the big-endian hex of the u32, as
//! in the corpus, e.g. `11000420`) either from CLI args or, if none are given,
//! one per line from stdin. For each it prints:
//!
//! ```text
//! WWWWWWWW\t<fARM64 disassembly>
//! ```
//!
//! Decoding uses the default [`DecoderOptions`]; formatting uses the zero-alloc
//! [`BufSink`] + [`FmtFormatter`] path (so this example builds on the default
//! `no_std`-targeting feature set too — it only needs `std` for IO).
//!
//! Usage:
//! ```text
//! cargo run --example disasm 11000420 d503201f
//! echo 11000420 | cargo run --example disasm
//! ```

use std::io::{self, BufRead, Write};

use fARM64::format::{BufSink, FmtFormatter, Formatter};
use fARM64::{Decoder, DecoderOptions};

/// Address each word is decoded at (matches the corpus anchor so PC-relative
/// targets line up with the golden test).
const ADDRESS: u64 = 0x8000_0000_0000_0004;

/// Decode one word and write `WWWWWWWW\t<text>` to `out`.
fn emit_word<W: Write>(token: &str, out: &mut W) -> io::Result<()> {
    let token = token.trim();
    if token.is_empty() {
        return Ok(());
    }
    // Accept an optional `0x` prefix and surrounding whitespace.
    let hex = token.trim_start_matches("0x").trim_start_matches("0X");
    let word = match u32::from_str_radix(hex, 16) {
        Ok(w) => w,
        Err(_) => {
            writeln!(out, "{token}\t<invalid hex>")?;
            return Ok(());
        }
    };

    // The per-group decoders are currently `todo!()` stubs that panic. Catch a
    // panic so a single unimplemented group does not abort the whole CLI run;
    // such words simply render as `<unimplemented>`. (Once decoders land this is
    // a transparent no-op — real instructions never panic.)
    let rendered = std::panic::catch_unwind(|| {
        let bytes = word.to_le_bytes();
        let mut dec = Decoder::new(&bytes, ADDRESS, DecoderOptions::default());
        let insn = dec.decode();

        // Zero-alloc format into a fixed stack buffer.
        let mut buf = [0u8; 256];
        let mut sink = BufSink::new(&mut buf);
        FmtFormatter::new().format(&insn, &mut sink);
        sink.as_str().to_string()
    })
    .unwrap_or_else(|_| String::from("<unimplemented>"));

    writeln!(out, "{word:08X}\t{rendered}")?;
    Ok(())
}

fn main() -> io::Result<()> {
    // Suppress the default panic backtrace from the (currently stubbed)
    // `todo!()` group decoders; each decode is wrapped in `catch_unwind`.
    std::panic::set_hook(Box::new(|_| {}));

    let stdout = io::stdout();
    let mut out = stdout.lock();

    let args: Vec<String> = std::env::args().skip(1).collect();
    if !args.is_empty() {
        for a in &args {
            emit_word(a, &mut out)?;
        }
    } else {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            let line = line?;
            // Allow several whitespace-separated words per line.
            for token in line.split_whitespace() {
                emit_word(token, &mut out)?;
            }
        }
    }
    out.flush()
}
