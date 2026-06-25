// Included into `simd_fp.rs` — Advanced SIMD cryptographic encoders.
//
// Inverse of `crate::decode::simd_fp::crypto`. The eight encoding classes are
// reconstructed from their fixed signatures plus the register operands read back
// from the structured operand list.

mod crypto {
    use super::*;

    /// Encode a crypto instruction. Returns `Ok(None)` if `code` is not a crypto
    /// code.
    pub(super) fn encode(insn: &Instruction, code: Code) -> Result<Option<u32>, EncodeError> {
        use Code::*;
        let w = match code {
            // cryptoaes: 0100_1110_00_10100_opcode_10 Rn Rd.
            AdvAese | AdvAesd | AdvAesmc | AdvAesimc => {
                let opcode = match code {
                    AdvAese => 4u32,
                    AdvAesd => 5,
                    AdvAesmc => 6,
                    _ => 7,
                };
                let rn = reg_num(insn, 1)?;
                let rd = reg_num(insn, 0)?;
                (0x4E << 24) | (0b10100 << 17) | (opcode << 12) | (0b10 << 10) | (rn << 5) | rd
            }
            // cryptosha3: 0101_1110_00 0 Rm 0 opcode 00 Rn Rd (opcode word<14:12>).
            AdvSha1c | AdvSha1p | AdvSha1m | AdvSha1su0 | AdvSha256h | AdvSha256h2
            | AdvSha256su1 => {
                let opcode = match code {
                    AdvSha1c => 0u32,
                    AdvSha1p => 1,
                    AdvSha1m => 2,
                    AdvSha1su0 => 3,
                    AdvSha256h => 4,
                    AdvSha256h2 => 5,
                    _ => 6, // AdvSha256su1
                };
                let rm = reg_num(insn, 2)?;
                let rn = reg_num(insn, 1)?;
                let rd = reg_num(insn, 0)?;
                (0x5E << 24) | (rm << 16) | (opcode << 12) | (rn << 5) | rd
            }
            // cryptosha2: 0101_1110_00_10100_opcode_10 Rn Rd (opcode word<16:12>).
            AdvSha1h | AdvSha1su1 | AdvSha256su0 => {
                let opcode = match code {
                    AdvSha1h => 0u32,
                    AdvSha1su1 => 1,
                    _ => 2, // AdvSha256su0
                };
                let rn = reg_num(insn, 1)?;
                let rd = reg_num(insn, 0)?;
                (0x5E << 24) | (0b10100 << 17) | (opcode << 12) | (0b10 << 10) | (rn << 5) | rd
            }
            // cryptosha512_3: 1100_1110_011 Rm 1 O opcode Rn Rd.
            //   base: word<31:24>=0xCE, word<23:21>=011, word<15>=1.
            //   O=word<14>, opcode=word<11:10>.
            AdvSha512h | AdvSha512h2 | AdvSha512su1 | AdvRax1 | AdvSm3partw1 | AdvSm3partw2
            | AdvSm4ekey => {
                let (o, opcode) = match code {
                    AdvSha512h => (0u32, 0u32),
                    AdvSha512h2 => (0, 1),
                    AdvSha512su1 => (0, 2),
                    AdvRax1 => (0, 3),
                    AdvSm3partw1 => (1, 0),
                    AdvSm3partw2 => (1, 1),
                    _ => (1, 2), // AdvSm4ekey
                };
                let rm = reg_num(insn, 2)?;
                let rn = reg_num(insn, 1)?;
                let rd = reg_num(insn, 0)?;
                (0xCE << 24)
                    | (0b011 << 21)
                    | (rm << 16)
                    | (1 << 15)
                    | (o << 14)
                    | (opcode << 10)
                    | (rn << 5)
                    | rd
            }
            // cryptosha512_2: 1100_1110_110_00000_10000_0 opcode Rn Rd.
            //   base: word<31:24>=0xCE, word<23:21>=110, word<20:16>=00000,
            //   word<15:12>=1000, word<11:10>=opcode.  (op2=8, op3 bits per dispatch)
            AdvSha512su0 | AdvSm4e => {
                let opcode = if code == AdvSha512su0 { 0u32 } else { 1 };
                let rn = reg_num(insn, 1)?;
                let rd = reg_num(insn, 0)?;
                (0xCE << 24)
                    | (0b110 << 21)
                    | (0b1000 << 12)
                    | (opcode << 10)
                    | (rn << 5)
                    | rd
            }
            // crypto4: 1100_1110_0 Op0 0 Rm 0 Ra Rn Rd.
            //   base word<31:24>=0xCE, word<23>=0, word<22:21>=Op0, word<15>=0.
            AdvEor3 | AdvBcax | AdvSm3ss1 => {
                let op0 = match code {
                    AdvEor3 => 0u32,
                    AdvBcax => 1,
                    _ => 2, // AdvSm3ss1
                };
                let rm = reg_num(insn, 2)?;
                let ra = reg_num(insn, 3)?;
                let rn = reg_num(insn, 1)?;
                let rd = reg_num(insn, 0)?;
                (0xCE << 24) | (op0 << 21) | (rm << 16) | (ra << 10) | (rn << 5) | rd
            }
            // crypto3_imm2 (SM3TT*): 1100_1110_010 Rm 10 imm2 opcode Rn Rd.
            //   base word<31:24>=0xCE, word<23:21>=010, word<15:14>=10,
            //   imm2=word<13:12>, opcode=word<11:10>.
            AdvSm3tt1a | AdvSm3tt1b | AdvSm3tt2a | AdvSm3tt2b => {
                let opcode = match code {
                    AdvSm3tt1a => 0u32,
                    AdvSm3tt1b => 1,
                    AdvSm3tt2a => 2,
                    _ => 3, // AdvSm3tt2b
                };
                let rm = reg_num(insn, 2)?;
                let imm2 = lane_of(insn, 2)? as u32; // Vm.S[imm2]
                let rn = reg_num(insn, 1)?;
                let rd = reg_num(insn, 0)?;
                if imm2 > 3 {
                    return Err(EncodeError::InvalidImmediate);
                }
                (0xCE << 24)
                    | (0b010 << 21)
                    | (rm << 16)
                    | (0b10 << 14)
                    | (imm2 << 12)
                    | (opcode << 10)
                    | (rn << 5)
                    | rd
            }
            // crypto3_imm6 (XAR): 1100_1110_100 Rm imm6 Rn Rd.
            //   base word<31:24>=0xCE, word<23:21>=100, imm6=word<15:10>.
            AdvXar => {
                let rm = reg_num(insn, 2)?;
                let imm6 = imm_u(insn, 3)? as u32;
                let rn = reg_num(insn, 1)?;
                let rd = reg_num(insn, 0)?;
                if imm6 > 0x3f {
                    return Err(EncodeError::InvalidImmediate);
                }
                (0xCE << 24) | (0b100 << 21) | (rm << 16) | (imm6 << 10) | (rn << 5) | rd
            }
            _ => return Ok(None),
        };
        Ok(Some(w))
    }
}
