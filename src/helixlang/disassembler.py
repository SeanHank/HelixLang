"""Disassembler: Chunk -> human-readable string."""
from __future__ import annotations

from helixlang.bytecode import Chunk
from helixlang.codon_table import OP_OPERAND_BYTES, Op


def disassemble(chunk: Chunk, name: str = "HelixLang Chunk") -> str:
    """Disassemble the entire chunk."""
    out: list[str] = [f"=== {name} ==="]

    # Gene offset table
    if chunk.gene_offsets:
        out.append("--- Gene Offsets ---")
        for gname, off in chunk.gene_offsets.items():
            out.append(f"  {gname:<20} @ {off:#06x} ({off})")

    out.append("--- Code ---")
    ip = 0
    while ip < len(chunk.code):
        op_byte = chunk.code[ip]
        try:
            op = Op(op_byte)
        except ValueError:
            out.append(f"  {ip:04d}  <unknown 0x{op_byte:02X}>")
            ip += 1
            continue
        nbytes = OP_OPERAND_BYTES[op]
        args = list(chunk.code[ip + 1:ip + 1 + nbytes])
        codon_idx = (chunk.codon_indices[ip]
                     if ip < len(chunk.codon_indices) else -1)
        line = chunk.lines[ip] if ip < len(chunk.lines) else 0
        codon_str = f"codon #{codon_idx}" if codon_idx >= 0 else ""
        line_str = f"line {line}" if line else ""
        loc = " ".join(s for s in (codon_str, line_str) if s)
        args_str = ' '.join(f'{a:02X}' for a in args)
        out.append(
            f"  {ip:04d}  {op.name:<22} {args_str:<8}  ; {loc}".rstrip())
        ip += 1 + nbytes

    if chunk.constants:
        out.append("--- Constants ---")
        for i, c in enumerate(chunk.constants):
            out.append(f"  [{i}] {c!r}")

    return '\n'.join(out)
