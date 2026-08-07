"""HelixLang interactive debugger.

Features:
- Bytecode-level breakpoints (by instruction offset)
- Source-level breakpoints (by line / codon index)
- Single-stepping (step into / step over / step out)
- Variable watches (protein concentrations, energy, GRN state)
- Call stack inspection
- Real-time disassembly display
- Conditional breakpoints
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from helixlang.ast_nodes import Program
from helixlang.bytecode import Chunk
from helixlang.codon_table import OP_OPERAND_BYTES, Op
from helixlang.vm import CellVM


@dataclass(slots=True)
class Breakpoint:
    """Breakpoint: settable by bytecode offset / source line / codon index, with optional condition expression."""

    offset: int | None = None       # bytecode offset
    line: int | None = None         # source line
    codon_index: int | None = None  # codon index
    condition: str | None = None    # condition expression, e.g. "energy < 50"
    enabled: bool = True
    hit_count: int = 0


@dataclass(slots=True)
class Watch:
    """Variable watch: evaluates a given expression and caches the most recent value."""

    name: str
    expression: str          # expression to evaluate, e.g. "energy" / "protein.3" / "grn.pigment"
    last_value: Any = None


@dataclass(slots=True)
class DebugState:
    """Debug state snapshot: the full observable VM state at a given moment."""

    ip: int                  # current instruction pointer
    op: str                  # current opcode name
    stack: list              # current stack
    cell_state: dict         # cell state
    grn_state: dict          # GRN state
    gene: str | None         # current gene
    line: int                # source line
    codon_index: int         # codon index


# Conditional breakpoint expression: variable name  comparison operator  numeric value
_COND_RE = re.compile(
    r'^\s*([A-Za-z_][\w]*(?:\.[A-Za-z_]\w*)?)\s*'
    r'(>=|<=|==|!=|>|<)\s*'
    r'(-?\d+(?:\.\d+)?)\s*$'
)


class HelixDebugger:
    """HelixLang interactive debugger.

    Implements instruction-level single-stepping by reusing CellVM's internal dispatch logic,
    layering breakpoint / watch / call stack capabilities on top without modifying the VM.
    """

    def __init__(self, vm: CellVM, program: Program):
        self.vm = vm
        self.program = program
        self._breakpoints: list[Breakpoint] = []
        self._watches: list[Watch] = []
        self._last_op: Op | None = None  # most recently executed opcode

    # ------------------------------------------------------------------ #
    # Startup
    # ------------------------------------------------------------------ #
    def start(self, gene_name: str | None = None) -> None:
        """Start execution: call the given gene (defaults to the first gene) by pushing the initial frame."""
        if gene_name is None:
            if not self.program.genes:
                return
            gene_name = self.program.genes[0].name
        self.vm._call_gene(gene_name)

    # ------------------------------------------------------------------ #
    # Breakpoint management
    # ------------------------------------------------------------------ #
    def set_breakpoint(self, offset: int | None = None, line: int | None = None,
                       codon_index: int | None = None,
                       condition: str | None = None) -> Breakpoint:
        """Set a breakpoint and return the created Breakpoint."""
        bp = Breakpoint(offset=offset, line=line, codon_index=codon_index,
                        condition=condition)
        self._breakpoints.append(bp)
        return bp

    def remove_breakpoint(self, bp: Breakpoint) -> None:
        """Remove a breakpoint."""
        if bp in self._breakpoints:
            self._breakpoints.remove(bp)

    def enable_breakpoint(self, bp: Breakpoint) -> None:
        """Enable a breakpoint."""
        bp.enabled = True

    def disable_breakpoint(self, bp: Breakpoint) -> None:
        """Disable a breakpoint."""
        bp.enabled = False

    def list_breakpoints(self) -> list[Breakpoint]:
        """List all breakpoints."""
        return list(self._breakpoints)

    def _hit_breakpoint(self) -> Breakpoint | None:
        """Check whether the current ip hits an enabled breakpoint (with its condition satisfied).

        On a hit, increment hit_count and return the breakpoint; otherwise return None.
        """
        vm = self.vm
        chunk = vm.chunk
        ip = vm.ip
        cur_line = chunk.lines[ip] if 0 <= ip < len(chunk.lines) else None
        cur_codon = (chunk.codon_indices[ip]
                     if 0 <= ip < len(chunk.codon_indices) else None)
        for bp in self._breakpoints:
            if not bp.enabled:
                continue
            matched = False
            if bp.offset is not None and bp.offset == ip:
                matched = True
            elif bp.line is not None and cur_line is not None and bp.line == cur_line:
                matched = True
            elif (bp.codon_index is not None and cur_codon is not None
                  and bp.codon_index == cur_codon):
                matched = True
            if not matched:
                continue
            if bp.condition is not None and not self._eval_condition(bp.condition):
                continue
            bp.hit_count += 1
            return bp
        return None

    # ------------------------------------------------------------------ #
    # Variable watches
    # ------------------------------------------------------------------ #
    def add_watch(self, name: str, expression: str) -> Watch:
        """Add a variable watch."""
        w = Watch(name=name, expression=expression)
        self._watches.append(w)
        return w

    def get_watches(self) -> list[Watch]:
        """Get the current values of all watched variables (updates last_value)."""
        for w in self._watches:
            w.last_value = self._eval_var(w.expression)
        return list(self._watches)

    # ------------------------------------------------------------------ #
    # Single-stepping
    # ------------------------------------------------------------------ #
    def _step_one(self) -> Op | None:
        """Execute one instruction and return that Op; return None if there is nothing executable (frames empty / out of bounds)."""
        vm = self.vm
        if not vm.frames:
            return None
        if vm.ip >= len(vm.chunk.code):
            # The current frame has executed to the end of the chunk, pop it
            vm.frames.pop()
            if vm.frames:
                vm.ip = vm.frames[-1].return_ip
            return None
        op_byte = vm.chunk.code[vm.ip]
        vm.ip += 1
        try:
            op = Op(op_byte)
        except ValueError:
            # Unknown opcode: skip it
            return None
        self._last_op = op
        vm._dispatch(op)
        return op

    def step(self) -> DebugState:
        """Single-step one instruction (step into) and return the current state."""
        self._step_one()
        return self.get_state()

    def step_over(self) -> DebugState:
        """Step over: execute one instruction; if it enters CALL_GENE, keep executing until back at the original frame depth."""
        vm = self.vm
        start_depth = len(vm.frames)
        self._step_one()
        while vm.frames and len(vm.frames) > start_depth:
            self._step_one()
        return self.get_state()

    def step_out(self) -> DebugState:
        """Step out: keep executing until returning from the current frame (frame depth decreases or frames become empty)."""
        vm = self.vm
        start_depth = len(vm.frames)
        if start_depth == 0:
            return self.get_state()
        while vm.frames and len(vm.frames) >= start_depth:
            self._step_one()
        return self.get_state()

    def continue_run(self) -> DebugState | None:
        """Continue execution until the next breakpoint or HALT is hit.

        Returns a DebugState on a breakpoint hit; returns None when execution reaches HALT (frames empty).
        """
        vm = self.vm
        if not vm.frames:
            return None
        # Take one step first to leave any breakpoint that may already be hit
        self._step_one()
        while vm.frames:
            bp = self._hit_breakpoint()
            if bp is not None:
                return self.get_state()
            self._step_one()
        return None

    # ------------------------------------------------------------------ #
    # State queries
    # ------------------------------------------------------------------ #
    def inspect(self, expr: str) -> Any:
        """Inspect the current value of an expression."""
        return self._eval_var(expr)

    def get_call_stack(self) -> list[dict]:
        """Return the call stack (innermost frame first)."""
        vm = self.vm
        n = len(vm.frames)
        result: list[dict] = []
        for i in range(n - 1, -1, -1):
            frame = vm.frames[i]
            depth = n - 1 - i
            # The innermost frame's "current ip" is vm.ip; the other frames use the next inner frame's return_ip
            cur_ip = vm.ip if i == n - 1 else vm.frames[i + 1].return_ip
            gene = frame.gene_name
            if gene in (None, "", "<call>"):
                resolved = self._gene_at_offset(cur_ip)
                if resolved is not None:
                    gene = resolved
            result.append({
                "depth": depth,
                "gene": gene,
                "ip": cur_ip,
                "return_ip": frame.return_ip,
            })
        return result

    def get_state(self) -> DebugState:
        """Return the current state snapshot."""
        vm = self.vm
        chunk = vm.chunk
        ip = vm.ip
        # Current opcode name
        if not vm.frames and self._last_op is not None:
            op_name = self._last_op.name
        elif 0 <= ip < len(chunk.code):
            try:
                op_name = Op(chunk.code[ip]).name
            except ValueError:
                op_name = f"<unknown 0x{chunk.code[ip]:02X}>"
        else:
            op_name = "<none>"
        # Line / codon
        line = chunk.lines[ip] if 0 <= ip < len(chunk.lines) else 0
        codon = (chunk.codon_indices[ip]
                 if 0 <= ip < len(chunk.codon_indices) else -1)
        # Current gene
        gene = vm.frames[-1].gene_name if vm.frames else None
        if vm.frames and gene in (None, "", "<call>"):
            resolved = self._gene_at_offset(ip)
            if resolved is not None:
                gene = resolved
        # Cell state
        cell = vm.cell
        cell_state = {
            "x": cell.x,
            "y": cell.y,
            "energy": cell.energy,
            "alive": cell.alive,
            "age": cell.age,
            "divisions": cell.divisions,
            "color": cell.color,
            "proteins": dict(cell.proteins),
        }
        # GRN state
        grn_state = {n: nd.level for n, nd in vm.grn.nodes.items()}
        return DebugState(
            ip=ip, op=op_name, stack=list(vm.stack),
            cell_state=cell_state, grn_state=grn_state,
            gene=gene, line=line, codon_index=codon,
        )

    # ------------------------------------------------------------------ #
    # Expression evaluation
    # ------------------------------------------------------------------ #
    def _eval_condition(self, condition: str) -> bool:
        """Evaluate a conditional breakpoint expression. If it cannot be parsed, treat it as always true."""
        m = _COND_RE.match(condition)
        if not m:
            return True
        var_name, op, val_str = m.groups()
        cur = self._eval_var(var_name)
        if cur is None:
            return False
        try:
            cur_n = float(cur)
        except (TypeError, ValueError):
            return False
        target = float(val_str)
        if op == '>':
            return cur_n > target
        if op == '<':
            return cur_n < target
        if op == '>=':
            return cur_n >= target
        if op == '<=':
            return cur_n <= target
        if op == '==':
            return cur_n == target
        if op == '!=':
            return cur_n != target
        return False

    def _eval_var(self, expr: str) -> Any:
        """Evaluate a variable expression, supporting energy / x / y / age / protein.X / grn.X / slot.N, etc."""
        vm = self.vm
        simple = {
            'energy': vm.cell.energy,
            'x': vm.cell.x,
            'y': vm.cell.y,
            'age': vm.cell.age,
            'alive': vm.cell.alive,
            'tick': vm.tick,
            'ip': vm.ip,
            'divisions': vm.cell.divisions,
        }
        if expr in simple:
            return simple[expr]
        if expr.startswith('protein.'):
            key_str = expr[len('protein.'):]
            try:
                key: int | str = int(key_str)
            except ValueError:
                key = key_str
            return vm.cell.proteins.get(key, 0.0)
        if expr.startswith('grn.'):
            name = expr[len('grn.'):]
            node = vm.grn.nodes.get(name)
            return node.level if node is not None else 0.0
        if expr.startswith('slot.'):
            try:
                idx = int(expr[len('slot.'):])
            except ValueError:
                return None
            if 0 <= idx < len(vm.cell.slots):
                return vm.cell.slots[idx]
            return None
        return None

    def _gene_at_offset(self, ip: int) -> str | None:
        """Find the gene name containing the given ip (searching by the gene_offsets ranges)."""
        chunk = self.vm.chunk
        if not chunk.gene_offsets:
            return None
        result: str | None = None
        for name, off in sorted(chunk.gene_offsets.items(),
                                key=lambda kv: kv[1]):
            if off <= ip:
                result = name
            else:
                break
        return result


# ---------------------------------------------------------------------- #
# Formatting output
# ---------------------------------------------------------------------- #
def format_state(state: DebugState) -> str:
    """Format the debug state as a readable string."""
    lines: list[str] = []
    lines.append(
        f"ip={state.ip}  op={state.op}  gene={state.gene}  "
        f"line={state.line}  codon={state.codon_index}"
    )
    lines.append(f"stack: {state.stack}")
    cs = state.cell_state
    lines.append(
        f"cell: pos=({cs['x']},{cs['y']}) energy={cs['energy']} "
        f"alive={cs['alive']} age={cs['age']} "
        f"divisions={cs['divisions']} color={cs['color']}"
    )
    lines.append(f"  proteins: {cs['proteins']}")
    if state.grn_state:
        grn_str = ", ".join(
            f"{k}={v:.3f}" for k, v in state.grn_state.items())
    else:
        grn_str = "(empty)"
    lines.append(f"grn: {grn_str}")
    return '\n'.join(lines)


def format_disasm_around(chunk: Chunk, ip: int, context: int = 5) -> str:
    """Format the disassembly around the current instruction.

    Marks the line of the ip instruction with ``>>>`` and shows ``context`` instructions above and below it.
    """
    # First scan all instruction start offsets
    starts: list[int] = []
    cur = 0
    while cur < len(chunk.code):
        starts.append(cur)
        try:
            op = Op(chunk.code[cur])
            nbytes = OP_OPERAND_BYTES[op]
        except ValueError:
            nbytes = 0
        cur += 1 + nbytes
    if not starts:
        return "<empty chunk>"
    # Find the nearest instruction start <= ip
    idx = 0
    for i, s in enumerate(starts):
        if s <= ip:
            idx = i
        else:
            break
    lo = max(0, idx - context)
    hi = min(len(starts), idx + context + 1)
    out: list[str] = []
    for i in range(lo, hi):
        s = starts[i]
        try:
            op = Op(chunk.code[s])
        except ValueError:
            out.append(f"  {s:04d}  <unknown 0x{chunk.code[s]:02X}>")
            continue
        nbytes = OP_OPERAND_BYTES[op]
        args = list(chunk.code[s + 1: s + 1 + nbytes])
        args_str = ' '.join(f'{a:02X}' for a in args)
        codon_idx = (chunk.codon_indices[s]
                     if s < len(chunk.codon_indices) else -1)
        line = chunk.lines[s] if s < len(chunk.lines) else 0
        loc_parts: list[str] = []
        if codon_idx >= 0:
            loc_parts.append(f"codon #{codon_idx}")
        if line:
            loc_parts.append(f"line {line}")
        loc = ' '.join(loc_parts)
        marker = ">>>" if i == idx else "   "
        out.append(
            f"{marker} {s:04d}  {op.name:<22} {args_str:<8}  ; {loc}".rstrip()
        )
    return '\n'.join(out)
