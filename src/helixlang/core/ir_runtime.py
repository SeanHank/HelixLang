"""IRRuntime: portable interpreter that executes the typed IR directly.

A second, independent CPU runtime alongside ``CellVM``.  It drives the same
cell model, GRN, field and :class:`~helixlang.core.vm.BioInstructionDispatcher`
effect semantics, but reads the program from the typed IR function stream
instead of a bytecode :class:`~helixlang.core.bytecode.Chunk`.

Design: the IR runtime keeps ``ip`` as an *instruction index* into a
linearised view of every gene's ORF (``Program -> functions[*].instrs``).
When ``_execute_pending`` picks the next :class:`IRInst` it:

1. records it as ``_current_inst`` and turns its operand into the byte view the
   dispatcher expects (``_read_u8`` / ``_read_u16``);
2. special-cases the three instructions whose bytecode operand *semantics*
   differ in the IR (``OP_PUSH_CONST`` carries the literal, ``OP_USE_PLUGIN``
   carries the ``(plugin, flags)`` pair, ``OP_CALL_GENE`` carries the resolved
   target gene name);
3. delegates every other opcode to ``BioInstructionDispatcher`` unchanged.

Because the effect semantics are shared verbatim, the IR runtime is bit-for-bit
equivalent to ``CellVM`` on the classic path (asserted by tests and validation
benchmarks).
"""
from __future__ import annotations

from typing import Any

from helixlang.core.codon_table import Op
from helixlang.core.ir import IRInst, IRProgram
from helixlang.core.vm import CellVM, Frame


class IRRuntime(CellVM):
    """Portable IR interpreter.  Drop-in for ``CellVM(chunk, program)`` on the
    classic (non-central-dogma) execution path: ``run(max_ticks) -> trace``."""

    def __init__(self, ir: IRProgram, program: Any, *, registry: Any = None):
        # do not call super().__init__ with a chunk: bypass the bytecode setup
        # but keep all cell/GRN/field/frames/trace infrastructure.
        from helixlang.plugins.runtime.cell import Cell
        from helixlang.plugins.runtime.grn import GRN

        self.ir = ir
        self.program = program
        self._registry = registry
        self.ip = 0
        self.stack: list = []
        self.frames: list[Frame] = []
        self.cell = Cell()
        self.grn = GRN()
        self.lsystems: dict[str, Any] = {}
        self.field: Any = None
        self.tick = 0
        self.debug = False
        self.trace: list[dict] = []
        self.daughters: list[Any] = []
        self._daughter_counter = 0
        self._gene_dna: dict[str, str] = {}
        self._gene_mrna: dict[str, float] = {}
        self._promoter_strengths: dict[str, float] = {}
        self._chromatin_modifier: dict[str, float] = {}
        self._crispr_edits: list[dict] = []
        self._epigenetic_marks: list[dict] = []
        self._evolution_history: list[dict] = []
        self._gem_dirty = False
        self._gem_gpr_map: dict[str, list[str]] = {}
        self._enzyme_kcat: dict[str, float] = {}
        self._metabolic_model = None
        self._growth_rate_gem = 0.0
        self._regulation_events: list[dict] = []
        self._binding_events: list[dict] = []
        self._signal_emissions = 0

        # Linearised instruction stream + gene start offsets (instr-index space)
        self._flat: list[IRInst] = []
        self._ir_offsets: dict[str, int] = {}
        for fn in ir.functions:
            self._ir_offsets[fn.name] = len(self._flat)
            self._flat.extend(fn.instrs)

        import random
        seed = int(self.program.config.sim.get("seed", "0"))
        self._rng = random.Random(seed)

        from helixlang.core.performance import SnapshotDownsampler
        self._snapshot_downsampler = SnapshotDownsampler()
        self._init_subsystems()
        from helixlang.core.vm import BioInstructionDispatcher
        self._dispatcher = BioInstructionDispatcher(self)
        for d in program.use_directives:
            self._use_plugin(d.plugin, tuple(d.flags))

        self._current_inst: IRInst | None = None
        self._current_operand_bytes: list[int] = []

    # -------- IR execution --------
    def _call_gene(self, name: str) -> None:
        off = self._ir_offsets.get(name)
        if off is None:
            return
        if len(self.frames) >= 256:
            return
        self.frames.append(Frame(return_ip=self.ip, gene_name=name))
        self.ip = off

    def _execute_pending(self) -> None:
        """Execute IR instructions until frames are empty or quota exhausted."""
        quota = self.program.config.ops_per_tick
        while self.frames and quota > 0:
            if len(self.frames) > 256:
                self.frames.clear()
                break
            if self.ip >= len(self._flat):
                self.frames.pop()
                if self.frames:
                    self.ip = self.frames[-1].return_ip
                break
            inst = self._flat[self.ip]
            self.ip += 1
            self._current_inst = inst
            self._current_operand_bytes = _operand_bytes(inst)
            op = inst.opcode
            if self.debug:
                print(f"[tick={self.tick} ip={self.ip - 1}] {op.name} "
                      f"stack={self.stack}")
            self._dispatch(op)
            quota -= 1

    def _dispatch(self, op: Op) -> None:
        inst = self._current_inst
        if inst is None:
            return
        if op is Op.OP_PUSH_CONST:
            self.stack.append(inst.operand)
            return
        if op is Op.OP_USE_PLUGIN:
            if isinstance(inst.operand, tuple) and len(inst.operand) == 2:
                self._use_plugin(str(inst.operand[0]),
                                 tuple(inst.operand[1]))
            return
        if op is Op.OP_CALL_GENE:
            target = str(inst.operand)
            off = self._ir_offsets.get(target)
            if off is not None:
                self.frames.append(Frame(return_ip=self.ip, gene_name="<call>"))
                self.ip = off
            return
        self._dispatcher.dispatch(op)

    # -------- operand byte view (shared dispatcher compatibility) --------
    def _read_u8(self) -> int:
        if self._current_operand_bytes:
            return self._current_operand_bytes.pop(0)
        return 0

    def _read_u16(self) -> int:
        hi = self._read_u8()
        lo = self._read_u8()
        return (hi << 8) | lo


def _operand_bytes(inst: IRInst) -> list[int]:
    """The u8/u16 byte view of an IR operand (for the shared dispatcher)."""
    operand = inst.operand
    if isinstance(operand, int):
        if OP_OPERAND_BYTES_NCONST(inst.opcode) >= 2:
            return [(operand >> 8) & 0xFF, operand & 0xFF]
        return [operand & 0xFF]
    return []


def OP_OPERAND_BYTES_NCONST(op: Op) -> int:
    from helixlang.core.codon_table import OP_OPERAND_BYTES

    return OP_OPERAND_BYTES[op]
