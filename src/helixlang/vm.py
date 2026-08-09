"""CellVM: stack-based bytecode VM + cell simulator.

Tick loop:
1. GRN update → trigger genes
2. Bytecode execution (limited by the ops_per_tick quota)
3. Morphology update
4. Morphology field → GRN feedback
5. Output snapshot

P0-1.2 central dogma pipeline (enabled when config.use_central_dogma=True):
1. Process bio instructions (#crispr / #evolve / #methylate / #histone / #quorum)
2. Transcription-translation coupling (DNA → mRNA → protein)
3. Morphology update and feedback
4. Snapshot

P2 architecture refactor: instruction dispatch logic (opcode dispatch + bio instruction handlers) extracted into
``BioInstructionDispatcher``, CellVM delegates to it, eliminating the God Class.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from helixlang.ast_nodes import BioInstruction, Program
from helixlang.bytecode import Chunk
from helixlang.cell import FEED_ENERGY_AMOUNT, Cell
from helixlang.codon_table import OP_OPERAND_BYTES, Op
from helixlang.grn import GRN
from helixlang.lsystem import LSystem
from helixlang.population import QUORUM_SIGNAL_THRESHOLD
from helixlang.reaction_diffusion import GrayScott

# ============================================================================
# Runtime opcode semantics (gameplay-unit constant registry)
# ============================================================================
# Values are dimensionless gameplay units unless cited otherwise. The
# behaviors these tune were wired in to replace documented no-op stubs;
# sources are cited at each constant.

#: regulatory-edge weight applied by the runtime ``OP_REGULATE`` opcode.
#: Grounding: the Jacob & Monod (1961) operon model and Ptashne (2004)
#: "Genetic Regulatory Networks" — a cell can dynamically rewire its
#: regulatory graph (e.g. lacI repression) rather than fixing it at
#: compile time. Sign is set by the operand's sign bit (0 = activate,
#: 1 = inhibit).
REGULATE_EDGE_WEIGHT = 1.0

#: expression-level boost applied by one ``OP_BIND`` protein-DNA binding
#: event (transcription-factor activator). Grounding: Berg & von Hippel
#: (1987 PNAS) binding specificity and McClure (1985) — a bound activator
#: raises the target promoter's effective output; binding is protein-limited
#: (no transcription factor available => no binding).
BIND_LEVEL_BOOST = 0.5

#: ``OP_EMIT_MORPHOGEN`` injects ``(id + 1) / EMIT_MORPHOGEN_SCALE`` into the
#: field's V channel (Turing 1952 reaction-diffusion morphogens; Pearson
#: 1993 measured presets). id=0 keeps a non-zero legacy emission.
EMIT_MORPHOGEN_SCALE = 256

#: ``OP_SIGNAL`` releases ``SIGNAL_EMISSION_AMOUNT * (1 + ch)`` (capped at
#: 1.0) of the V channel at the cell position — the quorum-sensing
#: autoinducer pool read by ``#quorum``. Grounding: Miller & Bassler
#: (2001), Xavier & Bassler (2003): AI-2 is secreted and sensed at uM
#: concentrations.
SIGNAL_EMISSION_AMOUNT = 0.25

#: ribosome density (ribosomes/100 nt mRNA) used by the central-dogma
#: pipeline (P0-1.2). Grounding: Ingolia 2009 (in vivo ribosome
#: profiling) — E. coli ribosomes load at ~1 per 100 nt; the value keeps
#: the legacy coupling magnitude.  Registered in ``units.CALIBRATED``.
RIBO_SOME_DENSITY_PER_100NT = 0.1

#: protein-yield coupling gain: ``protein_amount = mrna_level * yield *
#: aa_count``.  Grounding: Bernstein 2002 — one mRNA makes ~10^2-10^3
#: proteins over its lifetime; 0.1 is the legacy normalized gain.
PROTEIN_YIELD_PER_MRNA_AA = 0.1

#: GRN feedback gain: protein abundance raises the gene's level.
PROTEIN_TO_GRN_GAIN = 0.01

#: morphogen field V concentration -> pigment-gene activation gain.
MORPHOGEN_TO_GRN_GAIN = 0.1

#: constitutive promoter reference strength (no explicit promoter).
CONSTITUTIVE_PROMOTER_STRENGTH = 0.5


@dataclass(slots=True)
class Frame:
    """Call frame: saves the return ip and the gene name."""
    return_ip: int
    gene_name: str


class BioInstructionDispatcher:
    """Instruction dispatcher (P2 refactor: extracted from CellVM, eliminating the God Class).

    Responsibilities:
    - ``dispatch(op)``: opcode dispatch, routes the :class:`Op` to the corresponding handler logic.
    - ``process_bio_instructions()``: routes bio instructions such as ``#crispr`` / ``#evolve`` /
      ``#methylate`` / ``#histone`` / ``#quorum`` / ``#transcribe`` /
      ``#translate`` to the ``_handle_*`` handlers.

    This class holds a reference to CellVM and accesses/modifies VM state through it (stack, frames,
    ip, cell, grn, field, _gene_dna, _crispr_edits, etc.). Stack/frame/IP management
    and the execution loop remain the responsibility of CellVM; operand reads (``_read_u8`` / ``_read_u16``)
    are delegated back to CellVM to keep IP advancement consistent.
    """

    def __init__(self, vm: CellVM) -> None:
        self._vm = vm
        # Bio instruction handler table (instance-level, bound methods)
        self._bio_handlers = {
            "crispr": self._handle_crispr,
            "evolve": self._handle_evolve,
            "methylate": self._handle_methylate,
            "histone": self._handle_histone,
            "quorum": self._handle_quorum,
            "transcribe": self._handle_transcribe,
            "translate": self._handle_translate,
        }

    # -------- opcode dispatch --------
    def dispatch(self, op: Op) -> None:
        vm = self._vm
        match op:
            case Op.OP_START | Op.OP_NOP | Op.OP_TICK:
                pass
            case Op.OP_HALT | Op.OP_RETURN:
                if vm.frames:
                    vm.ip = vm.frames.pop().return_ip
            case Op.OP_PUSH_CONST:
                idx = vm._read_u8()
                if idx < len(vm.chunk.constants):
                    vm.stack.append(vm.chunk.constants[idx])
                else:
                    vm.stack.append(idx)
            case Op.OP_POP:
                if vm.stack:
                    vm.stack.pop()
            case Op.OP_DUP:
                if vm.stack:
                    vm.stack.append(vm.stack[-1])
            case Op.OP_SWAP:
                if len(vm.stack) >= 2:
                    vm.stack[-1], vm.stack[-2] = vm.stack[-2], vm.stack[-1]
            case Op.OP_BUILD_PROTEIN:
                kind = vm._read_u8()
                vm.cell.add_protein(kind)
            case Op.OP_BUILD_MEMBRANE:
                # Operand = target membrane permeability (0..255, clamped);
                # scales nutrient intake via Cell.feed (see Cell.feed).
                vm.cell.set_membrane_permeability(vm._read_u8())
            case Op.OP_BUILD_PIGMENT:
                vm.cell.color = (200, 50, 50)
            case Op.OP_MOVE:
                d = vm._read_u8()
                vm.cell.move(d)
            case Op.OP_SIGNAL:
                # Release a signal molecule (quorum-sensing autoinducer)
                # into the field at the cell position; count every emission
                # so the event is observable even without a field.
                ch = vm._read_u8()
                vm._signal_emissions += 1
                if vm.field:
                    vm.field.emit(
                        vm.cell.x % vm.field.n,
                        vm.cell.y % vm.field.n,
                        min(1.0, SIGNAL_EMISSION_AMOUNT * (1 + ch)),
                    )
            case Op.OP_DIVIDE:
                vm._read_u8()
                vm.cell.divide()
            case Op.OP_DIE:
                vm._read_u8()
                vm.cell.die()
            case Op.OP_FEED:
                vm._read_u8()
                vm.cell.feed(FEED_ENERGY_AMOUNT)
            case Op.OP_GROW_LSYSTEM:
                _k = vm._read_u8()
                if vm.lsystems:
                    ls = next(iter(vm.lsystems.values()))
                    pts = ls.iterate()
                    vm.cell.morphology_points.extend(pts)
            case Op.OP_DIFFUSE:
                vm._read_u8()
                if vm.field:
                    vm.field.step()
            case Op.OP_REACT:
                vm._read_u8()
                if vm.field:
                    for _ in range(vm.program.config.react_steps):
                        vm.field.step()
            case Op.OP_EMIT_MORPHOGEN:
                # Morphogen ID scales the injected amount: (id+1)/256,
                # id=0 keeps the legacy non-zero emission.
                m_id = vm._read_u8()
                if vm.field:
                    vm.field.emit(
                        vm.cell.x % vm.field.n,
                        vm.cell.y % vm.field.n,
                        (m_id + 1) / EMIT_MORPHOGEN_SCALE,
                    )
            case Op.OP_READ_MEM:
                slot = vm._read_u8()
                vm.stack.append(vm.cell.slots[slot])
            case Op.OP_WRITE_MEM:
                slot = vm._read_u8()
                if vm.stack:
                    vm.cell.slots[slot] = vm.stack.pop()
            case Op.OP_MODIFY_STATE:
                f = vm._read_u8()
                if f == 0:
                    vm.cell.color = (100, 200, 50)
                elif f == 1:
                    vm.cell.age += 1
                elif f == 2:
                    vm.cell.color = (200, 200, 50)
                elif f == 3:
                    vm.cell.color = (200, 50, 200)
            case Op.OP_REGULATE:
                # Runtime regulatory-edge (re)wiring (Jacob & Monod 1961):
                # source = the currently-executing gene, target = gene
                # selected by the low nibble of the operand, sign = bit 7
                # (0 = activate, 1 = inhibit). The edge is added or, if it
                # already exists, its weight is updated in place.
                mode = vm._read_u8()
                names = list(vm.grn.nodes)
                if names:
                    source = (vm.frames[-1].gene_name if vm.frames
                              else names[0])
                    if source not in vm.grn.nodes:
                        source = names[0]
                    target = names[(mode & 0x0F) % len(names)]
                    weight = (REGULATE_EDGE_WEIGHT
                              if not (mode & 0x80)
                              else -REGULATE_EDGE_WEIGHT)
                    existing = next(
                        (e for e in vm.grn.edges
                         if e.source == source and e.target == target),
                        None,
                    )
                    if existing is not None:
                        existing.weight = weight
                    else:
                        vm.grn.add_edge(source, target, weight)
                    vm._regulation_events.append({
                        "tick": vm.tick, "source": source,
                        "target": target, "weight": weight,
                    })
            case Op.OP_BIND:
                # Protein-DNA binding (Berg & von Hippel 1987): the current
                # gene's transcription factor binds the target promoter,
                # consuming one unit of protein and boosting the target's
                # expression level (protein-limited: no TF, no binding).
                site = vm._read_u8()
                names = list(vm.grn.nodes)
                if names:
                    binder: str | None = (vm.frames[-1].gene_name
                                          if vm.frames else None)
                    tf_kind: str | int | None = None
                    if binder in vm.cell.proteins:
                        tf_kind = binder
                    elif vm.cell.proteins:
                        tf_kind = next(iter(vm.cell.proteins))
                    if tf_kind is not None:
                        consumed = vm.cell.consume_protein(tf_kind, 1.0)
                        if consumed > 0:
                            target = names[site % len(names)]
                            lvl = vm.grn.nodes[target].level
                            vm.grn.set_level(
                                target, lvl + BIND_LEVEL_BOOST)
                            vm._binding_events.append({
                                "tick": vm.tick, "target": target,
                                "protein": str(tf_kind),
                                "boost": BIND_LEVEL_BOOST,
                            })
            case Op.OP_CALL_GENE:
                off = vm._read_u16()
                vm.frames.append(Frame(return_ip=vm.ip, gene_name="<call>"))
                vm.ip = off
            case Op.OP_JUMP:
                off = vm._read_u16()
                vm.ip += off
            case Op.OP_JUMP_IF_ZERO:
                off = vm._read_u16()
                v = vm.stack.pop() if vm.stack else 0
                if not v:
                    vm.ip += off
            case Op.OP_ADD:
                if len(vm.stack) >= 2:
                    b = vm.stack.pop()
                    a = vm.stack.pop()
                    vm.stack.append(a + b)
            case Op.OP_SUB:
                if len(vm.stack) >= 2:
                    b = vm.stack.pop()
                    a = vm.stack.pop()
                    vm.stack.append(a - b)
            case Op.OP_MUL:
                if len(vm.stack) >= 2:
                    b = vm.stack.pop()
                    a = vm.stack.pop()
                    vm.stack.append(a * b)
            case Op.OP_LT:
                if len(vm.stack) >= 2:
                    b = vm.stack.pop()
                    a = vm.stack.pop()
                    vm.stack.append(1 if a < b else 0)
            case Op.OP_NOT:
                if vm.stack:
                    a = vm.stack.pop()
                    vm.stack.append(0 if a else 1)
            case Op.OP_DEBUG:
                print(f"DEBUG: {vm.cell.dump()}")
            case _:
                # Unimplemented opcode: skip its operands (with bounds protection)
                nbytes = OP_OPERAND_BYTES.get(op, 0)
                vm.ip = min(vm.ip + nbytes, len(vm.chunk.code))

    # -------- bio instruction dispatch --------
    def process_bio_instructions(self) -> None:
        """Process bio instructions: CRISPR / evolution / epigenetics / quorum sensing, etc.

        Each tick, all instructions in ``program.bio_instructions`` are dispatched to the corresponding handlers.
        """
        for inst in self._vm.program.bio_instructions:
            handler = self._bio_handlers.get(inst.kind)
            if handler is not None:
                handler(inst)

    # -------- bio instruction handlers --------
    def _handle_crispr(self, inst: BioInstruction) -> None:
        """Handle the #crispr instruction: perform CRISPR editing on the target gene's DNA."""
        from helixlang.crispr import edit_gene
        vm = self._vm
        target = inst.target
        dna = vm._gene_dna.get(target, "")
        if not dna:
            return
        position = int(inst.params.get("position", 0))
        new_seq = inst.params.get("new_sequence", "")
        cas = inst.params.get("cas", "SpCas9")
        try:
            result = edit_gene(
                dna, target_position=position,
                new_sequence=new_seq, cas_variant=cas,
                rng=vm._rng,
            )
            vm._gene_dna[target] = result.edited_dna
            vm._crispr_edits.append({
                "tick": vm.tick,
                "target": target,
                "success": result.success,
                "edit_type": result.edit_type,
                "off_targets": len(result.off_targets),
            })
        except (ValueError, KeyError, RuntimeError) as exc:
            # Edit failed (e.g. no PAM site, position out of bounds, unknown Cas variant) — record rather than silently swallowing the error
            vm._crispr_edits.append({
                "tick": vm.tick,
                "target": target,
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
            })

    def _handle_evolve(self, inst: BioInstruction) -> None:
        """Handle the #evolve instruction: evolve the target gene by one generation."""
        from helixlang.evolution import mutate
        vm = self._vm
        target = inst.target
        dna = vm._gene_dna.get(target, "")
        if not dna:
            return
        rate = float(inst.params.get("mutation_rate", 0.01))
        indel_rate = float(inst.params.get("indel_rate", rate * 0.1))
        new_dna, mutations = mutate(
            dna, mutation_rate=rate, indel_rate=indel_rate, rng=vm._rng,
        )
        vm._gene_dna[target] = new_dna
        vm._evolution_history.append({
            "tick": vm.tick,
            "target": target,
            "mutations": len(mutations),
            "mutation_list": mutations[:5],  # keep only the first 5 entries
        })

    def _handle_methylate(self, inst: BioInstruction) -> None:
        """Handle the #methylate instruction: DNA methylation represses gene expression."""
        from helixlang.epigenetics import methylate_dna
        vm = self._vm
        target = inst.target
        dna = vm._gene_dna.get(target, "")
        if not dna:
            return
        methylase = inst.params.get("methylase", "dam")
        state = methylate_dna(dna, methylase=methylase)
        # Methylation reduces expression (~70% Bird 2002)
        meth_fraction = (state.methylated_sites / max(1, state.total_sites))
        repression = 1.0 - 0.7 * meth_fraction
        vm._chromatin_modifier[target] = repression
        vm._epigenetic_marks.append({
            "tick": vm.tick,
            "target": target,
            "type": "methylation",
            "methylase": methylase,
            "sites": state.total_sites,
            "methylated": state.methylated_sites,
            "repression": repression,
        })

    def _handle_histone(self, inst: BioInstruction) -> None:
        """Handle the #histone instruction: histone modification affects gene expression."""
        from helixlang.epigenetics import HISTONE_MARK_TYPES
        vm = self._vm
        target = inst.target
        mark = inst.params.get("mark", "H3K4me3")
        mark_info = HISTONE_MARK_TYPES.get(mark, {"score": 0.0})
        score = mark_info.get("score", 0.0)
        # Positive score = activation, negative = repression → converted into an expression modifier
        current = vm._chromatin_modifier.get(target, 1.0)
        vm._chromatin_modifier[target] = max(0.0, current + score * 0.5)
        vm._epigenetic_marks.append({
            "tick": vm.tick,
            "target": target,
            "type": "histone",
            "mark": mark,
            "score": score,
        })

    def _handle_quorum(self, inst: BioInstruction) -> None:
        """Handle the #quorum instruction: quorum sensing activates the target gene."""
        vm = self._vm
        threshold = float(
            inst.params.get("threshold", QUORUM_SIGNAL_THRESHOLD))
        activate = inst.params.get("activate", inst.target)
        # Signal field concentration
        if vm.field:
            x = vm.cell.x % vm.field.n
            y = vm.cell.y % vm.field.n
            signal = vm.field.v[x][y]
        else:
            signal = 0.0
        if signal >= threshold and activate in vm.grn.nodes:
            vm.grn.set_level(
                activate,
                max(vm.grn.nodes[activate].level, 1.0),
            )

    def _handle_transcribe(self, inst: BioInstruction) -> None:
        """Handle the #transcribe instruction: force transcription of the target gene."""
        vm = self._vm
        target = inst.target
        if target in vm.grn.nodes:
            vm.grn.set_level(target, 1.0)

    def _handle_translate(self, inst: BioInstruction) -> None:
        """Handle the #translate instruction: force translation of the target gene."""
        vm = self._vm
        target = inst.target
        if target in vm.cell.proteins:
            vm.cell.proteins[target] = (
                vm.cell.proteins.get(target, 0.0) + 1.0
            )


class CellVM:
    """Bytecode VM + cell simulator."""

    def __init__(self, chunk: Chunk, program: Program):
        self.chunk = chunk
        self.program = program
        self.ip = 0
        self.stack: list = []
        self.frames: list[Frame] = []
        # ``#config units=real`` (Tier 3) activates calibrated subsystems:
        # physical-unit Cell and GRN while keeping all default counts.
        self._real_units = self.program.config.units == "real"
        self.cell = Cell(calibrated=self._real_units)
        self.grn = GRN(calibrated=self._real_units)
        self.lsystems: dict[str, LSystem] = {}
        self.field: GrayScott | None = None
        self.tick = 0
        self.debug = False
        self.trace: list[dict] = []
        # P0-1.2 central dogma pipeline state
        self._gene_dna: dict[str, str] = {}        # gene_name → DNA sequence
        self._gene_mrna: dict[str, float] = {}     # gene_name → mRNA concentration
        self._promoter_strengths: dict[str, float] = {}  # promoter_name → strength
        self._chromatin_modifier: dict[str, float] = {}  # gene_name → expression modifier
        self._crispr_edits: list[dict] = []          # CRISPR edit records
        self._epigenetic_marks: list[dict] = []      # epigenetic modification records
        self._evolution_history: list[dict] = []     # evolution event records
        # Runtime regulation / binding / signal event records (opcode effects)
        self._regulation_events: list[dict] = []     # OP_REGULATE edge changes
        self._binding_events: list[dict] = []        # OP_BIND protein-DNA bindings
        self._signal_emissions: int = 0              # OP_SIGNAL emission count
        self._rng = random.Random(0)
        self._init_subsystems()
        # P2 refactor: instruction dispatch delegated to BioInstructionDispatcher
        self._dispatcher = BioInstructionDispatcher(self)

    # -------- initialization --------
    def _init_subsystems(self) -> None:
        prom_by_name = {p.name: p for p in self.program.promoters}
        # Promoters as GRN nodes; negative strength means constitutive (active from the start)
        for p in self.program.promoters:
            initial = 1.0 if p.strength < 0 else 0.0
            self.grn.add_gene(p.name, threshold=p.strength, initial_level=initial)
            self._promoter_strengths[p.name] = max(0.0, min(1.0, abs(p.strength)))
        # Genes as GRN nodes; use the promoter's strength if present, otherwise constitutive expression
        for g in self.program.genes:
            if g.promoter and g.promoter in prom_by_name:
                threshold = prom_by_name[g.promoter].strength
                initial = 0.0
                self._promoter_strengths[g.promoter] = max(
                    0.0, min(1.0, abs(prom_by_name[g.promoter].strength)))
            else:
                threshold = -1.0  # constitutive expression
                initial = 1.0     # active from the start
            self.grn.add_gene(g.name, threshold, initial_level=initial)
            # P0-1.2: cache the gene's DNA sequence (for the central dogma pipeline)
            self._gene_dna[g.name] = "".join(c.seq for c in g.codons)
            self._chromatin_modifier[g.name] = 1.0
        # Regulatory edges
        for r in self.program.regulations:
            self.grn.add_edge(r.source, r.target, r.strength)
        # L-system
        for name, decl in self.program.lsystems.items():
            rules = decl.rules.get(0, {})  # default to rule set 0
            self.lsystems[name] = LSystem(
                axiom=decl.axiom, rules=rules,
                angle=decl.angle, step=decl.step,
            )
        # Morphology field
        if self.program.field_decl is not None:
            f = self.program.field_decl
            self.field = GrayScott(
                n=f.size, F=f.F, k=f.k, Du=f.Du, Dv=f.Dv)

    # -------- main loop --------
    def run(self, max_ticks: int) -> list[dict]:
        """Run max_ticks ticks and return the trace.

        When ``program.config.use_central_dogma=True``, the central dogma pipeline
        (P0-1.2) is enabled: each tick first processes bio instructions, then does transcription-translation coupling,
        then morphology update and snapshot; otherwise the original GRN + bytecode execution path is used.
        """
        while self.tick < max_ticks and self.cell.alive:
            if self.program.config.use_central_dogma:
                # P0-1.2: central dogma pipeline
                self._process_bio_instructions()
                self._transcribe_translate()
                self._flush_morphology()
                self._feedback()
                self._snapshot()
            else:
                # Original GRN + bytecode path
                triggered = self.grn.step()
                for g in triggered:
                    self._call_gene(g)
                self._execute_pending()
                self._flush_morphology()
                self._feedback()
                self._snapshot()
            self.tick += 1
        return self.trace

    # -------- P0-1.2: central dogma pipeline --------
    def _transcribe_translate(self) -> None:
        """Transcription-translation coupling model: DNA → mRNA → protein.

        Based on the central_dogma module:
        1. Transcribe each gene's DNA (including promoter strength, TF effects, terminators)
        2. Translate the transcripts (including RBS, codon-specific rates, termination efficiency)
        3. Update intracellular protein concentrations and GRN gene expression levels
        """
        from helixlang.bio_data import get_species_trna
        from helixlang.central_dogma import (
            calculate_mrna_level,
            transcribe,
            translate,
        )

        species = self.program.config.species
        trna_abundance = get_species_trna(species)
        # Advance the GRN one step to get the current expression level of each gene
        self.grn.step()
        for gene in self.program.genes:
            dna = self._gene_dna.get(gene.name, "")
            if not dna:
                continue
            # Promoter strength (normalized to 0..1)
            prom_strength = self._get_promoter_strength(gene.promoter)
            # Epigenetic modifier (methylation/histone modification affects expression)
            chromatin = self._chromatin_modifier.get(gene.name, 1.0)
            effective_strength = max(0.0, min(1.0, prom_strength * chromatin))
            # Transcription factor effects (from GRN regulation)
            tf_effects = self._get_transcription_factors(gene.name)
            # Transcribe
            transcript = transcribe(
                dna,
                promoter_strength=effective_strength,
                transcription_factors=tf_effects or None,
            )
            # Translate
            result = translate(
                transcript,
                trna_abundance=trna_abundance,
                ribosome_density=RIBO_SOME_DENSITY_PER_100NT,
            )
            # mRNA level (kinetic steady state)
            mrna_level = calculate_mrna_level(transcript, time=float(self.tick + 1))
            self._gene_mrna[gene.name] = mrna_level
            # Update cell proteins (protein molecule count ∝ mRNA × ribosome density)
            if result.protein:
                protein_amount = (mrna_level * PROTEIN_YIELD_PER_MRNA_AA
                                  * len(result.protein))
                self.cell.proteins[gene.name] = protein_amount
                # Feedback to GRN: protein abundance raises the gene level
                if gene.name in self.grn.nodes:
                    self.grn.set_level(
                        gene.name,
                        min(1.0, self.grn.nodes[gene.name].level
                            + protein_amount * PROTEIN_TO_GRN_GAIN),
                    )

    def _get_promoter_strength(self, promoter_name: str | None) -> float:
        """Get the effective promoter strength (0..1)."""
        if promoter_name is None:
            return CONSTITUTIVE_PROMOTER_STRENGTH  # constitutive moderate expression
        return self._promoter_strengths.get(
            promoter_name, CONSTITUTIVE_PROMOTER_STRENGTH)

    def _get_transcription_factors(self, gene_name: str) -> dict[str, float]:
        """Get the transcription factor effects acting on the gene (from GRN regulatory edges).

        Returns {regulator_name: fold_change}, >1=activation, <1=repression.
        """
        tf_effects: dict[str, float] = {}
        for r in self.program.regulations:
            if r.target == gene_name:
                # Regulatory edge source gene level × strength → fold_change
                source_level = self.grn.nodes.get(r.source)
                level = (source_level.level if source_level else 0.0)
                # Positive strength → activation (fold > 1), negative → repression (fold < 1)
                fold = 1.0 + r.strength * level
                tf_effects[r.source] = max(0.0, fold)
        return tf_effects

    # -------- P0-1.1: bio instruction processing (delegated to the dispatcher) --------
    def _process_bio_instructions(self) -> None:
        """Process bio instructions (delegated to BioInstructionDispatcher)."""
        self._dispatcher.process_bio_instructions()

    # -------- call / frames --------
    def _call_gene(self, name: str) -> None:
        off = self.chunk.gene_offsets.get(name)
        if off is None:
            return
        # Frame depth cap: prevent unbounded frame accumulation from the GRN pushing frames across ticks
        if len(self.frames) >= 256:
            return
        self.frames.append(Frame(return_ip=self.ip, gene_name=name))
        self.ip = off

    def _execute_pending(self) -> None:
        """Execute bytecode until the frames are empty or the tick quota is exhausted."""
        quota = self.program.config.ops_per_tick
        while self.frames and quota > 0:
            # Frame depth guard: prevent unbounded frame accumulation across ticks (leftover from exhausted quota + GRN continuously pushing frames)
            if len(self.frames) > 256:
                self.frames.clear()
                break
            if self.ip >= len(self.chunk.code):
                self.frames.pop()
                if self.frames:
                    self.ip = self.frames[-1].return_ip
                break
            op_byte = self.chunk.code[self.ip]
            self.ip += 1
            try:
                op = Op(op_byte)
            except ValueError:
                # Unknown byte: cannot determine the operand length, skip only 1 byte; record it in debug mode
                if self.debug:
                    print(f"[tick={self.tick} ip={self.ip - 1}] "
                          f"<unknown 0x{op_byte:02X}>")
                continue
            if self.debug:
                print(f"[tick={self.tick} ip={self.ip - 1}] "
                      f"{op.name} stack={self.stack}")
            self._dispatch(op)
            quota -= 1
        # Frames not empty but quota exhausted: resume execution on the next tick (no forced frame pop)

    # -------- dispatch (thin wrapper, delegated to the dispatcher; keeps backward compatibility) --------
    def _dispatch(self, op: Op) -> None:
        self._dispatcher.dispatch(op)

    # -------- read operands --------
    def _read_u8(self) -> int:
        if self.ip >= len(self.chunk.code):
            return 0
        v = self.chunk.code[self.ip]
        self.ip += 1
        return v

    def _read_u16(self) -> int:
        if self.ip + 1 >= len(self.chunk.code):
            self.ip = len(self.chunk.code)
            return 0
        v = (self.chunk.code[self.ip] << 8) | self.chunk.code[self.ip + 1]
        self.ip += 2
        return v

    # -------- morphology / feedback --------
    def _flush_morphology(self) -> None:
        # Morphology updates are handled inline in _dispatch
        pass

    def _feedback(self) -> None:
        """Morphology field V concentration → extra activation of the pigment gene."""
        if self.field and "pigment" in self.grn.nodes:
            i = self.cell.x % self.field.n
            j = self.cell.y % self.field.n
            v = self.field.v[i][j]
            self.grn.set_level(
                "pigment",
                self.grn.nodes["pigment"].level + v * MORPHOGEN_TO_GRN_GAIN,
            )

    # -------- snapshot --------
    def _snapshot(self) -> None:
        snap = {
            "tick": self.tick,
            "x": self.cell.x,
            "y": self.cell.y,
            "energy": self.cell.energy,
            "alive": self.cell.alive,
            "proteins": dict(self.cell.proteins),
            "color": self.cell.color,
            "gene_levels": {n: nd.level for n, nd in self.grn.nodes.items()},
            "morphology_points_count": len(self.cell.morphology_points),
            "membrane_permeability": self.cell.membrane_permeability,
            "signal_emissions": self._signal_emissions,
            "regulation_edges": len(self.grn.edges),
            "binding_events": len(self._binding_events),
            "field_total_v": self.field.total_v() if self.field else 0.0,
            "units": self.program.config.units,
        }
        self.trace.append(snap)
