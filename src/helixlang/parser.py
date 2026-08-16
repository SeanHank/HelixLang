"""Recursive-descent Parser: token stream -> AST.

ORF identification: each #gene block runs from the first ATG to the first STOP.
Bare DNA blocks (without #gene) are wrapped as anonymous genes.

P0-1.1 language extension: supports biological instruction annotations
    #crispr target=gene position=50 new_sequence="GGGG" cas=SpCas9
    #evolve target=gene generations=100 mutation_rate=0.01
    #methylate target=gene methylase=dam
    #histone target=gene mark=H3K4me3
    #transcribe target=gene
    #quorum threshold=5.0 activate=gene

P0-1.3 type system integration: supports type annotations
    #type gene_name=Protein
"""
from __future__ import annotations

from helixlang.ast_nodes import (
    BioInstruction,
    Codon,
    EnzymeDecl,
    FieldDecl,
    Gene,
    LSystemDecl,
    MediaDecl,
    MorphogenFeedback,
    PoolDecl,
    Program,
    Promoter,
    Regulation,
)
from helixlang.errors import ParseError
from helixlang.lexer import Token

# Supported biological instruction kinds
BIO_INSTRUCTION_KINDS = frozenset({
    "crispr", "evolve", "methylate", "histone",
    "transcribe", "translate", "quorum",
})


class Parser:
    """Recursive-descent Parser."""

    def __init__(self, tokens: list[Token],
                 stop_codons: set[str] | None = None,
                 enable_type_check: bool = False):
        self.toks = [t for t in tokens if t.kind != "NEWLINE"]
        self.i = 0
        self.anon_counter = 0
        # Stop codon set (can be derived from the translation table); defaults to the standard table
        self.stop_codons = stop_codons or {"TAA", "TAG", "TGA"}
        self.enable_type_check = enable_type_check
        self._type_errors: list[str] = []

    # -------- Entry point --------
    def parse(self) -> Program:
        prog = Program()
        while self._peek() and self._peek().kind != "EOF":
            t = self._peek()
            if t.kind == "ANNOT_START":
                handler = {
                    "promoter": self._parse_promoter,
                    "gene": self._parse_gene,
                    "regulate": self._parse_regulate,
                    "lsystem": self._parse_lsystem,
                    "field": self._parse_field,
                    "morphogen": self._parse_morphogen,
                    "config": self._parse_config,
                    "type": self._parse_type_annotation,
                    "media": self._parse_media,
                    "enzyme": self._parse_enzyme,
                    "metabolite": self._parse_metabolite,
                    "sim": self._parse_sim,
                    "genome": self._parse_genome,
                    "species": self._parse_species,
                    "patch": self._parse_patch,
                }.get(t.value)
                # Biological instructions (P0-1.1)
                if t.value in BIO_INSTRUCTION_KINDS:
                    self._parse_bio_instruction(prog, t.value)
                    continue
                if handler is None:
                    raise ParseError(f"unknown annotation #{t.value}",
                                     line=t.line, col=t.col)
                handler(prog)
            elif t.kind == "CODON":
                prog.genes.append(self._wrap_anon_gene())
            else:
                raise ParseError(f"unexpected token {t.kind} ({t.value!r})",
                                 line=t.line, col=t.col)
        # P0-1.3: run type checking after parsing
        if self.enable_type_check:
            self._run_type_check(prog)
        return prog

    # -------- Biological instruction parsing (P0-1.1) --------
    def _parse_bio_instruction(self, prog: Program, kind: str) -> None:
        """Parse biological instruction annotations.

        Format: #crispr target=gene position=50 new_sequence="GGGG" cas=SpCas9
        """
        t = self._advance()  # ANNOT_START
        fields = self._collect_fields_until_block_end(allow_no_end=True)
        target = fields.get("target", "")
        if not target:
            raise ParseError(
                f"#{kind} requires target= field",
                line=t.line,
            )
        # Strip quotes from string values
        params: dict[str, str] = {}
        for k, v in fields.items():
            if v.startswith('"') and v.endswith('"'):
                params[k] = v[1:-1]
            else:
                params[k] = v
        prog.bio_instructions.append(BioInstruction(
            kind=kind, target=target, params=params, line=t.line,
        ))

    # -------- Annotations --------
    def _parse_promoter(self, prog: Program) -> None:
        t = self._advance()  # ANNOT_START
        fields = self._collect_fields_until_block_end(allow_no_end=True)
        if "name" not in fields:
            raise ParseError("#promoter missing name= field", line=t.line)
        if "strength" not in fields:
            raise ParseError(f"#promoter {fields['name']} missing strength=", line=t.line)
        try:
            strength = float(fields["strength"])
        except ValueError as e:
            raise ParseError(f"invalid strength {fields['strength']!r}: {e}",
                             line=t.line) from None
        prog.promoters.append(Promoter(
            name=fields["name"], strength=strength, fields=fields))

    def _parse_gene(self, prog: Program) -> None:
        t = self._advance()  # ANNOT_START
        fields = self._collect_fields_until_block_end()
        name = fields.get("name") or f"__anon_{self.anon_counter}"
        if name.startswith("__anon"):
            self.anon_counter += 1
        promoter = fields.get("promoter")
        # Collect CODON stream (before #end)
        codons: list[Codon] = []
        while self._peek() and self._peek().kind == "CODON":
            ct = self._advance()
            codons.append(Codon(seq=ct.value, index=ct.codon_index, line=ct.line))
        if not codons:
            raise ParseError(f"#gene {name!r} has no DNA codons", line=t.line)
        orf = self._extract_orf(codons, name, t.line)
        prog.genes.append(Gene(name=name, promoter=promoter,
                               codons=codons, orf=orf, fields=fields))
        # Consume #end (if present)
        if self._peek() and self._peek().kind == "ANNOT_END":
            self._advance()

    def _parse_regulate(self, prog: Program) -> None:
        t = self._advance()  # ANNOT_START
        # regulate line: source -> target strength=±0.9
        arrow_t = self._expect("ARROW")
        src, tgt = arrow_t.value.split("->", 1)
        src, tgt = src.strip(), tgt.strip()
        fields = self._collect_fields_until_block_end(allow_no_end=True)
        strength = 0.5
        if "strength" in fields:
            try:
                strength = float(fields["strength"])
            except ValueError:
                raise ParseError(f"invalid strength {fields['strength']!r}",
                                 line=t.line) from None
        prog.regulations.append(Regulation(source=src, target=tgt, strength=strength))

    def _parse_lsystem(self, prog: Program) -> None:
        self._advance()
        fields = self._collect_fields_until_block_end(allow_no_end=True)
        name = fields.get("name", "default")
        axiom = fields.get("axiom", "F")
        angle = float(fields.get("angle", "25"))
        step = float(fields.get("step", "1.0"))
        # rules field format: "0:F->F[+F]F[-F]F;1:F->FF"
        rules: dict[int, dict[str, str]] = {}
        rules_str = fields.get("rules", "")
        if rules_str:
            for entry in rules_str.split(";"):
                if not entry or ":" not in entry:
                    continue
                k_str, body = entry.split(":", 1)
                try:
                    k = int(k_str)
                except ValueError:
                    continue
                # body looks like "F->F[+F]F[-F]F,X->FX"
                d: dict[str, str] = {}
                for pair in body.split(","):
                    if "->" in pair:
                        sym, prod = pair.split("->", 1)
                        d[sym] = prod
                rules[k] = d
        prog.lsystems[name] = LSystemDecl(
            name=name, axiom=axiom, rules=rules, angle=angle, step=step)

    def _parse_field(self, prog: Program) -> None:
        self._advance()  # ANNOT_START
        fields = self._collect_fields_until_block_end(allow_no_end=True)
        size = int(fields.get("size", "32"))
        F = float(fields.get("F", "0.035"))
        k = float(fields.get("k", "0.065"))
        Du = float(fields.get("Du", "0.16"))
        Dv = float(fields.get("Dv", "0.08"))
        prog.field_decl = FieldDecl(size=size, F=F, k=k, Du=Du, Dv=Dv)

    def _parse_morphogen(self, prog: Program) -> None:
        """Parse #morphogen gene=<name> channel=V|U gain=<float> (G9).

        Declarative morphogen→gene feedback wiring, replacing the legacy
        hard-coded ``pigment`` gene feedback.
        """
        t = self._advance()  # ANNOT_START
        fields = self._collect_fields_until_block_end(allow_no_end=True)
        gene = fields.get("gene", "")
        if not gene:
            raise ParseError("#morphogen requires gene= field", line=t.line)
        channel = fields.get("channel", "V").upper()
        if channel not in ("U", "V"):
            raise ParseError(
                f"#morphogen channel must be 'U' or 'V', got {channel!r}",
                line=t.line)
        try:
            gain = float(fields.get("gain", "0.1"))
        except ValueError as e:
            raise ParseError(f"invalid gain {fields['gain']!r}: {e}",
                             line=t.line) from None
        prog.morphogen_feedback.append(MorphogenFeedback(
            gene=gene, channel=channel, gain=gain))

    def _parse_config(self, prog: Program) -> None:
        self._advance()
        fields = self._collect_fields_until_block_end(allow_no_end=True)
        if "ticks" in fields:
            prog.config.ticks = int(fields["ticks"])
        if "output" in fields:
            prog.config.output = [s.strip() for s in fields["output"].split(",") if s.strip()]
        if "table" in fields:
            prog.config.table = fields["table"]
        if "ops_per_tick" in fields:
            prog.config.ops_per_tick = int(fields["ops_per_tick"])
        if "react_steps" in fields:
            prog.config.react_steps = int(fields["react_steps"])
        # P0-1.2: central dogma pipeline switch
        if "use_central_dogma" in fields:
            prog.config.use_central_dogma = fields["use_central_dogma"].lower() in ("true", "1", "yes")
        if "species" in fields:
            prog.config.species = fields["species"]
        # Simulation backend selector (12-helix-language-wiring.md §6.1)
        if "backend" in fields:
            prog.config.backend = fields["backend"]
        # Every remaining #config key is a sim parameter: preserved verbatim
        # for the backend adapter (12-helix-language-wiring.md §7.1). The classic
        # pipeline never reads `sim`, so its behaviour is untouched.
        consumed = {
            "ticks", "output", "table", "ops_per_tick", "react_steps",
            "use_central_dogma", "species", "backend",
        }
        for k, v in fields.items():
            if k not in consumed:
                prog.config.sim[k] = v

    def _parse_media(self, prog: Program) -> None:
        """Parse #media nutrient=GLC concentration=10.0 [diffusion_um2_s=300].

        Growth-medium declaration consumed by the sim backends; inert (with a
        warning) under the classic backend (12-helix-language-wiring.md §6.4).
        """
        t = self._advance()  # ANNOT_START
        fields = self._collect_fields_until_block_end(allow_no_end=True)
        nutrient = fields.get("nutrient", "")
        if not nutrient:
            raise ParseError("#media requires nutrient= field", line=t.line)
        if "concentration" not in fields:
            raise ParseError(
                f"#media {nutrient!r} requires concentration= field", line=t.line)
        try:
            concentration = float(fields["concentration"])
        except ValueError as e:
            raise ParseError(
                f"invalid concentration {fields['concentration']!r}: {e}",
                line=t.line) from None
        diffusion: float | None = None
        if "diffusion_um2_s" in fields:
            try:
                diffusion = float(fields["diffusion_um2_s"])
            except ValueError as e:
                raise ParseError(
                    f"invalid diffusion_um2_s {fields['diffusion_um2_s']!r}: {e}",
                    line=t.line) from None
        prog.media.append(MediaDecl(
            nutrient=nutrient, concentration=concentration,
            diffusion_um2_s=diffusion))

    def _parse_enzyme(self, prog: Program) -> None:
        """Parse #enzyme gene=gltA reaction=CS [kcat=2800].

        Enzyme--reaction binding for enzyme-constrained FBA; inert under the
        classic backend (12-helix-language-wiring.md §6.5).
        """
        t = self._advance()  # ANNOT_START
        fields = self._collect_fields_until_block_end(allow_no_end=True)
        gene = fields.get("gene", "")
        reaction = fields.get("reaction", "")
        if not gene:
            raise ParseError("#enzyme requires gene= field", line=t.line)
        if not reaction:
            raise ParseError(
                f"#enzyme {gene!r} requires reaction= field", line=t.line)
        kcat: float | None = None
        if "kcat" in fields:
            try:
                kcat = float(fields["kcat"])
            except ValueError as e:
                raise ParseError(
                    f"invalid kcat {fields['kcat']!r}: {e}", line=t.line) from None
        prog.enzymes.append(EnzymeDecl(gene=gene, reaction=reaction, kcat=kcat))

    def _parse_metabolite(self, prog: Program) -> None:
        """Parse #metabolite name=glc__D init=0.5.

        Intracellular pool initialisation; requires
        ``#config metabolite_pools=true`` to take effect, inert under classic
        (12-helix-language-wiring.md §6.6).
        """
        t = self._advance()  # ANNOT_START
        fields = self._collect_fields_until_block_end(allow_no_end=True)
        name = fields.get("name", "")
        if not name:
            raise ParseError("#metabolite requires name= field", line=t.line)
        try:
            init = float(fields.get("init", "0.0"))
        except ValueError as e:
            raise ParseError(
                f"invalid init {fields['init']!r}: {e}", line=t.line) from None
        prog.pools.append(PoolDecl(name=name, init=init))

    def _parse_sim(self, prog: Program) -> None:
        """Parse #sim key=value ... (open extension point, wiring.md §8.6).

        Each #sim annotation merges its fields into ``Program.sim_extensions``,
        reserved for long-tail backends (e.g. ``#sim kind=spatial_dfba``).
        Inert until a backend registers it.
        """
        self._advance()  # ANNOT_START
        fields = self._collect_fields_until_block_end(allow_no_end=True)
        for k, v in fields.items():
            prog.sim_extensions[k] = v

    def _parse_genome(self, prog: Program) -> None:
        """Parse #genome source=... (doc/18-programmable-cell-population-simulation.md §13 Design 5, task 1).

        Fields are merged into ``Program.sim_extensions`` under a ``genome_``
        prefix (the same open extension point as ``#sim``) and turn the
        genome-scale backend on:

            #genome source=synth-4300 tf_map=regulon grn_mode=sparse
            #genome seed=7

        Keys consumed by ``sim_runtime._build_population_config``:
        ``genome`` (true/false), ``genome_source`` (``ecoli-mg1655`` |
        ``synth-4300`` | file path), ``genome_tf_map`` (``regulon`` |
        ``random`` | ``off``), ``genome_grn_mode`` (``sparse`` | ``full``),
        ``genome_active_gene_budget`` (per-cell per-tick budget, default
        512), ``genome_seed``.  Inert under the classic backend.
        """
        self._advance()  # ANNOT_START
        fields = self._collect_fields_until_block_end(allow_no_end=True)
        prog.sim_extensions["genome"] = "true"
        for k, v in fields.items():
            prog.sim_extensions[f"genome_{k}"] = v

    def _parse_species(self, prog: Program) -> None:
        """Parse #species name=... (doc/19 §5.3 A2; ecosystem spine).

        The ecosystem backend's species table, namespaced into
        ``Program.sim_extensions`` under a ``species.<name>.`` prefix (the
        same open extension point as ``#sim``/``#genome``):

            #species name=producer photo=true photo_vmax=0.01 cn_ratio=8
            #species name=consumer substrate=glucose vmax=0.02 ks=0.1
            #species name=predator diet=consumer:0.5 attack=consumer:0.001
            #species name=acetotroph substrate=acetate vmax=0.012 ks=0.05

        Supported fields (after ``name``): ``genome``, ``photo``,
        ``photo_vmax``, ``cn_ratio``, ``maintenance``,
        ``consumption.<sub>.vmax`` / ``consumption.<sub>.ks`` (dotted keys
        for multiple substrates), the flat ``substrate``/``vmax``/``ks``
        (plus ``substrate2``/``vmax2``/``ks2``) form,
        ``secretion=<sub>:<rate>``, ``diet=<prey>:<efficiency>`` and
        ``attack=<prey>:<rate>``.  The genotype may be given either as a
        ``genome=`` field or as a DNA code block on the following lines
        (analogous to ``#gene``): the block DNA is concatenated into the
        ``genome`` and must not be combined with a ``genome=`` field.
        Consumed by ``sim_runtime._run_ecosystem``; inert otherwise.
        """
        t = self._advance()  # ANNOT_START
        fields = self._collect_fields_until_block_end(allow_no_end=True)
        name = fields.get("name", "")
        if not name:
            raise ParseError("#species requires name= field", line=t.line)
        if "genome" in fields and self._peek().kind == "CODON":
            raise ParseError(
                f"#species {name}: use either a genome= field or a DNA "
                "code block, not both", line=t.line)
        if "genome" not in fields and self._peek().kind == "CODON":
            # DNA code block (analogous to #gene): every CODON token up to
            # the next annotation/#end forms the species genotype
            codons: list[str] = []
            while self._peek().kind == "CODON":
                ct = self._advance()
                codons.append(ct.value)
            fields["genome"] = "".join(codons)
            if self._peek().kind == "ANNOT_END":
                self._advance()
        for k, v in fields.items():
            if k != "name":
                prog.sim_extensions[f"species.{name}.{k}"] = v

    def _parse_patch(self, prog: Program) -> None:
        """Parse #patch name=... (doc/19 §5.3 A2; multi-environment, G10).

        The ecosystem backend's habitat table, namespaced into
        ``Program.sim_extensions`` under a ``patch.<name>.`` prefix:

            #patch name=water kind=water width=4 height=4
            #patch name=sediment kind=sediment anoxic=true moisture=0.6
            #patch name=chemostat kind=chemostat flow_rate=0.002
            #patch initial producer=100 consumer=10
            #patch substrate glucose initial=1.0 bulk=10.0 diffusion=600
            #patch scalar light initial=500 kind=light forcing=diurnal
            #patch scalar temperature initial=25 forcing=diurnal amplitude=3
            #patch dispersal sediment=0.0001
            #patch carrying_capacity=1e5

        Supported fields (after ``name``): ``kind``, ``width``, ``height``,
        ``carrying_capacity``, ``anoxic``, ``moisture``, ``clay``,
        ``cn_som``, ``cn_species``, ``initial_nh4_mm``, ``initial_no3_mm``,
        ``flow_rate``, ``fluctuation_period``, ``fluctuation_amplitude``,
        ``initial.<species>`` (biomass), ``substrate.<sub>.initial`` /
        ``.bulk`` / ``.diffusion`` / ``.carbon_per_mol``,
        ``scalar.<name>.kind`` /
        ``.initial`` / ``.forcing`` (``diurnal`` | ``seasonal``) /
        ``.amplitude``, and ``dispersal.<neighbor>``.  Consumed by
        ``sim_runtime._run_ecosystem``; inert otherwise.
        """
        t = self._advance()  # ANNOT_START
        fields = self._collect_fields_until_block_end(allow_no_end=True)
        name = fields.get("name", "")
        if not name:
            raise ParseError("#patch requires name= field", line=t.line)
        for k, v in fields.items():
            if k != "name":
                prog.sim_extensions[f"patch.{name}.{k}"] = v

    # -------- Type annotation parsing (P0-1.3) --------
    def _parse_type_annotation(self, prog: Program) -> None:
        """Parse #type annotations.

        Format: #type gene_name=Protein
        """
        self._advance()  # ANNOT_START
        fields = self._collect_fields_until_block_end(allow_no_end=True)
        for name, type_name in fields.items():
            prog.type_annotations[name] = type_name

    def _run_type_check(self, prog: Program) -> None:
        """Run type checking (P0-1.3).

        Checks:
        - whether the gene referenced by a type annotation exists
        - whether the source/target referenced by a regulation edge exists
        - whether the target referenced by a biological instruction exists
        """
        # Collect all defined symbols
        defined_genes = {g.name for g in prog.genes}
        defined_promoters = {p.name for p in prog.promoters}
        all_symbols = defined_genes | defined_promoters

        # Check type annotation references
        for name in prog.type_annotations:
            if name not in all_symbols:
                self._type_errors.append(
                    f"type annotation references undefined symbol {name!r}"
                )

        # Check regulation edge references
        for r in prog.regulations:
            if r.source not in all_symbols:
                self._type_errors.append(
                    f"regulation source {r.source!r} is undefined"
                )
            if r.target not in all_symbols:
                self._type_errors.append(
                    f"regulation target {r.target!r} is undefined"
                )

        # Check biological instruction references
        for inst in prog.bio_instructions:
            if inst.target not in all_symbols:
                self._type_errors.append(
                    f"#{inst.kind} target {inst.target!r} is undefined"
                )

        if self._type_errors:
            raise ParseError(
                "type check failed: " + "; ".join(self._type_errors)
            )

    # -------- Field collection --------
    def _collect_fields_until_block_end(self, allow_no_end: bool = False) -> dict[str, str]:
        """Collect FIELD tokens until ANNOT_END or the next annotation / EOF."""
        fields: dict[str, str] = {}
        while self._peek():
            t = self._peek()
            if t.kind == "ANNOT_END":
                self._advance()
                return fields
            if t.kind == "ANNOT_START":
                # Implicit end
                return fields
            if t.kind == "ARROW":
                # regulate's source->target is already handled in _parse_regulate
                return fields
            if t.kind == "FIELD":
                self._advance()
                key, _, val = t.value.partition("=")
                fields[key] = val
            elif t.kind == "EOF":
                if allow_no_end:
                    return fields
                raise ParseError("unexpected EOF inside annotation block",
                                 line=t.line, col=t.col)
            else:
                # CODON stream begins (gene block)
                return fields
        return fields

    # -------- ORF identification --------
    def _extract_orf(self, codons: list[Codon], gene_name: str,
                     start_line: int) -> list[Codon]:
        start_idx = None
        for i, c in enumerate(codons):
            if c.seq == "ATG":
                start_idx = i
                break
        if start_idx is None:
            raise ParseError(
                f"#gene {gene_name!r} has no START codon (ATG)",
                line=start_line,
            )
        for j in range(start_idx, len(codons)):
            if codons[j].seq in self.stop_codons:
                return codons[start_idx:j + 1]
        raise ParseError(
            f"#gene {gene_name!r} ORF not terminated by STOP codon",
            line=codons[-1].line if codons else start_line,
        )

    def _wrap_anon_gene(self) -> Gene:
        codons: list[Codon] = []
        while self._peek() and self._peek().kind == "CODON":
            ct = self._advance()
            codons.append(Codon(seq=ct.value, index=ct.codon_index, line=ct.line))
        name = f"__anon_{self.anon_counter}"
        self.anon_counter += 1
        orf = self._extract_orf(codons, name, codons[0].line if codons else 0)
        return Gene(name=name, promoter=None, codons=codons, orf=orf)

    # -------- token utilities --------
    def _peek(self, k: int = 0) -> Token:
        """Return the k-th lookahead token; returns the trailing EOF token when out of range (never None).

        The lexer always emits an ``EOF`` token at the end of the token stream,
        so ``self.toks`` is never empty. Out-of-range access clamps to that EOF
        token, so callers do not need to check for None.
        """
        idx = self.i + k
        if idx < len(self.toks):
            return self.toks[idx]
        return self.toks[-1]  # EOF token

    def _advance(self) -> Token:
        t = self.toks[self.i]
        self.i += 1
        return t

    def _expect(self, kind: str, value: str | None = None) -> Token:
        t = self._peek()
        if t.kind != kind or (value is not None and t.value != value):
            got = f"{t.kind} {t.value!r}" if t.kind != "EOF" else "EOF"
            raise ParseError(f"expected {kind} {value or ''}, got {got}",
                             line=t.line,
                             col=t.col)
        return self._advance()
