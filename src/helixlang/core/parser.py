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

from helixlang.core.ast_nodes import (
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
    ReactionDecl,
    Regulation,
    UseDecl,
)
from helixlang.core.errors import ParseError, UnknownKeywordError
from helixlang.core.lexer import Lexer, Token
from helixlang.core.use_stmt import UseError, parse_use_line

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
                    "reaction": self._parse_reaction,
                    "metabolite": self._parse_metabolite,
                    "sim": self._parse_sim,
                    "genome": self._parse_genome,
                    "species": self._parse_species,
                    "patch": self._parse_patch,
                    "gem": self._parse_gem,
                    "person": self._parse_person,
                    "trait": self._parse_trait,
                    "disease": self._parse_disease,
                    "disease_gene": self._parse_disease_gene,
                    "disease_metabolite": self._parse_disease_metabolite,
                    "drug": self._parse_drug,
                    "pd_effect": self._parse_pd_effect,
                    "qsp_binding": self._parse_qsp_binding,
                    "endocrine_config": self._parse_endocrine_config,
                    "immune_config": self._parse_immune_config,
                    "tumor_biopsy": self._parse_tumor_biopsy,
                }.get(t.value)
                # Biological instructions (P0-1.1)
                if t.value in BIO_INSTRUCTION_KINDS:
                    self._parse_bio_instruction(prog, t.value)
                    continue
                if handler is None:
                    # doc/36 F7: an unknown #keyword is never silently dropped —
                    # it is a hard SemanticError naming the keyword.
                    raise UnknownKeywordError(
                        f"unknown keyword #{t.value}", line=t.line, col=t.col)
                handler(prog)
                # After #gem, consume any inline DNA block (GENE_ID + CODON)
                if t.value == "gem" and self._peek() and \
                        self._peek().kind in ("GENE_ID", "CODON", "NEWLINE"):
                    # Skip leading NEWLINEs
                    while self._peek() and self._peek().kind == "NEWLINE":
                        self._advance()
                    # We're in an inline DNA block — collect genes + sequences
                    gene_entries: list[list[str]] = []
                    current_gene_id: str | None = None
                    codons: list[str] = []
                    while self._peek().kind in ("GENE_ID", "CODON", "NEWLINE"):
                        if self._peek().kind == "NEWLINE":
                            self._advance()
                            continue
                        if self._peek().kind == "GENE_ID":
                            if codons:
                                gene_entries.append([
                                    current_gene_id or f"gene_{len(gene_entries)}",
                                    "".join(codons),
                                ])
                                codons = []
                            gt = self._advance()
                            current_gene_id = gt.value
                        else:
                            ct = self._advance()
                            codons.append(ct.value)
                    if codons:
                        gene_entries.append([
                            current_gene_id or f"gene_{len(gene_entries)}",
                            "".join(codons),
                        ])
                    if gene_entries:
                        prog.sim_extensions["gem_inline_genes"] = gene_entries
                        prog.sim_extensions["gem_inline_genome"] = "".join(
                            s for _, s in gene_entries
                        )
                    # Consume #end if present
                    if self._peek() and self._peek().kind == "ANNOT_END":
                        self._advance()
                    # Consume trailing NEWLINE
                    if self._peek() and self._peek().kind == "NEWLINE":
                        self._advance()
            elif t.kind == "USERDIRECTIVE":
                prog.use_directives.append(self._parse_use(t))
            elif t.kind == "CODON":
                prog.genes.append(self._wrap_anon_gene())
            elif t.kind == "NEWLINE":
                pass  # skip blank lines
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

    # -------- Plugin opt-in: `use <plugin> [--flag ...]` (doc/36 §4) --------
    def _parse_use(self, t: Token) -> UseDecl:
        """Parse the raw remainder of a ``#use`` line into a :class:`UseDecl`."""
        self._advance()  # consume USERDIRECTIVE
        try:
            d = parse_use_line(t.value, line=t.line, col=t.col)
        except UseError as e:
            raise ParseError(f"#use: {e}", line=t.line, col=t.col) from e
        return UseDecl(plugin=d.plugin, flags=frozenset(d.flags),
                       model=d.model, line=d.line, col=d.col)

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
        # Pharmacogene declaration form (human-simulation backend):
        # "#gene name=... allele=... zygosity=..." with no DNA block records
        # one entry under sim_extensions["genes"] for genotype building.
        if "allele" in fields:
            if not fields.get("name"):
                raise ParseError("#gene requires name= field", line=t.line)
            entry = {k: self._clean_value(v) for k, v in fields.items()}
            existing = prog.sim_extensions.setdefault("genes", [])
            if isinstance(existing, list):
                existing.append(entry)
            else:
                prog.sim_extensions["genes"] = [entry]
            return
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
        """Parse #enzyme gene=gltA reaction=CS [kcat=2800] [km=0.5].

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
        km: float | None = None
        if "km" in fields:
            try:
                km = float(fields["km"])
            except ValueError as e:
                raise ParseError(
                    f"invalid km {fields['km']!r}: {e}", line=t.line) from None
        prog.enzymes.append(EnzymeDecl(gene=gene, reaction=reaction, kcat=kcat, km=km))

    def _parse_reaction(self, prog: Program) -> None:
        """Parse #reaction id=PGI name=PGI substrate=g6p product=f6p.

        Direct reaction definition for DSL-authored metabolic networks.
        Fields:
          id        (required) reaction identifier
          name      human-readable name
          substrate reactant metabolite id
          substrate_coeff  reactant stoichiometric coefficient (default -1)
          product   product metabolite id
          product_coeff    product stoichiometric coefficient (default 1)
          lower_bound  flux lower bound (default 0, negative = reversible)
          upper_bound  flux upper bound (default 1000)
          subsystem pathway / subsystem tag
          reversible   shorthand: if true, lower_bound = -upper_bound
        """
        t = self._advance()  # ANNOT_START
        fields = self._collect_fields_until_block_end(allow_no_end=True)
        rxn_id = fields.get("id", "")
        if not rxn_id:
            raise ParseError("#reaction requires id= field", line=t.line)
        name = fields.get("name", rxn_id)
        substrate = fields.get("substrate", "")
        product = fields.get("product", "")
        sub_coeff = float(fields.get("substrate_coeff", "-1"))
        prod_coeff = float(fields.get("product_coeff", "1"))
        lb = float(fields.get("lower_bound", "0"))
        ub = float(fields.get("upper_bound", "1000"))
        subsystem = fields.get("subsystem", "other")
        reversible = fields.get("reversible", "false").lower() in ("true", "1", "yes")
        if reversible:
            lb = -ub
        stoich: dict[str, float] = {}
        if substrate:
            stoich[substrate] = sub_coeff
        if product:
            stoich[product] = prod_coeff
        prog.reactions.append(ReactionDecl(
            id=rxn_id, name=name,
            substrate=substrate, substrate_coeff=sub_coeff,
            product=product, product_coeff=prod_coeff,
            lower_bound=lb, upper_bound=ub,
            subsystem=subsystem, reversible=reversible,
        ))

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

    def _parse_gem(self, prog: Program) -> None:
        """Parse #gem organism=... [genome=... | DNA block].

        GEM reconstruction pipeline declaration (doc/20 §6.5).
        Fields are namespaced into ``Program.sim_extensions`` under ``gem_``:

            #gem organism=e_coli_k12 genome=genome.fasta
            #gem organism=b_subtilis use_database=true include_spontaneous=true

        **Inline DNA support** (doc/20 §12): the genome can be given as a
        DNA code block (ATCG sequences) instead of a ``genome=`` file path:

            #gem organism=e_coli_k12
            ATGAAACGCATTAGCACCACCATTACCACCACCATCAC...
            #end

        The DNA block is concatenated and stored as ``gem_inline_genome``
        (a FASTA-formatted string written to a temp file at runtime).

        Supported fields: ``organism`` (organism identifier),
        ``genome`` (path to genome FASTA file),
        ``use_database`` (use known regulatory interactions, default true),
        ``include_spontaneous`` (include spontaneous reactions, default true),
        ``gapfill`` (run gap-filling, default true),
        ``target_organism`` (target organism for BRENDA lookup),
        ``medium`` (growth medium preset: ``glucose_minimal``, ``lb``,
        ``bg11``, or ``custom`` to use ``#media`` annotations;
        default ``glucose_minimal`),
        ``dynamic`` (enable dFBA, default false),
        ``duration`` (simulation hours, default 24.0),
        ``dt`` (time step hours, default 0.1),
        ``expression`` (enable expression inference, default false),
        ``use_full_model`` (load a full genome-scale model from BiGG
        instead of rebuilding from genome; default false.  Requires
        ``organism`` to be in the organism registry, e.g.
        ``e_coli_k12`` → iML1515, ``synechocystis_pcc6803`` → iJN678).

        Setting ``#config backend=gem`` triggers the pipeline automatically;
        using ``#gem`` alone is inert unless the backend is ``gem``.
        """
        t = self._advance()  # ANNOT_START
        fields = self._collect_fields_until_block_end(allow_no_end=True)
        if "organism" not in fields:
            raise ParseError(
                "#gem requires organism= field", line=t.line)

            # Inline DNA support: if no genome= field and DNA follows
            if "genome" not in fields and self._peek().kind in ("CODON", "GENE_ID"):
                # Collect DNA sequences, tracking gene IDs from #gene_id markers
                gene_entries: list[list[str]] = []
                current_gene_id: str | None = None
                codons: list[str] = []

                while self._peek().kind in ("CODON", "GENE_ID"):
                    if self._peek().kind == "GENE_ID":
                        # Save previous gene if any
                        if codons:
                            seq = "".join(codons)
                            gene_entries.append([
                                current_gene_id or f"gene_{len(gene_entries)}",
                                seq,
                            ])
                        codons = []
                    gt = self._advance()
                    current_gene_id = gt.value
                else:  # CODON
                    ct = self._advance()
                    codons.append(ct.value)

            # Save last gene
            if codons:
                seq = "".join(codons)
                gene_entries.append([
                    current_gene_id or f"gene_{len(gene_entries)}",
                    seq,
                ])

            if gene_entries:
                # Multiple genes: store as structured list
                fields["inline_genome"] = "".join(
                    seq for _, seq in gene_entries
                )
                # Store gene list in sim_extensions directly (Any-typed)
                prog.sim_extensions["gem_inline_genes"] = gene_entries
            else:
                fields["inline_genome"] = ""

            if self._peek().kind == "ANNOT_END":
                self._advance()

        if "genome" in fields and self._peek().kind == "CODON":
            raise ParseError(
                f"#gem {fields['organism']}: use either a genome= field or "
                "a DNA code block, not both", line=t.line)

        # Store all fields under gem_ prefix
        for k, v in fields.items():
            prog.sim_extensions[f"gem_{k}"] = v

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

    # -------- Human patient simulation annotations --------
    @staticmethod
    def _clean_value(v: str) -> str:
        """Strip surrounding double quotes from a lexer FIELD value."""
        if len(v) >= 2 and v.startswith('"') and v.endswith('"'):
            return v[1:-1]
        return v

    def _parse_person(self, prog: Program) -> None:
        """Parse #person name=... age=... (human patient simulation).

        Patient demographics for the human-simulation backend, merged into
        ``Program.sim_extensions`` under a ``person_`` prefix (the same open
        extension point as ``#sim``/``#genome``):

            #person name=John age=55 sex=male weight=82 height=175 \
ethnicity=european

        Values are stored verbatim as strings; inert until a backend
        consumes them.
        """
        self._advance()  # ANNOT_START
        fields = self._collect_fields_until_block_end(allow_no_end=True)
        for k, v in fields.items():
            prog.sim_extensions[f"person_{k}"] = self._clean_value(v)

    def _parse_trait(self, prog: Program) -> None:
        """Parse #trait smoking=former pack_years=10 ... (patient lifestyle).

        Lifestyle traits merged into ``Program.sim_extensions`` under a
        ``trait_`` prefix:

            #trait smoking=former pack_years=10 alcohol=5 exercise=moderate

        Values are stored verbatim as strings; inert until a backend
        consumes them.
        """
        self._advance()  # ANNOT_START
        fields = self._collect_fields_until_block_end(allow_no_end=True)
        for k, v in fields.items():
            prog.sim_extensions[f"trait_{k}"] = self._clean_value(v)

    def _parse_disease(self, prog: Program) -> None:
        """Parse #disease name=... category=... severity=0.7 onset_age=45.

        Disease declaration merged into ``Program.sim_extensions`` under a
        ``disease_`` prefix:

            #disease name="type 2 diabetes" category=metabolic_overload \
severity=0.7 onset_age=45

        Values are stored verbatim as strings; inert until a backend
        consumes them.
        """
        self._advance()  # ANNOT_START
        fields = self._collect_fields_until_block_end(allow_no_end=True)
        for k, v in fields.items():
            prog.sim_extensions[f"disease_{k}"] = self._clean_value(v)

    def _append_sim_list(
            self, prog: Program, key: str,
            fields: dict[str, str]) -> None:
        """Append one cleaned entry to a list-valued sim_extensions key.

        The list is created on first use so repeated annotations accumulate.
        """
        entry = {k: self._clean_value(v) for k, v in fields.items()}
        existing = prog.sim_extensions.setdefault(key, [])
        if isinstance(existing, list):
            existing.append(entry)
        else:
            prog.sim_extensions[key] = [entry]

    def _parse_disease_gene(self, prog: Program) -> None:
        """Parse #disease_gene gene=INSR type=downregulate activity=0.3.

        Gene perturbation associated with the declared ``#disease``; each
        annotation appends one dict to
        ``Program.sim_extensions["disease_genes"]`` (created on first use):

            #disease_gene gene=INSR type=downregulate activity=0.3

        Requires ``gene=``; inert until a backend consumes it.
        """
        t = self._advance()  # ANNOT_START
        fields = self._collect_fields_until_block_end(allow_no_end=True)
        if not fields.get("gene"):
            raise ParseError("#disease_gene requires gene= field",
                             line=t.line)
        self._append_sim_list(prog, "disease_genes", fields)

    def _parse_disease_metabolite(self, prog: Program) -> None:
        """Parse #disease_metabolite id=glucose type=accumulate concentration=7.8.

        Metabolite perturbation associated with the declared ``#disease``;
        each annotation appends one dict to
        ``Program.sim_extensions["disease_metabolites"]`` (created on first
        use):

            #disease_metabolite id=glucose type=accumulate concentration=7.8 \
normal=5.5

        Requires ``id=``; inert until a backend consumes it.
        """
        t = self._advance()  # ANNOT_START
        fields = self._collect_fields_until_block_end(allow_no_end=True)
        if not fields.get("id"):
            raise ParseError("#disease_metabolite requires id= field",
                             line=t.line)
        self._append_sim_list(prog, "disease_metabolites", fields)

    def _parse_drug(self, prog: Program) -> None:
        """Parse #drug name=metformin smiles=... formula=... dose=500 ...

        Drug declaration; each annotation appends one dict to
        ``Program.sim_extensions["drugs"]`` (created on first use):

            #drug name=metformin smiles=CN(C)C(=N)NC(=N)N formula=C4H11N5 \
mw=129.16 dose=500 route=oral interval=8 duration=90

        Requires ``name=``; inert until a backend consumes it.
        """
        t = self._advance()  # ANNOT_START
        fields = self._collect_fields_until_block_end(allow_no_end=True)
        if not fields.get("name"):
            raise ParseError("#drug requires name= field", line=t.line)
        self._append_sim_list(prog, "drugs", fields)

    def _parse_pd_effect(self, prog: Program) -> None:
        """Parse #pd_effect drug=metformin target=BIOMASSReaction ec50=5 ...

        Pharmacodynamic effect linking a previously declared ``#drug`` to a
        model target; each annotation appends one dict to
        ``Program.sim_extensions["pd_effects"]`` (created on first use):

            #pd_effect drug=metformin target=BIOMASSReaction ec50=5 emax=0.6 \
            hill=1.5

        Requires ``drug=``; inert until a backend consumes it.
        """
        t = self._advance()  # ANNOT_START
        fields = self._collect_fields_until_block_end(allow_no_end=True)
        if not fields.get("drug"):
            raise ParseError("#pd_effect requires drug= field", line=t.line)
        self._append_sim_list(prog, "pd_effects", fields)

    def _parse_qsp_binding(self, prog: Program) -> None:
        """Parse #qsp_binding drug=... kind=mass_action|tmdd|competitive ...

        QSP-style pharmacodynamic binding model (doc/31 §2.5).

            #qsp_binding drug=trastuzumab kind=tmdd kss_nM=2.0 emax=0.9
            #qsp_binding drug=imatinib kind=mass_action kd_nM=1.0 emax=0.85
            #qsp_binding drug=antagonist kind=competitive kd_agonist=10 ki=5

        Requires ``drug=`` and ``kind=``.
        """
        t = self._advance()
        fields = self._collect_fields_until_block_end(allow_no_end=True)
        if not fields.get("drug"):
            raise ParseError("#qsp_binding requires drug= field", line=t.line)
        if not fields.get("kind"):
            raise ParseError("#qsp_binding requires kind= field (mass_action|tmdd|competitive)", line=t.line)
        self._append_sim_list(prog, "qsp_bindings", fields)

    def _parse_endocrine_config(self, prog: Program) -> None:
        """Parse #endocrine_config axis=... severity=0.5 ...

        Configure endocrine axis parameters (doc/31 §2.6).

            #endocrine_config axis=diabetes severity=0.7
            #endocrine_config axis=addison severity=0.3
            #endocrine_config axis=hypothyroid severity=0.5
            #endocrine_config axis=stress level=0.8

        Requires ``axis=``.
        """
        t = self._advance()
        fields = self._collect_fields_until_block_end(allow_no_end=True)
        if not fields.get("axis"):
            raise ParseError("#endocrine_config requires axis= field", line=t.line)
        self._append_sim_list(prog, "endocrine_configs", fields)

    def _parse_immune_config(self, prog: Program) -> None:
        """Parse #immune_config parameter=value ...

        Configure immune system parameters (doc/31 §2.4).

            #immune_config infection_severity=0.8
            #immune_config autoimmune_activation=0.5
            #immune_config immunosuppression=0.3

        No required fields; all parameters optional.
        """
        self._advance()
        fields = self._collect_fields_until_block_end(allow_no_end=True)
        self._append_sim_list(prog, "immune_configs", fields)

    def _parse_tumor_biopsy(self, prog: Program) -> None:
        """Parse #tumor_biopsy mutation=EGFR_L858R amplification=HER2 ...

        Tumor molecular profile for biomarker-driven cancer therapy.
        Multiple values for the same key use comma separation:
            mutation=EGFR_L858R,TP53_R175H
            amplification=HER2
            fusion=EML4-ALK
            pd_l1_expression=0.6
            msi_status=MSS
            tmb_per_mb=5.2
            hr_status=HRD

        Stores into ``Program.sim_extensions["tumor_biopsy"]`` as a dict.
        """
        self._advance()  # ANNOT_START
        fields = self._collect_fields_until_block_end(allow_no_end=True)
        prog.sim_extensions["tumor_biopsy"] = fields

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
            elif t.kind == "CODON":
                # CODON stream begins (gene block)
                return fields
            elif t.kind == "GENE_ID":
                # Gene marker inside DNA block — return to caller
                return fields
            elif t.kind == "NEWLINE":
                self._advance()
            else:
                # Unknown token — return to caller
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


def parse_source(source: str, stop_codons: set[str] | None = None) -> Program:
    """Parse a helix program directly from source text.

    Convenience wrapper that lexes ``source`` and runs the
    recursive-descent parser, returning the resulting
    :class:`~helixlang.core.ast_nodes.Program`.
    """
    tokens = list(Lexer(source).tokens())
    return Parser(tokens, stop_codons=stop_codons).parse()
