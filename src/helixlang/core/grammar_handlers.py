"""Grammar handlers extracted from the Parser core (doc/41 Item 4).

The :class:`ParserGrammarMixin` holds the parser's non-core annotation
grammars — biological instructions, GEM/ecosystem declarations, and the
human-simulation annotations.  ``Parser`` inherits this mixin so the
``@AnnotationGrammar(parse=Parser._parse_media)`` hooks registered in
``parser.register_core_grammars`` keep resolving to the same bound methods
(doc/41 §4 extensible parser).  Moving these bodies out shrinks ``parser.py``
to its structural core: the token stream driver, ``#use``/``#type``/``#config``
plumbing, the generic ``#sim`` extension point, field collection, and ORF
identification.

Each method uses the ``Parser`` token surface via ``self`` (``_advance``,
``_peek``, ``_expect``, ``_collect_fields_until_block_end``, ``_clean_value``,
``_append_sim_list``, ``_extract_orf``, ``anon_counter``) which the core class
still owns.
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
)
from helixlang.core.errors import ParseError

# Supported biological instruction kinds
BIO_INSTRUCTION_KINDS = frozenset({
    "crispr", "evolve", "methylate", "histone",
    "transcribe", "translate", "quorum",
})


def _parse_float(value: str, what: str, line: int) -> float:
    """Parse a numeric annotation field; garbage is a typed ParseError, never
    a bare ValueError (doc/38 §10 fuzzing invariant: typed errors only)."""
    try:
        return float(value)
    except ValueError:
        raise ParseError(
            f"invalid {what} {value!r}", line=line) from None


def _parse_int(value: str, what: str, line: int) -> int:
    """Parse an integer annotation field as a typed ParseError (see above)."""
    try:
        return int(value)
    except ValueError:
        raise ParseError(
            f"invalid {what} {value!r}", line=line) from None


class ParserGrammarMixin:
    """Non-core annotation grammars (see module docstring)."""

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
        # Pharmacogene declaration form (human-simulation backend):
        # "#gene name=... allele=... zygosity=..." with no DNA block records
        # one entry under sim_extensions["genes"] for genotype building.
        if "allele" in fields:
            if not fields.get("name"):
                raise ParseError("#gene requires name= field", line=t.line)
            entry = {k: self._clean_value(v) for k, v in fields.items()}
            prog.extensions.extension_for("genes", entry).append("genes", entry)
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
        t = self._advance()
        fields = self._collect_fields_until_block_end(allow_no_end=True)
        name = fields.get("name", "default")
        axiom = fields.get("axiom", "F")
        angle = _parse_float(fields.get("angle", "25"), "angle", t.line)
        step = _parse_float(fields.get("step", "1.0"), "step", t.line)
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
        t = self._advance()  # ANNOT_START
        fields = self._collect_fields_until_block_end(allow_no_end=True)
        size = _parse_int(fields.get("size", "32"), "size", t.line)
        F = _parse_float(fields.get("F", "0.035"), "F", t.line)
        k = _parse_float(fields.get("k", "0.065"), "k", t.line)
        Du = _parse_float(fields.get("Du", "0.16"), "Du", t.line)
        Dv = _parse_float(fields.get("Dv", "0.08"), "Dv", t.line)
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
        sub_coeff = _parse_float(fields.get("substrate_coeff", "-1"), "substrate_coeff", t.line)
        prod_coeff = _parse_float(fields.get("product_coeff", "1"), "product_coeff", t.line)
        lb = _parse_float(fields.get("lower_bound", "0"), "lower_bound", t.line)
        ub = _parse_float(fields.get("upper_bound", "1000"), "upper_bound", t.line)
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
        prog.extensions.extension_for("genome", "true").set("genome", "true")
        for k, v in fields.items():
            prog.extensions.extension_for(f"genome_{k}", v)\
                .set(f"genome_{k}", v)

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
                key = f"species.{name}.{k}"
                prog.extensions.extension_for(key, v).set(key, v)

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
        default ``glucose_minimal``),
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

        if "genome" in fields and self._peek().kind == "CODON":
            raise ParseError(
                f"#gem {fields['organism']}: use either a genome= field or "
                "a DNA code block, not both", line=t.line)

        # Store all fields under gem_ prefix
        for k, v in fields.items():
            prog.extensions.extension_for(f"gem_{k}", v).set(f"gem_{k}", v)

        # Inline DNA block (doc/38 §5): consuming the GENE_ID/CODON stream that
        # follows a #gem line is a property of the ``gem`` grammar, not a
        # Parser special case.
        if self._peek() and self._peek().kind in ("GENE_ID", "CODON"):
            gene_entries: list[list[str]] = []
            current_gene_id: str | None = None
            codons: list[str] = []
            while self._peek() and self._peek().kind in ("GENE_ID", "CODON"):
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
                prog.extensions.extension_for("gem_inline_genes", gene_entries)\
                    .set("gem_inline_genes", gene_entries)
                prog.extensions.extension_for(
                    "gem_inline_genome", "").set(
                        "gem_inline_genome", "".join(
                            s for _, s in gene_entries)
                    )
            # Consume #end if present
            if self._peek() and self._peek().kind == "ANNOT_END":
                self._advance()

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
                key = f"patch.{name}.{k}"
                prog.extensions.extension_for(key, v).set(key, v)

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
            key = f"person_{k}"
            prog.extensions.extension_for(key, v).set(
                key, self._clean_value(v))

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
            key = f"trait_{k}"
            prog.extensions.extension_for(key, v).set(
                key, self._clean_value(v))

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
            key = f"disease_{k}"
            prog.extensions.extension_for(key, v).set(
                key, self._clean_value(v))

    def _append_sim_list(
            self, prog: Program, key: str,
            fields: dict[str, str]) -> None:
        """Append one cleaned entry to a list-valued sim_extensions key.

        The list is created on first use so repeated annotations accumulate.
        """
        entry = {k: self._clean_value(v) for k, v in fields.items()}
        prog.extensions.extension_for(key, entry).append(key, entry)

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
        prog.extensions.extension_for("tumor_biopsy", fields).set(
            "tumor_biopsy", fields)
