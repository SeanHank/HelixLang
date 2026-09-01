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

from typing import Any

from helixlang.core.ast_nodes import (
    Codon,
    Gene,
    Program,
    UseDecl,
)
from helixlang.core.errors import ParseError, UnknownKeywordError
from helixlang.core.grammar_handlers import BIO_INSTRUCTION_KINDS, ParserGrammarMixin
from helixlang.core.grammar_registry import (
    BEFORE_SIM,
    AnnotationGrammar,
    GrammarDescriptor,
    _parse_quantity_stmt,
    _quantity_decompile,
    dict_entry_decompile,
    gem_inline_decompile,
    grammar_registry,
    prefix_decompile,
    sim_entry_decompile,
)
from helixlang.core.language import LanguageConfig
from helixlang.core.lexer import Lexer, Token
from helixlang.core.use_stmt import UseError, parse_use_line


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


class _ParserTokenHooks:
    """The documented token surface a raw plugin grammar may drive."""

    __slots__ = ("_parser",)

    def __init__(self, parser: Parser) -> None:
        self._parser = parser

    def advance(self) -> Token:
        return self._parser._advance()

    def peek(self) -> Token | None:
        return self._parser._peek()

    def expect(self, kind: str, what: str) -> Token:
        return self._parser._expect(kind, what)

    def collect_fields(self, allow_no_end: bool = False) -> dict[str, str]:
        return self._parser._collect_fields_until_block_end(
            allow_no_end=allow_no_end)


class Parser(ParserGrammarMixin):
    """Recursive-descent Parser.

    The non-core annotation grammars — biological instructions, GEM/ecosystem
    declarations, and the human-simulation annotations — live in
    :class:`~helixlang.core.grammar_handlers.ParserGrammarMixin` (doc/41
    Item 4); this class keeps the structural core: token-stream dispatch, the
    ``#use``/``#config``/``#sim``/``#type`` plumbing, field collection and ORF
    identification.
    """

    def __init__(self, tokens: list[Token],
                 stop_codons: set[str] | None = None,
                 config: LanguageConfig | None = None,
                 enable_type_check: bool = False):
        self.toks = [t for t in tokens if t.kind != "NEWLINE"]
        self.i = 0
        self.anon_counter = 0
        # Stop-codon truth comes from the LanguageConfig (doc/38 §4), never a
        # hardcoded literal: the default is the standard table's OP_HALT set.
        # ``stop_codons`` remains a backward-compatible override.
        if config is None:
            config = LanguageConfig.for_table("standard")
        elif stop_codons is not None:
            raise ParseError(
                "pass exactly one of 'config' or 'stop_codons' to Parser")
        self.config = config
        self.stop_codons = stop_codons or set(config.stop_codons)
        self.enable_type_check = enable_type_check
        self._type_errors: list[str] = []

    # -------- Entry point --------
    def parse(self) -> Program:
        prog = Program()
        while self._peek() and self._peek().kind != "EOF":
            t = self._peek()
            if t.kind == "ANNOT_START":
                # Registry-driven dispatch (doc/38 §5): the annotation
                # grammars — including plugin grammars — live in
                # grammar_registry, not in a literal dict here.
                grammar = grammar_registry.get(t.value)
                if grammar is None or grammar.parse is None:
                    # doc/36 F7: an unknown #keyword is never silently dropped —
                    # it is a hard SemanticError naming the keyword.
                    raise UnknownKeywordError(
                        f"unknown keyword #{t.value}", line=t.line, col=t.col)
                if grammar.requires_use and grammar.requires_use not in {
                        d.plugin for d in prog.use_directives}:
                    # doc/41 §4.2: a plugin grammar is inert until its `#use`.
                    raise UnknownKeywordError(
                        f"unknown keyword #{t.value}; enable it with "
                        f"`#use {grammar.requires_use}`",
                        line=t.line, col=t.col)
                _before_grammar = self.i
                grammar.parse(self, prog)
                if self.i == _before_grammar:
                    raise ParseError(
                        f"grammar hook for #{t.value} made no progress "
                        f"(missing advance past the annotation marker)",
                        line=t.line, col=t.col)
                continue
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

    # -------- Annotations (core: #config / #sim; the rest live in the mixin) --------
    def _parse_config(self, prog: Program) -> None:
        t = self._advance()
        fields = self._collect_fields_until_block_end(allow_no_end=True)
        if "ticks" in fields:
            prog.config.ticks = _parse_int(fields["ticks"], "ticks", t.line)
        if "output" in fields:
            prog.config.output = [s.strip() for s in fields["output"].split(",") if s.strip()]
        if "table" in fields:
            prog.config.table = fields["table"]
        if "ops_per_tick" in fields:
            prog.config.ops_per_tick = _parse_int(fields["ops_per_tick"], "ops_per_tick", t.line)
        if "react_steps" in fields:
            prog.config.react_steps = _parse_int(fields["react_steps"], "react_steps", t.line)
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
            # ``#config sim <fields>`` documents that the following fields are
            # sim parameters (12-helix-language-wiring.md §6.1); the marker is
            # consumed, not stored as an empty-valued sim parameter.
            "sim",
        }
        for k, v in fields.items():
            if k in consumed:
                continue
            # doc/38 §10: an empty-valued sim parameter is a silent no-op that
            # also breaks decompile→reparse canonicality (``key= next=val``
            # merges in the lexer) — reject it with a typed ParseError instead.
            if not v:
                raise ParseError(
                    f"#config {k}= has an empty value (sim parameters must be "
                    f"non-empty; use a quoted \"\" if truly empty)",
                    line=t.line)
            prog.config.sim[k] = v
            # doc/41 Item 5 Ring 2: a unit-tagged numeric value (``key=5min``
            # or ``key=5 min`` once a single token) resolves to a Quantity in
            # the parallel ``config.quantities`` map. ``sim`` keeps the
            # verbatim string for source round-trip; an unknown unit is a
            # typed ParseError, not a silent no-op.
            from helixlang.core.dimensions import UnitError, parse_quantity
            try:
                parsed = parse_quantity(v)
            except UnitError:
                # Not a unit-carrying numeric literal (bare string or unknown
                # unit); leave it as a plain sim parameter.
                continue
            if parsed.unit is not None:
                prog.config.quantities[k] = parsed

    def _parse_sim(self, prog: Program) -> None:
        """Parse #sim key=value ... (open extension point, wiring.md §8.6).

        Each #sim annotation merges its fields into ``Program.sim_extensions``,
        reserved for long-tail backends (e.g. ``#sim kind=spatial_dfba``).
        Inert until a backend registers it.
        """
        self._advance()  # ANNOT_START
        fields = self._collect_fields_until_block_end(allow_no_end=True)
        for k, v in fields.items():
            prog.extensions.extension_for(k, v).set(k, v)

    # -------- Type annotation parsing (P0-1.3) --------
    def _parse_type_annotation(self, prog: Program) -> None:
        """Parse #type annotations.

        Format: #type gene_name=Protein, or a unit-carrying
        ``#type gene_name=Float<µM>`` (doc/38 §7.4).  A conflicting repeat of
        the same symbol (``Protein`` then ``Float``) is a hard error naming
        the symbol — never a silent overwrite (doc/38 §7.2 constant solving).
        """
        self._advance()  # ANNOT_START
        fields = self._collect_fields_until_block_end(allow_no_end=True)
        for name, type_name in fields.items():
            prior = prog.type_annotations.get(name)
            if prior is not None and prior != type_name:
                raise ParseError(
                    f"conflicting #type for {name!r}: {prior!r} vs "
                    f"{type_name!r}")
            try:
                from helixlang.core.dimensions import UnitError
                from helixlang.core.errors import SemanticError as _SE
                from helixlang.core.type_system import parse_type_annotation
                parse_type_annotation(type_name)
            except (_SE, UnitError) as exc:  # bad type or unknown dimension
                raise ParseError(f"#type {name}={type_name!r}: {exc}") from exc
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

    # -------- Token hooks (documented Parser extension surface, doc/41 §4.2) --------
    @property
    def token_hooks(self) -> _ParserTokenHooks:
        """Protected token methods for ``body="raw"`` plugin grammars.

        A plugin's raw parse callable receives ``parser`` and may use these to
        drive the shared token machinery without reaching into ``Parser``
        privates — the parser still owns the token plumbing, the plugin owns
        the shape of its body (doc/41 §4.2).
        """
        return _ParserTokenHooks(self)

    # -------- ORF identification --------
    def _extract_orf(self, codons: list[Codon], gene_name: str,
                     start_line: int) -> list[Codon]:
        start_idx = None
        for i, c in enumerate(codons):
            if c.seq in self.config.start_codons:
                start_idx = i
                break
        if start_idx is None:
            raise ParseError(
                f"#gene {gene_name!r} has no START codon "
                f"({'/'.join(sorted(self.config.start_codons))})",
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


def parse_source(source: str, stop_codons: set[str] | None = None,
                 config: LanguageConfig | None = None) -> Program:
    """Parse a helix program directly from source text.

    Convenience wrapper that lexes ``source`` and runs the
    recursive-descent parser, returning the resulting
    :class:`~helixlang.core.ast_nodes.Program`.

    Args:
        source: .helix source text.
        stop_codons: legacy override; do not combine with ``config``.
        config: :class:`LanguageConfig` providing table, stop/start codons and
            the amino-acid map (doc/38 §4).  Defaults to ``standard``.
    """
    tokens = list(Lexer(source).tokens())
    return Parser(tokens, stop_codons=stop_codons, config=config).parse()


def _bio_parse(kind: str) -> Any:
    """Adapter matching the P0-1.1 bio-instruction signature to a grammar hook."""

    def _parse(parser: Parser, prog: Program) -> None:
        parser._parse_bio_instruction(prog, kind)

    return _parse


def register_core_grammars() -> None:
    """Register the built-in core annotation grammars (doc/38 §5).

    This is the complete grammar table that used to live as the literal
    annotation-dispatch dict inside ``Parser.parse``, now exposed through the
    shared :data:`grammar_registry`.  Decompile hooks let ``hxbc`` write
    ``sim_extensions`` back in their original annotation form; plugins add
    grammars via ``helixlang.core.grammar_registry.register_grammar``.

    Idempotent: core grammars are registered once per process (at import and
    again via :func:`~helixlang.core.grammar_registry.ensure_core_grammars`).
    """
    global _GRAMMARS_REGISTERED
    if _GRAMMARS_REGISTERED:
        return
    # #quantity is a descriptor grammar (doc/41 §6): the registry compiles the
    # declarative shape, the parser supplies two tiny hooks, and dimension
    # checking lives in the semantic layer — no bespoke parser dispatch.
    grammar_registry.register_descriptor(GrammarDescriptor(
        keyword="quantity",
        body="raw",
        parse=_parse_quantity_stmt,
        decompile=_quantity_decompile,
    ))
    grammars = [
        # ---- structural annotations backed by typed AST sections ----
        AnnotationGrammar("promoter", parse=Parser._parse_promoter),
        # #gene is also the pharmacology genotype record (no DNA block).
        AnnotationGrammar("gene", parse=Parser._parse_gene,
                          decompile=sim_entry_decompile("genes", "gene"),
                          list_valued_keys=frozenset({"genes"})),
        AnnotationGrammar("regulate", parse=Parser._parse_regulate),
        AnnotationGrammar("lsystem", parse=Parser._parse_lsystem),
        AnnotationGrammar("field", parse=Parser._parse_field),
        AnnotationGrammar("morphogen", parse=Parser._parse_morphogen),
        AnnotationGrammar("config", parse=Parser._parse_config),
        AnnotationGrammar("type", parse=Parser._parse_type_annotation),
        AnnotationGrammar("media", parse=Parser._parse_media),
        AnnotationGrammar("enzyme", parse=Parser._parse_enzyme),
        AnnotationGrammar("reaction", parse=Parser._parse_reaction),
        AnnotationGrammar("metabolite", parse=Parser._parse_metabolite),
        AnnotationGrammar("sim", parse=Parser._parse_sim),
        AnnotationGrammar("genome", parse=Parser._parse_genome),
        AnnotationGrammar("species", parse=Parser._parse_species),
        AnnotationGrammar("patch", parse=Parser._parse_patch),
        AnnotationGrammar("gem", parse=Parser._parse_gem,
                          decompile=gem_inline_decompile(),
                          extension_keys=frozenset(
                              {"gem_inline_genes", "gem_inline_genome"})),
        # ---- biological instructions (P0-1.1) ----
        *(AnnotationGrammar(kind, parse=_bio_parse(kind))
          for kind in sorted(BIO_INSTRUCTION_KINDS)),
        # ---- human-simulation annotations round-tripping via sim_extensions.
        # Prefix-valued grammars are emitted ahead of the generic #sim fallback
        # in the canonical .helix layout (historic ordering preserved).
        AnnotationGrammar("disease", parse=Parser._parse_disease,
                          decompile=prefix_decompile("disease_", "disease"),
                          extension_prefixes=frozenset({"disease_"}),
                          sim_section=BEFORE_SIM),
        AnnotationGrammar("person", parse=Parser._parse_person,
                          decompile=prefix_decompile("person_", "person"),
                          extension_prefixes=frozenset({"person_"}),
                          sim_section=BEFORE_SIM),
        AnnotationGrammar("trait", parse=Parser._parse_trait,
                          decompile=prefix_decompile("trait_", "trait"),
                          extension_prefixes=frozenset({"trait_"}),
                          sim_section=BEFORE_SIM),
        # ---- list/dict-valued grammars (emitted after the generic #sim loop).
        AnnotationGrammar("drug", parse=Parser._parse_drug,
                          decompile=sim_entry_decompile("drugs", "drug"),
                          list_valued_keys=frozenset({"drugs"})),
        AnnotationGrammar("disease_gene", parse=Parser._parse_disease_gene,
                          decompile=sim_entry_decompile("disease_genes",
                                                        "disease_gene"),
                          list_valued_keys=frozenset({"disease_genes"})),
        AnnotationGrammar("disease_metabolite",
                          parse=Parser._parse_disease_metabolite,
                          decompile=sim_entry_decompile("disease_metabolites",
                                                        "disease_metabolite"),
                          list_valued_keys=frozenset({"disease_metabolites"})),
        AnnotationGrammar("pd_effect", parse=Parser._parse_pd_effect,
                          decompile=sim_entry_decompile("pd_effects", "pd_effect"),
                          list_valued_keys=frozenset({"pd_effects"})),
        AnnotationGrammar("qsp_binding", parse=Parser._parse_qsp_binding,
                          decompile=sim_entry_decompile("qsp_bindings",
                                                        "qsp_binding"),
                          list_valued_keys=frozenset({"qsp_bindings"})),
        AnnotationGrammar("endocrine_config",
                          parse=Parser._parse_endocrine_config,
                          decompile=sim_entry_decompile("endocrine_configs",
                                                        "endocrine_config"),
                          list_valued_keys=frozenset({"endocrine_configs"})),
        AnnotationGrammar("immune_config", parse=Parser._parse_immune_config,
                          decompile=sim_entry_decompile("immune_configs",
                                                        "immune_config"),
                          list_valued_keys=frozenset({"immune_configs"})),
        AnnotationGrammar("tumor_biopsy", parse=Parser._parse_tumor_biopsy,
                          decompile=dict_entry_decompile("tumor_biopsy",
                                                         "tumor_biopsy"),
                          extension_keys=frozenset({"tumor_biopsy"})),
    ]
    grammar_registry.register_all(grammars)
    _GRAMMARS_REGISTERED = True


_GRAMMARS_REGISTERED = False


# Register on import: Parser dispatch depends on the registry being populated
# before the first parse.  Idempotent (re-registration of the same grammar
# objects is a no-op), so ensure_core_grammars() can safely call this again.
register_core_grammars()
