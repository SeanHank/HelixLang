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
    FieldDecl,
    Gene,
    LSystemDecl,
    MorphogenFeedback,
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
