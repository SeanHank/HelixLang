"""Lexer: a dual-mode DNA scanner.

DNA mode: aggregates CODON tokens every 3 bases; annotation mode: recognizes #ident / field=values / -> / #end.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from helixlang.core.errors import LexError

#: #keywords always treated as annotations (never gene-ID markers), matched
#: case-exact against the lexed identifier (doc/38 §5).  Plugin grammars are
#: recognized dynamically through the shared grammar registry instead of this
#: tuple, so a new #keyword needs no lexer edit.
ANNOTATION_KEYWORDS = frozenset({
    "use", "gem", "config", "sim", "end", "species", "genome",
    "media", "patch", "type", "enzyme", "metabolite",
    "regulate", "export", "reaction",
    "gene", "promoter", "lsystem", "morphogen",
    "crispr", "evolve", "methylate", "histone",
    "transcribe", "translate", "quorum",
    "gff", "sequence", "table", "field",
    "person", "trait", "disease", "disease_gene",
    "disease_metabolite", "drug", "pd_effect",
    "qsp_binding", "endocrine_config", "immune_config",
    "tumor_biopsy",
})


def _is_annotation_keyword(name: str) -> bool:
    """Core keywords plus anything a (plugin) grammar registered.

    ``name`` is matched case-exact: a registered grammar keyword is an
    annotation (``ANNOT_START``); anything else that looks like ``#ident`` is a
    gene-ID marker (``GENE_ID``).
    """
    if name in ANNOTATION_KEYWORDS:
        return True
    from helixlang.core import grammar_registry as _grammar

    return _grammar.grammar_registry.contains(name)


@dataclass(slots=True)
class Token:
    kind: str   # CODON | ANNOT_START | ANNOT_END | FIELD | ARROW | NEWLINE | EOF
    value: str
    line: int
    col: int
    codon_index: int = -1

    def __repr__(self) -> str:
        ci = f" #{self.codon_index}" if self.codon_index >= 0 else ""
        return f"Token({self.kind},{self.value!r} L{self.line}:{self.col}{ci})"


class Lexer:
    """Dual-mode scanner."""

    BASES = set("ACGTacgt")

    def __init__(self, source: str):
        # Preserves original case (annotation field names/values are case-sensitive); DNA bases are uppercased inside _scan_dna
        self.src = source
        self.pos = 0
        self.line = 1
        self.col = 1
        self.codon_counter = 0
        self._dna_mode = False  # True when inside a DNA block

    # -------- Public entry point --------
    def tokens(self) -> Iterator[Token]:
        while self.pos < len(self.src):
            c = self.src[self.pos]
            if c == '#':
                # Check for gene ID marker: #identifier (no '=' follows)
                peek = self.src[self.pos + 1:self.pos + 30]
                m_end = 0
                while m_end < len(peek) and (
                    peek[m_end].isalnum() or peek[m_end] in ('_', '.')
                ):
                    m_end += 1
                if m_end > 0:
                    gene_id = peek[:m_end]
                    # Check if '=' appears soon (annotation) or not (gene ID)
                    after_id = peek[m_end:m_end + 5].lstrip()
                    if not after_id.startswith('=') and not \
                            _is_annotation_keyword(gene_id):
                        # Gene ID marker — emit GENE_ID token
                        start_line, start_col = self.line, self.col
                        for _ in range(m_end + 1):
                            self._advance()
                        yield Token("GENE_ID", gene_id,
                                    start_line, start_col)
                    else:
                        yield from self._scan_annotation()
                else:
                    yield from self._scan_annotation()
            elif c in self.BASES:
                self._dna_mode = True
                yield from self._scan_dna()
            elif c in ' \t\r':
                self._advance()
            elif c == '\\' and self._at_line_continuation():
                # Python-style line continuation at the top level (e.g. a
                # continuation line leading into a DNA block or annotation).
                self._skip_line_continuation()
            elif c == '\n':
                yield Token("NEWLINE", "\\n", self.line, self.col)
                self._advance(newline=True)
            else:
                raise LexError(f"unexpected char {c!r}",
                               line=self.line, col=self.col)
        yield Token("EOF", "", self.line, self.col)

    # -------- DNA scanning --------
    def _skip_dna_gap(self) -> None:
        """Skip whitespace/newlines (and backslash line continuations) between
        DNA bases.  Inside a DNA block they are insignificant."""
        while self.pos < len(self.src):
            c = self.src[self.pos]
            if c in ' \t\r':
                self._advance()
            elif c == '\n':
                self._advance(newline=True)
            elif c == '\\' and self._at_line_continuation():
                self._skip_line_continuation()
            else:
                break

    def _scan_dna(self) -> Iterator[Token]:
        buf: list[str] = []
        start_line, start_col = self.line, self.col
        # Skip leading whitespace/newlines
        self._skip_dna_gap()
        # Collect DNA bases, skipping whitespace/newlines between them
        while self.pos < len(self.src) and self.src[self.pos] in self.BASES:
            buf.append(self.src[self.pos])
            self._advance()
            # Skip whitespace/newlines within DNA block
            self._skip_dna_gap()
        if len(buf) % 3 != 0:
            raise LexError(
                f"DNA length {len(buf)} not multiple of 3",
                line=start_line, col=start_col,
            )
        for i in range(0, len(buf), 3):
            codon = ''.join(buf[i:i+3]).upper()  # DNA bases are uppercased
            yield Token("CODON", codon, start_line, start_col,
                        codon_index=self.codon_counter)
            self.codon_counter += 1

    # -------- Annotation scanning --------
    def _scan_annotation(self) -> Iterator[Token]:
        start_line, start_col = self.line, self.col
        self._advance()  # skip '#'
        # '#' followed by space / '\t' / another '#' / newline / EOF -> line comment
        if (self.pos >= len(self.src) or self.src[self.pos] in ' \t#\r\n'):
            self._skip_to_newline()
            return
        name = self._read_ident()
        if name == "":
            raise LexError("missing annotation name after '#'",
                           line=start_line, col=start_col)
        if name.upper() == "END":
            yield Token("ANNOT_END", "#end", start_line, start_col)
            # Skip to end of line
            self._skip_to_newline()
            return
        if name.lower() == "use":
            # `use` is the plugin opt-in statement (doc/36 §4).  Emit a dedicated
            # USERDIRECTIVE token carrying the raw remainder of the line so the
            # parser can canonicalize plugin name + capability flags.  A trailing
            # '\' joins the directive with the next physical line (Python-style).
            rest_start = self.pos
            parts: list[str] = []
            while True:
                while self.pos < len(self.src) and self.src[self.pos] != '\n':
                    self._advance()
                piece = self.src[rest_start:self.pos].rstrip('\r')
                if piece.endswith('\\'):
                    parts.append(piece[:-1])
                    if self.pos < len(self.src):
                        self._advance(newline=True)
                    while self.pos < len(self.src) \
                            and self.src[self.pos] in ' \t':
                        self._advance()
                    rest_start = self.pos
                    continue
                parts.append(piece)
                break
            rest = ''.join(parts).strip()
            yield Token("USERDIRECTIVE", rest, start_line, start_col)
            if self.pos < len(self.src) and self.src[self.pos] == '\n':
                self._advance(newline=True)
            return
        yield Token("ANNOT_START", name.lower(), start_line, start_col)
        # Fields on the same line
        yield from self._scan_fields_on_line()

    def _scan_fields_on_line(self) -> Iterator[Token]:
        # Skip whitespace until end of line or a non-field character
        # Field format: ident=value | ident->ident
        while self.pos < len(self.src):
            c = self.src[self.pos]
            if c == '\\' and self._at_line_continuation():
                # Python-style line continuation: join with the next line
                self._skip_line_continuation()
                continue
            if c == '\n':
                self._advance(newline=True)
                return
            if c in ' \t\r':
                self._advance()
                continue
            if c == '#':
                # A new annotation starts, stop the current line's fields
                return
            # Field
            start_line, start_col = self.line, self.col
            ident = self._read_ident()
            if ident == "":
                # Skip unknown characters
                self._advance()
                continue
            # Check whether what follows is =, ->, or nothing
            self._skip_spaces()
            if self.pos < len(self.src) and self.src[self.pos] == '=':
                self._advance()
                self._skip_spaces()
                value = self._read_value()
                yield Token("FIELD", f"{ident}={value}", start_line, start_col)
            elif self.pos + 1 < len(self.src) and self.src[self.pos:self.pos+2] == '->':
                self._advance()
                self._advance()
                self._skip_spaces()
                if self._at_line_continuation():
                    self._skip_line_continuation()
                target = self._read_ident()
                yield Token("ARROW", f"{ident}->{target}", start_line, start_col)
            else:
                # Bare identifier field (e.g., stdout)
                yield Token("FIELD", f"{ident}=", start_line, start_col)

    def _read_ident(self) -> str:
        start = self.pos
        while self.pos < len(self.src) and (self.src[self.pos].isalnum()
                                            or self.src[self.pos] in '_-.'):
            self._advance()
        return self.src[start:self.pos]

    def _read_value(self) -> str:
        # Value can be a number, string, or identifier; read until whitespace/newline/#
        if self.pos < len(self.src) and self.src[self.pos] == '"':
            # String
            self._advance()
            start = self.pos
            while self.pos < len(self.src) and self.src[self.pos] != '"':
                self._advance()
            s = self.src[start:self.pos]
            if self.pos < len(self.src):
                self._advance()  # closing "
            return f'"{s}"'
        # Unquoted value; a trailing '\' joins the value with the next line
        # (the backslash and the newline are deleted, Python-style).
        parts: list[str] = []
        start = self.pos
        while self.pos < len(self.src):
            c = self.src[self.pos]
            if c in ' \t\r\n#':
                break
            if c == '\\' and self._at_line_continuation():
                parts.append(self.src[start:self.pos])
                self._skip_line_continuation()
                start = self.pos
                continue
            self._advance()
        parts.append(self.src[start:self.pos])
        return ''.join(parts)

    def _skip_spaces(self) -> None:
        while self.pos < len(self.src) and self.src[self.pos] in ' \t':
            self._advance()

    def _at_line_continuation(self) -> bool:
        """True if the current char is a backslash that is the very last
        character of the physical line (Python-style line continuation: the
        backslash must be followed only by the newline, optionally a CRLF)."""
        if self.pos >= len(self.src) or self.src[self.pos] != '\\':
            return False
        nxt = self.src[self.pos + 1] if self.pos + 1 < len(self.src) else ''
        if nxt == '\n':
            return True
        if nxt == '\r' and self.pos + 2 < len(self.src) \
                and self.src[self.pos + 2] == '\n':
            return True
        return False

    def _skip_line_continuation(self) -> None:
        """Consume '\\' + newline and the next line's leading whitespace, as if
        the two physical lines had been joined into one logical line."""
        self._advance()                      # backslash
        if self.pos < len(self.src) and self.src[self.pos] == '\r':
            self._advance()                  # CR (CRLF line ending)
        if self.pos < len(self.src) and self.src[self.pos] == '\n':
            self._advance(newline=True)      # newline (updates line/col)
        while self.pos < len(self.src) and self.src[self.pos] in ' \t':
            self._advance()                  # next line's indentation

    def _skip_to_newline(self) -> None:
        while self.pos < len(self.src) and self.src[self.pos] != '\n':
            self._advance()
        if self.pos < len(self.src):
            self._advance(newline=True)

    # -------- Position management --------
    def _advance(self, newline: bool = False) -> None:
        c = self.src[self.pos] if self.pos < len(self.src) else ''
        self.pos += 1
        if newline or c == '\n':
            self.line += 1
            self.col = 1
        else:
            self.col += 1
