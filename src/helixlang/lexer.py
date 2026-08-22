"""Lexer: a dual-mode DNA scanner.

DNA mode: aggregates CODON tokens every 3 bases; annotation mode: recognizes #ident / field=values / -> / #end.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from helixlang.errors import LexError


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
                    if not after_id.startswith('=') and gene_id not in (
                        "gem", "config", "sim", "end", "species", "genome",
                        "media", "patch", "type", "enzyme", "metabolite",
                        "regulate", "export", "reaction",
                        "gene", "promoter", "lsystem", "morphogen",
                        "crispr", "evolve", "methylate", "histone",
                        "transcribe", "translate", "quorum",
                        "gff", "sequence", "table", "field",
                    ):
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
            elif c == '\n':
                yield Token("NEWLINE", "\\n", self.line, self.col)
                self._advance(newline=True)
            else:
                raise LexError(f"unexpected char {c!r}",
                               line=self.line, col=self.col)
        yield Token("EOF", "", self.line, self.col)

    # -------- DNA scanning --------
    def _scan_dna(self) -> Iterator[Token]:
        buf: list[str] = []
        start_line, start_col = self.line, self.col
        # Skip leading whitespace/newlines
        while self.pos < len(self.src) and self.src[self.pos] in ' \t\r\n':
            if self.src[self.pos] == '\n':
                self._advance(newline=True)
            else:
                self._advance()
        # Collect DNA bases, skipping whitespace/newlines between them
        while self.pos < len(self.src) and self.src[self.pos] in self.BASES:
            buf.append(self.src[self.pos])
            self._advance()
            # Skip whitespace/newlines within DNA block
            while self.pos < len(self.src) and self.src[self.pos] in ' \t\r\n':
                if self.src[self.pos] == '\n':
                    self._advance(newline=True)
                else:
                    self._advance()
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
        yield Token("ANNOT_START", name.lower(), start_line, start_col)
        # Fields on the same line
        yield from self._scan_fields_on_line()

    def _scan_fields_on_line(self) -> Iterator[Token]:
        # Skip whitespace until end of line or a non-field character
        # Field format: ident=value | ident->ident
        while self.pos < len(self.src):
            c = self.src[self.pos]
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
        start = self.pos
        while self.pos < len(self.src) and self.src[self.pos] not in ' \t\r\n#':
            self._advance()
        return self.src[start:self.pos]

    def _skip_spaces(self) -> None:
        while self.pos < len(self.src) and self.src[self.pos] in ' \t':
            self._advance()

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
