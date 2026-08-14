"""HelixLang error hierarchy. All exceptions carry codon-level position for diagnostics."""


class HelixError(Exception):
    """Base class for all HelixLang exceptions."""

    def __init__(self, msg: str, *, line: int = 0, col: int = 0, codon_index: int = -1):
        super().__init__(msg)
        self.msg = msg
        self.line = line
        self.col = col
        self.codon_index = codon_index

    def __str__(self) -> str:
        loc = f"line {self.line}" if self.line else "<unknown>"
        if self.codon_index >= 0:
            loc += f" codon #{self.codon_index}"
        return f"[{type(self).__name__} @ {loc}] {self.msg}"


class LexError(HelixError):
    """Lexical error (e.g., DNA length is not a multiple of 3)."""


class ParseError(HelixError):
    """Syntax error (e.g., annotation block unclosed, ORF missing terminator)."""


class SemanticError(HelixError):
    """Semantic error (e.g., duplicate symbols, references to undefined genes)."""


class CompileError(HelixError):
    """Compile-time error (e.g., unknown codon, constant pool out of bounds)."""


class RegulationError(HelixError):
    """Regulation graph error (e.g., regulation references an undefined gene)."""


class RuntimeHelixError(HelixError):
    """Runtime error (stack underflow, unknown opcode). Note: deliberately avoids Python's built-in RuntimeError."""


class BioError(HelixError):
    """Biology module error (invalid input or solver failure in biological computation layers such as CRISPR/metabolism/protein structure).

    Used to replace bare ``raise ValueError`` in the bio module, so biology-side errors
    join the unified :class:`HelixError` hierarchy, letting the server layer's
    ``@errorhandler`` handle them by severity (bad user input -> 400, implementation bug -> 500).
    """


class SimConfigError(HelixError):
    """Simulation-backend configuration error (bad `#config sim` value, unknown
    enum, malformed float/int/dict coercion) raised by :mod:`helixlang.sim_runtime`.

    Naming the offending key so the CLI/server can point at the source line.
    """
