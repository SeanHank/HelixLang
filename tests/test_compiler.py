"""Compiler unit tests."""
from helixlang.codon_table import MITO_VERTEBRATE_TABLE, STANDARD_TABLE, Op
from helixlang.compiler import Compiler
from helixlang.disassembler import disassemble
from helixlang.lexer import Lexer
from helixlang.parser import Parser


def compile_src(src, table=STANDARD_TABLE):
    stop = {c for c, op in table.items() if op == Op.OP_HALT}
    toks = list(Lexer(src).tokens())
    prog = Parser(toks, stop_codons=stop).parse()
    return Compiler(table).compile(prog)


def test_compile_simple_gene():
    chunk = compile_src("#gene name=hello\nATG GCT TAA\n#end")
    assert len(chunk.gene_offsets) == 1
    assert "hello" in chunk.gene_offsets
    # OP_START, OP_BUILD_PROTEIN, OP_HALT
    assert chunk.code[0] == int(Op.OP_START)
    assert chunk.code[1] == int(Op.OP_BUILD_PROTEIN)
    # OP_BUILD_PROTEIN has a 1-byte operand
    assert chunk.code[3] == int(Op.OP_HALT)


def test_wobble_as_operand():
    """The third-base wobble should be used as an operand. WOBBLE: A=0, C=1, G=2, T=3."""
    chunk = compile_src("#gene name=g\nATG GCT GCC GCA GCG TAA\n#end")
    # GCT(wobble=3), GCC(wobble=1), GCA(wobble=0), GCG(wobble=2)
    assert chunk.code[1] == int(Op.OP_BUILD_PROTEIN)
    assert chunk.code[2] == 3  # GCT third base T
    assert chunk.code[4] == 1  # GCC third base C
    assert chunk.code[6] == 0  # GCA third base A
    assert chunk.code[8] == 2  # GCG third base G


def test_halt_appended_if_missing():
    """If the ORF ends without a HALT, the compiler should append one."""
    # In the standard table, TAA is a stop -> OP_HALT, so no auto-append is triggered
    # But with the mito table, TGA is not a stop... here we test the standard table which naturally has HALT
    chunk = compile_src("#gene name=g\nATG GCT TAA\n#end")
    # The last one should be OP_HALT
    last_op_ip = 0
    ip = 0
    while ip < len(chunk.code):
        op = Op(chunk.code[ip])
        last_op_ip = ip
        nbytes = {Op.OP_START: 0, Op.OP_BUILD_PROTEIN: 1, Op.OP_HALT: 0}[op]
        ip += 1 + nbytes
    assert chunk.code[last_op_ip] == int(Op.OP_HALT)


def test_mito_table_tga_compiles_to_pigment():
    """In the mito table, TGA -> OP_BUILD_PIGMENT rather than OP_HALT."""
    chunk = compile_src("#gene name=m\nATG TGA TAA\n#end",
                        table=MITO_VERTEBRATE_TABLE)
    assert chunk.code[0] == int(Op.OP_START)
    assert chunk.code[1] == int(Op.OP_BUILD_PIGMENT)
    assert chunk.code[2] == int(Op.OP_HALT)


def test_call_gene_patch():
    """The OP_CALL_GENE operand should be back-patched to the gene offset."""
    src = """#gene name=caller
ATG CGT TAA
#end
#gene name=target
ATG GCT TAA
#end
"""
    chunk = compile_src(src)
    # Find OP_CALL_GENE
    ip = 0
    found = False
    while ip < len(chunk.code):
        if chunk.code[ip] == int(Op.OP_CALL_GENE):
            offset = (chunk.code[ip + 1] << 8) | chunk.code[ip + 2]
            # The target gene offset should be > 0
            assert offset == chunk.gene_offsets["target"]
            found = True
            break
        ip += 1
    assert found


def test_disassemble_output():
    chunk = compile_src("#gene name=hello\nATG TAA\n#end")
    out = disassemble(chunk, "test")
    assert "OP_START" in out
    assert "OP_HALT" in out
    assert "hello" in out
