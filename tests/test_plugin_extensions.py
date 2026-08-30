"""Phase E2: typed Program.extensions sections (doc/38 §6.3).

Verifies the parser routes every ``sim_extensions`` write through a typed,
owner-namespaced extension section; the sections remain a view over the legacy
dict so ``.helixc`` encoding, legacy decode and every sim_runtime reader see
byte-identical state during the E2→E4 migration window.
"""
from __future__ import annotations

import pytest

from helixlang.api.errors import UnknownKeywordError
from helixlang.core.language import LanguageConfig
from helixlang.core.lexer import Lexer
from helixlang.core.parser import Parser

HUMAN_SRC = """\
#config ticks=100 output=stdout
#person name=John age=55 sex=male weight=82 height=175
#trait smoking=former pack_years=10
#disease name="type 2 diabetes" category=metabolic_overload severity=0.7
#disease_gene gene=INSR type=downregulate activity=0.3
#drug name=metformin dose=850 mg=1000 target=INSR
#gene name=APOE allele=E3 zygosity=heterozygous
#tumor_biopsy site=lung mutation=EGFR_L858R pd_l1_expression=0.6
#end
#sim kind=spatial_dfba grid=8
"""

GENOME_SRC = """\
#config ticks=40 output=stdout
#genome source=synth-4300 tf_map=regulon grn_mode=sparse seed=7
#species name=producer photo=true photo_vmax=0.01 cn_ratio=8
#species name=consumer substrate=glucose vmax=0.02 ks=0.1
#patch name=sugar_depleted seed=4
#gem organism=e_coli_k12 use_database=true gapfill=false
"""


def parse(src: str):
    tokens = list(Lexer(src).tokens())
    return Parser(tokens, config=LanguageConfig.for_table("standard")).parse()


def test_section_ids_and_owner_routing():
    prog = parse(HUMAN_SRC)
    ext = prog.extensions
    assert ext.ids() == ["core", "gem", "human", "population"]
    assert ext.extension_for("person_age", "55").id == "human"
    assert ext.extension_for("genome_source", "x").id == "population"
    assert ext.extension_for("species.a.b", "1").id == "population"
    assert ext.extension_for("gem_dynamic", "true").id == "gem"
    assert ext.extension_for("kind", "population").id == "core"
    assert ext.extension_for("grid", "8").id == "core"   # #sim long-tail


def test_typed_attribute_access():
    prog = parse(HUMAN_SRC)
    human = prog.extensions.human
    assert human.get("person_age") == "55"
    assert human.get("person_sex") == "male"
    assert human.get("disease_name") == "type 2 diabetes"
    drugs = human.drugs
    assert isinstance(drugs, list) and drugs[0]["name"] == "metformin"
    genes = human.genes
    assert genes == [{"name": "APOE", "allele": "E3",
                      "zygosity": "heterozygous"}]
    biopsy = human.tumor_biopsy
    assert biopsy["site"] == "lung" and biopsy["mutation"] == "EGFR_L858R"
    dg = human.disease_genes
    assert dg == [{"gene": "INSR", "type": "downregulate", "activity": "0.3"}]
    with pytest.raises(AttributeError):
        _ = human.nope


def test_gem_and_population_sections():
    prog = parse(GENOME_SRC)
    gem = prog.extensions.gem
    assert gem.gem_organism == "e_coli_k12"
    assert gem.get("gem_gapfill") == "false"
    pop = prog.extensions.population
    assert pop.genome == "true"
    assert pop.get("genome_source") == "synth-4300"
    assert pop.get("species.producer.photo") == "true"
    assert pop.get("patch.sugar_depleted.seed") == "4"


def test_single_store_view_parity():
    prog = parse(HUMAN_SRC)
    ext = prog.extensions
    round_trip = dict(prog.sim_extensions)          # legacy dict
    section_proj = {}
    for sid in ("human", "gem", "population"):
        section_proj.update(ext.extension(sid).to_dict())
    open_ = ext.extension("core").to_dict()
    assert round_trip == section_proj | open_


def test_governed_section_rejects_undeclared_key():
    prog = parse("#config ticks=10 output=stdout\n")
    human = prog.extensions.human
    with pytest.raises(UnknownKeywordError):
        human.set("hexx", "x")            # matches no declared key/prefix
    with pytest.raises(UnknownKeywordError):
        human.set("tumor_biopsy", "scalar")   # declared map; scalar rejected
    with pytest.raises(UnknownKeywordError):
        human.append("totally_new", {"k": 1})


def test_governed_section_accepts_declared_and_core_open():
    prog = parse("#config ticks=10 output=stdout\n")
    human = prog.extensions.human
    human.set("tumor_biopsy", {"site": "lung"})     # declared map ok
    human.append("drugs", {"name": "x"})
    assert prog.sim_extensions["tumor_biopsy"]["site"] == "lung"
    core = prog.extensions.core
    core.set("anything_goes_for_longtail", "1")     # open escape hatch
    assert prog.sim_extensions["anything_goes_for_longtail"] == "1"


def test_plugin_grammar_routes_through_escape_hatch():
    import helixlang.plugins.annotation.vector  # noqa: F401
    prog = parse("#config ticks=10 output=stdout\n"
                 "#vector gene=TP53 plasmid=pUC19 payload_len=3821\n")
    assert prog.sim_extensions["vectors"] == [
        {"gene": "TP53", "plasmid": "pUC19", "payload_len": "3821"}]


def test_hxbc_round_trip_unchanged():
    from helixlang.core import hxbc
    for src in (HUMAN_SRC, GENOME_SRC):
        prog = parse(src)
        blob = hxbc.dumps_program(prog)
        prog2 = hxbc.loads_program(blob).program
        assert dict(prog.sim_extensions) == dict(prog2.sim_extensions)
        text = hxbc.decompile(prog2)
        prog3 = parse(text)
        assert dict(prog2.sim_extensions) == dict(prog3.sim_extensions)


def test_hxbc_plugin_ext_payload_round_trip():
    """Governed keys serialize as namespaced PLUGIN_EXT records and reload
    into the identical sim_extensions (doc/38 §6.7)."""
    from helixlang.core import hxbc
    for src in (HUMAN_SRC, GENOME_SRC):
        prog = parse(src)
        blob = hxbc.dumps_program(prog)
        assert bytes([0x0E]) in blob
        prog2 = hxbc.loads_program(blob).program
        assert dict(prog.sim_extensions) == dict(prog2.sim_extensions)
        assert dict(prog2.extensions.human.to_dict()) == dict(
            parse(src).extensions.human.to_dict())


def test_hxbc_legacy_flat_artifact_still_loads():
    """An artifact written with every key in the flat TAG (pre-migration)
    must decode unchanged — no PLUGIN_EXT records present."""
    from unittest import mock

    from helixlang.core import hxbc
    prog = parse(HUMAN_SRC)
    with mock.patch.object(hxbc, "split_sim_extensions",
                           return_value=({}, dict(prog.sim_extensions))):
        legacy_blob = hxbc.dumps_program(prog)
    assert bytes([0x0E]) not in legacy_blob
    prog2 = hxbc.loads_program(legacy_blob).program
    assert dict(prog.sim_extensions) == dict(prog2.sim_extensions)


def test_hxbc_plugin_ext_rejects_unknown_or_stale_abi():
    import hashlib

    from helixlang.core import hxbc
    from helixlang.core.errors import PluginMissingError
    from helixlang.core.extensions import PLUGIN_EXT_ABI
    from helixlang.core.hxbc import PluginBinaryError

    def reseal(blob: bytes) -> bytes:
        """Recompute the EOF SHA-256 after same-length byte surgery."""
        sealed = bytearray(blob)
        end = len(sealed) - 40
        sealed[end + 8:end + 40] = hashlib.sha256(sealed[12:end]).digest()
        return bytes(sealed)

    prog = parse(HUMAN_SRC)
    good = hxbc.dumps_program(prog)
    rec = bytes([0x0E]) + b"\x00\x00\x00\x05human"

    # Unknown extension id -> missing plugin.
    evil = reseal(good.replace(rec, bytes([0x0E]) + b"\x00\x00\x00\x05virus"))
    with pytest.raises(PluginMissingError):
        hxbc.loads_program(evil)

    # Stale abi_version (0) -> PluginBinaryError, never a wrong-result run.
    stale = reseal(good.replace(rec + PLUGIN_EXT_ABI.to_bytes(4, "big"),
                                rec + (0).to_bytes(4, "big")))
    with pytest.raises(PluginBinaryError):
        hxbc.loads_program(stale)


def test_hxbc_plugin_ext_deterministic_bytes():
    from helixlang.core import hxbc
    assert bytes([0x0E]) in (blob := hxbc.dumps_program(parse(HUMAN_SRC)))
    assert blob == hxbc.dumps_program(parse(HUMAN_SRC))


def test_extensions_read_surface_is_a_mapping():
    """Engine readers can swap ``program.sim_extensions`` → ``program.extensions``
    with byte-identical results (doc/38 §6.3 E4)."""
    from collections.abc import Mapping

    for src in (HUMAN_SRC, GENOME_SRC):
        prog = parse(src)
        ext = prog.extensions
        store = prog.sim_extensions
        assert isinstance(ext, Mapping)
        assert len(ext) == len(store)
        assert set(ext) == set(store)                      # __iter__
        assert dict(ext.items()) == store                  # Mapping.items()
        assert {**ext} == store                            # merge
        for k in list(store):                              # full parity
            assert (k in ext) == (k in store)
            assert ext[k] == store[k]
            assert ext.get(k) == store.get(k)
            assert ext.get(k, "missing-default") == store.get(k, "missing-default")
        assert ext.get("never_set_key", "dflt") == "dflt"
    assert parse(HUMAN_SRC).extensions.get("person_age", None) == "55"
    assert parse(GENOME_SRC).extensions.get("genome_source", None) == "synth-4300"


def test_extensions_setitem_routes_through_typed_section():
    """A ``__setitem__`` write is a sanctioned section write, never a silent
    raw dict poke."""
    prog = parse("#config ticks=10 output=stdout\n")
    prog.extensions["ticks"] = "99"                         # core-open key
    assert prog.sim_extensions["ticks"] == "99"
    prog.extensions["person_age"] = "60"                    # governed prefix
    assert prog.extensions.human.get("person_age") == "60"
    # An unclaimable key falls through to the core long-tail escape hatch
    # (route matches the owning section; unowned keys are core-open, so the
    # write is a lenient store poke — same as the legacy #sim long-tail rule).
    prog.extensions["hexx"] = "x"
    assert prog.extensions.core.get("hexx") == "x"
    assert prog.sim_extensions["hexx"] == "x"
    # Declared governed keys land on the owning typed section.
    prog.extensions["tumor_biopsy"] = {"site": "lung"}
    assert prog.extensions.human.tumor_biopsy["site"] == "lung"
    # A governed map key routed with a non-map value is *not* claimable by the
    # governed section and falls to core-open (lenient long-tail), exactly as
    # legacy flat writes behaved; direct section writes stay strict.
    prog.extensions["tumor_biopsy"] = "scalar"
    assert prog.sim_extensions["tumor_biopsy"] == "scalar"
    with pytest.raises(UnknownKeywordError):
        prog.extensions.human.set("tumor_biopsy", "scalar")
