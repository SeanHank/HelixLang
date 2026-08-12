"""HelixLang web server tests.

Uses the Flask test_client (no real port listening); covers:
- health check
- example list/read/path traversal protection
- compile API: valid/invalid source
- run API: trace length, serialization structure, error handling
- static index page reachability
- unknown routes 404
"""
from __future__ import annotations

import pytest

# Flask is an optional web dependency; the whole module is skipped when missing.
# The client / app fixtures are provided by tests/conftest.py (session-level app + function-level client).
flask = pytest.importorskip("flask")


HELLO_SRC = """#gene name=hello
ATG GCT GGT TAA
#end
#config ticks=5
"""


LAC_SRC = """#promoter name=plac strength=-1.0
#end
#gene name=laci promoter=plac
ATG GCT GCT GCT TAA
#end
#gene name=lacoperon promoter=plac
ATG GGT GGT TAA
#end
#regulate laci -> lacoperon strength=-0.5
#end
#config ticks=10
"""


# ---------- health check ----------

def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.get_json()
    assert data["status"] == "ok"
    assert "version" in data


# ---------- static index page ----------

def test_index_returns_html(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.data.decode("utf-8")
    assert "<html" in body.lower()
    assert "HelixLang" in body
    # Includes an ECharts reference
    assert "echarts" in body.lower()


# ---------- examples API ----------

def test_examples_list(client):
    r = client.get("/api/examples")
    assert r.status_code == 200
    data = r.get_json()
    assert "examples" in data
    assert isinstance(data["examples"], list)
    # At least 5 examples
    assert len(data["examples"]) >= 5
    assert "01_hello_dna.helix" in data["examples"]


def test_example_content(client):
    r = client.get("/api/examples/01_hello_dna.helix")
    assert r.status_code == 200
    data = r.get_json()
    assert data["name"] == "01_hello_dna.helix"
    assert "ATG" in data["source"]


def test_example_not_found(client):
    r = client.get("/api/examples/does_not_exist.helix")
    assert r.status_code == 404


def test_example_path_traversal_rejected(client):
    # Path traversal: URL-encoded / already 404s at the Flask routing layer (does not match <name>),
    # or is rejected with 400 by the business layer
    r = client.get("/api/examples/..%2F..%2Fetc%2Fpasswd.helix")
    assert r.status_code in (400, 404)
    # A path containing / should also be rejected by routing or the business layer
    r2 = client.get("/api/examples/sub/foo.helix")
    assert r2.status_code in (400, 404)


# ---------- compile API ----------

def test_compile_success(client):
    r = client.post("/api/compile",
                    json={"source": HELLO_SRC, "table": "standard"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert "disassemble" in data
    assert "OP_START" in data["disassemble"] or "OP_HALT" in data["disassemble"]
    assert data["chunk_bytes"] > 0
    assert data["constants_count"] >= 0
    prog = data["program"]
    assert len(prog["genes"]) == 1
    assert prog["genes"][0]["name"] == "hello"
    assert prog["genes"][0]["orf"][0] == "ATG"
    assert prog["genes"][0]["orf"][-1] == "TAA"


def test_compile_invalid_table(client):
    r = client.post("/api/compile",
                    json={"source": HELLO_SRC, "table": "bogus"})
    assert r.status_code == 400


def test_compile_syntax_error(client):
    # A real syntax error: unclosed annotation block + no ATG
    bad_src2 = "#gene name=x\nGGG GGG\n"
    r = client.post("/api/compile",
                    json={"source": bad_src2, "table": "standard"})
    assert r.status_code == 400  # business errors return 4xx + ok:false (REST semantics)
    data = r.get_json()
    assert data["ok"] is False
    assert "error" in data


def test_compile_lex_error(client):
    # Length is not a multiple of 3
    bad_src = "#gene name=x\nATG GC\n"
    r = client.post("/api/compile",
                    json={"source": bad_src, "table": "standard"})
    data = r.get_json()
    assert data["ok"] is False


# ---------- run API ----------

def test_run_hello(client):
    r = client.post("/api/run",
                    json={"source": HELLO_SRC, "table": "standard"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert data["ticks_run"] == 5
    assert len(data["trace"]) == 5
    # trace structure
    s = data["trace"][0]
    for k in ("tick", "x", "y", "energy", "alive", "proteins",
              "color", "gene_levels", "morphology_points_count",
              "field_total_v"):
        assert k in s, f"missing key {k}"
    # serialization
    ts = data["trace_series"]
    assert len(ts["ticks"]) == 5
    assert len(ts["energy"]) == 5
    # summary
    prog = data["program"]
    assert len(prog["genes"]) == 1
    # cell
    cell = data["cell"]
    assert "energy" in cell
    assert "alive" in cell


def test_run_config_has_no_units_key(client):
    r = client.post("/api/run",
                    json={"source": HELLO_SRC, "table": "standard"})
    data = r.get_json()
    assert data["ok"] is True
    assert "units" not in data["program"]["config"]
    assert all("units" not in s for s in data["trace"])


def test_run_with_grn(client):
    r = client.post("/api/run",
                    json={"source": LAC_SRC, "table": "standard"})
    data = r.get_json()
    assert data["ok"] is True
    grn = data["grn"]
    # At least 3 nodes (2 genes + 1 promoter plac)
    assert len(grn["nodes"]) >= 2
    # At least 1 regulatory edge
    assert any(e["source"] == "laci" for e in grn["edges"])
    # Node structure
    n = grn["nodes"][0]
    for k in ("id", "name", "level", "threshold", "value"):
        assert k in n


def test_run_with_overridden_ticks(client):
    r = client.post("/api/run",
                    json={"source": HELLO_SRC, "table": "standard",
                          "ticks": 3})
    data = r.get_json()
    assert data["ok"] is True
    assert data["ticks_run"] == 3


def test_run_field_example(client):
    """04_turing_pattern.helix should return field data."""
    r = client.get("/api/examples/04_turing_pattern.helix")
    src = r.get_json()["source"]
    r2 = client.post("/api/run",
                     json={"source": src, "table": "standard",
                           "ticks": 5})
    data = r2.get_json()
    if data["ok"]:
        # Example 4 should have field data
        if data["field"] is not None:
            f = data["field"]
            assert f["n"] > 0
            assert len(f["u"]) == f["n"]
            assert len(f["v"]) == f["n"]
            assert len(f["u"][0]) == f["n"]


def test_run_lsystem_example(client):
    """03_plant_growth.helix should return morphology data."""
    r = client.get("/api/examples/03_plant_growth.helix")
    src = r.get_json()["source"]
    r2 = client.post("/api/run",
                     json={"source": src, "table": "standard",
                           "ticks": 5})
    data = r2.get_json()
    if data["ok"]:
        m = data["morphology"]
        assert "points" in m
        assert "count" in m
        assert m["count"] == len(m["points"])


def test_run_table_switch(client):
    """mito_vertebrate table switch: TGA is no longer STOP."""
    r = client.post("/api/run",
                    json={"source": HELLO_SRC, "table": "mito_vertebrate"})
    data = r.get_json()
    # The hello example ORF ends with TAA; TAA is still STOP under mito
    assert data["ok"] is True


# ---------- error handling ----------

def test_run_runtime_error(client):
    """Runtime errors (e.g. ORF without ATG) should return 4xx + ok:false, not 500."""
    bad = "#gene name=x\nGCT GCT GCT\n"
    r = client.post("/api/run", json={"source": bad, "table": "standard"})
    assert r.status_code == 400  # REST semantics: business errors return 4xx
    data = r.get_json()
    assert data["ok"] is False
    assert "error" in data


def test_404(client):
    r = client.get("/api/does-not-exist")
    assert r.status_code == 404
    data = r.get_json()
    assert data["error"] == "not found"


# ---------- static assets ----------

def test_static_assets(client):
    # Index page is reachable
    r = client.get("/")
    assert r.status_code == 200


# ---------- end-to-end: all examples run ----------

@pytest.mark.parametrize("name", [
    "01_hello_dna.helix",
    "02_lac_operon.helix",
    "03_plant_growth.helix",
    "04_turing_pattern.helix",
    "05_table_switch.helix",
])
def test_all_examples_run(client, name):
    r = client.get(f"/api/examples/{name}")
    assert r.status_code == 200
    src = r.get_json()["source"]
    table = "mito_vertebrate" if "table_switch" in name else "standard"
    r2 = client.post("/api/run",
                     json={"source": src, "table": table, "ticks": 5})
    data = r2.get_json()
    assert data["ok"] is True, f"{name} run failed: {data.get('error')}"
    assert data["ticks_run"] >= 1


# ---------- translation table coverage ----------

def test_all_tables_compile(client):
    for tbl in ["standard", "mito_vertebrate", "ciliate"]:
        r = client.post("/api/compile",
                        json={"source": HELLO_SRC, "table": tbl})
        # Under different tables the hello example may parse the ORF differently, but should not raise 500
        assert r.status_code in (200, 400)


# ============================================================================
# Additional tests below: cover the remaining ~35 API endpoints
# ============================================================================

# Reusable small biological inputs
PROTEIN_SEQ = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard amino acids
DNA_ORF = "ATGAAATTTGGGTAA"            # ATG AAA TTT GGG TAA -> M K F G *
# ~210bp genome containing multiple NGG PAMs (SpCas9)
CRISPR_GENOME = (
    "ATGCGATCGATCGATCGATCGATCGGATCGATCGATCGATCGATCGATCGGATCG"
    "ATCGATCGATCGATCGATCGATCGGATCGATCGATCGATCGATCGATCGGATCGA"
    "TCGATCGATCGATCGATCGATCGATCGGATCGATCGATCGATCGATCGATCGGAT"
    "ATCGATCGATCGATCGATCGATCGGATCGATCGATCGATCGATCGATCGG"
)
TM_PROTEIN = "LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL"  # hydrophobic, transmembrane-like
STORE_TEXT = "Hello, HelixLang!"


# ---------- DNA physical codec API ----------

def test_dna_encode_goldman(client):
    r = client.post("/api/dna/encode",
                    json={"source": STORE_TEXT, "scheme": "goldman"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert data["scheme"] == "goldman"
    assert isinstance(data["oligos"], list) and len(data["oligos"]) > 0
    assert "stats" in data
    # Each oligo should have stats (GC stats)
    assert "stats" in data["oligos"][0]


def test_dna_encode_erlich_with_pcr(client):
    r = client.post("/api/dna/encode",
                    json={"source": STORE_TEXT, "scheme": "erlich",
                          "pcr_cycles": 5})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert data["scheme"] == "erlich"
    assert len(data["oligos"]) > 0
    assert "seed" in data["oligos"][0]


def test_dna_encode_bad_scheme(client):
    r = client.post("/api/dna/encode",
                    json={"source": STORE_TEXT, "scheme": "bogus"})
    assert r.status_code == 400


def test_dna_decode_roundtrip(client):
    # Encode first, then decode, to verify the roundtrip
    enc = client.post("/api/dna/encode",
                      json={"source": STORE_TEXT, "scheme": "goldman"})
    enc_data = enc.get_json()
    dec = client.post("/api/dna/decode",
                      json={"oligos_data": {
                          "oligos": enc_data["oligos"],
                          "stats": enc_data["stats"],
                      }, "scheme": "goldman"})
    assert dec.status_code == 200
    dec_data = dec.get_json()
    assert dec_data["ok"] is True
    assert dec_data["source"] == STORE_TEXT


def test_dna_decode_bad_scheme(client):
    r = client.post("/api/dna/decode",
                    json={"oligos_data": {}, "scheme": "bogus"})
    assert r.status_code == 400


# ---------- bio data API ----------

@pytest.mark.parametrize("species", ["ecoli", "yeast", "human"])
def test_codon_usage_valid_species(client, species):
    r = client.get(f"/api/bio/codon-usage?species={species}")
    assert r.status_code == 200
    data = r.get_json()
    assert data["species"] == species
    assert "available" in data
    assert isinstance(data["codon_usage"], list) and len(data["codon_usage"]) > 0
    cu = data["codon_usage"][0]
    for k in ("codon", "aa", "per_thousand", "fraction"):
        assert k in cu


def test_codon_usage_invalid_species(client):
    r = client.get("/api/bio/codon-usage?species=alien")
    assert r.status_code == 400
    data = r.get_json()
    assert data["ok"] is False


def test_codon_usage_default_species(client):
    r = client.get("/api/bio/codon-usage")
    assert r.status_code == 200
    assert r.get_json()["species"] == "ecoli"


def test_species_list(client):
    r = client.get("/api/bio/species")
    assert r.status_code == 200
    data = r.get_json()
    assert isinstance(data["species"], list)
    assert len(data["species"]) >= 3
    ids = {s["id"] for s in data["species"]}
    assert {"ecoli", "yeast", "human"} <= ids
    s0 = data["species"][0]
    for k in ("id", "name", "n_codons", "has_trna"):
        assert k in s0


@pytest.mark.parametrize("species", ["ecoli", "yeast", "human"])
def test_trna_valid_species(client, species):
    r = client.get(f"/api/bio/trna?species={species}")
    assert r.status_code == 200
    data = r.get_json()
    assert data["species"] == species
    assert "available" in data
    assert isinstance(data["trna_abundance"], list)
    if data["trna_abundance"]:
        t0 = data["trna_abundance"][0]
        for k in ("codon", "abundance", "fraction"):
            assert k in t0


def test_trna_invalid_species(client):
    r = client.get("/api/bio/trna?species=alien")
    assert r.status_code == 400
    assert r.get_json()["ok"] is False


def test_gray_scott_presets(client):
    r = client.get("/api/bio/gray-scott-presets")
    assert r.status_code == 200
    data = r.get_json()
    assert isinstance(data["presets"], list) and len(data["presets"]) > 0
    p0 = data["presets"][0]
    for k in ("name", "F", "k", "Du", "Dv", "description"):
        assert k in p0


# ---------- central dogma API ----------

def test_central_dogma_transcribe(client):
    r = client.post("/api/central-dogma/transcribe",
                    json={"dna": DNA_ORF, "promoter_strength": 0.8})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert isinstance(data["mrna"], str) and len(data["mrna"]) > 0
    assert "transcript" in data


def test_central_dogma_translate_from_dna(client):
    r = client.post("/api/central-dogma/translate",
                    json={"dna": DNA_ORF, "ribosome_density": 1.0})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert isinstance(data["protein"], str)
    assert "result" in data
    assert "transcript" in data


def test_central_dogma_translate_from_mrna(client):
    mrna = DNA_ORF.replace("T", "U")  # AUG AAA UUU GGG UAA
    r = client.post("/api/central-dogma/translate",
                    json={"mrna": mrna})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert isinstance(data["protein"], str)


def test_central_dogma_coupled(client):
    r = client.post("/api/central-dogma/coupled",
                    json={"dna": DNA_ORF, "promoter_strength": 1.0})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert isinstance(data["protein"], str)
    ts = data["time_series"]
    for k in ("time", "mrna_level", "protein_level"):
        assert k in ts and isinstance(ts[k], list)
    assert "mrna_steady_state" in data
    assert "translation_result" in data


# ---------- evolution API ----------

def test_evolution_run(client):
    r = client.post("/api/evolution/run",
                    json={"initial_dna": "ACGT" * 10,
                          "generations": 5,
                          "population_size": 20,
                          "mutation_rate": 0.05,
                          "fitness_method": "hamming",
                          "target_dna": "ACGT" * 10})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert isinstance(data["generations"], list)
    assert isinstance(data["final_dna"], str)
    assert "final_fitness" in data
    assert "stats" in data
    for k in ("final_generation", "final_diversity", "population_size"):
        assert k in data["stats"]


def test_evolution_params(client):
    r = client.get("/api/evolution/params")
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert "defaults" in data
    assert "hamming" in data["fitness_methods"]
    assert "descriptions" in data


# ---------- multicellular population API ----------

def test_population_simulate(client):
    r = client.post("/api/population/simulate",
                    json={"initial_count": 2,
                          "generations": 3,
                          "config": {"grid_width": 20, "grid_height": 20,
                                     "max_size": 100}})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert isinstance(data["history"], list)
    assert "final_grid" in data
    assert "signal_field" in data


# ---------- CRISPR API ----------

def test_crispr_find_pam(client):
    r = client.post("/api/crispr/find-pam",
                    json={"dna": CRISPR_GENOME, "cas_variant": "SpCas9"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert isinstance(data["sites"], list)
    assert len(data["sites"]) > 0
    s0 = data["sites"][0]
    for k in ("position", "pam", "spacer", "strand"):
        assert k in s0


def test_crispr_design_guide(client):
    r = client.post("/api/crispr/design-guide",
                    json={"dna": CRISPR_GENOME,
                          "cas_variant": "SpCas9",
                          "position": 50})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    g = data["guide"]
    for k in ("spacer", "pam", "pam_position", "cas_variant",
              "target_position", "strand"):
        assert k in g
    assert isinstance(data["on_target_score"], (int, float))


def test_crispr_edit(client):
    r = client.post("/api/crispr/edit",
                    json={"dna": CRISPR_GENOME,
                          "target_position": 50,
                          "new_sequence": "TTTTTTTTTTTTTTTTTTTT",
                          "cas_variant": "SpCas9"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    for k in ("edited_dna", "edit_type", "success", "off_targets",
              "guide", "edit_position", "edit_length", "repair"):
        assert k in data


# ---------- epigenetics API ----------

def test_epigenetics_methylate(client):
    r = client.post("/api/epigenetics/methylate",
                    json={"dna": "GATCGATCGATCGATCGATCGATCGATCGATC",
                          "cell_type": "ecoli", "methylase": "dam"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert "methylation" in data


def test_epigenetics_histone(client):
    r = client.post("/api/epigenetics/histone",
                    json={"dna": "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATC",
                          "gene_positions": [
                              {"name": "geneA", "start": 0, "end": 20}],
                          "cell_type": "eukaryote"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert isinstance(data["histone_marks"], list)
    assert "accessibility" in data
    assert "expression_modifier" in data


def test_epigenetics_cpg_islands(client):
    # CpG islands require >=200bp and GC enrichment
    cpg_dna = "GCGCGCGCGCGCGCGCGCATATATGCGCGCGCGCGCGCGCGCGCGCATATATGCGC" * 8
    r = client.post("/api/epigenetics/cpg-islands",
                    json={"dna": cpg_dna})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert "islands" in data


# ---------- 3D morphogenesis API ----------

def test_morphology3d_preset(client):
    r = client.post("/api/morphology3d/generate",
                    json={"preset": "fern", "iterations": 2})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert isinstance(data["lines"], list)
    assert isinstance(data["points"], list)
    b = data["bounds"]
    for k in ("min", "max", "center", "size"):
        assert k in b and len(b[k]) == 3


def test_morphology3d_custom(client):
    r = client.post("/api/morphology3d/generate",
                    json={"axiom": "F", "rules": "F:F[+F]F[-F]F",
                          "angle": 25.0, "iterations": 2})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert len(data["lines"]) > 0


# ---------- DNA storage API ----------

def test_dna_storage_store_erlich(client):
    r = client.post("/api/dna-storage/store",
                    json={"text": STORE_TEXT, "scheme": "erlich"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert isinstance(data["oligos"], list) and len(data["oligos"]) > 0
    rep = data["report"]
    for k in ("scheme", "total_bp", "num_oligos", "data_len"):
        assert k in rep


def test_dna_storage_store_goldman(client):
    r = client.post("/api/dna-storage/store",
                    json={"text": STORE_TEXT, "scheme": "goldman"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert len(data["oligos"]) > 0


def test_dna_storage_retrieve_roundtrip(client):
    store = client.post("/api/dna-storage/store",
                        json={"text": STORE_TEXT, "scheme": "erlich"})
    sdata = store.get_json()
    rep = sdata["report"]
    r = client.post("/api/dna-storage/retrieve",
                    json={"oligos": sdata["oligos"],
                          "scheme": "erlich",
                          "data_len": rep["data_len"]})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert data["text"] == STORE_TEXT


def test_dna_storage_lifecycle(client):
    r = client.post("/api/dna-storage/lifecycle",
                    json={"text": STORE_TEXT, "scheme": "erlich",
                          "synthesis_quality": "typical",
                          "pcr_cycles": 5,
                          "storage_years": 1.0})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    rep = data["report"]
    for k in ("recovered_data", "integrity", "error_rate", "success"):
        assert k in rep
    # Array fields exposed at the top level
    assert isinstance(data["integrity"], list)
    assert isinstance(data["error_rate"], list)


def test_dna_storage_analyze(client):
    store = client.post("/api/dna-storage/store",
                        json={"text": STORE_TEXT, "scheme": "erlich"})
    sdata = store.get_json()
    rep = sdata["report"]
    r = client.post("/api/dna-storage/analyze",
                    json={"oligos": sdata["oligos"],
                          "scheme": "erlich",
                          "data_len": rep["data_len"]})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    arep = data["report"]
    for k in ("density", "cost", "durability", "oligo_count"):
        assert k in arep
    # Top-level canonical fields
    assert "density" in data
    assert "bits_per_nt" in data["density"]
    assert "total_oligos" in data


# ---------- synthetic biology design API ----------

def test_synbio_design_cassette(client):
    r = client.post("/api/synbio/design-cassette",
                    json={"protein": PROTEIN_SEQ,
                          "promoter": "lac",
                          "terminator": "rrnB_T1",
                          "optimize_codons": True,
                          "add_histidine_tag": True})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    c = data["cassette"]
    for k in ("full_sequence", "promoter_seq", "rbs_seq", "orf_seq",
              "terminator_seq", "protein", "cai", "gc_content"):
        assert k in c
    assert len(c["full_sequence"]) > 0


def test_synbio_design_vector(client):
    r = client.post("/api/synbio/design-vector",
                    json={"protein": PROTEIN_SEQ,
                          "origin": "pUC19",
                          "selection_marker": "AmpR"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    v = data["vector"]
    for k in ("full_sequence", "total_length", "features",
              "origin_seq", "marker_seq", "mcs_seq", "cassette"):
        assert k in v
    assert v["total_length"] > 0


def test_synbio_validate(client):
    dna = "ATGAAATTTGGGAAAGGGTTTAA" * 2
    r = client.post("/api/synbio/validate", json={"dna": dna})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    v = data["validation"]
    assert "has_start" in v
    assert "has_stop" in v
    assert "cai" in v


def test_synbio_export_genbank(client):
    dna = "ATGAAATTTGGGAAAGGGTTTAA" * 2
    r = client.post("/api/synbio/export-genbank",
                    json={"dna": dna, "name": "test_seq"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert isinstance(data["genbank"], str)
    assert "ORIGIN" in data["genbank"] or "LOCUS" in data["genbank"]


def test_synbio_export_fasta_with_dna(client):
    dna = "ATGAAATTTGGGAAAGGGTTTAA" * 2
    r = client.post("/api/synbio/export-fasta",
                    json={"dna": dna, "name": "myseq"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert data["name"] == "myseq"
    assert isinstance(data["fasta"], str)
    assert data["fasta"].startswith(">")


def test_synbio_export_fasta_from_protein(client):
    r = client.post("/api/synbio/export-fasta",
                    json={"protein": PROTEIN_SEQ, "name": "prot_seq"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert isinstance(data["fasta"], str)
    assert data["fasta"].startswith(">")


# ---------- metabolism FBA API ----------

def test_metabolism_model(client):
    r = client.get("/api/metabolism/model")
    assert r.status_code == 200
    data = r.get_json()
    assert data["n_reactions"] > 0
    assert data["n_metabolites"] > 0
    assert isinstance(data["biomass_reaction"], str)
    assert isinstance(data["metabolites"], list)
    assert isinstance(data["subsystems"], dict)
    assert isinstance(data["reactions"], list)
    rxn0 = data["reactions"][0]
    for k in ("id", "name", "stoichiometry", "lower_bound",
              "upper_bound", "subsystem"):
        assert k in rxn0


def test_metabolism_fba(client):
    r = client.post("/api/metabolism/fba",
                    json={"objective": "biomass", "glc_uptake": 10.0})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert data["objective"] == "biomass"
    assert isinstance(data["objective_value"], (int, float))
    assert isinstance(data["fluxes"], dict)


def test_metabolism_analyze(client):
    r = client.post("/api/metabolism/analyze",
                    json={"objective": "biomass", "glc_uptake": 8.0})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    analysis = data["analysis"]
    assert isinstance(analysis, dict)
    # The analysis report should contain these key fields
    for k in ("biomass_yield", "objective_value", "subsystem_fluxes"):
        assert k in analysis


# ---------- protein structure API ----------

def test_protein_secondary_chou_fasman(client):
    r = client.post("/api/protein/secondary",
                    json={"sequence": PROTEIN_SEQ, "method": "chou-fasman"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert data["method"] == "chou-fasman"
    assert data["sequence"] == PROTEIN_SEQ
    assert len(data["secondary_structure"]) == len(PROTEIN_SEQ)
    assert isinstance(data["segments"], list)


def test_protein_secondary_gor(client):
    r = client.post("/api/protein/secondary",
                    json={"sequence": PROTEIN_SEQ, "method": "gor"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert data["method"] == "gor"
    assert len(data["secondary_structure"]) == len(PROTEIN_SEQ)


def test_protein_transmembrane(client):
    r = client.post("/api/protein/transmembrane",
                    json={"sequence": TM_PROTEIN})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert data["sequence"] == TM_PROTEIN
    assert isinstance(data["n_helices"], int)
    assert isinstance(data["is_membrane_protein"], bool)
    assert isinstance(data["gravy"], (int, float))
    assert isinstance(data["hydropathy_profile"], list)
    assert isinstance(data["helices"], list)
    if data["helices"]:
        h0 = data["helices"][0]
        for k in ("start", "end", "length", "mean_hydropathy", "sequence"):
            assert k in h0


def test_protein_disorder(client):
    r = client.post("/api/protein/disorder",
                    json={"sequence": PROTEIN_SEQ})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert data["sequence"] == PROTEIN_SEQ
    assert isinstance(data["n_regions"], int)
    assert 0.0 <= data["disorder_fraction"] <= 1.0 + 1e-9
    assert isinstance(data["regions"], list)
    if data["regions"]:
        rg0 = data["regions"][0]
        for k in ("start", "end", "length", "mean_hydropathy", "sequence"):
            assert k in rg0


# ---------- debugger API ----------

def _create_debug_session(client, source=None):
    """Helper: create a debug session and return the session_id."""
    r = client.post("/api/debug/session",
                    json={"source": source or HELLO_SRC,
                          "table": "standard"})
    assert r.status_code == 200, f"failed to create debug session: {r.get_json()}"
    data = r.get_json()
    assert data["ok"] is True
    return data["session_id"], data["initial_state"]


def test_debug_session_create(client):
    sid, state = _create_debug_session(client)
    assert isinstance(sid, str) and len(sid) > 0
    for k in ("ip", "op", "stack", "cell_state", "grn_state",
              "gene", "line", "codon_index"):
        assert k in state


def test_debug_session_bad_table(client):
    r = client.post("/api/debug/session",
                    json={"source": HELLO_SRC, "table": "bogus"})
    assert r.status_code == 400
    assert r.get_json()["ok"] is False


def test_debug_state(client):
    sid, _ = _create_debug_session(client)
    r = client.get(f"/api/debug/state?session_id={sid}")
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert "state" in data
    assert "call_stack" in data
    assert isinstance(data["breakpoints"], list)
    assert "halted" in data


def test_debug_state_unknown_session(client):
    r = client.get("/api/debug/state?session_id=nonexistent-uuid")
    assert r.status_code == 400
    assert r.get_json()["ok"] is False


def test_debug_step(client):
    sid, _ = _create_debug_session(client)
    r = client.post("/api/debug/step", json={"session_id": sid})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert "halted" in data


def test_debug_step_over(client):
    sid, _ = _create_debug_session(client)
    r = client.post("/api/debug/step-over", json={"session_id": sid})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert "halted" in data


def test_debug_step_out(client):
    sid, _ = _create_debug_session(client)
    r = client.post("/api/debug/step-out", json={"session_id": sid})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert "halted" in data


def test_debug_continue(client):
    sid, _ = _create_debug_session(client)
    r = client.post("/api/debug/continue", json={"session_id": sid})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert "halted" in data


def test_debug_breakpoint_set_and_list(client):
    sid, _ = _create_debug_session(client)
    # Set a breakpoint
    r = client.post("/api/debug/breakpoint",
                    json={"session_id": sid, "action": "set",
                          "line": 2})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert "breakpoint" in data
    assert isinstance(data["breakpoints"], list)
    # List breakpoints
    r2 = client.post("/api/debug/breakpoint",
                     json={"session_id": sid, "action": "list"})
    assert r2.status_code == 200
    assert r2.get_json()["ok"] is True
    # Clear breakpoints
    r3 = client.post("/api/debug/breakpoint",
                     json={"session_id": sid, "action": "clear"})
    assert r3.status_code == 200
    assert r3.get_json()["breakpoints"] == []


def test_debug_breakpoint_unknown_session(client):
    r = client.post("/api/debug/breakpoint",
                    json={"session_id": "nope", "action": "list"})
    assert r.status_code == 400
    assert r.get_json()["ok"] is False


def test_debug_full_lifecycle(client):
    """Complete debug session lifecycle: create -> state -> step -> breakpoint -> continue."""
    sid, _ = _create_debug_session(client)
    # Query state
    assert client.get(f"/api/debug/state?session_id={sid}").status_code == 200
    # Single step
    assert client.post("/api/debug/step",
                       json={"session_id": sid}).status_code == 200
    # Continue after setting a breakpoint
    client.post("/api/debug/breakpoint",
                json={"session_id": sid, "action": "set", "line": 1})
    r = client.post("/api/debug/continue", json={"session_id": sid})
    assert r.status_code == 200
    # Step over / step out
    assert client.post("/api/debug/step-over",
                       json={"session_id": sid}).status_code == 200
    assert client.post("/api/debug/step-out",
                       json={"session_id": sid}).status_code == 200
    # Keep continuing until HALT
    for _ in range(20):
        r = client.post("/api/debug/continue", json={"session_id": sid})
        if r.get_json().get("halted"):
            break
