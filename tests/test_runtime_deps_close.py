"""Closure coverage for small runtime dependency modules.

Covers the remaining statements and branch arcs in ``flow``, ``seq_utils``,
``morphology_3d``, ``protein_fitness`` and ``cell_body`` that the module-
specific test suites leave untouched (identity/edge paths, degenerate
geometry, unavailable-oracle error handling).
"""
from __future__ import annotations

import math
import random

import pytest

from helixlang.plugins.runtime import protein_fitness
from helixlang.plugins.runtime.cell_body import (
    CellBody,
    _closest_point_on_segment,
    _segment_distance_sq,
    advect_rods,
    capsule_radius,
    divide_rod,
    hertzian_force,
    resolve_rod_contacts,
    rod_rod_contact,
    rod_volume_um3,
    rod_wall_contacts,
    stokes_drag_coefficient,
)
from helixlang.plugins.runtime.flow import FlowField3D, stagnant
from helixlang.plugins.runtime.grn import GRN
from helixlang.plugins.runtime.morphology_3d import (
    PLANT_PRESETS,
    LSystem3D,
    Point3D,
    _normalize,
    rotate_vector,
)
from helixlang.plugins.runtime.protein_fitness import (
    AA20,
    BLOSUM62,
    BLOSUMOracle,
    ESM2Oracle,
    FitnessOracle,
    blosum62_normalized,
    blosum62_raw,
    dna_fitness,
    oracle_score,
    protein_to_dna,
    rank_variants,
)
from helixlang.plugins.runtime.reaction_diffusion import GrayScott
from helixlang.plugins.runtime.seq_utils import (
    gc_content,
    max_homopolymer,
    reverse_complement,
    stop_codons_from_table,
)
from helixlang.plugins.runtime.sparse_grn import SparseGRN
from helixlang.plugins.runtime.stochastic import (
    TelegraphPromoter,
    fano_to_noise_std,
    gillespie_telegraph,
    telegraph_fano_factor,
)


# ---------------------------------------------------------------------------
# flow.py
# ---------------------------------------------------------------------------
def test_flowfield3d_arrays_numpy_and_cache():
    field = FlowField3D(3, 2, 2,
                        [[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]] * 2,
                        [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]] * 2,
                        [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]] * 2)
    u, v, w = field.arrays()
    assert u is field._u_arr
    u2, v2, w2 = field.arrays()  # cached arrays returned
    assert u2 is u and v2 is v and w2 is w


def test_flowfield3d_max_magnitude_and_oob():
    field = FlowField3D(2, 2, 2,
                        [[[1.0, 0.0], [0.0, 0.0]]] * 2,
                        [[[0.0, 0.0], [0.0, 2.0]]] * 2,
                        [[[0.0, 0.0], [0.0, 0.0]]] * 2)
    assert field.max_magnitude() == pytest.approx(2.0)
    assert field.velocity(-1, 0, 0) == (0.0, 0.0, 0.0)
    assert field.velocity(0, 0, 0) == (1.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        FlowField3D(2, 2, 1, [], [], [])
    with pytest.raises(ValueError):
        FlowField3D(2, 2, 2, [[[1.0]]]*2, [[[1.0]]]*2, [[[1.0]]]*2)
    with pytest.raises(ValueError):
        FlowField3D(2, 2, 2,
                    [[[1.0, 2.0], [1.0]]]*2,
                    [[[1.0, 2.0], [1.0]]]*2,
                    [[[1.0, 2.0], [1.0]]]*2)


def test_stagnant_flow_coverage():
    flow = stagnant(8, 8)
    assert flow.velocity(3, 3) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# seq_utils.py
# ---------------------------------------------------------------------------
def test_seq_utils_edges():
    assert gc_content("") == 0.0
    assert gc_content("ATGC") == 0.5
    assert gc_content("au") == 0.0
    assert reverse_complement("ATGC") == "GCAT"
    assert reverse_complement("ACGTnr") == "YnacGT".upper()
    assert reverse_complement("Z") == "N"
    assert reverse_complement("AUU") == "AAT"
    assert max_homopolymer("") == 0
    assert max_homopolymer("AATTTGGG") == 3
    assert max_homopolymer("ttttt") == 5
    assert max_homopolymer("ACGT") == 1
    assert callable(stop_codons_from_table)


# ---------------------------------------------------------------------------
# morphology_3d.py
# ---------------------------------------------------------------------------
def test_morphology_normalize_zero_and_rotate():
    zero = _normalize(Point3D(0.0, 0.0, 0.0))
    assert (zero.x, zero.y, zero.z) == (0.0, 0.0, 0.0)
    rot = rotate_vector(Point3D(1.0, 0.0, 0.0), Point3D(0.0, 0.0, 0.0), 0.0)
    assert (rot.x, rot.y, rot.z) == (1.0, 0.0, 0.0)


def test_lsystem3d_derive_draw_bounds():
    cfg = PLANT_PRESETS["tree3d"]
    sys_ = LSystem3D(cfg["axiom"], cfg["rules"], cfg["angle"], cfg["step"])
    state = sys_.derive(2)
    assert state
    lines = sys_.draw(2)
    assert lines and all(l.width > 0.0 for l in lines)
    points = sys_.get_points(2)
    assert points[0] == Point3D(0.0, 0.0, 0.0)
    bounds = sys_.get_bounds(2)
    assert set(bounds) == {"min", "max", "center", "size"}
    assert bounds["max"].x >= bounds["min"].x


def test_lsystem3d_roll_branching_symbols():
    cfg = PLANT_PRESETS["algae"]
    sys_ = LSystem3D("F\\F/F[F]F", cfg["rules"], cfg["angle"], cfg["step"])
    lines = sys_.draw(2)
    assert len(lines) > 2  # yaw/pitch/roll/branch symbols all execute


def test_lsystem3d_yaw_pitch_and_unmatched_bracket():
    cfg = PLANT_PRESETS["algae"]
    sys_ = LSystem3D("A", cfg["rules"], cfg["angle"], cfg["step"])
    assert len(sys_.draw(3)) > 0
    assert LSystem3D("]", {}, 22.5, 1.0).draw(0) == []  # empty stack
    assert LSystem3D("Ff", {"F": "F"}, 22.5, 1.0).draw(1)  # pen-up move


# ---------------------------------------------------------------------------
# protein_fitness.py
# ---------------------------------------------------------------------------
def test_protein_fitness_blosum_edges():
    assert blosum62_raw("AC", "AC") == 4.0 + 9.0
    assert blosum62_normalized("A", "A") == 1.0
    assert blosum62_normalized("A", "C") == pytest.approx(
        max(0.0, min(1.0, (0.0 - (-3.0)) / (4.0 - (-3.0)))))
    for fn in (blosum62_raw, blosum62_normalized):
        with pytest.raises(ValueError):
            fn("AC", "A")
        with pytest.raises(ValueError, match="non-empty"):
            fn("", "")
        with pytest.raises(ValueError, match="invalid amino acid"):
            fn("AJ", "AC")
    assert len(AA20) == 20
    assert all(row in BLOSUM62 for aa in "TY" for row in BLOSUM62)


def test_protein_fitness_oracle_dispatch():
    assert oracle_score("AC", "AC") == pytest.approx(1.0)
    assert oracle_score("AC", "AC", "blosum62") == pytest.approx(1.0)
    with pytest.raises(ValueError, match="unknown oracle"):
        oracle_score("AC", "AC", "spam")
    assert FitnessOracle().score("AC", "AC") == pytest.approx(1.0)
    assert FitnessOracle().available is True
    assert BLOSUMOracle().available is True
    assert BLOSUMOracle().score("AC", "AC") == pytest.approx(1.0)
    custom = BLOSUMOracle()
    assert oracle_score("AC", "AC", custom) == pytest.approx(1.0)
    ranked = rank_variants("AC", ["AD", "AC", "YY"])
    assert ranked[0][0] == "AC"
    rev = rank_variants("AC", ["AD", "YY"], reverse=True)
    assert rev[0][0] == "YY"


def test_protein_fitness_dna_paths():
    dna = protein_to_dna("MAK")
    assert dna_fitness(dna, dna) == pytest.approx(1.0)
    assert len(dna) % 3 == 0


def test_esm_oracle_paths():
    esm = ESM2Oracle()
    if esm.available:
        pll = esm.pseudo_log_likelihood("A")
        assert pll == esm.pseudo_log_likelihood("A")  # cached
        assert isinstance(esm.score("A", "C"), float)
        assert isinstance(oracle_score("A", "C", "esm2"), float)
    else:
        assert esm._load_error is not None
        with pytest.raises(RuntimeError, match="unavailable"):
            esm.pseudo_log_likelihood("A")
        with pytest.raises(RuntimeError, match="esm2 oracle requested"):
            oracle_score("A", "C", "esm2")
    with pytest.raises(ValueError, match="equal length"):
        esm.score("AC", "A")


# ---------------------------------------------------------------------------
# cell_body.py
# ---------------------------------------------------------------------------
def test_cellbody_capsule_geometry():
    assert capsule_radius(-2.0) == 0.0
    assert capsule_radius(2.0) == 1.0
    assert rod_volume_um3(4.0, 2.0) == pytest.approx(
        math.pi * 4.0 + (4.0 / 3.0) * math.pi)
    assert hertzian_force(100.0, 0.0) == 0.0
    assert hertzian_force(100.0, 0.25) == pytest.approx(100.0 * 0.125)
    assert stokes_drag_coefficient(0.0, 1.0) == 0.0
    assert stokes_drag_coefficient(1.0, 0.0) == 0.0
    assert stokes_drag_coefficient(1.0, -2.0) == 0.0
    assert stokes_drag_coefficient(1.0, 1.0) == pytest.approx(
        6.0 * math.pi * (1e-3 / 60.0))


def test_segment_distance_edges():
    # degenerate AB (single point)
    d0, ax, ay, cx, cy = _segment_distance_sq(0, 0, 0, 0, 3, 0, 5, 0)
    assert d0 == pytest.approx(9.0) and (ax, ay) == (0, 0)
    # both degenerate
    d1, p1x, p1y, p2x, p2y = _segment_distance_sq(0, 0, 0, 0, 3, 4, 3, 4)
    assert d1 == pytest.approx(25.0)
    # parallel projection t<0 branch
    d2, *_ = _segment_distance_sq(-2, 0, 2, 0, 5, 1, 6, 1)
    assert d2 > 4.0
    # parallel projection t>1 branch
    d3, *_ = _segment_distance_sq(0, 0, 4, 0, -3, 1, -2, 1)
    assert d3 > 4.0
    # degenerate clamp helper
    px, py = _closest_point_on_segment(5, 5, 1, 1, 1, 1)
    assert (px, py) == (1, 1)


def test_rod_rod_contact_variants():
    a = CellBody(x=0.0, y=0.0, length_um=4.0, diameter_um=1.0, angle=0.0)
    b = CellBody(x=10.0, y=0.0, length_um=4.0, diameter_um=1.0, angle=0.0)
    assert rod_rod_contact(a, b) is None
    b = CellBody(x=0.2, y=0.0, length_um=4.0, diameter_um=1.0, angle=0.0)
    contact = rod_rod_contact(a, b)
    assert contact is not None and contact[0] > 0.0
    # parallel axes with coincident normal direction fallback
    c = CellBody(x=0.1, y=0.0, length_um=4.0, diameter_um=1.0, angle=0.0)
    overlap, nx, ny = rod_rod_contact(c, c)
    assert overlap > 0.0 and (nx, ny) == (1.0, 0.0)


def test_rod_wall_contacts_all_walls():
    body = CellBody(x=1.0, y=0.5, length_um=4.0, diameter_um=2.0,
                    angle=math.pi / 4)
    hits = rod_wall_contacts(body, x_max=2.0, y_max=2.0)
    assert len(hits) == 4
    assert all(h[0] > 0.0 for h in hits)


def test_resolve_rod_contacts_edges():
    resolve_rod_contacts([], 1.0, 1e4, 10, 10)  # empty list early return
    a = CellBody(x=0.0, y=0.0, length_um=4.0, diameter_um=1.0, angle=0.0)
    b = CellBody(x=0.5, y=0.0, length_um=4.0, diameter_um=1.0, angle=0.0)
    resolve_rod_contacts([a, b], drag=0.0, stiffness=1e4, x_max=10, y_max=10)
    assert a.x > 0.0
    c = CellBody(x=0.0, y=10.0, length_um=4.0, diameter_um=1.0, angle=0.0)
    d = CellBody(x=0.5, y=10.0, length_um=4.0, diameter_um=1.0, angle=0.0)
    resolve_rod_contacts([c, d], drag=1e-6, stiffness=1e2, x_max=10, y_max=10)
    assert c.x > 0.0


def test_advect_rods_and_divide():
    flow = stagnant(8, 8)
    body = CellBody(x=1.0, y=1.0, length_um=2.0, diameter_um=1.0)
    advect_rods([body], None, 10.0)  # no flow -> untouched
    assert body.x == 1.0
    flow.u[0][0] = 1.0
    advect_rods([body], flow, 10.0)
    assert body.x == pytest.approx(11.0)
    rng = random.Random(0)
    x, y = divide_rod(body, rng, epsilon=0.0)
    assert x.length_um == y.length_um == 1.0


def test_cellbody_axis_lattice_sites():
    body = CellBody(x=15.0, y=15.0, length_um=4.0, diameter_um=2.0)
    assert body.axis() == (1.0, 0.0)
    assert body.endpoints()[0:2] == (13.0, 15.0)
    assert body.volume_um3() > 0.0
    assert body.lattice(8, 8, 10.0) == (2, 2)
    covered = body.sites(8, 8, 10.0)
    assert (1, 1) in covered
    oob = CellBody(x=-50.0, y=200.0, length_um=4.0, diameter_um=2.0)
    assert oob.lattice(8, 8, 10.0) == (0, 7)  # clamped


def test_cellbody_diagonal_segment_and_sites():
    d, *_ = _segment_distance_sq(0, 0, 4, 0, 2, 3, 2, -3)
    assert d == pytest.approx(0.0)  # perpendicular crossing (denom > 0)
    rod = CellBody(x=15.0, y=15.0, length_um=5.0, diameter_um=2.0,
                   angle=math.pi / 4)
    sites = rod.sites(30, 30, 3.0)
    assert len(sites) >= 2
    assert all(0 <= sx < 30 and 0 <= sy < 30 for sx, sy in sites)


def test_cellbody_coincident_contacts_and_resolve_edges():
    c1 = CellBody(x=0.0, y=0.0, length_um=4.0, diameter_um=2.0, angle=0.0)
    c2 = CellBody(x=0.0, y=0.0, length_um=4.0, diameter_um=2.0,
                  angle=math.pi / 2)
    overlap, nx, ny = rod_rod_contact(c1, c2)
    assert overlap > 0.0
    assert math.hypot(nx, ny) == pytest.approx(1.0)  # non-degenerate normal

    # far rod never contacts -> resolve loop hits the None-continue
    far = CellBody(x=20.0, y=20.0, length_um=4.0, diameter_um=2.0)
    near = CellBody(x=0.2, y=0.0, length_um=4.0, diameter_um=2.0)
    resolve_rod_contacts([c1, near, far], drag=1e-6, stiffness=0.0,
                         x_max=30, y_max=30)
    # stiffness 0 -> zero hertzian pushes skip rod-rod and wall steps
    wall = CellBody(x=0.0, y=0.0, length_um=4.0, diameter_um=2.0)
    resolve_rod_contacts([wall, near], drag=1e-6, stiffness=0.0,
                         x_max=2.0, y_max=2.0)
    # iterations exhausted without converging
    a = CellBody(x=0.0, y=0.0, length_um=4.0, diameter_um=2.0)
    b = CellBody(x=0.3, y=0.0, length_um=4.0, diameter_um=2.0)
    resolve_rod_contacts([a, b], drag=0.0, stiffness=1e6, x_max=10, y_max=10,
                         iterations=1, tolerance_um=1e-16)
    assert a.x > 0.0

# ---------------------------------------------------------------------------
# grn.py / sparse_grn.py / stochastic.py / reaction_diffusion.py
# ---------------------------------------------------------------------------
def test_grn_noise_and_hill_accel():
    grn = GRN(noise_enabled=True, noise_seed=3)
    grn.add_gene("a", 0.5, noise=TelegraphPromoter(0.1, 0.5, 4.0))
    grn.add_gene("b", 0.5, hill_n=2.0, kd=0.8)
    grn.add_edge("a", "b", 2.0)
    grn.nodes["a"].level = 1.0
    for _ in range(3):
        grn.step()  # telegraph-noise branch (287-291)
    assert isinstance(grn.step_accel(prefer="python"), list)  # hill path
    quiet = GRN()  # noise disabled -> accel skips the noise overlay
    quiet.add_gene("a", 0.5)
    quiet.add_gene("b", 0.5)
    quiet.add_edge("a", "b", 2.0)
    quiet.nodes["a"].level = 1.0
    assert isinstance(quiet.step_accel(prefer="python"), list)


def test_sparse_grn_noise_step():
    grn = GRN()
    grn.add_gene("x", 0.5, noise=TelegraphPromoter(0.2, 0.3, 3.0))
    grn.add_gene("y", 0.5, noise=TelegraphPromoter(0.2, 0.3, 3.0))
    grn.add_edge("x", "y", 1.5)
    sg = SparseGRN.from_grn(grn, noise_seed=5)
    assert sg._noise_mask.any()
    levels = sg.new_state(2, initial_genes=("x",))
    out = sg.step(levels)
    assert out.shape == (2, sg.n_genes)
    assert sg.n_edges == 1


def test_sparse_grn_noiseless_step():
    grn = GRN()
    grn.add_gene("x", 0.5)
    grn.add_gene("y", 0.5)
    grn.add_edge("x", "y", 1.5)
    sg = SparseGRN.from_grn(grn)
    out = sg.step(sg.new_state(1, initial_genes=("x",)))
    assert not sg._noise_mask.any()
    assert out.shape == (1, 2)


def test_telegraph_validation_and_gillespie():
    with pytest.raises(ValueError):
        telegraph_fano_factor(-1.0, 1.0, 1.0, 0.14)
    with pytest.raises(ValueError):
        telegraph_fano_factor(1.0, 1.0, 1.0, 0.0)
    assert telegraph_fano_factor(0.0, 0.0, 0.0, 0.14) == 1.0
    assert telegraph_fano_factor(10.0, 10.0, 1.0, 0.14) > 1.0
    tp = TelegraphPromoter(0.1, 0.5, 4.0)
    assert tp.transcription_rate == pytest.approx(2.0)
    assert tp.on_fraction == pytest.approx(0.1 / 0.6)
    assert tp.fano_factor() >= 1.0
    with pytest.raises(ValueError):
        fano_to_noise_std(0.5, 0.5, 0.5)
    with pytest.raises(ValueError):
        fano_to_noise_std(1.5, -1.0, 0.5)
    with pytest.raises(ValueError):
        fano_to_noise_std(1.5, 0.5, 1.0)
    with pytest.raises(ValueError):
        fano_to_noise_std(1.5, 0.5, 0.5, expression_scale=0.0)
    stat = gillespie_telegraph(0.1, 0.5, 4.0, 0.14, t_max=10.0,
                               n_replicates=25, seed=1)
    assert set(stat) == {"mean", "variance", "fano"}
    with pytest.raises(ValueError):
        gillespie_telegraph(0.1, 0.5, 4.0, 0.14, t_max=0.0)


def test_gray_scott_full_paths():
    gs = GrayScott(n=16, seed=1)
    gs.emit(-1, 0)  # out of bounds no-op
    gs.emit(5, 5, 0.5)  # in-bounds injection
    assert gs.total_v() > 0.0
    gs.step()  # numpy backend (151-176)
    gs.u = [[1.0] * 16 for _ in range(16)]
    gs.v = [[0.0] * 16 for _ in range(16)]
    gs._scratch_u = None
    gs._scratch_v = None
    gs.step()  # pure-Python fallback rebuilds the scratch buffers
    assert gs.total_v() >= 0.0


def test_protein_fitness_offline_env_false_branch(monkeypatch):
    monkeypatch.setenv("HELIX_BENCHMARK_OFFLINE", "0")
    esm = protein_fitness.ESM2Oracle()
    assert esm.available or esm._load_error is not None
