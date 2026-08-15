"""POST /api/sim/run round-trip (doc/12-helix-language-wiring.md §9, W-4)."""
from __future__ import annotations

import pytest

from helixlang.server import create_app


@pytest.fixture()
def client():
    app = create_app()
    app.testing = True
    return app.test_client()


def test_api_sim_run_whole_cell(client):
    """A whole-cell program round-trips through /api/sim/run."""
    src = """
#promoter name=p_constitutive strength=-0.4
#gene name=gltA promoter=p_constitutive
ATG GCT GGT GCT TAA
#end
#media nutrient=GLC concentration=10.0
#config backend=whole_cell
#config division_rule=adder adder_volume_um3=1.6
#config ticks=5
#config output=age,energy,divisions
"""
    r = client.post("/api/sim/run", json={"source": src})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["backend"] == "whole_cell"
    assert body["columns"] == ["age", "energy", "divisions"]
    assert len(body["rows"]) == 5


def test_api_sim_run_fba(client):
    src = """
#media nutrient=GLC concentration=10.0
#config backend=fba
#config output=BIOMASS
"""
    r = client.post("/api/sim/run", json={"source": src})
    assert r.status_code == 200
    body = r.get_json()
    assert body["backend"] == "fba"
    assert body["rows"][0]["BIOMASS"] > 0.0


def test_api_sim_run_backend_override(client):
    """body.backend overrides the source's #config backend."""
    src = "#media nutrient=GLC concentration=10.0\n#config output=BIOMASS"
    r = client.post("/api/sim/run", json={"source": src, "backend": "fba"})
    assert r.status_code == 200
    assert r.get_json()["backend"] == "fba"


def test_api_sim_run_unknown_backend(client):
    r = client.post("/api/sim/run", json={"source": "#config backend=bogus"})
    assert r.status_code == 400
    assert r.get_json()["ok"] is False
    assert "backend" in r.get_json()["error"]
