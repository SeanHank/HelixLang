"""Shared pytest fixtures and path bootstrap.

Shared by all test modules to avoid redeclaring common fixtures (Flask test
client, example source code, the examples directory path, etc.).
"""
import sys
from pathlib import Path

import pytest

# Let tests find src/ even when the package is not installed
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


# ============================================================================
# Path fixtures
# ============================================================================

@pytest.fixture(scope="session")
def examples_dir() -> Path:
    """Path to the examples/ directory (session-scoped, shared by all tests)."""
    return EXAMPLES


@pytest.fixture(scope="session")
def src_dir() -> Path:
    """Path to the src/ directory."""
    return SRC


# ============================================================================
# HelixLang source fixtures (commonly used minimal compilable snippets)
# ============================================================================

@pytest.fixture
def hello_src() -> str:
    """Minimal compilable source: single gene hello + 5 ticks."""
    return "#gene name=hello\nATG GCT GGT TAA\n#end\n#config ticks=5\n"


@pytest.fixture
def lac_src() -> str:
    """lac operon example: promoter + two genes + negative regulation + 10 ticks."""
    return (
        "#promoter name=plac strength=-1.0\n#end\n"
        "#gene name=laci promoter=plac\nATG GCT GCT GCT TAA\n#end\n"
        "#gene name=lacoperon promoter=plac\nATG GGT GGT TAA\n#end\n"
        "#regulate laci -> lacoperon strength=-0.5\n#end\n"
        "#config ticks=10\n"
    )


@pytest.fixture
def table_switch_src() -> str:
    """Example of switching between variable translation tables."""
    return "#table vertebrate_mito\nATG TAA\n#end\n#config ticks=1\n"


# ============================================================================
# Flask web app fixtures
# ============================================================================

@pytest.fixture(scope="session")
def app():
    """Flask app (session-scoped).

    flask is an optional web dependency; tests depending on this fixture are
    skipped when it is missing.
    """
    pytest.importorskip("flask")
    from helixlang.server import create_app
    application = create_app()
    application.config["TESTING"] = True
    return application


@pytest.fixture
def client(app):
    """Flask test client (function-scoped, independent request context per test)."""
    with app.test_client() as c:
        yield c
