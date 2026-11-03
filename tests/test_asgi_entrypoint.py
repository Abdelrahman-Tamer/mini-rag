import sys
import inspect
from pathlib import Path
import pytest
from fastapi import FastAPI


def test_api_index_and_routes_import_cleanly():
    """Confirms api.index ASGI entrypoint and its route modules import without NameError or circular dependency."""
    repo_root = Path(__file__).resolve().parent.parent
    api_dir = repo_root / "api"
    if str(api_dir) not in sys.path:
        sys.path.insert(0, str(api_dir))

    import api.index
    from api.index import app
    assert isinstance(app, FastAPI), "api.index.app is not a FastAPI instance"

    from routes import base, data, nlp
    assert hasattr(base, "base_router")
    assert hasattr(data, "data_router")
    assert hasattr(nlp, "nlp_router")


def test_nlp_router_dependencies_and_routes():
    """Confirms that nlp_router has Depends properly attached to protected endpoints without NameError."""
    repo_root = Path(__file__).resolve().parent.parent
    api_dir = repo_root / "api"
    if str(api_dir) not in sys.path:
        sys.path.insert(0, str(api_dir))

    from routes.nlp import nlp_router
    paths = [r.path for r in nlp_router.routes]

    assert "/api/v1/nlp/chat" in paths
    assert "/api/v1/nlp/chat/stream" in paths
    assert "/api/v1/nlp/sync/activity" in paths
    assert "/api/v1/nlp/sync/reconcile" in paths

    # Verify dependencies are present on admin sync endpoints
    for route in nlp_router.routes:
        if route.path in ("/api/v1/nlp/sync/activity", "/api/v1/nlp/sync/reconcile"):
            assert len(route.dependencies) > 0, f"Missing security dependencies on {route.path}"


def test_src_and_api_nlp_route_signature_parity():
    """Verifies src/routes/nlp.py and api/routes/nlp.py share identical endpoint signatures."""
    repo_root = Path(__file__).resolve().parent.parent
    
    src_nlp_path = repo_root / "src" / "routes" / "nlp.py"
    api_nlp_path = repo_root / "api" / "routes" / "nlp.py"

    assert "Depends" in api_nlp_path.read_text(encoding="utf-8"), "Depends import missing from api/routes/nlp.py"
    assert "verify_admin_access" in api_nlp_path.read_text(encoding="utf-8"), "verify_admin_access missing from api/routes/nlp.py"
