"""Every page of the Streamlit app renders without an exception, with and without seeds."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PAGES = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "views").glob("*.py"))


@pytest.mark.parametrize("demo", [None, "indie"])
@pytest.mark.parametrize("page", PAGES)
def test_page_renders(engine, page, demo):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    if demo:
        at.query_params["demo"] = demo
    at.run()
    at.switch_page(page)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if demo:
        assert len(at.session_state["seeds"]) == 3
    assert len(at.markdown) > 0
