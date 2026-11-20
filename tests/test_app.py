"""UI smoke test with Streamlit's headless AppTest."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_runs_all_presets_and_flags_stale():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception
    assert any("Sourced cohort" in m.value for m in at.markdown)
    labels = {m.label: m.value for m in at.metric}
    assert labels["Flights receiving SAF"] == "3 of 12"
    at.sidebar.radio(key="w_info").set_value("No update (one group)").run()
    assert not at.exception
    for preset in ["Illustrative Stockholm simulation (12 synthetic rows)",
                   "Small teaching fixture (4 synthetic rows)"]:
        at.sidebar.selectbox(key="w_preset").set_value(preset).run()
        assert not at.exception
        assert any("synthetic" in m.value.lower() for m in at.markdown)
    at.sidebar.toggle(key="w_auto").set_value(False).run()
    at.sidebar.number_input(key="w_stock").set_value(0.0).run()
    assert any("previous run" in m.value for m in at.markdown)
    at.sidebar.button[0].click().run()
    assert not any("previous run" in m.value for m in at.markdown)
    assert {m.label: m.value for m in at.metric}["Flights receiving SAF"] == "0 of 4"
