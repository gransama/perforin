from pathlib import Path

import pytest

APP = str(Path(__file__).resolve().parents[1] / "src" / "ui" / "app.py")

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def _app():
    return AppTest.from_file(APP, default_timeout=120).run()


def _evaluate(at):
    at.button[0].click().run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def _verdict(at):
    return next(m.value for m in at.markdown if "class='verdict'" in m.value)


def test_initial_screen_asks_to_configure():
    at = _app()
    assert not at.exception
    assert any("Evaluar" in i.value for i in at.info)


def test_evaluation_shows_recommended_location():
    at = _evaluate(_app())
    v = _verdict(at)
    assert "📍" in v and any(d in v for d in ("GO", "CONDITIONAL", "NO-GO"))
    assert not at.warning                      # sin datos de respaldo


def test_manual_geology_mode_exposes_sliders():
    at = _app()
    at.checkbox[0].set_value(True).run()
    labels = [s.label for s in at.slider]
    assert "Profundidad (m)" in labels and "Permeabilidad (mD)" in labels
    _evaluate(at)


@pytest.mark.parametrize("agent", ["location", "geological", "forecasting", "economic"])
def test_fallback_when_agent_fails(agent):
    at = _app()
    at.multiselect[0].set_value([agent])
    _evaluate(at)
    assert at.warning                          # avisa que usa datos de respaldo
    assert "📍" in _verdict(at)
