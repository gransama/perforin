import os
import re
from pathlib import Path

import pytest

os.environ["PERFORIN_OFFLINE"] = "1"      # los tests no dependen de la red: el WTI sale del respaldo guardado

APP = str(Path(__file__).resolve().parents[1] / "src" / "ui" / "app.py")

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def _app():
    return AppTest.from_file(APP, default_timeout=120).run()


def _evaluate(at):
    at.button(key="run_btn").click().run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def _verdict(at):
    return next(m.value for m in at.markdown if "class='verdict'" in m.value)


def _van(at):
    """VAN10 (USD MM) que muestra el encabezado."""
    v = re.search(r"VAN10</div><div class='n'>USD (-?[\d.]+) MM", " ".join(m.value for m in at.markdown))
    return float(v.group(1))


COORDS = re.compile(r"-?\d+\.\d{4}, -?\d+\.\d{4}")


def test_initial_screen_asks_to_configure():
    at = _app()
    assert not at.exception
    assert any("Evaluar" in i.value for i in at.info)


def test_evaluation_shows_recommended_location():
    at = _evaluate(_app())
    v = _verdict(at)
    assert COORDS.search(v) and any(d in v for d in ("GO", "CONDITIONAL", "NO-GO"))
    assert not at.warning                      # sin datos de respaldo


def test_manual_geology_mode_exposes_sliders():
    at = _app()
    at.checkbox(key="manual").set_value(True).run()
    labels = [s.label for s in at.slider]
    assert "Profundidad (m)" in labels and "Permeabilidad (mD)" in labels
    _evaluate(at)


@pytest.mark.parametrize("agent", ["location", "geological", "forecasting", "economic"])
def test_fallback_when_agent_fails(agent):
    at = _app()
    at.multiselect(key="simulate_fail").set_value([agent])
    _evaluate(at)
    assert at.warning                          # avisa que usa datos de respaldo
    assert COORDS.search(_verdict(at))


@pytest.mark.parametrize("preset,expected", [("Neuquién Star · pozo fuerte", "GO"),
                                             ("Astral Marginal · pozo dudoso", "CONDITIONAL"),
                                             ("Cuyo Cuyo Dud · pozo malo", "NO-GO")])
def test_preset_loads_and_evaluates_in_one_click(preset, expected):
    at = _app()
    at.selectbox(key="preset").select(preset).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert f"<div class='t'>{expected} ·" in _verdict(at)      # exacto: "GO" no debe coincidir con "NO-GO"


def test_lower_royalty_improves_npv():
    at = _evaluate(_app())
    base = _van(at)
    at.slider(key="royalty").set_value(12.0).run()
    _evaluate(at)
    assert _van(at) > base


def test_crude_discount_and_export_duty_reduce_npv():
    at = _evaluate(_app())
    base = _van(at)
    at.slider(key="discount").set_value(10.0).run()
    at.slider(key="duty").set_value(10.0).run()
    _evaluate(at)
    assert _van(at) < base


def test_market_tab_uses_saved_wti_when_offline():
    at = _evaluate(_app())
    text = " ".join([m.value for m in at.markdown] + [c.value for c in at.caption])
    assert "dato de respaldo" in text and "WTI actual" in text
    assert "Tendencia" in text


def test_report_name_and_author_inputs_do_not_break_the_app():
    at = _app()
    at.text_input(key="project_name").set_value("Campaña Oveja Muerta 2026").run()
    at.text_input(key="author").set_value("Equipo Perforin").run()
    _evaluate(at)
    assert at.text_input(key="project_name").value == "Campaña Oveja Muerta 2026"


def test_example_does_not_erase_report_name():
    at = _app()
    at.text_input(key="project_name").set_value("Mi proyecto").run()
    at.selectbox(key="preset").select("Astral Marginal · pozo dudoso").run()
    assert at.text_input(key="project_name").value == "Mi proyecto"


def test_use_market_wti_button_sets_slider():
    at = _app()
    at.button(key="use_wti").click().run()
    assert not at.exception
    assert 20 <= at.slider(key="wti").value <= 120 and at.slider(key="wti").value != 65
