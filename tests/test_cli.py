import json

import pytest

from src import cli

SOLICITUD = {"basin": "Neuquién", "formation": "Oveja Muerta", "stimulation_type": "acidizing",
             "acid_volume_m3": 20, "proppant_tonnes": 0, "wti_price_usd": 65}


def _setup(tmp_path, solicitud=SOLICITUD):
    d = tmp_path / "corrida"
    d.mkdir()
    (d / "00_solicitud.json").write_text(json.dumps(solicitud, ensure_ascii=False), encoding="utf-8")
    return d


def test_full_run_writes_every_file_and_report(tmp_path, capsys):
    d = _setup(tmp_path)
    assert cli.main(["evaluar", "--dir", str(d)]) == 0
    for f in cli.FILES.values():
        assert (d / f).exists(), f
    out = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert [o.get("paso") for o in out[:5]] == cli.STEPS and all(o["estado"] == "ok" for o in out[:5])
    eco = json.loads((d / "05_economico.json").read_text(encoding="utf-8"))
    assert eco["decision"] in {"GO", "CONDITIONAL", "NO-GO"}
    informe = (d / "informe.md").read_text(encoding="utf-8")
    assert eco["decision"] in informe and "Ubicación recomendada" in informe and "Neuquién" in informe


def test_steps_match_the_python_pipeline(tmp_path):
    """El CLI da los mismos números que llamar a los agentes directamente."""
    from src.agents.economic_agent import EconomicAgent
    from src.agents.forecasting_agent import ForecastingAgent
    from src.agents.geological_agent import GeologicalAgent
    from src.agents.ingestion_agent import IngestionAgent

    d = _setup(tmp_path)
    assert cli.main(["evaluar", "--dir", str(d)]) == 0
    raw = json.loads((d / "01_pozo_crudo.json").read_text(encoding="utf-8"))
    well = IngestionAgent().run(raw)
    geo = GeologicalAgent().run(well)
    eco = EconomicAgent().run(ForecastingAgent().run(geo, well), well)
    got = json.loads((d / "05_economico.json").read_text(encoding="utf-8"))
    assert got["npv10_usd_mm"] == pytest.approx(eco["npv10_usd_mm"])
    assert got["decision"] == eco["decision"]


def test_single_steps_in_order(tmp_path):
    d = _setup(tmp_path)
    for name in cli.STEPS:
        assert cli.main(["paso", name, "--dir", str(d)]) == 0


def test_missing_previous_file_exits_2(tmp_path, capsys):
    d = _setup(tmp_path)
    assert cli.main(["paso", "geologico", "--dir", str(d)]) == 2
    assert "Falta 02_ingesta.json" in capsys.readouterr().err


def test_invalid_request_exits_2(tmp_path, capsys):
    d = _setup(tmp_path, {"basin": "Neuquién"})
    assert cli.main(["paso", "recomendar", "--dir", str(d)]) == 2
    assert "formation" in capsys.readouterr().err
    assert cli.main(["informe", "--dir", str(d)]) == 2


def test_invalid_well_exits_2(tmp_path, capsys):
    d = _setup(tmp_path)
    cli.main(["paso", "recomendar", "--dir", str(d)])
    raw = json.loads((d / "01_pozo_crudo.json").read_text(encoding="utf-8"))
    raw["depth_m"] = 99999
    (d / "01_pozo_crudo.json").write_text(json.dumps(raw), encoding="utf-8")
    assert cli.main(["paso", "ingesta", "--dir", str(d)]) == 2
    assert "depth_m" in capsys.readouterr().err


def test_request_file_with_bom_is_accepted(tmp_path):
    """PowerShell guarda los JSON con BOM; el CLI debe leerlos igual."""
    d = tmp_path / "corrida"
    d.mkdir()
    (d / "00_solicitud.json").write_bytes(b"\xef\xbb\xbf" + json.dumps(SOLICITUD, ensure_ascii=False).encode("utf-8"))
    assert cli.main(["paso", "recomendar", "--dir", str(d)]) == 0

def test_economic_step_leaves_the_report_ready(tmp_path):
    d = _setup(tmp_path)
    for name in cli.STEPS:
        assert cli.main(["paso", name, "--dir", str(d)]) == 0
    assert (d / "informe.md").exists()