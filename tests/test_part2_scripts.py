import json
from http.client import HTTPException
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace
from urllib.error import URLError

import pytest
from streamlit.testing.v1 import AppTest

from careerpath.schema import Profile

PROJECT_ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def collect_mod():
    return run_path(PROJECT_ROOT / "scripts" / "part2_collect.py")


@pytest.fixture(scope="module")
def report_mod():
    return run_path(PROJECT_ROOT / "scripts" / "part2_report.py")


def _sample_profiles(count: int, *, shift: bool = False) -> list[dict[str, object]]:
    base = [
        {
            "age": 30 + (i % 25) + (5 if shift else 0),
            "education": "Bachelors" if (i % 2 == 0 or shift) else "HS-grad",
            "workclass": "Private" if i % 10 != 0 else None,
            "occupation": (
                None
                if i % 15 == 0
                else (
                    "Prof-specialty"
                    if (shift and i % 3 != 0)
                    else ("Tech-support" if i % 2 == 0 else "Craft-repair")
                )
            ),
            "marital_status": "Never-married" if i % 2 == 0 else "Married-civ-spouse",
            "hours_per_week": 35 + (i % 15) + (6 if shift else 0),
        }
        for i in range(count)
    ]
    for item in base:
        Profile.model_validate(item)
    return base


def _write_adult_raw(data_dir: Path, rows: int = 160) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    occupations = ["Tech-support", "Craft-repair", "Prof-specialty", "Exec-managerial", "?"]
    educations = ["Bachelors", "HS-grad", "Masters", "Some-college"]
    train_lines = []
    test_lines = ["|1x3 Cross validator"]
    for i in range(rows):
        age = 24 + (i % 35)
        workclass = "Private" if i % 12 != 0 else "?"
        occ = occupations[i % len(occupations)]
        edu = educations[i % len(educations)]
        hours = 30 + (i % 25)
        income = ">50K" if i % 4 == 0 else "<=50K"
        fields = [
            str(age),
            workclass,
            "100000",
            edu,
            "10",
            "Never-married" if i % 2 == 0 else "Married-civ-spouse",
            occ,
            "Not-in-family",
            "White",
            "Male",
            "0",
            "0",
            str(hours),
            "United-States",
            income,
        ]
        train_lines.append(", ".join(fields))
        # Shift test age slightly so train/test rows have no exact overlap.
        test_fields = [str(age + 1), *fields[1:-1], f"{income}."]
        test_lines.append(", ".join(test_fields))

    (data_dir / "adult.data").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (data_dir / "adult.test").write_text("\n".join(test_lines) + "\n", encoding="utf-8")


def test_build_datasets_from_raw_and_offline_fixtures(
    collect_mod, tmp_path: Path
) -> None:
    raw_dir = tmp_path / "raw"
    evidence_dir = tmp_path / "evidence"
    _write_adult_raw(raw_dir, rows=150)

    ref, prod, provenance = collect_mod["build_datasets"](
        100,
        seed=42,
        data_dir=raw_dir,
        metadata_path=tmp_path / "no-meta.json",
        evidence_dir=evidence_dir,
    )
    assert len(ref) == 100 and len(prod) == 100
    assert "adult.data" in provenance["reference_source"]
    for row in ref + prod:
        Profile.model_validate(row)

    # Now verify deterministic offline fallback when raw files are absent.
    collect_mod["write_jsonl"](evidence_dir / "reference-inputs.jsonl", ref)
    collect_mod["write_jsonl"](evidence_dir / "production-inputs.jsonl", prod)
    ref_off, prod_off, prov_off = collect_mod["build_datasets"](
        100,
        seed=42,
        data_dir=tmp_path / "missing-raw",
        evidence_dir=evidence_dir,
    )
    assert ref_off == ref and prod_off == prod
    assert "Offline fixture" in prov_off["reference_source"]

    with pytest.raises(ValueError, match="100"):
        collect_mod["build_datasets"](50, data_dir=raw_dir)


def test_collect_cli_and_report_cli_end_to_end_mocked_http(
    collect_mod,
    report_mod,
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    raw_dir = tmp_path / "raw"
    evidence_dir = tmp_path / "evidence"
    _write_adult_raw(raw_dir, rows=140)

    class FakeResponse:
        status = 200

        def __init__(self, payload: dict[str, object]) -> None:
            prob = 0.65 if payload.get("occupation") == "Prof-specialty" else 0.18
            self._body = json.dumps(
                {
                    "income_class_proxy": ">50K" if prob >= 0.5 else "<=50K",
                    "probability_above_50k": prob,
                    "model_version": "adult-income-v1",
                }
            ).encode("utf-8")

        def read(self) -> bytes:
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def fake_open(request, *, timeout):
        assert timeout == 10.0
        assert request.full_url == "http://127.0.0.1:3000/predict"
        return FakeResponse(json.loads(request.data.decode("utf-8")))

    monkeypatch.setitem(
        collect_mod["main"].__globals__,
        "build_opener",
        lambda *args: SimpleNamespace(open=fake_open),
    )

    rc = collect_mod["main"](
        [
            "--count",
            "100",
            "--benchmark-count",
            "100",
            "--data-dir",
            str(raw_dir),
            "--metadata-path",
            str(tmp_path / "no-meta.json"),
            "--evidence-dir",
            str(evidence_dir),
        ]
    )
    assert rc == 0
    collect_summary = json.loads(capsys.readouterr().out)
    assert collect_summary["benchmark_sent"] == 100
    assert collect_summary["reference_sent"] == 100
    assert collect_summary["production_sent"] == 100

    # Also test server-log reconciliation on report.
    server_log = tmp_path / "inference.jsonl"
    server_log.write_text(
        "\n".join(
            json.dumps(
                {
                    "endpoint": "/predict",
                    "outcome": "success",
                    "input_capture": "complete",
                    "latency_ms": 4.2 + i * 0.05,
                }
            )
            for i in range(100)
        )
        + "\n",
        encoding="utf-8",
    )

    rc_report = report_mod["main"](
        [
            "--evidence-dir",
            str(evidence_dir),
            "--inference-log",
            str(server_log),
        ]
    )
    assert rc_report == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["latency_client"]["count"] == 100
    assert rep["latency_client"]["sla_passed"] is True
    assert rep["primary_input_psi"]["variable"] == "occupation"
    assert rep["primary_output_ks"]["variable"] == "probability_above_50k"
    for chart_path in rep["charts"].values():
        assert Path(chart_path).is_file()
    assert (evidence_dir / "summary.json").is_file()


def test_collect_handles_offline_server_and_invalid_url(
    collect_mod, report_mod, tmp_path: Path, monkeypatch
) -> None:
    raw_dir = tmp_path / "raw"
    _write_adult_raw(raw_dir, rows=120)

    def fail_open(request, *, timeout):
        raise URLError("connection refused")

    monkeypatch.setitem(
        collect_mod["main"].__globals__,
        "build_opener",
        lambda *args: SimpleNamespace(open=fail_open),
    )
    assert (
        collect_mod["main"](
            [
                "--count",
                "100",
                "--benchmark-count",
                "100",
                "--data-dir",
                str(raw_dir),
                "--evidence-dir",
                str(tmp_path / "ev"),
            ]
        )
        == 1
    )

    assert collect_mod["main"](["--url", "file:///tmp", "--prepare-only"]) == 1
    assert report_mod["main"](["--evidence-dir", str(tmp_path / "missing-ev")]) == 1

    bad_meta = tmp_path / "bad-meta.json"
    bad_meta.write_text(json.dumps({"data_sha256": {"adult.data": "0" * 64}}), encoding="utf-8")
    assert (
        collect_mod["main"](
            [
                "--prepare-only",
                "--count",
                "100",
                "--data-dir",
                str(raw_dir),
                "--metadata-path",
                str(bad_meta),
            ]
        )
        == 1
    )

    # Ensure HTTPException from urllib client is also handled cleanly.
    def fail_http(request, *, timeout):
        raise HTTPException("broken")

    with pytest.raises(RuntimeError, match="Failed to reach"):
        collect_mod["submit_profiles"](
            "http://127.0.0.1:3000",
            _sample_profiles(1),
            dataset_name="test",
            opener=SimpleNamespace(open=fail_http),
        )


def test_streamlit_dashboard_states_empty_error_and_success(
    tmp_path: Path, monkeypatch
) -> None:
    empty_dir = tmp_path / "empty_ev"
    empty_dir.mkdir()
    monkeypatch.setenv("CAREERPATH_PART2_EVIDENCE_DIR", str(empty_dir))

    app_empty = AppTest.from_file(str(PROJECT_ROOT / "scripts" / "dashboard.py")).run(timeout=20)
    assert not app_empty.exception
    assert len(app_empty.info) == 1
    assert "Nenhuma coleta encontrada" in app_empty.info[0].value

    bad_dir = tmp_path / "bad_ev"
    bad_dir.mkdir()
    (bad_dir / "latency-benchmark.jsonl").write_text("{broken\n", encoding="utf-8")
    (bad_dir / "reference-responses.jsonl").write_text("{}\n", encoding="utf-8")
    (bad_dir / "production-responses.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("CAREERPATH_PART2_EVIDENCE_DIR", str(bad_dir))

    app_err = AppTest.from_file(str(PROJECT_ROOT / "scripts" / "dashboard.py")).run(timeout=20)
    assert not app_err.exception
    assert len(app_err.error) == 1

    good_dir = tmp_path / "good_ev"
    good_dir.mkdir()
    ref_profiles = _sample_profiles(100, shift=False)
    prod_profiles = _sample_profiles(100, shift=True)
    bench_rows = [
        {"client_latency_ms": 8.0 + (i % 15) * 0.4, "status_code": 200} for i in range(100)
    ]
    ref_rows = [
        {**row, "probability_above_50k": 0.15 + (i % 10) * 0.02}
        for i, row in enumerate(ref_profiles)
    ]
    prod_rows = [
        {**row, "probability_above_50k": 0.40 + (i % 10) * 0.04}
        for i, row in enumerate(prod_profiles)
    ]

    for fname, rows in (
        ("latency-benchmark.jsonl", bench_rows),
        ("reference-responses.jsonl", ref_rows),
        ("production-responses.jsonl", prod_rows),
    ):
        (good_dir / fname).write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )
    monkeypatch.setenv("CAREERPATH_PART2_EVIDENCE_DIR", str(good_dir))

    app_ok = AppTest.from_file(str(PROJECT_ROOT / "scripts" / "dashboard.py")).run(timeout=30)
    assert not app_ok.exception
    assert len(app_ok.tabs) == 3
    assert len(app_ok.metric) >= 9
