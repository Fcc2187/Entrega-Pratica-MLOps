"""JSONL loading and numeric extraction helpers for AV2 Part 2 analytics."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd


def to_finite_floats(values: Sequence[object]) -> tuple[list[float], int]:
    """Convert numeric sequences to finite floats, counting skipped non-finite/missing items."""
    clean: list[float] = []
    skipped = 0
    for raw in values:
        if raw is None or isinstance(raw, bool) or pd.isna(raw):
            skipped += 1
            continue
        val = float(raw)
        if not math.isfinite(val):
            skipped += 1
            continue
        clean.append(val)
    return clean, skipped


def extract_numeric_column(records: Sequence[Mapping[str, object]], key: str) -> list[float]:
    """Extract finite float values for `key` from JSONL-decoded record mappings."""
    values: list[float] = []
    for row in records:
        val = row.get(key)
        if isinstance(val, int | float) and not isinstance(val, bool) and math.isfinite(val):
            values.append(float(val))
    return values


def load_jsonl_records(path: Path) -> list[dict[str, object]]:
    """Read UTF-8 JSONL file into a list of dicts, failing clearly if malformed."""
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    records: list[dict[str, object]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{lineno}: {exc.msg}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"Expected JSON object at {path}:{lineno}")
        records.append(obj)
    return records


def extract_inference_latencies(
    jsonl_path: Path,
    *,
    endpoint: str = "/predict",
    only_success: bool = False,
) -> list[float]:
    """Extract finite latency_ms values from monitoring JSONL events."""
    events = [
        event
        for event in load_jsonl_records(jsonl_path)
        if event.get("endpoint") == endpoint
        and (not only_success or event.get("outcome") == "success")
    ]
    return extract_numeric_column(events, "latency_ms")


def extract_valid_prediction_records(
    jsonl_path: Path,
    *,
    endpoint: str = "/predict",
) -> list[dict[str, object]]:
    """Extract flat input+output records from complete successful inference events."""
    events = load_jsonl_records(jsonl_path)
    rows: list[dict[str, object]] = []
    for event in events:
        if event.get("endpoint") != endpoint:
            continue
        if event.get("outcome") != "success" or event.get("input_capture") != "complete":
            continue
        inputs = event.get("input")
        outputs = event.get("output")
        if not isinstance(inputs, dict) or not isinstance(outputs, dict):
            continue
        rows.append(
            {
                **inputs,
                **outputs,
                "latency_ms": event.get("latency_ms"),
                "timestamp": event.get("timestamp"),
                "request_id": event.get("request_id"),
            }
        )
    return rows
