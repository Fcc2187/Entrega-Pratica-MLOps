"""Prepare Reference/Production datasets, benchmark /predict latency, and collect responses."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from http.client import HTTPException
from pathlib import Path
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import numpy as np
import pandas as pd

from careerpath.schema import Profile
from careerpath.train import FEATURES, load_adult, remove_test_overlap

DEFAULT_EVIDENCE_DIR = Path("docs/av2-p2/evidence")
DEFAULT_DATA_DIR = Path("data/raw")
DEFAULT_METADATA_PATH = Path("models/metadata.json")
MIN_SAMPLES = 100

# Realistic production shifts representing broader technical/managerial adoption.
OCCUPATION_SHIFT_WEIGHTS: dict[str, float] = {
    "Prof-specialty": 2.1,
    "Exec-managerial": 2.0,
    "Tech-support": 2.2,
    "Sales": 1.4,
    "Adm-clerical": 1.2,
    "Craft-repair": 0.85,
    "Machine-op-inspct": 0.65,
    "Handlers-cleaners": 0.50,
    "Farming-fishing": 0.45,
    "Priv-house-serv": 0.40,
}


class NoRedirect(HTTPRedirectHandler):
    """Disallow HTTP redirects on inference endpoint calls."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def validate_base_url(url: str) -> str:
    """Validate an ASCII HTTP/HTTPS base URL without credentials, query, or fragment."""
    if "?" in url or "#" in url or any(not 33 <= ord(char) <= 126 for char in url):
        raise ValueError("--url must use printable ASCII without spaces, query or fragment")
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("--url must be a valid HTTP or HTTPS base URL") from exc
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("--url must be an HTTP or HTTPS base URL")
    if port == 0 or parsed.username is not None:
        raise ValueError("--url must use a nonzero port and omit credentials")
    return url.rstrip("/")


def _clean_nullable(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text if text and text != "?" else None


def row_to_profile(row: Mapping[str, object]) -> dict[str, object]:
    """Convert an Adult dataset row to a validated API Profile dictionary."""
    payload: dict[str, object] = {
        "age": int(row["age"]),
        "education": str(row["education"]).strip(),
        "workclass": _clean_nullable(row["workclass"]),
        "occupation": _clean_nullable(row["occupation"]),
        "marital_status": str(row["marital-status"]).strip(),
        "hours_per_week": int(row["hours-per-week"]),
    }
    return Profile.model_validate(payload).model_dump()


def _verify_adult_hashes(data_dir: Path, metadata_path: Path) -> None:
    if not metadata_path.is_file():
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = metadata.get("data_sha256", {})
    for filename in ("adult.data", "adult.test"):
        target = data_dir / filename
        if filename in expected and target.is_file():
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
            if actual != expected[filename]:
                raise ValueError(f"SHA-256 mismatch for {target}")


def load_jsonl_profiles(path: Path, count: int) -> list[dict[str, object]]:
    """Load and validate at least `count` Profile objects from an offline JSONL fixture."""
    if not path.is_file():
        raise FileNotFoundError(f"Fixture not found: {path}")
    profiles: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        profiles.append(Profile.model_validate(json.loads(stripped)).model_dump())
        if len(profiles) >= count:
            break
    if len(profiles) < count:
        raise ValueError(f"Fixture {path} has {len(profiles)} valid profiles, expected >= {count}")
    return profiles


def build_datasets(
    count: int,
    *,
    seed: int = 42,
    data_dir: Path = DEFAULT_DATA_DIR,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    evidence_dir: Path = DEFAULT_EVIDENCE_DIR,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, str]]:
    """Build deterministic Reference (adult.data) and realistic Production (adult.test) sets."""
    if count < MIN_SAMPLES:
        raise ValueError(f"count must be >= {MIN_SAMPLES}, got {count}")

    train_path = data_dir / "adult.data"
    test_path = data_dir / "adult.test"
    if train_path.is_file() and test_path.is_file():
        _verify_adult_hashes(data_dir, metadata_path)
        raw_train = load_adult(train_path)
        raw_test = load_adult(test_path, test_file=True)
        clean_train, _ = remove_test_overlap(raw_train, raw_test)

        ref_sample = clean_train[FEATURES].sample(n=count, random_state=seed)
        reference = [row_to_profile(rec) for rec in ref_sample.to_dict(orient="records")]

        rng = np.random.default_rng(seed)
        test_df = raw_test[FEATURES].reset_index(drop=True)
        occ_series = test_df["occupation"].fillna("(missing)").astype(str)
        weights = occ_series.map(lambda val: OCCUPATION_SHIFT_WEIGHTS.get(val, 1.0)).to_numpy(
            dtype=float
        )
        probs = weights / weights.sum()
        chosen_idx = rng.choice(len(test_df), size=count, replace=False, p=probs)
        prod_df = test_df.iloc[chosen_idx].copy().reset_index(drop=True)
        age_delta = rng.integers(1, 7, size=count)
        hours_delta = rng.integers(2, 8, size=count)
        prod_df["age"] = np.clip(prod_df["age"].to_numpy() + age_delta, 17, 100)
        prod_df["hours-per-week"] = np.clip(
            prod_df["hours-per-week"].to_numpy() + hours_delta,
            1,
            99,
        )
        production = [row_to_profile(rec) for rec in prod_df.to_dict(orient="records")]
        provenance = {
            "reference_source": "UCI Adult train partition (data/raw/adult.data, overlap removed)",
            "production_source": (
                "UCI Adult test partition (data/raw/adult.test) with controlled realistic drift"
            ),
        }
        return reference, production, provenance

    ref_fixture = evidence_dir / "reference-inputs.jsonl"
    prod_fixture = evidence_dir / "production-inputs.jsonl"
    reference = load_jsonl_profiles(ref_fixture, count)
    production = load_jsonl_profiles(prod_fixture, count)
    provenance = {
        "reference_source": f"Offline fixture ({ref_fixture})",
        "production_source": f"Offline fixture ({prod_fixture})",
    }
    return reference, production, provenance


def write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    """Write sequence of JSON objects as UTF-8 JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(dict(record), allow_nan=False) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def submit_profiles(
    base_url: str,
    profiles: Sequence[Mapping[str, object]],
    *,
    dataset_name: str,
    opener: object | None = None,
    timeout: float = 10.0,
) -> list[dict[str, object]]:
    """Submit valid profiles sequentially to POST /predict and record client latency & outputs."""
    endpoint = f"{base_url}/predict"
    http_opener = opener if opener is not None else build_opener(NoRedirect())
    records: list[dict[str, object]] = []

    for index, profile in enumerate(profiles, start=1):
        body = json.dumps(dict(profile)).encode("utf-8")
        request = Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = perf_counter()
        try:
            with http_opener.open(request, timeout=timeout) as response:
                status = int(response.status)
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            status = int(error.code)
            error.close()
            raise RuntimeError(
                f"{dataset_name} request #{index} failed with HTTP {status} at {endpoint}"
            ) from error
        except (OSError, HTTPException, URLError) as error:
            raise RuntimeError(
                f"Failed to reach {endpoint} on {dataset_name} request #{index}: {error}"
            ) from error
        elapsed_ms = (perf_counter() - started) * 1000.0
        if status != 200:
            raise RuntimeError(
                f"{dataset_name} request #{index} returned unexpected HTTP {status}"
            )
        records.append(
            {
                "index": index,
                "dataset": dataset_name,
                "timestamp": datetime.now(UTC).isoformat(timespec="microseconds"),
                "status_code": status,
                "client_latency_ms": float(elapsed_ms),
                **dict(profile),
                **dict(payload),
            }
        )
    return records


def build_benchmark_payloads(
    reference: Sequence[dict[str, object]],
    production: Sequence[dict[str, object]],
    *,
    count: int,
) -> list[dict[str, object]]:
    """Interleave Reference and Production profiles to reach `count` benchmark calls."""
    if count < MIN_SAMPLES:
        raise ValueError(f"benchmark count must be >= {MIN_SAMPLES}, got {count}")
    pool: list[dict[str, object]] = []
    for ref_item, prod_item in zip(reference, production, strict=False):
        pool.append(ref_item)
        pool.append(prod_item)
    if not pool:
        raise ValueError("Cannot build benchmark payloads from empty pools")
    return [pool[i % len(pool)] for i in range(count)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:3000", help="Base URL of BentoML")
    parser.add_argument("--count", type=int, default=500, help="Samples per dataset (>= 100)")
    parser.add_argument(
        "--benchmark-count",
        type=int,
        default=500,
        help="Latency benchmark request count (>= 100)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only generate/validate reference and production inputs without calling HTTP",
    )
    args = parser.parse_args(argv)

    try:
        base_url = validate_base_url(args.url)
        reference, production, provenance = build_datasets(
            args.count,
            seed=args.seed,
            data_dir=args.data_dir,
            metadata_path=args.metadata_path,
            evidence_dir=args.evidence_dir,
        )
        args.evidence_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.evidence_dir / "reference-inputs.jsonl", reference)
        write_jsonl(args.evidence_dir / "production-inputs.jsonl", production)

        summary: dict[str, object] = {
            "seed": args.seed,
            "count": args.count,
            "provenance": provenance,
            "prepare_only": args.prepare_only,
        }

        if not args.prepare_only:
            bench_payloads = build_benchmark_payloads(
                reference, production, count=args.benchmark_count
            )
            bench_records = submit_profiles(base_url, bench_payloads, dataset_name="benchmark")
            ref_records = submit_profiles(base_url, reference, dataset_name="reference")
            prod_records = submit_profiles(base_url, production, dataset_name="production")

            write_jsonl(args.evidence_dir / "latency-benchmark.jsonl", bench_records)
            write_jsonl(args.evidence_dir / "reference-responses.jsonl", ref_records)
            write_jsonl(args.evidence_dir / "production-responses.jsonl", prod_records)
            summary.update(
                {
                    "url": base_url,
                    "benchmark_sent": len(bench_records),
                    "reference_sent": len(ref_records),
                    "production_sent": len(prod_records),
                }
            )
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"Erro na coleta da Parte 2: {error}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
