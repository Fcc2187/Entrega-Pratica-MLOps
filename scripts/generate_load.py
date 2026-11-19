"""Send reproducible synthetic profiles to the CareerPath prediction endpoint."""

import argparse
import json
from http.client import HTTPException
from pathlib import Path
from random import Random
from urllib.error import HTTPError
from urllib.request import Request, build_opener

from careerpath.http_utils import NoRedirect, validate_base_url

SCENARIOS = ("baseline", "shifted")
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def build_payloads(count: int, *, seed: int, scenario: str) -> list[dict[str, object]]:
    """Generate paired scenarios with the same categories and a numeric shift."""
    if count <= 0:
        raise ValueError("count must be positive")
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario}")

    templates = [
        json.loads((EXAMPLES / f"case-{index}.json").read_text(encoding="utf-8"))
        for index in (1, 2, 3)
    ]
    random = Random(seed)
    shift = 12 if scenario == "shifted" else 0
    return [
        {
            **random.choice(templates),
            "age": random.randint(22, 55) + shift,
            "hours_per_week": random.randint(28, 48) + shift,
        }
        for _ in range(count)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:3000", help="Base URL of the service")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario", choices=SCENARIOS, default="baseline")
    parser.add_argument("--invalid-every", type=int, default=0, help="0 disables invalid requests")
    args = parser.parse_args(argv)
    if args.count <= 0:
        parser.error("--count must be positive")
    if not 0 <= args.invalid_every <= args.count:
        parser.error("--invalid-every must be between 0 and --count")
    try:
        base_url = validate_base_url(args.url)
    except ValueError as error:
        parser.error(str(error))

    payloads = build_payloads(args.count, seed=args.seed, scenario=args.scenario)
    endpoint = f"{base_url}/predict"
    opener = build_opener(NoRedirect())
    counts = {
        "sent": 0,
        "2xx": 0,
        "4xx": 0,
        "5xx": 0,
        "other_status": 0,
        "network_errors": 0,
        "unexpected": 0,
    }
    for index, payload in enumerate(payloads, start=1):
        invalid = bool(args.invalid_every and index % args.invalid_every == 0)
        if invalid:
            payload["age"] = -1
        request = Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        counts["sent"] += 1
        try:
            with opener.open(request, timeout=10) as response:
                status = response.status
        except HTTPError as error:
            status = error.code
            error.close()
        except (OSError, HTTPException):
            counts["network_errors"] += 1
            counts["unexpected"] += 1
            continue

        group = f"{status // 100}xx"
        counts[group if group in counts else "other_status"] += 1
        counts["unexpected"] += status != (400 if invalid else 200)

    print(
        json.dumps(
            {
                "scenario": args.scenario,
                "seed": args.seed,
                "invalid_every": args.invalid_every,
                **counts,
            }
        )
    )
    return int(counts["unexpected"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
