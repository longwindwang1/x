"""Aggregate per-turn latency records into a p50/p95 report.

Usage:
    python evals/latency_report.py [logs/turns.jsonl]

Reads the JSONL written by the runtime (metrics.MetricsLog) and prints
percentiles per pipeline stage, split by turn source. This is the script
behind the latency numbers quoted in the README.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

FIELDS = [
    "asr_ms",
    "llm_ttft_ms",
    "llm_total_ms",
    "tts_first_chunk_ms",
    "first_audio_ms",
    "total_ms",
]


def percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    k = (len(values) - 1) * pct / 100.0
    lower = int(k)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (k - lower)


def main(path: str = "logs/turns.jsonl") -> int:
    log = Path(path)
    if not log.is_file():
        print(f"No log at {log} — run the server and hold a few conversations first.")
        return 1

    by_source: dict[str, list[dict]] = defaultdict(list)
    with log.open(encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            if not record.get("cancelled"):
                by_source[record.get("source", "?")].append(record)

    for source, records in sorted(by_source.items()):
        print(f"\n== source={source}  (n={len(records)}) ==")
        print(f"{'stage':<22}{'p50':>10}{'p95':>10}{'mean':>10}{'n':>6}")
        for field in FIELDS:
            values = [r[field] for r in records if field in r]
            if not values:
                continue
            print(
                f"{field:<22}{percentile(values, 50):>10.0f}{percentile(values, 95):>10.0f}"
                f"{statistics.fmean(values):>10.0f}{len(values):>6}"
            )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
