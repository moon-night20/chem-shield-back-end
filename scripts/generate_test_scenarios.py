"""Generate labeled synthetic scenarios for controlled ML evaluation only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import DATA_DIR, write_csv


def generate_test_scenarios(seed: int) -> pd.DataFrame:
    """Create normal and anomalous laboratory scenarios with evaluation labels."""

    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    timestamp = 1

    def append(
        rpm: float,
        target_rpm: float,
        temperature: float,
        power: float,
        pressure: float,
        label: int,
        scenario: str,
    ) -> None:
        nonlocal timestamp
        records.append(
            {
                "timestamp": timestamp,
                "rpm": rpm,
                "target_rpm": target_rpm,
                "temperature": temperature,
                "power": power,
                "pressure": pressure,
                "actual_label": label,
                "scenario": scenario,
            }
        )
        timestamp += 1

    for _ in range(100):
        append(
            750 + rng.normal(0, 2),
            750,
            30 + rng.normal(0, 0.4),
            6.5 + rng.normal(0, 0.1),
            3 + rng.normal(0, 0.04),
            0,
            "normal",
        )
    for _ in range(30):
        append(
            2000 + rng.normal(0, 10),
            750,
            30 + rng.normal(0, 0.4),
            6.5 + rng.normal(0, 0.1),
            3 + rng.normal(0, 0.04),
            1,
            "large_rpm_deviation",
        )
    for _ in range(30):
        append(
            750 + rng.normal(0, 2),
            750,
            58 + rng.normal(0, 0.5),
            11 + rng.normal(0, 0.2),
            4.8 + rng.normal(0, 0.05),
            1,
            "high_temperature_power_pressure",
        )
    for offset in range(40):
        append(
            750 + (offset * 12),
            750,
            30 + (offset * 0.7),
            6.5 + (offset * 0.1),
            3 + (offset * 0.04),
            1,
            "sudden_process_change",
        )

    return pd.DataFrame.from_records(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic evaluation scenarios for controlled academic experiments."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "test_scenarios.csv")
    args = parser.parse_args()

    scenarios = generate_test_scenarios(args.seed)
    output = write_csv(scenarios, args.output)
    print(f"Wrote {len(scenarios)} synthetic scenario observations to {output}")


if __name__ == "__main__":
    main()
