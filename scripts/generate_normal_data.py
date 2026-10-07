"""Generate synthetic normal-operation telemetry for laboratory experiments only."""

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


def generate_normal_data(samples: int, seed: int) -> pd.DataFrame:
    """Create stable synthetic telemetry within the supplied operating ranges.

    These bounds are for baseline generation only. The detector learns the
    generated distribution and does not apply the bounds as prediction rules.
    """

    rng = np.random.default_rng(seed)
    sample_index = np.arange(samples, dtype=float)
    target_rpm = 750 + 750 * np.sin(4 * np.pi * sample_index / samples)
    rpm = np.clip(target_rpm + rng.normal(0, 2.0, samples), 0, 1500)
    temperature = 30 + 30 * np.sin(14 * np.pi * sample_index / samples)
    temperature += rng.normal(0, 0.15, samples)
    temperature = np.clip(temperature, 0, 60)
    power = 6.5 + 5.5 * np.sin(26 * np.pi * sample_index / samples + 0.8)
    power += rng.normal(0, 0.04, samples)
    power = np.clip(power, 1, 12)
    pressure = 3 + 2 * np.sin(38 * np.pi * sample_index / samples + 1.7)
    pressure += rng.normal(0, 0.015, samples)
    pressure = np.clip(pressure, 1, 5)
    return pd.DataFrame(
        {
            "timestamp": range(1, samples + 1),
            "rpm": rpm,
            "target_rpm": target_rpm,
            "temperature": temperature,
            "power": power,
            "pressure": pressure,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic normal telemetry for controlled academic experiments."
    )
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "normal_data_large.csv")
    args = parser.parse_args()
    if args.samples < 10:
        parser.error("--samples must be at least 10")

    output = write_csv(generate_normal_data(args.samples, args.seed), args.output)
    print(f"Wrote {args.samples} synthetic normal observations to {output}")


if __name__ == "__main__":
    main()
