import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.generate_normal_data import generate_normal_data
from src.detector import PhysicalAnomalyDetector
from src.features import FEATURE_COLUMNS, TelemetryValidationError, build_feature_frame


class PhysicalAnomalyDetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.normal_data = generate_normal_data(500, seed=42)
        cls.detector = PhysicalAnomalyDetector(random_state=42).fit(cls.normal_data)

    def setUp(self) -> None:
        self.detector.reset_stream()

    def test_normal_telemetry_is_classified_as_normal(self) -> None:
        sample = {
            "rpm": 750,
            "target_rpm": 750,
            "temperature": 30,
            "power": 6.5,
            "pressure": 3,
        }

        result = self.detector.predict(sample)

        self.assertEqual(result["anomaly_status"], "NORMAL")
        self.assertGreaterEqual(result["anomaly_score"], 0.0)
        self.assertLessEqual(result["anomaly_score"], 1.0)

    def test_unusual_telemetry_is_classified_by_isolation_forest(self) -> None:
        sample = {
            "rpm": 2200,
            "target_rpm": 750,
            "temperature": 95,
            "power": 25,
            "pressure": 9,
        }

        result = self.detector.predict(sample)

        self.assertEqual(result["anomaly_status"], "ANOMALOUS")

    def test_training_and_prediction_use_the_same_feature_order(self) -> None:
        telemetry = generate_normal_data(12, seed=4)
        _, features = build_feature_frame(telemetry)

        self.assertEqual(tuple(self.detector.feature_columns), FEATURE_COLUMNS)
        self.assertEqual(tuple(features.columns), FEATURE_COLUMNS)
        self.assertEqual(
            FEATURE_COLUMNS,
            (
                "rpm",
                "temperature",
                "pressure",
                "power",
                "rpm_deviation",
                "rpm_rate_of_change",
                "temperature_rate_of_change",
                "pressure_rate_of_change",
            ),
        )

    def test_first_sample_rates_are_zero_and_following_rates_are_differences(self) -> None:
        telemetry = pd.DataFrame(
            {
                "rpm": [100, 105],
                "temperature": [20, 22],
                "pressure": [2, 2.5],
                "power": [5, 5.2],
                "target_rpm": [100, 100],
            }
        )

        _, features = build_feature_frame(telemetry)

        self.assertEqual(features.loc[0, "rpm_rate_of_change"], 0)
        self.assertEqual(features.loc[0, "temperature_rate_of_change"], 0)
        self.assertEqual(features.loc[0, "pressure_rate_of_change"], 0)
        self.assertEqual(features.loc[1, "rpm_rate_of_change"], 5)
        self.assertEqual(features.loc[1, "temperature_rate_of_change"], 2)
        self.assertAlmostEqual(features.loc[1, "pressure_rate_of_change"], 0.5)

    def test_normal_generator_covers_the_supplied_operating_ranges(self) -> None:
        telemetry = generate_normal_data(1000, seed=42)

        self.assertTrue(telemetry["rpm"].between(0, 1500).all())
        self.assertTrue(telemetry["temperature"].between(0, 60).all())
        self.assertTrue(telemetry["power"].between(1, 12).all())
        self.assertTrue(telemetry["pressure"].between(1, 5).all())
        self.assertLess(telemetry["rpm"].min(), 100)
        self.assertGreater(telemetry["rpm"].max(), 1400)
        self.assertLess(telemetry["temperature"].min(), 2)
        self.assertGreater(telemetry["temperature"].max(), 58)
        self.assertLess(telemetry["power"].min(), 1.5)
        self.assertGreater(telemetry["power"].max(), 11.5)
        self.assertLess(telemetry["pressure"].min(), 1.1)
        self.assertGreater(telemetry["pressure"].max(), 4.9)

    def test_invalid_telemetry_values_are_rejected(self) -> None:
        for column, value in (
            ("pressure", np.nan),
            ("power", np.inf),
            ("temperature", "not-a-number"),
        ):
            with self.subTest(column=column):
                sample = self.normal_data.iloc[[125]].copy()
                sample[column] = sample[column].astype(object)
                sample.loc[sample.index[0], column] = value
                with self.assertRaises(TelemetryValidationError):
                    self.detector.predict(sample)

    def test_missing_columns_and_insufficient_training_data_fail_clearly(self) -> None:
        with self.assertRaisesRegex(TelemetryValidationError, "missing required process columns"):
            build_feature_frame(pd.DataFrame({"rpm": [100]}))

        with self.assertRaisesRegex(TelemetryValidationError, "at least 10"):
            PhysicalAnomalyDetector().fit(generate_normal_data(9, seed=1))

    def test_saved_model_loads_without_retraining(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "isolation_forest.joblib"
            self.detector.save(model_path)
            loaded_detector = PhysicalAnomalyDetector.load(model_path)

            self.assertEqual(
                loaded_detector.predict(
                    {
                        "rpm": 750,
                        "target_rpm": 750,
                        "temperature": 30,
                        "power": 6.5,
                        "pressure": 3,
                    }
                )["anomaly_status"],
                "NORMAL",
            )


if __name__ == "__main__":
    unittest.main()
