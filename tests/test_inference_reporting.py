import unittest

from src.comparison.reporting import (
    estimate_eta_seconds,
    format_clock_duration,
    should_report_progress,
)


class InferenceReportingTests(unittest.TestCase):
    def test_clock_duration_is_fixed_width_without_24_hour_wrap(self):
        self.assertEqual(format_clock_duration(-1.0), "00:00:00")
        self.assertEqual(format_clock_duration(125.4), "00:02:05")
        self.assertEqual(format_clock_duration(90061.0), "25:01:01")

    def test_progress_reports_first_hundreds_and_final(self):
        reported = [
            index
            for index in range(1, 2400)
            if should_report_progress(index, 2399)
        ]
        self.assertEqual(reported[:3], [1, 100, 200])
        self.assertEqual(reported[-2:], [2300, 2399])

    def test_eta_uses_completed_object_average(self):
        self.assertEqual(estimate_eta_seconds(30.0, 2, 5), 45.0)
        self.assertEqual(estimate_eta_seconds(30.0, 5, 5), 0.0)


if __name__ == "__main__":
    unittest.main()
