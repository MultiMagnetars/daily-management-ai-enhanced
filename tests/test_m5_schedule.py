import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = ROOT_DIR / ".github" / "workflows" / "run.yml"


class M5ScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    def test_workflow_has_one_beijing_midnight_schedule_and_manual_dispatch(self):
        self.assertEqual(self.workflow.count("\n  schedule:\n"), 1)
        self.assertEqual(self.workflow.count("cron: '0 16 * * *'"), 1)
        self.assertIn("workflow_dispatch:", self.workflow)
        self.assertIn("validation_mode:", self.workflow)
        self.assertNotIn('cron: "30 17 * * *"', self.workflow)

    def test_production_date_calculation_uses_shanghai_and_three_day_lookback(self):
        self.assertIn(
            'today=$(TZ=Asia/Shanghai date "+%Y-%m-%d")',
            self.workflow,
        )
        self.assertIn(
            'start_date=$(TZ=Asia/Shanghai date -d "${today} - 3 days" "+%Y-%m-%d")',
            self.workflow,
        )
        self.assertEqual(self.workflow.count("TZ=Asia/Shanghai date"), 2)
        self.assertNotIn('today=$(date -u', self.workflow)
        self.assertNotIn('start_date=$(date -u', self.workflow)
        self.assertIn('${today} - 3 days', self.workflow)
        self.assertIn('echo "crawl_date=$today" >> "$GITHUB_OUTPUT"', self.workflow)

    def test_beijing_midnight_example_maps_to_expected_crawl_date(self):
        utc_time = datetime(2026, 8, 24, 16, tzinfo=timezone.utc)
        beijing_time = utc_time.astimezone(timezone(timedelta(hours=8)))
        self.assertEqual(beijing_time.strftime("%Y-%m-%d"), "2026-08-25")

    def test_scheduled_and_manual_normal_events_enter_build(self):
        build_block = self.workflow.split("  build:", 1)[1]
        self.assertIn("github.event_name != 'workflow_dispatch'", build_block)
        self.assertIn("inputs.validation_mode == 'normal'", build_block)

    def test_scix_jobs_remain_manual_validation_only(self):
        smoke_block = self.workflow.split("  scix-smoke:", 1)[1].split(
            "\n  scix-e2e:", 1
        )[0]
        e2e_block = self.workflow.split("  scix-e2e:", 1)[1].split(
            "\n  build:", 1
        )[0]
        manual_smoke = (
            "github.event_name == 'workflow_dispatch'"
            " && github.ref_name == 'astro-custom'"
            " && inputs.validation_mode == 'scix-smoke'"
        )
        manual_e2e = (
            "github.event_name == 'workflow_dispatch'"
            " && github.ref_name == 'astro-custom'"
            " && inputs.validation_mode == 'scix-e2e'"
        )
        self.assertIn(manual_smoke, smoke_block)
        self.assertIn(manual_e2e, e2e_block)
        self.assertNotIn("github.event_name != 'workflow_dispatch'", smoke_block)
        self.assertNotIn("github.event_name != 'workflow_dispatch'", e2e_block)

    def test_validation_modes_and_concurrency_remain_unchanged(self):
        self.assertIn("- normal", self.workflow)
        self.assertIn("- scix-smoke", self.workflow)
        self.assertIn("- scix-e2e", self.workflow)
        self.assertIn("cancel-in-progress: false", self.workflow)
        self.assertIn(
            "group: ${{ github.workflow }}-${{ github.ref }}-production",
            self.workflow,
        )

    def test_downstream_steps_use_crawl_date_output(self):
        self.assertGreaterEqual(
            self.workflow.count("steps.crawl_step.outputs.crawl_date"), 3
        )
        self.assertIn("crawl_date=$today", self.workflow)

    def test_workflow_is_yaml_parseable(self):
        yaml.safe_load(self.workflow)


if __name__ == "__main__":
    unittest.main()
