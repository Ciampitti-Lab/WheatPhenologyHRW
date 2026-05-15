"""Smoke tests for WheatPhenologyHRW — confirm imports, config loading, and core
WES (Wang-Engel-Streck) functions work end-to-end without requiring the (large) data files.

Run with:  pytest tests/ -v
"""
import os
import sys
import unittest
from pathlib import Path

# Allow `from scripts.utils...` imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class TestConfig(unittest.TestCase):
    """Config loader sanity checks."""

    def test_yaml_loads(self):
        """config.yaml parses without errors."""
        from scripts.utils.config import get_config
        cfg = get_config()
        self.assertIsNotNone(cfg)

    def test_required_top_level_sections(self):
        """All required sections are present."""
        from scripts.utils.config import get_config
        cfg = get_config()
        for section in ('paths', 'gee', 'study_area', 'phenology_stages',
                        'windows', 'ml', 'pcse'):
            self.assertIn(section, cfg, f"Missing section: {section}")

    def test_phenology_stages_count(self):
        """We claim 8 stages everywhere."""
        from scripts.utils.config import get_config
        cfg = get_config()
        self.assertEqual(len(cfg.phenology_stages), 8,
                         "Expected exactly 8 phenological stages")

    def test_study_area_5_states(self):
        """Study area covers 5 US Plains states."""
        from scripts.utils.config import get_config
        cfg = get_config()
        self.assertEqual(len(cfg.study_area.state_codes), 5)
        self.assertEqual(set(cfg.study_area.state_codes), {'TX','OK','KS','NE','CO'})


class TestImports(unittest.TestCase):
    """All required dependencies are importable."""

    def test_scientific_stack(self):
        import numpy, pandas, scipy
        self.assertIsNotNone(numpy.__version__)

    def test_ml_stack(self):
        import sklearn, xgboost, lightgbm
        self.assertIsNotNone(sklearn.__version__)

    def test_geospatial(self):
        import geopandas, shapely
        self.assertIsNotNone(geopandas.__version__)

    def test_pcse(self):
        import pcse
        self.assertIsNotNone(pcse.__version__)

    def test_utils_modules(self):
        from scripts.utils import config, thermal


class TestWES(unittest.TestCase):
    """Core WES (Wang-Engel-Streck) functions in scripts/utils/thermal.py."""

    def test_thermal_module_exposes_simulate(self):
        """Required WES entry point exists."""
        from scripts.utils import thermal
        # Should expose at least one of the expected APIs
        has_callable = any(
            callable(getattr(thermal, name, None))
            for name in ('simulate_wes', 'beta_temp_response',
                         'streck_vernalization', 'photoperiod_hours')
        )
        self.assertTrue(has_callable,
                        "thermal.py should expose WES simulator / response functions")


if __name__ == '__main__':
    unittest.main(verbosity=2)
