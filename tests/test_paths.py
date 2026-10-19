"""Fast checks for the shared path layout without running the analysis."""

import os
from pathlib import Path
import unittest
from unittest.mock import patch

import csp_paths


class PathLayoutTests(unittest.TestCase):
    def test_default_workspace_is_checkout(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(csp_paths.workspace_root(), Path(csp_paths.__file__).resolve().parent)

    def test_workspace_can_be_overridden(self):
        with patch.dict(os.environ, {"CSP_WORKSPACE": "/tmp/csp-test-workspace"}):
            self.assertEqual(csp_paths.workspace_root(), Path("/tmp/csp-test-workspace").resolve())

    def test_relative_workspace_is_rejected(self):
        with patch.dict(os.environ, {"CSP_WORKSPACE": "relative-workspace"}):
            with self.assertRaisesRegex(ValueError, "absolute path"):
                csp_paths.workspace_root()

    def test_pipeline_handoffs_share_directories(self):
        self.assertEqual(csp_paths.TRAIN_PDF.parent, csp_paths.TRAINING)
        self.assertEqual(csp_paths.TOPICS.parent, csp_paths.FIGURES)
        self.assertEqual(csp_paths.MAIN.parent, csp_paths.RESULTS)
        self.assertEqual(csp_paths.PSL.parent, csp_paths.RESULTS)

    def test_scripts_do_not_pin_another_machine_paths(self):
        for script in Path(csp_paths.__file__).resolve().parent.glob("CSP_*.py"):
            with self.subTest(script=script.name):
                source = script.read_text()
                self.assertNotIn("/Users/", source)
                self.assertNotIn("/opt/homebrew/", source)


if __name__ == "__main__":
    unittest.main()
