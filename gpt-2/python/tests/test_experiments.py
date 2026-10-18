"""Check experiment mistakes that could otherwise select the wrong data or run."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from gpt2_lab.config import load_experiment
from gpt2_lab.runs import prepare_run


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "data").mkdir()
        (self.root / "data" / "text.txt").write_text("The moon runs a bakery.\n", encoding="utf-8")
        (self.root / "configs").mkdir()
        self.config_path = self.root / "configs" / "example.json"
        self.values = {
            "name": "example", "description": "test", "schema_version": 1,
            "model": {}, "dataset": {"path": "../data/text.txt"}, "training": {},
        }

    def load(self):
        self.config_path.write_text(json.dumps(self.values), encoding="utf-8")
        return load_experiment(self.config_path)

    def test_relative_data_is_anchored_to_config(self):
        experiment = self.load()
        self.assertEqual(experiment.dataset_path, (self.root / "data" / "text.txt").resolve())
        self.assertTrue(experiment.plan()["dataset_exists"])

    def test_misspelled_training_setting_is_rejected(self):
        self.values["training"]["max_step"] = 50
        with self.assertRaisesRegex(ValueError, "Unknown TrainingConfig fields: max_step"):
            self.load()

    def test_incompatible_heads_and_invalid_numbers_are_rejected(self):
        for model in ({"n_embd": 31, "n_head": 4}, {"n_embd": True}, {"n_layer": 0}):
            with self.subTest(model=model):
                self.values["model"] = model
                with self.assertRaises(ValueError):
                    self.load()
        self.values["model"] = {}
        self.values["training"] = {"learning_rate": float("nan")}
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            self.load()

    def test_prepared_runs_are_distinct_and_snapshot_can_be_reloaded(self):
        experiment = self.load()
        first = prepare_run(experiment, self.root / "runs")
        before = (first / "manifest.json").read_bytes()
        second = prepare_run(experiment, self.root / "runs")
        self.assertNotEqual(first, second)
        self.assertEqual((first / "manifest.json").read_bytes(), before)
        snapshot = load_experiment(first / "config.json")
        self.assertEqual(snapshot.dataset_path, experiment.dataset_path)
        manifest = json.loads(before)
        self.assertEqual(manifest["status"], "prepared")
        self.assertEqual(manifest["config_sha256"], hashlib.sha256((first / "config.json").read_bytes()).hexdigest())
        self.assertEqual(manifest["dataset"]["sha256"], hashlib.sha256(experiment.dataset_path.read_bytes()).hexdigest())
        self.assertEqual({path.name for path in first.iterdir()}, {"manifest.json", "config.json"})

    def test_missing_dataset_does_not_create_a_run(self):
        self.values["dataset"]["path"] = "missing.txt"
        with self.assertRaisesRegex(ValueError, "Dataset file does not exist"):
            prepare_run(self.load(), self.root / "runs")
        self.assertFalse((self.root / "runs").exists())


if __name__ == "__main__":
    unittest.main()
