import tempfile
import unittest
from pathlib import Path

from visual_torque_toolkit.dataset import build_dataset, validate_dataset
from visual_torque_toolkit.generate import GenerationConfig


class DatasetTests(unittest.TestCase):
    def test_rendered_dataset_roundtrip(self) -> None:
        config = GenerationConfig(
            max_units=2,
            max_distance=2,
            multi_max_units=1,
            multi_scene_count=4,
            seed=11,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            summary = build_dataset(root, config)
            checked = validate_dataset(root)
            self.assertEqual(summary.scene_count, 20)
            self.assertEqual(checked["scene_count"], 20)
            self.assertEqual(checked["image_count"], 20)
            self.assertTrue((root / "statistics.json").is_file())

    def test_builder_refuses_nonempty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "existing.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                build_dataset(root, GenerationConfig(multi_scene_count=0), render_images=False)


if __name__ == "__main__":
    unittest.main()
