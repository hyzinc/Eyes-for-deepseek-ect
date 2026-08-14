import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "img2txt.py"


def make_test_image(path):
    img = Image.new("RGB", (120, 90), (135, 206, 235))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 30, 60, 70], fill=(200, 40, 40))
    draw.rectangle([70, 40, 110, 80], fill=(34, 139, 34))
    img.save(path)


class Img2TxtTests(unittest.TestCase):
    def test_transcript_contains_all_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "test.png"
            make_test_image(image)
            proc = subprocess.run(
                [
                    sys.executable, str(SCRIPT), str(image),
                    "--ascii", "40", "--palette", "6",
                    "--grid", "4x3", "--edge", "32",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("## META", proc.stdout)
            self.assertIn("## PALETTE", proc.stdout)
            self.assertIn("## COLOR GRID", proc.stdout)
            self.assertIn("## ASCII", proc.stdout)
            self.assertIn("## EDGE MAP", proc.stdout)

    def test_crop_updates_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "test.png"
            make_test_image(image)
            proc = subprocess.run(
                [
                    sys.executable, str(SCRIPT), str(image),
                    "--crop", "0,0,60,45", "--ascii", "24",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("- size: 60x45", proc.stdout)


if __name__ == "__main__":
    unittest.main()
