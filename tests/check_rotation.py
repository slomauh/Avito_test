"""Training-only small-angle rotation; offline regression checks."""

import json
import os
from pathlib import Path
import random
import tempfile
from unittest.mock import patch
import numpy as np
import pandas as pd
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "notebooks").is_dir():
    ROOT = Path("D:/Avito_test")
os.environ["AVITO_PROJECT_ROOT"] = str(ROOT)
ns = {}
for cell in json.loads((ROOT / "notebooks/_shared.ipynb").read_text(encoding="utf-8"))["cells"]:
    if cell["cell_type"] == "code":
        exec("".join(cell["source"]), ns)
im = Image.new("RGB", (120, 30), "white")
im.paste((0, 0, 0), (5, 5, 115, 25))
random.seed(42)
state = random.getstate()
assert ns["small_rotation"](im, 0, 0.5) is im
assert random.getstate() == state, "Old CFG must not consume extra RNG"
assert ns["small_rotation"](im, 7, 0) is im
with patch.object(random, "random", return_value=0), patch.object(
    random, "uniform", return_value=7
) as angle:
    rotated = ns["small_rotation"](im, 7, 0.5)
    angle.assert_called_once_with(-7, 7)
assert rotated.width >= im.width and rotated.height > im.height
assert rotated.getpixel((0, 0)) == (255, 255, 255)
with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "test.png"
    im.save(path)
    frame = pd.DataFrame([dict(path=str(path), crop_id="fixture")])
    cfg = dict(height=48, width=192, batch_size=1, rotation_degrees=7.0, rotation_probability=0.5)
    dataset = ns["make_loader"](frame, cfg, False).dataset
    with patch.dict(
        ns, {"augment": lambda *args: (_ for _ in ()).throw(AssertionError("Augmentation in dev"))}
    ):
        for i in [0, 1]:
            x, y, _ = dataset[i]
            original = im if i == 0 else im.transpose(Image.Transpose.ROTATE_180)
            assert y == i and torch.equal(x, ns["letterbox"](original))
    calls = []

    def spy(image, degrees, probability):
        calls.append((degrees, probability))
        return image

    with patch.dict(ns, {"augment": spy}):
        ns["make_loader"](frame, cfg, True).dataset[0]
        ns["make_loader"](
            frame, {k: v for k, v in cfg.items() if not k.startswith("rotation_")}, True
        ).dataset[0]
    assert calls == [(7.0, 0.5), (0.0, 0.5)]
print(
    "PASS rotation range, canvas, old RNG compatibility, training-only config, unchanged validation labels"
)
