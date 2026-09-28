"""Offline smoke check: notebook execution on artificial data, never Avito test data."""

from pathlib import Path
import ast
import gzip
import json
import os
import shutil
import sys
import tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
PROJECT = HERE / "project" if (HERE / "project").exists() else HERE.parent
QA = Path(tempfile.mkdtemp(prefix="avito-notebook-smoke-"))
kernel_root = QA / "jupyter/share/jupyter"
kernel_dir = kernel_root / "kernels/avito-smoke"
kernel_dir.mkdir(parents=True)
(kernel_dir / "kernel.json").write_text(
    json.dumps(
        {
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Avito smoke",
            "language": "python",
        }
    ),
    encoding="utf-8",
)
os.environ["JUPYTER_PATH"] = str(kernel_root) + os.pathsep + os.environ.get("JUPYTER_PATH", "")
os.environ["IPYTHONDIR"] = str(QA / "ipython")
os.environ["JUPYTER_RUNTIME_DIR"] = str(QA / "jupyter_runtime")
os.environ["MPLBACKEND"] = "Agg"
os.environ["PYTHONUTF8"] = "1"
KERNEL = "avito-smoke"
shutil.copytree(PROJECT / "notebooks", QA / "notebooks")
shutil.copy2(PROJECT / "avito_project.json", QA / "avito_project.json")
(QA / "data/gt").mkdir(parents=True)


def check_source():
    for path in (PROJECT / "notebooks").glob("*.ipynb"):
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
                assert cell.execution_count is None and not cell.outputs


def make_fixtures():
    rng = np.random.default_rng(123)
    for split, count in [("train", 30), ("validation", 24)]:
        images_dir = QA / "data" / split
        images_dir.mkdir()
        annotations = []
        for i in range(count):
            array = rng.integers(180, 256, (70, 230, 3), dtype=np.uint8)
            im = Image.fromarray(array)
            d = ImageDraw.Draw(im)
            d.text(
                (20, 20), f"TEST {split} {i} UP", fill=(0, 0, 0), font=ImageFont.load_default(22)
            )
            vertices = [[15, 15], [214, 15], [214, 49], [15, 49]]
            if i % 2:
                im = im.transpose(Image.Transpose.ROTATE_180)
                vertices = [[229 - x, 69 - y] for x, y in vertices]
            image_id = f"{split}_{i:03d}"
            # PNG bytes with .jpg extension avoid lossy reference differences.
            im.save(images_dir / f"{image_id}.jpg", format="PNG")
            word = {"text": f"TEST {i}", "legible": True, "vertical": False, "vertices": vertices}
            annotations.append(
                {
                    "image_id": image_id,
                    "image_width": 230,
                    "image_height": 70,
                    "paragraphs": [{"lines": [{"words": [word]}]}],
                }
            )
        with gzip.open(QA / f"data/gt/{split}.jsonl.gz", "wt", encoding="utf-8") as f:
            json.dump({"annotations": annotations}, f)


def run_notebook(name, additions=None, extra_cells=()):
    nb = nbformat.read(QA / "notebooks" / name, as_version=4)
    if name == "05_calibration_submission.ipynb":
        nb.cells[
            2
        ].source += "\nRUN_DIR=ROOT/'runs/mobilenet_v1'\nCHECKPOINT=RUN_DIR/'best.pt'\nMANIFEST=ROOT/'data/processed/hiertext_v1/manifest.csv'\nSUBMISSION_PATH=ROOT/'submissions/mobilenet_v1.csv'\nCALIBRATE=not (RUN_DIR/'deployment.json').exists()"

    for index, text in (additions or {}).items():
        nb.cells[index].source += "\n" + text
    for text in extra_cells:
        nb.cells.append(nbformat.v4.new_code_cell(text))
    print("EXECUTE", name, flush=True)
    client = NotebookClient(
        nb, timeout=240, kernel_name=KERNEL, resources={"metadata": {"path": str(QA)}}
    )
    try:
        client.execute()
    finally:
        nbformat.write(nb, QA / ("executed_" + name))
    print("PASS", name, flush=True)


check_source()
make_fixtures()
run_notebook("00_setup.ipynb")
run_notebook(
    "01_prepare_hiertext.ipynb",
    extra_cells=[
        r"""
# The same word in two native orientations must rectify identically.
im = np.arange(40*120*3, dtype=np.uint8).reshape(40,120,3)
pts = np.array([[4,4],[115,4],[115,35],[4,35]], np.float32)
upright = rectify_word(im, pts, margin=0)
rotated = rectify_word(im[::-1,::-1].copy(), np.array([119,39])-pts, margin=0)
assert np.array_equal(upright, rotated), 'Wrong polygon orientation'
assert manifest.groupby('group_id').split.nunique().max() == 1
assert set(manifest.split) == {'train', 'dev', 'calibration', 'holdout'}
# Same photograph added to train under another ID must be excluded from training.
raw_val = list(records(RAW_DIR / 'gt/validation.jsonl.gz'))[0]
raw_train = list(records(RAW_DIR / 'gt/train.jsonl.gz'))
copy = json.loads(json.dumps(raw_val))
copy['image_id'] = 'duplicate'
import shutil
shutil.copy2(RAW_DIR / 'validation' / (raw_val['image_id']+'.jpg'), RAW_DIR / 'train/duplicate.jpg')
raw_train.append(copy)
with gzip.open(RAW_DIR / 'gt/train.jsonl.gz', 'wt', encoding='utf-8') as f:
    json.dump({'annotations': raw_train}, f)
dups, report = prepare_hiertext(RAW_DIR, ROOT / 'data/processed/duplicate_check')
assert not (dups.source_id == 'duplicate').any()
assert report['filtered']['duplicate_source_removed'] > 0
"""
    ],
)
run_notebook(
    "04_train_mobilenet.ipynb",
    additions={
        2: """
MANIFEST = ROOT / 'data/processed/hiertext_v1/manifest.csv'
RUN_DIR = ROOT / 'runs/mobilenet_v1'
CFG.update(epochs=2, batch_size=8, pretrained=False, amp=False, allow_cpu=True, cpu_threads=2)
"""
    },
    extra_cells=[
        r"""
# Check serialization and restoring RNG for a real checkpoint.
state = torch.load(RUN_DIR/'last.pt', map_location='cpu', weights_only=True)
restore_rng(state['rng'])
a = (random.random(), np.random.rand(), torch.rand(2))
restore_rng(state['rng'])
b = (random.random(), np.random.rand(), torch.rand(2))
assert a[:2] == b[:2] and torch.equal(a[2], b[2])
# Resume the completed run must preserve history and weights.
before = sha256_file(RUN_DIR/'best.pt')
resumed = train_experiment(train_frame, dev_frame, CFG, RUN_DIR, data_identity, resume=True)
assert len(resumed) == 2 and sha256_file(RUN_DIR/'best.pt') == before
"""
    ],
)

# A small artificial submission, with numeric-looking IDs and reversed template order.
test_dir = QA / "data/test"
test_dir.mkdir(exist_ok=True)
for i in range(5):
    im = Image.new("RGB", (80 + i * 10, 24), (255, 255, 255))
    ImageDraw.Draw(im).text((3, 3), f"ID {i}", fill=(0, 0, 0))
    im.save(test_dir / f"{i:04d}.png")
import pandas as pd

pd.DataFrame({"image_id": [f"{i:04d}.png" for i in reversed(range(5))], "p_180": 0.5}).to_csv(
    QA / "data/sample_submission.csv", index=False
)

run_notebook(
    "05_calibration_submission.ipynb",
    additions={2: "BATCH_SIZE=4\nEXPECTED_COUNT=5\nCREATE_SUBMISSION=True\nBENCHMARK=True"},
    extra_cells=[
        r"""
assert 0 < deployment['temperature'] < 11
saved = pd.read_csv(SUBMISSION_PATH, dtype={'image_id':str})
assert saved.image_id.tolist() == ids
assert len(saved) == 5 and saved.p_180.between(0,1).all()
# Dataset and direct-image path must produce equal outputs for y=0.
frame = load_manifest(MANIFEST).query("split == 'dev'").head(2)
pred = predict_loader(model, make_loader(frame, cfg), device, rotated=True)
direct = predict_images(model, [ROOT/p for p in frame.path], cfg, deployment, device, batch_size=4)
np.testing.assert_allclose(direct, probabilities(pred.iloc[::2], deployment['temperature'], deployment['tta']), atol=1e-6)
# Invalid files/NaN/duplicate IDs should never become a seemingly valid submission.
for bad_p in [[np.nan]*5, [1.1]*5]:
    try:
        write_submission(ids, bad_p, ROOT/'submissions/bad.csv', 5)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid probabilities accepted')
Image.new('RGB',(10,10)).save(TEST_DIR/'extra.png')
try:
    test_files(TEST_DIR, SAMPLE_SUBMISSION, 5)
except ValueError:
    pass
else:
    raise AssertionError('Extra image accepted')
(TEST_DIR/'extra.png').unlink()
"""
    ],
)
# A fresh kernel must support submission without data preparation, calibration or internet.
run_notebook(
    "05_calibration_submission.ipynb",
    additions={
        2: """
BATCH_SIZE=4
EXPECTED_COUNT=5
CALIBRATE=False
CREATE_SUBMISSION=True
SUBMISSION_PATH=ROOT/'submissions/second.csv'
"""
    },
    extra_cells=[
        r"""
first = pd.read_csv(ROOT/'submissions/mobilenet_v1.csv')
second = pd.read_csv(ROOT/'submissions/second.csv')
np.testing.assert_allclose(first.p_180, second.p_180, atol=1e-10)
"""
    ],
)
print("ALL CHECKS PASSED. Evidence:", QA, flush=True)
if (HERE / "project").exists():
    (HERE / "last_qa_path.txt").write_text(str(QA), encoding="utf-8")
