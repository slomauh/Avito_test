from pathlib import Path
import ast, json, tempfile
import pandas as pd

root = Path(__file__).resolve().parents[1]
nb = json.loads((root / "notebooks/_shared.ipynb").read_text(encoding="utf-8"))
s = next(c["source"] for c in nb["cells"] if "def test_files(" in "".join(c["source"]))
if isinstance(s, list):
    s = "".join(s)
tree = ast.parse(s)
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "test_files")
scope = dict(
    Path=Path, pd=pd, EXTENSIONS={".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
)
exec(compile(ast.Module(body=[fn], type_ignores=[]), "<test_files>", "exec"), scope)
resolve = scope["test_files"]
with tempfile.TemporaryDirectory() as tmp:
    base = Path(tmp)
    images = base / "images"
    sample = base / "sample.csv"

    def template(ids):
        pd.DataFrame({"image_id": ids, "p_180": 0.5}).to_csv(sample, index=False)

    def fails(kind, contains):
        try:
            resolve(images, sample, 2)
        except kind as e:
            assert contains in str(e), str(e)
        else:
            raise AssertionError("Expected rejection: " + contains)

    template(["0002", "0001"])
    fails(FileNotFoundError, "не существует")
    images.mkdir()
    fails(ValueError, "не найдено изображений")
    sub = images / "nested"
    sub.mkdir()
    (images / "0001.jpg").touch()
    (sub / "0002.PNG").touch()
    ids, paths = resolve(images, sample, 2)
    assert ids == ["0002", "0001"] and [p.name for p in paths] == ["0002.PNG", "0001.jpg"]
    template(["0001.jpg", "0002.PNG"])
    assert resolve(images, sample, 2)[0] == ["0001.jpg", "0002.PNG"]
    template(["0001", "0001.jpg"])
    fails(ValueError, "один файл")
    template(["0002", "0001"])
    (images / "0001.png").touch()
    fails(ValueError, "Неоднозначный")
    (images / "0001.png").unlink()
    (sub / "0001.jpg").touch()
    fails(ValueError, "Одинаковые имена")
    (sub / "0001.jpg").unlink()
    (images / "extra.jpg").touch()
    fails(ValueError, "extra=")
    (images / "extra.jpg").unlink()
    template(["0002", "missing"])
    fails(ValueError, "missing=")
print("PASS: paths, empty folder, exact/stem IDs, order, ambiguity, duplicate and extra files")
