"""Tests the Flask API end-to-end and writes outputs/test_results.md.
Run:  python tests/test_api.py [--model models/best_model.pt] [--samples samples]
Samples: sample_*.tif (12-band). If sample_N_gt.png exists, IoU/F1 vs ground truth is reported.
"""
import argparse, glob, io, os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import create_app


def scores(pred, gt):
    tp, fp, fn = (pred & gt).sum(), (pred & ~gt).sum(), (~pred & gt).sum()
    return tp / (tp + fp + fn + 1e-7), 2 * tp / (2 * tp + fp + fn + 1e-7)


def post(client, fh, name, query=""):
    return client.post("/predict" + query, data={"file": (fh, name)}, content_type="multipart/form-data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/best_model.pt"); ap.add_argument("--samples", default="samples")
    a = ap.parse_args()
    client = create_app(a.model).test_client()
    files = sorted(glob.glob(os.path.join(a.samples, "sample_*.tif")))
    assert len(files) >= 3, f"need >= 3 samples in {a.samples}/, found {len(files)}"
    rows, ok = [], True
    assert client.get("/health").json["status"] == "ok" and client.get("/").status_code == 200

    for f in files:
        with open(f, "rb") as fh:
            r = post(client, fh, os.path.basename(f))
        assert r.status_code == 200 and r.mimetype == "image/png", (f, r.status_code)
        mask = np.array(Image.open(io.BytesIO(r.data)))
        vals = sorted(np.unique(mask).tolist())
        ratio = float(r.headers["X-Water-Ratio"])
        with open(f, "rb") as fh:
            j = post(client, fh, "x.tif", "?format=json").json
        passed = mask.shape == (128, 128) and set(vals) <= {0, 255} and abs(j["water_ratio"] - ratio) < 1e-3
        gt_path = f.replace(".tif", "_gt.png"); iou = f1 = "n/a"
        if os.path.exists(gt_path):
            iou, f1 = (f"{v:.3f}" for v in scores(mask > 127, np.array(Image.open(gt_path)) > 127))
        ok &= passed
        rows.append((os.path.basename(f), mask.shape, vals, f"{ratio:.3f}", iou, f1, "PASS" if passed else "FAIL"))

    assert client.post("/predict", data={}, content_type="multipart/form-data").status_code == 400
    bad = io.BytesIO(); np.save(bad, np.zeros((3, 128, 128), np.float32)); bad.seek(0)
    r = post(client, bad, "bad.npy"); assert r.status_code == 422 and "12-channel" in r.json["error"]
    odd = io.BytesIO(); np.save(odd, (np.random.rand(12, 100, 77) * 3000).astype(np.float32)); odd.seek(0)
    r = post(client, odd, "odd.npy")
    assert r.status_code == 200 and np.array(Image.open(io.BytesIO(r.data))).shape == (100, 77)

    os.makedirs("outputs", exist_ok=True)
    lines = ["| Sample | Mask shape | Unique values | Water ratio | IoU | F1 | Result |", "|---|---|---|---|---|---|---|"]
    lines += ["| " + " | ".join(map(str, x)) + " |" for x in rows]
    lines += ["", "Error cases: no file -> 400, 3-channel input -> 422, 100x77 input -> 200 (100x77 mask)."]
    open("outputs/test_results.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines)); print("\nALL TESTS PASSED" if ok else "\nSOME TESTS FAILED"); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
