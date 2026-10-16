"""Inference: 12-channel image (GeoTIFF or .npy) -> binary water mask.
CLI:  python inference.py samples/sample_1.tif --out mask.png
"""
import argparse, io
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from model import build_model

N_BANDS = 12
PAD_MULT = 32  # DeepLabV3+/UNet need H, W divisible by 32 -> pad, predict, crop


class Predictor:
    def __init__(self, ckpt_path, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.model = build_model(ck["arch"], ck["encoder"], len(ck["channels"]))
        self.model.load_state_dict(ck["state_dict"])
        self.model.to(self.device).eval()
        self.thr = float(ck["thr"])
        self.lo = np.asarray(ck["norm_lo"], np.float32).reshape(-1, 1, 1)
        self.hi = np.asarray(ck["norm_hi"], np.float32).reshape(-1, 1, 1)
        self.info = {"name": ck.get("name"), "arch": ck["arch"], "encoder": ck["encoder"],
                     "threshold": self.thr, "test_metrics": ck.get("test_metrics")}

    @staticmethod
    def read_image(data: bytes, filename: str = "") -> np.ndarray:
        """Bytes of a .tif/.tiff or .npy file -> float32 array (12, H, W)."""
        if filename.lower().endswith(".npy"):
            arr = np.load(io.BytesIO(data), allow_pickle=False)
        else:
            import rasterio
            with rasterio.MemoryFile(data) as mf, mf.open() as src:
                arr = src.read()
        arr = np.asarray(arr, np.float32)
        if arr.ndim != 3 or arr.shape[0] != N_BANDS:
            raise ValueError(f"Expected a 12-channel image shaped (12, H, W); got {tuple(arr.shape)}")
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def preprocess(self, arr):
        x = np.clip((arr - self.lo) / (self.hi - self.lo + 1e-6), 0, 1).astype(np.float32)  # same as training
        return torch.from_numpy(x).unsqueeze(0)

    @torch.no_grad()
    def predict(self, arr):
        """arr (12,H,W) raw reflectance -> (mask uint8 {0,1} (H,W), prob float32 (H,W))."""
        h, w = arr.shape[1:]
        x = self.preprocess(arr)
        ph, pw = (-h) % PAD_MULT, (-w) % PAD_MULT
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        prob = torch.sigmoid(self.model(x.to(self.device)))[0, 0, :h, :w].cpu().numpy()
        return (prob > self.thr).astype(np.uint8), prob


def rgb_preview(arr):
    """Percentile-stretched true-colour composite (Red, Green, Blue = bands 3, 2, 1) as uint8 HxWx3."""
    out = np.zeros((*arr.shape[1:], 3), np.float32)
    for k, b in enumerate((3, 2, 1)):
        lo, hi = np.percentile(arr[b], (2, 98))
        out[..., k] = np.clip((arr[b] - lo) / (hi - lo + 1e-6), 0, 1)
    return (out * 255).astype(np.uint8)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("--model", default="models/best_model.pt")
    ap.add_argument("--out", default="mask.png")
    a = ap.parse_args()
    p = Predictor(a.model)
    data = open(a.image, "rb").read()
    mask, _ = p.predict(p.read_image(data, a.image))
    Image.fromarray(mask * 255).save(a.out)
    print(f"saved {a.out} | shape {mask.shape} | water ratio {mask.mean():.3f}")
