"""SMOKE TEST ONLY: random-weight checkpoint + 3 synthetic 12-band samples to verify the API plumbing.
Masks from this checkpoint are meaningless. Use the real best_model.pt for real results."""
import os, sys
import numpy as np, torch, rasterio
from rasterio.transform import from_origin
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import build_model

# per-band lo/hi from the notebook's ChannelNormalizer (12 raw bands)
LO = [-1393, -1169, -722, -684, -412, -335, -251, 64, -9999, 8, 10, 0]
HI = [6390, 9659, 11368, 12041, 15841, 15252, 14647, 255, 4245, 4287, 100, 111]
torch.manual_seed(0)
m = build_model("DeepLabV3Plus", "resnet34", 12)
os.makedirs("models", exist_ok=True); os.makedirs("samples", exist_ok=True)
torch.save({"state_dict": m.state_dict(), "arch": "DeepLabV3Plus", "encoder": "resnet34", "channels": list(range(12)),
            "thr": 0.45, "norm_lo": np.array(LO, np.float32), "norm_hi": np.array(HI, np.float32),
            "name": "SMOKE_TEST_random_weights", "test_metrics": None}, "models/smoke_test.pt")
rng = np.random.RandomState(1)
yy, xx = np.mgrid[:128, :128]
for k in range(1, 4):
    water = ((yy - rng.randint(30, 100)) ** 2 + (xx - rng.randint(30, 100)) ** 2) < rng.randint(300, 1500)
    x = np.stack([rng.normal(400, 60, (128, 128)) for _ in range(12)]).astype(np.float32)
    x[4] = np.where(water, 900, 2500) + rng.normal(0, 150, (128, 128))   # NIR dark over water
    x[5] = np.where(water, 800, 2400) + rng.normal(0, 150, (128, 128))
    x[7], x[8], x[9], x[10], x[11] = 100, 150, 300, 30, 0
    with rasterio.open(f"samples/sample_{k}.tif", "w", driver="GTiff", height=128, width=128, count=12,
                       dtype="float32", transform=from_origin(0, 0, 1, 1)) as d:
        d.write(x)
print("wrote models/smoke_test.pt and samples/sample_1..3.tif")
