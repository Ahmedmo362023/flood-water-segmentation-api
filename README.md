# Flood Water Segmentation API

Flask REST API + HTML frontend that turns a 12-band satellite image into a binary water mask.

**Model:** DeepLabV3+ / ResNet34, ImageNet transfer learning (Week 2, run `P3_deeplab_r34_mean`).
Chosen as the best of Weeks 1-2 by **validation IoU** (0.722 vs 0.698 for the best from-scratch UNet);
selecting on validation, not test, avoids selection bias.

| Model | Val IoU | Test F1 | Test IoU |
|---|---|---|---|
| Week 1: UNet from scratch (12 ch) | 0.698 | 0.809 | 0.679 |
| Week 2: UNet-R34 transfer (mean init) | 0.699 | 0.807 | 0.676 |
| Week 2: UNet-R34 transfer (zero init) | 0.705 | 0.824 | 0.700 |
| **Week 2: DeepLabV3+-R34 transfer (selected)** | **0.722** | 0.820 | 0.696 |

The UNet-R34 zero-init variant has a slightly higher *test* IoU, but the gap is within noise on a 47-image test set.

## Setup
```bash
pip install -r requirements.txt
```
1. Run the last cell of the notebook, `notebook_export_cell.py`. It creates `best_model.pt` and `samples/`.
2. Put `best_model.pt` in `models/` and the sample files in `samples/`.

## Run
```bash
python app.py                      # http://127.0.0.1:5000
# production: gunicorn -w 2 -b 0.0.0.0:8000 "app:create_app()"
```

## API
| Endpoint | Description |
|---|---|
| `GET /` | HTML frontend |
| `GET /health` | model info |
| `POST /predict` | multipart field `file` (12-band `.tif` / `.npy`) |

```bash
# binary mask PNG (0 = land, 255 = water); water ratio in the X-Water-Ratio header
curl -F "file=@samples/sample_1.tif" http://127.0.0.1:5000/predict -o mask.png
# JSON: water_ratio, width, height, mask_png_base64, rgb_png_base64
curl -F "file=@samples/sample_1.tif" "http://127.0.0.1:5000/predict?format=json"
```
Errors: `400` no file, `422` unreadable file or not 12 channels. Any H x W is accepted (padded to a multiple of 32 internally).

## CLI inference
```bash
python inference.py samples/sample_1.tif --out mask.png
```

## Tests
```bash
python tests/test_api.py           # writes outputs/test_results.md
```
