"""Flask REST API for flood-water segmentation.
  GET  /          HTML frontend
  GET  /health    model info
  POST /predict   multipart field "file" (12-band .tif/.npy)
                  ?format=png (default) -> binary mask PNG (0 = land, 255 = water)
                  ?format=json          -> JSON with water ratio + base64 PNGs (mask, RGB preview)
"""
import base64, io, os
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image
from inference import Predictor, rgb_preview

MODEL_PATH = os.environ.get("MODEL_PATH", "models/best_model.pt")


def png_bytes(a):
    buf = io.BytesIO(); Image.fromarray(a).save(buf, format="PNG"); return buf.getvalue()


def create_app(model_path=MODEL_PATH):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
    predictor = Predictor(model_path)

    @app.get("/")
    def index():
        return render_template("index.html", info=predictor.info)

    @app.get("/health")
    def health():
        return jsonify(status="ok", model=predictor.info)

    @app.post("/predict")
    def predict():
        f = request.files.get("file")
        if f is None or f.filename == "":
            return jsonify(error="No file uploaded. Send multipart/form-data with field 'file'."), 400
        try:
            arr = predictor.read_image(f.read(), f.filename)
        except ValueError as e:
            return jsonify(error=str(e)), 422
        except Exception:
            return jsonify(error="Could not read the file. Upload a 12-band GeoTIFF (.tif) or .npy."), 422
        mask, _ = predictor.predict(arr)
        mask_png = png_bytes(mask * 255)
        ratio = float(mask.mean())
        if request.args.get("format", "png") == "json":
            return jsonify(width=int(mask.shape[1]), height=int(mask.shape[0]),
                           water_ratio=round(ratio, 4), threshold=predictor.thr,
                           mask_png_base64=base64.b64encode(mask_png).decode(),
                           rgb_png_base64=base64.b64encode(png_bytes(rgb_preview(arr))).decode())
        resp = send_file(io.BytesIO(mask_png), mimetype="image/png", download_name="mask.png")
        resp.headers["X-Water-Ratio"] = f"{ratio:.4f}"
        return resp

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error="File too large (max 50 MB)."), 413

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
