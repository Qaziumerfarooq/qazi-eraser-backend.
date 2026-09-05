import base64
import io
import json
import os
from flask import Flask, request, jsonify
import numpy as np
import onnxruntime as ort
from PIL import Image

app = Flask(__name__)

# Load Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.onnx")
_SESSION = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

def _inpaint(image, mask):
    src_w, src_h = image.size
    img_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    image_bgr = img_rgb[:, :, ::-1]
    image_pre = np.asarray(Image.fromarray(image_bgr).resize((512, 512), Image.LANCZOS), dtype=np.float32) / 255.0
    image_blob = np.transpose(image_pre, (2, 0, 1))[None].astype(np.float32)
    mask_r = np.asarray(mask.convert("L"), dtype=np.uint8)
    mask_resized = np.asarray(Image.fromarray(mask_r).resize((512, 512), Image.NEAREST), dtype=np.float32)
    mask_blob = (mask_resized > 0)[None, None].astype(np.float32)
    out = _SESSION.run(["output"], {"image": image_blob, "mask": mask_blob})[0][0]
    out = np.transpose(out, (1, 2, 0))
    out_u8 = np.clip(out, 0, 255).astype(np.uint8)
    result_rgb = out_u8[:, :, ::-1]
    return Image.fromarray(result_rgb, "RGB").resize((src_w, src_h), Image.LANCZOS)

@app.route("/erase", methods=["POST"])
def erase():
    try:
        data = request.json
        image = Image.open(io.BytesIO(base64.b64decode(data["image_b64"])))
        mask = Image.open(io.BytesIO(base64.b64decode(data["mask_b64"])))
        res = _inpaint(image, mask)
        buf = io.BytesIO()
        res.save(buf, format="JPEG", quality=85)
        return jsonify({"result_b64": base64.b64encode(buf.getvalue()).decode("ascii")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def health():
    return "ok"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8317)
