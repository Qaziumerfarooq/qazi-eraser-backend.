import base64
import io
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_PATH = "model.onnx"
PORT = int(os.environ.get("PORT", 8317))

print("Loading model...", flush=True)
_SESSION = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
print("Ready.", flush=True)

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

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def do_POST(self):
        try:
            length = int(self.headers['Content-Length'])
            body = json.loads(self.rfile.read(length))
            image = Image.open(io.BytesIO(base64.b64decode(body["image_b64"])))
            mask = Image.open(io.BytesIO(base64.b64decode(body["mask_b64"])))
            res = _inpaint(image, mask)
            buf = io.BytesIO()
            res.save(buf, format="JPEG", quality=85)
            raw = json.dumps({"result_b64": base64.b64encode(buf.getvalue()).decode("ascii")}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        except Exception as e:
            print(f"Error: {e}")
            self.send_response(500)
            self.end_headers()

if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
