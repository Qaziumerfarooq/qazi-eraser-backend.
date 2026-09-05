import base64
import io
import json
import threading
import time
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import onnxruntime as ort
from PIL import Image

# For Cloud Deployment
MODEL_PATH = "model.onnx"
SIZE = 512
HOST = "0.0.0.0"
# Hugging Face default port is 7860
PORT = int(os.environ.get("PORT", 7860))

print("[server] Loading model...", flush=True)
t0 = time.time()
SO = ort.SessionOptions()
SO.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
SO.intra_op_num_threads = 1
SO.log_severity_level = 3
_SESSION = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"], sess_options=SO)
print(f"[server] Model loaded in {time.time()-t0:.1f}s. Ready.", flush=True)
_LOCK = threading.Lock()

def _inpaint(image: Image.Image, mask: Image.Image) -> Image.Image:
    src_w, src_h = image.size
    img_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    image_bgr = img_rgb[:, :, ::-1]
    image_pre = np.asarray(
        Image.fromarray(image_bgr).resize((SIZE, SIZE), Image.LANCZOS), dtype=np.float32
    ) / 255.0
    image_blob = np.transpose(image_pre, (2, 0, 1))[None].astype(np.float32)

    mask_r = np.asarray(mask.convert("L"), dtype=np.uint8)
    mask_resized = np.asarray(
        Image.fromarray(mask_r).resize((SIZE, SIZE), Image.NEAREST), dtype=np.float32
    )
    mask_blob = (mask_resized > 0)[None, None].astype(np.float32)

    with _LOCK:
        out = _SESSION.run(["output"], {"image": image_blob, "mask": mask_blob})[0][0]

    out = np.transpose(out, (1, 2, 0))
    out_u8 = np.clip(out, 0, 255).astype(np.uint8)
    result_rgb = out_u8[:, :, ::-1]
    result = Image.fromarray(result_rgb, "RGB").resize((src_w, src_h), Image.LANCZOS)
    return result

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass

    def do_GET(self):
        if self.path == "/health" or self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/log":
            self.send_response(200)
            self.end_headers()
            return

        if self.path != "/erase":
            self.send_response(404)
            self.end_headers()
            return

        try:
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            payload = json.loads(body)

            image = Image.open(io.BytesIO(base64.b64decode(payload["image_b64"])))
            mask = Image.open(io.BytesIO(base64.b64decode(payload["mask_b64"])))

            result = _inpaint(image, mask)

            out_buf = io.BytesIO()
            result.save(out_buf, format="JPEG", quality=90)

            response = {
                "result_b64": base64.b64encode(out_buf.getvalue()).decode("ascii")
            }
            raw = json.dumps(response).encode("utf-8")

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
    print(f"[server] listening on port {PORT}", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
