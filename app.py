import gradio as gr
import base64
import io
import json
import numpy as np
import onnxruntime as ort
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

# Load Model
MODEL_PATH = "model.onnx"
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

# Create FastAPI app
app = FastAPI()

@app.post("/erase")
async def erase(request: Request):
    try:
        body = await request.json()
        image = Image.open(io.BytesIO(base64.b64decode(body["image_b64"])))
        mask = Image.open(io.BytesIO(base64.b64decode(body["mask_b64"])))
        res = _inpaint(image, mask)
        buf = io.BytesIO()
        res.save(buf, format="JPEG", quality=85)
        return {"result_b64": base64.b64encode(buf.getvalue()).decode("ascii")}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/health")
async def health():
    return "ok"

# Gradio dummy UI for Hugging Face compatibility
with gr.Blocks() as demo:
    gr.Markdown("# Qazi Eraser API is Running")

# Mount FastAPI onto Gradio
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
