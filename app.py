import gradio as gr
import base64
import io
import json
import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_PATH = "model.onnx"
SIZE = 512

SO = ort.SessionOptions()
SO.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
_SESSION = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"], sess_options=SO)

def _inpaint(image, mask):
    src_w, src_h = image.size
    img_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    image_bgr = img_rgb[:, :, ::-1]
    image_pre = np.asarray(Image.fromarray(image_bgr).resize((SIZE, SIZE), Image.LANCZOS), dtype=np.float32) / 255.0
    image_blob = np.transpose(image_pre, (2, 0, 1))[None].astype(np.float32)
    mask_r = np.asarray(mask.convert("L"), dtype=np.uint8)
    mask_resized = np.asarray(Image.fromarray(mask_r).resize((SIZE, SIZE), Image.NEAREST), dtype=np.float32)
    mask_blob = (mask_resized > 0)[None, None].astype(np.float32)
    out = _SESSION.run(["output"], {"image": image_blob, "mask": mask_blob})[0][0]
    out = np.transpose(out, (1, 2, 0))
    out_u8 = np.clip(out, 0, 255).astype(np.uint8)
    result_rgb = out_u8[:, :, ::-1]
    return Image.fromarray(result_rgb, "RGB").resize((src_w, src_h), Image.LANCZOS)

def api_erase(image_b64, mask_b64):
    try:
        image = Image.open(io.BytesIO(base64.b64decode(image_b64)))
        mask = Image.open(io.BytesIO(base64.b64decode(mask_b64)))
        res = _inpaint(image, mask)
        out_buf = io.BytesIO()
        res.save(out_buf, format="JPEG", quality=90)
        return json.dumps({"result_b64": base64.b64encode(out_buf.getvalue()).decode("ascii")})
    except Exception as e:
        return json.dumps({"error": str(e)})

# Mobile app will use this direct endpoint
with gr.Blocks() as demo:
    input_img = gr.Textbox(visible=False)
    input_mask = gr.Textbox(visible=False)
    output = gr.Textbox()
    btn = gr.Button("Erase", visible=False)
    btn.click(fn=api_erase, inputs=[input_img, input_mask], outputs=output)

demo.launch()
