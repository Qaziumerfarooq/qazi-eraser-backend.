import gradio as gr
import base64
import io
import numpy as np
import onnxruntime as ort
from PIL import Image

# Load Model
_SESSION = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])

def predict(image_b64, mask_b64):
    try:
        # Decode
        img_raw = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        mask_raw = Image.open(io.BytesIO(base64.b64decode(mask_b64))).convert("L")
        w, h = img_raw.size

        # Process
        img_np = np.asarray(img_raw.resize((512, 512)), dtype=np.float32) / 255.0
        img_blob = np.transpose(img_np[:, :, ::-1], (2, 0, 1))[None].astype(np.float32)
        mask_np = np.asarray(mask_raw.resize((512, 512)), dtype=np.float32)
        mask_blob = (mask_np > 0)[None, None].astype(np.float32)

        # Run AI
        out = _SESSION.run(["output"], {"image": img_blob, "mask": mask_blob})[0][0]

        # Post-process
        out = np.transpose(out, (1, 2, 0))
        res_img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)[:, :, ::-1])
        res_img = res_img.resize((w, h), Image.LANCZOS)

        # Encode
        buf = io.BytesIO()
        res_img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as e:
        return str(e)

# Gradio Interface (Standard for Hugging Face)
demo = gr.Interface(
    fn=predict,
    inputs=[gr.Textbox(), gr.Textbox()],
    outputs=gr.Textbox(),
    api_name="erase"
)

if __name__ == "__main__":
    demo.launch()
