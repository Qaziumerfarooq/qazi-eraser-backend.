import gradio as gr
import base64
import io
import numpy as np
import onnxruntime as ort
from PIL import Image

# Load the AI model
MODEL_PATH = "model.onnx"
_SESSION = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

def process_erase(image_b64, mask_b64):
    try:
        # 1. Decode inputs
        img_data = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        mask_data = Image.open(io.BytesIO(base64.b64decode(mask_b64))).convert("L")

        src_w, src_h = img_data.size

        # 2. Preprocess
        img_np = np.asarray(img_data.resize((512, 512), Image.LANCZOS), dtype=np.float32) / 255.0
        img_bgr = img_np[:, :, ::-1] # BGR for model
        img_blob = np.transpose(img_bgr, (2, 0, 1))[None].astype(np.float32)

        mask_np = np.asarray(mask_data.resize((512, 512), Image.NEAREST), dtype=np.float32)
        mask_blob = (mask_np > 0)[None, None].astype(np.float32)

        # 3. Run AI
        out = _SESSION.run(["output"], {"image": img_blob, "mask": mask_blob})[0][0]

        # 4. Postprocess
        out = np.transpose(out, (1, 2, 0))
        out_u8 = np.clip(out, 0, 255).astype(np.uint8)
        res_rgb = out_u8[:, :, ::-1] # BGR -> RGB
        res_img = Image.fromarray(res_rgb).resize((src_w, src_h), Image.LANCZOS)

        # 5. Encode result
        buf = io.BytesIO()
        res_img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as e:
        return f"Error: {str(e)}"

# Define a simple Gradio Interface for the API
demo = gr.Interface(
    fn=process_erase,
    inputs=[gr.Textbox(label="image_b64"), gr.Textbox(label="mask_b64")],
    outputs=gr.Textbox(label="result_b64"),
    title="Qazi Eraser API"
)

if __name__ == "__main__":
    demo.launch()
