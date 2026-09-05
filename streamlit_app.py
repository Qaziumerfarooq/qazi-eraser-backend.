import streamlit as st
import base64
import io
import json
import numpy as np
import onnxruntime as ort
from PIL import Image

# Force Model Path for Streamlit
MODEL_PATH = "model.onnx"

@st.cache_resource
def load_model():
    SO = ort.SessionOptions()
    SO.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"], sess_options=SO)

_SESSION = load_model()

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

st.title("Qazi Eraser API")
st.write("Server is live.")

# This acts as a simple API for the mobile app
query_params = st.query_params
if "image_b64" in query_params and "mask_b64" in query_params:
    try:
        img_data = base64.b64decode(query_params["image_b64"])
        mask_data = base64.b64decode(query_params["mask_b64"])
        image = Image.open(io.BytesIO(img_data))
        mask = Image.open(io.BytesIO(mask_data))
        res = _inpaint(image, mask)
        out_buf = io.BytesIO()
        res.save(out_buf, format="JPEG", quality=90)
        st.json({"result_b64": base64.b64encode(out_buf.getvalue()).decode("ascii")})
    except Exception as e:
        st.json({"error": str(e)})
