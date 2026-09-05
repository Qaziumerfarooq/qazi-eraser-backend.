import numpy as np
import onnxruntime as ort
from PIL import Image
import time

MODEL = r'D:\QaziEraser\backend\models\inpainting_lama_2025jan.onnx'
SIZE = 512

# Build a synthetic 256x320 test image: blue-ish background, with a red square "object"
W, H = 320, 256
img = np.zeros((H, W, 3), dtype=np.uint8)
img[:, :, 0] = 200   # B channel
img[:, :, 2] = 60    # R channel -> blue-teal background
img[90:170, 120:200] = (0, 0, 255)  # red square object (R high) -> BGR (0,0,255)

# Mask covering the red object with some padding
mask = np.zeros((H, W), dtype=np.uint8)
mask[80:180, 110:210] = 255

image_bgr = img  # OpenCV BGR order
image_preprocessed = np.asarray(Image.fromarray(image_bgr).resize((SIZE, SIZE), Image.LANCZOS), dtype=np.float32) / 255.0
# blobFromImage produces NCHW
image_blob = np.transpose(image_preprocessed, (2, 0, 1))[None].astype(np.float32)

mask_resized = np.asarray(Image.fromarray(mask).resize((SIZE, SIZE), Image.NEAREST), dtype=np.float32)
mask_blob = (mask_resized > 0)[None, None].astype(np.float32)

session = ort.InferenceSession(MODEL, providers=['CPUExecutionProvider'])
t = time.time()
outputs = session.run(['output'], {'image': image_blob, 'mask': mask_blob})
dt = time.time() - t
print(f"Inference time: {dt*1000:.1f} ms")

out = outputs[0][0]  # [3,512,512]
out = np.transpose(out, (1, 2, 0))
out_u8 = np.clip(out, 0, 255).astype(np.uint8)

# Crop back to original aspect ratio: model output is 512x512 (square-warped, same as opencv demo)
aspect = H / W
width = out_u8.shape[1]
height = int(width * aspect)
out_crop = np.asarray(Image.fromarray(out_u8).resize((W, H), Image.LANCZOS))

# Check masked region: average color should now be close to background (blue-teal), not red
region = out_crop[80:180, 110:210]
avg = region.reshape(-1, 3).mean(axis=0)
print("Avg color of erased region (BGR):", np.round(avg, 1))
bg = out_crop[0:40, 0:40].reshape(-1, 3).mean(axis=0)
print("Avg background color (BGR):       ", np.round(bg, 1))
orig_region = img[80:180, 110:210].reshape(-1, 3).mean(axis=0)
print("Original object color (BGR):      ", np.round(orig_region, 1))

Image.fromarray(out_crop, 'RGB').save(r'D:\QaziEraser\backend\result_test.png')
print("Result saved. Check that erased region no longer red.")

# Check if the object (red) was successfully removed -> R channel should drop in the region
removed = avg[2] < (orig_region[2] * 0.5)
print("Object removed successfully:" , removed)
