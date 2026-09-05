import onnx
from onnx import version_converter
import os

SRC = r"D:\QaziEraser\app\assets\models\inpainting_lama.onnx"
OUT = r"D:\QaziEraser\app\assets\models\inpainting_lama_op15.onnx"
TARGET = 15

m = onnx.load(SRC)
print("current opset:", [(o.domain, o.version) for o in m.opset_import])

converted = version_converter.convert_version(m, TARGET)
onnx.checker.check_model(converted)
onnx.save(converted, OUT)
print("saved", OUT)
print("new opset:", [(o.domain, o.version) for o in converted.opset_import])
print("size MB:", round(os.path.getsize(OUT)/1e6, 1))