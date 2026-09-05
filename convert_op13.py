import onnx
from onnx import helper, numpy_helper, TensorProto
import numpy as np
import os

SRC = r"D:\QaziEraser\app\assets\models\inpainting_lama.onnx"
OUT = r"D:\QaziEraser\app\assets\models\inpainting_lama_op13.onnx"

m = onnx.load(SRC)
g = m.graph
nodes = list(g.node)
new_nodes = []
removed = 0

def attr(n, name):
    for a in n.attribute:
        if a.name == name:
            return onnx.helper.get_attribute_value(a)
    return None

def new_init(name, npv):
    g.initializer.append(numpy_helper.from_array(npv, name))

# pre-scan needed initializer names
existing = {i.name for i in g.initializer}

for n in nodes:
    if n.op_type == "Clip":
        if len(n.input) >= 3 and n.input[2] and n.input[2] not in existing:
            new_init(n.input[2], np.float32(attr(n, "max")))
            existing.add(n.input[2])
        if len(n.input) >= 2 and n.input[1] and n.input[1] not in existing:
            new_init(n.input[1], np.float32(attr(n, "min")))
            existing.add(n.input[1])
        new_nodes.append(n)
    elif n.op_type == "Squeeze" and len(n.input) == 1:
        axes = attr(n, "axes")
        n2 = helper.make_node("Squeeze", n.input,
                              list(n.output), name=n.name)
        if axes is not None:
            if "axes_sq" + n.name not in existing:
                new_init("axes_sq" + n.name, np.array(axes, dtype=np.int64))
                existing.add("axes_sq" + n.name)
            n2.input.append("axes_sq" + n.name)
        for a in list(n2.attribute):
            n2.attribute.remove(a)
        new_nodes.append(n2); removed += 1
    elif n.op_type == "Unsqueeze" and len(n.input) == 1:
        axes = attr(n, "axes")
        if axes is None: new_nodes.append(n); continue
        n2 = helper.make_node("Unsqueeze", [n.input[0], "ux_ax" + n.name],
                              list(n.output), name=n.name)
        if "ux_ax" + n.name not in existing:
            new_init("ux_ax" + n.name, np.array(axes, dtype=np.int64))
            existing.add("ux_ax" + n.name)
        new_nodes.append(n2); removed += 1
    else:
        new_nodes.append(n)

del g.node[:]
for n in new_nodes:
    g.node.append(n)

# trim pad/axes inputs already handled above (axes attr removed manually)
m.opset_import[0].version = 13
onnx.checker.check_model(m)
onnx.save(m, OUT)
print("saved", OUT, "opset", m.opset_import[0].version, "size MB", round(os.path.getsize(OUT)/1e6, 1))
print("removed attr-style squeeze/unsqueeze:", removed)