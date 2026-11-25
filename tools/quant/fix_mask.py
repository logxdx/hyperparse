"""LayoutNet with FP16-safe attention masks: the two -1e9 mask constants (encoder self-attention,
image cross-attention) become -3e4. Softmax weights on masked entries stay exactly 0."""
import numpy as np, onnx
from onnx import numpy_helper

src = "/home/logx/.cache/huggingface/hub/models--logxdx--hyperparse-models/snapshots/f7ec3a8f436d999407be2ef8054502b4f3d38aac/tablenet.onnx"
dst = "/home/logx/.claude/jobs/784b7a0d/tmp/quant/tablenet_fixmask.onnx"
m = onnx.load(src)
n_fixed = 0
for node in m.graph.node:
    if node.op_type != "Constant" or not node.attribute or node.attribute[0].type != onnx.AttributeProto.TENSOR:
        continue
    v = numpy_helper.to_array(node.attribute[0].t)
    if v.dtype == np.float32 and v.size == 1 and float(v.ravel()[0]) == -1e9:
        node.attribute[0].t.CopyFrom(numpy_helper.from_array(np.array(-3e4, np.float32).reshape(v.shape), node.attribute[0].t.name))
        n_fixed += 1
assert n_fixed == 2, n_fixed
onnx.checker.check_model(m)
onnx.save(m, dst)
print("fixed", n_fixed, "->", dst)
