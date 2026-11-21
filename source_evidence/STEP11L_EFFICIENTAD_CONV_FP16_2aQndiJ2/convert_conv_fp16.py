import json
import sys
from pathlib import Path

import onnx
from onnx import TensorProto, helper, shape_inference

source, out = map(Path, sys.argv[1:])
report = {}

for category in ("bottle", "carpet"):
    stem = f"{category}_efficientad_s"
    model = onnx.load(source / f"{stem}_fp32.onnx")
    inferred = shape_inference.infer_shapes(model, strict_mode=True)
    types = {
        value.name: value.type.tensor_type.elem_type
        for value in (
            *inferred.graph.input, *inferred.graph.output, *inferred.graph.value_info
        )
        if value.type.HasField("tensor_type")
    }
    types.update({item.name: item.data_type for item in inferred.graph.initializer})

    nodes = []
    converted = []
    for i, original in enumerate(model.graph.node):
        if original.op_type != "Conv" or original.domain not in ("", "ai.onnx"):
            nodes.append(original)
            continue

        operands = [name for name in original.input if name]
        if (len(original.output) != 1 or
                any(types.get(name) != TensorProto.FLOAT for name in operands)):
            raise ValueError(f"{category}: unexpected Conv types at {original.name!r}")

        prefix = f"step11l_conv_{i}"
        node = onnx.NodeProto()
        node.CopyFrom(original)

        for j, name in enumerate(original.input):
            if not name:
                continue
            cast_name = f"{prefix}_in_{j}_half"
            nodes.append(helper.make_node(
                "Cast", [name], [cast_name], name=cast_name, to=TensorProto.FLOAT16,
            ))
            node.input[j] = cast_name

        node.output[0] = f"{prefix}_out_half"
        nodes.append(node)
        nodes.append(helper.make_node(
            "Cast", [node.output[0]], [original.output[0]],
            name=f"{prefix}_out_float", to=TensorProto.FLOAT,
        ))
        converted.append(original.name or f"node_{i}")

    if not converted:
        raise ValueError(f"{category}: no Conv nodes found")

    del model.graph.node[:]
    model.graph.node.extend(nodes)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, out / f"{stem}_conv_fp16.onnx")
    report[category] = {"converted_conv_count": len(converted), "conv_nodes": converted}
    print(f"{category}: CONV_FP16_NODES={len(converted)} / ONNX_CHECK=PASS")

(out / "conversion.json").write_text(json.dumps(report, indent=2) + "\n")
