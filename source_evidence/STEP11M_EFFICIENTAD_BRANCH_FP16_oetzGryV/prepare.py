import json
import re
import sys
from pathlib import Path

import onnx

onnx_dir, prior, out = map(Path, sys.argv[1:])
branches = ("teacher", "student", "autoencoder")
aliases = {"teacher": "teacher", "student": "student",
           "ae": "autoencoder", "autoencoder": "autoencoder"}
inventory = {}
unclassified = []

for category in ("bottle", "carpet"):
    model = onnx.load(onnx_dir / f"{category}_efficientad_s_fp32.onnx")
    groups = {branch: [] for branch in branches}
    for node in model.graph.node:
        if node.op_type != "Conv" or node.domain not in ("", "ai.onnx"):
            continue
        # Use both the module path and weight input name.
        operands = [node.name, *node.input[1:]]
        tokens = re.split(r"[^a-z0-9_]+", " ".join(operands).lower())
        hits = {aliases[token] for token in tokens if token in aliases}
        if len(hits) != 1 or not node.name:
            unclassified.append({"category": category, "node": node.name,
                                 "operands": operands})
            continue
        groups[hits.pop()].append(node.name)
    inventory[category] = groups

report = {"categories": inventory, "unclassified": unclassified}
text = json.dumps(report, indent=2)
(out / "conv_branch_inventory.json").write_text(text + "\n")
print(text)
if unclassified or any(not names for groups in inventory.values() for names in groups.values()):
    raise SystemExit("Conv branch classification incomplete. See inventory above.")

source = (prior / "convert_conv_fp16.py").read_text()
old = 'source, out = map(Path, sys.argv[1:])'
new = '''source, out, inventory_path = map(Path, sys.argv[1:4])
branch = sys.argv[4]
inventory = json.loads(inventory_path.read_text())["categories"]'''
if source.count(old) != 1:
    raise SystemExit("Unexpected STEP11L converter arguments")
source = source.replace(old, new, 1)

old = '        if original.op_type != "Conv" or original.domain not in ("", "ai.onnx"):'
new = '''        if (original.op_type != "Conv"
                or original.domain not in ("", "ai.onnx")
                or original.name not in inventory[category][branch]):'''
if source.count(old) != 1:
    raise SystemExit("Unexpected STEP11L Conv filter")
source = source.replace(old, new, 1)

target = out / "convert_branch_fp16.py"
compile(source, str(target), "exec")
target.write_text(source)
