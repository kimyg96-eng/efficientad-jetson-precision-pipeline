import json
import sys
from pathlib import Path

source, target = map(Path, sys.argv[1:])
prior = json.loads(source.read_text())
if prior["unclassified"]:
    raise ValueError("STEP11M inventory contains unclassified Conv nodes")

groups = {}
for category in ("bottle", "carpet"):
    names = prior["categories"][category]["autoencoder"]
    parts = {
        part: [name for name in names if name.startswith(f"/model/ae/{part}/")]
        for part in ("encoder", "decoder")
    }
    if (len(names) != 14 or len(set(names)) != 14
            or len(parts["encoder"]) != 6 or len(parts["decoder"]) != 8):
        raise ValueError(f"{category}: unexpected AE Conv inventory")
    groups[category] = parts

target.write_text(json.dumps({"categories": groups}, indent=2) + "\n")
