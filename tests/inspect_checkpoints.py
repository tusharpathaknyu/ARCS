"""Inspect all training checkpoints and history."""
import sys
import torch
import json
from pathlib import Path

# Add circuitgenie to path so old checkpoints can unpickle
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for d in ["checkpoints", "checkpoints_v2", "checkpoints_v3"]:
    p = Path(d)
    if not p.exists():
        continue
    files = sorted(p.iterdir())
    print(f"\n=== {d} ===")
    for f in files:
        if f.suffix == ".pt":
            ckpt = torch.load(f, map_location="cpu", weights_only=False)
            epoch = ckpt.get("epoch", "?")
            val_loss = ckpt.get("val_loss", "?")
            config = ckpt.get("config", {})
            if hasattr(config, "__dict__"):
                config = vars(config)
            if not isinstance(config, dict):
                config = {}
            vs = config.get("vocab_size", "?")
            dm = config.get("d_model", "?")
            nl = config.get("n_layers", "?")
            print(f"  {f.name}: epoch={epoch}, val_loss={val_loss}, vocab={vs}, d={dm}, layers={nl}")
        elif f.suffix == ".json":
            with open(f) as fh:
                data = json.load(fh)
            if isinstance(data, list) and len(data) > 0:
                last = data[-1]
                print(f"  {f.name}: {len(data)} records")
                print(f"    first: {data[0]}")
                print(f"    last:  {last}")
            elif isinstance(data, dict):
                print(f"  {f.name}: {json.dumps(data, indent=2)[:200]}")
