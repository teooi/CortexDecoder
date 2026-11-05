"""Start runs on the *deployed* app and exit immediately; they keep running if the laptop sleeps or closes.

    modal deploy code/v2/modal_app.py      # once, and again after any code change
    python3 code/v2/scripts/spawn_runs.py clip '{"timestamp": "20261031_000000", "n_layers": 8}' '{...}'

stage: vgg | clip | final | clip+final. Each JSON object is one run's config overrides (same keys as modal_app.py --extra).
Unlike `modal run --detach`, no local client stays attached, so nothing on this machine can cancel the calls.
"""
import json
import sys

import modal

APP = "cortexdecoder-v2"
FUNCTIONS = {"vgg": "train_vgg", "clip": "train_clip", "final": "final_diffusion", "clip+final": "train_clip_then_final"}


def main(stage, *configs):
    fn = modal.Function.from_name(APP, FUNCTIONS[stage])
    for raw in configs:
        overrides = json.loads(raw)
        call = fn.spawn(overrides)
        print(f"spawned {FUNCTIONS[stage]} {overrides} -> {call.object_id}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], *sys.argv[2:])
