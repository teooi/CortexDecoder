"""One-off: download COCO 2017 captions + NSD stim info to the volume and build nsd_captions.json.

    modal run code/v2/scripts/fetch_captions.py

Writes (under /mnt/braid2-flatmap-data/datasets):
    coco/annotations/captions_{train,val}2017.json
    nsd/nsd_stim_info_merged.csv
    coco/nsd_captions.json    {nsd image id (0-based, = pipeline img id): [5 COCO captions]}
"""
import modal

DATA_DIR = "/mnt/braid2-flatmap-data"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
NSD_STIM_INFO_URL = "https://natural-scenes-dataset.s3.amazonaws.com/nsddata/experiments/nsd/nsd_stim_info_merged.csv"

volume = modal.Volume.from_name("braid2-flatmap-data")
app = modal.App("cortexdecoder-fetch-captions", image=modal.Image.debian_slim(python_version="3.11"))


@app.function(volumes={DATA_DIR: volume}, timeout=30 * 60)
def fetch():
    import csv
    import io
    import json
    import os
    import urllib.request
    import zipfile
    from collections import defaultdict

    coco_dir, nsd_dir = f"{DATA_DIR}/datasets/coco/annotations", f"{DATA_DIR}/datasets/nsd"
    os.makedirs(coco_dir, exist_ok=True)
    os.makedirs(nsd_dir, exist_ok=True)

    print("downloading COCO annotations...")
    with urllib.request.urlopen(COCO_ANN_URL) as r:
        zf = zipfile.ZipFile(io.BytesIO(r.read()))
    captions = defaultdict(list)
    for split in ("train2017", "val2017"):
        raw = zf.read(f"annotations/captions_{split}.json")
        with open(f"{coco_dir}/captions_{split}.json", "wb") as f:
            f.write(raw)
        for ann in json.loads(raw)["annotations"]:
            captions[int(ann["image_id"])].append(ann["caption"].strip())

    print("downloading NSD stim info...")
    stim_path = f"{nsd_dir}/nsd_stim_info_merged.csv"
    urllib.request.urlretrieve(NSD_STIM_INFO_URL, stim_path)
    with open(stim_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 73000, len(rows)
    assert all(int(r["nsdId"]) == i for i, r in enumerate(rows)), "nsdId is not the 0-based row index"

    nsd_captions = {i: captions[int(r["cocoId"])] for i, r in enumerate(rows)}
    missing = [i for i, c in nsd_captions.items() if not c]
    assert not missing, f"{len(missing)} NSD images without captions, e.g. {missing[:5]}"
    with open(f"{DATA_DIR}/datasets/coco/nsd_captions.json", "w") as f:
        json.dump(nsd_captions, f)
    volume.commit()

    counts = sorted(len(c) for c in nsd_captions.values())
    print(f"nsd_captions.json: {len(nsd_captions)} images, captions per image min={counts[0]} max={counts[-1]}")
    for i in (0, 2950, 72999):
        print(i, rows[i]["cocoSplit"], rows[i]["cocoId"], nsd_captions[i][:2])
