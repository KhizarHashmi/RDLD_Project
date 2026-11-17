"""
p0_download_data.py -- P0 (Khizar, Workstream B): download the DocLayNet v1.2 files and model checkpoints the study used.

Everything is fetched at the exact revision we used, into the standard Hugging Face cache
(~/.cache/huggingface/hub), which is where p0_prepare_doclaynet.py and p7_prepare_train_data.py look for it.

  data/validation-*.parquet   all 7 shards   (validation page set, and RF-DETR's selection pages)
  data/test-*.parquet         all 6 shards   (official test page set)
  data/train-*.parquet        every 6th of the 72 shards (12 shards, RF-DETR fine-tuning only)

With --models it also downloads the ten released layout checkpoints and the RF-DETR base model at the revisions
used in the paper, so later runs read them from the cache. Files already in the cache are skipped.

Run:
  python scripts/p0_download_data.py              # dataset shards: 5.1 GB validation + test, 5.5 GB train
  python scripts/p0_download_data.py --models     # dataset shards and model checkpoints
  python scripts/p0_download_data.py --no-train   # skip the train shards if RF-DETR is not retrained

The ten released layout checkpoints come from the first seven model ids below (hantian/yolo-doclaynet holds the
four YOLO models). Roboflow/rf-detr-medium is the base that p7_finetune_rfdetr.py fine-tunes; its DINOv2 backbone is
part of that checkpoint.
"""
import sys

from huggingface_hub import hf_hub_download, snapshot_download

DATASET = "ds4sd/DocLayNet-v1.2"          # now hosted as docling-project/DocLayNet-v1.2; the old id redirects
DATASET_REV = "0daf93102e2efce76c3e11a274a5e0d0969391d3"

# model id -> revision used for every result in the paper (checked against the Hub on 18 Nov 2026)
MODELS = {
    "hantian/yolo-doclaynet": "49b97586dbd3bdae169e8f5e165710d0facf5f1e",
    "ds4sd/docling-layout-heron": "8f39ad3c0b4c58e9c2d2c84a38465abf757272d8",
    "docling-project/docling-layout-heron-101": "2e4993cf6bb211112084a2f80938f26138008917",
    "ds4sd/docling-layout-egret-medium": "77ede7cc7bed96d853c58f319734803d6ea2ea5c",
    "docling-project/docling-layout-egret-xlarge": "23857d16596e0106716b3162d132212d733769e7",
    "cmarkea/detr-layout-detection": "b30837562d210bde4a3a0bb246133f4fdeb29fbe",
    "Aryn/deformable-detr-DocLayNet": "d5503a90ae08dd43565de6984a5dd7924cad2400",
    "Roboflow/rf-detr-medium": "1b5b672408f86dd38e05dd3cf3f2e0834e545a59",
}


def files(train=True):
    out = [f"data/validation-{i:05d}-of-00007.parquet" for i in range(7)]
    out += [f"data/test-{i:05d}-of-00006.parquet" for i in range(6)]
    if train:
        out += [f"data/train-{i:05d}-of-00072.parquet" for i in range(0, 72, 6)]
    return out


def main():
    todo = files(train="--no-train" not in sys.argv)
    print(f"{DATASET} @ {DATASET_REV[:10]}: {len(todo)} files")
    for f in todo:
        hf_hub_download(DATASET, f, repo_type="dataset", revision=DATASET_REV)
        print("  ok", f)
    if "--models" in sys.argv:
        for repo, rev in MODELS.items():
            snapshot_download(repo, revision=rev, ignore_patterns=["*.md", ".gitattributes", "*.png", "*.jpg"])
            print(f"  ok {repo} @ {rev[:10]}")


if __name__ == "__main__":
    main()
