"""
infer_modality_auc_unified.py
------------------------------
Unified inference script for both sparse (masked=True) and dense (masked=False)
multimodal ResNet3D models.  Supports groupnorm and batchnorm only.

Sparse  → wraps model in MultiMaskSNIPWrapper, passes modality_tensor to forward.
Dense   → bare ResNet3D, does NOT pass modality_tensor to forward.

Usage:
    # sparse
    python3 infer_modality_auc_unified.py \
        --checkpoint  slurm_scripts/logs/<run>/fold_0/model.best.pth \
        --test_ids    slurm_scripts/logs/<run>/fold_0/test_ids.txt \
        --norm_type   groupnorm \
        --masked      True \
        --sparsity    0.7 \
        --db_host     10.245.12.58 \
        --fold        0

    # dense
    python3 infer_modality_auc_unified.py \
        --checkpoint  slurm_scripts/logs/<run>/fold_0/model.best.pth \
        --test_ids    slurm_scripts/logs/<run>/fold_0/test_ids.txt \
        --norm_type   batchnorm \
        --masked      False \
        --db_host     10.245.12.58 \
        --fold        0
"""

import argparse
import io
import sys

import torch
import numpy as np
from pymongo import MongoClient
from sklearn.metrics import roc_auc_score

from resnet import ResNet3D

# ── constants ─────────────────────────────────────────────────────────────────
DATABASE    = "multimodalSubnetworks"
COLLECTION  = "fbirn"        # overridden by --collection at runtime
DB_FIELDS   = ("falff", "smri", "dwi")
LABEL_FIELD = "gender_encoded"
N_CLASSES   = 1
CHANNELS    = 64

# smri→0, falff→1, dwi→2  (matches customMongoDataset map_modality_codes)
MOD_NAMES = {0: "smri", 1: "falff", 2: "dwi"}
MOD_CODES = {"smri": 0, "falff": 1, "dwi": 2}
# ─────────────────────────────────────────────────────────────────────────────


def mytransform(x: bytes) -> torch.Tensor:
    LZ4_MAGIC = b"\x04\x22\x4d\x18"
    if x[:4] == LZ4_MAGIC:
        import lz4.frame
        x = lz4.frame.decompress(x)
    return torch.load(io.BytesIO(x), weights_only=False)


def safe_normalize(img: torch.Tensor) -> torch.Tensor:
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-8:
        return torch.zeros_like(img)
    return (img - mn) / (mx - mn)


def load_ids(path: str):
    ids = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                ids.append(int(s))
            except ValueError:
                ids.append(s)
    return ids


def build_model(checkpoint_path: str, device: torch.device,
                norm_type: str, masked: bool, sparsity: float) -> torch.nn.Module:
    base_model = ResNet3D(
        in_channels=1,
        n_classes=N_CLASSES,
        channels=CHANNELS,
        norm_type=norm_type,
    )

    if masked:
        from src.masked_model import MultiMaskSNIPWrapper
        model = MultiMaskSNIPWrapper(base_model, sparsity=sparsity)
        model.prepare_for_loading([0, 1, 2])
    else:
        model = base_model

    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    all_keys = list(state_dict.keys())
    print(f"  Total checkpoint keys: {len(all_keys)}. First 5: {all_keys[:5]}")

    if masked:
        orig_keys = [k for k in all_keys if "original" in k]
        mask_keys = [k for k in all_keys if "mask_" in k]
        print(f"  'original' keys: {len(orig_keys)},  mask keys: {len(mask_keys)}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys  ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    print("Checkpoint loaded successfully.")

    model.to(device)
    model.eval()
    return model


def run_inference(model, test_ids, db_host, device, masked):
    client   = MongoClient(f"mongodb://{db_host}:27017")
    db       = client[DATABASE]
    bin_col  = db[COLLECTION + ".bin"]
    meta_col = db[COLLECTION + ".meta"]

    print(f"MongoDB connection OK — {meta_col.estimated_document_count()} docs in {COLLECTION}.meta")

    mod_preds  = {k: [] for k in MOD_NAMES}
    mod_labels = {k: [] for k in MOD_NAMES}
    all_preds, all_labels = [], []
    per_subj_preds, per_subj_labels = [], []
    skipped = 0
    processed = 0

    for subj_idx, subject_id in enumerate(test_ids):
        meta = meta_col.find_one(
            {"id": subject_id},
            {"id": 1, LABEL_FIELD: 1, "modalities": 1, "_id": 0},
        )
        if meta is None:
            print(f"  [WARN] No meta for '{subject_id}', skipping")
            skipped += 1
            continue

        label_val = meta.get(LABEL_FIELD)
        if label_val is None:
            print(f"  [WARN] No label for '{subject_id}', skipping")
            skipped += 1
            continue

        subject_mods = set(meta.get("modalities", [])).intersection(set(DB_FIELDS))
        if not subject_mods:
            print(f"  [WARN] No matching modalities for '{subject_id}', skipping")
            skipped += 1
            continue

        subj_probas = []
        for mod in subject_mods:
            mod_id = MOD_CODES[mod]

            chunks = list(bin_col.find(
                {"id": subject_id, "kind": mod},
                {"id": 1, "chunk": 1, "chunk_id": 1, "_id": 0},
            ))
            if not chunks:
                print(f"  [WARN] No chunks for {subject_id}/{mod}, skipping")
                continue
            chunks.sort(key=lambda c: c["chunk_id"])
            raw = b"".join(c["chunk"] for c in chunks)

            volume = mytransform(raw).float()
            volume = safe_normalize(volume)
            if volume.dim() == 3:
                volume = volume.unsqueeze(0)
            volume = volume.unsqueeze(0).to(device)

            modality_tensor = torch.tensor([mod_id], dtype=torch.long).to(device)

            with torch.no_grad():
                if masked:
                    y_hat = model.forward(volume, modality_tensor)
                else:
                    y_hat = model.forward(volume)
                proba = torch.sigmoid(y_hat).item()

            all_preds.append(proba)
            all_labels.append(int(label_val))
            mod_preds[mod_id].append(proba)
            mod_labels[mod_id].append(int(label_val))
            subj_probas.append(proba)
            processed += 1

        if subj_probas:
            per_subj_preds.append(float(np.mean(subj_probas)))
            per_subj_labels.append(int(label_val))

        if (subj_idx + 1) % 10 == 0:
            print(f"  Processed {subj_idx + 1}/{len(test_ids)} subjects "
                  f"({processed} pairs so far)...")
            sys.stdout.flush()

    print(f"\nDone: {processed} pairs from "
          f"{len(test_ids) - skipped}/{len(test_ids)} subjects ({skipped} skipped).")
    return mod_preds, mod_labels, all_preds, all_labels, per_subj_preds, per_subj_labels


def report(mod_preds, mod_labels, all_preds, all_labels, fold, norm_type, masked):
    mode = "sparse" if masked else "dense"
    print(f"\n{'='*60}")
    print(f"  Fold {fold} — {norm_type} {mode} — Test AUC (best checkpoint)")
    print(f"{'='*60}")
    for mod_id, name in MOD_NAMES.items():
        n = len(mod_labels[mod_id])
        if n == 0:
            print(f"  {name:<8}  n=0  (no samples)")
            continue
        try:
            auc = roc_auc_score(mod_labels[mod_id], mod_preds[mod_id])
            print(f"  {name:<8}  AUC = {auc:.4f}   (n={n})")
        except Exception as e:
            print(f"  {name:<8}  AUC = ERROR ({e})  (n={n})")
    if all_labels:
        try:
            overall = roc_auc_score(all_labels, all_preds)
            print(f"  {'overall':<8}  AUC = {overall:.4f}   (n={len(all_labels)})")
        except Exception as e:
            print(f"  {'overall':<8}  AUC = ERROR ({e})  (n={len(all_labels)})")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test_ids",   required=True)
    parser.add_argument("--norm_type",  default="groupnorm",
                        choices=["batchnorm", "groupnorm"])
    parser.add_argument("--masked",     required=True,
                        type=lambda x: x.lower() in ("true", "1", "yes"),
                        help="True for sparse (MultiMaskSNIPWrapper), False for dense")
    parser.add_argument("--sparsity",   default=0.7, type=float,
                        help="Sparsity level — only used when --masked True")
    parser.add_argument("--db_host",    default="10.245.12.58")
    parser.add_argument("--collection", default="fbirn")
    parser.add_argument("--fold",       default=0, type=int)
    args = parser.parse_args()

    global COLLECTION
    COLLECTION = args.collection

    mode = "sparse" if args.masked else "dense"
    sps_str = f"sparsity={args.sparsity}" if args.masked else "sparsity=N/A"
    print(f"Collection: {COLLECTION}  |  norm_type: {args.norm_type}  |  "
          f"mode: {mode}  |  {sps_str}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_ids = load_ids(args.test_ids)
    print(f"Loaded {len(test_ids)} test subjects. First 3: {test_ids[:3]}")

    model = build_model(args.checkpoint, device, args.norm_type,
                        args.masked, args.sparsity)

    print("\n--- Sanity check ---")
    for name, param in model.named_parameters():
        print(f"  First param [{name}]: mean={param.data.mean():.6f}, "
              f"std={param.data.std():.6f}, shape={list(param.data.shape)}")
        break
    print("--- end ---\n")

    mod_preds, mod_labels, all_preds, all_labels, per_subj_preds, per_subj_labels = \
        run_inference(model, test_ids, args.db_host, device, args.masked)

    report(mod_preds, mod_labels, all_preds, all_labels,
           args.fold, args.norm_type, args.masked)


if __name__ == "__main__":
    main()
