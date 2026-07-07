"""
infer_modality_auc.py
---------------------
Load a trained GroupNorm multimodal model from a saved checkpoint and evaluate
it on the held-out test set, reporting AUC separately for each modality
(fALFF, sMRI, DWI) and overall.

No training, no SNIP re-initialization, no weight updates.

Usage:
    python3 infer_modality_auc.py \
        --checkpoint  /path/to/fold_0/model.best.pth \
        --test_ids    /path/to/fold_0/test_ids.txt \
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
from src.masked_model import MultiMaskSNIPWrapper

# ── constants matching the original training run ──────────────────────────────
DATABASE   = "multimodalSubnetworks"
COLLECTION = "fbirn"
DB_FIELDS  = ("falff", "smri", "dwi")
LABEL_FIELD = "gender_encoded"

# map_modality_codes in customMongoDataset: smri→0, falff→1, dwi→2
MOD_NAMES  = {0: "smri", 1: "falff", 2: "dwi"}
MOD_CODES  = {"smri": 0, "falff": 1, "dwi": 2}

N_CLASSES  = 1
CHANNELS   = 64
SPARSITY   = 0.5
# ─────────────────────────────────────────────────────────────────────────────


def mytransform(x: bytes) -> torch.Tensor:
    """Decompress LZ4 if needed, then deserialize a torch tensor."""
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
    """Load subject IDs, preserving their original type (int if numeric)."""
    ids = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                ids.append(int(s))   # stored as int in MongoDB
            except ValueError:
                ids.append(s)        # stored as string
    return ids


def build_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Build architecture, register parametrizations, load checkpoint weights."""
    base_model = ResNet3D(
        in_channels=1,
        n_classes=N_CLASSES,
        channels=CHANNELS,
        norm_type="groupnorm",
    )
    model = MultiMaskSNIPWrapper(base_model, sparsity=SPARSITY)

    # Register dummy parametrizations so state_dict keys match.
    # Checkpoint uses integer IDs (mask_0, mask_1, mask_2), not field names.
    model.prepare_for_loading([0, 1, 2])

    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # ── Checkpoint key diagnostics (before loading) ──────────────────────────
    orig_keys = [k for k in state_dict.keys() if "original" in k]
    print(f"  Checkpoint 'original' keys ({len(orig_keys)} found): {orig_keys[:3]}{'...' if len(orig_keys) > 3 else ''}")
    if orig_keys:
        t = state_dict[orig_keys[0]]
        print(f"  Checkpoint [{orig_keys[0]}]: mean={t.mean():.6f}, std={t.std():.6f}, shape={list(t.shape)}")
    mask_keys = [k for k in state_dict.keys() if "mask_" in k]
    print(f"  Checkpoint mask keys ({len(mask_keys)} found): {mask_keys[:3]}{'...' if len(mask_keys) > 3 else ''}")
    all_keys = list(state_dict.keys())
    print(f"  Total checkpoint keys: {len(all_keys)}. First 5: {all_keys[:5]}")
    # ─────────────────────────────────────────────────────────────────────────

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys  ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    print("Checkpoint loaded successfully.")

    model.to(device)
    model.eval()
    return model


def run_inference(model, test_ids, db_host, device):
    """
    Query MongoDB directly (no DataLoader) and run forward passes one
    subject×modality at a time.  Returns per-modality, overall, and
    per-subject predictions (used by --split_halves).
    """
    client = MongoClient(f"mongodb://{db_host}:27017")
    db     = client[DATABASE]
    bin_col  = db[COLLECTION + ".bin"]
    meta_col = db[COLLECTION + ".meta"]

    # Quick connectivity check
    n_subjects_db = meta_col.estimated_document_count()
    print(f"MongoDB connection OK — {n_subjects_db} documents in {COLLECTION}.meta")

    mod_preds  = {k: [] for k in MOD_NAMES}
    mod_labels = {k: [] for k in MOD_NAMES}
    all_preds  = []
    all_labels = []

    # per-subject accumulator: list of (rank_position, mean_proba, label)
    # rank_position = order subject was encountered among non-skipped subjects
    per_subj_preds  = []   # mean predicted probability across modalities per subject
    per_subj_labels = []   # label per subject

    skipped = 0
    processed = 0

    for subj_idx, subject_id in enumerate(test_ids):
        # ── 1. fetch metadata ──────────────────────────────────────────────
        meta = meta_col.find_one(
            {"id": subject_id},
            {"id": 1, LABEL_FIELD: 1, "modalities": 1, "_id": 0},
        )
        if meta is None:
            print(f"  [WARN] No meta for subject '{subject_id}', skipping")
            skipped += 1
            continue

        label_val = meta.get(LABEL_FIELD)
        if label_val is None:
            print(f"  [WARN] No label '{LABEL_FIELD}' for '{subject_id}', skipping")
            skipped += 1
            continue

        subject_mods = set(meta.get("modalities", [])).intersection(set(DB_FIELDS))
        if not subject_mods:
            print(f"  [WARN] Subject '{subject_id}' has no matching modalities, skipping")
            skipped += 1
            continue

        # ── 2. iterate over available modalities ───────────────────────────
        subj_probas = []
        for mod in subject_mods:
            mod_id = MOD_CODES[mod]

            # fetch & reassemble chunks
            chunks = list(bin_col.find(
                {"id": subject_id, "kind": mod},
                {"id": 1, "chunk": 1, "chunk_id": 1, "_id": 0},
            ))
            if not chunks:
                print(f"  [WARN] No binary chunks for {subject_id}/{mod}, skipping")
                continue
            chunks.sort(key=lambda c: c["chunk_id"])
            raw = b"".join(c["chunk"] for c in chunks)

            # transform → tensor [1, 1, H, W, D]
            volume = mytransform(raw).float()
            volume = safe_normalize(volume)
            if volume.dim() == 3:
                volume = volume.unsqueeze(0)  # add channel dim
            volume = volume.unsqueeze(0).to(device)  # add batch dim

            modality_tensor = torch.tensor([mod_id], dtype=torch.long).to(device)

            # forward pass
            with torch.no_grad():
                y_hat = model.forward(volume, modality_tensor)
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
                  f"({processed} volume-modality pairs so far)...")
            sys.stdout.flush()

    print(f"\nDone: {processed} volume-modality pairs from "
          f"{len(test_ids) - skipped}/{len(test_ids)} subjects "
          f"({skipped} skipped).")
    return mod_preds, mod_labels, all_preds, all_labels, per_subj_preds, per_subj_labels


def split_halves_report(per_subj_preds, per_subj_labels, fold):
    """
    Simulate DDP rank-averaging: split subjects into two halves
    (interleaved, as DistributedSampler would), compute AUC on each half,
    then compare mean(half0_AUC, half1_AUC) vs pooled AUC.

    Training-logged AUC ≈ mean(rank0_AUC, rank1_AUC).
    Standalone inference AUC = pooled AUC on all subjects.
    If these differ, DDP averaging explains the gap.
    """
    n = len(per_subj_preds)
    preds  = np.array(per_subj_preds)
    labels = np.array(per_subj_labels)

    # Interleaved split (mirrors DistributedSampler: rank0=even, rank1=odd)
    idx0 = np.arange(0, n, 2)   # rank 0: positions 0, 2, 4, ...
    idx1 = np.arange(1, n, 2)   # rank 1: positions 1, 3, 5, ...

    def _auc(idx):
        y, p = labels[idx], preds[idx]
        if len(set(y)) < 2:
            return float("nan")
        return roc_auc_score(y, p)

    auc0    = _auc(idx0)
    auc1    = _auc(idx1)
    avg_auc = (auc0 + auc1) / 2
    pooled  = roc_auc_score(labels, preds) if len(set(labels)) >= 2 else float("nan")

    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  Fold {fold} — DDP Rank-Averaging Simulation")
    print(f"{bar}")
    print(f"  Total subjects  : {n}  (rank 0: {len(idx0)}, rank 1: {len(idx1)})")
    print(f"  Rank 0 AUC      : {auc0:.4f}   (subjects at even positions, n={len(idx0)})")
    print(f"  Rank 1 AUC      : {auc1:.4f}   (subjects at odd  positions, n={len(idx1)})")
    print(f"  Mean(rank AUCs) : {avg_auc:.4f}  ← what training logs reported")
    print(f"  Pooled AUC      : {pooled:.4f}  ← what standalone inference reports")
    print(f"  Gap             : {avg_auc - pooled:+.4f}")
    if avg_auc - pooled > 0.05:
        print(f"  → DDP averaging INFLATES the logged AUC by ~{avg_auc - pooled:.3f}")
    elif abs(avg_auc - pooled) <= 0.02:
        print(f"  → Gap is negligible; DDP averaging is NOT the main cause.")
    else:
        print(f"  → Moderate gap; DDP averaging may contribute but is not the sole cause.")
    print(f"{bar}\n")


def report(mod_preds, mod_labels, all_preds, all_labels, fold):
    print(f"\n{'='*55}")
    print(f"  Fold {fold} — GroupNorm model — Test set AUC")
    print(f"{'='*55}")
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
    if len(all_labels) > 0:
        try:
            overall = roc_auc_score(all_labels, all_preds)
            print(f"  {'overall':<8}  AUC = {overall:.4f}   (n={len(all_labels)})")
        except Exception as e:
            print(f"  {'overall':<8}  AUC = ERROR ({e})  (n={len(all_labels)})")
    print(f"{'='*55}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model.best.pth for this fold")
    parser.add_argument("--test_ids",   required=True,
                        help="Path to test_ids.txt saved during original training")
    parser.add_argument("--db_host",    default="10.245.12.58",
                        help="MongoDB host (default: 10.245.12.58)")
    parser.add_argument("--fold",       default=0, type=int,
                        help="Fold index (for display only)")
    parser.add_argument("--split_halves", action="store_true",
                        help="Simulate DDP rank-averaging: compute AUC on each "
                             "interleaved half of subjects separately, then compare "
                             "mean(rank0_AUC, rank1_AUC) vs pooled AUC. "
                             "Tests whether DDP averaging explains the training-log "
                             "vs standalone-inference AUC gap.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_ids = load_ids(args.test_ids)
    print(f"Loaded {len(test_ids)} test subjects from {args.test_ids}")
    print(f"First 3 IDs: {test_ids[:3]}")

    model = build_model(args.checkpoint, device)

    # ── Sanity check: verify weights and masks loaded non-trivially ───────────
    print("\n--- Weight/mask sanity check ---")
    for name, param in model.named_parameters():
        print(f"  First param  [{name}]: mean={param.data.mean():.6f}, std={param.data.std():.6f}, shape={list(param.data.shape)}")
        break
    for name, buf in model.named_buffers():
        sparsity = 1 - buf.float().mean().item()
        print(f"  First buffer [{name}]: sparsity={sparsity:.2%}, min={buf.float().min():.1f}, max={buf.float().max():.1f}, shape={list(buf.shape)}")
        break
    print("--- end sanity check ---\n")

    mod_preds, mod_labels, all_preds, all_labels, per_subj_preds, per_subj_labels = \
        run_inference(model, test_ids, args.db_host, device)

    report(mod_preds, mod_labels, all_preds, all_labels, args.fold)

    if args.split_halves:
        split_halves_report(per_subj_preds, per_subj_labels, args.fold)


if __name__ == "__main__":
    main()
