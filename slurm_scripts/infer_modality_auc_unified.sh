#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=200g
#SBATCH -p qTRDGPUL
#SBATCH -t 1800
#SBATCH --gres=gpu:A100:1
#SBATCH -J infer_unified
#SBATCH -D /home/users/elatash1/Work/Joanne_Multimodal/multimodal-subnetworks-main-groupnorm-claude
#SBATCH -o /home/users/elatash1/Work/Joanne_Multimodal/multimodal-subnetworks-main-groupnorm-claude/slurm_scripts/out_infer_unified_%A.out
#SBATCH -A psy53c17

sleep 10s
echo "Running on host: $HOSTNAME" >&2
echo "Job ID: $SLURM_JOB_ID" >&2
export TMPDIR=/tmp
source /data/users2/jwardell1/miniconda3/bin/activate mmsn312
echo "Using python from: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled

START=$(date +%s)
echo "Job started: $(date)"

BASEDIR="/home/users/elatash1/Work/Joanne_Multimodal/multimodal-subnetworks-main-groupnorm-claude"
LOGBASE="${BASEDIR}/slurm_scripts/logs/Copy_to_server"
DB_HOST="10.245.12.58"
COLLECTION="fbirn"

# ── Edit these to match the experiment_name used in training ──────────────────
# The full log dir is built as:
#   {LOGBASE}/{EXPERIMENT_NAME}_{COLLECTION}_('falff', 'smri', 'dwi')_('gender_encoded',)_masked_{MASKED}_sps_{SPS}
#
# For SPARSE runs (masked=True, sps=0.7):
EXP_SPARSE_GN="fbirn_multimodal_lz4_sps_60epochs_10cv_0.7_groupnorm"
EXP_SPARSE_BN="fbirn_multimodal_lz4_sps_60epochs_10cv_0.7_batchnorm"
#
# For DENSE runs (masked=False, sps=0.0):
EXP_DENSE_GN="fbirn_multimodal_lz4_sps_60epochs_10cv_0.0_groupnorm"
EXP_DENSE_BN="fbirn_multimodal_lz4_sps_60epochs_10cv_0.0_batchnorm"
# ─────────────────────────────────────────────────────────────────────────────

DBFIELDS="('falff', 'smri', 'dwi')"
METAFIELDS="('gender_encoded',)"

cd "${BASEDIR}"

run_folds() {
    local NORM_TYPE=$1
    local MASKED=$2      # True or False
    local SPARSITY=$3    # 0.7 or 0.0
    local LOGDIR=$4

    echo ""
    echo "######################################################################"
    echo "  norm_type=${NORM_TYPE}  masked=${MASKED}  sparsity=${SPARSITY}"
    echo "  logdir=${LOGDIR}"
    echo "######################################################################"

    for FOLD in 0 1 2 3 4 5 6 7 8 9; do
        CKPT="${LOGDIR}/fold_${FOLD}/model.best.pth"
        TEST_IDS="${LOGDIR}/fold_${FOLD}/test_ids.txt"

        if [ ! -f "$CKPT" ]; then
            echo "  [SKIP] Checkpoint not found: $CKPT"
            continue
        fi
        if [ ! -f "$TEST_IDS" ]; then
            echo "  [SKIP] test_ids.txt not found: $TEST_IDS"
            continue
        fi

        echo ""
        echo "===== ${NORM_TYPE} masked=${MASKED} — Fold ${FOLD} ====="
        python3 infer_modality_auc_unified.py \
            --checkpoint  "$CKPT" \
            --test_ids    "$TEST_IDS" \
            --norm_type   "$NORM_TYPE" \
            --masked      "$MASKED" \
            --sparsity    "$SPARSITY" \
            --db_host     "$DB_HOST" \
            --collection  "$COLLECTION" \
            --fold        "$FOLD"
    done
}

# ── Sparse runs (masked=True, sparsity=0.7) ───────────────────────────────────
LOGDIR_SPARSE_GN="${LOGBASE}/${EXP_SPARSE_GN}_${COLLECTION}_${DBFIELDS}_${METAFIELDS}_masked_True_sps_0.7"
LOGDIR_SPARSE_BN="${LOGBASE}/${EXP_SPARSE_BN}_${COLLECTION}_${DBFIELDS}_${METAFIELDS}_masked_True_sps_0.7"

run_folds "groupnorm" "True" "0.7" "${LOGDIR_SPARSE_GN}"
run_folds "batchnorm" "True" "0.7" "${LOGDIR_SPARSE_BN}"

# ── Dense runs (masked=False, sparsity=0.0) ───────────────────────────────────
LOGDIR_DENSE_GN="${LOGBASE}/${EXP_DENSE_GN}_${COLLECTION}_${DBFIELDS}_${METAFIELDS}_masked_False_sps_0.0"
LOGDIR_DENSE_BN="${LOGBASE}/${EXP_DENSE_BN}_${COLLECTION}_${DBFIELDS}_${METAFIELDS}_masked_False_sps_0.0"

run_folds "groupnorm" "False" "0.0" "${LOGDIR_DENSE_GN}"
run_folds "batchnorm" "False" "0.0" "${LOGDIR_DENSE_BN}"

END=$(date +%s)
echo ""
echo "Job ended: $(date)"
echo "Elapsed: $(( (END - START) / 3600 ))h $(( ((END - START) % 3600) / 60 ))m $(( (END - START) % 60 ))s"
echo "Job $SLURM_JOB_ID completed"
