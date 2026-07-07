#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=200g
#SBATCH -p YYYY
#SBATCH -t 1800
##SBATCH --nodelist=XXXX
#SBATCH --gres=gpu:A100:1
#SBATCH -J ukbm_infer_lisa
#SBATCH -o /home/users/elatash1/Work/Joanne_Multimodal/multimodal-subnetworks-main-groupnorm-claude/slurm_scripts/out_infer_%A.out
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

cd /home/users/elatash1/Work/Joanne_Multimodal/multimodal-subnetworks-main-groupnorm-claude

LOGDIR="slurm_scripts/logs/ukb_multimodal_lz4_sps0.5_ukb_('falff', 'smri', 'dwi')_('gender_encoded',)_masked_True_sps_0.5"

echo "===== Fold 0 ====="
python3 infer_modality_auc.py \
    --checkpoint "${LOGDIR}/fold_0/model.best.pth" \
    --test_ids   "${LOGDIR}/fold_0/test_ids.txt" \
    --db_host    10.245.12.58 \
    --fold       0 \
    --split_halves

echo "===== Fold 1 ====="
python3 infer_modality_auc.py \
    --checkpoint "${LOGDIR}/fold_1/model.best.pth" \
    --test_ids   "${LOGDIR}/fold_1/test_ids.txt" \
    --db_host    10.245.12.58 \
    --fold       1 \
    --split_halves

END=$(date +%s)
echo "Job ended: $(date)"
echo "Elapsed: $(( (END - START) / 3600 ))h $(( ((END - START) % 3600) / 60 ))m $(( (END - START) % 60 ))s"
echo "Job $SLURM_JOB_ID completed"
