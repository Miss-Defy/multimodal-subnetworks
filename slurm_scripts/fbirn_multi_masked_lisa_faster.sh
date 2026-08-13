#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=120g
#SBATCH -p qTRDGPUH
#SBATCH -t 7200
##SBATCH --nodelist=arctrddgxa004
#SBATCH --gres=gpu:A100:1
### SBATCH --gres=gpu:A100:1
#SBATCH -J fbirn_multimodal_d_gn
### cd /home/users/elatash1/Work/Joanne_Multimodal/multimodal-subnetworks-main-groupnorm-claude
#SBATCH -D /home/users/elatash1/Work/Joanne_Multimodal/multimodal-subnetworks-main-groupnorm-claude
#SBATCH -o /home/users/elatash1/Work/Joanne_Multimodal/multimodal-subnetworks-main-groupnorm-claude/slurm_scripts/out_20260804_%A_Joanne_60ep_5cv_d_multimodal_groupnorm_AdamW_channels.out
#SBATCH -A psy53c17
sleep 10s
echo "Running on host: $HOSTNAME" >&2
echo "Job ID: $SLURM_JOB_ID" >&2
echo "TMPDIR is: $TMPDIR" >&2
export TMPDIR=/tmp
source /data/users2/jwardell1/miniconda3/bin/activate mmsn312
echo "Using python from: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
# export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

export WANDB_MODE=disabled

export HYDRA_RUN_DIR=/home/users/elatash1/Work/Joanne_Multimodal/multimodal-subnetworks-main-groupnorm-claude/slurm_scripts/hydra_runs
mkdir -p $HYDRA_RUN_DIR

# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_DEBUG=INFO
ulimit -n 65536
# export TQDM_MININTERVAL=30

START=$(date +%s)
echo "Job started: $(date)"

dataset="fbirn"
modality="multimodal"
# modality="dwi"
NormType="groupnorm"
# for SPARSITY in 0.7; do
# for SPARSITY in 0.3 0.5 0.7; do
# experiment.dbfields=[falff,smri,dwi]
for SPARSITY in 0.0; do
    echo "===== Running sparsity=${SPARSITY} ====="
    python3 train_script_rev.py \
        --config-name new_conf \
        --config-dir conf \
        experiment.experiment_name=${dataset}_${modality}_lz4_sps_60epochs_10cv_AdamW_${SPARSITY}_${NormType} \
        experiment.collections=$dataset \
        experiment.dbfields=[falff,smri,dwi] \
        experiment.metafields=[gender_encoded] \
        model.masked=False \
        model.model_channels=64 \
        model.norm_type=${NormType} \
        model.sparsity=${SPARSITY} \
        hydra.run.dir=/home/users/elatash1/Work/Joanne_Multimodal/multimodal-subnetworks-main-groupnorm-claude/slurm_scripts/hydra_runs \
        paths.logdir=/home/users/elatash1/Work/Joanne_Multimodal/multimodal-subnetworks-main-groupnorm-claude/slurm_scripts/logs \
        experiment.numvolumes=4 \
        experiment.prefetches=16 \
        experiment.num_workers=20 \
        model.snip_batch_size=10 \
        experiment.cv_folds=5 \
        experiment.epochs=60 \
        experiment.train_num_workers=18 \
        experiment.train_prefetches=2 \
        experiment.train_prefetch_factor=4 \
        experiment.eval_num_workers=12 \
        experiment.eval_prefetches=2 \
        experiment.eval_prefetch_factor=4
    echo "===== Done sparsity=${SPARSITY} ====="
done

END=$(date +%s)
echo "Job ended: $(date)"
echo "Elapsed: $(( (END - START) / 3600 ))h $(( ((END - START) % 3600) / 60 ))m $(( (END - START) % 60 ))s"

sleep 10s
echo "Job $SLURM_JOB_ID completed"
