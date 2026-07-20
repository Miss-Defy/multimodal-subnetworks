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
#SBATCH -J ukbm_m_lisa
### cd /home/users/elatash1/Work/Joanne_Multimodal/multimodal-subnetworks-main-groupnorm-claude
#SBATCH -D /home/users/elatash1/Work/Joanne_Multimodal/multimodal-subnetworks-main-groupnorm-claude
#SBATCH -o /home/users/elatash1/Work/Joanne_Multimodal/multimodal-subnetworks-main-groupnorm-claude/slurm_scripts/out_Joanne_multimodal_%A_groupnorm_allSubj_sparsity7_10epochs.out 
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

dataset="ukb"
modality="multimodal"
for SPARSITY in 0.7; do
# for SPARSITY in 0.3 0.5 0.7; do
    echo "===== Running sparsity=${SPARSITY} ====="
    python3 train_script_rev.py \
        --config-name new_conf \
        --config-dir conf \
        experiment.experiment_name=${dataset}_${modality}_lz4_sps${SPARSITY}_groupnorm \
        experiment.collections=$dataset \
        experiment.dbfields=[falff,smri,dwi] \
        experiment.metafields=[gender_encoded] \
        model.masked=True \
        model.model_channels=64 \
        model.norm_type=groupnorm \
        model.sparsity=${SPARSITY} \
        hydra.run.dir=/home/users/elatash1/Work/Joanne_Multimodal/multimodal-subnetworks-main-groupnorm-claude/slurm_scripts/hydra_runs \
        paths.logdir=/home/users/elatash1/Work/Joanne_Multimodal/multimodal-subnetworks-main-groupnorm-claude/slurm_scripts/logs \
        experiment.numvolumes=4 \
        experiment.prefetches=16 \
        experiment.num_workers=20 \
        model.snip_batch_size=10 \
        experiment.cv_folds=5 \
        experiment.epochs=10
    echo "===== Done sparsity=${SPARSITY} ====="
done

END=$(date +%s)
echo "Job ended: $(date)"
echo "Elapsed: $(( (END - START) / 3600 ))h $(( ((END - START) % 3600) / 60 ))m $(( (END - START) % 60 ))s"

sleep 10s
echo "Job $SLURM_JOB_ID completed"
