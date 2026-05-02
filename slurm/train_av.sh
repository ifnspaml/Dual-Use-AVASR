#!/bin/bash
#SBATCH --job-name=whisper-release-test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32gb
#SBATCH --time=6-00:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu06

# print hostname
srun hostname

## Set the python environment you want to use for your code
PYTHON_VIRTUAL_ENVIRONMENT=avwhisper
CONDA_ROOT=/home/zhengyangli/anaconda3/
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT

cd /home/zhengyangli/work_fast/whisper-flamingo

# srun python -u whisper_ft_av.py config/visual/v_en_large.yaml
srun python -u whisper_ft_av.py config/audio-visual/lrs3/av_en_tiny_short_training_dual_use.yaml
#config/audio-visual/av_en_base_eff_short_training_moe_distil.yaml
