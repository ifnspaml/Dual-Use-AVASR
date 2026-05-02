#!/bin/bash
#SBATCH --job-name=whisper_tiny_en_finetune
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32gb
#SBATCH --time=6-00:00:00
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu06
#SBATCH --gres=gpu:a100:1

# print hostname
srun hostname

## Set the python environment you want to use for your code
PYTHON_VIRTUAL_ENVIRONMENT=avwhisper
CONDA_ROOT=/home/zhengyangli/anaconda3/
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT

cd /home/zhengyangli/work_fast/whisper-flamingo

# srun python -u whisper_ft_audio.py config/audio/audio_en_large.yaml
srun python -u whisper_ft_audio.py /home/zhengyangli/work_fast/whisper-flamingo/config/audio/audio_en_tiny.yaml
