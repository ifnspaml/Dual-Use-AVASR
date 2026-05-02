#!/bin/bash
#SBATCH --job-name=decode
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32gb
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1080:1

## Set the python environment you want to use for your code
PYTHON_VIRTUAL_ENVIRONMENT=avwhisper
CONDA_ROOT=/home/zhengyangli/anaconda3/
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT

cd /home/zhengyangli/work_fast/whisper-flamingo

srun hostname
echo $CUDA_VISIBLE_DEVICES
echo $1
echo $2
echo $3
echo $4
echo $5
echo $6
echo $7
echo $8
echo $9
echo ${10}
echo ${11}
echo ${12}
echo ${13}
echo ${14}
echo ${15}
echo ${16}
echo ${17}

python -u whisper_decode_video.py --lang $1 \
                                --model-type $2 \
                                --noise-snr $3 \
                                --noise-fn $4 \
                                --checkpoint-path $5 \
                                --beam-size $6 \
                                --modalities $7 \
                                --use_av_hubert_encoder $8 \
                                --av_fusion $9 \
                                --fp16 ${10} \
                                --decode-path ${11} \
                                --av-hubert-path ${12} \
                                --av-hubert-ckpt ${13} \
                                --task ${14} \
                                --normalizer ${15} \
                                --use-original-whisper ${16} \
                                --split ${17} \
                                $([ -n "${18}" ] && echo "--fusion-mode ${18}")