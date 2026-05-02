# Select model checkpoint
# Audio checkpoints
# checkpoint=/home/zhengyangli/work_fast/whisper-flamingo/models/checkpoint/whisper_en_medium_finetune_lrs3_433h_baseline_bs8_gacc1_a6000/step-90000-wer=0.0318-acc=0.9732.ckpt
checkpoint=/home/zhengyangli/work_fast/whisper-flamingo/models/checkpoint/whisper_lrs3_433h_tiny/step-90000-wer=0.0620-acc=0.9457.ckpt
# Audio-visual checkpoints
# checkpoint=


# Select model size
model=tiny.en

# Select modalities
modalities=asr # whisper
# modalities=avsr # whisper-flamingo

# Select av_fusion type
av_fusion="None" # asr - use this for audio only whisper models
# av_fusion="separate" # use this for whisper-flamingo models

# Select whether to use AV-HuBERT encoder
use_av_hubert_encoder=0 # for whisper / asr
# use_av_hubert_encoder=1 # for whisper-flamingo / avsr

noise_fn_test=//beegfs/data/shared/lrs3/noise/musan/tsv/babble/test.tsv # single lrs3 mixture
noise_fn_val=//beegfs/data/shared/lrs3/noise/musan/tsv/babble/valid.tsv


# Select AV-HuBERT checkpoint
av_hubert_ckpt=models/mavhubert_only_weights.pt # multilingual
# av_hubert_ckpt=models/large_noise_pt_noise_ft_433h_only_weights.pt # english

# Specify Paths
# checkpoint_root=models/
# checkpoint_root=models/checkpoint/
checkpoint_path=${checkpoint}
decode_path=decode_whisper_baseline/
fp16=1
av_hubert_path=//beegfs/work_fast/shared/li_shared/

# for ASR only: ignore checkpoint path and use original whisper weights
# use_original_whisper=0
use_original_whisper=1

task=transcribe # ASR
normalizer=fairseq

# for lang in en ar de el es fr it pt ru; do # all langs
# for lang in en es fr it pt; do # high resource langs
# for lang in ar de el ru; do # low resource langs
for split in test valid; do
    if [ "$split" == "test" ]; then
        noise_fn=$noise_fn_test
    else
        noise_fn=$noise_fn_val
    fi
    for lang in en; do
    # for beam_size in 1; do
    for beam_size in 1; do
        # for noise_snr in 0; do
        # for noise_snr in 1000; do
        for noise_snr in -10 -5 0 5 10 1000; do
                echo $modalities $lang $noise_snr $split
                sbatch slurm/whisper_decode.sh $lang \
                                        $model \
                                        $noise_snr \
                                        $noise_fn \
                                        $checkpoint_path \
                                        $beam_size \
                                        $modalities \
                                        $use_av_hubert_encoder \
                                        $av_fusion \
                                        $fp16 \
                                        $decode_path \
                                        $av_hubert_path \
                                        $av_hubert_ckpt \
                                        $task \
                                        $normalizer \
                                        $use_original_whisper \
                                        $split
        done
    done
done
done