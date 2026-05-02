# Check if all required arguments are provided
if [ $# -lt 4 ]; then
    echo "Usage: $0 <checkpoint_path> <use_av_hubert_encoder> <fusion_spec> <noisex_test>"
    echo "  fusion_spec (3rd arg):"
    echo "    Legacy (old checkpoints): effusion | separate | effusion-concat"
    echo "    New (fusion_mode): baseline | early_fusion | dual_use | dual_use_concat"
    echo "Example (legacy): $0 checkpoint/.../step-120000.ckpt 1 effusion 0"
    echo "Example (new dual_use): $0 checkpoint/.../step-20000.ckpt 1 dual_use 0"
    exit 1
fi

# Use the provided arguments
checkpoint=$1
use_av_hubert_encoder=$2
fusion_spec=$3
noisex_test=$4

# 3rd arg: legacy av_fusion (old checkpoints) OR new fusion_mode
case "$fusion_spec" in
    baseline|early_fusion|dual_use|dual_use_concat)
        fusion_mode=$fusion_spec
        av_fusion=effusion
        ;;
    effusion|separate|effusion-concat)
        fusion_mode=
        av_fusion=$fusion_spec
        ;;
    *)
        echo "Error: fusion_spec must be legacy (effusion|separate|effusion-concat) or fusion_mode (baseline|early_fusion|dual_use|dual_use_concat), got: $fusion_spec"
        exit 1
        ;;
esac

echo "Using checkpoint: $checkpoint"
echo "Using AV-HuBERT encoder: $use_av_hubert_encoder"
echo "Using fusion_spec: $fusion_spec (av_fusion=$av_fusion${fusion_mode:+, fusion_mode=$fusion_mode})"
echo "Using noisex_test: $noisex_test"

# Select model size
# model=large-v2
# model=small.en
# model=base.en
model=tiny.en

# Select modalities
# modalities=asr # whisper
modalities=avsr # whisper-flamingo


# Select multilingual or EN babble noise
# noise_fn=noise/babble/muavic/test.tsv # multilingual babble noise 
noise_fn_test=//beegfs/data/shared/lrs3/noise/musan/tsv/babble/test.tsv # single lrs3 mixture
noise_fn_val=//beegfs/data/shared/lrs3/noise/musan/tsv/babble/valid.tsv # single lrs3 mixture
noise_fn_noisx=//beegfs/work_fast/shared/li_shared/noisex/noisex_test.tsv # babble noise from the noisex dataset


# Select AV-HuBERT checkpoint
av_hubert_ckpt=models/large_vox_iter5.pt # english

checkpoint_root=models/
# Specify Paths
if [ "$noisex_test" == "1" ]; then
    decode_path=decode/noisex_test/
else
    decode_path=decode/
fi
# checkpoint_root=models/checkpoint/
checkpoint_path=${checkpoint_root}${checkpoint}
fp16=1
av_hubert_path=//beegfs/work_fast/zhengyangli/av_hubert/avhubert

# for ASR only: ignore checkpoint path and use original whisper weights
use_original_whisper=0
# use_original_whisper=1

task=transcribe # ASR
normalizer=fairseq

for split in test; do
    # Select noise function
    if [ "$noisex_test" == "0" ]; then
        if [ "$split" == "test" ]; then
            noise_fn=$noise_fn_test
        else
            noise_fn=$noise_fn_val
        fi
    else
        noise_fn=$noise_fn_noisx
    fi

    # Select language
    for lang in en; do # all langs
    for beam_size in 1; do
        for noise_snr in 0; do
        # for noise_snr in 1000; do
        # for noise_snr in -10 -5 0 5 10; do
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
                                $split \
                                $fusion_mode
        done
    done
done
done

