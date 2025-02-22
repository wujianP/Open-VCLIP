ROOT=/discobox/wjpeng/code/2024/Open-VCLIP
OUT_DIR=/discobox/wjpeng/ckp/betterCLIP/rebuttal/action_recognition/ucf101_ourclip_vitb32

LOAD_CKPT_FILE='/DDN_ROOT/wjpeng/ckp/betterCLIP/v2/vitb32-openai_ep10-step100_lr1e-6-warm800_common-laion400m-bs256_blip-y_llama-n_extra-wt0.2-mer-bs8-hn2_sep-com-extra/checkpoints/epoch_10.pt'
PATCHING_RATIO=1.0

conda activate /discobox/wjpeng/env/openvclip/
cd $ROOT
python -W ignore -u tools/run_net.py \
    --cfg configs/Kinetics/CLIP_vitb32_8x16_STAdapter.yaml \
    --opts DATA.PATH_TO_DATA_DIR $ROOT/zs_label_db/ucf101_full \
    DATA.PATH_PREFIX /dev/shm/ucf/UCF-101 \
    DATA.PATH_LABEL_SEPARATOR , \
    DATA.INDEX_LABEL_MAPPING_FILE $ROOT/zs_label_db/ucf101-index2cls.json \
    TRAIN.ENABLE False \
    OUTPUT_DIR $OUT_DIR \
    TEST.BATCH_SIZE 480 \
    NUM_GPUS 8 \
    DATA.DECODING_BACKEND "pyav" \
    MODEL.NUM_CLASSES 101 \
    TEST.CUSTOM_LOAD False \
    TEST.CUSTOM_LOAD_FILE $LOAD_CKPT_FILE \
    TEST.SAVE_RESULTS_PATH temp.pyth \
    TEST.NUM_ENSEMBLE_VIEWS 3 \
    TEST.NUM_SPATIAL_CROPS 1 \
    TEST.PATCHING_MODEL False \
    TEST.PATCHING_RATIO $PATCHING_RATIO


