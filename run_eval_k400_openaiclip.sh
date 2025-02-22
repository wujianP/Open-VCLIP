ROOT=/discobox/wjpeng/code/2024/Open-VCLIP
OUT_DIR=/discobox/wjpeng/ckp/betterCLIP/rebuttal/action_recognition/k400_openaiclip_vitb32

LOAD_CKPT_FILE='openai'
PATCHING_RATIO=1.0

conda activate /discobox/wjpeng/env/openvclip/
cd $ROOT
python -W ignore -u tools/run_net.py \
    --cfg configs/Kinetics/CLIP_vitb32_8x16_STAdapter.yaml \
    --opts DATA.PATH_TO_DATA_DIR $ROOT/label_db/weng_compress_full_splits \
    DATA.PATH_PREFIX /dev/shm/k400 \
    DATA.PATH_LABEL_SEPARATOR , \
    DATA.INDEX_LABEL_MAPPING_FILE $ROOT/label_db/k400-index2cls.json \
    TRAIN.ENABLE False \
    OUTPUT_DIR $OUT_DIR \
    TEST.BATCH_SIZE 480 \
    NUM_GPUS 8 \
    DATA.DECODING_BACKEND "pyav" \
    MODEL.NUM_CLASSES 400 \
    TEST.CUSTOM_LOAD False \
    TEST.CUSTOM_LOAD_FILE $LOAD_CKPT_FILE \
    TEST.SAVE_RESULTS_PATH temp.pyth \
    TEST.NUM_ENSEMBLE_VIEWS 3 \
    TEST.NUM_SPATIAL_CROPS 1 \
    TEST.PATCHING_MODEL False \
    TEST.PATCHING_RATIO $PATCHING_RATIO


