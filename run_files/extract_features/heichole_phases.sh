#!/bin/bash

# Common configurations
TASK="PHASES"
ARCH="MMViT"
ONLINE=True
NUM_FRAMES=24
SAMPLE_RATE='1,2,4,6'
DATASET="heichole"
TYPE="pytorch"
FRAME_DIR="./data/$DATASET/frames"
FRAME_LIST="./data/$DATASET/frame_lists"
MVIT_FEATS_ENABLE=True

export PYTHONPATH="./must:$PYTHONPATH"

# Function to run feature extraction experiment
run_feature_experiment() {
    TRAIN_FOLD=$1
    TEST_FOLD=$2
    BATCH_SIZE=$3
    CUDA_DEVICES=$4

    EXP_PREFIX="arch_$ARCH-frames_$NUM_FRAMES-sr_$SAMPLE_RATE-online_$ONLINE"
    EXPERIMENT_NAME=$EXP_PREFIX"/$TRAIN_FOLD"
    CONFIG_PATH="configs/$DATASET/${ARCH}_$TASK.yaml"
    OUTPUT_DIR="outputs/$DATASET/$TASK/$EXPERIMENT_NAME"
    ANNOT_DIR="./data/$DATASET/annotations/$TRAIN_FOLD"
    COCO_ANN_PATH="./data/$DATASET/annotations/$TRAIN_FOLD/${TEST_FOLD}_long-term_anns.json"
    CHECKPOINT="./model_weights/multiterm_frame_encoder/heichole/train/checkpoint_best_phases.pyth"
    
    # Features path
    MVIT_FEATS_PATH="./data/"$DATASET"/frames_features/$TRAIN_FOLD"

    mkdir -p $OUTPUT_DIR

    echo "Running feature extraction: TRAIN=$TRAIN_FOLD, TEST=$TEST_FOLD"

    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -B tools/run_net.py \
    --cfg $CONFIG_PATH \
    NUM_GPUS $(echo $CUDA_DEVICES | tr ',' '\n' | wc -l) \
    TRAIN.DATASET "heicholems" \
    TEST.DATASET "heicholems" \
    DATA.MULTI_SAMPLING_RATE $SAMPLE_RATE \
    DATA.NUM_FRAMES $NUM_FRAMES \
    DATA.ONLINE $ONLINE \
    TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
    TRAIN.CHECKPOINT_EPOCH_RESET True \
    TRAIN.CHECKPOINT_TYPE $TYPE \
    TEST.ENABLE True \
    TRAIN.ENABLE False \
    ENDOVIS_DATASET.FRAME_DIR $FRAME_DIR \
    ENDOVIS_DATASET.FRAME_LIST_DIR $FRAME_LIST \
    ENDOVIS_DATASET.TRAIN_LISTS "$TRAIN_FOLD.csv" \
    ENDOVIS_DATASET.TEST_LISTS "$TEST_FOLD.csv" \
    ENDOVIS_DATASET.ANNOTATION_DIR $ANNOT_DIR \
    ENDOVIS_DATASET.TEST_COCO_ANNS $COCO_ANN_PATH \
    ENDOVIS_DATASET.TRAIN_GT_BOX_JSON "train_long-term_anns.json" \
    ENDOVIS_DATASET.TEST_GT_BOX_JSON "$TEST_FOLD"_long-term_anns.json \
    MVIT_FEATS.PATH $MVIT_FEATS_PATH \
    MVIT_FEATS.ENABLE $MVIT_FEATS_ENABLE \
    TRAIN.BATCH_SIZE $BATCH_SIZE \
    TEST.BATCH_SIZE $BATCH_SIZE \
    OUTPUT_DIR "log_feats_$OUTPUT_DIR"
}

# Run feature extraction experiments
run_feature_experiment "train" "test" 12 "0,2,3"
run_feature_experiment "train" "train" 12 "0,2,3"