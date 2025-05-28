#!/bin/bash

# Common configurations
TASK="PHASES"
ARCH="MViT"
ONLINE=False
NUM_FRAMES=16
SAMPLE_RATE=120 # Stands for 4 seconds
DATASET="GraSP"
TYPE="pytorch"
CHECKPOINT="./model_weights/pretrained_models/k400_16.pyth"
FRAME_DIR="./data/$DATASET/frames"
FRAME_LIST="./data/$DATASET/frame_lists"

export PYTHONPATH="./must:$PYTHONPATH"

# Function to run an experiment
run_experiment() {
    TRAIN_FOLD=$1
    TEST_FOLD=$2
    BATCH_SIZE=$3

    EXP_PREFIX="arch_$ARCH-frames_$NUM_FRAMES-sr_$SAMPLE_RATE-online_$ONLINE"
    EXPERIMENT_NAME=$EXP_PREFIX"/"$TRAIN_FOLD
    CONFIG_PATH="configs/$DATASET/${ARCH}_$TASK.yaml"
    OUTPUT_DIR="outputs/$DATASET/$TASK/$EXPERIMENT_NAME"
    ANNOT_DIR="./data/$DATASET/annotations/$TRAIN_FOLD"
    COCO_ANN_PATH="./data/$DATASET/annotations/$TRAIN_FOLD/$TEST_FOLD_long-term_anns.json"

    mkdir -p $OUTPUT_DIR

    echo "Running experiment: TRAIN=$TRAIN_FOLD, TEST=$TEST_FOLD"

    CUDA_VISIBLE_DEVICES=0,2,3 python -B tools/run_net.py \
    --cfg $CONFIG_PATH \
    NUM_GPUS 3 \
    TRAIN.DATASET $DATASET \
    TEST.DATASET $DATASET \
    ENDOVIS_DATASET.TRAIN_USE_COLOR_AUGMENTATION True \
    TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
    TRAIN.CHECKPOINT_EPOCH_RESET True \
    TRAIN.CHECKPOINT_TYPE $TYPE \
    TEST.ENABLE False \
    TRAIN.ENABLE True \
    DATA.NUM_FRAMES $NUM_FRAMES \
    DATA.SAMPLING_RATE $SAMPLE_RATE \
    DATA.ONLINE $ONLINE \
    ENDOVIS_DATASET.FRAME_DIR $FRAME_DIR \
    ENDOVIS_DATASET.FRAME_LIST_DIR $FRAME_LIST \
    ENDOVIS_DATASET.TRAIN_LISTS "$TRAIN_FOLD.csv" \
    ENDOVIS_DATASET.TEST_LISTS "$TEST_FOLD.csv" \
    ENDOVIS_DATASET.ANNOTATION_DIR $ANNOT_DIR \
    ENDOVIS_DATASET.TEST_COCO_ANNS $COCO_ANN_PATH \
    ENDOVIS_DATASET.TRAIN_GT_BOX_JSON "train_long-term_anns.json" \
    ENDOVIS_DATASET.TEST_GT_BOX_JSON $TEST_FOLD"_long-term_anns.json" \
    TRAIN.BATCH_SIZE $BATCH_SIZE \
    TEST.BATCH_SIZE 69 \
    SOLVER.MAX_EPOCH 20 \
    OUTPUT_DIR $OUTPUT_DIR
}

# Run experiments
run_experiment "fold1" "fold2" 21
run_experiment "fold2" "fold1" 21
run_experiment "train" "test" 21
