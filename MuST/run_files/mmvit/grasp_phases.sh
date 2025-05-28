#!/bin/bash

# Common configurations
TASK="PHASES"
ARCH="MMViT"
ONLINE=False
NUM_FRAMES=16
SAMPLE_RATE='30,60,120,240'
DATASET="GraSP"
TYPE="pytorch"
FRAME_DIR="./data/$DATASET/frames"
FRAME_LIST="./data/$DATASET/frame_lists"

export PYTHONPATH="./must:$PYTHONPATH"

# Function to run an experiment
run_experiment() {
    TRAIN_FOLD=$1
    TEST_FOLD=$2
    BATCH_SIZE=$3

    EXP_PREFIX="arch_$ARCH-frames_$NUM_FRAMES-sr_$SAMPLE_RATE-online_$ONLINE"
    EXPERIMENT_NAME=$EXP_PREFIX"/$TRAIN_FOLD"
    CONFIG_PATH="configs/$DATASET/${ARCH}_$TASK.yaml"
    OUTPUT_DIR="outputs/$DATASET/$TASK/$EXPERIMENT_NAME"
    ANNOT_DIR="./data/$DATASET/annotations/$TRAIN_FOLD"
    COCO_ANN_PATH="./data/$DATASET/annotations/$TRAIN_FOLD/${TEST_FOLD}_long-term_anns.json"
    # Note: The MViT pretrained model on the same dataset.
    CHECKPOINT="./model_weights/mvit/GraSP/$TRAIN_FOLD/checkpoint_best_phases.pyth"

    mkdir -p $OUTPUT_DIR

    echo "Running experiment: TRAIN=$TRAIN_FOLD, TEST=$TEST_FOLD"

    CUDA_VISIBLE_DEVICES=0,2,3 python -B tools/run_net.py \
    --cfg $CONFIG_PATH \
    NUM_GPUS 3 \
    TRAIN.DATASET "Graspms" \
    TEST.DATASET "Graspms" \
    DATA.MULTI_SAMPLING_RATE $SAMPLE_RATE \
    DATA.NUM_FRAMES $NUM_FRAMES \
    DATA.ONLINE $ONLINE \
    TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
    TRAIN.CHECKPOINT_EPOCH_RESET True \
    TRAIN.CHECKPOINT_TYPE $TYPE \
    TEST.ENABLE False \
    TRAIN.ENABLE True \
    SOLVER.WARMUP_EPOCHS 0.0 \
    DATA.FIXED_RESIZE True \
    SOLVER.BASE_LR 0.0001 \
    SOLVER.WARMUP_START_LR 0.000125 \
    SOLVER.COSINE_END_LR 0.00001 \
    SOLVER.EARLY_STOPPING 5 \
    ENDOVIS_DATASET.FRAME_DIR $FRAME_DIR \
    ENDOVIS_DATASET.FRAME_LIST_DIR $FRAME_LIST \
    ENDOVIS_DATASET.TRAIN_LISTS "$TRAIN_FOLD.csv" \
    ENDOVIS_DATASET.TEST_LISTS "$TEST_FOLD.csv" \
    ENDOVIS_DATASET.ANNOTATION_DIR $ANNOT_DIR \
    ENDOVIS_DATASET.TEST_COCO_ANNS $COCO_ANN_PATH \
    ENDOVIS_DATASET.TRAIN_GT_BOX_JSON "$TRAIN_FOLD"_long-term_anns.json \
    ENDOVIS_DATASET.TEST_GT_BOX_JSON "$TEST_FOLD"_long-term_anns.json \
    TRAIN.BATCH_SIZE $BATCH_SIZE \
    TEST.BATCH_SIZE $BATCH_SIZE \
    OUTPUT_DIR $OUTPUT_DIR
}

# Run experiments
run_experiment "fold1" "fold2" 18
run_experiment "fold2" "fold1" 18
run_experiment "train" "test" 18 