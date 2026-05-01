#!/bin/bash

# ==== Settings ====
GPU_ID=0
DATE=$(date +%m%d) # auto complete
SEED_START=0
SEED_END=400
SEED_STEP=100
BACKBONE=gru

# ==== Tasks ====
tasks=(
        "memorymaze_9x9"
        "memorymaze_11x11"
        "memorymaze_13x13"
        "memorymaze_15x15"
)

# ==== Loop ====
for task in "${tasks[@]}"
do
    for seed in $(seq $SEED_START $SEED_STEP $SEED_END)
    do
        CUDA_VISIBLE_DEVICES=$GPU_ID MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU_ID python train.py \
            env=memorymaze \
            env.task=$task \
            logdir=logdir/${DATE}_${BACKBONE}_${task#memorymaze_}_$seed \
            model.compile=True \
            device=cuda:0 \
            buffer.storage_device=cuda:0 \
            backbone=${BACKBONE} \
            seed=$seed
    done
done
