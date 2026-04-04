#!/bin/bash
# UCVLA Stage 1: train per-user bias vectors on frozen RDT-2 FM backbone.
# Only user_bias (nn.Embedding) and bias_proj (nn.Linear) are trainable.
# Checkpoint saves ucvla_weights.pt — base RDT weights are NOT saved (unchanged).

export CFLAGS="-I/usr/include"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"

CUR_TIME=$(date +%Y%m%d_%H%M%S)

WANDB_PROJECT="rdt2-ucvla-stage1"
OUTPUT_DIR="outputs/ucvla/stage1/"

mkdir -p "./logs/${WANDB_PROJECT}"
LOGGING_DIR="./logs/${WANDB_PROJECT}"
LOGGING_FILE="${LOGGING_DIR}/${CUR_TIME}.log"

# Point to the finetuned RDT checkpoint from finetune_rdt.sh
RDT_FINETUNED_CHECKPOINT="outputs/rdt/rdt2-action-expert/checkpoint-40000"
VISION_LANGUAGE_MODEL_NAME_OR_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
WDS_CONFIG_FILE="configs/datasets/mug_handover.yaml"

N_USERS=3
D_BIAS=64
TRAIN_BATCH_SIZE=32
NUM_GPUS=2  # be considerate of other users

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

PYTHONPATH="/home/ywc/Codes/RDT2" uv run accelerate launch \
    --num_processes=$NUM_GPUS \
    rdt/main_ucvla_stage1.py \
    --deepspeed="scripts/zero1.json" \
    --config_path="./configs/rdt/post_train.yaml" \
    --pretrained_vision_language_model_name_or_path=$VISION_LANGUAGE_MODEL_NAME_OR_PATH \
    --pretrained_model_name_or_path=$RDT_FINETUNED_CHECKPOINT \
    --n_users=$N_USERS \
    --d_bias=$D_BIAS \
    --output_dir=$OUTPUT_DIR \
    --webdataset_config=$WDS_CONFIG_FILE \
    --train_batch_size=$TRAIN_BATCH_SIZE \
    --max_train_steps=50000 \
    --checkpointing_period=5000 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=100 \
    --learning_rate=1e-3 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=4 \
    --report_to=wandb 2>&1 | tee -a "$LOGGING_FILE"
