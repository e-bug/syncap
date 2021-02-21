#!/bin/bash

DATA_SPLIT="coco_heldout_1_chunk_plan"
IMAGE_NAME="image_features.hdf5"
MODEL_NAME="BOTTOM_UP_TOP_DOWN"
MODEL_ABBR="butd"
MODEL_SUFF=""

PROJ_DIR="$HOME/projects/syncap"
DATA_DIR="/data/syncap"
IMGS_DIR="$DATA_DIR/coco2014/images"
CAPS_DIR="/data/syncap/compgen/datasets/$DATA_SPLIT"
CKPT_DIR="/checkpoints/syncap/${DATA_SPLIT}/${MODEL_NAME}${MODEL_SUFF}"
LOG_DIR="$PROJ_DIR/logs/${DATA_SPLIT}/${MODEL_NAME}${MODEL_SUFF}"

mkdir -p $CKPT_DIR $LOG_DIR

train_args="""
	--image-features-filename $IMGS_DIR/$IMAGE_NAME \
  --dataset-splits-dir $CAPS_DIR \
  --checkpoints-dir $CKPT_DIR \
	--logging-dir $LOG_DIR \
	--objective GENERATION \
  --batch-size 50 \
	--max-epochs 120 \
	--epochs-early-stopping 5 \
	--max-caption-len 40 \
"""

model_args="""
butd
"""

source activate /envs/syncap

export PYTHONWARNINGS="ignore"

time python $PROJ_DIR/code/train.py $train_args $model_args

conda deactivate
