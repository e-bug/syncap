#!/bin/bash

DATA_SPLIT="coco_heldout_4_pos_inter"
IMAGE_NAME="image_features.hdf5"
MODEL_NAME="BOTTOM_UP_TOP_DOWN_RANKING_MEAN"
MODEL_SUFF=""
MODEL_ABBR="butr_mean"

PROJ_DIR="$HOME/projects/syncap"
DATA_DIR="/science/image/nlp-datasets/emanuele/data/syncap"
IMGS_DIR="$DATA_DIR/coco2014/images"
EXP_DIR="$PROJ_DIR/experiments/$DATA_SPLIT/${MODEL_ABBR}${MODEL_SUFF}"
CAPS_DIR="/science/image/nlp-datasets/emanuele/data/syncap/compgen/datasets/$DATA_SPLIT"
LOG_DIR="$PROJ_DIR/logs/${DATA_SPLIT}/${MODEL_NAME}${MODEL_SUFF}"
CKPT="/science/image/nlp-datasets/emanuele/checkpoints/syncap/${DATA_SPLIT}/${MODEL_NAME}${MODEL_SUFF}/checkpoint.best.pth.tar"


args="""
	--image-features-filename $IMGS_DIR/$IMAGE_NAME \
  --dataset-splits-dir $CAPS_DIR \
  --checkpoint $CKPT \
	--logging-dir $LOG_DIR \
	--output-path $EXP_DIR \
	--split val \
	--max-caption-len 40 \
	--beam-size 100 \
	--eval-beam-size 5 \
  --re-ranking
"""


source activate /science/image/nlp-datasets/emanuele/envs/syncap

export PYTHONWARNINGS="ignore"

time python $PROJ_DIR/code/eval.py $args

conda deactivate
