#!/bin/bash

DATA_SPLIT="coco_heldout_2_pos_inter"
MODEL_NAME="m2_transformer"
MODEL_ABBR="m2"
MODEL_SUFF=""

PROJ_DIR="$HOME/projects/syncap"
DATA_DIR="$/data/syncap/coco2014"
IMGS_DIR="$DATA_DIR/images/coco_detections.hdf5"
ANNS_DIR="$DATA_DIR/coco2014/annotations"
CAPS_DIR="/data/syncap/compgen/datasets/$DATA_SPLIT"
CKPT_DIR="/checkpoints/syncap/${DATA_SPLIT}/${MODEL_NAME}${MODEL_SUFF}"
LOG_DIR="$PROJ_DIR/logs/${DATA_SPLIT}"
EXP_DIR="$PROJ_DIR/experiments/$DATA_SPLIT/${MODEL_ABBR}${MODEL_SUFF}"
CKPT="/checkpoints/syncap/${DATA_SPLIT}/${MODEL_NAME}${MODEL_SUFF}/m2_transformer_best.pth"

args="""
  --batch_size 20 \
  --features_path $IMGS_DIR \
  --annotation_folder $ANNS_DIR \
  --id_folder $CAPS_DIR \
  --checkpoint $CKPT \
  --output_path $EXP_DIR \
  --split test \
  --max_len 40 \
  --beam_size 100 \
  --eval_beam_size 5 \
	--keep-special-tokens
"""

source activate m2release

time python $PROJ_DIR/code/meshed-memory-transformer/test.py $args

conda deactivate
