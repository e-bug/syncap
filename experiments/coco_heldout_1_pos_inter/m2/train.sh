#!/bin/bash

DATA_SPLIT="coco_heldout_1_pos_inter"
MODEL_NAME="m2_transformer"
MODEL_SUFF=""

PROJ_DIR="$HOME/projects/syncap"
DATA_DIR="$/science/image/nlp-datasets/emanuele/data/syncap/coco2014"
IMGS_DIR="$DATA_DIR/images/coco_detections.hdf5"
ANNS_DIR="$DATA_DIR/coco2014/annotations"
CAPS_DIR="/science/image/nlp-datasets/emanuele/data/syncap/compgen/datasets/${DATA_SPLIT}"
CKPT_DIR="/science/image/nlp-datasets/emanuele/checkpoints/syncap/${DATA_SPLIT}/${MODEL_NAME}${MODEL_SUFF}"
LOG_DIR="$PROJ_DIR/logs/${DATA_SPLIT}" 

mkdir -p $CKPT_DIR $LOG_DIR

train_args="""
  --exp_name ${MODEL_NAME}${MODEL_SUFF} \
  --batch_size 50 \
  --m 40 \
  --head 8 \
  --warmup 10000 \
	--max_len 40 \
  --features_path $IMGS_DIR \
  --annotation_folder $ANNS_DIR \
  --id_folder $CAPS_DIR \
  --checkpoint_path $CKPT_DIR \
  --logs_folder $LOG_DIR \
  --resume_last
"""

source activate m2release

time python $PROJ_DIR/code/meshed-memory-transformer/train.py $train_args

conda deactivate
