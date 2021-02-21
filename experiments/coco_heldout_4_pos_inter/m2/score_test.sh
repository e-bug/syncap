#!/bin/bash

DATA_SPLIT="coco_heldout_4_pos_inter"
MODEL_NAME="m2_transformer"
MODEL_ABBR="m2"
MODEL_SUFF=""
SPLIT="test"
heldout_pairs="black_bird small_dog white_boat big_truck eat_horse stand_child"

PROJ_DIR="$HOME/projects/syncap"
DATA_DIR="/data/syncap"
EXP_DIR="$PROJ_DIR/experiments/$DATA_SPLIT/${MODEL_ABBR}${MODEL_SUFF}"

RES_FN="$EXP_DIR/outputs/${SPLIT}.beam_100.json"
TOP_FN="$EXP_DIR/outputs/${SPLIT}.beam_100.top_5.json"
TGT_FN="$EXP_DIR/outputs/${SPLIT}.targets.json"
OUT_DIR="$EXP_DIR/results"

ANNS_DIR="$DATA_DIR/coco2014/annotations"
OCC_DIR="$DATA_DIR/compgen/concepts/occurrences"

args="""
  --results-fn $RES_FN \
	--top-results-fn $TOP_FN \
	--targets-fn $TGT_FN \
	--output-dir $OUT_DIR \
	--metrics coco recall \
	--annotations-dir $ANN_DIR \
	--annotations-split val2014 \
	--occurrences-dir $OCC_DIR \
	--heldout-pairs $heldout_pairs
"""

source activate /envs/syncap

export PYTHONWARNINGS="ignore"

python $PROJ_DIR/code/score.py $args

conda deactivate
