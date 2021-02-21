#!/bin/bash

DATA_SPLIT="coco_heldout_2"
MODEL_NAME="BOTTOM_UP_TOP_DOWN"
MODEL_ABBR="butd"
MODEL_SUFF=""

PROJ_DIR="$HOME/projects/syncap"
EXP_DIR="$PROJ_DIR/experiments/$DATA_SPLIT/${MODEL_ABBR}${MODEL_SUFF}"
SPLIT="val"

RES_FN="$EXP_DIR/outputs/${SPLIT}.beam_100.json"
OUT_DIR="$EXP_DIR/diversity"
ANN_FN="$OUT_DIR/annotations.${SPLIT}.json"
STATS_FN="$OUT_DIR/stats.${SPLIT}.json"
GLOBAL_FN="$OUT_DIR/global_recall.${SPLIT}.json"
LOCAL_FN="$OUT_DIR/local_recall.${SPLIT}.json"
NNPP_FN="$OUT_DIR/noun_pp_data.${SPLIT}.json"

mkdir -p $OUT_DIR

args="""
  $RES_FN \
	--annotations_file $ANN_FN \
	--stats_file $STATS_FN \
	--global_coverage_file $GLOBAL_FN \
	--local_coverage_file $LOCAL_FN \
	--noun_pp_file $NNPP_FN
	--data_split $DATA_SPLIT
"""

source activate /envs/syncap

export PYTHONWARNINGS="ignore"

cd $PROJ_DIR/code/MeasureDiversity
python analyze_my_system.py $args

conda deactivate
