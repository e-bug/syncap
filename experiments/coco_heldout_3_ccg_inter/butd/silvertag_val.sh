#!/bin/bash

DATA_SPLIT="coco_heldout_3_ccg_inter"
MODEL_NAME="BOTTOM_UP_TOP_DOWN"
MODEL_ABBR="butd"
MODEL_SUFF=""
SPLIT="val"

PROJ_DIR="$HOME/projects/syncap"
DATA_DIR="/science/image/nlp-datasets/emanuele/data/syncap"
EXP_DIR="$PROJ_DIR/experiments/$DATA_SPLIT/${MODEL_ABBR}${MODEL_SUFF}"

TOP_FN="$EXP_DIR/outputs/${SPLIT}.beam_100.top_5.json"
OUT_DIR="$EXP_DIR/results"

args="""
  --results-fn $TOP_FN \
	--output-dir $OUT_DIR
	--syntax-type ccg
"""

source activate /science/image/nlp-datasets/emanuele/envs/syncap

export PYTHONWARNINGS="ignore"

python $PROJ_DIR/code/tag_results.py $args

conda deactivate
