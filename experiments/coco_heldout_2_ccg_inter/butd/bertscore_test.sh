#!/bin/bash

DATA_SPLIT="coco_heldout_2_ccg_inter"
MODEL_NAME="BOTTOM_UP_TOP_DOWN"
MODEL_ABBR="butd"
MODEL_SUFF=""

PROJ_DIR="$HOME/projects/syncap"
DATA_DIR="/data/syncap"
ANNS_DIR="$DATA_DIR/coco2014/annotations"
EXP_DIR="$PROJ_DIR/experiments/$DATA_SPLIT/${MODEL_ABBR}${MODEL_SUFF}"

mkdir -p $EXP_DIR/bertscore

source activate /envs/bertscore-captioning

cd $PROJ_DIR/tools/improved-bertscore-for-image-captioning-evaluation/
python match_cand_refs.py \
	--refs_file $ANNS_DIR/captions_val2014.json \
	--cand_file $EXP_DIR/outputs/test.beam_100.json \
	--output_fn $EXP_DIR/bertscore/test.samples.json

python run_metric.py \
	--samples_fn $EXP_DIR/bertscore/test.samples.json \
	--output_name "test" \
	--output_path $EXP_DIR/bertscore \
	> $EXP_DIR/bertscore/test.out
	
conda deactivate
