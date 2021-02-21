#!/bin/bash

DATA_SPLIT="coco_heldout_4"
MODEL_NAME="BOTTOM_UP_TOP_DOWN_RANKING"
MODEL_SUFF=""
MODEL_ABBR="butr"

PROJ_DIR="$HOME/projects/syncap"
DATA_DIR="/science/image/nlp-datasets/emanuele/data/syncap"
ANNS_DIR="$DATA_DIR/coco2014/annotations"
EXP_DIR="$PROJ_DIR/experiments/$DATA_SPLIT/${MODEL_ABBR}${MODEL_SUFF}"

mkdir -p $EXP_DIR/bertscore

source activate /science/image/nlp-datasets/emanuele/envs/bertscore-captioning

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
