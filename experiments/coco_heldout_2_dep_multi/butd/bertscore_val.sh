#!/bin/bash

DATA_SPLIT="coco_heldout_2_dep_multi"
MODEL_NAME="BOTTOM_UP_TOP_DOWN"
MODEL_ABBR="butd"
MODEL_SUFF=""

PROJ_DIR="$HOME/projects/syncap"
DATA_DIR="/science/image/nlp-datasets/emanuele/data/syncap"
ANNS_DIR="$DATA_DIR/coco2014/annotations"
EXP_DIR="$PROJ_DIR/experiments/$DATA_SPLIT/${MODEL_ABBR}${MODEL_SUFF}"

mkdir -p $EXP_DIR/bertscore

source activate /science/image/nlp-datasets/emanuele/envs/bertscore-captioning

cd $PROJ_DIR/tools/improved-bertscore-for-image-captioning-evaluation/
python match_cand_refs.py \
	--refs_file $ANNS_DIR/captions_train2014.json \
	--cand_file $EXP_DIR/outputs/val.beam_100.json \
	--output_fn $EXP_DIR/bertscore/val.samples.json

python run_metric.py \
	--samples_fn $EXP_DIR/bertscore/val.samples.json \
	--output_name "val" \
	--output_path $EXP_DIR/bertscore \
	> $EXP_DIR/bertscore/val.out
	
conda deactivate
