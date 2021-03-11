#!/bin/bash

DATA_INDIR="/science/image/nlp-datasets/emanuele/data/syncap/coco2014"
IMG_INDIR="$DATA_INDIR/images"
CAP_INDIR="$DATA_INDIR/captions"
ANN_INDIR="$CAP_INDIR/annotations_trainval2014"
KARPATHY_INDIR="$CAP_INDIR/karpathy_splits"

PROJ_DIR="$HOME/projects/syncap"
OUTDIR="$PROJ_DIR/data/preparation"

CONCEPTS_DIR="$PROJ_DIR/data/concepts"
SYNONYMS_DIR="$CONCEPTS_DIR/synonyms"
OCCURRENCES_DIR="$CONCEPTS_DIR/occurrences"
concept_pairs="""
  black,cat,adj-noun    big,bird,adj-noun    red,bus,adj-noun
  small,plane,adj-noun  eat,man,verb-noun    lie,woman,verb-noun
	white,truck,adj-noun  small,cat,adj-noun   brown,dog,adj-noun
	big,plane,adj-noun    ride,woman,verb-noun fly,bird,verb-noun
	white,horse,adj-noun  big,cat,adj-noun     blue,bus,adj-noun
	small,table,adj-noun  hold,child,verb-noun stand,bird,verb-noun
	black,bird,adj-noun   small,dog,adj-noun   white,boat,adj-noun
	stand,child,verb-noun big,truck,adj-noun   eat,horse,verb-noun
"""

DATASETS_DIR="$PROJ_DIR/data/coco2014/captions"
ENCODED_DIR="$PROJ_DIR/data/datasets/encoded_datasets"

export CUDA_VISIBLE_DEVICES=0
export PYTHONWARNINGS="ignore"

mkdir -p $OUTDIR $OCCURRENCES_DIR $DATASETS_DIR

# Activate environment
source activate /science/image/nlp-datasets/emanuele/envs/syncap

# ==================================================================================================================== #
# Preprocess
# ==================================================================================================================== #
python $PROJ_DIR/code/prepare_data.py \
	--ann-dir $ANN_INDIR --img-dir $IMG_INDIR \
	--karpathy-dir $KARPATHY_INDIR \
	--output-dir $OUTDIR

python $PROJ_DIR/code/prepare_data.py \
	--img-dir $IMG_INDIR/trainval_36 \
	--output-dir $PROJ_DIR/data/coco2014/images

# ==================================================================================================================== #
# Syntax
# ==================================================================================================================== #
python $PROJ_DIR/code/prepare_data.py \
	--output-dir $OUTDIR \
	--syntax-tags ccg #lemma pos dep head chunk

# ==================================================================================================================== #
# Composition
# ==================================================================================================================== #
python $PROJ_DIR/code/prepare_data.py \
  --output-dir $OUTDIR \
	--synonyms-dir $SYNONYMS_DIR \
	--occurrences-dir $OCCURRENCES_DIR \
	--concept-pairs $concept_pairs

# ==================================================================================================================== #
# Dataset splits
# ==================================================================================================================== #
# Karpathy
#python $PROJ_DIR/code/prepare_data.py \
#	--split-type karpathy \
#	--captions-dir $KARPATHY_INDIR \
#	--split-dir $DATASETS_DIR/coco_karpathy

# Full
#python $PROJ_DIR/code/prepare_data.py \
#	--output-dir $OUTDIR \
#  --split-type full \
#  --captions-dir $KARPATHY_INDIR \
#  --split-dir $DATASETS_DIR/coco_full

# Heldout 1
heldout_pairs="black_cat big_bird red_bus small_plane eat_man lie_woman"
python $PROJ_DIR/code/prepare_data.py \
  --split-type heldout \
  --captions-dir $OCCURRENCES_DIR \
	--heldout-pairs $heldout_pairs \
	--split-dir $DATASETS_DIR/coco_heldout_1

# Heldout 2
heldout_pairs="brown_dog small_cat white_truck big_plane ride_woman fly_bird"
python $PROJ_DIR/code/prepare_data.py \
  --split-type heldout \
  --captions-dir $OCCURRENCES_DIR \
  --heldout-pairs $heldout_pairs \
  --split-dir $DATASETS_DIR/coco_heldout_2

# Heldout 3
heldout_pairs="white_horse big_cat blue_bus small_table hold_child stand_bird"
python $PROJ_DIR/code/prepare_data.py \
  --split-type heldout \
  --captions-dir $OCCURRENCES_DIR \
  --heldout-pairs $heldout_pairs \
  --split-dir $DATASETS_DIR/coco_heldout_3

# Heldout 4
heldout_pairs="black_bird small_dog white_boat big_truck eat_horse stand_child"
python $PROJ_DIR/code/prepare_data.py \
  --split-type heldout \
  --captions-dir $OCCURRENCES_DIR \
  --heldout-pairs $heldout_pairs \
  --split-dir $DATASETS_DIR/coco_heldout_4

# ==================================================================================================================== #
# Encode datasets
# ==================================================================================================================== #
# Karpathy
#python $PROJ_DIR/code/prepare_data.py \
#  --output-dir $OUTDIR \
#  --split-dir $DATASETS_DIR/coco_karpathy \
#  --vocabulary-size 10000

# Full
#python $PROJ_DIR/code/prepare_data.py \
#  --output-dir $OUTDIR \
#  --split-dir $DATASETS_DIR/coco_full \
#  --vocabulary-size 10000

# Heldout
for i in {1..4}; do
  python $PROJ_DIR/code/prepare_data.py \
	  --output-dir $OUTDIR \
	  --split-dir $DATASETS_DIR/coco_heldout_$i \
	  --vocabulary-size 10000
done

# ==================================================================================================================== #
# Encode syntax datasets
# ==================================================================================================================== #
# Interleaved
for split in heldout_1 heldout_2 heldout_3 heldout_4; do
  for synt in idle chunk pos dep ccg; do
    mkdir -p $DATASETS_DIR/coco_${split}_${synt}_inter
    cp $DATASETS_DIR/coco_${split}/dataset_splits.json $DATASETS_DIR/coco_${split}_${synt}_inter
    python $PROJ_DIR/code/prepare_data.py \
      --output-dir $OUTDIR \
      --split-dir $DATASETS_DIR/coco_${split}_${synt}_inter \
	    --syntax-tags $synt \
      --vocabulary-size 10000
  done
done

# Sequential
for split in heldout_1 heldout_2 heldout_3 heldout_4; do
  for synt in idle chunk pos dep ccg; do
    mkdir -p $DATASETS_DIR/coco_${split}_${synt}_seq
    cp $DATASETS_DIR/coco_${split}/dataset_splits.json $DATASETS_DIR/coco_${split}_${synt}_seq
    python $PROJ_DIR/code/prepare_data.py \
      --output-dir $OUTDIR \
      --split-dir $DATASETS_DIR/coco_${split}_${synt}_seq \
      --syntax-tags $synt \
      --vocabulary-size 10000
  done
done

# Multi-task
for split in heldout_1 heldout_2 heldout_3 heldout_4; do
  for synt in idle chunk pos dep ccg; do
    mkdir -p $DATASETS_DIR/coco_${split}_${synt}_multi
    cp $DATASETS_DIR/coco_${split}/dataset_splits.json $DATASETS_DIR/coco_${split}_${synt}_multi
    python $PROJ_DIR/code/prepare_data.py \
      --output-dir $OUTDIR \
      --split-dir $DATASETS_DIR/coco_${split}_${synt}_multi \
      --syntax-tags $synt \
      --vocabulary-size 10000
  done
done

# Deactivate environment
conda deactivate
