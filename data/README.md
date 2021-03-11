# Data Setup

We provide access to our preprocessed data and preprocessing scripts to replicate our setup.

## Preprocessed Data

- [coco2014](https://sid.erda.dk/sharelink/CVjfkjiOdO)
- [compgen](https://sid.erda.dk/sharelink/a4ocfn5VH5)

## Preprocessing Steps

`prepare_data.sh` provides sample execution for data preparation. It calls `prepare_data.py`.
Note that the output corpus directory might look different than the final one (see below).

---

The corpus directory looks as follows:
```text
corpus/
 |-- coco2014/
 |    |-- annotations/
 |    |    |-- captions_train2014.json
 |    |    |-- captions_val2014.json
 |    |    |-- instances_train2014.json
 |    |    |-- instances_val2014.json
 |    |-- images/
 |    |    |-- coco_detections.hdf5 (used by M2-TRM)
 |    |    |-- image_features.hdf5
 |    |    |-- images.hdf5
 |
 |-- compgen/
 |    |-- annotations/
 |    |    |-- captions.json
 |    |    |-- ccg_captions.json
 |    |    |-- chunk_captions.json
 |    |    |-- dep_captions.json
 |    |    |-- head_captions.json
 |    |    |-- idle_captions.json
 |    |    |-- lemma_captions.json
 |    |    |-- pos_captions.json
 |    |-- concepts/
 |    |    |-- occurrences/
 |    |    |    |-- big_bird.json
 |    |    |    |-- big_cat.json
 |    |    |    |-- ...
 |    |    |    |-- white_truck.json
 |    |    |-- synonyms/
 |    |    |    |-- adjectives/
 |    |    |    |    |-- big.json
 |    |    |    |    |-- black.json
 |    |    |    |    |-- ...
 |    |    |    |    |-- young.json
 |    |    |    |-- nouns/
 |    |    |    |-- verbs/
 |    |-- datasets/
 |    |    |-- coco_heldout_1/
 |    |    |    |-- captions.json  
 |    |    |    |-- dataset_splits.json  
 |    |    |    |-- encoded_captions.json  
 |    |    |    |-- word_map.json
 |    |    |-- coco_heldout_1_ccg_inter/
 |    |    |-- coco_heldout_1_ccg_multi/
 |    |    |-- coco_heldout_1_ccg_seq/
 |    |    |-- coco_heldout_1_chunk_inter/
 |    |    |-- ...
```
