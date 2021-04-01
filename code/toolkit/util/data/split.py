"""Create dataset splits by defining a set of heldout concept pairs."""

import os
import json

from toolkit.utils import (    
    DATA_COCO_SPLIT,
    OCCURRENCE_DATA,
    PAIR_OCCURRENCES,
    TRAIN_SPLIT, 
    VALID_SPLIT, 
    TEST_SPLIT, 
    HELDOUT_PAIRS,
    DATASET_SPLITS_FILENAME
)


def get_occurrences_splits(occurrences_fns): 
    test_images = set()
    val_images = set()
    for fn in occurrences_fns:
        with open(fn) as f:
            data = json.load(f)

        test_images |= {k for k, v in data[OCCURRENCE_DATA].items()
                        if v[PAIR_OCCURRENCES] >= 1 and v[DATA_COCO_SPLIT] == "val2014"}
        val_images |= {k for k, v in data[OCCURRENCE_DATA].items()
                       if v[PAIR_OCCURRENCES] >= 1 and v[DATA_COCO_SPLIT] == "train2014"}

    with open(occurrences_fns[0]) as f:
        data = json.load(f)

    train_images = {k for k, v in data[OCCURRENCE_DATA].items()
                    if k not in val_images and v[DATA_COCO_SPLIT] == "train2014"}

    return list(train_images), list(val_images), list(test_images)


def get_full_splits(karpathy_fn):
    with open(karpathy_fn) as f:
        images_data = json.load(f)["images"]

    train_images = [str(data["cocoid"]) for data in images_data if data["split"] == "train"]
    val_images = [str(data["cocoid"]) for data in images_data if data["split"] == "val"]
    test_images = [str(data["cocoid"]) for data in images_data if data["split"] == "test"]

    return train_images, val_images, test_images


def get_karpathy_splits(karpathy_fn):
    with open(karpathy_fn) as f:
        images_data = json.load(f)["images"]

    train_images = [str(data["cocoid"]) for data in images_data if data["split"] in {"train", "restval"}]
    val_images = [str(data["cocoid"]) for data in images_data if data["split"] == "val"]
    test_images = [str(data["cocoid"]) for data in images_data if data["split"] == "test"]

    return train_images, val_images, test_images


def create_dataset_splits(split_type, captions_dir, output_dir, heldout_pairs):
    if split_type == "karpathy":
        get_splits = get_karpathy_splits
        input_fn = os.path.join(captions_dir, "dataset_coco.json")
        print("Karpathy splits")
    elif split_type == "full":
        get_splits = get_full_splits
        input_fn = os.path.join(captions_dir, "dataset_coco.json")
        print("Full splits")
    elif split_type == "heldout":
        if not heldout_pairs:
            raise ValueError("Missing heldout-pairs")
        get_splits = get_occurrences_splits
        input_fn = [os.path.join(captions_dir, pair + ".json") 
                    for pair in heldout_pairs]
        print("{}".format(heldout_pairs))
    elif split_type == "robust":
        raise ValueError("Not implemented yet")
    else:
        raise ValueError("Invalid split-type. Options: karpathy, heldout, full")

    train_images, val_images, test_images = get_splits(input_fn)
    print("\tTrain set size: {}".format(len(train_images)))
    print("\tVal set size: {}".format(len(val_images)))
    print("\tTest set size: {}".format(len(test_images)))

    dataset_splits = {
        TRAIN_SPLIT: train_images,
        VALID_SPLIT: val_images,
        TEST_SPLIT: test_images,
        HELDOUT_PAIRS: heldout_pairs,
    }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, DATASET_SPLITS_FILENAME), "w") as f:
        json.dump(dataset_splits, f)
