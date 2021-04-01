"""Preprocess the COCO images and captions and store them in a hdf5 file"""

import os
import sys
import csv
import json
from tqdm import tqdm

import h5py
import base64
import numpy as np
from pycocotools.coco import COCO
from toolkit.util.data.syntax.stanfordnlp_annotator import StanfordNLPAnnotator

from toolkit.utils import (
    IMAGES_FILENAME,
    BU_FEATURES_FILENAME,
    CAPTIONS_FILENAME,
    DATA_COCO_SPLIT,
    DATA_CAPTIONS,
    DATA_CAPTION_LENGTHS,
    read_image,
)

csv.field_size_limit(sys.maxsize)


def preprocess_coco_images_and_captions(annotations_dir, images_dir, output_dir, captions_per_image):
    """
    Given COCO images in `images_dir` and corresponding annotations in `annotations_dir`,
    create preprocessed dataset in `output_dir`.
    The output directory will contain:
    - an HDF5 file with the preprocessed images called `IMAGES_FILENAME`
      with attributes `captions_per_image` and `max_caption_len`, and
      whose data points consist of (coco_id, shape, dtype, image)
    - a JSON file with the preprocessed captions called `IMAGES_META_FILENAME`
      with associated coco_ids to two lists, `DATA_CAPTIONS` and `DATA_CAPTION_LENGTHS` (before padding)
    - a JSON file with the word mappings called `WORD_MAP_FILENAME` mapping each word in the vocabulary to an integer
    """

    image2path = dict()
    image2metas = dict()
    max_caption_len = 0
    
    config = {"use_gpu": True}
    annotator = StanfordNLPAnnotator(config)

    for split in ["train2014", "val2014"]:
        ann_fn = "{}/annotations/captions_{}.json".format(annotations_dir, split)
        coco = COCO(ann_fn)

        images_data = coco.loadImgs(coco.getImgIds())
        all_captions = []
        for img in images_data:
            coco_id = img["id"]
            ann_ids = coco.getAnnIds(imgIds=[coco_id])
            anns = coco.loadAnns(ann_ids)[:captions_per_image]
            captions = [ann["caption"].replace("\n", "").strip().lower() for ann in anns]
            all_captions.extend([caption.strip() for caption in captions])

        print("Tokenizing all captions in {}...".format(split))
        all_sents = annotator.tokenize(all_captions, clean=True)
        assert len(all_sents) == len(images_data) * captions_per_image
        
        captions_list = [all_sents[ix: ix + captions_per_image] for ix in range(0, len(all_sents), captions_per_image)]
        for img, caption_list in zip(images_data, captions_list):
            coco_id = img["id"]
            caption_lens = [len(caption) for caption in caption_list]
            max_caption_len = max(max_caption_len, max(caption_lens))

            path = os.path.join(images_dir, split, img["file_name"])
            image2path[coco_id] = path
            image2metas[coco_id] = {
                DATA_CAPTIONS: caption_list,
                DATA_CAPTION_LENGTHS: caption_lens,
                DATA_COCO_SPLIT: split,
            }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save meta data to JSON file
    data_path = os.path.join(output_dir, CAPTIONS_FILENAME)
    print("Saving image meta data to {}".format(data_path))
    with open(data_path, "w") as f:
        json.dump(image2metas, f)

    # Create hdf5 file and dataset for the images
    images_dataset_path = os.path.join(output_dir, IMAGES_FILENAME)
    print("Creating image dataset at {}".format(images_dataset_path))
    with h5py.File(images_dataset_path, "a") as h5py_file:
        h5py_file.attrs["captions_per_image"] = captions_per_image
        h5py_file.attrs["max_caption_len"] = max_caption_len

        for coco_id, image_path in tqdm(image2path.items()):
            # Read image and save it to hdf5 file
            img = read_image(image_path)
            h5py_file.create_dataset(str(coco_id), (3, 256, 256), dtype="uint8", data=img)

        assert len(h5py_file.keys()) == len(image2metas)


def convert_BU_features(base_dir, output_dir):
    FIELDNAMES = ["image_id", "image_w", "image_h", "num_boxes", "boxes", "features"]
    feature_length = 2048

    output_fn = os.path.join(output_dir, BU_FEATURES_FILENAME)
    print("Saving features to {}".format(output_fn))
    output_file = h5py.File(output_fn, "w")

    count = 0
    for directory in os.listdir(base_dir):
        input_file = os.path.join(base_dir, directory)
        if os.path.isfile(input_file):
            print("Reading tsv: ", input_file)
            with open(input_file, "rt") as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter="\t", fieldnames=FIELDNAMES)
                for item in tqdm(reader):
                    image_id = item["image_id"]
                    item["num_boxes"] = int(item["num_boxes"])

                    image_features = np.frombuffer(base64.b64decode(item["features"]), 
                                                   dtype=np.float32).reshape((item["num_boxes"], -1))
                    # image_boxes = np.frombuffer(base64.decodestring(item["boxes"]),
                    #                             dtype=np.float32).reshape((item['num_boxes'],-1))
                    if image_id not in output_file:
                        output_file.create_dataset(
                            image_id,
                            (item["num_boxes"], feature_length),
                            dtype="f",
                            data=image_features,
                        )
                        count += 1

    output_file.close()
    print("Converted features for {} images".format(count))
