import sys
import os.path
import argparse

sys.path.append(os.path.abspath('../code'))
from toolkit.util.data.download import maybe_download_and_extract
from toolkit.util.data.preprocess import preprocess_coco_images_and_captions, convert_BU_features
from toolkit.util.data.syntax.annotate import annotate_captions
from toolkit.util.data.composition import count_concept_pair
from toolkit.util.data.split import create_dataset_splits
from toolkit.util.data.encode import encode_captions, encode_syntax_interleaved_captions, encode_syntax_planning_captions, encode_syntax_multitask_captions

TRAIN2014_IMG_URL = "http://images.cocodataset.org/zips/train2014.zip"
VAL2014_IMG_URL = "http://images.cocodataset.org/zips/val2014.zip"
TRAINVAL2014_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
KARPATHY_SPLITS_URL = "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"
FRCNN_FEATURES_URL = "https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip"


def check_args(args):
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group("Preprocessing")
    group.add_argument("--img-dir",
                       help="Path where to store COCO images")
    group.add_argument("--ann-dir",
                       help="Path where to store COCO annotations")
    group.add_argument("--karpathy-dir",
                       help="Path where to store Karpathy's caption splits")
    group.add_argument("--output-dir",
                       default=os.path.expanduser("../data/coco2014"),
                       help="Path to output preprocessed data")
    group.add_argument("--captions-per-image", type=int, default=5,
                       help="Maximum number of captions per image")
    group.add_argument("--word-map", type=str, default=None,
                       help="Path to existing word map file to be used")

    group = parser.add_argument_group("Syntactic tagging")
    group.add_argument("--syntax-tags", nargs="+", default=[],
                       help="List of captions' syntactic tags to be extracted")
    group.add_argument("--batch-size", type=int, default=None,
                       help="Number of captions to annotate in a batch (default: all)")
    group.add_argument("--batch-num", type=int, default=None,
                       help="Batch id")

    group = parser.add_argument_group("Compositionality")
    group.add_argument("--synonyms-dir", 
                       help="Path to concept synonyms data")
    group.add_argument("--occurrences-dir",
                       help="Path where to store concepts' occurrence counts")
    group.add_argument("--concept-pairs", nargs="+", default=[],
                       help="List of other,noun,type concept pair triplets")

    group = parser.add_argument_group("Dataset splitting")
    group.add_argument("--split-type", choices=["karpathy", "heldout", "full"],
                       help="Type of dataset split to extract")
    group.add_argument("--captions-dir", 
                       help="Path to karpathy splits or occurrences pairs")
    group.add_argument("--heldout-pairs", nargs="+", default=[],
                       help="adj-noun or verb-noun pairs to be held out")
    group.add_argument("--split-dir",
                       help="Path where to store resulting dataset split")

    group = parser.add_argument_group("Encoding")
    group.add_argument("--vocabulary-size", type=int, default=10000,
                       help="Number of words to be stored in the vocabulary")
    group.add_argument("--existing-word-map-path",
                       help="Path to existing word mapping")
    group.add_argument("--encoding-dir",
                       help="Path where to store encoded datasets")

    parsed_args = parser.parse_args(args)
    return parsed_args


if __name__ == "__main__":
    args = check_args(sys.argv[1:])

    maybe_download_and_extract(TRAIN2014_IMG_URL, args.img_dir)
    maybe_download_and_extract(VAL2014_IMG_URL, args.img_dir)
    maybe_download_and_extract(TRAINVAL2014_ANN_URL, args.ann_dir)
    maybe_download_and_extract(KARPATHY_SPLITS_URL, args.cap_dir)
    maybe_download_and_extract(FRCNN_FEATURES_URL, args.img_dir)

    preprocess_coco_images_and_captions(args.ann_dir, args.img_dir, args.output_dir, args.captions_per_image)

    convert_BU_features(args.img_dir, args.output_dir)

    for syntax_type in args.syntax_tags:
        annotate_captions(args.output_dir, syntax_type)

    for concept_triplet in args.concept_pairs:
        other, noun, concept_type = concept_triplet.split(",")
        others = "adjectives" if "adj" in concept_type else "verbs"
        noun_fn = os.path.join(args.synonyms_dir, "nouns", "{}.json".format(noun))
        other_fn = os.path.join(args.synonyms_dir, others, "{}.json".format(other))
        count_concept_pair(noun_fn, other_fn, concept_type, args.output_dir, args.occurrences_dir)

    create_dataset_splits(args.split_type, args.captions_dir, args.split_dir, args.heldout_pairs)

    encode_captions(args.output_dir, args.split_dir, args.vocabulary_size, args.existing_word_map_path)

    for syntax_type in args.syntax_tags:
       encode_syntax_interleaved_captions(args.output_dir, args.split_dir, syntax_type,
                                          args.vocabulary_size, args.existing_word_map_path)

    for syntax_type in args.syntax_tags:
        encode_syntax_planning_captions(args.output_dir, args.split_dir, syntax_type,
                                        args.vocabulary_size, args.existing_word_map_path)

    for syntax_type in args.syntax_tags:
        encode_syntax_multitask_captions(args.output_dir, args.split_dir, syntax_type,
                                         args.vocabulary_size, args.existing_word_map_path)
