"""Preprocess the COCO images and captions and store them in a hdf5 file"""

import os
import json
import itertools
from shutil import copy
from collections import Counter

from toolkit.utils import (
    DATASET_SPLITS_FILENAME,
    CAPTIONS_FILENAME,
    WORD_MAP_FILENAME,
    ENCODED_METAS_FILENAME,
    DATA_CAPTIONS,
    DATA_CAPTION_LENGTHS,
    TOKEN_START,
    TOKEN_END,
    TOKEN_UNKNOWN,
    TOKEN_PAD,
    TOKEN_SYNTAX,
    TRAIN_SPLIT,
    VALID_SPLIT,
    TEST_SPLIT,
    TAGGED_CAPTIONS,
    IDLE_TAGGED_CAPTIONS_FILENAME,
    CHUNK_TAGGED_CAPTIONS_FILENAME,
    POS_TAGGED_CAPTIONS_FILENAME,
    DEP_TAGGED_CAPTIONS_FILENAME,
    CCG_TAGGED_CAPTIONS_FILENAME,
)

syntax2filename = {
    "idle": IDLE_TAGGED_CAPTIONS_FILENAME,
    "chunk": CHUNK_TAGGED_CAPTIONS_FILENAME,
    "pos": POS_TAGGED_CAPTIONS_FILENAME,
    "dep": DEP_TAGGED_CAPTIONS_FILENAME,
    "ccg": CCG_TAGGED_CAPTIONS_FILENAME,
}


# ==================================================================================================================== #

# ==================================================================================================================== #
def create_word_map(words):
    """
    Create a dictionary of word -> index.
    """
    word_map = {w: i + 1 for i, w in enumerate(words)}
    # Mapping for special characters
    word_map[TOKEN_UNKNOWN] = len(word_map) + 1
    word_map[TOKEN_START] = len(word_map) + 1
    word_map[TOKEN_END] = len(word_map) + 1
    word_map[TOKEN_PAD] = 0
    return word_map


def extend_word_map(word_map, words):
    max_word_map = len(word_map)
    for ix, word in enumerate(words):
        word_map[word] = max_word_map + ix 
    return word_map


# ==================================================================================================================== #

# ==================================================================================================================== #
def encode_caption(caption, word_map, max_caption_len):
    return (
        [word_map[TOKEN_START]]
        + [word_map.get(word, word_map[TOKEN_UNKNOWN]) for word in caption]
        + [word_map[TOKEN_END]]
        + [word_map[TOKEN_PAD]] * (max_caption_len - len(caption))
    )


def encode_syntax_interleaved_caption(caption, tag_caption, word_map, max_caption_len):
    encoded_caption = [word_map[TOKEN_START]]
    for word, tag in zip(caption, tag_caption):
        next_word = word_map.get(word, word_map[TOKEN_UNKNOWN])
        if next_word == word_map[TOKEN_UNKNOWN]:
            # UNK the corresponding tag too
            next_tag = word_map[TOKEN_UNKNOWN]
        else:
            next_tag = word_map.get('_%s_' % tag.upper(), word_map[TOKEN_UNKNOWN])
        encoded_caption += [next_tag, next_word] 
    encoded_caption += [word_map[TOKEN_END]]
    encoded_caption += [word_map[TOKEN_PAD]] * 2 * (max_caption_len - len(caption))
    return encoded_caption


def encode_syntax_planning_caption(caption, tag_caption, word_map, max_caption_len):
    encoded_caption = [word_map[TOKEN_START]]
    encoded_tag_caption = [word_map[TOKEN_SYNTAX]]
    for word, tag in zip(caption, tag_caption):
        next_word = word_map.get(word, word_map[TOKEN_UNKNOWN])
        if next_word == word_map[TOKEN_UNKNOWN]:
            # UNK the corresponding tag too
            next_tag = word_map[TOKEN_UNKNOWN]
        else:
            next_tag = word_map.get('_%s_' % tag.upper(), word_map[TOKEN_UNKNOWN])
        encoded_caption += [next_word]
        encoded_tag_caption += [next_tag]
    encoded_caption = encoded_tag_caption + encoded_caption
    encoded_caption += [word_map[TOKEN_END]]
    encoded_caption += [word_map[TOKEN_PAD]] * 2 * (max_caption_len - len(caption))
    return encoded_caption


def encode_syntax_multitask_caption(caption, tag_caption, word_map, max_caption_len):
    encoded_caption = [word_map[TOKEN_START]]
    encoded_tag_caption = [word_map[TOKEN_SYNTAX]]
    for word, tag in zip(caption, tag_caption):
        next_word = word_map.get(word, word_map[TOKEN_UNKNOWN])
        if next_word == word_map[TOKEN_UNKNOWN]:
            # UNK the corresponding tag too
            next_tag = word_map[TOKEN_UNKNOWN]
        else:
            next_tag = word_map.get('_%s_' % tag.upper(), word_map[TOKEN_UNKNOWN])
        encoded_caption += [next_word]
        encoded_tag_caption += [next_tag]
    encoded_caption += [word_map[TOKEN_END]]
    encoded_tag_caption += [word_map[TOKEN_END]]
    encoded_caption += [word_map[TOKEN_PAD]] * (max_caption_len - len(caption))
    encoded_tag_caption += [word_map[TOKEN_PAD]] * (max_caption_len - len(caption))
    return encoded_caption, encoded_tag_caption


# ==================================================================================================================== #

# ==================================================================================================================== #
def encode_captions(captions_dir, dataset_split_dir, vocab_size, existing_wordmap_path):
    """
    Given COCO images in `images_dir` and corresponding annotations in 
    `annotations_dir`, create preprocessed dataset in `output_dir`.
    The output directory will contain:
    - an HDF5 file with the preprocessed images called `IMAGES_FILENAME`
      with attributes `captions_per_image` and `max_caption_len`, and
      whose data points consist of (coco_id, shape, dtype, image)
    - a JSON file with the preprocessed captions called `IMAGES_META_FILENAME`
      with associated coco_ids to two lists, `DATA_CAPTIONS` and 
      `DATA_CAPTION_LENGTHS` (before padding)
    - a JSON file with the word mappings called `WORD_MAP_FILENAME`
      mapping each word in the vocabulary to an integer
    """
    with open(os.path.join(dataset_split_dir, DATASET_SPLITS_FILENAME)) as f:
        dataset2imgs = json.load(f)
    with open(os.path.join(captions_dir, CAPTIONS_FILENAME)) as f:
        img2caption_metas = json.load(f)
    train_images = set(dataset2imgs[TRAIN_SPLIT])

    # Word mappings
    if existing_wordmap_path:
        print("Loading word mapping from {}".format(existing_wordmap_path))
        with open(existing_wordmap_path) as f:
            word_map = json.load(f)
        copy(existing_wordmap_path, dataset_split_dir)
    else:
        # Select the most frequent training words
        train_tokens = []
        for img in train_images:
            img_captions = img2caption_metas[img][DATA_CAPTIONS]
            train_tokens.extend(list(itertools.chain(*img_captions)))
        word_freq = Counter(train_tokens)
        words = [w for w, _ in word_freq.most_common(vocab_size)]

        # Create word map
        word_map = create_word_map(words)
        word_map_path = os.path.join(dataset_split_dir, WORD_MAP_FILENAME)
        print("Saving new word mapping to {}".format(word_map_path))
        with open(word_map_path, "w") as f:
            json.dump(word_map, f)

    # Max len in training captions
    train_caption_lens = []
    for img in train_images:
        img_caption_len = img2caption_metas[img][DATA_CAPTION_LENGTHS]
        train_caption_lens.extend(img_caption_len)
    max_caption_len = max(train_caption_lens)

    # Encode captions
    split_metas = dict()
    for img, metas in img2caption_metas.items():
        img_captions = metas[DATA_CAPTIONS]
        enc_captions = [encode_caption(caption, word_map, max_caption_len) for caption in img_captions]
            
        img_caption_lens = metas[DATA_CAPTION_LENGTHS]
        # extend caption length by 2 for start and end of sentence tokens
        enc_caption_lens = [l+2 for l in img_caption_lens]
        split_metas[img] = {DATA_CAPTIONS: enc_captions, DATA_CAPTION_LENGTHS: enc_caption_lens}
    split_path = os.path.join(dataset_split_dir, ENCODED_METAS_FILENAME)
    print("Saving encoded data to {}".format(split_path))
    with open(split_path, "w") as f:
        json.dump(split_metas, f)


def encode_syntax_interleaved_captions(captions_dir, dataset_split_dir, syntax_type, vocab_size, existing_wordmap_path):
    with open(os.path.join(dataset_split_dir, DATASET_SPLITS_FILENAME)) as f:
        dataset2imgs = json.load(f)
    with open(os.path.join(captions_dir, CAPTIONS_FILENAME)) as f:
        img2caption_metas = json.load(f)
    with open(os.path.join(captions_dir, syntax2filename[syntax_type])) as f:
        img2syntax_metas = json.load(f)
    train_images = set(dataset2imgs[TRAIN_SPLIT])

    # Word mappings
    if existing_wordmap_path:
        print("Loading word mapping from {}".format(existing_wordmap_path))
        with open(existing_wordmap_path) as f:
            word_map = json.load(f)
        copy(existing_wordmap_path, dataset_split_dir)
    else:
        # Select the most frequent training words
        train_tokens = []
        for img in train_images:
            img_captions = img2caption_metas[img][DATA_CAPTIONS]
            train_tokens.extend(list(itertools.chain(*img_captions)))
        word_freq = Counter(train_tokens)
        words = [w for w, _ in word_freq.most_common(vocab_size)]

        # Create word map
        word_map = create_word_map(words)
        word_map_path = os.path.join(dataset_split_dir, WORD_MAP_FILENAME)
        print("Saving new word mapping to {}".format(word_map_path))
        with open(word_map_path, "w") as f:
            json.dump(word_map, f)

    # Add syntax tags to vocabulary
    train_tags = []
    for img in train_images:
        img_tags = img2syntax_metas[img][TAGGED_CAPTIONS]
        train_tags.extend(map(lambda t: '_%s_' % t.upper(), itertools.chain(*img_tags)))
    word_map = extend_word_map(word_map, list(set(train_tags)))
    word_map_path = os.path.join(dataset_split_dir, WORD_MAP_FILENAME)
    print("Saving new word mapping to {}".format(word_map_path))
    with open(word_map_path, "w") as f:
        json.dump(word_map, f)

    # Max len in training captions
    train_caption_lens = []
    for img in train_images:
        img_caption_len = img2caption_metas[img][DATA_CAPTION_LENGTHS]
        train_caption_lens.extend(img_caption_len)
    max_caption_len = max(train_caption_lens)

    # Encode captions
    split_metas = dict()
    for img, metas in img2caption_metas.items():
        img_captions = metas[DATA_CAPTIONS]
        img_tags = img2syntax_metas[img][TAGGED_CAPTIONS]
        enc_captions = [encode_syntax_interleaved_caption(caption, tag_caption, word_map, max_caption_len)
                        for caption, tag_caption in zip(img_captions, img_tags)]

        img_caption_lens = metas[DATA_CAPTION_LENGTHS]
        # extend caption length by 2 for start and end of sentence tokens
        enc_caption_lens = [2*l+2 for l in img_caption_lens]
        split_metas[img] = {DATA_CAPTIONS: enc_captions, DATA_CAPTION_LENGTHS: enc_caption_lens}
    split_path = os.path.join(dataset_split_dir, ENCODED_METAS_FILENAME)
    print("Saving encoded data to {}".format(split_path))
    with open(split_path, "w") as f:
        json.dump(split_metas, f)


def encode_syntax_planning_captions(captions_dir, dataset_split_dir, syntax_type, vocab_size, existing_wordmap_path):
    with open(os.path.join(dataset_split_dir, DATASET_SPLITS_FILENAME)) as f:
        dataset2imgs = json.load(f)
    with open(os.path.join(captions_dir, CAPTIONS_FILENAME)) as f:
        img2caption_metas = json.load(f)
    with open(os.path.join(captions_dir, syntax2filename[syntax_type])) as f:
        img2syntax_metas = json.load(f)
    train_images = set(dataset2imgs[TRAIN_SPLIT])

    # Word mappings
    if existing_wordmap_path:
        print("Loading word mapping from {}".format(existing_wordmap_path))
        with open(existing_wordmap_path) as f:
            word_map = json.load(f)
        copy(existing_wordmap_path, dataset_split_dir)
    else:
        # Select the most frequent training words
        train_tokens = []
        for img in train_images:
            img_captions = img2caption_metas[img][DATA_CAPTIONS]
            train_tokens.extend(list(itertools.chain(*img_captions)))
        word_freq = Counter(train_tokens)
        words = [w for w, _ in word_freq.most_common(vocab_size)]

        # Create word map
        word_map = create_word_map(words)
        word_map_path = os.path.join(dataset_split_dir, WORD_MAP_FILENAME)
        print("Saving new word mapping to {}".format(word_map_path))
        with open(word_map_path, "w") as f:
            json.dump(word_map, f)

    # Add syntax tags to vocabulary
    train_tags = []
    for img in train_images:
        img_tags = img2syntax_metas[img][TAGGED_CAPTIONS]
        train_tags.extend(map(lambda t: '_%s_' % t.upper(), itertools.chain(*img_tags)))
    word_map = extend_word_map(word_map, list(set(train_tags)) + [TOKEN_SYNTAX])
    word_map_path = os.path.join(dataset_split_dir, WORD_MAP_FILENAME)
    print("Saving new word mapping to {}".format(word_map_path))
    with open(word_map_path, "w") as f:
        json.dump(word_map, f)

    # Max len in training captions
    train_caption_lens = []
    for img in train_images:
        img_caption_len = img2caption_metas[img][DATA_CAPTION_LENGTHS]
        train_caption_lens.extend(img_caption_len)
    max_caption_len = max(train_caption_lens)

    # Encode captions
    split_metas = dict()
    for img, metas in img2caption_metas.items():
        img_captions = metas[DATA_CAPTIONS]
        img_tags = img2syntax_metas[img][TAGGED_CAPTIONS]
        enc_captions = [encode_syntax_planning_caption(caption, tag_caption, word_map, max_caption_len)
                        for caption, tag_caption in zip(img_captions, img_tags)]

        img_caption_lens = metas[DATA_CAPTION_LENGTHS]
        # extend caption length by 2 for start and end of sentence tokens and by 1 for end of planning
        enc_caption_lens = [2*l+2+1 for l in img_caption_lens]
        split_metas[img] = {DATA_CAPTIONS: enc_captions, DATA_CAPTION_LENGTHS: enc_caption_lens}
    split_path = os.path.join(dataset_split_dir, ENCODED_METAS_FILENAME)
    print("Saving encoded data to {}".format(split_path))
    with open(split_path, "w") as f:
        json.dump(split_metas, f)


def encode_syntax_multitask_captions(captions_dir, dataset_split_dir, syntax_type, vocab_size, existing_wordmap_path):
    with open(os.path.join(dataset_split_dir, DATASET_SPLITS_FILENAME)) as f:
        dataset2imgs = json.load(f)
    with open(os.path.join(captions_dir, CAPTIONS_FILENAME)) as f:
        img2caption_metas = json.load(f)
    with open(os.path.join(captions_dir, syntax2filename[syntax_type])) as f:
        img2syntax_metas = json.load(f)
    train_images = set(dataset2imgs[TRAIN_SPLIT])

    # Word mappings
    if existing_wordmap_path:
        print("Loading word mapping from {}".format(existing_wordmap_path))
        with open(existing_wordmap_path) as f:
            word_map = json.load(f)
        copy(existing_wordmap_path, dataset_split_dir)
    else:
        # Select the most frequent training words
        train_tokens = []
        for img in train_images:
            img_captions = img2caption_metas[img][DATA_CAPTIONS]
            train_tokens.extend(list(itertools.chain(*img_captions)))
        word_freq = Counter(train_tokens)
        words = [w for w, _ in word_freq.most_common(vocab_size)]

        # Create word map
        word_map = create_word_map(words)
        word_map_path = os.path.join(dataset_split_dir, WORD_MAP_FILENAME)
        print("Saving new word mapping to {}".format(word_map_path))
        with open(word_map_path, "w") as f:
            json.dump(word_map, f)

    # Add syntax tags to vocabulary
    train_tags = []
    for img in train_images:
        img_tags = img2syntax_metas[img][TAGGED_CAPTIONS]
        train_tags.extend(map(lambda t: '_%s_' % t.upper(), itertools.chain(*img_tags)))
    word_map = extend_word_map(word_map, list(set(train_tags)) + [TOKEN_SYNTAX])
    word_map_path = os.path.join(dataset_split_dir, WORD_MAP_FILENAME)
    print("Saving new word mapping to {}".format(word_map_path))
    with open(word_map_path, "w") as f:
        json.dump(word_map, f)

    # Max len in training captions
    train_caption_lens = []
    for img in train_images:
        img_caption_len = img2caption_metas[img][DATA_CAPTION_LENGTHS]
        train_caption_lens.extend(img_caption_len)
    max_caption_len = max(train_caption_lens)

    # Encode captions
    split_metas = dict()
    for img, metas in img2caption_metas.items():
        img_captions = metas[DATA_CAPTIONS]
        if img in train_images:
            img_tags = img2syntax_metas[img][TAGGED_CAPTIONS]
            enc_captions = [encode_syntax_multitask_caption(caption, tag_caption, word_map, max_caption_len)
                            for caption, tag_caption in zip(img_captions, img_tags)]
            enc_captions = [caption for captions in enc_captions for caption in captions]
            img_caption_lens = metas[DATA_CAPTION_LENGTHS]
            # extend caption length by 2 for start and end of sentence tokens
            enc_caption_lens = [l+2 for l in img_caption_lens]
            enc_caption_lens = [enc_caption_lens[i//2] for i in range(len(enc_caption_lens)*2)] 
        else:
            enc_captions = [encode_caption(caption, word_map, max_caption_len) for caption in img_captions]
            img_caption_lens = metas[DATA_CAPTION_LENGTHS]
            # extend caption length by 2 for start and end of sentence tokens
            enc_caption_lens = [l+2 for l in img_caption_lens]
        split_metas[img] = {DATA_CAPTIONS: enc_captions, DATA_CAPTION_LENGTHS: enc_caption_lens}
    split_path = os.path.join(dataset_split_dir, ENCODED_METAS_FILENAME)
    print("Saving encoded data to {}".format(split_path))
    with open(split_path, "w") as f:
        json.dump(split_metas, f)
