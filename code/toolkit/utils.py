"""General utility functions and variables"""

import os
import logging
import shutil

import torch
import torch.nn as nn

from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import transforms
from tqdm import tqdm

# Special tokens
TOKEN_UNKNOWN = "<unk>"
TOKEN_START = "<start>"
TOKEN_END = "<end>"
TOKEN_PAD = "<pad>"
# TOKEN_WORDS = "_word_"
TOKEN_SYNTAX = "_syntax_"
TOKEN_MASK_TAG = "_masktag_"
TOKEN_MASK_WORD = "_maskword_"

# Normalization for images (cf. https://pytorch-zh.readthedocs.io/en/latest/torchvision/models.html)
IMAGENET_IMAGES_MEAN = [0.485, 0.456, 0.406]
IMAGENET_IMAGES_STD = [0.229, 0.224, 0.225]

# COCO attributes and filenames
DATA_CAPTIONS = "captions"
DATA_CAPTION_LENGTHS = "caption_lengths"
DATA_COCO_SPLIT = "coco_split"
CAPTIONS_FILENAME = "captions.json"
IMAGES_FILENAME = "images.hdf5"
BU_FEATURES_FILENAME = "image_features.hdf5"

# Syntax attributes and filenames
TOKEN_IDLE = "IDLE"
STANFORDNLP_DIR = os.path.expanduser('~/bin/stanfordnlp_resources')
STANFORDNLP_ANNOTATIONS_FILENAME = "stanfordnlp_annotataions.json"
STANFORDNLP_FIELD2IDX = {'id': 0, 'word': 1, 'lemma': 2, 'upos': 3, 'xpos': 4, 
                         'feats': 5, 'head': 6, 'deprel': 7, 'deps': 8, 
                         'misc': 9}
CAPTIONS_META = "captions_meta"
TAGGED_CAPTIONS = "tagged_captions"
CAPTIONS_META_FILENAME = "captions_meta.json"
IDLE_MAP_FILENAME = "idle_map.json"
CHUNK_MAP_FILENAME = "chunk_map.json"
POS_MAP_FILENAME = "pos_map.json"
DEP_MAP_FILENAME = "dep_map.json"
CCG_MAP_FILENAME = "ccg_map.json"
IDLE_TAGGED_CAPTIONS_FILENAME = "idle_captions.json"
LEMMA_TAGGED_CAPTIONS_FILENAME = "lemma_captions.json"
CHUNK_TAGGED_CAPTIONS_FILENAME = "chunk_captions.json"
POS_TAGGED_CAPTIONS_FILENAME = "pos_captions.json"
DEP_TAGGED_CAPTIONS_FILENAME = "dep_captions.json"
HEAD_TAGGED_CAPTIONS_FILENAME = "head_captions.json"
CCG_TAGGED_CAPTIONS_FILENAME = "ccg_captions.json"
DEPCCG_MODEL_FILENAME = os.path.expanduser("~/bin/depccg_resources/lstm_parser_elmo_finetune.tar.gz")

# Composition attributes
NOUNS = "nouns"
ADJECTIVES = "adjectives"
VERBS = "verbs"

OCCURRENCE_DATA = "occurrence_data"
PAIR_OCCURRENCES = "pair_occurrences"
NOUN_OCCURRENCES = "noun_occurrences"
VERB_OCCURRENCES = "verb_occurrences"
ADJECTIVE_OCCURRENCES = "adjective_occurrences"

RELATION_NOMINAL_SUBJECT = "nsubj"
RELATION_ADJECTIVAL_MODIFIER = "amod"
RELATION_CONJUNCT = "conj"
RELATION_RELATIVE_CLAUSE_MODIFIER = "acl:relcl"
RELATION_ADJECTIVAL_CLAUSE = "acl"
RELATION_OBJECT = "obj"
RELATION_INDIRECT_OBJECT = "iobj"
RELATION_OBLIQUE_NOMINAL = "obl"

# Dataset splits attributes
DATASET_SPLITS_FILENAME = "dataset_splits.json"
TRAIN_SPLIT = "train_images"
VALID_SPLIT = "val_images"
TEST_SPLIT = "test_images"
HELDOUT_PAIRS = "heldout_pairs"
WORD_MAP_FILENAME = "word_map.json"
ENCODED_METAS_FILENAME = "encoded_captions.json"

# Models
MODEL_SHOW_ATTEND_TELL = "SHOW_ATTEND_TELL"
MODEL_BOTTOM_UP_TOP_DOWN = "BOTTOM_UP_TOP_DOWN"
MODEL_BOTTOM_UP_TOP_DOWN_RANKING = "BOTTOM_UP_TOP_DOWN_RANKING"
MODEL_BOTTOM_UP_TOP_DOWN_RANKING_MEAN = "BOTTOM_UP_TOP_DOWN_RANKING_MEAN"
MODEL_BOTTOM_UP_TOP_DOWN_RANKING_WEIGHT = "BOTTOM_UP_TOP_DOWN_RANKING_WEIGHT"

# Training objectives
OBJECTIVE_GENERATION = "GENERATION"
OBJECTIVE_JOINT = "JOINT"


# ============================================================================ #
#                                     DATA                                     #
# ============================================================================ #
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


def encode_caption(caption, word_map, max_caption_len):
    """
    Map words in caption into corresponding indices
    after adding <start>, <stop> and <pad> tokens.
    """
    return (
        [word_map[TOKEN_START]]
        + [word_map.get(word, word_map[TOKEN_UNKNOWN]) for word in caption]
        + [word_map[TOKEN_END]]
        + [word_map[TOKEN_PAD]] * (max_caption_len - len(caption))
    )


def decode_caption(encoded_caption, word_map):
    rev_word_map = {v: k for k, v in word_map.items()}
    return " ".join(rev_word_map[ind].lower() for ind in encoded_caption)


def rm_caption_special_tokens(caption, word_map):
    """Remove start, end and padding tokens from encoded caption."""
    rev_word_map = {v: k for k, v in word_map.items()}
    return [tok for tok in caption
            if not (tok in {word_map[TOKEN_START], word_map[TOKEN_END], word_map[TOKEN_PAD]}
                    or rev_word_map[tok].startswith("_"))]


# ============================================================================ #
def read_image(path):
    img = imread(path)
    if len(img.shape) == 2:  # b/w image
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = imresize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    assert img.shape == (3, 256, 256)
    assert np.max(img) <= 255
    return img


def invert_normalization(image):
    image = torch.FloatTensor(image)
    inv_normalize = transforms.Normalize(
        mean=(-1 * np.array(IMAGENET_IMAGES_MEAN) / np.array(IMAGENET_IMAGES_STD)),
        std=(1 / np.array(IMAGENET_IMAGES_STD)),
    )
    return inv_normalize(image)


def show_img(img):
    plt.imshow(img.transpose(1, 2, 0))
    plt.axis("off")
    plt.show()


# ============================================================================ #
#
# ============================================================================ #
def get_objects_for_noun(pos_tagged_caption, nouns):
    dependencies = pos_tagged_caption.dependencies

    objects = {
        d[2].lemma
        for d in dependencies
        if (
            d[1] == RELATION_OBJECT
            or d[1] == RELATION_INDIRECT_OBJECT
            or d[1] == RELATION_OBLIQUE_NOMINAL
        )
        and d[0].lemma in nouns
    }
    return objects


def get_objects_for_verb(pos_tagged_caption, verbs):
    dependencies = pos_tagged_caption.dependencies

    objects = {
        d[2].lemma
        for d in dependencies
        if (
            d[1] == RELATION_OBJECT
            or d[1] == RELATION_INDIRECT_OBJECT
            or d[1] == RELATION_OBLIQUE_NOMINAL
        )
        and d[0].lemma in verbs
    }
    return objects


# ============================================================================ #

# ============================================================================ #
def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrink the learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate should be shrunk.
    :param shrink_factor: factor to multiply learning rate with.
    """

    logging.info("\nAdjusting learning rate.")
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr"] * shrink_factor
    logging.info(
        "The new learning rate is {}\n".format(optimizer.param_groups[0]["lr"])
    )


def clip_gradients(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def get_log_file_path(logging_dir, split="train"):
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    return os.path.join(logging_dir, split + ".log")


def save_checkpoint(checkpoints_dir, model_name, model,
                    epoch, epochs_since_last_improvement,
                    encoder_optimizer, decoder_optimizer,
                    generation_metric_score, is_best, **kwargs):
    """
    Save a model checkpoint.

    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update the encoder's weights
    :param decoder_optimizer: optimizer to update the decoder's weights
    :param validation_metric_score: validation set score for this epoch
    :param is_best: True, if this is the best checkpoint so far (will save the model to a dedicated file)
    """
    state = {
        "model_name": model_name,
        "epoch": epoch,
        "epochs_since_improvement": epochs_since_last_improvement,
        "gen_metric_score": generation_metric_score,
        "model": model,
        "encoder_optimizer": encoder_optimizer,
        "decoder_optimizer": decoder_optimizer,
    }
    for k, v in kwargs.items():
        state[k] = v
    filename = os.path.join(checkpoints_dir, "checkpoint.{}.pth.tar".format(epoch))
    torch.save(state, filename)
    shutil.copyfile(filename, os.path.join(checkpoints_dir, "checkpoint.last.pth.tar"))
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoints_dir, "checkpoint.best.pth.tar"))


class AverageMeter(object):
    """Class to keep track of most recent, average, sum, and count of a metric."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.uniform_(m.weight, -0.1, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def LSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def l2_norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


def load_pretrained_embedding_from_file(embed_path, word_map):
    """Return an embedding for the specified word map from the GloVe embedding file"""
    logging.info("\nLoading embeddings: {}".format(embed_path))
    with open(embed_path, "r") as f:
        embed_dim = len(f.readline().split(" ")) - 1
    vocab = set(word_map.keys())
    num_embeddings = len(vocab)
    padding_idx = word_map[TOKEN_PAD]

    embeddings = torch.FloatTensor(num_embeddings, embed_dim)
    # Initialize weights with random values (these will stay only if 
    # a word in the vocabulary does not exist in the loaded embeddings vocabulary)
    nn.init.uniform_(embeddings, -0.1, 0.1)
    nn.init.constant_(embeddings[padding_idx], 0)

    shared_tokens = set()
    with open(embed_path) as f:
        for line in tqdm(f.readlines()):
            line_split = line.split(" ")
            emb_word = line_split[0]  # word in pretrained embedding
            if emb_word in vocab:
                embedding = [float(t) for t in line_split[1:] if not t.isspace()]
                embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)
                shared_tokens.add(emb_word)

    missed_tokens = vocab - shared_tokens
    if missed_tokens:
        logging.info("""\nThe loaded embeddings did not contain an embedding 
                        for the following tokens: {}""".format(missed_tokens))

    return embeddings, embed_dim

