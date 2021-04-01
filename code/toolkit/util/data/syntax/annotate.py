"""Script for POS-tagging of all coco captions"""

import os
import json

from toolkit.util.data.syntax.idle_annotator import IdleAnnotator
from toolkit.util.data.syntax.depccg_annotator import DepCCGAnnotator
from toolkit.util.data.syntax.nltk_annotator import NLTKAnnotator
from toolkit.util.data.syntax.stanfordnlp_annotator import StanfordNLPAnnotator
from toolkit.utils import (
    CAPTIONS_FILENAME,
    IDLE_TAGGED_CAPTIONS_FILENAME,
    CHUNK_TAGGED_CAPTIONS_FILENAME,
    LEMMA_TAGGED_CAPTIONS_FILENAME,
    POS_TAGGED_CAPTIONS_FILENAME,
    DEP_TAGGED_CAPTIONS_FILENAME,
    HEAD_TAGGED_CAPTIONS_FILENAME,
    CCG_TAGGED_CAPTIONS_FILENAME,
)


def annotate_captions(data_dir, syntax_type, batch_size=None, batch_num=None):
    syntax_type = syntax_type.lower()
    if syntax_type == "idle":
        annotator = IdleAnnotator()
        tag_captions_fn = IDLE_TAGGED_CAPTIONS_FILENAME
    elif syntax_type == "lemma":
        config = {"use_gpu": True, "tokenize_pretokenized": True} #FIXME
        annotator = StanfordNLPAnnotator(config)
        tag_captions_fn = LEMMA_TAGGED_CAPTIONS_FILENAME
    elif syntax_type == "chunk":
        annotator = NLTKAnnotator()
        tag_captions_fn = CHUNK_TAGGED_CAPTIONS_FILENAME
    elif syntax_type == "pos":
        config = {"use_gpu": False, "tokenize_pretokenized": True} #FIXME
        annotator = StanfordNLPAnnotator(config)
        tag_captions_fn = POS_TAGGED_CAPTIONS_FILENAME
    elif syntax_type == "dep":
        config = {"use_gpu": False, "tokenize_pretokenized": True} #FIXME
        annotator = StanfordNLPAnnotator(config)
        tag_captions_fn = DEP_TAGGED_CAPTIONS_FILENAME
    elif syntax_type == "head":
        config = {"use_gpu": False, "tokenize_pretokenized": True} #FIXME
        annotator = StanfordNLPAnnotator(config)
        tag_captions_fn = HEAD_TAGGED_CAPTIONS_FILENAME
    elif syntax_type == "ccg":
        annotator = DepCCGAnnotator()
        tag_captions_fn = CCG_TAGGED_CAPTIONS_FILENAME
    else:
        raise ValueError("""Invalid syntax_type. Options: idle, lemma, chunk, pos, dep, ccg""")

    # Load dictionary with captions
    with open(os.path.join(data_dir, CAPTIONS_FILENAME)) as f:
        image2metas = json.load(f)
    
    # Batch
    if batch_size and batch_num is not None:
        ids = list(image2metas.keys())
        batch_ids = [ids[i:i+batch_size] for i in range(0, len(ids), batch_size)][batch_num]
        image2metas = {k: image2metas[k] for k in batch_ids}
        tag_captions_fn += '.%d' % batch_num

    # Annotate captions with syntax tags
    image2syntax_metas = annotator.annotate(image2metas, data_dir, syntax_type)
    
    # Save tag captions to JSON file
    data_path = os.path.join(data_dir, tag_captions_fn)
    print("Saving {} tag captions to {}".format(syntax_type, data_path))
    with open(data_path, "w") as f:
        json.dump(image2syntax_metas, f)


def merge_captions(data_dir, syntax_type, num_batches):
    if syntax_type == "idle":
        tag_captions_fn = IDLE_TAGGED_CAPTIONS_FILENAME
    elif syntax_type == "lemma":
        tag_captions_fn = LEMMA_TAGGED_CAPTIONS_FILENAME
    elif syntax_type == "chunk":
        tag_captions_fn = CHUNK_TAGGED_CAPTIONS_FILENAME
    elif syntax_type == "pos":
        tag_captions_fn = POS_TAGGED_CAPTIONS_FILENAME
    elif syntax_type == "dep":
        tag_captions_fn = DEP_TAGGED_CAPTIONS_FILENAME
    elif syntax_type == "head":
        tag_captions_fn = HEAD_TAGGED_CAPTIONS_FILENAME
    elif syntax_type == "ccg":
        tag_captions_fn = CCG_TAGGED_CAPTIONS_FILENAME
    else:
        raise ValueError("""Invalid syntax_type. Options: idle, lemma, chunk, pos, dep, ccg""")

    batches = []
    for i in range(num_batches):
        data_path = os.path.join(data_dir, tag_captions_fn + '.%d' % i)
        with open(data_path) as f:
            batches.append(json.load(f))
    
    image2syntax_metas = dict()
    for image2metas in batches:
        for img, metas in image2metas.items():
            image2syntax_metas[img] = metas

    data_path = os.path.join(data_dir, tag_captions_fn)
    print("Saving {} tag captions to {}".format(syntax_type, data_path))
    with open(data_path, "w") as f:
        json.dump(image2syntax_metas, f)
