"""Create occurrence statistics for a given concept pair"""
import os
import json
from tqdm import tqdm

from toolkit.utils import (
    TAGGED_CAPTIONS,
    PAIR_OCCURRENCES,
    ADJECTIVE_OCCURRENCES,
    NOUN_OCCURRENCES,
    NOUNS,
    ADJECTIVES,
    OCCURRENCE_DATA,
    DATA_COCO_SPLIT,
    POS_TAGGED_CAPTIONS_FILENAME,
    VERBS,
    VERB_OCCURRENCES,
    LEMMA_TAGGED_CAPTIONS_FILENAME,
    DEP_TAGGED_CAPTIONS_FILENAME,
    HEAD_TAGGED_CAPTIONS_FILENAME,
    RELATION_NOMINAL_SUBJECT,
    RELATION_ADJECTIVAL_MODIFIER,
    RELATION_CONJUNCT,
    RELATION_RELATIVE_CLAUSE_MODIFIER,
    RELATION_ADJECTIVAL_CLAUSE,
)


def get_adjectives_for_noun(caption_metas, noun_synonyms):
    lemma_toks = caption_metas["lemma"]
    pos_toks = caption_metas["pos"]
    dep_toks = caption_metas["dep"]

    head_ixs = [int(ix)-1 for ix in caption_metas["head"]]
    head_lemma_toks = [lemma_toks[ix] for ix in head_ixs]
    head_pos_toks = [pos_toks[ix] for ix in head_ixs]

    adjectives = ({lemma_toks[i] for i in range(len(dep_toks))
                   if dep_toks[i] == RELATION_ADJECTIVAL_MODIFIER
                   and head_lemma_toks[i] in noun_synonyms
                   and pos_toks[i] == "ADJ"} | 
                  {head_lemma_toks[i] for i in range(len(dep_toks))
                   if dep_toks[i] == RELATION_NOMINAL_SUBJECT
                   and lemma_toks[i] in noun_synonyms
                   and head_pos_toks[i] == "ADJ"})

    conjuncted_adjectives = set()
    for adjective in adjectives:
        conjuncted_adjectives.update(
            {lemma_toks[i] for i in range(len(dep_toks))
             if dep_toks[i] == RELATION_CONJUNCT
             and head_lemma_toks[i] == adjective
             and pos_toks[i] == "ADJ"} |
            {lemma_toks[i] for i in range(len(dep_toks))
             if dep_toks[i] == RELATION_ADJECTIVAL_MODIFIER
             and head_lemma_toks[i] == adjective
             and pos_toks[i] == "ADJ"}
        )

    return (adjectives | conjuncted_adjectives)


def get_verbs_for_noun(caption_metas, noun_synonyms):
    lemma_toks = caption_metas["lemma"]
    pos_toks = caption_metas["pos"]
    dep_toks = caption_metas["dep"]
    
    head_ixs = [int(ix)-1 for ix in caption_metas["head"]]
    head_lemma_toks = [lemma_toks[ix] for ix in head_ixs]
    head_pos_toks = [pos_toks[ix] for ix in head_ixs]

    verbs = ({head_lemma_toks[i] for i in range(len(dep_toks))
              if dep_toks[i] == RELATION_NOMINAL_SUBJECT
              and lemma_toks[i] in noun_synonyms
              and head_pos_toks[i] == "VERB"} |
             {lemma_toks[i] for i in range(len(dep_toks))
              if dep_toks[i] == RELATION_RELATIVE_CLAUSE_MODIFIER
              and head_lemma_toks[i] in noun_synonyms
              and pos_toks[i] == "VERB"} |
             {lemma_toks[i] for i in range(len(dep_toks))
              if dep_toks[i] == RELATION_ADJECTIVAL_CLAUSE
              and head_lemma_toks[i] in noun_synonyms
              and pos_toks[i] == "VERB"})

    return verbs


def has_concept_pair(caption_metas, noun_synonyms, other_synonyms, concept_type):
    """
    TODO noun & other are sets
    """
    lemmas = set(caption_metas["lemma"])
    has_noun = len(lemmas.intersection(noun_synonyms)) > 0
    has_other = len(lemmas.intersection(other_synonyms)) > 0

    if concept_type == "adj-noun":
        get_types_for_noun = get_adjectives_for_noun
    elif concept_type == "verb-noun":
        get_types_for_noun = get_verbs_for_noun
    else:
        raise ValueError("Invalid --concept-type. Options: adj-noun, verb-noun")
    caption_types_for_noun = get_types_for_noun(caption_metas, noun_synonyms)
    has_combination = bool(other_synonyms & caption_types_for_noun)

    return has_noun, has_other, has_combination


# ============================================================================ #
def count_concept_pair(noun_fn, other_fn, concept_type, data_dir, out_dir):
    """
    TODO
    """
    with open(noun_fn, "r") as f:
        noun_synonyms = json.load(f)
    first_noun = noun_synonyms[0]
    
    with open(other_fn, "r") as f:
        other_synonyms = json.load(f)
    first_other = other_synonyms[0]

    image2metas = dict()
    with open(os.path.join(data_dir, LEMMA_TAGGED_CAPTIONS_FILENAME)) as f:
        image2lemma_metas = json.load(f)
    with open(os.path.join(data_dir, POS_TAGGED_CAPTIONS_FILENAME)) as f:
        image2pos_metas = json.load(f)
    with open(os.path.join(data_dir, DEP_TAGGED_CAPTIONS_FILENAME)) as f:
        image2dep_metas = json.load(f)
    with open(os.path.join(data_dir, HEAD_TAGGED_CAPTIONS_FILENAME)) as f:
        image2head_metas = json.load(f)
    for id_, metas in image2lemma_metas.items():
        image2metas[id_] = dict()
        image2metas[id_]["lemma_captions"] = metas[TAGGED_CAPTIONS]
        image2metas[id_]["pos_captions"] = image2pos_metas[id_][TAGGED_CAPTIONS]
        image2metas[id_]["dep_captions"] = image2dep_metas[id_][TAGGED_CAPTIONS]
        image2metas[id_]["head_captions"]=image2head_metas[id_][TAGGED_CAPTIONS]
        image2metas[id_][DATA_COCO_SPLIT] = metas[DATA_COCO_SPLIT]
    captions_per_image = len(image2metas[id_]["lemma_captions"])

    if concept_type == "adj-noun":
        others = ADJECTIVES
        other_occurrences = ADJECTIVE_OCCURRENCES
    elif concept_type == "verb-noun":
        others = VERBS
        other_occurrences = VERB_OCCURRENCES
    else:
        raise ValueError("Invalid --concept-type. Options: adj-noun, verb-noun")
    
    print("Looking for pairs: {} - {}...".format(first_other, first_noun))
    data = dict()
    data[NOUNS] = noun_synonyms
    data[others] = other_synonyms
    
    noun_synonyms = set(noun_synonyms)
    other_synonyms = set(other_synonyms)
    occurrence_data = dict()
    for coco_id, metas in tqdm(image2metas.items()):
        occurrence_data[coco_id] = dict()
        occurrence_data[coco_id][NOUN_OCCURRENCES] = 0
        occurrence_data[coco_id][other_occurrences] = 0
        occurrence_data[coco_id][PAIR_OCCURRENCES] = 0
        occurrence_data[coco_id][DATA_COCO_SPLIT] = metas[DATA_COCO_SPLIT]

        for ix, lemma_caption in enumerate(metas["lemma_captions"]):
            caption_metas = dict()
            caption_metas["lemma"] = lemma_caption
            caption_metas["pos"] = metas["pos_captions"][ix]
            caption_metas["dep"] = metas["dep_captions"][ix]
            caption_metas["head"] = metas["head_captions"][ix]
            has_noun, has_other, has_combination = has_concept_pair(
                caption_metas, noun_synonyms, other_synonyms, concept_type)
            if has_noun:
                occurrence_data[coco_id][NOUN_OCCURRENCES] += 1
            if has_other:
                occurrence_data[coco_id][other_occurrences] += 1
            if has_combination:
                occurrence_data[coco_id][PAIR_OCCURRENCES] += 1

    data[OCCURRENCE_DATA] = occurrence_data

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    data_path = os.path.join(out_dir, "{}_{}.json".format(first_other, 
                                                          first_noun))
    print("Saving results to {}".format(data_path))
    with open(data_path, "w") as f:
        json.dump(data, f)

    for ncaptions in range(1, captions_per_image+1):
        noun_occurences = len([d for d in occurrence_data.values() if d[NOUN_OCCURRENCES] >= ncaptions])
        other_occurences = len([d for d in occurrence_data.values() if d[other_occurrences] >= ncaptions])
        pair_occurences = len([d for d in occurrence_data.values() if d[PAIR_OCCURRENCES] >= ncaptions])

        print("Found {}\timages where the noun occurs in at least {} captions"\
              .format(noun_occurences, ncaptions))
        print("Found {}\timages where the other occurs in at least {} captions"\
              .format(other_occurences, ncaptions))
        print("Found {}\timages where the pair occurs in at least {} captions"\
              .format(pair_occurences, ncaptions))
