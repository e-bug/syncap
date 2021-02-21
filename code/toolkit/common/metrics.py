"""Metrics for the image captioning task"""

import os
import json
from tqdm import tqdm
from collections import Counter

import matplotlib.pyplot as plt

import stanfordnlp
import numpy as np

from pycocotools.coco import COCO
from coco_caption.pycocoevalcap.bleu.bleu import Bleu
from coco_caption.pycocoevalcap.eval import COCOEvalCap

from toolkit.util.data.composition import has_concept_pair, get_adjectives_for_noun, get_verbs_for_noun
from toolkit.util.data.split import get_occurrences_splits
from toolkit.utils import (
    NOUNS,
    VERBS,
    ADJECTIVES,
    OCCURRENCE_DATA,
    PAIR_OCCURRENCES,
    decode_caption,
    rm_caption_special_tokens,
    STANFORDNLP_DIR
)

base_dir = os.path.dirname(os.path.abspath(__file__))


def coco_metrics(generated_captions_fn, annotations_dir, split):
    # Read generated captions
    # resFile = '/home/plz563/projects/syncap/experiments/coco_karpathy/butd/results_best_beam_5_test.json'
    # annotations_dir = '/home/plz563/data/coco2014/captions/annotations_trainval2014'
    ann_fn = "{}/annotations/captions_{}.json".format(annotations_dir, split)
    coco = COCO(ann_fn)
    cocoRes = coco.loadRes(generated_captions_fn)

    # create cocoEval object by taking coco and cocoRes
    cocoEval = COCOEvalCap(coco, cocoRes)

    # evaluate on a subset of images by setting
    # cocoEval.params['image_id'] = cocoRes.getImgIds()
    # please remove this line when evaluating the full validation set
    cocoEval.params['image_id'] = cocoRes.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    cocoEval.evaluate()

    # print output evaluation scores
    for metric, score in cocoEval.eval.items():
        print('%s: %.3f' % (metric, 100 * score))
    return cocoEval.eval


def calc_bleu(generated_captions_fn, target_captions_fn):

    with open(generated_captions_fn) as f:
        generated_captions = json.load(f)
    with open(target_captions_fn) as f:
        target_captions = json.load(f)
    id2caption = {meta['image_id']: [meta['caption']] for meta in generated_captions}
    id2targets = {meta['image_id']: meta['captions'] for meta in target_captions}

    bleu4 = Bleu(n=4)
    bleu_scores, _ = bleu4.compute_score(id2targets, id2caption)
    bleu_scores = [float("%.2f" % elem) for elem in bleu_scores]
    print("BLEU scores:", bleu_scores)
    return bleu_scores


def recall_pairs(generated_captions_fn, occurrences_dir, heldout_pairs, split):
    with open(generated_captions_fn) as f:
        generated_captions = json.load(f)
    id2captions = {meta['image_id']: meta['captions'] for meta in generated_captions}

    config = {'use_gpu': False, 'tokenize_pretokenized': True}
    nlp_pipeline = stanfordnlp.Pipeline(lang='en', models_dir=STANFORDNLP_DIR, **config)

    recall_scores = {}
    for pair in heldout_pairs:
        occurrences_fn = os.path.join(occurrences_dir, pair + ".json")
        occurrences_data = json.load(open(occurrences_fn, "r"))

        _, val_indices, test_indices = get_occurrences_splits([occurrences_fn])
        if split == "val2014":
            eval_indices = test_indices
        else:
            eval_indices = val_indices

        nouns = set(occurrences_data[NOUNS])
        if ADJECTIVES in occurrences_data:
            others = set(occurrences_data[ADJECTIVES])
            concept_type = "adj-noun"
        elif VERBS in occurrences_data:
            others = set(occurrences_data[VERBS])
            concept_type = "verb-noun"
        else:
            raise ValueError("No adjectives or verbs found in occurrences data!")
        recall_score = calc_recall(id2captions, eval_indices, nouns, others, concept_type,
                                   occurrences_data, nlp_pipeline)

        pair = os.path.basename(occurrences_fn).split(".")[0]
        recall_scores[pair] = recall_score
    # average_pair_recall = np.sum(list(recall_score["true_positives"].values())) / \
    #                       np.sum(list(recall_score["numbers"].values()))
    # logging.info("{}: {}".format(pair, np.round(average_pair_recall, 2)))

    # logging.info("Average: {}".format(average_recall(recall_scores)))
    # json.dump(recall_scores, open(output_file_name, "w"))
    return recall_scores


def calc_recall(generated_captions, eval_indices, nouns, others, concept_type, occurrences_data, nlp_pipeline):
    true_positives = dict.fromkeys(["N=1", "N=2", "N=3", "N=4", "N=5"], 0)
    numbers = dict.fromkeys(["N=1", "N=2", "N=3", "N=4", "N=5"], 0)
    adjective_frequencies = Counter()
    verb_frequencies = Counter()
    for coco_id in eval_indices:
        top_k_captions = generated_captions[int(coco_id)]
        count = occurrences_data[OCCURRENCE_DATA][coco_id][PAIR_OCCURRENCES]
        hit = False
        for caption in top_k_captions:
            if caption == "":
                continue
            tagged_caption = nlp_pipeline(caption).sentences[0]
            caption_metas = {"lemma": [w.lemma for w in tagged_caption.words],
                             "pos": [w.upos for w in tagged_caption.words],
                             "dep": [w.dependency_relation for w in tagged_caption.words],
                             "head": [w.governor for w in tagged_caption.words]
                             }
            _, _, contains_pair = has_concept_pair(caption_metas, nouns, others, concept_type)
            if contains_pair:
                hit = True

            noun_is_present = False
            for word in tagged_caption.words:
                if word.lemma in nouns:
                    noun_is_present = True
            if noun_is_present:
                adjectives = get_adjectives_for_noun(caption_metas, nouns)
                if len(adjectives) == 0:
                    adjective_frequencies["No adjective"] += 1
                adjective_frequencies.update(adjectives)

                verbs = get_verbs_for_noun(caption_metas, nouns)
                if len(verbs) == 0:
                    verb_frequencies["No verb"] += 1
                verb_frequencies.update(verbs)

        true_positives["N={}".format(count)] += int(hit)
        numbers["N={}".format(count)] += 1

    recall_score = {
        "true_positives": true_positives,
        "numbers": numbers,
        "adjective_frequencies": adjective_frequencies,
        "verb_frequencies": verb_frequencies,
    }
    return recall_score


def average_recall(recall_scores, min_importance=1):
    pair_recalls_summed = 0
    length = 0

    for i, pair in enumerate(recall_scores.keys()):
        average_pair_recall = np.sum(list(recall_scores[pair]["true_positives"].values())[min_importance - 1:]) / \
                              np.sum(list(recall_scores[pair]["numbers"].values())[min_importance - 1:])
        if not np.isnan(average_pair_recall):
            pair_recalls_summed += average_pair_recall
            length += 1

    recall = pair_recalls_summed / length
    return recall


def mrr_pairs(generated_captions_fn, occurrences_dir, heldout_pairs, split): # TODO
    with open(generated_captions_fn) as f:
        generated_captions = json.load(f)
    id2captions = {meta['image_id']: meta['captions'] for meta in generated_captions}

    config = {'use_gpu': False, 'tokenize_pretokenized': True}
    nlp_pipeline = stanfordnlp.Pipeline(lang='en', models_dir=STANFORDNLP_DIR, **config)

    mrr_scores = {}
    for pair in heldout_pairs:
        occurrences_fn = os.path.join(occurrences_dir, pair + ".json")
        occurrences_data = json.load(open(occurrences_fn, "r"))

        _, val_indices, test_indices = get_occurrences_splits([occurrences_fn])
        if split == "val2014":
            eval_indices = test_indices
        else:
            eval_indices = val_indices

        nouns = set(occurrences_data[NOUNS])
        if ADJECTIVES in occurrences_data:
            others = set(occurrences_data[ADJECTIVES])
            concept_type = "adj-noun"
        elif VERBS in occurrences_data:
            others = set(occurrences_data[VERBS])
            concept_type = "verb-noun"
        else:
            raise ValueError("No adjectives or verbs found in occurrences data!")
        mrr_score = calc_mrr(id2captions, eval_indices, nouns, others, concept_type, occurrences_data, nlp_pipeline)

        pair = os.path.basename(occurrences_fn).split(".")[0]
        mrr_scores[pair] = mrr_score
    return mrr_scores


def calc_mrr(generated_captions, eval_indices, nouns, others, concept_type, occurrences_data, nlp_pipeline):
    ranks = dict.fromkeys(["N=1", "N=2", "N=3", "N=4", "N=5"], 0.0)
    numbers = dict.fromkeys(["N=1", "N=2", "N=3", "N=4", "N=5"], 0)
    for coco_id in eval_indices:
        top_k_captions = generated_captions[int(coco_id)]
        count = occurrences_data[OCCURRENCE_DATA][coco_id][PAIR_OCCURRENCES]
        rank = 0
        for ix, caption in enumerate(top_k_captions):
            if caption == "":
                continue
            tagged_caption = nlp_pipeline(caption).sentences[0]
            caption_metas = {"lemma": [w.lemma for w in tagged_caption.words],
                             "pos": [w.upos for w in tagged_caption.words],
                             "dep": [w.dependency_relation for w in tagged_caption.words],
                             "head": [w.governor for w in tagged_caption.words]
                             }
            _, _, contains_pair = has_concept_pair(caption_metas, nouns, others, concept_type)
            if contains_pair and rank == 0:
                rank = 1/(ix+1)

        ranks["N={}".format(count)] += rank
        numbers["N={}".format(count)] += 1

    mrr_score = {
        "ranks": ranks,
        "numbers": numbers,
    }
    return mrr_score


def average_mrr(mrr_scores, min_importance=1):
    pair_mrrs_summed = 0
    length = 0

    for i, pair in enumerate(mrr_scores.keys()):
        average_pair_mrr = np.sum(list(mrr_scores[pair]["ranks"].values())[min_importance - 1:]) / \
                           np.sum(list(mrr_scores[pair]["numbers"].values())[min_importance - 1:])
        if not np.isnan(average_pair_mrr):
            pair_mrrs_summed += average_pair_mrr
            length += 1

    recall = pair_mrrs_summed / length
    return recall


def beam_occurrences(generated_beams, beam_size, word_map, heldout_pairs, max_print_length=20):
    for pair in heldout_pairs:
        occurrences_data_file = os.path.join(base_dir, "data", "occurrences", pair + ".json")
        occurrences_data = json.load(open(occurrences_data_file, "r"))

        _, _, test_indices = get_occurrences_splits([pair])

        nouns = set(occurrences_data[NOUNS])
        if ADJECTIVES in occurrences_data:
            adjectives = set(occurrences_data[ADJECTIVES])
        if VERBS in occurrences_data:
            verbs = set(occurrences_data[VERBS])

        max_length = max([beams[-1].size(1) for beams in generated_beams.values()])
        noun_occurrences = np.zeros(max_length)
        other_occurrences = np.zeros(max_length)
        pair_occurrences = np.zeros(max_length)

        num_beams = np.zeros(max_length)
        for coco_id in test_indices:
            beam = generated_beams[coco_id]
            for step, beam_timestep in enumerate(beam):
                noun_match = False
                other_match = False
                pair_match = False
                for branch in beam_timestep:
                    branch_words = set(decode_caption(branch.numpy(), word_map))
                    noun_occurs = bool(nouns & branch_words)

                    if ADJECTIVES in occurrences_data:
                        adjective_occurs = bool(adjectives & branch_words)
                        other_match = adjective_occurs
                    elif VERBS in occurrences_data:
                        verb_occurs = bool(verbs & branch_words)
                        other_match = verb_occurs
                    noun_match = noun_occurs
                    pair_match = noun_occurs and other_match

                noun_occurrences[step] += int(noun_match)
                other_occurrences[step] += int(other_match)
                pair_occurrences[step] += int(pair_match)
                num_beams[step] += 1

        # Print only occurrences up to max_print_length
        print_length = min(max_print_length, len(np.trim_zeros(num_beams)))

        name = os.path.basename(occurrences_data_file).split(".")[0]
        # logging.info("Beam occurrences for {}".format(name))
        # logging.info("Nouns: {}".format(noun_occurrences[:print_length]))
        # logging.info("Adjectives/Verbs: {}".format(other_occurrences[:print_length]))
        # logging.info("Pairs: {}".format(pair_occurrences[:print_length]))
        # logging.info("Number of beams: {}".format(num_beams[:print_length]))

        steps = np.arange(print_length)
        plt.plot(steps, noun_occurrences[:print_length] / num_beams[:print_length], label="nouns")
        plt.plot(steps, other_occurrences[:print_length] / num_beams[:print_length], label="adjectives/verbs")
        plt.plot(steps, pair_occurrences[:print_length] / num_beams[:print_length], label="pairs")
        plt.legend()
        plt.xlabel("timestep")
        plt.title("Recall@{} for {} in the decoding beam".format(beam_size, name))
        plt.show()
