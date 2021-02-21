import os
import sys
import json
import argparse
from tqdm import tqdm

# Chunker
import nltk
from nltk import pos_tag
from nltk.corpus import conll2000
from nltk.chunk import conlltags2tree, tree2conlltags
from toolkit.util.data.syntax.chunker import ClassifierChunkParser
nltk.download("conll2000")
nltk.download("averaged_perceptron_tagger")
data = conll2000.chunked_sents()
train_data = data[:10900]
chunker = ClassifierChunkParser(train_data)

# StanfordNLP
import stanfordnlp
from toolkit.utils import STANFORDNLP_DIR
config = {'use_gpu': False, 'tokenize_pretokenized': True}
nlp = stanfordnlp.Pipeline(lang="en", models_dir=STANFORDNLP_DIR, **config)

# DepCCG
from depccg.parser import EnglishCCGParser
from toolkit.utils import DEPCCG_MODEL_FILENAME
config = dict(binary_rules=None, unary_penalty=0.1, beta=0.00001, use_beta=True,
              use_category_dict=True, use_seen_rules=True, pruning_size=50, nbest=1,
              possible_root_cats=None, max_length=250, max_steps=100000, gpu=-1)
depccg = EnglishCCGParser.from_dir(DEPCCG_MODEL_FILENAME, load_tagger=True, **config)


def stanfordnlp_tagger(pretokenized_text, syntax_type):
    doc = nlp(pretokenized_text)
    sents_tags = []
    for sent in doc.sentences:
        sent_tags = []
        for w in sent.words:
            tag = w.upos if syntax_type == 'pos' else w.dependency_relation
            sent_tags.append(tag)
        sents_tags.append(sent_tags)
    return sents_tags


def depccg_tagger(pretokenized_text):
    captions = [" ".join(caption) for caption in pretokenized_text]
    results = depccg.parse_doc(captions)
    ann_captions = []
    for nbests in results:
        for tree, log_prob in nbests:
            v = tree.conll()
            ann_captions.append([(s.split('\t'))[2] for s in v.split('\n') if s != ''])
    return ann_captions


def chunker_tagger(pretokenized_text):
    tag_captions = []
    for caption in pretokenized_text:
        tag_caption = tree2conlltags(chunker.parse(pos_tag(caption)))
        tag_captions.append([elem[-1] for elem in tag_caption])
    return tag_captions


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(args.results_fn) as f:
        id2captions_list = json.load(f)
    id2captions = {d['image_id']: d['captions'] for d in id2captions_list}
    id2words = {k: [[tok for tok in caption.split() if not (tok.startswith('_') or tok.startswith('<'))] 
                    for caption in captions] 
                for k, captions in id2captions.items()}    

    id2silvertags = dict()
    for k, words_list in tqdm(id2words.items()):
        id2silvertags[k] = []
        pretokenized_text = words_list
        if args.syntax_type in {'pos', 'dep'}:
            sents_tags = stanfordnlp_tagger(pretokenized_text, args.syntax_type)
        elif args.syntax_type == 'chunk':
            sents_tags = chunker_tagger(pretokenized_text)
        elif args.syntax_type == 'ccg':
            sents_tags = depccg_tagger(pretokenized_text)
        else:
            raise ValueError("Invalid syntax-type")
        id2silvertags[k] = sents_tags
    
    out_fn = args.results_fn.split("/")[-1].replace("json", "silvertags.json")
    data_path = os.path.join(args.output_dir, out_fn)
    with open(data_path, 'w') as f:
        json.dump(id2silvertags, f)


def check_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-fn",
                        help="Path to JSON file of image -> caption output by the model.")
    parser.add_argument("--output-dir",
                        help="Directory where to store the results.")
    parser.add_argument("--syntax-type", default="pos", choices=["chunk", "pos", "dep", "ccg"],
                        help="Syntactic tag to be used for annotation.")
    parsed_args = parser.parse_args(args)
    return parsed_args 


if __name__ == "__main__":
    args = check_args(sys.argv[1:])
    main(args)
