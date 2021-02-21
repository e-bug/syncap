import random
from data import ImageDetectionsField, SyncapTextField, RawField
from data import HeldoutCOCO, DataLoader
import evaluation
from models.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory
import torch
from tqdm import tqdm
import argparse
import pickle
import numpy as np

import os
import json
from collections import defaultdict

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)


def rm_caption_special_tokens(caption):
    """Remove start, end and padding tokens from encoded caption."""
    tokens = caption.split()
    return " ".join([tok for tok in tokens if not (tok in {"<start>", "<end>", "<pad>"} or tok.startswith("_"))])


def my_tokenize(corpus):
    if isinstance(corpus, list) or isinstance(corpus, tuple):
        if isinstance(corpus[0], list) or isinstance(corpus[0], tuple):
            corpus = {i:c for i, c in enumerate(corpus)}
        else:
            corpus = {i: [c, ] for i, c in enumerate(corpus)}

    # prepare data for PTB Tokenizer
    tokenized_corpus = {}
    image_id = [k for k, v in list(corpus.items()) for _ in range(len(v))]
    sentences = [c for k, v in corpus.items() for c in v]
    lines = sentences

    # create dictionary for tokenized captions
    for k, line in zip(image_id, lines):
        if not k in tokenized_corpus:
            tokenized_corpus[k] = []
        tokenized_caption = line
        tokenized_corpus[k].append(tokenized_caption)

    return tokenized_corpus


def predict_captions(model, dataloader, text_field, max_len=20, beam_size=5, eval_beam_size=1):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(iter(dataloader)):
            #if it < 8:
            #    continue
            #if it > 8:
            #    break
            images = images.to(device)
            with torch.no_grad():
                out, _ = model.beam_search(images, max_len, text_field.vocab.stoi['<end>'], beam_size, out_size=eval_beam_size)
            caps_gen = [text_field.decode(o, join_words=False) for o in out]
            #caps_gen = text_field.decode(out, join_words=False)
            #print(caps_gt, '\n', caps_gen)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = [' '.join([k for k, g in itertools.groupby(gen_l)]) for gen_l in gen_i]
                gen['%d_%d' % (it, i)] = gen_i
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = my_tokenize(gts)  #evaluation.PTBTokenizer.tokenize(gts)
    gen = my_tokenize(gen)  #evaluation.PTBTokenizer.tokenize(gen)
    gts = {k: list(map(rm_caption_special_tokens, v)) for k, v in gts.items()}
    gen = {k: list(map(rm_caption_special_tokens, v)) for k, v in gen.items()}
    #scores, _ = evaluation.compute_scores(gts, gen)
    #return scores
    return gts, gen


if __name__ == '__main__':
    device = torch.device('cuda')

    parser = argparse.ArgumentParser(description='Meshed-Memory Transformer')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--features_path', type=str)
    parser.add_argument('--annotation_folder', type=str)
    parser.add_argument('--id_folder', type=str)
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--eval_beam_size', type=int, default=5)
    parser.add_argument("--max_len", type=int, default=20)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--checkpoint', required=True,
                        help="Path to checkpoint of trained model")
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    args = parser.parse_args()

    print('Meshed-Memory Transformer Evaluation')

    # Pipeline for image regions
    image_field = ImageDetectionsField(detections_path=args.features_path, max_detections=50, load_in_tmp=False)

    # Pipeline for text
    text_field = SyncapTextField(init_token='<start>', eos_token='<end>')

    # Create the dataset
    dataset = HeldoutCOCO(image_field, text_field, 'coco/images/', args.annotation_folder, args.id_folder)
    if args.split == 'val':
        _, test_dataset, _ = dataset.splits
    else:
        _, _, test_dataset = dataset.splits
#    text_field.vocab = pickle.load(open('vocab.pkl', 'rb'))
    text_field.build_vocab(vocab_name='%s/word_map.json' % args.id_folder, min_freq=1)

    # Create imageid2captions dictionary
    id2caps = json.load(open(args.id_folder + '/captions.json'))

    # Model and dataloaders
    encoder = MemoryAugmentedEncoder(3, 0, attention_module=ScaledDotProductAttentionMemory,
                                     attention_module_kwargs={'m': 40})
    decoder = MeshedDecoder(len(text_field.vocab), 108, 3, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<start>'], encoder, decoder).to(device)

    data = torch.load(args.checkpoint)
    model.load_state_dict(data['state_dict'])

    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size, num_workers=args.workers)

    gts_dict, gen_dict = predict_captions(model, dict_dataloader_test, text_field, args.max_len, args.beam_size, args.eval_beam_size)
    #print(scores)

    # Match captions with images
    gts_caps2ix = {' _ '.join(v): k for k, v in gts_dict.items()}
    caps2id = {' _ '.join(list(map(rm_caption_special_tokens, v))): k for k, v in id2caps.items()}
#    with open('gts_caps2ix', 'w') as f:
#        json.dump(gts_caps2ix, f)
#    with open('caps2id', 'w') as f:
#        json.dump(caps2id, f)
    ix2id = {ix: caps2id[caps] for caps, ix in gts_caps2ix.items()}

    target_captions = {ix2id[k]: v for k, v in gts_dict.items()}
    generated_captions = {ix2id[k]: v for k, v in gen_dict.items()}

    # Save results
    outputs_dir = os.path.join(args.output_path, "outputs")
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    split = args.split
    name = split + '.beam_' + str(args.beam_size)

    results = []
    for coco_id, top_k_captions in generated_captions.items():
        caption = top_k_captions[0]
        results.append({"image_id": int(coco_id), "caption": caption})
    results_output_file_name = os.path.join(outputs_dir, name + ".json")
    json.dump(results, open(results_output_file_name, "w"))

    results = []
    for coco_id, top_k_captions in generated_captions.items():
        captions = top_k_captions 
        results.append({"image_id": int(coco_id), "captions": captions})
    results_output_file_name = os.path.join(outputs_dir, name + ".top_%d" % args.eval_beam_size + ".json")
    json.dump(results, open(results_output_file_name, "w"))

    results = []
    for coco_id, all_captions_for_image in target_captions.items():
        captions = all_captions_for_image 
        results.append({"image_id": int(coco_id), "captions": captions})
    results_output_file_name = os.path.join(outputs_dir, split + ".targets.json")
    json.dump(results, open(results_output_file_name, "w"))

