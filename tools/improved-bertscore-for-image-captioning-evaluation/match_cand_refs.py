import json
import argparse
from collections import defaultdict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--refs_file')
    parser.add_argument('--cand_file')
    parser.add_argument('--output_fn')
    args = parser.parse_args()

    # Refs
    j = json.load(open(args.refs_file))
    anns = j['annotations']
    image2anns = defaultdict(list)
    for ann in anns:
        image2anns[ann['image_id']].append(ann['caption'].strip())

    # Cand
    j = json.load(open(args.cand_file))
    image2cand = defaultdict(list)
    for ann in j:
        image2cand[ann['image_id']].append(ann['caption']) 

    samples = {}
    for ix, img in enumerate(image2cand):
        d = dict()
        d['refs'] = image2anns[img] #[:5]
        d['cand'] = image2cand[img]
        samples[str(ix)] = d

    with open(args.output_fn, 'w') as f:
        json.dump(samples, f)

