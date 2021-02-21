import os
import sys
import json
import argparse
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')
from toolkit.common.metrics import recall_pairs, coco_metrics, calc_bleu

METRIC_COCO = "coco"
METRIC_BLEU = "bleu"
METRIC_RECALL = "recall"


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    split = args.results_fn.split("/")[-1].split(".")[0]
    for metric in args.metrics:
        output_fn = os.path.join(args.output_dir,
                                 metric + "." + ".".join(args.results_fn.split("/")[-1].split(".")[1:-1]) + "." + split)
        if metric == METRIC_BLEU:
            bleus = calc_bleu(args.results_fn, args.targets_fn)
            with open(output_fn, "w") as f:
                for bleu in bleus:
                    f.write("%f\n" % bleu)
        elif metric == METRIC_RECALL:
            pair2scores = recall_pairs(args.top_results_fn, args.occurrences_dir,
                                       args.heldout_pairs, args.annotations_split)
            k = int(os.path.basename(args.top_results_fn).split("top_")[-1].split(".")[0])
            for p, d in pair2scores.items():
                out_fn = os.path.join(args.output_dir, metric + "_%d" % k + "." + p + "." + ".".join(
                                        args.results_fn.split("/")[-1].split(".")[1:-1]) + "." + split)
                with open(out_fn, "w") as f:
                    json.dump(d, f)
        elif metric == METRIC_COCO:
            metric2score = coco_metrics(args.results_fn, args.annotations_dir, args.annotations_split)
            with open(output_fn, "w") as f:
                for m, score in metric2score.items():
                    f.write("%s: %f\n" % (m, score))
        else:
            raise ValueError("Invalid metric")


def check_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets-fn",
                        help="Path to JSON file of image -> ground-truth captions.")
    parser.add_argument("--results-fn",
                        help="Path to JSON file of image -> caption output by the model.")
    parser.add_argument("--top-results-fn",
                        help="Path to JSON file of image -> top-k captions output by the model. Used for recall.")
    parser.add_argument("--metrics", nargs="+", default="coco", 
                        choices=[METRIC_COCO, METRIC_BLEU, METRIC_RECALL],
                        help="List of metrics to be evaluated (space separated).")
    parser.add_argument("--output-dir",
                        help="Directory where to store the results.")

    # COCO
    parser.add_argument("--annotations-dir",
                        help="Path to COCO 2014 trainval annotations directory.")
    parser.add_argument("--annotations-split", choices=["train2014", "val2014"],
                        help="COCO 2014 trainval annotations split.")

    # Recall
    parser.add_argument("--occurrences-dir",
                        help="Path to concept pair occurrences directory.")
    parser.add_argument("--heldout-pairs", nargs="+",
                        help="List of heldout concept pairs to be evaluated (space separated).")
    
    parsed_args = parser.parse_args(args)
    return parsed_args 


if __name__ == "__main__":
    args = check_args(sys.argv[1:])
    main(args)
