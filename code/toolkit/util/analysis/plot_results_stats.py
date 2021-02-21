"""Plot some statistics on the nouns of the captions generated by a model"""

import sys
from collections import Counter

import argparse
import json

import matplotlib.pyplot as plt


def plot_noun_stats_results(eval_data):
    with open(eval_data, "r") as json_file:
        eval_data = json.load(json_file)

    for noun, stats in eval_data.items():

        adjective_frequencies = Counter(stats["adjective_frequencies"])
        verb_frequencies = Counter(stats["verb_frequencies"])

        total = sum(adjective_frequencies.values())
        no_adjective_freq = int(adjective_frequencies["No adjective"] / total * 100)
        del adjective_frequencies["No adjective"]

        total = sum(verb_frequencies.values())
        no_verb_freq = int(verb_frequencies["No verb"] / total * 100)
        del verb_frequencies["No verb"]

        fig, axes = plt.subplots(nrows=1, figsize=(30, 15))
        plt.suptitle("{}".format(noun) + " ({} captions)".format(total))

        axes.bar(
            [adj for adj, freq in adjective_frequencies.most_common(20)],
            [freq for adj, freq in adjective_frequencies.most_common(20)],
        )
        axes.set_title(
            "Adjectives (captions w/o adjective: {}%)".format(no_adjective_freq)
        )

        plt.show()


def check_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--eval-file", help="File containing evaluation data (as generated by eval.py)"
    )
    parsed_args = parser.parse_args(args)
    print(parsed_args)
    return parsed_args


if __name__ == "__main__":
    parsed_args = check_args(sys.argv[1:])
    plot_noun_stats_results(parsed_args.eval_file)