import argparse

from toolkit.utils import OBJECTIVE_GENERATION, OBJECTIVE_JOINT, OBJECTIVE_MULTI
from toolkit.models.show_attend_tell import SATModel
from toolkit.models.bottom_up_top_down import BUTDModel
from toolkit.models.bottom_up_top_down_ranking import BUTRModel
from toolkit.models.bottom_up_top_down_ranking_mean import BUTRMeanModel
from toolkit.models.bottom_up_top_down_ranking_weight import BUTRWeightModel

abbr2class = {"sat": SATModel, "butd": BUTDModel, "butr": BUTRModel,
              "butr_mean": BUTRMeanModel, "butr_weight": BUTRWeightModel}


def check_args(args):
    parser = argparse.ArgumentParser()
    add_training_args(parser)
    add_data_args(parser)
    add_optimization_args(parser)
    add_checkpoint_args(parser)
    add_model_args(parser)

    parsed_args = parser.parse_args(args)
    return parsed_args


def add_training_args(parser):
    group = parser.add_argument_group("Training")
    group.add_argument('--seed', default=1, type=int,
                        help="Seed")
    group.add_argument('--print-freq', default=100, type=int,
                        help="TODO")
    group.add_argument('--workers', default=1, type=int,
                        help="TODO")
    group.add_argument("--cpu", action="store_true")

    group.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    group.add_argument("--max-epochs", type=int, default=120,
                        help="Maximum number of training epochs")
    group.add_argument("--epochs-early-stopping", type=int, default=5)

    group.add_argument("--max-caption-len", type=int, default=50)

    group.add_argument("--criterion", default="cross_entropy")
    group.add_argument("--regularization", default=None)
    group.add_argument("--generation-criterion", default="cross_entropy")
    group.add_argument("--ranking-criterion", default="contrastive_loss")
    group.add_argument("--gradnorm-criterion", default="l1")

    group.add_argument("--mask-prob", type=float, default=0.0)
    group.add_argument("--mask-type", type=str, default=None, choices=["tags", "words", "both"])

    return group


def add_data_args(parser):
    group = parser.add_argument_group("Data")
    group.add_argument("--image-features-filename",
                        help="Folder where the preprocessed data is located")
    group.add_argument("--dataset-splits-dir",
                        help="Pickled file containing the dataset splits")
    group.add_argument("--embeddings", default=None,
                        help="""Path to a word GloVe embeddings file to be used
                                to initialize the decoder word embeddings""")
    return group


def add_optimization_args(parser):
    group = parser.add_argument_group("Optimization")
    group.add_argument("--objective", default=OBJECTIVE_GENERATION,
                        choices=[OBJECTIVE_GENERATION, OBJECTIVE_JOINT, OBJECTIVE_MULTI],
                        help="Training objective for which the loss is calculated")
    
    group.add_argument("--teacher-forcing", type=float,
                        help="Teacher forcing rate (used in the decoder)")
    group.add_argument("--dropout", type=float,
                        help="Dropout ratio in the decoder")
    
    group.add_argument("--encoder-learning-rate", type=float,
                        help="""Initial learning rate for the encoder
                                (used only if fine-tuning is enabled)""")
    group.add_argument("--decoder-learning-rate", type=float,
                        help="Initial learning rate for the decoder")
    group.add_argument("--grad-clip", type=float, default=10.0,
                        help="Gradient clip")
    
    group.add_argument("--gradnorm-alpha", type=float, default=2.5,
                        help="Gradnorm alpha")
    group.add_argument("--gradnorm-learning-rate", type=float, default=0.01,
                        help="Initial learning rate for the decoder")

    return group


def add_checkpoint_args(parser):
    group = parser.add_argument_group("Checkpointing")
    group.add_argument("--checkpoints-dir", default=None,
                       help="Path to checkpoint of previously trained model")
    group.add_argument("--logging-dir", default="logs",
                       help="Path where to store logs for a run of a model")
    return group


def add_model_args(parser):
    group = parser.add_argument_group("Model")
    group.add_argument("--embeddings-path", default=None,
                       help="""Path to a word GloVe embeddings file to be used
                               to initialize the decoder word embeddings""")
    group.add_argument("--image-normalize", default=None, choices=["imagenet", "scaleimagenet"],
                       help="TODO")

    group.add_argument("--fine-tune-encoder", action="store_true",
                        help="Fine tune the encoder")
    group.add_argument("--dont-fine-tune-word-embeddings", action="store_false",
                        dest="fine_tune_decoder_word_embeddings",
                        help="Do not fine tune the decoder word embeddings")
    group.add_argument("--dont-fine-tune-caption-embeddings", action="store_false",
                        dest="fine_tune_decoder_caption_embeddings",
                        help="Do not fine tune the decoder caption embeddings")
    group.add_argument("--dont-fine-tune-image-embeddings", action="store_false",
                        dest="fine_tune_decoder_image_embeddings",
                        help="Do not fine tune the decoder image embeddings")

    # Add model-specific arguments to the parser
    subparsers = parser.add_subparsers(dest='model')
    subparsers.required = True

    for k, v in abbr2class.items():
        model_parser = subparsers.add_parser(k)
        v.add_args(model_parser)
    
    return group

