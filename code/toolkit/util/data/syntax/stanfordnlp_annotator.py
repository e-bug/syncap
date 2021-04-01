import json
import os.path
import stanfordnlp
from tqdm import tqdm

from toolkit.util.data.syntax.annotator import SyntaxAnnotator
from toolkit.utils import (
    STANFORDNLP_DIR, 
    STANFORDNLP_ANNOTATIONS_FILENAME,
    DATA_CAPTIONS,
    CAPTIONS_META,
    TAGGED_CAPTIONS,
    DATA_COCO_SPLIT,
)


PUNCTUATIONS = {"''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-",
                ".", "?", "!", ",", ":", "-", "--", "...", ";"}


class StanfordNLPAnnotator(SyntaxAnnotator):
    def __init__(self, config=dict()):
        super().__init__()
        self.field2idx = {'id': 0, 'word': 1, 'lemma': 2, 'upos': 3, 'xpos': 4,
                          'feats': 5, 'head': 6, 'deprel': 7, 'deps': 8, 'misc': 9}
        self.type2field = {"pos": "upos", "dep": "deprel"}
        self.model = stanfordnlp.Pipeline(lang="en", models_dir=STANFORDNLP_DIR, **config)

    def _get_tag(self, token_meta, syntax_type):
        syntax_field = self.type2field.get(syntax_type, syntax_type)
        return token_meta.split('\t')[self.field2idx[syntax_field]]

    def annotate(self, image2metas, data_dir, syntax_type="pos"):
        ann_fn = os.path.join(data_dir, STANFORDNLP_ANNOTATIONS_FILENAME)
        if not os.path.exists(ann_fn):
            # Annotate captions and store them into disk
            print("Annotating captions...")
            coco_ids = list(image2metas.keys())
            captions = [caption for d in image2metas.values()
                        for caption in d[DATA_CAPTIONS]]
            captions_per_image = len(image2metas[coco_ids[0]][DATA_CAPTIONS])

            doc = self.model(captions)
            ann_captions = doc.conll_file.conll_as_string().strip().split("\n\n")
            assert len(captions) == len(ann_captions)

            ann_data = dict()
            for ix in range(0, len(ann_captions), captions_per_image):
                coco_id = coco_ids[ix // captions_per_image]
                img_sents = ann_captions[ix: ix + captions_per_image]
                img_meta_captions = [sent.split("\n") for sent in img_sents]
                ann_data[coco_id] = {
                    CAPTIONS_META: img_meta_captions,
                    DATA_COCO_SPLIT: image2metas[coco_id][DATA_COCO_SPLIT],
                }

            with open(ann_fn, "w") as f:
                json.dump(ann_data, f)
        else:
            # Read file with annotations
            with open(ann_fn) as f:
                ann_data = json.load(f)

        # Extract syntactic tag from annotations
        image2syntax_metas = dict()
        for coco_id, d in ann_data.items():
            tag_captions = [[self._get_tag(tok_meta, syntax_type)
                             for tok_meta in ann_caption]
                            for ann_caption in d[CAPTIONS_META]]
            image2syntax_metas[coco_id] = {
                TAGGED_CAPTIONS: tag_captions, 
                DATA_COCO_SPLIT: d[DATA_COCO_SPLIT],
            }
        return image2syntax_metas

    # TODO parallelize: run on all corpus and then make sure all words in original caption are in sent
    def tokenize(self, captions, clean=False):
        tok_sents = []
        for caption in tqdm(captions):
            doc = self.model(caption)
            tok_sent = []
            for sent in doc.conll_file.conll_as_string().strip().split("\n\n"):
                for tok in sent.split("\n"):
                    fields = tok.split("\t")
                    word = fields[1]
                    if not clean:
                        tok_sent.append(word)
                    elif not (word in PUNCTUATIONS or fields[4] in PUNCTUATIONS):
                        # Remove punctuation
                        tok_sent.append(word)
            tok_sents.append(tok_sent)
        return tok_sents
