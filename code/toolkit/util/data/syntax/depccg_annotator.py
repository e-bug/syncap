from depccg.parser import EnglishCCGParser

from toolkit.util.data.syntax.annotator import SyntaxAnnotator
from toolkit.utils import DATA_CAPTIONS, DATA_COCO_SPLIT, TAGGED_CAPTIONS, DEPCCG_MODEL_FILENAME


class DepCCGAnnotator(SyntaxAnnotator):
    def __init__(self):
        kwargs = dict(
            # A list of binary rules 
            # By default: depccg.combinator.en_default_binary_rules
            binary_rules=None,
            # Penalize an application of a unary rule by adding this value (negative log probability)
            unary_penalty=0.1,
            # Prune supertags with low probabilities using this value
            beta=0.00001,
            # Set False if not prune
            use_beta=True,
            # Use category dictionary
            use_category_dict=True,
            # Use seen rules
            use_seen_rules=True,
            # This also used to prune supertags
            pruning_size=50,
            # Nbest outputs
            nbest=1,
            # Limit categories that can appear at the root of a CCG tree
            # By default: S[dcl], S[wq], S[q], S[qem], NP.
            possible_root_cats=None,
            # Give up parsing long sentences
            max_length=250,
            # Give up parsing if it runs too many steps
            max_steps=100000,
            # You can specify a GPU
            gpu=-1
        )
        self.model = EnglishCCGParser.from_dir(DEPCCG_MODEL_FILENAME, load_tagger=True, **kwargs)

    def annotate(self, image2metas, data_dir, syntax_type="ccg"):
        # Annotate captions and store them into disk
        print("Annotating captions...")
        coco_ids = list(image2metas.keys())
        captions = [" ".join(caption) for d in image2metas.values()
                    for caption in d[DATA_CAPTIONS]]
        captions_per_image = len(image2metas[coco_ids[0]][DATA_CAPTIONS])

        results = self.model.parse_doc(captions)
        ann_captions = []
        for nbests in results:
            for tree, log_prob in nbests:
                v = tree.conll()
                ann_captions.append([(s.split('\t'))[2] for s in v.split('\n') if s != ''])

        image2syntax_metas = dict()
        for ix in range(0, len(ann_captions), captions_per_image):
            coco_id = coco_ids[ix // captions_per_image]
            tag_captions = ann_captions[ix: ix + captions_per_image]
            image2syntax_metas[coco_id] = {
                TAGGED_CAPTIONS: tag_captions,
                DATA_COCO_SPLIT: image2metas[coco_id][DATA_COCO_SPLIT],
            }

        return image2syntax_metas

    def redo(self, img2ix2caption):
        # Annotate captions and store them into disk
        print("Annotating captions...")
        captions = []
        for img, ix2caption in img2ix2caption.items():
            for ix, caption in ix2caption.items():
                captions.append(" ".join(caption))

        results = self.model.parse_doc(captions)
        ann_captions = []
        for nbests in results:
            for tree, log_prob in nbests:
                v = tree.conll()
                ann_captions.append([(s.split('\t'))[2] for s in v.split('\n') if s != ''])

        image2ix2tagcaption = dict()
        i = 0
        for img, ix2caption in img2ix2caption.items():
            image2ix2tagcaption[img] = dict()
            for ix in ix2caption:
                image2ix2tagcaption[img][ix] = ann_captions[i]
                i += 1

        return image2ix2tagcaption
