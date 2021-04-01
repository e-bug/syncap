import nltk
from nltk import pos_tag
from nltk.corpus import conll2000
from nltk.chunk import conlltags2tree, tree2conlltags
from toolkit.util.data.syntax.annotator import SyntaxAnnotator
from toolkit.util.data.syntax.chunker import ClassifierChunkParser
from toolkit.utils import DATA_CAPTIONS, DATA_COCO_SPLIT, TAGGED_CAPTIONS


class NLTKAnnotator(SyntaxAnnotator):
    def __init__(self):
        super().__init__()
        nltk.download("conll2000")
        nltk.download("averaged_perceptron_tagger")
        data = conll2000.chunked_sents()
        train_data = data[:10900]
        self.model = ClassifierChunkParser(train_data) 

    def annotate(self, image2metas, data_dir, syntax_type="chunk"):
        image2syntax_metas = dict()
        for coco_id, metas in image2metas.items():
            tag_captions = []
            for caption in metas[DATA_CAPTIONS]:
                tag_caption = tree2conlltags(self.model.parse(pos_tag(caption)))
                tag_captions.append([elem[-1] for elem in tag_caption])
            image2syntax_metas[coco_id] = {
                TAGGED_CAPTIONS: tag_captions,
                DATA_COCO_SPLIT: metas[DATA_COCO_SPLIT],
            }
        return image2syntax_metas

