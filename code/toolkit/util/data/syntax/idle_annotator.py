from toolkit.util.data.syntax.annotator import SyntaxAnnotator
from toolkit.utils import (
    TOKEN_IDLE, 
    DATA_CAPTIONS, 
    DATA_COCO_SPLIT, 
    TAGGED_CAPTIONS,
)


class IdleAnnotator(SyntaxAnnotator):
    def __init__(self):
        super().__init__()

    def annotate(self, image2metas, data_dir, syntax_type="idle"):
        image2syntax_metas = dict()
        for coco_id, metas in image2metas.items():
            tag_captions = [[TOKEN_IDLE] * len(caption)
                            for caption in metas[DATA_CAPTIONS]]
            image2syntax_metas[coco_id] = {
                TAGGED_CAPTIONS: tag_captions,
                DATA_COCO_SPLIT: metas[DATA_COCO_SPLIT],
            }
        return image2syntax_metas
 
