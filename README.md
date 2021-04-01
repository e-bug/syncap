# SynCap: Syntactic Planning in Compositional Image Captioning

This is the implementation of the approaches described in the paper:

> Emanuele Bugliarello and Desmond Elliott. 
> [The Role of Syntactic Planning in Compositional Image Captioning](https://arxiv.org/abs/2101.11911). 
> In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics, April 2021.

We provide the code for reproducing our results, processed data and pretrained models.


## Requirements
You can clone this repository with submodules included issuing: <br>
`git clone git@github.com:e-bug/syncap.git`

The requirements can be installed by setting up a conda environment: <br>
`conda env create -f environment.yml` followed by `source activate syncap`

To set up the syntactic taggers, run `bash setup_environment.sh`. 

Finally, install the environments for [M2-Transformer](code/meshed-memory-transformer/environment.yml) and 
[Improved BERTScore](tools/improved-bertscore-for-image-captioning-evaluation/requirements.txt) to use them.


## Data & Models
Check out [`data/README.md`](data/README.md) for links to preprocessed data and data preparation steps.

We also distribute our final [trained models](https://sid.erda.dk/sharelink/GccStABtV6).


## Training and Evaluation

Scripts for training and evaluating each model are provided in the corresponding `experiments/` directory 
(e.g., [`experiments/coco_heldout_1_pos_inter/butr_weight/train.sh`](experiments/coco_heldout_1_pos_inter/butr_weight/train.sh)).

We also provide SLURM wrappers that call the corresponding bash files (e.g., `train.cluster`).

In particular:
- `train.sh`: trains a model
- `val.sh`: generates captions for the validation set
- `score_val.sh`: computes R@5 for compositional generalization and the standard COCO metrics for the generated captions in the validation set
- `bertscore_val.sh`: computes the Improved BERTScore for the generated captions in the validation set ([Yi et al., 2020](https://doi.org/10.18653/v1/2020.acl-main.93))
- `rank_val.sh`: computes image--text retrieval performance for the generated captions in the validation set (ranking models only)
- `diversity_val.sh`: measures diversity metrics for the captions generated in the validation set ([van Miltenburg et al., 2018](https://www.aclweb.org/anthology/C18-1147))


## Description of this repository

- `code/`
  - `eval.py`: Generate captions for a given COCO split
  - `evalrank.py`: Evaluate image--text retrieval performance of ranking modules
  - `options.py`: Hyper-parameters that can be used by each model during training
  - `tag_results.py`: Annotate captions with specified type of syntactic tags
  - `train.py`: Train captioning models
- `data/`: Concept pairs data and data preprocessing (scripts and download links) 
- `experiments/`: Results for each model we trained and scripts to reproduce them
- `notebooks/`: iPython notebooks to analyze trained models
- `tools/`: Third-party software (Improved BERTScore)


## License

This work is licensed under the MIT license. See [`LICENSE`](LICENSE) for details. 
Third-party software and data sets are subject to their respective licenses. <br>
If you find our code/data/models or ideas useful in your research, please consider citing the paper:
```
@inproceedings{bugliarello-elliott-2021-role,
    title = "The Role of Syntactic Planning in Compositional Image Captioning",
    author = "Bugliarello, Emanuele and Elliott, Desmond",
    booktitle = "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics",
    month = apr,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
}
```


## Acknowledgments

Our code builds on top of the following excellent repositories:
- [compositional-image-captioning](https://github.com/mitjanikolaus/compositional-image-captioning)
- [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention)
- [meshed-memory-transformer](https://github.com/aimagelab/meshed-memory-transformer)
- [vsepp](https://github.com/fartashf/vsepp)
- [coco-caption](https://github.com/tylin/coco-caption)
- [improved-bertscore-for-image-captioning-evaluation](https://github.com/ck0123/improved-bertscore-for-image-captioning-evaluation)