#!/bin/bash

BIN_DIR=$HOME/bin
mkdir -p $BIN_DIR

source activate syncap


# Install StanfordNLP
dir="$BIN_DIR/stanfordnlp_resources"
if [ ! -d "$dir" ]; then
  python -c """
import stanfordnlp; \
stanfordnlp.download('en', resource_dir='$dir', version='0.2.0', force=True)
  """
fi

# Install NLTK
python -c """
import nltk; \
nltk.download('punkt', quiet=True); \
nltk.download('conll2000'); nltk.download('averaged_perceptron_tagger') 
"""

# Install depccg
dir="$BIN_DIR/depccg_resources"
if [ ! -d "$dir" ]; then
  mkdir -p $dir
  gdown -O $dir/lstm_parser_elmo_finetune.tar.gz \
    https://drive.google.com/uc?id=1UldQDigVq4VG2pJx9yf3krFjV0IYOwLr
fi

#./code/coco_caption/get_stanford_models.sh


conda deactivate
