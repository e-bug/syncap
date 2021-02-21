import os.path
import json

import torch
from torch import nn
import torchvision

from toolkit.utils import LSTMCell, load_pretrained_embedding_from_file
from toolkit.models.captioning_model import (
    CaptioningEncoderDecoderModel, CaptioningEncoder, CaptioningDecoder
)
from toolkit.utils import WORD_MAP_FILENAME


class SATModel(CaptioningEncoderDecoderModel):
    def __init__(self, args):
        super(SATModel, self).__init__()

        # Read word map
        word_map_filename = os.path.join(args.dataset_splits_dir, WORD_MAP_FILENAME)
        with open(word_map_filename) as f:
            word_map = json.load(f)

        # Pre-trained Embeddings
        if args.embeddings_path:
            embeddings, embed_dim = load_pretrained_embedding_from_file(args.embeddings_path, word_map)
        else:
            embeddings, embed_dim = None, args.embeddings_dim

        self.encoder = SATEncoder(
            encoded_image_size=args.encoded_image_size,
            encoder_training=args.encoder_training,
        )

        self.decoder = SATDecoder(
            word_map=word_map,
            embed_dim=embed_dim,
            encoder_output_dim=args.encoder_output_dim,
            pretrained_embeddings=embeddings,
            embeddings_freeze=args.embeddings_freeze,
            hidden_size=args.decoder_hidden_dim,
            attention_dim=args.attention_dim,
            dropout=args.dropout,
        )

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser"""
        group = parser.add_argument_group("Show, Attend and Tell")
        group.add_argument("--regularization", default="doubly_stochastic_attention")
        group.add_argument("--alpha-c", type=float, default=1.0,
                           help="Regularization parameter for doubly stochastic attention")

        group.add_argument("--encoded-image-size", default=14)
        group.add_argument("--encoder-output-dim", default=2048)
        group.add_argument("--encoder-training", default="freeze", 
                           choices=["freeze", "finetune", "train"])
        group.add_argument("--encoder-learning-rate", default=1e-4)

        group.add_argument("--embeddings-dim", default=512)
        group.add_argument("--attention-dim", default=512)
        group.add_argument("--decoder-hidden-dim", default=512)
        group.add_argument("--dropout", default=0.5)
        group.add_argument("--teacher-forcing", type=float, default=1)
        group.add_argument("--embeddings-freeze", default=False)
        group.add_argument("--decoder-learning-rate", default=4e-4)

        return group


class SATEncoder(CaptioningEncoder): 
    def __init__(self, encoded_image_size, encoder_training="freeze"):
        super(SATEncoder, self).__init__(encoded_image_size, encoder_training)

        resnet = torchvision.models.resnet152(pretrained=True)

        # Remove linear and pool layers, these are only used for classification
        modules = list(resnet.children())[:-2]
        self.model = nn.Sequential(*modules)

        # Resize input image to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.encoded_image_size, self.encoded_image_size))

        self.train_encoder()

    def forward(self, images):
        """
        Forward propagation.

        :param images: input images, shape: (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.model(images)  # output shape: (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out) # output shape: (batch_size, 2048, enc_image_size, enc_image_size)
        out = out.permute(0, 2, 3, 1)  # output shape: (batch_size, enc_image_size, enc_image_size, 2048)
        return out

    def finetune(self):
        """
        Enable or disable the computation of gradients for the convolutional blocks 2-4 of the encoder.
        :param enable_fine_tuning: Set to True to enable fine tuning
        """
        self.freeze()
        # The convolutional blocks 2-4 are found at position 5-7 in the model
        for c in list(self.model.children())[5:]:
            for p in c.parameters():
                p.requires_grad = True


class SATDecoder(CaptioningDecoder):
    def __init__(self, word_map, embed_dim=512, encoder_output_dim=2048,
                 pretrained_embeddings=None, embeddings_freeze=False,
                 hidden_size=512, attention_dim=512, dropout=0.1):            
        super(SATDecoder, self).__init__(word_map, embed_dim, encoder_output_dim,
                                         pretrained_embeddings, embeddings_freeze)
                
        self.hidden_size = hidden_size
        self.attention_dim = attention_dim
        self.dropout_ratio = dropout

        # LSTM layers
        self.decode_step = LSTMCell(
            input_size=encoder_output_dim + self.embed_dim, # if layer == 0 else hidden_size,
            hidden_size=hidden_size, 
            bias=True
        )
        #self.layers = nn.ModuleList([
        #    LSTMCell(input_size=encoder_output_dim + embed_dim if layer == 0 else hidden_size,
        #             hidden_size=hidden_size, bias=True)
        #    for layer in range(num_layers)
        #])

        # Attention Layer
        self.attention = AttentionLayer(encoder_output_dim, hidden_size, attention_dim)

        # Linear layers to find initial states of LSTMs
        self.init_h = nn.Linear(encoder_output_dim, hidden_size)
        self.init_c = nn.Linear(encoder_output_dim, hidden_size)

        # Gating scalars and sigmoid layer (cf. section 4.2.1 of the paper)
        self.f_beta = nn.Linear(hidden_size, encoder_output_dim)
        self.sigmoid = nn.Sigmoid()

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # Linear layers for output generation
        self.linear_o = nn.Linear(embed_dim, self.vocab_size)
        self.linear_h = nn.Linear(hidden_size, self.embed_dim)
        self.linear_z = nn.Linear(encoder_output_dim, self.embed_dim)

    def init_hidden_states(self, encoder_out):
        """
        Create the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, shape: (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_hidden_dim)
        c = self.init_c(mean_encoder_out)
        states = [h, c]
        return states

    def forward_step(self, encoder_output, prev_word_embeddings, states):
        """Perform a single decoding step."""

        # Initialize LSTM state
        h, c = states 

        # Attention Layer
        attention_weighted_encoding, alpha = self.attention(encoder_output, h)
        gating_scalars = self.sigmoid(self.f_beta(h))
        attention_weighted_encoding = gating_scalars * attention_weighted_encoding

        decoder_input = torch.cat((prev_word_embeddings, attention_weighted_encoding), dim=1)
        h, c = self.decode_step(decoder_input, (h, c))

        h_embedded = self.linear_h(h)
        attention_weighted_encoding_embedded = self.linear_z(attention_weighted_encoding)
        scores = self.linear_o(self.dropout(prev_word_embeddings + \
                                            h_embedded + \
                                            attention_weighted_encoding_embedded))

        states = [h, c]
        return scores, states, alpha


#    def loss(self, scores, target_captions, decode_lengths, alphas):
#        loss = self.loss_cross_entropy(scores, target_captions, decode_lengths)
#
#        # Add doubly stochastic attention regularization
#        loss += self.params["alpha_c"] * ((1.0 - alphas.sum(dim=1)) ** 2).mean()
#        return loss


class AttentionLayer(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(AttentionLayer, self).__init__()

        # Linear layer to transform encoded image
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        # Linear layer to transform decoder's output
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        # Linear layer to calculate values to be softmax-ed
        self.full_att = nn.Linear(attention_dim, 1)
        # ReLU layer
        self.relu = nn.ReLU()
        # Softmax layer to calculate attention weights
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, shape: (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, shape: (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # output shape: (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # output shape: (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # output shape: (batch_size, num_pixels)
        alpha = self.softmax(att)  # output shape: (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # output shape: (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


#def LSTMCell(input_size, hidden_size, **kwargs):
#    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
#    for name, param in m.named_parameters():
#        if 'weight' in name or 'bias' in name:
#            param.data.uniform_(-0.1, 0.1)
#    return m

#def base_architecture(args):
#    args.regularization = getattr(args, 'regularization', 'doubly_stochastic_attention')
#    args.alpha_c = getattr(args, 'alpha_c', 1.0)
#
#    args.encoded_image_size = getattr(args, 'encoded_image_size', 14)
#    args.encoder_output_dim = getattr(args, 'encoder_output_dim', 2048)
#    args.encoder_training = getattr(args, 'encoder_training', 'freeze')
#    args.encoder_learning_rate = getattr(args, 'encoder_learning_rate', 1e-4)
#
#    args.embeddings_dim = getattr(args, 'embeddings_dim', 512)
#    args.attention_dim = getattr(args, 'attention_dim', 512)
#    args.decoder_hidden_dim = getattr(args, 'decoder_hidden_dim', 512)
#    args.teacher_forcing = getattr(args, 'teacher_forcing', 1)
#    args.dropout = getattr(args, 'dropout', 0.5)
#    args.max_caption_len = getattr(args, 'max_caption_len', 20)
#    args.embeddings_freeze = getattr(args, 'embeddings_freeze', False)
#    args.decoder_learning_rate = getattr(args, 'decoder_learning_rate', 4e-4)

