import torch
import torch.nn as nn

import json
import os.path
import numpy as np

from toolkit.models.captioning_model import CaptioningEncoderDecoderModel, CaptioningDecoder
from toolkit.utils import (
    WORD_MAP_FILENAME,
    TOKEN_START,
    TOKEN_END,
    l2_norm,
    load_pretrained_embedding_from_file,
    LSTMCell,
    TOKEN_MASK_TAG,
    TOKEN_MASK_WORD,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BUTRWeightModel(CaptioningEncoderDecoderModel):
    def __init__(self, args):
        super(BUTRWeightModel, self).__init__()

        # Read word map
        word_map_filename = os.path.join(args.dataset_splits_dir, WORD_MAP_FILENAME)
        with open(word_map_filename) as f:
            word_map = json.load(f)

        # Pre-trained Embeddings
        if args.embeddings_path:
            embeddings, embed_dim = load_pretrained_embedding_from_file(args.embeddings_path, word_map)
        else:
            embeddings, embed_dim = None, args.embeddings_dim

        self.decoder = TopDownRankingDecoder(
            word_map=word_map,
            embed_dim=embed_dim,
            encoder_output_dim=args.encoder_output_dim,
            pretrained_embeddings=embeddings,
            embeddings_freeze=args.embeddings_freeze,
            joint_embed_dim=args.joint_embeddings_dim,
            language_encoding_lstm_dim=args.language_encoding_lstm_dim,
            image_embeddings_freeze=args.embeddings_freeze,
            caption_embeddings_freeze=args.embeddings_freeze,
            attention_dim=args.attention_dim,
            language_generation_lstm_dim=args.language_generation_lstm_dim,
            dropout=args.dropout,
        )

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser"""
        group = parser.add_argument_group("Bottom-Up Top-down Ranking")

        group.add_argument("--encoder-output-dim", default=2048)

        group.add_argument("--joint-embeddings-dim", default=1024)
        group.add_argument("--embeddings-dim", default=300)
        group.add_argument("--language-encoding-lstm-dim", default=1000)
        group.add_argument("--image-embeddings-freeze", default=False)
        group.add_argument("--caption-embeddings-freeze", default=False)
        group.add_argument("--attention-dim", default=512)
        group.add_argument("--language-generation-lstm-dim", default=1000)
        group.add_argument("--teacher-forcing", default=1)
        group.add_argument("--dropout", default=0.0)
        group.add_argument("--embeddings-freeze", default=False)
        group.add_argument("--decoder-learning-rate", type=float, default=1e-4)

        return group

    @staticmethod
    def get_top_ranked_captions_indices(embedded_image, embedded_captions):
        # Compute similarity of image to all captions
        d = np.dot(embedded_image, embedded_captions.T).flatten()
        inds = np.argsort(d)[::-1]
        return inds


class TopDownRankingDecoder(CaptioningDecoder):
    SHARED_PARAMS = [
       "embeddings.weight",
       "image_embedding.linear_image_embedding_weights.weight",
       "image_embedding.linear_image_embedding_weights.bias",
       "image_embedding.image_embedding.weight",
       "image_embedding.image_embedding.bias",
       "language_encoding_lstm.lstm_cell.weight_ih",
       "language_encoding_lstm.lstm_cell.weight_hh",
       "language_encoding_lstm.lstm_cell.bias_ih",
       "language_encoding_lstm.lstm_cell.bias_hh",
    ]

    def __init__(self, word_map, embed_dim=300, encoder_output_dim=2048,
                 pretrained_embeddings=None, embeddings_freeze=False,
                 joint_embed_dim=1024, language_encoding_lstm_dim=1000, 
                 image_embeddings_freeze=False, caption_embeddings_freeze=False,
                 attention_dim=512, language_generation_lstm_dim=1000,
                 dropout=0.0):
        super(TopDownRankingDecoder, self).__init__(word_map, embed_dim, encoder_output_dim,
                                                    pretrained_embeddings, embeddings_freeze)

        self.joint_embed_dim = joint_embed_dim
        self.language_encoding_lstm_size = language_encoding_lstm_dim
        self.image_embeddings_freeze = image_embeddings_freeze
        self.caption_embeddings_freeze = caption_embeddings_freeze
        self.attention_dim = attention_dim
        self.language_generation_lstm_size = language_generation_lstm_dim

        self.image_embedding = ImageEmbedding(joint_embed_dim, encoder_output_dim, image_embeddings_freeze)
        self.caption_attention = CaptionAttention(language_encoding_lstm_dim)
        self.caption_embedding = nn.Linear(language_encoding_lstm_dim, joint_embed_dim)

        self.language_encoding_lstm = LanguageEncodingLSTM(self.embed_dim, language_encoding_lstm_dim, 
                                                           caption_embeddings_freeze)
        self.language_generation_lstm = LanguageGenerationLSTM(joint_embed_dim, language_encoding_lstm_dim,
                                                               language_generation_lstm_dim)
        self.attention = VisualAttention(joint_embed_dim, language_encoding_lstm_dim, attention_dim)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # Linear layer to find scores over vocabulary
        self.fc = nn.Linear(language_generation_lstm_dim, self.vocab_size, bias=True)

        # Linear layers to find initial states of LSTMs
        self.init_h_lan_gen = nn.Linear(joint_embed_dim, self.language_generation_lstm.lstm_cell.hidden_size)
        self.init_c_lan_gen = nn.Linear(joint_embed_dim, self.language_generation_lstm.lstm_cell.hidden_size)

    def init_hidden_states(self, encoder_out):
        _, v_mean_embedded = self.image_embedding(encoder_out)
        h_lan_enc, c_lan_enc = self.language_encoding_lstm.init_state(v_mean_embedded.size(0))
        
        h_lan_gen = self.init_h_lan_gen(v_mean_embedded)
        c_lan_gen = self.init_c_lan_gen(v_mean_embedded)

        states = [h_lan_enc, c_lan_enc, h_lan_gen, c_lan_gen]
        return states

    def embed_captions(self, captions, decode_lengths):
        # Initialize LSTM state
        batch_size = captions.size(0)
        h_lan_enc, c_lan_enc = self.language_encoding_lstm.init_state(batch_size)

        # Tensor to store hidden activations
        lang_enc_hidden_activations = torch.zeros((batch_size, max(decode_lengths), self.language_encoding_lstm_size), device=device)

        for t in range(max(decode_lengths)):
            prev_words_embedded = self.embeddings(captions[:, t])
            h_lan_enc, c_lan_enc = self.language_encoding_lstm(h_lan_enc, c_lan_enc, prev_words_embedded)
            lang_enc_hidden_activations[decode_lengths >= t + 1, t] = h_lan_enc[decode_lengths >= t + 1]

        captions_attention = self.caption_attention(lang_enc_hidden_activations, decode_lengths)
        captions_embedded = self.caption_embedding(captions_attention)
        captions_embedded = l2_norm(captions_embedded)
        return captions_embedded

    def forward_step(self, encoder_output, prev_word_embeddings, states):
        h_lan_enc, c_lan_enc, h_lan_gen, c_lan_gen = states
        images_embedded, _ = self.image_embedding(encoder_output)

        h_lan_enc, c_lan_enc = self.language_encoding_lstm(h_lan_enc, c_lan_enc, prev_word_embeddings)
        v_hat = self.attention(images_embedded, h_lan_enc)
        h_lan_gen, c_lan_gen = self.language_generation_lstm(h_lan_gen, c_lan_gen, h_lan_enc, v_hat)
        scores = self.fc(self.dropout(h_lan_gen))
        
        states = [h_lan_enc, c_lan_enc, h_lan_gen, c_lan_gen]
        return scores, states, None

    def forward_ranking(self, encoder_output, captions, decode_lengths):
        """
        Forward propagation for the ranking task.

        """
        _, v_mean_embedded = self.image_embedding(encoder_output)
        captions_embedded = self.embed_captions(captions, decode_lengths)
        return v_mean_embedded, captions_embedded

    # ex forward_joint
    def forward(self, encoder_output, target_captions=None, decode_lengths=None,
                teacher_forcing=0.0, mask_prob=0.0, mask_type=None):
        """Forward pass for both ranking and caption generation."""

        batch_size = encoder_output.size(0)

        # Flatten image
        encoder_output = encoder_output.view(batch_size, -1, encoder_output.size(-1))

        # Initialize LSTM states
        states = self.init_hidden_states(encoder_output)
        lang_enc_hidden_activations = None
        if self.training:
            # Tensor to store hidden activations of the language encoding LSTM of the last timestep
            # These will be the caption embedding
            lang_enc_hidden_activations = torch.zeros((batch_size, max(decode_lengths), self.language_encoding_lstm_size), device=device)

        # Tensors to hold word prediction scores
        scores = torch.zeros((batch_size, max(decode_lengths), self.vocab_size), device=device)

        # FOR MULTITASK
        if self.training and target_captions is not None:
            prev_words = torch.ones((batch_size,), dtype=torch.int64, device=device) * target_captions[:, 0]
        else:
            # At the start, all 'previous words' are the <start> token
            prev_words = torch.full((batch_size,), self.word_map[TOKEN_START], dtype=torch.int64, device=device)

        target_clones = target_captions
        if self.training and mask_prob:
            # FOR MASK INTERLEAVED
            target_clones = target_clones.clone()
            tag_ix = self.word_map[TOKEN_MASK_TAG]
            word_ix = self.word_map[TOKEN_MASK_WORD]
            probs = np.random.uniform(0, 1, len(target_captions))
            tochange_ixs = [ix for ix, v in enumerate(probs < mask_prob) if v]
            mask_tag_ixs = np.array([np.random.choice(range(0, l - 1, 2)) for l in decode_lengths.tolist()])
            mask_tag_ixs = mask_tag_ixs[tochange_ixs]
            if mask_type in {"tags", "both"}:
                target_clones[tochange_ixs, mask_tag_ixs+1] = tag_ix
            if mask_type in {"words", "both"}:
                target_clones[tochange_ixs, mask_tag_ixs+2] = word_ix

        for t in range(max(decode_lengths)):

            if not self.training:
                # Find all sequences where an <end> token has been produced in the last timestep
                ind_end_token = torch.nonzero(prev_words == self.word_map[TOKEN_END]).view(-1).tolist()
                # Update the decode lengths accordingly
                decode_lengths[ind_end_token] = torch.min(
                    decode_lengths[ind_end_token],
                    torch.full_like(decode_lengths[ind_end_token], t, device=device),
                )

            # Check if all sequences are finished:
            incomplete_sequences_ixs = torch.nonzero(decode_lengths > t).view(-1)
            if len(incomplete_sequences_ixs) == 0:
                break

            # Forward prop.
            prev_words_embedded = self.embeddings(prev_words)
            scores_for_timestep, states, alphas_for_timestep = \
                self.forward_step(encoder_output, prev_words_embedded, states)

            # Update the previously predicted words
            prev_words = self.update_previous_word(scores_for_timestep, target_clones, t, teacher_forcing)

            scores[incomplete_sequences_ixs, t, :] = scores_for_timestep[incomplete_sequences_ixs]
            if self.training:
                h_lan_enc = states[0]
                lang_enc_hidden_activations[decode_lengths >= t + 1, t] = h_lan_enc[decode_lengths >= t + 1]

        captions_embedded = None
        v_mean_embedded = None
        if self.training:
            _, v_mean_embedded = self.image_embedding(encoder_output)
            captions_attention = self.caption_attention(lang_enc_hidden_activations, decode_lengths)
            captions_embedded = self.caption_embedding(captions_attention)
            captions_embedded = l2_norm(captions_embedded)

        extras = {'images_embedded': v_mean_embedded, 'captions_embedded': captions_embedded}

        return scores, decode_lengths, extras


class LanguageGenerationLSTM(nn.Module):
    def __init__(self, joint_embed_size, language_encoding_lstm_size, hidden_size):
        super(LanguageGenerationLSTM, self).__init__()
        self.lstm_cell = LSTMCell(language_encoding_lstm_size + joint_embed_size, hidden_size, bias=True)

    def forward(self, h2, c2, h_lang_enc, v_hat):
        input_features = torch.cat((h_lang_enc, v_hat), dim=1)
        h_out, c_out = self.lstm_cell(input_features, (h2, c2))
        return h_out, c_out


class VisualAttention(nn.Module):
    def __init__(self, joint_embed_size, language_encoding_lstm_size, hidden_size):
        super(VisualAttention, self).__init__()
        self.linear_image_features = nn.Linear(joint_embed_size, hidden_size, bias=False)
        self.linear_lang_enc = nn.Linear(language_encoding_lstm_size, hidden_size, bias=False)
        self.tanh = nn.Tanh()
        self.linear_attention = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, images_embedded, h_lang_enc):
        image_feats_embedded = self.linear_image_features(images_embedded)
        h_lang_enc_embedded = self.linear_lang_enc(h_lang_enc).unsqueeze(1)

        all_feats_emb = image_feats_embedded + h_lang_enc_embedded.repeat(1, images_embedded.size(1), 1)
        activate_feats = self.tanh(all_feats_emb)
        attention = self.linear_attention(activate_feats)
        normalized_attention = self.softmax(attention)

        weighted_feats = normalized_attention * images_embedded
        attention_weighted_image_features = weighted_feats.sum(dim=1)
        return attention_weighted_image_features


class CaptionAttention(nn.Module):
    # Modeled after https://github.com/juditacs/snippets/blob/master/deep_learning/masked_softmax.ipynb
    def __init__(self, hidden_size):
        super(CaptionAttention, self).__init__()
        self.linear_attention = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, h_lang_enc, decode_lengths):
        # h_lang_enc: B x T x H
        maxlen = max(decode_lengths)
        mask = torch.arange(maxlen, device=device)[None, :] < decode_lengths[:, None]  # B x T

        attention = self.linear_attention(h_lang_enc)  # B x T x 1
        attention[~mask] = float('-inf')
        normalized_attention = self.softmax(attention)  # B x T x 1

        weighted_h_lang_enc = normalized_attention * h_lang_enc  # B x T x H
        attention_weighted_h_lang_enc = weighted_h_lang_enc.sum(dim=1)  # B x H
        return attention_weighted_h_lang_enc


class LanguageEncodingLSTM(nn.Module):
    def __init__(self, word_embeddings_size, hidden_size, embeddings_freeze=False):
        super(LanguageEncodingLSTM, self).__init__()
        self.lstm_cell = LSTMCell(word_embeddings_size, hidden_size, bias=True)
        self.freeze(embeddings_freeze)

    def forward(self, h, c, prev_words_embedded):
        h_out, c_out = self.lstm_cell(prev_words_embedded, (h, c))
        return h_out, c_out

    def init_state(self, batch_size):
        h = torch.zeros((batch_size, self.lstm_cell.hidden_size), device=device)
        c = torch.zeros((batch_size, self.lstm_cell.hidden_size), device=device)
        return [h, c]

    def freeze(self, embeddings_freeze):
        # Disable gradients calculation
        for p in list(self.parameters()):
            p.requires_grad = not embeddings_freeze


class ImageEmbedding(nn.Module):
    def __init__(self, joint_embeddings_size, image_features_size, embeddings_freeze=False):
        super(ImageEmbedding, self).__init__()
        self.linear_image_embedding_weights = nn.Linear(joint_embeddings_size, 1)
        self.softmax = nn.Softmax(dim=1)
        self.image_embedding = nn.Linear(image_features_size, joint_embeddings_size)
        self.freeze(embeddings_freeze)

    def forward(self, encoder_output):
        images_embedded = self.image_embedding(encoder_output)

        weights = self.linear_image_embedding_weights(images_embedded)
        normalized_weights = self.softmax(weights)

        weighted_image_boxes = normalized_weights * images_embedded
        weighted_image_boxes_summed = weighted_image_boxes.sum(dim=1)

        v_mean_embedded = l2_norm(weighted_image_boxes_summed)
        return images_embedded, v_mean_embedded

    def freeze(self, embeddings_freeze):
        # Disable gradients calculation
        for p in list(self.parameters()):
            p.requires_grad = not embeddings_freeze
