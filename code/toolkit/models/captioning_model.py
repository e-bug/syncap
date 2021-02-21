import random
import numpy as np

import torch
from torch import nn

from toolkit.utils import Embedding, load_pretrained_embedding_from_file, WORD_MAP_FILENAME
from toolkit.utils import TOKEN_START, decode_caption, TOKEN_END, TOKEN_PAD, TOKEN_MASK_TAG, TOKEN_MASK_WORD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CaptioningEncoderDecoderModel(nn.Module):
    def __init__(self):
        super(CaptioningEncoderDecoderModel, self).__init__()
        self.encoder = None
        self.decoder = None

    def forward(self, images, encoded_captions=None, caption_lengths=None,
                teacher_forcing=0.0, mask_prob=0.0, mask_type=None):
        if self.encoder:
            encoder_out = self.encoder(images)
        else:
            encoder_out = images
        decoder_out = self.decoder(encoder_out, encoded_captions, caption_lengths,
                                   teacher_forcing, mask_prob, mask_type)
        return decoder_out

    def forward_multi(self, images, encoded_captions=None, caption_lengths=None,
                teacher_forcing=0.0, mask_prob=0.0, mask_type=None):
        if self.encoder:
            encoder_out = self.encoder(images)
        else:
            encoder_out = images
        decoder_out = self.decoder.forward_multi(encoder_out, encoded_captions, caption_lengths,
                                                 teacher_forcing)
        return decoder_out


class CaptioningEncoder(nn.Module):
    def __init__(self, encoded_image_size, encoder_training="freeze"):
        super(CaptioningEncoder, self).__init__()
        self.encoded_image_size = encoded_image_size
        self.encoder_training = encoder_training

    def train_encoder(self):
        """Enable/Disable gradients calculation"""
        if self.encoder_training == "freeze": 
            self.freeze()
        elif self.encoder_training == "finetune":
            self.finetune()
        elif self.encoder_training == "train":
            self.unfreeze()

    def unfreeze(self):
        # Enable gradients calculation
        for p in self.model.parameters():
            p.requires_grad = True

    def freeze(self):
        # Disable gradients calculation
        for p in self.model.parameters():
            p.requires_grad = False

    def finetune(self):
        raise NotImplementedError

    def forward(self, images):
        raise NotImplementedError


class CaptioningDecoder(nn.Module):
    def __init__(self, word_map, embed_dim, encoder_output_dim,
                 pretrained_embeddings=None, embeddings_freeze=False):
        super(CaptioningDecoder, self).__init__()

        self.encoder_output_dim = encoder_output_dim
        self.vocab_size = len(word_map)
        self.word_map = word_map

        # Embedding Layer
        self.embed_dim = embed_dim
        num_embeddings = len(word_map)
        padding_idx = word_map[TOKEN_PAD]
        self.embeddings = Embedding(num_embeddings, embed_dim, padding_idx)
        if pretrained_embeddings:
            self.embeddings.weight = nn.Parameter(pretrained_embeddings)
        self.embeddings.weight.requires_grad = not embeddings_freeze

    def update_previous_word(self, scores, target_words, t, teacher_forcing=0.0):
        if self.training:
            if random.random() < teacher_forcing:
                use_teacher_forcing = True
            else:
                use_teacher_forcing = False
        else:
            use_teacher_forcing = False

        if use_teacher_forcing:
            next_words = target_words[:, t + 1]
        else:
            next_words = torch.argmax(scores, dim=1)

        return next_words

    def forward(self, encoder_output, target_captions=None, decode_lengths=None,
                teacher_forcing=0.0, mask_prob=0.0, mask_type=None):
        """
        Forward propagation.

        :param encoder_output: output features of the encoder
        :param target_captions: encoded target captions, shape: (batch_size, max_caption_length)
        :param decode_lengths: caption lengths, shape: (batch_size, 1)
        :return: scores for vocabulary, decode lengths, weights
        """

        batch_size = encoder_output.size(0)

        # Flatten image
        encoder_output = encoder_output.view(batch_size, -1, encoder_output.size(-1))

        # Initialize LSTM state
        states = self.init_hidden_states(encoder_output)

        # Tensors to hold word prediction scores and alphas
        scores = torch.zeros((batch_size, max(decode_lengths), self.vocab_size), device=device)
        alphas = torch.zeros(batch_size, max(decode_lengths), encoder_output.size(1), device=device)

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
            if alphas_for_timestep is not None:
                alphas[incomplete_sequences_ixs, t, :] = alphas_for_timestep[incomplete_sequences_ixs]

        extras = {'alphas': alphas}

        return scores, decode_lengths, extras

    def forward_step(self, encoder_output, prev_words_embedded, hidden_states):
        raise NotImplementedError

    def forward_multi(self, encoder_output, target_captions=None, decode_lengths=None, teacher_forcing=0.0):
        """
        Forward propagation.

        :param encoder_output: output features of the encoder
        :param target_captions: encoded target captions, shape: (batch_size, max_caption_length)
        :param decode_lengths: caption lengths, shape: (batch_size, 1)
        :return: scores for vocabulary, decode lengths, weights
        """

        batch_size = encoder_output.size(0)

        # Flatten image
        encoder_output = encoder_output.view(batch_size, -1, encoder_output.size(-1))

        # Initialize LSTM state
        states = self.init_hidden_states(encoder_output)

        # Tensors to hold word prediction scores and alphas
        w_scores = torch.zeros((batch_size, max(decode_lengths), self.vocab_size), device=device)
        t_scores = torch.zeros((batch_size, max(decode_lengths), self.vocab_size), device=device)
        alphas = torch.zeros(batch_size, max(decode_lengths), encoder_output.size(1), device=device)

        # FOR MULTITASK
        if self.training and target_captions is not None:
            prev_words = torch.ones((batch_size,), dtype=torch.int64, device=device) * target_captions[:, 0]
        else:
            # At the start, all 'previous words' are the <start> token
            prev_words = torch.full((batch_size,), self.word_map[TOKEN_START], dtype=torch.int64, device=device)

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
                self.forward_multi_step(encoder_output, prev_words_embedded, states)

            # Update the previously predicted words
            w_scores_for_timestep, t_scores_for_timestep = scores_for_timestep
            prev_words = self.update_previous_word(w_scores_for_timestep, target_captions, t, teacher_forcing)

            w_scores[incomplete_sequences_ixs, t, :] = w_scores_for_timestep[incomplete_sequences_ixs]
            t_scores[incomplete_sequences_ixs, t, :] = t_scores_for_timestep[incomplete_sequences_ixs]
            if alphas_for_timestep is not None:
                alphas[incomplete_sequences_ixs, t, :] = alphas_for_timestep[incomplete_sequences_ixs]

        scores = [w_scores, t_scores]
        extras = {'alphas': alphas}

        return scores, decode_lengths, extras

    def forward_multi_step(self, encoder_output, prev_words_embedded, hidden_states):
        raise NotImplementedError
