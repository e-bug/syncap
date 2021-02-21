import torch
import torch.nn.functional as F

from toolkit.utils import TOKEN_START, TOKEN_END, TOKEN_PAD, rm_caption_special_tokens, decode_caption


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def beam_search(model, images, beam_size, max_caption_len=20, 
                store_alphas=False, store_beam=False, print_beam=False):
    """Generate and return the top k sequences using beam search."""

    # the max beam size is the dictionary size - 1, since we never select pad
    beam_size = min(beam_size, model.decoder.vocab_size - 1)
    current_beam_width = beam_size

    encoder_output = model.encoder(images) if model.encoder else images
    enc_image_size = encoder_output.size(1)
    encoder_dim = encoder_output.size()[-1]

    # Flatten encoding
    encoder_output = encoder_output.view(1, -1, encoder_dim)

    # We'll treat the problem as having a batch size of k
    encoder_output = encoder_output.expand(beam_size, encoder_output.size(1), encoder_dim)

    # Tensor to store top k sequences; now they're just <start>
    top_k_sequences = torch.full((beam_size, 1), model.decoder.word_map[TOKEN_START], dtype=torch.int64, device=device)
    # top_k_sequences = torch.full((beam_size, 1), model.decoder.word_map['<start_syntax>'], dtype=torch.int64, device=device)
    # FIXME top_k_sequences = torch.full((beam_size, 1), 10021, dtype=torch.int64, device=device)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(beam_size, device=device)

    if store_alphas:
        # Tensor to store top k sequences' alphas; now they're just 1s
        seqs_alpha = torch.ones(beam_size, 1, enc_image_size, enc_image_size).to(device)

    # Lists to store completed sequences, scores, and alphas and the full decoding beam
    complete_seqs = []
    complete_seqs_alpha = []
    complete_seqs_scores = []
    beam = []

    # Initialize hidden states
    states = model.decoder.init_hidden_states(encoder_output)

    # Start decoding
    for step in range(0, max_caption_len - 1):
        prev_words = top_k_sequences[:, step]

        prev_word_embeddings = model.decoder.embeddings(prev_words)
        predictions, states, alpha = model.decoder.forward_step(encoder_output, prev_word_embeddings, states)
#        print('predictions', predictions.size())
        #(predictions, _), states, alpha = model.decoder.forward_multi_step(encoder_output, prev_word_embeddings, states)
        scores = F.log_softmax(predictions, dim=1)
#        print('scores', scores.size())

        # Add the new scores
        scores = top_k_scores.unsqueeze(1).expand_as(scores) + scores
#        print('scores', scores.size())

        # For the first timestep, the scores from previous decoding are all the same, so in order to create 5
        # different sequences, we should only look at one branch
        if step == 0:
            scores = scores[0]

        # Find the top k of the flattened scores
        top_k_scores, top_k_words = scores.view(-1).topk(current_beam_width, 0, largest=True, sorted=True)
#        print('top_k_scores', top_k_scores.size(), 'top_k_words', top_k_words.size())

        # Convert flattened indices to actual indices of scores
        prev_seq_inds = top_k_words / model.decoder.vocab_size  # (k)
        next_words = top_k_words % model.decoder.vocab_size  # (k)
#        print('top_k_words', top_k_words)
#        print('prev_seq_inds', prev_seq_inds)
#        print('next_words', next_words) 

        # Add new words to sequences
        top_k_sequences = torch.cat((top_k_sequences[prev_seq_inds], next_words.unsqueeze(1)), dim=1)
#        print('top_k_sequences', top_k_sequences.size(), top_k_sequences)

        if print_beam:
            print_current_beam(top_k_sequences, top_k_scores, model.decoder.word_map)
        if store_beam:
            beam.append(top_k_sequences)

        # Store the new alphas
        if store_alphas:
            alpha = alpha.view(-1, enc_image_size, enc_image_size)
            seqs_alpha = torch.cat((seqs_alpha[prev_seq_inds], alpha[prev_seq_inds].unsqueeze(1)), dim=1)

        # Check for complete and incomplete sequences (based on the <end> token)
        incomplete_inds = torch.nonzero(next_words != model.decoder.word_map[TOKEN_END]).view(-1).tolist()
        complete_inds = torch.nonzero(next_words == model.decoder.word_map[TOKEN_END]).view(-1).tolist()
#        print('incomplete_inds', incomplete_inds, 'complete_inds', complete_inds)

        # Set aside complete sequences and reduce beam size accordingly
        if len(complete_inds) > 0:
            complete_seqs.extend(top_k_sequences[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
            if store_alphas:
                complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())

        # Stop if k captions have been completely generated
        current_beam_width = len(incomplete_inds)
        if current_beam_width == 0:
            break

        # Proceed with incomplete sequences
        top_k_sequences = top_k_sequences[incomplete_inds]
        for i in range(len(states)):
            states[i] = states[i][prev_seq_inds[incomplete_inds]]
        encoder_output = encoder_output[prev_seq_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds]
        if store_alphas:
            seqs_alpha = seqs_alpha[incomplete_inds]

    if len(complete_seqs) < beam_size:
        complete_seqs.extend(top_k_sequences.tolist())
        complete_seqs_scores.extend(top_k_scores)
        if store_alphas:
            complete_seqs_alpha.extend(seqs_alpha)
    
#    print(complete_seqs_scores, complete_seqs)

    sorted_sequences = [sequence for _, sequence in sorted(zip(complete_seqs_scores, complete_seqs), reverse=True)]
    sorted_alphas = None
    if store_alphas:
        sorted_alphas = [alpha for _, alpha in sorted(zip(complete_seqs_scores, complete_seqs_alpha), reverse=True)]
    return sorted_sequences, sorted_alphas, beam


def beam_search_conv(model, images, beam_size, max_caption_len=20,
                     store_alphas=False, store_beam=False, print_beam=False):
    """Generate and return the top k sequences using beam search."""

    # the max beam size is the dictionary size - 1, since we never select pad
    beam_size = min(beam_size, model.decoder.vocab_size - 1)
    current_beam_width = beam_size

    encoder_output = model.encoder(images) if model.encoder else images
    # enc_image_size = encoder_output[0].size(2)
    # encoder_dim = encoder_output[0].size(1)

    # Flatten encoding
    # encoder_output = encoder_output.view(1, -1, encoder_dim)

    # We'll treat the problem as having a batch size of k
    # encoder_output = encoder_output.expand(beam_size, encoder_output.size(1), encoder_dim)
    imgsfeats, imgsfc7 = encoder_output
    rep_imgsfeats = imgsfeats.repeat(beam_size, 1, 1, 1)
    rep_imgsfc7 = imgsfc7.repeat(beam_size, 1)
    encoder_output = (rep_imgsfeats, rep_imgsfc7)

    # Tensor to store top k sequences; now they're just <start>
    top_k_sequences = torch.full((beam_size, 1), model.decoder.word_map[TOKEN_START], dtype=torch.int64, device=device)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(beam_size, device=device)

    if store_alphas:
        # Tensor to store top k sequences' alphas; now they're just 1s
        # seqs_alpha = torch.ones(beam_size, 1, enc_image_size, enc_image_size).to(device)
        pass

    # Lists to store completed sequences, scores, and alphas and the full decoding beam
    complete_seqs = []
    complete_seqs_alpha = []
    complete_seqs_scores = []
    beam = []

    # Initialize hidden states
    # states = model.decoder.init_hidden_states(encoder_output)
    words = torch.full((beam_size, max_caption_len), model.decoder.word_map[TOKEN_PAD], dtype=torch.int64, device=device)
    # Start decoding
    for step in range(0, max_caption_len - 1):
        words[:, step] = top_k_sequences[:, step]

        # predictions, alpha = model.decoder.forward_step(encoder_output, words)
        # for k in beam_size:
        scores_for_timestep, alphas_for_timestep = model.decoder.forward_step(encoder_output, words)
        predictions = scores_for_timestep[:, step]
        # alpha = alphas_for_timestep[:, step]

        scores = F.log_softmax(predictions, dim=1)

        # Add the new scores
        scores = top_k_scores.unsqueeze(1).expand_as(scores) + scores

        # For the first timestep, the scores from previous decoding are all the same, so in order to create 5
        # different sequences, we should only look at one branch
        if step == 0:
            scores = scores[0]

        # Find the top k of the flattened scores
        top_k_scores, top_k_words = scores.view(-1).topk(current_beam_width, 0, largest=True, sorted=True)

        # Convert flattened indices to actual indices of scores
        prev_seq_inds = top_k_words / model.decoder.vocab_size  # (k)
        next_words = top_k_words % model.decoder.vocab_size  # (k)

        # Add new words to sequences
        top_k_sequences = torch.cat((top_k_sequences[prev_seq_inds], next_words.unsqueeze(1)), dim=1)

        if print_beam:
            print_current_beam(top_k_sequences, top_k_scores, model.decoder.word_map)
        if store_beam:
            beam.append(top_k_sequences)

        # Store the new alphas
        if store_alphas:
            # alpha = alpha.view(-1, enc_image_size, enc_image_size)
            # seqs_alpha = torch.cat((seqs_alpha[prev_seq_inds], alpha[prev_seq_inds].unsqueeze(1)), dim=1)
            pass

        # Check for complete and incomplete sequences (based on the <end> token)
        incomplete_inds = torch.nonzero(next_words != model.decoder.word_map[TOKEN_END]).view(-1).tolist()
        complete_inds = torch.nonzero(next_words == model.decoder.word_map[TOKEN_END]).view(-1).tolist()

        # Set aside complete sequences and reduce beam size accordingly
        if len(complete_inds) > 0:
            complete_seqs.extend(top_k_sequences[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
            if store_alphas:
                # complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
                pass

        # Stop if k captions have been completely generated
        current_beam_width = len(incomplete_inds)
        if current_beam_width == 0:
            break

        # Proceed with incomplete sequences
        top_k_sequences = top_k_sequences[incomplete_inds]
        # for i in range(len(states)):
        #     states[i] = states[i][prev_seq_inds[incomplete_inds]]
#        encoder_output = encoder_output[prev_seq_inds[incomplete_inds]]
        imgsfeats, imgsfc7 = encoder_output
        encoder_output = (imgsfeats[prev_seq_inds[incomplete_inds]], imgsfc7[prev_seq_inds[incomplete_inds]])
        words = words[prev_seq_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds]
        if store_alphas:
            seqs_alpha = seqs_alpha[incomplete_inds]

    if len(complete_seqs) < beam_size:
        complete_seqs.extend(top_k_sequences.tolist())
        complete_seqs_scores.extend(top_k_scores)
        if store_alphas:
            complete_seqs_alpha.extend(seqs_alpha)

    sorted_sequences = [sequence for _, sequence in sorted(zip(complete_seqs_scores, complete_seqs), reverse=True)]
    sorted_alphas = None
    if store_alphas:
        # sorted_alphas = [alpha for _, alpha in sorted(zip(complete_seqs_scores, complete_seqs_alpha), reverse=True)]
        pass
    return sorted_sequences, sorted_alphas, beam


def beam_search_conv_large(model, images, beam_size, max_caption_len=20,
                           store_alphas=False, store_beam=False, print_beam=False):
    """Generate and return the top k sequences using beam search."""

    # the max beam size is the dictionary size - 1, since we never select pad
    beam_size = min(beam_size, model.decoder.vocab_size - 1)
    current_beam_width = beam_size

    encoder_output = model.encoder(images) if model.encoder else images
    # enc_image_size = encoder_output[0].size(2)
    # encoder_dim = encoder_output[0].size(1)

    # Flatten encoding
    # encoder_output = encoder_output.view(1, -1, encoder_dim)

    # We'll treat the problem as having a batch size of k
    # encoder_output = encoder_output.expand(beam_size, encoder_output.size(1), encoder_dim)
    imgsfeats, imgsfc7 = encoder_output
    rep_imgsfeats = imgsfeats.repeat(beam_size, 1, 1, 1)
    rep_imgsfc7 = imgsfc7.repeat(beam_size, 1)
    encoder_output = (rep_imgsfeats, rep_imgsfc7)

    # Tensor to store top k sequences; now they're just <start>
    top_k_sequences = torch.full((beam_size, 1), model.decoder.word_map[TOKEN_START], dtype=torch.int64, device=device)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(beam_size, device=device)

    if store_alphas:
        # Tensor to store top k sequences' alphas; now they're just 1s
        # seqs_alpha = torch.ones(beam_size, 1, enc_image_size, enc_image_size).to(device)
        pass

    # Lists to store completed sequences, scores, and alphas and the full decoding beam
    complete_seqs = []
    complete_seqs_alpha = []
    complete_seqs_scores = []
    beam = []

    # Initialize hidden states
    # states = model.decoder.init_hidden_states(encoder_output)
    words = torch.full((beam_size, max_caption_len), model.decoder.word_map[TOKEN_PAD], dtype=torch.int64, device=device)
    # Start decoding
    for step in range(0, max_caption_len - 1):
        words[:, step] = top_k_sequences[:, step]

        # predictions, alpha = model.decoder.forward_step(encoder_output, words)
        predictions = torch.zeros((len(top_k_sequences), len(model.decoder.word_map)), device=device)
        for k in range(0,len(top_k_sequences),5):
#            encoder_output_k = (encoder_output[0][k, None], encoder_output[1][k, None])
#            words_k = words[k, None]
#            print(encoder_output_k[0].size(), words_k.size())
            scores_for_timestep, alphas_for_timestep = model.decoder.forward_step((encoder_output[0][k:k+5], encoder_output[1][k:k+5]),  words[k:k+5])
#            print(scores_for_timestep.size())
            predictions[k:k+5] = scores_for_timestep[:, step]
            # alpha = alphas_for_timestep[:, step]
            del scores_for_timestep

        scores = F.log_softmax(predictions, dim=1)

        # Add the new scores
        scores = top_k_scores.unsqueeze(1).expand_as(scores) + scores

        # For the first timestep, the scores from previous decoding are all the same, so in order to create 5
        # different sequences, we should only look at one branch
        if step == 0:
            scores = scores[0]

        # Find the top k of the flattened scores
        top_k_scores, top_k_words = scores.view(-1).topk(current_beam_width, 0, largest=True, sorted=True)

        # Convert flattened indices to actual indices of scores
        prev_seq_inds = top_k_words / model.decoder.vocab_size  # (k)
        next_words = top_k_words % model.decoder.vocab_size  # (k)

        # Add new words to sequences
        top_k_sequences = torch.cat((top_k_sequences[prev_seq_inds], next_words.unsqueeze(1)), dim=1)

        if print_beam:
            print_current_beam(top_k_sequences, top_k_scores, model.decoder.word_map)
        if store_beam:
            beam.append(top_k_sequences)

        # Store the new alphas
        if store_alphas:
            # alpha = alpha.view(-1, enc_image_size, enc_image_size)
            # seqs_alpha = torch.cat((seqs_alpha[prev_seq_inds], alpha[prev_seq_inds].unsqueeze(1)), dim=1)
            pass

        # Check for complete and incomplete sequences (based on the <end> token)
        incomplete_inds = torch.nonzero(next_words != model.decoder.word_map[TOKEN_END]).view(-1).tolist()
        complete_inds = torch.nonzero(next_words == model.decoder.word_map[TOKEN_END]).view(-1).tolist()

        # Set aside complete sequences and reduce beam size accordingly
        if len(complete_inds) > 0:
            complete_seqs.extend(top_k_sequences[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
            if store_alphas:
                # complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
                pass

        # Stop if k captions have been completely generated
        current_beam_width = len(incomplete_inds)
        if current_beam_width == 0:
            break

        # Proceed with incomplete sequences
        top_k_sequences = top_k_sequences[incomplete_inds]
        # for i in range(len(states)):
        #     states[i] = states[i][prev_seq_inds[incomplete_inds]]
#        encoder_output = encoder_output[prev_seq_inds[incomplete_inds]]
        imgsfeats, imgsfc7 = encoder_output
        encoder_output = (imgsfeats[prev_seq_inds[incomplete_inds]], imgsfc7[prev_seq_inds[incomplete_inds]])
        words = words[prev_seq_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds]
        if store_alphas:
            seqs_alpha = seqs_alpha[incomplete_inds]

    if len(complete_seqs) < beam_size:
        complete_seqs.extend(top_k_sequences.tolist())
        complete_seqs_scores.extend(top_k_scores)
        if store_alphas:
            complete_seqs_alpha.extend(seqs_alpha)

    sorted_sequences = [sequence for _, sequence in sorted(zip(complete_seqs_scores, complete_seqs), reverse=True)]
    sorted_alphas = None
    if store_alphas:
        # sorted_alphas = [alpha for _, alpha in sorted(zip(complete_seqs_scores, complete_seqs_alpha), reverse=True)]
        pass
    return sorted_sequences, sorted_alphas, beam


def nucleus_sampling(model, images, beam_size, top_p, print_beam=False):
    """Generate and return the top k sequences using nucleus sampling."""
    
    current_beam_width = beam_size

    encoder_output = model.encoder(images)
    encoder_dim = encoder_output.size()[-1]

    # Flatten encoding
    encoder_output = encoder_output.view(1, -1, encoder_dim)

    # We'll treat the problem as having a batch size of k
    encoder_output = encoder_output.expand(
        beam_size, encoder_output.size(1), encoder_dim
    )

    # Tensor to store top k sequences; now they're just <start>
    top_k_sequences = torch.full(
        (beam_size, 1), model.decoder.word_map[TOKEN_START], dtype=torch.int64, device=device
    )

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(beam_size, device=device)

    # Lists to store completed sequences, scores, and alphas and the full decoding beam
    complete_seqs = []
    complete_seqs_scores = []

    # Initialize hidden states
    states = model.decoder.init_hidden_states(encoder_output)

    # Start decoding
    for step in range(0, max_caption_len - 1):
        prev_words = top_k_sequences[:, step]

        prev_word_embeddings = model.decoder.embeddings(prev_words)
        predictions, states, alpha = model.decoder.forward_step(
            encoder_output, prev_word_embeddings, states
        )
        scores = F.log_softmax(predictions, dim=1)

        sorted_logits, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1
        ].clone()
        sorted_indices_to_remove[..., 0] = 0

        top_k_scores = torch.zeros(
            current_beam_width, dtype=torch.float, device=device
        )
        top_k_words = torch.zeros(
            current_beam_width, dtype=torch.long, device=device
        )

        for i in range(0, current_beam_width):
            scores[i][sorted_indices[i][sorted_indices_to_remove[i]]] = -float(
                "inf"
            )

            # Sample from the scores
            top_k_words[i] = torch.multinomial(torch.softmax(scores[i], -1), 1)
            top_k_scores[i] = scores[i][top_k_words[i]]

        # Add new words to sequences
        top_k_sequences = torch.cat(
            (top_k_sequences, top_k_words.unsqueeze(1)), dim=1
        )

        if print_beam:
            print_current_beam(top_k_sequences, top_k_scores, model.decoder.word_map)

        # Check for complete and incomplete sequences (based on the <end> token)
        incomplete_inds = (
            torch.nonzero(top_k_words != model.decoder.word_map[TOKEN_END]).view(-1).tolist()
        )
        complete_inds = (
            torch.nonzero(top_k_words == model.decoder.word_map[TOKEN_END]).view(-1).tolist()
        )

        # Set aside complete sequences and reduce beam size accordingly
        if len(complete_inds) > 0:
            complete_seqs.extend(top_k_sequences[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])

        # Stop if k captions have been completely generated
        current_beam_width = len(incomplete_inds)
        if current_beam_width == 0:
            break

        # Proceed with incomplete sequences
        top_k_sequences = top_k_sequences[incomplete_inds]
        for i in range(len(states)):
            states[i] = states[i][incomplete_inds]
        encoder_output = encoder_output[incomplete_inds]
        top_k_scores = top_k_scores[incomplete_inds]

    if len(complete_seqs) < beam_size:
        complete_seqs.extend(top_k_sequences.tolist())
        complete_seqs_scores.extend(top_k_scores)

    sorted_sequences = [
        sequence
        for _, sequence in sorted(
            zip(complete_seqs_scores, complete_seqs), reverse=True
        )
    ]
    return sorted_sequences, None, None


def beam_re_ranking(model, images, top_k_generated_captions, word_map):
    lengths = [len(caption) - 1 for caption in top_k_generated_captions]
    top_k_generated_captions = torch.tensor([top_k_generated_caption +
                                             [word_map[TOKEN_PAD]]*(max(lengths) + 1 - len(top_k_generated_caption))
                                             for top_k_generated_caption in top_k_generated_captions],
                                            device=device)
    encoded_features = model.encoder(images) if model.encoder else images

    image_embedded, image_captions_embedded = \
        model.decoder.forward_ranking(encoded_features, top_k_generated_captions, torch.tensor(lengths, device=device))
    image_embedded = image_embedded.detach().cpu().numpy()[0]
    image_captions_embedded = image_captions_embedded.detach().cpu().numpy()

    indices = model.get_top_ranked_captions_indices(image_embedded, image_captions_embedded)
    top_k_generated_captions = [top_k_generated_captions[i] for i in indices]

    return [caption.cpu().numpy() for caption in top_k_generated_captions]


def print_current_beam(top_k_sequences, top_k_scores, word_map):
    print("\n")
    for sequence, score in zip(top_k_sequences, top_k_scores):
        print("{} \t\t\t\t Score: {}".format(decode_caption(sequence.cpu().numpy(), word_map), score))
