"""Training script for the implemented image captioning models"""
import os
import sys
import logging
import numpy as np
from coco_caption.pycocoevalcap.bleu.bleu import Bleu

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pack_padded_sequence

from options import check_args
from toolkit.models.bottom_up_top_down import BUTDModel
from toolkit.models.bottom_up_top_down_ranking import BUTRModel
from toolkit.models.bottom_up_top_down_ranking_mean import BUTRMeanModel
from toolkit.models.bottom_up_top_down_ranking_weight import BUTRWeightModel
from toolkit.models.show_attend_tell import SATModel
from toolkit.data.datasets import get_data_loader
from toolkit.optim import create_optimizer
from toolkit.criterions import create_criterion, create_regularizer
from toolkit.utils import (
    AverageMeter,
    clip_gradients,
    decode_caption,
    save_checkpoint,
    get_log_file_path,
    rm_caption_special_tokens,
    MODEL_SHOW_ATTEND_TELL,
    MODEL_BOTTOM_UP_TOP_DOWN,
    MODEL_BOTTOM_UP_TOP_DOWN_RANKING,
    MODEL_BOTTOM_UP_TOP_DOWN_RANKING_MEAN,
    MODEL_BOTTOM_UP_TOP_DOWN_RANKING_WEIGHT,
    OBJECTIVE_GENERATION,
    OBJECTIVE_JOINT,
    # OBJECTIVE_MULTI,
    TOKEN_PAD
)

abbr2name = {"sat": MODEL_SHOW_ATTEND_TELL, "butd": MODEL_BOTTOM_UP_TOP_DOWN, "butr": MODEL_BOTTOM_UP_TOP_DOWN_RANKING,
             "butr_mean": MODEL_BOTTOM_UP_TOP_DOWN_RANKING_MEAN, "butr_weight": MODEL_BOTTOM_UP_TOP_DOWN_RANKING_WEIGHT}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # improve performance if inputs to model are fixed size


# ==================================================================================================================== #
#                                                        HELPERS                                                       #
# ==================================================================================================================== #
def build_model(args, model_name):
    if model_name == MODEL_SHOW_ATTEND_TELL:
        model = SATModel(args)
    elif model_name == MODEL_BOTTOM_UP_TOP_DOWN:
        model = BUTDModel(args)
    elif model_name == MODEL_BOTTOM_UP_TOP_DOWN_RANKING:
        model = BUTRModel(args)
    elif model_name == MODEL_BOTTOM_UP_TOP_DOWN_RANKING_MEAN:
        model = BUTRMeanModel(args)
    elif model_name == MODEL_BOTTOM_UP_TOP_DOWN_RANKING_WEIGHT:
        model = BUTRWeightModel(args)
    return model


def build_optimizers(args, model):
    encoder_optimizer = None
    if args.model in {MODEL_SHOW_ATTEND_TELL} and args.encoder_training != "freeze":
        encoder_optimizer = create_optimizer(model.encoder, args.encoder_learning_rate)
    decoder_optimizer = create_optimizer(model.decoder, args.decoder_learning_rate)
    return encoder_optimizer, decoder_optimizer


def calc_initial_losses(data_loader, model, teacher_forcing, gen_criterion, rank_criterion):

    model.train()

    # Do only one batch
    images, target_captions, caption_lengths = next(iter(data_loader))
    target_captions = target_captions.to(device)
    caption_lengths = caption_lengths.to(device)
    images = images.to(device)

    # Forward propagation
    decode_lengths = caption_lengths.squeeze(1) - 1
    scores, decode_lengths, extras = model(images, target_captions, decode_lengths, teacher_forcing)
    images_embedded = extras.get("images_embedded", None)
    captions_embedded = extras.get("captions_embedded", None)

    # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
    target_captions = target_captions[:, 1:]

    # Remove timesteps that we didn't decode at, or are pads
    decode_lengths, sort_ind = decode_lengths.sort(dim=0, descending=True)
    scores = pack_padded_sequence(scores[sort_ind], decode_lengths, batch_first=True)[0]
    targets = pack_padded_sequence(target_captions[sort_ind], decode_lengths, batch_first=True)[0]

    # Calculate losses
    loss_generation = gen_criterion(scores, targets)
    loss_ranking = rank_criterion(images_embedded, captions_embedded)

    logging.info("Initial generation loss: {}".format(loss_generation))
    logging.info("Initial ranking loss: {}".format(loss_ranking))

    return loss_generation, loss_ranking


# ==================================================================================================================== #
#                                                      TRAIN & VAL                                                     #
# ==================================================================================================================== #
def train(model, data_loader,
          encoder_optimizer, decoder_optimizer,
          criterion, reg_param, reg_func, grad_clip,
          epoch, teacher_forcing, print_freq,
          mask_prob, mask_type):
    """
    Perform one training epoch.

    """

    model.train()
    losses = AverageMeter()

    # Loop over training batches
    for i, (images, target_captions, caption_lengths) in enumerate(data_loader):
        target_captions = target_captions.to(device)
        caption_lengths = caption_lengths.to(device)
        images = images.to(device)

        # Forward propagation
        decode_lengths = caption_lengths.squeeze(1) - 1
        scores, decode_lengths, extras = model(images, target_captions, decode_lengths,
                                               teacher_forcing, mask_prob, mask_type)
        alphas = extras.get("alphas", None)  # B x T x H*W

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        target_captions = target_captions[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        decode_lengths, sort_ind = decode_lengths.sort(dim=0, descending=True)
        scores = pack_padded_sequence(scores[sort_ind], decode_lengths, batch_first=True)[0]
        targets = pack_padded_sequence(target_captions[sort_ind], decode_lengths, batch_first=True)[0]
        
        # Calculate loss
        loss = criterion(scores, targets)
        loss += reg_func(reg_param, alphas)

        # Backward propagation
        decoder_optimizer.zero_grad()
        if encoder_optimizer:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip:
            clip_gradients(decoder_optimizer, grad_clip)
            if encoder_optimizer:
                clip_gradients(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer:
            encoder_optimizer.step()

        # Keep track of metrics
        losses.update(loss.item(), sum(decode_lengths).item())

        # Log status
        if i % print_freq == 0:
            logging.info("Epoch: {0}[Batch {1}/{2}]\t"
                         "Loss: {loss.val:.4f} (Average: {loss.avg:.4f})\t".format(
                            epoch, i, len(data_loader), loss=losses))

    logging.info("\n * LOSS - {loss.avg:.3f}\n".format(loss=losses))


def train_joint(model, data_loader,
                encoder_optimizer, decoder_optimizer,
                gradnorm_optimizer, gradnorm_alpha,
                gen_criterion, rank_criterion, gradnorm_criterion, grad_clip,
                loss_weight_generation, loss_weight_ranking,
                initial_generation_loss, initial_ranking_loss,
                epoch, teacher_forcing, print_freq,
                mask_prob, mask_type):
    """
    Perform one training epoch for jointly learning to caption and rank.

    """

    model.train()
    losses = AverageMeter()

    # Loop over training batches
    loss_weights = [loss_weight_generation, loss_weight_ranking]
    for i, (images, target_captions, caption_lengths) in enumerate(data_loader):
        target_captions = target_captions.to(device)
        caption_lengths = caption_lengths.to(device)
        images = images.to(device)

        # Forward propagation
        decode_lengths = caption_lengths.squeeze(1) - 1
        scores, decode_lengths, extras = model(images, target_captions, decode_lengths,
                                               teacher_forcing, mask_prob, mask_type)
        images_embedded = extras.get("images_embedded", None)
        captions_embedded = extras.get("captions_embedded", None)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        target_captions = target_captions[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        decode_lengths, sort_ind = decode_lengths.sort(dim=0, descending=True)
        scores = pack_padded_sequence(scores[sort_ind], decode_lengths, batch_first=True)[0]
        targets = pack_padded_sequence(target_captions[sort_ind], decode_lengths, batch_first=True)[0]

        # Calculate losses
        loss_generation = gen_criterion(scores, targets)
        loss_ranking = rank_criterion(images_embedded, captions_embedded)
        loss = loss_weights[0] * loss_generation + loss_weights[1] * loss_ranking

        # Backward propagation
        decoder_optimizer.zero_grad()
        if encoder_optimizer:
            encoder_optimizer.zero_grad()
        loss.backward(retain_graph=True)

        # Get the gradients of the shared layers and calculate their l2-norm
        named_params = dict(model.decoder.named_parameters())
        shared_params = [param for param_name, param in named_params.items()
                         if param_name in model.decoder.SHARED_PARAMS and param.requires_grad]
        G1R = torch.autograd.grad(loss_generation, shared_params, retain_graph=True, create_graph=True)
        G1R_flattened = torch.cat([g.view(-1) for g in G1R])
        G1 = torch.norm(loss_weights[0] * G1R_flattened.data, 2).unsqueeze(0)
        G2R = torch.autograd.grad(loss_ranking, shared_params)
        G2R_flattened = torch.cat([g.view(-1) for g in G2R])
        G2 = torch.norm(loss_weights[1] * G2R_flattened.data, 2).unsqueeze(0)
        # Calculate the average gradient norm across all tasks
        G_avg = torch.div(torch.add(G1, G2), 2)

        # Calculate relative losses
        lhat1 = torch.div(loss_generation, initial_generation_loss)
        lhat2 = torch.div(loss_ranking, initial_ranking_loss)
        lhat_avg = torch.div(torch.add(lhat1, lhat2), 2)

        # Calculate relative inverse training rates
        inv_rate1 = torch.div(lhat1, lhat_avg)
        inv_rate2 = torch.div(lhat2, lhat_avg)

        # Calculate the gradient norm target for this batch
        C1 = G_avg * (inv_rate1 ** gradnorm_alpha)
        C2 = G_avg * (inv_rate2 ** gradnorm_alpha)

        # Calculate the gradnorm loss
        Lgrad = torch.add(gradnorm_criterion(G1, C1.data), gradnorm_criterion(G2, C2.data))

        # Backprop and perform an optimization step
        gradnorm_optimizer.zero_grad()
        Lgrad.backward()
        gradnorm_optimizer.step()

        # Clip gradients
        if grad_clip:
            clip_gradients(decoder_optimizer, args.grad_clip)
            if encoder_optimizer:
                clip_gradients(encoder_optimizer, args.grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer:
            encoder_optimizer.step()

        # Keep track of metrics
        losses.update(loss.item(), sum(decode_lengths).item())

        # Log status
        if i % print_freq == 0:
            logging.info(
                "Epoch: {0}[Batch {1}/{2}]\t"
                "Loss: {loss.val:.4f} (Average: {loss.avg:.4f})\t" 
                "Loss weights: Generation: {3:.4f} Ranking: {4:.4f}".format(
                    epoch, i, len(data_loader), loss_weights[0].item(), loss_weights[1].item(), loss=losses))

        # Re-normalize gradnorm weights
        # Enforce positive weights
        coef = 2 / torch.add(torch.abs(loss_weight_generation), torch.abs(loss_weight_ranking))
        loss_weights = [coef * torch.abs(loss_weight_generation), coef * torch.abs(loss_weight_ranking)]

    logging.info("\n * LOSS - {loss.avg:.3f}\n".format(loss=losses))


def validate(model, data_loader, max_caption_len, print_freq):
    """
    Perform validation of one training epoch.

    """
    word_map = model.decoder.word_map
    model.eval()

    target_captions = []
    generated_captions = []
    coco_ids = []
    bleu4 = Bleu(n=4)

    # Loop over batches
    for i, (images, all_captions_for_image, _, coco_id) in enumerate(data_loader):
        images = images.to(device)

        # Forward propagation
        decode_lengths = torch.full((images.size(0),), max_caption_len, dtype=torch.int64, device=device)
        scores, decode_lengths, alphas = model(images, None, decode_lengths)

        if i % print_freq == 0:
            logging.info("Validation: [Batch {0}/{1}]\t".format(i, len(data_loader)))

        # Target captions
        for j in range(all_captions_for_image.shape[0]):
            img_captions = [decode_caption(rm_caption_special_tokens(caption, word_map), word_map)
                            for caption in all_captions_for_image[j].tolist()]
            target_captions.append(img_captions)

        # Generated captions
        _, captions = torch.max(scores, dim=2)
        captions = [decode_caption(rm_caption_special_tokens(caption, word_map), word_map)
                    for caption in captions.tolist()]
        generated_captions.extend(captions)

        coco_ids.append(coco_id[0])

        assert len(target_captions) == len(generated_captions)

    id2targets = {coco_ids[ix]: target_captions[ix] for ix in range(len(coco_ids))}
    id2caption = {coco_ids[ix]: [generated_captions[ix]] for ix in range(len(coco_ids))}
    bleus, _ = bleu4.compute_score(id2targets, id2caption)
    bleu = bleus[-1]

    logging.info("\n * BLEU-4 - {bleu}\n".format(bleu=bleu))
    return bleu


# ==================================================================================================================== #
#                                                         MAIN                                                         #
# ==================================================================================================================== #
def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Setup training stats
    start_epoch = 0
    epochs_since_last_improvement = 0
    best_gen_metric_score = 0.0

    # Data loaders
    train_data_loader = get_data_loader("train", args.batch_size, args.dataset_splits_dir, args.image_features_filename,
                                        args.workers, args.image_normalize)
    val_data_loader = get_data_loader("val", 5, args.dataset_splits_dir, args.image_features_filename,
                                      args.workers, args.image_normalize)

    # Build model
    ckpt_filename = os.path.join(args.checkpoints_dir, "checkpoint.last.pth.tar")
    if os.path.isfile(ckpt_filename):
        # Load checkpoint and update training stats
        checkpoint = torch.load(ckpt_filename, map_location=device)

        start_epoch = checkpoint["epoch"] + 1
        epochs_since_last_improvement = checkpoint["epochs_since_improvement"]
        best_gen_metric_score = checkpoint["gen_metric_score"]

        model = checkpoint["model"]
        model_name = checkpoint["model_name"]
        encoder_optimizer = checkpoint["encoder_optimizer"]
        decoder_optimizer = checkpoint["decoder_optimizer"]
        if "encoder_training" in args and args.encoder_training != "freeze" and encoder_optimizer is None:
            if args.encoder_training == "finetune":
                model.encoder.finetune()
            elif args.encoder_training == "train":
                model.encoder.unfreeze()
            encoder_optimizer, _ = build_optimizers(args, model)

        if model_name == MODEL_BOTTOM_UP_TOP_DOWN_RANKING:
            if "image_embeddings_freeze" in args:
                model.decoder.image_embedding.freeze(args.image_embeddings_freeze)
            if "caption_embeddings_freeze" in args:
                model.decoder.language_encoding_lstm.freeze(args.caption_embeddings_freeze)

        if args.objective == OBJECTIVE_JOINT:
            loss_weight_generation = checkpoint.get("loss_weight_generation", None)
            loss_weight_ranking = checkpoint.get("loss_weight_ranking", None)
            gradnorm_optimizer = checkpoint.get("gradnorm_optimizer", None)
    else:
        # No checkpoint given, initialize the model
        model_name = abbr2name[args.model]
        model = build_model(args, model_name)
        encoder_optimizer, decoder_optimizer = build_optimizers(args, model)
        if args.objective == OBJECTIVE_JOINT:
            loss_weight_generation = torch.ones(1, requires_grad=True, device=device, dtype=torch.float)
            loss_weight_ranking = torch.ones(1, requires_grad=True, device=device, dtype=torch.float)
            gradnorm_optimizer = torch.optim.Adam([loss_weight_generation, loss_weight_ranking],
                                                  lr=args.gradnorm_learning_rate)

    # Move to GPU, if available
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # Log configuration
    logging.info("Model params: %s", vars(model))

    # Build criterions & regularizer
    if args.objective == OBJECTIVE_JOINT:
        gen_criterion = create_criterion(args.generation_criterion)
        rank_criterion = create_criterion(args.ranking_criterion)
        gradnorm_criterion = create_criterion(args.gradnorm_criterion)
        initial_gen_loss, initial_rank_loss = calc_initial_losses(train_data_loader, model, args.teacher_forcing,
                                                                  gen_criterion, rank_criterion)
    else:
        criterion = create_criterion(args.criterion)
        reg_param, reg_func = create_regularizer(args)

    # Start Training
    logging.info("Starting training on device: %s", device)
    for epoch in range(start_epoch, args.max_epochs):
        if epochs_since_last_improvement >= args.epochs_early_stopping:
            logging.info("No improvement since {} epochs, stopping training".format(epochs_since_last_improvement))
            break

        # Train for one epoch
        if args.objective == OBJECTIVE_GENERATION:
            train(model, train_data_loader, encoder_optimizer, decoder_optimizer,
                  criterion, reg_param, reg_func, args.grad_clip,
                  epoch, args.teacher_forcing, args.print_freq,
                  args.mask_prob, args.mask_type)
            extras = dict()
        elif args.objective == OBJECTIVE_JOINT:
            train_joint(model, train_data_loader, encoder_optimizer, decoder_optimizer,
                        gradnorm_optimizer, args.gradnorm_alpha,
                        gen_criterion, rank_criterion, gradnorm_criterion, args.grad_clip,
                        loss_weight_generation, loss_weight_ranking,
                        initial_gen_loss, initial_rank_loss,
                        epoch, args.teacher_forcing, args.print_freq,
                        args.mask_prob, args.mask_type)
            extras = {'loss_weight_generation': loss_weight_generation,
                      'loss_weight_ranking': loss_weight_ranking,
                      'gradnorm_optimizer': gradnorm_optimizer}

        # Validate
        gen_metric_score = validate(model, val_data_loader, args.max_caption_len, args.print_freq)

        # Update stats
        ckpt_is_best = gen_metric_score > best_gen_metric_score
        if ckpt_is_best:
            best_gen_metric_score = gen_metric_score
            epochs_since_last_improvement = 0
        else:
            epochs_since_last_improvement += 1
            logging.info("\nEpochs since last improvement: {}".format(epochs_since_last_improvement))
            logging.info("Best generation score: {}".format(best_gen_metric_score))

        # Save checkpoint
        save_checkpoint(args.checkpoints_dir, model_name, model, epoch, epochs_since_last_improvement,
                        encoder_optimizer, decoder_optimizer, gen_metric_score, ckpt_is_best, **extras)

    logging.info("\n\nFinished training.")


if __name__ == "__main__":
    args = check_args(sys.argv[1:])
    logging.basicConfig(filename=get_log_file_path(args.logging_dir, "train"), level=logging.INFO)
    logging.info(args)
    main(args)

