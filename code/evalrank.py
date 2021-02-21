"""Evaluate an image-text retrieval model on the specified evaluation set using recall metrics"""

from tqdm import tqdm
import argparse
import logging
import os.path
import sys
import numpy as np

import torch.optim
import torch.utils.data
import torch.backends.cudnn as cudnn

from toolkit.data.datasets import get_data_loader
from toolkit.utils import MODEL_SHOW_ATTEND_TELL


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # improve performance if inputs to model are fixed size


def encode_data(model, data_loader):
    """Encode all images and captions loadable by `data_loader`"""
    
    # numpy array to keep all the embeddings
    img_embs = None
    cap_embs = None
    
    ix = 0
    for image_features, all_captions_for_image, caption_lengths, coco_ids in tqdm(data_loader):
        encoder_out = image_features
        captions_for_image = all_captions_for_image[0]
        lengths = caption_lengths[0]
        img_emb, cap_emb = model.decoder.forward_ranking(encoder_out, captions_for_image, lengths)

        # initialize the numpy arrays given the size of the embeddings
        if img_embs is None:
            img_embs = np.zeros((len(data_loader)*len(lengths), img_emb.size(1)))
            cap_embs = np.zeros((len(data_loader)*len(lengths), cap_emb.size(1)))
    
        # preserve the embeddings by copying from gpu and converting to numpy
        ixs = list(range(ix, ix+len(lengths)))
        img_embs[ixs] = img_emb.data.cpu().numpy().repeat(len(lengths), axis=0).copy()
        cap_embs[ixs] = cap_emb.data.cpu().numpy().copy()

        ix += len(lengths)

    return img_embs, cap_embs


# From https://github.com/fartashf/vsepp/blob/master/evaluation.py
def i2t(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = images.shape[0] // 5
    index_list = []

    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        # Get query image
        im = images[5 * index].reshape(1, images.shape[1])

        # Compute scores
        if measure == 'order':
            raise NotImplementedError
            # bs = 100
            # if index % bs == 0:
            #     mx = min(images.shape[0], 5 * (index + bs))
            #     im2 = images[5 * index:mx:5]
            #     d2 = order_sim(torch.Tensor(im2).cuda(), torch.Tensor(captions).cuda())
            #     d2 = d2.cpu().numpy()
            # d = d2[index % bs]
        else:
            d = np.dot(im, captions.T).flatten()
        inds = np.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


# From https://github.com/fartashf/vsepp/blob/master/evaluation.py
def t2i(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = images.shape[0] // 5
    ims = np.array([images[i] for i in range(0, len(images), 5)])

    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)
    for index in range(npts):
        # Get query captions
        queries = captions[5 * index:5 * index + 5]

        # Compute scores
        if measure == 'order':
            raise NotImplementedError
            # bs = 100
            # if 5 * index % bs == 0:
            #     mx = min(captions.shape[0], 5 * index + bs)
            #     q2 = captions[5 * index:mx]
            #     d2 = order_sim(torch.Tensor(ims).cuda(), torch.Tensor(q2).cuda())
            #     d2 = d2.cpu().numpy()
            # d = d2[:, (5 * index) % bs:(5 * index) % bs + 5].T
        else:
            d = np.dot(queries, ims.T)
        inds = np.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = np.argsort(d[i])[::-1]
            ranks[5 * index + i] = np.where(inds[i] == index)[0][0]
            top1[5 * index + i] = inds[i][0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def evalrank(image_features_fn, dataset_splits_dir, split, checkpoint_path, output_path):
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_name = checkpoint["model_name"]
    logging.info("Model: {}".format(model_name))

    model = checkpoint["model"]
    model = model.to(device)
    model.eval()
    logging.info("Model params: {}".format(vars(model)))

    # DataLoader
    image_normalize = None
    if model_name == MODEL_SHOW_ATTEND_TELL:
        image_normalize = "imagenet"
    data_loader = get_data_loader(split, 1, dataset_splits_dir, image_features_fn, 1, image_normalize)

    print("Computing results...")
    img_embs, cap_embs = encode_data(model, data_loader)
    print("Images: %d, Captions: %d" % (img_embs.shape[0]/5, cap_embs.shape[0]))

    # Full evaluation
    r, rt = i2t(img_embs, cap_embs, return_ranks=True)
    ri, rti = t2i(img_embs, cap_embs, return_ranks=True)
    ar = (r[0] + r[1] + r[2]) / 3
    ari = (ri[0] + ri[1] + ri[2]) / 3
    rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
    
    # Save results
    name = split
    outputs_dir = os.path.join(output_path, "results")
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    output_fn = os.path.join(outputs_dir, name + ".ranking.out")
    with open(output_fn, 'w') as f:
        f.write("rsum: %.1f\n" % rsum)
        f.write("Average i2t Recall: %.1f\n" % ar)
        f.write("Image to text (Image Annotation): %.1f %.1f %.1f %.1f %.1f\n" % r)
        f.write("Average t2i Recall: %.1f\n" % ari)
        f.write("Text to image (Image Search): %.1f %.1f %.1f %.1f %.1f\n" % ri)


def check_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-features-filename",
                        help="Folder where the preprocessed image data is located")
    parser.add_argument("--dataset-splits-dir",
                        help="Pickled file containing the dataset splits")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to checkpoint of trained model")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--output-path",
                        help="Folder where to store outputs")
    parsed_args = parser.parse_args(args)
    return parsed_args


if __name__ == "__main__":
    args = check_args(sys.argv[1:])
    evalrank(
        image_features_fn=args.image_features_filename,
        dataset_splits_dir=args.dataset_splits_dir,
        split=args.split,
        checkpoint_path=args.checkpoint,
        output_path=args.output_path,
    )
