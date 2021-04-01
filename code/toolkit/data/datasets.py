"""PyTorch dataset classes for the image captioning training and testing datasets"""

import os
import h5py
import json
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from scipy.misc import imresize

from toolkit.utils import (
    DATA_CAPTIONS,
    DATA_CAPTION_LENGTHS,
    TRAIN_SPLIT,
    VALID_SPLIT,
    TEST_SPLIT,
    ENCODED_METAS_FILENAME,
    DATASET_SPLITS_FILENAME,
    IMAGENET_IMAGES_MEAN,
    IMAGENET_IMAGES_STD,
)


class CaptionDataset(Dataset):
    """
    PyTorch Dataset that provides batches of images of a given split
    """

    def __init__(self, dataset_splits_dir, features_fn, normalize=None, features_scale_factor=1):
        """
        :param data_dir: folder where data files are stored
        :param features_fn: Filename of the image features file
        :param split: split, indices of images that should be included
        :param normalize: PyTorch normalization transformation
        :param features_scale_factor: Additional scale factor, applied before normalization
        """
        self.image_features = h5py.File(features_fn, "r")
        self.features_scale_factor = features_scale_factor

        # Set PyTorch transformation pipeline
        self.transform = normalize

        # Load image meta data, including captions
        with open(os.path.join(dataset_splits_dir, ENCODED_METAS_FILENAME)) as f:
            self.image_metas = json.load(f)

        self.captions_per_image = len(next(iter(self.image_metas.values()))[DATA_CAPTIONS])

        with open(os.path.join(dataset_splits_dir, DATASET_SPLITS_FILENAME)) as f:
            self.split = json.load(f)

    def get_image_features(self, coco_id):
        image_data = self.image_features[coco_id][()]
        # scale the features with given factor
        image_data = image_data * self.features_scale_factor
        image = torch.FloatTensor(image_data)
        if self.transform:
            image = self.transform(image)
        return image

    def __getitem__(self, i):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class CaptionTrainDataset(CaptionDataset):
    """
    PyTorch training dataset that provides batches of images with a corresponding caption each.
    """

    def __init__(self, dataset_splits_dir, features_fn, normalize=None, features_scale_factor=1):
        super().__init__(dataset_splits_dir, features_fn,
                         normalize, features_scale_factor)
        self.split = self.split[TRAIN_SPLIT]

    def __getitem__(self, i):
        # Convert index depending on the dataset split
        coco_id = self.split[i // self.captions_per_image]
        caption_index = i % self.captions_per_image

        image = self.get_image_features(coco_id)
        caption = torch.LongTensor(self.image_metas[coco_id][DATA_CAPTIONS][caption_index])
        caption_length = torch.LongTensor([self.image_metas[coco_id][DATA_CAPTION_LENGTHS][caption_index]])

        return image, caption, caption_length

    def __len__(self):
        return len(self.split) * self.captions_per_image


class CaptionEvalDataset(CaptionDataset):
    """
    PyTorch test dataset that provides batches of images and all their corresponding captions.

    """

    def __init__(self, dataset_splits_dir, features_fn, normalize=None, features_scale_factor=1, eval_split="val"):
        super().__init__(dataset_splits_dir, features_fn,
                         normalize, features_scale_factor)
        if eval_split == "val":
            self.split = self.split[VALID_SPLIT]
        else:
            self.split = self.split[TEST_SPLIT]

    def __getitem__(self, i):
        # Convert index depending on the dataset split
        coco_id = self.split[i]

        image = self.get_image_features(coco_id)
        all_captions_for_image = torch.LongTensor(self.image_metas[coco_id][DATA_CAPTIONS])
        caption_lengths = torch.LongTensor(self.image_metas[coco_id][DATA_CAPTION_LENGTHS])

        return image, all_captions_for_image, caption_lengths, coco_id

    def __len__(self):
        return len(self.split)


class Scale(object):
  """Scale transform with list as size params"""

  def __init__(self, size): # interpolation=Image.BILINEAR):
    self.size = size

  def __call__(self, img):
    return imresize(img.numpy().transpose(1,2,0), (224,224))


def get_data_loader(split, batch_size, dataset_splits_dir, image_features_fn, workers, image_normalize=None):

    if not image_normalize:
        normalize = None
        features_scale_factor = 1
    if image_normalize == "imagenet":
        normalize = transforms.Normalize(mean=IMAGENET_IMAGES_MEAN, std=IMAGENET_IMAGES_STD)
        normalize = transforms.Compose([normalize])
        features_scale_factor = 1/255.0
    if image_normalize == "scaleimagenet":
        normalize = transforms.Normalize(mean=IMAGENET_IMAGES_MEAN, std=IMAGENET_IMAGES_STD)
        normalize = transforms.Compose([Scale([224,224]), transforms.ToTensor(), normalize])
        features_scale_factor = 1

    if split == "train":
        data_loader = torch.utils.data.DataLoader(
            CaptionTrainDataset(dataset_splits_dir, image_features_fn, normalize, features_scale_factor),
            batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True
        )
    elif split in {"val", "test"}:
        data_loader = torch.utils.data.DataLoader(
            CaptionEvalDataset(dataset_splits_dir, image_features_fn, normalize, features_scale_factor, split),
            batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True
        )
    else:
        raise ValueError("Invalid data_loader split. Options: train, val, test")

    return data_loader

