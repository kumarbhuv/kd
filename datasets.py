# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json
from PIL import Image
import torch
import pandas as pd
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.datasets import VisionDataset
from typing import Any, Callable, Optional, Tuple

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
import torchvision
import scipy.io
import PIL
from torch.utils.data import ConcatDataset

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == "CUB":
        print("reading from datapath", args.data_path)
        root = args.data_path
        if is_train:
            dataset = CUBDataset(image_root_path=root, transform=transform, split="train")
        else:
            dataset = CUBDataset(image_root_path=root, transform=transform, split="test")

        nb_classes = 200
        assert len(dataset.class_to_idx) == nb_classes

    elif args.data_set == "Aircraft":
        print("reading from datapath", args.data_path)
        root = args.data_path
        if is_train:
            dataset = FGVCAircraft(image_root_path=root, transform=transform, split="train")
        else:
            dataset = FGVCAircraft(image_root_path=root, transform=transform, split="test")

        nb_classes = 120
        assert len(dataset.class_to_idx) == nb_classes

    elif args.data_set == "DOGS":
        print("reading from datapath", args.data_path)
        root = args.data_path
        if is_train:
            dataset = DOGDataset(image_root_path=root, transform=transform, split="train")
        else:
            dataset = DOGDataset(image_root_path=root, transform=transform, split="test")

        nb_classes = 120
        assert len(dataset.class_to_idx) == nb_classes

    elif args.data_set == "FOOD":
        if is_train:
            train_df = pd.read_csv(f'{args.data_path}/annot/train_info.csv', names=['image_name', 'label'])
            train_df['path'] = train_df['image_name'].map(lambda x: os.path.join(f'{args.data_path}/train_set/', x))
            dataset = FOODDataset(train_df, transform=transform)
        else:
            val_df = pd.read_csv(f'{args.data_path}/annot/val_info.csv', names=['image_name', 'label'])
            val_df['path'] = val_df['image_name'].map(lambda x: os.path.join(f'{args.data_path}/val_set/', x))
            dataset = FOODDataset(val_df, transform=transform)
        nb_classes = 251
        # assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    print("Number of the class = %d" % nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

class CUBDataset(torchvision.datasets.ImageFolder):
    """
    Dataset class for CUB Dataset
    """

    def __init__(self, image_root_path, caption_root_path=None, split="train", *args, **kwargs):
        """
        Args:
            image_root_path:      path to dir containing images and lists folders
            caption_root_path:    path to dir containing captions
            split:          train / testz
            *args:
            **kwargs:
        """
        image_info = self.get_file_content(f"{image_root_path}/images.txt")
        self.image_id_to_name = {y[0]: y[1] for y in [x.strip().split(" ") for x in image_info]}
        split_info = self.get_file_content(f"{image_root_path}/train_test_split.txt")
        self.split_info = {self.image_id_to_name[y[0]]: y[1] for y in [x.strip().split(" ") for x in split_info]}
        self.split = "1" if split == "train" else "0"
        self.caption_root_path = caption_root_path

        super(CUBDataset, self).__init__(root=f"{image_root_path}/images", is_valid_file=self.is_valid_file,
                                         *args, **kwargs)

    def is_valid_file(self, x):
        return self.split_info[(x[len(self.root) + 1:])] == self.split

    @staticmethod
    def get_file_content(file_path):
        with open(file_path) as fo:
            content = fo.readlines()
        return content
    
class FGVCAircraft(VisionDataset):
    _URL = "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        split: str = "trainval",
        annotation_level: str = "variant",
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = verify_str_arg(split, "split", ("train", "val", "trainval", "test"))
        self._annotation_level = verify_str_arg(
            annotation_level, "annotation_level", ("variant", "family", "manufacturer")
        )

        self._data_path = os.path.join(self.root, "fgvc-aircraft-2013b")
        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        annotation_file = os.path.join(
            self._data_path,
            "data",
            {
                "variant": "variants.txt",
                "family": "families.txt",
                "manufacturer": "manufacturers.txt",
            }[self._annotation_level],
        )
        with open(annotation_file, "r") as f:
            self.classes = [line.strip() for line in f]

        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        image_data_folder = os.path.join(self._data_path, "data", "images")
        labels_file = os.path.join(self._data_path, "data", f"images_{self._annotation_level}_{self._split}.txt")

        self._image_files = []
        self._labels = []

        with open(labels_file, "r") as f:
            for line in f:
                image_name, label_name = line.strip().split(" ", 1)
                self._image_files.append(os.path.join(image_data_folder, f"{image_name}.jpg"))
                self._labels.append(self.class_to_idx[label_name])

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def _download(self) -> None:
        """
        Download the FGVC Aircraft dataset archive and extract it under root.
        """
        if self._check_exists():
            return
        download_and_extract_archive(self._URL, self.root)

    def _check_exists(self) -> bool:
        return os.path.exists(self._data_path) and os.path.isdir(self._data_path)

class DOGDataset(torchvision.datasets.ImageFolder):
    """
    Dataset class for DOG Dataset
    """

    def __init__(self, image_root_path, caption_root_path=None, split="train", *args, **kwargs):
        """
        Args:
            image_root_path:      path to dir containing images and lists folders
            caption_root_path:    path to dir containing captions
            split:          train / test
            *args:
            **kwargs:
        """
        image_info = self.get_file_content(f"{image_root_path}splits/file_list.mat")
        image_files = [o[0][0] for o in image_info]

        split_info = self.get_file_content(f"{image_root_path}/splits/{split}_list.mat")
        split_files = [o[0][0] for o in split_info]
        self.split_info = {}
        if split == 'train':
            for image in image_files:
                if image in split_files:
                    self.split_info[image] = "1"
                else:
                    self.split_info[image] = "0"
        elif split == 'test':
            for image in image_files:
                if image in split_files:
                    self.split_info[image] = "0"
                else:
                    self.split_info[image] = "1"

        self.split = "1" if split == "train" else "0"
        self.caption_root_path = caption_root_path

        super(DOGDataset, self).__init__(root=f"{image_root_path}Images", is_valid_file=self.is_valid_file,
                                         *args, **kwargs)

        ## modify class index as we are going to concat to first dataset
        self.class_to_idx = {class_: idx for idx, class_ in enumerate(self.class_to_idx)}

    def is_valid_file(self, x):
        return self.split_info[(x[len(self.root) + 1:])] == self.split

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(os.path.join(path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        ## modify target class index as we are going to concat to first dataset
        return img, target + 200

    @staticmethod
    def get_file_content(file_path):
        content = scipy.io.loadmat(file_path)
        return content['file_list']


class FOODDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform):
        self.dataframe = dataframe
        self.data_transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        return (
            self.data_transform(Image.open(row["path"])), row['label']
        )
