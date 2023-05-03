import os
import pdb
import yaml
import torch
import random
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from einops import rearrange
from einops.layers.torch import Rearrange

from utils.logger import Logger as Log

from dataset.mask_generator import TubeMaskingGenerator
from dataset.transforms import GroupMultiScaleCrop, GroupNormalize, Stack, ToTorchFormatTensor, RandomReverse, RandomHorizontalFlip


def list_all_files(rootdir):
    extention = '.png'
    _files = []
    lst = os.listdir(rootdir)
    for i in range(0,len(lst)):
        path = os.path.join(rootdir+'/',lst[i])
        pdb.set_trace()
        if os.path.isdir(path):
            _files.extend(list_all_files(path))
        if os.path.isfile(path):
            if '/find' not in path and path[-4:] == extention:
                _files.append(path)
    return _files

def write_files(root, lst):
    lst.sort()
    with open('unlabeled_list.txt', 'w') as f:
        for folder in lst:
            for i in range(22):
                f.write(f'{root}/{folder}/image_{i}.png\n')

def build_unlabeled():
    foler_lst = os.listdir('../../../dataset/dl/unlabeled')
    write_files('unlabeled', foler_lst)
    lines = []
    with open("unlabeled_list.txt", "r") as file:
        for line in file:
            lines.append(line)
    first = lines[:110000]
    second = lines[110000:]
    lines = []
    lines.extend(second)
    lines.extend(first)
    with open("unlabeled_list.txt", "w") as f:
        for line in lines:
            f.write(line)

def build_trainval_list(root):
    folder_lst = os.listdir(root)
    folder_lst.sort()
    pdb.set_trace()
    a = 1
    

class PredictDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        """
        Args:
            root: Location of the dataset folder, usually it is /unlabeled
            transform: the transform you want to applied to the images.
        """
        self.args = args
        broken_lst = [6751, 14879, 6814, 3110]
        self.video_lst = []
        self.type_lst = []
        for i in range((13000)):
            if (2000+i) not in broken_lst:
                self.video_lst.append(f"video_{2000+i}")

        self.resize = transforms.Resize((args.input_size, args.input_size))
        self.transform = transforms.Compose([
            Stack(p=self.args.reverse),
            transforms.ToTensor(),
            RandomHorizontalFlip(p=self.args.flip),
            GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # use imagenet default mean/std
            Rearrange("(t c) h w -> t c h w", t=22)
        ])

    def __len__(self):
        return len(self.video_lst)

    def __getitem__(self, idx):
        # first resize
        img_lst = [
            self.resize(Image.open(os.path.join(self.args.root, f'{self.video_lst[idx]}/image_{i}.png')).convert('RGB')) for i in range(22)
        ]

        img_lst = self.transform(img_lst)
        
        return (img_lst[:11], img_lst[11:])


class MAEUnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        """
        Args:
            root: Location of the dataset folder, usually it is /unlabeled
            transform: the transform you want to applied to the images.
        """
        self.args = args
        # broken_lst = [8326, 3768, 6751, 14879, 6814, 3776, 3110]
        broken_lst = [6751, 14879, 6814, 3110]
        self.file_lst = []
        with open(args.list_path, "r") as f:
            for line in f:
                self.file_lst.append(line.strip())

        self.transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=args.flip),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # use imagenet default mean/std
        ])

    def __len__(self):
        return len(self.file_lst)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.args.root, self.file_lst[idx])).convert('RGB')

        return self.transform(image)


class VMAEUnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        """
        This is the dataset for Video MAE pretraining
        Args:
            root: Location of the dataset folder, usually it is /unlabeled
            transform: the transform you want to applied to the images.
        """
        self.args = args
        self.video_lst = []
        self.type_lst = []

        broken_lst = [8326, 3768, 6751, 14879, 6814, 3776, 3110]
        for i in range(13000*2):
            if (2000+i//2) not in broken_lst:
                self.video_lst.append(f"video_{2000+i//2}")
                # 0: first half   1: second half
                self.type_lst.append(i%2) # first half or second half

        self.resize = transforms.Resize((args.input_size, args.input_size))
        # because the horizontal filp depends on the idx, we have to split transform into two parts
        normalize = GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.train_augmentation = GroupMultiScaleCrop(args.input_size, [1, .875, .75, .66])
        self.transform = transforms.Compose([                            
            self.train_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),  # [(C, T), H, W]
            RandomHorizontalFlip(p=args.flip),
            RandomReverse(p=args.reverse),
            normalize,
        ])
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                self.args.window_size, self.args.mask_ratio
            )

    def __len__(self):
        return len(self.video_lst)

    def __getitem__(self, idx):
        # the idx of labeled image is from 0
        assert self.type_lst[idx] in [0, 1]

        # first half
        if self.type_lst[idx] == 0:
            img_lst = [
                self.resize(Image.open(os.path.join(self.args.root, f'{self.video_lst[idx]}/image_{i}.png')).convert('RGB')) for i in range(12)
            ]
        # second half
        elif self.type_lst[idx] == 1:
            img_lst = [
                self.resize(Image.open(os.path.join(self.args.root, f'{self.video_lst[idx]}/image_{i+10}.png')).convert('RGB')) for i in range(12)
            ]

        # for pretraining stage, we set label to None
        imgs = self.transform((img_lst, None))[0]  # T*C, H, W

        imgs = rearrange(imgs, '(t c) h w -> c t h w', c=3)  # C, T, H, W
        
        return (imgs, self.masked_position_generator())


def process_mask(mask, num_classes=49):
    num_frames, h, w = mask.shape
    new_mask = np.zeros((num_frames, num_classes, h, w))
    for f in range(num_frames):
        # total cls num is 49
        for c in range(49):
            new_mask[f][c][np.where(mask[f] == c)] = 1

    return new_mask


class TrainDatset(torch.utils.data.Dataset):
    def __init__(self, args):
        # "../../../dataset/dl/"
        self.args = args

        # create the video list, each contains 22 frames
        self.video_lst = [f"train/video_{i}" for i in range(1000)]

        self.resize = transforms.Resize((args.input_size, args.input_size))
        self.resize_catmsk = transforms.Resize((14, 14))
        self.transform = transforms.Compose([
            Stack(p=0.0),
            transforms.ToTensor(),
            RandomHorizontalFlip(p=self.args.flip), 
            GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # use imagenet default mean/std
            Rearrange("(t c) h w -> t c h w", t=11)
        ])

    def __len__(self):
        return len(self.video_lst)

    def __getitem__(self, idx):
        # root: ../../../dataset/dl
        # load refined mask
        mask = np.load(os.path.join(self.args.root, f'{self.video_lst[idx]}/refined_mask.npy')) # [22, 160, 240]
        # convert mask to torch tensor
        mask = torch.from_numpy(mask[11:])  # [11, 160, 240]
        # resize mask for loss calculation 

        # mask = self.resize(mask)  #  [11, 224, 224]
        # """ deal with the smooth value when resizing """
        # mask[torch.where(mask>0.5)] = 1.0
        # mask[torch.where(mask<0.5)] = 0.0

        # process images
        img_lst = [
            self.resize(Image.open(os.path.join(self.args.root, f'{self.video_lst[idx]}/image_{i}.png')).convert('RGB')) for i in range(11)
        ]
        # transform images
        img_lst = self.transform(img_lst)  # [11, 3, 224, 224]

        return (img_lst, mask)


class ValDatset(torch.utils.data.Dataset):
    def __init__(self, args):
        # "../../../dataset/dl/"
        self.args = args

        # create the video list, each contains 22 frames
        self.video_lst = [f"val/video_{i+1000}" for i in range(1000)]

        self.resize = transforms.Resize((args.input_size, args.input_size))
        # no transformations
        self.transform = transforms.Compose([
            Stack(p=0.0),
            transforms.ToTensor(),
            GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # use imagenet default mean/std
            Rearrange("(t c) h w -> t c h w", t=11)
        ])

    def __len__(self):
        return len(self.video_lst)

    def __getitem__(self, idx):
        # root: ../../../dataset/dl
        # load mask
        mask = np.load(os.path.join(self.args.root, f'{self.video_lst[idx]}/refined_mask.npy'))
        # convert mask to torch tensor
        mask = torch.from_numpy(mask[11:])   # [11, 160, 240]

        # process images
        img_lst = [
            self.resize(Image.open(os.path.join(self.args.root, f'{self.video_lst[idx]}/image_{i}.png')).convert('RGB')) for i in range(11)
        ]
        # basic transform of images
        img_lst = self.transform(img_lst)  # [11, 3, 224, 224]

        return (img_lst, mask)


class HRDatset(torch.utils.data.Dataset):
    def __init__(self, args, train=True):
        # "../../../dataset/dl/"
        self.args = args
        self.train = train
        # create the video list, each contains 22 frames
        self.video_lst = []
        self.tensor = transforms.ToTensor()
        if train:
            for i in range(1000):
                self.video_lst.append(f"train/video_{i}")
            self.transform = transforms.Compose([
                # Stack(p=0.0),
                # transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            self.horizontal = RandomHorizontalFlip(p=self.args.flip)
        else:
            for i in range(1000):
                self.video_lst.append(f"val/video_{i+1000}")
            self.transform = transforms.Compose([
                # Stack(p=0.0),
                # transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])


    def __len__(self):
        return len(self.video_lst)

    def __getitem__(self, idx):
        # root: ../../../dataset/dl
        # load refined mask
        mask = np.load(os.path.join(self.args.root, f'{self.video_lst[idx]}/refined_mask.npy')) # [22, 160, 240]
        # convert mask to torch tensor
        mask = torch.from_numpy(mask)  # [22, 160, 240]

        # process images
        imgs = [
            self.tensor(Image.open(os.path.join(self.args.root, f'{self.video_lst[idx]}/image_{i}.png')).convert('RGB')).unsqueeze(0) for i in range(22)
        ]  # single img: [1, 3, 160, 240]

        imgs = torch.cat(imgs)  # [22, 3, 160, 240]
        # transform images
        imgs = self.transform(imgs)  # [22, 3, 224, 224]

        if self.train:
            imgs = self.horizontal(imgs)
            mask = self.horizontal(mask)

        return (imgs, mask)


class ValTestDatset(torch.utils.data.Dataset):
    def __init__(self, args):
        # "../../../dataset/dl/"
        self.args = args

        # create the video list, each contains 22 frames
        self.video_lst = [f"val/video_{i+1000}" for i in range(1000)]

        self.resize = transforms.Resize((160, 240))
        self.transform = transforms.Compose([
            Stack(p=0.0),
            transforms.ToTensor(),
            GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # use imagenet default mean/std
            Rearrange("(t c) h w -> t c h w", t=11)
        ])

    def __len__(self):
        return len(self.video_lst)

    def __getitem__(self, idx):
        # root: ../../../dataset/dl
        mask = np.load(os.path.join(self.args.root, f'{self.video_lst[idx]}/refined_mask.npy'))[-1] # [160, 240]
        # convert mask to torch tensor
        mask = torch.from_numpy(mask)  # [160, 240]

        # process images
        imgs = [
            self.resize(Image.open(os.path.join(self.args.root, f'{self.video_lst[idx]}/image_{i}.png')).convert('RGB')) for i in range(11)
        ]

        imgs = self.transform(imgs)  # [11, 3, 224, 224]

        return (imgs, mask)


class TestDatset(torch.utils.data.Dataset):
    def __init__(self, args):
        # "../../../dataset/dl/"
        self.args = args

        # create the video list, each contains 22 frames
        self.video_lst = [f"hidden/video_{i+15000}" for i in range(2000)]

        self.resize = transforms.Resize((160, 240))
        self.transform = transforms.Compose([
            Stack(p=0.0),
            transforms.ToTensor(),
            GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # use imagenet default mean/std
            Rearrange("(t c) h w -> t c h w", t=11)
        ])

    def __len__(self):
        return len(self.video_lst)

    def __getitem__(self, idx):
        # root: ../../../dataset/dl
        # process images
        img_lst = [
            self.resize(Image.open(os.path.join(self.args.root, f'{self.video_lst[idx]}/image_{i}.png')).convert('RGB')) for i in range(11)
        ]

        img_lst = self.transform(img_lst)  # [11, 3, 160, 240]

        return (img_lst, self.video_lst[idx])

