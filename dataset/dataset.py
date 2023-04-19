import os
import pdb
import yaml
import torch
import random
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from einops import rearrange

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
    

class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        """
        Args:
            root: Location of the dataset folder, usually it is /unlabeled
            transform: the transform you want to applied to the images.
        """
        self.args = args
        broken_lst = [8326, 3768, 6751, 14879, 6814, 3776, 3110]
        self.video_lst = []
        self.type_lst = []
        for i in range((13000)):
            if (2000+i//4) not in broken_lst:
                self.video_lst.append(f"video_{2000+i//4}")
                # 0: not flipped, first half   1: flipped, first half
                # 2: not flipped, second half  3: flipped, second half
                self.type_lst.append(i%4) # augmentation type

        self.post_process = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # use imagenet default mean/std
        ])

    def __len__(self):
        return len(self.video_lst)

    def __getitem__(self, idx):
        # the idx of labeled image is from 0
        assert self.type_lst[idx] in [0, 1, 2, 3]
        trans = transforms.RandomHorizontalFlip(p=self.type_lst[idx]%2)
        
        # first resize
        img_lst = [
            self.post_process(Image.open(os.path.join(self.args.root, f'{self.video_lst[idx]}/image_{i}.png')).convert('RGB')) for i in range(22)
        ]
        
        # not flipped, first half    # flipped, first half
        if self.type_lst[idx] == 0 or self.type_lst[idx] == 1:
            img_lst = torch.stack(img_lst)

        # not flipped, second half   # flipped, second half
        elif self.type_lst[idx] == 2 or self.type_lst[idx] == 3:
            img_lst.reverse()
            img_lst = torch.stack(img_lst)

        img_lst = trans(img_lst)

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
        try:
            image = Image.open(os.path.join(self.args.root, self.file_lst[idx])).convert('RGB')
        except:
            Log.info(f"Can't open {os.path.join(self.args.root, self.file_lst[idx])}")

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


class TrainDatset(torch.utils.data.Dataset):
    def __init__(self, args, transforms=None):
        # "../../../dataset/dl/"
        self.image_dir = args.root

        self.transforms = transforms
        # create the video list, each contains 22 frames
        self.video_lst = [f"train/video_{i//2}" for i in range(2000)]
        # 0: use first 11 images to predict 12-22 masks
        # 1: use 12-22 images to predict frist 11 masks
        self.index_lst = [i%2 for i in range(2000)]

    def __len__(self):
        return len(self.folder_lst)

    def __getitem__(self, idx):
        # select which video
        video = os.path.join(self.image_dir, self.folder_lst[idx])
        half = self.index_ls[idx]
        if half == 0: # first half
            pass


class ValDatset(torch.utils.data.Dataset):
    def __init__(self, args, transforms=None):
        # "../../../dataset/dl/"
        self.image_dir = args.root
        self.transforms = transforms
        # create the video list, each contains 22 frames
        self.video_lst = [f"val/video_{i//2+1000}" for i in range(2000)]
        # 0: use first 11 images to predict 12-22 masks
        # 1: use 12-22 images to predict frist 11 masks
        self.index_lst = [i%2 for i in range(2000)]

    def __len__(self):
        return len(self.folder_lst)

    def __getitem__(self, idx):
        video = os.path.join(self.image_dir, self.folder_lst[idx])
        half = self.index_ls[idx]
        return None


# if __name__ == "__main__":
    # dataset = UnlabeledDataset('../../../dataset/dl/unlabeled', None)


    
