import os
import pdb
import numpy as np

from tqdm import tqdm


def save():
    root = '../dataset/dl/'
    # train
    for i in tqdm(range(1000)):
        mask_path = f"{root}train/video_{i}/refined_mask.npy"
        save_path = f"{root}train/video_{i}/processed_mask.npy"
        mask = np.load(mask_path)
        num_frames, h, w = mask.shape
        new_mask = np.zeros((num_frames, 49, h, w))
        for f in range(num_frames):
            # total cls num is 49
            for c in range(49):
                new_mask[f][c][np.where(mask[f] == c)] = 1
    
        np.save(save_path, new_mask)

    # val
    for i in tqdm(range(1000)):
        mask_path = f"{root}val/video_{i+1000}/refined_mask.npy"
        save_path = f"{root}val/video_{i+1000}/processed_mask.npy"
        mask = np.load(mask_path)
        num_frames, h, w = mask.shape
        new_mask = np.zeros((num_frames, 49, h, w))
        for f in range(num_frames):
            # total cls num is 49
            for c in range(49):
                new_mask[f][c][np.where(mask[f] == c)] = 1
            
        np.save(save_path, new_mask)


def find(x, y, remove, mask):
    top_steps = 0
    tx, ty = x, y
    top_label = mask[(tx, ty)]
    # top search
    while top_label in remove:
        ty += 1
        try:
            top_label = mask[(tx, ty)]
        except:
            top_steps = 1000
            break
        top_steps += 1

    down_steps = 0
    dx, dy = x, y
    down_label = mask[(dx, dy)]
    # down search
    while down_label in remove:
        dy -= 1
        try:
            down_label = mask[(dx, ty)]
        except:
            down_steps = 1000
            break
        down_steps += 1

    left_steps = 0
    lx, ly = x, y
    left_label = mask[(lx, ly)]
    # left search
    while left_label in remove:
        lx += 1
        try:
            left_label = mask[(lx, ly)]
        except:
            left_steps = 1000
            break
        left_steps += 1

    right_steps = 0
    rx, ry = x, y
    right_label = mask[(rx, ry)]
    # right search
    while right_label in remove:
        rx -= 1
        try:
            right_label = mask[(rx, ry)]
        except:
            right_steps = 1000
            break
        right_steps += 1
    
    lst = [top_steps, down_steps, left_steps, right_steps]
    labels = [top_label, down_label, left_label, right_label]
    min_steps = min(lst)
    if lst.count(min_steps) == 1:
        # unique lowest steps
        assert labels[lst.index(min_steps)] not in remove
        return labels[lst.index(min_steps)]
    else:
        # has multiple lowest steps
        # 看众数
        counts = np.bincount(labels)
        if np.argmax(counts) not in remove:
            return np.argmax(counts)
        else:
            for i in labels:
                if i not in remove:
                    return i


if __name__ == "__main__":
    root = '../dataset/dl/'
    # root = '/dataset/'

    # train video lst
    for i in tqdm(range(1000)):
        mask_path = f"{root}train/video_{i}/mask.npy"
        save_path = f"{root}train/video_{i}/refined_mask.npy"

        mask = np.load(mask_path)

        num_frames, h, w = mask.shape

        # remove the error pixels
        uni_label, count = np.unique(mask, return_counts=True)

        remove_labels1 = uni_label[np.where(count <= 600)]
        remove_labels2 = uni_label[np.where(uni_label >= 49)]
        remove = np.concatenate((remove_labels1, remove_labels2))
        remove = np.unique(remove)

        for label in remove:
            location = np.where(mask==label)
            for i in np.unique(location[0]):  # go through the frames
                x, y = int(location[1][np.where(location[0]==i)].mean()), int(location[2][np.where(location[0]==i)].mean())
                # 上下左右寻找
                replace_label = find(x, y, remove, mask[i])
                # replace
                mask[i][np.where(mask[i]==label)] = replace_label

        np.save(save_path, mask)

    # val list
    for i in tqdm(range(1000)):
        mask_path = f"{root}val/video_{i+1000}/mask.npy"
        save_path = f"{root}val/video_{i+1000}/refined_mask.npy"

        mask = np.load(mask_path)

        num_frames, h, w = mask.shape

        # remove the error pixels
        uni_label, count = np.unique(mask, return_counts=True)

        remove_labels1 = uni_label[np.where(count <= 600)]
        remove_labels2 = uni_label[np.where(uni_label >= 49)]
        remove = np.concatenate((remove_labels1, remove_labels2))
        remove = np.unique(remove)

        for label in remove:
            location = np.where(mask==label)
            for i in np.unique(location[0]):  # go through the frames
                x, y = int(location[1][np.where(location[0]==i)].mean()), int(location[2][np.where(location[0]==i)].mean())
                # 上下左右寻找
                replace_label = find(x, y, remove, mask[i])
                # replace
                mask[i][np.where(mask[i]==label)] = replace_label

        np.save(save_path, mask)

    # test video lst
    for i in tqdm(range(1000)):
        mask_path = f"{root}train/video_{i}/mask.npy"
        save_path = f"{root}train/video_{i}/refined_mask.npy"

        mask = np.load(mask_path)

        num_frames, h, w = mask.shape

        # remove the error pixels
        uni_label, count = np.unique(mask, return_counts=True)

        remove_labels1 = uni_label[np.where(count <= 600)]
        remove_labels2 = uni_label[np.where(uni_label >= 49)]
        remove = np.concatenate((remove_labels1, remove_labels2))
        remove = np.unique(remove)

        for label in remove:
            location = np.where(mask==label)
            for i in np.unique(location[0]):  # go through the frames
                x, y = int(location[1][np.where(location[0]==i)].mean()), int(location[2][np.where(location[0]==i)].mean())
                # 上下左右寻找
                replace_label = find(x, y, remove, mask[i])
                # replace
                mask[i][np.where(mask[i]==label)] = replace_label

        np.save(save_path, mask)


    # save()


