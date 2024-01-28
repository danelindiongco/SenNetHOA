import os
import random
from tqdm import tqdm
import pandas as pd
import numpy as np
from glob import glob
import gc
import time
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import copy
import cv2

# pytorch imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from torch.cuda import amp
import torch.optim as optim
import albumentations as A
import segmentation_models_pytorch as smp

from colorama import Fore, Back, Style

c_ = Fore.GREEN
sr_ = Style.RESET_ALL

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import tifffile as tiff


class CFG:
    def __init__(self):
        self.seed = 42
        self.debug = False  # set debug=False for Full Training
        self.exp_name = 'baseline'
        self.comment = 'unet-efficientnet_b1-512x512'
        self.output_dir = './'
        self.model_name = 'Unet'
        self.backbone = 'efficientnet-b1'
        self.train_bs = 16
        self.valid_bs = 32
        self.img_size = [512, 512]
        self.epochs = 15
        self.n_accumulate = max(1, 64 // self.train_bs)
        self.lr = 2e-3
        self.scheduler = 'CosineAnnealingLR'
        self.min_lr = 1e-6
        self.T_max = int(2279 / (self.train_bs * self.n_accumulate) * self.epochs) + 50
        self.T_0 = 25
        self.warmup_epochs = 0
        self.wd = 1e-6
        self.n_fold = 5
        self.num_classes = 1
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self.gt_df = "dataset/train_rles.csv"
        self.data_root = "dataset"
        self.train_groups = ["kidney_1_dense"]
        self.valid_groups = ["kidney_3_dense"]
        self.loss_func = "DiceLoss"

        self.data_transforms = {
            "train": A.Compose([
                A.Resize(*self.img_size, interpolation=cv2.INTER_CUBIC),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=45, p=0.5),
                A.RandomScale(scale_limit=(0.8, 1.25), interpolation=cv2.INTER_CUBIC, p=0.5),
                A.RandomCrop(*self.img_size, p=1), ], p=1.0),  # using *self operator for iterative unpacking od li
            "valid": A.Compose([
                A.Resize(*self.img_size, interpolation=cv2.INTER_NEAREST), ], p=1.0)
        }

        self.optimizers = 'adam'


class BuildDataset(Dataset):
    def __init__(self, img_paths, msk_paths=[], transforms=None):
        self.img_paths = img_paths
        self.msk_paths = msk_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = self.load_img(img_path)

        if len(self.msk_paths) > 0:
            msk_path = self.msk_paths[index]
            msk = self.load_msk(msk_path)
            if self.transforms:
                data = self.transforms(image=img, mask=msk)
                img = data['image']
                msk = data['mask']
            img = np.transpose(img, (2, 0, 1))
            return torch.tensor(img), torch.tensor(msk)

        else:
            orig_size = img.shape
            if self.transforms:
                data = self.transforms(image=img)
                img = data['image']
            img = np.transpose(img, (2, 0, 1))
            return torch.tensor(img), torch.tensor(np.array([orig_size[0], orig_size[1]]))

    def load_img(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = np.tile(img[..., None], [1, 1, 3])  # gray to rgb
        img = img.astype('float32')  # original is uint16
        mx = np.max(img)
        if mx:
            img /= mx  # scale image to [0, 1]
        return img

    def load_msk(self, path):
        msk = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        msk = msk.astype('float32')
        msk /= 255.0
        return msk


def set_seed(seed=42):
    """Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def show_images(images, titles=None, cmap='gray'):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(20, 10))
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    for idx, ax in enumerate(axes):
        ax.imshow(images[idx], cmap=cmap)
        if titles:
            ax.set_title(titles[idx])
        ax.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    config = CFG()

    set_seed(config.seed)

    train_groups = config.train_groups
    valid_groups = config.valid_groups

    gt_df = pd.read_csv(config.gt_df)
    train_images_path = os.path.join(config.data_root, 'train', config.train_groups[0], 'images')
    train_labels_path = os.path.join(config.data_root, 'train', config.train_groups[0], 'labels')

    valid_images_path = os.path.join(config.data_root, 'test', config.valid_groups[0], 'images')
    valid_labels_path = os.path.join(config.data_root, 'test', config.valid_groups[0], 'labels')

    image_files = sorted(
        [os.path.join(train_images_path, f) for f in os.listdir(train_images_path) if f.endswith('.tif')])
    label_files = sorted(
        [os.path.join(train_labels_path, f) for f in os.listdir(train_labels_path) if f.endswith('.tif')])

    # ____________________________________________________________________________________________
    first_image = tiff.imread(image_files[981])
    first_label = tiff.imread(label_files[981])

    show_images([first_image, first_label], titles=['Train Image', 'Train Label'])

    # ____________________________________________________________________________________________
    train_image_files, val_image_files, train_mask_files, val_mask_files = train_test_split(image_files, label_files,
                                                                                            test_size=0.2,
                                                                                            random_state=config.seed)
    train_dataset = BuildDataset(train_image_files, train_mask_files, transforms=config.data_transforms['train'])
    valid_dataset = BuildDataset(val_image_files, val_mask_files, transforms=config.data_transforms['valid'])

    train_loader = DataLoader(train_dataset, batch_size=config.train_bs, num_workers=0, shuffle=True, pin_memory=True,
                              drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=config.valid_bs, num_workers=0, shuffle=False, pin_memory=True)

    for batch_idx, (batch_images, batch_masks) in enumerate(train_loader):
        print("Batch", batch_idx + 1)
        print("Image batch shape:", batch_images.shape)
        print("Mask batch shape:", batch_masks.shape)

        for image, mask, image_path, mask_path in zip(batch_images, batch_masks, train_image_files, train_mask_files):
            image = image.permute((1, 2, 0)).numpy() * 255.0
            image = image.astype('uint8')
            mask = (mask * 255).numpy().astype('uint8')

            image_filename = os.path.basename(image_path)
            mask_filename = os.path.basename(mask_path)

            plt.figure(figsize=(15, 10))

            plt.subplot(2, 4, 1)
            plt.imshow(image, cmap='gray')
            plt.title(f"Original Image - {image_filename}")

            plt.subplot(2, 4, 2)
            plt.imshow(mask, cmap='gray')
            plt.title(f"Mask Image - {mask_filename}")

            plt.tight_layout()
            plt.show()
        break

