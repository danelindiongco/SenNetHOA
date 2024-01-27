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


if __name__ == '__main__':
    CFG = CFG()

    set_seed(CFG.seed)

    train_groups = CFG.train_groups
    valid_groups = CFG.valid_groups

    gt_df = pd.read_csv(CFG.gt_df)
    gt_df["img_path"] = gt_df["img_path"].apply(lambda x: os.path.join(CFG.data_root, x))
    gt_df["msk_path"] = gt_df["msk_path"].apply(lambda x: os.path.join(CFG.data_root, x))

    train_df = gt_df.query("group in @train_groups").reset_index(drop=True)
    valid_df = gt_df.query("group in @valid_groups").reset_index(drop=True)

    train_img_paths = train_df["img_path"].values.tolist()
    train_msk_paths = train_df["msk_path"].values.tolist()

    valid_img_paths = valid_df["img_path"].values.tolist()
    valid_msk_paths = valid_df["msk_path"].values.tolist()

    if CFG.debug:
        train_img_paths = train_img_paths[:CFG.train_bs * 5]
        train_msk_paths = train_msk_paths[:CFG.train_bs * 5]
        valid_img_paths = valid_img_paths[:CFG.valid_bs * 3]
        valid_msk_paths = valid_msk_paths[:CFG.valid_bs * 3]

    train_dataset = BuildDataset(train_img_paths, train_msk_paths, transforms=CFG.data_transforms['train'])
    valid_dataset = BuildDataset(valid_img_paths, valid_msk_paths, transforms=CFG.data_transforms['valid'])

    train_loader = DataLoader(train_dataset, batch_size=CFG.train_bs, num_workers=0, shuffle=True, pin_memory=True,
                              drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_bs, num_workers=0, shuffle=False, pin_memory=True)

    sample_ids = [random.randint(0, len(train_img_paths)) for _ in range(5)]
    for sample_id in sample_ids:
        data_name = train_df.loc[sample_id]["id"]
        img, msk = train_dataset[sample_id]
        img = img.permute((1, 2, 0)).numpy() * 255.0
        img = img.astype('uint8')
        msk = (msk * 255).numpy().astype('uint8')
        plt.figure(figsize=(9, 4))
        print(data_name)
        plt.axis('off')
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.subplot(1, 3, 2)
        plt.imshow(msk)
        plt.subplot(1, 3, 3)
        plt.imshow(img, cmap='bone')
        plt.imshow(msk, alpha=0.5)
        plt.show()

