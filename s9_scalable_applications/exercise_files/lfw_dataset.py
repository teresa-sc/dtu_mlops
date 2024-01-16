"""LFW dataloading."""
import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image
from torchvision.utils import make_grid
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

class LFWDataset(Dataset):
    """Initialize LFW dataset."""

    def __init__(self, path_to_folder: str, transform) -> None:
        """Initialize LFW dataset."""
        self.data_path = path_to_folder
        self.data = glob.glob(os.path.join(self.data_path, "*/*.jpg"))
        self.transform = transform

    def __len__(self):
        """Return length of dataset."""
        data_len = len(self.data)
        return data_len

    def __getitem__(self, index: int) -> torch.Tensor:
        """Get item from dataset."""

        img = Image.open(self.data[index])
        return self.transform(img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-path_to_folder", default="/home/teresa/courses/lfw", type=str)
    parser.add_argument("-batch_size", default=128, type=int)
    parser.add_argument("-num_workers", default=1, type=int)
    parser.add_argument("-visualize_batch", action="store_true")
    parser.add_argument("-get_timing", action="store_true")
    parser.add_argument("-batches_to_check", default=100, type=int)

    args = parser.parse_args()

    lfw_trans = transforms.Compose([transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)), transforms.ToTensor()])

    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)

    # Define dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if args.visualize_batch:
        all_images = []
        for batch_idx, _batch in enumerate(dataloader):
            if batch_idx > 0:
                break
            all_images.append(_batch)
        show(make_grid(all_images))


    if args.get_timing:
        # lets do some repetitions
        res = []
        for i in range(5):
            start = time.time()
            for batch_idx, _batch in enumerate(dataloader):
                if batch_idx > args.batches_to_check:
                    print(i, "done")
                    break
            end = time.time()

            res.append(end - start)

        res = np.array(res)
        print("Timing: ", np.mean(res),"+-",np.std(res))
