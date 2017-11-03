# coding: utf-8

import numpy as np

import torch
import torch.utils.data

import torchvision
import torchvision.models
import torchvision.transforms


def to_tensor():
    def _to_tensor(image):
        if len(image.shape) == 3:
            return torch.from_numpy(
                image.transpose(2, 0, 1).astype(np.float32))
        else:
            return torch.from_numpy(image[None, :, :].astype(np.float32))

    return _to_tensor


def normalize(mean, std):
    mean = np.array(mean)
    std = np.array(std)

    def _normalize(image):
        image = np.asarray(image).astype(np.float32) / 255.
        image = (image - mean) / std
        return image

    return _normalize


def random_erase(p, area_ratio_range, min_aspect_ratio, max_attempt=20):
    sl, sh = area_ratio_range
    rl, rh = min_aspect_ratio, 1. / min_aspect_ratio

    def _random_erase(image):
        image = np.asarray(image).copy()

        if np.random.random() > p:
            return image

        h, w = image.shape[:2]
        image_area = h * w

        for _ in range(max_attempt):
            mask_area = np.random.uniform(sl, sh) * image_area
            aspect_ratio = np.random.uniform(rl, rh)
            mask_h = int(np.sqrt(mask_area * aspect_ratio))
            mask_w = int(np.sqrt(mask_area / aspect_ratio))

            if mask_w < w and mask_h < h:
                x0 = np.random.randint(0, w - mask_w)
                y0 = np.random.randint(0, h - mask_h)
                x1 = x0 + mask_w
                y1 = y0 + mask_h
                image[y0:y1, x0:x1] = np.random.uniform(0, 1)
                break

        return image

    return _random_erase


def get_loader(batch_size, num_workers, config):
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        normalize(mean, std),
        random_erase(
            config['random_erase_prob'],
            config['random_erase_area_ratio_range'],
            config['random_erase_min_aspect_ratio'],
            config['random_erase_max_attempt']),
        to_tensor(),
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ])

    dataset_dir = '~/.torchvision/datasets/CIFAR10'
    train_dataset = torchvision.datasets.CIFAR10(
        dataset_dir, train=True, transform=train_transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(
        dataset_dir, train=False, transform=test_transform, download=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, test_loader
