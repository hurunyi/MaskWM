# This code snippet is adapted from the Watermark Anything project by Facebook Research:
# https://github.com/facebookresearch/watermark-anything
# Original source file: watermark_anything/augmentation/geometric.py
# License: MIT License
# Copyright (c) Meta Platforms, Inc. and affiliates.


import random

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, image, mask, *args, **kwargs):
        return image, mask


class Rotate(nn.Module):
    def __init__(self, min_angle=-10, max_angle=10):
        super(Rotate, self).__init__()
        self.min_angle = min_angle
        self.max_angle = max_angle

    def get_random_angle(self):
        if self.min_angle is None or self.max_angle is None:
            raise ValueError("min_angle and max_angle must be provided")
        return torch.randint(self.min_angle, self.max_angle + 1, size=(1,)).item()

    def forward(self, image, mask, angle=None):
        if angle is None:
            angle = self.get_random_angle()
        image = F.rotate(image, angle)
        mask = F.rotate(mask, angle)
        return image, mask


class Resize(nn.Module):
    def __init__(self, min_size=None, max_size=None):
        super(Resize, self).__init__()
        self.min_size = min_size  # float between 0 and 1, representing the total area of the output image compared to the input image
        self.max_size = max_size

    def get_random_size(self, h, w):
        if self.min_size is None or self.max_size is None:
            raise ValueError("min_size and max_size must be provided")
        output_size = (
            torch.randint(int(self.min_size * h), int(self.max_size * h) + 1, size=(1,)).item(),
            torch.randint(int(self.min_size * w), int(self.max_size * w) + 1, size=(1,)).item()
        )
        return output_size

    def forward(self, image, mask, size=None):
        h, w = image.shape[-2:]
        if size is None:
            output_size = self.get_random_size(h, w)
        else:
            output_size = (int(size * h), int(size * w))
        image = F.resize(image, output_size, antialias=True)
        mask = F.resize(mask, output_size, interpolation=InterpolationMode.NEAREST)
        return image, mask


class Crop(nn.Module):
    def __init__(self, min_size=None, max_size=None):
        super(Crop, self).__init__()
        self.min_size = min_size
        self.max_size = max_size

    def get_random_size(self, h, w):
        if self.min_size is None or self.max_size is None:
            raise ValueError("min_size and max_size must be provided")
        output_size = (
            torch.randint(int(self.min_size * h), int(self.max_size * h) + 1, size=(1,)).item(),
            torch.randint(int(self.min_size * w), int(self.max_size * w) + 1, size=(1,)).item()
        )
        return output_size

    def forward(self, image, mask, size=None):
        h, w = image.shape[-2:]
        if size is None:
            output_size = self.get_random_size(h, w)
        else:
            output_size = (int(size * h), int(size * w))
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=output_size)
        image = F.crop(image, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)
        return image, mask


class UpperLeftCrop(nn.Module):
    def __init__(self, min_size=None, max_size=None):
        super(UpperLeftCrop, self).__init__()
        self.min_size = min_size
        self.max_size = max_size

    def get_random_size(self, h, w):
        if self.min_size is None or self.max_size is None:
            raise ValueError("min_size and max_size must be provided")
        output_size = (
            torch.randint(int(self.min_size * h), int(self.max_size * h) + 1, size=(1,)).item(),
            torch.randint(int(self.min_size * w), int(self.max_size * w) + 1, size=(1,)).item()
        )
        return output_size

    def forward(self, image, mask, size=None):
        h, w = image.shape[-2:]
        if size is None:
            output_size = self.get_random_size(h, w)
        else:
            output_size = (int(size * h), int(size * w))
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=output_size)
        image = F.crop(image, 0, 0, h, w)
        mask = F.crop(mask, 0, 0, h, w)
        return image, mask


class Perspective(nn.Module):
    def __init__(self, min_distortion_scale=0.1, max_distortion_scale=0.5, random_seed=None):
        super(Perspective, self).__init__()
        self.min_distortion_scale = min_distortion_scale
        self.max_distortion_scale = max_distortion_scale
        self.random_seed = random_seed
    
    def get_random_distortion_scale(self):
        if self.min_distortion_scale is None or self.max_distortion_scale is None:
            raise ValueError("min_distortion_scale and max_distortion_scale must be provided")
        return self.min_distortion_scale + torch.rand(1).item() * \
               (self.max_distortion_scale - self.min_distortion_scale)

    def forward(self, image, mask, distortion_scale=None):
        if distortion_scale is None:
            distortion_scale = self.get_random_distortion_scale()
        else:
            distortion_scale = distortion_scale
        width, height = image.shape[-1], image.shape[-2]
        startpoints, endpoints = self.get_perspective_params(width, height, distortion_scale, self.random_seed)
        image = F.perspective(image, startpoints, endpoints)
        mask = F.perspective(mask, startpoints, endpoints)
        return image, mask

    @staticmethod
    def get_perspective_params(width, height, distortion_scale, random_seed=None):
        half_height = height // 2
        half_width = width // 2
        
        if random_seed is not None:
            torch.manual_seed(random_seed)
            random.seed(random_seed)
        
        topleft = [
            int(torch.randint(0, int(distortion_scale * half_width) + 1, size=(1,)).item()),
            int(torch.randint(0, int(distortion_scale * half_height) + 1, size=(1,)).item())
        ]
        topright = [
            int(torch.randint(width - int(distortion_scale * half_width) - 1, width, size=(1,)).item()),
            int(torch.randint(0, int(distortion_scale * half_height) + 1, size=(1,)).item())
        ]
        botright = [
            int(torch.randint(width - int(distortion_scale * half_width) - 1, width, size=(1,)).item()),
            int(torch.randint(height - int(distortion_scale * half_height) - 1, height, size=(1,)).item())
        ]
        botleft = [
            int(torch.randint(0, int(distortion_scale * half_width) + 1, size=(1,)).item()),
            int(torch.randint(height - int(distortion_scale * half_height) - 1, height, size=(1,)).item())
        ]
        startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        endpoints = [topleft, topright, botright, botleft]
        return startpoints, endpoints


class HorizontalFlip(nn.Module):
    def __init__(self):
        super(HorizontalFlip, self).__init__()

    def forward(self, image, mask, *args, **kwargs):
        image = F.hflip(image)
        mask = F.hflip(mask)
        return image, mask


class CropResize(nn.Module):
    def __init__(self, min_ratio=0.5, max_ratio=1):
        super(CropResize, self).__init__()
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def get_random_ratio(self):
        if self.min_ratio is None or self.max_ratio is None:
            raise ValueError("min_ratio and max_ratio must be provided")
        h_ratio = torch.empty(1).uniform_(self.min_ratio, self.max_ratio).item()
        w_ratio = torch.empty(1).uniform_(self.min_ratio, self.max_ratio).item()
        return h_ratio, w_ratio

    def forward(self, image, mask):
        _, _, h, w = image.shape
        h_ratio, w_ratio = self.get_random_ratio()
        crop_h, crop_w = int(h * h_ratio), int(w * w_ratio)

        top = torch.randint(0, h - crop_h + 1, (1,)).item()
        left = torch.randint(0, w - crop_w + 1, (1,)).item()

        image = F.crop(image, top, left, crop_h, crop_w)
        mask = F.crop(mask, top, left, crop_h, crop_w)

        image = F.resize(image, (h, w))
        mask = F.resize(mask, (h, w), interpolation=InterpolationMode.NEAREST)

        return image, mask
