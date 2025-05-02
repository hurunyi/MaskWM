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


class CropResizePad(nn.Module):
    def __init__(self, resize_min=None, resize_max=None, crop_min=None, crop_max=None):
        super(CropResizePad, self).__init__()
        self.resize_min = resize_min
        self.resize_max = resize_max
        self.crop_min = crop_min
        self.crop_max = crop_max

    def get_random_resize_size(self, h, w):
        if self.resize_min is None or self.resize_max is None:
            raise ValueError("resize_min and resize_max must be provided")
        output_size = (
            torch.randint(int(self.resize_min * h), int(self.resize_max * h) + 1, size=(1,)).item(),
            torch.randint(int(self.resize_min * w), int(self.resize_max * w) + 1, size=(1,)).item()
        )
        return output_size

    def get_random_crop_size(self, h, w):
        if self.crop_min is None or self.crop_max is None:
            raise ValueError("crop_min and crop_max must be provided")
        output_size = (
            torch.randint(int(self.crop_min * h), min(int(self.crop_max * h) + 1, h), size=(1,)).item(),
            torch.randint(int(self.crop_min * w), min(int(self.crop_max * w) + 1, w), size=(1,)).item()
        )
        return output_size

    def forward(self, images, masks, sizes=None, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
        batch_size, channels, original_h, original_w = images.shape

        # Resize with random proportions
        if sizes is None:
            new_size = self.get_random_resize_size(original_h, original_w)
            crop_size = self.get_random_crop_size(*new_size)
        else:
            new_size = int(sizes[0] * original_h), int(sizes[1] * original_w)
            crop_size_0, crop_size_1 = sizes[2], sizes[3]
            crop_size = int(crop_size_0 * new_size[0]), int(crop_size_1 * new_size[1])

        images = F.resize(images, new_size, antialias=False)
        masks = F.resize(masks, new_size, interpolation=InterpolationMode.NEAREST)
        crop_h, crop_w = crop_size

        # Ensure the crop size is valid for the resized images
        crop_h = min(crop_h, new_size[0])
        crop_w = min(crop_w, new_size[1])

        i, j, h, w = transforms.RandomCrop.get_params(images[0], output_size=(crop_h, crop_w))
        cropped_images = images[:, :, i:i + h, j:j + w]
        cropped_masks = masks[:, :, i:i + h, j:j + w]

        # Ensure the crop size is valid for padding
        if h > original_h or w > original_w:
            raise ValueError("Crop size is larger than the original image size.")

        # Randomly place the crop and add padding
        pad_top = random.randint(0, max(0, original_h - h))
        pad_left = random.randint(0, max(0, original_w - w))
        pad_bottom = original_h - h - pad_top
        pad_right = original_w - w - pad_left

        # Generate a random color for each channel within the normalized range
        min_val, max_val = images.min().item(), images.max().item()
        pad_color = (max_val - min_val) * torch.rand((batch_size, channels, 1, 1), dtype=images.dtype,
                                                     device=images.device) + min_val

        # Create a padded image with the random color
        padded_images = pad_color.repeat(1, 1, original_h, original_w)
        padded_masks = torch.zeros((batch_size, masks.shape[1], original_h, original_w), dtype=masks.dtype,
                                   device=masks.device)

        # Place the cropped images and masks into the padded versions
        padded_images[:, :, pad_top:pad_top + h, pad_left:pad_left + w] = cropped_images
        padded_masks[:, :, pad_top:pad_top + h, pad_left:pad_left + w] = cropped_masks

        return padded_images, padded_masks


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
