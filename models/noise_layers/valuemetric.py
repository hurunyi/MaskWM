import torch
import torch.nn as nn
import numpy as np
from kornia.filters import GaussianBlur2d, MedianBlur
import torchvision.transforms.functional as TF


class Identity(nn.Module):
	def __init__(self):
		super(Identity, self).__init__()

	def forward(self, image, mask):
		return image, mask


class MF(nn.Module):

	def __init__(self, kernel=3):
		super(MF, self).__init__()
		self.middle_filter = MedianBlur((kernel, kernel))

	def forward(self, image, mask):
		image = (image + 1) / 2
		filtered = self.middle_filter(image)
		filtered = filtered * 2 - 1
		return filtered, mask


class GF(nn.Module):

	def __init__(self, sigma=1.0, kernel=5):
		super(GF, self).__init__()
		self.gaussian_filter = GaussianBlur2d((kernel, kernel), (sigma, sigma))

	def forward(self, image, mask):
		image = (image + 1) / 2
		filtered = self.gaussian_filter(image)
		filtered = filtered * 2 - 1
		return filtered, mask


class GN(nn.Module):

	def __init__(self, mu=0, sigma=0.1):
		super(GN, self).__init__()
		self.mu = mu
		self.sigma = sigma

	def gaussian_noise(self, image, mu, sigma):
		noise = torch.Tensor(np.random.normal(mu, sigma/2, image.shape)).to(image.device)
		out = image + noise
		return torch.clamp(out, 0, 1)

	def forward(self, image, mask):
		image = (image + 1) / 2
		noised = self.gaussian_noise(image, self.mu, self.sigma)
		noised = noised * 2 - 1
		return noised, mask


class SP(nn.Module):

	def __init__(self, prob=0.1):
		super(SP, self).__init__()
		self.prob = prob

	def sp_noise(self, image, prob):
		prob_zero = prob / 2
		prob_one = 1 - prob_zero
		rdn = torch.rand(image.shape).to(image.device)

		output = torch.where(rdn > prob_one, torch.zeros_like(image).to(image.device), image)
		output = torch.where(rdn < prob_zero, torch.ones_like(output).to(output.device), output)

		return output

	def forward(self, image, mask):
		image = (image + 1) / 2
		noised = self.sp_noise(image, self.prob)
		noised = noised * 2 - 1
		return noised, mask


class BrightnessAdjustment(nn.Module):
    def __init__(self, brightness_range=(0.7, 1.3)):
        super(BrightnessAdjustment, self).__init__()
        self.brightness_range = brightness_range
        
    def forward(self, image, mask):
        image = (image + 1) / 2
        
        factor = torch.FloatTensor(1).uniform_(self.brightness_range[0], self.brightness_range[1]).item()
        image = TF.adjust_brightness(image, factor)
        
        image = image * 2 - 1
        return image, mask


class ContrastAdjustment(nn.Module):
    def __init__(self, contrast_range=(0.7, 1.3)):
        super(ContrastAdjustment, self).__init__()
        self.contrast_range = contrast_range
        
    def forward(self, image, mask):
        image = (image + 1) / 2
        
        factor = torch.FloatTensor(1).uniform_(self.contrast_range[0], self.contrast_range[1]).item()
        image = TF.adjust_contrast(image, factor)
        
        image = image * 2 - 1
        return image, mask


class HueAdjustment(nn.Module):
    def __init__(self, hue_range=(-0.1, 0.1)):
        super(HueAdjustment, self).__init__()
        self.hue_range = hue_range
        
    def forward(self, image, mask):
        image = (image + 1) / 2
        
        factor = torch.FloatTensor(1).uniform_(self.hue_range[0], self.hue_range[1]).item()
        image = TF.adjust_hue(image, factor)
        
        image = image * 2 - 1
        return image, mask


class SaturationAdjustment(nn.Module):
    def __init__(self, saturation_range=(0.7, 1.3)):
        super(SaturationAdjustment, self).__init__()
        self.saturation_range = saturation_range
        
    def forward(self, image, mask):
        image = (image + 1) / 2
        
        factor = torch.FloatTensor(1).uniform_(self.saturation_range[0], self.saturation_range[1]).item()
        image = TF.adjust_saturation(image, factor)
        
        image = image * 2 - 1
        return image, mask


class ImageAdjustment(nn.Module):
    def __init__(self, brightness_range=(0.7, 1.3), contrast_range=(0.7, 1.3), 
                 hue_range=(-0.1, 0.1), saturation_range=(0.7, 1.3)):
        super(ImageAdjustment, self).__init__()
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.hue_range = hue_range
        self.saturation_range = saturation_range
        
    def forward(self, image, mask):
        image = (image + 1) / 2
        
        adjustment_type = torch.randint(0, 4, (1,)).item()
        
        if adjustment_type == 0:  # 亮度
            factor = torch.FloatTensor(1).uniform_(self.brightness_range[0], self.brightness_range[1]).item()
            image = TF.adjust_brightness(image, factor)
        elif adjustment_type == 1:  # 对比度
            factor = torch.FloatTensor(1).uniform_(self.contrast_range[0], self.contrast_range[1]).item()
            image = TF.adjust_contrast(image, factor)
        elif adjustment_type == 2:  # 色调
            factor = torch.FloatTensor(1).uniform_(self.hue_range[0], self.hue_range[1]).item()
            image = TF.adjust_hue(image, factor)
        else:  # 饱和度
            factor = torch.FloatTensor(1).uniform_(self.saturation_range[0], self.saturation_range[1]).item()
            image = TF.adjust_saturation(image, factor)
        
        image = image * 2 - 1
        return image, mask


class Resize(nn.Module):
	def __init__(self, scale_factor=0.5):
		super(Resize, self).__init__()
		self.scale_factor = scale_factor
	
	def forward(self, image, mask):
		image = (image + 1) / 2
		
		original_size = image.shape[-2:]
		small_size = [int(dim * self.scale_factor) for dim in original_size]
		
		small_image = TF.resize(image, small_size, antialias=True)
		resized_image = TF.resize(small_image, original_size, antialias=True)
		
		resized_image = resized_image * 2 - 1
		
		return resized_image, mask

