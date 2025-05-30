from torch import nn
from .noise_layers import *


class Noise(nn.Module):

	def __init__(self, layers):
		super(Noise, self).__init__()
		self.noise = eval(layers)

	def forward(self, image, mask):
		results = self.noise(image, mask)
		return results
