from . import Identity
import torch.nn as nn
import random


def get_random_int(int_range: [int]):
	return random.randint(int_range[0], int_range[1])


class Combined(nn.Module):

	def __init__(self, list=None):
		super(Combined, self).__init__()
		if list is None:
			list = [Identity()]
		self.list = list

	def forward(self, image, mask):
		id = get_random_int([0, len(self.list) - 1])
		return self.list[id](image, mask)
