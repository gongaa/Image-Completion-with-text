import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.u_net import UNet
from nets.encoder_decoder import EncoderDecoder
from net_utils import *

class Model(nn.Module):
	'''
	'''
	def __init__(self, n_classes, model_name="u_net"):
		super(Model, self).__init__()
		self.n_classes = n_classes
		self.reconst_loss = None 
		self.model_name = model_name
		if self.model_name == "u_net":
			self.layer = UNet(3, n_classes)
		elif self.model_name == "encoder_decoder":
			self.layer = EncoderDecoder(3, n_classes)

	def forward(self, img, seg, mask, seg_gt=None):
		assert img.size(1) == 3, img.size()
		assert len(seg.size()) == 4, seg.size()
		assert len(mask.size()) == 3, mask.size()
		mask = mask.unsqueeze(1)

		img = img*mask
		seg = seg*mask

		if self.training:
			output = self.layer(img, seg, mask)
		else:
			output = self.layer(img, seg, mask)
		# print(torch.mean(output))
		# print(torch.var(output))
		output = output*(1-mask) + seg

		if self.training:
			self.reconst_loss = F.cross_entropy(output, seg_gt, reduction='sum')
			elems = (1-mask).nonzero().numel()
			self.reconst_loss = self.reconst_loss / elems

		# return squeeze_seg(output, self.n_classes), self.reconst_loss
		return output, self.reconst_loss

	def _init_weights(self):
		def normal_init(m, mean, stddev, truncated=False):
			"""
			weight initalizer: truncated normal and random normal.
			"""
			# x is a parameter
			if truncated:
				m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
			else:
				m.weight.data.normal_(mean, stddev)
				m.bias.data.zero_()

		self.layer._init_weights()

	def create_architecture(self):
		self._init_weights()
