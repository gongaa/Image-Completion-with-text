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

	def forward(self, img, seg, mask, gt_data=None):
		assert seg.size(1) == self.n_classes, seg.size()
		assert img.size(1) == 3, img.size()
		output = self.layer(img, seg, mask)
		if self.training:
			self.reconst_loss = F.cross_entropy(output, gt_data, reduction='sum')
			elems = (1-mask).nonzero().numel()
			self.reconst_loss = self.reconst_loss / elems
		# return squeeze_seg(output, self.n_classes), self.reconst_loss
		return output, self.reconst_loss


