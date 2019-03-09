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

	def forward(self, img, seg, mask):
		assert img.size(1) == 3, img.size()
		assert len(seg.size()) == 3, seg.size()
		assert len(mask.size()) == 3, mask.size()
		mask = mask.unsqueeze(1)
		seg_one_hot = transform_seg_one_hot(seg, self.n_classes)
		img = img*mask
		seg_one_hot = seg_one_hot*mask
		if self.training:
			output = self.layer(img, seg_one_hot, mask)
		else:
			output = self.layer(img, seg_one_hot, mask)

		if self.training:
			self.reconst_loss = F.cross_entropy(output, seg, reduction='sum')
			elems = (1-mask).nonzero().numel()
			self.reconst_loss = self.reconst_loss / elems
		# return squeeze_seg(output, self.n_classes), self.reconst_loss
		return output, self.reconst_loss


