import torch
import torch.nn as nn
import torch.nn.functional as F

from u_net import UNet
from encoder_decoder import EncoderDecoder
from net_utils import *

__all__ = ['simple29_unet', 'simple29_encoderdecoder']

class Simple(nn.Module):
	'''
	'''
	def __init__(self, n_classes, model_name="u_net"):
		super(Simple, self).__init__()
		self.n_classes = n_classes
		self.reconst_loss = None 
		self.model_name = model_name
		self.weight = [0.19, 0.45, 0.29, 0.13, 0.2, 0.33, 0.48, 0.14, 0.36, 0.34, 1.0, 0.43, 0.66, 0.33, 0.51, 0.41, 0.17, 0.31, 0.19, 0.33, 0.57, 0.21, 0.48, 0.49, 0.75, 0.88, 0.49, 0.61, 0.42]
		if self.model_name == "u_net":
			self.layer = UNet(3, n_classes)
		elif self.model_name == "encoder_decoder":
			self.layer = EncoderDecoder(3, n_classes)

	def forward(self, mask, onehot, img=None, seg_gt=None):
		# assert img.size(1) == 3, img.size()
		assert len(seg.size()) == 4, seg.size()
		assert len(mask.size()) == 3, mask.size()
		mask = mask.unsqueeze(1)

		img = img*mask
		seg = seg*mask


		if self.training:
			output = self.layer(img, seg, mask)
		else:
			output = self.layer(img, seg, mask)

		# print(output[0,:,200,200])

		output = output*(1-mask) + seg #transform_seg_one_hot(seg_gt, self.n_classes)*mask

		if self.training:
			self.reconst_loss = F.cross_entropy(input=output, weight=self.weight, target=seg_gt, reduction='sum')
			# target of shape (N,H,W), where each element in [0,C-1], input of shape (N,C,H,W)
			elems = (1-mask).nonzero().size(0)
			self.reconst_loss = self.reconst_loss / elems

		# return squeeze_seg(output, self.n_classes), self.reconst_loss
		return output, self.reconst_loss


def simple29_unet():
	return Simple(29, 'u_net')

def simple29_encoderdecoder():
	return Simple(29, 'encoder_decoder')