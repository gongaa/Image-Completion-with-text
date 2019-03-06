import torch
import torch.nn as nn
import torch.nn.functional as F

from U_net.u_net import UNet



def unsqueeze_seg(seg, n_cls):
	'''
		input  (n, h, w)
		output (n, n_cls, h, w)
	'''
	bs, H, W = seg.size()

	out_seg = torch.zeros((bs, n_cls, h, w))
	for i in torch.arange(bs):
		for h in torch.arange(H):
			for w in torch.arange(W):
				lbl_ind = seg[i, h, w]
				out_seg[i, lbl_ind, h, w] = 1
	return out_seg 


def squeeze_seg(seg, n_cls):
	'''
		input  (n, n_cls, h, w)
		output (n, h, w)
	'''
	bs, c, H, W = seg.size()
	assert c == n_cls , "input seg channel is not n_cls"
	lbls = torch.nonzero(seg)
	out_seg = torch.zeros((bs, H, W))

	for loc in lbls:
		out_seg[loc[0], loc[2], loc[3]] = loc[1]
	return out_seg


class UNetModel(nn.Module):
	'''
	'''
	def __init__(self, n_classes):
		super(Unet_model, self).__init__(self, n_classes)
		self._n_classes = n_classes
		self._reconst_loss = None 
		self.layer = UNet(3, n_classes)

	def forward(self, img, seg, mask, gt_data=None):
		assert seg.size(1) == self._n_classes, seg.size()
		assert img.size(0) == 3, img.size()
		output = self.layer(img, seg, mask)
		if self.training:
			self._reconst_loss = F.cross_entropy(output, gt_data, reduction='sum')
			elems = (1-mask).nonzero().numel()
			self._reconst_loss = self._reconst_loss / elems
		return squeeze_seg(output, self.n_classes), self._reconst_loss


