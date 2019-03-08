import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


def adjust_learning_rate(optimizer, decay=0.1):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']


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

def squeeze_seg_np(seg, n_cls):
	'''
		input  (n, n_cls, h, w)
		output (n, h, w)
	'''
	bs, c, H, W = seg.shape
	assert c == n_cls , "input seg channel is not n_cls"
	lbls = np.transpose(np.nonzero(seg))
	out_seg = np.zeros((bs, H, W))

	for loc in lbls:
		out_seg[loc[0], loc[2], loc[3]] = loc[1]
	return out_seg


def mask2box(mask):
	'''
		input: mask of shape (bs, h, w) outer region is 1
		output: bbox (bs, 4) of (h1, w1, h2, w2) 
	'''
	inner_mask = (1-mask)
	bs = mask.size(0)
	output = []
	for i in range(bs):
		nonzero_indices = torch.nonzero(inner_mask[i]) # inner region indices of shape (k, 2) 
		min_hw,_ = torch.min(nonzero_indices, dim=0)
		max_hw,_ = torch.max(nonzero_indices, dim=0)
		i_hw = torch.cat([min_hw.view(1,2), max_hw.view(1,2)], dim=1)
		output.append(i_hw)
	return torch.cat(output, dim=0)


