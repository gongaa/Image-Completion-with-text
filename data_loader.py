import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data.sampler import Sampler

from cityscape_utils import *
from net_utils import *

class Samplerian(Sampler):
	def __init__(self, train_size, batch_size):
		self.num_data = train_size
		self.num_per_batch = int(train_size / batch_size) 
		# num per batch is actually the num of batches
		self.batch_size = batch_size
		self.range = torch.arange(0,batch_size).view(1, batch_size).long()
		self.leftover_flag = False
		if train_size % batch_size:
			self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
			self.leftover_flag = True

	def __iter__(self):
		rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
		# rand_num = torch.arange(self.num_per_batch).view(-1,1) * self.batch_size
		self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

		self.rand_num_view = self.rand_num.view(-1) # only shuffle the batches

		if self.leftover_flag:
			self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

		return iter(self.rand_num_view)

	def __len__(self):
		return self.num_data

class DataLoaderian(data.Dataset):
	'''
	'''
	def __init__(self, imgs, segs, masks, n_classes, training=True):
		'''
			imgs:  (n, h, w, 3)
			segs:  (n, h, w, 3)
			masks: (n, h, w)
		'''
		size, h, w, channel = imgs.shape
		assert segs.shape == (size, h, w, channel), segs.shape
		assert masks.shape == (size, h, w), masks.shape
		self.training = training
		self.imgs = imgs
		self.masks = masks
		self.n_classes = n_classes
		self.segs = self._transform_segs(segs)
		np.save("Dataset/Cityscape/train_data_dataloaderain_segs", self.segs)
		sys.stdout.write("squeezing segs .... ")
		tic = time.time()
		self.gts = squeeze_seg_np(self.segs, self.n_classes)
		np.save("Dataset/Cityscape/train_data_gt", self.gts)
		print("cost {:.2f}-s".format(time.time()-tic))

	def _transform_segs(self, segs):
		'''
			input segs (n, h, w, 3)
			return segs (n, n_classes, h, w)
		'''
		N,H,W,C = segs.shape
		new_segs = np.zeros((N, self.n_classes, H, W))
		sys.stdout.write("transforming segs ...... ") 
		tic = time.time()
		for b in range(N):
			for h in range(H):
				for w in range(W):
					cls = seg_color2index[tuple(segs[b, h, w].tolist())] 
					new_segs[b, cls, h, w] = 1
		toc = time.time()
		sys.stdout.write("done! cost {:.2f}-s\n".format(toc-tic))
		return new_segs

	def __getitem__(self, index):
		'''
			return 
				img  	(3, h, w)
				seg:	(n_classes, h, w)
				mask:	(h, w)	
		'''
		img = torch.from_numpy(self.imgs[index]).permute(2,0,1).contiguous()
		seg = torch.from_numpy(self.segs[index])
		mask = torch.from_numpy(self.masks[index])
		gt = torch.from_numpy(self.gts[index])
		return img, seg, mask, gt

	def __len__(self):
		return self.imgs.shape[0]


