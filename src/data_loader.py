import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data.sampler import Sampler

from cityscape_utils import *
from cfg import cfg
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
			segs:  (n, h, w)
			masks: (n, h, w)
		'''
		size, h, w, channel = imgs.shape
		assert segs.shape == (size, h, w), segs.shape
		assert masks.shape == (size, h, w), masks.shape
		self.training = training
		self.imgs = imgs
		self.masks = masks
		self.n_classes = n_classes
		self.segs = segs

		# np.save("Dataset/Cityscape/train_data_dataloaderain_segs", self.segs)
		# sys.stdout.write("squeezing segs .... ")
		# tic = time.time()
		# self.gts = squeeze_seg_np(self.segs, self.n_classes)
		# np.save("Dataset/Cityscape/train_data_gt", self.gts)
		# print("cost {:.2f}-s".format(time.time()-tic))

	# def _transform_segs(self, segs):
	# 	'''
	# 		input segs (n, h, w)
	# 		return segs (n, n_classes, h, w)
	# 	'''
	# 	N,H,W = segs.shape
	# 	one_hot_index = np.eye(self.n_classes)
	# 	segs_one_hot = one_hot_index[segs].transpose(0,3,1,2)

	# 	return segs_one_hot

	def __getitem__(self, index):
		'''
			return 
				img  	(3, h, w)
				seg:	(h, w)
				mask:	(h, w)	
		'''
		img = torch.from_numpy(self.imgs[index]- cfg.CITYSCAPE.PIXEL_MEANS).permute(2,0,1).contiguous().float()
		seg = torch.from_numpy(self.segs[index])
		mask = torch.from_numpy(self.masks[index])
		return img, seg, mask

	def __len__(self):
		return self.imgs.shape[0]


img_h = 1024
img_w = 2048
cls = 29
min_area = 20000
max_area = 1024*2048/4
min_total_missing_area = 2000

def add_mask_onehot(file, file_index):
    with open(file, 'r') as f:
        text = json.load(f)
    mask = np.ones((img_h,img_w))
    objects_list = text['objects']
    # sort object according to their idx
    objects_list = sorted(objects_list, key=lambda k: k['idx'])
    num_obj = len(objects_list)
    while True:
        rand_idx = randint(0, num_obj-1)
        # check whether area fullfill requirement
        x_1, x_2, y_1, y_2 = objects_list[rand_idx]['x_min'], objects_list[rand_idx]['x_max'], objects_list[rand_idx]['y_min'], objects_list[rand_idx]['y_max']
        bbox_area = (x_2-x_1)*(y_2-y_1)
        if bbox_area > max_area or bbox_area < min_area: 
            continue
        # generate random crop
#         print(rand_idx)
        w = x_2-x_1
        h = y_2-y_1
#         print(x_1,x_2,y_1,y_2)
        rand_w = randint(w//2, w)
        rand_h = randint(h//2, h)
        rand_x1 = randint(x_1, x_2-rand_w)
        rand_y1 = randint(y_1, y_2-rand_h)
        rand_x2 = rand_x1 + rand_w
        rand_y2 = rand_y1 + rand_h
        mask[rand_y1:rand_y2, rand_x1:rand_x2] = 0
#         print("axis",rand_x1,rand_x2,rand_y1,rand_y2)
        break

    one_hot = np.zeros(cls)
    # generate one-hot vector
    for idx in objects_list[rand_idx]['intersection_id']:
        # check whether its bbox is fully contained in cropped window
        # check whether its area is bigger than the threshold
        x_1,x_2,y_1,y_2 = objects_list[idx]['x_min'],objects_list[idx]['x_max'],objects_list[idx]['y_min'],objects_list[idx]['y_max']
        if x_1>rand_x1 and x_2<rand_x2 and y_1>rand_y1 and y_2<rand_y2 and (x_2-x_1)*(y_2-y_1)>min_total_missing_area:
            lin_idx = label2index[objects_list[idx]['label']] # label that lin set
            one_hot[lin_idx] = one_hot[lin_idx] + 1 
        
    return mask.astype(bool), one_hot.astype(np.int8)