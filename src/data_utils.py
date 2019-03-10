import numpy as np
import os
import sys
import pickle
import json
import time
import glob
from scipy.misc import imread

from cityscape_utils import *
'''
	input:			 dataset name(str)

	return np data:	
		n_classes	: int

		train_imgs	: (n_t, h, w, 3)
		train_segs	: (n_t, h, w)
		train_masks	: (n_t, h, w)  missing region is 0, known region is 1 

		val_imgs	: (n_v, h, w, 3)
		val_segs	: (n_v, h, w)
		val_masks	: (n_v, h, w)
'''
def get_data(dataset, epoch=0, flip=True, val=True):
	### explicitly set flip = True #######
	if dataset == "cityscape":
		dataset_train_dir = "Dataset/Cityscape_level1/train/"
		dataset_val_dir = "Dataset/Cityscape/val_data"

		seg_dir = "Dataset/Cityscape/gtFine"
		n_classes = 29
		if not val:
			if os.path.exists(dataset_train_dir+"_segs.npy"): # if stored file exist
				sys.stdout.write("loading cityscape train dataset from {} ".format(dataset_train_dir))
				train_masks = np.load(dataset_train_dir+"mask_one_hot/epoch"+str(epoch)+"_mask.npy")
				train_onehots = np.load(dataset_train_dir+"mask_one_hot/epoch"+str(epoch)+"_onehot.npy")
				train_imgs = None	# for training segmentation network, we don't need this for now
				train_segs = np.load(dataset_train_dir+"_seg.npy")	# ground-truth label
				print("loaded")
				train_size = train_segs.shape[0]

			return n_classes, train_imgs, train_segs, train_masks, None, None ,None

		else:
			if os.path.exists(dataset_val_dir+"_segs.npy"): # if stored file exist
				sys.stdout.write("loading cityscape flipped val dataset from {} ".format(dataset_val_dir))
				val_segs = np.load(dataset_val_dir+"_segs.npy")
				val_imgs = np.load(dataset_val_dir+"_imgs.npy")
				val_masks = np.load(dataset_val_dir+"_masks.npy")
				print("loaded")
				val_size = val_segs.shape[0]
			else: # else, loading val data by hand, which takes about 10 mins
				val_seg_dir = os.path.join(seg_dir, 'val')
				val_seg_files = glob.glob( val_seg_dir+ '/**/*labelIds.png', recursive=True)
				val_segs = []
				val_imgs = []
				val_masks = []
				val_size = len(val_seg_files)
				for o,val_seg_file in enumerate(val_seg_files):
					val_img_file = val_seg_file.replace("gtFine_labelIds", "leftImg8bit")
					val_img_file = val_img_file.replace("gtFine", "leftImg8bit")
					val_mask_file = val_seg_file.replace("labelIds.png", "polygons_mask.json")

					seg = np.vectorize(seg_id2index.get)(imread(val_seg_file))
					assert seg.shape == (1024, 2048), seg.shape
					img = imread(val_img_file, mode="RGB")
					assert img.shape == (1024, 2048, 3), img.shape
					with open(val_mask_file, 'r') as f:
						file = json.load(f)
						mask = np.array(file['mask'])
					assert mask.shape == (1024, 2048), mask.shape
					val_segs.append(seg.astype(np.uint8))
					val_imgs.append(img)
					val_masks.append(mask.astype(np.uint8))
					sys.stdout.write("\rloading cityscape val dataset <<<<<<< {}/{} >>>>>>".format(o+1, val_size))
					sys.stdout.flush()
				print()
				val_segs = np.array(val_segs)
				val_imgs = np.array(val_imgs)
				val_masks = np.array(val_masks)
				### store file into dir
				print("saving val data into {}".format(dataset_val_dir))
				np.save(dataset_val_dir+"_segs",val_segs)
				np.save(dataset_val_dir+"_imgs",val_imgs)
				np.save(dataset_val_dir+"_masks",val_masks)		

			return n_classes, None, None, None, val_imgs, val_segs, val_masks
	else:
		raise Exception("dataset not defined")


