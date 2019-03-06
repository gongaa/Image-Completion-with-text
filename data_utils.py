import numpy as np
import os
import sys
import json
import glob
from scipy.misc import imread
'''
	input:			 dataset name(str)

	return np data:	
		n_classes	: int

		train_imgs	: (n_t, h, w, 3)
		train_segs	: (n_t, h, w, n_classes)
		train_masks	: (n_t, h, w)  missing region is 0, known region is 1 

		val_imgs	: (n_v, h, w, 3)
		val_segs	: (n_v, h, w, n_classes)
		val_masks	: (n_v, h, w)
'''
def get_data(dataset, flip=True, val=True):
	if dataset == "cityscape":
		seg_dir = "Dataset/Cityscape/gtFine"
		n_classes = 29

		train_seg_dir = os.path.join(seg_dir, 'train')
		train_seg_files = glob.glob( train_seg_dir+ '/**/*color.png', recursive=True)
		train_segs = []
		train_imgs = []
		train_masks = []
		train_size = len(train_seg_files)

		for o,train_seg_file in enumerate(train_seg_files):
			train_img_file = train_seg_file.replace("gtFine", "leftImage8bit")
			train_mask_file = train_seg_file.replace("color.png", "polygons_mask.json")

			seg = imread(train_seg_file, mode='RGB')
			img = imread(train_img_file, mode="RGB")
			with open(train_mask_file, 'r') as f:
				file = json.load(f)
				mask = file['mask']
			train_segs.append(seg)
			train_masks.apend(np.array(mask))
			sys.stdout.write("\rloading cityscape train dataset <<<<<<< {}/{} >>>>>>".format(o+1, train_size))
			sys.stdout.flush()
		print()
		
		if flip:
			for i in range(train_size):
				train_flip_seg  = train_segs[i].flip(1)
				train_flip_mask = train_masks[i].flip(1)
				train_flip_img  = train_imgs[i].flip(1)
				train_segs.append(train_flip_seg)
				train_imgs.append(train_flip_img)
				train_masks.append(train_flip_mask)
				sys.stdout.write("\rflipping cityscape train dataset <<<<<<< {}/{} >>>>>>".format(i+1, train_size))
				sys.stdout.flush()
			print()
		train_segs = np.array(train_segs)
		train_imgs = np.array(train_imgs)
		train_masks = np.array(train_masks)

		np.random.shuffle(train_segs)
		np.random.shuffle(train_imgs)
		np.random.shuffle(train_masks)

		if not val:
			return n_classes, train_imgs, train_segs, train_masks, None, None ,None

		val_seg_dir = os.path.join(seg_dir, 'val')
		val_seg_files = glob.glob( val_seg_dir+ '/**/*color.png', recursive=True)
		val_segs = []
		val_imgs = []
		val_masks = []
		val_size = len(val_seg_files)
		for o,val_seg_file in enumerate(val_seg_files):
			val_img_file = val_seg_file.replace("gtFine", "leftImage8bit")
			val_mask_file = val_seg_file.replace("color.png", "polygons_mask.json")

			seg = imread(val_seg_file, mode='RGB')
			img = imread(val_img_file, mode="RGB")
			with open(val_mask_file, 'r') as f:
				file = json.load(f)
				mask = file['mask']
			val_segs.append(seg)
			val_masks.apend(np.array(mask))
			sys.stdout.write("\rloading cityscape val dataset <<<<<<< {}/{} >>>>>>".format(o+1, val_size))
			sys.stdout.flush()
		print()
		val_segs = np.array(val_segs)
		val_imgs = np.array(val_imgs)
		val_masks = np.array(val_masks)

		return n_classes, train_imgs, train_segs, train_masks, val_imgs, val_segs, val_masks
	else:
		raise Exception("dataset not defined")


