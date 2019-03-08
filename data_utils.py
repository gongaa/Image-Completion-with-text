import numpy as np
import os
import sys
import pickle
import json
import time
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
	### explicitly set flip = True #######
	if dataset == "cityscape":
		dataset_train_dir = "Dataset/Cityscape/train_data"
		dataset_val_dir = "Dataset/Cityscape/val_data"

		seg_dir = "Dataset/Cityscape/gtFine"
		n_classes = 29

		if os.path.exists(dataset_train_dir+"_segs.npy"): # if stored file exist
			sys.stdout.write("loading cityscape train dataset from {} ".format(dataset_train_dir))
			train_segs = np.load(dataset_train_dir+"_segs.npy")
			train_imgs = np.load(dataset_train_dir+"_imgs.npy")
			train_masks = np.load(dataset_train_dir+"_masks.npy")
			print("loaded")
			train_size = train_segs.shape[0]
		else:  # else , loading one time, which cost 1 hour
			train_seg_dir = os.path.join(seg_dir, 'train')
			train_seg_files = glob.glob( train_seg_dir+ '/**/*color.png', recursive=True)
			train_segs = []
			train_imgs = []
			train_masks = []
			train_size = len(train_seg_files)

			for o,train_seg_file in enumerate(train_seg_files):
				# if o > 1:
				# 	break
				train_img_file = train_seg_file.replace("gtFine_color", "leftImg8bit")
				train_img_file = train_img_file.replace("gtFine", "leftImg8bit")
				train_mask_file = train_seg_file.replace("color.png", "polygons_mask.json")

				# tic = time.time()
				seg = imread(train_seg_file, mode='RGB')
				img = imread(train_img_file, mode="RGB")
				# toc = time.time() 
				# img_load_time = toc - tic
				with open(train_mask_file, 'r') as f:
					file = json.load(f)
					mask = file['mask']
				# mask_load_time = time.time()-toc
				train_segs.append(seg)
				train_imgs.append(img)
				train_masks.append(np.array(mask))
				sys.stdout.write("\rloading cityscape train dataset <<<<<<< {}/{} >>>>>> ".format(o+1, train_size))
				sys.stdout.flush()
			print()
			train_segs = np.array(train_segs)
			train_imgs = np.array(train_imgs)
			train_masks = np.array(train_masks)

			print("saving train data into {}".format(dataset_train_dir))	
			np.save(dataset_train_dir+"_segs",train_segs)
			np.save(dataset_train_dir+"_imgs",train_imgs)
			np.save(dataset_train_dir+"_masks",train_masks)
			

		if flip:
			train_append_imgs = []
			train_append_segs = []
			train_append_masks = []
			for i in range(train_size):
				train_flip_seg  = np.flip(train_segs[i],1)
				train_flip_mask = np.flip(train_masks[i],1)
				train_flip_img  = np.flip(train_imgs[i],1)
				train_append_segs.append(train_flip_seg)
				train_append_img.append(train_flip_img)
				train_append_masks.append(train_flip_mask)
				sys.stdout.write("\rflipping cityscape train dataset <<<<<<< {}/{} >>>>>>".format(i+1, train_size))
				sys.stdout.flush()
			print()
			train_append_imgs = np.array(train_append_imgs)
			train_append_segs = np.array(train_append_segs)
			train_append_masks = np.array(train_append_masks)

			train_segs = np.concatenate([train_segs, train_append_segs], axis=0)
			train_imgs = np.concatenate([train_imgs, train_append_imgs], axis=0)
			train_masks = np.concatenate([train_masks, train_append_masks], axis=0)

		np.random.shuffle(train_segs)
		np.random.shuffle(train_imgs)
		np.random.shuffle(train_masks)

		if not val:
			return n_classes, train_imgs, train_segs, train_masks, None, None ,None

		if os.path.exists(dataset_val_dir+"_segs.npy"): # if stored file exist
			sys.stdout.write("loading cityscape flipped val dataset from {} ".format(dataset_val_dir))
			val_segs = np.load(dataset_val_dir+"_segs.npy")
			val_imgs = np.load(dataset_val_dir+"_imgs.npy")
			val_masks = np.load(dataset_val_dir+"_masks.npy")
			print("loaded")
			val_size = val_segs.shape[0]
		else: # else, loading val data by hand, which takes about 10 mins
			val_seg_dir = os.path.join(seg_dir, 'val')
			val_seg_files = glob.glob( val_seg_dir+ '/**/*color.png', recursive=True)
			val_segs = []
			val_imgs = []
			val_masks = []
			val_size = len(val_seg_files)
			for o,val_seg_file in enumerate(val_seg_files):
				# if o > 1:
				# 	break
				val_img_file = val_seg_file.replace("gtFine_color", "leftImg8bit")
				val_img_file = val_img_file.replace("gtFine", "leftImg8bit")
				val_mask_file = val_seg_file.replace("color.png", "polygons_mask.json")

				seg = imread(val_seg_file, mode='RGB')
				img = imread(val_img_file, mode="RGB")
				with open(val_mask_file, 'r') as f:
					file = json.load(f)
					mask = file['mask']
				val_segs.append(seg)
				val_imgs.append(img)
				val_masks.append(np.array(mask))
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

		return n_classes, train_imgs, train_segs, train_masks, val_imgs, val_segs, val_masks
	else:
		raise Exception("dataset not defined")


