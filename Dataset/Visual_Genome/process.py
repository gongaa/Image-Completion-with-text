'''
Author:	Lin.Z
Date:	26/2/2019

vg1_2_meta.h5
	-gt
		-train
		-test(25858)
			-index (not consecutive)
				-obj_boxes
					-n*4 array (x1, y1, x2, y2) ? 
				-rlp_labels
					-n*3 array (s_index, p_index, o_index)
				-sub_boxes
					-n*4 array (x1, y1, x2, y2)
	-meta
		-cls
			-idx2name(201, from 0)
			-name2idx(201, from 0)
		-imid2path(108077 image path)
			-index(from 1)
				-img path, e.g.("VG_100K_2/1.jpg")
		-pre
			-idx2name(100, from 0)

'''
import h5py
import pickle
import json
import numpy as np
import sys

def find_img_indices(vg_file, mode="train"):
	imid2path_dict = vg_file['meta']['imid2path']
	if mode=="train":
		img_dict = vg_file['gt']['train']
		store_file = "train_imgs.npy"
	else:
		img_dict = vg_file['gt']['test']
		store_file = 'test_imgs.npy'
	img_list = []
	length = len(img_dict.keys())
	for o, i in enumerate(img_dict.keys()):
		# if o > 3:
		# 	break
		# print(str(np.array(imid2path_dict[i])))
		img_list.append(str(np.array(imid2path_dict[i])))
		text = " \r {} / {}".format(o+1, length)
		sys.stdout.write(text)
	with open(store_file,'wb') as f:
		pickle.dump(img_list, f)


def find_img_coco_id(vg_file, img_id2coco_id, mode='train'):
	'''
		train: 35062 / 73794
		test : 12385 / 25858
	'''
	if mode=="train":
		img_dict = vg_file['gt']['train']
		store_file = "train_imgs_img_id2coco_id.txt"
	else:
		img_dict = vg_file['gt']['test']
		store_file = 'test_imgs_img_id2coco_id.txt'
	id2coco_id_dict = {}
	length = len(img_dict.keys())
	for o, i in enumerate(img_dict.keys()):
		if img_id2coco_id[int(i)] is not None:
			id2coco_id_dict[i] = img_id2coco_id[int(i)]
			# assert img_id2coco_id[int(i)] not in coco_id_list, [ i, int(i), img_id2coco_id[int(i)] ]
		sys.stdout.write("\r {} / {}".format(o+1, length))
	with open(store_file,'wb') as f:
		pickle.dump(id2coco_id_dict, f)

vg_file = h5py.File("vg1_2_meta.h5", 'r')
# with open("ori_data/image_data.json") as f:
#     image_data = json.load(f)
# for i in vg_file['gt']['test'].keys():
# 	print(i)
with open("ori_data/img_id2coco_id.txt", 'rb') as f:
	img_id2coco_id = pickle.load(f)
	find_img_coco_id(vg_file, img_id2coco_id, mode='train')
	find_img_coco_id(vg_file, img_id2coco_id, mode='test')

# print(np.array(vg_file['meta']))
# for i,j in vg_file['meta']['pre'].items():
# 	print(i)
# 	print(j)
# 	break
# find_img_indices(vg_file, 'test')

vg_file.close()




