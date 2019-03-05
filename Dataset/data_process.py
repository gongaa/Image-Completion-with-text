#import h5py
import pickle
import json
import numpy as np
import numpy.random as nr
import sys
import os
import collections

#import matplotlib.pyplot as plt
#import matplotlib.patches as patches
#import matplotlib.image as mpimg
#from PIL import Image
import glob


def print_json(data, indent=0):
    if not isinstance(data, dict) and not isinstance(data, list):
        return
    if isinstance(data, list):
    	print_json(data[0], indent+1)
    	return
    for i in data.keys():
        print('\t'*indent, i)
        print_json(data[i], indent+1)
    return

def print_h5(data, indent=0):
    if not isinstance(data, h5py._hl.group.Group) and not isinstance(data, h5py._hl.files.File):
        return
    if len(data.keys())>20:
    	for key, item in data.items():
    		print_h5(item, indent+1)
    		return
    	
    for i in data.keys():
        print('\t'*indent, i)
        print_h5(data[i], indent+1)
    return

def check_repe_relations(vg):
	'''
		check relation repetition for original vg h5 file
	'''
	for mode in ['train', 'test']:
		print(mode)
		for o,(kk,i) in enumerate(vg['gt'][mode].items()):
			rels =len(i['obj_boxes'])
			values = []
			for k in range(i['obj_boxes'].value.shape[0]):
				# print(vg['meta']['imid2path'][kk].value,end="\t")
				value = "{} {} {}\t{}\t{}".format(vg['meta']['cls']['idx2name'][str(i['rlp_labels'].value[k,0])].value,\
					vg['meta']['pre']['idx2name'][str(i['rlp_labels'].value[k,1])].value,\
					vg['meta']['cls']['idx2name'][str(i['rlp_labels'].value[k,2])].value,\
					str(i['sub_boxes'].value[k].tolist()),\
					str(i['obj_boxes'].value[k].tolist()))
				assert value not in values, value
				values.append(value)
			sys.stdout.write("\r {} / {}".format(o+1, len(vg['gt'][mode].items())))
		print()

def coco_instance_category_id2name(coco_instances, category_id):
	for cat in coco_instances['categories']:
		if cat['id'] == category_id:
			return cat['supercategory'] + '/' + cat['name']
	raise Exception("Invalid coco category id!", category_id)

def coco_instance_id2bboxes_phrases(coco_instances, coco_id):
	bboxes_phrases = []
	for coco_ann in coco_instances['annotations']:
		if coco_ann['image_id'] == coco_id:
			obj_name = coco_instance_category_id2name(coco_instances, coco_ann['category_id'])
			phrase = (obj_name, coco_ann['bbox'])
			bboxes_phrases.append(phrase)
	if len(bboxes_phrases) == 0:
		raise Exception("coco instances can not found by this id!", coco_id)
	return bboxes_phrases

def vg_obj_id2name(vg, obj_id):
	result = vg['meta/cls/idx2name'][str(obj_id)].value
	assert isinstance(result, str)
	return result

def vg_pre_id2name(vg, pre_id):
	result = vg['meta/pre/idx2name'][str(pre_id)].value
	assert isinstance(result, str)
	return result

def construct_coco_id2size(coco_instances):
	coco_id2size_dict = {}
	o = 0
	for img in coco_instances['images']:
		assert not coco_id2size_dict.__contains__(img['id']), img['id']
		coco_id2size_dict[img['id']] = (img['height'], img['width'])
		o+=1
		sys.stdout.write("\r {} / {}".format(o, len(coco_instances['images'])))
	print()
	return coco_id2size_dict

def compare_bbox(vg, coco_instance, coco_stuff):
	'''
		for same name, make sure the bbox is the same
		record those bbox is the same but having different names
	'''
	pass

############################# functions for Cityscape dataset ######################
'''
	sizes:
		128, 256, 512

'''
def crop_box(ann, label, sizes = [128, 256, 512], img_h=1024, img_w=2048):
	'''
		all img size (1024, 2048) h,w
		locations_list
		size = (h, w)
		return a mask (h, w)
	'''
	objs = ann['objects']
	obj_lbls = [obj['label'] for obj in objs]
	obj_polygons = [np.array(obj['polygon']) for obj in objs]
	if label not in obj_lbls:
		return None
	selected_obj_indices = [t for t in range(len(obj_lbls)) if obj_lbls[t] == label]
	selected_obj_polygons = [obj_polygons[t] for t in selected_obj_indices]
	selected_obj_bbox = np.array([( np.amin(polyg[:, 0]), np.amin(polyg[:, 1]), np.amax(polyg[:,0]), np.amax(polyg[:, 1]) ) \
												for polyg in selected_obj_polygons ])
	size = sizes[nr.randint(0, len(sizes))]
	box = search_crop_box(selected_obj_bbox, size, img_h, img_w)
	mask = box2mask(img_h, img_w, box)
	return mask

def box2mask(img_h, img_w, box):
	if box is None:
		return None
	mask = np.ones((img_h, img_w))
	mask[box[1]:box[3]+1, box[0]:box[2]+1] = 0
	return mask

def check_box_sanity(box, img_h, img_w):
	'''
		box = (x1, y1, x2, y2)
		x in [100, img_w - 100 - 1]
		y in [100, img_h - 256 - 1]
	'''
	assert box[0] < box[2] and box[1] < box[3], box
	return( box[0] >=100 and box[2] < img_w-100 and \
			box[1] >= 100 and box[3] < img_h-256 )  

def search_crop_box(obj_boxes, size, img_h, img_w):
	X_MIN = 100
	X_MAX = img_w-100-1
	Y_MIN = 100
	Y_MAX = img_h-256-1

	for obj_box in obj_boxes:
		x2_max = min(obj_box[2] + size -1, X_MAX)
		x2_min = max(obj_box[0], X_MIN + size - 1)
		if x2_min > x2_max:
			continue
		x2 = nr.random_integers(x2_min, x2_max+1)

		y2_max = min(obj_box[3] + size -1, Y_MAX)
		y2_min = max(obj_box[1], Y_MIN + size - 1)
		if y2_min > y2_max:
			continue
		y2 = nr.random_integers(y2_min, y2_max)
		return np.array([x2-size+1, y2-size+1, x2, y2])
	return None

############################ do a summary of the image statistics #################
cat_dict = { \
	'car':0, 
	"truck":1,\
	"bus":2,\
	# originally "on rails", but has no tram
	"train":3,\
	"motorcycle":4,\
	"bicycle":5,\
	"caravan":6,\
	"trailer":7,\
	"person":8,\
	"rider":9,\
	"pole":10,\
	"polegroup":11,\
	"traffic sign":12,\
	"traffic light":13,\
	"vegetation":14,\
	"cargroup":15,\
	"bicyclegroup":16,\
	"persongroup":17,\
	"polegroup":18,\
	"motorcyclegroup":19,\
	"ridergroup":20,\
	"truckgroup":21 \
}

train_files = glob.glob("Cityscape/gtFine/train" + '/**/*gons.json', recursive=True)
train_file = train_files[0]
val_files = glob.glob("Cityscape/gtFine/val" + '/**/*gons.json', recursive=True)
# val_file = val_files[0]

cat_list = [\
	"truckgroup",
	"motorcyclegroup",
	"ridergroup",
	"caravan" ,
	"trailer",
	"train",
	"polegroup"	,			
	"bus",
	"truck",
	"bicyclegroup",
	"persongroup",
	"motorcycle",
	"rider",
	"cargroup",
	"bicycle",
	"traffic light",
	"person",
	"vegetation",
	"traffic sign",
	"car",
	"pole"
]
cat_cnt_dict = {}
for cat in cat_list:
	cat_cnt_dict[cat] = 0

# nr.shuffle(train_files)
# count = 0
# for train_file in train_files:
# 	with open(train_file, 'r') as f:
# 		ann = json.load(f)
# 	mask=None
# 	for label in cat_list:
# 		if cat_cnt_dict[label] < 185:
# 			mask = crop_box(ann, label)
# 			if mask is not None:
# 				cat_cnt_dict[label]+=1
# 				break
# 	# for overflow cat cases
# 	if mask is None:
# 		rand_cat_indices = np.arange(len(cat_list))
# 		nr.shuffle(rand_cat_indices)
# 		rand_cat_list = cat_list[rand_cat_indices]
# 		for label in rand_cat_list:
# 			mask = crop_box(ann, label)
# 			if mask is not None:
# 				cat_cnt_dict[label]+=1
# 				break
# 	ann['mask'] = mask.tolist()
# 	train_file = train_file.replace('.json', '_mask.json')
# 	with open(train_file, 'w') as f:
# 		json.dump(ann, f)
# 	count+=1
# 	sys.stdout.write("\r {} / {}".format(count, len(train_files)))
# print()
# for key, value in cat_cnt_dict.items():
# 	print(key, value)

nr.shuffle(val_files)
count = 0
for val_file in val_files:
	with open(val_file, 'r') as f:
		ann = json.load(f)
	mask=None
	for label in cat_list:
		if cat_cnt_dict[label] < 185:
			mask = crop_box(ann, label)
			if mask is not None:
				cat_cnt_dict[label]+=1
				break
	# for overflow cat cases
	if mask is None:
		rand_cat_indices = np.arange(len(cat_list))
		nr.shuffle(rand_cat_indices)
		rand_cat_list = cat_list[rand_cat_indices]
		for label in rand_cat_list:
			mask = crop_box(ann, label)
			if mask is not None:
				cat_cnt_dict[label]+=1
				break
	ann['mask'] = mask.tolist()
	val_file = val_file.replace('.json', '_mask.json')
	with open(val_file, 'w') as f:
		json.dump(ann, f)
	count+=1
	sys.stdout.write("\r {} / {}".format(count, len(val_files)))
print()
for key, value in cat_cnt_dict.items():
	print(key, value)




# nr.shuffle(val_files)


############################		 check h and w 			#######################
############################ all height 1024 all width 2048 #######################
# heights = set()
# widths = set()
# for train_file in train_files:
# 	with open(train_file, 'r') as f:
# 		ann = json.load(f)
# 		heights.add(ann['imgHeight'])
# 		widths.add(ann['imgWidth'])
# for val_file in val_files:
# 	with open(val_file, 'r') as f:
# 		ann = json.load(f)
# 		heights.add(ann['imgHeight'])
# 		widths.add(ann['imgWidth'])


# with open(val_file, 'r') as f:
# 	val_ann = json.load(f)
# 	print_json(val_ann)
# 	for obj in val_ann['objects']:
# 		print(val_file)
# 		print(val_ann['imgHeight'])
# 		print(val_ann['imgWidth'])
# 		print(obj['label'], obj['polygon'])
# 		break


################################# store cat list ####################
# cat2img_count_train = {}
# cat2img_count_val = {}
# cat2count_train = {}
# cat2count_val = {}
# for cat in list(cat_dict.keys()):
# 	cat2img_count_train[cat] = 0
# 	cat2img_count_val[cat] = 0
# 	cat2count_train[cat] = 0
# 	cat2count_val[cat] = 0
#
# for file in train_files:
# 	with open(file, 'r') as f:
# 		instance = json.load(f)
# 	cat_flag = {cat:0 for cat in list(cat_dict.keys())}
# 	for obj in instance['objects']:
# 		# if 'group' in obj['label'] and obj['label'] != "pole group":
# 		# 	# print(file, obj['label'])
# 		# 	if obj['label'] not in group_list:
# 		# 		group_list.append(obj['label'])
# 		if obj['label'] in cat_list:
# 			cat2count_train[obj['label']] +=1
# 			cat_flag[obj['label']] = 1
# 	for cat in list(cat_dict.keys()):
# 		cat2img_count_train[cat] += cat_flag[cat]
#
# for file in val_files:
# 	with open(file, 'r') as f:
# 		instance = json.load(f)
# 	cat_flag = {cat:0 for cat in list(cat_dict.keys())}
# 	for obj in instance['objects']:
# 		# if 'group' in obj['label'] and obj['label'] != "pole group":
# 		# 	# print(file, obj['label'])
# 		# 	if obj['label'] not in group_list:
# 		# 		group_list.append(obj['label'])
# 		if obj['label'] in cat_list:
# 			cat2count_val[obj['label']] +=1
# 			cat_flag[obj['label']] = 1
# 	for cat in list(cat_dict.keys()):
# 		cat2img_count_val[cat] += cat_flag[cat]
# print("cat_name", "\t", "cnt_t", "\t", "cnt_v", "\t", "cnt_s", "\t", "icnt_t", "\t", "icnt_v", "\t", "icnt_s")
# for cat in list(cat_dict.keys()):
# 	print(cat, "\t", cat2count_train[cat], "\t",cat2count_val[cat], "\t",cat2count_train[cat] + cat2count_val[cat], "\t",\
# 			cat2img_count_train[cat], "\t",cat2img_count_val[cat], "\t",cat2img_count_train[cat] + cat2img_count_val[cat])
# 		if 'bridge' == obj['label']:
# 			print(file)
# for i in group_list:
# 	print(i)
# person_count = 0
# car_count = 0
# 	elif obj['label'] == 'car':
# 		car_count+=1
# print("person", person_count)
# print("car", car_count)






# with open("MS_COCO/coco/annotations/instances_val2017.json", 'r') as f:
# 	coco_instances = json.load(f)


# with open("PIC/relation/relations_val.json", 'r') as f:
# 	coco_instances = json.load(f)
# with open("PIC/categories_list/label_categories.json", 'r') as f:
# 	coco_relations = json.load(f)
# for o,i in enumerate(coco_relations):
# 	print(i['id'], i['name'])

# print_json(coco_instances)
# for o,i in enumerate(coco_instances):
# 	if o < 10:
# 		for j in i['relations']:
# 			print(j['subject'],j['relation'], j['object'])


# with open("MS_COCO/coco/annotations/instances_train2017.json", 'r') as f:
# 	coco_train_instances = json.load(f)

# vg = h5py.File('Visual_Genome/vg_new.h5', 'r')

# vg = h5py.File("vg_resized.h5", 'r')



#################### construct coco_id2size #################
# with open('coco_id2size.pkl', 'wb') as f:
# 	train_part = construct_coco_id2size(coco_train_instances)
# 	val_part = construct_coco_id2size(coco_instances)
# 	train_part.update(val_part)
# 	pickle.dump(train_part, f)
	




#################### add imid2size ##################
# with open("vg_id2size.pkl", 'rb') as vg_f:
# 	vg_id2size = pickle.load(vg_f)
# 	# with open("coco_id2size.pkl", 'rb') as coco_f:
# 	# 	coco_id2size = pickle.load(coco_f)
# 	vg.copy('meta', vg_resized)
# 	o=0
# 	for vg_id in vg['gt'].keys():
# 		vg_resized.create_dataset('meta/imid2size/{}'.format(vg_id), data=vg_id2size[int(vg_id)])
# 		sys.stdout.write("\r {} / {}".format(o+1, len(vg['gt'].keys())))
# 		o+=1



############################# transform bbox ##################
# with open("coco_id2size.pkl", 'rb') as coco_f:
# 	coco_id2size = pickle.load(coco_f)
# 	print(len(coco_id2size))
# 	# vg_resized.create_group('gt')
# 	o=0
# 	for vg_id, ann in vg['gt'].items():
# 		prename = "gt/{}".format(vg_id)
# 		try:
# 			vg_resized.create_group(prename)
# 			vg.copy(prename+'/rlp_labels', vg_resized[prename])
# 		except:
# 			print(vg_resized[prename+'/rlp_labels'])
		
# 		coco_id = vg_resized['meta/imid2cocoid'][vg_id].value
# 		coco_size = coco_id2size[coco_id]
# 		vg_size = vg_resized['meta/imid2size'][vg_id].value
# 		prop = float(coco_size[1]) / vg_size[1]
# 		try:
# 			#### change sub boxes
# 			sub_boxes = ann['sub_boxes'].value
# 			new_sub_boxes = sub_boxes*prop
# 			vg_resized.create_dataset(prename+"/sub_boxes", data=new_sub_boxes)			

# 			#### change obj boxes
# 			obj_boxes = ann['obj_boxes'].value
# 			new_obj_boxes = obj_boxes*prop
# 			vg_resized.create_dataset(prename+"/obj_boxes", data=new_obj_boxes)
# 		except:
# 			pass

# 		o+=1
# 		sys.stdout.write("\r {} / {}".format(o, len(vg['gt'].items())))


# vg_resized.close()


# check coco instance bbox
# for i in range(5):
# 	print(coco_instances['annotations'][i]['bbox'])

# check vg bbox
# for img in vg['gt'].values():
# 	print(img['obj_boxes'].value) 
# 	break





########################################### check the same img ############################
# count = 0
# for o,(vg_id, vg_ann) in enumerate(vg['gt'].items()):
# 	try:
# 		coco_id = int(vg['meta/imid2cocoid'][vg_id].value)
# 		# get height and width
# 		for coco_img in coco_instances['images']:
# 			if coco_img['id'] == coco_id:
# 				coco_height = coco_img['height']
# 				coco_width = coco_img['width']
# 		coco_bboxes_phrases = coco_instance_id2bboxes_phrases(coco_instances, coco_id)
# 		print("coco done")
# 	except:
# 		continue
# 	count+=1
# 	if count < 5:
# 		continue
# 	if count > 10:
# 		break
# 	############################# construct vg boxes ####################
# 	vg_img_name = vg['meta/imid2path'][vg_id].value
# 	relations_num = len(vg_ann)
# 	vg_bboxes_phrases = []
# 	for i in range(relations_num):
# 		phrase = ( vg_obj_id2name(vg, vg_ann['rlp_labels'].value[i][0]), \
# 					vg_pre_id2name(vg, vg_ann['rlp_labels'].value[i][1]), \
# 					vg_obj_id2name(vg, vg_ann['rlp_labels'].value[i][2]),\
# 					vg_ann['sub_boxes'].value[i],\
# 					vg_ann['obj_boxes'].value[i] )
# 		vg_bboxes_phrases.append(phrase)
# 		print("{} / {}".format(i, relations_num))

	
# 	############################# print text information ##########################
# 	print(vg_img_name)
# 	for t in coco_bboxes_phrases:
# 		print(t)
# 	for t in vg_bboxes_phrases:
# 		print(t)

# 	print(coco_height, coco_width)
# 	assert coco_id2size[coco_id][0] == coco_height and coco_id2size[coco_id][1] == coco_width, [coco_height, coco_width, coco_id2size[coco_id]]
# 	img = Image.open('Visual_Genome/{}'.format(vg_img_name))
# 	img = img.resize(( coco_width, coco_height))
# 	im = np.array(img, dtype=np.uint8)

# 	########################### minor debugging ###############################
# 	# print("\r {} / {}  {}".format(o, len(vg['gt'].items()), count), end=" ")

# 	# if vg_img_name != 'VG_100K_2/2415993.jpg':
# 	# if abs(int("{:.0f}".format(coco_width*im.shape[0]/im.shape[1]))-coco_height) >1:
# 	# 	count+=1
# 	# assert abs(int("{:.0f}".format(coco_width*im.shape[0]/im.shape[1]))-coco_height) <=2, [coco_width/im.shape[1]*im.shape[0], coco_height, coco_width, im.shape, vg_img_name]
# 	# assert abs(int(coco_width/coco_height*100) - int(im.shape[1]/im.shape[0]*100)) <=1, [coco_height, coco_width, im.shape, vg_img_name]

# 	####################### draw bbox #####################

# 	# Create figure and axes
# 	fig,ax = plt.subplots(1)

# 	# Display the image
# 	ax.imshow(im)

# 	# Create a Rectangle patch
# 	for t in coco_bboxes_phrases:
# 		box = t[1]
# 		rect = patches.Rectangle((box[0],box[1]),box[2],box[3],linewidth=1,edgecolor='r',facecolor='none')
# 		# Add the patch to the Axes
# 		ax.add_patch(rect)
# 	for t in vg_bboxes_phrases:
# 		sub_box = t[3]
# 		obj_box = t[4]
# 		rect = patches.Rectangle((sub_box[0],sub_box[1]),sub_box[2]-sub_box[0],sub_box[3]-sub_box[1],linewidth=1,edgecolor='b',facecolor='none')
# 		# Add the patch to the Axes
# 		ax.add_patch(rect)
# 		rect = patches.Rectangle((obj_box[0],obj_box[1]),obj_box[2]-obj_box[0],obj_box[3]-obj_box[1],linewidth=1,edgecolor='g',facecolor='none')
# 		# Add the patch to the Axes
# 		ax.add_patch(rect)

# 	plt.show()

# vg.close()
