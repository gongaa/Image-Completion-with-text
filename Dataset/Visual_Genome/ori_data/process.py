'''
Author: Lin.Z
'''

'''

	todo
		find coco id for our split dataset 
'''


import json
import pickle
import sys
# import ijson

'''
	synsets are dicts

'''

'''
	object synsets
'''
# with open("object_synsets.json") as f:
# 	obj_syns = json.load(f)
# print(len(obj_syns))

# obj_synsets = obj_syns.values()
# obj_synsets_set = set(obj_synsets)
# print(len(obj_synsets_set))

# obj_rev_synsets = dict()
# for i,j in obj_syns.items():
# 	if j not in obj_rev_synsets.keys():
# 		obj_rev_synsets[j] = set()
# 	obj_rev_synsets[j].add(i)
# for o,(i,j) in enumerate(obj_rev_synsets.items()):
# 	if o < 50:
# 		print(i,"\t",j)



# for o,(i,j) in enumerate(obj_syns.items()):
# 	if o < 10:
# 		print(i, "\t", j)

'''
	attribute synsets
'''
# with open("attribute_synsets.json") as f:
# 	attr_syns = json.load(f)
# print(len(attr_syns))
# for o,(i,j) in enumerate(attr_syns.items()):
# 	if o < 10:
# 		print(i, "\t", j)

'''
	relationship synsets
'''
# with open("relationship_synsets.json") as f:
# 	rela_syns = json.load(f)
# print(len(rela_syns))
# for o,(i,j) in enumerate(rela_syns.items()):
# 	if o < 10:
# 		print(i, "\t", j)

'''
	objects
'''
# obj_parse = ijson.parse(open("objects.json"))
# print(type(obj_parse))
# obj_dt = list(obj_parse)

# for o, obj_dt in enumerate(ijson.items("objects.json")):
# 	if o < 10:
# 		print(obj_dt)


'''
	image_data
'''

def img_id2coco_id(image_data, file):
	mydict = {}
	o = 0
	length = len(image_data)
	for i in image_data:
		assert not mydict.__contains__(i['image_id'])
		mydict[i['image_id']] = i['coco_id']
		if i['coco_id'] is not None and i['coco_id'] in mydict.values():
			mydict[i['image_id']] = None		
		sys.stdout.write("\r {} / {}".format(o+1, length))
		o+=1
	with open(file, 'wb') as f:
		pickle.dump(mydict, f)

with open("image_data.json") as f:
    image_data = json.load(f)
img_id2coco_id(image_data, "img_id2coco_id.txt")
# print(type(image_data[0]['image_id']))




