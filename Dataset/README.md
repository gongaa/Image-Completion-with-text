Author: Lin.Z

Dataset:
1. MS COCO(may discarded)
2. VRD
	clean dataset
	5000 	images, all annotated with sub/obj bboxes and standard (s-p-o) phrases
	100 	object categories
	70 		redicates
	37,993 	relation annotations
	6,672 	unique relations 
	24.25 	predicates per object category
3. VG
	cleaned version
	99,658	images
	200		object categories
	100		predicates
	1174692 relation annotations
	19,237 	unique relations
	57		predicates per object category

problems now:
	if we want to generate layout first, then we need to do scene/object segmentation to every image

todo:
	1. find state of art(at least reliable) scene and also object segmentation tools
		segment on our dataset(VG/VRD)
		after this the ms coco dataset can be dropped, and dataset is good enough
	2. find joint objects/predicates in VG/VRD
	3. find bounding box with joint objects/predicates and crop them from images

	following 3 steps above, if the result is good enough, then we have our dataset.


todo new(27/2):
	1. we have already found the imgs in VG that have coco id
	2. check ms coco's segmentation classes(things + stuffs)
	3. find images in ms coco that have the correspoinding id
	4. retrieve those images' segmentation
	5. combine segmentation with vg dataset to form our 47,447 imgs dataset
		1> link s,o wit segmentation
		2> link environment with segmentation
	6. build our memory bank (s-p-o)->(s-seg, o_seg)
	



