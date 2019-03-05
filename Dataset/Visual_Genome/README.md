## Visual Genome dataset

* preparations
  * images preparation
	https://visualgenome.org/api/v0/api_home.html
	download VG_100K(part 1) and VG_100K_2(part 2) here 
	download image meta data into ori_data
  * annotations preparation
	[VTransE](https://github.com/zawlin/cvpr17_vtranse)
	download cleaned VG dataset annotations here


* content

  * **train_imgs.npy**: image paths of imgs used in train set
								pickle dump
								e.g. "VG_100K_2/1.jpg"

  * **test_imgs.npy**: image paths of imgs used in test set
							 	save as above

  * **train_imgs_img_id2coco_id.txt**: dict of {img_id->coco_id} in train images. Only contain images having coco id

  * **test_imgs_img_id2coco_id.txt**: dict of {img_id->coco_id} in test images. Only contain images having coco id

  * **vg1_2_meta.h5**: downloaded annotations from vtranse website

  * **ori_data**

  * **img_id2coco_id.txt**: img id -> coco id for image_data.json. strangely some images have the same coco id. 
									if that happens, we make the second image corresponds to None
  * **complete_info_list.json**: same format as image_data.json download from VG website, but filtered to those ones in COCO and VTransE at the same time