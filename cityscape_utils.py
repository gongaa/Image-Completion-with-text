seg_index2label={
	0:'unlabeled'            ,
    # 'ego vehicle'          (  0,  0,  0) ,
    # 'rectification border' (  0,  0,  0) ,
    # 'out of roi'           (  0,  0,  0) ,
    # 'static'               (  0,  0,  0) ,
    1:'dynamic'              ,
    2:'ground'                ,
    3:'road'                  ,
    4:'sidewalk'              ,
    5:'parking'             ,
    6:'rail track'            ,
    7:'building'              ,
    8:'wall'                 ,
    9:'fence'                ,
    10:'guard rail'           ,
    11:'bridge'               ,
    12:'tunnel'                ,

    13:'pole'                 ,	#
    # 'polegroup'            (153,153,153) ,
    14:'traffic light'         ,	#
    15:'traffic sign'         ,	#
    16:'vegetation'           ,	#

    17:'terrain'              ,
    18:'sky'                   ,

    19:'person'                ,	#
    20:'rider'                ,	#
    21:'car'                   ,	#
    22:'truck'                 ,	#
    23:'bus'                   ,	#
    24:'caravan'              ,	#
    25:'trailer'               ,	#
    26:'train'                 ,	#
    27:'motorcycle'            ,	#
    28:'bicycle'               
}


# 30 colors -> 29 classes
seg_color2index = {
	(  0,  0,  0 ):0,
	(  0,  0,142 ):0,
    (111, 74,  0):1,
    ( 81,  0, 81):2,
    (128, 64,128):3, 			#'road'
    (244, 35,232):4, 			#'sidewalk'            
    (250,170,160):5,			#'parking'              
    (230,150,140):6,			#rail track'            
    ( 70, 70, 70):7,			#building'              
    (102,102,156):8,			#wall'                  
    (190,153,153):9,			#fence'                 
    (180,165,180):10,			#guard rail'            
    (150,100,100):11,			#bridge'                
    (150,120, 90):12,			#tunnel'                

    (153,153,153):13,			#pole'                  
    (250,170, 30):14,			#traffic light'         
    (220,220,  0):15,			#traffic sign'          
    (107,142, 35):16,			#vegetation'            

    (152,251,152):17,			#terrain'              
    ( 70,130,180):18,			#sky'                   

    (220, 20, 60):19,			#person'                
    (255,  0,  0):20,			#rider'               
    (  0,  0,142):21,			#car'                   
    (  0,  0, 70):22,			#truck'                
    (  0, 60,100):23,			#bus'                   
    (  0,  0, 90):24,			#caravan'               
    (  0,  0,110):25,			#trailer'               
    (  0, 80,100):26,			#train'                
    (  0,  0,230):27,			#motorcycle'           
    (119, 11, 32):28			#bicycle'             	
}