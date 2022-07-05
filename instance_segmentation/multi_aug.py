
import numpy as np
import cv2
import albumentations as A
import yaml
import time
import os
import random
import copy
import sys
from multiprocessing import Process,Queue
from threading import Thread
from datetime import datetime

sys.path.append(os.getcwd())
from instance_segmentation.augmentation import *
from instance_segmentation.utils import *

# annotation_id_queue = Queue()
# image_id_queue = Queue()
def load_log():
    with open("instance_segmentation/instance_seg_log.yaml","r") as f:
        return yaml.load(f,yaml.FullLoader)

def make_albumentation_pipeline(log):
    pipeline=[]
    pipeline.append(
        A.OneOf(
                [
                    A.GlassBlur(p=1,max_delta=round(log["blur_strength"]/6)),
                    A.MedianBlur(p=1,blur_limit=round(log["blur_strength"]*3)),
                    A.MotionBlur(p=1,blur_limit=round(log["blur_strength"]*3)),
                    A.GaussianBlur(p=1,blur_limit=(3,round(log["blur_strength"]*3)))
                ],
                p=log["blur_probability"]/100
                ),
    )

    pipeline.append(
        A.OneOf(
                [
                    A.GaussNoise(p=1,var_limit=(100,log["noise_strength"]*30),mean=10,per_channel=False),
                    A.GaussNoise(p=1,var_limit=(100,log["noise_strength"]*30),mean=10,per_channel=True)
                ],
                p=log["noise_probability"]
                ),
    )

    pipeline.append(
        A.RandomBrightnessContrast(p=log["brightness_probability"]/100,brightness_limit=log["brightness_strength"]/50)
    )

    pipeline.append(
        A.HorizontalFlip(p=log["horizentalflip_probability"]/100)
    )

    pipeline.append(
        A.HorizontalFlip(p=log["verticalflip_probability"]/100)
    )
    
    pipeline.append(
        A.Downscale(p=log["downscale_probability"]/100,scale_max=0.999,scale_min=1-log["downscale_strength"]/50),
    )
    
    pipeline.append(
        A.OneOf(
                    [
                        A.ColorJitter(p=1,),
                        A.ISONoise(p=1,),
                        A.FancyPCA(p=1,),
                        A.HueSaturationValue(p=1,
                                                hue_shift_limit=(-int(log["colorjitter_strength"])*0.5,int(log["colorjitter_strength"])*0.5),
                                                sat_shift_limit=(-int(log["colorjitter_strength"])*0.6,int(log["colorjitter_strength"])*0.6),
                                                val_shift_limit=(-int(log["colorjitter_strength"])*0.5,int(log["colorjitter_strength"])*0.5))
                    ],
                    p=log["colorjitter_probability"]/100
                ),
    )

    pipeline.append(
        A.OpticalDistortion(
                p=log["opticaldistortion_probability"]/100,
                distort_limit=log["opticaldistortion_strength"]/40,
                shift_limit=log["opticaldistortion_strength"]/80,
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            ),
    )
    
    pipeline.append(
        A.Compose(
                    [
                        A.Affine(p=1,),
                        A.RandomResizedCrop(p=1,height=1280,width=1280,scale=(1-log["affine_strength"]/100,1))
                    ],
                    p=log["affine_probability"]/100,
                )
    )
    
    pipeline.append(
        A.Perspective(p=1,scale=(0.05,0.05+log["perspective_strength"]/100))
    )
    
    pipelines = []
    for i in range(20):
        random.shuffle(pipeline)
        temp = copy.deepcopy(A.Compose(pipeline))
        pipelines.append(temp)
    return pipelines
        
        
def make_custom_aug_pipeline(log):
    custom_pipeline=[]
    custom_pipeline.append(
        [
            random_part_resize,
            {
                "p":log["resize_probability"]/100
            }
        ]
    )

    custom_pipeline.append(
        [
            random_dropout,
            {
                "p": log["dropout_probability"]/100,
                "min_instance":0,
                "max_instance":log["dropout_strength"],
            }
        ]
    )
    
    # custom_pipeline.append(
    #     [
    #         random_background_change,
    #         {
    #             "p":0,
    #             "patch_image":None
    #         }
    #     ]
    # )
    custom_pipeline.append(
        [
            random_grid_distortion,
            {
                "p":log["griddistortion_probability"]/100,
                "distort_intensity":log["griddistortion_strength"]/100,
            }
        ]
    )
    
    custom_pipeline.append(
        [
            random_gradation,
            {
                "p":log["gradation_probability"]/100,
                "gradation_intensity":log["gradation_strength"]/100
                
            }
        ]
    )
    return custom_pipeline

def execute_one_aug(image,masks,pre_categories_about_one_mask,custom_pipeline,pipelines):
    random.shuffle(custom_pipeline)
    random.shuffle(pipelines)
    temp = [-1]+list(range(len(custom_pipeline)))
    random.shuffle(temp)
    for index in temp:
        if index==-1:
            transformed = pipelines[0](image=image,masks=masks)
            image = transformed['image']
            masks = transformed['masks']
        else:
            
            image,masks,pre_categories_about_one_mask = custom_pipeline[index][0](image,masks,pre_categories_about_one_mask,**custom_pipeline[index][1])
    return image,masks,pre_categories_about_one_mask

def temp_run(process_index,annotation_id_queue,image_id_queue,output_queue):
    log = load_log()
    label_file_path = log["label_file_path"]
    images_folder = log["images_folder"]
    save_folder = log["save_folder"]
    folder_count=1
    custom_pipeline = make_custom_aug_pipeline(log)
    pipelines = make_albumentation_pipeline(log)
    coco_data,categories = read_json(label_file_path,images_folder)
    coco_data,categories = read_json(label_file_path,images_folder)
    today = str(datetime.now())
    today = today.split(" ")[0]
    while True:
        if not os.path.isdir(save_folder+"/{}-{}".format(today,folder_count)):
            break
        else:
            folder_count+=1
    full_save_path = save_folder+"/"+today+"-"+str(folder_count-1)
    
    images=[]
    annotations=[]
    print("시작한다!!!!!!!!!!!!")
    for index,item in enumerate(coco_data):
        pre_image_origin,pre_masks_origin,pre_categories_about_one_mask = item
        for j in range(int(int(log["aug_nums"])//int(log["nums_process"]))+1):
        # for j in range(1):
            number = image_id_queue.get()
            t = "_".join(str(time.time()).split("."))
            result_image = copy.deepcopy(pre_image_origin)
            result_masks = copy.deepcopy(pre_masks_origin)
            result_categories_about_one_mask = copy.deepcopy(pre_categories_about_one_mask)
            result_image,result_masks,pre_categories_about_one_mask = execute_one_aug(result_image,result_masks,result_categories_about_one_mask,custom_pipeline,pipelines)
            result_polygons,result_categories_about_one_mask,result_image = masks2polygons(result_masks,pre_categories_about_one_mask,result_image=result_image,dummy_image_list=None)
            for polygon,one_category_num in zip(result_polygons,result_categories_about_one_mask):
                annotation_number = annotation_id_queue.get()
                x,y,w,h = cv2.boundingRect(polygon)
                area = int(cv2.contourArea(polygon))
                seg = [ int(i) for i in polygon.reshape(-1)]
                # seg = list(polygon.reshape(-1))
                annotations.append({
                    "iscrowd":0,
                    "image_id":number,
                    "bbox":[[x,y,w,h]],
                    "segmentation":[seg],
                    "category_id":one_category_num,
                    "id":annotation_number,
                    "area":area,
                })
                annotation_number+=1
                del x,y,w,h,area,seg
            images.append({
                    "height":result_image.shape[0],
                    "width":result_image.shape[1],
                    "id":number,
                    "file_name":str(process_index)+"_"+t+".jpg",
                })
            cv2.drawContours(result_image,result_polygons,-1,(255,0,0),thickness=2,)
            print(full_save_path+"/{}{}.jpg".format(str(process_index)+"_",t))
            print(result_image.shape)
            cv2.imwrite(full_save_path+"/{}{}.jpg".format(str(process_index)+"_",t),result_image)
            del result_image,result_masks, result_polygons,result_categories_about_one_mask
    result_object = {
        "images":images,
        "annotations":annotations,
        "categories":categories,
        }
    output_queue.put(result_object)

def main():
    log = load_log()
    process_num = int(log["nums_process"])
    save_folder = log["save_folder"]
    label_file_path = log["label_file_path"]
    images_folder = log["images_folder"]
    save_folder = log["save_folder"]
    today = str(datetime.now())
    today = today.split(" ")[0]
    output_queue = Queue()
    annotation_id_queue = Queue()
    image_id_queue = Queue()
    for i in range(10**7):
        annotation_id_queue.put(i)
        image_id_queue.put(i)
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    if label_file_path:
        if images_folder:
            coco_data,categories = read_json(label_file_path,images_folder)
        else:
            raise Exception("이미지폴더 경로가 유효하지않음")
    else:
        raise Exception("라벨파일의 경로가 유효하지않음")
    folder_count=1
    while True:
        if not os.path.isdir(save_folder+"/{}-{}".format(today,folder_count)):
            os.mkdir(save_folder+"/{}-{}".format(today,folder_count))
            full_save_path = save_folder+"/{}-{}".format(today,folder_count)
            break
        else:
            folder_count+=1
    
    p_list =[]    
    for index in range(process_num):
        p = Process(target=temp_run,args=(index,annotation_id_queue,image_id_queue,output_queue))
        p.start()
        p_list.append(p)

    images = []
    annotations = []
    process_num = process_num
    while_count=0
    while True:
        if process_num<=while_count:
            break
        result_object = output_queue.get()
        images+=result_object["images"]
        annotations+=result_object["annotations"]
        while_count+=1
    result_object = {
        "images":images,
        "annotations":annotations,
        "categories":categories,
        }
    with open('{}/data.json'.format(full_save_path), 'w') as f:
        json_string = json.dump(result_object, f, indent=2)
    
    
    
    
if __name__=="__main__":
    main()