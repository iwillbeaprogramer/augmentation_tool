import json
import cv2
import os
import numpy as np
import albumentations as A
import shutil
import random
import time
import collections
import copy

def polygons2masks(shape,polygons):
    """
    polygons [array,array,array,array,...,array]
    array = (n,2)
    """
    H,W = shape
    result = []
    for polygon in polygons:
        temp = np.zeros((H,W)).astype(np.uint8)
        cv2.fillPoly(temp,[polygon.astype(np.int64)],255)
        result.append(temp)
    return result

def masks2areas(masks):
    areas=[]
    for mask in masks:
        areas.append(int(np.sum(mask/255.)))
    return areas

def masks2polygons(masks,categories_about_one_mask,result_image,border_cut=True,median_value=0,dummy_image_list=None):
    if len(masks[0].shape)==2:
        H,W=masks[0].shape
    else:
        H,W,_=masks[0].shape
    polygons = []
    filtered_categories_num=[]
    for mask,category_num in zip(masks,categories_about_one_mask):
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if len(contours)>=1:
            temp = contours[-1].reshape(-1,2)
            if np.all(temp[:,0]>=2) and np.all(temp[:,0]<=mask.shape[1]-2) and np.all(temp[:,1]>=2) and np.all(temp[:,1]<=mask.shape[0]-2):
                polygons.append(temp)
                filtered_categories_num.append(category_num)
            else:
                result_image[mask==255] = np.array([random.randint(0,255),random.randint(0,255),random.randint(0,255)]).astype(np.uint8)
    return polygons,filtered_categories_num,result_image

def read_json(path,image_path_prefix=""):
    result = []
    with open(path) as f:
        json_object = json.load(f)
    # for key in json_object:
    #     print(type(json_object[key]))
    images = json_object['images']
    annotations = json_object['annotations']
    categories = json_object['categories']
    for item in images:
        filename = item["file_name"]
        full_path = os.path.join(image_path_prefix,filename)
        image = cv2.imread(full_path)
        target_segmentations = [ [np.array(one_annotation['segmentation']).reshape(-1,2),one_annotation['category_id']] for one_annotation in annotations if one_annotation['image_id']==item['id']]
        mask_categories = [ item[1] for item in target_segmentations]
        target_segmentations = [ item[0] for item in target_segmentations]
        target_segmentations_masks = polygons2masks((item['height'],item['width']),target_segmentations)
        result.append([image,target_segmentations_masks,mask_categories])
    return result,categories