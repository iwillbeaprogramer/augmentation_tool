import numpy as np
import cv2
import os
import random
import copy
import albumentations as A
import gdown
import zipfile
def masks_sum(masks):
    if len(masks)>=2:
        masks[1] += masks[0]
        return masks_sum(masks[1:])
    elif len(masks)==0:
        return []
    else:
        return masks[0]

def mixupdata_download():
    google_path = 'https://drive.google.com/uc?id='
    file_id = '1_GIB9z5vH6sZasWkRNf0LcfPJvCjQWFa'
    output = '../mixupdata.zip'
    gdown.download(google_path+file_id,output,quiet=False,verify=False,)
    zip = zipfile.ZipFile("../mixupdata.zip").extractall("../")
    
def random_background_change(image,masks,labels,p=0.5,patch_image=None,):
    if random.random()>p:
        return image,masks,labels
    else:
        if patch_image is not None:
            temp = os.listdir(patch_image)
            temp = [ os.path.join(patch_image,filename) for filename in temp]
            patch_nums = random.randint(1,10)
            result_image = image.copy()
            result_masks = copy.deepcopy(masks)
            instances = masks_sum(result_masks)
            background = 255-instances
            random_indexes = np.random.choice(len(temp)-1,patch_nums)
            patch_images = [ cv2.imread(temp[random_index]) for random_index in random_indexes]
            xs,ys = np.where(background[:,:,]==255)
            position_indexs = np.random.choice(len(xs)-1,patch_nums)
            positions = [ (xs[i],ys[i]) for i in position_indexs]
            x_scale,y_scale = random.randint(3,5),random.randint(3,5)
            for index,(x,y) in enumerate(positions):
                x_start,y_start = x,y
                x_end,y_end = x+1,y+1
                while True:
                    a = random.randint(1,10)*x_scale
                    b = random.randint(1,10)*y_scale
                    if np.any(background[y_start:y_end+b,x_start:x_end+a]!=255):
                        result_image[y_start:y_end,x_start:x_end,:] = cv2.resize(patch_images[index],(x_end-x_start,y_end-y_start))
                        break
                    elif (x_end+a>=image.shape[0]-1) or (y_end+b>=image.shape[0]-1):
                        try:
                            result_image[y_start:y_end,x_start:x_end,:] = cv2.resize(patch_images[index],(x_end-x_start,y_end-y_start))
                            break
                        except:
                            break
                    elif random.random()<=0.005:
                        result_image[y_start:y_end,x_start:x_end,:] = cv2.resize(patch_images[index],(x_end-x_start,y_end-y_start))
                        break
                    else:
                        x_end+=a
                        y_end+=b
            return result_image,masks,labels
        else:
            patch_nums = random.randint(1,10)
            result_image = image.copy()
            result_masks = copy.deepcopy(masks)
            instances = masks_sum(result_masks)
            background = 255-instances
            random_indexes = np.random.choice(len(patch_image)-1,patch_nums)
            xs,ys = np.where(background[:,:,]==255)
            position_indexs = np.random.choice(len(xs)-1,patch_nums)
            positions = [ (xs[i],ys[i]) for i in position_indexs]
            x_scale,y_scale = random.randint(3,5),random.randint(3,5)
            for index,(x,y) in enumerate(positions):
                x_start,y_start = x,y
                x_end,y_end = x+1,y+1
                while True:
                    a = random.randint(1,10)*x_scale
                    b = random.randint(1,10)*y_scale
                    if np.any(background[y_start:y_end+b,x_start:x_end+a]!=255):
                        result_image[y_start:y_end,x_start:x_end,:] = (random.random(0,255),random.random(0,255),random.random(0,255))
                        break
                    elif (x_end+a>=image.shape[0]-1) or (y_end+b>=image.shape[0]-1):
                        try:
                            result_image[y_start:y_end,x_start:x_end,:] = (random.random(0,255),random.random(0,255),random.random(0,255))
                            break
                        except:
                            break
                    elif random.random()<=0.005:
                        result_image[y_start:y_end,x_start:x_end,:] = (random.random(0,255),random.random(0,255),random.random(0,255))
                        break
                    else:
                        x_end+=a
                        y_end+=b
            return result_image,masks,labels



def random_dropout(image,masks,labels,p=0.5,min_instance=1,max_instance=5,mixup_image=False):
    if random.random()>p:
        return image,masks,labels
    else:
        return_image = image.copy()
        dropout_nums = random.randint(min_instance,max_instance)
        masks_num = len(masks)
        if dropout_nums>masks_num:
            if masks_num<=1:
                return image,masks,labels
            else:
                dropout_nums = 1
        temp = np.random.choice(len(masks),dropout_nums)
        return_masks = [ mask for index,mask in enumerate(masks) if index not in temp]
        result_labels = [ label for index,label in enumerate(labels) if index not in temp]
        for index in temp:
            mask = masks[index]
            if mixup_image:
                random_indexes = np.random.choice(len(mixup_image)-1,1)
                temp_image = cv2.resize([ cv2.imread(mixup_image[random_index]) for random_index in random_indexes][0],(image.shape[1],image.shape[0]))
                return_image[mask==255] = temp_image[mask==255]
            else :
                if random.random()>1:
                    color= np.array([random.randint(0,103)+random.choice([0,152]),random.randint(0,103)+random.choice([0,152]),random.randint(0,103)+random.choice([0,152])]).astype(np.uint8)
                    return_image[mask==255] = color
                else:
                    y,x = np.where(mask==255)
                    if len(y)==0 or len(x)==0:
                        continue
                    x_min,x_max,y_min,y_max = np.min(x),np.max(x),np.min(y),np.max(y)
                    color= np.array([random.randint(0,93)+random.choice([0,162]),random.randint(0,93)+random.choice([0,162]),random.randint(0,93)+random.choice([0,162])]).astype(np.uint8)
                    return_image[y_min:y_max,x_min:x_max,:]=color
                    
        return return_image,return_masks,result_labels

def random_mixup(image,masks,labels,p=0.5,HISTOGRAM_IMAGE_LIST=None):
    if random.random()<p:
        image_ = image.copy()
        a = random.random()*0.25+0.75
        if isinstance(HISTOGRAM_IMAGE_LIST,list):
            pass
        else:
            return image,masks,labels
        if random.random()<0.5:
            random_indexes = np.random.choice(len(HISTOGRAM_IMAGE_LIST)-1,1)
            mixup_image = [ cv2.imread(HISTOGRAM_IMAGE_LIST[random_index]) for random_index in random_indexes][0]
            return cv2.addWeighted(image_,a,cv2.resize(mixup_image,(image_.shape[1],image_.shape[0])),1-a,0),masks,labels
        else:
            random_indexes = np.random.choice(len(HISTOGRAM_IMAGE_LIST)-1,4)
            images_list = [ cv2.imread(HISTOGRAM_IMAGE_LIST[random_index]) for random_index in random_indexes]
            h,w,_ = image_.shape
            h,w = round(h*(random.random()*0.4+0.3))+10,round(w*(random.random()*0.4+0.3))+10
            temp1 = cv2.resize(images_list[0],(w,h))
            temp2 = cv2.resize(images_list[1],(image_.shape[1]-w,h))
            temp3 = cv2.resize(images_list[2],(w,image_.shape[0]-h))
            temp4 = cv2.resize(images_list[3],(image_.shape[1]-w,image_.shape[0]-h))
            result1 = np.concatenate([temp1,temp2],axis=1)
            result2 = np.concatenate([temp3,temp4],axis=1)
            mixup_image = np.concatenate([result1,result2],axis=0)
        return cv2.addWeighted(image_,a,cv2.resize(mixup_image,(image_.shape[1],image_.shape[0])),1-a,0),masks,labels
    else:
        return image,masks,labels


def random_part_resize(image,masks,labels,p=0.5):
    if random.random()<p:
        if random.random()<0.5:
            H,W = image.shape[:2]
            part_num = random.randint(5,9)
            term = 1/part_num
            random_size = [random.random()*0.8+0.6 for i in range(part_num)] # part_num
            slice_points = [0]+[ round(H*(term*i+term*0.5*0.5 + random.random()*term*0.5)) for i in range(part_num-1)]+[H] #part_num+1
            part_list = []
            for index in range(len(slice_points)-1):
                temp = image[slice_points[index]:slice_points[index+1],:,:]
                part_list.append(cv2.resize(temp,(temp.shape[1],round(temp.shape[0]*random_size[index]))))
            result_masks_list = []
            for mask_index in range(len(masks)):
                mask = masks[mask_index]
                mask_list = []
                for index in range(len(slice_points)-1):
                    temp = mask[slice_points[index]:slice_points[index+1],:,]
                    mask_list.append(cv2.resize(temp,(temp.shape[1],round(temp.shape[0]*random_size[index]))))
                result_masks_list.append(np.concatenate(mask_list,axis=0).astype(np.uint8))
            result_image = np.concatenate(part_list,axis=0).astype(np.uint8)
            return result_image,result_masks_list,labels
        else:
            H,W = image.shape[:2]
            part_num = random.randint(5,9)
            term = 1/part_num
            random_size = [random.random()*0.8+0.6 for i in range(part_num)] # part_num
            slice_points = [0]+[ round(W*(term*i+term*0.5*0.5 + random.random()*term*0.5)) for i in range(part_num-1)]+[W] #part_num+1
            part_list = []
            for index in range(len(slice_points)-1):
                temp = image[:,slice_points[index]:slice_points[index+1],:]
                part_list.append(cv2.resize(temp,(round(temp.shape[1]*random_size[index]),round(temp.shape[0]))))
            result_masks_list = []
            for mask_index in range(len(masks)):
                mask = masks[mask_index]
                mask_list = []
                for index in range(len(slice_points)-1):
                    temp = mask[:,slice_points[index]:slice_points[index+1],]
                    mask_list.append(cv2.resize(temp,(round(temp.shape[1]*random_size[index]),round(temp.shape[0]))))
                result_masks_list.append(np.concatenate(mask_list,axis=1).astype(np.uint8))
            result_image = np.concatenate(part_list,axis=1).astype(np.uint8)
            return result_image,result_masks_list,labels
    else:
        return image,masks,labels


def make_gradation_images(rgb=(255,255,255)):
    if not os.path.isdir("./gradation_images"):
        os.mkdir("./gradation_images")
        WIDTH = 1280
        HEIGHT = 720
        for i in [1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7]:
            dummy_array = np.zeros((2*HEIGHT,2*WIDTH))
            m1,m2 = HEIGHT/WIDTH,-HEIGHT/WIDTH
            for x in range(-WIDTH,WIDTH):
                for y in range(-HEIGHT,HEIGHT):
                    if y<=m1*x:
                        if y>=m2*x:
                            #1
                            value = (-255/WIDTH)*x+255
                            dummy_array[(y+HEIGHT),(x+WIDTH)]=value
                        else: 
                            #4
                            value = (255/HEIGHT)*y+255
                            dummy_array[(y+HEIGHT),(x+WIDTH)]=value
                    else:
                        if y>=m2*x:
                            value = (-255/HEIGHT)*y+255
                            dummy_array[(y+HEIGHT),(x+WIDTH)]=value
                        else:
                            #3
                            value = (255/WIDTH)*x+255
                            dummy_array[(y+HEIGHT),(x+WIDTH)]=value
            # dummy_array = np.cos(5*np.pi*(dummy_array/255))/2+0.5
            dummy_array = (dummy_array/255)**i
            dummy_array = (dummy_array)*255
            dummy_array = dummy_array.astype(np.uint8)
            cv2.imwrite("./gradation_images/sample_center_gradation_{}.jpg".format(i),dummy_array)

        for j in list(range(3,30,2)):
            dummy_array_cos = np.zeros((2*HEIGHT,2*WIDTH))
            dummy_array_sin = np.zeros((2*HEIGHT,2*WIDTH))
            m1,m2 = HEIGHT/WIDTH,-HEIGHT/WIDTH
            for x in range(-WIDTH,WIDTH):
                for y in range(-HEIGHT,HEIGHT):
                    if y<=m1*x:
                        if y>=m2*x:
                            #1
                            value = (-255/WIDTH)*x+255
                            dummy_array_cos[(y+HEIGHT),(x+WIDTH)]=value
                            dummy_array_sin[(y+HEIGHT),(x+WIDTH)]=value
                        else: 
                            #4
                            value = (255/HEIGHT)*y+255
                            dummy_array_cos[(y+HEIGHT),(x+WIDTH)]=value
                            dummy_array_sin[(y+HEIGHT),(x+WIDTH)]=value
                    else:
                        if y>=m2*x:
                            value = (-255/HEIGHT)*y+255
                            dummy_array_cos[(y+HEIGHT),(x+WIDTH)]=value
                            dummy_array_sin[(y+HEIGHT),(x+WIDTH)]=value
                        else:
                            #3
                            value = (255/WIDTH)*x+255
                            dummy_array_cos[(y+HEIGHT),(x+WIDTH)]=value
                            dummy_array_sin[(y+HEIGHT),(x+WIDTH)]=value
            dummy_array_cos = np.cos(j*np.pi*(dummy_array_cos/255))/2+0.5
            dummy_array_other = np.sin(np.sin(np.sin(np.sin(np.sin(j*np.pi*(dummy_array_sin/255))/2+0.5))))
            dummy_array_cos = (dummy_array_cos)*255
            dummy_array_cos = dummy_array_cos.astype(np.uint8)
            dummy_array_other = (dummy_array_other)*255
            dummy_array_other = dummy_array_other.astype(np.uint8)
            
            cv2.imwrite("./gradation_images/sample_center_gradation_cos{}.jpg".format(j),dummy_array_cos)
            cv2.imwrite("./gradation_images/sample_center_gradation_other{}.jpg".format(j),dummy_array_other)
    else:
        pass
    


def random_gradation(image,masks,labels,p=0.5,gradation_intensity=0.25,):
    make_gradation_images()
    if random.random()<p:
        gradation_image_path = "./gradation_images"
        image_list = os.listdir(gradation_image_path)
        randomindex = random.randint(0,len(image_list)-1)
        gradation_image = os.path.join(gradation_image_path,image_list[randomindex])
        gradation_image = cv2.imread(gradation_image)
        w,h = gradation_image.shape[1]//2,gradation_image.shape[0]//2
        w_random,h_random = random.randint(0,w-1),random.randint(0,h-1)
        temp = gradation_image[h_random:(h_random+h),w_random:(w_random+w),:]
        temp = cv2.resize(temp,dsize=(image.shape[1],image.shape[0]))
        ratio = random.random()*gradation_intensity+1-gradation_intensity
        result = cv2.addWeighted(image,ratio,temp,1-ratio,0)
        return result,masks,labels
    else:
        return image,masks,labels



def liquify(img, cx1,cy1, cx2,cy2,half):
    x, y, w, h = cx1-half, cy1-half, half*2, half*2
    roi = img[y:y+h, x:x+w].copy()
    out = roi.copy()

    offset_cx1,offset_cy1 = cx1-x, cy1-y
    offset_cx2,offset_cy2 = cx2-x, cy2-y
    
    tri1 = [[ (0,0), (w, 0), (offset_cx1, offset_cy1)], # 상,top
            [ [0,0], [0, h], [offset_cx1, offset_cy1]], # 좌,left
            [ [w, 0], [offset_cx1, offset_cy1], [w, h]], # 우, right
            [ [0, h], [offset_cx1, offset_cy1], [w, h]]] # 하, bottom

    tri2 = [[ [0,0], [w,0], [offset_cx2, offset_cy2]], # 상, top
            [ [0,0], [0, h], [offset_cx2, offset_cy2]], # 좌, left
            [ [w,0], [offset_cx2, offset_cy2], [w, h]], # 우, right
            [ [0,h], [offset_cx2, offset_cy2], [w, h]]] # 하, bottom

    
    for i in range(4):
        matrix = cv2.getAffineTransform( np.float32(tri1[i]), \
                                         np.float32(tri2[i]))
        warped = cv2.warpAffine( roi.copy(), matrix, (w, h), \
            None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        mask = np.zeros((h, w), dtype = np.uint8)
        cv2.fillConvexPoly(mask, np.int32(tri2[i]), (255,255,255))
        
        warped = cv2.bitwise_and(warped, warped, mask=mask)
        out = cv2.bitwise_and(out, out,
                              mask=cv2.bitwise_not(mask))
        out = out + warped

    img[y:y+h, x:x+w] = out
    return img 

def random_grid_distortion(image,masks,labels,p=0.5,distort_intensity=0.20,):
    if random.random()<p:
        random_box_num = random.randint(5,10)
        data = []
        H,W,_ = image.shape
        h_step,w_step = H//random_box_num,W//random_box_num
        for y_axis in range(random_box_num):
            for x_axis in range(random_box_num):
                # distort_intensity = (random.random()*0.5+0.5)*distort_intensity
                x = random.randint(x_axis*w_step,round((x_axis+0.5)*w_step))
                y = random.randint(y_axis*h_step,round((y_axis+0.5)*h_step))
                toward_x,toward_y = random.random()*2-1,random.random()*2-1
                data.append([x,y,x+w_step//2,y+h_step//2,round(distort_intensity*toward_x*w_step//2),round(distort_intensity*toward_y*h_step//2)])
        
        result_masks = []
        for item in data:
            x1,y1,x2,y2,tx,ty = item
            x_mid,y_mid = (x1+x2)//2,(y1+y2)//2
            half = min((y2-y1)//2,(x2-x1)//2)
            
            result_image = liquify(image,x_mid,y_mid,x_mid+tx,y_mid+ty,half)
            masks = [ liquify(mask,x_mid,y_mid,x_mid+tx,y_mid+ty,half) for index,mask in enumerate(masks)]
        return result_image,masks,labels
            
    else:
        return image,masks,labels
    
    
def random_mosaic(image_list,masks_list,labels_list,p=0.5,option = "resize"):
    if False:
    # if p<random.random():
        return image_list[0],masks_list[0],labels_list[0]
    else:
        temp = list(range(4))
        np.random.shuffle(temp)
        if option =='crop':
            
            return
        else:
            h = max([ image_list[index].shape[0] for index in temp ])
            w = max([ image_list[index].shape[1] for index in temp ])
            black = np.zeros((h,w)).astype(np.uint8)
            center_x = round(w*0.5 + w*(np.random.uniform()*0.2-0.1))
            center_y = round(h*0.5 + h*(np.random.uniform()*0.2-0.1))
            # x,y,w,h
            last_masks = []
            last_labels = []
            image_sizes=[(0,0,center_x,center_y),(center_x,0,w-center_x,center_y),(0,center_y,center_x,h-center_y),(center_x,center_y,w-center_x,h-center_y)]
            for index,size in enumerate(image_sizes):
                target_image = image_list[temp[index]]
                target_masks = masks_list[temp[index]]
                target_labels = labels_list[temp[index]]
                image_list[temp[index]] = cv2.resize(target_image,size[2:4])
                for mask_index,mask in enumerate(target_masks):
                    black = np.zeros((h,w)).astype(np.uint8)
                    dummy=cv2.resize(mask,size[2:4])
                    black[size[1]:size[1]+size[3],size[0]:size[0]+size[2]]=dummy
                    last_masks.append(black)
                    last_labels.append(labels_list[temp[index]][mask_index])
                
                    
            t1 = np.concatenate([image_list[temp[0]],image_list[temp[1]]],axis=1)
            t2 = np.concatenate([image_list[temp[2]],image_list[temp[3]]],axis=1)
            last_image = np.concatenate([t1,t2],axis=0)
            return last_image,last_masks,last_labels