from threading import Thread
from PySide6.QtWidgets import QFrame,QLabel,QPushButton,QVBoxLayout,QCheckBox,QSlider,QHBoxLayout,QGridLayout,QWidget,QFileDialog,QLineEdit
from PySide6.QtCore import Qt,QSize
from PySide6.QtGui import QPixmap
from multiprocessing import Process,Queue

import albumentations as A
from instance_segmentation.utils import read_json
import cv2
import random
import copy
import time
from datetime import datetime
from instance_segmentation.augmentation import *
from instance_segmentation.utils import masks2polygons
import yaml
import json
from instance_segmentation.multi_aug import main

class InstanceSegmentation_Tab(QFrame):
    def __init__(self,WIDTH,HEIGHT):
        super().__init__()
        self.WIDTH=WIDTH
        self.HEIGHT=HEIGHT
        self.images_for_aug = None
        self.total_view = QHBoxLayout(self)
        self.label = QLabel("TEST")
        self.main_view = QLabel(self)
        self.total_view.addWidget(self.main_view)
        self.main_view.setPixmap(QPixmap("imgs/Readme.png").scaled(QSize(int(self.WIDTH*0.7), int(self.HEIGHT*0.7)), aspectMode=Qt.KeepAspectRatio))
        # self.main_view.setScaledContents(True)
        self.grid_layout = QGridLayout()
        self.total_view.addLayout(self.grid_layout)
        self.aug_count=0
        self.default_probability=50
        self.probatility_list=[]
        self.background_path=None
        self.save_folder = "./output"
        
        self.make_select_images_directory_button()
        self.make_load_button()
        self.make_save_path_button()
        self.make_background_images_button()
        self.make_aug_per_image_button()
        self.make_process_button()
        self.make_about_menu()
        self.make_total_probability()
        self.make_random_resize_aug()
        self.make_dropout_aug()
        self.make_background_change_aug()
        self.make_griddistortion_aug()
        self.make_brightness_aug()
        self.make_blur_aug()
        self.make_noise_aug()
        self.make_horizental_flip_aug()
        self.make_vertical_flip_aug()
        self.make_downscale_aug()
        self.make_mixup_aug()
        self.make_colorjitter_aug()
        self.make_optical_distortion_aug()
        self.make_gradation_aug()
        self.make_affine_aug()
        self.make_perspective_aug()
        self.make_run_button()
        self.load_log()
        
    
    def load_log(self):
        if os.path.isfile("instance_segmentation/instance_seg_log.yaml"):
            with open("instance_segmentation/instance_seg_log.yaml","r") as f:
                self.log = yaml.load(f,yaml.FullLoader)
            self.background_path = self.log["background_path"]
            self.images_folder = self.log["images_folder"]
            self.label_file_path = self.log["label_file_path"]
            self.save_folder = self.log["save_folder"]
            self.line_edit.setText(self.log["aug_nums"])
            self.process_edit.setText(str(self.log["nums_process"]))
            
            
            self.resize_slider_probability.setValue(self.log["resize_probability"])
            self.resize_slider.setValue(self.log["resize_strength"]),
            self.resize_checkbox.setChecked(self.log["resize_apply"])
            
            self.dropout_slider_probability.setValue(self.log["dropout_probability"]),
            self.dropout_slider.setValue(self.log["dropout_strength"])
            self.dropout_checkbox.setChecked(self.log["dropout_apply"])
            
            
            self.background_slider_probability.setValue(self.log['background_probability'])
            self.background_slider.setValue(self.log['background_strength'])
            self.background_checkbox.setChecked(self.log["background_apply"])
            
            self.griddistortion_slider_probability.setValue(self.log["griddistortion_probability"])
            self.griddistortion_slider.setValue(self.log["griddistortion_strength"]),
            self.griddistortion_checkbox.setChecked(self.log["griddistortion_apply"])
            
            
            self.brightness_slider_probability.setValue(self.log["brightness_probability"]),
            self.brightness_slider.setValue(self.log["brightness_strength"]),
            self.brightness_checkbox.setChecked(self.log["brightness_apply"])
            
            self.blur_slider_probability.setValue(self.log["blur_probability"]),
            self.blur_slider.setValue(self.log["blur_strength"]),
            self.blur_checkbox.setChecked(self.log["blur_apply"])
            
            self.noise_slider_probability.setValue(self.log["noise_probability"]),
            self.noise_slider.setValue(self.log["noise_strength"]),
            self.noise_checkbox.setChecked(self.log["noise_apply"])
            
            self.horizentalflip_slider_probability.setValue(self.log["horizentalflip_probability"]),
            self.horizentalflip_checkbox.setChecked(self.log["horizentalflip_apply"])
            
            self.verticalflip_slider_probability.setValue(self.log["verticalflip_probability"]),
            self.verticalflip_checkbox.setChecked(self.log["verticalflip_apply"])
            
            self.downscale_slider_probability.setValue(self.log["downscale_probability"]),
            self.downscale_slider.setValue(self.log["downscale_strength"]),
            self.downscale_checkbox.setChecked(self.log["downscale_apply"])
            
            self.mixup_slider_probability.setValue(self.log["mixup_probability"]),
            self.mixup_slider.setValue(self.log["mixup_strength"]),
            self.mixup_checkbox.setChecked(self.log["mixup_apply"])
            
            self.colorjitter_slider_probability.setValue(self.log["colorjitter_probability"]),
            self.colorjitter_slider.setValue(self.log["colorjitter_strength"]),
            self.colorjitter_checkbox.setChecked(self.log["colorjitter_apply"])
            
            self.optical_slider_probability.setValue(self.log["opticaldistortion_probability"]),
            self.optical_slider.setValue(self.log["opticaldistortion_strength"]),
            self.optical_checkbox.setChecked(self.log["opticaldistortion_apply"])
            
            self.gradation_slider_probability.setValue(self.log["gradation_probability"]),
            self.gradation_slider.setValue(self.log["gradation_strength"]),
            self.gradation_checkbox.setChecked(self.log["gradation_apply"])
            
            
            self.affine_slider_probability.setValue(self.log["affine_probability"]),
            self.affine_slider.setValue(self.log["affine_strength"]),
            self.affine_checkbox.setChecked(self.log["affine_apply"])
            
            self.perspective_slider_probability.setValue(self.log["perspective_probability"]),
            self.perspective_slider.setValue(self.log["perspective_strength"]),
            self.perspective_checkbox.setChecked(self.log["perspective_apply"])
    
    def save_log(self):
        instance_seg_log = {
            "images_folder" : self.images_folder,
            "label_file_path" : self.label_file_path,
            "background_path" : self.background_path,
            "save_folder" : self.save_folder,
            "nums_process" : int(self.process_edit.text()),
            "aug_nums":self.line_edit.text(),
            
            "resize_probability":self.resize_slider_probability.value(),
            "resize_strength":self.resize_slider.value(),
            "resize_apply":self.resize_checkbox.isChecked(),
            
            "dropout_probability":self.dropout_slider_probability.value(),
            "dropout_strength":self.dropout_slider.value(),
            "dropout_apply":self.dropout_checkbox.isChecked(),
            
            "background_probability":self.background_slider_probability.value(),
            "background_strength":self.background_slider.value(),
            "background_apply":self.background_checkbox.isChecked(),
            
            "griddistortion_probability":self.griddistortion_slider_probability.value(),
            "griddistortion_strength":self.griddistortion_slider.value(),
            "griddistortion_apply":self.griddistortion_checkbox.isChecked(),
            
            "brightness_probability":self.brightness_slider_probability.value(),
            "brightness_strength":self.brightness_slider.value(),
            "brightness_apply":self.brightness_checkbox.isChecked(),
            
            "blur_probability":self.blur_slider_probability.value(),
            "blur_strength":self.blur_slider.value(),
            "blur_apply":self.blur_checkbox.isChecked(),
            
            "noise_probability":self.noise_slider_probability.value(),
            "noise_strength":self.noise_slider.value(),
            "noise_apply":self.noise_checkbox.isChecked(),
            
            "horizentalflip_probability":self.horizentalflip_slider_probability.value(),
            "horizentalflip_apply":self.horizentalflip_checkbox.isChecked(),
            
            "verticalflip_probability":self.verticalflip_slider_probability.value(),
            "verticalflip_apply":self.verticalflip_checkbox.isChecked(),
            
            "downscale_probability":self.downscale_slider_probability.value(),
            "downscale_strength":self.downscale_slider.value(),
            "downscale_apply":self.downscale_checkbox.isChecked(),
            
            "mixup_probability":self.mixup_slider_probability.value(),
            "mixup_strength":self.mixup_slider.value(),
            "mixup_apply":self.mixup_checkbox.isChecked(),
            
            "colorjitter_probability":self.colorjitter_slider_probability.value(),
            "colorjitter_strength":self.colorjitter_slider.value(),
            "colorjitter_apply":self.colorjitter_checkbox.isChecked(),
            
            "opticaldistortion_probability":self.optical_slider_probability.value(),
            "opticaldistortion_strength":self.optical_slider.value(),
            "opticaldistortion_apply":self.optical_checkbox.isChecked(),
            
            "gradation_probability":self.resize_slider_probability.value(),
            "gradation_strength":self.gradation_slider.value(),
            "gradation_apply":self.gradation_checkbox.isChecked(),
            
            "affine_probability":self.resize_slider_probability.value(),
            "affine_strength":self.affine_slider.value(),
            "affine_apply":self.affine_checkbox.isChecked(),
            
            "perspective_probability":self.resize_slider_probability.value(),
            "perspective_strength":self.perspective_slider.value(),
            "perspective_apply":self.perspective_checkbox.isChecked(),
        }
        with open("instance_segmentation/instance_seg_log.yaml","w") as f:
            yaml.dump(instance_seg_log,f)
    
            
        
    def AUGMENTATION_RUN(self):
        self.save_log()
        if not os.path.isdir(self.save_folder):
            os.mkdir(self.save_folder)
        if self.label_file_path:
            if self.images_folder:
                coco_data,categories = read_json(self.label_file_path,self.images_folder)
            else:
                raise Exception("이미지폴더 경로가 유효하지않음")
        else:
            raise Exception("라벨파일의 경로가 유효하지않음")
        self.make_albumentation_pipeline()
        self.make_custom_aug_pipeline()
        images=[]
        annotations=[]
        number=1
        annotation_number = 1
        today = str(datetime.now())
        today = today.split(" ")[0]
        folder_count=1
        while True:
            if not os.path.isdir(self.save_folder+"/{}_{}".format(today,folder_count)):
                os.mkdir(self.save_folder+"/{}_{}".format(today,folder_count))
                full_save_path = self.save_folder+"/{}_{}".format(today,folder_count)
                break
            else:
                folder_count+=1
        for index,item in enumerate(coco_data):
            pre_image_origin,pre_masks_origin,pre_categories_about_one_mask = item
            #############################
            a = np.random.choice(range(len(coco_data)),3)
            i0,m0,l0 = coco_data[a[0]]
            i1,m1,l1 = coco_data[a[1]]
            i2,m2,l2 = coco_data[a[2]]
            images_list = [pre_image_origin,i0,i1,i2]
            masks_list = [pre_masks_origin,m0,m1,m2]
            labels_list = [pre_categories_about_one_mask,l0,l1,l2]
            pre_image_origin,pre_masks_origin,pre_categories_about_one_mask = random_mosaic(images_list,masks_list,labels_list)
            #############################
            for j in range(int(self.line_edit.text())):
                t = "_".join(str(time.time()).split("."))
                result_image = copy.deepcopy(pre_image_origin)
                result_masks = copy.deepcopy(pre_masks_origin)
                result_labels = copy.deepcopy(pre_categories_about_one_mask)
                result_image,result_masks,result_labels = self.execute_one_aug(result_image,result_masks,result_labels)
                result_polygons,result_categories_about_one_mask,result_image = masks2polygons(result_masks,result_labels,result_image=result_image,dummy_image_list=None)
                for polygon,one_category_num in zip(result_polygons,result_categories_about_one_mask):
                    x,y,w,h = cv2.boundingRect(polygon)
                    area = int(cv2.contourArea(polygon))
                    seg = [ int(i) for i in polygon[::4,:].reshape(-1)]
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
                    number+=1
                cv2.imwrite(full_save_path+"/{}.jpg".format(t),result_image)
                images.append({
                        "height":result_image.shape[0],
                        "width":result_image.shape[1],
                        "id":number,
                        "file_name":t+".jpg",
                    })
                del result_image,result_masks, result_polygons,result_categories_about_one_mask
                
            try:
                del i0,i1,i2,m0,m1,m2,l0,l1,l2
            except:
                pass
            
        result_object = {
            "images":images,
            "annotations":annotations,
            "categories":categories,
            }
        with open('{}/data.json'.format(full_save_path), 'w') as f:
            json_string = json.dump(result_object, f, indent=2)
            
    def MULTI_AUGMENTATION_RUN(self):
        self.save_log()
        self.load_log()
        if self.log["nums_process"]<2:
            self.AUGMENTATION_RUN()
        else:
            main()
        

    def execute_one_aug(self,image,masks,labels):
        random.shuffle(self.custom_pipeline)
        random.shuffle(self.pipelines)
        temp = [-1]+list(range(len(self.custom_pipeline)))
        random.shuffle(temp)
        for index in temp:
            if index==-1:
                transformed = self.pipelines[0](image=image,masks=masks)
                image = transformed['image']
                masks = transformed['masks']
            else:
                image,masks,labels = self.custom_pipeline[index][0](image,masks,labels,**self.custom_pipeline[index][1])
        return image,masks,labels
    
    def preprocess(self,images_list,masks_list,labels_list):
        return images_list,masks_list,labels_list
        
    def make_custom_aug_pipeline(self):
        custom_pipeline=[]
        if self.resize_checkbox.isChecked():
            custom_pipeline.append(
                [
                    random_part_resize,
                    {
                        "p":self.resize_slider_probability.value()/100
                    }
                ]
            )
        
        if self.dropout_checkbox.isChecked():
            custom_pipeline.append(
                [
                    random_dropout,
                    {
                        "p": self.dropout_slider_probability.value()/100,
                        "min_instance":0,
                        "max_instance":self.dropout_slider.value(),
                    }
                ]
            )
            
        if self.background_checkbox.isChecked(): # X
            if self.background_slider_probability.value()/100>1e-3:
                if self.background_path is None:
                    # print("!?!?!?")
                    if not os.path.isdir("../mixup_data"):
                        mixupdata_download()
                    self.background_path = "../mixup_data"
            custom_pipeline.append(
                [
                    random_background_change,
                    {
                        "p":self.background_slider_probability.value()/100,
                        "patch_image":self.background_path
                    }
                ]
            )
        if self.griddistortion_checkbox.isChecked():
            custom_pipeline.append(
                [
                    random_grid_distortion,
                    {
                        "p":self.griddistortion_slider_probability.value()/100,
                        "distort_intensity":self.griddistortion_slider.value()/100,
                    }
                ]
            )
            
        if self.gradation_checkbox.isChecked():
            custom_pipeline.append(
                [
                    random_gradation,
                    {
                        "p":self.gradation_slider_probability.value()/100,
                        "gradation_intensity":self.gradation_slider.value()/100
                        
                    }
                ]
            )
        self.custom_pipeline = custom_pipeline
    
    def make_albumentation_pipeline(self):
        pipeline=[]
        if self.blur_checkbox.isChecked():
            pipeline.append(
                A.OneOf(
                        [
                            A.GlassBlur(p=1,max_delta=round(self.blur_slider.value()/6)),
                            A.MedianBlur(p=1,blur_limit=round(self.blur_slider.value()*3)),
                            A.MotionBlur(p=1,blur_limit=round(self.blur_slider.value()*3)),
                            A.GaussianBlur(p=1,blur_limit=(3,round(self.blur_slider.value()*3)))
                        ],
                        p=self.blur_slider_probability.value()/100
                        ),
            )
        
        if self.noise_checkbox.isChecked():
            pipeline.append(
                A.OneOf(
                        [
                            A.GaussNoise(p=1,var_limit=(100,self.noise_slider.value()*30),mean=10,per_channel=False),
                            A.GaussNoise(p=1,var_limit=(100,self.noise_slider.value()*30),mean=10,per_channel=True)
                        ],
                        p=self.noise_slider_probability.value()
                        ),
            )
        
        if self.brightness_checkbox.isChecked():
            pipeline.append(
                A.RandomBrightnessContrast(p=self.brightness_slider_probability.value()/100,brightness_limit=self.brightness_slider.value()/50)
            )
        
        if self.horizentalflip_checkbox.isChecked():
            pipeline.append(
                A.HorizontalFlip(p=self.horizentalflip_slider_probability.value()/100)
            )
        
        if self.verticalflip_checkbox.isChecked():
            pipeline.append(
                A.HorizontalFlip(p=self.verticalflip_slider_probability.value()/100)
            )
            
        if self.downscale_checkbox.isChecked():
            pipeline.append(
                A.Downscale(p=self.downscale_slider_probability.value()/100,scale_max=0.999,scale_min=1-self.downscale_slider.value()/50),
            )
            
        if self.colorjitter_checkbox.isChecked():
            pipeline.append(
                A.OneOf(
                            [
                                A.ColorJitter(p=1,),
                                A.ISONoise(p=1,),
                                A.FancyPCA(p=1,),
                                A.HueSaturationValue(p=1,
                                                     hue_shift_limit=(-int(self.colorjitter_slider.value())*0.5,int(self.colorjitter_slider.value())*0.5),
                                                     sat_shift_limit=(-int(self.colorjitter_slider.value())*0.6,int(self.colorjitter_slider.value())*0.6),
                                                     val_shift_limit=(-int(self.colorjitter_slider.value())*0.5,int(self.colorjitter_slider.value())*0.5))
                            ],
                            p=self.colorjitter_slider_probability.value()/100
                        ),
            )
        
        if self.optical_checkbox.isChecked():
            pipeline.append(
                A.OpticalDistortion(
                        p=self.optical_slider_probability.value()/100,
                        distort_limit=self.optical_slider.value()/40,
                        shift_limit=self.optical_slider.value()/80,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0
                    ),
            )
            
        if self.affine_checkbox.isChecked():
            pipeline.append(
                A.Compose(
                            [
                                A.Affine(p=1,),
                                A.RandomResizedCrop(p=1,height=1280,width=1280,scale=(1-self.affine_slider.value()/100,1))
                            ],
                            p=self.affine_slider_probability.value()/100,
                        )
            )
            
        if self.perspective_checkbox.isChecked():
            pipeline.append(
                A.Perspective(p=1,scale=(0.05,0.05+self.perspective_slider.value()/100))
            )
        
        pipelines = []
        for i in range(20):
            random.shuffle(pipeline)
            temp = copy.deepcopy(A.Compose(pipeline))
            pipelines.append(temp)
        self.pipelines = pipelines
        
        
    def make_total_probability(self):
        self.total_probability_minus = QPushButton("-")
        self.total_probability_minus.clicked.connect(self.probability_minus)
        self.total_probability_plus = QPushButton("+")
        self.total_probability_plus.clicked.connect(self.probability_plus)
        self.total_probability = QLabel(str(self.default_probability))
        self.total_probability.setAlignment(Qt.AlignCenter)
        self.grid_layout.addWidget(self.total_probability_minus,self.aug_count,0)
        self.grid_layout.addWidget(self.total_probability_plus,self.aug_count,2)
        self.grid_layout.addWidget(self.total_probability,self.aug_count,1)
        self.aug_count+=1
        
    def probability_minus(self,):
        self.default_probability-=1
        self.total_probability.setText(str(self.default_probability))
        for slider in self.probatility_list:
            slider.setValue(slider.value()-1)
    
    def probability_plus(self,):
        self.default_probability+=1
        self.total_probability.setText(str(self.default_probability))
        for slider in self.probatility_list:
            slider.setValue(slider.value()+1)
        
    def load_label_file(self):
        fname = QFileDialog.getOpenFileName(self,filter="*.json")
        self.label_file_path=fname[0]
    
    def load_images_path(self):
        self.images_folder = QFileDialog.getExistingDirectory(self)
    
    def load_save_path(self):
        self.save_folder = QFileDialog.getExistingDirectory(self)
        self.save_folder +="/output"
        
    def load_images_folder_for_mixup(self):
        self.images_for_aug = QFileDialog.getExistingDirectory(self)
    
    def connect_aug_count(self):
        self.aug_count.setText()
    
    def make_aug_per_image_button(self):
        self.nums_label = QLabel("Aug 개수 : ")
        self.grid_layout.addWidget(self.nums_label,self.aug_count,0)
        self.nums_real = QLabel("")
        self.grid_layout.addWidget(self.nums_real,self.aug_count,1)
        self.line_edit = QLineEdit()
        self.line_edit.setText("10")
        self.line_edit.textChanged.connect(self.nums_real.setText)
        self.grid_layout.addWidget(self.line_edit,self.aug_count,2)
        self.aug_count+=1
        
    def make_process_button(self):
        self.nums_process = QLabel("Process 개수 : ")
        self.grid_layout.addWidget(self.nums_process,self.aug_count,0)
        self.nums_process_show = QLabel("")
        self.grid_layout.addWidget(self.nums_process_show,self.aug_count,1)
        self.process_edit = QLineEdit()
        self.process_edit.setText("5")
        self.process_edit.textChanged.connect(self.nums_process_show.setText)
        self.grid_layout.addWidget(self.process_edit,self.aug_count,2)
        self.aug_count+=1
    
    def make_background_images_button(self):
        background_image_path_button = QPushButton("BackGround 이미지 경로")
        background_image_path_button.clicked.connect(self.load_images_folder_for_mixup)
        self.grid_layout.addWidget(background_image_path_button,self.aug_count,0,1,3)
        self.aug_count+=1
    
    def make_save_path_button(self):
        save_path_button = QPushButton("Aug된 이미지 저장할 경로")
        save_path_button.clicked.connect(self.load_save_path)
        self.grid_layout.addWidget(save_path_button,self.aug_count,0,1,3)
        self.aug_count+=1
    
    def make_select_images_directory_button(self):
        load_button = QPushButton("Aug할 이미지들 경로")
        load_button.clicked.connect(self.load_images_path)
        self.grid_layout.addWidget(load_button,self.aug_count,0,1,3)
        self.aug_count+=1
        
    def make_load_button(self):
        load_button = QPushButton("COCO 라벨 경로")
        load_button.clicked.connect(self.load_label_file)
        self.grid_layout.addWidget(load_button,self.aug_count,0,1,3)
        self.aug_count+=1
        
    def make_run_button(self):
        run_button = QPushButton("Aug 실행")
        run_button.clicked.connect(self.MULTI_AUGMENTATION_RUN)
        self.grid_layout.addWidget(run_button,self.aug_count,0,1,3)
        self.aug_count+=1
        self.start_time = time.time()
    
    def make_about_menu(self,maximum_height=30):
        name_label = QLabel("Agumentation Name")
        name_label.setMaximumHeight(maximum_height)
        probability_label = QLabel("Probability")
        probability_label.setMaximumHeight(maximum_height)
        strength_label = QLabel("How Strongly")
        strength_label.setMaximumHeight(maximum_height)
        self.grid_layout.addWidget(name_label,self.aug_count,0)
        self.grid_layout.addWidget(probability_label,self.aug_count,1)
        self.grid_layout.addWidget(strength_label,self.aug_count,2)
        self.aug_count+=1

    
    def make_random_resize_aug(self):
        self.resize_checkbox = QCheckBox("Random Part Resize")
        self.resize_checkbox.setChecked(True)
        self.resize_slider_probability = QSlider(Qt.Horizontal)
        self.resize_slider_probability.setRange(0,100)
        self.resize_slider_probability.setValue(50)
        self.probatility_list.append(self.resize_slider_probability)
        
        self.resize_slider = QSlider(Qt.Horizontal)
        self.resize_slider.setRange(0,30)
        self.resize_slider.setValue(15)
        self.resize_slider.setSingleStep(1)
        
        self.grid_layout.addWidget(self.resize_checkbox,self.aug_count,0)
        self.grid_layout.addWidget(self.resize_slider_probability,self.aug_count,1)
        self.grid_layout.addWidget(self.resize_slider,self.aug_count,2)
        self.aug_count+=1
        
    def make_dropout_aug(self):
        self.dropout_checkbox = QCheckBox("Random Dropout")
        self.dropout_checkbox.setChecked(True)
        self.dropout_slider_probability = QSlider(Qt.Horizontal)
        self.dropout_slider_probability.setRange(0,100)
        self.dropout_slider_probability.setValue(50)
        self.probatility_list.append(self.dropout_slider_probability)
        
        self.dropout_slider = QSlider(Qt.Horizontal)
        self.dropout_slider.setRange(0,100)
        self.dropout_slider.setValue(5)
        self.dropout_slider.setSingleStep(1)
        
        
        self.grid_layout.addWidget(self.dropout_checkbox,self.aug_count,0)
        self.grid_layout.addWidget(self.dropout_slider_probability,self.aug_count,1)
        self.grid_layout.addWidget(self.dropout_slider,self.aug_count,2)
        self.aug_count+=1
        
        
    def make_background_change_aug(self):
        self.background_checkbox = QCheckBox("Random Background Change")
        self.background_checkbox.setChecked(False)
        self.background_slider_probability = QSlider(Qt.Horizontal)
        self.background_slider_probability.setRange(0,100)
        self.background_slider_probability.setValue(50)
        self.probatility_list.append(self.background_slider_probability)
        
        self.background_slider = QSlider(Qt.Horizontal)
        self.background_slider.setRange(0,10)
        self.background_slider.setValue(5)
        self.background_slider.setSingleStep(1)
        
        self.grid_layout.addWidget(self.background_checkbox,self.aug_count,0)
        self.grid_layout.addWidget(self.background_slider_probability,self.aug_count,1)
        self.grid_layout.addWidget(self.background_slider,self.aug_count,2)
        self.aug_count+=1
        
    def make_griddistortion_aug(self):
        self.griddistortion_checkbox = QCheckBox("Random Grid Distortion")
        self.griddistortion_checkbox.setChecked(True)
        self.griddistortion_slider_probability = QSlider(Qt.Horizontal)
        self.griddistortion_slider_probability.setRange(0,100)
        self.griddistortion_slider_probability.setValue(50)
        self.probatility_list.append(self.griddistortion_slider_probability)
        
        self.griddistortion_slider = QSlider(Qt.Horizontal)
        self.griddistortion_slider.setRange(0,40)
        self.griddistortion_slider.setValue(15)
        
        self.grid_layout.addWidget(self.griddistortion_checkbox,self.aug_count,0)
        self.grid_layout.addWidget(self.griddistortion_slider_probability,self.aug_count,1)
        self.grid_layout.addWidget(self.griddistortion_slider,self.aug_count,2)
        self.aug_count+=1
        
    def make_brightness_aug(self):
        self.brightness_checkbox = QCheckBox("Random Brightness")
        self.brightness_checkbox.setChecked(True)
        self.brightness_slider_probability = QSlider(Qt.Horizontal)
        self.brightness_slider_probability.setRange(0,100)
        self.brightness_slider_probability.setValue(50)
        self.probatility_list.append(self.brightness_slider_probability)
        
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(0,50)
        self.brightness_slider.setValue(25)
        
        self.grid_layout.addWidget(self.brightness_checkbox,self.aug_count,0)
        self.grid_layout.addWidget(self.brightness_slider_probability,self.aug_count,1)
        self.grid_layout.addWidget(self.brightness_slider,self.aug_count,2)
        self.aug_count+=1
        
    def make_blur_aug(self):
        self.blur_checkbox = QCheckBox("Random Blur")
        self.blur_checkbox.setChecked(True)
        self.blur_slider_probability = QSlider(Qt.Horizontal)
        self.blur_slider_probability.setRange(0,100)
        self.blur_slider_probability.setValue(50)
        self.probatility_list.append(self.blur_slider_probability)
        
        self.blur_slider = QSlider(Qt.Horizontal)
        self.blur_slider.setRange(3,9)
        self.blur_slider.setValue(5)
        self.blur_slider.setSingleStep(2)
        
        self.grid_layout.addWidget(self.blur_checkbox,self.aug_count,0)
        self.grid_layout.addWidget(self.blur_slider_probability,self.aug_count,1)
        self.grid_layout.addWidget(self.blur_slider,self.aug_count,2)
        self.aug_count+=1
        
    def make_noise_aug(self):
        self.noise_checkbox = QCheckBox("Random Noise")
        self.noise_checkbox.setChecked(True)
        self.noise_slider_probability = QSlider(Qt.Horizontal)
        self.noise_slider_probability.setRange(0,100)
        self.noise_slider_probability.setValue(50)
        self.probatility_list.append(self.noise_slider_probability)
        
        self.noise_slider = QSlider(Qt.Horizontal)
        self.noise_slider.setRange(0,50)
        self.noise_slider.setValue(25)
        self.noise_slider.setSingleStep(1)
        self.grid_layout.addWidget(self.noise_checkbox,self.aug_count,0)
        self.grid_layout.addWidget(self.noise_slider_probability,self.aug_count,1)
        self.grid_layout.addWidget(self.noise_slider,self.aug_count,2)
        self.aug_count+=1
        
    def make_horizental_flip_aug(self):
        self.horizentalflip_checkbox = QCheckBox("Random Horizental Flip")
        self.horizentalflip_checkbox.setChecked(True)
        self.horizentalflip_slider_probability = QSlider(Qt.Horizontal)
        self.horizentalflip_slider_probability.setRange(0,100)
        self.horizentalflip_slider_probability.setValue(50)
        self.probatility_list.append(self.horizentalflip_slider_probability)
        self.grid_layout.addWidget(self.horizentalflip_checkbox,self.aug_count,0)
        self.grid_layout.addWidget(self.horizentalflip_slider_probability,self.aug_count,1)
        self.aug_count+=1
        
    def make_vertical_flip_aug(self):
        self.verticalflip_checkbox = QCheckBox("Random Vertical Flip")
        self.verticalflip_checkbox.setChecked(True)
        self.verticalflip_slider_probability = QSlider(Qt.Horizontal)
        self.verticalflip_slider_probability.setRange(0,100)
        self.verticalflip_slider_probability.setValue(50)
        self.probatility_list.append(self.verticalflip_slider_probability)
        self.grid_layout.addWidget(self.verticalflip_checkbox,self.aug_count,0)
        self.grid_layout.addWidget(self.verticalflip_slider_probability,self.aug_count,1)
        self.aug_count+=1
        
    def make_downscale_aug(self):
        self.downscale_checkbox = QCheckBox("Random DownScale")
        self.downscale_checkbox.setChecked(True)
        self.downscale_slider_probability = QSlider(Qt.Horizontal)
        self.downscale_slider_probability.setRange(0,100)
        self.downscale_slider_probability.setValue(50)
        self.probatility_list.append(self.downscale_slider_probability)
        
        self.downscale_slider = QSlider(Qt.Horizontal)
        self.downscale_slider.setRange(0,80)
        self.downscale_slider.setValue(40)
        
        self.grid_layout.addWidget(self.downscale_checkbox,self.aug_count,0)
        self.grid_layout.addWidget(self.downscale_slider_probability,self.aug_count,1)
        self.grid_layout.addWidget(self.downscale_slider,self.aug_count,2)
        self.aug_count+=1
        
    def make_mixup_aug(self):
        self.mixup_checkbox = QCheckBox("Random Mixup")
        self.mixup_checkbox.setChecked(False)
        self.mixup_slider_probability = QSlider(Qt.Horizontal)
        self.mixup_slider_probability.setRange(0,100)
        self.mixup_slider_probability.setValue(50)
        self.probatility_list.append(self.mixup_slider_probability)
        
        self.mixup_slider = QSlider(Qt.Horizontal)
        self.mixup_slider.setRange(0,100)
        self.mixup_slider.setValue(50)
        
        self.grid_layout.addWidget(self.mixup_checkbox,self.aug_count,0)
        self.grid_layout.addWidget(self.mixup_slider_probability,self.aug_count,1)
        self.grid_layout.addWidget(self.mixup_slider,self.aug_count,2)
        self.aug_count+=1
    
    def make_colorjitter_aug(self):
        self.colorjitter_checkbox = QCheckBox("Random Color Transform")
        self.colorjitter_checkbox.setChecked(True)
        self.colorjitter_slider_probability = QSlider(Qt.Horizontal)
        self.colorjitter_slider_probability.setRange(0,100)
        self.colorjitter_slider_probability.setValue(50)
        self.probatility_list.append(self.colorjitter_slider_probability)
        
        self.colorjitter_slider = QSlider(Qt.Horizontal)
        self.colorjitter_slider.setRange(0,100)
        self.colorjitter_slider.setValue(50)
        
        self.grid_layout.addWidget(self.colorjitter_checkbox,self.aug_count,0)
        self.grid_layout.addWidget(self.colorjitter_slider_probability,self.aug_count,1)
        self.grid_layout.addWidget(self.colorjitter_slider,self.aug_count,2)
        self.aug_count+=1
        
    def make_optical_distortion_aug(self):
        self.optical_checkbox = QCheckBox("Random Optical Transform")
        self.optical_checkbox.setChecked(True)
        self.optical_slider_probability = QSlider(Qt.Horizontal)
        self.optical_slider_probability.setRange(0,100)
        self.optical_slider_probability.setValue(50)
        self.probatility_list.append(self.optical_slider_probability)
        
        self.optical_slider = QSlider(Qt.Horizontal)
        self.optical_slider.setRange(0,10)
        self.optical_slider.setValue(5)
        self.grid_layout.addWidget(self.optical_checkbox,self.aug_count,0)
        self.grid_layout.addWidget(self.optical_slider_probability,self.aug_count,1)
        self.grid_layout.addWidget(self.optical_slider,self.aug_count,2)
        self.aug_count+=1
    
    def make_gradation_aug(self):
        self.gradation_checkbox = QCheckBox("Random gradation Aug")
        self.gradation_checkbox.setChecked(True)
        self.gradation_slider_probability = QSlider(Qt.Horizontal)
        self.gradation_slider_probability.setRange(0,100)
        self.gradation_slider_probability.setValue(50)
        self.probatility_list.append(self.gradation_slider_probability)
        
        self.gradation_slider = QSlider(Qt.Horizontal)
        self.gradation_slider.setRange(0,35)
        self.gradation_slider.setValue(15)
        self.grid_layout.addWidget(self.gradation_checkbox,self.aug_count,0)
        self.grid_layout.addWidget(self.gradation_slider_probability,self.aug_count,1)
        self.grid_layout.addWidget(self.gradation_slider,self.aug_count,2)
        self.aug_count+=1
    
    def make_affine_aug(self):
        self.affine_checkbox = QCheckBox("Random Affine Transform")
        self.affine_checkbox.setChecked(True)
        self.affine_slider_probability = QSlider(Qt.Horizontal)
        self.affine_slider_probability.setRange(0,100)
        self.affine_slider_probability.setValue(50)
        self.probatility_list.append(self.affine_slider_probability)
        
        self.affine_slider = QSlider(Qt.Horizontal)
        self.affine_slider.setRange(0,30)
        self.affine_slider.setValue(15)
        
        self.grid_layout.addWidget(self.affine_checkbox,self.aug_count,0)        
        self.grid_layout.addWidget(self.affine_slider_probability,self.aug_count,1)
        self.grid_layout.addWidget(self.affine_slider,self.aug_count,2)
        self.aug_count+=1

    def make_perspective_aug(self):
        self.perspective_checkbox = QCheckBox("Random Perspective Transform")
        self.perspective_checkbox.setChecked(True)
        self.perspective_slider_probability = QSlider(Qt.Horizontal)
        self.perspective_slider_probability.setRange(0,100)
        self.perspective_slider_probability.setValue(50)
        self.probatility_list.append(self.perspective_slider_probability)
        
        self.perspective_slider = QSlider(Qt.Horizontal)
        self.perspective_slider.setRange(0,100)
        self.perspective_slider.setValue(15)
        
        self.grid_layout.addWidget(self.perspective_checkbox,self.aug_count,0)
        self.grid_layout.addWidget(self.perspective_slider_probability,self.aug_count,1)
        self.grid_layout.addWidget(self.perspective_slider,self.aug_count,2)
        self.aug_count+=1