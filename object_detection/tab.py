from PySide6.QtWidgets import QFrame,QHBoxLayout,QLabel,QGridLayout,QPushButton,QSlider,QFileDialog,QCheckBox
from PySide6.QtGui import QPixmap
from PySide6.QtCore import QSize,Qt
import time,os,sys,yaml


class OjbectDetection_Tab(QFrame):
    def __init__(self,WIDTH,HEIGHT):
        super().__init__()
        self.WIDTH=WIDTH
        self.HEIGHT=HEIGHT
        self.total_view = QHBoxLayout(self)
        self.main_view = QLabel(self)
        self.total_view.addWidget(self.main_view)
        self.main_view.setPixmap(QPixmap("imgs/Readme.png").scaled(QSize(int(self.WIDTH*0.7), int(self.HEIGHT*0.7)), aspectMode=Qt.KeepAspectRatio))
        self.grid_layout = QGridLayout()
        self.total_view.addLayout(self.grid_layout)
        self.aug_count=0
        self.images_folder = None
        self.labels_folder = None
        self.save_folder = None
        self.images_for_aug = None
        self.probatility_list=[]
        
        self.load_log()
        

        self.make_select_images_directory_button()
        self.make_load_button()
        self.make_save_path_button()
        self.make_background_images_button()
        self.make_about_menu()
        
        self.make_mixup_aug()
        self.make_random_brighitness()
        self.make_blur_aug()
        self.make_colorjitter_aug()
        self.make_downscale_aug()
        self.make_gradation_aug()
        self.make_horizental_flip_aug()
        self.make_vertical_flip_aug()
        self.make_affine_aug()
        self.make_perspective_aug()
        
        
        self.make_run_button()
        self.save_log()
            
    def make_random_brighitness(self):
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
        
        
        
        
    def make_run_button(self):
        run_button = QPushButton("Aug 실행")
        run_button.clicked.connect(self.MULTI_AUGMENTATION_RUN)
        self.grid_layout.addWidget(run_button,self.aug_count,0,1,3)
        self.aug_count+=1
        self.start_time = time.time()
    
    def MULTI_AUGMENTATION_RUN(self):
        return
        
    def make_select_images_directory_button(self):
        load_button = QPushButton("Aug할 이미지들 경로")
        load_button.clicked.connect(self.load_images_path)
        self.grid_layout.addWidget(load_button,self.aug_count,0,1,3)
        self.aug_count+=1
    def load_images_path(self):
        self.images_folder = QFileDialog.getExistingDirectory(self)
        
    def make_load_button(self):
        load_button = QPushButton("Aug할 라벨들 경로")
        load_button.clicked.connect(self.load_label_file)
        self.grid_layout.addWidget(load_button,self.aug_count,0,1,3)
        self.aug_count+=1
    def load_label_file(self):
        self.labels_folder = QFileDialog.getExistingDirectory(self)
    
    def make_save_path_button(self):
        save_path_button = QPushButton("Aug된 이미지 저장할 경로")
        save_path_button.clicked.connect(self.load_save_path)
        self.grid_layout.addWidget(save_path_button,self.aug_count,0,1,3)
        self.aug_count+=1
    def load_save_path(self):
        self.save_folder = QFileDialog.getExistingDirectory(self)
        self.save_folder +="/output"
        
    def make_background_images_button(self):
        background_image_path_button = QPushButton("BackGround 이미지 경로")
        background_image_path_button.clicked.connect(self.load_images_folder_for_mixup)
        self.grid_layout.addWidget(background_image_path_button,self.aug_count,0,1,3)
        self.aug_count+=1
    def load_images_folder_for_mixup(self):
        self.images_for_aug = QFileDialog.getExistingDirectory(self)
        
        
        
    def save_log(self):
        log = {
            "images_folder":self.images_folder if self.images_folder else None,
            "labels_folder":self.labels_folder if self.labels_folder else None,
            "save_folder":self.save_folder if self.save_folder else None,
            "images_for_aug":self.images_for_aug if self.images_for_aug else None,
        }
        with open("object_detection/object_detection_log.yaml","w") as f:
            yaml.dump(log,f)
            
    def load_log(self):
        if os.path.isfile("object_detection/object_detection_log.yaml"):
            with open("object_detection/object_detection_log.yaml") as f:
                self.log = yaml.load(f,yaml.FullLoader)
            self.images_folder = self.log["images_folder"]
            self.labels_folder = self.log["labels_folder"]
            self.save_folder = self.log["save_folder"]
            self.images_for_aug = self.log["images_for_aug"]
            
    
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
    