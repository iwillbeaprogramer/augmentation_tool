from PySide6.QtWidgets import QFrame,QHBoxLayout,QLabel,QGridLayout,QPushButton,QSlider,QFileDialog
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
        self.load_log()
        self.make_select_images_directory_button()
        self.make_load_button()
        self.make_save_path_button()
        self.make_background_images_button()
        
        self.make_run_button()
        
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
            "images_folder":self.images_folder,
            "labels_folder":self.labels_folder,
            "save_folder":self.save_folder,
            "images_for_aug":self.images_for_aug,
        }
        with open("object_detection/object_detection_log.yaml","w") as f:
            yaml.dump(log,f)
    def load_log(self):
        with open("object_detection/object_detection_log.yaml") as f:
            self.log = yaml.load(f,yaml.FullLoader)
        self.images_folder = self.log["images_folder"]
        self.labels_folder = self.log["labels_folder"]
        self.save_folder = self.log["save_folder"]
        self.images_for_aug = self.log["images_for_aug"]