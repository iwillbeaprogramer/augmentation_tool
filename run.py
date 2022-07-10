from PySide6.QtWidgets import QApplication,QMainWindow,QTabWidget
from PySide6.QtGui import QPixmap
import sys
import multiprocessing
from qt_material import apply_stylesheet
import yaml

from instance_segmentation import InstanceSegmentation_Tab
from object_detection.tab import OjbectDetection_Tab


class Main_window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Augmentation")
        self.WIDTH=int(1920*0.8)
        self.HEIGHT=int(1280*0.8)
        self.setFixedSize(self.WIDTH,self.HEIGHT)
        tabs = QTabWidget()
        self.setCentralWidget(tabs)
        self.instance_seg_tab = InstanceSegmentation_Tab(self.WIDTH,self.HEIGHT)
        object_detection_tap = OjbectDetection_Tab()
        tabs.addTab(self.instance_seg_tab,"Instance Segmentation")
        tabs.addTab(object_detection_tap,"Object Detection")
        
    def closeEvent(self, event):
        event.accept()
        sys.exit(app.exec())
        
        
        
if __name__=="__main__":
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    window = Main_window()
    apply_stylesheet(app,theme="dark_teal.xml")
    stylesheet = app.styleSheet()
    # with open("style.css") as style:
    #     app.setStyleSheet(stylesheet + style.read().format(**os.environ))
    window.show()
    app.exec()