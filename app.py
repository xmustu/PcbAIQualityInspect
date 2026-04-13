import sys
import math
import cv2
import json
import os
import csv
import numpy as np
import threading
import  multiprocessing as mp
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import QMediaPlaylist, QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QGraphicsVideoItem

from GUI.ui_mainwindow import Ui_MainWindow
from GUI.utils import *
from GUI.threads import *


class MainGUI(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainGUI, self).__init__()
        self.setupUi(self)


        # -----------------------------------------------------------------
        # Data labeling tab settings with dl_ prefix
        # -----------------------------------------------------------------
        self.dl_video_root = ''
        self.dl_video_path_list = []
        self.dl_selected_video_path = ''
        self.dl_image_path_list = []

        self.dl_save_extract_all_images_path = ''
        self.dl_selected_image_path = ''
        self.dl_input_img = None
        self.dl_augment_img = None

  
        self.dl_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.dl_player.setVideoOutput(self.dl_videoWidget)
        self.dl_videoWidget.setStyleSheet('background-color: black')

        # Show select image
        self.dl_inputGraphicsScene = QGraphicsScene()
        self.dl_inputGraphicsScene.setBackgroundBrush(Qt.gray)
        self.dl_inputGraphicsView.setScene(self.dl_inputGraphicsScene)
        self.dl_inputGraphicsScene.setSceneRect(QRectF())
        self.dl_inputGraphicsView.fitInView(self.dl_inputGraphicsScene.sceneRect(), Qt.KeepAspectRatio)

        self.dl_augmentGraphicsScene = QGraphicsScene()
        self.dl_augmentGraphicsScene.setBackgroundBrush(Qt.gray)
        self.dl_augmentGraphicsView.setScene(self.dl_augmentGraphicsScene)
        self.dl_augmentGraphicsScene.setSceneRect(QRectF())
        self.dl_augmentGraphicsView.fitInView(self.dl_augmentGraphicsScene.sceneRect(), Qt.KeepAspectRatio)

        # Event and Slot
        self.dl_openFolderpushButton.clicked.connect(self.handle_dl_open_video_folder_button)
        self.dl_playPushButton.clicked.connect(self.handle_dl_play_video_button)
        self.dl_replayPushButton.clicked.connect(self.handle_dl_replay_video_button)
        self.dl_pausePushButton.clicked.connect(self.handle_dl_pause_video_button)
        self.dl_interceptPushButton.clicked.connect(self.handle_dl_intercept_image_button)
        self.dl_savePathPushButton.clicked.connect(self.handle_dl_save_path_button)
        self.dl_interceptAllImagesPushButton.clicked.connect(self.handle_dl_intercept_all_image_button)
        self.dl_deletePushButton.clicked.connect(self.handle_dl_delete_image_button)
        self.dl_augPushButton.clicked.connect(self.handle_dl_aug_image_button)
        self.dl_resetPushButton.clicked.connect(self.handle_dl_reset_image_button)
        self.dl_augAllPushButton.clicked.connect(self.handle_dl_aug_all_button)
        self.dl_resetAllPushButton.clicked.connect(self.handle_dl_reset_all_button)

        self.dl_videoListWidget.itemClicked.connect(self.handle_dl_video_list_widget_item_clicked)
        self.dl_imageListWidget.itemClicked.connect(self.handle_dl_image_list_widget_item_clicked)

        self.dl_extractAllFramesThread = ExtractAllFramesThread()
        self.dl_extractAllFramesThread.signal.connect(self.handle_dl_extractAllFramesThread)

        self.dl_AugmentationAllFramesThread = AugmentationAllFramesThread()
        self.dl_AugmentationAllFramesThread.signal.connect(self.handle_dl_AugmentationAllFramesThread)



        # -----------------------------------------------------------------
        # Model training tab settings with mt_ prefix
        # -----------------------------------------------------------------


        # -----------------------------------------------------------------
        # Model quantize tab settings with mq_ prefix
        # -----------------------------------------------------------------


        # -----------------------------------------------------------------
        # Quality inspect tab settings with qi_ prefix
        # -----------------------------------------------------------------

        self.qi_video_root = ''
        self.qi_video_path_list = []
        self.qi_selected_video_path = ''
        self.fps = -1

        self.device = None
        self.model_settings = None
        
        # cls / seg model pth file path
        self.cls_model_param_path = ''
        self.seg_model_param_path = ''

        self.play_thread = VideoPlayThread()
        self.play_thread.send_img.connect(lambda x: self.qi_show_image(x, self.qi_input_VideoLabel))
        self.play_thread.send_seg.connect(lambda x: self.qi_show_image(x, self.qi_segmentVideoLabellabel))

        self.qi_video_cap = None

        self.qi_openPushButton.clicked.connect(self.handle_qi_openPushButton)
        # open JSON setting file
        self.qi_openSettingpushButton.clicked.connect(self.handle_qi_openSettingpushButton)

        self.qi_loadSegmentModelpushButton.clicked.connect(self.handle_qi_loadSegmentModelpushButton)
        self.qi_loadClassifierModelpushButton.clicked.connect(self.handle_qi_loadClassifierModelpushButton)
        self.qi_startPushButton.clicked.connect(self.handle_qi_startPushButton)
        self.qi_initModelPushButton.clicked.connect(self.handle_qi_initModelPushButton)
        self.qi_endPushButton.clicked.connect(self.handle_qi_endPushButton)
        self.qi_videoListWidget.itemClicked.connect(self.handle_qi_videoListWidget)


        # self.qi_deviceComboBox.currentIndexChanged.connect(self.handle_qi_deviceComboBox)
        self.qi_segmodelcomboBox.currentIndexChanged.connect(self.handle_qi_segmodelcomboBox)
        self.qi_classmodelcomboBox.currentIndexChanged.connect(self.handle_qi_classmodelcomboBox)

        # Init device and models
        if torch.cuda.is_available():
            self.deviceinfolabel.setText('GPU')
            self.device = torch.device('cuda')
        else:
            self.deviceinfolabel.setText('当前设备不支持GPU加速')
            self.device = torch.device('cpu')
        
        

    """
    --------------------------------------------------------------------
    Data labeling tab functions with dl_ prefix
    --------------------------------------------------------------------
    """
    def handle_dl_open_video_folder_button(self):
        print('Open video folder')
        
        self.dl_video_root = str(QFileDialog.getExistingDirectory(self, 'Open video root'))
        if self.dl_video_root == '' or not os.path.exists(self.dl_video_root):
            print('Video root not exist')
            return
        
        # Get all video files in this video root
        self.dl_video_path_list = [f for f in os.listdir(self.dl_video_root) if '.mp4' in f]
        if len(self.dl_video_path_list) == 0:
            print('None video found!')
            return
        
        # Show video file name in list widget
        self.dl_videoListWidget.clear()
        self.dl_imageListWidget.clear()
        self.dl_selected_video_path = ''
        for f in self.dl_video_path_list:
            v_item = QListWidgetItem('{}'.format(f))
            self.dl_videoListWidget.addItem(v_item)

    def handle_dl_video_list_widget_item_clicked(self):
        print('handle_video_list_widget_item_clicked')
        idx = self.dl_videoListWidget.currentRow()
        video_name = self.dl_video_path_list[idx]
        self.dl_selected_video_path = os.path.join(self.dl_video_root, video_name)
        print(self.dl_selected_video_path)
        if not os.path.exists(self.dl_selected_video_path):
            print('{} not exist'.format(self.dl_selected_video_path))
            return
        self.dl_player.setMedia(QMediaContent(QUrl.fromLocalFile(self.dl_selected_video_path)))

    def handle_dl_play_video_button(self):
        print('Play video')
        if self.dl_player.state() == QMediaPlayer.PlayingState:
            self.dl_player.pause()
        else:
            self.dl_player.play()

    def handle_dl_replay_video_button(self):
        print('Replay video')
        pass

    def handle_dl_pause_video_button(self):
        print('Pause video')
        pass

    def handle_dl_intercept_image_button(self):
        print('Intercept image')
        pass

    def handle_dl_save_path_button(self):
        self.dl_save_extract_all_images_path = str(QFileDialog.getExistingDirectory(self, 'Open extract all images path'))
        if self.dl_save_extract_all_images_path == '' or not os.path.exists(self.dl_save_extract_all_images_path):
            print('Video root not exist')
            return

    def handle_dl_intercept_all_image_button(self):
        print('Intercept all images')
        # Get selected video path
        if not os.path.exists(self.dl_selected_video_path):
            print('{} not exist'.format(self.dl_selected_video_path))
            return
        
        # Get time interval (int)
        frame_interval = int(self.dl_frameIntervalLineEdit.text())
        print('Frame interval: {} '.format(frame_interval))

        # Extracts all image from a video
        if not os.path.exists(self.dl_save_extract_all_images_path):
            print('{} not exist'.format(self.dl_save_extract_all_images_path))
            return
        # Extract all images from video
        video_path = self.dl_selected_video_path
        save_path = self.dl_save_extract_all_images_path
        extension = '.mp4'
        is_deduplicated = False
        ssim_threshold = 0.90
        self.dl_extractAllFramesThread.set_params(video_path, frame_interval, save_path, extension, is_deduplicated, ssim_threshold)
        self.dl_extractAllFramesThread.start()

        state = extract_all_images(self.dl_selected_video_path, frame_interval, self.dl_save_extract_all_images_path)
        if state == 'OK':
            print('Extract all images successed!')
        elif state == 'Error':
            print('Extract all images failed!')
        
    def handle_dl_delete_image_button(self):
        print('Delete image')
        if not os.path.exists(self.dl_selected_image_path):
            print('deleted {} not exist'.format(self.dl_selected_image_path))
            return
        os.remove(self.dl_selected_image_path)
        # update gui -- image list
        self.dl_imageListWidget.clear()
        self.dl_image_path_list = [f for f in os.listdir(self.dl_save_extract_all_images_path) if '.png' in f]
        self.dl_selected_image_path = ''
        for f in self.dl_image_path_list:
            v_item = QListWidgetItem('{}'.format(f))
            self.dl_imageListWidget.addItem(v_item)
        
        # update gui -- input / aug view
        self.dl_inputGraphicsScene.clear()
        self.dl_augmentGraphicsScene.clear()

    def handle_dl_extractAllFramesThread(self, state):
        print('handle_extractAllFramesThread done: {}'.format(state))
        self.dl_imageListWidget.clear()
        self.dl_image_path_list = [f for f in os.listdir(self.dl_save_extract_all_images_path) if '.png' in f]
        self.dl_selected_image_path = ''
        for f in self.dl_image_path_list:
            v_item = QListWidgetItem('{}'.format(f))
            self.dl_imageListWidget.addItem(v_item)

    def handle_dl_AugmentationAllFramesThread(self, state):
        print('handle_dl_AugmentationAllFramesThread Done!')

    def handle_dl_image_list_widget_item_clicked(self):
        print('handle_image_list_widget_item_clicked')
        idx = self.dl_imageListWidget.currentRow()
        image_name = self.dl_image_path_list[idx]
        self.dl_selected_image_path = os.path.join(self.dl_save_extract_all_images_path, image_name)
        print(self.dl_selected_image_path)
        if not os.path.exists(self.dl_selected_image_path):
            print('{} not exist'.format(self.dl_selected_image_path))
            return
        # show select image in gui

        self.dl_inputGraphicsScene.clear()
        self.dl_augmentGraphicsScene.clear()
        self.dl_input_img = None
        self.dl_augment_img = None

        self.dl_input_img = cv2.imread(self.dl_selected_image_path)

        qimg = QImage(self.dl_input_img.data, self.dl_input_img.shape[1], self.dl_input_img.shape[0], self.dl_input_img.shape[1]*3, QImage.Format_RGB888).rgbSwapped()


        qimg_pix = QPixmap.fromImage(qimg)
        self.dl_inputGraphicsScene.addPixmap(qimg_pix)
        self.dl_inputGraphicsScene.update()

    def handle_dl_aug_image_button(self):
        print('Aug image')
        self.dl_augmentGraphicsScene.clear()
        if self.dl_input_img is None:
            print('dl_input_img is none')
            return
        
        self.dl_augment_img = do_frame_augmentation(self.dl_input_img)

        qimg = QImage(self.dl_augment_img.data, self.dl_augment_img.shape[1], self.dl_augment_img.shape[0], self.dl_augment_img.shape[1]*3, QImage.Format_RGB888).rgbSwapped()


        qimg_pix = QPixmap.fromImage(qimg)
        self.dl_augmentGraphicsScene.addPixmap(qimg_pix)
        self.dl_augmentGraphicsScene.update()

    def handle_dl_reset_image_button(self):
        print('Reset image')
        self.dl_augmentGraphicsScene.clear()
        self.dl_augment_img = None

    def handle_dl_aug_all_button(self):
        print('Aug all images')
        if not os.path.exists(self.dl_save_extract_all_images_path):
            print('{} not exist'.format(self.dl_save_extract_all_images_path))
            return
        # Augment all iamges
        self.dl_AugmentationAllFramesThread.set_params(all_images_save_path=self.dl_save_extract_all_images_path)
        self.dl_AugmentationAllFramesThread.start()
        
    def handle_dl_reset_all_button(self):
        print('Reset all images')
        pass

    """
    --------------------------------------------------------------------
    Model training tab functions with mt_ prefix
    --------------------------------------------------------------------
    """



    """
    --------------------------------------------------------------------
    Model quantize tab settings with mq_ prefix
    --------------------------------------------------------------------
    """



    """
    --------------------------------------------------------------------
    Quality inspect tab settings with qi_ prefix
    --------------------------------------------------------------------
    """

    def handle_qi_openPushButton(self):
        print('Open video folder')
        
        self.qi_video_root = str(QFileDialog.getExistingDirectory(self, 'Open video root'))
        if self.qi_video_root == '' or not os.path.exists(self.qi_video_root):
            print('Video root not exist')
            return
        
        # Get all video files in this video root
        self.qi_video_path_list = [f for f in os.listdir(self.qi_video_root) if '.mp4' in f]
        if len(self.qi_video_path_list) == 0:
            print('None video found!')
            return
        
        # Show video file name in list widget
        self.qi_videoListWidget.clear()
        self.qi_selected_video_path = ''
        for f in self.qi_video_path_list:
            v_item = QListWidgetItem('{}'.format(f))
            self.qi_videoListWidget.addItem(v_item)
    
    def handle_qi_videoListWidget(self):
        print('handle_video_list_widget_item_clicked')
        idx = self.qi_videoListWidget.currentRow()
        video_name = self.qi_video_path_list[idx]
        self.qi_selected_video_path = os.path.join(self.qi_video_root, video_name)
        print(self.qi_selected_video_path)
        if not os.path.exists(self.qi_selected_video_path):
            print('{} not exist'.format(self.qi_selected_video_path))
            return
        
    def handle_qi_openSettingpushButton(self):
        print('Open model setting json file')
        json_path, _ = QFileDialog.getOpenFileName(None, "Open xml file", QDir.currentPath())
        if not os.path.exists(json_path):
            print('{} not exist')
            return
        if ".json" not in json_path:
            print('not correct json file')
            return
        
        try:
            with open(json_path, 'r') as json_file:
                self.model_settings = json.load(json_file)

            # Reflash segmentation / classifier combo box
            if 'segmentations' in self.model_settings:
                self.qi_segmodelcomboBox.clear()
                for arch in self.model_settings['segmentations']:
                    self.qi_segmodelcomboBox.addItem(arch)
               
            if 'classifiers' in self.model_settings:
                self.qi_classmodelcomboBox.clear()
                for arch in self.model_settings['classifiers']:
                    self.qi_classmodelcomboBox.addItem(arch)

        except Exception as e:
            print(repr(e))

    def handle_qi_loadSegmentModelpushButton(self):
        print('Load segmentation model params')
        self.seg_model_param_path, _ = QFileDialog.getOpenFileName(None, "Open .pth file", QDir.currentPath())
        if not os.path.exists(self.seg_model_param_path):
            print('{} not exist'.format(self.seg_model_param_path))
            return
        if ".pth" not in self.seg_model_param_path:
            print('not correct .pth file')
            return
        print('Load {} file'.format(self.seg_model_param_path))

    def handle_qi_loadClassifierModelpushButton(self):
        print('Load classification model params')
        self.cls_model_param_path, _ = QFileDialog.getOpenFileName(None, "Open .pth file", QDir.currentPath())
        if not os.path.exists(self.cls_model_param_path):
            print('{} not exist'.format(self.cls_model_param_path))
            return
        if ".pth" not in self.cls_model_param_path:
            print('not correct .pth file')
            return
        print('Load {} file'.format(self.cls_model_param_path))
    
    def handle_qi_segmodelcomboBox(self, idx):
        print('Segmentaiton model: {}'.format(self.qi_segmodelcomboBox.currentText()))
        # segment_arch = self.qi_segmodelcomboBox.currentText()
        # print('segment arch: {}'.format(segment_arch))
        # self.segmentation_model = create_model(segment_arch)
        # self.segmentation_model = self.segmentation_model.to(self.device)

    def handle_qi_classmodelcomboBox(self, idx):
         print('Classification model: {}'.format(self.qi_classmodelcomboBox.currentText()))
        # class_arch = self.qi_classmodelcomboBox.currentText()
        # print('class arch: {}'.format(class_arch))
        # self.classifier_model = create_model(class_arch)
        # self.classifier_model = self.classifier_model.to(self.device)

    def handle_qi_initModelPushButton(self):
        print('Start inializate models')

        try:
            segment_arch = self.qi_segmodelcomboBox.currentText()
            class_arch = self.qi_classmodelcomboBox.currentText()
            print('seg arch: {}, cls arch: {}'.format(segment_arch, class_arch))
            self.play_thread.init_model(self.device, segment_arch=segment_arch, class_arch=class_arch, seg_param_path=self.seg_model_param_path, cls_param_path=self.cls_model_param_path)
            print('Finish inializate models')

        except Exception as e:
            print(repr(e))

    def handle_qi_startPushButton(self):
        if not os.path.exists(self.qi_selected_video_path):
            print('{} not exist'.format(self.qi_selected_video_path))
            return
        self.play_thread.set_params(self.qi_selected_video_path)
        self.play_thread.start()

    def handle_qi_endPushButton(self):
        self.play_thread.stop()
    
    def qi_show_image(self, image_src, qlabel):
        try:
            ih, iw, _ = image_src.shape
            w = qlabel.geometry().width()
            h = qlabel.geometry().height()
            # keep original aspect ratio
            if iw/w > ih/h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                image_src_ = cv2.resize(image_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                image_src_ = cv2.resize(image_src, (nw, nh))

            frame = cv2.cvtColor(image_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1], QImage.Format_RGB888)
            qlabel.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print('qi_show image', repr(e))


if __name__ == "__main__":
    os.environ["QT_ENABLE_HIGHDPI_SCALING"]   = "1"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    os.environ["QT_SCALE_FACTOR"]             = "1"
    
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)

    gui = MainGUI()
    gui.show()
    sys.exit(app.exec_())