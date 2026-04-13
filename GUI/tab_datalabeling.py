import sys
import math
import cv2
import os
import csv
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import QMediaPlaylist, QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QGraphicsVideoItem

from GUI.ui_mainwindow import Ui_MainWindow
from GUI.utils import *
from GUI.threads import *


class Tab_Datalabel(object):
    dl_video_root = ''
    dl_video_path_list = []
    dl_selected_video_path = ''
    dl_image_path_list = []

    dl_save_extract_all_images_path = ''
    dl_selected_image_path = ''
    dl_input_img = None
    dl_augment_img = None

    dl_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)

    dl_inputGraphicsScene = QGraphicsScene()
    dl_inputGraphicsScene.setBackgroundBrush(Qt.gray)

    dl_augmentGraphicsScene = QGraphicsScene()
    dl_augmentGraphicsScene.setBackgroundBrush(Qt.gray)

    dl_extractAllFramesThread = ExtractAllFramesThread()

    dl_AugmentationAllFramesThread = AugmentationAllFramesThread()