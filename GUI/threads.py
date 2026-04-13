from PyQt5.QtCore import QThread, pyqtSignal
import os
import cv2
import numpy as np
import time
import torch
import torch.nn.functional as F

from GUI.utils import extract_all_images, do_frame_augmentation
from GUI.utils import create_model, ToTensor
from GUI.tools import *
from GUI.utils import *


class ExtractAllFramesThread(QThread):
    signal = pyqtSignal(str)

    video_path = ''
    frame_interval = 10
    save_path = ''
    extension = '.mp4'
    is_deduplicated = False
    ssim_threshold = 0.90

    def __init__(self):
        QThread.__init__(self)

    def set_params(self, video_path, frame_interval, save_path, extension, is_deduplicated, ssim_threshold):
        self.video_path = video_path
        self.frame_interval = frame_interval
        self.save_path = save_path
        self.extension = extension
        self.is_deduplicated = is_deduplicated
        self.ssim_threshold = ssim_threshold
    
    def run(self):
        state = extract_all_images(self.video_path, self.frame_interval, self.save_path, self.extension, self.is_deduplicated, self.ssim_threshold)
        self.signal.emit(state)


class AugmentationAllFramesThread(QThread):
    signal = pyqtSignal(str)

    all_images_save_path = ''

    def __init__(self):
        QThread.__init__(self)

    def set_params(self, all_images_save_path):
        self.all_images_save_path = all_images_save_path
    
    def run(self):
        image_names = [f for f in os.listdir(self.all_images_save_path) if '.png' in f]
        for img_nm in image_names:
            img_p = os.path.join(self.all_images_save_path, img_nm)
            input_img = cv2.imread(img_p)
            aug_img = do_frame_augmentation(input_img)
            cv2.imwrite(img_p, aug_img)
        state = 'OK'
        self.signal.emit(state)


class VideoPlayThread(QThread):
    input_video_path = ''
    video_cap = None
    fps = -1

    device = None
    segmentation_model = None
    classifier_model = None

    send_img = pyqtSignal(np.ndarray)
    send_seg = pyqtSignal(np.ndarray) 
    send_raw = pyqtSignal(np.ndarray)
    send_statistic = pyqtSignal(dict)
    # emit：detecting/pause/stop/finished/error msg
    send_msg = pyqtSignal(str)
    send_percent = pyqtSignal(int)
    send_fps = pyqtSignal(str)

    def __init__(self):
        QThread.__init__(self)

    def set_params(self, input_video_path):
        self.input_video_path = input_video_path
        if self.video_cap:
            self.video_cap.release()
        self.video_cap = cv2.VideoCapture(self.input_video_path)
    
    @torch.no_grad()
    def init_model(self, device, segment_arch, class_arch, seg_param_path, cls_param_path):
        # clear old models in thread
        self.segmentation_model = None
        self.classifier_model = None
        if not os.path.exists(seg_param_path):
            print('seg .pth path {} not exist'.format(seg_param_path))
            return
        if not os.path.exists(cls_param_path):
            print('cls .pth path {} not exist'.format(cls_param_path))
            return
        
        try:
            self.device = device

            # segmentation model
            self.segmentation_model = create_model(segment_arch)
            seg_sd = safe_torch_load(seg_param_path, map_location='cpu')
            seg_sd = clean_state_dict_for_seg(seg_sd)
            self.segmentation_model.load_state_dict(seg_sd, strict=False)
            print("[seg] weights loaded (strict=False)")
            self.segmentation_model.eval().to(self.device)

            # classification model
            self.classifier_model = create_model(class_arch)
            cls_sd = safe_torch_load(cls_param_path, map_location='cpu')
            load_backbone_only(self.classifier_model, cls_sd)
            self.classifier_model.eval().to(self.device)

            if self.segmentation_model is not None and self.classifier_model is not None:
                print('init models finised!')
            else:
                if self.segmentation_model is None:
                    print('seg model is none')
                elif self.classifier_model is None:
                    print('cls model is none')

            nwarmup = 50
            with torch.inference_mode():
                if device.type == 'cuda':
                    x = torch.randn(2, 3, 384, 480, device=device)
                    for _ in range(nwarmup):
                        _ = self.segmentation_model(x)
                    
                    x = torch.randn(2, 3, 224, 224, device=device)
                    for _ in range(nwarmup):
                        _ = self.classifier_model(x)
                    torch.cuda.synchronize()

        except Exception as e:
            print(repr(e))

    def run(self):
        frame_num = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        for _ in range(frame_num):
            ret, frame = self.video_cap.read()
            if ret:
                seg_label, result = run_seg_and_classify(frame, 
                                            seg_model=self.segmentation_model, 
                                            cls_model=self.classifier_model,
                                            device=self.device,
                                            half=False)
                
                # Show image in input / result QLable
                seg_img = make_new_input_image(frame, seg_label, result)
                self.send_img.emit(frame)
                self.send_seg.emit(seg_img)

                # time.sleep(1 / fps) 

