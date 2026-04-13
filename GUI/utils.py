import os
import cv2
import ntpath
import time
from skimage.metrics import structural_similarity
import numpy as np
import torch
import torch.nn as nn

from Segmentation.models.bisenetv2 import BiSeNetV2
from Segmentation.models.bisenetv1 import BiSeNetV1
from Classifier.models.efficientnet import efficientnet_b0, efficientnet_v2_m


def extract_all_images(video_path: str, frame_interval: int, save_path: str, extension: str = '.mp4', is_deduplicated: bool = False, ssim_threshold: float = 0.98):
    """
    Extract all imges from a video with frame interval, formating .png file and saving to save_path folder.
    Return a extract state: OK or Error
    """
    if not os.path.exists(video_path):
        print('video path {} not exist'.format(video_path))
        return 'Error'
    if not os.path.exists(save_path):
        print('save path {} not exist'.format(save_path))
        return 'Error'
    if frame_interval <= 0:
        print('frame interval {} should large than 0'.format(frame_interval))
        return 'Error'
    
    # Get the name of video file
    video_name = ntpath.basename(video_path)
    if video_name == '' or extension not in video_name:
        print('video name {} error'.format(video_name))
        return 'Error'
    video_name = video_name.replace(extension, '') # remove file extension

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print('video {} not opened'.format(video_path))
        return 'Error'
    
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Extract all frames with frame_interval
    current_idx = 1
    is_end = False
    while current_idx <= num_frames:
        # move to the current frame
        video.set(cv2.CAP_PROP_POS_FRAMES, current_idx)
        success, frame = video.read()
        if success:
            p = os.path.join(save_path, '{}_{}.png'.format(video_name, current_idx))
            cv2.imwrite(p, frame)
        else:
            print('frame {} not successly read'.format(current_idx))
        
        if is_end:
            break

        current_idx += frame_interval
        if current_idx > num_frames:
            # save the last frame
            current_idx = num_frames
            is_end = True

    # Remove highly similar images -- sorted image names
    if is_deduplicated:
        image_names = [f for f in os.listdir(save_path) if '.png' in f]
        num_frames = len(image_names)
        print('Total frame num: {}'.format(num_frames))

        # Group imges based on the frame similarity
        frame_groups = []
        st_idx, ed_idx = 0, 0
        while ed_idx <= num_frames-1 and st_idx <= num_frames-1:
            st_img_path = os.path.join(save_path, image_names[st_idx])
            st_img = cv2.imread(st_img_path)
            st_img_gray = cv2.cvtColor(st_img, cv2.COLOR_BGR2GRAY)
            gp_list = []
            ed_idx = st_idx + 1
            is_similar = True
            gp_list.append(image_names[st_idx])
            while is_similar and ed_idx <= num_frames-1:
                # print('st_idx: {}, ed_idx: {}'.format(st_idx, ed_idx))
                ed_img_path = os.path.join(save_path, image_names[ed_idx])
                ed_img = cv2.imread(ed_img_path)
                ed_img_gray = cv2.cvtColor(ed_img, cv2.COLOR_BGR2GRAY)

                score, diff = structural_similarity(st_img_gray, ed_img_gray, win_size=101, full=True)
                # print('score: {}'.format(score))

                if score >= ssim_threshold:
                    gp_list.append(image_names[ed_idx])
                    ed_idx += 1
                    continue
                else:
                    # if not, group end
                    is_similar = False
                    frame_groups.append(gp_list)
                    break
            st_idx = ed_idx
        # print(frame_groups)

        # check all images being correctly group
        total_num = 0
        for gp in frame_groups:
            if len(gp) == 0:
                continue
            total_num += len(gp)
        if num_frames != total_num:
            print('all image are not correctly group, orignal num: {}, after num: {}'.format(num_frames, total_num))
        else:
            print('All images are correctly grouped')

        # Only save the best PSNR image in each group
        if len(frame_groups) > 0:
            for idx in range(len(frame_groups)):
                gp = frame_groups[idx]
                if len(gp) == 0:
                    continue
                else:
                    # only save the first image in each group
                    for gp_idx in range(1, len(gp)):
                        img_name = gp[gp_idx]
                        img_path = os.path.join(save_path, img_name)
                        os.remove(img_path)
  
    return 'OK'


def do_frame_augmentation(input_image: np.array):
    """
    Frame augmentation
    """
    if input_image is None:
        print('input image is none')
        return None
    return input_image


def create_model(arch):
    model = None

    # Segmentaiton model
    if arch == 'BiSeNetV2':
        num_classes = 3
        model = BiSeNetV2(n_classes=num_classes, aux_mode='eval')

    # Classification model
    elif arch == 'EfficientV2':
        num_classes = 3
        model = efficientnet_v2_m(weights=None)
        
        in_features = model.classifier[1].in_features
        
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, num_classes)
        )
    
    elif arch == 'EfficientB0':
        num_classes = 3
        model = efficientnet_b0(weights=None)
        
        in_features = model.classifier[1].in_features

        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, num_classes)
        )

    return model


def safe_torch_load(path, map_location='cpu'):
    """优先用 weights_only=True；不支持则回退。"""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def ensure_num_classes(model: nn.Module, num_classes: int = 3):
    """
    把分类头改成 num_classes。兼容 EfficientNet/EfficientV2、ResNet 风格。
    """
    # EfficientNet / EfficientV2 常见：model.classifier[-1]
    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
        last = model.classifier[-1]
        if isinstance(last, nn.Linear) and last.out_features != num_classes:
            in_f = last.in_features
            model.classifier[-1] = nn.Linear(in_f, num_classes)
            print(f"[cls] reset classifier to nn.Linear({in_f}, {num_classes})")
            return

    # ResNet 风格：model.fc
    if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear) and model.fc.out_features != num_classes:
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
        print(f"[cls] reset fc to nn.Linear({in_f}, {num_classes})")


def load_backbone_only(model: nn.Module, state):
    """
    仅加载与模型形状匹配的参数（自动丢弃分类头等尺寸不一致的层）。
    """
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']

    msd = model.state_dict()
    filt = {k: v for k, v in state.items() if (k in msd and msd[k].shape == v.shape)}
    missing = [k for k in msd.keys() if k not in filt]
    dropped = [k for k in state.keys() if k not in filt]

    msg = model.load_state_dict(filt, strict=False)
    print(f"[cls] loaded backbone only: matched={len(filt)}, missing={len(missing)}, dropped={len(dropped)}")
    if getattr(msg, "missing_keys", None) is not None or getattr(msg, "unexpected_keys", None) is not None:
        print(f"[cls] load_state_dict report -> missing={len(getattr(msg,'missing_keys',[]))}, "
              f"unexpected={len(getattr(msg,'unexpected_keys',[]))}")


def clean_state_dict_for_seg(state):
    """可选：清理 seg ckpt 里的 flops/params 之类的统计 buffer（有就删）。"""
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    drop_tags = ('total_ops', 'total_params')
    return {k: v for k, v in state.items() if not any(tag in k for tag in drop_tags)}

class ToTensor(object):
    '''
    mean and std should be of the channel order 'bgr'
    '''
    def __init__(self, mean=(0, 0, 0), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, im_lb):
        im, lb = im_lb['im'], im_lb['lb']
        im = im.transpose(2, 0, 1).astype(np.float32)
        im = torch.from_numpy(im).div_(255)
        dtype, device = im.dtype, im.device
        mean = torch.as_tensor(self.mean, dtype=dtype, device=device)[:, None, None]
        std = torch.as_tensor(self.std, dtype=dtype, device=device)[:, None, None]
        im = im.sub_(mean).div_(std).clone()
        if not lb is None:
            lb = torch.from_numpy(lb.astype(np.int64).copy()).clone()
        return dict(im=im, lb=lb)


"""
============================================================================================
Unit test
============================================================================================
"""
def test_extract_all_images():
    video_path = "E:/LP/Codes/PCBAIQualityInspect/Videos/20250716_150953.mp4"
    frame_interval = 5
    save_path = "E:/LP/Codes/PCBAIQualityInspect/OriginalImages"

    st_time = time.time()
    state = extract_all_images(video_path, frame_interval, save_path, is_deduplicated=False)
    print('Runing time: {:.4f} s'.format(time.time() - st_time))
    print(state)






if __name__ == "__main__":
    test_extract_all_images()