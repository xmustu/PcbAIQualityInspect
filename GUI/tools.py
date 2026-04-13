# conjunction.py (for PyQt)
# -*- coding: utf-8 -*-

"""
分割 + 分类 的轻量级组合推理脚本（自包含、PyQt可直接调用）。
- 不依赖项目的 lib.* 模块；
- 假设外部传入的 seg_model / cls_model 已经是 CUDA + FP16（.half().cuda().eval()）；
- ROI 固定 224x224；每帧所有 ROI 一次性批量分类（不再分块）。

对外核心函数：
    run_seg_and_classify(image_bgr: np.ndarray,
                         seg_model: nn.Module,
                         cls_model: nn.Module,
                         device: torch.device,
                         half: bool)
      -> Tuple[np.ndarray, Dict[str, List]]

返回：
    seg_label: HxW 的 np.int32（0=背景,1=焊丝,2=焊盘）
    result: {"rectangles": [(x,y,w,h), ...], "labels": ["pre"/"ok"/"ng", ...]}
"""

from pathlib import Path
from typing import List, Tuple, Optional, Dict
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import numpy as np
from typing import Dict, List, Optional

# ---------------- 常量 ----------------
LABELS_STR = ["pre", "ok", "ng"]   # 分类标签（索引 0/1/2）
PAD_SIZE   = 224                   # 固定 ROI 尺寸
_DEFAULT_MEAN_BGR = (0.4441, 0.4441, 0.4416)
_DEFAULT_STD_BGR  = (0.2366, 0.2366, 0.2357)

# 分类侧（ImageNet）均值方差（RGB顺序）
_CLS_MEAN_RGB = (0.485, 0.456, 0.406)
_CLS_STD_RGB  = (0.229, 0.224, 0.225)

# ROI 后处理写死的阈值
_MIN_W = 5
_MIN_H = 5
_MIN_AREA_ABS_RATIO = 0.0002   # 0.02% of frame
_MIN_AREA_ABS_PIX   = 64
_CENTER_RATIO = 0.2            # 中心优先的方形区域边长比例（20%）
_TOPK = 12                     # 最多保留 12 个 ROI

# ---------------- 实用函数 ----------------
def round32(x: int) -> int:
    return (x + 31) // 32 * 32

def _try_load_runtime_mean_std_rgb() -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    尝试从 ./temp/runtime_cfg.py 读取 mean/std（按 BGR 存储），转换为 RGB 返回；
    若失败则使用默认值。
    """
    here = Path(__file__).resolve().parent
    candidate = here / "temp" / "runtime_cfg.py"
    mean_bgr, std_bgr = _DEFAULT_MEAN_BGR, _DEFAULT_STD_BGR
    if candidate.exists():
        try:
            spec_name = "runtime_cfg_module_local"
            import importlib.util
            spec = importlib.util.spec_from_file_location(spec_name, str(candidate))
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            cfg = getattr(module, "cfg", None)
            if isinstance(cfg, dict):
                mean_bgr = tuple(cfg.get("mean", _DEFAULT_MEAN_BGR))
                std_bgr  = tuple(cfg.get("std",  _DEFAULT_STD_BGR))
        except Exception:
            pass
    mean_rgb = (mean_bgr[2], mean_bgr[1], mean_bgr[0])
    std_rgb  = (std_bgr[2],  std_bgr[1],  std_bgr[0])
    return mean_rgb, std_rgb

# 轻量 ToTensor（RGB + 归一化）——仅分割侧使用
class _ToTensorRGB(object):
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std
    def __call__(self, im_rgb_uint8: np.ndarray) -> torch.Tensor:
        x = im_rgb_uint8.astype(np.float32) / 255.0
        x = (x - np.array(self.mean, dtype=np.float32)) / np.array(self.std, dtype=np.float32)
        x = np.transpose(x, (2, 0, 1)).copy()  # (3,H,W)
        return torch.from_numpy(x)

# 在导入阶段准备分割所需的 ToTensor，减少重复创建
_MEAN_RGB, _STD_RGB = _try_load_runtime_mean_std_rgb()
_TO_TENSOR = _ToTensorRGB(mean=_MEAN_RGB, std=_STD_RGB)

# ---------------- 核心子模块 ----------------
@torch.inference_mode()
def _seg_infer_mask(img_bgr: np.ndarray,
                    seg_model: nn.Module,
                    device: torch.device,
                    half: bool) -> np.ndarray:
    """
    语义分割：返回 HxW 的类别索引（0=背景,1=焊丝,2=焊盘）
    假设 seg_model 为 CUDA+FP16；输入也用 FP16 喂入。
    - 仅右/下 pad 到 32 的倍数（不插值），前向后再裁回原图尺寸。
    - 兼容 seg_model 返回 (logits, ...) 或 仅 logits 两种写法。
    """
    if device.type != "cuda":
        raise RuntimeError("seg_infer 期望在 CUDA 上运行（并使用 FP16）。")
    if seg_model.training:
        seg_model.eval()

    H, W = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    im = _TO_TENSOR(img_rgb).unsqueeze(0).to(device=device, dtype=torch.float16 if half else torch.float32)

    # 仅 pad 到 32 的倍数（右、下）
    pad_w = (32 - (W % 32)) % 32
    pad_h = (32 - (H % 32)) % 32
    if pad_w or pad_h:
        im = F.pad(im, (0, pad_w, 0, pad_h), mode='replicate')

    out = seg_model(im)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    logits = logits[:, :, :H, :W]  # 裁回

    pred_idx = logits.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.int32)
    return pred_idx

def _extract_pad_rois(img_bgr: np.ndarray,
                      mask_idx: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    """
    提取焊盘连通域(=2)，裁剪 ROI（不在 CPU resize），交给 GPU 统一插值到 224。
    返回：patches_bgr, rects[(x,y,w,h), ...]
    - 形态学：Open(3x3,1) + Close(5x5,1)
    - 过滤：w<5 或 h<5 丢弃；area < max(64, 0.0002*H*W) 丢弃
    - 中心优先：center_ratio=0.2；若中心有 ROI 则只用中心内的
    - Top-K：最多取 12 个（按面积从大到小）
    """
    H, W = mask_idx.shape[:2]
    bin_pad = (mask_idx == 2).astype(np.uint8) * 255

    # 轻量形态学（O(HW)，不会明显增加计算量）
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bin_pad = cv2.morphologyEx(bin_pad, cv2.MORPH_OPEN,  k_open,  iterations=1)
    bin_pad = cv2.morphologyEx(bin_pad, cv2.MORPH_CLOSE, k_close, iterations=1)

    contours, _ = cv2.findContours(bin_pad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 过滤 + 扩边为近似正方框
    boxes = []   # (x1,y1,x2,y2, area)
    area_thr = max(_MIN_AREA_ABS_PIX, int(_MIN_AREA_ABS_RATIO * H * W))
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < _MIN_W or h < _MIN_H:
            continue
        area = w * h
        if area < area_thr:
            continue
        # 适度扩边，近似正方
        cx, cy = x + w // 2, y + h // 2
        half = int(max(w, h) * 0.6)
        x1, y1 = max(0, cx - half), max(0, cy - half)
        x2, y2 = min(W, cx + half), min(H, cy + half)
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2, area))

    if not boxes:
        return [], []

    # 中心优先（若中心内非空则只用中心内）
    cx1 = int((1.0 - _CENTER_RATIO) * 0.5 * W)
    cy1 = int((1.0 - _CENTER_RATIO) * 0.5 * H)
    cx2, cy2 = W - cx1, H - cy1
    center_boxes = []
    for (x1, y1, x2, y2, area) in boxes:
        bx = (x1 + x2) // 2
        by = (y1 + y2) // 2
        if cx1 <= bx <= cx2 and cy1 <= by <= cy2:
            center_boxes.append((x1, y1, x2, y2, area))
    use_boxes = center_boxes if len(center_boxes) > 0 else boxes

    # Top-K（面积降序）
    use_boxes.sort(key=lambda b: b[4], reverse=True)
    use_boxes = use_boxes[:_TOPK]

    # 裁剪 ROI（不在 CPU resize）
    patches, rects = [], []
    for (x1, y1, x2, y2, _area) in use_boxes:
        roi = img_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        patches.append(roi)
        rects.append((x1, y1, x2 - x1, y2 - y1))
    return patches, rects

# ---------- GPU 单帧批量化预处理（一次 H2D + GPU 归一化/插值） ----------
def _stack_and_pad_uint8_rgb(patches_bgr: List[np.ndarray]) -> Optional[np.ndarray]:
    """
    CPU: 将不同大小的 BGR patch
      -> 转 RGB
      -> 堆叠成 (N, Hmax, Wmax, 3) 的 uint8 大数组，空位用 0 填充
    （仅颜色转换和堆叠，不做归一化/resize）
    """
    if len(patches_bgr) == 0:
        return None
    hs, ws, rgbs = [], [], []
    for p in patches_bgr:
        rgb = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        rgbs.append(rgb)
        hs.append(h); ws.append(w)
    Hmax, Wmax = max(hs), max(ws)
    N = len(rgbs)
    arr = np.zeros((N, Hmax, Wmax, 3), dtype=np.uint8)
    for i, rgb in enumerate(rgbs):
        h, w = rgb.shape[:2]
        arr[i, :h, :w, :] = rgb
    return arr  # NHWC uint8

def _make_mean_std_buffers(device: torch.device, half: bool):
    dtype = torch.float16 if (half and device.type == "cuda") else torch.float32
    mean = torch.tensor(_CLS_MEAN_RGB, device=device, dtype=dtype).view(1, 3, 1, 1)
    std  = torch.tensor(_CLS_STD_RGB,  device=device, dtype=dtype).view(1, 3, 1, 1)
    return mean, std

def _preprocess_patches_for_cls(patches_bgr: List[np.ndarray],
                                device: torch.device,
                                half: bool) -> Optional[torch.Tensor]:
    """
    GPU 批量化预处理：
      CPU：BGR->RGB + 堆叠/零填充 -> (N,Hmax,Wmax,3) uint8
      GPU：一次 H2D -> /255 -> (x-mean)/std -> interpolate(224) -> channels_last
      返回 (N,3,224,224) (half/float)
    """
    if len(patches_bgr) == 0:
        return None

    arr = _stack_and_pad_uint8_rgb(patches_bgr)  # NHWC uint8
    if arr is None:
        return None

    # NHWC->NCHW（uint8）并尽量使用 pinned memory，加速 H2D
    x_cpu = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
    if device.type == "cuda":
        try:
            x_cpu = x_cpu.pin_memory()
        except Exception:
            pass

    x = x_cpu.to(device, non_blocking=True)
    x = x.half() if (half and device.type == "cuda") else x.float()

    mean_buf, std_buf = _make_mean_std_buffers(device, half)
    x = x / 255.0
    x = (x - mean_buf) / std_buf
    x = F.interpolate(x, size=(PAD_SIZE, PAD_SIZE), mode='bilinear', align_corners=False)

    if device.type == "cuda":
        x = x.to(memory_format=torch.channels_last)
    return x

@torch.inference_mode()
def _cls_infer_batch(cls_model: nn.Module,
                     x: Optional[torch.Tensor]) -> List[Tuple[int, float]]:
    """
    分类前向（一次性整批），返回 [(pred_idx, prob), ...]。
    """
    if x is None or x.shape[0] == 0:
        return []
    if cls_model.training:
        cls_model.eval()
    logits = cls_model(x)               # 期望 cls_model 也是 half + CUDA
    probs = torch.softmax(logits, dim=1)
    conf, pred = torch.max(probs, dim=1)
    return [(int(pi), float(ci)) for pi, ci in zip(pred.tolist(), conf.tolist())]

# ---------------- 对外主函数（PyQt 调用） ----------------
@torch.inference_mode()
def run_seg_and_classify(
    image_bgr: np.ndarray,
    seg_model: nn.Module,
    cls_model: nn.Module,
    device: torch.device,
    half: bool,
) -> Tuple[np.ndarray, Dict[str, List]]:
    """
    一次性组合推理：分割 → 提取焊盘 ROI → 整批分类。
    约定：
      - seg_model、cls_model 已经 .half().cuda().eval()；
      - device 为 torch.device('cuda')。
    """
    if device.type != "cuda":
        raise RuntimeError("本函数仅支持在 CUDA 设备上推理。")

    # 1) 分割
    st = time.time()
    seg_label = _seg_infer_mask(image_bgr, seg_model, device, half=half)
    print(" _seg_infer_mask {:.6f}".format(time.time() - st))

    # 2) 提取 ROI（不在 CPU resize）
    st = time.time()
    patches_bgr, rects = _extract_pad_rois(image_bgr, seg_label)
    print(" _extract_pad_rois {:.6f}".format(time.time() - st))
    print(" patches_bgr:", len(patches_bgr))

    # 3) 整批分类（GPU 预处理 + 前向）
    st = time.time()
    labels: List[str] = []
    if len(patches_bgr) > 0:
        x = _preprocess_patches_for_cls(patches_bgr, device, half=half)
        preds = _cls_infer_batch(cls_model, x)
        for idx, _conf in preds:
            labels.append(LABELS_STR[idx] if 0 <= idx < len(LABELS_STR) else "?")
    print(" _preprocess_patches_for_cls {:.6f}".format(time.time() - st))

    result = {
        "rectangles": rects,   # [(x,y,w,h), ...]
        "labels": labels       # ["pre"/"ok"/"ng", ...]
    }
    return seg_label, result




# 颜色定义（BGR）
_COLOR_WIRE_BGR = (0, 0, 255)      # 分割：焊丝=1 -> 红
_COLOR_PAD_BGR  = (0, 255, 0)      # 分割：焊盘=2 -> 绿
_LABEL_COLORS   = {                # 分类标签颜色
    "pre": (0, 165, 255),          # 橙
    "ok":  (0, 255, 0),            # 绿
    "ng":  (0, 0, 255),            # 红
}

def make_new_input_image(
    input_image: np.ndarray,
    seg_label: np.ndarray,
    result: Dict[str, List],
    draw_seg: bool = True,
    alpha_wire: float = 0.45,
    alpha_pad: float = 0.35,
) -> np.ndarray:
    """
    把分割透明膜与分类结果叠加到输入图上，返回 new_input_image。

    参数
    ----
    input_image : BGR uint8, shape(H, W, 3)
    seg_label   : int32/int64, shape(H, W)，值域：0=背景,1=焊丝,2=焊盘
    result      : dict，至少包含：
                  - "rectangles": List[(x,y,w,h), ...]
                  - "labels":     List[str]（"pre"/"ok"/"ng"...）
                  可选：
                  - "probs" / "scores" / "confs": List[float]（与 rectangles 对齐）
    draw_seg    : 是否叠加透明膜（默认 True）
    alpha_wire  : 焊丝通道透明度
    alpha_pad   : 焊盘通道透明度

    返回
    ----
    new_input_image : BGR uint8
    """
    assert input_image.ndim == 3 and input_image.shape[2] == 3, "input_image 必须是 BGR(H,W,3)"
    H, W = seg_label.shape[:2]
    vis = input_image.copy()

    # ---- 1) 叠加分割透明膜 ----
    if draw_seg and seg_label.shape[0] == input_image.shape[0] and seg_label.shape[1] == input_image.shape[1]:
        out = vis.astype(np.float32)
        m1 = (seg_label == 1)
        m2 = (seg_label == 2)
        if m1.any():
            out[m1] = (1 - alpha_wire) * out[m1] + alpha_wire * np.array(_COLOR_WIRE_BGR, dtype=np.float32)
        if m2.any():
            out[m2] = (1 - alpha_pad) * out[m2] + alpha_pad * np.array(_COLOR_PAD_BGR, dtype=np.float32)
        vis = np.clip(out + 0.5, 0, 255).astype(np.uint8)

    # ---- 2) 画分类框 + 文本（类别/概率）----
    rects: List = result.get("rectangles", []) or []
    labels: List = result.get("labels", []) or []

    # 兼容不同字段名的置信度
    probs: Optional[List[float]] = None
    for k in ("probs", "scores", "confs", "confidences"):
        if k in result and isinstance(result[k], list):
            probs = result[k]
            break

    n = min(len(rects), len(labels))
    for i in range(n):
        x, y, w, h = rects[i]
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        name = str(labels[i])
        color = _LABEL_COLORS.get(name, (255, 255, 255))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # 文本：类别 + 概率（若有）
        if probs is not None and i < len(probs) and isinstance(probs[i], (float, int)):
            text = f"{name} {float(probs[i]):.2f}"
        else:
            text = name

        # 文本背景条
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        tx1, ty1 = x1, max(0, y1 - th - 8)      # 尽量画在框上方
        tx2, ty2 = x1 + tw + 8, ty1 + th + 6
        # 若超出顶部，改画到框内
        if ty1 < 0:
            ty1 = y1
            ty2 = y1 + th + 6
        cv2.rectangle(vis, (tx1, ty1), (tx2, ty2), color, -1)
        cv2.putText(vis, text, (tx1 + 3, ty2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2, cv2.LINE_AA)

    return vis