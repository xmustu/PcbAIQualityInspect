import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torchvision.transforms as transforms
from torchvision.transforms import ColorJitter

import os
import json
import cv2
import numpy as np
import random
from typing import List, Dict, Tuple, Union
from tqdm import tqdm  # 用于显示处理进度，可选但推荐
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import ColorJitter
from copy import deepcopy


class PadExtractor:
    def __init__(self, pad_size: int = 224):
        """
        初始化焊盘提取器
        :param pad_size: 截取后焊盘图像的固定尺寸（适配模型输入，如224×224）
        :param center_ratio: 中心区域判定比例（图像中心占比，如0.3表示30%范围）
        """
        self.pad_size = pad_size  # 输出焊盘图像尺寸（正方形）
        # 标签映射（后续可根据模型需求调整，默认返回字符串标签）
        self.label_mapping = {"OK": "OK", "NG": "NG", "pre": "pre"}

    def _get_image_center(self, image_shape: Tuple[float, float]) -> Tuple[float, float]:
        """计算图像中心点坐标（H, W）→ (center_y, center_x)"""
        img_h, img_w = image_shape[:2]
        return (img_h // 2, img_w // 2)

    def _is_center_pad(self, pad: Dict, image_center: Tuple[float, float]) -> bool:
        """
        调整后核心逻辑：判断「图像中心点」是否落在「焊盘的圆形区域内」
        判定公式：图像中心点到焊盘中心点的距离 ≤ 焊盘半径 → 图像中心在焊盘内
        :param pad: 焊盘信息（含x=中心x, y=中心y, r=半径）
        :param image_center: 图像中心点（y, x）
        :return: 是否为中央焊盘（包含图像中心点的焊盘）
        """
        # 焊盘中心点（x=水平坐标，y=垂直坐标）
        pad_center_x = pad["x"]
        pad_center_y = pad["y"]
        pad_radius = pad["r"]

        # 图像中心点（y=垂直坐标，x=水平坐标）
        img_center_y, img_center_x = image_center

        # 计算「图像中心点」到「焊盘中心点」的欧氏距离
        distance = np.sqrt(
            (img_center_x - pad_center_x) ** 2 +  # x方向距离（水平）
            (img_center_y - pad_center_y) ** 2  # y方向距离（垂直）
        )

        # 距离 ≤ 焊盘半径 → 图像中心点在焊盘内 → 是中央焊盘
        return distance <= pad_radius

    # 获取焊盘标注
    def _read_pad_annotations(self, json_path: str) -> List[Dict]:
        """
        调整后：解析shapes字段中的hanpan标注（circle类型）
        JSON中circle的points格式：[[x1,y1], [x2,y2]] → 两点构成圆形的直径（对角线）
        计算逻辑：中心点=( (x1+x2)/2, (y1+y2)/2 ), 半径=两点距离/2
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON标注文件不存在：{json_path}")

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON格式错误（{json_path}）：{str(e)}")

        # 提取shapes字段，筛选label=hanpan且shape_type=circle的标注
        shapes = data.get("shapes", [])
        if not isinstance(shapes, list) or len(shapes) == 0:
            raise ValueError(f"JSON文件 {json_path} 中无有效shapes标注")

        valid_pads = []
        for shape in shapes:
            # 仅保留"hanpan"标签且"circle"类型的标注
            if shape.get("label") == "hanpan" and shape.get("shape_type") == "circle":
                points = shape.get("points", [])
                # circle标注需包含2个点（构成直径）
                if len(points) == 2 and all(len(p) == 2 for p in points):
                    x1, y1 = points[0]  # 第一个点坐标
                    x2, y2 = points[1]  # 第二个点坐标（与第一个点构成直径）

                    # 计算圆形焊盘的中心点（x,y）
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    # 计算半径（两点间距离的1/2，即直径/2）
                    radius = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / 2

                    # 过滤半径过小的无效标注（避免异常值）
                    if radius > 1:  # 半径大于1像素才视为有效焊盘
                        valid_pads.append({
                            "x": int(round(center_x)),  # 转为整数像素坐标
                            "y": int(round(center_y)),
                            "r": int(round(radius))  # 焊盘半径
                        })
                else:
                    print(f"警告：跳过无效hanpan标注（points格式错误）：{shape}")
            # 忽略非hanpan标注（如hansi）
            elif shape.get("label") != "hanpan":
                continue

        if len(valid_pads) == 0:
            raise ValueError(f"JSON文件 {json_path} 中无有效hanpan标注（circle类型）")

        return valid_pads

    def _get_flags_label(self, json_path: str) -> str:
        """从JSON文件中提取flags标签（返回OK/NG/pre）"""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        flags = data.get("flags", {})
        # 标签优先级：pre > NG > OK（避免多标签同时为True的歧义）
        if flags.get("pre", False):
            return self.label_mapping["pre"]
        elif flags.get("NG", False):
            return self.label_mapping["NG"]
        elif flags.get("OK", False):
            return self.label_mapping["OK"]
        else:
            raise ValueError(f"JSON文件 {json_path} 的flags字段无有效标签（pre/OK/NG均为False或缺失）")


    def extract_center_pad(self, image_path: str, json_path: str) -> Tuple[np.ndarray, str]:
        """
        核心函数：根据图像路径和JSON路径，截取中央焊盘并获取flags标签
        :param image_path: 原始图像文件路径（如.png/.jpg）
        :param json_path: 对应JSON标注文件路径
        :return: (center_pad_image, flags_label) → 中央焊盘图像（RGB格式，shape=(pad_size, pad_size, 3)）、标签
        """
        # 1. 读取原始图像（OpenCV默认BGR格式，后续转为RGB）
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"原始图像文件不存在：{image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"图像读取失败（可能格式不支持）：{image_path}")

        # 2. 读取焊盘标注和flags标签
        valid_pads = self._read_pad_annotations(json_path)
        # print("valid_pads:",valid_pads)
        flags_label = self._get_flags_label(json_path)

        if len(valid_pads) > 1:
            # 3. 筛选中心区域内的焊盘
            image_center = self._get_image_center(image.shape)
            # print("image_center:",image_center)
            center_pads = [pad for pad in valid_pads if self._is_center_pad(pad, image_center)]
            # print("center_pads:",center_pads)

            #处理无符合条件焊盘的情况
            if len(center_pads) == 0:
                raise ValueError(f"图像 {os.path.basename(image_path)} 无包含中心点的焊盘（所有hanpan均不覆盖图像中心）")

            # 4. 选择距离图像中心最近的焊盘（若有多个中心焊盘）
            def calculate_distance(pad):
                # 焊盘中心点 (pad['x'], pad['y'])，图像中心点 (image_center_y, image_center_x)
                pad_x, pad_y = pad['x'], pad['y']
                img_center_y, img_center_x = image_center
                # 计算欧氏距离（无需开根号，比较平方值结果一致，更高效）
                return (pad_x - img_center_x) ** 2 + (pad_y - img_center_y) ** 2

            # 按距离从小到大排序，取第一个（最近的）焊盘
            target_pad = min(center_pads, key=calculate_distance)
            # print("target_pad:",target_pad)

            # 4. 选择距离图像中心最近的焊盘（若有多个中心焊盘）
            pad_x, pad_y, pad_r = target_pad["x"], target_pad["y"], target_pad["r"]
        else:
            target_pad = valid_pads[0]
            pad_x, pad_y, pad_r = target_pad["x"], target_pad["y"], target_pad["r"]

        # 5. 截取焊盘区域（避免超出图像边界）
        # 计算截取的矩形边界（左、右、上、下）
        img_h, img_w = image.shape[:2]
        x1 = max(0, pad_x - pad_r)  # 左边界（不小于0）
        x2 = min(img_w - 1, pad_x + pad_r)  # 右边界（不大于图像宽度-1）
        y1 = max(0, pad_y - pad_r)  # 上边界（不小于0）
        y2 = min(img_h - 1, pad_y + pad_r)  # 下边界（不大于图像高度-1）

        # 截取焊盘（OpenCV切片格式：[y_start:y_end, x_start:x_end]）
        pad_crop = image[y1:y2 + 1, x1:x2 + 1]  # +1是因为切片左闭右开
        # print("pad_crop:",pad_crop)


        # 6. 调整焊盘尺寸为固定大小（适配模型输入）
        pad_resized = cv2.resize(
            pad_crop,
            (self.pad_size, self.pad_size),
            interpolation=cv2.INTER_LINEAR  # 双线性插值，适合图像缩放
        )

        # 7. 转换为RGB格式（OpenCV默认BGR，模型通常需要RGB）
        pad_rgb = cv2.cvtColor(pad_resized, cv2.COLOR_BGR2RGB)

        return pad_rgb, flags_label

    def batch_extract(self, image_dir: str, json_dir: str) -> List[Dict[str, Union[np.ndarray, str, str]]]:
        """
        批量处理图像文件夹和JSON文件夹，返回所有处理后的结果
        :param image_dir: 原始图像文件夹路径
        :param json_dir: 对应JSON标注文件夹路径
        :return: 处理结果列表，每个元素包含"pad_image"（焊盘图像）、"flags_label"（标签）、"image_path"（原始图像路径）
        """
        # 检查输入文件夹是否存在
        for dir_path in [image_dir, json_dir]:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"文件夹不存在：{dir_path}")

        # 获取所有图像文件（支持.png/.jpg/.jpeg格式）
        image_suffixes = (".png", ".jpg", ".jpeg")
        image_filenames = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(image_suffixes)
        ]

        if len(image_filenames) == 0:
            raise ValueError(f"图像文件夹 {image_dir} 中无有效图像文件（支持格式：{image_suffixes}）")

        # 批量处理每个图像
        batch_results = []
        for filename in tqdm(image_filenames, desc="批量提取焊盘与标签"):
            # 拼接图像和JSON路径（假设图像和JSON文件名前缀一致，如001.png对应001.json）
            image_basename = os.path.splitext(filename)[0]
            image_path = os.path.join(image_dir, filename)
            json_path = os.path.join(json_dir, f"{image_basename}.json")
            # print("image_path:",image_path)
            # print("json_paht:",json_path)

            try:
                # 提取中央焊盘和标签
                pad_image, flags_label = self.extract_center_pad(image_path, json_path)
                # 保存结果
                batch_results.append({
                    "image_path": image_path,  # 原始图像路径（便于追溯）
                    "pad_image": pad_image,  # 截取后的中央焊盘图像（RGB，shape=(pad_size,pad_size,3)）
                    "flags_label": flags_label  # 对应的flags标签（OK/NG/pre）
                })
            except Exception as e:
                # 跳过处理失败的样本，并打印警告
                print(f"警告：样本 {image_basename} 处理失败，原因：{str(e)}")
                continue

        if len(batch_results) == 0:
            raise RuntimeError("所有样本处理失败，请检查数据格式或参数配置")

        print(f"\n批量处理完成：共处理 {len(image_filenames)} 个样本，成功 {len(batch_results)} 个")
        return batch_results

# --------------------------  数据集类（新增，适配PadExtractor输出） --------------------------
class PCBPadDataset(Dataset):
    def __init__(self, processed_data: list, transform=None):
        """
        对接PadExtractor的数据集类
        :param processed_data: PadExtractor.batch_extract()返回的列表，每个元素是{"pad_image": 焊盘图, "flags_label": 标签, "image_path": 原始路径}
        :param transform: 图像预处理/增强变换
        """
        self.processed_data = processed_data
        self.transform = transform

        # 标签映射（pre/OK/NG三分类，与你的任务匹配）
        self.label_map = {"pre": 0, "OK": 1, "NG": 2}
        # 反向映射（评估时解析标签用）
        self.inv_label_map = {v: k for k, v in self.label_map.items()}

        # 校验标签有效性（避免未知标签报错）
        for data in self.processed_data:
            if data["flags_label"] not in self.label_map:
                raise ValueError(f"未知标签：{data['flags_label']}，请在label_map中添加对应映射")

    def __len__(self) -> int:
        # 返回数据集总样本数
        return len(self.processed_data)

    def __getitem__(self, idx: int) -> tuple:
        # 按索引获取单个样本
        data = self.processed_data[idx]

        # 1. 焊盘图像（PadExtractor输出为RGB格式，shape: (224,224,3)）
        pad_image = data["pad_image"]
        # 2. 标签（字符串→整数，适配模型输入）
        label = self.label_map[data["flags_label"]]
        # 3. 原始图像路径（用于后续错误样本追溯）
        image_path = data["image_path"]

        # 4. 图像预处理（如转为Tensor、归一化、增强等）
        if self.transform:
            pad_image = self.transform(pad_image)

        # 返回：处理后的图像、标签、原始路径
        return pad_image, label, image_path


def augment_ng_samples_in_train(train_data, ng_augment_times=5, pad_size=224):
    """
    仅对划分后的训练集中的NG样本做定向增强（核心函数）
    不改动验证集/测试集，不修改原有数据划分逻辑

    Args:
        train_data: 划分后的训练集数据列表（每个元素含"pad_image", "flags_label", "image_path"）
        ng_augment_times: 单个NG样本生成的增强样本数（1个原始→1+ng_augment_times个）
        pad_size: 焊盘图像尺寸（与模型输入匹配）

    Returns:
        augmented_train_data: 增强后的训练集（NG样本扩充，pre/OK样本不变）
    """
    # -------------------------- 1. 拆分训练集中的三类样本 --------------------------
    pre_data = [d for d in train_data if d["flags_label"] == "pre"]
    ok_data = [d for d in train_data if d["flags_label"] == "OK"]
    ng_data = [d for d in train_data if d["flags_label"] == "NG"]

    print(f"[训练集增强前] 类别分布：pre={len(pre_data)} | OK={len(ok_data)} | NG={len(ng_data)}")
    if len(ng_data) == 0:
        raise ValueError("训练集中无NG样本，无需增强！请检查数据划分逻辑")

    # -------------------------- 2. 定义NG样本专属增强策略 --------------------------
    # 针对NG样本的细微缺陷特征，设计更贴合真实场景的增强（比pre/OK更丰富）
    ng_aug_transform = transforms.Compose([
        transforms.ToTensor(),  # 转为Tensor（若原始是PIL图像）
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # 微小形变（模拟焊盘位置/角度差异）
        transforms.RandomAffine(
            degrees=12,  # 小范围旋转（避免缺陷特征失真）
            translate=(0.04, 0.04),  # 轻微平移
            scale=(0.92, 1.08),  # 小幅缩放
            shear=4  # 轻微剪切
        ),
        # 噪声与细节模拟（匹配真实成像缺陷）
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.5)),  # 轻度模糊
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.06)),  # 随机擦除小区域（模拟局部缺陷）
        # 颜色微调（适应不同光照环境）
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        # 翻转（保持对称性，不改变缺陷本质）
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        # 转回PIL图像（与原始pad_image格式保持一致，避免后续处理报错）
        transforms.ToPILImage()
    ])

    # -------------------------- 3. 定向增强NG样本 --------------------------
    augmented_ng = []
    for ng_sample in ng_data:
        # 保留原始NG样本（避免丢失真实缺陷特征）
        augmented_ng.append(deepcopy(ng_sample))

        # 生成ng_augment_times个增强样本
        original_pad_img = ng_sample["pad_image"]
        for i in range(ng_augment_times):
            # 对原始NG焊盘图做增强
            aug_img = ng_aug_transform(original_pad_img)

            # 构建新的增强样本（标记路径，便于后续追溯）
            new_ng_sample = deepcopy(ng_sample)
            new_ng_sample["pad_image"] = aug_img
            new_ng_sample["image_path"] = f"{ng_sample['image_path']}_ng_aug_{i + 1}"  # 增强标识

            augmented_ng.append(new_ng_sample)

    # -------------------------- 4. 重组增强后的训练集 --------------------------
    # pre/OK样本不变，仅替换NG样本为“原始+增强”版本
    augmented_train_data = pre_data + ok_data + augmented_ng
    # 打乱顺序（避免训练时先学pre/OK、后学NG，保证类别随机性）
    random.shuffle(augmented_train_data)

    # 打印增强结果
    final_ng_count = len([d for d in augmented_train_data if d["flags_label"] == "NG"])
    print(f"[训练集增强后] 类别分布：pre={len(pre_data)} | OK={len(ok_data)} | NG={final_ng_count}")
    print(f"[训练集增强后] 总样本数：{len(augmented_train_data)}（新增{final_ng_count - len(ng_data)}个NG增强样本）")

    return augmented_train_data


def get_data_loaders(
        image_dir: str,
        json_dir: str,
        pad_size: int = 224,
        batch_size: int = 16,
        test_size: float = 0,
        val_size: float = 0.2,
        num_workers: int = 4,
        ng_augment_times: int = 8
):
    """生成训练/验证/测试数据加载器"""
    # 1. 用PadExtractor批量处理数据（复用已有逻辑）
    pad_extractor = PadExtractor(pad_size=pad_size)
    print("正在用PadExtractor处理焊盘数据...")
    processed_data = pad_extractor.batch_extract(image_dir=image_dir, json_dir=json_dir)

    # 2. 分层划分数据集（保证标签分布一致，避免数据泄露）
    if test_size == 0.0:
        train_data, val_data = train_test_split(
            processed_data, test_size=val_size, random_state=42,
            stratify=[d["flags_label"] for d in processed_data]
        )
        print(f"数据划分完成：训练集{len(train_data)}个 | 验证集{len(val_data)}个")
    else:
        # 先分训练+验证集 / 测试集
        train_val_data, test_data = train_test_split(
            processed_data, test_size=test_size, random_state=42,
            stratify=[d["flags_label"] for d in processed_data]
        )
        # 再分训练集 / 验证集
        train_data, val_data = train_test_split(
            train_val_data, test_size=val_size, random_state=42,
            stratify=[d["flags_label"] for d in train_val_data]
        )
        print(f"数据划分完成：训练集{len(train_data)}个 | 验证集{len(val_data)}个 | 测试集{len(test_data)}个")

    augmented_train_data = augment_ng_samples_in_train(
        train_data=train_data,
        ng_augment_times=ng_augment_times,  # 1个NG样本生成9个增强样本（共10个）
        pad_size=224
    )

    augmented_val_data = augment_ng_samples_in_train(
        train_data=val_data,
        ng_augment_times=ng_augment_times,  # 1个NG样本生成9个增强样本（共10个）
        pad_size=224
    )


    # 3. 图像预处理变换（训练集增强，验证/测试集仅归一化）
    train_transform = transforms.Compose([
        transforms.ToTensor(),  # HWC→CHW，像素值归一化到[0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 适配预训练模型
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.RandomRotation(degrees=10),  # 随机旋转±10度
        ColorJitter()
    ])
    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. 后续创建Dataset和DataLoader（用增强后的训练集）
    # 训练集：用augmented_train_data（增强后的）
    train_dataset = PCBPadDataset(  # 你原有的Dataset类
        processed_data=augmented_train_data,  # 传入增强后的训练集
        transform=train_transform  # 你的训练集变换
    )
    # 验证集：用val_data（原始划分的，无增强）
    val_dataset = PCBPadDataset(
        processed_data=augmented_val_data,  # 原始验证集
        transform=val_test_transform  # 你的验证集变换
    )

    # 新增：统计训练集每个类别的样本数量（pre:0, OK:1, NG:2）
    class_counts = [0, 0, 0]  # 对应pre、OK、NG的样本数
    for data in train_dataset.processed_data:
        label = train_dataset.label_map[data["flags_label"]]  # 转为整数标签
        class_counts[label] += 1
    print(f"训练集类别分布：pre={class_counts[0]}, OK={class_counts[1]}, NG={class_counts[2]}")

    # 计算类别权重：权重 = 总样本数 / (类别数 * 该类样本数)（常用加权公式）
    total_samples = sum(class_counts)
    class_weights = torch.tensor([
        total_samples / (3 * class_counts[0]),  # pre的权重
        total_samples / (3 * class_counts[1]),  # OK的权重
        total_samples / (3 * class_counts[2])   # NG的权重（样本最少，权重最大）
    ], dtype=torch.float32)


    print(f"计算的类别权重：pre={class_weights[0]:.4f}, OK={class_weights[1]:.4f}, NG={class_weights[2]:.4f}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,drop_last=True
    )
    # test_loader = DataLoader(
    #     test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    # )

    return train_loader, val_loader, train_dataset.inv_label_map, class_weights


if __name__ == "__main__":
    # 1. 配置参数（替换为你的实际路径）
    IMAGE_DIR = r"./data/image"  # 原始图像文件夹
    JSON_DIR = r"./data/json"  # 对应JSON标注文件夹
    TARGET_PAD_SIZE = 224  # 焊盘输出尺寸（适配模型输入）

    # 2. 初始化焊盘提取器
    pad_extractor = PadExtractor(pad_size=TARGET_PAD_SIZE)

    # 3. 批量提取焊盘和flags标签
    try:
        processed_data = pad_extractor.batch_extract(
            image_dir=IMAGE_DIR,
            json_dir=JSON_DIR
        )

        # 4. 示例：查看前3个处理结果（后续可直接传入模型训练）
        print(f"\n查看前3个处理结果：")
        for i, data in enumerate(processed_data[:3]):
            print(f"\n样本 {i + 1}：")
            print(f"  原始图像路径：{data['image_path']}")
            print(f"  flags标签：{data['flags_label']}")
            print(f"  焊盘图像形状：{data['pad_image'].shape}（H, W, C）")
            print(f"  焊盘图像数据类型：{data['pad_image'].dtype}")

    except Exception as e:
        print(f"处理失败：{str(e)}")