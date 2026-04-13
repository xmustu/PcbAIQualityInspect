import os
import json
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from fontTools.ttLib.tables.C_P_A_L_ import Color
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast  # 混合精度训练
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import torchvision.transforms as transforms
from torchvision.transforms import ColorJitter

from models.efficientnet import efficientnet_b0, efficientnet_v2_m, EfficientNet_B0_Weights, EfficientNet_V2_M_Weights

def create_model(arch):

    model = None

    # Classification model
    if arch == 'EfficientV2':
        num_classes = 3
        model = model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)
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
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features

        model.classifier = nn.Sequential(
                    nn.Dropout(p=0.3, inplace=True),
                    nn.Linear(in_features, 128),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(128),
                    nn.Linear(128, num_classes)
                )

    return model

# -------------------------- 4. 模型评估（测试集） --------------------------
def evaluate_model(model, test_loader, device, inv_label_map):
    """评估测试集性能，输出分类报告、混淆矩阵、错误样本"""
    model.eval()
    all_preds, all_labels, all_paths = [], [], []

    num_correct, num_total = 0, 0

    with torch.no_grad():
        for images, labels, paths in tqdm(test_loader, desc="Evaluation:"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            num_total += labels.size(0)
            num_correct += (preds == labels).sum().item()

            # 收集结果（转为CPU格式）
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_paths.extend(paths)

    # Accuracy
    accuracy = num_correct / num_total
    print('Accuracy: {:.4f}'.format(accuracy))

    # 计算核心指标
    test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    class_names = [inv_label_map[i] for i in range(len(inv_label_map))]

    # 打印评估结果
    print("\n" + "=" * 50)
    print("测试集评估结果")
    print("=" * 50)
    print(f"测试准确率：{test_acc:.4f}")
    print("\n分类报告（精确率/召回率/F1）：")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix(all_labels, all_preds),
        annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Lable")
    plt.title("PCB Confusion Matrix")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix/confusion_matrix_aug.png")
    print("\n混淆矩阵已保存至：confusion_matrix_aug.png")

    # # 记录错误样本（便于后续分析）
    # error_samples = [
    #     {
    #         "原始图像路径": paths[i],
    #         "真实标签": inv_label_map[all_labels[i]],
    #         "预测标签": inv_label_map[all_preds[i]]
    #     }
    #     for i in range(len(all_labels)) if all_preds[i] != all_labels[i]
    # ]
    # if error_samples:
    #     with open("error_samples.json", "w", encoding="utf-8") as f:
    #         json.dump(error_samples, f, ensure_ascii=False, indent=2)
    #     print(f"错误样本详情已保存至：error_samples.json（共{len(error_samples)}个）")

    return test_acc



def plot_train_history(history, save_path="results/train_history/train_history_aug.png"):
    """绘制训练/验证损失、准确率曲线"""
    plt.figure(figsize=(12, 5))
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="train loss", linewidth=2)
    plt.plot(history["val_loss"], label="val loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="train acc", linewidth=2)
    plt.plot(history["val_acc"], label="val acc", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    # 保存图片
    # plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n训练历史曲线已保存至:{save_path}")
