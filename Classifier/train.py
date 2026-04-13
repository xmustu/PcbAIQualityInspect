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
import copy

from argparse import ArgumentParser

from datasets import get_data_loaders


from utils import plot_train_history, evaluate_model
from models.efficientnet import efficientnet_b0, efficientnet_v2_m
from utils import create_model

# from models.impr_efficientnet import efficientnet_b0, efficientnet_v2_m



def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs: int = 50, early_stop_patience: int = 20, save_dir: str = "classifier_model_checkpoints"):
    """Training model"""
    os.makedirs(save_dir, exist_ok=True)

    best_val_acc = 0.0
    best_model = None
    early_stop_counter = 0
    train_history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    } 

    st_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        train_history["train_loss"].append(epoch_train_loss)
        train_history["train_acc"].append(epoch_train_acc)

        # -------------------------- Eval stage --------------------------
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels, _  in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)

                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

        # val performance
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        train_history["val_loss"].append(epoch_val_loss)
        train_history["val_acc"].append(epoch_val_acc)

        scheduler.step(epoch_val_acc)  

        # save best model
        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            best_model = copy.deepcopy(model)
            # best_model_path = os.path.join(save_dir, f"best_model_epoch{epoch + 1}_acc{best_val_acc:.4f}.pth")
            # torch.save(model.state_dict(), best_model_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"Eaily stop step: {early_stop_counter}/{early_stop_patience}")

        # 打印epoch总结
        print(f"Epoch {epoch + 1} - Train loss: {epoch_train_loss:.4f} | accuracy: {epoch_train_acc:.4f} | Val loss: {epoch_val_loss:.4f} | accuracy: {epoch_val_acc:.4f}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}\n")

        # # 触发早停
        # if early_stop_counter >= early_stop_patience:
        #     print(f" 早停触发（{early_stop_patience}轮无提升），训练结束")
        #     break

    # Training time
    total_time = time.time() - st_time
    print(f"Training time: {total_time // 60:.0f}分{total_time % 60:.0f}s | Best val accuracy:{best_val_acc:.4f}")

    if best_model:
        best_model_path = os.path.join(save_dir, f"best_model_epoch{epoch + 1}_acc{best_val_acc:.4f}.pth")
        torch.save(best_model.state_dict(), best_model_path)
    return train_history, best_model_path


def main(args):
    # Set random seeds
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print('GPU info: {} | memory: {:.3f} GB'.format(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory/1024**3))

    # Load train / val datasets
    train_loader, val_loader, inv_label_map, class_weights = get_data_loaders(
        image_dir=args.image_dir,
        json_dir=args.json_dir,
        pad_size=args.pad_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        ng_augment_times=args.ng_augment_times
    )
    print('train len: {}, val len: {}'.format(len(train_loader), len(val_loader)))

    # Create model
    model = create_model(args.arch)
    model = model.to(device)

    # Hyper-params
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.8, patience=3)

    # Training models
    train_history, best_model_path = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epoch,
        save_dir=args.save_dir
    )

    plot_train_history(train_history)

    print("\n" + "=" * 50)
    print(f"Load best model: {best_model_path}")

    # best_model = copy.deepcopy(model)
    best_model = create_model(args)
    best_model = best_model.to(device)
    best_model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    evaluate_model(best_model, val_loader, device, inv_label_map)


def parse_args():
    parser = ArgumentParser(description='Semantic segmentation with pytorch')

    # Data setting
    parser.add_argument('--data_root', type=str, default="./data", help="path of datasets")
    parser.add_argument('--image_dir', type=str, default="./data/image", help="input image_dir")
    parser.add_argument('--json_dir', type=str, default="./data/json", help="label json_dir")
    parser.add_argument('--batch_size', type=int, default=16, help="batch size")
    parser.add_argument('--pad_size', type=int, default=224, help="pad_size")
    parser.add_argument('--ng_augment_times', type=int, default=8, help="ng_augment_times")
    
    
    
    # Model setting
    parser.add_argument('--device', type=str, default="cuda", choices=['cuda', 'cpu'],help="label json")
    parser.add_argument('--arch', type=str, default="EfficientB0", choices=['EfficientB0', 'EfficientV2', 'resnet34','resnet18', 'resnet50'],help="model architectures")
    parser.add_argument('--pretrained', type=bool, default=True, choices=[True, False],help="pretrained")
    parser.add_argument('--num_classes', type=int, default=3, help="num_classes")
    

    # Hyper-params setting 
    parser.add_argument('--epoch', type=int, default=80, help="number of epoch")
    parser.add_argument('--lr', type=float, default=1e-3, help="initial learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="weight_decay")
    parser.add_argument('--num_workers', type=int, default=4, help="num_workers")
    parser.add_argument('--optim', type=str.lower, default='adam', choices=['sgd', 'adam', 'adamw'], help="select optimizer")

    # Others

    parser.add_argument('--save_dir', type=str, default="./pcb_model/pcb_model_aug_checkpoints", help="chekc points save dir")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    start_time = time.time()
    args = parse_args()
    print('[Args]: ', args)

    main(args)

    print('=' * 80)
    print('Training time: {:.2f} s'.format(time.time() - start_time))
    print('=' * 80)

