import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.efficientnet import EfficientNet_B0_Weights, EfficientNet_V2_M_Weights
from torchvision.models.resnet import ResNet34_Weights, ResNet18_Weights, ResNet50_Weights, ResNet101_Weights




def create_extractor(arch, pretrained):
    """
        Create Feature extractors
    """
    extractor = None
    in_features = 0
    if arch == 'efficientb0':
        # efficientb0
        if pretrained:
            extractor = models.efficientnet_b0(weights=EfficientNet_B0_Weights)
        else:
            extractor = models.efficientnet_b0()

        in_features = extractor.classifier[1].in_features
    elif arch == 'efficientv2m':
        # efficientv2m
        if pretrained:
            extractor = models.efficientnet_v2_m(weights=EfficientNet_V2_M_Weights)
        else:
            extractor = models.efficientnet_v2_m()

        in_features = extractor.classifier[1].in_features


    return extractor, in_features


class PCBClassifier(nn.Module):
    def __init__(self, arch, num_classes=3, pretrained=True):
        super(PCBClassifier, self).__init__()
        self.arch = arch
        self.num_classes = num_classes
        self.pretrained = pretrained

        self.extractor, self.in_features = create_extractor(self.arch, self.pretrained)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(self.in_features, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, num_classes)
        )
    
    def initialize_weights(self):
        pass

    def forward(self, x):
        feat = self.extractor(x)
        feat = feat.flatten()
        pred = self.classifier(feat)
        return pred


def create_model(args):
    model = None
    if args.arch == 'efficientb0':
        # efficientb0
        if args.pretrained:
            model = models.efficientnet_b0(weights=EfficientNet_B0_Weights)
        else:
            model = models.efficientnet_b0()
        in_features = model.classifier[1].in_features

        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, args.num_classes)
        )

    elif args.arch == 'efficientv2m':
        # efficientv2m
        if args.pretrained:
            model = models.efficientnet_v2_m(weights=EfficientNet_V2_M_Weights)
        else:
            model = models.efficientnet_v2_m()

        in_features = model.classifier[1].in_features

        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, args.num_classes)
        )

    elif 'resnet' in args.arch:
        if args.pretrained:
            if args.arch == 'resnet34':
                model = models.resnet34(weights=ResNet34_Weights)
            elif args.arch == 'resnet18':
                model = models.resnet18(weights=ResNet18_Weights)
            elif args.arch == 'resnet50':
                model = models.resnet50(weights=ResNet50_Weights)
        else:
            if args.arch == 'resnet34':
                model = models.resnet34()
            elif args.arch == 'resnet18':
                model = models.resnet18()
            elif args.arch == 'resnet50':
                model = models.resnet50()
        
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, args.num_classes)
        )
   
    return model


