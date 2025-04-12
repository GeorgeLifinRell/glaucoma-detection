import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder3D(nn.Module):
    def __init__(self):
        super(CNNEncoder3D, self).__init__()
        
        # Layer 1
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # Layer 2
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm3d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Layer 3
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.relu3 = nn.ReLU(inplace=True)
        
        # Layer 4
        self.conv4 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm3d(256)
        self.relu4 = nn.ReLU(inplace=True)
        
        # Layer 5
        self.conv5 = nn.Conv3d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm3d(512)
        self.relu5 = nn.ReLU(inplace=True)
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        
        # Fully Connected Layer (outputs a single logit for binary classification)
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc(x)
        return x

def build_model():
    model = CNNEncoder3D()
    return model