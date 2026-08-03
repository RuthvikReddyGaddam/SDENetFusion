import torch
import torch.nn as nn

class PredictionHead(nn.Module):
    def __init__(self, in_channels=64, classification=True, num_classes = 4):
        super().__init__()
        self.seg_head = nn.Conv2d(in_channels=in_channels,out_channels=1, kernel_size=1)
        self.classification = classification
        if self.classification:
            self.class_head = nn.Conv2d(in_channels=in_channels,out_channels=num_classes, kernel_size=1)
    def forward(self, x):
        seg_map = self.seg_head(x)
        if self.classification:
            class_map = self.class_head(x)
            return seg_map, class_map
        return seg_map