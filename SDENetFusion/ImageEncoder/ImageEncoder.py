import torch
import torch.nn as nn
from  SDENetFusion.ImageEncoder.SDEBlock.SDEBlock import SDEBlock

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.convBlock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.convBlock(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        skip = self.conv(x)
        out = self.maxpool(skip)
        return skip, out
        
class ImageEncoder(nn.Module):
    FEATURE_NAMES = ("skip_l3", "skip_l2", "skip_l1", "bottleneck")
    
    def __init__(self, in_channels=3, out_channels=(64, 128, 128, 256), r=3):
        super().__init__()
        self.sde_block = SDEBlock(in_channels=in_channels, dec_in_channels=in_channels, r=r)
        sde_out_channels = self.sde_block.calculate_C_sd()
        
        blocks = {}
        blocks[self.FEATURE_NAMES[0]] = EncoderBlock(in_channels=sde_out_channels, out_channels = out_channels[0])
        
        for index, name in enumerate(self.FEATURE_NAMES[1:], start=1):
            blocks[name] = EncoderBlock(in_channels=out_channels[index-1], out_channels=out_channels[index])
        
        self.image_encoder_blocks = nn.ModuleDict(blocks) 
        
        
    def forward(self, x):
        image_features = {}
        
        x = self.sde_block(x, x)
        
        for name, block in self.image_encoder_blocks.items():
            if name == "bottleneck":
                skip, _ = block(x)
                image_features[name] = skip
            else:
                skip, x = block(x)
                image_features[name] = skip
        return image_features
    
    
if __name__ == "__main__":
    dummy_input = torch.randn(2, 3, 256, 256) 
    
    encoder = ImageEncoder(in_channels=3, out_channels=(64, 128, 128, 256), r=3)
    skips = encoder(dummy_input)
    print("--- Encoder Layer Output Shapes ---")
    print(f"Input Shape:        {dummy_input.shape}")
    print(f"Skip 256 Level:     {skips['skip_l3'].shape}")
    print(f"Skip 128 Level:     {skips['skip_l2'].shape}")
    print(f"Skip 128 Level:     {skips['skip_l1'].shape}")
    print(f"Bottleneck Output:  {skips['bottleneck'].shape}")