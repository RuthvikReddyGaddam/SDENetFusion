import torch
import torch.nn as nn
from SDENetFusion.EdgeMapEncoder.EdgeMap import color_structure_tensor_edges

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.basic_block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=stride,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )

        if in_channels != out_channels or stride != 1:
            self.identity = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.identity = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.identity(x)

        out = self.basic_block(x)
        out = out + identity
        out = self.relu(out)

        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.basic_block1 = BasicBlock(
            in_channels,
            out_channels,
            stride=stride
        )

        self.basic_block2 = BasicBlock(
            out_channels,
            out_channels,
            stride=1
        )

    def forward(self, x):
        x = self.basic_block1(x)
        x = self.basic_block2(x)
        return x


class EdgeMapEncoder(nn.Module):
    FEATURE_NAMES  = ("skip_l3", "skip_l2", "skip_l1", "bottleneck")
    def __init__(
        self,
        in_channels=3,
        out_channels=(64, 128, 128, 256)
    ):
        super().__init__()

        blocks = {}
        blocks[self.FEATURE_NAMES[0]] = ResidualBlock(
                                            in_channels=in_channels, 
                                            out_channels=out_channels[0],
                                            stride=1
                                        )
                                        
        for index, name in enumerate(self.FEATURE_NAMES[1:], start=1):
            blocks[name] = ResidualBlock(
            in_channels=out_channels[index - 1],
            out_channels=out_channels[index],
            stride=2
        )
        
        self.edge_encoder_blocks = nn.ModuleDict(blocks)

    def get_edge_map(self, x):
        # Save device so we can move the result back
        device = x.device

        # (B, C, H, W) -> (B, H, W, C)
        x = x.detach().cpu().permute(0, 2, 3, 1).numpy()

        edgemap = []

        for img in x:
            _, Jxx, Jyy, Jxy = color_structure_tensor_edges(img)

            edge = torch.stack([
                torch.from_numpy(Jxx),
                torch.from_numpy(Jyy),
                torch.from_numpy(Jxy)
            ], dim=0)  # (3, H, W)

            edgemap.append(edge)

        # (B, 3, H, W)
        edgemap = torch.stack(edgemap, dim=0).to(device)

        return edgemap

    def forward(self, x):
        edge_features = {}
        x = self.get_edge_map(x)

        for name, block in self.edge_encoder_blocks.items():
            
            x = block(x)
            edge_features[name] = x

        return edge_features


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt

    dummy_input = torch.rand(
        2,
        3,
        256,
        256,
        dtype=torch.float32,
    )

    encoder = EdgeMapEncoder(
        in_channels=3,
        out_channels=(64, 128, 128, 256),
    )

    skips = encoder(dummy_input)

    print("---Edge Encoder Layer Output Shapes ---")
    print(f"Input Shape:        {dummy_input.shape}")
    print(f"Skip 256 Level:     {skips['skip_l3'].shape}")
    print(f"Skip 128 Level:     {skips['skip_l2'].shape}")
    print(f"Skip 64 Level:      {skips['skip_l1'].shape}")
    print(f"Bottleneck Output:  {skips['bottleneck'].shape}")

    image_path = (
        r"C:\Users\gadda\Documents\FeatureFusionMoNuSAC\SDENet-Fusion\SDENetFusion\GeometricEncoder\slicSamples\image.png"
    )

    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )

    # OpenCV loads BGR. Convert to RGB to match your dataset pipeline.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_tensor = (
        torch.from_numpy(img)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        / 255.0
    )

    edge_tensor = encoder.get_edge_map(img_tensor)

    edge_numpy = (
        edge_tensor.squeeze(0)
        .detach()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )

    titles = ["Jxx", "Jyy", "Jxy"]

    plt.figure(figsize=(15, 5))

    for index in range(3):
        plt.subplot(1, 3, index + 1)
        plt.imshow(edge_numpy[:, :, index], cmap="gray")
        plt.title(titles[index])
        plt.axis("off")

    plt.tight_layout()
    plt.show()