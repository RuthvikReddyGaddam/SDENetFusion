import torch
from torch import nn

from SDENetFusion.ImageEncoder.ImageEncoder import ImageEncoder
from SDENetFusion.EdgeMapEncoder.EdgeMapEncoder import EdgeMapEncoder
from SDENetFusion.GeometricEncoder.GeometricEncoder import GeometricEncoder
from SDENetFusion.Decoder.Decoder import Decoder
from SDENetFusion.Decoder.PredictionHead import PredictionHead


class Model(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        num_segments: int = 1000,
        graph_feature_dim: int = 128,
        num_heads: int = 4,
    ):
        super().__init__()

        encoder_channels = (64, 128, 128, 256)

        self.image_encoder = ImageEncoder(
            in_channels=in_channels,
            out_channels=encoder_channels,
            r=3,
        )

        self.edge_encoder = EdgeMapEncoder(
            in_channels=in_channels,
            out_channels=encoder_channels,
        )

        self.graph_encoder = GeometricEncoder(
            n_segments=num_segments,
            compactness=10,
            feature_dim=graph_feature_dim,
            num_heads=num_heads,
            image_in_channels=encoder_channels,
            gat_dropout=0.0,
        )

        # Decoder returns the final decoder feature map:
        # [B, 64, H, W]
        self.decoder = Decoder(
        num_heads = 4,
        channels = (256,128,128,64),
        skip_channels = (128,128,64),
        edge_channels = (128,128,64),
        graph_channels = 128,
        embed_dim = 128,
        r = (
            16,
            16,
            16,
        ),
        pooled_sizes = (
            (8,8),    # Bottleneck
            (8,8),    # skip_l1
            (16,16),    # skip_l2
            (32,32),    # skip_l3
        ),
        num_multimodal_layers = 1,
        num_image_layers = 1,
        dropout = 0.0,
        )

        # Prediction is applied only here.
        self.prediction_head = PredictionHead(
            in_channels=64,
            classification=False,
            num_classes=num_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:
                Input image:
                [B, 3, H, W]

        Returns:
            Segmentation logits:
                [B, num_classes, H, W]
        """

        if x.ndim != 4:
            raise ValueError(
                "Input must have shape [B, C, H, W], "
                f"but received {x.shape}."
            )

        image_features = self.image_encoder(x)
        edge_features = self.edge_encoder(x)

        graph_outputs = self.graph_encoder(
            x,
            image_features,
        )

        decoder_features = self.decoder(
            image_features=image_features,
            edge_features=edge_features,
            graph_outputs=graph_outputs,
        )

        logits = self.prediction_head(
            decoder_features
        )

        return logits


if __name__ == "__main__":
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = Model(
        in_channels=3,
        num_classes=1,
        num_segments=1000,
        graph_feature_dim=128,
        num_heads=4,
    ).to(device)

    model.eval()

    dummy_input = torch.randn(
        1,
        3,
        256,
        256,
        device=device,
    )

    with torch.no_grad():
        output = model(dummy_input)

    print("=" * 55)
    print("MODEL FORWARD PASS SUCCESSFUL")
    print("=" * 55)
    print("Input shape:", list(dummy_input.shape))
    print("Output shape:", list(output.shape))
    print("Expected shape:", [1, 1, 256, 256])
    from torchinfo import summary

    summary(
        model,
        input_size=(16, 3, 256, 256),
        depth=6,
    )