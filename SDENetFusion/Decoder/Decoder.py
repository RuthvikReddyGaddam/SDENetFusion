from typing import Dict, Tuple

import torch
import torch.nn as nn

from SDENetFusion.Decoder.FusionBlock import FusionBlock
from SDENetFusion.Decoder.DecoderBlock import DecoderBlock
# from Model.Decoder.PredictionHead import PredictionHead


class Decoder(nn.Module):
    """
    Decoder order:

        bottleneck
            ↓
        skip_l1
            ↓
        skip_l2
            ↓
        skip_l3
            ↓
        prediction head
    """

    FEATURE_NAMES = (
        "skip_l1",
        "skip_l2",
        "skip_l3",
    )

    def __init__(
        self,
        num_heads: int = 4,
        channels: Tuple[int, ...] = (
            256,
            128,
            128,
            64,
        ),
        skip_channels: Tuple[int, ...] = (
            128,
            128,
            64,
        ),
        edge_channels: Tuple[int, ...] = (
            128,
            128,
            64,
        ),
        graph_channels: int = 128,
        embed_dim: int = 128,
        r: Tuple[int, ...] = (
            16,
            16,
            16,
        ),
        pooled_sizes: Tuple[Tuple[int, int], ...] = (
            (8, 8),    # Bottleneck
            (8, 8),    # skip_l1
            (8, 8),    # skip_l2
            (8, 8),    # skip_l3
        ),
        num_multimodal_layers: int = 1,
        num_image_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self._validate_configuration(
            channels=channels,
            skip_channels=skip_channels,
            edge_channels=edge_channels,
            r=r,
            pooled_sizes=pooled_sizes,
        )

        # ------------------------------------------------------
        # Bottleneck fusion
        # ------------------------------------------------------
        bottleneck_pool_height, bottleneck_pool_width = (
            pooled_sizes[0]
        )

        self.bottleneck_fusion = FusionBlock(
            image_in_channels=channels[0],
            edge_in_channels=channels[0],
            graph_in_channels=graph_channels,
            embed_dim=embed_dim,
            num_heads=num_heads,
            pooled_height=bottleneck_pool_height,
            pooled_width=bottleneck_pool_width,
            num_multimodal_layers=num_multimodal_layers,
            num_image_layers=num_image_layers,
            dropout=dropout,
        )

        # ------------------------------------------------------
        # Decoder stages
        # ------------------------------------------------------
        decoder_blocks = {}

        for index, feature_name in enumerate(
            self.FEATURE_NAMES
        ):
            pooled_height, pooled_width = (
                pooled_sizes[index + 1]
            )

            decoder_blocks[feature_name] = DecoderBlock(
                skip_channels=skip_channels[index],
                decoder_channels=channels[index],
                out_channels=channels[index + 1],
                edge_channels=edge_channels[index],
                graph_channels=graph_channels,
                r=r[index],
                embed_dim=embed_dim,
                num_heads=num_heads,
                pooled_height=pooled_height,
                pooled_width=pooled_width,
                num_multimodal_layers=(
                    num_multimodal_layers
                ),
                num_image_layers=num_image_layers,
                dropout=dropout,
                k_size=3,
                stride=1,
                padding=1,
            )

        self.decoder_blocks = nn.ModuleDict(
            decoder_blocks
        )

        # ------------------------------------------------------
        # Segmentation prediction head
        # ------------------------------------------------------
        # if use_prediction_head:
        #     self.prediction_head = PredictionHead(
        #         in_channels=channels[-1],
        #         classification=False,
        #         num_classes=num_classes,
        #     )
        # else:
        #     self.prediction_head = nn.Identity()

    @staticmethod
    def _validate_configuration(
        channels: Tuple[int, ...],
        skip_channels: Tuple[int, ...],
        edge_channels: Tuple[int, ...],
        r: Tuple[int, ...],
        pooled_sizes: Tuple[Tuple[int, int], ...],
    ) -> None:
        if len(channels) != 4:
            raise ValueError(
                "channels must contain four values: "
                "(bottleneck, decoder_l1, decoder_l2, decoder_l3)."
            )

        if len(skip_channels) != 3:
            raise ValueError(
                "skip_channels must contain three values for "
                "skip_l1, skip_l2, and skip_l3."
            )

        if len(edge_channels) != 3:
            raise ValueError(
                "edge_channels must contain three values for "
                "skip_l1, skip_l2, and skip_l3."
            )

        if len(r) != 3:
            raise ValueError(
                "r must contain three values, one for each "
                "decoder block."
            )

        if len(pooled_sizes) != 4:
            raise ValueError(
                "pooled_sizes must contain four (height, width) "
                "pairs: bottleneck, skip_l1, skip_l2, skip_l3."
            )

        for pooled_size in pooled_sizes:
            if len(pooled_size) != 2:
                raise ValueError(
                    "Every pooled size must be a "
                    "(height, width) pair."
                )

            if pooled_size[0] <= 0 or pooled_size[1] <= 0:
                raise ValueError(
                    "Pooled height and width must be positive."
                )

    @staticmethod
    def _validate_feature_keys(
        features: Dict[str, torch.Tensor],
        feature_group_name: str,
    ) -> None:
        required_keys = {
            "skip_l1",
            "skip_l2",
            "skip_l3",
            "bottleneck",
        }

        missing_keys = required_keys.difference(
            features.keys()
        )

        if missing_keys:
            raise KeyError(
                f"{feature_group_name} is missing keys: "
                f"{sorted(missing_keys)}."
            )

    def forward(
        self,
        image_features: Dict[str, torch.Tensor],
        edge_features: Dict[str, torch.Tensor],
        graph_outputs: Dict,
    ) -> torch.Tensor:
        """
        Expected graph_outputs structure:

        {
            "graph_features": {
                "skip_l1": [B, N, C],
                "skip_l2": [B, N, C],
                "skip_l3": [B, N, C],
                "bottleneck": [B, N, C],
            },
            "node_mask": [B, N]
        }

        Returns:
            Segmentation logits when use_prediction_head=True.

            Final decoder feature map when
            use_prediction_head=False.
        """

        self._validate_feature_keys(
            image_features,
            "image_features",
        )

        self._validate_feature_keys(
            edge_features,
            "edge_features",
        )

        if "graph_features" not in graph_outputs:
            raise KeyError(
                "graph_outputs must contain 'graph_features'."
            )

        if "node_mask" not in graph_outputs:
            raise KeyError(
                "graph_outputs must contain 'node_mask'."
            )

        graph_features = graph_outputs["graph_features"]
        graph_mask = graph_outputs["node_mask"]

        self._validate_feature_keys(
            graph_features,
            "graph_outputs['graph_features']",
        )

        if graph_mask.ndim != 2:
            raise ValueError(
                "graph_outputs['node_mask'] must have shape "
                f"[B, N], but received {graph_mask.shape}."
            )

        # ------------------------------------------------------
        # 1. Bottleneck fusion
        # ------------------------------------------------------
        x = self.bottleneck_fusion(
            image_feature=image_features["bottleneck"],
            edge_feature=edge_features["bottleneck"],
            graph_feature=graph_features["bottleneck"],
            graph_valid_mask=graph_mask,
        )

        # ------------------------------------------------------
        # 2. Decoder stages
        # ------------------------------------------------------
        for feature_name in self.FEATURE_NAMES:
            x = self.decoder_blocks[feature_name](
                x=x,
                image=image_features[feature_name],
                edge=edge_features[feature_name],
                graph=graph_features[feature_name],
                mask=graph_mask,
            )

        # ------------------------------------------------------
        # 3. Prediction head
        # ------------------------------------------------------
        # output = self.prediction_head(x)

        # return output
        return x


if __name__ == "__main__":
    print("Initializing multi-encoder feature maps...")

    batch_size = 2
    num_nodes = 1000
    graph_channels = 128
    num_classes = 1

    # ----------------------------------------------------------
    # Image encoder features
    # ----------------------------------------------------------
    image_feature_maps = {
        "skip_l3": torch.randn(
            batch_size,
            64,
            256,
            256,
        ),
        "skip_l2": torch.randn(
            batch_size,
            128,
            128,
            128,
        ),
        "skip_l1": torch.randn(
            batch_size,
            128,
            64,
            64,
        ),
        "bottleneck": torch.randn(
            batch_size,
            256,
            32,
            32,
        ),
    }

    # ----------------------------------------------------------
    # Edge encoder features
    # ----------------------------------------------------------
    edge_feature_maps = {
        "skip_l3": torch.randn(
            batch_size,
            64,
            256,
            256,
        ),
        "skip_l2": torch.randn(
            batch_size,
            128,
            128,
            128,
        ),
        "skip_l1": torch.randn(
            batch_size,
            128,
            64,
            64,
        ),
        "bottleneck": torch.randn(
            batch_size,
            256,
            32,
            32,
        ),
    }

    # ----------------------------------------------------------
    # Padded graph features
    # ----------------------------------------------------------
    padded_graph_features = {
        "skip_l3": torch.randn(
            batch_size,
            num_nodes,
            graph_channels,
        ),
        "skip_l2": torch.randn(
            batch_size,
            num_nodes,
            graph_channels,
        ),
        "skip_l1": torch.randn(
            batch_size,
            num_nodes,
            graph_channels,
        ),
        "bottleneck": torch.randn(
            batch_size,
            num_nodes,
            graph_channels,
        ),
    }

    # True = valid node, False = padded node.
    graph_node_mask = torch.zeros(
        batch_size,
        num_nodes,
        dtype=torch.bool,
    )

    # Example: image 1 has 970 valid nodes.
    graph_node_mask[0, :970] = True

    # Example: image 2 has all 1000 nodes.
    graph_node_mask[1, :1000] = True

    graph_outputs = {
        "graph_features": padded_graph_features,
        "node_mask": graph_node_mask,
    }

    decoder_pipeline = Decoder(
        num_heads=4,
        channels=(256, 128, 128, 64),
        skip_channels=(128, 128, 64),
        edge_channels=(128, 128, 64),
        graph_channels=graph_channels,
        embed_dim=128,
        r=(16, 16, 16),
        pooled_sizes=(
            (8, 8),
            (8, 8),
            (8, 8),
            (8, 8),
        ),
        num_multimodal_layers=1,
        num_image_layers=1,
        dropout=0.0
    )

    decoder_pipeline.eval()

    print("Executing decoder forward pass...")

    with torch.no_grad():
        segmentation_output = decoder_pipeline(
            image_features=image_feature_maps,
            edge_features=edge_feature_maps,
            graph_outputs=graph_outputs,
        )

    print("\n" + "=" * 55)
    print("PIPELINE VERIFICATION SUCCESSFUL")
    print("=" * 55)
    print(
        "Segmentation output shape:",
        list(segmentation_output.shape),
    )
    print(
        "Expected shape:",
        [
            batch_size,
            num_classes,
            256,
            256,
        ],
    )