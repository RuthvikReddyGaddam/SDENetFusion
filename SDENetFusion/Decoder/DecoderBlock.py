import torch
import torch.nn as nn

from SDENetFusion.ImageEncoder.SDEBlock.SDEBlock import SDEBlock
from SDENetFusion.ImageEncoder.ImageEncoder import ConvBlock
from SDENetFusion.Decoder.FusionBlock import FusionBlock


class UpBlock(nn.Module):
    def __init__(
        self,
        skip_channels: int,
        decoder_channels: int,
        out_channels: int,
        r: int,
        k_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()

        self.sde_block = SDEBlock(
            in_channels=skip_channels,
            dec_in_channels=decoder_channels,
            r=r,
            k_size=k_size,
            padding=padding,
            stride=stride,
            input=False,
        )

        self.sde_block_out_channels = (
            self.sde_block.calculate_C_sd()
        )

        self.conv_block = nn.Sequential(
            ConvBlock(
                in_channels=self.sde_block_out_channels,
                out_channels=out_channels,
                kernel_size=k_size,
                stride=stride,
                padding=padding,
            ),
            ConvBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=k_size,
                stride=stride,
                padding=padding,
            ),
        )

    def forward(
        self,
        image: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image:
                CNN skip feature map.

            x:
                Decoder feature map from the previous/deeper stage.

        Returns:
            Upsampled and fused decoder feature map.
        """

        x = self.sde_block(image, x)
        x = self.conv_block(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        skip_channels: int,
        decoder_channels: int,
        out_channels: int,
        edge_channels: int,
        graph_channels: int,
        r: int,
        embed_dim: int,
        num_heads: int,
        pooled_height: int,
        pooled_width: int,
        num_multimodal_layers: int,
        num_image_layers: int,
        dropout: float,
        k_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()

        self.upblock = UpBlock(
            skip_channels=skip_channels,
            decoder_channels=decoder_channels,
            out_channels=out_channels,
            r=r,
            k_size=k_size,
            stride=stride,
            padding=padding,
        )

        self.fusion_block = FusionBlock(
            image_in_channels=out_channels,
            edge_in_channels=edge_channels,
            graph_in_channels=graph_channels,
            embed_dim=embed_dim,
            num_heads=num_heads,
            pooled_height=pooled_height,
            pooled_width=pooled_width,
            num_multimodal_layers=num_multimodal_layers,
            num_image_layers=num_image_layers,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        image: torch.Tensor,
        edge: torch.Tensor,
        graph: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:
                Previous decoder feature:
                [B, C_decoder, H/2, W/2]

            image:
                Image encoder skip feature:
                [B, C_skip, H, W]

            edge:
                Edge encoder feature:
                [B, C_edge, H, W]

            graph:
                Padded graph feature:
                [B, N_graph, C_graph]

            mask:
                Graph valid-node mask:
                [B, N_graph]

                True  = valid graph node
                False = padded graph node

        Returns:
            Fused decoder feature:
                [B, C_out, H, W]
        """

        x = self.upblock(
            image=image,
            x=x,
        )

        x = self.fusion_block(
            image_feature=x,
            edge_feature=edge,
            graph_feature=graph,
            graph_valid_mask=mask,
        )

        return x