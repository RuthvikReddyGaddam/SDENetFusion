import torch
import torch.nn as nn
import torch.nn.functional as F

class PMA(nn.Module):
    """
    Pooling by Multihead Attention.

    Input:
        x:          [B, N, C]
        valid_mask: [B, N]
                    True  = valid node
                    False = padded node

    Output:
        [B, num_seeds, C]
    """

    def __init__(
        self,
        embed_dim=128,
        num_heads=4,
        num_seeds=128,
        dropout=0.0,
    ):
        super().__init__()

        self.seed_tokens = nn.Parameter(
        torch.empty(1, num_seeds, embed_dim))

        nn.init.trunc_normal_(
            self.seed_tokens,
            std=0.02,
        )

        self.norm_queries = nn.LayerNorm(embed_dim)
        self.norm_inputs = nn.LayerNorm(embed_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_ffn = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x,
        valid_mask=None,
    ):
        batch_size = x.shape[0]

        seeds = self.seed_tokens.expand(
            batch_size,
            -1,
            -1,
        )

        query = self.norm_queries(seeds)
        key_value = self.norm_inputs(x)

        key_padding_mask = None

        if valid_mask is not None:
            # PyTorch convention:
            # True  = ignore this key
            # False = use this key
            key_padding_mask = ~valid_mask.bool()

        attention_output, _ = self.cross_attention(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        pooled_tokens = seeds + attention_output

        pooled_tokens = pooled_tokens + self.ffn(
            self.norm_ffn(pooled_tokens)
        )

        return pooled_tokens


class MultimodalAttentionLayer(nn.Module):
    """
    Transformer-style joint self-attention over:

        [image tokens | edge tokens | graph tokens]

    All valid tokens can interact with one another.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        feedforward_multiplier: int = 4,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )

        hidden_dim = embed_dim * feedforward_multiplier

        # Pre-normalization for self-attention.
        self.norm1 = nn.LayerNorm(embed_dim)

        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.attention_dropout = nn.Dropout(dropout)

        # Pre-normalization for feed-forward network.
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

        self.feed_forward_dropout = nn.Dropout(dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            tokens:
                [B, N_total, embed_dim]

            key_padding_mask:
                [B, N_total]

                PyTorch convention:
                    False = valid token
                    True  = padded token

        Returns:
            Updated tokens:
                [B, N_total, embed_dim]
        """

        if tokens.ndim != 3:
            raise ValueError(
                "tokens must have shape [B, N, C], "
                f"but received {tokens.shape}."
            )

        if key_padding_mask is not None:
            expected_shape = tokens.shape[:2]

            if key_padding_mask.shape != expected_shape:
                raise ValueError(
                    "key_padding_mask must have shape "
                    f"{expected_shape}, but received "
                    f"{key_padding_mask.shape}."
                )

        # ------------------------------------------------------
        # Multi-head self-attention
        # ------------------------------------------------------
        normalized_tokens = self.norm1(tokens)

        attention_output, _ = self.self_attention(
            query=normalized_tokens,
            key=normalized_tokens,
            value=normalized_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        tokens = tokens + self.attention_dropout(attention_output)

        # ------------------------------------------------------
        # Feed-forward network
        # ------------------------------------------------------
        normalized_tokens = self.norm2(tokens)

        feed_forward_output = self.feed_forward(normalized_tokens)

        tokens = tokens + self.feed_forward_dropout(
            feed_forward_output
        )

        # Keep padded token positions at zero.
        if key_padding_mask is not None:
            tokens = tokens.masked_fill(
                key_padding_mask.unsqueeze(-1),
                0.0,
            )

        return tokens


class FusionBlock(nn.Module):
    """
    Multimodal fusion block for image, edge, and graph features.

    Expected inputs
    ---------------
    image_feature:
        [B, C_image, H, W]

    edge_feature:
        [B, C_edge, H_edge, W_edge]

    graph_feature:
        [B, N_graph, C_graph]

    graph_valid_mask:
        [B, N_graph]

        True  = valid graph node
        False = padded graph node

    Processing
    ----------
    1. Project image, edge, and graph features to embed_dim.
    2. Pool image and edge features into spatial tokens.
    3. Add positional and modality embeddings.
    4. Concatenate all modalities.
    5. Apply joint multimodal self-attention.
    6. Extract the image token section.
    7. Apply image-only self-attention.
    8. Reshape image tokens into a feature map.
    9. Upsample to the original image-feature resolution.
    10. Add a learnable residual connection.
    """

    def __init__(
        self,
        image_in_channels: int,
        edge_in_channels: int,
        graph_in_channels: int = 128,
        embed_dim: int = 128,
        num_heads: int = 4,
        pooled_height: int = 16,
        pooled_width: int = 16,
        num_multimodal_layers: int = 2,
        num_image_layers: int = 1,
        dropout: float = 0.1,
        num_seeds: int = 128
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )

        if pooled_height <= 0 or pooled_width <= 0:
            raise ValueError(
                "pooled_height and pooled_width must be positive."
            )

        self.embed_dim = embed_dim
        self.pooled_height = pooled_height
        self.pooled_width = pooled_width
        self.num_spatial_tokens = pooled_height * pooled_width

        # ------------------------------------------------------
        # Feature projections
        # ------------------------------------------------------
        self.image_projection = nn.Sequential(
            nn.Conv2d(
                image_in_channels,
                embed_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        self.edge_projection = nn.Sequential(
            nn.Conv2d(
                edge_in_channels,
                embed_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        self.graph_projection = nn.Sequential(
            nn.Linear(graph_in_channels, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )

        # ------------------------------------------------------
        # Adaptive pooling
        # ------------------------------------------------------
        self.image_pool = nn.AdaptiveAvgPool2d(
            (pooled_height, pooled_width)
        )

        self.edge_pool = nn.AdaptiveAvgPool2d(
            (pooled_height, pooled_width)
        )

        self.graph_pool = PMA(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_seeds=num_seeds,
            dropout=dropout,
        )
                
        # ------------------------------------------------------
        # Positional embeddings
        # ------------------------------------------------------
        # Image and edge tokens correspond to spatial positions.
        self.image_position_embedding = nn.Parameter(
            torch.zeros(
                1,
                self.num_spatial_tokens,
                embed_dim,
            )
        )

        self.edge_position_embedding = nn.Parameter(
            torch.zeros(
                1,
                self.num_spatial_tokens,
                embed_dim,
            )
        )

        # Graph positional embeddings are intentionally omitted because
        # graph-node ordering is arbitrary.

        # ------------------------------------------------------
        # Modality embeddings
        # ------------------------------------------------------
        self.image_modality_embedding = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        self.edge_modality_embedding = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        self.graph_modality_embedding = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        # ------------------------------------------------------
        # Multimodal attention
        # ------------------------------------------------------
        self.multimodal_attention_layers = nn.ModuleList(
            [
                MultimodalAttentionLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_multimodal_layers)
            ]
        )

        self.multimodal_output_norm = nn.LayerNorm(embed_dim)

        # ------------------------------------------------------
        # Image-only self-attention
        # ------------------------------------------------------
        if num_image_layers > 0:
            image_attention_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                activation="relu",
                batch_first=True,
                norm_first=True,
            )

            self.image_self_attention = nn.TransformerEncoder(
                encoder_layer=image_attention_layer,
                num_layers=num_image_layers,
                norm=nn.LayerNorm(embed_dim),
            )
        else:
            self.image_self_attention = nn.Identity()

        # ------------------------------------------------------
        # Convert fused tokens back to a feature map
        # ------------------------------------------------------
        self.output_projection = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                image_in_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(image_in_channels),
        )

        # Controls how strongly the attention branch affects the
        # original CNN feature map.
        self.residual_scale = nn.Parameter(
            torch.tensor(0.1)
        )

        self.output_activation = nn.ReLU(inplace=True)

        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        nn.init.trunc_normal_(
            self.image_position_embedding,
            std=0.02,
        )

        nn.init.trunc_normal_(
            self.edge_position_embedding,
            std=0.02,
        )

        nn.init.trunc_normal_(
            self.image_modality_embedding,
            std=0.02,
        )

        nn.init.trunc_normal_(
            self.edge_modality_embedding,
            std=0.02,
        )

        nn.init.trunc_normal_(
            self.graph_modality_embedding,
            std=0.02,
        )

    @staticmethod
    def feature_map_to_tokens(
        feature_map: torch.Tensor,
    ) -> torch.Tensor:
        """
        Converts:

            [B, C, H, W]

        into:

            [B, H*W, C]
        """

        if feature_map.ndim != 4:
            raise ValueError(
                "feature_map must have shape [B, C, H, W], "
                f"but received {feature_map.shape}."
            )

        return feature_map.flatten(2).transpose(1, 2)

    def tokens_to_feature_map(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Converts:

            [B, pooled_height*pooled_width, embed_dim]

        into:

            [B, embed_dim, pooled_height, pooled_width]
        """

        batch_size, num_tokens, channels = tokens.shape

        if num_tokens != self.num_spatial_tokens:
            raise ValueError(
                f"Expected {self.num_spatial_tokens} image tokens, "
                f"but received {num_tokens}."
            )

        if channels != self.embed_dim:
            raise ValueError(
                f"Expected token dimension {self.embed_dim}, "
                f"but received {channels}."
            )

        return (
            tokens.transpose(1, 2)
            .reshape(
                batch_size,
                self.embed_dim,
                self.pooled_height,
                self.pooled_width,
            )
        )

    

    def forward(
        self,
        image_feature: torch.Tensor,
        edge_feature: torch.Tensor,
        graph_feature: torch.Tensor,
        graph_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image_feature:
                [B, C_image, H, W]

            edge_feature:
                [B, C_edge, H_edge, W_edge]

            graph_feature:
                [B, N_graph, C_graph]

            graph_valid_mask:
                [B, N_graph]

                True  = valid graph node
                False = padded graph node

        Returns:
            fused_image_feature:
                [B, C_image, H, W]
        """

        if image_feature.ndim != 4:
            raise ValueError(
                "image_feature must have shape [B, C, H, W], "
                f"but received {image_feature.shape}."
            )

        if edge_feature.ndim != 4:
            raise ValueError(
                "edge_feature must have shape [B, C, H, W], "
                f"but received {edge_feature.shape}."
            )

        if graph_feature.ndim != 3:
            raise ValueError(
                "graph_feature must have shape [B, N, C], "
                f"but received {graph_feature.shape}."
            )

        batch_size = image_feature.shape[0]

        if edge_feature.shape[0] != batch_size:
            raise ValueError(
                "Image and edge feature batch sizes do not match."
            )

        if graph_feature.shape[0] != batch_size:
            raise ValueError(
                "Image and graph feature batch sizes do not match."
            )

        if graph_valid_mask.shape != graph_feature.shape[:2]:
            raise ValueError(
                "graph_valid_mask must match the first two graph "
                f"dimensions. Expected {graph_feature.shape[:2]}, "
                f"received {graph_valid_mask.shape}."
            )

        original_height = image_feature.shape[-2]
        original_width = image_feature.shape[-1]

        # ------------------------------------------------------
        # 1. Project feature dimensions
        # ------------------------------------------------------
        projected_image = self.image_projection(image_feature)
        projected_edge = self.edge_projection(edge_feature)
        projected_graph = self.graph_projection(graph_feature)

        graph_valid_mask = graph_valid_mask.to(device=projected_graph.device, dtype=torch.bool)
        

        # ------------------------------------------------------
        # 2. Pool image and edge features
        # ------------------------------------------------------
        pooled_image = self.image_pool(projected_image)
        pooled_edge = self.edge_pool(projected_edge)

        # ------------------------------------------------------
        # 3. Convert spatial feature maps to tokens
        # ------------------------------------------------------
        image_tokens = self.feature_map_to_tokens(pooled_image)
        edge_tokens = self.feature_map_to_tokens(pooled_edge)
        

        # Shapes:
        #
        # image_tokens:
        # [B, pooled_height * pooled_width, embed_dim]
        #
        # edge_tokens:
        # [B, pooled_height * pooled_width, embed_dim]
        #
        # graph_tokens:
        # [B, N_graph, embed_dim]

        # ------------------------------------------------------
        # 4. Add positional embeddings
        # ------------------------------------------------------
        image_tokens = (
            image_tokens
            + self.image_position_embedding
        )

        edge_tokens = (
            edge_tokens
            + self.edge_position_embedding
        )
        
        graph_tokens = self.graph_pool(x=projected_graph,
                                       valid_mask=graph_valid_mask)

        # ------------------------------------------------------
        # 5. Add modality embeddings
        # ------------------------------------------------------
        image_tokens = (
            image_tokens
            + self.image_modality_embedding
        )

        edge_tokens = (
            edge_tokens
            + self.edge_modality_embedding
        )

        graph_tokens = (
            graph_tokens
            + self.graph_modality_embedding
        )

        # ------------------------------------------------------
        # 6. Concatenate modalities
        # ------------------------------------------------------
        combined_tokens = torch.cat(
            [
                image_tokens,
                edge_tokens,
                graph_tokens,
            ],
            dim=1,
        )


        # ------------------------------------------------------
        # 7. Joint multimodal self-attention
        # ------------------------------------------------------
        for attention_layer in self.multimodal_attention_layers:
            combined_tokens = attention_layer(
                tokens=combined_tokens
            )

        combined_tokens = self.multimodal_output_norm(
            combined_tokens
        )

        # ------------------------------------------------------
        # 8. Extract updated image tokens
        # ------------------------------------------------------
        image_token_count = image_tokens.shape[1]

        fused_image_tokens = combined_tokens[
            :,
            :image_token_count,
            :,
        ]

        # ------------------------------------------------------
        # 9. Image-only self-attention
        # ------------------------------------------------------
        fused_image_tokens = self.image_self_attention(
            fused_image_tokens
        )

        # ------------------------------------------------------
        # 10. Convert tokens back into feature map
        # ------------------------------------------------------
        fused_image_map = self.tokens_to_feature_map(
            fused_image_tokens
        )

        # ------------------------------------------------------
        # 11. Upsample to original image-feature resolution
        # ------------------------------------------------------
        fused_image_map = F.interpolate(
            fused_image_map,
            size=(original_height, original_width),
            mode="bilinear",
            align_corners=False,
        )

        # ------------------------------------------------------
        # 12. Project back to original image channels
        # ------------------------------------------------------
        fused_image_map = self.output_projection(
            fused_image_map
        )

        # ------------------------------------------------------
        # 13. Residual fusion
        # ------------------------------------------------------
        output = (
            image_feature
            + self.residual_scale * fused_image_map
        )

        return self.output_activation(output)


if __name__ == "__main__":
    batch_size = 2

    image_feature = torch.randn(
        batch_size,
        128,
        128,
        128,
    )

    edge_feature = torch.randn(
        batch_size,
        128,
        128,
        128,
    )

    graph_feature = torch.randn(
        batch_size,
        1000,
        128,
    )

    # First image has 970 valid nodes.
    # Second image has 1000 valid nodes.
    graph_valid_mask = torch.zeros(
        batch_size,
        1000,
        dtype=torch.bool,
    )

    graph_valid_mask[0, :970] = True
    graph_valid_mask[1, :1000] = True

    fusion_block = FusionBlock(
        image_in_channels=128,
        edge_in_channels=128,
        graph_in_channels=128,
        embed_dim=128,
        num_heads=4,
        pooled_height=16,
        pooled_width=16,
        num_multimodal_layers=2,
        num_image_layers=1,
        dropout=0.1,
    )

    output = fusion_block(
        image_feature=image_feature,
        edge_feature=edge_feature,
        graph_feature=graph_feature,
        graph_valid_mask=graph_valid_mask,
    )

    print("Image feature:", image_feature.shape)
    print("Edge feature:", edge_feature.shape)
    print("Graph feature:", graph_feature.shape)
    print("Graph mask:", graph_valid_mask.shape)
    print("Fused output:", output.shape)