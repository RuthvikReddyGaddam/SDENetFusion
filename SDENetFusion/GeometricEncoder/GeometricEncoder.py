import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from SDENetFusion.ImageEncoder.ImageEncoder import ConvBlock
from SDENetFusion.GeometricEncoder.SLIC import superpixel_segmentation


class ImageFeatureConv(nn.Module):
    """
    Projects every CNN feature-map level to the same channel dimension.
    """
    def __init__(
        self,
        in_channels=(64, 128, 128, 256),
        out_channels=128,
    ):
        super().__init__()

        feature_names = ("skip_l3", "skip_l2", "skip_l1", "bottleneck")

        if len(in_channels) != len(feature_names):
            raise ValueError(
                f"in_channels must contain {len(feature_names)} values, "
                f"but received {len(in_channels)}."
            )

        self.projections = nn.ModuleDict(
            {
                name: ConvBlock(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    padding=0,
                )
                for name, input_channels in zip(feature_names, in_channels)
            }
        )

    def forward(
        self,
        image_features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        missing = set(self.projections) - set(image_features)

        if missing:
            raise KeyError(
                f"Missing image feature maps: {sorted(missing)}. "
                f"Available keys: {sorted(image_features)}"
            )

        return {
            name: projection(image_features[name])
            for name, projection in self.projections.items()
        }


class GATBlock(nn.Module):
    """
    GATv2 block that preserves the input feature dimension, allowing
    a residual connection.
    """

    def __init__(
        self,
        feature_dim=128,
        heads=4,
        dropout=0.0,
    ):
        super().__init__()

        if feature_dim % heads != 0:
            raise ValueError(
                f"feature_dim ({feature_dim}) must be divisible by heads ({heads})."
            )

        head_dim = feature_dim // heads

        self.norm = nn.LayerNorm(feature_dim)

        self.gat = GATv2Conv(
            in_channels=feature_dim,
            out_channels=head_dim,
            heads=heads,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
        )

        self.activation = nn.ReLU(inplace=True)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        identity = x
        x = self.norm(x)
        x = self.gat(x, edge_index)
        x = x + identity
        return self.activation(x)


class GeometricEncoder(nn.Module):
    """
    Pipeline
    --------
    1. Create a SLIC superpixel map for every image in the batch.
    2. Build a separate adjacency graph for every superpixel map.
    3. Project each CNN feature map to `feature_dim` channels.
    4. Mean-pool pixels belonging to each superpixel into node features.
    5. Apply one GAT block to every encoder level.
    6. Return padded node features with shape [B, max_nodes, feature_dim].

    Because SLIC can produce a slightly different number of segments for
    each image, outputs are padded to the maximum node count in the batch.
    `node_mask` identifies valid nodes.
    """

    FEATURE_NAMES = ("skip_l3", "skip_l2", "skip_l1", "bottleneck")

    def __init__(
        self,
        n_segments=1000,
        compactness=10,
        feature_dim=128,
        num_heads=4,
        image_in_channels=(64, 128, 128, 256),
        gat_dropout=0.0,
    ):
        super().__init__()

        self.n_segments = n_segments
        self.compactness = compactness
        self.feature_dim = feature_dim

        self.image_feature_conv = ImageFeatureConv(
            in_channels=image_in_channels,
            out_channels=feature_dim,
        )

        self.gat_blocks = nn.ModuleDict(
            {
                name: GATBlock(
                    feature_dim=feature_dim,
                    heads=num_heads,
                    dropout=gat_dropout,
                )
                for name in self.FEATURE_NAMES
            }
        )

    @staticmethod
    def _make_labels_contiguous(
        superpixel_map: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """
        Converts arbitrary SLIC labels into contiguous node IDs 0..N-1.
        """
        _, inverse = torch.unique(
            superpixel_map.reshape(-1),
            sorted=True,
            return_inverse=True,
        )

        contiguous_map = inverse.reshape_as(superpixel_map).long()
        num_nodes = int(contiguous_map.max().item()) + 1

        return contiguous_map, num_nodes

    def get_superpixel_maps(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, list[int]]:
        """
        Args:
            images: [B, C, H, W]

        Returns:
            superpixel_maps: [B, H, W], with labels starting at zero
            node_counts: number of superpixels in each image
        """
        if images.ndim != 4:
            raise ValueError(
                f"images must have shape [B, C, H, W], got {images.shape}."
            )

        maps = []
        node_counts = []

        for image in images.detach().cpu():
            image_np = image.permute(1, 2, 0).contiguous().numpy()

            segments_np = superpixel_segmentation(
                image_np,
                n_segments=self.n_segments,
                compactness=self.compactness
            )

            segments = torch.as_tensor(segments_np, dtype=torch.long)
            # segments, num_nodes = self._make_labels_contiguous(segments)
            num_nodes = int(segments.max().item()) + 1

            maps.append(segments)
            node_counts.append(num_nodes)

        superpixel_maps = torch.stack(maps, dim=0).to(images.device)

        return superpixel_maps, node_counts

    @staticmethod
    def get_edges_from_superpixel_map(
        superpixel_map: torch.Tensor,
    ) -> torch.Tensor:
        """
        Builds an undirected region-adjacency graph for one image.

        Args:
            superpixel_map: [H, W]

        Returns:
            edge_index: [2, E]
        """
        if superpixel_map.ndim != 2:
            raise ValueError(
                "superpixel_map must have shape [H, W], "
                f"got {superpixel_map.shape}."
            )

        # These four comparisons find all 8-connected boundaries without
        # repeatedly comparing both directions.
        neighbor_pairs = (
            (superpixel_map[:, :-1], superpixel_map[:, 1:]),     # right
            (superpixel_map[:-1, :], superpixel_map[1:, :]),     # down
            (superpixel_map[:-1, :-1], superpixel_map[1:, 1:]),  # down-right
            (superpixel_map[:-1, 1:], superpixel_map[1:, :-1]),  # down-left
        )

        edges = []

        for source, target in neighbor_pairs:
            source = source.reshape(-1)
            target = target.reshape(-1)

            boundary_mask = source != target

            if boundary_mask.any():
                source = source[boundary_mask]
                target = target[boundary_mask]

                # Add both directions for an undirected adjacency graph.
                edges.append(torch.stack((source, target), dim=0))
                edges.append(torch.stack((target, source), dim=0))

        if not edges:
            return torch.empty(
                (2, 0),
                dtype=torch.long,
                device=superpixel_map.device,
            )

        return torch.unique(torch.cat(edges, dim=1), dim=1)

    def build_batched_edge_index(
        self,
        superpixel_maps: torch.Tensor,
        node_counts: list[int],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Builds one global PyG edge index while preventing edges between images.
        """
        global_edges = []
        local_edges = []
        node_offset = 0

        for batch_index, num_nodes in enumerate(node_counts):
            edge_index = self.get_edges_from_superpixel_map(
                superpixel_maps[batch_index]
            )

            local_edges.append(edge_index)

            if edge_index.numel() > 0:
                global_edges.append(edge_index + node_offset)

            node_offset += num_nodes

        if global_edges:
            batched_edge_index = torch.cat(global_edges, dim=1)
        else:
            batched_edge_index = torch.empty(
                (2, 0),
                dtype=torch.long,
                device=superpixel_maps.device,
            )

        return batched_edge_index, local_edges

    def create_feature_region_map(
        self,
        feature_height: int,
        feature_width: int,
        image_height: int,
        image_width: int,
        device=None,
    ) -> torch.Tensor:
        """
        Creates an [image_height, image_width] tensor where each region
        corresponding to one feature-map cell has a unique label from 1 to N.

        Example:
            feature map: 32x32
            image map:   256x256

            Each feature cell becomes one labeled 8x8 region.
        """
        if image_height % feature_height != 0:
            raise ValueError("image_height must be divisible by feature_height.")

        if image_width % feature_width != 0:
            raise ValueError("image_width must be divisible by feature_width.")

        region_height = image_height // feature_height
        region_width = image_width // feature_width

        labels = torch.arange(
            0,
            feature_height * feature_width,
            device=device,
            dtype=torch.long,
        ).reshape(feature_height, feature_width)

        region_map = labels.repeat_interleave(
            region_height,
            dim=0,
        ).repeat_interleave(
            region_width,
            dim=1,
        )

        return region_map

    

    def feature_map_to_nodes(
        self,
        feature_map: torch.Tensor,
        superpixel_maps: torch.Tensor,
        node_counts: list[int],
    ) -> list[torch.Tensor]:

        batch_size, channels, feature_h, feature_w = feature_map.shape
        sp_batch_size, image_h, image_w = superpixel_maps.shape

        region_map = self.create_feature_region_map(
            feature_height=feature_h,
            feature_width=feature_w,
            image_height=image_h,
            image_width=image_w,
            device=feature_map.device,
        )

        node_feature_list = []

        for batch_idx, num_nodes in enumerate(node_counts):

            feature = (
                feature_map[batch_idx]
                .permute(1, 2, 0)
                .reshape(feature_h * feature_w, channels)
            )

            region_ids = region_map.reshape(-1).long()

            superpixel_ids = (
                superpixel_maps[batch_idx]
                .reshape(-1)
                .long()
                .to(feature_map.device)
            )

            projected_features = feature[region_ids]

            graph_feature_sum = feature.new_zeros(
                (num_nodes, channels)
            )

            graph_feature_sum.index_add_(
                0,
                superpixel_ids,
                projected_features,
            )

            node_pixel_counts = torch.bincount(
                superpixel_ids,
                minlength=num_nodes,
            ).to(
                device=feature.device,
                dtype=feature.dtype,
            )

            graph_feature = (
                graph_feature_sum
                / node_pixel_counts.clamp_min(1).unsqueeze(1)
            )

            node_feature_list.append(graph_feature)

        return node_feature_list
            
    def pad_node_features(
        self,
        node_features: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pads variable-size node tensors into [B, max_nodes, C].

        Returns:
            padded_features: [B, max_nodes, C]
            node_mask: [B, max_nodes], True for valid nodes
        """
        if not node_features:
            raise ValueError("node_features cannot be empty.")

        batch_size = len(node_features)

        max_nodes = max(
            features.shape[0]
            for features in node_features
        )

        feature_dim = node_features[0].shape[1]

        padded = node_features[0].new_zeros(
            (
                batch_size,
                max_nodes,
                feature_dim,
            )
        )

        node_mask = torch.zeros(
            (
                batch_size,
                max_nodes,
            ),
            dtype=torch.bool,
            device=node_features[0].device,
        )

        for batch_index, features in enumerate(node_features):
            num_nodes = features.shape[0]

            padded[
                batch_index,
                :num_nodes,
            ] = features

            node_mask[
                batch_index,
                :num_nodes,
            ] = True

        return padded, node_mask
    
    def forward(
        self,
        images: torch.Tensor,
        image_features: dict[str, torch.Tensor],
    ) -> dict[str, object]:
        """
        Flow:
            1. Project all CNN hierarchy features to feature_dim.
            2. Convert each spatial hierarchy to superpixel node features.
            3. Average node features across all CNN hierarchies.
            4. Pass the averaged node features through sequential GAT blocks.
            5. Save the output from every GAT block.
            6. Pad outputs to [B, max_nodes, feature_dim].

        Returns:
            {
                "node_features":
                    Averaged multiscale node features before GNN,
                    [B, max_nodes, feature_dim].

                "graph_features":
                    Final GNN output,
                    [B, max_nodes, feature_dim].

                "multiscale_graph_features":
                    Dictionary containing the output from every GAT block:
                    {
                        "gat_0": [B, max_nodes, feature_dim],
                        "gat_1": [B, max_nodes, feature_dim],
                        ...
                    }

                "node_mask":
                    [B, max_nodes].

                "node_counts":
                    list[int].

                "superpixel_maps":
                    [B, H, W].

                "edge_index":
                    [2, total_edges].

                "local_edge_indices":
                    list of [2, E_i] tensors.
            }
        """
        # ---------------------------------------------------------
        # 1. Project all CNN feature maps to the same feature_dim.
        # ---------------------------------------------------------
        projected_features = self.image_feature_conv(image_features)

        # Example:
        #
        # skip_l3:    [B, 64, H, W]       -> [B, D, H, W]
        # skip_l2:    [B, 128, H/2, W/2]  -> [B, D, H/2, W/2]
        # skip_l1:    [B, 128, H/4, W/4]  -> [B, D, H/4, W/4]
        # bottleneck: [B, 256, H/8, W/8]  -> [B, D, H/8, W/8]

        # ---------------------------------------------------------
        # 2. Create superpixel nodes and graph edges.
        # ---------------------------------------------------------
        superpixel_maps, node_counts = self.get_superpixel_maps(images)

        global_edge_index, local_edge_indices = self.build_batched_edge_index(
            superpixel_maps,
            node_counts,
        )

        # Ensure edge indices are on the same device as the features.
        global_edge_index = global_edge_index.to(images.device)

        # ---------------------------------------------------------
        # 3. Convert every hierarchy to node features.
        # ---------------------------------------------------------
        hierarchy_node_features: dict[str, list[torch.Tensor]] = {}

        for feature_name in self.FEATURE_NAMES:
            hierarchy_node_features[feature_name] = self.feature_map_to_nodes(
                feature_map=projected_features[feature_name],
                superpixel_maps=superpixel_maps,
                node_counts=node_counts,
            )

        # Structure:
        #
        # hierarchy_node_features = {
        #     "skip_l1": [
        #         [N_0, D],
        #         [N_1, D],
        #         ...
        #     ],
        #     "skip_l2": [
        #         [N_0, D],
        #         [N_1, D],
        #         ...
        #     ],
        #     ...
        # }

        # ---------------------------------------------------------
        # 4. Average hierarchy node features for each image.
        # ---------------------------------------------------------
        averaged_node_feature_list: list[torch.Tensor] = []

        for batch_idx in range(len(node_counts)):
            # Collect the same image's node features from every hierarchy.
            #
            # Each tensor has shape:
            # [N_i, D]
            current_image_hierarchy_features = [
                hierarchy_node_features[feature_name][batch_idx]
                for feature_name in self.FEATURE_NAMES
            ]

            # Validate that every hierarchy generated the same node layout.
            reference_shape = current_image_hierarchy_features[0].shape

            for feature_name, current_features in zip(
                self.FEATURE_NAMES,
                current_image_hierarchy_features,
            ):
                if current_features.shape != reference_shape:
                    raise ValueError(
                        f"Node feature shape mismatch for image {batch_idx}, "
                        f"hierarchy '{feature_name}'. Expected "
                        f"{reference_shape}, received {current_features.shape}."
                    )

            # [num_hierarchies, N_i, D]
            stacked_node_features = torch.stack(
                current_image_hierarchy_features,
                dim=0,
            )

            # Average across the hierarchy dimension.
            #
            # [num_hierarchies, N_i, D]
            #                  ↓ mean(dim=0)
            #              [N_i, D]
            averaged_node_features = stacked_node_features.mean(dim=0)

            averaged_node_feature_list.append(averaged_node_features)

        # Example:
        #
        # averaged_node_feature_list = [
        #     [N_0, D],
        #     [N_1, D],
        #     ...
        # ]

        # ---------------------------------------------------------
        # 5. Pad averaged node features to B x max_nodes x D.
        # ---------------------------------------------------------
        padded_node_features, node_mask = self.pad_node_features(
            averaged_node_feature_list
        )

        # padded_node_features:
        # [B, max_nodes, D]
        #
        # node_mask:
        # [B, max_nodes]

        # ---------------------------------------------------------
        # 6. Flatten valid nodes for PyG.
        # ---------------------------------------------------------
        flat_node_features = torch.cat(
            averaged_node_feature_list,
            dim=0,
        )

        # Shape:
        # [sum(node_counts), D]
        #
        # PyG sees this as one large disconnected graph. The shifted
        # global_edge_index prevents edges between separate images.

        # ---------------------------------------------------------
        # 7. Pass through sequential GAT blocks.
        # ---------------------------------------------------------
        multiscale_graph_features: dict[str, torch.Tensor] = {}

        for name, block in self.gat_blocks.items():
            flat_node_features = block(
                flat_node_features,
                global_edge_index
            )

            # Restore the GAT output into one tensor per image.
            current_split_features = list(
                torch.split(
                    flat_node_features,
                    node_counts,
                    dim=0,
                )
            )

            # Pad this GAT level to:
            # [B, max_nodes, D]
            current_padded_features, _ = self.pad_node_features(
                current_split_features
            )

            # Save the graph representation produced by this GAT block.
            multiscale_graph_features[
                name
            ] = current_padded_features

        # ---------------------------------------------------------
        # 8. Final GNN representation.
        # ---------------------------------------------------------
        final_split_graph_features = list(
            torch.split(
                flat_node_features,
                node_counts,
                dim=0,
            )
        )

        final_graph_features, _ = self.pad_node_features(
            final_split_graph_features
        )

        return {
            "node_features": padded_node_features,
            "graph_features": multiscale_graph_features,
            "final_graph_features": final_graph_features,
            "node_mask": node_mask,
            "node_counts": node_counts,
            "superpixel_maps": superpixel_maps,
            "edge_index": global_edge_index,
            "local_edge_indices": local_edge_indices,
        }
        
if __name__ == "__main__":
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from PIL import Image
    from skimage.segmentation import mark_boundaries

    from SDENetFusion.ImageEncoder.ImageEncoder import ImageEncoder

    # ============================================================
    # Configuration
    # ============================================================
    image_path = Path(
        r"C:\Users\gadda\Documents\FeatureFusionMoNuSAC"
        r"\SDENet-Fusion\Datasets\Glas\train\img\train_9.bmp"
    )

    patch_size = 256
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # ============================================================
    # 1. Load the complete RGB image
    # ============================================================
    if not image_path.exists():
        raise FileNotFoundError(
            f"Could not find image: {image_path.resolve()}"
        )

    full_image_pil = Image.open(image_path).convert("RGB")

    full_image_np = np.asarray(
        full_image_pil,
        dtype=np.float32,
    ) / 255.0

    image_height, image_width = full_image_np.shape[:2]

    if (
        image_height < patch_size
        or image_width < patch_size
    ):
        raise ValueError(
            f"Image must be at least {patch_size}×{patch_size}. "
            f"Received {image_height}×{image_width}."
        )

    # ============================================================
    # 2. Extract first/top-left 256×256 patch
    # ============================================================
    patch_np = full_image_np[
        0:patch_size,
        0:patch_size,
        :,
    ]

    patch_tensor = (
        torch.from_numpy(patch_np)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
        .to(device=device, dtype=torch.float32)
    )

    print("Full image shape:", full_image_np.shape)
    print("Patch shape:", patch_tensor.shape)
    print(
        "Patch range:",
        float(patch_tensor.min()),
        float(patch_tensor.max()),
    )

    # ============================================================
    # 3. Create image and geometric encoders
    # ============================================================
    image_encoder = ImageEncoder(
        in_channels=3,
        out_channels=(64, 128, 128, 256),
        r=3,
    ).to(device)

    geometric_encoder = GeometricEncoder(
        n_segments=1000,
        compactness=10,
        feature_dim=128,
        num_heads=4,
        image_in_channels=(64, 128, 128, 256),
        gat_dropout=0.0,
    ).to(device)

    image_encoder.eval()
    geometric_encoder.eval()

    # ============================================================
    # 4. Run the patch through the image and graph encoders
    # ============================================================
    with torch.no_grad():
        image_features = image_encoder(patch_tensor)

        graph_outputs = geometric_encoder(
            images=patch_tensor,
            image_features=image_features,
        )

    # ============================================================
    # 5. Print image encoder outputs
    # ============================================================
    print("\n" + "=" * 70)
    print("IMAGE ENCODER OUTPUTS")
    print("=" * 70)

    for name, feature in image_features.items():
        print(
            f"{name:12s}: "
            f"shape={tuple(feature.shape)}, "
            f"min={feature.min().item():.5f}, "
            f"max={feature.max().item():.5f}, "
            f"mean={feature.mean().item():.5f}"
        )

    # ============================================================
    # 6. Print every graph output
    # ============================================================
    print("\n" + "=" * 70)
    print("GEOMETRIC ENCODER OUTPUTS")
    print("=" * 70)

    for key, value in graph_outputs.items():
        if torch.is_tensor(value):
            print(
                f"{key:24s}: "
                f"shape={tuple(value.shape)}, "
                f"dtype={value.dtype}, "
                f"device={value.device}"
            )

        elif isinstance(value, dict):
            print(f"{key:24s}: dictionary")

            for level_name, level_tensor in value.items():
                print(
                    f"    {level_name:16s}: "
                    f"shape={tuple(level_tensor.shape)}, "
                    f"min={level_tensor.min().item():.5f}, "
                    f"max={level_tensor.max().item():.5f}, "
                    f"mean={level_tensor.mean().item():.5f}"
                )

        elif isinstance(value, list):
            print(
                f"{key:24s}: list with "
                f"{len(value)} entries"
            )

            for index, item in enumerate(value):
                if torch.is_tensor(item):
                    print(
                        f"    [{index}]: "
                        f"shape={tuple(item.shape)}"
                    )
                else:
                    print(f"    [{index}]: {item}")

        else:
            print(f"{key:24s}: {value}")

    # ============================================================
    # 7. Extract outputs for the first image
    # ============================================================
    superpixel_map = (
        graph_outputs["superpixel_maps"][0]
        .detach()
        .cpu()
        .numpy()
    )

    node_count = graph_outputs["node_counts"][0]

    node_mask = (
        graph_outputs["node_mask"][0]
        .detach()
        .cpu()
        .numpy()
    )

    local_edge_index = (
        graph_outputs["local_edge_indices"][0]
        .detach()
        .cpu()
        .numpy()
    )

    initial_node_features = (
        graph_outputs["node_features"][0, :node_count]
        .detach()
        .cpu()
    )

    graph_feature_levels = {
        name: tensor[0, :node_count].detach().cpu()
        for name, tensor
        in graph_outputs["graph_features"].items()
    }

    print("\nValid nodes from mask:", int(node_mask.sum()))
    print("Node count:", node_count)
    print("Directed graph edges:", local_edge_index.shape[1])

    # ============================================================
    # 8. Calculate superpixel centroids
    # ============================================================
    centroids = np.zeros(
        (node_count, 2),
        dtype=np.float32,
    )

    for node_id in range(node_count):
        rows, columns = np.where(
            superpixel_map == node_id
        )

        if rows.size == 0:
            continue

        # x coordinate
        centroids[node_id, 0] = columns.mean()

        # y coordinate
        centroids[node_id, 1] = rows.mean()

    # ============================================================
    # 9. Remove reverse duplicates for visualization
    # ============================================================
    undirected_edges = set()

    for source, target in local_edge_index.T:
        source = int(source)
        target = int(target)

        if source == target:
            continue

        undirected_edges.add(
            tuple(sorted((source, target)))
        )

    print("Undirected graph edges:", len(undirected_edges))

    # ============================================================
    # 10. Calculate node degrees
    # ============================================================
    node_degrees = np.zeros(
        node_count,
        dtype=np.int64,
    )

    for source, target in undirected_edges:
        node_degrees[source] += 1
        node_degrees[target] += 1

    print("Minimum node degree:", node_degrees.min())
    print("Maximum node degree:", node_degrees.max())
    print("Average node degree:", node_degrees.mean())

    # ============================================================
    # Helper: map one scalar per node back to image space
    # ============================================================
    def node_values_to_image(
        node_values,
        labels,
    ):
        node_values = np.asarray(node_values)

        return node_values[labels]

    # ============================================================
    # Helper: compute L2 feature magnitude for every node
    # ============================================================
    def node_feature_norms(node_features):
        return torch.linalg.vector_norm(
            node_features,
            ord=2,
            dim=1,
        ).numpy()

    # ============================================================
    # 11. Build visualization maps
    # ============================================================
    initial_node_norms = node_feature_norms(
        initial_node_features
    )

    initial_norm_map = node_values_to_image(
        initial_node_norms,
        superpixel_map,
    )

    degree_map = node_values_to_image(
        node_degrees,
        superpixel_map,
    )

    graph_norm_maps = {}

    for level_name, level_features in graph_feature_levels.items():
        level_norms = node_feature_norms(level_features)

        graph_norm_maps[level_name] = node_values_to_image(
            level_norms,
            superpixel_map,
        )

    # ============================================================
    # 12. Superpixel boundary overlay
    # ============================================================
    boundary_overlay = mark_boundaries(
        patch_np,
        superpixel_map,
        color=(0, 1, 0),
        mode="thick",
    )

    # ============================================================
    # 13. Plot patch, SLIC map, graph and degree map
    # ============================================================
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(16, 16),
    )

    axes[0, 0].imshow(patch_np)
    axes[0, 0].set_title("Top-left 256×256 input patch")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(boundary_overlay)
    axes[0, 1].set_title(
        f"SLIC superpixels: {node_count} nodes"
    )
    axes[0, 1].axis("off")

    axes[1, 0].imshow(boundary_overlay)

    for source, target in undirected_edges:
        source_x, source_y = centroids[source]
        target_x, target_y = centroids[target]

        axes[1, 0].plot(
            [source_x, target_x],
            [source_y, target_y],
            linewidth=0.7,
            alpha=0.55,
            color="blue",
        )

    axes[1, 0].scatter(
        centroids[:, 0],
        centroids[:, 1],
        s=8,
        color="cyan",
        edgecolors="white",
        linewidths=0.25,
    )

    axes[1, 0].set_title(
        f"Superpixel graph\n"
        f"{node_count} nodes, "
        f"{len(undirected_edges)} undirected edges"
    )
    axes[1, 0].axis("off")

    degree_display = axes[1, 1].imshow(
        degree_map,
        cmap="viridis",
    )

    axes[1, 1].set_title(
        "Node degree mapped to superpixels"
    )
    axes[1, 1].axis("off")

    figure.colorbar(
        degree_display,
        ax=axes[1, 1],
        fraction=0.046,
        pad=0.04,
    )

    figure.tight_layout()
    plt.show()

    # ============================================================
    # 14. Plot initial and GAT node-feature magnitudes
    # ============================================================
    number_of_graph_levels = len(graph_norm_maps)
    total_panels = 1 + number_of_graph_levels

    figure, axes = plt.subplots(
        1,
        total_panels,
        figsize=(5 * total_panels, 5),
    )

    if total_panels == 1:
        axes = [axes]

    initial_display = axes[0].imshow(
        initial_norm_map,
        cmap="magma",
    )

    axes[0].set_title(
        "Averaged multiscale node features\nbefore GAT"
    )
    axes[0].axis("off")

    figure.colorbar(
        initial_display,
        ax=axes[0],
        fraction=0.046,
        pad=0.04,
    )

    for axis_index, (
        level_name,
        level_map,
    ) in enumerate(
        graph_norm_maps.items(),
        start=1,
    ):
        display = axes[axis_index].imshow(
            level_map,
            cmap="magma",
        )

        axes[axis_index].set_title(
            f"{level_name}\nGAT node-feature L2 norm"
        )
        axes[axis_index].axis("off")

        figure.colorbar(
            display,
            ax=axes[axis_index],
            fraction=0.046,
            pad=0.04,
        )

    figure.tight_layout()
    plt.show()

    # ============================================================
    # 15. Plot node-feature matrices
    #
    # Each row represents one superpixel.
    # Each column represents one feature channel.
    # ============================================================
    feature_matrices = {
        "before_gat": initial_node_features.numpy(),
        **{
            name: features.numpy()
            for name, features
            in graph_feature_levels.items()
        },
    }

    number_of_matrices = len(feature_matrices)

    figure, axes = plt.subplots(
        number_of_matrices,
        1,
        figsize=(16, 4 * number_of_matrices),
    )

    if number_of_matrices == 1:
        axes = [axes]

    for axis, (
        level_name,
        feature_matrix,
    ) in zip(
        axes,
        feature_matrices.items(),
    ):
        display = axis.imshow(
            feature_matrix,
            aspect="auto",
            cmap="coolwarm",
        )

        axis.set_title(
            f"{level_name}: node-feature matrix "
            f"{feature_matrix.shape}"
        )

        axis.set_xlabel("Feature channel")
        axis.set_ylabel("Superpixel node")

        figure.colorbar(
            display,
            ax=axis,
            fraction=0.02,
            pad=0.02,
        )

    figure.tight_layout()
    plt.show()

    # ============================================================
    # 16. Plot feature distribution statistics
    # ============================================================
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
    )

    level_names = list(feature_matrices.keys())

    feature_means = [
        feature_matrix.mean()
        for feature_matrix in feature_matrices.values()
    ]

    feature_standard_deviations = [
        feature_matrix.std()
        for feature_matrix in feature_matrices.values()
    ]

    axes[0].bar(
        level_names,
        feature_means,
    )

    axes[0].set_title("Mean node-feature activation")
    axes[0].set_ylabel("Mean")
    axes[0].tick_params(
        axis="x",
        rotation=30,
    )

    axes[1].bar(
        level_names,
        feature_standard_deviations,
    )

    axes[1].set_title(
        "Node-feature activation standard deviation"
    )
    axes[1].set_ylabel("Standard deviation")
    axes[1].tick_params(
        axis="x",
        rotation=30,
    )

    figure.tight_layout()
    plt.show()

    # ============================================================
    # 17. Save the primary graph visualization
    # ============================================================
    output_path = Path(
        "first_256_patch_graph_visualization.png"
    )

    figure, axis = plt.subplots(
        figsize=(12, 12),
    )

    axis.imshow(boundary_overlay)

    for source, target in undirected_edges:
        source_x, source_y = centroids[source]
        target_x, target_y = centroids[target]

        axis.plot(
            [source_x, target_x],
            [source_y, target_y],
            linewidth=0.8,
            alpha=0.55,
            color="blue",
        )

    axis.scatter(
        centroids[:, 0],
        centroids[:, 1],
        s=10,
        color="cyan",
        edgecolors="white",
        linewidths=0.3,
    )

    axis.set_title(
        "First 256×256 patch: "
        "superpixels and region-adjacency graph"
    )

    axis.axis("off")
    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        "\nSaved graph visualization to:",
        output_path.resolve(),
    )

    plt.show()