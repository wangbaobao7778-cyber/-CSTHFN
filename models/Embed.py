import torch
import torch.nn as nn
from typing import Tuple


class CustomPatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 1, embed_dim: int = 64, roi_mode: str = 'original'):
        super().__init__()
        self.embed_dim = embed_dim
        self.roi_mode = roi_mode

        # Define patch definitions: (h, w, y_start, x_start)
        if roi_mode == 'original':
            self.patch_definitions = [
                (2, 3, 0, 0), (1, 3, 0, 3), (2, 3, 0, 6),
                (3, 3, 2, 0), (4, 3, 1, 3), (3, 3, 2, 6)
            ]
        elif roi_mode == 'full':
            self.patch_definitions = [
                (5, 9, 0, 0)
            ]
        elif roi_mode == 'hemi_4_5':
            self.patch_definitions = [
                (5, 4, 0, 0),
                (5, 5, 0, 4)
            ]
        elif roi_mode == 'hemi_5_4':
            self.patch_definitions = [
                (5, 5, 0, 0),
                (5, 4, 0, 5)
            ]
        elif roi_mode == 'three_columns':
            self.patch_definitions = [
                (5, 3, 0, 0),
                (5, 3, 0, 3),
                (5, 3, 0, 6)
            ]
        elif roi_mode == 'grid_1x3':
            self.patch_definitions = []
            for row in range(5):
                for col_idx in range(3):
                    self.patch_definitions.append((1, 3, row, col_idx * 3))
        else:
            raise ValueError(f"Unknown roi_mode: {roi_mode}")

        self.num_patches = len(self.patch_definitions)

        self.patch_projs = nn.ModuleList()
        for h, w, _, _ in self.patch_definitions:
            self.patch_projs.append(nn.Conv2d(in_channels, embed_dim, kernel_size=(h, w)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch_embeddings = []
        for i, (h, w, y_start, x_start) in enumerate(self.patch_definitions):
            image_patch = x[..., y_start: y_start + h, x_start: x_start + w]
            patch_proj_output = self.patch_projs[i](image_patch)
            flattened_patch = patch_proj_output.flatten(2).transpose(1, 2)
            patch_embeddings.append(flattened_patch)

        return torch.cat(patch_embeddings, dim=1)


class VideoPatchEmbeddingWrapper(nn.Module):
    """
    Adapter: Converts 5D video data to 3D sequential data for TimeSformer.
    Includes dimensionality reshaping, spatial embedding, and time embedding.
    """

    def __init__(self, patch_embed_module, num_frames, embed_dim):
        super().__init__()
        self.patch_embed = patch_embed_module
        self.num_frames = num_frames
        self.embed_dim = embed_dim

        self.time_embed = nn.Parameter(torch.zeros(1, num_frames, 1, embed_dim))
        nn.init.trunc_normal_(self.time_embed, std=0.02)

    def forward(self, x):
        """
        Input x: [Batch_Size, T, C, H, W]
        """
        B, T, C, H, W = x.shape

        x_merged = x.reshape(B * T, C, H, W)

        spatial_tokens = self.patch_embed(x_merged)

        N = spatial_tokens.shape[1]

        video_tokens = spatial_tokens.view(B, T, N, self.embed_dim)

        video_tokens = video_tokens + self.time_embed

        return video_tokens


class LocalFeatureAggregation(nn.Module):
    """
    LFA (Local Feature Aggregation) Module
    Replaces the simple PatchEmbedding. Extracts and aggregates local spatial features
    using convolutional layers before mapping to the embedding dimension.
    """

    def __init__(self, in_channels: int = 1, embed_dim: int = 64, roi_mode: str = 'original'):
        super().__init__()
        self.embed_dim = embed_dim
        self.roi_mode = roi_mode

        # Define patch definitions: (h, w, y_start, x_start)
        if roi_mode == 'original':
            self.patch_definitions = [
                (2, 3, 0, 0), (1, 3, 0, 3), (2, 3, 0, 6),
                (3, 3, 2, 0), (4, 3, 1, 3), (3, 3, 2, 6)
            ]
        elif roi_mode == 'full':
            self.patch_definitions = [(5, 9, 0, 0)]
        elif roi_mode == 'hemi_4_5':
            self.patch_definitions = [(5, 4, 0, 0), (5, 5, 0, 4)]
        elif roi_mode == 'hemi_5_4':
            self.patch_definitions = [(5, 5, 0, 0), (5, 4, 0, 5)]
        elif roi_mode == 'three_columns':
            self.patch_definitions = [(5, 3, 0, 0), (5, 3, 0, 3), (5, 3, 0, 6)]
        elif roi_mode == 'grid_1x3':
            self.patch_definitions = []
            for row in range(5):
                for col_idx in range(3):
                    self.patch_definitions.append((1, 3, row, col_idx * 3))
        else:
            raise ValueError(f"Unknown roi_mode: {roi_mode}")

        self.num_patches = len(self.patch_definitions)
        self.local_extractors = nn.ModuleList()

        # Build LFA extractors for each ROI
        for h, w, _, _ in self.patch_definitions:
            # LFA component: local spatial convolutions to aggregate features
            # instead of a single linear projection.
            local_conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels * 8, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels * 8),
                nn.GELU(),
                nn.Conv2d(in_channels * 8, embed_dim, kernel_size=(h, w))
            )
            self.local_extractors.append(local_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch_embeddings = []
        for i, (h, w, y_start, x_start) in enumerate(self.patch_definitions):
            image_patch = x[..., y_start: y_start + h, x_start: x_start + w]
            # Aggregate local features
            patch_proj_output = self.local_extractors[i](image_patch)
            flattened_patch = patch_proj_output.flatten(2).transpose(1, 2)
            patch_embeddings.append(flattened_patch)

        return torch.cat(patch_embeddings, dim=1)


class VideoPatchEmbeddingWrapper(nn.Module):
    """
    Adapter: Converts 5D video data to 3D sequential data for the Transformer.
    """

    def __init__(self, patch_embed_module, num_frames, embed_dim):
        super().__init__()
        self.patch_embed = patch_embed_module
        self.num_frames = num_frames
        self.embed_dim = embed_dim

        self.time_embed = nn.Parameter(torch.zeros(1, num_frames, 1, embed_dim))
        nn.init.trunc_normal_(self.time_embed, std=0.02)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x_merged = x.reshape(B * T, C, H, W)

        # Pass through LFA
        spatial_tokens = self.patch_embed(x_merged)
        N = spatial_tokens.shape[1]

        video_tokens = spatial_tokens.view(B, T, N, self.embed_dim)
        video_tokens = video_tokens + self.time_embed

        return video_tokens


if __name__ == '__main__':
    BATCH_SIZE = 2
    FRAMES = 8
    CHANNELS = 1
    HEIGHT = 5
    WIDTH = 9
    EMBED_DIM = 128

    base_embed = CustomPatchEmbedding(in_channels=CHANNELS, embed_dim=EMBED_DIM)
    video_adapter = VideoPatchEmbeddingWrapper(
        patch_embed_module=base_embed,
        num_frames=FRAMES,
        embed_dim=EMBED_DIM
    )

    input_video = torch.randn(BATCH_SIZE, FRAMES, CHANNELS, HEIGHT, WIDTH)
    print(f"Input video shape: {input_video.shape}")

    final_tokens = video_adapter(input_video)
    print(final_tokens.shape)

    expected_seq_len = FRAMES * base_embed.num_patches

    print(f"TimeSformer input shape: {final_tokens.shape}")
    print(f"Expected shape: [{BATCH_SIZE}, {expected_seq_len}, {EMBED_DIM}]")

    if final_tokens.shape == (BATCH_SIZE, expected_seq_len, EMBED_DIM):
        print("Success: Shape verification passed. Ready for Transformer Block!")
    else:
        print("Error: Shape mismatch. Please check the code.")