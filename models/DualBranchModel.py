import torch
import torch.nn as nn
import torch.nn.functional as F

from .TemporalFusionModule import TemporalFusionModule
from .Timesformer import CausalPSTABlock, BIE
from .Embed import LocalFeatureAggregation, VideoPatchEmbeddingWrapper


class DualBranchRecurrentModel(nn.Module):
    def __init__(self,
                 embed_dim=64,
                 num_heads=4,
                 depth=3,
                 k_memory=20,
                 num_classes=2,
                 chunk_size=10,
                 drop=0.,
                 attn_drop=0.,
                 roi_mode='original',
                 keep_ratio=0.4,
                 edl_mode=False):
        super().__init__()

        self.edl_mode = edl_mode
        self.depth = depth
        self.chunk_size = chunk_size
        self.embed_dim = embed_dim

        # =========================================================
        # 1. Embedding Layers (Using LFA)
        # =========================================================
        self.patch_embed_hbo = LocalFeatureAggregation(in_channels=1, embed_dim=embed_dim, roi_mode=roi_mode)
        num_patches = self.patch_embed_hbo.num_patches
        self.video_wrapper_hbo = VideoPatchEmbeddingWrapper(
            patch_embed_module=self.patch_embed_hbo,
            num_frames=chunk_size,
            embed_dim=embed_dim
        )

        self.patch_embed_hbr = LocalFeatureAggregation(in_channels=1, embed_dim=embed_dim, roi_mode=roi_mode)
        self.video_wrapper_hbr = VideoPatchEmbeddingWrapper(
            patch_embed_module=self.patch_embed_hbr,
            num_frames=chunk_size,
            embed_dim=embed_dim
        )

        # =========================================================
        # 2. Backbone Construction (CausalPSTA + BIE)
        # =========================================================
        self.hbo2_blocks = nn.ModuleList([
            CausalPSTABlock(embed_dim, num_heads, chunk_size, num_patches, drop=drop, attn_drop=attn_drop) for _ in
            range(depth)
        ])
        self.hbr_blocks = nn.ModuleList([
            CausalPSTABlock(embed_dim, num_heads, chunk_size, num_patches, drop=drop, attn_drop=attn_drop) for _ in
            range(depth)
        ])

        self.bie_layers = nn.ModuleList([
            BIE(embed_dim, num_heads) for _ in range(depth)
        ])

        # =========================================================
        # 3. Fusion Modules & Classifier
        # =========================================================
        self.fusion_hbo2 = TemporalFusionModule(embed_dim, k_memory)
        self.fusion_hbr = TemporalFusionModule(embed_dim, k_memory)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_classes)
        )

    def forward_one_step(self, hbo2_emb, hbr_emb, A_causal_oxy=None, A_causal_dxy=None):
        x1 = hbo2_emb
        x2 = hbr_emb
        B, T, N, D = x1.shape

        x1 = x1.view(B, T * N, D)
        x2 = x2.view(B, T * N, D)

        for i in range(self.depth):
            x1 = self.hbo2_blocks[i](x1, A_causal_oxy)
            x2 = self.hbr_blocks[i](x2, A_causal_dxy)

            x1, x2 = self.bie_layers[i](x1, x2)

        f_t_hbo2 = self.fusion_hbo2(x1)
        f_t_hbr = self.fusion_hbr(x2)

        return f_t_hbo2, f_t_hbr

    def forward(self, hbo2_raw, hbr_raw, A_causal_oxy=None, A_causal_dxy=None, return_features=False):
        if hbo2_raw.dim() == 4:
            hbo2_raw = hbo2_raw.unsqueeze(2)
            hbr_raw = hbr_raw.unsqueeze(2)

        batch_size, time_steps, _, _, _ = hbo2_raw.shape

        self.fusion_hbo2.reset_memory()
        self.fusion_hbr.reset_memory()

        final_hbo2 = None
        final_hbr = None
        step = self.chunk_size

        for t in range(0, time_steps, step):
            if t + step > time_steps:
                break

            chunk_hbo2 = hbo2_raw[:, t: t + step, ...]
            chunk_hbr = hbr_raw[:, t: t + step, ...]

            emb_hbo2 = self.video_wrapper_hbo(chunk_hbo2)
            emb_hbr = self.video_wrapper_hbr(chunk_hbr)

            out_hbo2, out_hbr = self.forward_one_step(
                emb_hbo2, emb_hbr,
                A_causal_oxy=A_causal_oxy,
                A_causal_dxy=A_causal_dxy
            )

            if t == time_steps - step:
                final_hbo2 = out_hbo2
                final_hbr = out_hbr

        if final_hbo2.dim() == 3:
            feat_1 = final_hbo2.mean(dim=1)
            feat_2 = final_hbr.mean(dim=1)
        else:
            feat_1 = final_hbo2
            feat_2 = final_hbr

        combined_feat = torch.cat([feat_1, feat_2], dim=-1)

        logits = self.classifier(combined_feat)

        if self.edl_mode:
            output = F.softplus(logits)
        else:
            output = logits

        if return_features:
            return output, combined_feat
        return output


if __name__ == "__main__":
    B = 2
    T = 20
    H, W = 5, 9
    D = 64
    CHUNK = 10

    model = DualBranchRecurrentModel(
        embed_dim=D,
        num_heads=4,
        depth=2,
        k_memory=5,
        num_classes=2,
        chunk_size=CHUNK
    )

    input_hbo2 = torch.randn(B, T, H, W)
    input_hbr = torch.randn(B, T, H, W)

    # 模拟外部传入的因果矩阵
    A_causal = torch.randn(B, 6, 6)

    print(f"Input Raw Shape: {input_hbo2.shape}")
    output = model(input_hbo2, input_hbr, A_causal, A_causal)
    print(f"Output Logits Shape: {output.shape}")

    loss = output.sum()
    loss.backward()
    print("Backward pass successful. Gradients flowing gracefully!")