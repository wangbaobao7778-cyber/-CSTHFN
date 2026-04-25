import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CausalSpatialAttention(nn.Module):
    """
    Causal Prior Spatial Attention Mechanism.
    Injects the Causal Matrix A_causal directly into the attention weights as a spatial bias.
    """

    def __init__(self, dim, num_heads, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Learnable parameter to control the influence of causal prior
        self.causal_lambda = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x, A_causal=None, num_frames=1):
        # x: [B*T, N, D]
        B_T, N, D = x.shape
        B = B_T // num_frames

        qkv = self.qkv(x).reshape(B_T, N, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Calculate standard attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B*T, num_heads, N, N]

        # Inject Causal Prior Bias
        if A_causal is not None:
            # A_causal shape: [B, N, N]
            # Broadcast A_causal to match attn shape: [B*T, num_heads, N, N]
            A_causal_expanded = A_causal.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N, N]
            A_causal_expanded = A_causal_expanded.expand(B, num_frames, self.num_heads, N, N)
            A_causal_expanded = A_causal_expanded.reshape(B_T, self.num_heads, N, N)

            # Dynamic injection: Attn = Softmax(QK^T/sqrt(d) + lambda * A_causal)
            attn = attn + self.causal_lambda * A_causal_expanded

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_T, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CausalPSTABlock(nn.Module):
    """
    Causal Prior Spatial-Temporal Attention Block (CausalPSTA)
    """

    def __init__(self, dim, num_heads, num_frames, num_patches, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.num_frames = num_frames
        self.num_patches = num_patches

        # --- Temporal Attention ---
        self.norm1 = nn.LayerNorm(dim)
        self.temporal_attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)

        # --- Causal Spatial Attention ---
        self.norm2 = nn.LayerNorm(dim)
        self.causal_spatial_attn = CausalSpatialAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )

        # --- MLP ---
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x, A_causal=None):
        B, Total, D = x.shape
        T = self.num_frames
        N = self.num_patches

        # === 1. Temporal Attention ===
        residual = x
        x = self.norm1(x)

        x = x.view(B, T, N, D).permute(0, 2, 1, 3).reshape(B * N, T, D)
        x, _ = self.temporal_attn(x, x, x)
        x = x.view(B, N, T, D).permute(0, 2, 1, 3).reshape(B, T * N, D)

        x = x + residual

        # === 2. Causal Spatial Attention ===
        residual = x
        x = self.norm2(x)

        x = x.view(B, T, N, D).reshape(B * T, N, D)

        # Apply custom causal attention
        x_attn = self.causal_spatial_attn(x, A_causal=A_causal, num_frames=T)

        x = x_attn.view(B, T, N, D).reshape(B, T * N, D)
        x = x + residual

        # === 3. MLP ===
        x = x + self.mlp(self.norm3(x))
        return x


class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(dim, dim)

    def forward(self, x_original, x_cross):
        z = self.gate_net(torch.cat([x_original, x_cross], dim=-1))
        out = z * x_original + (1 - z) * self.proj(x_cross)
        return out


class BIE(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.cross_attn_1 = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.cross_attn_2 = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)

        self.gate_1 = GatedFusion(dim)
        self.gate_2 = GatedFusion(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x1, x2):
        x1_norm = self.norm1(x1)
        x2_norm = self.norm2(x2)

        attn_out_1, _ = self.cross_attn_1(query=x1_norm, key=x2_norm, value=x2_norm)
        attn_out_2, _ = self.cross_attn_2(query=x2_norm, key=x1_norm, value=x1_norm)

        out1 = self.gate_1(x1, attn_out_1)
        out2 = self.gate_2(x2, attn_out_2)

        return out1, out2