import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

class TemporalFusionModule(nn.Module):

    def __init__(self, dim, k_memory_size):
        super().__init__()
        self.k = k_memory_size
        self.dim = dim

        self.memory_linear = nn.Linear(dim * k_memory_size, dim)

        self.fusion_layer = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )

        # Feature pool using deque
        self.feature_pool = deque(maxlen=k_memory_size)

    def reset_memory(self):

        self.feature_pool.clear()

    def forward(self, f_t_raw):

        batch_size, num_patches, dim = f_t_raw.shape

        # 1. Fill the pool if it's not full (cold start)
        if len(self.feature_pool) == 0:
            for _ in range(self.k):
                self.feature_pool.append(torch.zeros_like(f_t_raw))

        # 2. Retrieve k features from the pool
        memory_stack = torch.stack(list(self.feature_pool), dim=2)

        # 3. Process memory with Linear layer
        memory_flat = memory_stack.view(batch_size, num_patches, -1)
        memory_processed = self.memory_linear(memory_flat)

        # 4. Fusion
        f_t_fused = self.fusion_layer(torch.cat([f_t_raw, memory_processed], dim=-1))

        # 5. Update pool
        self.feature_pool.append(f_t_fused)

        return f_t_fused