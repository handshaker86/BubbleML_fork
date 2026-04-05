import torch
import torch.nn as nn


class PhysicsAttentionStructuredMesh2D(nn.Module):
    """Physics-Attention for structured 2D meshes.

    Implements the Slice -> Attend -> Deslice mechanism:
    1. Slice: Soft-cluster N mesh points into G learned physical states
    2. Attend: Standard multi-head attention among G slice tokens
    3. Deslice: Distribute slice-level features back to mesh points
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0,
                 slice_num=64, H=85, W=85, kernel=3):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.H = H
        self.W = W

        self.in_project_x = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_fx = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        torch.nn.init.orthogonal_(self.in_project_slice.weight)

        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (B, N, C) where N = H * W
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2).contiguous()  # B C H W

        # (1) Slice
        fx_mid = (self.in_project_fx(x)
                  .permute(0, 2, 3, 1).contiguous()
                  .reshape(B, N, self.heads, self.dim_head)
                  .permute(0, 2, 1, 3).contiguous())  # B H N C
        x_mid = (self.in_project_x(x)
                 .permute(0, 2, 3, 1).contiguous()
                 .reshape(B, N, self.heads, self.dim_head)
                 .permute(0, 2, 1, 3).contiguous())  # B H N C

        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5)
        )  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        # (2) Attention among slice tokens
        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v)  # B H G D

        # (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = out_x.permute(0, 2, 1, 3).contiguous().reshape(B, N, -1)
        return self.to_out(out_x)
