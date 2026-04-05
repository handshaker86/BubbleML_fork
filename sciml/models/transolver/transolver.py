import torch
import torch.nn as nn

from .physics_attention import PhysicsAttentionStructuredMesh2D

ACTIVATION = {
    'gelu': nn.GELU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'relu': nn.ReLU,
    'leaky_relu': lambda: nn.LeakyReLU(0.1),
    'softplus': nn.Softplus,
    'elu': nn.ELU,
    'silu': nn.SiLU,
}


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super().__init__()
        if act not in ACTIVATION:
            raise NotImplementedError(f"Activation '{act}' not supported")
        act_fn = ACTIVATION[act]
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act_fn())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([
            nn.Sequential(nn.Linear(n_hidden, n_hidden), act_fn())
            for _ in range(n_layers)
        ])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class TransolverBlock(nn.Module):
    """Transformer encoder block with Physics-Attention."""

    def __init__(self, num_heads, hidden_dim, dropout, act='gelu',
                 mlp_ratio=4, last_layer=False, out_dim=1,
                 slice_num=32, H=85, W=85):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = PhysicsAttentionStructuredMesh2D(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num, H=H, W=W
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
                       n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


class Transolver(nn.Module):
    """Transolver adapted for BubbleML (B, C, H, W) I/O convention.

    Internally operates on (B, N, C) tokens where N = H * W,
    using Physics-Attention (Slice -> Attend -> Deslice).
    """

    def __init__(self, in_channels, out_channels, domain_rows, domain_cols,
                 n_hidden=256, n_layers=5, n_head=8, slice_num=32,
                 mlp_ratio=1, dropout=0.0, act='gelu'):
        super().__init__()
        self.H = int(domain_rows)
        self.W = int(domain_cols)
        self.n_hidden = n_hidden

        self.preprocess = MLP(in_channels, n_hidden * 2, n_hidden,
                              n_layers=0, res=False, act=act)

        self.placeholder = nn.Parameter(
            (1 / n_hidden) * torch.rand(n_hidden, dtype=torch.float)
        )

        self.blocks = nn.ModuleList([
            TransolverBlock(
                num_heads=n_head,
                hidden_dim=n_hidden,
                dropout=dropout,
                act=act,
                mlp_ratio=mlp_ratio,
                out_dim=out_channels,
                slice_num=slice_num,
                H=self.H,
                W=self.W,
                last_layer=(i == n_layers - 1),
            )
            for i in range(n_layers)
        ])

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: (B, C_in, H, W)
        B, C, H, W = x.shape

        # Reshape to token sequence: (B, N, C_in)
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Preprocess + placeholder
        fx = self.preprocess(x) + self.placeholder[None, None, :]

        # Transformer blocks
        for block in self.blocks:
            fx = block(fx)

        # Reshape back: (B, N, C_out) -> (B, C_out, H, W)
        fx = fx.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return fx
