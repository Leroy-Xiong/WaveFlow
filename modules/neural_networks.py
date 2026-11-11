import torch
import torch.nn as nn


class SimpleVelocityNet(nn.Module):
    """
    A simple MLP to approximate the velocity field.
    Inputs: t (1 dim), x (D dims)
    Outputs: v (D dims)
    """
    def __init__(self, dim, hidden_dim=4096):
        super().__init__()
        # Input: dimension of data + 1 dimension for time
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.SiLU(),  # Swish activation
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x, t):
        # Concatenate x and t along the feature dimension
        # x: [B, D], t: [B] or [B, 1] -> input: [B, D+1]
        # Note: Parameter order (x, t) matches FlowDiffusion.compute_loss() calling convention
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # [B] -> [B, 1]
        x_in = torch.cat([x, t], dim=-1)
        return self.net(x_in)