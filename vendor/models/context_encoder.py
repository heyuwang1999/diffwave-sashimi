"""
Context encoder for short-clip continuation (milestone 2), engine-local copy.

Maps a context waveform (the audio preceding the target) to a fixed-size
conditioning vector that is added to the diffusion-step embedding inside Sashimi
(global / FiLM-style conditioning — the robust baseline; cross-attention is a
planned later upgrade, see ../../ROADMAP_V2.md).

Self-contained (torch only) dilated-conv trunk + attention pooling. The same
design lives in the repo-root `music_continuation/` package as the standalone
scaffold; this copy is the one actually wired into the model.
"""
import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    def __init__(self, d, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d, d, kernel_size=3, dilation=dilation, padding=dilation),
            nn.GroupNorm(min(8, d), d),
            nn.GELU(),
        )

    def forward(self, x):
        return x + self.net(x)


class ContextEncoder(nn.Module):
    """Waveform context -> (B, d_cond) summary vector."""

    def __init__(self, d_cond=512, d_model=64, n_blocks=6, downsample=16, pool="attention"):
        super().__init__()
        self.stem = nn.Conv1d(1, d_model, kernel_size=downsample * 2, stride=downsample, padding=downsample // 2)
        self.blocks = nn.ModuleList(_ConvBlock(d_model, dilation=2 ** i) for i in range(n_blocks))
        self.pool_kind = pool
        if pool == "attention":
            self.attn_score = nn.Conv1d(d_model, 1, kernel_size=1)
        self.proj = nn.Sequential(nn.Linear(d_model, d_cond), nn.GELU(), nn.Linear(d_cond, d_cond))

    def forward(self, context):
        # context: (B, 1, L_ctx) -> (B, d_cond)
        x = self.stem(context)
        for blk in self.blocks:
            x = blk(x)
        if self.pool_kind == "attention":
            w = torch.softmax(self.attn_score(x), dim=-1)
            summary = (x * w).sum(dim=-1)
        else:
            summary = x.mean(dim=-1)
        return self.proj(summary)
