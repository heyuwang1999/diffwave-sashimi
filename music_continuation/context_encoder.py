"""
Context encoder for short-clip music continuation (milestone 2).

STATUS: scaffold. This module is intentionally NOT wired into the vendored
training loop yet — milestone 1 (unconditional generation) must train cleanly
first. It is self-contained (depends only on torch) so it can be unit-tested in
isolation before integration.

What it does
------------
Maps a *context* waveform (the audio immediately preceding the target clip) to a
fixed-size conditioning vector `cond` of dimension `d_cond`. That vector is then
injected into the DiffWave/SaShiMi backbone the same way the diffusion-step
embedding is: project per block and broadcast-add across time (FiLM-style).

Why a global summary first
--------------------------
The context is the *past*; the target is the *future*. They are not time-aligned,
so a single summary vector ("style / key / energy / timbre of what just played")
is the honest baseline. A time-aligned cross-attention variant is the planned
follow-up (see DESIGN.md) once this underfits local timbral evolution.

Integration sketch (do this in milestone 2, in the vendored model)
-------------------------------------------------------------------
1. In `Sashimi.__init__`, when conditioning on context, build
   `self.context_encoder = ContextEncoder(d_cond=diffusion_step_embed_dim_out)`
   and reuse the existing `diffusion_step_embed` broadcast-add path: in each
   `DiffWaveBlock`, add `self.fc_cond(cond).unsqueeze(-1)` (mirror `fc_t`).
2. In `train.py`, draw `(context, target)` pairs and compute
   `cond = net.context_encoder(context)`, threading `cond` through `forward`.
3. Apply `cond = cond * bernoulli(keep_prob)` (context dropout) so the same model
   also does unconditional generation and the context lift is measurable.

Swapping in S4: the dilated-conv trunk below can be replaced by the vendored
`models.s4.S4` blocks for true long-range context modeling; kept as convs here so
the scaffold has no pykeops/cwd dependency.
"""
import torch
import torch.nn as nn


class FiLM(nn.Module):
    """Feature-wise linear modulation: produce per-channel (scale, shift) from cond.

    Helper for the eventual integration; lets a block do `x = gamma * x + beta`
    instead of a plain additive bias if richer conditioning is wanted.
    """

    def __init__(self, d_cond, d_feature):
        super().__init__()
        self.to_scale = nn.Linear(d_cond, d_feature)
        self.to_shift = nn.Linear(d_cond, d_feature)
        # Start as identity (scale=1, shift=0) so adding the module is a no-op.
        nn.init.zeros_(self.to_scale.weight); nn.init.zeros_(self.to_scale.bias)
        nn.init.zeros_(self.to_shift.weight); nn.init.zeros_(self.to_shift.bias)

    def forward(self, x, cond):
        # x: (B, C, L), cond: (B, d_cond)
        gamma = (1.0 + self.to_scale(cond)).unsqueeze(-1)
        beta = self.to_shift(cond).unsqueeze(-1)
        return gamma * x + beta


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
    """Waveform context -> fixed-size conditioning vector.

    Parameters
    ----------
    d_cond : int
        Output conditioning dimension (match the backbone's
        `diffusion_step_embed_dim_out`, e.g. 512, for the broadcast-add path).
    d_model : int
        Internal channel width.
    n_blocks : int
        Number of dilated residual conv blocks (dilation doubles each block,
        giving an exponentially growing receptive field over the context).
    downsample : int
        Strided downsampling applied up front to keep compute modest on a ~2 s
        context at 22 kHz.
    pool : {"attention", "mean"}
        How to collapse the time axis into the summary vector.
    """

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
            w = torch.softmax(self.attn_score(x), dim=-1)  # (B, 1, T)
            summary = (x * w).sum(dim=-1)                  # (B, d_model)
        else:
            summary = x.mean(dim=-1)
        return self.proj(summary)


if __name__ == "__main__":
    # Smoke test: shapes only (CPU-friendly).
    enc = ContextEncoder(d_cond=512, d_model=64)
    ctx = torch.randn(3, 1, 44032)  # 3 context clips, ~2 s @ 22050 Hz
    cond = enc(ctx)
    assert cond.shape == (3, 512), cond.shape
    film = FiLM(d_cond=512, d_feature=64)
    feats = torch.randn(3, 64, 2752)
    out = film(feats, cond)
    assert out.shape == feats.shape
    print("ContextEncoder + FiLM smoke test OK:", cond.shape, out.shape)
