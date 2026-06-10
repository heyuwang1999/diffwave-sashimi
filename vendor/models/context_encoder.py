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
import torch.nn.functional as F


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

    def __init__(self, d_cond=512, d_model=64, n_blocks=6, downsample=16, pool="attention",
                 n_context_tokens=64):
        super().__init__()
        self.stem = nn.Conv1d(1, d_model, kernel_size=downsample * 2, stride=downsample, padding=downsample // 2)
        self.blocks = nn.ModuleList(_ConvBlock(d_model, dilation=2 ** i) for i in range(n_blocks))
        self.pool_kind = pool
        if pool == "attention":
            self.attn_score = nn.Conv1d(d_model, 1, kernel_size=1)
        self.proj = nn.Sequential(nn.Linear(d_model, d_cond), nn.GELU(), nn.Linear(d_cond, d_cond))
        # For cross-attention conditioning: a fixed-length token sequence.
        self.n_context_tokens = n_context_tokens
        self.token_proj = nn.Linear(d_model, d_cond)

    def _features(self, context):
        x = self.stem(context)
        for blk in self.blocks:
            x = blk(x)
        return x  # (B, d_model, T)

    def forward(self, context):
        # context: (B, 1, L_ctx) -> (B, d_cond) global summary
        x = self._features(context)
        if self.pool_kind == "attention":
            w = torch.softmax(self.attn_score(x), dim=-1)
            summary = (x * w).sum(dim=-1)
        else:
            summary = x.mean(dim=-1)
        return self.proj(summary)

    def tokens(self, context):
        # context -> (B, n_context_tokens, d_cond) memory for cross-attention
        x = self._features(context)
        x = F.adaptive_avg_pool1d(x, self.n_context_tokens)  # (B, d_model, n_tokens)
        return self.token_proj(x.transpose(1, 2))            # (B, n_tokens, d_cond)


class CrossAttention(nn.Module):
    """Multi-head cross-attention: queries from the diffusion activations attend to
    a sequence of context tokens. Output projection is zero-initialized so the
    module starts as an identity residual (stable to add to a trained/untrained net)."""

    def __init__(self, d_query, d_context, n_heads=4, d_head=32):
        super().__init__()
        inner = n_heads * d_head
        self.n_heads, self.d_head = n_heads, d_head
        self.norm = nn.LayerNorm(d_query)
        self.to_q = nn.Linear(d_query, inner, bias=False)
        self.to_k = nn.Linear(d_context, inner, bias=False)
        self.to_v = nn.Linear(d_context, inner, bias=False)
        self.to_out = nn.Linear(inner, d_query)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, x, context_tokens):
        # x: (B, d_query, L); context_tokens: (B, M, d_context) -> (B, d_query, L)
        B, C, L = x.shape
        h = self.norm(x.transpose(1, 2))                 # (B, L, C)
        q, k, v = self.to_q(h), self.to_k(context_tokens), self.to_v(context_tokens)

        def heads(t):
            return t.view(t.shape[0], t.shape[1], self.n_heads, self.d_head).transpose(1, 2)

        out = F.scaled_dot_product_attention(heads(q), heads(k), heads(v))  # (B, nh, L, dh)
        out = out.transpose(1, 2).reshape(B, L, self.n_heads * self.d_head)
        return self.to_out(out).transpose(1, 2)          # (B, d_query, L)
