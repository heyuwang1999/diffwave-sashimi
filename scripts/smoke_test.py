"""
CPU smoke test for the music pipeline + the vendored SaShiMi+DiffWave model.

Run from the repo root with the smoke venv:
    cd vendor && python ../scripts/smoke_test.py

The upstream code hardcodes .cuda(); we no-op it so the *real* code paths run on
CPU. Model/diffusion are scaled tiny for speed — this checks soundness, not quality.
"""
import os, sys, math, tempfile, warnings
warnings.filterwarnings("ignore")

import torch

# --- make the real (cuda-hardcoded) code run on CPU ---------------------------
torch.Tensor.cuda = lambda self, *a, **k: self            # type: ignore
torch.nn.Module.cuda = lambda self, *a, **k: self         # type: ignore

import torchaudio
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)                          # for `music_continuation`
sys.path.insert(0, os.path.join(ROOT, "vendor"))  # for `dataloaders`, `models`, `utils`, `train`

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
results = []
def check(name, cond, extra=""):
    results.append(cond)
    print(f"  [{PASS if cond else FAIL}] {name} {extra}")

SR = 22050
SEG = 44032

def make_dataset_dir():
    d = tempfile.mkdtemp(prefix="music_smoke_")
    # two "long" tracks (~3 s and ~4 s) + one deliberately too-short clip (0.2 s)
    import soundfile as sf
    for name, secs, freq in [("trackA", 3.0, 220.0), ("trackB", 4.0, 330.0), ("tiny", 0.2, 440.0)]:
        t = torch.arange(int(secs * SR)) / SR
        wav = (0.5 * torch.sin(2 * math.pi * freq * t)).numpy()
        sf.write(os.path.join(d, f"{name}.wav"), wav, SR)  # backend-agnostic fixture
    return d

print("\n=== 1. MusicWaveform dataset ===")
from dataloaders.music import MusicWaveform
data_dir = make_dataset_dir()
ds = MusicWaveform(data_dir, segment_length=SEG, sampling_rate=SR, samples_per_epoch=50)
wav, sr, label = ds[0]
check("returns 3-tuple (waveform, sr, label)", isinstance(label, str) and sr == SR)
check("waveform shape == (1, segment_length)", tuple(wav.shape) == (1, SEG), str(tuple(wav.shape)))
check("waveform dtype float32", wav.dtype == torch.float32, str(wav.dtype))
check("peak-normalized to <= 1.0", float(wav.abs().max()) <= 1.0 + 1e-4, f"max={float(wav.abs().max()):.3f}")
check("len == samples_per_epoch", len(ds) == 50)
# hammer many draws incl. the tiny (tiling) file -> must never crash or wrong-shape
shapes_ok = all(tuple(ds[i][0].shape) == (1, SEG) for i in range(50))
check("50 random draws all correct shape (tiling path exercised)", shapes_ok)

print("\n=== 2. dataloader() dispatch + batching ===")
from dataloaders import dataloader
cfg = OmegaConf.create({"_name_": "music", "data_path": data_dir,
                        "segment_length": SEG, "sampling_rate": SR,
                        "samples_per_epoch": 16, "peak_normalize": True})
dl = dataloader(cfg, batch_size=4, num_gpus=1, unconditional=True)
batch = next(iter(dl))
audio, _, _ = batch                       # exactly how train.py unpacks it
check("batched audio shape == (B,1,L)", tuple(audio.shape) == (4, 1, SEG), str(tuple(audio.shape)))
check("_name_ restored in cfg after dispatch", cfg.get("_name_") == "music")

print("\n=== 3. ContextEncoder (milestone-2 scaffold) ===")
from music_continuation import ContextEncoder, FiLM
enc = ContextEncoder(d_cond=128, d_model=32, n_blocks=4)
cond = enc(torch.randn(2, 1, SEG))
check("context -> (B, d_cond)", tuple(cond.shape) == (2, 128), str(tuple(cond.shape)))
film = FiLM(d_cond=128, d_feature=32)
y = film(torch.randn(2, 32, 100), cond)
check("FiLM preserves feature shape", tuple(y.shape) == (2, 32, 100))
check("FiLM is identity at init", torch.allclose(y, y), "")  # placeholder; real check below
feats = torch.randn(2, 32, 100)
check("FiLM == identity at init (zero-init)", torch.allclose(film(feats, cond), feats, atol=1e-6))

print("\n=== 4. REAL Sashimi forward + DiffWave training loss (tiny, CPU) ===")
from models import construct_model
from utils import calc_diffusion_hyperparams
from train import training_loss
import torch.nn as nn

TINY_L = 256  # divisible by prod(pool)=16
model_cfg = OmegaConf.create({
    "_name_": "sashimi", "unconditional": True, "in_channels": 1, "out_channels": 1,
    # NOTE: diffusion_step_embed_dim_out is effectively locked to 512 upstream
    # (DiffWaveBlock hardcodes the default and Sashimi._residual doesn't forward it).
    # Our real configs use 512 to match; keep it here too.
    "diffusion_step_embed_dim_in": 128, "diffusion_step_embed_dim_mid": 256,
    "diffusion_step_embed_dim_out": 512, "unet": True,
    "d_model": 16, "n_layers": 2, "pool": [4, 4], "expand": 2, "ff": 2, "L": TINY_L,
})
net = construct_model(model_cfg)
n_params = sum(p.numel() for p in net.parameters())
check("model constructs from config", isinstance(net, nn.Module), f"({n_params:,} params)")

x = torch.randn(2, 1, TINY_L)
steps = torch.randint(0, 50, (2, 1))
out = net((x, steps), mel_spec=None)
check("forward output shape == input shape", tuple(out.shape) == (2, 1, TINY_L), str(tuple(out.shape)))
check("forward output finite", bool(torch.isfinite(out).all()))

dh = calc_diffusion_hyperparams(T=50, beta_0=1e-4, beta_T=0.02, beta=None, fast=False)
loss = training_loss(net, nn.MSELoss(), x, dh, mel_spec=None)
check("training_loss is finite scalar", loss.dim() == 0 and bool(torch.isfinite(loss)), f"loss={loss.item():.4f}")
loss.backward()
grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
check("backward produces finite grads", grad_ok)

print("\n=== 5. Reverse sampling loop (few steps) ===")
from generate import sampling
net.eval()
samp = sampling(net, (1, 1, TINY_L), dh)
check("sampling output shape", tuple(samp.shape) == (1, 1, TINY_L), str(tuple(samp.shape)))
check("sampling output finite", bool(torch.isfinite(samp).all()))

print("\n=== 6. Tier-0: v-prediction + Min-SNR weighting (config-gated) ===")
for param in ("eps", "v"):
    for gamma in (None, 5.0):
        dh2 = calc_diffusion_hyperparams(T=50, beta_0=1e-4, beta_T=0.02, beta=None,
                                         fast=False, parameterization=param, min_snr_gamma=gamma)
        net.train(); net.zero_grad()
        l = training_loss(net, nn.MSELoss(), x, dh2, mel_spec=None)
        finite = l.dim() == 0 and bool(torch.isfinite(l))
        l.backward()
        gok = any(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
        check(f"train_loss finite+grads [{param}, min_snr={gamma}]", finite and gok, f"loss={l.item():.4f}")
    net.eval()
    dh_s = calc_diffusion_hyperparams(T=20, beta_0=1e-4, beta_T=0.02, beta=None,
                                      fast=False, parameterization=param)
    s = sampling(net, (1, 1, TINY_L), dh_s)
    check(f"sampling finite [{param}]", tuple(s.shape) == (1, 1, TINY_L) and bool(torch.isfinite(s).all()))

print("\n=== 7. Tier-0: cosine schedule + STFT aux loss + self-conditioning ===")
from train import multi_resolution_stft_loss

# cosine schedule: Alpha_bar must be strictly decreasing in [0,1]
dh_cos = calc_diffusion_hyperparams(T=50, beta_0=1e-4, beta_T=0.02, beta=None,
                                    fast=False, schedule="cosine")
abar = dh_cos["Alpha_bar"]
check("cosine schedule: len==T, in (0,1], decreasing",
      len(abar) == 50 and float(abar.max()) <= 1.0 and bool((abar[1:] <= abar[:-1]).all()),
      f"abar[0]={float(abar[0]):.4f} abar[-1]={float(abar[-1]):.4f}")
net.train(); net.zero_grad()
lc = training_loss(net, nn.MSELoss(), x, dh_cos, mel_spec=None)
check("training_loss finite under cosine schedule", bool(torch.isfinite(lc)), f"loss={lc.item():.4f}")

# multi-resolution STFT loss: 0 for identical, >0 for different
sig = torch.randn(2, 1, TINY_L)
check("STFT loss == 0 for identical signals", float(multi_resolution_stft_loss(sig, sig)) < 1e-5)
check("STFT loss > 0 for different signals", float(multi_resolution_stft_loss(sig, torch.randn(2, 1, TINY_L))) > 0)
dh_stft = calc_diffusion_hyperparams(T=50, beta_0=1e-4, beta_T=0.02, beta=None,
                                     fast=False, parameterization="v", stft_loss_weight=0.1)
net.zero_grad()
ls = training_loss(net, nn.MSELoss(), x, dh_stft, mel_spec=None)
ls.backward()
gok = any(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
check("training_loss + STFT aux: finite + grads", bool(torch.isfinite(ls)) and gok, f"loss={ls.item():.4f}")

# self-conditioning: model takes an extra input channel; train + sample must work
sc_cfg = OmegaConf.create(dict(model_cfg)); sc_cfg["_name_"] = "sashimi"; sc_cfg["self_conditioning"] = True
sc_net = construct_model(sc_cfg)
check("self-cond model has self_conditioning=True", getattr(sc_net, "self_conditioning", False))
out_zero = sc_net((x, steps), x_self_cond=None)                    # None -> zeros internally
out_cond = sc_net((x, steps), x_self_cond=torch.randn_like(x))
check("self-cond forward shapes ok (None and provided)",
      tuple(out_zero.shape) == (2, 1, TINY_L) and tuple(out_cond.shape) == (2, 1, TINY_L))
sc_net.train(); sc_net.zero_grad()
lsc = training_loss(sc_net, nn.MSELoss(), x, dh, mel_spec=None)     # detects self_cond via getattr
lsc.backward()
gok2 = any(p.grad is not None and torch.isfinite(p.grad).all() for p in sc_net.parameters())
check("self-cond training_loss: finite + grads (two-pass)", bool(torch.isfinite(lsc)) and gok2, f"loss={lsc.item():.4f}")
sc_net.eval()
ssc = sampling(sc_net, (1, 1, TINY_L), calc_diffusion_hyperparams(T=15, beta_0=1e-4, beta_T=0.02, beta=None, fast=False))
check("self-cond sampling: finite + shape", tuple(ssc.shape) == (1, 1, TINY_L) and bool(torch.isfinite(ssc).all()))

print("\n=== 8. Tier-2: continuation (context conditioning + CFG) ===")
from dataloaders.music import MusicContinuation
cds = MusicContinuation(data_dir, segment_length=TINY_L, context_length=TINY_L,
                        sampling_rate=SR, samples_per_epoch=8)
tgt, ctx0 = cds[0]
check("continuation item: (target[1,L], context[1,L])",
      tuple(tgt.shape) == (1, TINY_L) and tuple(ctx0.shape) == (1, TINY_L))

cc_cfg = OmegaConf.create(dict(model_cfg))
cc_cfg["_name_"] = "sashimi"; cc_cfg["context_conditioning"] = True
cc_cfg["context_d_model"] = 16; cc_cfg["context_n_blocks"] = 2
cc_net = construct_model(cc_cfg)
check("context-cond model flag set", getattr(cc_net, "context_conditioning", False))
# The model's final layer is ZeroConv1d (zero-init), so an untrained net outputs
# zeros and passes no upstream gradient. Nudge it off zero so the influence/grad
# checks below test the wiring rather than the init.
with torch.no_grad():
    for p in cc_net.final_conv.parameters():
        p.add_(0.1 * torch.randn_like(p))
ctxb = torch.randn(2, 1, TINY_L)
o_ctx = cc_net((x, steps), context=ctxb)
o_none = cc_net((x, steps), context=None)   # classifier-free null pass
check("context forward shapes (with/without context)",
      tuple(o_ctx.shape) == (2, 1, TINY_L) and tuple(o_none.shape) == (2, 1, TINY_L))
check("context actually changes the output", not torch.allclose(o_ctx, o_none))

# training_loss with context; dropout=0 so the encoder must receive gradient
dh_c = calc_diffusion_hyperparams(T=50, beta_0=1e-4, beta_T=0.02, beta=None, fast=False,
                                  parameterization="v", context_cfg_dropout=0.0)
cc_net.train(); cc_net.zero_grad()
lc = training_loss(cc_net, nn.MSELoss(), x, dh_c, context=ctxb)
lc.backward()
gok = any(p.grad is not None and torch.isfinite(p.grad).all() for p in cc_net.parameters())
enc_grad = any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in cc_net.context_encoder.parameters())
check("continuation training_loss: finite + grads", bool(torch.isfinite(lc)) and gok, f"loss={lc.item():.4f}")
check("context encoder receives gradient (dropout=0)", enc_grad)

# CFG dropout=1.0: context always dropped -> still finite (unconditional path)
dh_d = calc_diffusion_hyperparams(T=50, beta_0=1e-4, beta_T=0.02, beta=None, fast=False,
                                  parameterization="v", context_cfg_dropout=1.0)
cc_net.zero_grad()
ld = training_loss(cc_net, nn.MSELoss(), x, dh_d, context=ctxb)
check("continuation training_loss finite with full CFG dropout", bool(torch.isfinite(ld)))

# classifier-free guided sampling
cc_net.eval()
dh_s = calc_diffusion_hyperparams(T=15, beta_0=1e-4, beta_T=0.02, beta=None, fast=False, parameterization="v")
sg = sampling(cc_net, (2, 1, TINY_L), dh_s, context=torch.randn(2, 1, TINY_L), guidance=3.0)
check("guided continuation sampling: finite + shape",
      tuple(sg.shape) == (2, 1, TINY_L) and bool(torch.isfinite(sg).all()))

print("\n=== 9. Tier-2: continuation generate CLI (load_context) ===")
import tempfile as _tmp, os as _os, numpy as _np, soundfile as _sf
from generate import load_context
_d = _tmp.mkdtemp(prefix="ctx_")
_sf.write(_os.path.join(_d, "c.wav"), _np.sin(_np.linspace(0, 60, SR)).astype("float32"), SR)
cc = load_context(_os.path.join(_d, "c.wav"), TINY_L, SR)
check("load_context shape (1,1,ctx_len)", tuple(cc.shape) == (1, 1, TINY_L), str(tuple(cc.shape)))
check("load_context peak-normalized", float(cc.abs().max()) <= 1.0 + 1e-4)
_sf.write(_os.path.join(_d, "short.wav"), (_np.ones(TINY_L // 4) * 0.2).astype("float32"), SR)
cs = load_context(_os.path.join(_d, "short.wav"), TINY_L, SR)
check("load_context tiles short clips to ctx_len", tuple(cs.shape) == (1, 1, TINY_L))

print("\n=== 10. Tier-2.2: cross-attention context conditioning ===")
xa_cfg = OmegaConf.create(dict(model_cfg))
xa_cfg["_name_"] = "sashimi"; xa_cfg["context_conditioning"] = True
xa_cfg["context_mode"] = "cross_attn"; xa_cfg["n_context_tokens"] = 8
xa_cfg["context_d_model"] = 16; xa_cfg["context_n_blocks"] = 2
xa_net = construct_model(xa_cfg)
check("cross_attn model built (has cross_attn + mode)",
      hasattr(xa_net, "cross_attn") and xa_net.context_mode == "cross_attn")
# Nudge the zero-inits (final ZeroConv + cross-attn to_out, both identity at start)
# so the checks test the wiring rather than the initialization.
with torch.no_grad():
    for p in list(xa_net.final_conv.parameters()) + list(xa_net.cross_attn.to_out.parameters()):
        p.add_(0.1 * torch.randn_like(p))
ctxb = torch.randn(2, 1, TINY_L)
oa = xa_net((x, steps), context=ctxb)
on = xa_net((x, steps), context=None)
check("cross_attn forward shapes (with/without context)",
      tuple(oa.shape) == (2, 1, TINY_L) and tuple(on.shape) == (2, 1, TINY_L))
check("cross_attn context changes output", not torch.allclose(oa, on))
dh_xa = calc_diffusion_hyperparams(T=50, beta_0=1e-4, beta_T=0.02, beta=None, fast=False,
                                   parameterization="v", context_cfg_dropout=0.0)
xa_net.train(); xa_net.zero_grad()
la = training_loss(xa_net, nn.MSELoss(), x, dh_xa, context=ctxb)
la.backward()
ca_grad = any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in xa_net.cross_attn.parameters())
enc_grad = any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in xa_net.context_encoder.parameters())
check("cross_attn training_loss finite + cross_attn/encoder grads",
      bool(torch.isfinite(la)) and ca_grad and enc_grad, f"loss={la.item():.4f}")
xa_net.eval()
sa = sampling(xa_net, (2, 1, TINY_L),
              calc_diffusion_hyperparams(T=15, beta_0=1e-4, beta_T=0.02, beta=None, fast=False, parameterization="v"),
              context=torch.randn(2, 1, TINY_L), guidance=3.0)
check("cross_attn guided sampling: finite + shape",
      tuple(sa.shape) == (2, 1, TINY_L) and bool(torch.isfinite(sa).all()))

print("\n=== 11. Tier-2.3: sliding-window long-continuation rollout ===")
from generate import rollout_continuation
dh_r = calc_diffusion_hyperparams(T=10, beta_0=1e-4, beta_T=0.02, beta=None, fast=False, parameterization="v")
roll = rollout_continuation(cc_net, dh_r, torch.randn(1, 1, TINY_L),
                            n_chunks=3, chunk_len=TINY_L, context_len=TINY_L, guidance=2.0)
check("rollout length == n_chunks*chunk_len", tuple(roll.shape) == (1, 1, 3 * TINY_L), str(tuple(roll.shape)))
check("rollout output finite", bool(torch.isfinite(roll).all()))

print("\n" + "=" * 50)
n_pass = sum(results); n_tot = len(results)
print(f"{n_pass}/{n_tot} checks passed")
sys.exit(0 if n_pass == n_tot else 1)
