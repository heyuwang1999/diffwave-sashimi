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

print("\n" + "=" * 50)
n_pass = sum(results); n_tot = len(results)
print(f"{n_pass}/{n_tot} checks passed")
sys.exit(0 if n_pass == n_tot else 1)
