# diffwave-sashimi — music continuation

Raw-waveform music generation and short-clip **continuation**, built on
**SaShiMi + DiffWave** (state-space diffusion over audio waveforms).

This project wraps the official reference implementation
([`albertfgu/diffwave-sashimi`](https://github.com/albertfgu/diffwave-sashimi),
from *"It's Raw! Audio Generation with State-Space Models"*, Goel et al. 2022)
as the model/training engine, and adds a thin, owned layer on top: a music data
pipeline, configs, and (milestone 2) a continuation conditioning path.

See **[DESIGN.md](DESIGN.md)** for the architecture decision, tradeoffs, and the
continuation roadmap.

## Layout

```
vendor/                         # upstream engine (cloned, run commands from here)
  dataloaders/music.py          #  + our chunking dataset for a few long tracks
  dataloaders/__init__.py       #  + "music" registered in the dispatch
  configs/dataset/music.yaml    #  + milestone-1 configs
  configs/model/sashimi_music.yaml
  configs/experiment/music.yaml
scripts/prepare_data.py         # validate / re-encode / hold-out split (optional)
scripts/smoke_test.py           # CPU soundness test (dataset + real model fwd/loss/sampling)
scripts/evaluate.py             # A/B metrics: fidelity (FMD) + continuation faithfulness
scripts/eval_smoke_test.py      # behavioral test for the metrics
music_eval/                     # model-free eval metrics (mel-Frechet, seam, style)
music_continuation/             # milestone-2 design + context-encoder scaffold
ROADMAP_V2.md                   # prioritized plan to beat the official baseline
notebooks/colab_quickstart.ipynb
requirements-colab.txt
```

Everything we own is `vendor/dataloaders/music.py`, the three `configs/*/music*.yaml`
files, one registration line in `vendor/dataloaders/__init__.py`, and the
`scripts/` + `music_continuation/` packages. The upstream model code is untouched.

## Milestone 1 — unconditional generation (ready)

Target spec (your choices): **22.05 kHz mono, ~2 s clips, single A100/L4**.

1. **Get the data in.** Drop your few long tracks into `vendor/data/music/`
   (any of wav/flac/mp3/ogg/m4a). They are resampled to 22.05 kHz and
   random-cropped to 2 s on the fly. Optional one-time clean-up + stats:
   ```bash
   python scripts/prepare_data.py --in_dir raw_audio --out_dir vendor/data/music --sr 22050 --holdout_frac 0.1
   ```
2. **Train** (from `vendor/`, where Hydra and the relative imports expect to run):
   ```bash
   cd vendor
   python train.py experiment=music                       # defaults: d_model=64, batch 8
   python train.py experiment=music model.d_model=128     # bump quality once it's training
   python train.py experiment=music wandb.mode=online     # turn on logging + sample uploads
   ```
   Checkpoints → `vendor/exp/<run>/checkpoint/`. Samples auto-generate every
   `iters_per_ckpt`.
3. **Generate** unconditional samples from a checkpoint:
   ```bash
   cd vendor && python generate.py experiment=music generate.ckpt_iter=max generate.n_samples=8
   ```

On Colab, see `notebooks/colab_quickstart.ipynb`. Install deps with
`pip install -r requirements-colab.txt` (torch/torchaudio come with Colab).

### Key constraint
`dataset.segment_length` **must be divisible by `prod(model.pool)`** (16 for
`pool=[4,4]`). Default `44032 = 16 × 2752 ≈ 1.997 s`. The SaShiMi UNet pools the
sequence and needs integer lengths at every stage.

## Milestone 2 — continuation (designed, not yet wired)

Encode the *preceding* audio with a context encoder and inject it through
DiffWave's existing per-block conditioning input (the same hook the vocoder uses
for mel spectrograms). Scaffold + full design in
[`music_continuation/`](music_continuation/) and [DESIGN.md](DESIGN.md).
