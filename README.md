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

## Recommended: train once → generate any length (outpainting)

For "train on a few hours of clips, then generate audio of any length," use the
**outpainting** model (`experiment=music_outpaint`). One model is trained to both
generate unconditionally *and* continue from a clean in-sequence prefix; at
inference it rolls out seamlessly to any duration by carrying each window's
waveform tail as the next window's clean context (sample-continuous, no seams).
It covers **both** pure generation and continuation.

```bash
cd vendor
python train.py experiment=music_outpaint train.batch_size_per_gpu=4
# pure generation, any length:
python generate.py experiment=music_outpaint generate.ckpt_iter=max generate.gen_seconds=30
# continue a clip:
python generate.py experiment=music_outpaint generate.ckpt_iter=max generate.gen_seconds=30 \
    generate.context_path=data/music_holdout/<clip>.wav
# faster sampling: add  generate.sampler=ddim generate.sampling_steps=30
```

The single Colab notebook `notebooks/colab_quickstart.ipynb` runs this end to end.
The milestones below document the other (earlier) approaches the repo also supports.

## Milestone 1 — unconditional generation (ready)

Target spec (your choices): **22.05 kHz mono, ~2 s clips, single A100/L4**.

1. **Get the data in.** Point `prepare_data.py` at a single long file *or* a
   folder, in any format (wav/flac/mp3/m4a/ogg). It decodes → mono → resamples to
   22.05 kHz → reserves a held-out tail, then training random-crops 2 s clips on
   the fly. With ffmpeg present (default on Colab) it streams with **bounded
   memory**, so a multi-hour file works on a normal runtime:
   ```bash
   python scripts/prepare_data.py --in_dir /path/to/mymix.mp3 --out_dir vendor/data/music --sr 22050 --holdout_frac 0.1
   ```
   (The training dataset loads the cleaned audio into RAM — fine up to a few hours
   of mono 22.05 kHz, ~1.3 GB for 4 h.)
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

## Milestone 2 — continuation (implemented)

A context encoder summarizes the preceding audio and conditions generation
(global / FiLM or bottleneck cross-attention), trained with classifier-free
guidance. Train and generate:
```bash
cd vendor
python train.py experiment=music_continuation          # global conditioning
python train.py experiment=music_continuation_xattn     # cross-attention variant
python generate.py experiment=music_continuation generate.ckpt_iter=max \
    generate.context_path=data/music_holdout/<track>.wav generate.guidance=3.0 generate.gen_seconds=60
```

### Custom output length
Unconditional samples are fixed at `segment_length` (~2 s) by design. For
**arbitrary length**, use continuation: `generate.gen_seconds=N` (or
`generate.rollout_chunks=K`) slides the window — generate a chunk, feed it back
as context, repeat — to produce audio of any duration (trimmed to exactly
`gen_seconds`). Design details in [ROADMAP_V2.md](ROADMAP_V2.md) and
[DESIGN.md](DESIGN.md).
