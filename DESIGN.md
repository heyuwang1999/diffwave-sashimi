# Design & roadmap

## Architecture decision

**Chosen: wrap the official `albertfgu/diffwave-sashimi` engine, add a thin owned layer.**

This is your recommended baseline (SaShiMi-backed DiffWave, raw waveform),
realized in the lowest-risk way.

### Options considered

| Option | Summary | Verdict |
|---|---|---|
| **A. Wrap upstream** | Vendor the official repo; add data + configs + continuation hook | **Chosen** |
| B. Clean reimplementation | Rebuild DiffWave + S4 backbone from scratch | Rejected for now — the S4 kernel is subtle; high time-to-baseline with no quality upside |
| C. Latent/token diffusion (e.g. EnCodec + transformer) | Different class of system | Rejected — abandons raw-waveform modeling, adds a heavy codec dependency; fails the project's "don't change the system class" guardrail |

### Why A remains the best starting point (your decision policy, 1→5)

1. **Working baseline fastest** — known-good code; pretrained SC09/LJSpeech
   checkpoints exist to sanity-check the pipeline before touching custom data.
2. **On the recommended direction** — it *is* SaShiMi+DiffWave, not an analogue.
3. **Debuggability** — the model is trusted; bugs localize to our small wrapper.
4. **Extensibility** — music-understanding objectives attach at the wrapper /
   conditioning layer without forking model internals.
5. We don't pay for sophistication we don't yet need.

### Tradeoff accepted
We live with upstream's style and its Hydra config system rather than rewriting
internals. Commands run from `vendor/` because Hydra's `config_path` and the
relative imports (`from models import ...`) are anchored there. We keep our
additions physically isolated and clearly commented so upstream stays pullable.

### Target operating point
22.05 kHz mono, ~2 s clips (`segment_length=44032`), single A100/L4. This is
~2.75× the sequence length of the paper's SC09 setting (16 kHz, 1 s) — squarely
the long-sequence regime S4/SaShiMi is designed for. We start at `d_model=64`
for memory safety and scale to 128 once training is confirmed.

## Diffusion + backbone (as used)

- **DiffWave**: DDPM with ε-prediction, `T=200`, linear β schedule
  `[1e-4, 0.02]`. Training noises a clip to a random step `t` and regresses the
  added noise (`train.py:training_loss`). Sampling is the standard reverse loop
  (`generate.py:sampling`).
- **SaShiMi backbone**: a U-Net of S4 blocks. Each `DiffWaveBlock` is
  `LN → +diffusion-step-embed → bidirectional S4 → (+conditioning) → residual →
  LN → FF → residual`. Pooling (`pool=[4,4]`) gives a 3-stage multi-scale UNet.

## Milestone 2 — continuation conditioning (design)

**Problem.** Given a context clip `c` (the seconds *before* the target), generate
a coherent continuation `x`.

**Insertion point.** `DiffWaveBlock.forward(x, diffusion_step_embed, mel_spec=...)`
already adds a per-block conditioning signal: it upsamples `mel_spec` to the
audio length, `conv1x1`s it to `d_model`, and adds it to the S4 output
(`sashimi.py:160-175`). The diffusion-step embedding is injected the same way
but **broadcast** (a single vector added across time, `sashimi.py:151-152`).
These two patterns give us two clean conditioning styles without touching the
diffusion math or the UNet plumbing.

**Approach (baseline → richer):**

1. **Global (FiLM-style) conditioning — baseline.**
   A `ContextEncoder` maps `c` → a single `d_cond`-vector summary. Project it
   per block (like `fc_t`) and **broadcast-add** to the block activations.
   - Why first: the context is the *past*, the target is the *future* — they are
     not time-aligned, so a global summary ("what kind of music, key, energy,
     timbre") is the honest baseline and is robust.
   - Plumbing: pass a `cond` tensor alongside `mel_spec` (or reuse the kwarg)
     and add `self.fc_cond(cond).unsqueeze(-1)` in the block.

2. **Encoder choice.** Reuse the same S4/SaShiMi machinery for the encoder so the
   model "hears" structure over the full context window (S4's strength), then
   pool over time to the summary vector. Start with a lightweight stack
   (a few S4 or dilated-conv blocks + mean/attention pool) — see
   `music_continuation/context_encoder.py`.

3. **Time-aligned / cross-attention — later.** If global conditioning underfits
   local timbral evolution, add cross-attention from each block's activations to
   a *sequence* of context features (not a single vector). This is the natural
   bridge toward the music-understanding objective.

4. **Data.** A continuation dataset yields `(context, target)` = two adjacent
   crops from the same track (target immediately follows context). The held-out
   tail produced by `prepare_data.py --holdout_frac` is reserved for eval so we
   never continue from trained audio.

5. **Training.** Identical ε-prediction loss; the only change is the model now
   receives `cond = ContextEncoder(context)`. Optionally **context dropout**
   (randomly zero the context) so one model does both unconditional and
   conditioned generation and we can measure the lift from context.

**Why this stays aligned.** Still raw-waveform DiffWave+SaShiMi; continuation is
an additive conditioning path, not a new system class.

## Research direction (after baseline): music understanding

The cross-attention context path (3) is where structure/timbre understanding
lives. Candidate extensions, in increasing ambition:
- auxiliary objectives on the context encoder (predict key/tempo/chroma) to shape
  a musically meaningful summary;
- longer context via S4's long-range strength (continuation conditioned on far
  more than 2 s);
- hierarchical conditioning (bar/phrase-level summary + local features).

All of these attach to the encoder/conditioning layer and leave the diffusion
backbone intact.

## Known notes / gotchas (verified via `scripts/smoke_test.py`)
- Install `pykeops` (in `requirements-colab.txt`) so S4 uses the fast Cauchy
  kernel; otherwise it warns and falls back to a slow path (still correct).
- Do **not** pip-install torch/torchaudio on Colab — use the preinstalled
  CUDA-matched build.
- `s4.py` imports `pytorch_lightning` (only `rank_zero_only`); it's in the reqs.
- **Upstream `sc.py` is import-fragile** (pulls in `torchvision` and a
  `torchaudio.datasets.utils` API removed in torchaudio≥2.x). We made the dataset
  imports in `dataloaders/__init__.py` lazy so the `music` path never touches it.
- **torchaudio I/O backend**: torchaudio≥2.11 routes load/save through
  `torchcodec`. The music loader (`load_audio`) falls back to `soundfile`
  (wav/flac/ogg) when the torchaudio backend is missing; mp3/m4a still need a
  working torchaudio backend (torchcodec/ffmpeg).
- **`diffusion_step_embed_dim_out` is effectively locked to 512** upstream
  (`DiffWaveBlock` hardcodes the default and `Sashimi._residual` doesn't forward
  it). Our configs use 512 to match. Changing it requires patching the model.

## Smoke test
`scripts/smoke_test.py` runs on CPU (no GPU/pykeops needed; it no-ops `.cuda()`)
and exercises: the dataset (incl. the short-file tiling path), the `dataloader()`
dispatch + batching, the milestone-2 `ContextEncoder`/`FiLM`, and a *real*
forward + DiffWave training loss + backward + reverse-sampling loop through the
vendored Sashimi model on a tiny config. Run from `vendor/`:
`python ../scripts/smoke_test.py`. All 19 checks pass.
