# v2 roadmap — beating official DiffWave+SaShiMi (sample-level, no text)

Locked scope (from project decisions):
- **Optimize for: audio fidelity/cleanliness + continuation faithfulness.**
- **Compute: single A100/L4, bounded hours.** Favor recipe/training upgrades over
  scale. No Tier-3 cascade / multi-GPU for now.
- **Sequencing: baseline first, then layer upgrades with A/B measurement.**
- Hard constraints unchanged: raw waveform (sample level), no text input.

## Realistic ceiling
MusicGen/AudioLM-class quality comes from scale + tokens + text + huge data and
is out of reach single-GPU/raw-waveform/no-text. The achievable, audible win is
over the **2022-era official recipe** (ε-prediction, linear-β, T=200 ancestral,
S4), which leaves a lot on the table. Most gains live in the diffusion recipe and
the continuation formulation, not model size.

## Plan (each step is independently A/B-measured against the previous)

**M1 — baseline (done, ready to train).** Unconditional SaShiMi+DiffWave, 22.05 kHz,
~2 s. Establishes the reference numbers in the eval harness.

**Tier 0 — fidelity recipe upgrades (do first; low risk, no architecture change).**
Order by ROI:
1. **v-prediction + Min-SNR-γ loss weighting** (Salimans 2022; Hang 2023) — better
   fidelity and faster convergence (matters under bounded hours).
2. **EDM/Karras noise schedule + preconditioning** (Karras 2022).
3. **Self-conditioning** (Chen 2022 / RIN) — near-free quality bump.
4. **Multi-resolution STFT auxiliary loss** on predicted x₀ — sharper highs.
5. **DPM-Solver++ / DDIM sampling** — ~20–50 steps at parity with T=200 (faster
   iteration, doesn't change training).
Each lands behind a config flag so we can toggle and compare.

**Tier 2 — continuation faithfulness (the goal).**
1. **Cross-attention conditioning** from the noisy target to a context encoder
   (upgrade of the current FiLM/broadcast scaffold in `music_continuation/`).
2. **Classifier-free guidance on context** (train with random context dropout;
   scale guidance at inference) — the biggest lever for context adherence.
3. **Diffusion Forcing / AR-diffusion** (Chen 2024): per-frame noise levels with
   clean past + noisy future → stable arbitrarily-long rollout continuation.

**Tier 1 — backbone (only if Tier 0/2 plateau).** Swap S4→Mamba/S5 selective SSM;
add windowed self-attention at the UNet bottleneck (modern SSM+attention hybrid).

**Tier 3 — deferred** (cascade / super-resolution, RIN) until more compute.

## Evaluation protocol (`music_eval/`, `scripts/evaluate.py`)
A/B requires fixed metrics computed the same way each round.

Fidelity (unconditional + continuation outputs vs a held-out real reference set):
- **FMD — Fréchet Mel Distance**: Fréchet distance between log-mel frame
  statistics of generated vs real audio. A dependency-free FAD proxy (no learned
  embedding). *Gold standard FAD (VGGish/CLAP/PANNs embedding) is a documented
  drop-in upgrade when an embedding model is available — see metrics.py.*
- **Reference mel-stat distance**: L2 between mean/std log-mel vectors.

Continuation faithfulness (uses the held-out tails reserved by `prepare_data.py`):
- **Seam smoothness**: spectral jump at the context→continuation boundary,
  normalized by the internal frame-to-frame jump (~1 = inaudible seam, ≫1 = click).
- **Style consistency**: 1 − cosine similarity of mean log-mel between context and
  continuation (0 = same timbre/energy).
- **Distance-to-true-continuation**: for held-out clips we have the *real* future;
  compare distributionally (stochastic, so treated as a soft signal).

Always also: **listen** (the metrics are proxies, not ground truth) and report
GPU-hours per result so ROI stays visible under the bounded-compute constraint.
