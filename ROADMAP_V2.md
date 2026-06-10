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
Order by ROI (✅ = implemented, config-gated, smoke-tested):
1. ✅ **v-prediction + Min-SNR-γ loss weighting** (Salimans 2022; Hang 2023) —
   `diffusion.parameterization=v`, `diffusion.min_snr_gamma=5.0`.
2. ✅ **Cosine noise schedule** (Nichol & Dhariwal 2021) — `diffusion.schedule=cosine`.
   (Full EDM/Karras preconditioning deferred — bigger reformulation, lower marginal
   gain than the items here.)
3. ✅ **Self-conditioning** (Chen 2022) — `model.self_conditioning=true` (changes
   input channels → use a distinct `train.name`; not checkpoint-compatible with baseline).
4. ✅ **Multi-resolution STFT auxiliary loss** on x₀ (Yamamoto 2020) —
   `diffusion.stft_loss_weight=0.1` (directly targets cleanliness).
5. ⬜ **DPM-Solver++ / DDIM sampling** — deprioritized (speed, not your ranked axis);
   easy drop-in later for faster iteration.
Each is a config flag, defaults == original DiffWave, so the baseline is unchanged.

**Tier 2 — continuation faithfulness (the goal).** (✅ = implemented, smoke-tested)
0. ✅ **Global context conditioning**: a ContextEncoder summarizes the preceding
   audio; the vector is added to the diffusion-step embedding (FiLM-style, the
   robust baseline). `experiment=music_continuation` (model.context_conditioning).
1. ✅ **Classifier-free guidance**: train with context dropout
   (`diffusion.context_cfg_dropout`), scale at inference (`sampling(..., guidance=w)`).
2. ✅ **Cross-attention conditioning**: the UNet bottleneck attends to a sequence
   of context tokens (richer than the global vector). `model.context_mode=cross_attn`
   or `experiment=music_continuation_xattn`. A/B vs the global baseline.
3. ⬜ **Diffusion Forcing / AR-diffusion** (Chen 2024): per-frame noise levels with
   clean past + noisy future → stable arbitrarily-long rollout continuation.

✅ **Continuation generate.py CLI**: `generate.context_path=<clip> generate.guidance=3.0`
conditions on the last `context_length` samples of the clip and writes both the
continuation and a context+continuation concat (`*_full.wav`) for listening.
(Auto-samples during training remain unconditional — harmless.)

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
