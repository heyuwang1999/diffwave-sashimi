"""Behavioral smoke test for music_eval metrics (CPU). Run from repo root:
    python scripts/eval_smoke_test.py
Checks the metrics move in the right direction on controlled synthetic signals.
"""
import math, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from music_eval import (
    frechet_mel_distance, reference_mel_distance, seam_smoothness, style_consistency,
)

SR = 22050
PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
res = []
def check(name, cond, extra=""):
    res.append(cond); print(f"  [{PASS if cond else FAIL}] {name} {extra}")

def tone(freq, secs=1.0, amp=0.5, phase=0.0):
    t = torch.arange(int(secs * SR)) / SR
    return amp * torch.sin(2 * math.pi * freq * t + phase)

def noise(secs=1.0, amp=0.5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return amp * (2 * torch.rand(int(secs * SR), generator=g) - 1)

print("=== FMD (Frechet Mel Distance) ===")
tones_a = [tone(220, phase=p) for p in (0.0, 0.3, 0.6, 0.9)]
tones_b = [tone(220, phase=p) for p in (0.1, 0.4, 0.7, 1.0)]
noises = [noise(seed=i) for i in range(4)]
fmd_same = frechet_mel_distance(tones_a, tones_b, SR)
fmd_diff = frechet_mel_distance(tones_a, noises, SR)
check("FMD(tone,tone) small", fmd_same < 5.0, f"={fmd_same:.3f}")
check("FMD(tone,noise) >> FMD(tone,tone)", fmd_diff > fmd_same * 5, f"diff={fmd_diff:.1f} same={fmd_same:.3f}")
check("FMD is non-negative", fmd_same >= -1e-3 and fmd_diff >= 0)

print("=== reference_mel_distance ===")
d_id = reference_mel_distance(tone(220), tone(220), SR)
d_dif = reference_mel_distance(tone(220), tone(880), SR)
check("ref dist identical ~0", d_id < 1e-3, f"={d_id:.5f}")
check("ref dist different > identical", d_dif > d_id, f"diff={d_dif:.3f}")

print("=== seam_smoothness (continuation join) ===")
# continuous sine spanning the boundary -> smooth seam
long = tone(220, secs=2.0)
ctx_smooth, con_smooth = long[: SR], long[SR:]          # split a single continuous tone
# discontinuity: context is a tone, continuation jumps to a different freq + phase
ctx_disc, con_disc = tone(220, 1.0), tone(523, 1.0, phase=1.7)
s_smooth = seam_smoothness(ctx_smooth, con_smooth, SR)
s_disc = seam_smoothness(ctx_disc, con_disc, SR)
check("smooth seam ~ internal (close to 1)", s_smooth < 3.0, f"={s_smooth:.2f}")
check("discontinuous seam > smooth seam", s_disc > s_smooth, f"disc={s_disc:.2f} smooth={s_smooth:.2f}")

print("=== style_consistency ===")
c_same = style_consistency(tone(220), tone(220, phase=0.5), SR)
c_diff = style_consistency(tone(220), tone(1760), SR)
check("same timbre -> ~0", c_same < 1e-2, f"={c_same:.4f}")
check("different timbre > same", c_diff > c_same, f"diff={c_diff:.4f}")

print("\n" + "=" * 40)
print(f"{sum(res)}/{len(res)} checks passed")
sys.exit(0 if all(res) else 1)
