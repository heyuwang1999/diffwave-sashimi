#!/usr/bin/env python3
"""
Evaluate generated audio for A/B comparison of model variants (see ROADMAP_V2.md).

Two modes:

  # Fidelity: generated set vs a held-out real reference set
  python scripts/evaluate.py fidelity --gen_dir vendor/exp/<run>/waveforms/<iter> \\
      --ref_dir vendor/data/music --sr 22050

  # Continuation: context clips -> model continuations, optionally vs true future.
  # Files are matched by stem across the directories.
  python scripts/evaluate.py continuation --context_dir ctx --cont_dir gen \\
      [--ref_dir vendor/data/music_holdout] --sr 22050

Metrics are model-free proxies; always also LISTEN. Report GPU-hours alongside.
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from music_eval import (  # noqa: E402
    frechet_mel_distance, reference_mel_distance, seam_smoothness, style_consistency,
)

AUDIO_EXTENSIONS = (".wav", ".flac", ".ogg", ".mp3", ".m4a")


def load_audio(path):
    try:
        import torchaudio
        return torchaudio.load(str(path))
    except Exception:
        import soundfile as sf
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(data.T).contiguous(), sr


def list_wavs(d):
    return sorted(p for p in Path(d).glob("**/*") if p.suffix.lower() in AUDIO_EXTENSIONS)


def load_set(d):
    return [load_audio(p)[0].mean(0) for p in list_wavs(d)]


def by_stem(d):
    return {p.stem: p for p in list_wavs(d)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    f = sub.add_parser("fidelity")
    f.add_argument("--gen_dir", required=True)
    f.add_argument("--ref_dir", required=True)
    f.add_argument("--sr", type=int, default=22050)

    c = sub.add_parser("continuation")
    c.add_argument("--context_dir", required=True)
    c.add_argument("--cont_dir", required=True)
    c.add_argument("--ref_dir", default=None, help="True continuations (held-out), matched by stem.")
    c.add_argument("--sr", type=int, default=22050)

    args = ap.parse_args()

    if args.mode == "fidelity":
        gen, ref = load_set(args.gen_dir), load_set(args.ref_dir)
        if not gen or not ref:
            raise SystemExit("Need audio in both --gen_dir and --ref_dir.")
        fmd = frechet_mel_distance(gen, ref, args.sr)
        print(f"n_gen={len(gen)} n_ref={len(ref)}")
        print(f"FMD (Frechet Mel Distance, lower better): {fmd:.4f}")

    elif args.mode == "continuation":
        ctx, con = by_stem(args.context_dir), by_stem(args.cont_dir)
        ref = by_stem(args.ref_dir) if args.ref_dir else {}
        stems = sorted(set(ctx) & set(con))
        if not stems:
            raise SystemExit("No matching stems between --context_dir and --cont_dir.")
        seam, style, refd = [], [], []
        for s in stems:
            cwav = load_audio(ctx[s])[0].mean(0)
            gwav = load_audio(con[s])[0].mean(0)
            seam.append(seam_smoothness(cwav, gwav, args.sr))
            style.append(style_consistency(cwav, gwav, args.sr))
            if s in ref:
                refd.append(reference_mel_distance(gwav, load_audio(ref[s])[0].mean(0), args.sr))
        mean = lambda xs: float(torch.tensor(xs).nanmean()) if xs else float("nan")
        print(f"matched pairs: {len(stems)}")
        print(f"seam smoothness   (~1 ideal, >>1 = audible seam): {mean(seam):.3f}")
        print(f"style consistency (0 = same timbre, lower better): {mean(style):.4f}")
        if refd:
            print(f"dist-to-true-cont (lower better, soft signal):    {mean(refd):.4f}")


if __name__ == "__main__":
    main()
