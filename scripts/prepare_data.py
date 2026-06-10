#!/usr/bin/env python3
"""
Prepare a music dataset for unconditional SaShiMi+DiffWave training.

The training dataset (`vendor/dataloaders/music.py`) chunks long files on the
fly, so preprocessing is OPTIONAL. But running this once is recommended because
it:
  1. validates every file decodes,
  2. reports total duration and how many ~2 s segments you effectively have,
  3. (optionally) re-encodes everything to clean mono WAV at the target sample
     rate, which makes Colab training start faster and avoids relying on an mp3
     decode backend at train time,
  4. (optionally) reserves the tail of each track as a held-out split for the
     milestone-2 continuation evaluation (so eval context is never trained on).

Examples
--------
  # Just inspect what you have:
  python scripts/prepare_data.py --in_dir raw_audio --report_only

  # Re-encode to data/music/ (mono, 22050 Hz wav), holding out the last 10%:
  python scripts/prepare_data.py --in_dir raw_audio --out_dir vendor/data/music \\
      --sr 22050 --holdout_frac 0.1
"""
import argparse
from pathlib import Path

import torch
import torchaudio

AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif")


def load_audio(path):
    """(channels, time) float32 tensor + sr; torchaudio with soundfile fallback."""
    try:
        return torchaudio.load(path)
    except Exception as e:
        try:
            import soundfile as sf
        except ImportError:
            raise RuntimeError(f"Cannot load {path!r} ({e}); install soundfile or torchcodec.")
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        return torch.from_numpy(data.T).contiguous(), sr


def save_wav(path, wav, sr):
    """Save a (channels, time) tensor as WAV; torchaudio with soundfile fallback."""
    try:
        torchaudio.save(path, wav, sr)
    except Exception:
        import soundfile as sf
        sf.write(path, wav.T.cpu().numpy(), sr)  # soundfile expects (time, channels)


def list_audio(in_dir):
    p = Path(in_dir)
    if p.is_file():
        return [p]
    return sorted(f for f in p.glob("**/*") if f.suffix.lower() in AUDIO_EXTENSIONS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_dir", required=True, help="Folder (recursive) or single file of source audio.")
    ap.add_argument("--out_dir", default=None, help="Where to write cleaned wavs. Omit for --report_only.")
    ap.add_argument("--sr", type=int, default=22050, help="Target sample rate.")
    ap.add_argument("--segment_length", type=int, default=44032,
                    help="Clip length in samples, for the segment-count report. Must be divisible by prod(pool).")
    ap.add_argument("--holdout_frac", type=float, default=0.0,
                    help="Fraction of the END of each track to reserve as a held-out split (for continuation eval).")
    ap.add_argument("--report_only", action="store_true", help="Only print stats; write nothing.")
    args = ap.parse_args()

    files = list_audio(args.in_dir)
    if not files:
        raise SystemExit(f"No audio files found under {args.in_dir!r} (extensions: {AUDIO_EXTENSIONS})")

    out_train = Path(args.out_dir) if args.out_dir else None
    out_holdout = (out_train.parent / (out_train.name + "_holdout")) if (out_train and args.holdout_frac > 0) else None
    if not args.report_only and out_train:
        out_train.mkdir(parents=True, exist_ok=True)
        if out_holdout:
            out_holdout.mkdir(parents=True, exist_ok=True)

    total_sec = 0.0
    resamplers = {}
    print(f"Found {len(files)} file(s).")
    for f in files:
        wav, sr = load_audio(str(f))
        wav = wav.mean(dim=0, keepdim=True)  # mono
        if sr != args.sr:
            if sr not in resamplers:
                resamplers[sr] = torchaudio.transforms.Resample(sr, args.sr)
            wav = resamplers[sr](wav)
        dur = wav.shape[1] / args.sr
        total_sec += dur
        print(f"  {f.name:40s} {dur:8.1f}s  ({wav.shape[1]} samples @ {args.sr} Hz)")

        if args.report_only or out_train is None:
            continue

        n = wav.shape[1]
        split = int(n * (1.0 - args.holdout_frac)) if args.holdout_frac > 0 else n
        train_part = wav[:, :split]
        # peak-normalize for consistent loudness across the corpus
        peak = train_part.abs().max()
        if peak > 1e-8:
            train_part = train_part / peak
        save_wav(str(out_train / (f.stem + ".wav")), train_part, args.sr)
        if out_holdout is not None and split < n:
            hold = wav[:, split:]
            hpeak = hold.abs().max()
            if hpeak > 1e-8:
                hold = hold / hpeak
            save_wav(str(out_holdout / (f.stem + ".wav")), hold, args.sr)

    n_seg = int(total_sec * args.sr // args.segment_length)
    print("-" * 60)
    print(f"Total: {total_sec/60:.1f} min  ->  ~{n_seg} non-overlapping "
          f"{args.segment_length/args.sr:.2f}s segments")
    if n_seg < 200:
        print("WARNING: very little data. Unconditional waveform diffusion typically "
              "needs at least tens of minutes of audio to sound musical.")
    if not args.report_only and out_train:
        print(f"Wrote cleaned wavs to: {out_train}")
        if out_holdout:
            print(f"Held-out (continuation eval) wavs: {out_holdout}")
        print(f"\nTrain with:\n  cd vendor && python train.py experiment=music "
              f"dataset.data_path={out_train.resolve()}")


if __name__ == "__main__":
    main()
