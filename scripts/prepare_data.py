#!/usr/bin/env python3
"""
Prepare a music dataset for SaShiMi+DiffWave training.

Handles everything from raw source audio to train-ready clips, automatically:
  - any format (wav/flac/mp3/m4a/ogg/...) and any length, including a single
    multi-hour file,
  - decode -> mono -> resample to the target rate,
  - reserve the tail of each track as a held-out split for continuation eval.

Two engines (auto-selected):
  - "ffmpeg" (default when ffmpeg is on PATH): streams through ffmpeg + soundfile
    with BOUNDED MEMORY, so a 4-hour file works on a normal Colab runtime. Robust
    to mp3/m4a. Recommended for long files.
  - "memory": the simple all-in-RAM path (torchaudio/soundfile). Fine for short
    files / quick checks; can OOM on multi-hour sources.

Per-clip peak normalization happens at train time (in the dataset), so this step
does not need to normalize.

Examples
--------
  # One long file on Drive -> train-ready clips, last 10% held out:
  python scripts/prepare_data.py --in_dir /content/drive/MyDrive/mymix.mp3 \\
      --out_dir vendor/data/music --sr 22050 --holdout_frac 0.1

  # Just inspect duration / segment count:
  python scripts/prepare_data.py --in_dir raw_audio --report_only
"""
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif", ".opus")


def list_audio(in_dir):
    p = Path(in_dir)
    if p.is_file():
        return [p]
    return sorted(f for f in p.glob("**/*") if f.suffix.lower() in AUDIO_EXTENSIONS)


def has_ffmpeg():
    return shutil.which("ffmpeg") is not None


def ffprobe_duration(src):
    """Duration in seconds via ffprobe, without decoding the audio."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nokey=1:noprint_wrappers=1", str(src)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out) if out else 0.0


def ffmpeg_to_mono_wav(src, dst, sr):
    """Decode any format -> mono, target-rate, 16-bit PCM wav. Streams in ffmpeg
    (bounded memory), so multi-hour files are fine."""
    subprocess.run(
        ["ffmpeg", "-y", "-vn", "-i", str(src), "-ac", "1", "-ar", str(sr),
         "-c:a", "pcm_s16le", str(dst)],
        check=True, capture_output=True,
    )


def stream_split(src_wav, train_path, holdout_path, holdout_frac, block=1 << 20):
    """Split a wav into train (head) + optional holdout (tail) by streaming blocks
    through soundfile, so the whole file is never held in RAM at once. Returns the
    total duration in seconds."""
    import soundfile as sf
    info = sf.info(str(src_wav))
    n, sr, ch, sub = info.frames, info.samplerate, info.channels, info.subtype
    split = int(n * (1.0 - holdout_frac)) if holdout_frac > 0 else n
    with sf.SoundFile(str(src_wav)) as f_in:
        with sf.SoundFile(str(train_path), "w", samplerate=sr, channels=ch, subtype=sub) as f_tr:
            remaining = split
            while remaining > 0:
                blk = f_in.read(min(block, remaining), dtype="float32", always_2d=True)
                if len(blk) == 0:
                    break
                f_tr.write(blk)
                remaining -= len(blk)
        if holdout_path is not None and split < n:
            with sf.SoundFile(str(holdout_path), "w", samplerate=sr, channels=ch, subtype=sub) as f_ho:
                while True:
                    blk = f_in.read(block, dtype="float32", always_2d=True)
                    if len(blk) == 0:
                        break
                    f_ho.write(blk)
    return n / sr


# ---- memory engine (all-in-RAM fallback; uses torch/torchaudio) -------------

def load_audio(path):
    import torch, torchaudio
    try:
        return torchaudio.load(path)
    except Exception as e:
        try:
            import soundfile as sf
        except ImportError:
            raise RuntimeError(f"Cannot load {path!r} ({e}); install soundfile or ffmpeg.")
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        return torch.from_numpy(data.T).contiguous(), sr


def save_wav(path, wav, sr):
    import torchaudio
    try:
        torchaudio.save(path, wav, sr)
    except Exception:
        import soundfile as sf
        sf.write(path, wav.T.cpu().numpy(), sr)


def process_memory(f, out_train, out_holdout, sr, holdout_frac):
    import torch, torchaudio
    wav, in_sr = load_audio(str(f))
    wav = wav.mean(dim=0, keepdim=True)
    if in_sr != sr:
        wav = torchaudio.transforms.Resample(in_sr, sr)(wav)
    dur = wav.shape[1] / sr
    if out_train is not None:
        n = wav.shape[1]
        split = int(n * (1.0 - holdout_frac)) if holdout_frac > 0 else n
        save_wav(str(out_train / (f.stem + ".wav")), wav[:, :split], sr)
        if out_holdout is not None and split < n:
            save_wav(str(out_holdout / (f.stem + ".wav")), wav[:, split:], sr)
    return dur


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_dir", required=True, help="Folder (recursive) or a single audio file.")
    ap.add_argument("--out_dir", default=None, help="Where to write cleaned wavs. Omit for --report_only.")
    ap.add_argument("--sr", type=int, default=22050, help="Target sample rate.")
    ap.add_argument("--segment_length", type=int, default=44032,
                    help="Clip length in samples, for the segment-count report. Divisible by prod(pool).")
    ap.add_argument("--holdout_frac", type=float, default=0.0,
                    help="Fraction of the END of each track reserved as a held-out split.")
    ap.add_argument("--engine", choices=["auto", "ffmpeg", "memory"], default="auto",
                    help="auto = ffmpeg if available (memory-bounded; best for long files) else memory.")
    ap.add_argument("--report_only", action="store_true", help="Only print stats; write nothing.")
    args = ap.parse_args()

    files = list_audio(args.in_dir)
    if not files:
        raise SystemExit(f"No audio files found under {args.in_dir!r} (extensions: {AUDIO_EXTENSIONS})")

    engine = args.engine
    if engine == "auto":
        engine = "ffmpeg" if has_ffmpeg() else "memory"
    print(f"engine: {engine} ({len(files)} file(s))")

    out_train = Path(args.out_dir) if args.out_dir else None
    out_holdout = (out_train.parent / (out_train.name + "_holdout")) if (out_train and args.holdout_frac > 0) else None
    if not args.report_only and out_train:
        out_train.mkdir(parents=True, exist_ok=True)
        if out_holdout:
            out_holdout.mkdir(parents=True, exist_ok=True)

    total_sec = 0.0
    for f in files:
        if args.report_only:
            dur = ffprobe_duration(f) if engine == "ffmpeg" else process_memory(f, None, None, args.sr, 0.0)
        elif engine == "ffmpeg":
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td) / "clean.wav"
                ffmpeg_to_mono_wav(f, tmp, args.sr)             # bounded-memory decode+resample
                dur = stream_split(tmp, out_train / (f.stem + ".wav"),
                                   (out_holdout / (f.stem + ".wav")) if out_holdout else None,
                                   args.holdout_frac)            # bounded-memory split
        else:
            dur = process_memory(f, out_train, out_holdout, args.sr, args.holdout_frac)
        total_sec += dur
        print(f"  {f.name:40s} {dur/60:7.1f} min")

    n_seg = int(total_sec * args.sr // args.segment_length)
    print("-" * 60)
    print(f"Total: {total_sec/60:.1f} min  ->  ~{n_seg} non-overlapping "
          f"{args.segment_length/args.sr:.2f}s segments")
    if n_seg < 200:
        print("WARNING: little data; unconditional waveform diffusion usually needs tens of minutes+.")
    if not args.report_only and out_train:
        print(f"Wrote cleaned wavs to: {out_train}")
        if out_holdout:
            print(f"Held-out (continuation eval) wavs: {out_holdout}")
        print(f"\nTrain with:\n  cd vendor && python train.py experiment=music "
              f"dataset.data_path={out_train.resolve()}")


if __name__ == "__main__":
    main()
