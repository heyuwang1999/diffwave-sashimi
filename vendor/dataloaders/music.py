"""
Project addition (not part of upstream diffwave-sashimi).

A simple raw-waveform dataset for *unconditional* music generation when the
source data is "a few long audio files" rather than many pre-cut clips.

Design notes
------------
- The whole dataset is loaded into RAM once at construction time. A handful of
  long tracks is tiny in float32 (e.g. 30 min @ 22.05 kHz mono ~= 160 MB), so
  this is both simple and fast, and avoids per-item disk decode/resample.
- Each ``__getitem__`` returns a *random* crop, so one "epoch" is just a fixed
  number of random segments (``samples_per_epoch``); the notion of epoch is
  decoupled from file count. Files are sampled proportionally to their length.
- Returns a ``(waveform, sampling_rate, label)`` 3-tuple to match the
  unconditional batch unpacking in ``train.py`` (``audio, _, _ = data``).

The continuation dataset (milestone 2) lives separately so this stays minimal.
"""

import os
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset

AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif")


def load_audio(path):
    """Load audio as a (channels, time) float32 tensor and its sample rate.

    Prefers torchaudio; falls back to soundfile (libsndfile) when torchaudio's
    I/O backend is unavailable -- e.g. torchaudio>=2.11 routes I/O through
    torchcodec, which may not be installed. soundfile reliably handles
    wav/flac/ogg; mp3/m4a require a working torchaudio backend (torchcodec/ffmpeg).
    """
    try:
        return torchaudio.load(path)
    except Exception as e:
        try:
            import soundfile as sf
        except ImportError:
            raise RuntimeError(
                f"Could not load {path!r} via torchaudio ({e}); install `soundfile` "
                f"(wav/flac/ogg) or `torchcodec` (for mp3/m4a)."
            )
        data, sr = sf.read(path, dtype="float32", always_2d=True)  # (time, channels)
        return torch.from_numpy(data.T).contiguous(), sr


def _list_audio_files(data_path):
    """Accept a directory (searched recursively) or a single file path."""
    p = Path(data_path)
    if p.is_file():
        return [str(p)]
    files = sorted(
        str(f)
        for f in p.glob("**/*")
        if f.suffix.lower() in AUDIO_EXTENSIONS
    )
    return files


class MusicWaveform(Dataset):
    """Random fixed-length crops from a small set of (possibly long) audio files.

    Parameters
    ----------
    data_path : str
        Directory of audio files (searched recursively) or a single file.
    segment_length : int
        Number of samples per training clip. MUST be divisible by the product
        of the SaShiMi ``pool`` factors (e.g. 16 for pool=[4,4]); otherwise the
        UNet pooling will not produce integer lengths.
    sampling_rate : int
        Target sample rate. All inputs are resampled to this rate.
    samples_per_epoch : int, optional
        How many random crops constitute one epoch. Defaults to roughly the
        number of non-overlapping segments across all files (min 1000) so
        checkpoints/logging cadence stays reasonable on tiny datasets.
    peak_normalize : bool
        If True, scale each crop so its max abs value is ~1.0 (per-crop).
    seed : int, optional
        Only affects the deterministic fallback; per-worker randomness comes
        from torch's DataLoader worker seeding.
    """

    def __init__(
        self,
        data_path,
        segment_length=44032,
        sampling_rate=22050,
        samples_per_epoch=None,
        peak_normalize=True,
        **kwargs,
    ):
        super().__init__()
        self.segment_length = int(segment_length)
        self.sampling_rate = int(sampling_rate)
        self.peak_normalize = peak_normalize

        files = _list_audio_files(data_path)
        if not files:
            raise FileNotFoundError(
                f"No audio files found under '{data_path}'. "
                f"Supported extensions: {AUDIO_EXTENSIONS}"
            )

        self.clips = []        # list of 1-D float tensors (mono), at target SR
        self.labels = []       # filename stem, kept for parity / future use
        total_samples = 0
        resamplers = {}        # cache transforms by source sample rate

        for fp in files:
            wav, sr = load_audio(fp)                 # (channels, time)
            wav = wav.mean(dim=0, keepdim=True)      # -> mono (1, time)
            if sr != self.sampling_rate:
                if sr not in resamplers:
                    resamplers[sr] = torchaudio.transforms.Resample(sr, self.sampling_rate)
                wav = resamplers[sr](wav)
            wav = wav.squeeze(0).contiguous()        # (time,)
            # Skip silent/empty files
            if wav.numel() == 0:
                continue
            self.clips.append(wav)
            self.labels.append(Path(fp).stem)
            total_samples += wav.numel()

        if not self.clips:
            raise RuntimeError(f"All audio files under '{data_path}' were empty.")

        # Sampling weights proportional to clip length (longer files seen more).
        lengths = torch.tensor([c.numel() for c in self.clips], dtype=torch.float)
        self.file_weights = lengths / lengths.sum()

        if samples_per_epoch is None:
            samples_per_epoch = max(1000, int(total_samples // self.segment_length))
        self.samples_per_epoch = int(samples_per_epoch)

        print(
            f"[MusicWaveform] {len(self.clips)} file(s), "
            f"{total_samples / self.sampling_rate / 60:.1f} min @ {self.sampling_rate} Hz, "
            f"segment_length={self.segment_length} "
            f"({self.segment_length / self.sampling_rate:.2f}s), "
            f"samples_per_epoch={self.samples_per_epoch}"
        )

    def __len__(self):
        return self.samples_per_epoch

    def _random_crop(self, wav):
        n = wav.numel()
        if n < self.segment_length:
            # Tile to reach the target length (robust for clips far shorter than
            # segment_length, where reflect-padding would error). Then random-crop
            # the tiled signal so we still see varied phase across draws.
            reps = (self.segment_length + n - 1) // n + 1
            wav = wav.repeat(reps)
            n = wav.numel()
        if n == self.segment_length:
            return wav
        start = int(torch.randint(0, n - self.segment_length + 1, (1,)).item())
        return wav[start:start + self.segment_length]

    def __getitem__(self, index):
        # `index` is ignored for the file choice; each draw is a fresh random crop.
        file_idx = int(torch.multinomial(self.file_weights, 1).item())
        wav = self._random_crop(self.clips[file_idx])

        if self.peak_normalize:
            peak = wav.abs().max()
            if peak > 1e-8:
                wav = wav / peak

        waveform = wav.unsqueeze(0)  # (1, segment_length) -> matches C=1 convention
        return waveform, self.sampling_rate, self.labels[file_idx]


class MusicContinuation(Dataset):
    """Continuation pairs: (target, context) where `target` is the segment that
    immediately follows `context` within the same track. For milestone-2
    short-clip continuation. Returns (target[1,L], context[1,L_ctx]) so the train
    loop can unpack `audio, context = data`.

    Files shorter than context_length+segment_length are tiled up to fit.
    """

    def __init__(self, data_path, segment_length=44032, context_length=None,
                 sampling_rate=22050, samples_per_epoch=None, peak_normalize=True, **kwargs):
        super().__init__()
        self.segment_length = int(segment_length)
        self.context_length = int(context_length) if context_length else int(segment_length)
        self.sampling_rate = int(sampling_rate)
        self.peak_normalize = peak_normalize
        self.span = self.context_length + self.segment_length

        files = _list_audio_files(data_path)
        if not files:
            raise FileNotFoundError(f"No audio files found under '{data_path}'.")

        self.clips, self.labels = [], []
        total = 0
        resamplers = {}
        for fp in files:
            wav, sr = load_audio(fp)
            wav = wav.mean(dim=0, keepdim=True)
            if sr != self.sampling_rate:
                if sr not in resamplers:
                    resamplers[sr] = torchaudio.transforms.Resample(sr, self.sampling_rate)
                wav = resamplers[sr](wav)
            wav = wav.squeeze(0).contiguous()
            if wav.numel() == 0:
                continue
            # tile up tracks too short to hold one (context+target) span
            if wav.numel() < self.span:
                reps = (self.span + wav.numel() - 1) // wav.numel() + 1
                wav = wav.repeat(reps)
            self.clips.append(wav)
            self.labels.append(Path(fp).stem)
            total += wav.numel()
        if not self.clips:
            raise RuntimeError(f"All audio files under '{data_path}' were empty.")

        lengths = torch.tensor([c.numel() for c in self.clips], dtype=torch.float)
        self.file_weights = lengths / lengths.sum()
        if samples_per_epoch is None:
            samples_per_epoch = max(1000, int(total // self.segment_length))
        self.samples_per_epoch = int(samples_per_epoch)
        print(f"[MusicContinuation] {len(self.clips)} file(s), {total/self.sampling_rate/60:.1f} min, "
              f"context={self.context_length} target={self.segment_length} "
              f"({self.span/self.sampling_rate:.2f}s span), samples_per_epoch={self.samples_per_epoch}")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index):
        file_idx = int(torch.multinomial(self.file_weights, 1).item())
        wav = self.clips[file_idx]
        n = wav.numel()
        # start of the target; context is the span immediately before it
        lo, hi = self.context_length, n - self.segment_length
        start = self.context_length if hi <= lo else int(torch.randint(lo, hi + 1, (1,)).item())
        context = wav[start - self.context_length:start]
        target = wav[start:start + self.segment_length]
        if self.peak_normalize:
            # joint peak over the pair keeps relative levels consistent
            peak = torch.cat([context, target]).abs().max()
            if peak > 1e-8:
                context, target = context / peak, target / peak
        return target.unsqueeze(0), context.unsqueeze(0)
