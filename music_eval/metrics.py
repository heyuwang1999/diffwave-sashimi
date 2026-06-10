"""
Audio generation metrics (sample level, no text/reference required).

All metrics operate on mono float waveforms in [-1, 1] as 1-D or (1, T) tensors.
They are intentionally model-free so they run anywhere; the fidelity metric is a
Fréchet distance over log-mel frame statistics (a proxy for FAD that needs no
learned embedding). For true FAD, replace `mel_features` with a pretrained audio
embedding (VGGish / PANNs / CLAP) and reuse `frechet_distance` unchanged.
"""
import numpy as np
import torch
import torchaudio


def _to_mono_1d(wav):
    if wav.dim() == 2:
        wav = wav.mean(dim=0)
    return wav.float()


def mel_features(wav, sr, n_fft=1024, hop=256, n_mels=80):
    """Waveform -> (n_frames, n_mels) log-mel features."""
    wav = _to_mono_1d(wav)
    melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, power=2.0
    )(wav)                              # (n_mels, n_frames)
    return torch.log(melspec + 1e-6).T  # (n_frames, n_mels)


def _gaussian_stats(feats):
    """feats: (N, D) -> (mu: D, cov: D x D)."""
    mu = feats.mean(dim=0)
    centered = feats - mu
    n = max(feats.shape[0] - 1, 1)
    cov = (centered.T @ centered) / n
    return mu, cov


def frechet_distance(mu1, cov1, mu2, cov2, eps=1e-6):
    """Fréchet distance between two multivariate Gaussians (the FID/FAD formula).

    Uses the standard FID stabilization: ridge-regularize the covariances and
    fall back to an offset product if the matrix sqrt is non-finite. The result
    is clamped at 0 (Fréchet distance is provably non-negative; tiny negatives
    are pure sqrtm round-off on near-singular covariances)."""
    from scipy.linalg import sqrtm
    mu1, mu2 = mu1.cpu().numpy(), mu2.cpu().numpy()
    cov1, cov2 = cov1.cpu().numpy(), cov2.cpu().numpy()
    d = cov1.shape[0]
    cov1 = cov1 + eps * np.eye(d)
    cov2 = cov2 + eps * np.eye(d)
    diff = mu1 - mu2
    covmean, _ = sqrtm(cov1 @ cov2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(d) * eps
        covmean, _ = sqrtm((cov1 + offset) @ (cov2 + offset), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fd = float(diff @ diff + np.trace(cov1) + np.trace(cov2) - 2.0 * np.trace(covmean))
    return max(fd, 0.0)


def _stack_frames(wavs, sr, **mel_kw):
    return torch.cat([mel_features(w, sr, **mel_kw) for w in wavs], dim=0)


def frechet_mel_distance(gen_wavs, ref_wavs, sr, **mel_kw):
    """FMD: Fréchet distance over log-mel frames of generated vs reference sets.
    Lower is better; 0 means the two sets share mel-feature statistics."""
    g = _stack_frames(gen_wavs, sr, **mel_kw)
    r = _stack_frames(ref_wavs, sr, **mel_kw)
    return frechet_distance(*_gaussian_stats(g), *_gaussian_stats(r))


def reference_mel_distance(gen, ref, sr, **mel_kw):
    """L2 between (mean ++ std) log-mel summary vectors of two clips. Lower better."""
    a = mel_features(gen, sr, **mel_kw)
    b = mel_features(ref, sr, **mel_kw)
    va = torch.cat([a.mean(0), a.std(0)])
    vb = torch.cat([b.mean(0), b.std(0)])
    return float((va - vb).pow(2).mean().sqrt())


def style_consistency(context, continuation, sr, **mel_kw):
    """1 - cosine similarity of mean log-mel (context vs continuation).
    0 = identical timbre/energy profile; larger = stylistic drift."""
    a = mel_features(context, sr, **mel_kw).mean(0)
    b = mel_features(continuation, sr, **mel_kw).mean(0)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0)
    return float(1.0 - cos)


def seam_smoothness(context, continuation, sr, window_s=0.5, n_fft=1024, hop=256, n_mels=80):
    """Spectral jump at the context->continuation boundary, normalized by the
    average internal frame-to-frame jump. ~1.0 = seam as smooth as the interior;
    >>1.0 = audible click/discontinuity at the join."""
    ctx = _to_mono_1d(context)
    con = _to_mono_1d(continuation)
    w = int(window_s * sr)
    join = torch.cat([ctx[-w:], con[:w]], dim=0)
    feats = mel_features(join, sr, n_fft=n_fft, hop=hop, n_mels=n_mels)  # (F, M)
    if feats.shape[0] < 3:
        return float("nan")
    deltas = (feats[1:] - feats[:-1]).pow(2).sum(-1).sqrt()  # (F-1,)
    seam_idx = min(max(w // hop, 1), deltas.shape[0] - 1)
    seam_jump = deltas[seam_idx - 1:seam_idx + 1].mean()
    internal = deltas.mean()
    return float(seam_jump / (internal + 1e-8))
