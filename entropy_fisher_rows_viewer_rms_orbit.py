#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vectorized Entropy/Fisher/Carrier Rows Viewer
---------------------------------
Standalone vectorized experiment app for one audio file.

Workflow:
    1) Run: python3 entropy_fisher_rows_viewer.py
    2) Open WAV/MP3/FLAC/OGG file
    3) Choose segment size from 256..65536
    4) Inspect 7 rectangular rows:
        row 1: Shannon entropy per segment
        row 2: Fisher carrier max per segment
        row 3: Shannon / max(Fisher) per segment
        row 4: Permutation entropy per segment
        row 5: Permutation / max(Fisher) per segment
        row 6: Variation entropy per segment
        row 7: Variation / max(Fisher) per segment
        row 8: Permutation entropy of RMS envelope
        row 9: Permutation(RMS) / max(Fisher)
        row 10: Shannon entropy of RMS envelope
        row 11: Shannon(RMS) / max(Fisher)
        row 12: Variation entropy × Permutation(RMS)
        row 13: Permutation(RMS) × Variation × Fisher
    5) Click a row to plot its full curve below.

Dependencies:
    pip install numpy scipy matplotlib soundfile
Optional MP3 support may depend on your soundfile/libsndfile build.
If MP3 fails, convert to WAV first.
"""

import os
import traceback
import time
import math
import subprocess
import tempfile
import shutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:
    import soundfile as sf
except Exception as exc:
    sf = None
    SF_IMPORT_ERROR = exc
else:
    SF_IMPORT_ERROR = None

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except Exception:
    sd = None
    SOUNDDEVICE_AVAILABLE = False

EPS = 1e-12
DEFAULT_CPU_WORKERS = max(1, (os.cpu_count() or 2) - 1)


def mono_audio(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        x = np.mean(x, axis=1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > EPS:
        x = x / peak
    return x


def segment_audio(x: np.ndarray, seg_size: int) -> np.ndarray:
    seg_size = int(seg_size)
    nseg = len(x) // seg_size
    if nseg <= 0:
        return np.empty((0, seg_size), dtype=np.float64)
    y = x[: nseg * seg_size]
    return y.reshape(nseg, seg_size)


def moving_average_1d(x: np.ndarray, win: int) -> np.ndarray:
    """Small centered moving average used only for zero-crossing stabilization."""
    x = np.asarray(x, dtype=np.float64)
    win = int(max(1, win))
    if win <= 1 or x.size < 3:
        return x.copy()
    win = min(win, x.size)
    if win % 2 == 0:
        win += 1
    k = np.ones(win, dtype=np.float64) / float(win)
    return np.convolve(x, k, mode="same")


def zero_crossing_f0_segments(
    S: np.ndarray,
    sr: int,
    smooth_samples: int = 5,
    deadband_rel: float = 0.02,
    min_halfwave_ms: float = 0.15,
    max_halfwave_ms: float = 80.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate local fundamental carrier from zero-crossing half-wave lengths.

    For every audio segment this detector finds stabilized zero crossings
    around DC and measures the median half-wave length. The corresponding
    fundamental estimate is

        f0 ~= sr / (2 * median_halfwave_samples)

    because one full carrier period contains two half-waves.

    Returns
    -------
    f0_hz : ndarray
        Segment-wise fundamental estimate in Hz.
    halfwave_s : ndarray
        Segment-wise median half-wave duration in seconds.
    zc_count : ndarray
        Number of accepted zero-crossing half-waves per segment.
    """
    S = np.asarray(S, dtype=np.float64)
    nseg = S.shape[0]
    f0_hz = np.zeros(nseg, dtype=np.float64)
    halfwave_s = np.zeros(nseg, dtype=np.float64)
    zc_count = np.zeros(nseg, dtype=np.float64)

    if S.size == 0 or sr is None or int(sr) <= 0:
        return f0_hz, halfwave_s, zc_count

    sr = int(sr)
    min_len = max(1.0, float(sr) * float(min_halfwave_ms) / 1000.0)
    max_len = max(min_len + 1.0, float(sr) * float(max_halfwave_ms) / 1000.0)

    for si in range(nseg):
        x = np.asarray(S[si], dtype=np.float64)
        if x.size < 4:
            continue

        # Remove local DC so crossings are measured around the current carrier axis.
        x = x - float(np.median(x))
        x = moving_average_1d(x, smooth_samples)

        scale = float(np.percentile(np.abs(x), 95.0)) if x.size else 0.0
        if not np.isfinite(scale) or scale <= EPS:
            continue

        # Small hysteresis/deadband suppresses chatter exactly around zero.
        band = max(EPS, float(deadband_rel) * scale)
        sign = np.zeros(x.size, dtype=np.int8)
        sign[x > band] = 1
        sign[x < -band] = -1

        nz = np.where(sign != 0)[0]
        if nz.size < 2:
            continue

        first = int(nz[0])
        sign[:first] = sign[first]
        for i in range(first + 1, sign.size):
            if sign[i] == 0:
                sign[i] = sign[i - 1]

        crossings = np.where(sign[:-1] * sign[1:] < 0)[0].astype(np.float64) + 0.5
        if crossings.size < 2:
            continue

        half_lengths = np.diff(crossings)
        valid = half_lengths[(half_lengths >= min_len) & (half_lengths <= max_len)]
        if valid.size == 0:
            continue

        med_half = float(np.median(valid))
        halfwave_s[si] = med_half / float(sr)
        f0_hz[si] = float(sr) / (2.0 * med_half + EPS)
        zc_count[si] = float(valid.size)

    # Fill occasional missing segments by interpolation so the displayed row is continuous.
    good = np.isfinite(f0_hz) & (f0_hz > 0)
    if np.any(good) and not np.all(good):
        idx = np.arange(nseg, dtype=np.float64)
        f0_hz[~good] = np.interp(idx[~good], idx[good], f0_hz[good])
        hs_good = np.isfinite(halfwave_s) & (halfwave_s > 0)
        if np.any(hs_good):
            halfwave_s[~hs_good] = np.interp(idx[~hs_good], idx[hs_good], halfwave_s[hs_good])

    return (
        np.nan_to_num(f0_hz, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(halfwave_s, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(zc_count, nan=0.0, posinf=0.0, neginf=0.0),
    )


def zero_crossing_f0_curve_for_segment(
    x: np.ndarray,
    sr: int,
    smooth_samples: int = 5,
    deadband_rel: float = 0.02,
    min_period_ms: float = 0.30,
    max_period_ms: float = 160.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Full-wave zero-crossing fundamental tracker for one displayed segment.

    One frequency value is assigned per full carrier wave, not per half-wave:

        period_samples = crossing[k+2] - crossing[k]
        f0_hz          = sr / period_samples

    The raw f0 curve is returned together with a 0..1 normalized version so it
    can be overlaid in the lower segment plot.  This is not FFT/STFT; it is a
    direct zero-crossing carrier detector around the local DC/median axis.

    Returns
    -------
    f0_norm : ndarray
        Display-normalized full-wave f0 curve, 0..1.
    crossings : ndarray
        Zero-crossing positions in samples, fractional .5 positions.
    f0_curve_hz : ndarray
        Raw full-wave f0 curve in Hz.
    period_curve_s : ndarray
        Raw full-wave period curve in seconds.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 6 or sr is None or int(sr) <= 0:
        z = np.zeros(n, dtype=np.float64)
        return z, np.array([], dtype=np.float64), z.copy(), z.copy()

    sr = int(sr)

    # Stabilize the local zero axis. Median is safer than mean for asymmetric
    # musical waveform segments.
    y = x - float(np.median(x))
    y = moving_average_1d(y, smooth_samples)

    scale = float(np.percentile(np.abs(y), 95.0)) if y.size else 0.0
    if not np.isfinite(scale) or scale <= EPS:
        z = np.zeros(n, dtype=np.float64)
        return z, np.array([], dtype=np.float64), z.copy(), z.copy()

    # Deadband prevents rapid chatter exactly around zero.
    band = max(EPS, float(deadband_rel) * scale)
    sign = np.zeros(n, dtype=np.int8)
    sign[y > band] = 1
    sign[y < -band] = -1

    nz = np.where(sign != 0)[0]
    if nz.size < 3:
        z = np.zeros(n, dtype=np.float64)
        return z, np.array([], dtype=np.float64), z.copy(), z.copy()

    first = int(nz[0])
    sign[:first] = sign[first]
    for i in range(first + 1, n):
        if sign[i] == 0:
            sign[i] = sign[i - 1]

    crossings = np.where(sign[:-1] * sign[1:] < 0)[0].astype(np.float64) + 0.5
    f0_curve = np.zeros(n, dtype=np.float64)
    period_curve = np.zeros(n, dtype=np.float64)

    if crossings.size < 3:
        return f0_curve, crossings, f0_curve.copy(), period_curve

    min_period = max(1.0, float(sr) * float(min_period_ms) / 1000.0)
    max_period = max(min_period + 1.0, float(sr) * float(max_period_ms) / 1000.0)

    intervals = []
    values = []

    # Full wave: crossing k to crossing k+2.  Consecutive full-wave intervals
    # overlap by one half-wave, which gives a smooth continuous carrier curve.
    for k in range(crossings.size - 2):
        c0 = float(crossings[k])
        c2 = float(crossings[k + 2])
        period_samples = c2 - c0
        if period_samples < min_period or period_samples > max_period:
            continue

        a = int(max(0, np.floor(c0)))
        b = int(min(n, np.ceil(c2)))
        if b <= a:
            continue

        f0 = float(sr) / (period_samples + EPS)
        period_s = period_samples / float(sr)
        intervals.append((a, b, f0, period_s))
        values.append(f0)

    if not intervals:
        return np.zeros(n, dtype=np.float64), crossings, f0_curve, period_curve

    # Overlapping intervals are averaged sample-wise for a smoother tracker.
    counts = np.zeros(n, dtype=np.float64)
    for a, b, f0, period_s in intervals:
        f0_curve[a:b] += f0
        period_curve[a:b] += period_s
        counts[a:b] += 1.0

    valid = counts > 0
    f0_curve[valid] /= counts[valid]
    period_curve[valid] /= counts[valid]

    # Fill edges and any small gaps by interpolation.
    if np.any(valid):
        idx = np.arange(n, dtype=np.float64)
        vi = idx[valid]
        f0_curve[~valid] = np.interp(idx[~valid], vi, f0_curve[valid])
        period_curve[~valid] = np.interp(idx[~valid], vi, period_curve[valid])

    # Light smoothing after full-wave averaging. Normalization happens after this.
    win = max(3, min(129, n // 96))
    if win % 2 == 0:
        win += 1
    if win > 3:
        f0_curve = moving_average_1d(f0_curve, win)
        period_curve = moving_average_1d(period_curve, win)

    vals = np.asarray(values, dtype=np.float64)
    lo = float(np.percentile(vals, 5.0))
    hi = float(np.percentile(vals, 95.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + EPS:
        lo = float(np.min(vals))
        hi = float(np.max(vals))

    if hi <= lo + EPS:
        f0_norm = np.zeros_like(f0_curve)
    else:
        f0_norm = np.clip((f0_curve - lo) / (hi - lo + EPS), 0.0, 1.0)

    return (
        np.nan_to_num(f0_norm, nan=0.0, posinf=0.0, neginf=0.0),
        crossings,
        np.nan_to_num(f0_curve, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(period_curve, nan=0.0, posinf=0.0, neginf=0.0),
    )


def moving_rms_envelope(x: np.ndarray, win: int) -> np.ndarray:
    """Smooth RMS envelope over the full waveform, normalized to peak 1."""
    x = np.asarray(x, dtype=np.float64)
    win = int(max(4, win))
    kernel = np.ones(win, dtype=np.float64) / float(win)
    env = np.sqrt(np.convolve(x * x, kernel, mode="same"))
    peak = float(np.max(env)) if env.size else 0.0
    if peak > EPS:
        env = env / peak
    return np.nan_to_num(env, nan=0.0, posinf=0.0, neginf=0.0)


def three_point_gradient_1d(x: np.ndarray) -> np.ndarray:
    """Three-point centered gradient for one 1D signal.

    Interior samples use the 3-point stencil:
        g[i] = (x[i+1] - x[i-1]) / 2

    The first and last samples use one-sided differences so the output
    length stays identical to the input length.
    """
    x = np.asarray(x, dtype=np.float64)
    g = np.zeros_like(x, dtype=np.float64)
    if x.size < 2:
        return g
    if x.size == 2:
        g[0] = x[1] - x[0]
        g[1] = x[1] - x[0]
        return g
    g[1:-1] = 0.5 * (x[2:] - x[:-2])
    g[0] = x[1] - x[0]
    g[-1] = x[-1] - x[-2]
    return g


def three_point_gradient_axis1(S: np.ndarray) -> np.ndarray:
    """Three-point centered gradient along axis=1 for segment matrices."""
    S = np.asarray(S, dtype=np.float64)
    G = np.zeros_like(S, dtype=np.float64)
    if S.ndim != 2 or S.shape[1] < 2:
        return G
    if S.shape[1] == 2:
        d = S[:, 1] - S[:, 0]
        G[:, 0] = d
        G[:, 1] = d
        return G
    G[:, 1:-1] = 0.5 * (S[:, 2:] - S[:, :-2])
    G[:, 0] = S[:, 1] - S[:, 0]
    G[:, -1] = S[:, -1] - S[:, -2]
    return G


def normalized_hist_entropy(seg: np.ndarray, bins: int = 64) -> float:
    """Amplitude-distribution Shannon entropy, normalized to [0,1]."""
    if seg.size < 2:
        return 0.0
    hist, _ = np.histogram(seg, bins=bins, range=(-1.0, 1.0), density=False)
    p = hist.astype(np.float64)
    s = float(np.sum(p))
    if s <= 0:
        return 0.0
    p = p / s
    p = p[p > 0]
    h = -float(np.sum(p * np.log(p + EPS)))
    return h / np.log(float(bins) + EPS)


def permutation_entropy(seg: np.ndarray, order: int = 3, delay: int = 1) -> float:
    """Normalized permutation entropy of ordinal patterns."""
    x = np.asarray(seg, dtype=np.float64)
    n = len(x)
    span = (order - 1) * delay
    if n <= span + 1:
        return 0.0

    # Count ordinal patterns. For order=3, only 6 patterns, cheap enough.
    counts = {}
    for i in range(n - span):
        window = x[i : i + span + 1 : delay]
        # Stable argsort reduces random flips for ties.
        pat = tuple(np.argsort(window, kind="mergesort"))
        counts[pat] = counts.get(pat, 0) + 1

    c = np.asarray(list(counts.values()), dtype=np.float64)
    p = c / (np.sum(c) + EPS)
    h = -float(np.sum(p * np.log(p + EPS)))
    # max = log(order!)
    maxh = float(np.log(math.factorial(order)))
    return h / (maxh + EPS)


def variation_entropy(seg: np.ndarray, bins: int = 64) -> float:
    """Entropy of local absolute variation |dx|, normalized to [0,1]."""
    x = np.asarray(seg, dtype=np.float64)
    if len(x) < 3:
        return 0.0
    d = np.abs(np.diff(x))
    # Robust range: map to 0..p99 to avoid one click dominating.
    hi = float(np.percentile(d, 99.0))
    if hi <= EPS or not np.isfinite(hi):
        return 0.0
    hist, _ = np.histogram(np.clip(d, 0.0, hi), bins=bins, range=(0.0, hi), density=False)
    p = hist.astype(np.float64)
    s = float(np.sum(p))
    if s <= 0:
        return 0.0
    p = p / s
    p = p[p > 0]
    h = -float(np.sum(p * np.log(p + EPS)))
    return h / np.log(float(bins) + EPS)




def _entropy_from_counts(counts: np.ndarray, max_states: int) -> np.ndarray:
    """Vectorized normalized Shannon entropy from row-wise counts."""
    counts = np.asarray(counts, dtype=np.float64)
    totals = np.sum(counts, axis=1, keepdims=True)
    p = counts / (totals + EPS)
    h = -np.sum(np.where(p > 0, p * np.log(p + EPS), 0.0), axis=1)
    return h / (np.log(float(max_states)) + EPS)


def vectorized_hist_entropy_segments(S: np.ndarray, bins: int = 64) -> np.ndarray:
    """Fast amplitude-distribution Shannon entropy for all segments."""
    S = np.asarray(S, dtype=np.float64)
    if S.size == 0:
        return np.zeros(S.shape[0], dtype=np.float64)
    n, m = S.shape
    # fixed [-1, 1] binning, matching the scalar function
    idx = np.floor((np.clip(S, -1.0, 1.0) + 1.0) * (bins / 2.0)).astype(np.int64)
    idx = np.clip(idx, 0, bins - 1)
    offsets = (np.arange(n, dtype=np.int64) * bins)[:, None]
    counts = np.bincount((idx + offsets).ravel(), minlength=n * bins).reshape(n, bins)
    return _entropy_from_counts(counts, bins)


def vectorized_hist_entropy_segments_01(S: np.ndarray, bins: int = 64) -> np.ndarray:
    """Fast Shannon entropy for non-negative envelope segments in fixed [0, 1]."""
    S = np.asarray(S, dtype=np.float64)
    if S.size == 0:
        return np.zeros(S.shape[0], dtype=np.float64)
    n, _m = S.shape
    idx = np.floor(np.clip(S, 0.0, 1.0) * bins).astype(np.int64)
    idx = np.clip(idx, 0, bins - 1)
    offsets = (np.arange(n, dtype=np.int64) * bins)[:, None]
    counts = np.bincount((idx + offsets).ravel(), minlength=n * bins).reshape(n, bins)
    return _entropy_from_counts(counts, bins)


def vectorized_variation_entropy_segments(S: np.ndarray, bins: int = 64) -> np.ndarray:
    """Fast entropy of local absolute variation |dx| for all segments."""
    S = np.asarray(S, dtype=np.float64)
    n = S.shape[0]
    if S.shape[1] < 3 or n == 0:
        return np.zeros(n, dtype=np.float64)
    D = np.abs(np.diff(S, axis=1))
    hi = np.percentile(D, 99.0, axis=1)
    valid = np.isfinite(hi) & (hi > EPS)
    out = np.zeros(n, dtype=np.float64)
    if not np.any(valid):
        return out
    Dv = D[valid]
    hiv = hi[valid]
    # row-wise binning to 0..hi
    scaled = Dv / (hiv[:, None] + EPS)
    idx = np.floor(np.clip(scaled, 0.0, 1.0) * bins).astype(np.int64)
    idx = np.clip(idx, 0, bins - 1)
    nv = idx.shape[0]
    offsets = (np.arange(nv, dtype=np.int64) * bins)[:, None]
    counts = np.bincount((idx + offsets).ravel(), minlength=nv * bins).reshape(nv, bins)
    out[valid] = _entropy_from_counts(counts, bins)
    return out


def _ordinal_code(args: np.ndarray, order: int) -> np.ndarray:
    """Encode ordinal pattern rows as base-order integers."""
    weights = (order ** np.arange(order - 1, -1, -1, dtype=np.int64))
    return np.sum(args.astype(np.int64) * weights, axis=-1)


def vectorized_permutation_entropy_segments(S: np.ndarray, order: int = 3, delay: int = 1, chunk_segments: int = 512) -> np.ndarray:
    """Chunked vectorized permutation entropy over all segments.

    Uses sliding_window_view inside chunks, so it avoids Python loops over every
    small window while keeping memory bounded for long audio files.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    S = np.asarray(S, dtype=np.float64)
    nseg, seg_len = S.shape
    span = (int(order) - 1) * int(delay)
    if nseg == 0 or seg_len <= span + 1:
        return np.zeros(nseg, dtype=np.float64)

    order = int(order)
    delay = int(delay)
    n_patterns = math.factorial(order)
    # Codes live in base-order space; not all codes are valid permutations,
    # but bincount over this code space is simple and fast. Entropy is still
    # normalized by log(order!), exactly as in the scalar version.
    code_space = order ** order
    out = np.zeros(nseg, dtype=np.float64)

    for start in range(0, nseg, int(chunk_segments)):
        stop = min(nseg, start + int(chunk_segments))
        X = S[start:stop]
        if delay == 1:
            W = sliding_window_view(X, window_shape=order, axis=1)
        else:
            # Build delayed windows from a longer sliding window then subsample.
            W0 = sliding_window_view(X, window_shape=span + 1, axis=1)
            W = W0[..., ::delay]
        # W shape: chunk x windows x order
        args = np.argsort(W, axis=-1, kind="mergesort")
        codes = _ordinal_code(args, order)
        cnum = codes.shape[0]
        offsets = (np.arange(cnum, dtype=np.int64) * code_space)[:, None]
        counts_full = np.bincount((codes + offsets).ravel(), minlength=cnum * code_space).reshape(cnum, code_space)
        # Remove zero invalid-pattern columns automatically by entropy formula.
        counts = counts_full[:, np.sum(counts_full, axis=0) > 0]
        totals = np.sum(counts, axis=1, keepdims=True)
        p = counts / (totals + EPS)
        h = -np.sum(np.where(p > 0, p * np.log(p + EPS), 0.0), axis=1)
        out[start:stop] = h / (np.log(float(n_patterns)) + EPS)

    return out


def _perm_entropy_worker(args):
    """Worker for multiprocessing permutation entropy chunks.

    Returns (start_index, entropy_vector). Kept top-level so it is pickleable
    with spawn/fork multiprocessing.
    """
    start, X, order, delay, chunk_segments = args
    y = vectorized_permutation_entropy_segments(
        X, order=int(order), delay=int(delay), chunk_segments=int(chunk_segments)
    )
    return int(start), y


def vectorized_permutation_entropy_segments_parallel(
    S: np.ndarray,
    order: int = 3,
    delay: int = 1,
    chunk_segments: int = 512,
    n_workers: int = 1,
) -> np.ndarray:
    """Permutation entropy using all requested CPU cores.

    The core inside each process is still vectorized with NumPy sliding windows
    and bincount; multiprocessing only splits the segment rows across cores.
    This is the heavy part of the app, especially for large audio files and
    small segment sizes, so this gives real multi-core usage.
    """
    S = np.asarray(S, dtype=np.float64)
    if S.size == 0:
        return np.zeros(S.shape[0], dtype=np.float64)

    nseg = S.shape[0]
    n_workers = int(max(1, min(int(n_workers or 1), nseg, os.cpu_count() or 1)))
    if n_workers <= 1 or nseg < 2 * n_workers:
        return vectorized_permutation_entropy_segments(
            S, order=order, delay=delay, chunk_segments=chunk_segments
        )

    # Chunk by segment rows. Use contiguous copies so subprocesses receive
    # compact arrays and keep memory predictable.
    splits = np.array_split(np.arange(nseg), n_workers)
    tasks = []
    for idxs in splits:
        if idxs.size == 0:
            continue
        start = int(idxs[0])
        stop = int(idxs[-1]) + 1
        tasks.append((start, np.ascontiguousarray(S[start:stop]), order, delay, chunk_segments))

    out = np.zeros(nseg, dtype=np.float64)
    # fork is faster on Linux. Fall back to spawn where fork is unavailable.
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=len(tasks), mp_context=ctx) as ex:
        futures = [ex.submit(_perm_entropy_worker, task) for task in tasks]
        for fut in as_completed(futures):
            start, vals = fut.result()
            out[start:start + len(vals)] = vals

    return out


def fisher_curve(values: np.ndarray) -> np.ndarray:
    """Fisher-like 1D gradient intensity over a segment-level curve."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) < 2:
        return np.zeros_like(v)
    g = np.gradient(v)
    f = (g * g) / (np.abs(v) + EPS)
    return np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)


def normalize01(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    if v.size == 0:
        return v
    lo = float(np.percentile(v, 1.0))
    hi = float(np.percentile(v, 99.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + EPS:
        lo = float(np.min(v))
        hi = float(np.max(v))
    if hi <= lo + EPS:
        return np.zeros_like(v)
    return np.clip((v - lo) / (hi - lo + EPS), 0.0, 1.0)


def ratio_to_global_max(numerator: np.ndarray, carrier: np.ndarray) -> np.ndarray:
    """Local numerator divided by the global maximum of a carrier curve.

    This preserves the original concept used in the Sound MP experiments:
    local entropy-like activity is measured relative to the strongest
    structural carrier in the whole track, rather than divided point-by-point.
    """
    a = np.asarray(numerator, dtype=np.float64)
    b = np.asarray(carrier, dtype=np.float64)
    b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
    denom = float(np.nanmax(np.abs(b))) if b.size else 0.0
    if not np.isfinite(denom) or denom <= EPS:
        denom = EPS
    return a / (denom + EPS)


def ratio_to_segment_max(numerator: np.ndarray, carrier_max: np.ndarray) -> np.ndarray:
    """Local Shannon divided by the local maximum carrier for the same segment.

    This is the corrected experimental MSFR-style ratio requested here:
        segment Shannon / max(carrier inside the same segment)
    not a point-by-point division and not a global-track maximum.
    """
    a = np.asarray(numerator, dtype=np.float64)
    b = np.asarray(carrier_max, dtype=np.float64)
    b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
    return a / (np.abs(b) + EPS)


def vectorized_segment_fisher_max(
    S: np.ndarray,
    mode: str = "sample",
    R: np.ndarray | None = None,
    shannon: np.ndarray | None = None,
    permutation: np.ndarray | None = None,
    variation: np.ndarray | None = None,
) -> np.ndarray:
    """Maximum Fisher-like activity inside each segment.

    Modes
    -----
    sample:
        Original carrier:
            F(t) = (dx/dt)^2 / (|x(t)| + eps)

    rms:
        RMS-stabilized carrier:
            F(t) = (dx/dt)^2 / (RMS_env(t) + eps)

        This suppresses artificial explosions near zero crossings of the
        audio waveform and measures fast changes relative to the local
        energy envelope.

    multi_rms:
        Multidimensional RMS-stabilized carrier:
            z(t) = [audio(t), RMS_env(t), H_seg, P_seg, V_seg]
            F(t) = ||dz/dt||^2 / (RMS_env(t) + eps)

        H/P/V are segment-level trajectories, so their temporal gradients are
        computed across segments and injected as constant contributions inside
        each corresponding segment.
    """
    S = np.asarray(S, dtype=np.float64)
    if S.size == 0:
        return np.zeros(S.shape[0], dtype=np.float64)
    if S.shape[1] < 3:
        return np.zeros(S.shape[0], dtype=np.float64)

    mode = str(mode or "sample").lower()
    G_audio = three_point_gradient_axis1(S)

    if R is None or np.shape(R) != np.shape(S):
        # Fallback local RMS envelope from each segment if a full-track RMS
        # envelope was not supplied.
        seg_rms = np.sqrt(np.mean(S * S, axis=1, keepdims=True) + EPS)
        R = np.repeat(seg_rms, S.shape[1], axis=1)
    else:
        R = np.asarray(R, dtype=np.float64)
        R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)

    if mode == "sample":
        numerator = G_audio * G_audio
        denominator = np.abs(S) + EPS
    elif mode == "rms":
        numerator = G_audio * G_audio
        denominator = R + EPS
    elif mode == "multi_rms":
        G_rms = three_point_gradient_axis1(R)
        numerator = (G_audio * G_audio) + 0.50 * (G_rms * G_rms)

        # Add segment-level entropy carrier dynamics.  These are not local
        # sample curves; they describe how the chosen segment changes relative
        # to neighboring segments.  They stabilize the Fisher carrier as a
        # multidimensional structural novelty detector.
        for feat, weight in (
            (shannon, 0.25),
            (permutation, 0.25),
            (variation, 0.25),
        ):
            if feat is None:
                continue
            f = np.asarray(feat, dtype=np.float64)
            if f.shape[0] != S.shape[0]:
                continue
            df = np.gradient(np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0))
            # Robust scale so one entropy family does not dominate only because
            # of units.
            scale = float(np.percentile(np.abs(df), 95.0)) + EPS
            df = df / scale
            numerator = numerator + float(weight) * (df[:, None] ** 2)

        denominator = R + EPS
    else:
        numerator = G_audio * G_audio
        denominator = np.abs(S) + EPS

    F = numerator / denominator
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    # Compress extreme clicks/transients but keep ordering.
    return np.log1p(np.max(F, axis=1))


def _subwindow_view_segments(S: np.ndarray, n_sub: int = 16) -> np.ndarray:
    """Return non-overlapping subwindows: shape nseg x n_sub x sub_len."""
    S = np.asarray(S, dtype=np.float64)
    nseg, seg_len = S.shape
    n_sub = int(max(1, min(n_sub, seg_len)))
    sub_len = seg_len // n_sub
    if sub_len < 4:
        return S[:, None, :]
    cut = n_sub * sub_len
    return S[:, :cut].reshape(nseg, n_sub, sub_len)


def vectorized_local_permutation_mean_max(S: np.ndarray, order: int = 3, delay: int = 1, n_sub: int = 16, n_workers: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Mean and max local permutation entropy inside each segment.

    For each segment we split into subwindows, compute permutation entropy per
    subwindow, then return mean and max across subwindows.  The mean is drawn as
    the carrier row; the max is used as denominator in Shannon/max(Permutation).
    """
    W = _subwindow_view_segments(S, n_sub=n_sub)
    nseg, nsub, sub_len = W.shape
    flat = W.reshape(nseg * nsub, sub_len)
    pe = vectorized_permutation_entropy_segments_parallel(flat, order=order, delay=delay, chunk_segments=512, n_workers=n_workers)
    pe = pe.reshape(nseg, nsub)
    return np.mean(pe, axis=1), np.max(pe, axis=1)


def vectorized_local_variation_mean_max(S: np.ndarray, bins: int = 64, n_sub: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Mean and max local variation entropy inside each segment."""
    W = _subwindow_view_segments(S, n_sub=n_sub)
    nseg, nsub, sub_len = W.shape
    flat = W.reshape(nseg * nsub, sub_len)
    ve = vectorized_variation_entropy_segments(flat, bins=bins)
    ve = ve.reshape(nseg, nsub)
    return np.mean(ve, axis=1), np.max(ve, axis=1)


class EntropyRowsViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Entropy/Fisher Rows Viewer — single audio experiment")
        self.root.geometry("1500x980")

        self.audio = None
        self.sr = None
        self.path = None
        self.features = {}
        self.row_names = [
            "Shannon entropy",
            "Fisher max in segment",
            "Shannon / max(Fisher)",
            "Permutation entropy",
            "Permutation / max(Fisher)",
            "Variation entropy",
            "Variation / max(Fisher)",
            "Permutation entropy of RMS envelope",
            "Permutation(RMS) / max(Fisher)",
            "Shannon entropy of RMS envelope",
            "Shannon(RMS) / max(Fisher)",
            "Variation × Permutation(RMS)",
            "Permutation(RMS) × Variation × Fisher",
            "Zero-crossing f0 carrier",
        ]
        self.selected_row = 0
        self.selected_segment = 0
        self._is_playing = False
        self._last_play_tmp = None
        self._play_process = None
        self._play_after_id = None
        self._play_start_clock = None
        self._play_duration = 0.0
        self._play_cursor = None
        self._recompute_after_id = None
        self.play_source_var = None
        self._rms_cache = {}          # {ms: full-track RMS envelope}
        self._rms_cache_sr = None
        self._rms_cache_audio_len = 0

        self._build_ui()
        self._draw_empty()

    def _build_ui(self):
        # Two-row command panel so all controls fit on narrower screens.
        controls = tk.Frame(self.root)
        controls.pack(side=tk.TOP, fill=tk.X)

        row1 = tk.Frame(controls)
        row1.pack(side=tk.TOP, fill=tk.X)

        row2 = tk.Frame(controls)
        row2.pack(side=tk.TOP, fill=tk.X)

        tk.Button(row1, text="Open audio file", command=self.open_audio).pack(side=tk.LEFT, padx=6, pady=4)
        tk.Button(row1, text="Compute / Recompute all", command=self.compute_and_draw).pack(side=tk.LEFT, padx=6, pady=4)
        tk.Button(row1, text="Play selected segment", command=self.play_selected_segment).pack(side=tk.LEFT, padx=6, pady=4)
        tk.Button(row1, text="Stop", command=self.stop_playback).pack(side=tk.LEFT, padx=2, pady=4)

        tk.Label(row1, text="Play source").pack(side=tk.LEFT, padx=(10, 4))
        self.play_source_var = tk.StringVar(value="audio")
        self.play_source_menu = tk.OptionMenu(
            row1,
            self.play_source_var,
            "audio",
            "RMS envelope",
            "local Fisher",
            "F_RMS dX2/RMS",
            "3-point gradient Fisher",
            "segment Shannon",
            "segment Shannon(RMS)",
            "selected row",
        )
        self.play_source_menu.config(width=16)
        self.play_source_menu.pack(side=tk.LEFT, padx=4)

        tk.Label(row1, text="CPU workers").pack(side=tk.LEFT, padx=(10, 4))
        self.cpu_workers_var = tk.IntVar(value=DEFAULT_CPU_WORKERS)
        tk.Spinbox(
            row1,
            from_=1,
            to=max(1, os.cpu_count() or 1),
            increment=1,
            width=4,
            textvariable=self.cpu_workers_var,
        ).pack(side=tk.LEFT, padx=4)

        tk.Label(row1, text="Segment size").pack(side=tk.LEFT, padx=(12, 4))
        self.seg_var = tk.IntVar(value=4096)
        self.seg_scale = tk.Scale(
            row1,
            from_=256,
            to=65536,
            resolution=255,
            orient=tk.HORIZONTAL,
            variable=self.seg_var,
            length=360,
            command=lambda _=None: self.schedule_recompute("Segment size changed"),
        )
        self.seg_scale.pack(side=tk.LEFT, padx=4)

        tk.Label(row1, text="Bins").pack(side=tk.LEFT, padx=(12, 4))
        self.bins_var = tk.IntVar(value=64)
        tk.Spinbox(row1, from_=16, to=256, increment=16, width=6, textvariable=self.bins_var).pack(side=tk.LEFT)

        tk.Label(row1, text="Permutation order").pack(side=tk.LEFT, padx=(12, 4))
        self.order_var = tk.IntVar(value=3)
        tk.Spinbox(row1, from_=3, to=6, increment=1, width=4, textvariable=self.order_var).pack(side=tk.LEFT)

        tk.Label(row2, text="Fisher mode").pack(side=tk.LEFT, padx=(6, 4))
        self.fisher_mode_var = tk.StringVar(value="sample")
        self.fisher_mode_menu = tk.OptionMenu(
            row2,
            self.fisher_mode_var,
            "sample",
            "rms",
            "multi_rms",
            command=lambda _=None: self.schedule_recompute("Fisher mode changed"),
        )
        self.fisher_mode_menu.config(width=10)
        self.fisher_mode_menu.pack(side=tk.LEFT, padx=4)

        tk.Label(row2, text="RMS window").pack(side=tk.LEFT, padx=(14, 4))
        self.rms_ms_var = tk.IntVar(value=50)
        for value, label in (
            (10, "10 ms → attacks"),
            (20, "20 ms → speech"),
            (50, "50 ms → music"),
            (100, "100 ms → phrases"),
        ):
            tk.Radiobutton(
                row2,
                text=label,
                variable=self.rms_ms_var,
                value=value,
                command=lambda: self.schedule_recompute("RMS window changed"),
            ).pack(side=tk.LEFT, padx=2)

        # Spinboxes do not have a command callback, so trace them and recompute all rows.
        self.bins_var.trace_add("write", lambda *_: self.schedule_recompute("Bins changed"))
        self.order_var.trace_add("write", lambda *_: self.schedule_recompute("Permutation order changed"))
        # Extra trace: some Tk themes/platforms may not fire Radiobutton.command
        # reliably when values are changed programmatically. This guarantees
        # that changing 10/20/50/100 ms recomputes FRMS and all rows before redraw.
        self.rms_ms_var.trace_add("write", lambda *_: self.schedule_recompute("RMS window changed"))

        self.status_var = tk.StringVar(value="Open one audio file.")
        tk.Label(self.root, textvariable=self.status_var, anchor="w").pack(side=tk.BOTTOM, fill=tk.X)

        self.fig = Figure(figsize=(15.0, 10.2), dpi=100)
        # Four visual bands:
        #   1) rectangular entropy/Fisher rows
        #   2) selected coefficient curve + normalized RMS preview overlaid
        #   3) selected-segment waveform/RMS view with Fisher modifiers shifted below
        #   4) RMS-centered Fisher harmonic orbit, segmented by two half-waves
        gs = self.fig.add_gridspec(4, 1, height_ratios=[1.0, 1.0, 1.75, 0.92])
        self.ax_rows = self.fig.add_subplot(gs[0, 0])
        self.ax_selected = self.fig.add_subplot(gs[1, 0])
        self.ax_audio = self.ax_selected
        self.ax_segment = self.fig.add_subplot(gs[2, 0])
        self.ax_orbit = self.fig.add_subplot(gs[3, 0])

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_click)

    def _draw_empty(self):
        for ax in (self.ax_rows, self.ax_selected, self.ax_segment, self.ax_orbit):
            ax.clear()
        self.ax_rows.text(0.5, 0.5, "Open audio file", transform=self.ax_rows.transAxes, ha="center", va="center")
        self.ax_rows.set_title("Entropy/Fisher rectangular map")
        self.ax_selected.set_title("Selected row + normalized RMS preview")
        self.ax_segment.set_title("Selected segment waveform + RMS; Fisher modifiers shifted below")
        self.ax_orbit.set_title("RMS-centered Fisher harmonic orbit")
        self.canvas.draw_idle()

    def get_rms_window_ms(self) -> int:
        """Return selected RMS analysis window in milliseconds."""
        try:
            return int(self.rms_ms_var.get())
        except Exception:
            return 50

    def get_rms_window_samples(self) -> int:
        """Return selected RMS window in samples, tied to real time, not segment size."""
        if self.sr is None:
            return 1024
        win = int(round(float(self.sr) * float(self.get_rms_window_ms()) / 1000.0))
        return int(max(8, win))

    def rms_window_samples_for_ms(self, ms: int) -> int:
        """Convert an RMS window in milliseconds to samples."""
        if self.sr is None:
            return 1024
        win = int(round(float(self.sr) * float(ms) / 1000.0))
        return int(max(8, win))

    def build_rms_cache(self):
        """Precompute full-track RMS envelopes for all selectable windows.

        This is fully vectorized NumPy convolution.  It is intentionally done
        once after loading the audio, because changing 10/20/50/100 ms should
        only select a cached envelope instead of recomputing convolution.
        """
        self._rms_cache = {}
        self._rms_cache_sr = self.sr
        self._rms_cache_audio_len = 0 if self.audio is None else len(self.audio)

        if self.audio is None or self.sr is None:
            return

        for ms in (10, 20, 50, 100):
            win = self.rms_window_samples_for_ms(ms)
            self._rms_cache[int(ms)] = moving_rms_envelope(self.audio, win)

    def get_cached_rms_envelope(self, ms: int | None = None) -> np.ndarray:
        """Return cached full-track RMS envelope for selected ms.

        If the cache is missing or stale, rebuild all four envelopes.
        """
        if ms is None:
            ms = self.get_rms_window_ms()
        ms = int(ms)

        cache_stale = (
            not self._rms_cache
            or self._rms_cache_sr != self.sr
            or self._rms_cache_audio_len != (0 if self.audio is None else len(self.audio))
            or ms not in self._rms_cache
        )
        if cache_stale:
            self.build_rms_cache()

        if ms in self._rms_cache:
            return self._rms_cache[ms]

        # Defensive fallback for unexpected values.
        return moving_rms_envelope(self.audio, self.rms_window_samples_for_ms(ms))

    def mark_stale(self):
        if self.audio is not None:
            self.status_var.set("Parameter changed — press Compute / Recompute all.")

    def schedule_recompute(self, reason="Parameter changed"):
        """Debounced full recompute after GUI parameter changes.

        Every parameter that affects the rows or lower overlay recomputes the
        full feature set before redrawing.  This avoids stale maps after changing
        segment size, Fisher mode, RMS window, bins or permutation order.
        """
        if self.audio is None:
            return
        self.status_var.set(f"{reason} — recomputing all graphs...")
        if self._recompute_after_id is not None:
            try:
                self.root.after_cancel(self._recompute_after_id)
            except Exception:
                pass
        self._recompute_after_id = self.root.after(300, self._scheduled_compute_and_draw)

    def _scheduled_compute_and_draw(self):
        self._recompute_after_id = None
        if self.audio is not None:
            self.compute_and_draw()

    def open_audio(self):
        if sf is None:
            messagebox.showerror("soundfile missing", f"Could not import soundfile:\n{SF_IMPORT_ERROR}")
            return
        path = filedialog.askopenfilename(
            title="Open audio file",
            filetypes=[
                ("Audio files", "*.wav *.flac *.ogg *.aiff *.aif *.mp3"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            x, sr = sf.read(path, always_2d=False)
            self.audio = mono_audio(x)
            self.sr = int(sr)
            self.path = path
            self.status_var.set("Building cached RMS envelopes: 10/20/50/100 ms...")
            self.root.update_idletasks()
            self.build_rms_cache()
            self.status_var.set(f"Loaded: {os.path.basename(path)} | sr={sr} | duration={len(self.audio)/sr:.2f}s | RMS cache ready")
            self.compute_and_draw()
        except Exception as exc:
            traceback.print_exc()
            messagebox.showerror("Open failed", str(exc))

    def get_cpu_workers(self) -> int:
        try:
            return int(max(1, min(int(self.cpu_workers_var.get()), os.cpu_count() or 1)))
        except Exception:
            return DEFAULT_CPU_WORKERS

    def compute_features(self):
        if self.audio is None:
            return
        t0 = time.perf_counter()
        seg_size = int(self.seg_var.get())
        bins = int(self.bins_var.get())
        order = int(self.order_var.get())
        n_workers = self.get_cpu_workers()
        S = segment_audio(self.audio, seg_size)
        if S.shape[0] < 2:
            raise RuntimeError("Audio is too short for this segment size.")

        # Vectorized feature extraction over all segments.
        # Shannon: one bincount over all samples.
        # Variation entropy: row-wise 99-percentile normalization + one bincount.
        # Permutation entropy: chunked stride windows + bincount over ordinal codes.
        shannon = vectorized_hist_entropy_segments(S, bins=bins)

        perm, _perm_max_unused = vectorized_local_permutation_mean_max(S, order=order, delay=1, n_sub=16, n_workers=n_workers)
        varent, _var_max_unused = vectorized_local_variation_mean_max(S, bins=bins, n_sub=16)
        rms = np.sqrt(np.mean(S * S, axis=1))

        # Zero-crossing carrier detector:
        #   median half-wave length -> f0 = sr / (2 * halfwave_length).
        # This is not FFT; it is a direct fundamental indicator from crossings
        # around the local zero/DC axis of each segment.
        zc_f0_hz, zc_halfwave_s, zc_count = zero_crossing_f0_segments(S, self.sr)

        # RMS-envelope entropy carriers:
        #   Permutation(RMS) and Shannon(RMS) are computed from the smoothed RMS envelope.
        #   The same envelope can also be used as a Fisher stabilizer.
        rms_ms = self.get_rms_window_ms()
        rms_win = self.rms_window_samples_for_ms(rms_ms)
        # Cached full-track RMS envelope.  Switching RMS 10/20/50/100 ms selects
        # this precomputed vector; no convolution is repeated here.
        rms_env = self.get_cached_rms_envelope(rms_ms)
        R = segment_audio(rms_env, seg_size)
        # R has the same number of segments as S because both come from self.audio.
        # Still trim defensively in case of future edits.
        ncommon = min(S.shape[0], R.shape[0])
        if ncommon != S.shape[0]:
            S = S[:ncommon]
            shannon = shannon[:ncommon]
            perm = perm[:ncommon]
            varent = varent[:ncommon]
            rms = rms[:ncommon]
            zc_f0_hz = zc_f0_hz[:ncommon]
            zc_halfwave_s = zc_halfwave_s[:ncommon]
            zc_count = zc_count[:ncommon]
            R = R[:ncommon]

        fisher_mode = self.fisher_mode_var.get() if hasattr(self, "fisher_mode_var") else "sample"

        # Common-Fisher carrier logic:
        #   We keep ONE structural carrier: Fisher max inside each segment.
        #   It can be computed in three modes:
        #       sample    : (dx/dt)^2 / (|x| + eps)
        #       rms       : (dx/dt)^2 / (RMS_env + eps)
        #       multi_rms : ||d[audio,RMS,H,P,V]/dt||^2 / (RMS_env + eps)
        #   Shannon, Permutation and Variation are divided by this same
        #   segment-level Fisher carrier.
        fisher_max = vectorized_segment_fisher_max(
            S,
            mode=fisher_mode,
            R=R,
            shannon=shannon,
            permutation=perm,
            variation=varent,
        )

        perm_rms = vectorized_permutation_entropy_segments_parallel(R, order=order, delay=1, chunk_segments=512, n_workers=n_workers)
        shannon_rms = vectorized_hist_entropy_segments_01(R, bins=bins)

        # Coupled rhythm-variation carrier:
        #   Variation(audio) × Permutation(RMS)
        # This emphasizes places where the local variation is high AND the RMS
        # event order is complex. It is intentionally not divided by Fisher;
        # Fisher remains visible in the common carrier rows above.
        variation_x_perm_rms = varent * perm_rms

        # Strong triple event carrier:
        #   Permutation(RMS) × Variation(audio) × Fisher(audio)
        # It lights up only when rhythmic order complexity, local variation,
        # and the audio Fisher structural slope are simultaneously high.
        perm_rms_x_variation_x_fisher = perm_rms * varent * fisher_max

        rows_raw = [
            shannon,
            fisher_max,
            ratio_to_segment_max(shannon, fisher_max),
            perm,
            ratio_to_segment_max(perm, fisher_max),
            varent,
            ratio_to_segment_max(varent, fisher_max),
            perm_rms,
            ratio_to_segment_max(perm_rms, fisher_max),
            shannon_rms,
            ratio_to_segment_max(shannon_rms, fisher_max),
            variation_x_perm_rms,
            perm_rms_x_variation_x_fisher,
            zc_f0_hz,
        ]
        rows_norm = np.vstack([normalize01(r) for r in rows_raw])

        self.features = {
            "segments": S,
            "seg_size": seg_size,
            "time_segments": np.arange(len(shannon)) * seg_size / float(self.sr),
            "shannon": shannon,
            "perm": perm,
            "variation": varent,
            "perm_rms": perm_rms,
            "shannon_rms": shannon_rms,
            "variation_x_perm_rms": variation_x_perm_rms,
            "perm_rms_x_variation_x_fisher": perm_rms_x_variation_x_fisher,
            "zc_f0_hz": zc_f0_hz,
            "zc_halfwave_s": zc_halfwave_s,
            "zc_count": zc_count,
            "fisher_max": fisher_max,
            "fisher_mode": fisher_mode,
            "rms_window_ms": rms_ms,
            "rms_window_samples": rms_win,
            "rms_env_segments": R,
            "perm_max": _perm_max_unused,
            "variation_max": _var_max_unused,
            "rms": rms,
            "rows_raw": rows_raw,
            "rows_norm": rows_norm,
            "compute_time": time.perf_counter() - t0,
            "cpu_workers": n_workers,
        }

    def compute_and_draw(self):
        if self.audio is None:
            messagebox.showerror("No audio", "Open audio file first.")
            return
        try:
            self.compute_features()
            self.draw_all()
            n = self.features["rows_norm"].shape[1]
            self.status_var.set(
                f"Computed {n} segments | seg={self.features['seg_size']} | "
                f"time={self.features.get('compute_time', 0.0):.3f}s | Fisher={self.features.get('fisher_mode', 'sample')} | RMS={self.features.get('rms_window_ms', 50)} ms cached | ZC f0 row | vectorized NumPy + multicore permutation | click row to inspect curve."
            )
        except Exception as exc:
            traceback.print_exc()
            messagebox.showerror("Compute failed", str(exc))

    def draw_all(self):
        rows = self.features.get("rows_norm")
        if rows is None:
            return
        t = self.features["time_segments"]

        self.ax_rows.clear()
        nrows = len(self.row_names)
        im = self.ax_rows.imshow(
            rows,
            aspect="auto",
            origin="upper",
            interpolation="nearest",
            vmin=0,
            vmax=1,
            extent=[t[0], t[-1] if len(t) else 1, nrows, 0],
        )
        self.ax_rows.set_yticks(np.arange(nrows) + 0.5)
        self.ax_rows.set_yticklabels(self.row_names, fontsize=9)
        self.ax_rows.set_xlabel("time (s)")
        self.ax_rows.set_title(
            f"Entropy/Fisher rows | {os.path.basename(self.path) if self.path else ''} | seg={self.features['seg_size']} | Fisher={self.features.get('fisher_mode', 'sample')} | RMS={self.features.get('rms_window_ms', 50)} ms"
        )
        self.ax_rows.axhline(self.selected_row, color="white", linewidth=0.8, alpha=0.7)
        self.ax_rows.axhline(self.selected_row + 1, color="white", linewidth=0.8, alpha=0.7)

        # Keep one colorbar only by clearing old figure colorbars is awkward; skip to keep UI clean.
        self.draw_selected_curve()
        self.draw_audio_preview()
        self.draw_segment_waveform_overlay()
        self.draw_fisher_rms_orbit()
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def draw_selected_curve(self):
        """Overlay the selected normalized row and the normalized RMS preview."""
        self.ax_selected.clear()
        rows_raw = self.features["rows_raw"]
        rows_norm = self.features["rows_norm"]
        t = self.features["time_segments"]
        idx = int(np.clip(self.selected_row, 0, len(self.row_names) - 1))

        selected_curve = normalize01(rows_norm[idx])
        rms_curve = normalize01(self.features["rms"])

        self.ax_selected.plot(t, selected_curve, linewidth=1.25, label=f"selected: {self.row_names[idx]}")
        self.ax_selected.plot(t, rms_curve, linewidth=1.05, alpha=0.85, label="RMS preview norm")

        if len(t):
            si = int(np.clip(self.selected_segment, 0, len(t) - 1))
            self.ax_selected.axvline(t[si], linewidth=1.2, linestyle=":", label=f"selected seg {si}")

        self.ax_selected.set_ylim(-0.05, 1.05)
        self.ax_selected.grid(True, alpha=0.3, linestyle="--")
        self.ax_selected.set_title(f"Selected row over RMS preview: {self.row_names[idx]}")
        self.ax_selected.set_xlabel("time (s)")
        self.ax_selected.set_ylabel("normalized value")

        raw = np.asarray(rows_raw[idx])
        txt = f"raw min={np.min(raw):.4g}, median={np.median(raw):.4g}, max={np.max(raw):.4g}"
        self.ax_selected.text(
            0.01, 0.92, txt, transform=self.ax_selected.transAxes,
            fontsize=8, ha="left", va="top", bbox=dict(boxstyle="round", alpha=0.15),
        )
        self.ax_selected.legend(fontsize=8, loc="upper right")

    def draw_audio_preview(self):
        """The old third plot is now overlaid into draw_selected_curve()."""
        return


    def compute_selected_segment_curves(self):
        """Return normalized waveform and analysis curves for the selected segment.

        The same helper is used both for drawing and for audio audition of
        synthetic analysis carriers, so the curves you hear match the curves
        you see in the lower plot.
        """
        if self.audio is None or not self.features:
            raise RuntimeError("Open and compute an audio file first.")

        seg_size = int(self.features.get("seg_size", self.seg_var.get()))
        rows_norm = self.features.get("rows_norm")
        rows_raw = self.features.get("rows_raw")
        if rows_norm is None or rows_norm.shape[1] == 0:
            raise RuntimeError("No computed segments available.")

        nseg = rows_norm.shape[1]
        si = int(np.clip(self.selected_segment, 0, nseg - 1))
        self.selected_segment = si
        row = int(np.clip(self.selected_row, 0, len(self.row_names) - 1))

        a = si * seg_size
        b = min(a + seg_size, len(self.audio))
        seg = np.asarray(self.audio[a:b], dtype=np.float64)
        if seg.size < 2:
            raise RuntimeError("Selected segment is empty or outside audio.")

        seg = seg - float(np.mean(seg))
        peak = float(np.max(np.abs(seg)) + EPS)
        seg_plot = seg / peak
        tt = np.arange(seg_plot.size, dtype=np.float64) / float(self.sr)

        coeff01 = float(rows_norm[row, si])

        # Local RMS envelope with the fixed physical RMS window used by the last compute.
        # It is a slice from the full-track cached RMS envelope, so changing
        # 10/20/50/100 ms selects a precomputed vector instead of rerunning
        # convolution while redrawing/listening.
        rms_ms_used = int(self.features.get("rms_window_ms", self.get_rms_window_ms()))
        frame = int(self.features.get("rms_window_samples", self.rms_window_samples_for_ms(rms_ms_used)))
        try:
            full_env = self.get_cached_rms_envelope(rms_ms_used)
            env = np.asarray(full_env[a:b], dtype=np.float64)
            if env.size != seg_plot.size:
                env = np.resize(env, seg_plot.size)
        except Exception:
            frame_local = int(max(8, min(frame, max(8, seg_plot.size))))
            kernel = np.ones(frame_local, dtype=np.float64) / float(frame_local)
            env = np.sqrt(np.convolve(seg_plot * seg_plot, kernel, mode="same"))
        env = env / (np.max(env) + EPS)

        # Explicit RMS-stabilized Fisher: F_RMS(t) = dX(t)^2 / (RMS(t)+eps)
        # Keep the RAW positive values before normalization, because the new
        # coupled carrier is defined as:
        #     ZC_f0_raw(t) * F_RMS_raw(t)
        # and only then normalized to 0..1 for display.
        dx_rms = three_point_gradient_1d(seg_plot)
        fisher_dx2_over_rms_raw = (dx_rms * dx_rms) / (env + 1e-3)
        fisher_dx2_over_rms_raw = np.nan_to_num(
            fisher_dx2_over_rms_raw, nan=0.0, posinf=0.0, neginf=0.0
        )
        fisher_dx2_over_rms = fisher_dx2_over_rms_raw - float(np.min(fisher_dx2_over_rms_raw))
        fisher_dx2_over_rms = fisher_dx2_over_rms / (float(np.max(fisher_dx2_over_rms)) + EPS)

        # Three-point gradient Fisher / slope-energy detector:
        #     F_grad(t) = ((X[t+1]-X[t-1])/2)^2
        # This keeps only the numerator of the Fisher-like expression.
        # It shows where the waveform slope is large without division by
        # either instantaneous sample amplitude or RMS envelope.
        fisher_gradient = dx_rms * dx_rms
        fisher_gradient = np.nan_to_num(fisher_gradient, nan=0.0, posinf=0.0, neginf=0.0)
        fisher_gradient = fisher_gradient - float(np.min(fisher_gradient))
        fisher_gradient = fisher_gradient / (float(np.max(fisher_gradient)) + EPS)

        # Local Fisher according to selected Fisher mode.
        fisher_mode = self.features.get("fisher_mode", "sample")
        d = np.diff(seg_plot, prepend=seg_plot[0])
        if fisher_mode == "sample":
            local_fisher = (d * d) / (np.abs(seg_plot) + 1e-3)
            fisher_label = "local Fisher: sample denom"
        elif fisher_mode == "rms":
            local_fisher = (d * d) / (env + 1e-3)
            fisher_label = "local Fisher: RMS denom"
        elif fisher_mode == "multi_rms":
            dr = np.diff(env, prepend=env[0])
            local_fisher = ((d * d) + 0.50 * (dr * dr)) / (env + 1e-3)
            fisher_label = "local multidim Fisher: RMS stabilized"
        else:
            local_fisher = (d * d) / (np.abs(seg_plot) + 1e-3)
            fisher_label = "local Fisher"
        local_fisher = np.nan_to_num(local_fisher, nan=0.0, posinf=0.0, neginf=0.0)

        fwin = max(3, min(129, seg_plot.size // 256))
        if fwin % 2 == 0:
            fwin += 1
        if fwin > 3:
            fk = np.ones(fwin, dtype=np.float64) / float(fwin)
            local_fisher = np.convolve(local_fisher, fk, mode="same")

        local_fisher_norm = local_fisher - float(np.min(local_fisher))
        local_fisher_norm = local_fisher_norm / (float(np.max(local_fisher_norm)) + EPS)

        # Segment Shannon as a constant normalized line.
        try:
            shannon_all = np.asarray(self.features.get("shannon", []), dtype=np.float64)
            if shannon_all.size > si:
                shannon_seg = float(shannon_all[si])
                sh_lo = float(np.nanmin(shannon_all))
                sh_hi = float(np.nanmax(shannon_all))
                if np.isfinite(sh_lo) and np.isfinite(sh_hi) and sh_hi > sh_lo + EPS:
                    shannon_norm = (shannon_seg - sh_lo) / (sh_hi - sh_lo + EPS)
                else:
                    shannon_norm = 0.0
            else:
                shannon_seg = 0.0
                shannon_norm = 0.0
        except Exception:
            shannon_seg = 0.0
            shannon_norm = 0.0
        shannon_norm = float(np.clip(shannon_norm, 0.0, 1.0))
        segment_shannon_line = np.full_like(seg_plot, shannon_norm, dtype=np.float64)

        # Segment Shannon of the RMS envelope as a second constant line.
        # This lets us compare whether the entropy comes mostly from the raw
        # waveform distribution or from the local energy envelope.
        try:
            shannon_rms_all = np.asarray(self.features.get("shannon_rms", []), dtype=np.float64)
            if shannon_rms_all.size > si:
                shannon_rms_seg = float(shannon_rms_all[si])
                sr_lo = float(np.nanmin(shannon_rms_all))
                sr_hi = float(np.nanmax(shannon_rms_all))
                if np.isfinite(sr_lo) and np.isfinite(sr_hi) and sr_hi > sr_lo + EPS:
                    shannon_rms_norm = (shannon_rms_seg - sr_lo) / (sr_hi - sr_lo + EPS)
                else:
                    shannon_rms_norm = 0.0
            else:
                shannon_rms_seg = 0.0
                shannon_rms_norm = 0.0
        except Exception:
            shannon_rms_seg = 0.0
            shannon_rms_norm = 0.0

        shannon_rms_norm = float(np.clip(shannon_rms_norm, 0.0, 1.0))
        segment_shannon_rms_line = np.full_like(seg_plot, shannon_rms_norm, dtype=np.float64)

        # Local zero-crossing full-wave fundamental index for the currently
        # visible segment. One raw value is assigned per full wave, measured
        # between crossing[k] and crossing[k+2].
        zc_f0_norm, zc_crossings, zc_f0_curve_hz, zc_period_curve_s = zero_crossing_f0_curve_for_segment(
            seg_plot,
            self.sr,
        )

        # Coupled carrier requested here:
        # multiply the RAW f0 curve and RAW F_RMS curve first, then normalize.
        zc_frms_raw = np.nan_to_num(
            zc_f0_curve_hz * fisher_dx2_over_rms_raw,
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        zc_frms = zc_frms_raw.copy()
        if zc_frms.size and float(np.max(zc_frms) - np.min(zc_frms)) > EPS:
            # Mild smoothing keeps the curve readable without changing the
            # definition. It is still normalized after raw multiplication.
            zwin = max(3, min(129, zc_frms.size // 96))
            if zwin % 2 == 0:
                zwin += 1
            if zwin > 3:
                zc_frms = moving_average_1d(zc_frms, zwin)
            lo = float(np.percentile(zc_frms, 2.0))
            hi = float(np.percentile(zc_frms, 98.0))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + EPS:
                lo = float(np.min(zc_frms))
                hi = float(np.max(zc_frms))
            zc_frms_norm = np.clip((zc_frms - lo) / (hi - lo + EPS), 0.0, 1.0)
        else:
            zc_frms_norm = np.zeros_like(zc_frms)

        valid_f0 = zc_f0_curve_hz[np.isfinite(zc_f0_curve_hz) & (zc_f0_curve_hz > 0)]
        valid_period = zc_period_curve_s[np.isfinite(zc_period_curve_s) & (zc_period_curve_s > 0)]
        zc_f0_median_hz = float(np.median(valid_f0)) if valid_f0.size else 0.0
        zc_period_median_s = float(np.median(valid_period)) if valid_period.size else 0.0

        raw = np.asarray(rows_raw[row], dtype=np.float64) if rows_raw is not None else rows_norm[row]
        raw_val = float(raw[si]) if raw.size else 0.0
        selected_row_line = np.full_like(seg_plot, coeff01, dtype=np.float64)

        return {
            "si": si,
            "row": row,
            "a": a,
            "b": b,
            "tt": tt,
            "seg_plot": seg_plot,
            "env": env,
            "local_fisher_norm": local_fisher_norm,
            "local_fisher_label": fisher_label,
            "fisher_dx2_over_rms": fisher_dx2_over_rms,
            "fisher_dx2_over_rms_raw": fisher_dx2_over_rms_raw,
            "zc_f0_norm": zc_f0_norm,
            "zc_crossings": zc_crossings,
            "zc_f0_curve_hz": zc_f0_curve_hz,
            "zc_period_curve_s": zc_period_curve_s,
            "zc_f0_median_hz": zc_f0_median_hz,
            "zc_period_median_s": zc_period_median_s,
            "zc_frms_norm": zc_frms_norm,
            "zc_frms_raw": zc_frms_raw,
            "fisher_gradient_norm": fisher_gradient,
            "segment_shannon_line": segment_shannon_line,
            "shannon_norm": shannon_norm,
            "segment_shannon_rms_line": segment_shannon_rms_line,
            "shannon_rms_norm": shannon_rms_norm,
            "selected_row_line": selected_row_line,
            "coeff01": coeff01,
            "raw_val": raw_val,
            "rms_ms_used": rms_ms_used,
            "frame": frame,
        }

    def draw_segment_waveform_overlay(self):
        """Show the real waveform inside the selected segment and analysis overlays."""
        self.ax_segment.clear()

        if self.audio is None or not self.features:
            self.ax_segment.set_title("Selected segment waveform + coefficient overlay")
            return

        try:
            c = self.compute_selected_segment_curves()
        except Exception as exc:
            self.ax_segment.text(0.5, 0.5, str(exc), transform=self.ax_segment.transAxes, ha="center", va="center")
            return

        tt = c["tt"]
        seg_plot = c["seg_plot"]
        env = c["env"]
        si = c["si"]
        row = c["row"]
        abs_time = c["a"] / float(self.sr)

        # Main audio/RMS layer stays in the normal amplitude field.
        self.ax_segment.plot(tt, seg_plot, linewidth=0.5, alpha=0.42, label="audio waveform")
        self.ax_segment.plot(tt, env, linewidth=0.8, alpha=0.82, label=f"local RMS envelope ({c['rms_ms_used']} ms)")

        # Fisher modifiers are shifted below the audio/RMS field.
        # We subtract max(RMS) from every point and use small additional offsets
        # so the two Fisher variants do not hide the audio waveform.
        rms_floor = float(np.nanmax(env)) if env.size else 1.0
        fisher_scale = 0.34
        local_fisher_down = fisher_scale * c["local_fisher_norm"] - rms_floor
        frms_down = fisher_scale * c["fisher_dx2_over_rms"] - rms_floor - 0.20

        self.ax_segment.plot(
            tt, local_fisher_down, linewidth=0.55, alpha=0.95,
            label=c["local_fisher_label"] + " ↓",
        )
        self.ax_segment.fill_between(tt, local_fisher_down, -rms_floor, alpha=0.045)

        self.ax_segment.plot(
            tt, frms_down, linewidth=0.55, alpha=0.90,
            label=f"F_RMS = dX² / RMS({c['rms_ms_used']} ms) ↓",
        )

        gradient_down = fisher_scale * c["fisher_gradient_norm"] - rms_floor - 0.40
        self.ax_segment.plot(
            tt, gradient_down, linewidth=0.55, alpha=0.95, color="blue",
            label="F_grad = dX² ↓",
        )

        # Segment-level Shannon lines remain in the upper normalized field.
        self.ax_segment.plot(
            tt, c["segment_shannon_line"], linewidth=0.5, alpha=0.98,
            label=f"segment Shannon={c['shannon_norm']:.3f}",
        )

        self.ax_segment.plot(
            tt, c["segment_shannon_rms_line"], linewidth=0.5, alpha=0.98,
            linestyle="--", label=f"segment Shannon(RMS)={c['shannon_rms_norm']:.3f}",
        )

        # Zero-crossing carrier and coupled carrier in the normal 0..1 field.
        # ZC f0 is the full-wave frequency tracker; ZC f0 × F_RMS is computed
        # from raw values first and only then normalized.
        if "zc_f0_norm" in c:
            self.ax_segment.plot(
                tt, c["zc_f0_norm"], linewidth=0.70, alpha=0.95,
                label="ZC f0 full-wave norm",
            )

        if "zc_frms_norm" in c:
            self.ax_segment.plot(
                tt, c["zc_frms_norm"], linewidth=0.85, alpha=0.98,
                label="ZC f0 × F_RMS norm",
            )

        if "zc_crossings" in c and len(c["zc_crossings"]):
            tz = np.asarray(c["zc_crossings"], dtype=np.float64) / float(self.sr)
            tz = tz[(tz >= 0.0) & (tz <= (tt[-1] if tt.size else 0.0))]
            if tz.size:
                self.ax_segment.scatter(
                    tz, np.zeros_like(tz), s=7, alpha=0.55, marker="o",
                    label="zero crossings",
                )

        self.ax_segment.axhline(-rms_floor, linewidth=0.7, linestyle=":", alpha=0.55, label="-max RMS offset")

        self.ax_segment.set_title(
            f"Selected audio segment {si} | t={abs_time:.3f}s | {self.row_names[row]} | "
            f"norm={c['coeff01']:.3f}, raw={c['raw_val']:.4g} | RMS={c['rms_ms_used']} ms | "
            f"ZC T={1000.0*c.get('zc_period_median_s', 0.0):.2f} ms, f0={c.get('zc_f0_median_hz', 0.0):.1f} Hz"
        )
        self.ax_segment.set_xlabel("time inside selected segment (s)")
        self.ax_segment.set_ylabel("audio/RMS; Fisher shifted below -max RMS")
        self.ax_segment.set_ylim(-1.62, 1.08)
        self.ax_segment.grid(True, alpha=0.3, linestyle="--")

        self._play_cursor = self.ax_segment.axvline(
            0.0,
            color="red",
            linewidth=2.0,
            alpha=0.90,
            label="play cursor",
            zorder=50,
        )

        self.ax_segment.legend(fontsize=8, loc="upper right")


    def draw_fisher_rms_orbit(self):
        """RMS-centered Fisher harmonic orbit below the waveform plot.

        The orbit width is a local high-frequency/Fisher density measured inside
        one fundamental period, where one period is defined by two zero-crossing
        half-waves: crossing[k] -> crossing[k+2].

        Center line: local RMS envelope.
        Upper/lower orbit: RMS ± k * HF_density.
        HF_density: number of Fisher-RMS peaks per full-wave period.
        """
        self.ax_orbit.clear()

        if self.audio is None or not self.features:
            self.ax_orbit.set_title("RMS-centered Fisher harmonic orbit")
            return

        try:
            c = self.compute_selected_segment_curves()
        except Exception as exc:
            self.ax_orbit.text(
                0.5, 0.5, str(exc), transform=self.ax_orbit.transAxes,
                ha="center", va="center"
            )
            return

        tt = np.asarray(c["tt"], dtype=np.float64)
        if tt.size < 4:
            self.ax_orbit.set_title("RMS-centered Fisher harmonic orbit")
            return

        env = np.asarray(c["env"], dtype=np.float64)
        env = np.nan_to_num(env, nan=0.0, posinf=0.0, neginf=0.0)
        if np.max(env) > EPS:
            env = env / (np.max(env) + EPS)

        # Fisher source for local harmonic density.  F_RMS is better here than
        # sample-denominator Fisher, because it is stabilized by the RMS envelope.
        fisher_src = np.asarray(c.get("fisher_dx2_over_rms", c.get("local_fisher_norm", env)), dtype=np.float64)
        fisher_src = np.nan_to_num(fisher_src, nan=0.0, posinf=0.0, neginf=0.0)
        if fisher_src.size != tt.size:
            fisher_src = np.resize(fisher_src, tt.size)
        if np.max(fisher_src) > np.min(fisher_src) + EPS:
            fisher_src = (fisher_src - np.min(fisher_src)) / (np.max(fisher_src) - np.min(fisher_src) + EPS)
        else:
            fisher_src = np.zeros_like(tt)

        # Simple dependency-free peak detector.
        # A peak is a local maximum above a robust threshold.
        if fisher_src.size >= 3:
            thr = float(np.median(fisher_src) + 0.45 * np.std(fisher_src))
            peak_mask = np.zeros_like(fisher_src, dtype=bool)
            peak_mask[1:-1] = (
                (fisher_src[1:-1] > fisher_src[:-2])
                & (fisher_src[1:-1] >= fisher_src[2:])
                & (fisher_src[1:-1] > thr)
            )
            peak_idx = np.where(peak_mask)[0]
        else:
            peak_idx = np.array([], dtype=int)

        crossings = np.asarray(c.get("zc_crossings", []), dtype=np.float64)
        crossings = crossings[np.isfinite(crossings)]
        crossings = crossings[(crossings >= 0.0) & (crossings < tt.size - 1)]

        hf_density = np.zeros_like(tt, dtype=np.float64)
        period_markers = []

        # One fundamental period = two half-waves = crossing[k] to crossing[k+2].
        if crossings.size >= 3:
            for k in range(crossings.size - 2):
                a = int(max(0, np.floor(crossings[k])))
                b = int(min(tt.size, np.ceil(crossings[k + 2])))
                if b <= a + 1:
                    continue
                period_s = float((crossings[k + 2] - crossings[k]) / float(self.sr))
                if period_s <= EPS:
                    continue
                n_peaks = int(np.sum((peak_idx >= a) & (peak_idx < b)))
                # Peaks per full period.  This is a local high-frequency
                # distribution index locked to the zero-crossing carrier.
                hf_density[a:b] = float(n_peaks) / period_s
                if k % 2 == 0:
                    period_markers.append(a)
        else:
            # Fallback: use already computed coupled carrier when crossings are sparse.
            hf_density = np.asarray(c.get("zc_frms_norm", fisher_src), dtype=np.float64)
            if hf_density.size != tt.size:
                hf_density = np.resize(hf_density, tt.size)

        if np.max(hf_density) > np.min(hf_density) + EPS:
            hf_norm = (hf_density - np.min(hf_density)) / (np.max(hf_density) - np.min(hf_density) + EPS)
        else:
            hf_norm = np.zeros_like(tt)

        # Add a small contribution from ZC_f0×F_RMS to keep the orbit continuous
        # inside periods with no detected sharp peaks.
        zc_frms = np.asarray(c.get("zc_frms_norm", np.zeros_like(tt)), dtype=np.float64)
        if zc_frms.size != tt.size:
            zc_frms = np.resize(zc_frms, tt.size)
        zc_frms = np.nan_to_num(zc_frms, nan=0.0, posinf=0.0, neginf=0.0)
        orbit_width = 0.75 * hf_norm + 0.25 * zc_frms
        if np.max(orbit_width) > np.min(orbit_width) + EPS:
            orbit_width = (orbit_width - np.min(orbit_width)) / (np.max(orbit_width) - np.min(orbit_width) + EPS)

        orbit_gain = 0.34
        upper = env + orbit_gain * orbit_width
        lower = env - orbit_gain * orbit_width

        self.ax_orbit.plot(tt, env, linewidth=1.0, alpha=0.95, label=f"RMS center ({c['rms_ms_used']} ms)")
        self.ax_orbit.plot(tt, upper, linewidth=0.90, alpha=0.95, label="RMS + Fisher period density")
        self.ax_orbit.plot(tt, lower, linewidth=0.90, alpha=0.95, label="RMS - Fisher period density")
        self.ax_orbit.fill_between(tt, env, upper, alpha=0.18)
        self.ax_orbit.fill_between(tt, lower, env, alpha=0.18)

        if peak_idx.size:
            y_peaks = env[peak_idx] + orbit_gain * 0.10
            self.ax_orbit.scatter(tt[peak_idx], y_peaks, s=10, alpha=0.85, label="Fisher peaks")

        for a in period_markers:
            if 0 <= a < tt.size:
                self.ax_orbit.axvline(tt[a], linewidth=0.65, linestyle="--", alpha=0.35)

        self.ax_orbit.set_title(
            "RMS-centered Fisher harmonic orbit | "
            "2 zero-crossing half-waves = 1 local fundamental period"
        )
        self.ax_orbit.set_xlabel("time inside selected segment (s)")
        self.ax_orbit.set_ylabel("RMS ± HF density")
        self.ax_orbit.grid(True, alpha=0.3, linestyle="--")
        self.ax_orbit.legend(fontsize=8, loc="upper right")



    def start_play_cursor(self, duration):
        """Start moving the red cursor in the lower waveform plot."""
        self.stop_play_cursor_only()
        self._play_duration = float(max(0.0, duration))
        self._play_start_clock = time.perf_counter()

        if self._play_cursor is None:
            self._play_cursor = self.ax_segment.axvline(
                0.0, color="red", linewidth=2.0, alpha=0.90, zorder=50
            )
        else:
            self._play_cursor.set_xdata([0.0, 0.0])
            self._play_cursor.set_visible(True)

        self.canvas.draw_idle()
        self.update_play_cursor()

    def update_play_cursor(self):
        """Timer callback: move cursor according to elapsed playback time."""
        if not self._is_playing or self._play_start_clock is None:
            return

        elapsed = time.perf_counter() - self._play_start_clock
        if elapsed >= self._play_duration:
            if self._play_cursor is not None:
                self._play_cursor.set_xdata([self._play_duration, self._play_duration])
                self.canvas.draw_idle()
            self._is_playing = False
            self._play_after_id = None
            return

        if self._play_cursor is not None:
            self._play_cursor.set_xdata([elapsed, elapsed])
            self.canvas.draw_idle()

        self._play_after_id = self.root.after(20, self.update_play_cursor)

    def stop_play_cursor_only(self):
        """Stop the GUI timer without stopping audio playback."""
        if self._play_after_id is not None:
            try:
                self.root.after_cancel(self._play_after_id)
            except Exception:
                pass
            self._play_after_id = None

    def _normalize_play_signal(self, y: np.ndarray) -> np.ndarray:
        """Prepare any selected curve as audible float32 signal."""
        y = np.asarray(y, dtype=np.float64)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        if y.size < 2:
            raise RuntimeError("Selected play source is empty.")
        y = y - float(np.mean(y))
        peak = float(np.max(np.abs(y)) + EPS)
        if peak > 0:
            y = 0.95 * y / peak
        return y.astype(np.float32)

    def get_selected_segment_audio(self, pad_ms=0.0):
        """Return the selected audio/analysis source as float32 audio and sample rate."""
        if self.audio is None or not self.features:
            raise RuntimeError("Open and compute an audio file first.")

        source = "audio"
        if self.play_source_var is not None:
            try:
                source = str(self.play_source_var.get())
            except Exception:
                source = "audio"

        c = self.compute_selected_segment_curves()
        si = c["si"]
        t0 = c["a"] / float(self.sr)
        t1 = c["b"] / float(self.sr)

        if source == "audio":
            y = c["seg_plot"]
        elif source == "RMS envelope":
            y = c["env"]
        elif source == "local Fisher":
            y = c["local_fisher_norm"]
        elif source == "F_RMS dX2/RMS":
            y = c["fisher_dx2_over_rms"]
        elif source == "3-point gradient Fisher":
            y = c["fisher_gradient_norm"]
        elif source == "segment Shannon":
            # Constant DC would be inaudible after centering; make a soft tone
            # with amplitude controlled by the segment Shannon coefficient.
            amp = float(c["shannon_norm"])
            tt = c["tt"]
            y = amp * np.sin(2.0 * np.pi * 440.0 * tt)
        elif source == "segment Shannon(RMS)":
            # Same idea, but amplitude is controlled by Shannon of the RMS envelope.
            amp = float(c["shannon_rms_norm"])
            tt = c["tt"]
            y = amp * np.sin(2.0 * np.pi * 660.0 * tt)
        elif source == "selected row":
            amp = float(c["coeff01"])
            tt = c["tt"]
            y = amp * np.sin(2.0 * np.pi * 330.0 * tt)
        else:
            y = c["seg_plot"]

        y = self._normalize_play_signal(y)
        return y, int(self.sr), si, t0, t1

    def play_selected_segment(self):
        """Audition the selected segment directly from the GUI.

        Preferred backend: sounddevice, if installed.
        Fallback: write a temporary WAV and use ffplay, paplay or aplay.
        """
        try:
            y, sr, si, t0, t1 = self.get_selected_segment_audio(pad_ms=0.0)
            self.stop_playback(silent=True)

            duration = len(y) / float(sr)

            if SOUNDDEVICE_AVAILABLE:
                sd.play(y, sr, blocking=False)
                self._is_playing = True
                self.start_play_cursor(duration)
                source = self.play_source_var.get() if self.play_source_var is not None else "audio"
                self.status_var.set(
                    f"Playing {source} | segment {si} | {t0:.3f}–{t1:.3f}s | backend=sounddevice"
                )
                return

            tmp = tempfile.NamedTemporaryFile(prefix="entropy_selected_segment_", suffix=".wav", delete=False)
            tmp_path = tmp.name
            tmp.close()
            sf.write(tmp_path, y, sr)
            self._last_play_tmp = tmp_path

            player_cmd = None
            if shutil.which("ffplay"):
                player_cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", tmp_path]
            elif shutil.which("paplay"):
                player_cmd = ["paplay", tmp_path]
            elif shutil.which("aplay"):
                player_cmd = ["aplay", tmp_path]

            if player_cmd is None:
                raise RuntimeError(
                    "No playback backend found. Install one of:\n"
                    "  pip install sounddevice\n"
                    "or system player:\n"
                    "  sudo apt install ffmpeg\n"
                    "or:\n"
                    "  sudo apt install alsa-utils"
                )

            self._play_process = subprocess.Popen(player_cmd)
            self._is_playing = True
            self.start_play_cursor(len(y) / float(sr))
            source = self.play_source_var.get() if self.play_source_var is not None else "audio"
            self.status_var.set(
                f"Playing {source} | segment {si} | {t0:.3f}–{t1:.3f}s | backend={player_cmd[0]}"
            )

        except Exception as exc:
            traceback.print_exc()
            messagebox.showerror("Playback failed", str(exc))

    def stop_playback(self, silent=False):
        """Stop playback and freeze/hide the moving cursor."""
        try:
            if SOUNDDEVICE_AVAILABLE:
                sd.stop()

            if self._play_process is not None:
                try:
                    self._play_process.terminate()
                except Exception:
                    pass
                self._play_process = None

            self.stop_play_cursor_only()
            self._is_playing = False
            self._play_start_clock = None

            if self._play_cursor is not None:
                try:
                    self._play_cursor.set_xdata([0.0, 0.0])
                    self.canvas.draw_idle()
                except Exception:
                    pass

            if not silent:
                self.status_var.set("Playback stopped.")
        except Exception:
            pass

    def on_click(self, event):
        if not self.features:
            return

        # Click on the heatmap: select both row and segment/time.
        if event.inaxes == self.ax_rows:
            if event.ydata is None:
                return
            row = int(np.floor(event.ydata))
            if 0 <= row < len(self.row_names):
                self.selected_row = row
            if event.xdata is not None:
                t = self.features.get("time_segments", np.array([]))
                if len(t):
                    self.selected_segment = int(np.clip(np.searchsorted(t, float(event.xdata)), 0, len(t) - 1))
            self.draw_all()
            self.status_var.set(
                f"Selected row {self.selected_row + 1}: {self.row_names[self.selected_row]} | "
                f"segment {self.selected_segment}"
            )
            return

        # Click on selected curve or RMS preview: select segment/time only.
        if event.inaxes in (self.ax_selected, self.ax_audio, self.ax_orbit) and event.xdata is not None:
            t = self.features.get("time_segments", np.array([]))
            if len(t):
                self.selected_segment = int(np.clip(np.searchsorted(t, float(event.xdata)), 0, len(t) - 1))
                self.draw_all()
                self.status_var.set(f"Selected segment {self.selected_segment}")


def main():
    root = tk.Tk()
    EntropyRowsViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
