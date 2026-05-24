#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gc
import time
import traceback
import re
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


from scipy.linalg import eigh
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_pdf import PdfPages


# ============================================================
# OPTIONAL MEMORY LOGGING
# ============================================================

try:
    import psutil
    PSUTIL_AVAILABLE = True
except Exception:
    PSUTIL_AVAILABLE = False


# ============================================================
# CPU-ONLY BACKEND
# ============================================================

# CUDA/CuPy is intentionally disabled for this workstation.
# Quadro K2000 / sm_30 is not compatible with the current CuPy/CUDA11 NVRTC path.
cp = None
GPU_AVAILABLE = False
print("CPU-only backend: NumPy/SciPy + multiprocessing", flush=True)


EPS = 1e-12
EXTS = {".flac", ".wav"}
BINS = 128
DEFAULT_CPU_WORKERS = max(1, (os.cpu_count() or 2) - 1)

# Number of intra-segment points used in vertical packet MP scan.
# For every real segment, each song contributes this many local feature points.
PACKET_POINTS = 64


# ============================================================
# LOGGING
# ============================================================

def log(msg):
    print(msg, flush=True)


def log_memory(label="MEM"):
    if not PSUTIL_AVAILABLE:
        return
    try:
        p = psutil.Process(os.getpid())
        ram = p.memory_info().rss / 1024**3
        print(f"[{label}] RAM used: {ram:.3f} GB", flush=True)
    except Exception:
        pass


# ============================================================
# FILES / AUDIO
# ============================================================

def list_audio_files(folder: Path):
    files = sorted([p for p in folder.rglob("*") if p.suffix.lower() in EXTS])
    if not files:
        raise RuntimeError("No .flac/.wav files found.")
    return files


def read_mono(path):
    y, sr = sf.read(str(path))

    if y.ndim > 1:
        y = y.mean(axis=1)

    y = y.astype(np.float64)
    y -= y.mean()

    return y, sr

def strip_flag_emojis(text):
    """Remove regional-indicator flag emojis that break Matplotlib PDF fonts."""
    if text is None:
        return ""
    return re.sub(r"[\U0001F1E6-\U0001F1FF]+", "", str(text))




def dc_hysteresis_gate(d, band=0.10):
    """Return a binary low-level gate for a Skorokhod-like distance curve.

    Parameters
    ----------
    d : array-like
        Distance curve, normally D[j] = |tau_song[j] - tau_ref[j]|.
    band : float
        Hysteresis half-width around the DC level.  The ON threshold is
        DC + band and the OFF threshold is DC - band.

    Returns
    -------
    gate : ndarray
        0 when OFF and -0.1 when ON, so it can be drawn under the main curve.
    dc, hi, lo : float
        Median DC level and the two hysteresis thresholds.
    state01 : ndarray
        Logical 0/1 state for CSV export.
    """
    d = np.asarray(d, dtype=np.float64)
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

    if d.size == 0:
        return d.copy(), 0.0, float(band), -float(band), d.astype(np.int32)

    dc = float(np.nanmedian(d))
    hi = dc + float(band)
    lo = dc - float(band)

    state = 0
    state01 = np.zeros_like(d, dtype=np.int32)
    gate = np.zeros_like(d, dtype=np.float64)

    for i, x in enumerate(d):
        if state == 0 and x > hi:
            state = 1
        elif state == 1 and x < lo:
            state = 0

        state01[i] = state
        gate[i] = -0.10 if state else 0.0

    return gate, dc, hi, lo, state01


def clean_song_label_from_path(path, fallback_index=None, max_len=42):
    """Return one compact heatmap/PDF label per song row.

    Keeps only the file number and the useful song name.  It removes common
    YouTube/Eurovision suffixes and the mono marker, so the heatmap does not
    become unreadable with long repeated titles.
    """
    stem = Path(path).stem
    label = strip_flag_emojis(stem)

    # Remove mono/conversion suffixes.
    for suffix in ("_mono", "-mono", " mono", "_MONO", "-MONO"):
        if label.endswith(suffix):
            label = label[: -len(suffix)]

    # Remove very repetitive webpage/video decorations.
    cuts = [
        "#Eurovision",
        "Official Music Video",
        "Official Video",
        "National Final Performance",
        "Grand Final",
        "Eurovision 2026",
    ]
    for c in cuts:
        pos = label.lower().find(c.lower())
        if pos >= 0:
            label = label[:pos]

    # Normalize separators/spaces.
    label = label.replace("｜", "|").replace("_", " ")
    label = " ".join(label.split())
    label = label.strip(" -_|.")

    # Try to keep the leading file number if present. Otherwise add rank/index.
    m = re.match(r"^\s*(\d{1,3})\s*[-_. ]+(.+)$", label)
    if m:
        num = int(m.group(1))
        name = m.group(2).strip(" -_|.")
        out = f"{num:02d} | {name}"
    else:
        if fallback_index is None:
            out = label
        else:
            out = f"{int(fallback_index):02d} | {label}"

    if len(out) > max_len:
        out = out[: max_len - 1] + "…"
    return out


# ============================================================
# SHANNON / FISHER FEATURES
# ============================================================

def shannon_entropy_blocks(blocks, bins=BINS):
    n_seg = blocks.shape[0]

    b_min = blocks.min(axis=1, keepdims=True)
    b_max = blocks.max(axis=1, keepdims=True)

    norm = (blocks - b_min) / (b_max - b_min + EPS)
    idx = np.clip((norm * bins).astype(np.int32), 0, bins - 1)

    hist = np.zeros((n_seg, bins), dtype=np.float64)
    rows = np.arange(n_seg)[:, None]
    np.add.at(hist, (rows, idx), 1.0)

    p = hist / (hist.sum(axis=1, keepdims=True) + EPS)

    return -np.sum(p * np.log(p + EPS), axis=1)


def fisher_blocks(blocks):
    grad = np.abs(np.diff(blocks, axis=1))
    intensity = np.abs(blocks[:, :-1])

    F = grad / (intensity + 1e-3)

    return np.percentile(F, 95, axis=1) + EPS


def audio_energy_blocks(blocks):
    """Direct audio-domain feature per segment: log1p(RMS)."""
    blocks = np.asarray(blocks, dtype=np.float64)
    rms = np.sqrt(np.mean(blocks ** 2, axis=1) + EPS)
    rms = np.nan_to_num(rms, nan=0.0, posinf=0.0, neginf=0.0)
    return np.log1p(rms)



# ============================================================
# INTRA-SEGMENT SHANNON/FISHER PEAK POSITIONS
# ============================================================

def _sf_score_1d(chunk, bins=BINS):
    """Fast Shannon/Fisher-like score for one short 1D window."""
    x = np.asarray(chunk, dtype=np.float64)
    if x.size < 8:
        return 0.0

    x = x - np.mean(x)
    peak = float(np.max(np.abs(x)) + EPS)
    x = x / peak

    # Shannon entropy of amplitude distribution.
    hist, _ = np.histogram(x, bins=bins, range=(-1.0, 1.0))
    p = hist.astype(np.float64)
    p = p / (np.sum(p) + EPS)
    H = -float(np.sum(p * np.log(p + EPS)))

    # Fisher-like structural response inside the same local window.
    g = np.abs(np.diff(x))
    inten = np.abs(x[:-1])
    F = float(np.percentile(g / (inten + 1e-3), 95)) + EPS

    ratio = H / max(abs(F), 1e-9)
    ratio = np.nan_to_num(ratio, nan=0.0, posinf=1e6, neginf=0.0)
    return float(np.log1p(np.clip(ratio, 0.0, 1e6)))


def segment_internal_sf_peaks(path, min_segments, seg_size, n_probe=255, win_div=64):
    """For each real audio segment, find the internal Shannon/Fisher maximum.

    Returns
    -------
    tau : ndarray, shape (min_segments,)
        Peak position inside each segment, normalized to [0, 1].
    amp : ndarray, shape (min_segments,)
        Shannon/Fisher peak score inside each segment.

    This is intentionally not a waveform alignment. It extracts one event position
    per segment, so later we can compare a song to the collective K=1 reference by:
        D[j] = |tau_song[j] - tau_ref[j]|.
    """
    y, sr = read_mono(Path(path))
    y = y.astype(np.float64, copy=False)
    y = y / (np.sqrt(np.mean(y ** 2)) + EPS)
    y = y[:int(min_segments) * int(seg_size)]

    if y.size < int(min_segments) * int(seg_size):
        tmp = np.zeros(int(min_segments) * int(seg_size), dtype=np.float64)
        tmp[:y.size] = y
        y = tmp

    blocks = y.reshape(int(min_segments), int(seg_size))

    n_probe = int(max(16, n_probe))
    seg_size = int(seg_size)
    win = int(max(32, min(seg_size, seg_size // max(4, int(win_div)))))
    half = max(1, win // 2)

    centers = np.linspace(0, seg_size - 1, n_probe).astype(np.int64)
    tau = np.zeros(int(min_segments), dtype=np.float64)
    amp = np.zeros(int(min_segments), dtype=np.float64)

    for si in range(int(min_segments)):
        seg = blocks[si]
        best_score = -np.inf
        best_center = 0

        for c in centers:
            a = max(0, int(c) - half)
            b = min(seg_size, int(c) + half)
            score = _sf_score_1d(seg[a:b])
            if score > best_score:
                best_score = score
                best_center = int(c)

        tau[si] = best_center / max(1, seg_size - 1)
        amp[si] = 0.0 if not np.isfinite(best_score) else float(best_score)

    return tau, amp


def _segment_peak_worker(args):
    """Worker: intra-segment Shannon/Fisher peaks for one song.

    Returns normalized peak position tau[segment] and peak score amp[segment].
    This is intentionally top-level so ProcessPoolExecutor can pickle it.
    """
    idx, path_str, min_segments, seg_size, n_probe = args
    tau, amp = segment_internal_sf_peaks(
        Path(path_str),
        int(min_segments),
        int(seg_size),
        n_probe=int(n_probe),
    )
    return idx, tau, amp, Path(path_str).name


def compute_all_segment_peak_maps(files, min_segments, seg_size, status_callback=None, n_probe=255, n_workers=1):
    """Compute intra-segment Shannon/Fisher peak positions for all songs.

    Parallelized per song.  Each worker reads exactly one audio file, scans all
    real segments, finds the intra-segment Shannon/Fisher maximum, and returns:
        tau_all[song, segment] in [0, 1]
        amp_all[song, segment] = local Shannon/Fisher peak score

    The PDF assembler later uses these cached matrices and does not re-read audio.
    """
    n = len(files)
    T = int(min_segments)
    n_workers = int(n_workers or 1)
    n_workers = max(1, min(n_workers, n))

    tau_all = np.zeros((n, T), dtype=np.float64)
    amp_all = np.zeros((n, T), dtype=np.float64)

    log("=" * 70)
    log("INTRA-SEGMENT SHANNON/FISHER PEAK MAPS — PARALLEL PER SONG")
    log(f"Files       : {n}")
    log(f"Segments    : {T}")
    log(f"Seg size    : {seg_size}")
    log(f"Probe points: {n_probe}")
    log(f"CPU workers : {n_workers}")
    log(f"Feature mode: {feature_mode}")
    log(f"Geometry    : {matrix_geometry}")
    log("=" * 70)

    if n_workers <= 1:
        for i, p in enumerate(files, 1):
            msg = f"[{i}/{n}] intra-segment SF peaks: {Path(p).name}"
            log(msg)
            if status_callback:
                status_callback(msg)
            tau, amp = segment_internal_sf_peaks(p, T, seg_size, n_probe=n_probe)
            tau_all[i - 1, :] = tau
            amp_all[i - 1, :] = amp
    else:
        tasks = [(i, str(p), T, int(seg_size), int(n_probe)) for i, p in enumerate(files)]
        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = mp.get_context("spawn")

        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            futures = [ex.submit(_segment_peak_worker, task) for task in tasks]
            done = 0
            for fut in as_completed(futures):
                idx, tau, amp, name = fut.result()
                tau_all[idx, :] = tau
                amp_all[idx, :] = amp
                done += 1
                msg = f"[{done}/{n}] intra-segment SF peaks done: {name}"
                log(msg)
                if status_callback:
                    status_callback(msg)

    amp_all = np.nan_to_num(amp_all, nan=0.0, posinf=0.0, neginf=0.0)
    tau_all = np.nan_to_num(tau_all, nan=0.0, posinf=0.0, neginf=0.0)
    return tau_all, amp_all


def weighted_reference_peak_position(tau_all, weights):
    """Collective reference peak position per segment from all songs.

    tau_all: (songs, segments), normalized [0,1]
    weights: (songs, segments), typically positive K=1 MP reconstruction density
    """
    tau_all = np.asarray(tau_all, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.maximum(w, 0.0) + EPS
    return np.sum(w * tau_all, axis=0) / (np.sum(w, axis=0) + EPS)


def sanitize_vector(v, label="vector"):
    """Remove NaN/Inf from one feature vector without destroying its shape."""
    v = np.asarray(v, dtype=np.float64)
    bad = ~np.isfinite(v)
    if np.any(bad):
        good = v[np.isfinite(v)]
        fill = float(np.median(good)) if good.size else 0.0
        log(f"WARNING: {label}: replaced {int(bad.sum())} NaN/Inf values with {fill:.6g}")
        v = v.copy()
        v[bad] = fill
    return v


def sanitize_matrix(X, label="matrix"):
    """Make MP/PCA input finite and robust against constant/invalid columns."""
    X = np.asarray(X, dtype=np.float64)
    if not np.all(np.isfinite(X)):
        bad = ~np.isfinite(X)
        log(f"WARNING: {label}: replacing {int(bad.sum())} NaN/Inf values")
        X = X.copy()
        # Replace bad values column-by-column with the finite column median.
        for j in np.where(np.any(bad, axis=0))[0]:
            col = X[:, j]
            finite = np.isfinite(col)
            fill = float(np.median(col[finite])) if np.any(finite) else 0.0
            col[~finite] = fill
            X[:, j] = col
    return X


def song_entropy_fisher_vectors(path, seg_size, min_segments):
    """Return separate Shannon and Fisher-max trajectories for one song."""
    y, sr = read_mono(path)
    y /= np.sqrt(np.mean(y ** 2)) + EPS
    y = y[:min_segments * seg_size]
    blocks = y.reshape(min_segments, seg_size)

    H = shannon_entropy_blocks(blocks)
    F = fisher_blocks(blocks)

    H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)

    H = sanitize_vector(H, label=Path(path).name + " [Shannon]")
    F = sanitize_vector(F, label=Path(path).name + " [Fisher max]")

    del y, blocks
    gc.collect()
    return H, F, sr


def build_entropy_fisher_matrices(files, seg_size, min_segments, status_callback=None, n_workers=1):
    """Build H and F matrices for Shannon-Fisher diagnostic space."""
    H_list = []
    F_list = []
    for i, p in enumerate(files, 1):
        msg = f"[{i}/{len(files)}] Shannon/Fisher memory: {p.name}"
        log(msg)
        if status_callback:
            status_callback(msg)
        H, F, _ = song_entropy_fisher_vectors(p, seg_size, min_segments)
        H_list.append(H)
        F_list.append(F)

    H = sanitize_matrix(np.vstack(H_list).astype(np.float64, copy=False), label="Shannon matrix H")
    F = sanitize_matrix(np.vstack(F_list).astype(np.float64, copy=False), label="Fisher matrix F")
    return H, F


def segment_hf_fisher_lag_vectors(y, seg_size, min_segments, n_sub=64):
    """For each segment: X=max H/F position; Y=signed Fisher lag.

    y_lag = x_hf - x_fisher.
    Positive means Fisher max is earlier than H/F max.
    Negative means Fisher max is later than H/F max.
    """
    y = np.asarray(y, dtype=np.float64)
    seg_size = int(seg_size)
    min_segments = int(min_segments)
    y = y[:min_segments * seg_size]
    if y.size < min_segments * seg_size:
        tmp = np.zeros(min_segments * seg_size, dtype=np.float64)
        tmp[:y.size] = y
        y = tmp

    n_sub = int(max(8, min(int(n_sub), max(8, seg_size // 8))))
    sub_size = int(seg_size // n_sub)
    if sub_size < 8:
        n_sub = max(1, seg_size // 8)
        sub_size = max(8, seg_size // max(1, n_sub))

    usable = n_sub * sub_size
    blocks = y.reshape(min_segments, seg_size)[:, :usable]
    subblocks = blocks.reshape(min_segments * n_sub, sub_size)

    Hsub = shannon_entropy_blocks(subblocks).reshape(min_segments, n_sub)
    Fsub = fisher_blocks(subblocks).reshape(min_segments, n_sub)
    Hsub = np.nan_to_num(Hsub, nan=0.0, posinf=0.0, neginf=0.0)
    Fsub = np.nan_to_num(Fsub, nan=0.0, posinf=0.0, neginf=0.0)

    HF = Hsub / (np.abs(Fsub) + 1e-9)
    HF = np.nan_to_num(HF, nan=0.0, posinf=0.0, neginf=0.0)

    idx_hf = np.argmax(HF, axis=1)
    idx_f = np.argmax(Fsub, axis=1)

    x_hf = (idx_hf + 0.5) / max(1, n_sub)
    x_f = (idx_f + 0.5) / max(1, n_sub)
    y_lag = x_hf - x_f

    hf_amp = np.max(HF, axis=1)
    f_amp = np.max(Fsub, axis=1)
    return x_hf, y_lag, hf_amp, f_amp


def segment_hf_fisher_com_pm_vectors(y, seg_size, min_segments, n_sub=64):
    """For each segment compute signed H/F max and Fisher max relative to H/F COM.

    Returns
    -------
    x_hf_pm : ndarray
        pos(max H/F) - COM(H/F). Positive = later/right of COM.

    y_f_pm : ndarray
        pos(max Fisher) - COM(H/F). Positive = later/right of COM.

    hf_amp, f_amp : ndarray
        amplitudes for optional interpretation.

    com_hf : ndarray
        H/F center of mass position in segment, normalized 0..1.
    """
    y = np.asarray(y, dtype=np.float64)
    seg_size = int(seg_size)
    min_segments = int(min_segments)
    y = y[:min_segments * seg_size]
    if y.size < min_segments * seg_size:
        tmp = np.zeros(min_segments * seg_size, dtype=np.float64)
        tmp[:y.size] = y
        y = tmp

    n_sub = int(max(8, min(int(n_sub), max(8, seg_size // 8))))
    sub_size = int(seg_size // n_sub)
    if sub_size < 8:
        n_sub = max(1, seg_size // 8)
        sub_size = max(8, seg_size // max(1, n_sub))

    usable = n_sub * sub_size
    blocks = y.reshape(min_segments, seg_size)[:, :usable]
    subblocks = blocks.reshape(min_segments * n_sub, sub_size)

    Hsub = shannon_entropy_blocks(subblocks).reshape(min_segments, n_sub)
    Fsub = fisher_blocks(subblocks).reshape(min_segments, n_sub)

    Hsub = np.nan_to_num(Hsub, nan=0.0, posinf=0.0, neginf=0.0)
    Fsub = np.nan_to_num(Fsub, nan=0.0, posinf=0.0, neginf=0.0)

    HF = Hsub / (np.abs(Fsub) + 1e-9)
    HF = np.nan_to_num(HF, nan=0.0, posinf=0.0, neginf=0.0)

    pos = (np.arange(n_sub, dtype=np.float64) + 0.5) / max(1, n_sub)

    # Use non-negative shifted HF weights for stable center of mass.
    W = HF - np.min(HF, axis=1, keepdims=True)
    W = W + 1e-12
    com_hf = np.sum(W * pos[None, :], axis=1) / (np.sum(W, axis=1) + 1e-12)

    idx_hf = np.argmax(HF, axis=1)
    idx_f = np.argmax(Fsub, axis=1)

    pos_hf = pos[idx_hf]
    pos_f = pos[idx_f]

    x_hf_pm = pos_hf - com_hf
    y_f_pm = pos_f - com_hf

    hf_amp = np.max(HF, axis=1)
    f_amp = np.max(Fsub, axis=1)

    return x_hf_pm, y_f_pm, hf_amp, f_amp, com_hf


def song_sf_vector(path, seg_size, min_segments, feature_mode="entropy"):
    """Return one temporal feature trajectory for one song.

    feature_mode="entropy":
        v[j] = log(1 + H(segment_j) / F(segment_j))

    feature_mode="audio":
        v[j] = log(1 + RMS(segment_j))
    """
    y, sr = read_mono(path)

    y /= np.sqrt(np.mean(y ** 2)) + EPS
    y = y[:min_segments * seg_size]

    blocks = y.reshape(min_segments, seg_size)

    if feature_mode == "audio":
        v = audio_energy_blocks(blocks)
        v = sanitize_vector(v, label=Path(path).name + " [audio/RMS]")
        del y, blocks
        gc.collect()
        return v, sr

    H = shannon_entropy_blocks(blocks)
    F = fisher_blocks(blocks)

    # Robust Shannon/Fisher feature.
    # With long segments Fisher can become extremely small/invalid;
    # direct log1p(H/F) may produce NaN/Inf.
    H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
    F = np.nan_to_num(F, nan=EPS, posinf=EPS, neginf=EPS)
    F = np.maximum(np.abs(F), 1e-9)

    ratio = H / F
    ratio = np.nan_to_num(ratio, nan=0.0, posinf=1e6, neginf=0.0)
    ratio = np.clip(ratio, 0.0, 1e6)

    v = np.log1p(ratio)
    v = sanitize_vector(v, label=Path(path).name + " [entropy H/F]")

    del y, blocks, H, F
    gc.collect()

    return v, sr




def _song_sf_worker(args):
    """Top-level worker for multiprocessing feature extraction."""
    if len(args) == 5:
        idx, path_str, seg_size, min_segments, feature_mode = args
    else:
        idx, path_str, seg_size, min_segments = args
        feature_mode = "entropy"
    v, sr = song_sf_vector(Path(path_str), seg_size, min_segments, feature_mode=feature_mode)
    return idx, v, sr, Path(path_str).name


def xrec_to_segment_gain(rec_seg, dry, wet, rec_lo=None, rec_hi=None):
    """Convert the K-mode MP inverse reconstruction row to per-segment gains.

    This is the direct filter requested by the user:
        for every audio segment s, take Xrec_K[file, s] and use it as
        the coefficient for that exact audio segment.

    No interpolation is used here.  The exported WAV is reconstructed
    segment-by-segment from the selected K eigenmodes.
    """
    rec = np.asarray(rec_seg, dtype=np.float64)
    rec = np.nan_to_num(rec, nan=0.0, posinf=0.0, neginf=0.0)

    if rec_lo is None:
        rec_lo = float(np.percentile(rec, 2.0))
    if rec_hi is None:
        rec_hi = float(np.percentile(rec, 98.0))

    if abs(rec_hi - rec_lo) < EPS:
        coeff01 = np.ones_like(rec)
    else:
        coeff01 = (rec - rec_lo) / (rec_hi - rec_lo + EPS)
        coeff01 = np.clip(coeff01, 0.0, 1.0)

    # Slight contrast: makes K differences visible/audible without destroying audio.
    coeff01 = coeff01 ** 1.25

    return dry + wet * coeff01


def _render_song_worker(args):
    """Top-level worker for multiprocessing audio rendering.

    Important: this function does NOT interpolate a smooth envelope.
    It applies one MP-reconstructed coefficient per original audio segment.
    """
    (
        idx,
        path_str,
        rec_seg,
        min_segments,
        seg_size,
        out_dir_str,
        n_modes,
        dry,
        wet,
        rec_lo,
        rec_hi,
        normalize_mode,
    ) = args

    p = Path(path_str)
    out_dir = Path(out_dir_str)

    y, sr = read_mono(p)

    work_len = min_segments * seg_size
    y_work = y[:work_len]
    rest = y[work_len:]

    # One coefficient per segment, reconstructed from selected K modes.
    seg_gain = xrec_to_segment_gain(rec_seg, dry, wet, rec_lo, rec_hi)

    # Direct segment-by-segment reconstruction:
    # segment_new[s] = segment_original[s] * Xrec_K_coefficient[s]
    blocks = y_work.reshape(min_segments, seg_size).copy()
    blocks *= seg_gain[:, None]
    y_filtered_work = blocks.reshape(-1)

    if len(rest) > 0:
        # Remaining tail was not part of the MP matrix; keep only dry component.
        y_filtered = np.concatenate([y_filtered_work, rest * dry])
    else:
        y_filtered = y_filtered_work

    # Keep the MP attenuation/amplification audible. Do NOT RMS-normalize
    # back to the original level by default, because that can hide the filter.
    y_filtered -= y_filtered.mean()

    peak = np.max(np.abs(y_filtered)) + EPS
    if normalize_mode == "peak":
        y_filtered = 0.98 * y_filtered / peak
    elif normalize_mode == "rms":
        in_rms = np.sqrt(np.mean(y ** 2)) + EPS
        out_rms = np.sqrt(np.mean(y_filtered ** 2)) + EPS
        y_filtered = y_filtered * (in_rms / out_rms)
        peak = np.max(np.abs(y_filtered)) + EPS
        if peak > 0.98:
            y_filtered = 0.98 * y_filtered / peak
    else:
        # peak_protect: only attenuate if clipping would occur.
        if peak > 0.98:
            y_filtered = 0.98 * y_filtered / peak

    y_filtered = np.clip(y_filtered, -0.98, 0.98)

    out_path = out_dir / f"{p.stem}_SEGMENT_MP_INVERSE_K{n_modes}_seg{seg_size}.wav"
    sf.write(str(out_path), y_filtered.astype(np.float32), sr)

    # Export segment coefficients for audit/debug.
    coeff_path = out_dir / f"{p.stem}_SEGMENT_COEFF_K{n_modes}_seg{seg_size}.csv"
    pd.DataFrame({
        "segment": np.arange(min_segments),
        "mp_reconstructed_value": np.asarray(rec_seg, dtype=np.float64),
        "segment_gain": seg_gain,
    }).to_csv(coeff_path, index=False)

    min_len = min(len(y), len(y_filtered))
    diff_rms = float(np.sqrt(np.mean((y[:min_len] - y_filtered[:min_len]) ** 2)))
    in_rms = float(np.sqrt(np.mean(y[:min_len] ** 2)) + EPS)
    rel_diff_db = 20.0 * np.log10(diff_rms / in_rms + EPS)

    return idx, p.name, str(out_path), rel_diff_db


def song_packet_feature_matrix(path, seg_size, min_segments, feature_mode="entropy", packet_points=PACKET_POINTS):
    """Return intra-segment feature curves for one song.

    Output shape:
        (min_segments, packet_points)

    For vertical packet MP:
        for each segment t, all songs form a packet matrix

            P_t.shape = (songs, packet_points)

        Then MP/PCA is performed independently for every t.

    feature_mode="audio":
        each intra-segment point is log1p(RMS) over a sub-window.

    feature_mode="entropy":
        each intra-segment point is log1p(H/F) over a sub-window.
    """
    y, sr = read_mono(Path(path))
    y = y.astype(np.float64, copy=False)
    y /= np.sqrt(np.mean(y ** 2)) + EPS
    y = y[:int(min_segments) * int(seg_size)]

    if y.size < int(min_segments) * int(seg_size):
        tmp = np.zeros(int(min_segments) * int(seg_size), dtype=np.float64)
        tmp[:y.size] = y
        y = tmp

    seg_size = int(seg_size)
    min_segments = int(min_segments)

    # Keep subwindows reasonably long.
    packet_points = int(max(4, min(int(packet_points), max(4, seg_size // 16))))
    sub_size = int(seg_size // packet_points)
    if sub_size < 8:
        packet_points = max(1, seg_size // 8)
        sub_size = max(8, seg_size // max(1, packet_points))

    usable = packet_points * sub_size
    blocks = y.reshape(min_segments, seg_size)[:, :usable]
    subblocks = blocks.reshape(min_segments * packet_points, sub_size)

    if feature_mode == "audio":
        vals = audio_energy_blocks(subblocks)
    else:
        H = shannon_entropy_blocks(subblocks)
        F = fisher_blocks(subblocks)
        H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
        F = np.nan_to_num(F, nan=EPS, posinf=EPS, neginf=EPS)
        F = np.maximum(np.abs(F), 1e-9)
        ratio = np.clip(np.nan_to_num(H / F, nan=0.0, posinf=1e6, neginf=0.0), 0.0, 1e6)
        vals = np.log1p(ratio)

    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    return vals.reshape(min_segments, packet_points), sr


def _song_packet_worker(args):
    idx, path_str, seg_size, min_segments, feature_mode, packet_points = args
    M, sr = song_packet_feature_matrix(
        Path(path_str),
        int(seg_size),
        int(min_segments),
        feature_mode=feature_mode,
        packet_points=int(packet_points),
    )
    return idx, M, sr, Path(path_str).name


def build_vertical_packet_tensor(files, seg_size, status_callback=None, n_workers=1, feature_mode="entropy", packet_points=PACKET_POINTS):
    """Build tensor for vertical packet MP scan.

    Returns
    -------
    A : ndarray, shape (songs, segments, packet_points)
        A[song, segment, intra_point]
    min_segments : int
    """
    t0 = time.time()
    n_workers = int(n_workers or 1)
    n_workers = max(1, min(n_workers, len(files)))

    frames = [sf.info(str(p)).frames for p in files]
    min_frames = min(frames)
    max_frames = max(frames)
    min_segments = int(min_frames // int(seg_size))

    log("=" * 70)
    log("BUILD VERTICAL PACKET TENSOR")
    log(f"Files         : {len(files)}")
    log(f"Seg size      : {seg_size}")
    log(f"Min segments  : {min_segments}")
    log(f"Packet points : {packet_points}")
    log(f"Feature mode  : {feature_mode}")
    log(f"CPU workers   : {n_workers}")
    log("=" * 70)

    if min_segments < 2:
        raise RuntimeError("Too few segments for vertical packet scan. Reduce Segment size.")

    mats = [None] * len(files)

    if n_workers <= 1:
        for i, p in enumerate(files, 1):
            msg = f"[{i}/{len(files)}] Packet features: {p.name}"
            log(msg)
            if status_callback:
                status_callback(msg)
            M, _ = song_packet_feature_matrix(p, seg_size, min_segments, feature_mode=feature_mode, packet_points=packet_points)
            mats[i - 1] = M
    else:
        tasks = [(i, str(p), int(seg_size), int(min_segments), feature_mode, int(packet_points)) for i, p in enumerate(files)]
        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            futures = [ex.submit(_song_packet_worker, task) for task in tasks]
            done = 0
            for fut in as_completed(futures):
                idx, M, sr, name = fut.result()
                mats[idx] = M
                done += 1
                msg = f"[{done}/{len(files)}] Packet features done: {name}"
                log(msg)
                if status_callback:
                    status_callback(msg)

    A = np.stack(mats, axis=0).astype(np.float64, copy=False)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

    log(f"Packet tensor shape: {A.shape} = songs × segments × packet_points")
    log(f"Packet feature extraction time: {time.time() - t0:.2f} s")
    log("=" * 70)
    return A, min_segments


def vertical_packet_reconstruct_from_tensor(A, n_modes):
    """Run MP/PCA independently for every vertical time packet.

    A.shape = (songs, segments, packet_points)

    For every segment t:
        P_t = A[:, t, :]       # songs × intra-segment points
        C_t = z(P_t) z(P_t)^T / packet_points
        eigen-decompose C_t
        score songs by selected K coherent modes

    Returns a result dict compatible with the GUI:
        Xrec.shape = (songs, segments)
    """
    t0 = time.time()
    A = np.asarray(A, dtype=np.float64)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

    n_songs, n_segments, n_points = A.shape
    k_req = int(max(1, n_modes))
    k = int(min(k_req, n_songs))

    heat = np.zeros((n_songs, n_segments), dtype=np.float64)
    evals_all = []
    evecs_first = None
    lam_plus_all = []
    lam_minus_all = []
    n_signal_all = []

    for t in range(n_segments):
        P = A[:, t, :]  # songs × packet_points

        # z-score every intra-point across songs, because we compare songs
        # within the same vertical packet.
        mu = P.mean(axis=0, keepdims=True)
        sd = P.std(axis=0, keepdims=True)
        sd[(~np.isfinite(sd)) | (sd < EPS)] = 1.0
        Pz = (P - mu) / sd
        Pz = np.nan_to_num(Pz, nan=0.0, posinf=0.0, neginf=0.0)

        C = (Pz @ Pz.T) / max(1, n_points)
        C = 0.5 * (C + C.T)
        C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)

        evals, evecs = eigh(C, check_finite=False)
        evals = evals[::-1]
        evecs = evecs[:, ::-1]

        if evecs_first is None:
            evecs_first = evecs.copy()

        q = n_songs / max(1, n_points)
        positive = evals[evals > 1e-10]
        sigma2 = float(np.median(positive)) if positive.size >= 3 else float(np.median(evals))
        if not np.isfinite(sigma2) or sigma2 <= EPS:
            sigma2 = float(np.mean(positive)) if positive.size else 1.0
        if not np.isfinite(sigma2) or sigma2 <= EPS:
            sigma2 = 1.0

        lam_minus = sigma2 * (1.0 - np.sqrt(q)) ** 2
        lam_plus = sigma2 * (1.0 + np.sqrt(q)) ** 2
        n_signal = int(np.sum(evals > lam_plus))

        # Use only strictly positive modes. In packet scans C can have
        # numerical zero modes; including them makes colors unstable when K changes.
        valid_modes = int(np.sum(evals > 1e-10))
        kk = int(min(k, evecs.shape[1], max(1, valid_modes)))
        weights = evals[:kk] / (np.sum(evals[:kk]) + EPS)
        score = np.sum((evecs[:, :kk] ** 2) * weights[None, :], axis=1)

        # Column participation: each vertical packet sums to 100%.
        score = np.maximum(score, 0.0)
        heat[:, t] = 100.0 * score / (np.sum(score) + EPS)

        evals_all.append(evals)
        lam_minus_all.append(lam_minus)
        lam_plus_all.append(lam_plus)
        n_signal_all.append(n_signal)

    evals_mean = np.mean(np.asarray(evals_all), axis=0)
    scores_mean = np.mean(heat, axis=1)

    log("=" * 70)
    log("VERTICAL PACKET MP SCAN")
    log(f"Songs          : {n_songs}")
    log(f"Segments       : {n_segments}")
    log(f"Packet points  : {n_points}")
    log(f"Render modes K : {k}")
    log(f"Mean λ+        : {np.mean(lam_plus_all):.6g}")
    log(f"Mean signal modes per packet: {np.mean(n_signal_all):.3f}")
    log(f"Scan time      : {time.time() - t0:.2f} s")
    log("=" * 70)

    return {
        "Xz": heat,
        "Xrec": heat,
        "Xrec_z": heat,
        "Xrender": heat,
        "evals": evals_mean,
        "evecs": evecs_first if evecs_first is not None else np.eye(n_songs),
        "lam_minus": float(np.mean(lam_minus_all)) if lam_minus_all else 0.0,
        "lam_plus": float(np.mean(lam_plus_all)) if lam_plus_all else 0.0,
        "n_signal": int(round(float(np.mean(n_signal_all)))) if n_signal_all else 0,
        "scores": scores_mean,
        "n_modes_used": k,
        "packet_tensor": A,
        "packet_points": n_points,
        "vertical_packet": True,
    }



def build_matrix(files, seg_size, status_callback=None, n_workers=1, feature_mode="entropy", matrix_geometry="song_time"):
    t0 = time.time()

    n_workers = int(n_workers or 1)
    n_workers = max(1, min(n_workers, len(files)))

    log("=" * 70)
    log("BUILD MATRIX")
    log(f"Files       : {len(files)}")
    log(f"Seg size    : {seg_size}")
    log(f"CPU workers : {n_workers}")
    log("=" * 70)

    frames = [sf.info(str(p)).frames for p in files]

    min_frames = min(frames)
    max_frames = max(frames)
    min_segments = min_frames // seg_size

    log(f"Shortest frames : {min_frames}")
    log(f"Longest frames  : {max_frames}")
    log(f"Min segments    : {min_segments}")
    log(f"Songs           : {len(files)}")

    if min_segments <= len(files):
        raise RuntimeError(
            f"MP boundary error:\n"
            f"segments={min_segments}, songs={len(files)}\n"
            f"Need segments > songs.\n"
            f"Reduce segment size.\n"
            f"Current segment size = {seg_size}"
        )

    if min_segments < 64:
        raise RuntimeError(
            f"Too few segments for stable MP:\n"
            f"segments={min_segments}\n"
            f"Use smaller segment size: 1024, 2048, 4096."
        )

    # Practical RMT stability warning: MP formally needs T > N, but if T is
    # close to N the estimate is coarse and the archetype may become unstable.
    if min_segments < 3 * len(files):
        log(
            f"WARNING: only {min_segments} segments for {len(files)} songs "
            f"(T/N={min_segments/len(files):.2f}). "
            "MP is valid but coarse; consider smaller Segment size for stability."
        )

    X = [None] * len(files)

    if n_workers <= 1:
        for i, p in enumerate(files, 1):
            msg = f"[{i}/{len(files)}] Feature extraction: {p.name}"
            log(msg)
            log_memory("before song")

            if status_callback:
                status_callback(msg)

            v, _ = song_sf_vector(p, seg_size, min_segments, feature_mode=feature_mode)
            X[i - 1] = v

            log_memory("after song")
            gc.collect()
    else:
        tasks = [(i, str(p), seg_size, min_segments, feature_mode) for i, p in enumerate(files)]

        # fork is fastest on Linux, but spawn is safer if fork is unavailable.
        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = mp.get_context("spawn")

        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            futures = [ex.submit(_song_sf_worker, task) for task in tasks]

            done = 0
            for fut in as_completed(futures):
                idx, v, sr, name = fut.result()
                X[idx] = v
                done += 1

                msg = f"[{done}/{len(files)}] Feature extraction done: {name}"
                log(msg)
                if status_callback:
                    status_callback(msg)

    X = np.vstack(X).astype(np.float64, copy=False)
    X = sanitize_matrix(X, label="feature matrix X")

    # Matrix geometry:
    #   song_time:
    #       X.shape = (songs, segments)
    #       MP compares whole temporal trajectories between songs.
    #
    #   vertical_packet:
    #       X.shape = (segments, songs)
    #       MP compares vertical time slices. Each row is one temporal segment
    #       represented by the vector of all songs at that same segment index.
    #       This is the user's originally imagined "vertical grouping" mode.
    if matrix_geometry == "vertical_packet":
        X = X.T.copy()
        X = sanitize_matrix(X, label="vertical segment matrix X.T")

    log(f"Matrix shape: {X.shape}")
    log(f"Feature extraction time: {time.time() - t0:.2f} s")
    log_memory("after matrix")
    log("=" * 70)

    return X, min_segments


# ============================================================
# MP RECONSTRUCTION
# ============================================================

def mp_reconstruct(X, n_modes, use_gpu=False):
    t0 = time.time()

    # CPU-only path: no CuPy/NVRTC probing, no CUDA fallback messages.
    log("=" * 70)
    log("MP RECONSTRUCTION — CPU ONLY")
    log(f"Input X shape : {X.shape}")
    log_memory("before MP")

    X = sanitize_matrix(X, label="MP input X")

    mean_cpu = X.mean(axis=0, keepdims=True)
    std_cpu = X.std(axis=0, keepdims=True)

    # Columns with zero or invalid variance cannot be z-normalized normally.
    # Keep them finite by using std=1.0; after centering they become near-zero columns.
    bad_std = (~np.isfinite(std_cpu)) | (std_cpu < EPS)
    if np.any(bad_std):
        log(f"WARNING: MP input: {int(bad_std.sum())} constant/invalid segment columns; using std=1.0")
        std_cpu = std_cpu.copy()
        std_cpu[bad_std] = 1.0

    Xz_cpu = (X - mean_cpu) / std_cpu
    Xz_cpu = np.nan_to_num(Xz_cpu, nan=0.0, posinf=0.0, neginf=0.0)

    C = (Xz_cpu @ Xz_cpu.T) / Xz_cpu.shape[1]
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    C = 0.5 * (C + C.T)
    evals, evecs = eigh(C, check_finite=False)

    evals = evals[::-1]
    evecs = evecs[:, ::-1]

    N, T = X.shape

    # ============================================================
    # MP boundary handling
    # ============================================================
    # Original song×time mode usually has T > N:
    #     N = songs, T = temporal segments
    #
    # New vertical packet mode has the opposite shape:
    #     N = segments, T = songs
    #
    # Then C is rank-deficient because there are fewer observations than rows.
    # This is still usable as a low-rank PCA/MP decomposition, but many
    # eigenvalues are exactly/near zero.  Therefore sigma2 must be estimated
    # from the nonzero eigenvalue bulk, not from the full median.
    # ============================================================

    q = N / max(1, T)

    positive_evals = evals[evals > 1e-10]
    if positive_evals.size >= 3:
        sigma2 = float(np.median(positive_evals))
    else:
        sigma2 = float(np.median(evals))

    if not np.isfinite(sigma2) or sigma2 <= EPS:
        sigma2 = float(np.mean(positive_evals)) if positive_evals.size else 1.0
    if not np.isfinite(sigma2) or sigma2 <= EPS:
        sigma2 = 1.0

    lam_minus = sigma2 * (1.0 - np.sqrt(q)) ** 2
    lam_plus = sigma2 * (1.0 + np.sqrt(q)) ** 2

    n_signal = int(np.sum(evals > lam_plus))
    k = min(max(1, n_modes), evecs.shape[1])

    V = evecs[:, :k]

    # Low-rank MP inverse reconstruction in z-domain.
    # This is the part controlled directly by the selected eigenvectors/modes.
    Xrec_z = V @ (V.T @ Xz_cpu)

    # Reconstruction mapped back to the original feature scale, used for plots/tables.
    Xrec = Xrec_z * std_cpu + mean_cpu

    # IMPORTANT FOR AUDIO RENDER:
    # Do not render from Xrec directly, because the column mean can dominate and
    # visually/audibly hide the effect of changing K.  Render from the centered
    # low-rank MP component instead.  This makes K=1, K=2, ... K=n produce
    # genuinely different gain fields.
    Xrender = Xrec_z.astype(np.float64, copy=False)

    # ranking по MP signal modes, не по render k
    score_modes = max(1, n_signal)
    weights = evals[:score_modes] / (evals[:score_modes].sum() + EPS)
    scores = np.sum((evecs[:, :score_modes] ** 2) * weights[None, :], axis=1)

    log(f"N rows         : {N}")
    log(f"T columns      : {T}")
    log(f"q=N/T          : {q:.8f}")
    if T <= N:
        log("WARNING: rank-deficient MP/PCA mode: T <= N. Using nonzero eigenvalue bulk estimate.")
    log(f"MP lambda minus: {lam_minus:.8f}")
    log(f"MP lambda plus : {lam_plus:.8f}")
    log(f"Signal modes   : {n_signal}")
    log(f"Render modes   : {k}")
    log(f"MP time        : {time.time() - t0:.3f} s")
    log_memory("after MP")
    log("=" * 70)

    return {
        "Xz": Xz_cpu,
        "Xrec": Xrec,
        "Xrec_z": Xrec_z,
        "Xrender": Xrender,
        "evals": evals,
        "evecs": evecs,
        "lam_minus": lam_minus,
        "lam_plus": lam_plus,
        "n_signal": n_signal,
        "scores": scores,
        "n_modes_used": k,
    }


# ============================================================
# LOCAL SEGMENT EIGENVALUE ANALYSIS
# ============================================================

def local_segment_eigs(Xz, center_segment, window_size=48):
    """
    Compute a moving local MP/PCA eigen-spectrum around a selected vertical segment.

    Important: one exact vertical segment X[:, t] is only one column across songs,
    therefore its covariance is rank-1 and the spectrum is almost always the same.
    To make the eigen-plot meaningful and visibly segment-dependent, we use a
    narrow moving vertical slab X[:, left:right] around the clicked segment.

    Use a small local window, typically N_songs+2 .. 64.  If the window is almost
    the whole song, the graph will look global and will barely move.
    """
    Xz = np.asarray(Xz, dtype=np.float64)
    N, T = Xz.shape

    if T <= N:
        raise RuntimeError(f"Need T > N for local MP spectrum. Got T={T}, N={N}.")

    c = int(np.clip(center_segment, 0, T - 1))

    # Minimum for a valid MP covariance: T_local > N.
    min_w = min(T, N + 2)

    # Keep the local window narrow enough to move when clicking different segments.
    # Large windows close to T create nearly identical spectra.
    requested = int(window_size)
    default_w = min(T, max(min_w, min(64, T)))
    if requested <= 0:
        w = default_w
    else:
        w = int(np.clip(requested, min_w, T))

    # If user accidentally selected almost the whole song, reduce automatically
    # to a local slab so the plot changes with the clicked vertical segment.
    if T > min_w and w > 0.75 * T:
        w = default_w

    # Moving centered window. Near borders it clamps, so the first few clicks may
    # share the same window; this is mathematically expected.
    left = c - w // 2
    left = int(np.clip(left, 0, max(0, T - w)))
    right = int(left + w)

    Xw = Xz[:, left:right]
    Tw = Xw.shape[1]

    # Local re-centering per song over the selected vertical slab.
    Xw = Xw - Xw.mean(axis=1, keepdims=True)
    row_std = Xw.std(axis=1, keepdims=True)
    row_std[(~np.isfinite(row_std)) | (row_std < EPS)] = 1.0
    Xw = Xw / row_std
    Xw = np.nan_to_num(Xw, nan=0.0, posinf=0.0, neginf=0.0)

    C = (Xw @ Xw.T) / max(1, Tw)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    C = 0.5 * (C + C.T)

    evals, evecs = eigh(C, check_finite=False)
    evals = evals[::-1]
    evecs = evecs[:, ::-1]

    q = N / Tw
    sigma2 = np.median(evals)
    lam_minus = sigma2 * (1.0 - np.sqrt(q)) ** 2
    lam_plus = sigma2 * (1.0 + np.sqrt(q)) ** 2
    n_signal = int(np.sum(evals > lam_plus))

    return {
        "evals": evals,
        "evecs": evecs,
        "lam_minus": lam_minus,
        "lam_plus": lam_plus,
        "n_signal": n_signal,
        "center": c,
        "left": left,
        "right": right,
        "window": Tw,
        "q": q,
    }



def vertical_segment_eigs(Xz, center_segment, global_lam_minus=None, global_lam_plus=None):
    """Eigen-spectrum of the exact clicked vertical segment across all songs.

    The clicked heatmap column is the vector v = X[:, segment], one value per song.
    We form its vertical outer-product covariance C = v v^T / N and compute the
    eigenvalues.  This is intentionally a rank-1 spectrum: it answers exactly
    what was requested -- the covariance/eigenvalue structure of all segments
    that lie vertically one under another at the clicked time coordinate.

    Marchenko-Pastur separation line: a single column cannot estimate its own
    MP bulk reliably, so we draw the global MP lambda +/- from the current full
    analysis as the separation reference.
    """
    Xz = np.asarray(Xz, dtype=np.float64)
    N, T = Xz.shape
    c = int(np.clip(center_segment, 0, T - 1))

    v = Xz[:, c].astype(np.float64, copy=True)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    v = v - np.mean(v)

    s = float(np.std(v))
    if not np.isfinite(s) or s < EPS:
        s = 1.0
    v = v / s

    C = np.outer(v, v) / max(1, N)
    C = 0.5 * (C + C.T)
    evals, evecs = eigh(C, check_finite=False)
    evals = evals[::-1]
    evecs = evecs[:, ::-1]

    lam_minus = float(global_lam_minus) if global_lam_minus is not None else 0.0
    lam_plus = float(global_lam_plus) if global_lam_plus is not None else 0.0
    n_signal = int(np.sum(evals > lam_plus)) if lam_plus > 0 else int(np.sum(evals > EPS))

    return {
        "evals": evals,
        "evecs": evecs,
        "lam_minus": lam_minus,
        "lam_plus": lam_plus,
        "n_signal": n_signal,
        "center": c,
        "vector": v,
        "N": N,
        "rank_note": "single-column vertical covariance is rank-1 by construction",
    }


# ============================================================
# HEATMAP
# ============================================================

def prepare_heatmap(X):
    X_abs = np.log1p(np.abs(X))

    vmax = np.percentile(X_abs, 99.0)

    if vmax <= 0:
        vmax = X_abs.max() + EPS

    X_abs = np.clip(X_abs, 0, vmax)

    return 100.0 * X_abs / (X_abs.sum(axis=1, keepdims=True) + EPS)


# ============================================================
# RENDER
# ============================================================

def render_outputs(
    files,
    Xrec,
    min_segments,
    seg_size,
    out_dir,
    n_modes,
    dry,
    wet,
    status_callback=None,
    n_workers=1
):
    t0 = time.time()

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    n_workers = int(n_workers or 1)
    n_workers = max(1, min(n_workers, len(files)))

    log("=" * 70)
    log("RENDER OUTPUTS")
    log(f"Output dir  : {out_dir}")
    log(f"Render K    : {n_modes}")
    log(f"Seg size    : {seg_size}")
    log(f"Dry/Wet     : {dry:.2f} / {wet:.2f}")
    log(f"CPU workers : {n_workers}")
    log(f"Xrec shape  : {None if Xrec is None else Xrec.shape}")
    log("Render source: DIRECT segment-by-segment MP inverse reconstruction from selected K modes")
    log("=" * 70)

    if Xrec is None:
        raise RuntimeError("Missing Xrec. Run MP reconstruction before rendering.")
    if Xrec.shape[0] != len(files):
        raise RuntimeError(f"row mismatch: Xrec rows={Xrec.shape[0]}, files={len(files)}")

    Xrec_arr = np.asarray(Xrec, dtype=np.float64)
    rec_lo = float(np.percentile(Xrec_arr, 2.0))
    rec_hi = float(np.percentile(Xrec_arr, 98.0))
    log(f"Segment coefficient scale from Xrec_K: p02={rec_lo:.6g}, p98={rec_hi:.6g}")

    # Save full coefficient field for all files/segments.
    coeff_matrix_path = out_dir / f"ALL_FILES_MP_RECONSTRUCTED_SEGMENT_VALUES_K{n_modes}_seg{seg_size}.csv"
    pd.DataFrame(Xrec_arr).to_csv(coeff_matrix_path, index=False)
    log(f"Saved all reconstructed segment coefficients: {coeff_matrix_path}")

    normalize_mode = "peak_protect"

    if n_workers <= 1:
        for i, p in enumerate(files, 1):
            msg = f"[{i}/{len(files)}] Rendering: {p.name}"
            log(msg)
            log_memory("before render song")

            if status_callback:
                status_callback(msg)

            _render_song_worker((
                i - 1,
                str(p),
                Xrec[i - 1],
                min_segments,
                seg_size,
                str(out_dir),
                n_modes,
                dry,
                wet,
                rec_lo,
                rec_hi,
                normalize_mode,
            ))

            gc.collect()
            log_memory("after render song")
    else:
        tasks = [
            (
                i,
                str(p),
                Xrec[i],
                min_segments,
                seg_size,
                str(out_dir),
                n_modes,
                dry,
                wet,
                rec_lo,
                rec_hi,
                normalize_mode,
            )
            for i, p in enumerate(files)
        ]

        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = mp.get_context("spawn")

        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            futures = [ex.submit(_render_song_worker, task) for task in tasks]
            done = 0
            for fut in as_completed(futures):
                idx, name, out_path, rel_diff_db = fut.result()
                done += 1
                msg = f"[{done}/{len(files)}] Rendered: {name} | diff={rel_diff_db:.1f} dB"
                log(msg)
                if status_callback:
                    status_callback(msg)

    log(f"RENDER DONE in {time.time() - t0:.2f} s")
    return out_dir



# ============================================================
# SEGMENT BEAT / ENERGY CHART
# ============================================================

def segment_beat_features(path, seg_index, seg_size):
    """Return waveform and simple beat/energy envelope for one audio segment."""
    y, sr = read_mono(Path(path))
    start = int(seg_index) * int(seg_size)
    end = min(start + int(seg_size), len(y))

    if start >= len(y):
        raise RuntimeError(f"Selected segment {seg_index} starts outside audio length.")

    seg = y[start:end].astype(np.float64, copy=False)
    if len(seg) < 8:
        raise RuntimeError("Selected segment is too short for beat chart.")

    seg = seg - np.mean(seg)
    seg_peak = np.max(np.abs(seg)) + EPS
    seg_norm = seg / seg_peak

    # Short-time RMS envelope inside this exact segment.
    frame = max(64, min(1024, int(seg_size) // 16))
    hop = max(16, frame // 4)

    if len(seg_norm) < frame:
        frame = max(8, len(seg_norm) // 2)
        hop = max(4, frame // 4)

    starts = np.arange(0, max(1, len(seg_norm) - frame + 1), hop, dtype=int)
    if len(starts) == 0:
        starts = np.array([0], dtype=int)

    rms = []
    centers = []
    for st in starts:
        chunk = seg_norm[st:st + frame]
        if len(chunk) == 0:
            continue
        rms.append(np.sqrt(np.mean(chunk ** 2)))
        centers.append(st + len(chunk) / 2.0)

    centers = np.asarray(centers, dtype=np.float64)
    rms = np.asarray(rms, dtype=np.float64)

    if len(rms) > 0:
        rms = rms / (np.max(rms) + EPS)
        if len(rms) >= 3:
            distance = max(1, len(rms) // 16)
            prominence = max(0.05, 0.20 * float(np.std(rms)))
            peaks, _ = find_peaks(rms, distance=distance, prominence=prominence)
        else:
            peaks = np.array([], dtype=int)
    else:
        peaks = np.array([], dtype=int)

    t = np.arange(len(seg_norm), dtype=np.float64) / float(sr)
    env_t = centers / float(sr)
    peak_t = env_t[peaks] if len(peaks) else np.array([], dtype=np.float64)
    peak_y = rms[peaks] if len(peaks) else np.array([], dtype=np.float64)

    return {
        "segment": seg_norm,
        "t": t,
        "env_t": env_t,
        "rms": rms,
        "peak_t": peak_t,
        "peak_y": peak_y,
        "sr": sr,
        "start_sample": start,
        "end_sample": end,
        "duration": len(seg_norm) / float(sr),
    }



# ============================================================
# ARCHETYPE COLLAGE RENDER
# ============================================================

def render_archetype_collage(
    files,
    Xfield,
    min_segments,
    seg_size,
    out_dir,
    n_modes,
    status_callback=None,
    crossfade_ms=30.0,
    use_abs=False,
):
    """Render a new audio track from the current MP/K archetype field.

    For every segment t:
        winner[t] = argmax_song(score[song, t])
        output gets the real audio segment from winner[t].

    Fixes compared to the previous export:
    - uses positive reconstructed feature participation by default, not abs(Xrender);
    - uses a real overlap crossfade when the winning source changes;
    - avoids per-segment RMS leveling so the source dynamics are preserved;
    - exports winner CSV + summary CSV for auditing dominance.
    """
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    if Xfield is None:
        raise RuntimeError("Missing MP field. Run Analyze MP first.")

    Xfield = np.asarray(Xfield, dtype=np.float64)
    Xfield = np.nan_to_num(Xfield, nan=0.0, posinf=0.0, neginf=0.0)

    if Xfield.shape[0] != len(files):
        raise RuntimeError(f"Xfield rows={Xfield.shape[0]} but files={len(files)}")

    T = int(min(min_segments, Xfield.shape[1]))
    if T <= 1:
        raise RuntimeError("Too few segments for archetype collage.")

    field = Xfield[:, :T]
    if use_abs:
        score = np.abs(field)
        score_mode = "abs(field)"
    else:
        score = field.copy()
        score_mode = "argmax(field)"

    score = np.nan_to_num(score, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    winners = np.argmax(score, axis=0).astype(int)
    winner_scores = score[winners, np.arange(T)]

    log("=" * 70)
    log("RENDER ARCHETYPE COLLAGE")
    log(f"Segments     : {T}")
    log(f"Segment size : {seg_size}")
    log(f"MP modes     : {n_modes}")
    log(f"Crossfade ms : {crossfade_ms}")
    log(f"Score mode   : {score_mode}")
    log("Selection    : argmax over current MP/K archetype field")
    log("=" * 70)

    if status_callback:
        status_callback("Loading audio files for archetype collage...")

    audios = []
    srs = []
    for i, p in enumerate(files, 1):
        y, sr = read_mono(p)
        audios.append(y.astype(np.float32, copy=False))
        srs.append(sr)
        if status_callback:
            status_callback(f"Loaded [{i}/{len(files)}]: {p.name}")

    sr0 = int(srs[0])
    if any(int(sr) != sr0 for sr in srs):
        raise RuntimeError("All files must have the same sample rate for collage rendering.")

    cross = int(round(sr0 * float(crossfade_ms) / 1000.0))
    cross = int(max(0, min(cross, seg_size // 3)))
    fade_in = np.linspace(0.0, 1.0, cross, endpoint=False, dtype=np.float32) if cross else None
    fade_out = 1.0 - fade_in if cross else None

    pieces = []
    last_winner = -1
    switches = 0

    for t in range(T):
        wi = int(winners[t])
        y = audios[wi]
        a = t * seg_size
        b = a + seg_size
        seg = y[a:b]
        if len(seg) < seg_size:
            tmp = np.zeros(seg_size, dtype=np.float32)
            tmp[:len(seg)] = seg
            seg = tmp
        else:
            seg = seg.astype(np.float32, copy=True)

        # Only protect extreme clipping; do not loudness-normalize every segment.
        pk = float(np.max(np.abs(seg)) + EPS)
        if pk > 0.98:
            seg = 0.98 * seg / pk

        if not pieces:
            pieces.append(seg)
        else:
            if cross > 0 and wi != last_winner:
                switches += 1
                prev = pieces[-1]
                if len(prev) >= cross and len(seg) >= cross:
                    # True overlap crossfade: previous tail + new head.
                    mixed = prev[-cross:] * fade_out + seg[:cross] * fade_in
                    pieces[-1] = np.concatenate([prev[:-cross], mixed]).astype(np.float32, copy=False)
                    pieces.append(seg[cross:])
                else:
                    pieces.append(seg)
            else:
                # Same winner in consecutive segments: keep the natural continuity.
                pieces.append(seg)

        last_winner = wi

        if status_callback and (t % max(1, T // 50) == 0):
            status_callback(f"Collage segment {t+1}/{T}: {files[wi].name}")

    out = np.concatenate(pieces).astype(np.float32, copy=False)
    out -= float(np.mean(out))
    peak = float(np.max(np.abs(out)) + EPS)
    if peak > 0.98:
        out = 0.98 * out / peak

    out_path = out_dir / f"ARCHETYPE_COLLAGE_K{n_modes}_seg{seg_size}_xfade{int(crossfade_ms)}ms.wav"
    sf.write(str(out_path), out.astype(np.float32), sr0)

    csv_path = out_dir / f"ARCHETYPE_COLLAGE_K{n_modes}_seg{seg_size}_winners.csv"
    pd.DataFrame({
        "segment": np.arange(T),
        "winner_index": winners,
        "winner_song": [Path(files[i]).name for i in winners],
        "winner_score": winner_scores,
        "source_start_sample": np.arange(T) * int(seg_size),
        "source_end_sample": (np.arange(T) + 1) * int(seg_size),
    }).to_csv(csv_path, index=False)

    counts = np.bincount(winners, minlength=len(files))
    summary_path = out_dir / f"ARCHETYPE_COLLAGE_K{n_modes}_seg{seg_size}_summary.csv"
    pd.DataFrame({
        "song_index": np.arange(len(files)),
        "song": [Path(p).name for p in files],
        "segments_used": counts,
        "percent": 100.0 * counts / max(1, T),
        "mean_score_when_used": [
            float(np.mean(winner_scores[winners == i])) if np.any(winners == i) else 0.0
            for i in range(len(files))
        ],
    }).sort_values("segments_used", ascending=False).to_csv(summary_path, index=False)

    log(f"Collage switches : {switches}")
    log(f"Output duration  : {len(out) / sr0:.2f} s")
    log(f"Saved collage    : {out_path}")
    log(f"Saved winners    : {csv_path}")
    log(f"Saved summary    : {summary_path}")
    log(f"Collage time     : {time.time() - t0:.2f} s")
    log("=" * 70)

    return out_path, csv_path, summary_path

# ============================================================
# GUI
# ============================================================

class MPOfflineRenderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Offline SEGMENT MP Inverse Render — CPU Multi-Core")
        self.root.geometry("1700x950")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.folder = None
        self.files = []
        self.names = []
        self.display_names = []

        self.X = None
        self.Xrec = None
        self.H_matrix = None
        self.F_matrix = None
        self.min_segments = None
        self.result = None

        self.df_rank = None
        self.df_eig = None
        self.cbar = None
        self.heat_marker = None
        self.heat_window_patch = None
        self.heat_row_marker = None
        self.current_rank_indices = None
        self.selected_file_index = None
        self.selected_rank_row = None
        self.current_ranked_heat = None
        self.current_ranked_names = None
        self.current_plot_seg = None
        self.local_eig_window = 256

        self.last_analyzed_seg = None
        self.last_reconstructed_modes = None
        self.feature_mode = "entropy"
        self.matrix_geometry = "song_time"
        self.row_order_mode = "mp_rank"
        self.entropy_diag_mode = "sf_space"
        self.current_ranked_heat = None
        self.current_ranked_names = None
        self.current_plot_seg = None

        self._mode_update_after_id = None

        top = tk.Frame(root)
        top.pack(side=tk.TOP, fill=tk.X)

        # ============================================================
        # THREE-ROW TOOLBAR LAYOUT
        # ============================================================
        # Row 1: file/actions
        # Row 2: compute/segmentation controls
        # Row 3: render + MP geometry controls
        # This prevents the toolbar from going off-screen on narrower displays.
        top_row1 = tk.Frame(top)
        top_row1.pack(side=tk.TOP, fill=tk.X)

        top_row2 = tk.Frame(top)
        top_row2.pack(side=tk.TOP, fill=tk.X)

        top_row3 = tk.Frame(top)
        top_row3.pack(side=tk.TOP, fill=tk.X)

        tk.Button(top_row1, text="Select folder", command=self.select_folder).pack(side=tk.LEFT, padx=5, pady=3)
        tk.Button(top_row1, text="Analyze ENTROPY MP", command=self.analyze_entropy).pack(side=tk.LEFT, padx=5, pady=3)
        tk.Button(top_row1, text="Analyze AUDIO MP", command=self.analyze_audio).pack(side=tk.LEFT, padx=5, pady=3)
        tk.Button(top_row1, text="Render filtered files", command=self.render).pack(side=tk.LEFT, padx=5, pady=3)
        tk.Button(top_row1, text="Render archetype collage", command=self.render_archetype_collage).pack(side=tk.LEFT, padx=5, pady=3)
        tk.Button(top_row1, text="Export K1-16 PDF", command=self.export_entropy_maps_pdf).pack(side=tk.LEFT, padx=5, pady=3)
        tk.Button(top_row1, text="Save CSV", command=self.save_csv).pack(side=tk.LEFT, padx=5, pady=3)

        self.use_gpu = tk.BooleanVar(value=False)
        tk.Label(top_row2, text="CPU-only mode").pack(side=tk.LEFT, padx=10)

        tk.Label(top_row2, text="CPU workers").pack(side=tk.LEFT, padx=5)
        self.cpu_workers = tk.IntVar(value=DEFAULT_CPU_WORKERS)
        tk.Spinbox(
            top_row2,
            from_=1,
            to=max(1, os.cpu_count() or 1),
            width=4,
            textvariable=self.cpu_workers
        ).pack(side=tk.LEFT, padx=5)

        tk.Label(top_row2, text="Segment").pack(side=tk.LEFT, padx=5)
        self.seg_size = tk.IntVar(value=1024)

        tk.Scale(
            top_row2,
            from_=256,
            to=65536,
            resolution=256,
            orient=tk.HORIZONTAL,
            length=220,
            variable=self.seg_size
        ).pack(side=tk.LEFT, padx=5)

        tk.Label(top_row2, text="MP modes").pack(side=tk.LEFT, padx=5)
        self.n_modes = tk.IntVar(value=6)

        tk.Scale(
            top_row2,
            from_=1,
            to=35,
            resolution=1,
            orient=tk.HORIZONTAL,
            length=160,
            variable=self.n_modes
        ).pack(side=tk.LEFT, padx=5)

        tk.Label(top_row3, text="Dry %").pack(side=tk.LEFT, padx=5)
        self.dry_percent = tk.IntVar(value=0)

        tk.Scale(
            top_row3,
            from_=0,
            to=100,
            resolution=5,
            orient=tk.HORIZONTAL,
            length=120,
            variable=self.dry_percent
        ).pack(side=tk.LEFT, padx=5)

        tk.Label(top_row3, text="Wet %").pack(side=tk.LEFT, padx=5)
        self.wet_percent = tk.IntVar(value=100)

        tk.Scale(
            top_row3,
            from_=0,
            to=100,
            resolution=5,
            orient=tk.HORIZONTAL,
            length=120,
            variable=self.wet_percent
        ).pack(side=tk.LEFT, padx=5)

        # Matrix geometry radio buttons.
        # "song × time" is the original implementation.
        # "vertical packet" is the vertical grouping mode:
        # each temporal segment becomes a row/vector across all songs.
        tk.Label(top_row3, text="MP geometry").pack(side=tk.LEFT, padx=(12, 4))
        self.geometry_var = tk.StringVar(value="song_time")
        tk.Radiobutton(
            top_row3,
            text="song×time",
            variable=self.geometry_var,
            value="song_time",
            command=self.on_geometry_changed
        ).pack(side=tk.LEFT, padx=2)
        tk.Radiobutton(
            top_row3,
            text="vertical packet",
            variable=self.geometry_var,
            value="vertical_packet",
            command=self.on_geometry_changed
        ).pack(side=tk.LEFT, padx=2)

        tk.Label(top_row3, text="Row order").pack(side=tk.LEFT, padx=(14, 4))
        self.row_order_var = tk.StringVar(value="mp_rank")
        tk.Radiobutton(
            top_row3,
            text="file order",
            variable=self.row_order_var,
            value="file_order",
            command=self.on_row_order_changed
        ).pack(side=tk.LEFT, padx=2)
        tk.Radiobutton(
            top_row3,
            text="MP rank",
            variable=self.row_order_var,
            value="mp_rank",
            command=self.on_row_order_changed
        ).pack(side=tk.LEFT, padx=2)

        tk.Label(top_row3, text="Entropy view").pack(side=tk.LEFT, padx=(14, 4))
        self.entropy_diag_var = tk.StringVar(value="sf_space")
        tk.Radiobutton(
            top_row3,
            text="H-F space",
            variable=self.entropy_diag_var,
            value="sf_space",
            command=self.on_entropy_diag_changed
        ).pack(side=tk.LEFT, padx=2)
        tk.Radiobutton(
            top_row3,
            text="Fisher lag",
            variable=self.entropy_diag_var,
            value="fisher_lag",
            command=self.on_entropy_diag_changed
        ).pack(side=tk.LEFT, padx=2)
        tk.Radiobutton(
            top_row3,
            text="COM ±",
            variable=self.entropy_diag_var,
            value="com_pm",
            command=self.on_entropy_diag_changed
        ).pack(side=tk.LEFT, padx=2)

        self.seg_size.trace_add("write", self.mark_segment_changed)
        self.n_modes.trace_add("write", self.schedule_mode_update)
        self.dry_percent.trace_add("write", self.mark_render_changed)
        self.wet_percent.trace_add("write", self.mark_render_changed)

        self.info = tk.Label(root, text="No folder selected", anchor="w")
        self.info.pack(side=tk.TOP, fill=tk.X, padx=10)

        # Segment selector: this sits above the plots and spans almost the full graph width.
        # It selects a segment/time position, updates the local eigenvalue spectrum,
        # and draws a vertical red marker on the heatmap.
        segpick = tk.Frame(root)
        segpick.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(2, 2))

        tk.Label(segpick, text="Selected segment").pack(side=tk.LEFT, padx=(0, 6))
        self.selected_segment = tk.IntVar(value=0)
        self.segment_pick_scale = tk.Scale(
            segpick,
            from_=0,
            to=0,
            resolution=1,
            orient=tk.HORIZONTAL,
            length=1200,
            variable=self.selected_segment,
            showvalue=True,
            command=self.on_segment_slider
        )
        self.segment_pick_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        tk.Label(segpick, text="local eig window").pack(side=tk.LEFT, padx=(8, 4))
        self.local_window_var = tk.IntVar(value=48)
        self.local_window_spin = tk.Spinbox(
            segpick,
            from_=8,
            to=512,
            increment=8,
            width=6,
            textvariable=self.local_window_var,
            command=self.on_segment_slider
        )
        self.local_window_spin.pack(side=tk.LEFT, padx=5)

        tk.Label(segpick, text="view chunks").pack(side=tk.LEFT, padx=(8, 4))
        self.view_chunks_var = tk.IntVar(value=1)
        self.view_chunks_spin = tk.Spinbox(
            segpick,
            from_=1,
            to=64,
            increment=1,
            width=5,
            textvariable=self.view_chunks_var,
            command=self.refresh_selected_audio_view
        )
        self.view_chunks_spin.pack(side=tk.LEFT, padx=5)
        self.view_chunks_var.trace_add("write", lambda *args: self.refresh_selected_audio_view())

        tk.Label(segpick, text="phase segs").pack(side=tk.LEFT, padx=(8, 4))
        self.phase_segments_var = tk.IntVar(value=64)
        self.phase_segments_spin = tk.Spinbox(
            segpick,
            from_=8,
            to=512,
            increment=8,
            width=5,
            textvariable=self.phase_segments_var,
            command=self.refresh_selected_audio_view
        )
        self.phase_segments_spin.pack(side=tk.LEFT, padx=5)
        self.phase_segments_var.trace_add("write", lambda *args: self.refresh_selected_audio_view())

        self.fig = Figure(figsize=(16, 9), dpi=100)

        gs = self.fig.add_gridspec(
            3, 2,
            # A little more vertical space for the lower signal/phase plots.
            height_ratios=[1.0, 1.55, 1.05],
            # The lower layout is additionally adjusted by _adjust_bottom_axes_for_legends().
            width_ratios=[1.70, 0.95],
            hspace=0.58,
            wspace=0.48
        )
        self.ax_eig = self.fig.add_subplot(gs[0, 0])
        self.ax_density = self.fig.add_subplot(gs[0, 1])
        self.ax_heat = self.fig.add_subplot(gs[1, :])
        # Dedicated left label axis for the waterfall rows.
        # This avoids Matplotlib tick-label accumulation/overlap in the heatmap axis.
        self.ax_heat_labels = self.fig.add_axes([0.010, 0.390, 0.175, 0.265])
        self.ax_heat_labels.axis("off")
        self.ax_beat = self.fig.add_subplot(gs[2, 0])
        self.ax_phase = self.fig.add_subplot(gs[2, 1])
        self.ax_beat.set_title("Click a heatmap cell to show the selected audio segment beat/energy chart")
        self.ax_beat.set_xlabel("Time inside selected segment [s]")
        self.ax_beat.set_ylabel("Normalized amplitude / energy")
        self.ax_beat.grid(True, alpha=0.3)
        self.ax_phase.set_title("Korolyuk / phase portrait of MP segment coefficients")
        self.ax_phase.set_xlabel("MP coefficient[n]")
        self.ax_phase.set_ylabel("MP coefficient[n+1]")
        self.ax_phase.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.mpl_connect("button_press_event", self.on_heatmap_click)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # --------------------------------------------------------

    def set_status(self, text):
        # GUI status only. Worker functions already print progress to the terminal.
        self.info.config(text=text)
        self.root.update_idletasks()

    # --------------------------------------------------------

    def on_segment_slider(self, *args):
        """Update local eigenvalue plot, heatmap marker, and selected audio overlay."""
        if self.X is None or self.result is None:
            return
        try:
            if self.selected_file_index is None:
                self.update_segment_view(redraw=False)
            else:
                self.plot_selected_song_mp_summary(
                    file_idx=self.selected_file_index,
                    seg_idx=int(self.selected_segment.get()),
                    redraw=False
                )
            self.refresh_selected_audio_view(redraw=False)
            self.canvas.draw_idle()
        except Exception as e:
            self.info.config(text=f"Segment/audio update failed: {e}")

    # --------------------------------------------------------

    def mark_segment_changed(self, *args):
        if self.X is not None:
            self.info.config(text="Segment changed — press Analyze MP again.")
            self.Xrec = None
            self.result = None
            self.last_reconstructed_modes = None

    # --------------------------------------------------------

    def mark_render_changed(self, *args):
        if self.Xrec is not None:
            self.info.config(text="Dry/Wet changed — lower filtered overlay and Render will use new values.")
            self.refresh_selected_audio_view()

    # --------------------------------------------------------

    def schedule_mode_update(self, *args):
        if self.X is None:
            return

        if self._mode_update_after_id is not None:
            try:
                self.root.after_cancel(self._mode_update_after_id)
            except Exception:
                pass

        self._mode_update_after_id = self.root.after(250, self.update_modes_from_slider)

    # --------------------------------------------------------

    def update_modes_from_slider(self):
        self._mode_update_after_id = None

        if self.X is None:
            return

        try:
            current_seg = int(self.seg_size.get())

            if current_seg != self.last_analyzed_seg:
                self.info.config(text="Segment changed — press Analyze MP again.")
                return

            k = int(self.n_modes.get())

            self.set_status(f"MP modes changed to {k} — recomputing reconstruction...")

            if self.matrix_geometry == "vertical_packet" and self.result is not None and "packet_tensor" in self.result:
                self.result = vertical_packet_reconstruct_from_tensor(
                    self.result["packet_tensor"],
                    k
                )
                self.X = self.result["Xrec"]
                self.Xrec = self.result["Xrec"]
            else:
                self.result = mp_reconstruct(
                    self.X,
                    k,
                    use_gpu=False
                )

                self.Xrec = self.result["Xrec"]

            self.last_reconstructed_modes = k

            self.update_tables_and_plot(current_seg, k)
            self.refresh_selected_audio_view()

            self.set_status(f"Updated reconstruction, graph and lower audio overlay with {k} modes.")

        except Exception:
            print("\n" + "=" * 80, flush=True)
            print("ERROR WHILE UPDATING MP MODES", flush=True)
            print("=" * 80, flush=True)
            traceback.print_exc()
            print("=" * 80 + "\n", flush=True)
            self.info.config(text="Error while updating MP modes. See terminal.")

    # --------------------------------------------------------

    def select_folder(self):
        folder = filedialog.askdirectory(title="Select MONO audio folder")

        if not folder:
            return

        self.folder = Path(folder)
        self.files = list_audio_files(self.folder)
        self.names = [p.stem for p in self.files]
        self.display_names = [clean_song_label_from_path(p, i + 1) for i, p in enumerate(self.files)]

        self.X = None
        self.Xrec = None
        self.result = None
        self.df_rank = None
        self.df_eig = None
        self.last_analyzed_seg = None
        self.last_reconstructed_modes = None
        self.current_ranked_heat = None
        self.current_ranked_names = None
        self.current_plot_seg = None
        self.current_rank_indices = None
        self.selected_file_index = None
        self.selected_rank_row = None

        self.info.config(text=f"Selected: {self.folder} | files: {len(self.files)}")

        log(f"Selected folder: {self.folder}")
        log(f"Files: {len(self.files)}")

    # --------------------------------------------------------

    def analyze_entropy(self):
        """Run Shannon/Fisher entropy-domain MP analysis."""
        self.feature_mode = "entropy"
        self.matrix_geometry = self.geometry_var.get() if hasattr(self, "geometry_var") else "song_time"
        self.analyze()

    # --------------------------------------------------------

    def analyze_audio(self):
        """Run direct RMS audio-domain MP analysis."""
        self.feature_mode = "audio"
        self.matrix_geometry = self.geometry_var.get() if hasattr(self, "geometry_var") else "song_time"
        self.analyze()

    # --------------------------------------------------------

    def on_geometry_changed(self):
        """Mark current analysis stale when changing matrix geometry."""
        if hasattr(self, "geometry_var"):
            self.matrix_geometry = self.geometry_var.get()
        if self.X is not None:
            self.info.config(text="MP geometry changed — press Analyze ENTROPY MP or Analyze AUDIO MP again.")
            self.Xrec = None
            self.result = None
            self.last_reconstructed_modes = None

    # --------------------------------------------------------

    def on_row_order_changed(self):
        """Redraw current heatmap with fixed file order or MP-rank order."""
        if hasattr(self, "row_order_var"):
            self.row_order_mode = self.row_order_var.get()

        if self.result is not None and self.X is not None:
            try:
                seg = int(self.seg_size.get())
                k = int(self.n_modes.get())
                self.update_tables_and_plot(seg, k)
                self.refresh_selected_audio_view()
                self.info.config(
                    text=f"Row order changed to {self.row_order_mode}. Heatmap redrawn without recomputing MP."
                )
            except Exception as e:
                traceback.print_exc()
                self.info.config(text=f"Row order update failed: {e}")

    # --------------------------------------------------------

    def on_entropy_diag_changed(self):
        """Switch bottom-right entropy diagnostic view without recomputing."""
        if hasattr(self, "entropy_diag_var"):
            self.entropy_diag_mode = self.entropy_diag_var.get()
        if self.feature_mode == "entropy" and self.selected_file_index is not None:
            try:
                self.refresh_selected_audio_view(redraw=True)
                self.info.config(text=f"Entropy diagnostic changed to {self.entropy_diag_mode}.")
            except Exception as e:
                traceback.print_exc()
                self.info.config(text=f"Entropy diagnostic update failed: {e}")

    # --------------------------------------------------------

    def analyze(self):
        if self.folder is None:
            messagebox.showerror("Error", "Select folder first.")
            return

        try:
            seg = int(self.seg_size.get())
            k = int(self.n_modes.get())

            mode_label = "Shannon/Fisher entropy" if self.feature_mode == "entropy" else "direct audio RMS"
            geom_label = "song×time" if self.matrix_geometry == "song_time" else "vertical packet vertical grouping"
            self.set_status(f"Building {mode_label} matrix [{geom_label}]...")
            log_memory("before build")

            workers = int(self.cpu_workers.get())

            if self.matrix_geometry == "vertical_packet":
                # True vertical packet scan:
                # for every temporal segment, build songs × intra-segment-points
                # and run a separate MP/PCA packet analysis.
                A, self.min_segments = build_vertical_packet_tensor(
                    self.files,
                    seg,
                    status_callback=self.set_status,
                    n_workers=workers,
                    feature_mode=self.feature_mode,
                    packet_points=PACKET_POINTS,
                )

                self.last_analyzed_seg = seg

                self.set_status("Running vertical packet MP scan...")
                log_memory("before vertical packet MP")

                self.result = vertical_packet_reconstruct_from_tensor(A, k)
                self.X = self.result["Xrec"]
                self.Xrec = self.result["Xrec"]
                self.last_reconstructed_modes = k
            else:
                self.X, self.min_segments = build_matrix(
                    self.files,
                    seg,
                    status_callback=self.set_status,
                    n_workers=workers,
                    feature_mode=self.feature_mode,
                    matrix_geometry=self.matrix_geometry
                )

                self.last_analyzed_seg = seg

                self.set_status("Running MP/PCA reconstruction...")
                log_memory("before reconstruct")

                self.result = mp_reconstruct(
                    self.X,
                    k,
                    use_gpu=False
                )

                self.Xrec = self.result["Xrec"]
                self.last_reconstructed_modes = k

            # Auto-select a narrow local eig window so the top-left spectrum
            # actually changes when clicking different vertical segments.
            try:
                auto_w = min(self.min_segments, max(len(self.files) + 2, min(64, self.min_segments)))
                self.local_window_var.set(int(auto_w))
            except Exception:
                pass

            # Keep separate Shannon and Fisher-max trajectories for
            # entropy-specific Shannon-Fisher diagnostic space.
            self.H_matrix = None
            self.F_matrix = None
            if self.feature_mode == "entropy":
                try:
                    self.set_status("Building Shannon/Fisher diagnostic matrices...")
                    self.H_matrix, self.F_matrix = build_entropy_fisher_matrices(
                        self.files,
                        seg,
                        self.min_segments,
                        status_callback=self.set_status,
                        n_workers=workers,
                    )
                except Exception:
                    traceback.print_exc()
                    self.H_matrix = None
                    self.F_matrix = None

            self.update_tables_and_plot(seg, k)

            self.set_status(
                f"Done [{self.feature_mode.upper()} | {self.matrix_geometry} | {self.row_order_mode}]. "
                f"Segments={self.min_segments}, "
                f"MP signal modes={self.result['n_signal']}, "
                f"render modes={k}"
            )

            log_memory("after analyze")

        except Exception:
            print("\n" + "=" * 80, flush=True)
            print("ERROR IN ANALYZE", flush=True)
            print("=" * 80, flush=True)
            traceback.print_exc()
            print("=" * 80 + "\n", flush=True)

            gc.collect()
            messagebox.showerror("Error", "Analyze failed. See terminal traceback.")

    # --------------------------------------------------------

    def update_tables_and_plot(self, seg, k):
        evals = self.result["evals"]
        evecs = self.result["evecs"]
        lam_minus = self.result["lam_minus"]
        lam_plus = self.result["lam_plus"]
        n_signal = self.result["n_signal"]
        scores = self.result["scores"]

        # ------------------------------------------------------------
        # Row ordering
        # ------------------------------------------------------------
        # mp_rank:
        #   previous behavior — rows sorted by global MP score / mean packet
        #   participation.
        #
        # file_order:
        #   fixed original file/album order.
        # ------------------------------------------------------------
        self.row_order_mode = (
            self.row_order_var.get()
            if hasattr(self, "row_order_var")
            else getattr(self, "row_order_mode", "mp_rank")
        )

        if self.row_order_mode == "file_order":
            rank = np.arange(len(scores), dtype=int)
        else:
            rank = np.argsort(scores)[::-1]

        self.current_rank_indices = rank
        ranked_names = [self.display_names[i] for i in rank]

        # Visualization:
        # - song×time: show the historical row-normalized reconstructed field.
        # - vertical_packet: Xrec is already a songs×segments participation map
        #   produced column-by-column by the packet MP scan. Do NOT run
        #   prepare_heatmap again, because that row-normalizes and destroys
        #   the packet coloring until K is changed several times.
        if self.matrix_geometry == "vertical_packet":
            src_heat = self.Xrec if self.Xrec is not None else self.X
            ranked_heat = np.asarray(src_heat, dtype=np.float64)[rank, :]
            ranked_heat = np.nan_to_num(ranked_heat, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            if self.Xrec is not None:
                ranked_heat = prepare_heatmap(self.Xrec)[rank, :]
            else:
                ranked_heat = prepare_heatmap(self.X)[rank, :]

        self.df_rank = pd.DataFrame({
            "rank": np.arange(1, len(self.names) + 1),
            "item": ranked_names,
            "mp_score": scores[rank],
            "pc1_abs": np.abs(evecs[rank, 0]),
            "pc1_signed": evecs[rank, 0],
            "segment_size": seg,
            "render_modes": k,
            "mp_signal_modes": n_signal,
            "feature_mode": self.feature_mode,
            "matrix_geometry": self.matrix_geometry,
            "row_order": self.row_order_mode,
        })

        self.df_eig = pd.DataFrame({
            "mode_index": np.arange(1, len(evals) + 1),
            "eigenvalue": evals,
            "above_mp": evals > lam_plus,
            "segment_size": seg,
            "render_modes": k,
            "feature_mode": self.feature_mode,
            "matrix_geometry": self.matrix_geometry,
            "row_order": self.row_order_mode,
        })

        # Preserve selected song marker after row-order changes by finding
        # the visible row of the currently selected file.
        if self.selected_file_index is not None:
            try:
                matches = np.where(rank == int(self.selected_file_index))[0]
                self.selected_rank_row = int(matches[0]) if len(matches) else None
            except Exception:
                self.selected_rank_row = None

        self.draw(
            ranked_names,
            ranked_heat,
            evals,
            lam_minus,
            lam_plus,
            n_signal,
            seg,
            k
        )

        if self.selected_file_index is not None:
            try:
                self.plot_selected_song_mp_summary(
                    file_idx=self.selected_file_index,
                    seg_idx=int(self.selected_segment.get()),
                    redraw=False
                )
            except Exception:
                pass

    # --------------------------------------------------------

    def ensure_current_reconstruction(self, force=False):
        if self.X is None:
            raise RuntimeError("Run Analyze MP first.")

        current_seg = int(self.seg_size.get())

        if current_seg != self.last_analyzed_seg:
            raise RuntimeError("Segment size changed. Press Analyze MP again before rendering.")

        current_k = int(self.n_modes.get())

        if force or self.Xrec is None or current_k != self.last_reconstructed_modes:
            self.set_status(f"Computing MP inverse reconstruction for K={current_k} before render...")

            if self.matrix_geometry == "vertical_packet" and self.result is not None and "packet_tensor" in self.result:
                self.result = vertical_packet_reconstruct_from_tensor(
                    self.result["packet_tensor"],
                    current_k
                )
                self.X = self.result["Xrec"]
                self.Xrec = self.result["Xrec"]
            else:
                self.result = mp_reconstruct(
                    self.X,
                    current_k,
                    use_gpu=False
                )

                self.Xrec = self.result["Xrec"]

            self.last_reconstructed_modes = current_k

            self.update_tables_and_plot(current_seg, current_k)

        return current_k

    # --------------------------------------------------------

    def render(self):
        try:
            # Force a fresh inverse MP reconstruction immediately before rendering,
            # so the rendered WAV files are guaranteed to use the current MP modes.
            k = self.ensure_current_reconstruction(force=True)

            seg = int(self.seg_size.get())

            dry = float(self.dry_percent.get()) / 100.0
            wet = float(self.wet_percent.get()) / 100.0

            out_dir = self.folder.parent / f"{self.folder.name}_MP_{self.feature_mode.upper()}_{self.matrix_geometry}_LOWRANK_AUDIO_K{k}_seg{seg}_D{int(dry*100)}_W{int(wet*100)}"

            workers = int(self.cpu_workers.get())

            # Render from the centered K-mode MP inverse component.
            # This is the field that changes directly when the MP modes slider changes.
            X_for_audio = self.result.get("Xrender", self.result.get("Xrec", self.Xrec))

            render_outputs(
                files=self.files,
                Xrec=X_for_audio,
                min_segments=self.min_segments,
                seg_size=seg,
                out_dir=out_dir,
                n_modes=k,
                dry=dry,
                wet=wet,
                status_callback=self.set_status,
                n_workers=workers
            )

            self.set_status(f"Rendered files saved to: {out_dir}")
            messagebox.showinfo("Done", f"Rendered files saved to:\n{out_dir}")

        except Exception:
            print("\n" + "=" * 80, flush=True)
            print("ERROR IN RENDER", flush=True)
            print("=" * 80, flush=True)
            traceback.print_exc()
            print("=" * 80 + "\n", flush=True)

            gc.collect()
            messagebox.showerror("Error", "Render failed. See terminal traceback.")


    # --------------------------------------------------------

    def render_archetype_collage(self):
        """Create a new WAV by selecting the strongest MP/archetype song per segment."""
        try:
            k = self.ensure_current_reconstruction(force=True)
            seg = int(self.seg_size.get())
            out_dir = self.folder.parent / f"{self.folder.name}_ARCHETYPE_COLLAGE_K{k}_seg{seg}"

            # Use reconstructed feature field for winner selection.
            # Xrender is centered/signed and good for filtering, but for collage
            # it can select anti-phase negative components as false winners.
            X_for_collage = self.result.get("Xrec", self.Xrec)

            out_path, csv_path, summary_path = render_archetype_collage(
                files=self.files,
                Xfield=X_for_collage,
                min_segments=self.min_segments,
                seg_size=seg,
                out_dir=out_dir,
                n_modes=k,
                status_callback=self.set_status,
                crossfade_ms=30.0,
                use_abs=False,
            )

            self.set_status(f"Archetype collage saved: {out_path}")
            messagebox.showinfo(
                "Archetype collage done",
                f"Saved WAV:\n{out_path}\n\nWinners CSV:\n{csv_path}\n\nSummary CSV:\n{summary_path}"
            )

        except Exception:
            print("\n" + "=" * 80, flush=True)
            print("ERROR IN ARCHETYPE COLLAGE", flush=True)
            print("=" * 80, flush=True)
            traceback.print_exc()
            print("=" * 80 + "\n", flush=True)
            gc.collect()
            messagebox.showerror("Error", "Archetype collage failed. See terminal traceback.")

    # --------------------------------------------------------

    def _adjust_bottom_axes_for_legends(self):
        """Stable manual layout for the GUI plots.

        The previous automatic tight_layout could push the heatmap/waterfall
        and eigenvalue plots out of the visible region after segment-slider
        updates, especially with long labels and a large selected-segment index.
        This fixed layout keeps:
          - top-left: selected segment eigenvalues;
          - top-right: local eigen-density;
          - middle: waterfall / MP heatmap;
          - bottom-left: audio/entropy overlay;
          - bottom-right: phase portrait.
        """
        try:
            # Top row
            self.ax_eig.set_position([0.060, 0.735, 0.505, 0.205])
            self.ax_density.set_position([0.665, 0.735, 0.245, 0.205])

            # Waterfall / heatmap row.  Labels are drawn in a separate
            # narrow axis at left, so the heatmap itself has no y tick labels.
            if hasattr(self, "ax_heat_labels"):
                self.ax_heat_labels.set_position([0.010, 0.390, 0.175, 0.265])
            self.ax_heat.set_position([0.190, 0.390, 0.690, 0.265])
            if self.cbar is not None:
                try:
                    self.cbar.ax.set_position([0.900, 0.390, 0.012, 0.265])
                except Exception:
                    pass

            # Bottom row
            self.ax_beat.set_position([0.055, 0.075, 0.525, 0.205])
            self.ax_phase.set_position([0.680, 0.075, 0.205, 0.205])
        except Exception:
            pass

    def draw(self, ranked_names, ranked_heat, evals, lam_minus, lam_plus, n_signal, seg, k):
        if self.cbar is not None:
            try:
                self.cbar.remove()
            except Exception:
                pass
            self.cbar = None

        while len(self.fig.axes) > 6:
            ax = self.fig.axes[-1]
            if ax is getattr(self, "ax_heat_labels", None):
                break
            self.fig.delaxes(ax)

        self.ax_eig.clear()
        self.ax_density.clear()
        self.ax_heat.clear()
        if hasattr(self, "ax_heat_labels"):
            self.ax_heat_labels.clear()
            self.ax_heat_labels.axis("off")
        self.ax_beat.clear()
        self.ax_phase.clear()
        self.ax_beat.set_title("Click a heatmap cell to show the selected audio segment beat/energy chart")
        self.ax_beat.set_xlabel("Time inside selected segment [s]")
        self.ax_beat.set_ylabel("Normalized amplitude / energy")
        self.ax_beat.grid(True, alpha=0.3)
        self.ax_phase.set_title("Korolyuk / phase portrait of MP segment coefficients")
        self.ax_phase.set_xlabel("MP coefficient[n]")
        self.ax_phase.set_ylabel("MP coefficient[n+1]")
        self.ax_phase.grid(True, alpha=0.3)

        # Store the currently displayed heatmap so the segment slider can update
        # only the marker/eigenvalue plots without recomputing the whole figure.
        self.current_ranked_heat = ranked_heat
        self.current_ranked_names = ranked_names
        self.current_plot_seg = seg

        # Keep selected segment in range and stretch the slider over the plot width.
        max_seg = max(0, ranked_heat.shape[1] - 1)
        try:
            self.segment_pick_scale.config(to=max_seg)
            if int(self.selected_segment.get()) > max_seg:
                self.selected_segment.set(max_seg)
        except Exception:
            pass

        if self.matrix_geometry == "vertical_packet":
            # Packet map is already percent participation per column.
            # Use a robust but stable scale from the first draw.
            vmax = np.percentile(ranked_heat, 99.0)
            if vmax <= 0:
                vmax = ranked_heat.max() + EPS
            # Do not let a single packet column blow the whole color scale.
            # Typical mean is 100 / number_of_songs.
            mean_part = 100.0 / max(1, ranked_heat.shape[0])
            vmax = max(vmax, mean_part * 3.0)
        else:
            vmax = np.percentile(ranked_heat, 99.5)

            if vmax <= 0:
                vmax = ranked_heat.max() + EPS

        im = self.ax_heat.imshow(
            ranked_heat,
            aspect="auto",
            interpolation="nearest",
            cmap="turbo",
            vmin=0,
            vmax=vmax
        )

        self.ax_heat.set_title(f"MP reconstructed {self.feature_mode.upper()} field — {self.matrix_geometry} — {self.row_order_mode} — seg={seg}, modes={k}")
        self.ax_heat.set_xlabel("Segment index")
        self.ax_heat.set_ylabel("Song rank" if self.matrix_geometry == "song_time" else "Segment rank")

        # IMPORTANT FIX:
        # Do NOT use heatmap y tick labels for song names.  With repeated GUI
        # redraws and many rows, Matplotlib can visually stack old tick-label
        # artists and create a black wall of repeated titles.  The heatmap axis
        # therefore shows no y tick labels; a separate label axis at left is
        # cleared and redrawn once per refresh, with exactly one text object per row.
        self.ax_heat.set_yticks([])
        self.ax_heat.set_yticklabels([])
        self.ax_heat.tick_params(axis="y", which="both", left=False, labelleft=False)
        self.ax_heat.minorticks_off()
        self.ax_heat.set_ylim(len(ranked_names) - 0.5, -0.5)

        if hasattr(self, "ax_heat_labels"):
            self.ax_heat_labels.clear()
            self.ax_heat_labels.axis("off")
            self.ax_heat_labels.set_xlim(0.0, 1.0)
            self.ax_heat_labels.set_ylim(len(ranked_names) - 0.5, -0.5)
            self.ax_heat_labels.set_facecolor("white")
            for row_i, name in enumerate(ranked_names):
                # One and only one label per row.  Clip it to the left label axis.
                self.ax_heat_labels.text(
                    0.98,
                    row_i,
                    str(name),
                    ha="right",
                    va="center",
                    fontsize=4.6,
                    family="DejaVu Sans Mono",
                    clip_on=True,
                )

        self.cbar = self.fig.colorbar(im, ax=self.ax_heat, fraction=0.018, pad=0.01)
        self.cbar.set_label("% participation")

        # Draw local eigenvalues for the selected segment and add the red marker.
        self.update_segment_view(redraw=False)

        # Fixed manual layout prevents waterfall/eigen plots from being pushed
        # off-screen by long labels after slider/click updates.
        self._adjust_bottom_axes_for_legends()
        self.canvas.draw_idle()

    # --------------------------------------------------------

    def _draw_heatmap_selection_markers(self, segment_index=None, rank_row=None):
        """Draw/refresh red selection markers on the heatmap robustly.

        This centralizes marker handling so click and slider updates behave
        identically in both song×time and vertical packet modes.
        """
        if self.current_ranked_heat is None:
            return

        n_rows, n_cols = self.current_ranked_heat.shape

        if segment_index is None:
            try:
                segment_index = int(self.selected_segment.get())
            except Exception:
                segment_index = 0

        segment_index = int(np.clip(segment_index, 0, max(0, n_cols - 1)))

        if rank_row is None:
            rank_row = self.selected_rank_row

        if rank_row is not None:
            rank_row = int(np.clip(int(rank_row), 0, max(0, n_rows - 1)))

        # Remove old artists every time. Some Matplotlib backends keep stale
        # references if the same axis is redrawn repeatedly.
        for attr in ("heat_marker", "heat_window_patch", "heat_row_marker"):
            artist = getattr(self, attr, None)
            if artist is not None:
                try:
                    artist.remove()
                except Exception:
                    pass
                setattr(self, attr, None)

        self.heat_marker = self.ax_heat.axvline(
            segment_index,
            color="red",
            linewidth=2.2,
            zorder=20,
        )

        if rank_row is not None:
            self.heat_row_marker = self.ax_heat.axhline(
                rank_row,
                color="red",
                linewidth=1.4,
                alpha=0.95,
                zorder=21,
            )

        # Use a very narrow span around the selected column.
        self.heat_window_patch = self.ax_heat.axvspan(
            segment_index - 0.5,
            segment_index + 0.5,
            color="red",
            alpha=0.10,
            zorder=19,
        )

    # --------------------------------------------------------

    def plot_selected_song_mp_summary(self, file_idx=None, seg_idx=None, redraw=True):
        """Top-row song-centric MP diagnostics.

        Left:
            selected song participation across MP modes/eigenvectors.

        Right:
            selected song original feature trajectory vs MP reconstruction.
        """
        if self.result is None or self.X is None:
            return

        if file_idx is None:
            file_idx = self.selected_file_index
        if seg_idx is None:
            try:
                seg_idx = int(self.selected_segment.get())
            except Exception:
                seg_idx = 0

        if file_idx is None:
            return

        file_idx = int(np.clip(int(file_idx), 0, self.X.shape[0] - 1))
        seg_idx = int(np.clip(int(seg_idx), 0, self.X.shape[1] - 1))

        evals = np.asarray(self.result.get("evals", []), dtype=np.float64)
        evecs = self.result.get("evecs", None)

        self.ax_eig.clear()
        self.ax_density.clear()

        # ------------------------------------------------------------
        # LEFT: selected song participation in the MP modes
        # ------------------------------------------------------------
        if evecs is not None:
            evecs = np.asarray(evecs, dtype=np.float64)
            max_modes = int(min(16, evecs.shape[1], len(evals) if len(evals) else evecs.shape[1]))
            modes = np.arange(1, max_modes + 1)
            part = np.abs(evecs[file_idx, :max_modes])

            bars = self.ax_eig.bar(modes, part, width=0.8, alpha=0.85)

            try:
                k_now = int(self.n_modes.get())
                if 1 <= k_now <= max_modes:
                    bars[k_now - 1].set_color("red")
                    bars[k_now - 1].set_alpha(0.95)
            except Exception:
                pass

            if len(evals) >= max_modes and max_modes > 0:
                ev = np.asarray(evals[:max_modes], dtype=np.float64)
                ev_norm = ev / (np.max(np.abs(ev)) + EPS)
                self.ax_eig.plot(
                    modes,
                    ev_norm,
                    marker="o",
                    linewidth=1.0,
                    alpha=0.75,
                    label="λ normalized"
                )
                self.ax_eig.legend(fontsize=7, loc="best")

            name = self.display_names[file_idx] if file_idx < len(self.display_names) else f"song {file_idx}"
            if len(name) > 42:
                name = name[:39] + "..."

            self.ax_eig.set_title(f"Selected song MP participation | {name}", fontsize=10)
            self.ax_eig.set_xlabel("MP mode number")
            self.ax_eig.set_ylabel("|participation|")
            self.ax_eig.set_xticks(modes)
            self.ax_eig.grid(True, alpha=0.25)
        else:
            self.ax_eig.text(
                0.5, 0.5,
                "No eigenvectors available",
                ha="center",
                va="center",
                transform=self.ax_eig.transAxes,
            )
            self.ax_eig.set_title("Selected song MP participation", fontsize=10)

        # ------------------------------------------------------------
        # RIGHT: original vs reconstructed trajectory for selected song
        # ------------------------------------------------------------
        x_original = np.asarray(self.X[file_idx, :], dtype=np.float64)

        x_rec = None
        if self.Xrec is not None and np.shape(self.Xrec) == np.shape(self.X):
            x_rec = np.asarray(self.Xrec[file_idx, :], dtype=np.float64)
        elif isinstance(self.result, dict) and "Xrec" in self.result and np.shape(self.result["Xrec"]) == np.shape(self.X):
            x_rec = np.asarray(self.result["Xrec"][file_idx, :], dtype=np.float64)

        t = np.arange(len(x_original))
        self.ax_density.plot(t, x_original, linewidth=1.0, alpha=0.55, label="original feature")
        if x_rec is not None:
            self.ax_density.plot(t, x_rec, linewidth=1.6, alpha=0.90, label="MP reconstructed")

        self.ax_density.axvline(seg_idx, color="red", linewidth=1.6, alpha=0.9, label="selected segment")
        self.ax_density.set_title("Selected song trajectory | original vs MP reconstruction", fontsize=10)
        self.ax_density.set_xlabel("Segment")
        self.ax_density.set_ylabel("feature / MP value")
        self.ax_density.grid(True, alpha=0.25)
        self.ax_density.legend(fontsize=7, loc="best")

        if redraw:
            self._adjust_bottom_axes_for_legends()
            self.canvas.draw_idle()

    # --------------------------------------------------------

    def update_segment_view(self, redraw=True):
        """
        Update the top eigenvalue plots for the selected segment and draw a
        vertical red marker on the heatmap.

        The eigen-spectrum is local: it is computed from a segment window around
        the selected index, because a single segment column alone cannot define
        a stable covariance spectrum.
        """
        if self.result is None or self.current_ranked_heat is None:
            return

        Xz = self.result.get("Xz", None)
        if Xz is None:
            return

        center = int(self.selected_segment.get())
        window = int(self.local_window_var.get())

        # Exact vertical-column eigenvalues for the clicked segment.
        # This uses only the values that are vertically aligned under the clicked
        # heatmap column: one Shannon/Fisher value per song.
        # C = v v^T / N is rank-1 by construction, so the first eigenvalue shows
        # the strength of that vertical collective fluctuation and the remaining
        # eigenvalues are the residual numerical spectrum.
        local = vertical_segment_eigs(
            Xz,
            center,
            global_lam_minus=self.result.get("lam_minus"),
            global_lam_plus=self.result.get("lam_plus"),
        )
        evals = local["evals"]
        evecs_local = local.get("evecs", None)
        lam_minus = local["lam_minus"]
        lam_plus = local["lam_plus"]
        n_signal = local["n_signal"]
        center = local["center"]
        win = 1
        left = center
        right = center + 1

        selected_file_idx = self.selected_file_index
        selected_participation = None
        if selected_file_idx is not None and evecs_local is not None:
            try:
                selected_file_idx = int(np.clip(int(selected_file_idx), 0, evecs_local.shape[0] - 1))
                selected_participation = float(evecs_local[selected_file_idx, 0])
            except Exception:
                selected_participation = None

        # --- Exact vertical-column eigenvalue curve for the selected segment ---
        self.ax_eig.clear()
        x = np.arange(1, len(evals) + 1)
        self.ax_eig.plot(x, evals, marker="o", linewidth=1.0)
        if lam_plus > 0:
            self.ax_eig.axhline(lam_plus, linestyle="--", linewidth=1.0, label=f"global MP λ+={lam_plus:.5f}")
        if lam_minus > 0:
            self.ax_eig.axhline(lam_minus, linestyle=":", linewidth=1.0, label=f"global MP λ-={lam_minus:.5f}")
        title_extra = ""
        if selected_file_idx is not None and selected_participation is not None:
            try:
                short_name = self.display_names[selected_file_idx]
                if len(short_name) > 34:
                    short_name = short_name[:31] + "..."
                title_extra = f" | selected: {short_name}, eig1={selected_participation:+.3f}"
            except Exception:
                title_extra = f" | selected eig1={selected_participation:+.3f}"

        self.ax_eig.set_title(
            f"Exact vertical-column eigenvalues | segment {center} | modes>{lam_plus:.3g}: {n_signal}{title_extra}",
            fontsize=10
        )
        self.ax_eig.set_xlabel("Mode")
        self.ax_eig.set_ylabel("eigenvalue of C=v vᵀ/N")
        self.ax_eig.grid(True, alpha=0.3)
        self.ax_eig.legend(fontsize=8, loc="best")

        # --- Vertical vector distribution + MP reference ---
        self.ax_density.clear()
        vcol = local.get("vector")
        try:
            xbars = np.arange(1, len(vcol) + 1)
            bars = self.ax_density.bar(xbars, vcol, width=0.85, alpha=0.85)

            # Highlight selected song. This makes the upper MP plots respond
            # to heatmap row clicks, not only to segment-column changes.
            if selected_file_idx is not None and 0 <= selected_file_idx < len(vcol):
                try:
                    bars[selected_file_idx].set_color("red")
                    bars[selected_file_idx].set_alpha(0.95)
                    self.ax_density.scatter(
                        [selected_file_idx + 1],
                        [vcol[selected_file_idx]],
                        s=55,
                        facecolors="none",
                        edgecolors="black",
                        linewidths=1.3,
                        zorder=10,
                        label="selected song"
                    )
                except Exception:
                    pass

            self.ax_density.axhline(0, linewidth=0.8, alpha=0.45)
            self.ax_density.set_title(f"Vertical segment vector across songs | seg {center}", fontsize=10)
            self.ax_density.set_xlabel("Song index in original file order")
            self.ax_density.set_ylabel("z-scored feature value")
            self.ax_density.grid(True, alpha=0.25)
            if selected_file_idx is not None:
                self.ax_density.legend(fontsize=7, loc="best")
        except Exception:
            self.ax_density.hist(evals, bins=min(20, len(evals)))
            if lam_plus > 0:
                self.ax_density.axvline(lam_plus, linestyle="--")
            self.ax_density.set_title(f"Vertical eigenvalue histogram | seg {center}")

        # --- Red marker on heatmap ---
        self._draw_heatmap_selection_markers(segment_index=center, rank_row=self.selected_rank_row)

        self.info.config(
            text=f"Selected segment={center} | moving vertical slab={left}:{right} ({win}) | local signal modes={n_signal} | MP λ+={lam_plus:.6f}"
        )

        if redraw:
            self._adjust_bottom_axes_for_legends()
            self.canvas.draw_idle()

    # --------------------------------------------------------

    def on_heatmap_click(self, event):
        """Click on heatmap: select song row + segment column and draw audio view.

        Works in both:
          - song×time mode;
          - true vertical packet scan mode.

        In true vertical packet mode the heatmap is still songs × segments.
        Only the way each column is computed is different.
        """
        if event.inaxes != self.ax_heat:
            return
        if self.current_ranked_heat is None or self.current_rank_indices is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        try:
            seg_idx = int(round(event.xdata))
            rank_row = int(round(event.ydata))

            n_rows, n_cols = self.current_ranked_heat.shape
            seg_idx = int(np.clip(seg_idx, 0, n_cols - 1))
            rank_row = int(np.clip(rank_row, 0, n_rows - 1))

            # In both geometries, current_rank_indices maps visible row -> song index.
            file_idx = int(self.current_rank_indices[rank_row])
            file_idx = int(np.clip(file_idx, 0, len(self.files) - 1))

            self.selected_segment.set(seg_idx)
            self.selected_file_index = file_idx
            self.selected_rank_row = rank_row

            # Draw immediately; do not rely only on Tk variable trace.
            self._draw_heatmap_selection_markers(segment_index=seg_idx, rank_row=rank_row)

            self.plot_selected_segment_beat(
                file_idx=file_idx,
                seg_idx=seg_idx,
                rank_row=rank_row,
                redraw=False
            )
            self.plot_selected_song_mp_summary(
                file_idx=file_idx,
                seg_idx=seg_idx,
                redraw=False
            )
            self._draw_heatmap_selection_markers(segment_index=seg_idx, rank_row=rank_row)
            self._adjust_bottom_axes_for_legends()
            self.canvas.draw_idle()

            self.info.config(
                text=(
                    f"Selected song-centric MP view [{self.matrix_geometry}]: song={self.files[file_idx].name} | "
                    f"rank row={rank_row + 1} | segment={seg_idx}"
                )
            )
        except Exception as e:
            traceback.print_exc()
            self.info.config(text=f"Heatmap click failed: {e}")

    # --------------------------------------------------------

    def refresh_selected_audio_view(self, redraw=True):
        """Redraw lower original/filtered waveform and song-centric top plots."""
        if self.selected_file_index is None:
            return
        try:
            seg_idx = int(self.selected_segment.get())
            self.plot_selected_segment_beat(
                self.selected_file_index,
                seg_idx,
                self.selected_rank_row,
                redraw=False
            )
            self.plot_selected_song_mp_summary(
                file_idx=self.selected_file_index,
                seg_idx=seg_idx,
                redraw=redraw
            )
        except Exception as e:
            self.info.config(text=f"Audio overlay update failed: {e}")

    # --------------------------------------------------------

    def _selected_filtered_audio_segment(self, file_idx, seg_idx, seg_size, view_chunks, original_norm_peak):
        """Return MP-filtered overlay for the selected visible audio chunks.

        This mirrors the export logic:
            gain[s] = dry + wet * normalized_MP_reconstruction[file, s]
            filtered_segment[s] = original_segment[s] * gain[s]
        The result is only scaled for display, not for saving.
        """
        if self.result is None:
            return None, None

        X_for_audio = self.result.get("Xrender", self.result.get("Xrec", self.Xrec))
        if X_for_audio is None:
            return None, None

        X_for_audio = np.asarray(X_for_audio, dtype=np.float64)
        if file_idx < 0 or file_idx >= X_for_audio.shape[0]:
            return None, None

        start_seg = int(max(0, seg_idx))
        end_seg = int(min(start_seg + int(view_chunks), X_for_audio.shape[1]))
        if end_seg <= start_seg:
            return None, None

        dry = float(self.dry_percent.get()) / 100.0
        wet = float(self.wet_percent.get()) / 100.0

        rec_lo = float(np.percentile(X_for_audio, 2.0))
        rec_hi = float(np.percentile(X_for_audio, 98.0))
        rec_seg = X_for_audio[file_idx, start_seg:end_seg]
        gain = xrec_to_segment_gain(rec_seg, dry, wet, rec_lo=rec_lo, rec_hi=rec_hi)

        y, sr = read_mono(self.files[file_idx])
        start_sample = start_seg * seg_size
        end_sample = min(end_seg * seg_size, len(y))
        y_vis = y[start_sample:end_sample].astype(np.float64, copy=False)
        if len(y_vis) == 0:
            return None, None

        # Apply one gain per original segment, exactly like export.
        filtered_parts = []
        pos = 0
        for g in gain:
            chunk = y_vis[pos:pos + seg_size]
            if len(chunk) == 0:
                break
            filtered_parts.append(chunk * float(g))
            pos += seg_size
        if not filtered_parts:
            return None, None
        y_filt = np.concatenate(filtered_parts)

        # Display scale: use the original visible peak so the red overlay is comparable.
        y_filt = y_filt - np.mean(y_filt)
        y_filt_norm = y_filt / (float(original_norm_peak) + EPS)
        t = np.arange(len(y_filt_norm), dtype=np.float64) / float(sr)
        return t, y_filt_norm

    # --------------------------------------------------------

    def plot_selected_shannon_fisher_space(self, file_idx, seg_idx, rank_row=None):
        """Bottom-right diagnostic for ENTROPY mode: Shannon vs Fisher max."""
        self.ax_phase.clear()

        if self.H_matrix is None or self.F_matrix is None:
            self.ax_phase.text(
                0.5, 0.5,
                "Shannon/Fisher matrices not available.\nRun ENTROPY analysis again.",
                ha="center",
                va="center",
                transform=self.ax_phase.transAxes,
            )
            self.ax_phase.set_title("Shannon-Fisher space")
            return

        try:
            file_idx = int(np.clip(int(file_idx), 0, self.H_matrix.shape[0] - 1))
            seg_idx = int(np.clip(int(seg_idx), 0, self.H_matrix.shape[1] - 1))

            H = np.asarray(self.H_matrix[file_idx, :], dtype=np.float64)
            F = np.asarray(self.F_matrix[file_idx, :], dtype=np.float64)
            H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
            F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)

            c = np.arange(len(H))
            self.ax_phase.plot(H, F, linewidth=0.8, alpha=0.35, label="song trajectory")
            self.ax_phase.scatter(H, F, c=c, s=18, alpha=0.85, label="segments")
            self.ax_phase.scatter([H[0]], [F[0]], s=65, marker="o", label="song start", zorder=6)
            self.ax_phase.scatter([H[-1]], [F[-1]], s=75, marker="x", label="song end", zorder=6)
            self.ax_phase.scatter(
                [H[seg_idx]], [F[seg_idx]],
                s=130,
                facecolors="none",
                edgecolors="red",
                linewidths=2.0,
                label=f"selected seg {seg_idx}",
                zorder=8,
            )

            if 0 <= seg_idx < len(H) - 1:
                self.ax_phase.annotate(
                    "",
                    xy=(H[seg_idx + 1], F[seg_idx + 1]),
                    xytext=(H[seg_idx], F[seg_idx]),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color="red"),
                )

            name = self.display_names[file_idx] if file_idx < len(self.display_names) else f"song {file_idx}"
            if len(name) > 34:
                name = name[:31] + "..."

            self.ax_phase.set_title(f"Shannon-Fisher space | {name}", fontsize=10)
            self.ax_phase.set_xlabel("Shannon entropy H(segment)")
            self.ax_phase.set_ylabel("Fisher max / structural activity F(segment)")
            self.ax_phase.grid(True, alpha=0.25)
            self.ax_phase.legend(fontsize=7, loc="best")

        except Exception as e:
            traceback.print_exc()
            self.ax_phase.text(
                0.5, 0.5,
                f"Shannon-Fisher plot failed:\n{e}",
                ha="center",
                va="center",
                transform=self.ax_phase.transAxes,
            )
            self.ax_phase.set_title("Shannon-Fisher space")

    # --------------------------------------------------------

    def plot_selected_fisher_lag_space(self, file_idx, seg_idx, rank_row=None):
        """ENTROPY diagnostic: position of H/F max vs signed Fisher lag."""
        self.ax_phase.clear()
        try:
            seg_size = int(self.last_analyzed_seg or self.seg_size.get())
            min_segments = int(self.min_segments or 0)
            if min_segments <= 0:
                raise RuntimeError("No analyzed segments available.")

            y, sr = read_mono(self.files[file_idx])
            y = y.astype(np.float64, copy=False)
            y /= np.sqrt(np.mean(y ** 2)) + EPS

            x_hf, y_lag, hf_amp, f_amp = segment_hf_fisher_lag_vectors(
                y, seg_size, min_segments, n_sub=64
            )

            seg_idx = int(np.clip(int(seg_idx), 0, len(x_hf) - 1))
            c = np.arange(len(x_hf))

            self.ax_phase.axhline(0.0, linewidth=0.9, alpha=0.55)
            self.ax_phase.scatter(x_hf, y_lag, c=c, s=18, alpha=0.85, label="segments")
            self.ax_phase.plot(x_hf, y_lag, linewidth=0.6, alpha=0.25, label="trajectory")
            self.ax_phase.scatter([x_hf[0]], [y_lag[0]], s=65, marker="o", label="song start", zorder=6)
            self.ax_phase.scatter([x_hf[-1]], [y_lag[-1]], s=75, marker="x", label="song end", zorder=6)
            self.ax_phase.scatter(
                [x_hf[seg_idx]], [y_lag[seg_idx]],
                s=140, facecolors="none", edgecolors="red",
                linewidths=2.0, label=f"selected seg {seg_idx}", zorder=8
            )

            if 0 <= seg_idx < len(x_hf) - 1:
                self.ax_phase.annotate(
                    "", xy=(x_hf[seg_idx + 1], y_lag[seg_idx + 1]),
                    xytext=(x_hf[seg_idx], y_lag[seg_idx]),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color="red"),
                )

            name = self.display_names[file_idx] if file_idx < len(self.display_names) else f"song {file_idx}"
            if len(name) > 34:
                name = name[:31] + "..."

            self.ax_phase.set_title(f"Fisher lag from H/F max | {name}", fontsize=10)
            self.ax_phase.set_xlabel("position of max H/F inside segment")
            self.ax_phase.set_ylabel("signed lag: + Fisher leads, - Fisher follows")
            self.ax_phase.set_xlim(-0.02, 1.02)
            self.ax_phase.grid(True, alpha=0.25)
            self.ax_phase.legend(fontsize=7, loc="best")

        except Exception as e:
            traceback.print_exc()
            self.ax_phase.text(
                0.5, 0.5, f"Fisher-lag plot failed:\n{e}",
                ha="center", va="center", transform=self.ax_phase.transAxes,
            )
            self.ax_phase.set_title("Fisher lag from H/F max")

    # --------------------------------------------------------

    def plot_selected_hf_fisher_com_pm_space(self, file_idx, seg_idx, rank_row=None):
        """ENTROPY diagnostic: signed H/F max and Fisher max around H/F COM.

        X:
            max(H/F) position - COM(H/F)

        Y:
            max(Fisher) position - COM(H/F)

        Both coordinates are signed:
            negative = earlier / left of H/F center of mass
            positive = later / right of H/F center of mass
        """
        self.ax_phase.clear()
        try:
            seg_size = int(self.last_analyzed_seg or self.seg_size.get())
            min_segments = int(self.min_segments or 0)
            if min_segments <= 0:
                raise RuntimeError("No analyzed segments available.")

            y, sr = read_mono(self.files[file_idx])
            y = y.astype(np.float64, copy=False)
            y /= np.sqrt(np.mean(y ** 2)) + EPS

            x_hf_pm, y_f_pm, hf_amp, f_amp, com_hf = segment_hf_fisher_com_pm_vectors(
                y, seg_size, min_segments, n_sub=64
            )

            seg_idx = int(np.clip(int(seg_idx), 0, len(x_hf_pm) - 1))
            c = np.arange(len(x_hf_pm))

            self.ax_phase.axhline(0.0, linewidth=0.9, alpha=0.55)
            self.ax_phase.axvline(0.0, linewidth=0.9, alpha=0.55)

            self.ax_phase.scatter(
                x_hf_pm,
                y_f_pm,
                c=c,
                s=18,
                alpha=0.85,
                label="segments",
            )
            self.ax_phase.plot(
                x_hf_pm,
                y_f_pm,
                linewidth=0.6,
                alpha=0.25,
                label="trajectory",
            )

            self.ax_phase.scatter(
                [x_hf_pm[0]], [y_f_pm[0]],
                s=65,
                marker="o",
                label="song start",
                zorder=6,
            )
            self.ax_phase.scatter(
                [x_hf_pm[-1]], [y_f_pm[-1]],
                s=75,
                marker="x",
                label="song end",
                zorder=6,
            )
            self.ax_phase.scatter(
                [x_hf_pm[seg_idx]], [y_f_pm[seg_idx]],
                s=140,
                facecolors="none",
                edgecolors="red",
                linewidths=2.0,
                label=f"selected seg {seg_idx}",
                zorder=8,
            )

            if 0 <= seg_idx < len(x_hf_pm) - 1:
                self.ax_phase.annotate(
                    "",
                    xy=(x_hf_pm[seg_idx + 1], y_f_pm[seg_idx + 1]),
                    xytext=(x_hf_pm[seg_idx], y_f_pm[seg_idx]),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color="red"),
                )

            name = self.display_names[file_idx] if file_idx < len(self.display_names) else f"song {file_idx}"
            if len(name) > 34:
                name = name[:31] + "..."

            self.ax_phase.set_title(f"H/F and Fisher around H/F COM | {name}", fontsize=10)
            self.ax_phase.set_xlabel("max H/F − COM(H/F)  [− earlier, + later]")
            self.ax_phase.set_ylabel("max Fisher − COM(H/F)  [− earlier, + later]")
            self.ax_phase.grid(True, alpha=0.25)
            self.ax_phase.legend(fontsize=7, loc="best")

        except Exception as e:
            traceback.print_exc()
            self.ax_phase.text(
                0.5, 0.5,
                f"COM ± plot failed:\n{e}",
                ha="center",
                va="center",
                transform=self.ax_phase.transAxes,
            )
            self.ax_phase.set_title("H/F and Fisher around H/F COM")

    # --------------------------------------------------------

    def plot_selected_phase_portrait(self, file_idx, seg_idx, rank_row=None):
        """Draw phase portrait from consecutive MP coefficients for the WHOLE selected song.

        This is the corrected interpretation requested by the user:
        for the selected song, take the full sequence of MP reconstructed segment
        coefficients across the whole song:

            c[0], c[1], c[2], ... c[T-1]

        and plot the consecutive-segment phase portrait:

            x = c[n]
            y = c[n+1]

        The selected segment is shown as a red marker on the same full-song
        trajectory, but the portrait itself is not limited to a local window.
        Changing K changes the entire row c[n] and therefore the portrait.
        """
        if not hasattr(self, "ax_phase"):
            return

        self.ax_phase.clear()
        self.ax_phase.grid(True, alpha=0.3)

        if self.result is None:
            self.ax_phase.set_title("Run Analyze MP first")
            return

        X_for_phase = self.result.get("Xrender", self.result.get("Xrec", self.Xrec))
        if X_for_phase is None:
            self.ax_phase.set_title("No MP reconstruction available")
            return

        X_for_phase = np.asarray(X_for_phase, dtype=np.float64)
        file_idx = int(np.clip(file_idx, 0, X_for_phase.shape[0] - 1))
        T = X_for_phase.shape[1]
        if T < 3:
            self.ax_phase.set_title("Too few segments for full-song phase portrait")
            return

        seg_idx = int(np.clip(seg_idx, 0, T - 2))

        # Full-song MP coefficient trajectory for the selected song.
        z = X_for_phase[file_idx, :].astype(np.float64, copy=False)
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

        # Robust normalization only for plotting. It does not affect the export.
        z = z - np.median(z)
        scale = np.percentile(np.abs(z), 95.0) + EPS
        z = z / scale

        x = z[:-1]
        y = z[1:]
        idx = np.arange(len(x))
        dz = np.diff(z)

        # Full trajectory over all consecutive segments in the selected song.
        self.ax_phase.plot(x, y, linewidth=0.65, alpha=0.45, label="whole song trajectory")
        sc = self.ax_phase.scatter(x, y, c=idx, s=15, alpha=0.78, label="segments")

        # Start/end markers of the whole song trajectory.
        self.ax_phase.scatter([x[0]], [y[0]], s=45, marker="o", label="song start")
        self.ax_phase.scatter([x[-1]], [y[-1]], s=55, marker="x", label="song end")

        # Selected segment marker on the full-song portrait.
        sx = x[seg_idx]
        sy = y[seg_idx]
        self.ax_phase.scatter(
            [sx], [sy],
            s=100,
            marker="o",
            facecolors="none",
            edgecolors="red",
            linewidths=2.0,
            label=f"selected seg {seg_idx}→{seg_idx+1}"
        )
        self.ax_phase.annotate(
            f"{seg_idx}",
            xy=(sx, sy),
            xytext=(6, 6),
            textcoords="offset points",
            color="red",
            fontsize=8
        )

        self.ax_phase.axhline(0, linewidth=0.8, alpha=0.4)
        self.ax_phase.axvline(0, linewidth=0.8, alpha=0.4)
        self.ax_phase.set_xlabel("MP coefficient c[n]")
        self.ax_phase.set_ylabel("MP coefficient c[n+1]")

        k = int(self.n_modes.get())
        title_name = self.files[file_idx].stem if self.files else f"song {file_idx+1}"
        if len(title_name) > 38:
            title_name = title_name[:35] + "..."
        self.ax_phase.set_title(
            f"Full-song phase portrait | song={file_idx+1}, K={k}, segments=0:{T-1}",
            fontsize=10
        )
        self.ax_phase.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=7,
            borderaxespad=0.0
        )

        # Compact diagnostic over the whole song trajectory.
        r = np.sqrt(x*x + y*y)
        ang = np.unwrap(np.arctan2(y, x))
        if len(r):
            self.ax_phase.text(
                0.02, 0.02,
                f"whole song\nmean r={np.mean(r):.3f}\nstd Δc={np.std(dz):.3f}\nturn={ang[-1]-ang[0]:.2f} rad",
                transform=self.ax_phase.transAxes,
                fontsize=8,
                va="bottom",
                bbox=dict(boxstyle="round", alpha=0.15)
            )

    # --------------------------------------------------------

    def plot_selected_segment_beat(self, file_idx, seg_idx, rank_row=None, redraw=True):
        """Draw original waveform + MP-filtered overlay + RMS/beat envelope."""
        if self.folder is None or not self.files:
            return

        file_idx = int(np.clip(file_idx, 0, len(self.files) - 1))
        seg_idx = int(max(0, seg_idx))
        seg_size = int(self.seg_size.get())
        view_chunks = int(max(1, self.view_chunks_var.get())) if hasattr(self, "view_chunks_var") else 1
        pth = self.files[file_idx]

        # Build a visible multi-chunk segment for analysis/plotting.
        y, sr = read_mono(pth)
        start_seg = seg_idx
        end_seg = min(seg_idx + view_chunks, len(y) // seg_size + 1)
        start = start_seg * seg_size
        end = min(end_seg * seg_size, len(y))
        if start >= len(y) or end <= start:
            raise RuntimeError(f"Selected segment {seg_idx} starts outside audio length.")

        seg = y[start:end].astype(np.float64, copy=False)
        if len(seg) < 8:
            raise RuntimeError("Selected segment is too short for beat chart.")

        seg = seg - np.mean(seg)
        seg_peak = np.max(np.abs(seg)) + EPS
        seg_norm = seg / seg_peak
        t = np.arange(len(seg_norm), dtype=np.float64) / float(sr)

        # Short-time RMS envelope inside the visible segment/chunks.
        frame = max(64, min(1024, int(seg_size) // 16))
        hop = max(16, frame // 4)
        if len(seg_norm) < frame:
            frame = max(8, len(seg_norm) // 2)
            hop = max(4, frame // 4)
        starts = np.arange(0, max(1, len(seg_norm) - frame + 1), hop, dtype=int)
        if len(starts) == 0:
            starts = np.array([0], dtype=int)
        rms = []
        centers = []
        for st in starts:
            chunk = seg_norm[st:st + frame]
            if len(chunk) == 0:
                continue
            rms.append(np.sqrt(np.mean(chunk ** 2)))
            centers.append(st + len(chunk) / 2.0)
        centers = np.asarray(centers, dtype=np.float64)
        rms = np.asarray(rms, dtype=np.float64)
        if len(rms) > 0:
            rms = rms / (np.max(rms) + EPS)
            if len(rms) >= 3:
                distance = max(1, len(rms) // 16)
                prominence = max(0.05, 0.20 * float(np.std(rms)))
                peaks, _ = find_peaks(rms, distance=distance, prominence=prominence)
            else:
                peaks = np.array([], dtype=int)
        else:
            peaks = np.array([], dtype=int)
        env_t = centers / float(sr)
        peak_t = env_t[peaks] if len(peaks) else np.array([], dtype=np.float64)
        peak_y = rms[peaks] if len(peaks) else np.array([], dtype=np.float64)

        # MP-filtered overlay for exactly the current K/Dry/Wet/current visible chunks.
        ft, filt = self._selected_filtered_audio_segment(
            file_idx=file_idx,
            seg_idx=seg_idx,
            seg_size=seg_size,
            view_chunks=view_chunks,
            original_norm_peak=seg_peak,
        )

        self.ax_beat.clear()
        self.ax_beat.plot(t, seg_norm, linewidth=0.45, alpha=0.58, label="original waveform")
        if ft is not None and filt is not None:
            self.ax_beat.plot(ft, filt, linewidth=0.45, color="red", alpha=0.85, label="MP filtered overlay")
        if len(rms):
            self.ax_beat.plot(env_t, rms, linewidth=0.9, alpha=0.85, label="short-time RMS / beat energy")
        if len(peak_t):
            self.ax_beat.scatter(peak_t, peak_y, s=12, marker="o", label="energy peaks")

        # ------------------------------------------------------------
        # Shannon/Fisher mask marker inside the selected visible audio.
        # This answers: where does the entropy mask find its strongest
        # local Shannon/Fisher response, and does it coincide with a beat?
        #
        # It is computed on short overlapping frames inside the currently
        # displayed segment/chunks, not on the whole song. The green marker
        # is the maximum of:
        #       MSF_like = log1p( Shannon(frame) / Fisher(frame) )
        # where Fisher is the 95th percentile of |diff(frame)|/(|frame|+eps).
        # ------------------------------------------------------------
        try:
            sf_frame = max(128, min(4096, int(seg_size) // 8))
            sf_hop = max(32, sf_frame // 4)
            if len(seg_norm) < sf_frame:
                sf_frame = max(16, len(seg_norm) // 2)
                sf_hop = max(8, sf_frame // 4)

            sf_starts = np.arange(0, max(1, len(seg_norm) - sf_frame + 1), sf_hop, dtype=int)
            if len(sf_starts) == 0:
                sf_starts = np.array([0], dtype=int)

            sf_blocks = []
            sf_centers = []
            for st in sf_starts:
                chunk = seg_norm[st:st + sf_frame]
                if len(chunk) < 8:
                    continue
                if len(chunk) < sf_frame:
                    tmp = np.zeros(sf_frame, dtype=np.float64)
                    tmp[:len(chunk)] = chunk
                    chunk = tmp
                sf_blocks.append(chunk)
                sf_centers.append(st + sf_frame // 2)

            if sf_blocks:
                sf_blocks = np.asarray(sf_blocks, dtype=np.float64)
                Hloc = shannon_entropy_blocks(sf_blocks)
                Floc = fisher_blocks(sf_blocks)
                # Robust local Shannon/Fisher curve for plotting.
                Floc = np.nan_to_num(Floc, nan=EPS, posinf=EPS, neginf=EPS)
                Floc = np.maximum(np.abs(Floc), 1e-9)
                Hloc = np.nan_to_num(Hloc, nan=0.0, posinf=0.0, neginf=0.0)
                ratio_loc = np.clip(np.nan_to_num(Hloc / Floc, nan=0.0, posinf=1e6, neginf=0.0), 0.0, 1e6)
                SFmask = np.log1p(ratio_loc)
                SFmask = np.nan_to_num(SFmask, nan=0.0, posinf=0.0, neginf=0.0)

                # Visualize the full Shannon/Fisher entropy mask curve over the
                # currently visible audio.  It is normalized only for plotting
                # so it can be compared directly with the waveform and beat RMS.
                sf_t = np.asarray(sf_centers, dtype=np.float64) / float(sr)
                sf_curve = SFmask - np.nanmin(SFmask)
                sf_curve = sf_curve / (np.nanmax(sf_curve) + EPS)
                self.ax_beat.plot(
                    sf_t,
                    sf_curve,
                    color="darkgreen",
                    linewidth=0.75,
                    alpha=0.95,
                    label="Shannon/Fisher mask H/F"
                )

                # Also visualize Fisher-only response as a thinner dashed green
                # curve. This helps compare whether H/F and Fisher peak on the
                # same beat/onset or detect different structure.
                Floc_safe = np.nan_to_num(Floc, nan=0.0, posinf=0.0, neginf=0.0)
                f_curve = Floc_safe - np.nanmin(Floc_safe)
                f_curve = f_curve / (np.nanmax(f_curve) + EPS)
                self.ax_beat.plot(
                    sf_t,
                    f_curve,
                    color="seagreen",
                    linewidth=0.65,
                    alpha=0.85,
                    linestyle="--",
                    label="Fisher-only response"
                )

                # 1) Filled green dot: maximum of the current entropy mask H/F.
                jmax = int(np.argmax(SFmask))
                imax = int(np.clip(sf_centers[jmax], 0, len(seg_norm) - 1))
                tmax = imax / float(sr)
                ymax = float(sf_curve[jmax])

                self.ax_beat.axvline(tmax, color="darkgreen", linewidth=0.55, alpha=0.65)
                self.ax_beat.scatter(
                    [tmax], [ymax],
                    s=42,
                    marker="o",
                    color="darkgreen",
                    edgecolors="black",
                    linewidths=0.55,
                    zorder=9,
                    label="max Shannon/Fisher H/F"
                )
                self.ax_beat.annotate(
                    f"max H/F\n{tmax:.3f}s",
                    xy=(tmax, ymax),
                    xytext=(8, 10),
                    textcoords="offset points",
                    color="darkgreen",
                    fontsize=7,
                    bbox=dict(boxstyle="round", alpha=0.15)
                )

                # 2) Hollow green circle: maximum of Fisher only.
                # This marks the strongest local structural transition/onset-like response.
                Floc_safe = np.nan_to_num(Floc, nan=0.0, posinf=0.0, neginf=0.0)
                jfmax = int(np.argmax(Floc_safe))
                ifmax = int(np.clip(sf_centers[jfmax], 0, len(seg_norm) - 1))
                tfmax = ifmax / float(sr)
                yfmax = float(f_curve[jfmax])

                self.ax_beat.axvline(tfmax, color="seagreen", linewidth=0.55, alpha=0.50, linestyle="--")
                self.ax_beat.scatter(
                    [tfmax], [yfmax],
                    s=64,
                    marker="o",
                    facecolors="none",
                    edgecolors="seagreen",
                    linewidths=1.2,
                    zorder=10,
                    label="max Fisher only"
                )
                self.ax_beat.annotate(
                    f"max F\n{tfmax:.3f}s",
                    xy=(tfmax, yfmax),
                    xytext=(8, -18),
                    textcoords="offset points",
                    color="seagreen",
                    fontsize=7,
                    bbox=dict(boxstyle="round", alpha=0.12)
                )
        except Exception as sf_marker_error:
            log(f"WARNING: Shannon/Fisher marker failed: {sf_marker_error}")

        title_name = pth.name
        if len(title_name) > 95:
            title_name = title_name[:92] + "..."
        row_txt = "" if rank_row is None else f"rank row={rank_row + 1}, "
        k = int(self.n_modes.get())
        dry = int(self.dry_percent.get())
        wet = int(self.wet_percent.get())
        self.ax_beat.set_title(
            f"Audio segment | song={file_idx + 1}, seg={seg_idx}, chunks={view_chunks}, "
            f"K={k}, Dry/Wet={dry}/{wet}",
            fontsize=10
        )
        self.ax_beat.set_xlabel("Time inside selected visible segment(s) [s]")
        self.ax_beat.set_ylabel("Normalized waveform / energy")
        self.ax_beat.set_ylim(-1.08, 1.08)
        self.ax_beat.grid(True, alpha=0.3)
        self.ax_beat.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=7,
            borderaxespad=0.0
        )

        self.info.config(
            text=f"Selected: song={pth.name} | segment={seg_idx} | view chunks={view_chunks} | "
                 f"segment size={seg_size} samples | visible duration={len(seg_norm)/float(sr):.3f} s"
        )
        if self.feature_mode == "entropy":
            self.entropy_diag_mode = (
                self.entropy_diag_var.get()
                if hasattr(self, "entropy_diag_var")
                else getattr(self, "entropy_diag_mode", "sf_space")
            )
            if self.entropy_diag_mode == "fisher_lag":
                self.plot_selected_fisher_lag_space(file_idx, seg_idx, rank_row)
            elif self.entropy_diag_mode == "com_pm":
                self.plot_selected_hf_fisher_com_pm_space(file_idx, seg_idx, rank_row)
            else:
                self.plot_selected_shannon_fisher_space(file_idx, seg_idx, rank_row)
        else:
            self.plot_selected_phase_portrait(file_idx, seg_idx, rank_row)

        if redraw:
            self._adjust_bottom_axes_for_legends()
            self.canvas.draw_idle()

    # --------------------------------------------------------

    def export_entropy_maps_pdf(self):
        """Export MP entropy reconstruction atlas to PDF.

        PDF structure:
          Page 1  : fixed song ranking / heatmap row order
          Page 2  : K=1..16 reconstructed heatmaps stacked vertically
          Page 3  : max Fisher-like fluctuation over segments
          Page 4+ : one page per song:
                    top    - REAL segment-density curve from the current analysis, no 255 resampling
                    bottom - phase portrait K=1 and phase portrait K=6
        """
        if self.X is None or self.folder is None:
            messagebox.showerror("Error", "Run Analyze MP first.")
            return

        try:
            seg = int(self.seg_size.get())
            if seg != self.last_analyzed_seg:
                messagebox.showerror("Error", "Segment size changed. Press Analyze MP first.")
                return

            max_k = min(16, self.X.shape[0])
            K_A = 1
            K_B = min(6, self.X.shape[0])

            out_dir = self.folder.parent / f"{self.folder.name}_PDF_K1_{max_k}_seg{seg}"
            out_dir.mkdir(exist_ok=True)

            pdf_path = out_dir / f"MP_entropy_REAL_segment_density_K1_{max_k}_seg{seg}_K1_vs_K{K_B}_phase.pdf"

            self.set_status(f"Preparing PDF maps K=1..{max_k} ...")

            # Fixed ranking for all PDF pages, based on the current/global MP score.
            base_result = self.result if self.result is not None else mp_reconstruct(
                self.X,
                int(self.n_modes.get()),
                use_gpu=False
            )

            base_scores = base_result["scores"]
            rank = np.argsort(base_scores)[::-1]
            ranked_names = [self.display_names[i] for i in rank]

            # ------------------------------------------------------------
            # Build K=1..16 heatmaps with common color scale
            # ------------------------------------------------------------
            maps = []
            infos = []
            global_vmax = 0.0

            for k in range(1, max_k + 1):
                self.set_status(f"Computing PDF heatmap K={k}/{max_k} ...")
                r = mp_reconstruct(self.X, k, use_gpu=False)
                heat = prepare_heatmap(r["Xrec"])[rank, :]
                maps.append(heat)
                infos.append((k, r["n_signal"], r["lam_plus"]))

                v = float(np.percentile(heat, 99.5))
                if np.isfinite(v):
                    global_vmax = max(global_vmax, v)

            if global_vmax <= 0:
                global_vmax = max(float(np.max(h)) for h in maps) + EPS

            # ------------------------------------------------------------
            # Precompute K=1 and K=6 reconstructions for per-song pages
            # ------------------------------------------------------------
            self.set_status(f"Computing per-song comparison fields K=1 and K={K_B} ...")

            res_k1 = mp_reconstruct(self.X, K_A, use_gpu=False)
            res_k6 = mp_reconstruct(self.X, K_B, use_gpu=False)

            X_entropy_raw = sanitize_matrix(self.X, label="PDF song entropy raw X")
            Xrec_k1 = sanitize_matrix(res_k1["Xrec"], label="PDF reconstructed X K=1")
            Xrec_k6 = sanitize_matrix(res_k6["Xrec"], label=f"PDF reconstructed X K={K_B}")

            # ------------------------------------------------------------
            # Skorokhod-like peak displacement basis.
            # For every real audio segment and every song we find the internal
            # maximum of a local Shannon/Fisher score. This gives:
            #   tau_song[file, segment] in [0, 1]
            # Then the collective K=1 reference peak position is computed as
            # a K=1-weighted average across all songs for every segment:
            #   tau_ref[segment]
            # The per-song distance exported later is the simplest form:
            #   D[file, segment] = |tau_song[file, segment] - tau_ref[segment]|
            # ------------------------------------------------------------
            self.set_status("Computing intra-segment Shannon/Fisher peak positions for Skorokhod distance ...")
            peak_tau_all, peak_amp_all = compute_all_segment_peak_maps(
                self.files,
                self.min_segments,
                seg,
                status_callback=self.set_status,
                n_probe=255,
                n_workers=int(self.cpu_workers.get()),
            )

            # K=1 density is used only as a positive collective weight.
            k1_weight = np.zeros_like(Xrec_k1, dtype=np.float64)
            for _wi in range(Xrec_k1.shape[0]):
                _v = np.asarray(Xrec_k1[_wi, :], dtype=np.float64)
                _v = np.nan_to_num(_v, nan=0.0, posinf=0.0, neginf=0.0)
                _lo = float(np.percentile(_v, 2.0))
                _hi = float(np.percentile(_v, 98.0))
                if abs(_hi - _lo) < EPS:
                    k1_weight[_wi, :] = 1.0
                else:
                    k1_weight[_wi, :] = np.clip((_v - _lo) / (_hi - _lo + EPS), 0.0, 1.0)
            peak_tau_ref = weighted_reference_peak_position(peak_tau_all, k1_weight)
            peak_amp_ref = np.sum((k1_weight + EPS) * peak_amp_all, axis=0) / (np.sum(k1_weight + EPS, axis=0) + EPS)
            peak_distance_all = np.abs(peak_tau_all - peak_tau_ref[None, :])
            peak_distance_samples_all = peak_distance_all * float(seg)

            def _norm01(v):
                v = np.asarray(v, dtype=np.float64)
                v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

                lo = float(np.percentile(v, 2.0))
                hi = float(np.percentile(v, 98.0))

                if abs(hi - lo) < EPS:
                    return np.zeros_like(v)

                return np.clip((v - lo) / (hi - lo + EPS), 0.0, 1.0)

            def _resample_to_255(v, n=255):
                v = np.asarray(v, dtype=np.float64)

                if v.size == 0:
                    return np.zeros(n, dtype=np.float64)

                if v.size == 1:
                    return np.full(n, float(v[0]), dtype=np.float64)

                x_old = np.linspace(0.0, 1.0, v.size)
                x_new = np.linspace(0.0, 1.0, n)

                return np.interp(x_new, x_old, v)

            def _phase_xy(v):
                c = np.asarray(v, dtype=np.float64)
                c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)

                c = c - np.median(c)
                scale = float(np.percentile(np.abs(c), 95.0) + EPS)
                c = c / scale

                if len(c) < 3:
                    return np.array([]), np.array([]), np.array([]), c

                x = c[:-1]
                y = c[1:]
                idx = np.arange(len(x))

                return x, y, idx, c

            def _draw_phase(ax, fig, v, title, k_label):
                x, y, idx, c = _phase_xy(v)

                if len(x) > 0:
                    ax.plot(
                        x,
                        y,
                        color="0.45",
                        linewidth=0.35,
                        alpha=0.35,
                        label="trajectory",
                    )

                    sc = ax.scatter(
                        x,
                        y,
                        c=idx,
                        cmap="turbo",
                        s=10,
                        alpha=0.85,
                        label="segments",
                    )

                    ax.scatter([x[0]], [y[0]], s=40, marker="o", label="start")
                    ax.scatter([x[-1]], [y[-1]], s=50, marker="x", label="end")

                    cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.015)
                    cb.set_label("segment index", fontsize=7)

                    radius = np.sqrt(x * x + y * y)
                    dc = np.diff(c)

                    ax.text(
                        0.02,
                        0.02,
                        f"{k_label}\nmean r={np.mean(radius):.3f}\nstd r={np.std(radius):.3f}\nstd Δc={np.std(dc):.3f}",
                        transform=ax.transAxes,
                        fontsize=8,
                        va="bottom",
                        bbox=dict(boxstyle="round", alpha=0.16),
                    )

                ax.axhline(0, linewidth=0.6, alpha=0.35)
                ax.axvline(0, linewidth=0.6, alpha=0.35)
                ax.set_aspect("equal", adjustable="box")
                ax.set_title(title, fontsize=9)
                ax.set_xlabel("MP entropy coefficient c[n]")
                ax.set_ylabel("MP entropy coefficient c[n+1]")
                ax.grid(True, alpha=0.22)
                ax.legend(fontsize=7, loc="best")

            # ------------------------------------------------------------
            # PAGE 1: catalog / fixed ranking used in all heatmaps
            # ------------------------------------------------------------
            fig_catalog = plt.figure(figsize=(13.5, 19.0), dpi=150)
            ax_cat = fig_catalog.add_subplot(111)
            ax_cat.axis("off")

            title = (
                f"MP Entropy Atlas — Song ranking / fixed heatmap row order\n"
                f"segment={seg} samples | songs={len(ranked_names)} | "
                f"K range=1..{max_k} | per-song comparison: K=1 vs K={K_B} | "
                f"MP signal modes={infos[0][1]} | λ+={infos[0][2]:.4f}"
            )
            ax_cat.set_title(title, fontsize=15, fontweight="bold", loc="left", pad=18)

            n = len(ranked_names)
            ncols = 2 if n <= 42 else 3
            rows_per_col = int(np.ceil(n / ncols))
            x_positions = np.linspace(0.02, 0.68, ncols)
            y0 = 0.94
            dy = 0.86 / max(1, rows_per_col)

            for col in range(ncols):
                start_i = col * rows_per_col
                end_i = min(n, (col + 1) * rows_per_col)
                x = x_positions[col]

                ax_cat.text(
                    x,
                    y0 + 0.02,
                    "Rank | song",
                    fontsize=10,
                    fontweight="bold",
                    transform=ax_cat.transAxes,
                )

                for local_i, i in enumerate(range(start_i, end_i)):
                    name = str(ranked_names[i]).replace("｜", "|")
                    if len(name) > 66:
                        name = name[:63] + "..."

                    line = f"№{i + 1:02d} | {name}"

                    ax_cat.text(
                        x,
                        y0 - local_i * dy,
                        line,
                        fontsize=8.2,
                        family="DejaVu Sans Mono",
                        va="top",
                        transform=ax_cat.transAxes,
                    )

            ax_cat.text(
                0.02,
                0.035,
                "The ranking is fixed for all heatmaps and per-song pages. Rows are sorted by MP score from the current/global reconstruction.",
                fontsize=9,
                alpha=0.75,
                transform=ax_cat.transAxes,
            )

            # ------------------------------------------------------------
            # PAGE 2: K=1..16 heatmaps
            # ------------------------------------------------------------
            fig_h = max(18.0, 1.15 * max_k)
            fig_heat, axes = plt.subplots(
                max_k,
                1,
                figsize=(13.5, fig_h),
                dpi=150,
                sharex=True,
                constrained_layout=False,
            )

            if max_k == 1:
                axes = [axes]

            im = None
            for ax, heat, (k, nsig, lplus) in zip(axes, maps, infos):
                im = ax.imshow(
                    heat,
                    aspect="auto",
                    interpolation="nearest",
                    cmap="turbo",
                    vmin=0,
                    vmax=global_vmax,
                )

                ax.set_ylabel(f"K={k}", rotation=0, labelpad=24, fontsize=8, va="center")
                ax.set_yticks([])
                ax.set_title(
                    f"K={k:02d} | MP signal modes={nsig} | λ+={lplus:.4f}",
                    fontsize=8,
                    loc="left",
                    pad=1,
                )
                ax.grid(False)

            axes[-1].set_xlabel("Segment index")

            fig_heat.suptitle(
                f"MP entropy reconstructed maps, K=1..{max_k}, segment={seg} samples | fixed song order from page 1",
                fontsize=12,
                y=0.995,
            )

            fig_heat.subplots_adjust(
                left=0.055,
                right=0.90,
                top=0.975,
                bottom=0.035,
                hspace=0.18,
            )

            if im is not None:
                cax = fig_heat.add_axes([0.915, 0.08, 0.015, 0.86])
                cb = fig_heat.colorbar(im, cax=cax)
                cb.set_label("% participation", fontsize=9)

            # Collect all 255-point intermediate entropy/Fisher coefficients for audit/export.
            real_density_rows = []
            fisher_full_rows = []
            peak_distance_rows = []


            # ------------------------------------------------------------
            # PAGE 3: maximum Fisher-like fluctuation across songs per segment
            # ------------------------------------------------------------
            # For every song we compute a Fisher-like coefficient over the original
            # Shannon/Fisher trajectory X: |d(norm01(X))/dt|. Then for every segment
            # we take the maximum across all songs. This shows where the ensemble
            # has the strongest segment-level fluctuations.
            fisher_matrix = []
            for song_i_all in range(X_entropy_raw.shape[0]):
                raw_i = _norm01(X_entropy_raw[song_i_all, :])
                fisher_i = np.abs(np.gradient(raw_i))
                fisher_i = np.nan_to_num(fisher_i, nan=0.0, posinf=0.0, neginf=0.0)
                fisher_matrix.append(fisher_i)
            fisher_matrix = np.asarray(fisher_matrix, dtype=np.float64)
            fisher_max_by_segment = np.max(fisher_matrix, axis=0)
            fisher_argmax_song = np.argmax(fisher_matrix, axis=0)
            fisher_mean_by_segment = np.mean(fisher_matrix, axis=0)

            fig_fisher_max = plt.figure(figsize=(13.5, 6.2), dpi=150)
            ax_fm = fig_fisher_max.add_subplot(111)
            seg_x = np.arange(fisher_max_by_segment.size)
            ax_fm.plot(seg_x, fisher_max_by_segment, linewidth=1.1, color="darkgreen", label="max Fisher-like |dX/dt| across songs")
            ax_fm.plot(seg_x, fisher_mean_by_segment, linewidth=0.8, color="seagreen", alpha=0.75, linestyle="--", label="mean Fisher-like |dX/dt| across songs")
            # Mark the strongest global fluctuation points.
            if fisher_max_by_segment.size:
                top_n = min(12, fisher_max_by_segment.size)
                top_idx = np.argsort(fisher_max_by_segment)[-top_n:][::-1]
                ax_fm.scatter(top_idx, fisher_max_by_segment[top_idx], s=24, color="red", label="top maxima")
                for jj in top_idx[:6]:
                    song_lab = strip_flag_emojis(self.display_names[int(fisher_argmax_song[jj])]) if hasattr(self, "display_names") else str(int(fisher_argmax_song[jj]) + 1)
                    if len(song_lab) > 28:
                        song_lab = song_lab[:25] + "..."
                    ax_fm.annotate(
                        f"{int(jj)}\n{song_lab}",
                        xy=(jj, fisher_max_by_segment[jj]),
                        xytext=(4, 8),
                        textcoords="offset points",
                        fontsize=7,
                        color="red",
                    )
            ax_fm.set_title("Maximum Fisher-like entropy fluctuation per segment across all songs", fontsize=12, fontweight="bold")
            ax_fm.set_xlabel("Original segment index")
            ax_fm.set_ylabel("max |d normalized Shannon/Fisher X / d segment|")
            ax_fm.grid(True, alpha=0.28)
            ax_fm.legend(loc="best", fontsize=8)
            fig_fisher_max.tight_layout()

            # ------------------------------------------------------------
            # PAGE 4: collective K=1 reference peak positions used for
            # Skorokhod-like peak-displacement distance.
            # ------------------------------------------------------------
            fig_peak_ref = plt.figure(figsize=(13.5, 6.2), dpi=150)
            ax_pr = fig_peak_ref.add_subplot(111)
            px = np.arange(peak_tau_ref.size)
            ax_pr.plot(px, peak_tau_ref, color="crimson", linewidth=1.05, label="collective K=1 reference peak position τ_ref")
            ax_pr.plot(px, peak_amp_ref / (np.max(peak_amp_ref) + EPS), color="darkgreen", linewidth=0.75, alpha=0.78, linestyle="--", label="reference peak Shannon/Fisher score, normalized")
            ax_pr.set_title("Collective K=1 intra-segment Shannon/Fisher peak reference", fontsize=12, fontweight="bold")
            ax_pr.set_xlabel("Real segment index")
            ax_pr.set_ylabel("Peak position inside segment, τ ∈ [0, 1]")
            ax_pr.set_ylim(-0.05, 1.05)
            ax_pr.grid(True, alpha=0.28)
            ax_pr.legend(loc="best", fontsize=8)
            fig_peak_ref.tight_layout()

            # ------------------------------------------------------------
            # Save PDF: catalog + atlas + Fisher page + reference peaks + per-song pages
            # ------------------------------------------------------------
            with PdfPages(pdf_path) as pdf:
                pdf.savefig(fig_catalog, bbox_inches="tight")
                pdf.savefig(fig_heat, bbox_inches="tight")
                pdf.savefig(fig_fisher_max, bbox_inches="tight")
                pdf.savefig(fig_peak_ref, bbox_inches="tight")

                for out_rank, song_i in enumerate(rank, 1):
                    self.set_status(f"Writing PDF song page {out_rank}/{len(rank)} ...")

                    raw = X_entropy_raw[song_i, :]
                    rec1 = Xrec_k1[song_i, :]
                    rec6 = Xrec_k6[song_i, :]

                    # REAL display density: no resampling to 255.
                    # These arrays preserve the original segment grid from the current analysis.
                    raw_real = _norm01(raw)
                    rec1_real = _norm01(rec1)
                    rec6_real = _norm01(rec6)

                    fisher_proxy = np.abs(np.gradient(raw_real))
                    fisher_real = _norm01(fisher_proxy)

                    for seg_i, fval in enumerate(np.asarray(fisher_proxy, dtype=np.float64)):
                        fisher_full_rows.append({
                            "rank": int(out_rank),
                            "song_index": int(song_i + 1),
                            "song": strip_flag_emojis(str(self.display_names[song_i])),
                            "file_stem": str(self.names[song_i]),
                            "segment_index": int(seg_i),
                            "fisher_coefficient_full_resolution": float(fval),
                        })

                    # Skorokhod-like peak displacement for this song:
                    # D[j] = |tau_song[j] - tau_ref[j]|.
                    tau_song = peak_tau_all[song_i, :len(raw_real)]
                    amp_song = peak_amp_all[song_i, :len(raw_real)]
                    tau_ref_song = peak_tau_ref[:len(raw_real)]
                    skor_dist = peak_distance_all[song_i, :len(raw_real)]
                    skor_dist_samples = peak_distance_samples_all[song_i, :len(raw_real)]

                    # DC/hysteresis detector over the Skorokhod-like distance curve.
                    # The gate is drawn at 0/-0.1, below the main distance graph.
                    # It turns ON above median(D)+0.1 and turns OFF below median(D)-0.1.
                    skor_gate, skor_dc, skor_hi, skor_lo, skor_gate_state = dc_hysteresis_gate(
                        skor_dist,
                        band=0.10,
                    )

                    for seg_i in range(len(raw_real)):
                        real_density_rows.append({
                            "rank": int(out_rank),
                            "song_index": int(song_i + 1),
                            "song": strip_flag_emojis(str(self.display_names[song_i])),
                            "file_stem": str(self.names[song_i]),
                            "segment_index": int(seg_i),
                            "raw_shannon_fisher_density": float(raw_real[seg_i]),
                            "mp_reconstructed_K1_density": float(rec1_real[seg_i]),
                            f"mp_reconstructed_K{K_B}_density": float(rec6_real[seg_i]),
                            "fisher_coefficient_density": float(fisher_real[seg_i]),
                            "fisher_like_gradient_full_resolution": float(fisher_proxy[seg_i]),
                        })
                        peak_distance_rows.append({
                            "rank": int(out_rank),
                            "song_index": int(song_i + 1),
                            "song": strip_flag_emojis(str(self.display_names[song_i])),
                            "file_stem": str(self.names[song_i]),
                            "segment_index": int(seg_i),
                            "tau_song_peak_position_0_1": float(tau_song[seg_i]),
                            "tau_reference_K1_peak_position_0_1": float(tau_ref_song[seg_i]),
                            "skorokhod_peak_distance_0_1": float(skor_dist[seg_i]),
                            "skorokhod_peak_distance_samples": float(skor_dist_samples[seg_i]),
                            "skorokhod_dc_median": float(skor_dc),
                            "skorokhod_hysteresis_hi": float(skor_hi),
                            "skorokhod_hysteresis_lo": float(skor_lo),
                            "skorokhod_hysteresis_state": int(skor_gate_state[seg_i]),
                            "skorokhod_hysteresis_plot_value": float(skor_gate[seg_i]),
                            "song_peak_shannon_fisher_score": float(amp_song[seg_i]),
                            "reference_peak_shannon_fisher_score": float(peak_amp_ref[seg_i]),
                        })

                    fig_song = plt.figure(figsize=(13.5, 12.0), dpi=150)

                    gs_song = fig_song.add_gridspec(
                        3,
                        2,
                        height_ratios=[1.0, 0.72, 1.15],
                        width_ratios=[1.0, 1.0],
                        hspace=0.42,
                        wspace=0.30,
                    )

                    ax_t = fig_song.add_subplot(gs_song[0, :])
                    ax_skor = fig_song.add_subplot(gs_song[1, :])
                    ax_p1 = fig_song.add_subplot(gs_song[2, 0])
                    ax_p6 = fig_song.add_subplot(gs_song[2, 1])

                    song_name = strip_flag_emojis(str(self.display_names[song_i]).replace("｜", "|"))

                    fig_song.suptitle(
                        f"№{out_rank:02d} | song index={song_i + 1} | {song_name}",
                        fontsize=13,
                        fontweight="bold",
                        y=0.985,
                    )

                    # ----------------------------------------------------
                    # Top temporal graph: REAL segment density, no resampling.
                    # ----------------------------------------------------
                    xx = np.arange(len(raw_real))

                    ax_t.plot(
                        xx,
                        raw_real,
                        color="darkgreen",
                        linewidth=0.9,
                        label="original Shannon/Fisher density",
                    )

                    ax_t.plot(
                        xx,
                        rec1_real,
                        color="red",
                        linewidth=0.8,
                        alpha=0.9,
                        label="MP reconstructed K=1",
                    )

                    ax_t.plot(
                        xx,
                        rec6_real,
                        color="navy",
                        linewidth=0.8,
                        alpha=0.85,
                        linestyle="--",
                        label=f"MP reconstructed K={K_B}",
                    )

                    ax_t.plot(
                        xx,
                        fisher_real,
                        color="seagreen",
                        linewidth=0.55,
                        alpha=0.75,
                        linestyle=":",
                        label="Fisher-like |dX/dsegment| density",
                    )

                    ax_t.set_title(
                        "Real segment entropy/Fisher density: original vs MP reconstruction",
                        fontsize=10,
                    )
                    ax_t.set_xlabel("Real segment index from current analysis")
                    ax_t.set_ylabel("Normalized density")
                    if len(xx) > 0:
                        ax_t.set_xlim(0, max(1, len(xx) - 1))
                    ax_t.set_ylim(-0.05, 1.08)
                    ax_t.grid(True, alpha=0.22)
                    ax_t.legend(
                        loc="center left",
                        bbox_to_anchor=(1.01, 0.5),
                        fontsize=8,
                        borderaxespad=0.0,
                    )

                    # ----------------------------------------------------
                    # Middle graph: Skorokhod-like intra-segment peak displacement
                    # ----------------------------------------------------
                    ax_skor.plot(
                        xx,
                        skor_dist,
                        color="crimson",
                        linewidth=0.95,
                        label=r"$D_j=|\tau_{song,j}-\tau_{ref,j}|$",
                    )
                    ax_skor.plot(
                        xx,
                        tau_song,
                        color="0.30",
                        linewidth=0.55,
                        alpha=0.65,
                        linestyle=":",
                        label=r"song peak position $\tau_{song}$",
                    )
                    ax_skor.plot(
                        xx,
                        tau_ref_song,
                        color="navy",
                        linewidth=0.75,
                        alpha=0.72,
                        linestyle="--",
                        label=r"collective K=1 reference $\tau_{ref}$",
                    )

                    # DC detector with hysteresis around the distance curve.
                    # It is intentionally plotted below zero as a rectangular signal
                    # so it does not hide D[j], tau_song, or tau_ref.
                    ax_skor.axhline(
                        skor_dc,
                        color="black",
                        linewidth=0.7,
                        alpha=0.55,
                        label=f"DC median={skor_dc:.3f}",
                    )
                    ax_skor.axhline(
                        skor_hi,
                        color="black",
                        linewidth=0.5,
                        alpha=0.35,
                        linestyle="--",
                        label=f"hysteresis +0.10={skor_hi:.3f}",
                    )
                    ax_skor.axhline(
                        skor_lo,
                        color="black",
                        linewidth=0.5,
                        alpha=0.35,
                        linestyle=":",
                        label=f"hysteresis -0.10={skor_lo:.3f}",
                    )
                    ax_skor.step(
                        xx,
                        skor_gate,
                        where="post",
                        color="black",
                        linewidth=1.15,
                        alpha=0.92,
                        label="DC hysteresis gate 0/-0.1",
                    )
                    ax_skor.set_title(
                        "Skorokhod-like peak displacement: intra-segment Shannon/Fisher maximum vs collective K=1 reference",
                        fontsize=9,
                    )
                    ax_skor.set_xlabel("Real segment index")
                    ax_skor.set_ylabel("Peak distance / position inside segment")
                    if len(xx) > 0:
                        ax_skor.set_xlim(0, max(1, len(xx) - 1))
                    ax_skor.set_ylim(-0.13, 1.05)
                    ax_skor.grid(True, alpha=0.24)
                    ax_skor.legend(
                        loc="center left",
                        bbox_to_anchor=(1.01, 0.5),
                        fontsize=7,
                        borderaxespad=0.0,
                    )

                    # ----------------------------------------------------
                    # Bottom phase portraits: K=1 and K=6
                    # ----------------------------------------------------
                    _draw_phase(
                        ax_p1,
                        fig_song,
                        rec1,
                        "Phase portrait: MP coefficient c[n] vs c[n+1], K=1",
                        "K=1",
                    )

                    _draw_phase(
                        ax_p6,
                        fig_song,
                        rec6,
                        f"Phase portrait: MP coefficient c[n] vs c[n+1], K={K_B}",
                        f"K={K_B}",
                    )

                    fig_song.subplots_adjust(
                        left=0.08,
                        right=0.88,
                        top=0.92,
                        bottom=0.07,
                    )

                    pdf.savefig(fig_song, bbox_inches="tight")
                    plt.close(fig_song)

            plt.close(fig_catalog)
            plt.close(fig_heat)
            plt.close(fig_fisher_max)
            plt.close(fig_peak_ref)

            # ------------------------------------------------------------
            # CSV audit exports
            # ------------------------------------------------------------
            order_path = out_dir / f"MP_entropy_reconstructed_maps_K1_{max_k}_seg{seg}_song_order.csv"
            pd.DataFrame({
                "rank": np.arange(1, len(ranked_names) + 1),
                "song_index": rank + 1,
                "song": ranked_names,
                "mp_score": base_scores[rank],
            }).to_csv(order_path, index=False)

            real_density_path = out_dir / f"MP_entropy_REAL_segment_density_K1_vs_K{K_B}_seg{seg}.csv"
            pd.DataFrame(real_density_rows).to_csv(real_density_path, index=False)

            fisher_full_path = out_dir / f"MP_fisher_coefficients_full_resolution_seg{seg}.csv"
            pd.DataFrame(fisher_full_rows).to_csv(fisher_full_path, index=False)

            fisher_max_path = out_dir / f"MP_fisher_max_by_segment_seg{seg}.csv"
            pd.DataFrame({
                "segment_index": np.arange(fisher_max_by_segment.size),
                "fisher_max_across_songs": fisher_max_by_segment,
                "fisher_mean_across_songs": fisher_mean_by_segment,
                "argmax_song_index": fisher_argmax_song + 1,
                "argmax_song": [strip_flag_emojis(self.display_names[int(i)]) for i in fisher_argmax_song],
            }).to_csv(fisher_max_path, index=False)

            peak_ref_path = out_dir / f"MP_K1_collective_reference_peak_positions_seg{seg}.csv"
            pd.DataFrame({
                "segment_index": np.arange(peak_tau_ref.size),
                "tau_reference_K1_peak_position_0_1": peak_tau_ref,
                "reference_peak_shannon_fisher_score": peak_amp_ref,
            }).to_csv(peak_ref_path, index=False)

            peak_distance_path = out_dir / f"MP_skorokhod_peak_distance_by_song_seg{seg}.csv"
            pd.DataFrame(peak_distance_rows).to_csv(peak_distance_path, index=False)

            comparison_path = out_dir / f"MP_entropy_K1_vs_K{K_B}_per_song_summary_seg{seg}.csv"

            rows = []
            for out_rank, song_i in enumerate(rank, 1):
                raw = _norm01(X_entropy_raw[song_i, :])
                rec1 = _norm01(Xrec_k1[song_i, :])
                rec6 = _norm01(Xrec_k6[song_i, :])

                rows.append({
                    "rank": out_rank,
                    "song_index": int(song_i + 1),
                    "song": strip_flag_emojis(self.display_names[song_i]),
                    "file_stem": self.names[song_i],
                    "fisher_coeff_mean": float(np.mean(np.abs(np.gradient(raw)))),
                    "fisher_coeff_max": float(np.max(np.abs(np.gradient(raw)))),
                    "corr_raw_K1": float(np.corrcoef(raw, rec1)[0, 1]) if np.std(raw) > EPS and np.std(rec1) > EPS else 0.0,
                    "corr_raw_K6": float(np.corrcoef(raw, rec6)[0, 1]) if np.std(raw) > EPS and np.std(rec6) > EPS else 0.0,
                    "rmse_raw_K1": float(np.sqrt(np.mean((raw - rec1) ** 2))),
                    "rmse_raw_K6": float(np.sqrt(np.mean((raw - rec6) ** 2))),
                })

            pd.DataFrame(rows).to_csv(comparison_path, index=False)

            self.set_status(f"Saved PDF maps: {pdf_path}")

            messagebox.showinfo(
                "PDF exported",
                f"Saved PDF:\n{pdf_path}\n\n"
                f"Includes: catalog page, K1-16 heatmap atlas, max-Fisher segment page, and one per-song page with:\n"
                f"top: original Shannon/Fisher + K=1 + K={K_B}\n"
                f"bottom: phase portrait K=1 and K={K_B}\n\n"
                f"Song order CSV:\n{order_path}\n\n"
                f"Real segment-density CSV:\n{real_density_path}\n\n"
                f"Full-resolution Fisher coefficient CSV:\n{fisher_full_path}\n\n"
                f"Max Fisher-by-segment CSV:\n{fisher_max_path}\n\n"
                f"K=1 reference peak positions CSV:\n{peak_ref_path}\n\n"
                f"Skorokhod peak-distance CSV:\n{peak_distance_path}\n\n"
                f"Comparison CSV:\n{comparison_path}"
            )

        except Exception:
            print("\n" + "=" * 80, flush=True)
            print("ERROR IN PDF EXPORT", flush=True)
            print("=" * 80, flush=True)
            traceback.print_exc()
            print("=" * 80 + "\n", flush=True)
            messagebox.showerror("Error", "PDF export failed. See terminal traceback.")

    # --------------------------------------------------------

    def save_csv(self):
        if self.df_rank is None:
            messagebox.showerror("Error", "Run Analyze MP first.")
            return

        seg = int(self.seg_size.get())
        k = int(self.n_modes.get())

        self.df_rank.to_csv(f"mp_offline_{self.feature_mode}_{self.matrix_geometry}_{self.row_order_mode}_ranking_K{k}_seg{seg}.csv", index=False)
        self.df_eig.to_csv(f"mp_offline_{self.feature_mode}_{self.matrix_geometry}_{self.row_order_mode}_eigenvalues_K{k}_seg{seg}.csv", index=False)

        messagebox.showinfo("Saved", "CSV files saved.")

    # --------------------------------------------------------

    def on_close(self):
        try:
            self.X = None
            self.Xrec = None
            self.result = None
            self.df_rank = None
            self.df_eig = None

            if self.cbar is not None:
                self.cbar.remove()
                self.cbar = None

            self.fig.clear()
            self.canvas.get_tk_widget().destroy()

            gc.collect()

            if GPU_AVAILABLE:
                cp.get_default_memory_pool().free_all_blocks()

        except Exception:
            pass

        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass

        os._exit(0)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = MPOfflineRenderGUI(root)
    root.mainloop()
