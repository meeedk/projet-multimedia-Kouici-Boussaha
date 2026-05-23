import math
import logging
from typing import List, Tuple, Dict

import numpy as np

logger = logging.getLogger("mpeg4.metrics")


def mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    o = original.astype(np.float64)
    r = reconstructed.astype(np.float64)
    return float(np.mean((o - r) ** 2))


def psnr(original: np.ndarray, reconstructed: np.ndarray, max_val: float = 255.0) -> float:
    err = mse(original, reconstructed)
    if err == 0.0:
        return float("inf")
    return 10.0 * math.log10((max_val ** 2) / err)


def compute_frame_metrics(
    originals: List[np.ndarray],
    reconstructed: List[np.ndarray],
) -> List[Dict[str, float]]:
    results = []
    for i, (orig, recon) in enumerate(zip(originals, reconstructed)):
        e = mse(orig, recon)
        p = psnr(orig, recon)
        results.append({"frame": i, "mse": e, "psnr": p})
    return results


def compression_ratio(original_bytes: int, compressed_bytes: int) -> float:
    if compressed_bytes == 0:
        return float("inf")
    return original_bytes / compressed_bytes


def aggregate_metrics(
    frame_metrics: List[Dict],
    frame_types: List[str],
    orig_bytes: int,
    comp_bytes: int,
) -> Dict:
    avg_psnr = float(np.mean([m["psnr"] for m in frame_metrics
                               if not math.isinf(m["psnr"])]))
    avg_mse  = float(np.mean([m["mse"]  for m in frame_metrics]))
    cr       = compression_ratio(orig_bytes, comp_bytes)
    n_i = frame_types.count("I")
    n_p = frame_types.count("P")

    summary = {
        "avg_psnr":         avg_psnr,
        "avg_mse":          avg_mse,
        "compression_ratio": cr,
        "n_iframes":        n_i,
        "n_pframes":        n_p,
        "frame_metrics":    frame_metrics,
    }

    logger.info(
        f"Metrics | PSNR={avg_psnr:.2f} dB  MSE={avg_mse:.2f}  "
        f"CR={cr:.2f}×  I={n_i}  P={n_p}"
    )
    return summary


def sweep_qf(
    encode_fn,
    qf_values: List[float],
) -> Tuple[List[float], List[float], List[float]]:
    psnr_vals, cr_vals = [], []
    for qf in qf_values:
        p, c = encode_fn(qf)
        psnr_vals.append(p)
        cr_vals.append(c)
        logger.info(f"  QF={qf:.1f}  PSNR={p:.2f} dB  CR={c:.2f}×")
    return qf_values, psnr_vals, cr_vals


def sweep_gop(
    encode_fn,
    gop_values: List[int],
) -> Tuple[List[int], List[float], List[float]]:
    psnr_vals, cr_vals = [], []
    for gop in gop_values:
        p, c = encode_fn(gop)
        psnr_vals.append(p)
        cr_vals.append(c)
        logger.info(f"  GOP={gop}  PSNR={p:.2f} dB  CR={c:.2f}×")
    return gop_values, psnr_vals, cr_vals
