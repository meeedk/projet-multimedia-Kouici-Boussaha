import numpy as np
from typing import List, Tuple

from .config import (
    MACROBLOCK_SIZE,
    SEARCH_WINDOW,
    QUANTISATION_FACTOR,
)
from .dct_codec import (
    dct2d, idct2d,
    get_q_matrix, quantise, dequantise,
    split_into_blocks, merge_blocks, BLOCK_SIZE,
)
from .utils import zigzag_scan, inverse_zigzag, pad_to_multiple, crop




def sad(block_a: np.ndarray, block_b: np.ndarray) -> float:
    return float(np.sum(np.abs(block_a.astype(np.float32) - block_b.astype(np.float32))))




def estimate_motion(
    current_frame_Y: np.ndarray,
    reference_frame_Y: np.ndarray,
    mb_size: int = MACROBLOCK_SIZE,
    search_window: int = SEARCH_WINDOW,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    H, W = current_frame_Y.shape
    
    cur  = pad_to_multiple(current_frame_Y,   mb_size)
    ref  = pad_to_multiple(reference_frame_Y, mb_size)
    pH, pW = cur.shape

    n_mb_h = pH // mb_size
    n_mb_w = pW // mb_size
    motion_vectors = np.zeros((n_mb_h, n_mb_w, 2), dtype=np.int32)
    predicted       = np.zeros_like(cur, dtype=np.float32)

    for i in range(n_mb_h):
        for j in range(n_mb_w):
            
            r0, c0 = i * mb_size, j * mb_size
            cur_mb = cur[r0:r0 + mb_size, c0:c0 + mb_size]

            best_sad = np.inf
            best_dy, best_dx = 0, 0

            
            for dy in range(-search_window, search_window + 1):
                for dx in range(-search_window, search_window + 1):
                    ry = r0 + dy
                    rx = c0 + dx
                    if ry < 0 or rx < 0 or ry + mb_size > pH or rx + mb_size > pW:
                        continue
                    ref_mb = ref[ry:ry + mb_size, rx:rx + mb_size]
                    s = sad(cur_mb, ref_mb)
                    if s < best_sad:
                        best_sad = s
                        best_dy, best_dx = dy, dx

            motion_vectors[i, j] = (best_dy, best_dx)
            ry = r0 + best_dy
            rx = c0 + best_dx
            predicted[r0:r0 + mb_size, c0:c0 + mb_size] = \
                ref[ry:ry + mb_size, rx:rx + mb_size]

    residuals = (cur.astype(np.float32) - predicted).astype(np.float32)
    return motion_vectors, residuals, n_mb_h, n_mb_w




def encode_residuals(
    residuals: np.ndarray,
    qf: float = QUANTISATION_FACTOR,
) -> Tuple[List[np.ndarray], int, int]:
    q_matrix = get_q_matrix("luma", qf)
    blocks, n_rows, n_cols = split_into_blocks(residuals)
    coeff_blocks: List[np.ndarray] = []
    for blk in blocks:
        dct_blk = dct2d(blk)                
        q_blk   = quantise(dct_blk, q_matrix)
        coeff_blocks.append(zigzag_scan(q_blk))
    return coeff_blocks, n_rows, n_cols


def decode_residuals(
    coeff_blocks: List[np.ndarray],
    n_rows: int,
    n_cols: int,
    orig_h: int,
    orig_w: int,
    qf: float = QUANTISATION_FACTOR,
) -> np.ndarray:
    q_matrix = get_q_matrix("luma", qf)
    recon_blocks = np.zeros((len(coeff_blocks), BLOCK_SIZE, BLOCK_SIZE), dtype=np.float32)
    for i, vec in enumerate(coeff_blocks):
        q_blk   = inverse_zigzag(vec)
        dct_blk = dequantise(q_blk, q_matrix)
        recon_blocks[i] = idct2d(dct_blk)
    merged = merge_blocks(recon_blocks, n_rows, n_cols)
    return crop(merged, orig_h, orig_w)




def encode_pframe(
    current_Y: np.ndarray,
    current_Cb: np.ndarray,
    current_Cr: np.ndarray,
    ref_Y: np.ndarray,
    ref_Cb: np.ndarray,
    ref_Cr: np.ndarray,
    qf: float = QUANTISATION_FACTOR,
    mb_size: int = MACROBLOCK_SIZE,
    search_window: int = SEARCH_WINDOW,
) -> dict:
    orig_h, orig_w = current_Y.shape

    mv, res_Y, n_mb_h, n_mb_w = estimate_motion(
        current_Y, ref_Y, mb_size, search_window
    )

    
    pH = n_mb_h * mb_size
    pW = n_mb_w * mb_size
    cur_pad = pad_to_multiple(current_Y, mb_size)
    ref_pad = pad_to_multiple(ref_Y,     mb_size)

    
    predicted_Y = np.zeros((pH, pW), dtype=np.float32)
    for i in range(n_mb_h):
        for j in range(n_mb_w):
            r0, c0 = i * mb_size, j * mb_size
            dy, dx = int(mv[i, j, 0]), int(mv[i, j, 1])
            ry, rx = r0 + dy, c0 + dx
            ry = max(0, min(ry, pH - mb_size))
            rx = max(0, min(rx, pW - mb_size))
            predicted_Y[r0:r0+mb_size, c0:c0+mb_size] = \
                ref_pad[ry:ry+mb_size, rx:rx+mb_size]

    residuals_Y = cur_pad.astype(np.float32) - predicted_Y
    res_Y_coeffs, res_Y_rows, res_Y_cols = encode_residuals(residuals_Y, qf)

    
    from .dct_codec import encode_plane
    Cb_coeffs, Cb_rows, Cb_cols = encode_plane(current_Cb, "chroma", qf)
    Cr_coeffs, Cr_rows, Cr_cols = encode_plane(current_Cr, "chroma", qf)

    return {
        "type":        "P",
        "orig_h":      orig_h,
        "orig_w":      orig_w,
        "qf":          qf,
        "n_mb_h":      n_mb_h,
        "n_mb_w":      n_mb_w,
        "mb_size":     mb_size,
        "motion_vectors": mv,
        "res_Y_coeffs": res_Y_coeffs,
        "res_Y_rows":   res_Y_rows,
        "res_Y_cols":   res_Y_cols,
        "Cb_coeffs":   Cb_coeffs, "Cb_rows": Cb_rows, "Cb_cols": Cb_cols,
        "Cr_coeffs":   Cr_coeffs, "Cr_rows": Cr_rows, "Cr_cols": Cr_cols,
    }




def decode_pframe(
    data: dict,
    ref_Y: np.ndarray,
    ref_Cb: np.ndarray,
    ref_Cr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    qf      = data["qf"]
    orig_h  = data["orig_h"]
    orig_w  = data["orig_w"]
    mb_size = data["mb_size"]
    mv      = data["motion_vectors"]
    n_mb_h  = data["n_mb_h"]
    n_mb_w  = data["n_mb_w"]

    pH = n_mb_h * mb_size
    pW = n_mb_w * mb_size
    ref_pad = pad_to_multiple(ref_Y, mb_size)

    
    predicted_Y = np.zeros((pH, pW), dtype=np.float32)
    for i in range(n_mb_h):
        for j in range(n_mb_w):
            r0, c0 = i * mb_size, j * mb_size
            dy, dx = int(mv[i, j, 0]), int(mv[i, j, 1])
            ry, rx = r0 + dy, c0 + dx
            ry = max(0, min(ry, pH - mb_size))
            rx = max(0, min(rx, pW - mb_size))
            predicted_Y[r0:r0+mb_size, c0:c0+mb_size] = \
                ref_pad[ry:ry+mb_size, rx:rx+mb_size]

    
    res_Y = decode_residuals(
        data["res_Y_coeffs"], data["res_Y_rows"], data["res_Y_cols"],
        pH, pW, qf
    )
    recon_Y_full = np.clip(predicted_Y + res_Y, 0, 255)
    recon_Y      = crop(recon_Y_full, orig_h, orig_w)

    
    from .dct_codec import decode_plane
    chroma_h = ref_Cb.shape[0]
    chroma_w = ref_Cb.shape[1]
    Cb = decode_plane(data["Cb_coeffs"], data["Cb_rows"], data["Cb_cols"],
                      chroma_h, chroma_w, "chroma", qf)
    Cr = decode_plane(data["Cr_coeffs"], data["Cr_rows"], data["Cr_cols"],
                      chroma_h, chroma_w, "chroma", qf)

    return recon_Y, Cb, Cr
