import numpy as np
from scipy.fft import dctn, idctn
from typing import List, Tuple

from .config import (
    JPEG_LUMA_Q_MATRIX,
    JPEG_CHROMA_Q_MATRIX,
    QUANTISATION_FACTOR,
    BLOCK_SIZE,
)
from .utils import zigzag_scan, inverse_zigzag, pad_to_multiple, crop




def dct2d(block: np.ndarray) -> np.ndarray:
    return dctn(block.astype(np.float32), norm="ortho", type=2)


def idct2d(block: np.ndarray) -> np.ndarray:
    return idctn(block.astype(np.float32), norm="ortho", type=2)




def get_q_matrix(channel: str = "luma", qf: float = QUANTISATION_FACTOR) -> np.ndarray:
    base = JPEG_LUMA_Q_MATRIX if channel == "luma" else JPEG_CHROMA_Q_MATRIX
    return np.clip(base * qf, 1, None).astype(np.float32)


def quantise(dct_block: np.ndarray, q_matrix: np.ndarray) -> np.ndarray:
    return np.round(dct_block / q_matrix).astype(np.int16)


def dequantise(q_block: np.ndarray, q_matrix: np.ndarray) -> np.ndarray:
    return (q_block.astype(np.float32) * q_matrix)




def split_into_blocks(plane: np.ndarray, block_size: int = BLOCK_SIZE) -> Tuple[np.ndarray, int, int]:
    padded = pad_to_multiple(plane, block_size)
    h, w   = padded.shape
    n_rows = h // block_size
    n_cols = w // block_size
    blocks = (
        padded
        .reshape(n_rows, block_size, n_cols, block_size)
        .transpose(0, 2, 1, 3)
        .reshape(-1, block_size, block_size)
        .astype(np.float32)
    )
    return blocks, n_rows, n_cols


def merge_blocks(
    blocks: np.ndarray, n_rows: int, n_cols: int, block_size: int = BLOCK_SIZE
) -> np.ndarray:
    return (
        blocks
        .reshape(n_rows, n_cols, block_size, block_size)
        .transpose(0, 2, 1, 3)
        .reshape(n_rows * block_size, n_cols * block_size)
    )




def encode_plane(
    plane: np.ndarray,
    channel: str = "luma",
    qf: float = QUANTISATION_FACTOR,
) -> Tuple[List[np.ndarray], int, int]:
    q_matrix = get_q_matrix(channel, qf)
    blocks, n_rows, n_cols = split_into_blocks(plane)
    coeff_blocks: List[np.ndarray] = []
    for blk in blocks:
        blk_shifted = blk - 128.0           
        dct_blk     = dct2d(blk_shifted)
        q_blk       = quantise(dct_blk, q_matrix)
        coeff_blocks.append(zigzag_scan(q_blk))
    return coeff_blocks, n_rows, n_cols


def decode_plane(
    coeff_blocks: List[np.ndarray],
    n_rows: int,
    n_cols: int,
    orig_h: int,
    orig_w: int,
    channel: str = "luma",
    qf: float = QUANTISATION_FACTOR,
) -> np.ndarray:
    q_matrix = get_q_matrix(channel, qf)
    recon_blocks = np.zeros((len(coeff_blocks), BLOCK_SIZE, BLOCK_SIZE), dtype=np.float32)
    for i, vec in enumerate(coeff_blocks):
        q_blk       = inverse_zigzag(vec)
        dct_blk     = dequantise(q_blk, q_matrix)
        blk         = idct2d(dct_blk) + 128.0
        recon_blocks[i] = np.clip(blk, 0, 255)
    merged = merge_blocks(recon_blocks, n_rows, n_cols)
    return crop(merged, orig_h, orig_w)




def encode_iframe(
    Y: np.ndarray,
    Cb: np.ndarray,
    Cr: np.ndarray,
    qf: float = QUANTISATION_FACTOR,
) -> dict:
    orig_h, orig_w = Y.shape

    Y_coeffs,  Y_rows,  Y_cols  = encode_plane(Y,  "luma",   qf)
    Cb_coeffs, Cb_rows, Cb_cols = encode_plane(Cb, "chroma", qf)
    Cr_coeffs, Cr_rows, Cr_cols = encode_plane(Cr, "chroma", qf)

    return {
        "type":    "I",
        "orig_h":  orig_h,
        "orig_w":  orig_w,
        "qf":      qf,
        "Y_coeffs":  Y_coeffs,  "Y_rows":  Y_rows,  "Y_cols":  Y_cols,
        "Cb_coeffs": Cb_coeffs, "Cb_rows": Cb_rows, "Cb_cols": Cb_cols,
        "Cr_coeffs": Cr_coeffs, "Cr_rows": Cr_rows, "Cr_cols": Cr_cols,
    }


def decode_iframe(data: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    qf     = data["qf"]
    orig_h = data["orig_h"]
    orig_w = data["orig_w"]

    Y  = decode_plane(data["Y_coeffs"],  data["Y_rows"],  data["Y_cols"],
                      orig_h, orig_w, "luma",   qf)
    Cb = decode_plane(data["Cb_coeffs"], data["Cb_rows"], data["Cb_cols"],
                      orig_h, orig_w // 2 if data["Cb_cols"] < data["Y_cols"] else orig_w,
                      "chroma", qf)
    Cr = decode_plane(data["Cr_coeffs"], data["Cr_rows"], data["Cr_cols"],
                      orig_h, orig_w // 2 if data["Cr_cols"] < data["Y_cols"] else orig_w,
                      "chroma", qf)
    return Y, Cb, Cr
