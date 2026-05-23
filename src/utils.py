import logging
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2


def setup_logging(log_level: str = "INFO", log_file: str | None = None) -> logging.Logger:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("mpeg4")


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def load_frames(folder: str, max_frames: int = 0) -> List[np.ndarray]:
    exts = {".png", ".jpg", ".jpeg"}
    paths = sorted(
        p for p in Path(folder).iterdir()
        if p.suffix.lower() in exts
    )
    if max_frames > 0:
        paths = paths[:max_frames]

    frames: List[np.ndarray] = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            raise IOError(f"Cannot read image: {p}")
        frames.append(img)
    return frames


def save_frame(frame_bgr: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, frame_bgr)




def _build_zigzag_index(n: int = 8) -> np.ndarray:
    indices: List[Tuple[int, int]] = []
    for diag in range(2 * n - 1):
        if diag % 2 == 0:
            r = min(diag, n - 1)
            c = diag - r
            while r >= 0 and c < n:
                indices.append((r, c))
                r -= 1
                c += 1
        else:
            c = min(diag, n - 1)
            r = diag - c
            while c >= 0 and r < n:
                indices.append((r, c))
                r += 1
                c -= 1
    return np.array(indices)


_ZZ_IDX = _build_zigzag_index(8)               
_ZZ_ROW = _ZZ_IDX[:, 0]
_ZZ_COL = _ZZ_IDX[:, 1]


_IZZ = np.zeros((8, 8), dtype=np.int32)
for _k, (_r, _c) in enumerate(zip(_ZZ_ROW, _ZZ_COL)):
    _IZZ[_r, _c] = _k


def zigzag_scan(block: np.ndarray) -> np.ndarray:
    return block[_ZZ_ROW, _ZZ_COL].copy()


def inverse_zigzag(vec: np.ndarray) -> np.ndarray:
    block = np.zeros((8, 8), dtype=vec.dtype)
    block[_ZZ_ROW, _ZZ_COL] = vec
    return block


def pad_to_multiple(arr: np.ndarray, multiple: int) -> np.ndarray:
    h, w = arr.shape[:2]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if arr.ndim == 2:
        return np.pad(arr, ((0, ph), (0, pw)), mode="edge")
    return np.pad(arr, ((0, ph), (0, pw), (0, 0)), mode="edge")


def crop(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    return arr[:h, :w] if arr.ndim == 2 else arr[:h, :w, :]
