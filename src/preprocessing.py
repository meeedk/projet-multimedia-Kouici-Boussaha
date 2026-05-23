import numpy as np
import cv2
from typing import Tuple




def bgr_to_ycbcr(frame_bgr: np.ndarray) -> np.ndarray:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    R, G, B = frame_rgb[..., 0], frame_rgb[..., 1], frame_rgb[..., 2]

    Y  =  16.0 +  65.481*R/255 + 128.553*G/255 +  24.966*B/255
    Cb = 128.0 -  37.797*R/255 -  74.203*G/255 + 112.000*B/255
    Cr = 128.0 + 112.000*R/255 -  93.786*G/255 -  18.214*B/255

    return np.stack([Y, Cb, Cr], axis=-1).astype(np.float32)


def ycbcr_to_bgr(frame_ycbcr: np.ndarray) -> np.ndarray:
    Y  = frame_ycbcr[..., 0]
    Cb = frame_ycbcr[..., 1]
    Cr = frame_ycbcr[..., 2]

    
    Y_  = Y  - 16.0
    Cb_ = Cb - 128.0
    Cr_ = Cr - 128.0

    R = 255 * (0.00456621 * Y_                    + 0.00625893 * Cr_)
    G = 255 * (0.00456621 * Y_ - 0.00153632 * Cb_ - 0.00318811 * Cr_)
    B = 255 * (0.00456621 * Y_ + 0.00791071 * Cb_)

    rgb = np.clip(np.stack([R, G, B], axis=-1), 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)




def chroma_subsample_420(
    Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = Y.shape
    Cb_sub = cv2.resize(Cb, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    Cr_sub = cv2.resize(Cr, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    return Y, Cb_sub, Cr_sub


def chroma_upsample_420(
    Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = Y.shape
    Cb_up = cv2.resize(Cb, (w, h), interpolation=cv2.INTER_LINEAR)
    Cr_up = cv2.resize(Cr, (w, h), interpolation=cv2.INTER_LINEAR)
    return Y, Cb_up, Cr_up


def split_ycbcr(frame_ycbcr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return frame_ycbcr[..., 0], frame_ycbcr[..., 1], frame_ycbcr[..., 2]


def merge_ycbcr(
    Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray
) -> np.ndarray:
    return np.stack([Y, Cb, Cr], axis=-1).astype(np.float32)
