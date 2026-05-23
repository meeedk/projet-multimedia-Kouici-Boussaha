import logging
import os
from typing import List, Tuple

import numpy as np

from .entropy_codec    import read_bitstream
from .dct_codec        import decode_iframe
from .motion_estimation import decode_pframe
from .preprocessing    import (
    chroma_upsample_420, merge_ycbcr, ycbcr_to_bgr,
)
from .utils            import save_frame

logger = logging.getLogger("mpeg4.decoder")


def decode_video(
    input_bin: str,
    output_dir: str | None = None,
    chroma_subsampled: bool = True,
    orig_h: int | None = None,
    orig_w: int | None = None,
) -> List[np.ndarray]:
    logger.info(f"Decoding bitstream: {input_bin}")
    frames_data = read_bitstream(input_bin)
    n = len(frames_data)
    logger.info(f"  {n} frames to decode")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    bgr_frames: List[np.ndarray] = []
    prev_Y = prev_Cb = prev_Cr = None

    for idx, fd in enumerate(frames_data):
        ftype = fd["type"]
        logger.debug(f"  Frame {idx:04d}  {ftype}")

        if ftype == "I":
            Y, Cb, Cr = decode_iframe(fd)
            prev_Y, prev_Cb, prev_Cr = Y.copy(), Cb.copy(), Cr.copy()
        else:  
            assert prev_Y is not None, "P-frame with no reference!"
            Y, Cb, Cr = decode_pframe(fd, prev_Y, prev_Cb, prev_Cr)
            prev_Y, prev_Cb, prev_Cr = Y.copy(), Cb.copy(), Cr.copy()

        
        if chroma_subsampled and Cb.shape != Y.shape:
            Y, Cb, Cr = chroma_upsample_420(Y, Cb, Cr)

        
        ycbcr = merge_ycbcr(Y, Cb, Cr)
        bgr   = ycbcr_to_bgr(ycbcr)

        
        fh = orig_h or fd.get("orig_h", bgr.shape[0])
        fw = orig_w or fd.get("orig_w", bgr.shape[1])
        bgr = bgr[:fh, :fw]

        bgr_frames.append(bgr)

        if output_dir:
            path = os.path.join(output_dir, f"frame_{idx:04d}.png")
            save_frame(bgr, path)

    logger.info("Decoding complete.")
    return bgr_frames
