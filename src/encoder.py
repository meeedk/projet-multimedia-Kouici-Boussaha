import logging
import os
from typing import List, Tuple

import numpy as np

from .config import (
    GOP_SIZE, QUANTISATION_FACTOR, CHROMA_SUBSAMPLING,
    SEARCH_WINDOW, MACROBLOCK_SIZE, COMPRESSED_BIN,
)
from .preprocessing import (
    bgr_to_ycbcr, split_ycbcr,
    chroma_subsample_420,
)
from .dct_codec    import encode_iframe
from .motion_estimation import encode_pframe
from .entropy_codec import write_bitstream

logger = logging.getLogger("mpeg4.encoder")


def encode_video(
    frames_bgr: List[np.ndarray],
    output_bin: str = COMPRESSED_BIN,
    gop_size: int = GOP_SIZE,
    qf: float = QUANTISATION_FACTOR,
    chroma_subsampling: bool = CHROMA_SUBSAMPLING,
    search_window: int = SEARCH_WINDOW,
    mb_size: int = MACROBLOCK_SIZE,
) -> Tuple[List[dict], List[str], int]:
    n = len(frames_bgr)
    logger.info(f"Encoding {n} frames  GOP={gop_size}  QF={qf}  "
                f"chroma_sub={chroma_subsampling}")

    frames_data: List[dict] = []
    frame_types: List[str]  = []

    
    prev_Y: np.ndarray | None = None
    prev_Cb: np.ndarray | None = None
    prev_Cr: np.ndarray | None = None

    for idx, bgr in enumerate(frames_bgr):
        is_iframe = (idx % gop_size == 0)
        logger.debug(f"  Frame {idx:04d}  {'I' if is_iframe else 'P'}")

        
        ycbcr = bgr_to_ycbcr(bgr)
        Y, Cb, Cr = split_ycbcr(ycbcr)

        if chroma_subsampling:
            Y, Cb, Cr = chroma_subsample_420(Y, Cb, Cr)

        
        if is_iframe or prev_Y is None:
            fd = encode_iframe(Y, Cb, Cr, qf=qf)
            frame_types.append("I")
        else:
            fd = encode_pframe(
                Y, Cb, Cr,
                prev_Y, prev_Cb, prev_Cr,
                qf=qf, mb_size=mb_size, search_window=search_window,
            )
            frame_types.append("P")

        frames_data.append(fd)

        
        if is_iframe or prev_Y is None:
            from .dct_codec import decode_iframe
            prev_Y, prev_Cb, prev_Cr = decode_iframe(fd)
        else:
            from .motion_estimation import decode_pframe
            prev_Y, prev_Cb, prev_Cr = decode_pframe(fd, prev_Y, prev_Cb, prev_Cr)

    
    os.makedirs(os.path.dirname(output_bin) or ".", exist_ok=True)
    bitstream_sz = write_bitstream(frames_data, output_bin)

    i_count = frame_types.count("I")
    p_count = frame_types.count("P")
    logger.info(f"Encoding complete: {i_count} I-frames, {p_count} P-frames")

    return frames_data, frame_types, bitstream_sz
