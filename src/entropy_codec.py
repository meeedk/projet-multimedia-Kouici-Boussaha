import io
import os
import pickle
import struct
import zlib
import logging
from typing import List, Any

import numpy as np

logger = logging.getLogger("mpeg4.entropy")

MAGIC = b"MP4S"
FRAME_TYPE_I = 0
FRAME_TYPE_P = 1




def _serialise_frame(frame_data: dict) -> bytes:
    raw = pickle.dumps(frame_data, protocol=4)
    return zlib.compress(raw, level=9)


def _deserialise_frame(payload: bytes) -> dict:
    raw = zlib.decompress(payload)
    return pickle.loads(raw)




def write_bitstream(frames_data: List[dict], output_path: str) -> int:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    n = len(frames_data)
    logger.info(f"Writing bitstream: {n} frames → {output_path}")

    with open(output_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", n))

        for fd in frames_data:
            ftype = FRAME_TYPE_I if fd["type"] == "I" else FRAME_TYPE_P
            payload = _serialise_frame(fd)
            f.write(struct.pack("<B", ftype))
            f.write(struct.pack("<I", len(payload)))
            f.write(payload)

    size = os.path.getsize(output_path)
    logger.info(f"Bitstream written: {size / 1024:.1f} KB")
    return size


def read_bitstream(input_path: str) -> List[dict]:
    logger.info(f"Reading bitstream: {input_path}")
    with open(input_path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(f"Invalid bitstream magic: {magic!r}")

        n = struct.unpack("<I", f.read(4))[0]
        logger.info(f"  {n} frames in bitstream")

        frames: List[dict] = []
        for i in range(n):
            ftype   = struct.unpack("<B", f.read(1))[0]
            pay_len = struct.unpack("<I", f.read(4))[0]
            payload = f.read(pay_len)
            fd = _deserialise_frame(payload)
            frames.append(fd)

    return frames




def original_size_bytes(frames_bgr: list) -> int:
    return sum(f.nbytes for f in frames_bgr)


def compressed_size_bytes(bin_path: str) -> int:
    return os.path.getsize(bin_path)
