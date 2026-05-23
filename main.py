import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.config    import (
    INPUT_FRAMES_DIR, COMPRESSED_BIN, RECON_FRAMES_DIR,
    OUTPUTS_DIR, FIGURES_DIR, PLOTS_DIR, LOGS_DIR,
    GOP_SIZE, QUANTISATION_FACTOR, CHROMA_SUBSAMPLING,
    SEARCH_WINDOW, MACROBLOCK_SIZE,
)
from src.utils      import setup_logging, ensure_dirs, load_frames
from src.encoder    import encode_video
from src.decoder    import decode_video
from src.metrics    import compute_frame_metrics, aggregate_metrics
from src.entropy_codec import original_size_bytes, compressed_size_bytes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Simplified MPEG-4 Video Encoder/Decoder Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",  "-i", default=INPUT_FRAMES_DIR,
                   help="Folder containing input .png/.jpg frames")
    p.add_argument("--output", "-o", default=COMPRESSED_BIN,
                   help="Output .bin file path")
    p.add_argument("--recon-dir",    default=RECON_FRAMES_DIR,
                   help="Directory for reconstructed frames")
    p.add_argument("--qf",           type=float, default=QUANTISATION_FACTOR,
                   help="Quantisation factor")
    p.add_argument("--gop",          type=int,   default=GOP_SIZE,
                   help="GOP size (every N-th frame is an I-frame)")
    p.add_argument("--search",       type=int,   default=SEARCH_WINDOW,
                   help="Motion estimation search window (±pixels)")
    p.add_argument("--no-chroma-sub", action="store_true",
                   help="Disable 4:2:0 chroma subsampling")
    p.add_argument("--max-frames",   type=int,   default=0,
                   help="Limit number of input frames (0 = all)")
    p.add_argument("--evaluate",     action="store_true",
                   help="Run QF and GOP sweeps and save evaluation plots")
    p.add_argument("--decode",       metavar="BIN",
                   help="Decode-only mode: path to existing .bin file")
    p.add_argument("--visualise",    action="store_true",
                   help="Generate pipeline visualisation figures")
    p.add_argument("--log-level",    default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def run_encode_decode(args) -> None:
    """Full encode + decode pass."""
    ensure_dirs(OUTPUTS_DIR, RECON_FRAMES_DIR, FIGURES_DIR, PLOTS_DIR, LOGS_DIR)

    logger = setup_logging(args.log_level, LOGS_DIR + "/encoder.log")
    logger.info("=== MPEG-4 Encoder Pipeline ===")

    
    logger.info(f"Loading frames from: {args.input}")
    frames_bgr = load_frames(args.input, max_frames=args.max_frames)
    if not frames_bgr:
        logger.error("No frames found. Aborting.")
        sys.exit(1)
    logger.info(f"Loaded {len(frames_bgr)} frames  "
                f"({frames_bgr[0].shape[1]}×{frames_bgr[0].shape[0]})")

    chroma_sub = not args.no_chroma_sub

    
    frames_data, frame_types, bin_sz = encode_video(
        frames_bgr,
        output_bin          = args.output,
        gop_size            = args.gop,
        qf                  = args.qf,
        chroma_subsampling  = chroma_sub,
        search_window       = args.search,
        mb_size             = MACROBLOCK_SIZE,
    )

    
    orig_h, orig_w = frames_bgr[0].shape[:2]
    recon_frames = decode_video(
        args.output,
        output_dir         = args.recon_dir,
        chroma_subsampled  = chroma_sub,
        orig_h             = orig_h,
        orig_w             = orig_w,
    )

    
    import cv2
    orig_resized = [cv2.resize(f, (orig_w, orig_h)) for f in frames_bgr]
    frame_metrics = compute_frame_metrics(orig_resized, recon_frames)
    orig_sz = original_size_bytes(frames_bgr)
    comp_sz = compressed_size_bytes(args.output)
    summary = aggregate_metrics(frame_metrics, frame_types, orig_sz, comp_sz)

    print("\n" + "="*50)
    print(f"  Frames        : {len(frames_bgr)}")
    print(f"  I-frames      : {summary['n_iframes']}")
    print(f"  P-frames      : {summary['n_pframes']}")
    print(f"  Avg PSNR      : {summary['avg_psnr']:.2f} dB")
    print(f"  Avg MSE       : {summary['avg_mse']:.2f}")
    print(f"  Orig size     : {orig_sz / 1024:.1f} KB")
    print(f"  Compressed    : {comp_sz / 1024:.1f} KB")
    print(f"  Comp. ratio   : {summary['compression_ratio']:.2f}×")
    print("="*50 + "\n")

    
    if args.evaluate:
        _run_sweeps(frames_bgr, args, chroma_sub, logger)

    
    if args.visualise:
        _run_visualisation(frames_bgr, recon_frames, frames_data, frame_types,
                           frame_metrics, chroma_sub, logger)


def _run_sweeps(frames_bgr, args, chroma_sub, logger):
    import tempfile, shutil
    from src.encoder       import encode_video
    from src.decoder       import decode_video
    from src.metrics       import compute_frame_metrics, aggregate_metrics
    from src.entropy_codec import original_size_bytes, compressed_size_bytes
    from src.visualisation import plot_qf_sweep, plot_gop_sweep
    import cv2

    orig_h, orig_w = frames_bgr[0].shape[:2]
    orig_sz = original_size_bytes(frames_bgr)
    tmp_dir = tempfile.mkdtemp()

    def _eval(qf=args.qf, gop=args.gop):
        bin_path = os.path.join(tmp_dir, "tmp.bin")
        _, ftypes, _ = encode_video(
            frames_bgr, bin_path, gop_size=gop, qf=qf,
            chroma_subsampling=chroma_sub, search_window=args.search
        )
        recon = decode_video(bin_path, chroma_subsampled=chroma_sub,
                             orig_h=orig_h, orig_w=orig_w)
        fm = compute_frame_metrics(frames_bgr, recon)
        sm = aggregate_metrics(fm, ftypes, orig_sz,
                               compressed_size_bytes(bin_path))
        return sm["avg_psnr"], sm["compression_ratio"]

    
    logger.info("Running QF sweep …")
    qf_vals  = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    p_qf, c_qf = [], []
    for qf in qf_vals:
        p, c = _eval(qf=qf)
        p_qf.append(p); c_qf.append(c)

    plot_qf_sweep(qf_vals, p_qf, c_qf,
                  os.path.join(PLOTS_DIR, "qf_sweep.png"))

    
    logger.info("Running GOP sweep …")
    gop_vals  = [1, 2, 4, 8, 16]
    p_gop, c_gop = [], []
    for gop in gop_vals:
        p, c = _eval(gop=gop)
        p_gop.append(p); c_gop.append(c)

    plot_gop_sweep(gop_vals, p_gop, c_gop,
                   os.path.join(PLOTS_DIR, "gop_sweep.png"))

    shutil.rmtree(tmp_dir)
    logger.info("Evaluation sweeps complete.")


def _run_visualisation(frames_bgr, recon_frames, frames_data, frame_types,
                       frame_metrics, chroma_sub, logger):
    from src.visualisation import (
        plot_ycbcr_channels, plot_dct_block, plot_motion_vectors,
        plot_residuals, plot_pipeline_overview, plot_psnr_per_frame,
    )
    from src.preprocessing import bgr_to_ycbcr, split_ycbcr
    from src.dct_codec     import dct2d, get_q_matrix, quantise, idct2d
    from src.motion_estimation import estimate_motion, pad_to_multiple
    import cv2, numpy as np

    logger.info("Generating visualisations …")
    frame0 = frames_bgr[0]
    recon0 = recon_frames[0]

    
    plot_ycbcr_channels(frame0, os.path.join(FIGURES_DIR, "ycbcr_channels.png"))

    
    ycbcr = bgr_to_ycbcr(frame0)
    Y0    = ycbcr[..., 0]
    h, w  = Y0.shape
    br, bc = h // 2 // 8 * 8, w // 2 // 8 * 8
    raw_blk = Y0[br:br+8, bc:bc+8]
    qm      = get_q_matrix("luma")
    dct_blk = dct2d(raw_blk - 128)
    q_blk   = quantise(dct_blk, qm).astype(np.float32)
    rec_blk = np.clip(idct2d(q_blk * qm) + 128, 0, 255)
    plot_dct_block(raw_blk, dct_blk, q_blk, rec_blk,
                   os.path.join(FIGURES_DIR, "dct_block.png"))

    
    p_idx = next((i for i, t in enumerate(frame_types) if t == "P"), None)

    if p_idx is not None:
        ref_ycbcr = bgr_to_ycbcr(frames_bgr[p_idx - 1])
        cur_ycbcr = bgr_to_ycbcr(frames_bgr[p_idx])
        ref_Y_full = ref_ycbcr[..., 0]
        cur_Y_full = cur_ycbcr[..., 0]
        mv, residuals, _, _ = estimate_motion(cur_Y_full, ref_Y_full)

        plot_motion_vectors(frames_bgr[p_idx], mv,
                            save_path=os.path.join(FIGURES_DIR, "motion_vectors.png"))
        plot_residuals(residuals[:frame0.shape[0], :frame0.shape[1]],
                       recon_frames[p_idx],
                       save_path=os.path.join(FIGURES_DIR, "residuals.png"))

        
        Y_s, Cb_s, Cr_s = split_ycbcr(ycbcr)
        plot_pipeline_overview(
            orig_bgr     = frame0,
            Y=Y_s, Cb=Cb_s, Cr=Cr_s,
            block_raw    = raw_blk,
            block_dct    = dct_blk,
            block_q      = q_blk,
            block_recon  = rec_blk,
            residuals    = residuals[:frame0.shape[0], :frame0.shape[1]],
            recon_bgr    = recon0,
            motion_vectors = mv,
            mb_size      = 16,
            save_path    = os.path.join(FIGURES_DIR, "pipeline_overview.png"),
        )

    plot_psnr_per_frame(frame_metrics, frame_types,
                        os.path.join(PLOTS_DIR, "psnr_per_frame.png"))

    logger.info("Visualisations saved.")


def main() -> None:
    args = parse_args()

    if args.decode:
        
        ensure_dirs(RECON_FRAMES_DIR, LOGS_DIR)
        log = setup_logging(args.log_level, LOGS_DIR + "/decoder.log")
        recon = decode_video(args.decode, output_dir=args.recon_dir)
        log.info(f"Decoded {len(recon)} frames → {args.recon_dir}")
    else:
        run_encode_decode(args)


if __name__ == "__main__":
    main()
