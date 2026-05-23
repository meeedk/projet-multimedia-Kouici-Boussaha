import os
import sys
import logging
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))

from src.utils import setup_logging, ensure_dirs
from src.config import (
    INPUT_FRAMES_DIR, COMPRESSED_BIN, RECON_FRAMES_DIR,
    OUTPUTS_DIR, FIGURES_DIR, PLOTS_DIR, LOGS_DIR,
)




def make_synthetic_frames(n: int = 16, height: int = 128, width: int = 160) -> list:
    frames = []
    for i in range(n):
        
        t = i / max(n - 1, 1)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[..., 0] = np.linspace(int(50 * t), int(200 * t),  width, dtype=np.uint8)
        frame[..., 1] = np.linspace(int(100),     int(200),       width, dtype=np.uint8)
        frame[..., 2] = np.linspace(int(200 * (1-t)), int(50), width, dtype=np.uint8)

        
        cx = int(width  * 0.1 + width  * 0.8 * t)
        cy = height // 2
        cv2.circle(frame, (cx, cy), 12, (255, 80, 20), -1)

        frames.append(frame)
    return frames




def main():
    ensure_dirs(INPUT_FRAMES_DIR, OUTPUTS_DIR, RECON_FRAMES_DIR,
                FIGURES_DIR, PLOTS_DIR, LOGS_DIR)
    log = setup_logging("INFO", os.path.join(LOGS_DIR, "demo.log"))
    log.info("=== MPEG-4 Demo ===")

    
    log.info("Generating synthetic frames …")
    frames_bgr = make_synthetic_frames(n=16)
    for idx, f in enumerate(frames_bgr):
        cv2.imwrite(os.path.join(INPUT_FRAMES_DIR, f"frame_{idx:04d}.png"), f)
    log.info(f"  {len(frames_bgr)} frames written to {INPUT_FRAMES_DIR}")

    
    from src.encoder import encode_video
    log.info("Encoding …")
    frames_data, frame_types, bin_sz = encode_video(
        frames_bgr,
        output_bin         = COMPRESSED_BIN,
        gop_size           = 4,
        qf                 = 2.0,
        chroma_subsampling = True,
    )

    
    from src.decoder import decode_video
    log.info("Decoding …")
    orig_h, orig_w = frames_bgr[0].shape[:2]
    recon_frames = decode_video(
        COMPRESSED_BIN,
        output_dir        = RECON_FRAMES_DIR,
        chroma_subsampled = True,
        orig_h            = orig_h,
        orig_w            = orig_w,
    )

    
    from src.metrics       import compute_frame_metrics, aggregate_metrics
    from src.entropy_codec import original_size_bytes, compressed_size_bytes

    frame_metrics = compute_frame_metrics(frames_bgr, recon_frames)
    orig_sz = original_size_bytes(frames_bgr)
    comp_sz = compressed_size_bytes(COMPRESSED_BIN)
    summary = aggregate_metrics(frame_metrics, frame_types, orig_sz, comp_sz)

    print("\n" + "="*52)
    print("  Demo Results")
    print("="*52)
    print(f"  Frames            : {len(frames_bgr)}")
    print(f"  I-frames / P-frames: {summary['n_iframes']} / {summary['n_pframes']}")
    print(f"  Avg PSNR          : {summary['avg_psnr']:.2f} dB")
    print(f"  Avg MSE           : {summary['avg_mse']:.2f}")
    print(f"  Original size     : {orig_sz / 1024:.1f} KB")
    print(f"  Compressed size   : {comp_sz / 1024:.1f} KB")
    print(f"  Compression ratio : {summary['compression_ratio']:.2f}×")
    print("="*52 + "\n")

    
    log.info("Generating figures …")

    from src.visualisation import (
        plot_ycbcr_channels, plot_dct_block, plot_motion_vectors,
        plot_residuals, plot_pipeline_overview, plot_psnr_per_frame,
        plot_qf_sweep, plot_gop_sweep,
    )
    from src.preprocessing import bgr_to_ycbcr, split_ycbcr
    from src.dct_codec     import dct2d, get_q_matrix, quantise, idct2d
    from src.motion_estimation import estimate_motion

    frame0 = frames_bgr[0]
    recon0 = recon_frames[0]

    
    plot_ycbcr_channels(frame0, os.path.join(FIGURES_DIR, "ycbcr_channels.png"))

    
    Y0 = bgr_to_ycbcr(frame0)[..., 0]
    h, w = Y0.shape
    br, bc = (h // 2 // 8) * 8, (w // 2 // 8) * 8
    raw_blk = Y0[br:br+8, bc:bc+8]
    qm = get_q_matrix("luma")
    dct_blk = dct2d(raw_blk - 128)
    q_blk   = quantise(dct_blk, qm).astype(np.float32)
    rec_blk = np.clip(idct2d(q_blk * qm) + 128, 0, 255)
    plot_dct_block(raw_blk, dct_blk, q_blk, rec_blk,
                   os.path.join(FIGURES_DIR, "dct_block.png"))

    
    ref_Y = bgr_to_ycbcr(frames_bgr[0])[..., 0]
    cur_Y = bgr_to_ycbcr(frames_bgr[1])[..., 0]
    mv, residuals, _, _ = estimate_motion(cur_Y, ref_Y)

    plot_motion_vectors(frames_bgr[1], mv,
                        save_path=os.path.join(FIGURES_DIR, "motion_vectors.png"))
    plot_residuals(residuals[:orig_h, :orig_w], recon_frames[1],
                   save_path=os.path.join(FIGURES_DIR, "residuals.png"))

    
    Y_s, Cb_s, Cr_s = split_ycbcr(bgr_to_ycbcr(frame0))
    plot_pipeline_overview(
        orig_bgr=frame0, Y=Y_s, Cb=Cb_s, Cr=Cr_s,
        block_raw=raw_blk, block_dct=dct_blk,
        block_q=q_blk, block_recon=rec_blk,
        residuals=residuals[:orig_h, :orig_w],
        recon_bgr=recon0, motion_vectors=mv, mb_size=16,
        save_path=os.path.join(FIGURES_DIR, "pipeline_overview.png"),
    )

    
    plot_psnr_per_frame(frame_metrics, frame_types,
                        os.path.join(PLOTS_DIR, "psnr_per_frame.png"))

    
    log.info("QF sweep …")
    import tempfile, shutil
    from src.encoder       import encode_video as enc
    from src.decoder       import decode_video as dec_v
    from src.metrics       import compute_frame_metrics as cfm, aggregate_metrics as am
    from src.entropy_codec import original_size_bytes as osb, compressed_size_bytes as csb

    def quick_eval(qf=2.0, gop=4):
        td = tempfile.mkdtemp()
        bp = os.path.join(td, "t.bin")
        _, ft, _ = enc(frames_bgr, bp, gop_size=gop, qf=qf, chroma_subsampling=True)
        rf = dec_v(bp, chroma_subsampled=True, orig_h=orig_h, orig_w=orig_w)
        sm = am(cfm(frames_bgr, rf), ft, osb(frames_bgr), csb(bp))
        shutil.rmtree(td)
        return sm["avg_psnr"], sm["compression_ratio"]

    qf_vals = [0.5, 1.0, 2.0, 4.0, 8.0]
    p_qf, c_qf = zip(*[quick_eval(qf=q) for q in qf_vals])
    plot_qf_sweep(list(qf_vals), list(p_qf), list(c_qf),
                  os.path.join(PLOTS_DIR, "qf_sweep.png"))

    gop_vals = [1, 2, 4, 8]
    p_gop, c_gop = zip(*[quick_eval(gop=g) for g in gop_vals])
    plot_gop_sweep(list(gop_vals), list(p_gop), list(c_gop),
                   os.path.join(PLOTS_DIR, "gop_sweep.png"))

    log.info("Demo complete. All figures saved.")
    print(f"\nFigures saved to : {FIGURES_DIR}")
    print(f"Plots saved to   : {PLOTS_DIR}")
    print(f"Recon frames     : {RECON_FRAMES_DIR}")
    print(f"Compressed bin   : {COMPRESSED_BIN}\n")


if __name__ == "__main__":
    main()
