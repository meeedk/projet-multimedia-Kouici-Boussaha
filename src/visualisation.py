import os
import logging
import warnings
from typing import List, Dict, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2

logger = logging.getLogger("mpeg4.visualisation")

CMAP_DCT    = "RdBu_r"
CMAP_GREY   = "gray"
FIG_DPI     = 120
TITLE_FS    = 10
LABEL_FS    = 8


def _save(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {path}")


def _show_img(ax: plt.Axes, img: np.ndarray, title: str,
              cmap=None, vmin=None, vmax=None) -> None:
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=TITLE_FS)
    ax.axis("off")



def plot_ycbcr_channels(
    frame_bgr: np.ndarray,
    save_path: str,
) -> None:
    from .preprocessing import bgr_to_ycbcr, split_ycbcr
    ycbcr  = bgr_to_ycbcr(frame_bgr)
    Y, Cb, Cr = split_ycbcr(ycbcr)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    fig.suptitle("Color Space — YCbCr Decomposition", fontsize=12, fontweight="bold")

    _show_img(axes[0], cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), "Original (RGB)")
    _show_img(axes[1], Y,  "Y (Luma)",    cmap=CMAP_GREY, vmin=16, vmax=235)
    _show_img(axes[2], Cb, "Cb (Chroma)", cmap="coolwarm", vmin=16, vmax=240)
    _show_img(axes[3], Cr, "Cr (Chroma)", cmap="coolwarm", vmin=16, vmax=240)

    for ax in axes[1:]:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        im  = ax.get_images()[0]
        fig.colorbar(im, cax=cax)

    plt.tight_layout()
    _save(fig, save_path)



def plot_dct_block(
    block_pixels: np.ndarray,
    dct_coeffs: np.ndarray,
    q_coeffs: np.ndarray,
    recon_block: np.ndarray,
    save_path: str,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    fig.suptitle("DCT Block Processing (8×8)", fontsize=12, fontweight="bold")

    _show_img(axes[0], block_pixels, "Raw pixels",   cmap=CMAP_GREY, vmin=0,    vmax=255)
    _show_img(axes[1], dct_coeffs,   "DCT coeffs",   cmap=CMAP_DCT)
    _show_img(axes[2], q_coeffs,     "Quantised",     cmap=CMAP_DCT)
    _show_img(axes[3], recon_block,  "Reconstructed", cmap=CMAP_GREY, vmin=0,    vmax=255)

    for ax, data in zip(axes[1:3], [dct_coeffs, q_coeffs]):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="8%", pad=0.05)
        im  = ax.get_images()[0]
        fig.colorbar(im, cax=cax)

    plt.tight_layout()
    _save(fig, save_path)



def plot_motion_vectors(
    frame_bgr: np.ndarray,
    motion_vectors: np.ndarray,
    mb_size: int = 16,
    save_path: str = "",
) -> None:
    rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(rgb)
    ax.set_title("Motion Vectors (P-frame)", fontsize=TITLE_FS, fontweight="bold")

    n_mb_h, n_mb_w = motion_vectors.shape[:2]
    for i in range(n_mb_h):
        for j in range(n_mb_w):
            dy, dx = motion_vectors[i, j]
            y_center = i * mb_size + mb_size // 2
            x_center = j * mb_size + mb_size // 2
            if dy != 0 or dx != 0:
                ax.annotate(
                    "", xy=(x_center + dx, y_center + dy),
                    xytext=(x_center, y_center),
                    arrowprops=dict(arrowstyle="->", color="lime", lw=0.8),
                )
            else:
                ax.plot(x_center, y_center, "r.", markersize=2, alpha=0.5)

    ax.axis("off")
    plt.tight_layout()
    _save(fig, save_path)



def plot_residuals(
    residuals: np.ndarray,
    reconstructed_bgr: np.ndarray,
    save_path: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("P-frame Residuals & Reconstruction", fontsize=12, fontweight="bold")

    vmax = max(abs(residuals.min()), abs(residuals.max())) + 1e-5
    im = axes[0].imshow(residuals, cmap=CMAP_DCT, vmin=-vmax, vmax=vmax)
    axes[0].set_title("Residual map (Y channel)", fontsize=TITLE_FS)
    axes[0].axis("off")
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    _show_img(axes[1], cv2.cvtColor(reconstructed_bgr, cv2.COLOR_BGR2RGB),
              "Reconstructed frame")
    plt.tight_layout()
    _save(fig, save_path)



def plot_qf_sweep(
    qf_values: List[float],
    psnr_values: List[float],
    cr_values: List[float],
    save_path: str,
) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 5))
    fig.suptitle("Effect of Quantisation Factor", fontsize=12, fontweight="bold")

    color1, color2 = "
    ax1.set_xlabel("Quantisation Factor (QF)", fontsize=LABEL_FS)
    ax1.set_ylabel("PSNR (dB)", color=color1, fontsize=LABEL_FS)
    l1, = ax1.plot(qf_values, psnr_values, "o-", color=color1, linewidth=2,
                   markersize=6, label="PSNR (dB)")
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Compression Ratio", color=color2, fontsize=LABEL_FS)
    l2, = ax2.plot(qf_values, cr_values, "s--", color=color2, linewidth=2,
                   markersize=6, label="Compression Ratio")
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.legend(handles=[l1, l2], loc="upper right", fontsize=LABEL_FS)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, save_path)


def plot_gop_sweep(
    gop_values: List[int],
    psnr_values: List[float],
    cr_values: List[float],
    save_path: str,
) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 5))
    fig.suptitle("Effect of GOP Size", fontsize=12, fontweight="bold")

    color1, color2 = "
    ax1.set_xlabel("GOP Size (G)", fontsize=LABEL_FS)
    ax1.set_ylabel("PSNR (dB)", color=color1, fontsize=LABEL_FS)
    l1, = ax1.plot(gop_values, psnr_values, "o-", color=color1, linewidth=2,
                   markersize=6, label="PSNR (dB)")
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Compression Ratio", color=color2, fontsize=LABEL_FS)
    l2, = ax2.plot(gop_values, cr_values, "s--", color=color2, linewidth=2,
                   markersize=6, label="Compression Ratio")
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.legend(handles=[l1, l2], loc="upper right", fontsize=LABEL_FS)
    ax1.set_xticks(gop_values)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, save_path)


def plot_psnr_per_frame(
    frame_metrics: List[Dict],
    frame_types: List[str],
    save_path: str,
) -> None:
    idxs   = [m["frame"] for m in frame_metrics]
    psnrs  = [min(m["psnr"], 60) for m in frame_metrics]   
    colors = ["

    fig, ax = plt.subplots(figsize=(max(8, len(idxs) * 0.4), 4))
    ax.bar(idxs, psnrs, color=colors, edgecolor="none")
    ax.set_xlabel("Frame index",  fontsize=LABEL_FS)
    ax.set_ylabel("PSNR (dB)",    fontsize=LABEL_FS)
    ax.set_title("Per-frame PSNR", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    patches = [
        mpatches.Patch(color="
        mpatches.Patch(color="
    ]
    ax.legend(handles=patches, fontsize=LABEL_FS)
    plt.tight_layout()
    _save(fig, save_path)



def plot_pipeline_overview(
    orig_bgr: np.ndarray,
    Y: np.ndarray,
    Cb: np.ndarray,
    Cr: np.ndarray,
    block_raw: np.ndarray,
    block_dct: np.ndarray,
    block_q: np.ndarray,
    block_recon: np.ndarray,
    residuals: np.ndarray,
    recon_bgr: np.ndarray,
    motion_vectors: np.ndarray,
    mb_size: int,
    save_path: str,
) -> None:
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle("Simplified MPEG-4 Encoder Pipeline — Overview",
                 fontsize=15, fontweight="bold", y=1.01)

    gs = GridSpec(3, 5, figure=fig, hspace=0.45, wspace=0.3)

    _show_img(fig.add_subplot(gs[0, 0]),
              cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB), "① Original Frame")
    _show_img(fig.add_subplot(gs[0, 1]),
              Y,  "② Y (Luma)",    cmap=CMAP_GREY, vmin=16, vmax=235)
    _show_img(fig.add_subplot(gs[0, 2]),
              Cb, "② Cb (Chroma)", cmap="coolwarm")
    _show_img(fig.add_subplot(gs[0, 3]),
              Cr, "② Cr (Chroma)", cmap="coolwarm")
    _show_img(fig.add_subplot(gs[0, 4]),
              cv2.cvtColor(recon_bgr, cv2.COLOR_BGR2RGB), "⑤ Reconstructed Frame")

    _show_img(fig.add_subplot(gs[1, 0]), block_raw,   "③ Raw Block (8×8)",
              cmap=CMAP_GREY, vmin=0, vmax=255)
    _show_img(fig.add_subplot(gs[1, 1]), block_dct,   "③ DCT Coefficients",
              cmap=CMAP_DCT)
    _show_img(fig.add_subplot(gs[1, 2]), block_q,     "③ Quantised Coeffs",
              cmap=CMAP_DCT)
    _show_img(fig.add_subplot(gs[1, 3]), block_recon, "③ Reconstructed Block",
              cmap=CMAP_GREY, vmin=0, vmax=255)
    vmax_r = max(abs(residuals.min()), abs(residuals.max())) + 1e-5
    ax_res = fig.add_subplot(gs[1, 4])
    im_res = ax_res.imshow(residuals, cmap=CMAP_DCT, vmin=-vmax_r, vmax=vmax_r,
                           interpolation="nearest")
    ax_res.set_title("⑤ Residual Map", fontsize=TITLE_FS)
    ax_res.axis("off")
    fig.colorbar(im_res, ax=ax_res, fraction=0.046, pad=0.04)

    ax_mv = fig.add_subplot(gs[2, :])
    rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    ax_mv.imshow(rgb)
    ax_mv.set_title("④ Motion Vectors Overlay (P-frame)", fontsize=TITLE_FS,
                    fontweight="bold")
    n_mb_h, n_mb_w = motion_vectors.shape[:2]
    for i in range(n_mb_h):
        for j in range(n_mb_w):
            dy, dx = motion_vectors[i, j]
            yc = i * mb_size + mb_size // 2
            xc = j * mb_size + mb_size // 2
            if dy != 0 or dx != 0:
                ax_mv.annotate(
                    "", xy=(xc + dx, yc + dy), xytext=(xc, yc),
                    arrowprops=dict(arrowstyle="->", color="lime", lw=0.8),
                )
    ax_mv.axis("off")

    plt.tight_layout()
    _save(fig, save_path)
