QUANTISATION_FACTOR: float = 2.0        
GOP_SIZE: int = 8                        
SEARCH_WINDOW: int = 8                   
BLOCK_SIZE: int = 8                      
MACROBLOCK_SIZE: int = 16               


CHROMA_SUBSAMPLING: bool = True          


import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR            = os.path.join(BASE_DIR, "data")
INPUT_FRAMES_DIR    = os.path.join(DATA_DIR, "input_frames")
RECON_FRAMES_DIR    = os.path.join(DATA_DIR, "reconstructed_frames")
OUTPUTS_DIR         = os.path.join(DATA_DIR, "outputs")
RESULTS_DIR         = os.path.join(BASE_DIR, "results")
FIGURES_DIR         = os.path.join(RESULTS_DIR, "figures")
PLOTS_DIR           = os.path.join(RESULTS_DIR, "plots")
LOGS_DIR            = os.path.join(RESULTS_DIR, "logs")

COMPRESSED_BIN      = os.path.join(OUTPUTS_DIR, "compressed.bin")


import numpy as np
JPEG_LUMA_Q_MATRIX = np.array([
    [16, 11, 10, 16, 24,  40,  51,  61],
    [12, 12, 14, 19, 26,  58,  60,  55],
    [14, 13, 16, 24, 40,  57,  69,  56],
    [14, 17, 22, 29, 51,  87,  80,  62],
    [18, 22, 37, 56, 68,  109, 103, 77],
    [24, 35, 55, 64, 81,  104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.float32)

JPEG_CHROMA_Q_MATRIX = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=np.float32)


LOG_LEVEL = "INFO"
LOG_FILE  = os.path.join(LOGS_DIR, "encoder.log")
