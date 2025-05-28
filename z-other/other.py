import sys
import torch
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import plotly
import tqdm
import argparse
import rasterio
import geopandas
import shapely

# Check Python version
print(f"Python version: {sys.version}")

# Check PyTorch and CUDA
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Check other libraries
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Matplotlib version: {matplotlib.__version__}")
print(f"Seaborn version: {sns.__version__}")
print(f"Plotly version: {plotly.__version__}")
print(f"TQDM version: {tqdm.__version__}")
print(f"Rasterio version: {rasterio.__version__}")
print(f"GeoPandas version: {geopandas.__version__}")
print(f"Shapely version: {shapely.__version__}")

# Get cuDNN version
if torch.cuda.is_available():
    cudnn_version = torch.backends.cudnn.version()
    print(f"cuDNN version: {cudnn_version}")