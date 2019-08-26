from .model import design_matrix, tess_earth_vector, visibility, smooth, MAP
from .animate import animate, render
import os
import urllib.request
from tqdm import tqdm

# Download the TESS data
base = "https://archive.stsci.edu/missions/tess/models/"
files = [
    "TESS_EPH_PRE_LONG_2018176_01.bsp",
    "TESS_EPH_PRE_LONG_2018183_01.bsp",
    "TESS_EPH_PRE_LONG_2018186_01.bsp",
    "TESS_EPH_PRE_LONG_2018190_01.bsp",
    "TESS_EPH_PRE_LONG_2018193_01.bsp",
    "TESS_EPH_PRE_LONG_2018197_01.bsp",
    "TESS_EPH_PRE_LONG_2018200_01.bsp",
    "TESS_EPH_PRE_LONG_2018204_01.bsp",
    "TESS_EPH_PRE_LONG_2018207_01.bsp",
    "TESS_EPH_PRE_LONG_2018211_01.bsp",
    "TESS_EPH_PRE_LONG_2018214_01.bsp",
    "TESS_EPH_PRE_LONG_2018218_01.bsp",
    "TESS_EPH_PRE_LONG_2018221_01.bsp",
    "TESS_EPH_PRE_LONG_2018225_01.bsp",
    "TESS_EPH_PRE_LONG_2018228_01.bsp",
    "TESS_EPH_PRE_LONG_2018232_01.bsp",
    "TESS_EPH_PRE_LONG_2018235_01.bsp",
    "TESS_EPH_PRE_LONG_2018239_01.bsp",
    "TESS_EPH_PRE_LONG_2018242_01.bsp",
    "TESS_EPH_PRE_LONG_2018246_01.bsp",
    "TESS_EPH_PRE_LONG_2018249_01.bsp",
    "TESS_EPH_PRE_LONG_2018253_01.bsp",
    "TESS_EPH_PRE_LONG_2018256_01.bsp",
    "TESS_EPH_PRE_LONG_2018260_01.bsp",
    "TESS_EPH_PRE_LONG_2018263_01.bsp",
    "TESS_EPH_PRE_LONG_2018268_01.bsp",
    "TESS_EPH_PRE_LONG_2018270_01.bsp",
    "TESS_EPH_PRE_LONG_2018274_01.bsp",
    "TESS_EPH_PRE_LONG_2018277_01.bsp",
    "TESS_EPH_PRE_LONG_2018282_01.bsp",
    "TESS_EPH_PRE_LONG_2018285_01.bsp",
    "TESS_EPH_PRE_LONG_2018288_01.bsp",
    "TESS_EPH_PRE_LONG_2018291_01.bsp",
    "TESS_EPH_PRE_LONG_2018295_01.bsp",
    "TESS_EPH_PRE_LONG_2018298_01.bsp",
    "TESS_EPH_PRE_LONG_2018302_01.bsp",
    "TESS_EPH_PRE_LONG_2018305_01.bsp",
    "TESS_EPH_PRE_LONG_2018309_01.bsp",
    "TESS_EPH_PRE_LONG_2018312_01.bsp",
    "TESS_EPH_PRE_LONG_2018316_01.bsp",
    "TESS_EPH_PRE_LONG_2018319_01.bsp",
    "tess2018338154046-41240_naif0012.tls",
    "tess2018338154429-41241_de430.bsp",
]

# Check if all the files are present
download = False
for file in files:
    f = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", file)
    if not os.path.exists(f):
        download = True
        break

# Download?
if download:
    print("Acquiring TESS ephemerides...")
    for file in tqdm(files):
        f = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", file)
        if not os.path.exists(f):
            urllib.request.urlretrieve(base + file, f)
