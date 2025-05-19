import pickle
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from pathlib import Path
from scipy.optimize import lsq_linear
from rw_data_processing import *
from Data_synthesize import *
from tqdm import tqdm

plt.style.use("./rw_visualization.mplstyle")

if __name__ == "__main__":
    # Color
    current_palette = sns.color_palette()