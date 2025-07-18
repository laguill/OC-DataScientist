from pathlib import Path

from itables import init_notebook_mode, show


init_notebook_mode(all_interactive=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import seaborn as sns

from IPython.display import Markdown
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ydata_profiling import ProfileReport

from scripts import analysis_tools as tools


tools.set_options()

csv_file: Path = Path("data/raw/2016_Building_Energy_Benchmarking.csv")
