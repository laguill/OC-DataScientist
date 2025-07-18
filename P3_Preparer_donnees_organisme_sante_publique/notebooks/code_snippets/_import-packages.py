from pathlib import Path

from itables import init_notebook_mode, show


init_notebook_mode(all_interactive=True)
import altair as alt
import matplotlib.pyplot as plt
import missingno as msno
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

from scripts import analysis_tools as tools


tools.set_options()

csv_file: Path = Path("data/raw/fr.openfoodfacts.org.products.csv")
