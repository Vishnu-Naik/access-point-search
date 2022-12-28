
import pandas as pd
import seaborn as sns
from pathlib import Path
import os
import time

CUR_DIR = Path(__file__).parent.absolute()
DATA_CSV_FILE_NAME = r'exo_hip_right_2022_12_13-18_02.csv'
DATA_STORAGE_DIR = r'data'
DATA_CSV_PATH = os.path.join(CUR_DIR, DATA_STORAGE_DIR, DATA_CSV_FILE_NAME)

df = pd.read_csv(DATA_CSV_PATH)
data_read = df.to_numpy().flatten()

dataset = pd.DataFrame({
    'data1': data_read,
    'data2': data_read})

sns.set_theme(style="darkgrid")

# Plot the responses for different datasets
line_plot = sns.lineplot(data=dataset)

# sns_fig = sns.get_figure()
fig = line_plot.get_figure()
filename = os.path.join(CUR_DIR, 'Plots/Data', 'exo_hip_right_' + time.strftime('%Y_%m_%d-%H_%M'))
fig.show()
# fig.savefig(filename + '.png')
