# Package imports
import os
import pathlib
import numpy as np
import pandas as pd

# Seed
np.random.seed(123)

# Define target function with one input and one output
def oscillator(x):
    return np.cos((x - 5) / 2) ** 2 * x * 2, -1 * np.cos((x - 5) / 2) ** 2 * x * 2

# set numpy data and stack another dimension onto y
n_data = 200
X_data = np.random.uniform(-10, 10, size=n_data)[:,np.newaxis]
y_data = np.hstack(oscillator(X_data))
y_data = y_data + np.random.normal(scale=1.0, size=y_data.shape) * [2, 0.5]

# Set output path
filepath = str(pathlib.Path(__file__).parent.resolve()) + "/tmp"
if not os.path.exists(filepath):
    os.mkdir(filepath)

# Save to DataFrame
df = pd.DataFrame({'x': X_data.flatten(), 'y0': y_data[:, 0].flatten(), 'y1': y_data[:, 1].flatten()})
df.to_csv(filepath + "/data.csv", index=False)