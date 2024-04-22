import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('calculate_shuffle_distance_bones.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(dataset.head())