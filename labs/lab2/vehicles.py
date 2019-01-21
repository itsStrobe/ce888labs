import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

df = pd.read_csv('./vehicles.csv')
old = df.as_matrix(columns=['Current fleet'])
new = df.as_matrix(columns=['New Fleet'])
new = new[~np.isnan(new).any(axis=1)]

old_idx = np.arange(1, old.size + 1).transpose()
new_idx = np.arange(1, new.size + 1).transpose()

# Scatter Plots
plt.scatter(old_idx, old, c='b') # Old data red
old_patch = mpatches.Patch(color='blue', label='Current Fleet')
plt.scatter(new_idx, new, c='r') # New data blue
new_patch = mpatches.Patch(color='red', label='New Fleet')
plt.legend(handles=[old_patch, new_patch])
plt.savefig('ScatterPlot_Vehicles.png')
plt.show()

# Histograms
plt.hist(old, color='b')
plt.legend(labels=['Current Fleet'])
plt.savefig('Histogram_Current.png')
plt.show()
plt.hist(new, color='r')
plt.legend(labels=['New Fleet'])
plt.savefig('Histogram_New.png')
plt.show()
