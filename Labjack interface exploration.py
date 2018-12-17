import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import io
import pandas as pd
import seaborn as sns
from scipy import signal
import scipy
import os

#parent folder containing lj files
target_dir = '/Users/davidprotter/Downloads/11_29_18 rig test/Labjack Logs/11_29_18 rig test lj/'
os.chdir(target_dir)

#ID dat files only, sort list
names = [fname for fname in os.listdir(target_dir) if fname.endswith('.dat')]
names.sort()

lj = pd.read_table(names[0], skiprows = 10)

#assemble all files into a single pandas df
for file in names[1:]:
     lj = lj.append(pd.read_table(file, skiprows = 10))

#voltage threshold over which we count "on"
threshold = 2.0

#make a copy of the lj df so we can use the same references without adding columns
thresholded_cols = {}
lj_signal = lj.copy()

#voltage lines we care about
voltages = ['v0', 'v1', 'v2', 'v4']

for col in lj.columns:
    if col.startswith('v'):
        if col in voltages:
            lj_signal[col] = lj[col] > threshold
            thresholded_cols[col] = 0
        else:
            lj.drop(columns = col, inplace = True)
            lj_signal.drop(columns = col, inplace = True)
    elif col.startswith('y'):
        print('dropping %s'%col)
        lj.drop(columns = col, inplace = True)
        lj_signal.drop(columns = col, inplace = True)


for col in thresholded_cols.keys():
    thresholded_cols[col] = lj_signal[col].idxmax()

thresholded_cols

"""see how well we id the start points"""
sli = lj[:140000]

fig, ax = plt.subplots()
ax.plot(sli['v0'], alpha = 0.5)
ax.plot(sli['v1'], alpha = 0.5)
ax.vlines(94576, ymin = 0, ymax = 2.6)
plt.show()

for k in thresholded_cols:
    v = thresholded_cols[k]
    ax = plt.subplot()
    ax.plot(sli[k])
    ax.vlines(v, ymin = 0, ymax = 3)
    ax.set_title(k)
    plt.show()

t0_voltage = 'v2'


data = lj_signal['v0'].values

data_back = data.copy()
data_back[1:] = data[:-1]
data_back[0] = False

data_fwd = data.copy()
data_fwd[:-1] = data[1:]
data_fwd[-1] = False


trailing_edge = np.logical_and(np.logical_and(data, data_fwd), data_back == False)

trail_edge_indices = trailing_edge.nonzero()[0]


start, finish = 140000, 140020
sli = lj[start:finish]
te = trail_edge_indices[np.logical_and(trail_edge_indices > start, trail_edge_indices < finish)]

fig, ax = plt.subplots()
ax.plot(sli['v0'], alpha = 0.5)
ax.plot(sli['v1'], alpha = 0.5)
ax.vlines(te, ymin = 0, ymax = 2.6)
plt.show()


def trailing_edge_detector(bool_trace):
    '''takes a boolean trace and identifies trailing edges. Returns a boolean
    array, as well as a np array of the indices'''
    if isinstance(bool_trace, pd.core.series.Series):
        data = bool_trace.values
    elif isinstance(bool_trace, np.ndarray):
        data = bool_trace

    data_back = data.copy()
    data_back[1:] = data[:-1]
    data_back[0] = False

    data_fwd = data.copy()
    data_fwd[:-1] = data[1:]
    data_fwd[-1] = False
    data
    data_fwd
    data_back

    trailing_edge = np.logical_and(np.logical_and(data, data_back), np.invert(data_fwd))

    trailing_edge_indices = trailing_edge.nonzero()[0]

    return trailing_edge, trailing_edge_indices



def voltage_alignment(v1, v2):

'''
#yep this works to detect the trailing edge
data = np.asarray([False, True, True, False])

data_back = data.copy()
data_back[1:] = data[:-1]
data_back[0] = False

data_fwd = data.copy()
data_fwd[:-1] = data[1:]
data_fwd[-1] = False
data
data_fwd
data_back

trailing_edge = np.logical_and(np.logical_and(data, data_back), np.invert(data_fwd))
'''


data = lj_signal['v0'].values

data_back = data.copy()
data_back[1:] = data[:-1]
data_back[0] = False

data_fwd = data.copy()
data_fwd[:-1] = data[1:]
data_fwd[-1] = False
data
data_fwd
data_back

trailing_edge = np.logical_and(np.logical_and(data, data_back), np.invert(data_fwd))

trail_edge_indices = trailing_edge.nonzero()[0]

start, finish = 140000, 140050
sli = lj[start:finish]
te = trail_edge_indices[np.logical_and(trail_edge_indices > start, trail_edge_indices < finish)]

_, te2 = trailing_edge_detector(data)
te2 = trail_edge_indices[np.logical_and(te2 > start, te2 < finish)]
fig, ax = plt.subplots()
ax.plot(sli['v0'], alpha = 0.5)
ax.plot(sli['v1'], alpha = 0.5)
ax.vlines(te, ymin = 0, ymax = 2.6)
ax.vlines(te2, ymin = 0, ymax = 2.7, color = 'red')
plt.show()
