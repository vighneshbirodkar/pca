import csv
import numpy as np
from matplotlib import pyplot as plt
from tga import TGA
from pcp import pcp
from numpy.linalg import norm, matrix_rank
from datetime import datetime

OUTLIERS = [
        "2015-03-03 21:07:53",
        "2015-03-16 02:57:53",
        "2015-03-31 03:27:53",
        "2015-04-14 23:12:53"
]

FORMAT = '%Y-%m-%d %H:%M:%S'
START_TIME_STR = '2015-02-26 21:42:53'
START_TIME = datetime.strptime(START_TIME_STR, FORMAT)

OUTLIER_IDX = []
for o in OUTLIERS:
    t = datetime.strptime(o, FORMAT)
    diff = t - START_TIME
    idx = diff.total_seconds()/300
    idx = int(idx)
    OUTLIER_IDX += [idx]

def do_plot(values):
    for value in values:
        plt.figure(figsize=(12, 8))
        plt.plot(value)

def make_windows_overlap(data, size):
    rows = data.shape[0]
    
    arr = np.array([data[i:i+size] for i in range(rows - size)])
    return arr

def make_windows_non_overlap(data, size):
    rows = data.shape[0]/size
    cols = size
    arr = np.array([data[i*cols:(i + 1)*cols] for i in range(rows)])
    return arr

DD = 23
HH = 02
MM = 02

fn = "/home/vighnesh/data/Twitter_volume_AAPL.csv"
tweet_data = np.loadtxt(open(fn,"r"),delimiter=",", skiprows=1, usecols=[1])
plt.plot(tweet_data)
print("Original shape = ", tweet_data.shape)

M = make_windows_non_overlap(tweet_data, 30)
print("NNZ original = ", np.count_nonzero(M))
print("Matrix shape = ", M.shape)

L, S, _ = pcp(M, verbose=True, delta=1e-4, svd_method='approximate')
L = L.flatten()
S = S.flatten()


outlier_values = tweet_data[OUTLIER_IDX]
plt.scatter(OUTLIER_IDX, outlier_values, color='red')

plt.xlim(0, 16000)
plt.ylim(0, 16000)
plt.figure()
N = np.abs(S)
plt.plot(N, label='Mag of low rank')
plt.figure()
plt.plot(L, color='orange')



plt.show()