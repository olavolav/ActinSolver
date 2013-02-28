# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

INPUT_FILE_NAME = "out-tail.dat"

# load data from file
data = np.loadtxt( INPUT_FILE_NAME )
data, times = data[:,1], data[:,0]

plt.xlabel('time (s)')
plt.ylabel('photon count (a.u.)')
plt.plot(times, data)
plt.show()
