# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

# INPUT_FILE_NAME = "out-tail.dat"
INPUT_FILE_NAME = "2013-02-15/actin488_1uM_001_"

# load data from file
data = np.loadtxt( INPUT_FILE_NAME )
data, times = data[:,1], data[:,0]

plt.xlabel('time (s)')
plt.ylabel('photon count (a.u.)')
plt.plot(times, data)
plt.show()
