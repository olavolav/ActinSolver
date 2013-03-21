# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

# INPUT_FILE_NAME = "data/out-tail.dat"
# INPUT_FILE_NAME = "data/2013-02-15/actin488_1uM_001_"
INPUT_FILE_NAME = "data/2013-02-19/actin488_1uM_v4_012_"

PLOT_WHOLE_DATASET = False
LENGTH_OF_SUBSETS = 25000

# load data from file
data = np.loadtxt( INPUT_FILE_NAME )
data, times = data[:,1], data[:,0]
print('-> {n} samples loaded.'.format(n=data.size))

print('Plotting...')
if(PLOT_WHOLE_DATASET):
  plt.xlabel('time (s)')
  plt.ylabel('photon count (a.u.)')
  plt.plot(times, data)
  plt.show()
  
  plt.hist(data, 30, normed=1)
  plt.show()
else:
  plt.subplot(221)
  # plt.xlabel('time (s)')
  plt.ylabel('photon count (a.u.)')
  plt.plot(times[:LENGTH_OF_SUBSETS], data[:LENGTH_OF_SUBSETS])

  plt.subplot(222)
  plt.hist(data[:LENGTH_OF_SUBSETS], 30, normed=1)

  plt.subplot(223)
  plt.xlabel('time (s)')
  plt.ylabel('photon count (a.u.)')
  plt.plot(times[-LENGTH_OF_SUBSETS:], data[-LENGTH_OF_SUBSETS:])
  
  plt.subplot(224)
  plt.hist(data[-LENGTH_OF_SUBSETS:], 30, normed=1)
  plt.show()
