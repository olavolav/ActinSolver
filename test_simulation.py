# -*- coding: utf-8 -*-

from ActinSimulator import *
import matplotlib.pyplot as plt
import numpy as np

sim = ActinSimulator(3.0, 0.5, 0.5, 0.005, 0.001, 0.5)
signal_times, signal_amplitudes = sim.generate_time_series_for_duration(5.0)

plt.xlabel('time (s)')
plt.ylabel('photon count (a.u.)')
plt.plot(signal_times, signal_amplitudes)
plt.show()

