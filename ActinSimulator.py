# -*- coding: utf-8 -*-

import random
import numpy as np
import math as math

class ActinSimulator:
  """A simulator of actin fluorescence time series to test the Bayesian analysis."""
  
  def __init__(self, a_rate, a_amplitude_mean, n_sigma, tau_k, tau_i, f_baseline=0.0):
    self.rate = a_rate
    self.amplitude_mean = a_amplitude_mean
    self.noise_stddev = n_sigma
    self.tau_kernel = tau_k
    self.tau_image = tau_i
    self.baseline = f_baseline
  
  def generate_time_series(self, samples):
    signal_times = np.linspace(0.0, samples*self.tau_image, samples)
    signal_amplitudes = np.zeros_like(signal_times)
    
    # 1.) Generate (hidden) spike data
    for i in np.arange(samples):
      if random.random() < self.__spiking_probability():
        # there was a spike at that time point, now we have to determine its amplitude
        signal_amplitudes[i] = self.amplitude_mean # <------------------------- for testing only!!!!!
    
    # 2.) Convolve with kernel
    kernel = self.__generate_kernel()
    signal_amplitudes = np.convolve(signal_amplitudes, kernel, 'same')
    
    # 3.) Add noise
    for i in np.arange(samples):
      signal_amplitudes[i] += self.noise_stddev * random.random() # <---------- for testing only!!!!!
    
    return [signal_times, signal_amplitudes]
  
  def generate_time_series_for_duration(self, time):
    return self.generate_time_series( round(time / self.tau_image) )
  
  def __spiking_probability(self):
    return self.rate * self.tau_image
  
  def __generate_kernel(self):
    kernel_center_index = round(self.tau_kernel/self.tau_image * 5.0)
    kernel = np.zeros(2*kernel_center_index + 1)
    for i in np.arange( kernel.size ):
      distance_in_s = (i-kernel_center_index) * self.tau_image
      kernel[i] = 1.0*math.exp( -abs(distance_in_s)/self.tau_kernel )
    
    return kernel
  
