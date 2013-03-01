# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import math as math

PI = math.pi
baseline = 0.5 # photon count baseline [photons]
std_noise = 0.1 # width of camera noise [photons]
tau = 0.001 # time scale of exponential decay of signal [s]
tau_image = 0.001 # width of one sample, inverse of imaging rate [s]
gamma = 0.0002 # step size factor of gradient descent
epsilon = 0.1 # accuracy limit

print("------ Actin spike solver, OS, Feb 2013 ------")
INPUT_FILE_NAME = "out-tail.dat"

print("Loading data from file '{f}'...".format(f=INPUT_FILE_NAME))
data = np.loadtxt( INPUT_FILE_NAME )
data, times = data[:,1], data[:,0]
samples = data.size
print(" -> done, {s} samples found ({st}s).".format(s=samples,st=samples*tau_image))

print("Initializing...")
kernel_length = 21 # in samples, must be an odd number
print(" -> kernel array width = {ks} samples ({kt}s)".format(ks=kernel_length,kt=kernel_length*tau_image))
kernel = np.zeros(kernel_length)
kernel_center_index = (kernel_length-1)/2
for i in range(kernel_length):
  distance_in_s = (i-kernel_center_index) * tau_image
  kernel[i] = 1.0*math.exp( -abs(distance_in_s)/tau )
  # print("DEBUG: kernel[{index}] = {k}".format(index=i,k=kernel[i]))

# Initialize latent data vector to base line level
latent_data = np.zeros_like(data)
print(" -> done.")

def log_likelihood(observation,latent_data):
  ll = -1.0*samples*math.log(math.sqrt(2.0*PI))
  model_prediction = np.convolve(latent_data,kernel,'same')
  for t in np.arange(samples):
    ll += -1.0*math.pow(std_noise, -2) * math.pow( data[t] - baseline - model_prediction[t], 2)
  return ll

log_l = log_likelihood(data, latent_data)
print(" -> init log likelihood: {ll}".format(ll=log_l))

log_likelihood_history = np.zeros(0)
print("Starting gradient ascent...")
for step in range(9):
  model_prediction = np.convolve(latent_data,kernel,'same')
  delta_latent = 0.00001/(step+1) # whatever, just to compute the gradient
  shifted_latent_data = latent_data + delta_latent
  shifted_model_prediction = np.convolve(shifted_latent_data,kernel,'same')
  for t in range(samples):
    old_error = math.pow(data[t] - baseline - model_prediction[t], 2)
    new_error = math.pow(data[t] - baseline - shifted_model_prediction[t], 2)
    delta_loglikelihood = -1.0/math.pow(std_noise,2) * (new_error - old_error)
    # print("DEBUG: at t={time}: old lat = {lat}, shifted lat = {lat1}, data' = {dat}, old model = {mod0}, new mod = {mod1}, delta_ll = {d}".format(time=t,lat=latent_data[t],lat1=shifted_latent_data[t],dat=data[t]-baseline,mod0=model_prediction[t],mod1=shifted_model_prediction[t],d=delta_loglikelihood))
    # update estimate of latent variables
    latent_data[t] = max(0, latent_data[t] + gamma * delta_loglikelihood/delta_latent)
    # print("DEBUG: -> new lat = {lat}".format(lat=latent_data[t]))
  log_l = log_likelihood(data, latent_data)
  log_likelihood_history = np.append(log_likelihood_history,log_l)
  print(" -> after step {s}: log_l = {ll}".format(s=step+1,ll=log_l))

print("Plotting result...")
plt.subplot(311)
# plt.xlabel('time (s)')
plt.ylabel('photon count')
plt.plot(times, data)
plt.subplot(312)
# plt.xlabel('time (s)')
plt.ylabel('actin count')
plt.plot(times, latent_data, 'g')
plt.subplot(313)
plt.xlabel('iteration')
plt.ylabel('log. likelihood')
plt.plot(range(1,log_likelihood_history.size+1), log_likelihood_history, 'ro-')
plt.show()

# for s in range(samples):
#   print("latent result at t = {t}: {r}".format(t=s,r=latent_data[s]))
