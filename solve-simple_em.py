# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import math as math

PI = math.pi
baseline = 0.5 # photon count baseline [photons]
std_noise = 0.1 # width of camera noise [photons]
tau = 0.001 # time scale of exponential decay of signal [s]
tau_image = 0.001 # width of one sample, inverse of imaging rate [s]
gamma = 0.0005 # step size factor of gradient descent
gamma_for_tau = 0.000000000005 # step size factor of gradient descent
epsilon = 0.1 # accuracy limit
ACCURACY_GOAL_OF_LOG_LIKELHOOD = 1.5

print("------ Actin spike solver, OS, Feb 2013 ------")
INPUT_FILE_NAME = "out-tail.dat"

print("Loading data from file '{f}'...".format(f=INPUT_FILE_NAME))
data = np.loadtxt( INPUT_FILE_NAME )
data, times = data[:,1], data[:,0]
samples = data.size
print(" -> done, {s} samples found ({st}s).".format(s=samples,st=samples*tau_image))

print("Initializing...")
kernel_length = 21 # in samples, must be an odd number
kernel = np.zeros(kernel_length)
print(" -> kernel array width = {ks} samples ({kt}s)".format(ks=kernel_length,kt=kernel_length*tau_image))
def recompute_kernel(tau):
  kernel_center_index = (kernel_length-1)/2
  for i in range(kernel_length):
    distance_in_s = (i-kernel_center_index) * tau_image
    kernel[i] = 1.0*math.exp( -abs(distance_in_s)/tau )
    # print("DEBUG: kernel[{index}] = {k}".format(index=i,k=kernel[i]))
recompute_kernel(tau)

# Initialize latent data vector to base line level
latent_data = np.zeros_like(data)
print(" -> done.")

def log_likelihood_based_on_latent_data(latent_data):
  ll = -1.0*samples*math.log(math.sqrt(2.0*PI))
  model_prediction = np.convolve(latent_data,kernel,'same')
  for t in np.arange(samples):
    ll += -1.0*math.pow(std_noise, -2) * math.pow( data[t] - baseline - model_prediction[t], 2)
  return ll

current_log_l = log_likelihood_based_on_latent_data(latent_data)
print(" -> init log likelihood: {ll}".format(ll=current_log_l))

log_likelihood_history = np.zeros(1)
log_likelihood_history[0] = current_log_l
old_log_l = 0.0
for em_step in range(2):
  print("Expectation-maximization step #{e}".format)
  print("E: Optimizing latent variables (gradient ascent)...")

  gradient_step = 1
  while(gradient_step == 1 or abs(old_log_l - current_log_l) > ACCURACY_GOAL_OF_LOG_LIKELHOOD):
    old_log_l = current_log_l
    model_prediction = np.convolve(latent_data,kernel,'same')
    # adjust latent variables
    delta_latent = 0.0001/gradient_step # whatever, just to compute the gradient
    shifted_latent_data = latent_data + delta_latent
    shifted_model_prediction = np.convolve(shifted_latent_data, kernel, 'same')
    for t in range(samples):
      old_error = math.pow(data[t] - baseline - model_prediction[t], 2)
      new_error = math.pow(data[t] - baseline - shifted_model_prediction[t], 2)
      delta_loglikelihood = -1.0/math.pow(std_noise,2) * (new_error - old_error)
      # print("DEBUG: at t={time}: old lat = {lat}, shifted lat = {lat1}, data' = {dat}, old model = {mod0}, new mod = {mod1}, delta_ll = {d}".format(time=t,lat=latent_data[t],lat1=shifted_latent_data[t],dat=data[t]-baseline,mod0=model_prediction[t],mod1=shifted_model_prediction[t],d=delta_loglikelihood))
      # update estimate of latent variables
      latent_data[t] = max(0, latent_data[t] + gamma/gradient_step * delta_loglikelihood/delta_latent)
      # print("DEBUG: -> new lat = {lat}".format(lat=latent_data[t]))
    current_log_l = log_likelihood_based_on_latent_data(latent_data)
    print(" -> after E-step #{s}: log_l = {ll}".format(s=gradient_step,ll=current_log_l))
    gradient_step += 1
  
  print("M: Optimizing parameters (gradient ascent)...")
  gradient_step = 1
  while(gradient_step == 1 or abs(old_log_l - current_log_l) > ACCURACY_GOAL_OF_LOG_LIKELHOOD):
    old_log_l = current_log_l
    # adjust parameters
    delta_tau = 0.00001/gradient_step # whatever
    recompute_kernel(tau + delta_tau)
    current_log_l = log_likelihood_based_on_latent_data(latent_data)
    # print("DEBUG: log_l changed from {old} to {new}".format(old=old_log_l,new=current_log_l))
    # update tau
    tau = tau + gamma_for_tau/gradient_step * (current_log_l - old_log_l)/delta_tau
    recompute_kernel(tau)
    print(" -> new tau = {t}".format(t=tau))
    print(" -> after M-step #{s}: log_l = {ll}".format(s=gradient_step,ll=current_log_l))
    gradient_step += 1

  log_likelihood_history = np.append(log_likelihood_history,current_log_l)
  print(" -> Result after EM-step #{s}: log_l = {ll}".format(s=em_step+1,ll=current_log_l))


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
plt.plot(range(log_likelihood_history.size), log_likelihood_history, 'ro-')
plt.show()

# for s in range(samples):
#   print("latent result at t = {t}: {r}".format(t=s,r=latent_data[s]))
