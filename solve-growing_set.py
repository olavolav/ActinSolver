# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import math as math

PI = math.pi
INITIAL_GUESS_OF_BASELINE = baseline = 0.1 # photon count baseline [photons]
INITIAL_GUESS_OF_STD_NOISE = std_noise = 0.05 # width of camera noise [photons]
INITIAL_GUESS_OF_TAU = tau = 0.010 # time scale of exponential decay of signal [s]
TAU_IMAGE = 0.001 # width of one sample, inverse of imaging rate [s]
gamma = 0.000001 # step size factor of gradient descent
# std_tau = 0.0002 # width of possible tau's [s]
ACCURACY_GOAL_OF_LOG_LIKELHOOD = 1.0
HIGHEST_POSSIBLE_EVENT_RATE = 3.0

USE_SIMULATED_DATA = True
PLOT_RESULTS_AT_MAX_EVENT_NUMBER = False
PLOT_RESULTS_AT_OPTIMAL_EVENT_NUMBER = True

print("------ Actin spike solver, OS, March 2013 ------")
INPUT_FILE_NAME = "data/out-tail.dat"

if(USE_SIMULATED_DATA):
  print("Simulating data...")
  from ActinSimulator import *
  sim = ActinSimulator(1.0, 15, INITIAL_GUESS_OF_STD_NOISE, 0.111, 0.005, TAU_IMAGE, 0.555)
  times, data = sim.generate_time_series_for_duration(5.0)
else:
  print("Loading data from file '{f}'...".format(f=INPUT_FILE_NAME))
  data = np.loadtxt( INPUT_FILE_NAME )
  data, times = data[:,1], data[:,0]

samples = data.size
print(" -> done, {s} samples ({st}s).".format(s=samples,st=samples*TAU_IMAGE))

print("Initializing...")
kernel_length = 21 # in samples, must be an odd number
kernel = np.zeros(kernel_length)
print(" -> kernel array width = {ks} samples ({kt}s)".format(ks=kernel_length,kt=kernel_length*TAU_IMAGE))
def recompute_kernel(tau):
  kernel_center_index = (kernel_length-1)/2
  for i in np.arange(kernel_length):
    distance_in_s = (i-kernel_center_index) * TAU_IMAGE
    kernel[i] = 1.0*math.exp( -abs(distance_in_s)/tau )
    # print("DEBUG: kernel[{index}] = {k}".format(index=i,k=kernel[i]))

recompute_kernel(tau)

def log_prior_on_tau(tau2):
  # return -1.0*math.log(math.sqrt(2*PI) * std_tau) - math.pow((tau2-INITIAL_GUESS_OF_TAU)/std_tau, 2)
  return 0.0 # we skip this for the moment

def log_prior_on_latent_data(latent_data):
  # ll = 0.0
  # for t in np.arange(latent_data.size):
  #   if(latent_data[t] > 0.0):
  #     ll += latent_data[t] * math.log(latent_data[t])
  # return -1.0*alpha*ll
  return 0.0 # we skip this for the moment

# Initialize latent data vector to base line level
latent_data = np.zeros_like(data)
print(" -> done.")

def log_likelihood_based_on_latent_data(latent_data, baseline, std_noise):
  ll = -1.0*samples*math.log(math.sqrt(2.0*PI)*std_noise)
  model_prediction = np.convolve(latent_data,kernel,'same') + baseline
  for t in np.arange(samples):
    # ll += -1.0*math.pow(std_noise, -2) * math.pow( data[t] - model_prediction[t], 2)
    ll -= 0.5*math.pow((data[t] - model_prediction[t])/std_noise, 2)
  return ll + log_prior_on_tau(tau) + log_prior_on_latent_data(latent_data)

def log_likelihood_based_on_set_of_events(e_times, e_amplitudes, baseline, std_noise):
  set_latent_data_array_based_on_set_of_events(e_times, e_amplitudes)
  return log_likelihood_based_on_latent_data(latent_data, baseline, std_noise)

def set_latent_data_array_based_on_set_of_events(e_times, e_amplitudes):
  latent_data.fill(0.0) # clear
  for i in range(event_times.size):
    latent_data[ event_times[i] ] = event_amplitudes[i]

def model_prediction_based_on_set_of_events(e_times, e_amplitudes, tau, baseline, std_noise):
  model_prediction = np.zeros_like(data)
  set_latent_data_array_based_on_set_of_events(e_times, e_amplitudes)
  recompute_kernel(tau)
  model_prediction = np.convolve(latent_data,kernel,'same') + baseline
  return model_prediction

event_times = np.zeros(0, int) # time given in terms of samples index
event_amplitudes = np.zeros(0, float) # amplitude of a given event

current_log_l = log_likelihood_based_on_set_of_events(event_times, event_amplitudes, baseline, std_noise)
print(" -> init log likelihood: {ll}".format(ll=current_log_l))
inferred_parameters_history = {}
log_likelihood_history = np.zeros(0)
# log_likelihood_history[0] = current_log_l
old_log_l = 0.0

max_number_of_events = int(TAU_IMAGE*samples*HIGHEST_POSSIBLE_EVENT_RATE)
print('')
for assumed_number_of_events in range(0, max_number_of_events+1):
  print("\n------ Growth step: assuming {e} events (max is {maxe}) ------".format(e=assumed_number_of_events,maxe=max_number_of_events))
  # print('DEBUG: log_l_0 = {ll}'.format(ll=log_likelihood_based_on_latent_data(latent_data, baseline, std_noise)))

  if(assumed_number_of_events > 0):
    print("--- find point of largest pos. prediction error to add latent event...")
    model_prediction = model_prediction_based_on_set_of_events(event_times, event_amplitudes, tau, baseline, std_noise)
    max_positive_error = data[0] - model_prediction[0]
    max_positive_error_sample_index = 0
    for t in range(1, samples): # inefficient!
      if(data[t] - model_prediction[t] > max_positive_error):
        max_positive_error = data[t] - model_prediction[t]
        max_positive_error_sample_index = t
    print(' -> found point of largest positive deviation at sample #{s} ({t}s)'.format(s=max_positive_error_sample_index, t=max_positive_error_sample_index*TAU_IMAGE))
    # adding event at that point
    event_times = np.append(event_times, max_positive_error_sample_index)
    event_amplitudes = np.append(event_amplitudes, max_positive_error)
    current_log_l = log_likelihood_based_on_set_of_events(event_times, event_amplitudes, baseline, std_noise)
    print(' -> log_l with this new event: {ll}'.format(ll=current_log_l))
  
  print("--- optimize all other parameters...") # TODO: These should be optimized simultaneously!
  set_latent_data_array_based_on_set_of_events(event_times, event_amplitudes)
  gradient_step = 1
  while(gradient_step == 1 or abs(old_log_l - current_log_l) > ACCURACY_GOAL_OF_LOG_LIKELHOOD):
    old_log_l = current_log_l
    
    # parameter: baseline
    baseline_shifted = baseline + 0.0001/gradient_step # whatever, just to compute the gradient
    new_log_l = log_likelihood_based_on_latent_data(latent_data, baseline_shifted, std_noise) # since latent data unchanged
    baseline_updated = min(2.0, max(0.0001, baseline + gamma*(new_log_l-current_log_l)/(0.0001/gradient_step)))
    new_log_l = log_likelihood_based_on_latent_data(latent_data, baseline_updated, std_noise) # since latent data unchanged
    if(new_log_l > current_log_l):
      print(' -> found better baseline at {x} (log_l -> {ll})'.format(x=baseline_updated,ll=new_log_l))
      baseline = baseline_updated
      current_log_l = new_log_l
    
    # parameter: standard deviation of noise
    std_noise_shifted = std_noise + 0.0001/gradient_step # whatever, just to compute the gradient
    new_log_l = log_likelihood_based_on_latent_data(latent_data, baseline, std_noise_shifted) # since latent data unchanged
    std_noise_updated = min(0.2, max(0.0001, std_noise + gamma*(new_log_l-current_log_l)/(0.0001/gradient_step))) 
    new_log_l = log_likelihood_based_on_latent_data(latent_data, baseline, std_noise_updated) # since latent data unchanged
    if(new_log_l > current_log_l):
      print(' -> found better std. of noise at {x} (log_l -> {ll})'.format(x=std_noise_updated,ll=new_log_l))
      std_noise = std_noise_updated
      current_log_l = new_log_l
    
    # parameter: tau
    if(assumed_number_of_events > 0):
      tau_shifted = tau + 0.00001/gradient_step # whatever, just to compute the gradient
      recompute_kernel(tau_shifted)
      new_log_l = log_likelihood_based_on_latent_data(latent_data, baseline, std_noise)
      # print('DEBUG: for tau I: (tau -> {t}s) new_log_l = {x}'.format(x=new_log_l,t=tau_shifted))
      tau_updated = min(0.2, max(0.0001, tau + gamma/100.0*(new_log_l-current_log_l)/(0.00001/gradient_step))) 
      recompute_kernel(tau_updated)
      new_log_l = log_likelihood_based_on_latent_data(latent_data, baseline, std_noise)
      # print('DEBUG: for tau II: (tau -> {t}s) new_log_l = {x}'.format(x=new_log_l,t=tau_updated))
      if(new_log_l > current_log_l):
        print(' -> found better tau at {x}s (log_l -> {ll})'.format(x=tau_updated,ll=new_log_l))
        tau = tau_updated
        current_log_l = new_log_l
      # else:
      #   print(' - no update of tau to {x}, would have decreased log_l from {old} to {new}'.format(x=tau_updated, old=current_log_l, new=new_log_l))
      recompute_kernel(tau)
    
    # parameter: all event amplitudes
    for i_event in range(event_times.size):
      event_amplitudes_shifted = np.array(event_amplitudes, float)
      event_amplitudes_shifted[i_event] += 0.01/gradient_step # whatever, just to compute the gradient
      new_log_l = log_likelihood_based_on_set_of_events(event_times, event_amplitudes_shifted, baseline, std_noise)
      event_amplitudes_shifted[i_event] = min(10.0, max(1.0, event_amplitudes_shifted[i_event] + gamma*(new_log_l-current_log_l)/(0.01/gradient_step)))
      new_log_l = log_likelihood_based_on_set_of_events(event_times, event_amplitudes_shifted, baseline, std_noise)
      if(new_log_l > current_log_l):
        print(' -> found better amplitude for event #{i}: {x} (log_l -> {ll})'.format(i=i_event, x=tau_updated, ll=new_log_l))
        event_amplitudes[i_event] = event_amplitudes_shifted[i_event]
        current_log_l = new_log_l
    
    gradient_step += 1
    
  log_likelihood_history = np.append(log_likelihood_history,current_log_l)
  inferred_parameters_history[assumed_number_of_events] = [tau, baseline, std_noise, event_times, event_amplitudes]
  print(" -> Result after growth step #{s}: log_l = {ll}".format(s=assumed_number_of_events,ll=current_log_l))

print("\n------ Results at max. event number ------")
print("> Inferred parameters:")
print("> tau = {y}s (initial guess was {x}s)".format(y=tau,x=INITIAL_GUESS_OF_TAU))
print("> baseline F_0 = {y} (initial guess was {x})".format(y=baseline,x=INITIAL_GUESS_OF_BASELINE))
print("> sigma_F = {y} (initial guess was {x})".format(y=std_noise,x=INITIAL_GUESS_OF_STD_NOISE))
# print("> most likely number of events: {n}".format(n=log_likelihood_history.argmax()))
# print inferred_parameters_history[log_likelihood_history.argmax()]

if(PLOT_RESULTS_AT_MAX_EVENT_NUMBER):
  print("Plotting result at the end...")
  plt.subplot(411)
  # plt.xlabel('time (s)')
  plt.ylabel('photon count')
  plt.title('Results at max. event number')
  plt.plot(times, data)

  plt.subplot(412)
  # plt.xlabel('time (s)')
  plt.ylabel('actin count')
  plt.plot(event_times*TAU_IMAGE + min(times), event_amplitudes, 'go')
  plt.xlim(min(times), max(times))
  plt.ylim(0.0, 1.1*max(event_amplitudes))

  plt.subplot(413)
  recompute_kernel(tau)
  model_prediction = model_prediction_based_on_set_of_events(event_times, event_amplitudes, tau, 0.0, std_noise)
  # plt.xlabel('time (s)')
  plt.ylabel('photon clean')
  plt.plot(times, model_prediction, 'r')

  plt.subplot(414)
  plt.xlabel('assumed number of events (max. likelihood is at {g})'.format(g=log_likelihood_history.argmax()))
  plt.ylabel('log. likelihood')
  plt.plot(range(log_likelihood_history.size), log_likelihood_history, 'ro-')
  plt.show()


print("\n------ Results at the most likely event number ------")
tau, baseline, std_noise, event_times, event_amplitudes = inferred_parameters_history[log_likelihood_history.argmax()]
recompute_kernel(tau)
set_latent_data_array_based_on_set_of_events(event_times, event_amplitudes)
print("> Inferred parameters:")
print("> tau = {y}s (initial guess was {x}s)".format(y=tau,x=INITIAL_GUESS_OF_TAU))
print("> baseline F_0 = {y} (initial guess was {x})".format(y=baseline,x=INITIAL_GUESS_OF_BASELINE))
print("> sigma_F = {y} (initial guess was {x})".format(y=std_noise,x=INITIAL_GUESS_OF_STD_NOISE))
print("> most likely number of events: {n}".format(n=log_likelihood_history.argmax()))
# print inferred_parameters_history[log_likelihood_history.argmax()]

if(PLOT_RESULTS_AT_OPTIMAL_EVENT_NUMBER):
  print("Plotting result at the optimal number of events...")
  plt.subplot(411)
  # plt.xlabel('time (s)')
  plt.ylabel('photon count')
  plt.title('Results at optimal event number ({g})'.format(g=log_likelihood_history.argmax()))
  plt.plot(times, data)

  plt.subplot(412)
  # plt.xlabel('time (s)')
  plt.ylabel('actin count')
  plt.plot(event_times*TAU_IMAGE + min(times), event_amplitudes, 'go')
  plt.xlim(min(times), max(times))
  plt.ylim(0.0, 1.1*max(event_amplitudes))

  plt.subplot(413)
  recompute_kernel(tau)
  model_prediction = model_prediction_based_on_set_of_events(event_times, event_amplitudes, tau, 0.0, std_noise)
  # plt.xlabel('time (s)')
  plt.ylabel('photon clean')
  plt.plot(times, model_prediction, 'r')

  plt.subplot(414)
  plt.xlabel('assumed number of events')
  plt.ylabel('log. likelihood')
  plt.plot(range(log_likelihood_history.size), log_likelihood_history, 'ro-')
  plt.show()
