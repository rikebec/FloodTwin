#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 13:06:02 2025

This code can be used to conduct a point-based data assimilation employing an Ensemble Kalman Filter.
Here we assimilate soil moisture data into a simple conceptual hydrological model.
The soil moisture data is SAR-derived soil moisture from the Sentinel-1 mission 
and was downloaded from https://land.copernicus.eu/en/products/soil-moisture/daily-surface-soil-moisture-v1.0
The model is setup with precipitation and potential evaporation data from the 
UK-COSMOS site at Hollin Hill

@author of the data assimilation code: Rike Becker 
@author of the model code: Anthanasios Paschalis
"""

import numpy as np
import matplotlib.pyplot as plt
#from scipy.stats import norm
import pandas as pd
import Conceptual_Model

#%%
def initialize_ensemble(N, Param):
    """
    Initialize the ensemble for the new hydrological model.
    N: Number of ensemble members
    Param: Model parameters 
    """
    (Zr, n1, Ks, Ksb, Zg, n2, b, c, d, T1, T2, DT) = Param
    
    # Randomly initialize states (assuming a range based on model constraints)
    S1_ensemble = np.random.uniform(0, 1, size=N)  # Soil moisture (normalized 0-1)
    S2_ensemble = np.random.uniform(0, 1, size=N)  
    V1_ensemble = np.random.uniform(0, 1000, size=N) # Initial water volumes (adjust range)
    V2_ensemble = np.random.uniform(0, 1000, size=N)  
    
    return np.column_stack((S1_ensemble, S2_ensemble, V1_ensemble, V2_ensemble))

#%%
def forecast_ensemble(ensemble, Param, R, PET):
    """
    Propagate the ensemble forward using the new model.
    ensemble: Current ensemble (Nx4 array)
    Param: Model parameters
    R: Precipitation time series
    PET: Potential evapotranspiration time series
    """
    N = ensemble.shape[0]
    updated_ensemble = np.zeros_like(ensemble)
    
    for i in range(N):
        Ini = tuple(ensemble[i])  # Extract (S1, S2, V1, V2) for this member
        Qt, Q1_a, Q2_a, S1_a, S2_a, V1_a, V2_a = Conceptual_Model.Model(Param, Ini, R, PET)
        # Use the last timestep as the new state
        updated_ensemble[i] = [S1_a[-1], S2_a[-1], V1_a[-1], V2_a[-1]]

    return updated_ensemble

#%%

def update_ensemble(ensemble, observation1, obs_error1, observation2=None, obs_error2=None, time1=None, time2=None, current_time=None):
    """
    Update the ensemble based on observations from two different datasets, considering their timestamps.
    """
    #S1_ensemble = ensemble[:, 0]  # S1 values (soil moisture from first variable)
    S1_ensemble = np.uniform(0, 1, size = N)
    ensemble_var = np.var(S1_ensemble)
    
    # Apply Kalman update for the first dataset if the observation time matches
    if time1 is not None and time1 == current_time:
        kalman_gain1 = ensemble_var / (ensemble_var + obs_error1**2)
        updated_S1_1 = S1_ensemble + kalman_gain1 * (observation1 - S1_ensemble)
        ensemble[:, 0] = updated_S1_1  # Update S1 in ensemble
    
    # Apply Kalman update for the second dataset if the observation time matches
    if time2 is not None and time2 == current_time:
        kalman_gain2 = ensemble_var / (ensemble_var + obs_error2**2)
        updated_S1_2 = S1_ensemble + kalman_gain2 * (observation2 - S1_ensemble)
        ensemble[:, 0] = updated_S1_2  # Update S1 in ensemble

    return ensemble

#%% Set model parameters

# Parameters
N = 100  # Number of ensemble members

Zr = 100.0   # [mm]
n1 = 3.7     # [-]
Ks = 15.0*24    # [mm/day] 
Ksb = 60*24    # [mm/day]
Zg = 1500.0  # [mm]
n2 = 2     # [-] 

b = 7.0     # [-]
c = (5.0*1e-4)*24 # [1/day]
d = 8.0      # [-]

T1 = 7.0/24     # [day]
T2 = 80.0/24   # [day]
DT = 1.0     # [day]
    
S1 = 0.5    # [-]
S2 = 0.5     # [-]
V1 = 0.0     # [mm]
V2 = 0.0     # [mm]
    
Param = (Zr,n1,Ks,Ksb,Zg,n2,b,c,d,T1,T2,DT)
Ini = (S1,S2,V1,V2)

#%% load the data
I_dir1 ='~/.../COSMOS_data/' #load COSMOS soil moisture data
I_dir2 ='~/.../SAR_soil_moisture_data_at_COSMOS_sites/' #load SAR data extracted at the same COSMOS site

# Precipitation
R = pd.read_csv(I_dir1+'Precip_HOLLN.csv')['precip'].to_numpy()  # precipitation (mm/day)
R = np.tile(R, 2) # add two years for warm-up
R = pd.Series(R).interpolate(method='linear').to_numpy() #to get rid of any nan values, using a simple linear interpolation
# Potential ET
PET = pd.read_csv(I_dir1+'PET_HOLLN.csv')['pe'].to_numpy() #  potential ET (mm/day)
PET = np.tile(PET, 2)
PET = pd.Series(PET).interpolate(method='linear').to_numpy()
# Soil moisture remote sensing (SAR/Sentinel-1)
observations = pd.read_csv(I_dir2+'SM_SAR_HOLLN_2023_2024.csv')['ssm'].to_numpy() # read the SAR soil moisture data at HOLLIN HILL (here for the years 2023-2024)
observations = observations/100
observations = np.tile(observations, 2)
# Soil moisture COSMOS
cosmos_soil_moisture = pd.read_csv(I_dir1+'SM_HOLLN.csv') #COSMOS soil moisture
cosmos_soil_moisture = cosmos_soil_moisture['soil_moisture'].to_numpy()  # COSMOS soil moisture
cosmos_soil_moisture = (cosmos_soil_moisture/100)
cosmos_soil_moisture = np.tile(cosmos_soil_moisture, 2)

# Assume time1 and time2 correspond to the timestamps of observations_1 and observations_2 datasets
time1 = pd.read_csv(I_dir2+'SM_SAR_HOLLN_2023_2024.csv')['date'].to_numpy()  # Timestamps for first dataset
time2 = pd.read_csv(I_dir1+'SM_HOLLN.csv')['datetime'].to_numpy()  # Timestamps for second dataset

observations_1 = observations # SAR data
observations_2 = cosmos_soil_moisture # COSMOS data

#%% Assimilation
time_steps = len(R)

# Simulate true soil moisture for each time step
true_soil_moisture = np.zeros(time_steps)
true_soil_moisture[0] = 0.5  # Initial S1
ini_S2 = np.zeros(time_steps)
ini_S2[0] = 0.5
ini_V1 = np.zeros(time_steps)
ini_V1[0] = 0
ini_V2 = np.zeros(time_steps)
ini_V2[0] = 0


# Initialize ensemble
ensemble = initialize_ensemble(N, Param)

for t in range(time_steps):  # Start from t=0
    if t == 0:
        Ini = (true_soil_moisture[t], ini_S2[t], ini_V1[t], ini_V2[t])  # Use initial values directly
    else:
        Ini = (true_soil_moisture[t-1], ini_S2[t-1], ini_V1[t-1], ini_V2[t-1])  

    # Ensure R and PET are passed correctly
    Qt, Q1_a, Q2_a, S1_a, S2_a, V1_a, V2_a = Conceptual_Model.Model(
        Param, Ini, R[max(0, t-1):t+1], PET[max(0, t-1):t+1]
    )

    # Store only the last timestep values
    true_soil_moisture[t] = S1_a[-1]
    ini_S2[t] = S2_a[-1]
    ini_V1[t] = V1_a[-1]
    ini_V2[t] = V2_a[-1]


# Main loop to handle missing values in both datasets
assimilated_soil_moisture = np.zeros(time_steps)

for t in range(time_steps):
    # Current time step
    current_time = t

    # Initialize observations for both datasets
    obs1 = None
    obs2 = None
    obs_error1 = 0.5  
    obs_error2 = 0.1

    # Check if the first observation dataset has data for this time step
    if t < len(time1) and time1[t] == current_time:
        obs1 = observations_1[t]  # Observation from the first dataset
    else:
        obs1 = None  # No observation for this dataset at time t

    # Check if the second observation dataset has data for this time step
    if t < len(time2) and time2[t] == current_time:
        obs2 = observations_2[t]  # Observation from the second dataset
    else:
        obs2 = None  # No observation for this dataset at time t

    # Forecast step (propagate using the full model)
    if t > 0:
        ensemble = forecast_ensemble(ensemble, Param, np.array([R[t-1], R[t]]), np.array([PET[t-1], PET[t]]))
    else:
        ensemble = forecast_ensemble(ensemble, Param, np.array([R[t]]), np.array([PET[t]]))

    # Assimilate data for both datasets if available
    if obs1 is not None and obs2 is not None:
        # Both observations are available, assimilate both
        ensemble = update_ensemble(ensemble, obs1, obs_error1, obs2, obs_error2)
    elif obs1 is not None:
        # Only the first observation is available, assimilate it
        ensemble = update_ensemble(ensemble, obs1, obs_error1)
    elif obs2 is not None:
        # Only the second observation is available, assimilate it
        ensemble = update_ensemble(ensemble, obs2, obs_error2)

    # Store the assimilated S1 (ensemble mean) for this time step
    assimilated_soil_moisture[t] = np.nanmean(ensemble[:, 0])  # Extract S1 only for assimilation

#%% Get goodness-of-fit value
# simulated soil moisture fit to COSMOS data

NSE_mod = 1-(np.sum((true_soil_moisture - cosmos_soil_moisture) ** 2))/(np.sum((np.mean(cosmos_soil_moisture)-cosmos_soil_moisture) ** 2))
print('NSE_model =', f"{NSE_mod:.2f}")
RMSE_mod = np.sqrt(np.mean((true_soil_moisture-cosmos_soil_moisture)**2))
print('RMSE_model =', f"{RMSE_mod:.2f}")

# assimilated soil moisture fit to COSMOS data
NSE_da = 1-(np.sum((assimilated_soil_moisture - cosmos_soil_moisture) ** 2))/(np.sum((np.mean(cosmos_soil_moisture)-cosmos_soil_moisture) ** 2))
print('NSE_assimilation =', f"{NSE_da:.2f}")
RMSE_da = np.sqrt(np.mean((assimilated_soil_moisture-cosmos_soil_moisture)**2))
print('RMSE_assimilation =', f"{RMSE_da:.2f}")

#%% Plot results without assimilation

# Define the start index for the second half to exclude warm-up period
half_index = len(true_soil_moisture) // 2  
plt.figure(figsize=(10, 6))

# Slice all time series to keep only the second half
plt.plot(true_soil_moisture[half_index:], label="Simulated soil moisture", linestyle='dashed')
plt.plot(observations[half_index:], label="Observations - SAR data", linestyle='none', marker='o', markersize=4)
plt.plot(cosmos_soil_moisture[half_index:], label="Observations - COSMOS data", color='black', linestyle='none', marker='x', markersize=4)


plt.legend()
plt.xlabel("Time Step")
plt.ylabel("Surface soil moisture - top 10 cm (%)")
plt.title("Soil moisture without assimilation")
plt.show()

#%% Plot results with assimilation
plt.figure(figsize=(15, 6))

plt.plot(true_soil_moisture[half_index:], label="Simulated soil moisture", linestyle='dashed')
plt.plot(cosmos_soil_moisture[half_index:], label="Observations - COSMOS data", color='black', linestyle='none', marker='x', markersize=4)
plt.plot(observations[half_index:], label="Observations - SAR data", linestyle='none', marker='o', markersize=4)
plt.plot(assimilated_soil_moisture[half_index:], label="Assimilated soil moisture", color='red')

plt.legend()
plt.xlabel("Time Step")
plt.ylabel("Surface soil moisture - top 10 cm (%)")
plt.title("Soil moisture assimilation with EnKF at Hollin Hill")

plt.show()