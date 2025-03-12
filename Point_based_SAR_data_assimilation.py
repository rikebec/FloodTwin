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
    Param: Model parameters (including field capacity-like constraints)
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
def update_ensemble(ensemble, observation, obs_error):
    """
    Update only S1 in the ensemble using the observation.
    ensemble: Nx4 array (S1, S2, V1, V2)
    observation: Observed soil moisture (S1)
    obs_error: Observation error standard deviation
    """
    S1_ensemble = ensemble[:, 0]  # Extract S1 values
    ensemble_var = np.var(S1_ensemble)
    kalman_gain = ensemble_var / (ensemble_var + obs_error**2)
    
    # Apply Kalman update only to S1
    updated_S1 = S1_ensemble + kalman_gain * (observation - S1_ensemble)

    # Update the ensemble while keeping S2, V1, and V2 unchanged
    updated_ensemble = ensemble.copy()
    updated_ensemble[:, 0] = updated_S1  # Update only S1

    return updated_ensemble

#%%
I_dir ='~/Data/'

# Parameters
N = 100  # Number of ensemble members
obs_error = 0.1  # Observation error (%)

Zr = 100.0   # [mm]
n1 = 3.7     # [-]
Ks = 15.0*24    # [mm/h]
Ksb = 60*24    # [mm/h]
Zg = 1500.0  # [mm]
n2 = 2     # [-]

b = 7.0     # [-]
c = (5.0*1e-4)*24 # [1/h]
d = 8.0      # [-]

T1 = 7.0/24     # [day]
T2 = 80.0/24    # [day]
DT = 1.0     # [day]
    
S1 = 0.5    # [-]
S2 = 0.5     # [-]
V1 = 0.0     # [mm]
V2 = 0.0     # [mm]
    
Param = (Zr,n1,Ks,Ksb,Zg,n2,b,c,d,T1,T2,DT)
Ini = (S1,S2,V1,V2)

# load data
R = pd.read_csv(I_dir+'Precip_HOLLN.csv')['precip'].to_numpy()  # precipitation (mm/day)
R = np.tile(R, 2)
R = pd.Series(R).interpolate(method='linear').to_numpy() #to get rid of any nan values, using a simple linear interpolation

PET = pd.read_csv(I_dir+'PET_HOLLN.csv')['pe'].to_numpy() #  potential ET (mm/day)
PET = np.tile(PET, 2)
PET = pd.Series(PET).interpolate(method='linear').to_numpy()

observations = pd.read_csv(I_dir+'SM_SAR_HOLLN_2023_2024.csv')['ssm'].to_numpy()
observations = observations/100
observations = np.tile(observations, 2)

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


for t in range(time_steps):  # Start from t=0
    if t == 0:
        Ini = (true_soil_moisture[t], ini_S2[t], ini_V1[t], ini_V2[t])  # Use initial values directly
    else:
        Ini = (true_soil_moisture[t-1], ini_S2[t-1], ini_V1[t-1], ini_V2[t-1])  
    '''
    if any(np.isnan(Ini)):
        print(f"Warning: NaN detected in Ini at t={t}. Resetting initial values.")
        Ini = (0.5, 0.5, 0, 0)  # Reasonable default values
    '''
    # Ensure R and PET are passed correctly
    Qt, Q1_a, Q2_a, S1_a, S2_a, V1_a, V2_a = Conceptual_Model.Model(
        Param, Ini, R[max(0, t-1):t+1], PET[max(0, t-1):t+1]
    )

    # Store only the last timestep values
    true_soil_moisture[t] = S1_a[-1]
    ini_S2[t] = S2_a[-1]
    ini_V1[t] = V1_a[-1]
    ini_V2[t] = V2_a[-1]
    
    # Debugging output
    #print(f"t={t}, S1={true_soil_moisture[t]:.3f}, R={R[t]:.2f}, PET={PET[t]:.2f}")


'''
for t in np.arange(1,time_steps):
    #print(np.array([R[t]]))
    Ini = (true_soil_moisture[t-1],ini_S2[t-1], ini_V1[t-1], ini_V2[t-1])  # Initial conditions, assuming S2=0.9, V1=0, V2=0
    Qt, Q1_a, Q2_a, S1_a, S2_a, V1_a, V2_a = Model_B.Model(Param, Ini, np.array([R[t-1],R[t]]), np.array([PET[t-1],PET[t]]))
    true_soil_moisture[t] = S1_a[-1]
    ini_S2[t] = S2_a[-1]
    ini_V1[t] = V1_a[-1]
    ini_V2[t] = V2_a[-1]
    # Store the last S1 value from the simulation
    #print(S1_a)
'''

'''
Qt, Q1_a, Q2_a, S1_a, S2_a, V1_a, V2_a = Model_B.Model(Param, Ini, R, PET)
'''
# Initialize ensemble
ensemble = initialize_ensemble(N, Param)

# Perform EnKF assimilation
assimilated_soil_moisture = np.zeros(time_steps)


for t in range(time_steps):
    # Forecast step (propagate using the full model)
    #ensemble = forecast_ensemble(ensemble, Param, np.array([R[t]]), np.array([PET[t]]))
    #ensemble = forecast_ensemble(ensemble, Param, np.array([R[t-1],R[t]]), np.array([PET[t-1],PET[t]]))
    
    if t > 0:
        ensemble = forecast_ensemble(ensemble, Param, np.array([R[t-1], R[t]]), np.array([PET[t-1], PET[t]]))
    else:
        ensemble = forecast_ensemble(ensemble, Param, np.array([R[t]]), np.array([PET[t]]))

    # Update step (assimilate only S1)
    if not np.isnan(observations[t]):
        ensemble = update_ensemble(ensemble, np.array([observations[t]]), obs_error)
    
    # Store the assimilated S1 (ensemble mean)
    assimilated_soil_moisture[t] = np.nanmean(ensemble[:, 0]) # Extract S1 only


cosmos_soil_moisture = pd.read_csv(I_dir+'SM_HOLLN.csv') #COSMOS soil moisture
cosmos_soil_moisture = cosmos_soil_moisture['soil_moisture'].to_numpy()  # COSMOS soil moisture
cosmos_soil_moisture = (cosmos_soil_moisture/100)
cosmos_soil_moisture = np.tile(cosmos_soil_moisture, 2)


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
# Define the start index for the second half
half_index = len(true_soil_moisture) // 2  
plt.figure(figsize=(10, 6))
# Slice all time series to keep only the second half
plt.plot(true_soil_moisture[half_index:], label="Simulated Soil Moisture", linestyle='dashed')
plt.plot(observations[half_index:], label="Observations - SAR data", linestyle='none', marker='o', markersize=4)
#plt.plot(assimilated_soil_moisture[half_index:], label="Assimilated Soil Moisture", color='red')
plt.plot(cosmos_soil_moisture[half_index:], label="COSMOS Soil Moisture", color='black')
'''
plt.plot(true_soil_moisture, label="True Soil Moisture", linestyle='dashed')
plt.plot(observations, label="Observations - SAR data", linestyle='none', marker='o', markersize=4)
#plt.plot(assimilated_soil_moisture, label="Assimilated Soil Moisture", color='red')
plt.plot(cosmos_soil_moisture, label="COSMOS Soil Moisture", color='black')
'''
plt.legend()
plt.xlabel("Time Step")
plt.ylabel("Surface soil moisture - top 10 cm (%)")
plt.title("Soil moisture without assimilation")
plt.show()


#%% Plot results with assimilation
plt.figure(figsize=(10, 6))
plt.plot(true_soil_moisture[half_index:], label="Simulated Soil Moisture", linestyle='dashed')
plt.plot(observations[half_index:], label="Observations - SAR data", linestyle='none', marker='o', markersize=4)
plt.plot(assimilated_soil_moisture[half_index:], label="Assimilated Soil Moisture", color='red')
plt.plot(cosmos_soil_moisture[half_index:], label="COSMOS Soil Moisture", color='black')
'''
plt.plot(true_soil_moisture, label="True Soil Moisture", linestyle='dashed')
plt.plot(observations, label="Observations - SAR data", linestyle='none', marker='o', markersize=4)
plt.plot(assimilated_soil_moisture, label="Assimilated Soil Moisture", color='red')
plt.plot(cosmos_soil_moisture, label="COSMOS Soil Moisture", color='black')
'''
plt.legend()
plt.xlabel("Time Step")
plt.ylabel("Surface soil moisture - top 10 cm (%)")
plt.title("Soil moisture assimilation with EnKF at Hollin Hill")


# Add a text box with the NSE and RMSE values
textstr = '\n'.join((
    f'NSE model: {NSE_mod:.2f}',
    f'RMSE model: {RMSE_mod:.2f}',
    f'NSE assimilation: {NSE_da:.2f}',
    f'RMSE assimilation: {RMSE_da:.2f}'
))

# Adjust the position and formatting of the text box
plt.gca().text(0.02, 0.03, textstr, transform=plt.gca().transAxes, fontsize=10,
               verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))

plt.show()

