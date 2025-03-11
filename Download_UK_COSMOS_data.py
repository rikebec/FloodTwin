#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:22:30 2024

@author: Rike Becker

This code is based on pre-written functions for accessing COSMOS data and slightly adapted for the work within the FloodTwin project.
It is is used to download precipitation, potential evapotranspiration and soil moisture data at specific UK-COMSOS sites.
This data is then used in the data assimilation procedure.

Credit is given to the code published by CEH:
# Please see https://cosmos-api.ceh.ac.uk/python_examples for code examples
# Please see https://cosmos-api.ceh.ac.uk/docs for more details
"""

from datetime import datetime
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%###########################################################################
########################## Defining Functions ################################
##############################################################################

BASE_URL = 'https://cosmos-api.ceh.ac.uk'

def get_api_response(url, csv=False):
    # Send request and read response
    print(url)
    response = requests.get(url)

    if csv:
        return response
    else:
        # Decode from JSON to Python dictionary
        return json.loads(response.content)
    

def get_collection_parameter_info(params):
    df = pd.DataFrame.from_dict(params)
    df = df.T[['label', 'description', 'unit', 'sensorInfo']]

    df['unit_symbol'] = df['unit'].apply(lambda x: x['symbol']['value'])
    df['unit_label'] = df['unit'].apply(lambda x: x['label'])
    df['sensor_depth'] = df['sensorInfo'].apply(lambda x: None if pd.isna(x) else x['sensor_depth']['value'])

    df = df.drop(['sensorInfo', 'unit'], axis=1)

    return df

#%%###########################################
######### Downloading a collection ###########
##############################################

collection_30M_url = f'{BASE_URL}/collections/30M'
collection_30M_meta = get_api_response(collection_30M_url)

# Get the information about the parameter names from the metadata dictionary
collection_30M_params = collection_30M_meta['parameter_names']
collection_30M_params_df = get_collection_parameter_info(collection_30M_params)

def read_json_collection_data(json_response):

    master_df = pd.DataFrame()

    for site_data in json_response['coverages']:
        # Read the site ID
        site_id = site_data['dct:identifier']

        # Read the time stamps of each data point
        time_values = pd.DatetimeIndex(site_data['domain']['axes']['t']['values'])

        # Now read the values for each requested parameter at each of the time stamps
        param_values = {param_name: param_data['values'] for param_name, param_data in site_data['ranges'].items()}

        # And put everything into a dataframe
        site_df = pd.DataFrame.from_dict(param_values)
        site_df['datetime'] = time_values
        site_df['site_id'] = site_id
        
        #site_df = site_df.set_index(['datetime', 'site_id'])
        master_df = pd.concat([master_df, site_df])

    return master_df

#%%#########################################################
#### query data for specific dates and specific sites ######
############################################################

def format_datetime(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

# An example of using specific start and end dates
start_date = format_datetime(datetime(2023, 1, 1))
end_date = format_datetime(datetime(2024, 11, 19))

query_date_range = f'{start_date}/{end_date}'

site_ids = ['HOLLN']
query_url = f'{BASE_URL}/collections/1D/locations/{(site_ids[0])}?datetime={query_date_range}' #daily data (1D) replace with 30M if 30-min data is wanted
resp = get_api_response(query_url)

df = read_json_collection_data(resp)
df.head()
df.keys()
df.shape

#%% create and save table
variables = ['datetime', 'tdt1_vwc', 'tdt2_vwc'] # for HOLLN only data is available from 10cm depth, for SPENF and RISEH in 5cm depth
data = df[variables]
data_table = pd.DataFrame(data)
data_table['datetime'] = pd.to_datetime(data_table['datetime']).dt.date
data_table['soil_moisture'] = (data_table.tdt1_vwc+data_table.tdt2_vwc)/2
print(data_table)

sns.set_style("whitegrid")
plt.figure(figsize=(10, 5))
sns.lineplot(x=data_table['datetime'], y=data_table['soil_moisture'], color='b')
plt.xlabel("Date")
plt.ylabel("Soil Moisture [mm]")
plt.title("Time")
plt.xticks(rotation=45)
plt.show()

data_table = data_table.iloc[:, [0, 3]]

data_table.to_csv('/.../SM_HOLLN.csv', index=False)

#%%######################################
#### query data for specific params #####
#########################################

# Specify a subset of paramater names
param_names = ['tdt3_vwc', 'tdt4_vwc'] # these change between daily and 30M resolution. check keys!

query_url = f'{BASE_URL}/collections/1D/locations/{(site_ids[0])}?datetime={query_date_range}&parameter-name={",".join(param_names)}'
resp = get_api_response(query_url)

df = read_json_collection_data(resp)
df.head()

#%% 
df['datetime'] = pd.to_datetime(df['datetime'])
#df['datetime'] = df['datetime'].dt.date
pd.api.types.is_datetime64_any_dtype(df['datetime']) # to check if datetime is in date format

#%%#############
#### plots #####
################

df_sm_precip = df[["tdt2_vwc", "precip"]]
axs = df_sm_precip.plot(figsize=(12,8), subplots=True)
plt.show()
#fig.savefig("VWC_UK_COSMOS.png")
