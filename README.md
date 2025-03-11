# Data assimilation in a digital twin framework
## Content of the repository
This repository contains code to assimilate SAR-derived soil moisture data into a conceptual hydrological model employing an Ensemble Kalman Filter. 
It serves as a prove of concept to show how observations from different platforms can be used to improve model performance in a digital twin framework. The code was created in the scope of the NERC funded UK research project 'FloodTwin'.
More information on the project can be found [**here**](https://www.hull.ac.uk/work-with-us/more/media-centre/news/2024/innovative-digital-twin-project-will-transform-flooding-forecasting-and-decision-making).


Be aware: the content will be frequently updated in the next days. (Last edits: March 11th, 2025)

## Point-based data assimilation
This code can be used to conduct a point-based data assimilation (e.g. assimilation of in-situ measurements). Here we assimilate soil moisture data into a simple conceptual hydrological model to update and improve soil moisture simulations. The soil moisture data is SAR-derived soil moisture from the Sentinel-1 mission and was downloaded from the [**Copernicus Land Monitoring Service**](https://land.copernicus.eu/en/products/soil-moisture/daily-surface-soil-moisture-v1.0).
The model is setup with precipitation and potential evaporation data from the UK-COSMOS site at Hollin Hill. 
To download the UK-COSMOS data, use the code 'Download_UK_Cosmos_data.py' published in this repository.

## Spatially distributed data assimilation
...
