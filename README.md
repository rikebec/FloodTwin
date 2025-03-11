# Data assimilation in a digital twin framework
## Content of the repository
This repository contains code to assimilate soil moisture data into a conceptual hydrological model employing an Ensemble Kalman Filter.
It serves as a prove of concept to show how observations from different platforms and of different spatial resolutions can be used to improve model performance in a digital twin framework. In such a framework it is often required to use various data products (e.g. from point-scale in-situ measurements to large-scale remotely sensed data) and use these data sets to continously update the forecasting performance of the digital twin. The code presented here, is a first step to integrate data sets of different spatial and temporal resolutions to update the performance of a hydrological model, whenever and wherever new data becomes available. This approach can be upscaled to a spatially distributed model and employed with more complex physically-based models, used in digital twin set-ups. The code was created in the scope of the NERC funded UK research project [**FloodTwin**](https://www.hull.ac.uk/work-with-us/more/media-centre/news/2024/innovative-digital-twin-project-will-transform-flooding-forecasting-and-decision-making).

![Image](images/Figure_1.png)
Be aware: the content will be frequently updated in the next days. (Last edits: March 11th, 2025)

## Point-based data assimilation
The codes for point-based data assimilation can be used to conduct data assimilation at a specific point of interest. Here we assimilate soil moisture data into a simple conceptual hydrological model to update and improve soil moisture simulations. One data set consists of SAR-derived soil moisture from the Sentinel-1 mission and was downloaded from the [**Copernicus Land Monitoring Service**](https://land.copernicus.eu/en/products/soil-moisture/daily-surface-soil-moisture-v1.0). The second data set consist of soil moisture in-situ measurements collected at UK-COSMOS sites. To download the UK-COSMOS data, use the code 'Download_UK_Cosmos_data.py' published in this repository. The model is setup with precipitation and potential evaporation data from the UK-COSMOS site at Hollin Hill, which can be downloaded using the same script (i.e. 'Download_UK_COSMOS_data.py'). 


## Spatially distributed data assimilation
...
