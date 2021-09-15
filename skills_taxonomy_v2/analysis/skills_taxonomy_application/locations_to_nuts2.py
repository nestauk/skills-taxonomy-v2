"""
Map Job ID locations to nuts codes and nuts names

In this script, latitude and longitude metadata associated with job ids are mapped onto nuts codes. 
It outputs a dictionary to S3 where the key is the associated job id and its values are a list of geographies incl.
lat, long, nuts code, nuts name, county and country. 

Usage:

python -i skills_taxonomy_v2/analysis/skills_taxonomy_application/locations_to_nuts2.py --config_path 'skills_taxonomy_v2/config/skills_taxonomy_application/2021.09.14.yaml'

"""
from argparse import ArgumentParser
import logging
import yaml

from tqdm import tqdm
import pandas as pd
import boto3
import os
import re
import requests
import numpy as np
import pyproj
from urllib.request import urlretrieve
from zipfile import ZipFile
from shapely.geometry import Point

import geopandas as gpd
from geopandas import GeoDataFrame

from skills_taxonomy_v2.getters.s3_data import load_s3_data, save_to_s3
from skills_taxonomy_v2 import PROJECT_DIR, BUCKET_NAME

logger = logging.getLogger(__name__)
s3 = boto3.resource("s3")

def parse_arguments(parser):

    parser.add_argument(
        "--config_path",
        help="Path to config file",
        default="skills_taxonomy_v2/config/skills_taxonomy_application/2021.09.14.yaml",
    )

    return parser.parse_args()


def get_nuts_shapefile(shape_url, shapefile_path, nuts_file):
    """gets relevant nuts file and converts it to geopandas dataframe.

    Args:
            shape_url: URL of zipped geojson file
            shapefile_path: local path to output relevant nuts geojson
            nuts_file: name of relevant nuts geojson file  

    Returns:
    geo dataframe of UK nuts2 codes. 
    """    
    full_shapefile_path = str(PROJECT_DIR) + shapefile_path
    if not os.path.isdir(full_shapefile_path):
        os.mkdir(full_shapefile_path)

    zip_path , _ = urlretrieve(shape_url)
    with ZipFile(zip_path, 'r') as zip_files:
        for zip_names in zip_files.namelist():
            if zip_names == nuts_file:
                zip_files.extract(zip_names, path=full_shapefile_path)
                nuts_geo = gpd.read_file(full_shapefile_path + nuts_file)
                nuts_geo = nuts_geo[nuts_geo['CNTR_CODE'] == 'UK'].reset_index(drop = True)
    
    return nuts_geo

def get_job_adverts_with_skills(sentence_outputs_path, bucket_name):
    """gets job adverts with skills.

    Args:
        sentence_outputs_path: s3 filepath to sentences we have skills for.
        bucket_name: name of relevant S3 bucket. 
            
    Returns:
        A dictionary where the key is a job id we have skills for 
        and its value is a list of location information incl. 
        lat/long, country name and county name associated to the 
        job id. 
    """
    sentence_data = load_s3_data(s3, bucket_name, sentence_outputs_path)
    sentence_data = pd.DataFrame(sentence_data)
    sentence_data = sentence_data[sentence_data['Cluster number']!=-1]
    # The job adverts that we have skills for
    job_ids = set(sentence_data['job id'].tolist())
    # Load the job advert location data - only for the job adverts we have skills for
    # Each one is really big (77secs to load)! There are 13 files
    job_id_loc_dict = {}
    for i, file_name in enumerate(tqdm(range(0,13))):
        file_loc_dict = load_s3_data(
            s3,
            bucket_name,
            f'outputs/tk_data_analysis/metadata_location/{file_name}.json')
        job_id_loc_dict.update({k:v for k,v in file_loc_dict.items() if k in job_ids})

    return job_id_loc_dict

def map_job_adverts_with_skills_to_nuts(job_id_loc_dict, nuts_geo, epsg = int):
    """maps lat/longs associated to job adverts with skills to nuts codes.

    Args:
        job_id_loc_dict: output from get_job_adverts_with_skills()
        nuts_geo: output from get_nuts_shapefile()
        epsg: Coordinate Reference System (CRS) code   
            
    Returns:
        A dictionary where the key is a job id we have skills for 
        and its value is a list of location information incl. 
        lat/long, country and county name, nuts code and nuts name 
        associated to the job id. 
    """
    df = pd.DataFrame(job_id_loc_dict).T
    df[['lat', 'long']] = df[1].str.split(',', 1, expand=True).astype('float64')
    geometry = [Point(xy) for xy in zip(df.long, df.lat)]

    gdf = GeoDataFrame(df, geometry=geometry)
    gdf = gdf.set_crs(epsg=epsg, inplace=True)
    gdf = gdf[(gdf.lat.notna()) | (gdf.long.notna())] #get rid of nas 
    job_nuts = gpd.sjoin(gdf, nuts_geo, how='left')

    return job_nuts[[0, 1, 2, 3, 'NUTS_ID', 'NUTS_NAME']].T.to_dict('list')

if __name__ == "__main__":

    parser = ArgumentParser()
    args = parse_arguments(parser)

    with open(args.config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    #get parameters
    params = config["params"]
    shape_url = params["shape_url"]
    shapefile_path = params["output_path"]
    nuts_file = params["nuts_file"]
    sentence_outputs_path = params["sentence_outputs_path"]
    epsg = params["epsg"]
    sentences_outputs_nuts_path = params["sentences_outputs_nuts_path"]

    nuts_geo = get_nuts_shapefile(shape_url, shapefile_path, nuts_file)
    job_id_loc_dict = get_job_adverts_with_skills(sentence_outputs_path, BUCKET_NAME)
    job_adverts_nuts = map_job_adverts_with_skills_to_nuts(job_id_loc_dict, nuts_geo, epsg)
    save_to_s3(s3, BUCKET_NAME, job_adverts_nuts, sentences_outputs_nuts_path)