from fastapi import FastAPI, Request
from typing import List, Union
from pydantic import BaseModel # To serialize the data into JSON for API responses
from fastapi.middleware.cors import CORSMiddleware
import re
import pandas as pd
import joblib
import argparse
from mhfp.encoder import MHFPEncoder
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
import numpy as np
#TODO: Fix import
# from chelombus.dynamic_tmap.src.fingerprint_calculator import calculate_mqn_fp
import tmap as tm
from rdkit import Chem

app = FastAPI()

# Allow requests from the local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  #TODO: Suitable for develop, restricted in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def calculate_mqn_fp(smiles: str) -> np.array:
    """Calculate MQN fingerprint for a single SMILES string.
    :param smiles: SMILES string for the molecule
    :return: np.array of fingerprint
    """
    try:
        fingerprint = rdMolDescriptors.MQNs_(Chem.MolFromSmiles(smiles))
        return np.array(fingerprint)
    except Exception as e:
        print(f"Error processing SMILES '{smiles}': {e}")
        return None
# Base Model for Coordinates 
class Coordinates(BaseModel):
    x: float
    y: float
    z: float

class ResponseModel(BaseModel):
    """
    Class to define the API response. Should contain a list of Coordinates objects
    under the coordinates key
    """
    coordinates: Union[Coordinates, List[Coordinates]]


def _get_coordinates_by_node_number(node_number) -> tuple:
    """
    Searches the CSV file for a row where 'cluster_name' contains 'cluster_<node_number>_TMAP.html'
    and returns the corresponding x, y, and z coordinates.
    
    :param csv_file: The path to the CSV file. This csv contains the coordinates (x,y,z) in the Primary TMAP
    for each cluster node. This way we can map our node_number with the actual location of that node in the 
    Primary TMAP HTML file.  
    :param node_number: The integer node number to search for.

    A tuple (x, y, z) if found, or None if not found.
    """
    # Load the CSV file
    cluster_coordinates_df= pd.read_csv('database_cluster_coordinates.csv')
    
    # Construct the target cluster name
    target_name = f"cluster_{node_number}_TMAP.html"
    
    # Search for the row with the matching cluster name
    matching_row = cluster_coordinates_df[cluster_coordinates_df['cluster_name'] == target_name]
    
    if not matching_row.empty:
        # Extract the x, y, z values
        x = matching_row.iloc[0]['x']
        y = matching_row.iloc[0]['y']
        z = matching_row.iloc[0]['z']
        return (x, y, z)
    else:
        return None  # Return None if no matching row is found
    

def find_cluster_from_smiles(csv_file: str, smiles: str):
    """
    Returns the (x, y, z) coordinate for the first matching cluster ID
    found in `csv_file` for the input SMILES.
    :param csv_file: CSV file that includes the PCA ranges for each cluster. We find the cluster based on this.
    :param smiles: smiles string to find its cluster

    In order to get the PCA ranges from the database where we have all our data we can do it with the following query
    "SELECT cluster_id, MIN(PCA_1) AS min_PCA_1, MAX(PCA_1) AS max_PCA_1, MIN(PCA_2) AS min_PCA_2, MAX(PCA_2) 
    AS max_PCA_2, MIN(PCA_3) AS min_PCA_3, MAX(PCA_3) AS max_PCA_3 FROM clustered_enamine 
    GROUP BY cluster_id ORDER BY cluster_id ASC"
    as this can take a while we load it from a csv file with that data
    """
    # 1) Load the CSV file
    cluster_ranges_dataframe = pd.read_csv(csv_file)

    # 2) Load your trained PCA model
    pca = joblib.load("ipca_model.joblib")

    # 3) Compute PCA coordinates
    fingerprint = calculate_mqn_fp(smiles=smiles).reshape(1, -1)
    pca_coordinates = pca.transform(fingerprint).reshape(3, 1)

    # 4) Filter dataframe
    matching_clusters = cluster_ranges_dataframe[
        (cluster_ranges_dataframe['min_PCA_1'] <= np.float64(pca_coordinates[0])) & (cluster_ranges_dataframe['max_PCA_1'] >= np.float64(pca_coordinates[0])) &
        (cluster_ranges_dataframe['min_PCA_2'] <= np.float64(pca_coordinates[1])) & (cluster_ranges_dataframe['max_PCA_2'] >= np.float64(pca_coordinates[1])) &
        (cluster_ranges_dataframe['min_PCA_3'] <= np.float64(pca_coordinates[2])) & (cluster_ranges_dataframe['max_PCA_3'] >= np.float64(pca_coordinates[2]))
    ]

    # 5) Get the list of matching cluster IDs (node_number)
    if matching_clusters.empty:
        return None

    node_numbers = matching_clusters['cluster_id'].tolist()

    # Just use the first matched node_number
    node_number = node_numbers[0]

    coordinate = _get_coordinates_by_node_number(node_number)
    # coordinate is either (x, y, z) or None
    
    return coordinate

@app.post("/api/find-cluster-from-jmse", response_model=ResponseModel)
async def find_cluster_from_jmse(smiles: dict) -> ResponseModel:
    """
    Function to find the cluster of a smiles that was pasted or drawn in the
    JMSE box in the frontend

    :param smiles: JSON body from the frontend containing the SMILES 
    string from the JMSE box
    """
    point = []
    smiles_coordinate = find_cluster_from_smiles(
        csv_file="cluster_ranges.csv", 
        smiles=smiles['smiles'], 
    )

    if smiles_coordinate:
        x, y, z = smiles_coordinate
        point.append(Coordinates(x=x, y=y, z=z))
    else: 
        # If there's no match, return None
        return ResponseModel(coordinates=None)

    return ResponseModel(coordinates=point)
    
@app.post("/api/find-cluster-from-file", response_model=ResponseModel)
async def find_cluster_from_file(request: dict) -> ResponseModel:
    """
    :param request: JSON body from the frontned.
    """
    points = []

    file_content = request.get("file_content", "")
    smiles_list= file_content.split("\n")

    for smiles in smiles_list:
        neighbor_smiles = smiles.strip()
        if not neighbor_smiles:
            continue  # Skip empty lines

        # find_cluster_from_smiles will return either (x, y, z) or None
        smiles_coordinate = find_cluster_from_smiles(
            csv_file="cluster_ranges.csv", 
            smiles=neighbor_smiles,
        )

        # If coordinate is found, append to our 'points' list
        if smiles_coordinate:
            x, y, z = smiles_coordinate
            points.append(Coordinates(x=x, y=y, z=z))
        else:
            # If there's no match, you could choose to do nothing
            return ResponseModel(coordinates=None)
     
    return ResponseModel(coordinates=points)