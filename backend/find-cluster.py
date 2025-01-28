from fastapi import FastAPI, Request
from typing import List
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import re
import pandas as pd
import joblib
import argparse
from mhfp.encoder import MHFPEncoder
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
import numpy as np
import tmap as tm
from rdkit import Chem

app = FastAPI()

# Allow requests from the local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Suitable for develop, restricted in production TODO
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Base Model for Coordinates 
class Coordinates(BaseModel):
    x: float
    y: float
    z: float

class ResponseModel(BaseModel):
    coordinates: List[Coordinates]


def calculate_mqn_fp(smiles: str) -> np.array:
    """
    Calculate MQN fingerprint for a single SMILES string.
    :param smiles: SMILES string for the molecule
  
    :return: np.array of fingerprint
    """
    try:
        fingerprint = rdMolDescriptors.MQNs_(Chem.MolFromSmiles(smiles))
        return np.array(fingerprint)
    except Exception as e:
        print(f"Error processing SMILES '{smiles}': {e}")
        return None
import pandas as pd
import numpy as np
import joblib

import re

def get_coordinates_by_node_number(node_number):
    """
    Searches the CSV file for a row where 'cluster_name' contains 'cluster_<node_number>_TMAP.html'
    and returns the corresponding x, y, and z coordinates.
    
    Args:
    - csv_file: The path to the CSV file.
    - node_number: The integer node number to search for.

    Returns:
    - A tuple (x, y, z) if found, or None if not found.
    """
    # Load the CSV file
    df = pd.read_csv('database_coordinates.csv')
    
    # Construct the target cluster name
    target_name = f"cluster_{node_number}_TMAP.html"
    
    # Search for the row with the matching cluster name
    matching_row = df[df['cluster_name'] == target_name]
    
    if not matching_row.empty:
        # Extract the x, y, z values
        x = matching_row.iloc[0]['x']
        y = matching_row.iloc[0]['y']
        z = matching_row.iloc[0]['z']
        return (x, y, z)
    else:
        return None  # Return None if no matching row is found
    

def find_clusters_og(csv_file: str, smiles: str, js_file: str):
    # 1) Load the CSV file
    df = pd.read_csv(csv_file)

    # 2) Load your trained PCA model
    pca = joblib.load("ipca_model.joblib")

    # 3) Compute PCA coordinates
    fingerprint = calculate_mqn_fp(smiles=smiles).reshape(1, -1)
    pca_coordinates = pca.transform(fingerprint).reshape(3, 1)

    # 4) Filter dataframe
    matching_clusters = df[
        (df['min_PCA_1'] <= np.float64(pca_coordinates[0])) & (df['max_PCA_1'] >= np.float64(pca_coordinates[0])) &
        (df['min_PCA_2'] <= np.float64(pca_coordinates[1])) & (df['max_PCA_2'] >= np.float64(pca_coordinates[1])) &
        (df['min_PCA_3'] <= np.float64(pca_coordinates[2])) & (df['max_PCA_3'] >= np.float64(pca_coordinates[2]))
    ]

    # 5) Get the list of matching cluster IDs (node_number)
    node_numbers = matching_clusters['cluster_id'].tolist()

    coordinate = get_coordinates_by_node_number(node_numbers[0])

    print(coordinate) # printed like this (89.885, 201.782, 375.0)


def find_clusters(csv_file: str, smiles: str, js_file: str):
    """
    Returns the (x, y, z) coordinate for the first matching cluster ID
    found in `csv_file` for the input SMILES.
    """
    # 1) Load the CSV file
    df = pd.read_csv(csv_file)

    # 2) Load your trained PCA model
    pca = joblib.load("ipca_model.joblib")

    # 3) Compute PCA coordinates
    fingerprint = calculate_mqn_fp(smiles=smiles).reshape(1, -1)
    pca_coordinates = pca.transform(fingerprint).reshape(3, 1)

    # 4) Filter dataframe
    matching_clusters = df[
        (df['min_PCA_1'] <= np.float64(pca_coordinates[0])) & (df['max_PCA_1'] >= np.float64(pca_coordinates[0])) &
        (df['min_PCA_2'] <= np.float64(pca_coordinates[1])) & (df['max_PCA_2'] >= np.float64(pca_coordinates[1])) &
        (df['min_PCA_3'] <= np.float64(pca_coordinates[2])) & (df['max_PCA_3'] >= np.float64(pca_coordinates[2]))
    ]

    # 5) Get the list of matching cluster IDs (node_number)
    if matching_clusters.empty:
        return None

    node_numbers = matching_clusters['cluster_id'].tolist()

    # Just use the first matched node_number
    node_number = node_numbers[0]

    coordinate = get_coordinates_by_node_number(node_number)
    # coordinate is either (x, y, z) or None
    
    return coordinate

@app.post("/api/find-cluster", response_model=ResponseModel)
async def find_cluster(smiles: dict):
     # We'll accumulate all (x, y, z) in a list of Coordinates
     points = []

     # Read neighbors.txt (one SMILES per line)
     with open('neighbors.txt', 'r') as f:
         for line in f:
             neighbor_smiles = line.strip()
             if not neighbor_smiles:
                 continue  # Skip empty lines

             # find_clusters will return either (x, y, z) or None
             coordinate = find_clusters(
                 csv_file="cluster_ranges.csv",            # or args.csv_file if you parse from CLI
                 smiles=neighbor_smiles,
                 js_file="../frontend/representatives_TMAP.js"
             )

             # If coordinate is found, append to our 'points' list
             if coordinate:
                 x, y, z = coordinate
                 points.append(Coordinates(x=x, y=y, z=z))
             else:
                 # If there's no match, you could choose to do nothing
                 # or append a placeholder
                 pass
     
     return ResponseModel(coordinates=points)

# @app.post("/api/find-cluster", response_model=ResponseModel)
# async def find_cluster(smiles: dict):
#     # Implement your logic to generate multiple (x, y, z) points based on the SMILES string
#     # For demonstration, returning some dummy points
#     points = [
#         Coordinates(x=510.0, y=220.0, z=350.0),
#         Coordinates(x=540.0, y=250.0, z=350.0),
#     ]
#     return ResponseModel(coordinates=points)


if __name__ == "__main__":
    #  parser = argparse.ArgumentParser(description="Find clusters for a given point.")
    #  parser.add_argument("csv_file", help="Path to the CSV file containing cluster ranges")
   
    #  args = parser.parse_args()
    find_cluster()
    #  list_clusters = []
    #  with open('neighbors.txt', 'r') as f:
    #      for neighbor in f:
    #          neighbor = neighbor.strip()  # Remove any leading/trailing whitespace, such as newlines
    #          neighbor_cluster = find_clusters(args.csv_file, smiles=neighbor, js_file='../frontend/representatives_TMAP.js')
    #          list_clusters.append(str(neighbor_cluster))

