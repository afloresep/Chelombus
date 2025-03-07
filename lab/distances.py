# Script to test everything related to distances in the 42 dimensional space in respect of the 3D space. 
import joblib
import numpy as np
import sys
import os
from scipy.stats import spearmanr
# Get the absolute path of dynamic_tmap/src
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dynamic_tmap", "src"))
sys.path.append(base_dir)
from fingerprint_calculator import FingerprintCalculator, calculate_mqn_fp
# from dynamic_tmap.src.fingerprint_calculator import FingerprintCalculator, calculate_mqn_fp
def get_smiles_list(): 
    # Load the smiles from file
    smiles_list = []
    with open("/Users/afloresep/work/chelombus/data/nn_search.txt", "r") as file: 
        for line in file:
            smiles_list.append(line)
    return smiles_list

def calculate_fg(smiles_list):
    fp_calculator = FingerprintCalculator(smiles_list=smiles_list, fingerprint_type="mqn")
    fp = fp_calculator.calculate_fingerprints()

    return fp

if __name__=="__main__":
    # Get smiles from file
    smiles_list = get_smiles_list()
    
    # Get 42 dimensional array from smiles_lsit
    print("Computing 42D fingerprints in parallel...")
    fp_42d = calculate_fg(smiles_list)
    print(f"Done. Shape of fingerprints array: {fp_42d.shape}")


    # Reduce 42D fingerprints to 3D
    #Load ipca model
    ipca_model = joblib.load("/Users/afloresep/work/chelombus/backend/data/ipca_model.joblib")
    print("Transforming fingerprints..:")
    fp_3d = ipca_model.transform(fp_42d)

    # Reference molecule 
    ref_fingerprint = calculate_mqn_fp("C#CC(C)(C)N(C(=O)C1=C[C@H](O)[C@H](NC(=O)C2=NOC3(CCC3)C2)C1)C1CC1")
    ref_fingerprint_3d = ipca_model.transform(ref_fingerprint.reshape(1, -1))

    # compute distances
    print("\n COmputing L1 and L2 distances in 42D")
    distances_42d_l2 = np.linalg.norm(fp_42d - ref_fingerprint, axis=1)
    distances_42d_l1 = np.linalg.norm(fp_42d- ref_fingerprint, ord=1, axis=1)


    # Sort them from smallest to largest distance
    print("Sorting 42D...")
    sorted_indices_42d_l2 = np.argsort(distances_42d_l2)
    sorted_indices_42d_l1 = np.argsort(distances_42d_l1)

    sorted_distances_42d_l2 = distances_42d_l2[sorted_indices_42d_l2]
    sorted_distances_42d_l1 = distances_42d_l1[sorted_indices_42d_l1]


    # ---------------------------------------
    # E) Compute L1 and L2 distances in 3D
    # ---------------------------------------
    print("\nComputing L1 and L2 distances in 3D...")
    distances_3d_l2 = np.linalg.norm(fp_3d - ref_fingerprint_3d, axis=1)
    distances_3d_l1 = np.linalg.norm(fp_3d- ref_fingerprint_3d, ord=1, axis=1)

    sorted_indices_3d_l2 = np.argsort(distances_3d_l2)
    sorted_indices_3d_l1 = np.argsort(distances_3d_l1)

    sorted_distances_3d_l2 = distances_3d_l2[sorted_indices_3d_l2]
    sorted_distances_3d_l1 = distances_3d_l1[sorted_indices_3d_l1]


    # ---------------------------------------
    # E) Get the rank (index) of each molecule by each distance
    #    "Rank" means the position of the molecule when sorted 
    #    by increasing distance (0 = closest, max = furthest).
    # ---------------------------------------
    sorted_indices_42d_l2 = np.argsort(distances_42d_l2)  # array of molecule indices from smallest to largest L2 distance
    sorted_indices_42d_l1 = np.argsort(distances_42d_l1)
    sorted_indices_3d_l2  = np.argsort(distances_3d_l2)
    sorted_indices_3d_l1  = np.argsort(distances_3d_l1)

    # To convert these sorted indices into ranks for each molecule:
    rank_l2_42d = np.empty_like(sorted_indices_42d_l2)
    rank_l1_42d = np.empty_like(sorted_indices_42d_l1)
    rank_l2_3d  = np.empty_like(sorted_indices_3d_l2)
    rank_l1_3d  = np.empty_like(sorted_indices_3d_l1)

    # Fill in the rank arrays with the position of each molecule.
    rank_l2_42d[sorted_indices_42d_l2] = np.arange(len(sorted_indices_42d_l2))
    rank_l1_42d[sorted_indices_42d_l1] = np.arange(len(sorted_indices_42d_l1))
    rank_l2_3d[sorted_indices_3d_l2]   = np.arange(len(sorted_indices_3d_l2))
    rank_l1_3d[sorted_indices_3d_l1]   = np.arange(len(sorted_indices_3d_l1))

    # ---------------------------------------
    # F) Build a table (DataFrame) with all the requested columns
    # ---------------------------------------
    import pandas as pd
    df = pd.DataFrame({
        "molecule": smiles_list,
        "L1_3D": distances_3d_l1,
        "L1_42D": distances_42d_l1,
        "L2_3D": distances_3d_l2,
        "L2_42D": distances_42d_l2,
        "index L1_42D": rank_l1_42d,
        "index L1_3D": rank_l1_3d,
        "index L2_42D": rank_l2_42d,
        "index L2_3D": rank_l2_3d
    }) 

    df.to_csv("distances.csv", index=False)