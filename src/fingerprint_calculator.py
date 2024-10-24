from multiprocessing import Pool
from config import N_JOBS
from rdkit import Chem
from mhfp.encoder import MHFPEncoder
from rdkit.Chem import rdMolDescriptors                                                           
import numpy as np

def init_worker(): 
    """iniitalizer for worker processes to set up global variables"""
    global encoder
    encoder = MHFPEncoder(512)

def _calculate_fingerprint(smiles): 
    """Calculate fp for a single SMILES string"""
    try:
        # mol = Chem.MolFromSMiles(smiles)
        # if mol is None: 
        #     raise ValueError(f"Invalid SMILES string: {smiles}")
        fingerprint = rdMolDescriptors.MQNs_(Chem.MolFromSmiles(smiles))
        return np.array(fingerprint)
    
    except Exception as e: 
        print(f"error processing SMILES '{smiles}: {e}")
        return None

class FingerprintCalculator:
    def calculate_fingerprints(self, smiles_list):
        with Pool(processes=N_JOBS, initializer=init_worker) as pool: 
            fingerprints = pool.map(_calculate_fingerprint, smiles_list)
        return np.array(fingerprints)
    # handle exception for invalid SMILES string