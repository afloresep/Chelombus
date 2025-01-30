from multiprocessing import Pool
from functools import partial
from typing import Optional
from config import N_JOBS
from rdkit import Chem
from mhfp.encoder import MHFPEncoder
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
import numpy as np
import tmap as tm
import hashlib
from mapchiral.mapchiral import encode as mapc_enc
import numpy.typing as npt

# Define the fingerprint functions at the module level
def calculate_mhfp_fp(smiles: str, permutations: int) -> np.array:
    """Calculate MHFP fingerprint for a single SMILES string.
    :param smiles: SMILES string for the molecule
    :return: np.array of fingerprint"""
    try:
        encoder = MHFPEncoder(permutations)
        return np.array(encoder.encode(smiles))
    except Exception as e:
        print(f"Error processing SMILES '{smiles}': {e}")
        return None

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

def calculate_morgan_fp(smiles: str, radius: int, fp_size: int) -> np.array:
    """Calculate Morgan fingerprint for a single SMILES string.
    :param smiles: SMILES string for the molecule
    :return: np.array of fingerprint"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string.")

        # Generate standard Morgan fingerprint
        morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=fp_size)
        return np.array(morgan_fp)

    except Exception as e:
        print(f"Error processing SMILES '{smiles}': {e}")
        return None

def calculate_mapc_fp(smiles: str, radius: int, fp_size: int) -> np.array:
    """Calculate MAP Chiral fingerprint for a single SMILES string.
    :param smiles: SMILES string for the molecule
    :return: np.array of fingerprint"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string.")

        fingerprint = mapc_enc(
            mol,
            max_radius=radius,
            n_permutations=fp_size,
            mapping=False
        )
        return np.array(fingerprint)

    except Exception as e:
        print(f"Error processing SMILES '{smiles}': {e}")
        return None

class FingerprintCalculator:
    def __init__(
        self,
        smiles_list: list,
        fingerprint_type: str,
        permutations: Optional[int] = 512,
        fp_size: Optional[int] = 1024,
        radius: Optional[int] = 2,
    ):
        self.smiles_list = smiles_list
        self.fingerprint_type = fingerprint_type.lower()
        self.permutations = permutations
        self.fp_size = fp_size
        self.radius = radius

        # Map fingerprint types to functions
        self.fingerprint_function_map = {
            'mhfp': calculate_mhfp_fp,
            'mqn': calculate_mqn_fp,
            'mapc': calculate_mapc_fp,
            'morgan': calculate_morgan_fp,
        }

    def calculate_fingerprints(self) -> npt.NDArray:
        # Get the appropriate fingerprint function
        func_to_apply = self.fingerprint_function_map.get(self.fingerprint_type)

        if func_to_apply is None:
            raise ValueError(f"Unsupported fingerprint type: {self.fingerprint_type}")

        # Prepare the function with necessary parameters
        if self.fingerprint_type == 'mhfp':
            func_to_apply = partial(func_to_apply, permutations=self.permutations)
        elif self.fingerprint_type == 'morgan':
            func_to_apply = partial(func_to_apply, radius=self.radius, fp_size=self.fp_size)
        elif self.fingerprint_type == 'mapc':
            func_to_apply = partial(func_to_apply, radius=self.radius, fp_size=self.fp_size)
        # No need to modify 'mqn' as it doesn't require extra parameters

        # Use multiprocessing Pool to calculate fingerprints in parallel
        with Pool(processes=16) as pool:
            fingerprints = pool.map(func_to_apply, self.smiles_list)

        return np.array(fingerprints)

"""
Example of usage:
# Initiate
fp_calculator = FingerprintCalculator(your_dataframe['smiles'], 'mqn')
# This will calculate the MQN fingerprints.

fp_calculator = FingerprintCalculator(smiles_list, 'mhfp', permutations=1024)
# This will calculate the MHFP with 1024 permutations (default is 512)

# Calculate fingerprints
fingerprints = fp_calculator.calculate_fingerprints()
# Returns np.array with the fingerprint vectors of same length as smiles_list
"""
