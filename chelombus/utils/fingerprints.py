#This module provides functionality to compute molecular fingerprints from SMILES strings 
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from rdkit import Chem, DataStructs
import os 
from typing import List
import numpy as np
import numpy.typing as npt
import logging
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors

logger = logging.getLogger(__name__)


def _calculate_mqn_fp(smiles: str, **params) -> npt.NDArray | None:
    """Calculate MQN fingerprint for a single SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning(
            "\nSMILES '%s' could not be parsed into a molecule. Skipping MQN fingerprint.", smiles
        )
        return None
    try:
        fingerprint = rdMolDescriptors.MQNs_(mol)
        return np.array(fingerprint, dtype=np.int16)
    except Exception as e:
        logger.warning("Error processing SMILES '%s': %s. Skipping entry.", smiles, e)
        return None
    

def _calculate_morgan_fp(smiles: str, **params) -> npt.NDArray | None:
    """
    Calculate a Morgan fingerprint for a single SMILES string.

    This function uses RDKit to convert the input SMILES string into a molecular object,
    then computes the Morgan fingerprint based on the provided parameters.

    Args:
        smiles (str): A valid SMILES representation of a molecule.
        **params: Keyword parameters required for fingerprint calculation.
            Expected keys:
                - fpSize (int): Size of the fingerprint (number of bits).
                - radius (int): Radius parameter for the Morgan algorithm.
                - to_numpy (bool): Convert fingerprint to numpy array. Default True,
                  otherwise returned as rdkit.DataStructs.cDataStructs.ExplicitBitVect

    Returns:
        np.array: An array representing the fingerprint, or None if the SMILES
            could not be parsed or an error occurred.
    """
    fpSize = params.get('fpSize')
    radius = params.get('radius')
    to_numpy = params.get('to_numpy', True)

    if fpSize is None or radius is None:
        raise ValueError("Missing required parameters: 'fpSize' and/or 'radius'.")

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"SMILES '{smiles}' could not be converted to a molecule. Skipping.")
            return None

        fp = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fpSize).GetFingerprint(mol)

        if to_numpy:
            fp_arr = np.zeros((fpSize,), dtype='uint8')
            DataStructs.ConvertToNumpyArray(fp, fp_arr)
            return fp_arr

        return fp

    except Exception as e:
        logger.error(f"Error processing SMILES '{smiles}': {e}")
        return None

class FingerprintCalculator:
    """
    A class to compute molecular fingerprints from a list of SMILES strings.
    Currently, only the 'morgan' and 'mqn' fingerprints type are supported.
    """
    def __init__(self):
        
        # Map fingerprint types to functions
        self.fingerprint_function_map = {
            #TODO: Support for other fingerprints
            'morgan': _calculate_morgan_fp,
            'mqn': _calculate_mqn_fp,
        }

    def FingerprintFromSmiles(self, smiles:List | str, fp:str, nprocesses:int | None = None,
                              return_valid_idx: bool = False, **params):
        """
        Generate fingerprints for a list of SMILES strings in parallel.

        The method selects the appropriate fingerprint function based on the 'fp' parameter,
        binds additional keyword parameters using functools.partial, and then applies the function
        across the SMILES list in a pool of worker processes.

        On Windows and macOS, worker processes re-import the calling script, so
        a script that calls this must do so under ``if __name__ == "__main__":``.
        Notebooks are not affected.

        Some SMILES may fail to parse; those rows are dropped. By default you get
        back only the surviving fingerprints, and you cannot tell *which* inputs
        were dropped - so the result no longer lines up with your SMILES list. If
        you carry other per-molecule data (an id, the SMILES themselves), pass
        ``return_valid_idx=True`` to also get the indices of the inputs that
        survived, and re-align your data with them::

            valid_idx, fps = calc.FingerprintFromSmiles(smiles, "mqn", return_valid_idx=True)
            ids_ok    = [ids[i] for i in valid_idx]
            smiles_ok = [smiles[i] for i in valid_idx]

        Args:
            smiles_list (list): A list of SMILES strings.
            fp (str): The fingerprint type to compute (e.g., 'morgan').
            nprocesses (int): Number of worker processes. Defaults to all CPU cores
                (at most 61 on Windows). Use 1 to compute in the current process.
            return_valid_idx (bool): If True, return ``(valid_idx, fingerprints)``
                where ``valid_idx`` is an int64 array of the input positions that
                parsed OK. If False (default), return only the fingerprints.
            **params: Additional keyword parameters for the fingerprint function.
                For 'morgan', required keys are:
                    - fpSize (int): Number of bits in the fingerprint.
                    - radius (int): Radius for the Morgan fingerprint.

        Returns:
            npt.NDArray: fingerprints with shape (n_valid, fpSize); or, if
            ``return_valid_idx`` is True, a tuple ``(valid_idx, fingerprints)``.

        Raises:
            ValueError: If an unsupported fingerprint type is requested.
            RuntimeError: If a worker process dies, e.g. a script on Windows or
                macOS that is missing the ``__main__`` guard.
        """
        func = self.fingerprint_function_map.get(fp)
        if func is None:
            raise ValueError(f"Unsupported fingerprint type: '{fp}'")

        # Bind the additional parameters to the selected function.
        part_func = partial(func, **params)

        if isinstance(smiles, str):
            smiles = [smiles]
        if nprocesses == 1:
            # No worker processes: avoids pool startup cost and spawn issues
            fingerprints = [part_func(s) for s in smiles]
        else:
            # A few chunks per worker keeps them busy without per-item overhead
            n_workers = nprocesses or os.cpu_count() or 1
            chunksize = max(1, len(smiles) // (n_workers * 4))
            executor = ProcessPoolExecutor(max_workers=nprocesses)
            try:
                fingerprints = list(executor.map(part_func, smiles, chunksize=chunksize))
            except BrokenProcessPool as e:
                # multiprocessing.Pool would respawn crashed workers forever here
                raise RuntimeError(
                    "A fingerprint worker process crashed. On Windows and macOS this "
                    "usually means the script calling FingerprintFromSmiles is missing "
                    "an `if __name__ == \"__main__\":` guard around its code."
                ) from e
            finally:
                # Results are collected; don't block while idle workers exit
                executor.shutdown(wait=False, cancel_futures=True)

        if return_valid_idx:
            # Keep the input positions that produced a fingerprint, so the
            # caller can re-align ids / SMILES that travel with the molecules.
            valid_idx = [i for i, fp_ in enumerate(fingerprints) if fp_ is not None]
            fps = np.array([fingerprints[i] for i in valid_idx])
            return np.asarray(valid_idx, dtype=np.int64), fps

        fingerprints = [fp for fp in fingerprints if fp is not None]

        # Free memory
        del smiles
        return np.array(fingerprints)
