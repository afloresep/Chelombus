import numpy as np
import pytest
import logging
from functools import partial
from chelombus.streamer.data_streamer import DataStreamer
from chelombus.utils.fingerprints import _calculate_morgan_fp, _calculate_mqn_fp,  FingerprintCalculator

# A fixture for common fingerprint parameters.
@pytest.fixture
def fp_params():
    return {'fpSize': 2048, 'radius': 2}


def test_calculate_morgan_fp_valid(fp_params):
    """
    Test that _calculate_morgan_fp returns a NumPy array of the correct shape
    when given a valid SMILES string.
    """
    smiles = "CCO"  
    fingerprint = _calculate_morgan_fp(smiles, to_numpy=True, **fp_params)
    assert isinstance(fingerprint, np.ndarray)
    assert fingerprint.shape[0] == fp_params['fpSize']


def test_calculate_mqn_fp_valid(fp_params):
    """
    Test that _calculate_mqn_fp returns a NumPy array of the correct shape
    when given a valid SMILES string.
    """
    smiles = "CCO"  
    fingerprint = _calculate_mqn_fp(smiles)
    assert isinstance(fingerprint, np.ndarray)
    assert fingerprint.shape[0] == 42, "MQN can only be 42 dimensions"


def test_calculate_morgan_fp_invalid(fp_params):
    """
    Test that _calculate_morgan_fp returns None for an invalid SMILES string.
    This ensures data integrity by not inserting random fingerprints.
    """
    smiles = "invalid_smiles"
    fingerprint = _calculate_morgan_fp(smiles, **fp_params)

    assert fingerprint is None, "Invalid SMILES should return None, not a fingerprint"


def test_calculate_morgan_fp_missing_parameters():
    """
    Test that _calculate_morgan_fp raises a ValueError when required parameters are missing.
    """
    smiles = "CCO"
    with pytest.raises(ValueError):
        _calculate_morgan_fp(smiles)  # No fpSize or radius provided.

def test_calculate_morgan_fp_logs_warning(fp_params, caplog):
    """
    Test that a warning is logged when an invalid SMILES string is processed.
    """
    smiles = "invalid_smiles"
    with caplog.at_level(logging.WARNING):
        _calculate_morgan_fp(smiles, **fp_params)
    
    # Check that a warning message was logged.
    assert any("could not be converted" in record.message for record in caplog.records)


def test_fingerprint_calculator_valid(fp_params):
    """
    Test that FingerprintCalculator.FingerprintFromSmiles returns a NumPy array
    with fingerprints for a list of valid SMILES strings.
    """
    calculator = FingerprintCalculator()
    smiles_list = ["CCO", "C1CCCCC1", "O=C=O"]
    fingerprints = calculator.FingerprintFromSmiles(smiles_list, 'morgan', to_numpy=True, nprocesses=1,**fp_params)
    
    assert isinstance(fingerprints, np.ndarray)
    assert fingerprints.shape == (len(smiles_list), fp_params['fpSize']) 
    assert fingerprints[0].nbytes == 2048


def test_fingerprint_calculator_invalid_fp(fp_params):
    """
    Test that FingerprintCalculator.FingerprintFromSmiles raises a ValueError
    when an unsupported fingerprint type is requested.
    """
    calculator = FingerprintCalculator()
    smiles_list = ["CCO", "C1CCCCC1"]
    with pytest.raises(ValueError):
        calculator.FingerprintFromSmiles(smiles_list, 'unsupported', nprocesses=1, **fp_params)

def test_fingerprint_with_data_parser(tmp_path):
    # Create a temporary file with known content.
    file_content = ["CCC(C)C(NC(=O)NC1=CC(C)=CC=C1C)C(=O)NC1=NC=CC=C1C1CC1\n", 
                    "CC(C)(C)OC(=O)N[C@H](CC(=O)NC1=NC=CC=C1C1CC1)CC1=CC=CC=C1\n",
                    "O=C(NC1=NC=CC=C1C1CC1)C1=CC=C(N2CCN(C[C@H](O)CO)CC2)C=C1\n",
                    "CC1=CC=CC=C1[C@@H](CC(=O)NC1=NC=CC=C1C1CC1)NC(=O)OC(C)(C)C\n  ", 
                    "CC(C)(C)OC(=O)NCCOC1=CC=C(C(=O)NC2=NC=CC=C2C2CC2)C=C1\n  ", 
                    "CC(C)(C)OC(=O)N1CCC(C2=C(C(=O)NC3=NC=CC=C3C3CC3)NC=C2)C1\n  ", 
                    "CC(C)(C)OC(=O)N1C[C@H](N2C=CN=N2)C[C@@H]1C(=O)NC1=NC=CC=C1C1CC1\n"]

    file_path = tmp_path / "test.txt"
    file_path.write_text("".join(file_content))

    result = [] 
    streamer = DataStreamer()
    calculator = FingerprintCalculator()
    chunksize = 2
    # Expected chunks: [first two lines], [next two lines], [last line]
    for smiles_chunk in streamer.parse_input(str(file_path), chunksize):
        fp_chunk = calculator.FingerprintFromSmiles(smiles_chunk, fp='morgan', fpSize=1024, radius=3, nprocesses=1)
        result.append(fp_chunk)

    assert len(result) == 4 
    assert result[0].shape == (2, 1024)
    assert result[3].shape == (1, 1024)