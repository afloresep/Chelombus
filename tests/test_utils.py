import os
import numpy as np
import pandas as pd
import pytest

# Import the helper function; adjust the module name if necessary.
from chelombus.utils.helper_functions import save_chunk

@pytest.fixture
def sample_fp_chunk():
    """Fixture providing a small fingerprint numpy array for testing."""
    return np.array([[0, 1, 1], [1, 0, 0]])

def test_save_chunk_npy(tmp_path, sample_fp_chunk):
    """
    Test that save_chunk correctly saves a numpy array as a .npy file.
    """
    output_dir = tmp_path / "output"
    chunk_index = 0
    file_format = 'npy'
    
    # Call the helper function.
    file_path = save_chunk(sample_fp_chunk, str(output_dir), chunk_index, file_format=file_format)
    
    # Check that the file exists and has the correct extension.
    assert os.path.exists(file_path)
    assert file_path.endswith(".npy")
    
    # Load the saved file and verify the contents.
    loaded_array = np.load(file_path)
    np.testing.assert_array_equal(loaded_array, sample_fp_chunk)

def test_save_chunk_parquet(tmp_path, sample_fp_chunk):
    """
    Test that save_chunk correctly saves a numpy array as a .parquet file.
    """
    output_dir = tmp_path / "output"
    chunk_index = 1
    file_format = 'parquet'
    
    # Save the fingerprint chunk.
    file_path = save_chunk(sample_fp_chunk, str(output_dir), chunk_index, file_format=file_format)
    
    # Check that the file exists and has the correct extension.
    assert os.path.exists(file_path)
    assert file_path.endswith(".parquet")
    
    # Load the parquet file using pandas.
    df = pd.read_parquet(file_path)
    loaded_array = df.values
    
    # Compare the loaded data with the original array.
    np.testing.assert_array_equal(loaded_array, sample_fp_chunk)

def test_save_chunk_invalid_format(tmp_path, sample_fp_chunk):
    """
    Test that save_chunk raises a ValueError when an unsupported file format is provided.
    """
    output_dir = tmp_path / "output"
    chunk_index = 2
    file_format = 'unsupported'
    
    with pytest.raises(ValueError, match="Unsupported file format"):
        save_chunk(sample_fp_chunk, str(output_dir), chunk_index, file_format=file_format)

def test_directory_creation(tmp_path, sample_fp_chunk):
    """
    Test that save_chunk creates the output directory if it does not exist.
    """
    # Define an output directory that does not exist yet.
    output_dir = tmp_path / "non_existent_directory"
    chunk_index = 3
    file_format = 'npy'
    
    # Call the helper function.
    file_path = save_chunk(sample_fp_chunk, str(output_dir), chunk_index, file_format=file_format)
    
    # Verify that the output directory and file were created.
    assert os.path.exists(output_dir)
    assert os.path.exists(file_path)