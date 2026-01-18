import pytest
import numpy as np

from chelombus.encoder.encoder import PQEncoder 

def test_fit_successful_with_normal_split():
    """
    Test that the fit method runs successfully with proper input data
    and that the codebook is set with the expected shape and the 
    encoder_is_trained flag becomes True.
    """
    np.random.seed(42)
    N, D = 100, 16
    X_train = np.random.rand(N, D).astype(np.float32)
    k = 4
    m = 4
    
    encoder = PQEncoder(k=k, m=m, iterations=10)
    encoder.fit(X_train)
    assert encoder.encoder_is_trained is True, "Encoder should be trained after successful fit."
    
    for idx, centroids_array in enumerate(encoder.codewords):
        assert encoder.codewords[idx].shape == (k, D // m), (
            f"Expected codebook shape to be {(k, D//m)}, "
            f"got {encoder.codewords[idx].shape}"
        )
        # Ensure dtype is float per the code
        assert np.issubdtype(centroids_array.dtype, np.floating), (
                f"Codebook {idx} has dtype {centroids_array.dtype}, "
                "expected a floating type."
        )


def test_fit_raises_assertionerror_for_non2d_input():
    """
    Test that providing a 1D input array to fit raises an AssertionError.
    """
    X_train_1d = np.array([1, 2, 3])  # 1D input
    encoder = PQEncoder(k=4, m=2, iterations=10)
    
    with pytest.raises(AssertionError, match="The input can only be a matrix"):
        encoder.fit(X_train_1d)


def test_fit_raises_assertionerror_for_3d_input():
    """
    Test that providing a 3D input array to fit raises an AssertionError.
    """
    X_train_3d = np.random.rand(10, 4, 4)
    encoder = PQEncoder(k=4, m=2, iterations=10)
    
    with pytest.raises(AssertionError, match="The input can only be a matrix"):
        encoder.fit(X_train_3d)


def test_fit_raises_assertionerror_for_k_too_large():
    """
    Test that if k >= N, an assertion error is raised.
    """
    N, D = 8, 16
    X_train = np.random.rand(N, D).astype(np.float32)
    
    # Here k == 8, so the assertion says it must be strictly less (k < N).
    encoder = PQEncoder(k=N, m=4, iterations=10)
    
    with pytest.raises(AssertionError, match="should be more than the number of centroids"):
        encoder.fit(X_train)


def test_fit_raises_assertionerror_for_incorrect_dim_division():
    """
    Test that if D is not divisible by m, an assertion error is raised.
    """
    N, D = 100, 10
    X_train = np.random.rand(N, D).astype(np.float32)
    # Here m=3, but D=10 which is not divisible by 3
    encoder = PQEncoder(k=4, m=3, iterations=10)
    
    with pytest.raises(AssertionError, match="dimension should be divisible by the number of subvectors"):
        encoder.fit(X_train)


def test_fit_raises_assertionerror_for_multiple_fits():
    """
    Test that calling fit on an already fitted encoder raises an assertion error.
    """
    N, D = 100, 16
    X_train = np.random.rand(N, D).astype(np.float32)
    encoder = PQEncoder(k=4, m=4, iterations=10)
    
    encoder.fit(X_train)  # First fit
    assert encoder.encoder_is_trained is True
    
    with pytest.raises(AssertionError, match="Encoder can only be fitted once"):
        encoder.fit(X_train)  # Second fit should raise error

import pytest
import numpy as np

@pytest.fixture
def trained_pq_encoder():
    """
    Fixture that returns a mock PQEncoder with:
      - m = 8 (eight subvectors)
      - k = 256 (each subvector has 256 possible centroids)
      - og_D = 1024 (imagine the original dimension is 1024)
      - D_subvector = 128 (since 1024 / 8 = 128)
      - codebook_cluster_centers has shape (8, 256, 128)
    This fixture stands in for a real, already-trained PQEncoder.
    """
    fingerprints = np.random.randint(0,2, size=(1000, 1024)) 
    pq_encoder = PQEncoder(k=256, m=2, iterations=10)
    pq_encoder.fit(fingerprints)

    return pq_encoder
def test_inverse_transform_shape(trained_pq_encoder):
    """
    Check that inverse_transform returns an array of shape (N, m * D_subvector).
    """
    pq_encoder = trained_pq_encoder
    # Suppose we have 3 samples, each encoded with m=2 codes
    X_codes = np.array([
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=int)

    X_inversed = pq_encoder.inverse_transform(X_codes)
    assert X_inversed.shape == (3, pq_encoder.m * pq_encoder.D_subvector), (
        f"Expected shape (3, {pq_encoder.m * pq_encoder.D_subvector}), got {X_inversed.shape}"
    )

def test_inverse_transform_dimension_mismatch(trained_pq_encoder):
    """
    If the dimension of X_codes does not match pq_encoder.m, 
    it should raise an AssertionError.
    """
    pq_encoder = trained_pq_encoder
    # m=2 in our mock encoder, so we'll supply an array with shape (N, 3)
    X_codes_wrong_dim = np.array([
        [0, 1, 0],
        [1, 0, 1]
    ], dtype=int)

    with pytest.raises(AssertionError):
        _ = pq_encoder.inverse_transform(X_codes_wrong_dim)

def test_inverse_transform_binary_option(trained_pq_encoder):
    """
    Check that setting binary=True returns a binary array (int8) 
    by thresholding at 0.6.
    """
    pq_encoder = trained_pq_encoder
    # Suppose N=2 for quick test
    X_codes = np.random.randint(0, high=256, size=(2, pq_encoder.m), dtype=np.int32)
    X_inversed = pq_encoder.inverse_transform(X_codes, binary=True)
    
    # Check shape is (2, 1024) if that's expected
    assert X_inversed.shape == (2, 1024)
    # Check it's int8
    assert X_inversed.dtype == np.int8
    # Check it contains only 0 or 1
    unique_vals = np.unique(X_inversed)
    assert set(unique_vals).issubset({0, 1}), f"Found values other than 0 or 1: {unique_vals}"


# @pytest.mark.parametrize("binary_flag", [False, True])
# def test_round_trip(trained_pq_encoder, binary_flag):
#     """
#     (Optional) Example of a round-trip test (transform → inverse_transform). 
#     Here we just demonstrate how you'd do it in principle if you had access
#     to the real encoder's transform method. For the fixture's sake, we mock
#     a transform → code → inverse_transform sequence to show conceptual usage.
#     """
#     pq_encoder = trained_pq_encoder
#     # Suppose we had an original data matrix of shape (2, 4)
#     # but we skip an actual transform because we only have a mock.
#     # Instead, we'll just define some "codes" that could be produced by transform.
#     X_original_codes = np.array([
#         [0, 1],  # meaning cluster-0 in subvector-0, cluster-1 in subvector-1
#         [1, 0]
#     ], dtype=int)

#     X_reconstructed = pq_encoder.inverse_transform(X_original_codes, binary=binary_flag)
#     # Check shapes
#     expected_shape = (2, pq_encoder.m * pq_encoder.m)
#     assert X_reconstructed.shape == expected_shape, (
#         f"Round-trip shape mismatch: got {X_reconstructed.shape}, expected {expected_shape}"
#     )
#     # Optionally check values or binary thresholding 
#     if binary_flag:
#         assert X_reconstructed.dtype == np.int8, "Expected binary reconstruction to be int8 dtype."
#     else:
#         assert X_reconstructed.dtype == float, "Expected float dtype for non-binary reconstruction."
