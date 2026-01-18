import numpy as np
from .encoder_base import PQEncoderBase
from tqdm import tqdm
from sklearn.cluster import KMeans

class PQEncoder(PQEncoderBase):
    """
    Class to encode high-dimensional vectors into PQ-codes.
    """
    def __init__(self, k:int=256, m:int=8, iterations=20):
        """ Initializes the encoder with trained sub-block centroids.

        Args:
            k (int): Number of centroids. Default is 256. We assume that all subquantizers 
            have the same finit number (k') of reproduction values. So k = (k')^m 
            m (int): Number of distinct subvectors of the input X vector. 
            m is a subvector of dimension D' = D/m where D is the 
            dimension of the input vector X. D is therefore a multiple of m
            iterations (int): Number of iterations for the k-means

        High values of k increase the computational cost of the quantizer as well as memory 
        usage of strogin the centroids (k' x D floating points). Using k=256 and m=8 is often
        a reasonable choice. 

        Reference: DOI: 10.1109/TPAMI.2010.57
        """

        self.m = m 
        self.k = k 
        self.iterations = iterations
        self.encoder_is_trained = False
        self.codebook_dtype =  np.uint8 if self.k <= 2**8 else (np.uint16 if self.k<= 2**16 else np.uint32)
        self.pq_trained = []

        """ The codebook is defined as the Cartesian product of all centroids. 
        Storing the codebook C explicitly is not efficient, the full codebook would be (k')^m  centroids
        Instead we store the K' centroids of all the subquantizers (k' · m). We can later simulate the full 
        codebook by combining the centroids from each subquantizer. 
        """

        
    def fit(self, X_train:np.array, verbose:int=1, **kwargs)->None:
        """ KMeans fitting of every subvector matrix from the X_train matrix. Populates 
        the codebook by storing the cluster centers of every subvector

        X_train is the input matrix. For a vector that has dimension D  
        then X_train is a matrix of size (N, D)
        where N is the number of rows (vectors) and D the number of columns (dimension of
        every vector i.e. fingerprint in the case of molecular data)

        Args:
           X_train(np.array): Input matrix to train the encoder.  
           verbose(int): Level of verbosity. Default is 1
           **kwargs: Optional keyword arguments passed to the underlying KMeans `fit()` function.
        """
        
        assert X_train.ndim == 2, "The input can only be a matrix (X.ndim == 2)"
        N, D = X_train.shape # N number of input vectors, D dimension of the vectors
        assert self.k < N, "the number of training vectors (N for N,D = X_train.shape) should be more than the number of centroids (K)"
        assert D % self.m == 0, f"Vector (fingeprint) dimension should be divisible by the number of subvectors (m). Got {D} / {self.m}" 
        self.D_subvector = int(D / self.m) # Dimension of the subvector. 
        self.og_D = D # We save the original dimensions of the input vector (fingerprint) for later use
        assert self.encoder_is_trained == False, "Encoder can only be fitted once"
        
        self.codewords= np.zeros((self.m, self.k, self.D_subvector), dtype=float)
        
        iterable = range(self.m)
        if verbose > 0: 
            iterable = tqdm(iterable, desc='Training PQ-codes', total=self.m)    
            
        # Divide the input vector into m subvectors 
        subvector_dim = int(D / self.m) 

        for subvector_idx in iterable:
            X_train_subvector = X_train[:, subvector_dim * subvector_idx : subvector_dim * (subvector_idx + 1)] 
            # For every subvector, run KMeans and store the centroids in the codebook 
            kmeans = KMeans(n_clusters=self.k, init='k-means++', max_iter=self.iterations, **kwargs).fit(X_train_subvector)
            self.pq_trained.append(kmeans)

            # Results for training 1M 1024 dimensional fingerprints were: 5 min 58 s, with k=256, m=4, iterations=200, this is much faster than using scipy
            # Store the cluster_centers coordinates in the codebook
            self.codewords[subvector_idx] = kmeans.cluster_centers_ # Store the cluster_centers coordinates in the codebook

        self.encoder_is_trained = True
        del X_train # remove initial training data from memory


    def transform(self, X:np.array, verbose:int=1, **kwargs) -> np.array:
        """
        Transforms the input matrix X into its PQ-codes.

        For each sample in X, the input vector is split into `m` equal-sized subvectors. 
        Each subvector is assigned to the nearest cluster centroid
        and the index of the closest centroid is stored. 

        The result is a compact representation of X, where each sample is encoded as a sequence of centroid indices.

        Args:
            X (np.ndarray): Input data matrix of shape (n_samples, n_features), 
                            where n_features must be divisible by the number of subvectors `m`.
            verbose(int): Level of verbosity. Default is 1
            **kwargs: Optional keyword arguments passed to the underlying KMeans `predict()` function.

        Returns:
            np.ndarray: PQ codes of shape (n_samples, m), where each element is the index of the nearest centroid 
                        for the corresponding subvector.
        """

        assert self.encoder_is_trained == True, "PQEncoder must be trained before calling transform"

        N, D = X.shape
        # Store the index of the Nearest centroid for each subvector
        pq_codes = np.zeros((N, self.m), dtype=self.codebook_dtype)

        iterable = range(self.m)
        # If our original vector is 1024 and our m (splits) is 8 then each subvector will be of dim= 1024/8 = 128
        if verbose > 0: 
            iterable = tqdm(iterable, desc='Generating PQ-codes', total=self.m)

        subvector_dim = int(D / self.m)

        for subvector_idx in iterable:
            X_train_subvector = X[:, subvector_dim * subvector_idx : subvector_dim * (subvector_idx + 1)] 
            # For every subvector, run KMeans.predict(). Then look in the codebook for the index of the cluster that is closest
            # Appends the centroid index to the pq_code.  
            pq_codes[:, subvector_idx] =  self.pq_trained[subvector_idx].predict(X_train_subvector, **kwargs)
            
        # Free memory 
        del  X

        # Return pq_codes (labels of the centroids for every subvector from the X_test data)
        return pq_codes

    def fit_transform(self, X:np.array, verbose:int=1, **kwargs) -> np.array:
        """Fit and transforms the input matrix `X` into its PQ-codes

        The encoder is trained on the matrix and then for each sample in X,
          the input vector is split into `m` equal-sized vectors subvectors composed 
          byt the index of the closest centroid. Returns a compact representation of X, 
          where each sample is encoded as a sequence of centroid indices (i.e PQcodes)

        Args:
            X (np.array): Input data matrix of shape (n_samples, n_features)
            verbose (int, optional): Level of verbosity. Defaults to 1.
            **kwargs: Optional keyword. These arguments will be passed to the underlying KMeans
            predict() function. 

        Returns:
            np.array: PQ codes of shape (n_samples, m), where each element is the index of the nearest
            centroid for the corresponding subvector
        """

        self.fit(X, verbose, **kwargs)
        return self.transform(X, verbose)


    def inverse_transform(self, 
                          pq_codes:np.array, 
                          binary=False, 
                          round=True):
        """ Inverse transform. From PQ-code to the original vector. 
        This process is lossy so we don't expect to get the exact same data.
        If binary=True then the vectors will be returned in binary. 
        This is useful for the case where our original vectors were binary. 
        With binary=True then the returned vectors are transformed from 
        [0.32134, 0.8232, 0.0132, ... 0.1432, 1.19234] to 
        [0, 1, 0, ..., 0, 1]
        If round=True then the vector will be approximated to integer values
        (useful in cases where we expeect to have a count-based fingerprint)

        Args:
            pq_code: (np.array): Input data of PQ codes to be transformed into the
            original vectors.
            binary: (bool): Wheter to return the vectors rounded to 0s and 1s. Default is False
            round: (bool): Round inversed vector to integers values. Default is True  
        """
        
        # Get shape of the input matrix of PQ codes
        N, D = pq_codes.shape

        # The dimension of the PQ vectors should be the same 
        # as the number of splits (subvectors) from the original data 

        assert D == (self.m), f"The dimension D of the PQ-codes (N,D) should be the same as the number of the subvectors or splits (m) . Got D = {D} for m = {self.m}"
        assert D == (self.og_D  / self.D_subvector), f"The dimension D of the PQ-codes (N,D) should be the same as the original vector dimension divided the subvector dimension"

        X_inversed = np.empty((N, D*self.D_subvector), dtype=float)
        for subvector_idx in range(self.m):
            X_inversed[:, subvector_idx*self.D_subvector:((subvector_idx+1)*self.D_subvector)] = self.codewords[subvector_idx][pq_codes[:, subvector_idx], :]

        if binary:
            reconstructed_binary = (X_inversed>= 0.6).astype('int8')
            return reconstructed_binary 

        if round: return X_inversed.astype(int)
        else: return X_inversed