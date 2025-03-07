import time 
import random 
import os 
from sklearn.neighbors import NearestNeighbors
from typing import List
import pandas as pd
import numpy as np
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)

from typing import Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from faerun import Faerun
from config import OUTPUT_FILE_PATH, TMAP_NAME, TMAP_NODE_SIZE, TMAP_K, TMAP_POINT_SCALE
from src.fingerprint_calculator import FingerprintCalculator
import tmap as tm
import logging

 # Class to generate physicochemical properties from smiles 
class TmapConstructor:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def _calculate_threshold(self, data) -> float:
        """ 
        Typically we could have very different values in the molecular properties (e.g. number of rings on a very diverse dataframe)
        which leads to lose of information due to outliers having extreme color values and the rest falling into the same range. 
        This function calculates a threshold using IQR method to cut the outliers value based on percentiles.
        """
        q1, q3 = np.percentile(data, [15, 85])
        iqr = q3 - q1 
        threshold = q3 + 1.5*iqr
        return threshold

    def _mol_properties_from_smiles(self, smiles: str) -> tuple:
        """ Get molecular properties from a single SMILES string"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        hac = mol.GetNumHeavyAtoms()
        num_aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        fraction_aromatic_atoms = num_aromatic_atoms / hac if hac > 0 else 0
        number_of_rings = rdMolDescriptors.CalcNumRings(mol)
        molecular_weight = Descriptors.ExactMolWt(mol)
        clogP = Descriptors.MolLogP(mol)
        fraction_Csp3 = Descriptors.FractionCSP3(mol)
        
        return (hac, fraction_aromatic_atoms, number_of_rings, clogP, fraction_Csp3, molecular_weight)
   
    def mol_properties_from_df(self)->tuple:
        self.dataframe[['hac', 'frac_aromatic', 'num_rings', 'clogp', 'frac_csp3', 'MW']]  = self.dataframe['smiles'].apply(
            self._mol_properties_from_smiles
        ).parallel_apply(pd.Series)

        # Drop rows with any None or NaN values in the property columns
        # df_clean = self.dataframe.dropna(subset=['hac', 'frac_aromatic', 'num_rings', 'clogp', 'frac_csp3'])
    
        # Calculate thresholds for each property using the clean DataFrame
        hac_threshold = self._calculate_threshold(self.dataframe['hac'])
        frac_threshold= self._calculate_threshold(self.dataframe['frac_aromatic'])
        rings_threshold=self._calculate_threshold(self.dataframe['num_rings'])
        clogp_threshold=self._calculate_threshold(self.dataframe['clogp'])
        csp3_threshold =self._calculate_threshold(self.dataframe['frac_csp3'])
        mw_threshold = self._calculate_threshold(self.dataframe['MW'])
    
        # TODO: Filter the Dataframe based on threshold? If we do this we drop some of the values or 'lose' their actual value by 
        # changing it for the threshold. 

        # Filter the DataFrame based on the thresholds
        filtered_df = self.dataframe[
            (self.dataframe['hac'] <= hac_threshold) &
            (self.dataframe['frac_aromatic'] <= frac_threshold) &
            (self.dataframe['num_rings'] <= rings_threshold) &
            (self.dataframe['clogp'] <= clogp_threshold) &
            (self.dataframe['frac_csp3'] <= csp3_threshold) & 
            (self.dataframe['MW'] <= mw_threshold)
        ]
    
        # filtered_hac = filtered_df['hac'].tolist()
        # filtered_frac_aromatic = filtered_df['frac_aromatic'].tolist()
        # filtered_num_rings = filtered_df['num_rings'].tolist()
        # filtered_clogp = filtered_df['clogp'].tolist()
        # filtered_frac_csp3 = filtered_df['frac_csp3'].tolist()
        # filtered_mw = filtered_df['MW'].tolist()
        
        # Extract filtered properties as lists
        filtered_hac = self.dataframe['hac'].tolist()
        filtered_frac_aromatic = self.dataframe['frac_aromatic'].tolist()
        filtered_num_rings = self.dataframe['num_rings'].tolist()
        filtered_clogp = self.dataframe['clogp'].tolist()
        filtered_frac_csp3 = self.dataframe['frac_csp3'].tolist()
        filtered_mw = self.dataframe['MW'].tolist()
    
        # Return the list of lists
        return np.array((np.array(filtered_hac), np.array(filtered_frac_aromatic), np.array(filtered_num_rings), np.array(filtered_clogp), np.array(filtered_frac_csp3), np.array(filtered_mw)))
    
 
class TmapGenerator:
    def __init__(
            self,
            df_path: pd.DataFrame,  
            fingerprint_type: str = 'mhfp', 
            permutations: Optional[int]= 512, 
            output_name: str = TMAP_NAME,
            fp_size: int = 1024, 
            categ_cols: Optional[List] = None
        ):
        """
        param: fingerprint_type : Type of molecular fingerprint to be used on the TMAP. Options: {'mhfp', 'mqn', 'morgan', 'mapc'
        param: permutations: On MHFP number of permutations to be used in the MinHash

        TODO: Is KNN necessary? 
        param: method: Type of method to be used in TMAP. It can be either LSH (normal TMAP) or KNN? We could just create the TMAP always with LSH....
        
        param: output_name: name for the TMAP files. In case of the dynamic TMAP should inherit name from the cluster_id 
        param: categ_cols: List with the column names in the dataframe to be included as labels in the TMAP. These are typically your categorical columns
        e.g. 'Target_type', 'Organism' ...  
        """
        self.df_path = df_path
        self.fingerprint_type = fingerprint_type
        self.permutations = permutations
        self.output_name = output_name
        self.fp_size = fp_size
        self.categ_cols = categ_cols 
        self.dataframe = pd.read_csv(df_path) 
        
        self.tmap_name = 'representatives'
        # Initialize helper classes
        self.fingerprint_calculator = FingerprintCalculator(self.dataframe['smiles'], self.fingerprint_type, permutations=self.permutations, fp_size=self.fp_size)
        self.tmap_constructor = TmapConstructor(self.dataframe)


    def tmap_from_vectors(self): 
        """
        Script for generating a simple TMAP using vectors (e.g. PCA coordinates) instead of SMILES
        Mainly to be used to create the primary TMAP. 
        """
        pca_columns = ['PCA_1', 'PCA_2', 'PCA_3']
        pca_values = self.dataframe[pca_columns].values # array shape (125000 , 3)
        nbrs = NearestNeighbors(n_neighbors=21, metric='manhattan').fit(pca_values)
    
        distances, indices = nbrs.kneighbors(pca_values)

        edge_list = []
        for i in range(len(pca_values)):
            for neighbor_idx, dist in zip(indices[i, 1:], distances[i, 1:]):
                edge_list.append((i, neighbor_idx, float(dist)))

        # Get the coordinates and Layout Configuration
        cfg = tm.LayoutConfiguration()
        cfg.node_size = TMAP_NODE_SIZE 
        cfg.mmm_repeats = 2
        cfg.sl_extra_scaling_steps = 10
        cfg.k = TMAP_K 
        cfg.sl_scaling_type = tm.RelativeToAvgLength
        self.x, self.y, self.s, self.t, _ = tm.layout_from_edge_list(len(pca_values), edge_list, cfg)

        logging.info("Creating labels")
        start = time.time()
        labels = []
        for i, row in self.dataframe.iterrows():
            # Add link to generate the link between primary TMAP node and secondary TMAP 
            if self.categ_cols is not None:
                label = '__'.join(str(row[col]) for col in self.categ_cols)
                # Create a clickable link with cluster_id that points to the Flask endpoint
                link = f'<a href="/generate/{label}" target="_blank">{label}</a>'
                labels.append(row['smiles'] + '__' + link)
            # If no categ_cols (node for secondary TMAP) then no link is needed
            else:
                labels.append(row['smiles'])
        descriptors = self.tmap_constructor.mol_properties_from_df()
        print(len(descriptors[2, :]))
        end = time.time()
        logging.info(f"Labels took {end - start} seconds to create")

        # Plotting
        logging.info("Setting up TMAP and plotting")
        start = time.time()
        f = Faerun(
            view="front",
            coords=False,
            title="",
            clear_color="#FFFFFF",
        )

        f.add_scatter(
            "Descriptors",
            {
                "x": self.x,
                "y": self.y,
                "c": descriptors, 
                "labels":labels,
            },
            shader="smoothCircle",
            point_scale= TMAP_POINT_SCALE,
            max_point_size= 20,
            interactive=True,
            # legend_labels=[], # TODO: Get list of list of labels. This sould be something like [df[col] for col in self.categ_col]
            # categorical= bool_categorical, #TODO: Add support for categorical columns. 
            series_title= ['HAC', 'Fraction Aromatic Atoms', 'Number of Rings', 'clogP', 'Fraction Csp3', 'MW'], 
            has_legend=True,           
            colormap=['hsv', 'hsv', 'hsv', 'hsv', 'gist_ncar', 'gist_rainbow'],
        
            categorical=[False, False, False, False, False, False],
        )

        f.add_tree(self.tmap_name+"_TMAP_tree", {"from": self.s, "to": self.t}, point_helper="Descriptors")
        f.plot(self.tmap_name+"_TMAP", template='smiles')
        end = time.time()
        logging.info(f"Plotting took {end - start} seconds") 

    def tmap_little(self): 
        """
        Script for generating a simple TMAP using fingerprint calculated from SMILES in dataframe
        """
        start = time.time() 
        logging.info("Calculating fingerprints")
        fingerprints = self.fingerprint_calculator.calculate_fingerprints()
        end = time.time()
        logging.info(f"Fingeprints calculations took {end - start} seconds")

        logging.info("Constructing LSH Forest")
        start = time.time()
        self.construct_lsh_forest(fingerprints)
        end = time.time()
        logging.info(f"LSH was constructed in {end - start} seconds")

        logging.info("Creating labels")
        start = time.time()
        labels = []
        for i, row in self.dataframe.iterrows():
            if self.categ_cols is not None:
                label = '__'.join(str(row[col]) for col in self.categ_cols)
                # Create a clickable link with cluster_id that points to the Flask endpoint
                link = f'<a href="/generate/{label}" target="_blank">{label}</a>'
                labels.append(row['smiles'] + '__' + link)
            else:
                labels.append(row['smiles'])
        descriptors = self.tmap_constructor.mol_properties_from_df()
        end = time.time()
        logging.info(f"Labels took {end - start} seconds to create")

        # Plotting
        logging.info("Setting up TMAP and plotting")
        start = time.time()
        f = Faerun(
            view="front",
            coords=False,
            title="",
            clear_color="#FFFFFF",
        )

        f.add_scatter(
            self.tmap_name+"_TMAP",
            {
                "x": self.x,
                "y": self.y,
                "c": descriptors, 
                "labels":labels,
            },
            shader="smoothCircle",
            point_scale= TMAP_POINT_SCALE,
            max_point_size= 20,
            interactive=True,
            # legend_labels=[], # TODO: Get list of list of labels. This sould be something like [df[col] for col in self.categ_col]
            # categorical= bool_categorical, #TODO: Add support for categorical columns. 
            series_title= ['HAC', 'Fraction Aromatic Atoms', 'Number of Rings', 'clogP', 'Fraction Csp3', 'MW'], 
            has_legend=True,           
            colormap=['viridis', 'viridis', 'viridis', 'viridis', 'viridis', 'viridis'],
            categorical=[False, False, True, False, False, False],
        )

        f.add_tree(self.tmap_name+"_TMAP_tree", {"from": self.s, "to": self.t}, point_helper=self.tmap_name+"_TMAP")
        f.plot(self.tmap_name+"_TMAP", template='smiles')
        end = time.time()
        logging.info(f"Plotting took {end - start} seconds")

    def _get_pca_vectors(self):
        """
        The vectors used for the TMAP layout will be the PCA Components. Typically to be used for the representatives cluster TMAP
        """
        #TODO: Get 3D Coordinates from self.representative_dataframe
        # Then use those as vectors for the TMAP. Not sure if worth it. Check with JL

    def generate_representatives_tmap(self, type_vector: str='fingerprint'):
        """
        Hierarchical TMAP of the clusters. Each point/node in the TMAP is a representative molecule of the cluster. 
        The vectors used for the TMAP layout will be either the 'fingerprint' of the representative molecule 
        or the PCA values (3D Coordinates). Default is 'fingerprint' 
        """
        if type_vector == 'fingerprint':
            self._get_fingerprint_vectors()
        elif type_vector == 'coordinates':
            self._get_pca_vectors()
        self.label = 'cluster_id'

    @staticmethod
    def construct_lsh_forest(fingerprints) -> None:
        import tmap as tm
        tm_fingerprints  = [tm.VectorUint(fp) for fp in fingerprints] #TMAP requires fingerprints to be passed as VectorUint

        # LSH Indexing and coordinates generation
        lf = tm.LSHForest(512)
        lf.batch_add(tm_fingerprints)
        lf.index()

        # Get the coordinates and Layout Configuration
        cfg = tm.LayoutConfiguration()
        cfg.node_size = TMAP_NODE_SIZE 
        cfg.mmm_repeats = 2
        cfg.sl_extra_scaling_steps = 10
        cfg.k = TMAP_K 
        cfg.sl_scaling_type = tm.RelativeToAvgLength
        start = time.time()
        x, y, s, t, _ = tm.layout_from_lsh_forest(lf, cfg)
        end = time.time()
        logging.info(f'Layout from lsh forest took {(end-start)} seconds')
        return x, y, s, t 

    def plot_faerun(self, fingerprints):
        logging.info("Constructing LSH Forest...")
        self.construct_lsh_forest(fingerprints) 
        f = Faerun(view="front", 
                    coords=False, 
                    title= "", 
                    clear_color="#FFFFFF")
        
        def safe_create_categories(series):
            return Faerun.create_categories(series.fillna('Unknown').astype(str))
        
        # Create categories
        labels = []
        for i, row in self.dataframe.iterrows():
            if self.categ_cols != None:
                label = '__'.join(str(row[col]) for col in self.categ_cols)
                labels.append(row['smiles']+'__'+label)
            else:
                labels.append(row['smiles'])

        logging.info("Plotting...")
        properties = self.tmap_constructor.mol_properties_from_df() 

        # Categorical = [True] * categorical_columns + [False]*numerical_columns 
        numerical_col= [False]*5 # These are 5 by default. 5 molecular properties 
        categorical_col = [True]*len(self.categ_cols)
        bool_categorical = categorical_col + numerical_col  # List of booleans required to indicate if a label is categorical or numerical
        
        cluster_labels, cluster_data = safe_create_categories(self.dataframe['cluster_id'])

        colormap = ['tab10' if value else 'viridis' for value in bool_categorical]
        # properties.insert(0, cluster_data)
        series_title = self.categ_cols + ['HAC', 'Fraction Aromatic Atoms', 'Number of Rings', 'clogP', 'Fraction Csp3']
        c = random.sample(range(1,len(properties[2])*2), len(properties[2]))
        f.add_scatter(
            "mapc_targets", 
            {
                "x": self.x, 
                "y": self.y, 
                "c": c, 
                "labels": self.dataframe['smiles'], 
            }, 
        shader="smoothCircle",
        point_scale= TMAP_POINT_SCALE,
        max_point_size=20,
        interactive=True,
        legend_labels=[], # TODO: Get list of list of labels. This sould be something like [df[col] for col in self.categ_col]
        categorical= bool_categorical, 
        colormap= colormap, 
        series_title= series_title, 
        has_legend=True,
) 
        # Add tree
        
        f.add_tree("mhfp_tmap_node140_TMAP_tree", {"from": self.s, "to": self.t}, point_helper="mhfp_tmap_node140_TMAP")
        f.plot('mhfp_tmap_node140_TMAP', template='smiles')
        # Plot

    def generate_cluster_tmap(self, cluster_id: str):
        """
        Generate the TMAP 'on the fly' based on the cluster_id given. It will look for the csv file for now but the idea is to retrieve from database
        cluster_id (str int_int_int) Find the cluster_id which TMAP we will do. It has to be in format PCA1_PCA2_PCA3. 
        e.g. 0_12_10. Right now it finds the csv file with this label. In the future it will retrieve it from the database
        """

        import clickhouse_connect
        import pandas as pd 

        client = clickhouse_connect.get_client(host='localhost', port=8123)

        logging.info(f'Getting data for cluster_id = {cluster_id}')
        # Define query 
        self.dataframe = pd.DataFrame(client.query(f"SELECT * FROM clustered_enamine WHERE cluster_id == {cluster_id}").result_rows,
                                 columns=["smiles", "PCA_1", "PCA_2", "PCA_3", "cluster_id"])

        name = f'cluster_{cluster_id}'
        self.tmap_name = name

        logging.info(f'TMAP of {len(self.dataframe)}')
        # Re-Initialize Fingerprint Calculator
        self.fingerprint_calculator = FingerprintCalculator(self.dataframe['smiles'], self.fingerprint_type, permutations=self.permutations, fp_size=self.fp_size)
        self.tmap_constructor = TmapConstructor(self.dataframe)
        self.tmap_little()
        logging.info('TMAP DONE')


class ClickhouseTMAP():
    def __init__(
            self,
            fingerprint_type: str = 'mhfp', 
            permutations: Optional[int]= 512, 
            output_name: str = TMAP_NAME,
            fp_size: int = 1024, 
            categ_cols: Optional[List] = None
        ):
        """
        param: fingerprint_type : Which molecular fingerprint to be used on the TMAP. Options: {'mhfp', 'mqn', 'morgan', 'mapc'
        param: permutations: On MHFP number of permutations to be used in the MinHash
        param: output_name: name for the TMAP files. In case of the dynamic TMAP should inherit name from the cluster_id 
        param: categ_cols: List with the column names in the dataframe to be included as labels in the TMAP. These are typically your categorical columns
        e.g. 'Target_type', 'Organism' ...  
        """
        self.fingerprint_type = fingerprint_type
        self.permutations = permutations
        self.output_name = output_name
        self.fp_size = fp_size
        self.categ_cols = categ_cols 
        
        self.tmap_name = 'representatives'
        # Initialize helper classes

    def generate_tmap_from_cluster_id(self, cluster_id, client): 

        import clickhouse_connect
        import pandas as pd

        ##########################################
        # Connect to clickchouse database
        ##########################################

        logging.info(f'Getting data for cluster_id = {cluster_id}')

        # Create dataframe from DB and name for the TMAP
        self.dataframe = pd.DataFrame(client.query(f"SELECT * FROM clustered_enamine WHERE cluster_id == {cluster_id}").result_rows,
                                 columns=["smiles", "PCA_1", "PCA_2", "PCA_3", "cluster_id"])
        name = f'cluster_{cluster_id}'
        self.tmap_name = name

        logging.info(f'TMAP of {len(self.dataframe)} points')

        # Initialize Fingerprint Calculator
        self.fingerprint_calculator = FingerprintCalculator(self.dataframe['smiles'], self.fingerprint_type, permutations=self.permutations, fp_size=self.fp_size)
        tmap_constructor = TmapConstructor(self.dataframe)

        ################################
        # Create TMAP from DB dataframe
        ################################

        start = time.time() 
        logging.info("Calculating fingerprints")
        fingerprints = self.fingerprint_calculator.calculate_fingerprints()
        end = time.time()
        logging.info(f"Fingeprints calculations took {end - start} seconds")

        logging.info("Constructing LSH Forest")
        start = time.time()
        x, y, s, t = TmapGenerator.construct_lsh_forest(fingerprints)
        end = time.time()
        logging.info(f"LSH was constructed in {end - start} seconds")

        logging.info("Creating labels")
        start = time.time()
        labels = []
        for i, row in self.dataframe.iterrows():
            if self.categ_cols is not None:
                label = '__'.join(str(row[col]) for col in self.categ_cols)
                # Create a clickable link with cluster_id that points to the Flask endpoint
                link = f'<a href="/generate/{label}" target="_blank">{label}</a>'
                labels.append(row['smiles'] + '__' + link)
            else:
                labels.append(row['smiles'])
        descriptors = tmap_constructor.mol_properties_from_df()
        end = time.time()
        logging.info(f"Labels took {end - start} seconds to create")

        # Plotting
        logging.info("Setting up TMAP and plotting")
        start = time.time()
        f = Faerun(
            view="front",
            coords=False,
            title="",
            clear_color="#FFFFFF",
        )

        f.add_scatter(
            "Descriptors",
            {
                "x": x,
                "y": y,
                "c": descriptors, 
                "labels":labels,
            },
            shader="smoothCircle",
            point_scale= TMAP_POINT_SCALE,
            max_point_size= 20,
            interactive=True,
            # legend_labels=[], # TODO: Get list of list of labels. This sould be something like [df[col] for col in self.categ_col]
            # categorical= bool_categorical, #TODO: Add support for categorical columns. 
            series_title= ['HAC', 'Fraction Aromatic Atoms', 'Number of Rings', 'clogP', 'Fraction Csp3', 'MW'], 
            has_legend=True,           
            colormap=['hsv', 'hsv', 'hsv', 'hsv', 'hsv', 'hsv'],
        
            categorical=[False, False, False, False, False, False],
        )
        f.add_tree(self.tmap_name+"_TMAP_tree", {"from": s, "to": t}, point_helper="Descriptors")
        f.plot(self.tmap_name+"_TMAP", template='smiles')
        
        logging.info(f"Plotting took {end - start} seconds") 
        logging.info('TMAP DONE')
        
        import gc 
        del self.dataframe, descriptors, x, y, labels, s, t 
        gc.collect()
