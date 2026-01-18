# Script to calculate the fingeprints from SMILES
import os
import numpy as np
from chelombus.streamer.data_streamer import DataStreamer
from chelombus.utils.fingerprints import  FingerprintCalculator
from chelombus.utils.helper_functions import format_time, save_chunk
from chelombus.encoder.encoder import PQEncoder
import time
import gc
import logging
logger = logging.getLogger(__name__)
import argparse
import joblib
import glob
import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process SMILES to Fingerprints with flexible options.")
    parser.add_argument('--input-path', type=str, help="The path to a file or directory containing SMILES files.") 
    parser.add_argument('--output-path', type=str, help="Output path for the fingerprints, pq-codes and other data generated")
    parser.add_argument('--chunksize', type=str, default=100_000, help=" The number of lines to include in each yielded chunk. Defualt 100k ")
    parser.add_argument('--col-idx', type=int, default=0, help="Column index for the smiles in the input data. This is for cases where the input data contains multiple columns. Default is 0 ")
    parser.add_argument('--file-format', type=str, default='parquet', help="File format for the fingerprints output file. Default is parquet")
    parser.add_argument('--fpSize', type=int, default=1024, help="Fingerprint Size. Default is 1024 bytes")
    parser.add_argument('--radius', type=int, default=3, help="Radius for the morgan fingeprint. Default is 3.")
    parser.add_argument('--verbose', type=int, default=1, help="Level of verbosity. Default is 1")
    parser.add_argument('--save-with-smiles', type=bool, default=True, help="Whether to save the fingerprints with their respective SMILES or not ")

    return parser.parse_args()


def calculate_fingerprints(input_path:str, 
                           chunksize:int, 
                           output_path:str, 
                           file_format:str,
                           fpSize:int,
                           radius:int, 
                           col_idx:int,
                           verbose:int, 
                           save_with_smiles:bool):

    if verbose > 0:
        logging.info("SMILES input path: ", input_path)
        logging.info("Fingerprint output path: ", output_path)
        logging.info("chunksize: ", chunksize)
        logging.info("Column index for SMILES: ", col_idx)
        logging.info("Output File Format: ", file_format)
        logging.info("fpSize: ", fpSize)
        logging.info("radius: ", radius)
        logging.info("Save with smiles: ", save_with_smiles)


    if save_with_smiles:
        assert file_format.lower() =='parquet', 'Saving with smiles is only compatible with `parquet` file format'
    
    ds = DataStreamer()
    fp_calc = FingerprintCalculator()
    smiles_chunk_idx = 0
    s = time.time() 
    for smiles_chunk in ds.parse_input(input_path, chunksize, verbose=verbose, col_idx=col_idx):
        try: 
            fp_chunk = fp_calc.FingerprintFromSmiles(smiles_chunk, fp='morgan', fpSize=fpSize, radius=radius)
        except Exception as e:
            print(f"An Exception has ocurred in chunk {smiles_chunk_idx}: {e}")
            continue
        try:
            # Make a separated folder for the fp within the output_path
            fp_dir = os.path.join(output_path, 'fingerprints')
            if save_with_smiles:
                smiles_df = pd.DataFrame({
                    'smiles': smiles_chunk
                })

                fp_df = pd.DataFrame(fp_chunk, columns=[f'fp_{binary_idx}' for binary_idx in range(fpSize)]) 

                final_df = pd.concat([smiles_df, fp_df], axis=1)
                save_chunk(final_df, fp_dir, chunk_index=smiles_chunk_idx, file_format=file_format)
        except Exception as e:
            print(f"An exception has ocurred while saving chunk: {smiles_chunk_idx}: {e}")
            continue
        del smiles_chunk, fp_chunk, fp_df, smiles_df
        smiles_chunk_idx += 1
        print(f"\rFingerprints calculated: {chunksize*smiles_chunk_idx:,}", end='', flush=True)
        gc.collect()
    e = time.time()
    print("Processed completed!")
    print(f"Fingerprint calculations for {smiles_chunk_idx*chunksize} molecules took: {format_time(e-s)}")


def main():
    args = parse_arguments()
    calculate_fingerprints(input_path=args.input_path, 
                           chunksize=args.chunksize, 
                           output_path=args.output_path, 
                           file_format = args.file_format, 
                           fpSize=args.fpSize, 
                           radius=args.radius, 
                           verbose=args.verbose, 
                           col_idx = args.col_idx, 
                           save_with_smiles = args.save_with_smiles)

if __name__=="__main__":
    main()