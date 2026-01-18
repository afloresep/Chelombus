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


"""
The basic logic behind this is to create our training data by going through
every fingerprint chunk we have, choose a random sample of that file and move to the next one. 
There's probably better / faster ways of doing this. e.g. choose X random files and pick N samples from that
instead of going through every file. However, I want to make sure that the sampling is as random and diverse as
possible. And giving the amount of time that the whole pipeline takes spending one more hour sampling does not seem
excesive

Args:
- fp_path (str): Path to the folder containig all fingerprint chunk files
- training_size (int): Number of fingerprints to have in the final training data file 
- file_format (str): If the fp_path contains other files than the fingerprints, filter to only those files that matches file_format. Default is None (pass all files)
-

    Did some testing on size and speed saving with different methods. Overall np.save or joblib are the fastest methods.
 -------------------------------------------------------
| method               | saving time   | Loading time   |
| ----------------------------------------------------- |
| np.save              | Save: 27.704s | Load: 6.802s   |
| np.savez_compressed  | Save: 44.703s | Load: 39.083s  |
| pickle               | Save: 58.335s | Load: 6.892s   |
| joblib               | Save: 21.249s | Load: 8.224s   |
| Parquet (pyarrow)    | Save: 107.587s| Load: 12.236s  |
  ------------------------------------------------------
"""  


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process with flexible options.")
    parser.add_argument('--input-path', type=str, help="The path to a file or directory containing SMILES files.") 
    parser.add_argument('--output-path', type=str, default='.', help="Output path for the training data generated")
    parser.add_argument('--training-size', type=int, default=100_000, help="Amount of fingerprints to include in the training data")
    parser.add_argument('--file-filter', type=str, default= None, help="File format for the fingeprint files. It will filter all files in the `--input-path` by that file format (e.g. parquet)")
    parser.add_argument('--fpSize', type=int, default=1024, help="Size of the fingerprint that is going to be read. Default is 1024")

    return parser.parse_args()


def get_training_data(fp_path:str, 
                      output_path:str,
                      training_size:int,
                      fpSize:int, 
                      file_format:str=None, 
                      ):

    # Create output directory if it doesnt exist
    os.makedirs(output_path, exist_ok=True)

    # Folder where the parquet files are stored 
    if file_format is not None:
        fp_path= fp_path+f'/*.{file_format.lower()}'

        try:
            files = glob.glob(fp_path)
        except Exception as e:
            raise ValueError(f"Failed when listing fingerprint files. Raise exception {e}")


    if file_format is None:
        files = os.listdir(fp_path)
    
    file_ext = files[0].split('.')[-1].lower()
    if file_ext not in ['parquet', 'npy']:
        raise ValueError('Only {.parquet, .npy} are supported for the fingeprint chunks', file_ext)

    number_files = int(len(files))

    # Number of fingerprints to get from each fp chunk
    fp_sample_size=int(training_size/(number_files))
    assert fp_sample_size > 0, "The number of fingeprints per file (training_size / number of files) must be greater than 0"
    
    training_data = np.empty([(fp_sample_size*number_files), fpSize], dtype='uint8')
    
    if file_ext == 'parquet':
        for i, file in enumerate(files):
            # Handle both glob results (full paths) and listdir results (filenames only)
            file_path = file if file_format else os.path.join(fp_path, file)
            fp_df = pd.read_parquet(file_path, engine='pyarrow')
            assert len(fp_df) > fp_sample_size, "The number of fingerprints in the file must be larger than the fraction sample size"
            fp_df_sample = fp_df.sample(fp_sample_size)
            arr = fp_df_sample.to_numpy()
            assert arr.shape[1] == fpSize, f"The fingerprint size of the read fingerprint should be the same as the selected fingerprint size for the training data. Read fpSize={arr.shape[1]} but got args.fpSize={fpSize}"
            training_data[i * int(fp_sample_size):(i + 1) * int(fp_sample_size), :] = arr
            print(f"\rTraining size: {(i + 1) * int(fp_sample_size):,}/{fp_sample_size * number_files:,}. Files used: {i + 1}/{number_files}", end='', flush=True)

    if file_ext == 'npy':
        for i, file in enumerate(files):
            fp_arr = np.load(file if file_format else os.path.join(fp_path, file))
            assert fp_arr.shape[0] > fp_sample_size, "The number of fingerprints in the file must be larger than the fraction sample size"
            fp_sample_indices = np.random.choice(fp_arr.shape[0], size=fp_sample_size, replace=False)
            training_data[i * int(fp_sample_size):(i + 1) * int(fp_sample_size), :] = fp_arr[fp_sample_indices]
            print(f"\rTraining size: {(i + 1) * int(fp_sample_size):,}/{fp_sample_size * number_files:,}. Files used: {i + 1}/{number_files}", end='', flush=True) 

    
    print(f"\nSaving {training_data.shape} training samples in {output_path}")
    training_data_output_path = os.path.join(output_path, 'training_data.joblib')
    print(training_data_output_path)
    joblib.dump(training_data, training_data_output_path)


def main():
    args = parse_arguments()

    s = time.time()
    get_training_data(fp_path=args.input_path,
                      output_path=args.output_path,
                      training_size=args.training_size, 
                      fpSize=args.fpSize,
                      file_format=args.file_filter)
    e = time.time()

    print("Getting training data took: ", format_time(e-s))
    
if __name__ == "__main__":
    main()



