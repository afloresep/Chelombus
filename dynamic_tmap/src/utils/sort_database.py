#!/usr/bin/env python3

import os
import sys
import xxhash
import time
from threading import Thread, Event

# Configuration
SEED = 42
NUM_OUTPUT_FILES = 10
OUTPUT_DIR = '/mnt/10tb_hdd/cleaned_enamine_10b'
OUTPUT_PREFIX = 'output_file_'
HASH_FUNC = xxhash.xxh64
BUFFER_SIZE = 32 * 1024 * 1024  # 16MB buffer

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global line counter
total_lines = 0
stop_event = Event()

def process_line(line, output_file_handles):
    molecule_smiles = line.strip().split()[0] # only take first column = smiles
    hash_value = HASH_FUNC(molecule_smiles.encode('utf-8'), seed=SEED).intdigest()
    file_index = hash_value % NUM_OUTPUT_FILES
    output_file_handles[file_index].write(f'{hash_value}\t{molecule_smiles}\n')

def process_input_file(input_file):
    global total_lines

    # Open output files in append mode
    output_file_paths = [os.path.join(OUTPUT_DIR, f'{OUTPUT_PREFIX}{i}.cxsmiles') for i in range(NUM_OUTPUT_FILES)]
    output_file_handles = [open(path, 'a', buffering=BUFFER_SIZE) for path in output_file_paths]

    with open(input_file, 'r', buffering=BUFFER_SIZE) as infile:
        for line in infile:
            process_line(line, output_file_handles)
            total_lines += 1

    # Close output file handles
    for f in output_file_handles:
        f.close()

def display_progress():
    """Function to print progress every second."""
    start_time = time.time()
    while not stop_event.is_set():
        elapsed_time = time.time() - start_time
        print(f"\rLines Processed: {total_lines:,} (Elapsed Time: {elapsed_time:.2f} seconds)", end="")
        time.sleep(1)
    print()  # Move to a new line after stopping

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: process_molecules.py input_file1 [input_file2 ... input_fileN]")
        sys.exit(1)

    input_files = sys.argv[1:]

    # Start the progress display thread
    progress_thread = Thread(target=display_progress)
    progress_thread.start()

    try:
        # Check if the input is a folder, if true then go through the files in the folder
        if len(input_files) == 1 and os.path.isdir(input_files[0]):
        # Process each input file
            for input_file in os.listdir(input_files[0]):
                print(f"Processing {input_file}")
                process_input_file(os.path.join(os.path.abspath(input_files[0]), input_file))
        else:
            for input_file in input_files:
                process_input_file(input_file)
    finally:
        # Stop the progress display and join the thread
        stop_event.set()
        progress_thread.join()
        print("\nProcessing complete.")
