import os
import time
import xxhash
import random
import string

# Configuration
TEST_FILE = "test_large_smiles.txt"
NUM_LINES = 50_000_000  # 5 million lines (~1GB)
BUFFER_SIZES = [16 * 1024 * 1024, 32 * 1024 * 1024, 64 * 1024 * 1024, 128 * 1024 * 1024, 256 * 1024 * 1024]
NUM_OUTPUT_FILES = 10
OUTPUT_DIR = "test_output"
OUTPUT_PREFIX = "output_file_"
SEED = 42
HASH_FUNC = xxhash.xxh64

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_random_smiles():
    """Generate a random SMILES-like string for testing."""
    chars = string.ascii_letters + "1234567890=CNOS()[]+"
    return ''.join(random.choices(chars, k=random.randint(10, 50)))

def create_test_file():
    """Create a large test file with random SMILES strings."""
    print("Generating test file (this may take a few minutes)...")
    with open(TEST_FILE, "w", buffering=128 * 1024 * 1024) as f:
        for _ in range(NUM_LINES):
            smiles = generate_random_smiles()
            f.write(f"11958573449277712313\t{smiles}\tm_282970\t289.299\t21\t0.169\t7\t2\t2\t0.417\t109.060\t0.815\tTrue\tTrue\tM\tXRXRUTHABDVAIK-UHFFFAOYSA-N\n")
    print(f"Test file '{TEST_FILE}' created with {NUM_LINES} lines.")

def process_test_file(buffer_size):
    """Read and process the test file using the given buffer size."""
    output_file_paths = [
        os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}{i}.cxsmiles")
        for i in range(NUM_OUTPUT_FILES)
    ]
    output_file_handles = [
        open(path, 'w', buffering=buffer_size) for path in output_file_paths
    ]

    start_time = time.time()

    with open(TEST_FILE, 'r', buffering=buffer_size) as infile:
        for line in infile:
            parts = line.split()
            if len(parts) < 2:
                continue
            smiles = parts[1]
            hash_value = HASH_FUNC(smiles.encode('utf-8'), seed=SEED).intdigest()
            file_index = hash_value % NUM_OUTPUT_FILES
            output_file_handles[file_index].write(smiles + "\n")

    # Close output files
    for f in output_file_handles:
        f.close()

    elapsed_time = time.time() - start_time
    print(f"Buffer {buffer_size // (1024*1024)}MB: {elapsed_time:.2f} sec")
    return elapsed_time

def main():
    """Run tests for different buffer sizes and find the fastest."""
    if not os.path.exists(TEST_FILE):
        create_test_file()

    results = {}
    for buffer_size in BUFFER_SIZES:
        elapsed_time = process_test_file(buffer_size)
        results[buffer_size] = elapsed_time

    # Find the best buffer size
    best_buffer = min(results, key=results.get)
    print("\n==== Benchmark Results ====")
    for buf, time_taken in results.items():
        print(f"Buffer {buf // (1024 * 1024)}MB: {time_taken:.2f} sec")

    print(f"\nBest buffer size: {best_buffer // (1024 * 1024)}MB")

if __name__ == "__main__":
    main()
