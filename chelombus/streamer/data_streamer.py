import os
from typing import Iterator, List, Optional

from chelombus.utils.helper_functions import _process_input

class DataStreamer:
    """
    Class for streaming large datasets in manageable chunks.
    """
    def parse_input(self, input_path: str, chunksize: Optional[int] = None, verbose: int = 0, smiles_col: int = 0) -> Iterator[List[str]]:
        """
        Reads input data from a file or a directory of files and yields the data in chunks.

        This method processes each file provided by the input path, which can be either a single file
        or a directory containing multiple files. Lines in plain-text files are split and the column at
        `smiles_col` is used as the SMILES string. SDF/SD files are handled in a streaming fashion via
        RDKit and converted to SMILES on the fly. For each file, the extracted SMILES strings are placed
        in a buffer until it reaches `chunksize`, at which point the buffer is yielded and cleared. If
        `chunksize` is None or there are remaining lines after processing a file, those are yielded as a
        final chunk.

        Args:
            input_path (str): The path to a file or directory containing input files.
            chunksize (Optional[int]): The number of lines to include in each yielded chunk. If None, 
                the entire file content is yielded as a single chunk.
            verbose (int): Level of verbosity. Default is 0.
            smiles_col (int): Column index for text inputs in which SMILES appear among several columns.
                Ignored for SDF inputs where SMILES strings are generated from molecule blocks.

        Yields:
            List[str]: A list of lines from the input file(s), where the list length will be equal to 
            `chunksize` except possibly for the final chunk.
        """ 
        # Sanity check
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Data source {input_path} not found.")
            
        buffer: List[str] = []

        def _flush_buffer() -> Iterator[List[str]]:
            if buffer:
                yield buffer[:]
                buffer.clear()

        for file_path in _process_input(input_paths=input_path):
            if verbose == 1:
                print("Processing file:", file_path)

            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext in {".sdf", ".sd"}:
                records = self._stream_sdf(file_path)
            else:
                records = self._stream_text(file_path, smiles_col)

            for smiles in records:
                buffer.append(smiles)
                if chunksize is not None and len(buffer) == int(chunksize):
                    yield buffer[:]
                    buffer.clear()

            # Process remaining items in the buffer for the current file
            yield from _flush_buffer()

    def _stream_text(self, file_path: str, smiles_col: int) -> Iterator[str]:
        with open(file_path, "r") as file:
            for line in file:
                try:
                    yield line.split()[smiles_col]
                except Exception as e:
                    print(f"\nAn exception occured with line: {line}. Raised Error: {e}\n")

    def _stream_sdf(self, file_path: str) -> Iterator[str]:
        try:
            from rdkit import Chem
        except ImportError as exc:
            raise ImportError(
                "RDKit is required to read SDF files. Please install rdkit to stream SDF inputs."
            ) from exc

        # ForwardSDMolSupplier streams molecules without loading the full file in memory
        with open(file_path, "rb") as file_handle:
            supplier = Chem.ForwardSDMolSupplier(file_handle)
            for mol in supplier:
                if mol is None:
                    continue
                try:
                    yield Chem.MolToSmiles(mol)
                except Exception as e:
                    print(f"\nFailed to convert molecule in {file_path} to SMILES: {e}\n")
