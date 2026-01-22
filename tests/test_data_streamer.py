import pytest
from pathlib import Path
from chelombus.streamer.data_streamer import DataStreamer


def test_parse_input_with_chunksize(tmp_path: Path):
    # Create a temporary file with known content.
    file_content = ["line1\n", "line2\n", "line3\n", "line4\n", "line5\n"]
    file_path = tmp_path / "test.txt"
    file_path.write_text("".join(file_content))
    
    streamer = DataStreamer()
    chunksize = 2
    # Expected chunks: [first two lines], [next two lines], [last line]
    result = list(streamer.parse_input(str(file_path), chunksize))
    assert result == [["line1", "line2"],["line3", "line4"], ["line5"]]
    
def test_parse_input_without_chunksize(tmp_path: Path):
    # Test when chunksize is None: the entire file should be returned as one chunk.
    file_content = ["line1\n", "line2\n", "line3\n"]
    file_path = tmp_path / "test.txt"
    file_path.write_text("".join(file_content))
    
    streamer = DataStreamer()
    result = list(streamer.parse_input(str(file_path), None))
    # Expect a single chunk containing all lines.
    assert result == [["line1", "line2", "line3"]]
    
def test_parse_input_file_not_found(tmp_path: Path):
    # Test that a FileNotFoundError is raised when the file does not exist.
    non_existent_path = tmp_path / "nonexistent.txt"
    streamer = DataStreamer()
    with pytest.raises(FileNotFoundError):
        list(streamer.parse_input(str(non_existent_path), 2))


def test_parse_input_omits_invalid_line_with_col_idx(tmp_path, monkeypatch):
    file_content = "valid_line1, valid_line11\n invalid_line\nvalid_line2m valid_line22\n"
    test_file = tmp_path / "test.txt"
    test_file.write_text(file_content)

    # The parse_input function calls _process_input from its globals, so we patch that.
    monkeypatch.setitem(DataStreamer.parse_input.__globals__, '_process_input', lambda input_paths: [str(test_file)])

    ds = DataStreamer()
    # We test col_idx=1, should return the second argument in the line (is doing line.split()[1])
    # Since we don't have a second argument in the second line, it will return a index out of list 
    # so it should omit the line and go to the next one, still return a chunksize = 2
    result = list(ds.parse_input(input_path=str(test_file), chunksize=2, verbose=0, smiles_col=1))
    
    # The expected result is one chunk with both valid lines.
    assert result == [["valid_line11", "valid_line22"]]


def test_parse_input_omits_invalid_line_without_chunksize(tmp_path, monkeypatch):
    # This test checks behavior when chunksize is None (i.e. yield all valid lines at once).
    file_content = "valid_line1\n\nvalid_line2\n"
    test_file = tmp_path / "test.txt"
    test_file.write_text(file_content)

    monkeypatch.setitem(DataStreamer.parse_input.__globals__, '_process_input', lambda input_paths: [str(test_file)])
    
    ds = DataStreamer()
    # When chunksize is None, the entire file (except invalid lines) is yielded as one chunk.
    result = list(ds.parse_input(input_path=str(test_file), chunksize=None, verbose=0))
    
    # Expected: one chunk containing only the valid lines.
    assert result == [["valid_line1", "valid_line2"]]
