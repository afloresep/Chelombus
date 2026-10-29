import pytest

pytest.importorskip("tmap")

from chelombus.utils.visualization import create_tmap

SMILES = ["C" * i + "O" for i in range(1, 30)] + ["c1ccccc1" + "C" * i for i in range(1, 20)]


@pytest.mark.parametrize("fingerprint", ["mqn", "morgan"])
def test_create_tmap_writes_html(fingerprint, tmp_path, monkeypatch):
    """create_tmap works for both count (MQN) and binary (Morgan) fingerprints."""
    monkeypatch.chdir(tmp_path)

    create_tmap(SMILES, fingerprint=fingerprint, properties=["mw", "logp"],
                tmap_name="test_map", k_neighbors=5)

    assert (tmp_path / "test_map.html").exists()
