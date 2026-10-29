import pytest

pytest.importorskip("tmap")

from chelombus.utils.visualization import create_tmap, representative_tmap

SMILES = ["C" * i + "O" for i in range(1, 30)] + ["c1ccccc1" + "C" * i for i in range(1, 20)]


@pytest.mark.parametrize("fingerprint", ["mqn", "morgan"])
def test_create_tmap_writes_html(fingerprint, tmp_path, monkeypatch):
    """create_tmap works for both count (MQN) and binary (Morgan) fingerprints."""
    monkeypatch.chdir(tmp_path)

    create_tmap(SMILES, fingerprint=fingerprint, properties=["mw", "logp"],
                tmap_name="test_map", k_neighbors=5)

    assert (tmp_path / "test_map.html").exists()


@pytest.mark.parametrize("fingerprint", ["mqn", "morgan"])
def test_create_tmap_skips_invalid_smiles(fingerprint, tmp_path, monkeypatch):
    """Unparseable SMILES are dropped instead of misaligning property layers."""
    monkeypatch.chdir(tmp_path)

    create_tmap(SMILES + ["not_a_smiles"], fingerprint=fingerprint, properties=["mw"],
                tmap_name="test_map", k_neighbors=5)

    assert (tmp_path / "test_map.html").exists()


def test_representative_tmap_aligns_cluster_ids(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    smiles = SMILES[:10] + ["not_a_smiles"] + SMILES[10:]
    representative_tmap(smiles, cluster_ids=list(range(len(smiles))),
                        properties=["mw"], tmap_name="rep", k_neighbors=5)

    assert (tmp_path / "rep.html").exists()


def test_cli_cluster_file_without_smiles(tmp_path, monkeypatch):
    """--cluster-file alone is enough (it used to also require --smiles)."""
    import pandas as pd
    from chelombus.utils import visualization

    monkeypatch.chdir(tmp_path)
    pd.DataFrame({"smiles": SMILES, "cluster_id": [i % 3 for i in range(len(SMILES))]}).to_csv(
        tmp_path / "reps.csv", index=False)
    monkeypatch.setattr("sys.argv", ["chelombus-tmap", "--cluster-file", "reps.csv",
                                     "--properties", "mw", "--output", "cli_map"])

    visualization.main()

    assert (tmp_path / "cli_map.html").exists()
