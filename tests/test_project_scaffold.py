from pathlib import Path


def test_core_files_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "README.md").is_file()
    assert (root / "requirements.txt").is_file()
    assert (root / "src" / "inspect_mat.py").is_file()
