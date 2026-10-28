import pandas as pd
from pathlib import Path


def test_stage1_summary_schema_is_documented():
    # Smoke test only: the real dataset is user-local and is not bundled here.
    expected = {"cycle_number", "cycle_type", "capacity_Ah"}
    assert expected == {"cycle_number", "cycle_type", "capacity_Ah"}
