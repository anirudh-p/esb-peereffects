import pandas as pd

from Cleaning import merge_bus_district


def test_merge_auto_detect_lea():
    # bus-level has "LEA Code", district has "LEA_ID" -> auto-detect should find both
    bus = pd.DataFrame({"LEA Code": [1, 1, 2], "bus_val": [10, 11, 12]})
    district = pd.DataFrame({"LEA_ID": [1, 2], "district_val": [100, 200]})

    merged = merge_bus_district(bus, district)

    # merged should contain district columns and preserve bus row count
    assert "district_val" in merged.columns
    assert len(merged) == 3
    # values should match on LEA id
    assert set(merged[merged["LEA Code"] == 1]["district_val"]) == {100}


def test_merge_with_explicit_cols():
    bus = pd.DataFrame({"bus_lea": [10, 20], "b": [1, 2]})
    district = pd.DataFrame({"district_lea": [10, 20], "d": [9, 8]})

    merged = merge_bus_district(bus, district, bus_lea_col="bus_lea", district_lea_col="district_lea")

    assert len(merged) == 2
    assert "d" in merged.columns
    assert merged.loc[merged["bus_lea"] == 10, "d"].iloc[0] == 9


def test_merge_raises_on_right_duplicates():
    # if the district (right) table has duplicate LEA ids, validate='m:1' should raise
    bus = pd.DataFrame({"leaid": [1, 2]})
    district = pd.DataFrame({"leaid": [1, 1], "val": [5, 6]})

    raised = False
    try:
        merge_bus_district(bus, district)
    except pd.errors.MergeError:
        raised = True
    assert raised
