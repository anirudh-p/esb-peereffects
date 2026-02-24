import unittest
import pandas as pd

from Cleaning import merge_bus_district


class TestCleaning(unittest.TestCase):
    def test_merge_auto_detect_lea(self):
        bus = pd.DataFrame({"LEA Code": [1, 1, 2], "bus_val": [10, 11, 12]})
        district = pd.DataFrame({"LEA_ID": [1, 2], "district_val": [100, 200]})

        merged = merge_bus_district(bus, district)

        self.assertIn("district_val", merged.columns)
        self.assertEqual(len(merged), 3)
        self.assertEqual(set(merged[merged["LEA Code"] == 1]["district_val"]), {100})

    def test_merge_with_explicit_cols(self):
        bus = pd.DataFrame({"bus_lea": [10, 20], "b": [1, 2]})
        district = pd.DataFrame({"district_lea": [10, 20], "d": [9, 8]})

        merged = merge_bus_district(bus, district, bus_lea_col="bus_lea", district_lea_col="district_lea")

        self.assertEqual(len(merged), 2)
        self.assertIn("d", merged.columns)
        self.assertEqual(merged.loc[merged["bus_lea"] == 10, "d"].iloc[0], 9)

    def test_merge_raises_on_right_duplicates(self):
        bus = pd.DataFrame({"leaid": [1, 2]})
        district = pd.DataFrame({"leaid": [1, 1], "val": [5, 6]})

        with self.assertRaises(pd.errors.MergeError):
            merge_bus_district(bus, district)


if __name__ == "__main__":
    unittest.main()
