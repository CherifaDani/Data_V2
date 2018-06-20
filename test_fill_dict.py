import unittest
from data_utils import fill_dict_from_df

class TestFillDict(unittest.TestCase):
    def test_var(self):
        # Test if the variable name exists
        self.assertIn(variable_name, dfs['SQL_Name'])