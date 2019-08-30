import os
import unittest

from xde.data_set_util import DataSetUtil

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES_PATH = os.path.join(ROOT_DIR, 'fixtures')
TRAINING_DATA_CSV = os.path.join(FIXTURES_PATH, 'xde_data_set.csv')
TRAINING_DATA_CSV_NB_ROWS = 20


class TestXDESurrogateModel(unittest.TestCase):

    def test_split_train_val(self):
        train_pct = 0.1
        train_set, val_set = DataSetUtil.split_train_val(TRAINING_DATA_CSV, train_pct=train_pct)
        expected_train_rows = int(round(TRAINING_DATA_CSV_NB_ROWS * train_pct))
        expected_bal_rows = TRAINING_DATA_CSV_NB_ROWS - expected_train_rows
        self.assertEqual(expected_train_rows, len(train_set))
        self.assertEqual(expected_bal_rows, len(val_set))
        nb_projects_in_common = len(set(train_set["Project"]).intersection(set(val_set["Project"])))
        self.assertEqual(0, nb_projects_in_common)
