import csv
import os
import unittest

import numpy as np

from xde.data_set_encoder import DataSetEncoder

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES_PATH = os.path.join(ROOT_DIR, 'fixtures')
XDE_POSSIBLE_VALUES_CSV = os.path.join(FIXTURES_PATH, 'xde_possible_values.csv')
XDE_DATA_SET_CSV = os.path.join(FIXTURES_PATH, 'xde_data_set.csv')
PREDICTOR_MODEL_INPUTS_CSV = os.path.join(FIXTURES_PATH, 'predictor_model_inputs.csv')
PREDICTOR_MODEL_OUTPUTS_CSV = os.path.join(FIXTURES_PATH, 'predictor_model_outputs.csv')

XDE_COLUMNS = "Budget,BRD - Functional,STTM's - Technical,Datawarehouse 1,Datawarehouse 2,Datawarehouse 3," \
              "Data Visualization 1,Data Visualization 2,Data Visualization 3,Project Duration,Testing Schedule," \
              "Test Strategy & Design - Static,Test Strategy & Design - Dynamic,Functional Testing,Manual Testing," \
              "Automation Testing,Agile testing,Test Coverage,Cost,Speed,Quality"
XDE_BUDGET_POSSIBLE_VALUES = ["Fixed Bid- < 100K", "Fixed Bid-100K to 250K", "Fixed Bid-250K to 500K",
                              "Fixed Bid-500K to 1 Million", "Fixed Bid-1 million to 5 Million",
                              "Fixed Bid-5 Million to 10 Million"]


class TestDataSetEncoder(unittest.TestCase):

    def test_create_encoders(self):
        encoders = DataSetEncoder.get_possible_values_dict(XDE_POSSIBLE_VALUES_CSV)
        self.assertEquals(21, len(encoders))
        self.assertEquals(6, len(encoders["Budget"]))

    def test_encode(self):
        encoder = DataSetEncoder()
        possible_values_dict = DataSetEncoder.get_possible_values_dict(XDE_POSSIBLE_VALUES_CSV)
        df = encoder.encode(possible_values_dict, XDE_DATA_SET_CSV)
        self.assertIsNotNone(df["Budget"])
        self.assertEquals([0, 1, 0, 0, 0, 0], df["Budget"][0])
        self.assertIsNotNone(df["BRD - Functional"])
        self.assertEquals([1, 0], df["BRD - Functional"][0])
        self.assertEquals([0, 1], df["BRD - Functional"][5])
        self.assertEquals([1, 0], df["BRD - Functional"][6])
        # Project 8 has BRD - Functional set to NO (all uppercase). Make sure it's correctly encoded
        self.assertEqual(8, df["Project"][7])
        self.assertEquals([0, 1], df["BRD - Functional"][7])

    def test_decode(self):
        possible_values_dict = DataSetEncoder.get_possible_values_dict(XDE_POSSIBLE_VALUES_CSV)
        columns = ["Cost", "Speed", "Quality"]
        encoder = DataSetEncoder()
        # Test with 1 row
        encoded_cost = np.array([[1., 0., 0., 0., 0.]])
        encoded_speed = np.array([[1., 0., 0., 0., 0.]])
        encoded_quality = np.array([[1., 0., 0., 0., 0.]])
        encoded = [encoded_cost, encoded_speed, encoded_quality]
        decoded = encoder.decode(encoded, columns, possible_values_dict)
        # Expecting 3 lists back (cost, speed, quality), 1 row each
        self.assertEquals(3, len(decoded))
        self.assertEquals(1, len(decoded[0]))
        self.assertEquals(1, len(decoded[1]))
        self.assertEquals(1, len(decoded[2]))
        row = 0
        self.assertEquals("0.00 to 0.49", decoded[0][row])
        self.assertEquals("0.00 to 0.49", decoded[1][row])
        self.assertEquals("0.00 to 0.49", decoded[2][row])
        # Test with 2 rows
        encoded_cost = np.array([[0., 0., 0., 0., 1.], [0., 0., 0., 1., 0.]])
        encoded_speed = np.array([[0., 0., 0., 1., 0.], [0., 0., 1., 0., 0.]])
        encoded_quality = np.array([[0., 0., 1., 0., 0.], [0., 1., 0., 0., 0.]])
        encoded = [encoded_cost, encoded_speed, encoded_quality]
        decoded = encoder.decode(encoded, columns, possible_values_dict)
        self.assertEquals(3, len(decoded))
        self.assertEquals(2, len(decoded[0]))
        self.assertEquals(2, len(decoded[1]))
        self.assertEquals(2, len(decoded[2]))

        row = 0
        self.assertEquals("0.90 to 1.00", decoded[0][row])
        self.assertEquals("0.70 to 0.89", decoded[1][row])
        self.assertEquals("0.60 to 0.69", decoded[2][row])
        row = 1
        self.assertEquals("0.70 to 0.89", decoded[0][row])
        self.assertEquals("0.60 to 0.69", decoded[1][row])
        self.assertEquals("0.50 to 0.59", decoded[2][row])

    def test_get_possible_values_dict(self):
        pvd = DataSetEncoder.get_possible_values_dict(XDE_POSSIBLE_VALUES_CSV)
        cols = list(pvd.keys())
        expected_cols = XDE_COLUMNS.split(',')
        self.assertListEqual(expected_cols, cols)
        self.assertListEqual(XDE_BUDGET_POSSIBLE_VALUES, pvd["Budget"])

    def test_generate_model_description(self):
        nb_hidden_units = 10
        use_bias = True
        possible_values_dict = DataSetEncoder.get_possible_values_dict(XDE_POSSIBLE_VALUES_CSV)
        inputs_csv = PREDICTOR_MODEL_INPUTS_CSV
        outputs_csv = PREDICTOR_MODEL_OUTPUTS_CSV
        experiment_params = DataSetEncoder.generate_model_description(possible_values_dict,
                                                                      inputs_csv=inputs_csv,
                                                                      outputs_csv=outputs_csv,
                                                                      nb_hidden_units=nb_hidden_units,
                                                                      use_bias=use_bias)
        self.assertIsNotNone(experiment_params)
        # Check the inputs
        with open(inputs_csv, 'r') as inputs_csv_file:
            # Read the CSV and only look at the first row
            reader = csv.reader(inputs_csv_file, delimiter=',')
            expected_inputs = next(reader)

        actual_inputs = experiment_params["network"]["inputs"]
        self.assertEqual(len(expected_inputs), len(actual_inputs), "Missing some input columns!")
        actual_input_names = [model_input["name"] for model_input in actual_inputs]
        for expected_input in expected_inputs:
            self.assertTrue(expected_input in actual_input_names, "Missing input column {}!".format(expected_input))

        # Check the first input: Budget
        self.assertEqual("Budget", actual_inputs[0]["name"])
        self.assertListEqual(XDE_BUDGET_POSSIBLE_VALUES, actual_inputs[0]["values"])
        self.assertEqual(len(XDE_BUDGET_POSSIBLE_VALUES), actual_inputs[0]["size"])

        # Check the outputs.
        with open(outputs_csv, 'r') as outputs_csv_file:
            # Read the CSV and only look at the first row
            reader = csv.reader(outputs_csv_file, delimiter=',')
            expected_outputs = next(reader)

        # Make sure the output columns are not in the input columns
        for expected_output in expected_outputs:
            self.assertTrue(expected_output not in actual_input_names,
                            "Output column {} should not be in input columns!".format(expected_output))

        # Check the output columns
        actual_outputs = experiment_params["network"]["outputs"]
        self.assertEqual(len(expected_outputs), len(actual_outputs), "Missing some output columns!")
        actual_output_names = [model_output["name"] for model_output in actual_outputs]
        for expected_output in expected_outputs:
            self.assertTrue(expected_output in actual_output_names,
                            "Missing output column {}!".format(expected_output))

        # Checkout hidden units and bias
        self.assertEqual(nb_hidden_units, experiment_params["network"]["nb_hidden_units"])
        self.assertEqual(use_bias, experiment_params["network"]["use_bias"])
