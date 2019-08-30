import json
import os
from unittest import mock
from unittest import TestCase
from unittest.mock import Mock

from numpy.testing import assert_array_equal
from keras.models import load_model

from xde.surrogate_model import XDESurrogateModel


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES_PATH = os.path.join(ROOT_DIR, 'fixtures')
MODEL_JSON = os.path.join(FIXTURES_PATH, 'predictor_model.json')
MODEL_PNG = MODEL_JSON.replace(".json", ".png")
MODEL_H5 = MODEL_JSON.replace(".json", ".h5")

TRAINING_DATA_CSV = os.path.join(FIXTURES_PATH, 'xde_data_set.csv')
TRAINING_DATA_ENCODED_CSV = os.path.join(FIXTURES_PATH, 'xde_encoded_data_set.csv')
TRAINING_DATA_CSV_NB_ROWS = 20
INPUT_COLS = ["Budget", "BRD - Functional", "STTM's - Technical", "Datawarehouse 1", "Datawarehouse 2",
              "Datawarehouse 3", "Data Visualization 1", "Data Visualization 2", "Data Visualization 3",
              "Project Duration", "Testing Schedule", "Test Strategy & Design - Static",
              "Test Strategy & Design - Dynamic", "Functional Testing", "Manual Testing", "Automation Testing",
              "Agile testing", "Test Coverage"]
OUTPUT_COLS = ["Cost", "Speed", "Quality"]
OUTPUT_COLS_LENGTH = [5, 5, 5]


class TestXDESurrogateModel(TestCase):

    def test_create_inputs_outputs(self):
        with open(MODEL_JSON) as json_data:
            experiment_params = json.load(json_data)

        surrogate_model = XDESurrogateModel()
        inputs, outputs = surrogate_model.create_inputs_outputs(experiment_params, TRAINING_DATA_ENCODED_CSV)
        # Check the inputs
        self.assertEqual(18, len(INPUT_COLS))
        self.assertEqual(18, len(inputs))
        self.assertEqual(TRAINING_DATA_CSV_NB_ROWS, inputs[0].shape[0])
        self.assertEqual(6, inputs[0].shape[1])
        self.assertEqual(TRAINING_DATA_CSV_NB_ROWS, inputs[1].shape[0])
        self.assertEqual(2, inputs[1].shape[1])
        # Check the outputs: Cost, Speed, Quality. 5 choices each.
        self.assertEqual(3, len(OUTPUT_COLS))
        self.assertEqual(3, len(outputs))
        # Cost
        self.assertEqual(TRAINING_DATA_CSV_NB_ROWS, outputs[0].shape[0])
        self.assertEqual(5, outputs[0].shape[1])
        # Speed
        self.assertEqual(TRAINING_DATA_CSV_NB_ROWS, outputs[1].shape[0])
        self.assertEqual(5, outputs[1].shape[1])
        # Quality
        self.assertEqual(TRAINING_DATA_CSV_NB_ROWS, outputs[2].shape[0])
        self.assertEqual(5, outputs[2].shape[1])

    @mock.patch("esp_sdk.v1_0.generated.population_service_pb2_grpc.PopulationServiceStub")
    def test_train_model(self, mock_esp_service):
        mock_esp_service.return_value = Mock()
        base_model = load_model(MODEL_H5)
        mock_esp_service.return_value.request_base_model.return_value = base_model

        # Get the experiment params
        with open(MODEL_JSON) as json_data:
            experiment_params = json.load(json_data)

        # Train the model
        surrogate_model = XDESurrogateModel()
        keras_model = surrogate_model.train_from_data(mock_esp_service.return_value,
                                                      experiment_params,
                                                      TRAINING_DATA_ENCODED_CSV,
                                                      batch_size=20,
                                                      epochs=6000,
                                                      verbose=0)

        save = False
        if save:
            keras_model.save(format(MODEL_H5))

        self._check_model(keras_model, TRAINING_DATA_ENCODED_CSV, MODEL_JSON)

    def test_persisted_model(self):
        from keras.models import load_model
        keras_model = load_model(MODEL_H5)
        self._check_model(keras_model, TRAINING_DATA_ENCODED_CSV, MODEL_JSON)

    @staticmethod
    def _check_model(model, training_data_csv, model_json):
        with open(model_json) as json_data:
            experiment_params = json.load(json_data)

        surrogate_model = XDESurrogateModel()
        data, labels = surrogate_model.create_inputs_outputs(experiment_params, training_data_csv)
        predictions = model.predict(data)
        # Check the predictions
        for label, prediction in zip(labels, predictions):
            assert_array_equal(label, prediction, "Trained model produced a bad prediction")
        print("Model is good!")
