import json
import os
import pandas as pd
from unittest import mock
from unittest import TestCase
from unittest.mock import Mock
import numpy as np
from keras.models import load_model

from xde.evaluator import XDEEvaluator

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES_PATH = os.path.join(ROOT_DIR, 'fixtures')
CONTEXT_CSV = os.path.join(FIXTURES_PATH, 'context.csv')
DECISIONS_CSV = os.path.join(FIXTURES_PATH, "decisions.csv")
PREDICTOR_MODEL_H5 = os.path.join(FIXTURES_PATH, 'predictor_model.h5')
PRESCRIPTOR_MODEL_H5 = os.path.join(FIXTURES_PATH, 'prescriptor_model.h5')
EXPERIMENT_PARAMS = os.path.join(FIXTURES_PATH, 'prescriptor_model.json')
TRAINING_DATA_CSV = os.path.join(FIXTURES_PATH, 'xde_data_set.csv')
XDE_POSSIBLE_VALUES_CSV = os.path.join(FIXTURES_PATH, 'xde_possible_values.csv')
MODEL_PNG = EXPERIMENT_PARAMS.replace(".json", ".png")


class TestXDEEvaluator(TestCase):

    @mock.patch("esp_sdk.v1_0.esp_service.EspService.__new__", autospect=True)
    def test_evaluate_candidate(self, mock_esp_service):
        # Mock the ESP service
        mock_esp_service.return_value = Mock()
        base_model = load_model(PRESCRIPTOR_MODEL_H5)
        mock_esp_service.return_value.request_base_model.return_value = base_model

        # Create an evaluator
        # Load the predictor
        with open(EXPERIMENT_PARAMS) as json_data:
            experiment_params = json.load(json_data)
        predictor = load_model(PREDICTOR_MODEL_H5)
        # Context
        with open(CONTEXT_CSV) as csv_file:
            context_df = pd.read_csv(csv_file, sep=',')
        context_names = list(context_df.columns)
        # Decisions
        with open(DECISIONS_CSV) as csv_file:
            decisions_df = pd.read_csv(csv_file, sep=',')
        decisions_names = list(decisions_df.columns)
        # Evaluator
        evaluator = XDEEvaluator(predictor, context_names, decisions_names,
                                 XDE_POSSIBLE_VALUES_CSV, TRAINING_DATA_CSV,
                                 experiment_params)

        # Get an individual
        candidate = load_model(PRESCRIPTOR_MODEL_H5)

        # And evaluate it
        fitness = evaluator.evaluate_candidate(candidate)
        self._check_fitness(fitness)

    def _check_fitness(self, fitness):
        nb_metrics = 3
        best_score_per_metric = 4
        best_score_per_sample = best_score_per_metric * nb_metrics
        nb_sample = 20
        lowest_score = 0
        highest_score = best_score_per_sample * nb_sample
        # Fitness should be greater than lowest score
        self.assertGreaterEqual(fitness, lowest_score)
        # And lower than highest score
        self.assertLessEqual(fitness, highest_score)

    def test_compute_fitness(self):
        """
        Test XDEEvaluator compute_fitness
        We have 3 predictions: Cost, Schedule, Quality
        For the 3 the predictions we have 5 possibilities:
            0.0 to 0.3
            0.31 to 0.5
            0.51 to 0.7
            0.71 to 0.8
            0.81 to 0.9
            0.91 to 1
        For each prediction, higher is better
        Score: 0 if found the worst solution, 5 if found the best
        :return: nothing
        """

        # Evaluating the results of a prediction on 3 samples
        # Best case: low cost, high speed and high quality
        cost = np.array([[0., 0., 0., 0., 0., 1.], [0., 0., 0., 0., 0., 1.], [0., 0., 0., 0., 0., 1.]])
        speed = np.array([[0., 0., 0., 0., 0., 1.], [0., 0., 0., 0., 0., 1.], [0., 0., 0., 0., 0., 1.]])
        quality = np.array([[0., 0., 0., 0., 0., 1.], [0., 0., 0., 0., 0., 1.], [0., 0., 0., 0., 0., 1.]])

        predictions = [cost, speed, quality]
        fitness = XDEEvaluator.compute_fitness(predictions)
        self.assertEquals(45, fitness, "Best score should be 45 points: 5 for each prediction for each sample, "
                                       "but was {}".format(fitness))

        # Average case
        # Cost: 1 + 2 + 5 = 8
        cost = np.array([[0., 1., 0., 0., 0., 0.], [0., 0., 1., 0., 0., 0.], [0., 0., 0., 0., 0., 1.]])
        # Speed: 1 + 2 + 3 = 6
        speed = np.array([[0., 1., 0., 0., 0., 0.], [0., 0., 1., 0., 0., 0.], [0., 0., 0., 1., 0., 0.]])
        # Quality: 2 + 3 + 4 = 9
        quality = np.array([[0., 0., 1., 0., 0., 0.], [0., 0., 0., 1., 0., 0.], [0., 0., 0., 0., 1., 0.]])

        predictions = [cost, speed, quality]
        fitness = XDEEvaluator.compute_fitness(predictions)
        self.assertEquals(23, fitness, "Best score should be 23 points: cost: 8 + speed: 6 + quality: 9, "
                                       "but was {}".format(fitness))

        # Worst case
        # Worst score is 0: 0 for each prediction: highest cost, lowest speed and lowest quality
        cost = np.array([[1., 0., 0., 0., 0., 0.], [1., 0., 0., 0., 0., 0.], [1., 0., 0., 0., 0., 0.]])
        speed = np.array([[1., 0., 0., 0., 0., 0.], [1., 0., 0., 0., 0., 0.], [1., 0., 0., 0., 0., 0.]])
        quality = np.array([[1., 0., 0., 0., 0., 0.], [1., 0., 0., 0., 0., 0.], [1., 0., 0., 0., 0., 0.]])

        predictions = [cost, speed, quality]
        fitness = XDEEvaluator.compute_fitness(predictions)
        self.assertEquals(0, fitness, "Best score should be 0 points: 0 for each prediction for each sample, "
                                      "but was {}".format(fitness))
