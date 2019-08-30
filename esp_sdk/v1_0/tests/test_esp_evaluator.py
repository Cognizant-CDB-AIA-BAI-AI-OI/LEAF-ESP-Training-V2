import json
import os
import pickle
import unittest
from unittest.mock import patch

from esp_sdk.v1_0.esp_evaluator import EspNNWeightsEvaluator
from esp_sdk.v1_0.extension_packaging import ExtensionPackaging
from esp_sdk.v1_0.generated.population_structs_pb2 import Candidate
from esp_sdk.v1_0.generated.population_structs_pb2 import PopulationResponse

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES_PATH = os.path.join(ROOT_DIR, 'fixtures')
EXPERIMENT_JSON = os.path.join(FIXTURES_PATH, 'experiment_params.json')

C1_MODEL = "c1_model"
C2_MODEL = "c2_model"

C1_SCORE = 111
C2_SCORE = 222
E1_SCORE = 333

SCORES = {C1_MODEL: C1_SCORE,
          C2_MODEL: C2_SCORE}

C1_TIME = 444
C2_TIME = 555

TIMES = {C1_MODEL: C1_TIME,
         C2_MODEL: C2_TIME}


class EspNNWeightsEvaluatorForTests(EspNNWeightsEvaluator):
    """
    Evaluator for test purposes.
    """

    def __init__(self, experiment_params):
        super().__init__(experiment_params)
        self.nb_candidates_evaluated = 0

    def get_keras_model(self, weights):
        return weights

    def evaluate_candidate(self, candidate):
        return SCORES[candidate]


class EspNNWeightsMultiMetricsEvaluatorForTests(EspNNWeightsEvaluator):
    """
    Evaluator the return multiple metrics, for test purposes
    """

    def __init__(self, experiment_params):
        super().__init__(experiment_params)
        self.nb_candidates_evaluated = 0

    def get_keras_model(self, weights):
        return weights

    def evaluate_candidate(self, candidate):
        metrics = {"score": SCORES[candidate],
                   "time": TIMES[candidate]}
        return metrics


class TestEspNNWeightsEvaluator(unittest.TestCase):

    def setUp(self):
        # Executed before each test
        self.extension_packaging = ExtensionPackaging()
        with open(EXPERIMENT_JSON) as json_data:
            self.experiment_params = json.load(json_data)

    # Mock where the class is used, i.e. in esp_evaluator
    @patch('esp_sdk.v1_0.esp_evaluator.EspService', autospec=True)
    def test_constructor(self, esp_service_mock):
        evaluator = EspNNWeightsEvaluatorForTests(self.experiment_params)
        self.assertIsNotNone(evaluator)
        # Make sure we called the ESP service mock to get a base model
        self.assertEqual(1, esp_service_mock.return_value.request_base_model.call_count)

    @patch('esp_sdk.v1_0.esp_evaluator.EspService', autospec=True)
    def test_evaluate_population(self, esp_service_mock):
        # If nothing is specified in the experiment_params, re-evaluate elites
        self._evaluate_population(esp_service_mock, reevaluate_elites=True)

    @patch('esp_sdk.v1_0.esp_evaluator.EspService', autospec=True)
    def test_reevaluate_elites(self, esp_service_mock):
        # Reevaluate elites
        self.experiment_params["LEAF"]["reevaluate_elites"] = True
        self._evaluate_population(esp_service_mock, reevaluate_elites=True)

    @patch('esp_sdk.v1_0.esp_evaluator.EspService', autospec=True)
    def test_do_not_reevaluate_elites(self, esp_service_mock):
        # Do NOT reevaluate elites
        self.experiment_params["LEAF"]["reevaluate_elites"] = False
        self._evaluate_population(esp_service_mock, reevaluate_elites=False)

    def _evaluate_population(self, esp_service_mock, reevaluate_elites):
        evaluator = EspNNWeightsEvaluatorForTests(self.experiment_params)
        # Make sure we called the ESP service mock to get a base model
        self.assertEqual(1, esp_service_mock.return_value.request_base_model.call_count)

        # Create a population
        response = self._create_population_response()
        # And evaluate it
        evaluator.evaluate_population(response)

        # Check c1
        c = response.population[0]
        metrics_json = self.extension_packaging.from_extension_bytes(c.metrics)
        score = metrics_json['score']
        if reevaluate_elites:
            # This candidate is an elite: we want to make sure it has been re-evaluated and it's score
            # is the re-evaluated score, not the elite score.
            self.assertEqual(C1_SCORE, score, "This elite candidate should have been re-evaluated")
        else:
            # This candidate is an elite, and we make sure we have NOT re-evaluated it
            self.assertEqual(E1_SCORE, score, "This elite candidate should still have its elite score")

        # Check c2
        c = response.population[1]
        score_json = self.extension_packaging.from_extension_bytes(c.metrics)
        score = score_json['score']
        self.assertEqual(C2_SCORE, score)

    def _create_population_response(self):
        population = []

        # elite
        c1 = Candidate()
        c1.id = "1_1"
        c1.interpretation = pickle.dumps(C1_MODEL)
        # This is an elite: it has already been evaluated anc already contains a score
        c1.metrics = self.extension_packaging.to_extension_bytes({"score": E1_SCORE})
        c1.identity = pickle.dumps("C1 identity")
        population.append(c1)

        # new candidate
        c2 = Candidate()
        c2.id = "2_1"
        c2.interpretation = pickle.dumps(C2_MODEL)
        # c2.metrics = self.extension_packaging.to_extension_bytes(None)
        c2.identity = pickle.dumps("C2 identity")
        population.append(c2)

        response = PopulationResponse(population=population)
        return response

    @patch('esp_sdk.v1_0.esp_evaluator.EspService', autospec=True)
    def test_evaluate_population_multi_metrics(self, esp_service_mock):
        evaluator = EspNNWeightsMultiMetricsEvaluatorForTests(self.experiment_params)
        # Make sure we called the ESP service mock to get a base model
        self.assertEqual(1, esp_service_mock.return_value.request_base_model.call_count)

        # Create a population
        response = self._create_population_response()
        # And evaluate it
        evaluator.evaluate_population(response)

        # Check c1
        c = response.population[0]
        metrics_json = self.extension_packaging.from_extension_bytes(c.metrics)
        score = metrics_json['score']
        self.assertEqual(C1_SCORE, score)
        time = metrics_json['time']
        self.assertEquals(C1_TIME, time)

        # Check c2
        c = response.population[1]
        score_json = self.extension_packaging.from_extension_bytes(c.metrics)
        score = score_json['score']
        self.assertEqual(C2_SCORE, score)
        time = metrics_json['time']
        self.assertEquals(C1_TIME, time)
