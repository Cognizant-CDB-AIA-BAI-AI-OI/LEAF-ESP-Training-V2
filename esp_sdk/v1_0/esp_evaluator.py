import io
import pickle
from abc import ABC, abstractmethod

from keras.models import load_model

from esp_sdk.v1_0.esp_service import EspService
from esp_sdk.v1_0.extension_packaging import ExtensionPackaging


class EspEvaluator(ABC):
    """
    An abstract class to evaluate ESP populations.
    """

    def __init__(self):
        self._extension_packaging = ExtensionPackaging()

    @abstractmethod
    def evaluate_candidate(self, candidate):
        """
        Evaluates a single Keras model.
        :param candidate: a Keras model
        :return: a dictionary of metrics
        """
        pass

    @abstractmethod
    def evaluate_population(self, population_response):
        """
        Evaluates the candidates in an ESP PopulationResponse
        and updates the PopulationResponse with the candidates fitness.
        :param population_response: an ESP PopulationResponse
        :return: nothing. A dictionary containing the metrics is assigned to the candidates metrics as a UTF-8
        encoded string (bytes) within the passed response
        """
        pass

    def _encode_metrics(self, metrics):
        if not isinstance(metrics, dict):
            # For backward compatibility: if the evaluation didn't return a dictionary,
            # convert the returned value to a float and put it in a dictionary.
            # metric might be a numpy object, like numpy.int64 or numpy.float32. Convert it to float.
            score = float(metrics)
            # Create a dictionary
            metrics = {"score": score}
        encoded_metrics = self._extension_packaging.to_extension_bytes(metrics)
        return encoded_metrics

    def _decode_metrics(self, encoded_metrics):
        metrics = self._extension_packaging.from_extension_bytes(encoded_metrics)
        return metrics


class EspKerasNNEvaluator(EspEvaluator, ABC):
    """
    A class that evaluates ESP populations made of Keras models.
    """
    def __init__(self, experiment_params):
        super().__init__()
        self.reevaluate_elites = experiment_params["LEAF"].get("reevaluate_elites", True)

    def evaluate_population(self, response):
        for candidate in response.population:
            # Do not re-evaluate elites unless specified
            previous_metrics = self._decode_metrics(candidate.metrics)
            if self.reevaluate_elites or not previous_metrics:
                # Convert the received bytes to a Keras model
                model_bytes = candidate.interpretation
                model_file = io.BytesIO(model_bytes)
                keras_model = load_model(model_file)
                metrics = self.evaluate_candidate(keras_model)
                candidate.metrics = self._encode_metrics(metrics)


class EspNNWeightsEvaluator(EspEvaluator, ABC):
    """
    A class that evaluates ESP populations made of neural network weights.
    """
    def __init__(self, experiment_params):
        super().__init__()
        self.reevaluate_elites = experiment_params["LEAF"].get("reevaluate_elites", True)
        esp_service = EspService(experiment_params)
        print("Requesting base model...")
        self.base_model = esp_service.request_base_model()
        print("Based model received.")

    def get_keras_model(self, weights):
        """
        Converts neural networks weights into a Keras model
        :param weights: a dictionary of neural network weights
        :return: a Keras model
        """
        for layer_name, layer_weights in weights.items():
            self.base_model.get_layer(layer_name).set_weights(layer_weights)
        return self.base_model

    def evaluate_population(self, response):
        for candidate in response.population:
            # Do not re-evaluate elites unless specified
            previous_metrics = self._decode_metrics(candidate.metrics)
            if self.reevaluate_elites or not previous_metrics:
                weights_bytes = candidate.interpretation
                weights_dict = pickle.loads(weights_bytes)
                keras_model = self.get_keras_model(weights_dict)
                metrics = self.evaluate_candidate(keras_model)
                candidate.metrics = self._encode_metrics(metrics)
