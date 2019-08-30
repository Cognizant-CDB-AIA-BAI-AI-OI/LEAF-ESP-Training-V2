import copy
import io

import grpc
from google.protobuf.json_format import ParseDict
from keras.models import load_model
from tenacity import retry, stop_after_attempt, wait_random, retry_if_exception_type

from esp_sdk.v1_0.extension_packaging import ExtensionPackaging
from esp_sdk.v1_0.generated.population_service_pb2_grpc import PopulationServiceStub
from esp_sdk.v1_0.generated.population_structs_pb2 import ExistingPopulationRequest
from esp_sdk.v1_0.generated.population_structs_pb2 import PopulationRequest

# retry gRPC calls this many times
NB_RETRIES = 3
# Increase max message size so that they can contain full Keras models.
# Otherwise the default max message size is around 4 MB.
# Note that message sizes have to be configured on the server side too.
MAX_MESSAGE_LENGTH_BYTES = 50 * 1024 * 1024  # 50 MB
DEFAULT_GRPC_OPTIONS = {'grpc.max_send_message_length': MAX_MESSAGE_LENGTH_BYTES,
                        'grpc.max_receive_message_length': MAX_MESSAGE_LENGTH_BYTES}


class EspService(object):

    def __init__(self, experiment_params):
        """
        A class that can interact with the ESP service
        :param experiment_params: the experiment parameters dictionary
        """
        self.experiment_params = experiment_params
        self.extension_packaging = ExtensionPackaging()
        self.experiment_params_as_bytes = self.extension_packaging.to_extension_bytes(experiment_params)
        self.experiment_id = experiment_params["LEAF"]["experiment_id"]
        self.version = experiment_params["LEAF"]["version"]

        # gRPC connection
        esp_host = experiment_params["LEAF"]["esp_host"]
        esp_port = experiment_params["LEAF"]["esp_port"]
        grpc_options = experiment_params["LEAF"].get("grpc_options", DEFAULT_GRPC_OPTIONS).items()
        print("ESP service: {}:{}".format(esp_host, esp_port))
        print("gRPC options:")
        for pair in grpc_options:
            print("  {}: {}".format(pair[0], pair[1]))
        channel = grpc.insecure_channel('{}:{}'.format(esp_host, esp_port), options=grpc_options)
        print("Ready to connect.")
        self.esp_service = PopulationServiceStub(channel)

    def get_next_population(self, prev_response):
        """
        Returns a new generation for a given experiment.
        :param prev_response: the previous generation, *with* evaluation metrics for each Candidate
        :return: a new generation, as a PopulationResponse object
        """
        # Prepare a request for next generation
        request_params = {'version': self.version, 'experiment_id': self.experiment_id}
        request = ParseDict(request_params, PopulationRequest())
        request.config = self.experiment_params_as_bytes
        if prev_response:
            request.evaluated_population_response.CopyFrom(prev_response)

        # Ask for next generation
        response = self._next_population_with_retry(request)
        return response

    def get_previous_population(self, experiment_id, checkpoint_id):
        """
        Returns the population corresponding to the passed experiment_id and checkpoint_id
        :param experiment_id: the experiment id
        :param checkpoint_id: the checkpoint id returned by a previous call to get_next_population
        :return: a previous generation, as a PopulationResponse object
        """
        # Prepare a GetPopulation request
        request_params = {'version': self.version, 'experiment_id': experiment_id, 'checkpoint_id': checkpoint_id}
        request = ParseDict(request_params, ExistingPopulationRequest())
        # Ask for a previous generation
        response = self.esp_service.GetPopulation(request)
        return response

    @retry(stop=stop_after_attempt(NB_RETRIES), wait=wait_random(1, 3), retry=retry_if_exception_type(grpc.RpcError))
    def _next_population_with_retry(self, request):
        print("Sending NextPopulation request")
        response = self.esp_service.NextPopulation(request)
        print("NextPopulation response received.")
        return response

    @retry(stop=stop_after_attempt(NB_RETRIES), wait=wait_random(1, 3), retry=retry_if_exception_type(grpc.RpcError))
    def _get_population_with_retry(self, request):
        print("Sending GetPopulation request")
        response = self.esp_service.GetPopulation(request)
        print("GetPopulation response received.")
        return response

    def request_base_model(self):
        # Update the request to query for 1 Keras model
        params = copy.deepcopy(self.experiment_params)
        params["evolution"] = {"population_size": 1}
        params["LEAF"]["representation"] = "KerasNN"

        # Prepare a request for next generation
        request_params = {'version': self.version, 'experiment_id': self.experiment_id}
        request = ParseDict(request_params, PopulationRequest())
        request.config = self.extension_packaging.to_extension_bytes(params)

        # Ask for the base model
        response = self._next_population_with_retry(request)

        # Convert the received bytes to a Keras model
        model_bytes = response.population[0].interpretation
        model_file = io.BytesIO(model_bytes)
        keras_model = load_model(model_file)

        # return the base model
        return keras_model

    def extract_candidates_info(self, response):
        """
        Prints Candidate details from a population
        :param response: a PopulationResponse from the ESP service
        :return: a dictionary representing a candidate
        """
        candidates_info = []
        for candidate in response.population:
            c = {"id": candidate.id,
                 "identity": candidate.identity.decode('UTF-8'),
                 "metrics": self.extension_packaging.from_extension_bytes(candidate.metrics),
                 "model": candidate.interpretation}
            candidates_info.append(c)
        return candidates_info

    @staticmethod
    def print_population_response(response):
        """
        Prints out the details of a population represented by a PopulationResponse object
        :param response: a PopulationResponse object returned by the ESP API
        """
        print("PopulationResponse:")
        print("  Generation: {}".format(response.generation_count))
        print("  Population size: {}".format(len(response.population)))
        print("  Checkpoint id: {}".format(response.checkpoint_id))
        print("  Evaluation stats: {}".format(response.evaluation_stats.decode('UTF-8')))

    def print_candidates(self, response, sort_candidates=True):
        """
        Prints the candidates details
        :param response: an evaluated PopulationResponse
        :param sort_candidates: if True, sort the candidates by score, lowest first, to always see the best
        candidates at the bottom of the logs
        """
        # Interpret the response received from the ESP service
        candidates_info = self.extract_candidates_info(response)
        if sort_candidates:
            # Sort the candidates by score, lowest first to always see the best candidates at the bottom of the log
            candidates_info = sorted(candidates_info, key=lambda k: k["metrics"]["score"], reverse=False)

        print("Evaluated candidates:")
        for candidate in candidates_info:
            print("Id: {} Identity: {} Metrics: {}".format(candidate["id"],
                                                           candidate["identity"],
                                                           candidate["metrics"]))
        print("")
