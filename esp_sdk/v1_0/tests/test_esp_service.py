from unittest import TestCase
from unittest.mock import patch, Mock

import grpc
from tenacity import RetryError

from esp_sdk.v1_0.esp_service import EspService, NB_RETRIES

EXPERIMENT_PARAMS = {
    'LEAF': {
        'experiment_id': 'test_experiment_id',
        'version': '1.0',
        'esp_host': 'localhost',
        'esp_port': '50051'
    }
}

EXPERIMENT_PARAMS_WITH_GRPC_OPTIONS = {
    'LEAF': {
        'experiment_id': 'test_experiment_id',
        'version': '1.0',
        'esp_host': 'localhost',
        'esp_port': '50051',
        'grpc_options': {
            'grpc.max_send_message_length': 111,
            'grpc.max_receive_message_length': 222
        }
    }
}


class TestEspService(TestCase):

    @patch('esp_sdk.v1_0.esp_service.PopulationServiceStub', autospec=True)
    def test_retry(self, grpc_mock):
        """
        Make sure that the service makes the right number of attempts at the gRPC call, then gives up, and raises
        an exception
        :param grpc_mock: Injected mock
        """
        service = EspService(EXPERIMENT_PARAMS)
        next_population_mock = Mock(side_effect=grpc.RpcError('expected'))
        grpc_mock.return_value.NextPopulation = next_population_mock

        self.assertRaises(RetryError, service.get_next_population, None)

        expected_times_called = NB_RETRIES

        self.assertEqual(expected_times_called, next_population_mock.call_count)

    @patch('esp_sdk.v1_0.esp_service.PopulationServiceStub', autospec=True)
    def test_success(self, grpc_mock):
        """
        Make sure that the service proceeds when the gRPC call succeeds.
        :param grpc_mock: Injected mock
        """
        service = EspService(EXPERIMENT_PARAMS)

        # Just use some placeholder text
        response_text = 'test_response'
        next_population_mock = Mock()
        next_population_mock.return_value = response_text
        grpc_mock.return_value.NextPopulation = next_population_mock

        response = service.get_next_population(None)

        self.assertEqual(response_text, response)
        self.assertEqual(1, next_population_mock.call_count)

    def test_grpc_options(self):
        """
        Make sure gRPC options can be passed to the gRPC service.
        :return: nothing
        """
        # No options
        service = EspService(EXPERIMENT_PARAMS)
        self.assertIsNotNone(service)
        # Can't really test the options themselves.

        # With gRPC options
        service = EspService(EXPERIMENT_PARAMS_WITH_GRPC_OPTIONS)
        self.assertIsNotNone(service)
        # Can't really test the options themselves.
