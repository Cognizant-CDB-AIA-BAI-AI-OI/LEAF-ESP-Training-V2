import numpy as np
import pandas as pd

from esp_sdk.v1_0.esp_evaluator import EspNNWeightsEvaluator

from xde.data_set_encoder import DataSetEncoder
from xde.surrogate_model import XDESurrogateModel


class XDEEvaluator(EspNNWeightsEvaluator):
    """
    A class that computes a fitness for Prescriptor candidates.
    """

    def __init__(self, predictor, context_names, actions_names, possible_values_csv, evaluation_samples_csv,
                 experiment_params):
        super().__init__(experiment_params)
        self.predictor = predictor
        possible_values_dict = DataSetEncoder.get_possible_values_dict(possible_values_csv)
        with open(evaluation_samples_csv) as df_file:
            evaluation_samples_df = pd.read_csv(df_file, keep_default_na=False)
        encoded_evaluation_samples_df = DataSetEncoder.encode_df(possible_values_dict, evaluation_samples_df, False)
        # Get the app columns from the experiment params (network) and the evaluation samples
        self.contexts, _ = XDESurrogateModel.create_inputs_outputs_df(experiment_params, encoded_evaluation_samples_df)
        self.context_names = context_names
        self.actions_names = actions_names

    def evaluate_candidate(self, candidate):
        """
        Evaluates a Keras model candidate and returns it fitness.
        :param candidate: a Keras neural network model to evaluate
        :return: a fitness score
        """
        # Generate actions for the evaluation contexts
        actions = candidate.predict(self.contexts)

        # Append the contexts and actions in the order expected by the Predictor
        contexts_and_actions = self.aggregate_predictor_inputs(self.contexts, actions)

        # Use the Predictor to predict the Cost, Schedule and Quality
        outcomes = self.predictor.predict(contexts_and_actions)

        # Compute a fitness from these outcomes
        fitness = XDEEvaluator.compute_fitness(outcomes)
        return fitness

    def aggregate_predictor_inputs(self, contexts, actions):
        """
        Aggregates the contexts and actions in the order expected by the Predictor.
        :param contexts: the encoded contexts
        :param actions: the encoded actions
        :return: An aggregation of contexts and actions that can be passed to the Predictor
        """
        context_dict = {z[0]: z[1] for z in zip(self.context_names, contexts)}
        actions_dict = {z[0]: z[1] for z in zip(self.actions_names, actions)}

        # Go in the order of the Predictor inputs and get the data either from contexts or actions
        predictor_inputs = []
        for predictor_input_name in self.predictor.input_names:
            # Ignore the suffix "_input" that comes back from Keras input names
            col = predictor_input_name[:-6]
            if col in self.context_names:
                predictor_inputs.append(context_dict[col])
            elif col in self.actions_names:
                predictor_inputs.append(actions_dict[col])
            else:
                raise ValueError("Unknown Predictor input {}".format(col))
        return predictor_inputs

    @staticmethod
    def compute_fitness(predictions):
        """
        Predictions contains 3 columns: Cost, Schedule and Quality
        We want to maximize them.
        """
        cost, schedule, quality = predictions
        c = sum([np.argmax(x) for x in cost])
        s = sum([np.argmax(x) for x in schedule])
        q = sum([np.argmax(x) for x in quality])
        f = c + s + q
        return f
