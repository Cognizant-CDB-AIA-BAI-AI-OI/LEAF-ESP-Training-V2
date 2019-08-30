import numpy as np
import pandas as pd

from esp_sdk.v1_0.esp_evaluator import EspNNWeightsEvaluator

from data_util.data_set_encoder import DataSetEncoder
from data_util.surrogate_model import SurrogateModel


class Evaluator(EspNNWeightsEvaluator):
    """
    A class that computes a fitness for Prescriptor candidates.
    """

    def __init__(self, predictor, context_names, decisions_names, possible_values_csv, evaluation_samples_csv,
                 experiment_params):
        super().__init__(experiment_params)
        self.predictor = predictor
        possible_values_dict = DataSetEncoder.get_possible_values_dict(possible_values_csv)
        #print ("@!@!@!...possible_values_dict   ", possible_values_dict)
        with open(evaluation_samples_csv) as df_file:
            evaluation_samples_df = pd.read_csv(df_file, keep_default_na=False)
        encoded_evaluation_samples_df = DataSetEncoder.encode_df(possible_values_dict, evaluation_samples_df, False)
        #print ("encoded_evaluation_samples_df ..... ",encoded_evaluation_samples_df)
        # Get the app columns from the experiment params (network) and the evaluation samples
        self.contexts, _ = SurrogateModel.create_inputs_outputs_df(experiment_params, encoded_evaluation_samples_df)
        #print ("s _ ..... ",  _)
        self.context_names = context_names
        self.decisions_names = decisions_names
        #print ("self.context_names ...",self.context_names)
        #print ("self.decisions_names ...",self.decisions_names)

    def evaluate_candidate(self, candidate):
        """
        Evaluates a Keras model candidate and returns it fitness.
        :param candidate: a Keras neural network model to evaluate
        :return: a fitness score
        """
        # Generate decisions for the evaluation contexts
        #print ("candidate... ", candidate)
        decisions = candidate.predict(self.contexts)
        #print ("!@!@!@~~@! decisions... ", decisions)
        #print ("!!!@@@!!@!@@@!@ contexts... ", self.contexts)

        # Append the contexts and decisions in the order expected by the Predictor
        contexts_and_decisions = self.aggregate_predictor_inputs(self.contexts, decisions)

        # Use the Predictor to predict the cost, speed and quality
        #print ("contexts_and_decisions ... ", contexts_and_decisions)
        metrics = self.predictor.predict(contexts_and_decisions)

        # Compute a fitness from these metrics
        fitness = Evaluator.compute_fitness(metrics)
        return fitness

    def aggregate_predictor_inputs(self, contexts, decisions):
        """
        Aggregates the contexts and decisions in the order expected by predictor.
        :param contexts: the encoded contexts
        :param decisions: the encoded decisions
        :return: An aggregation of contexts and decisions that can be passed to the predictor
        """
        context_dict = {z[0]: z[1] for z in zip(self.context_names, contexts)}
        decisions_dict = {z[0]: z[1] for z in zip(self.decisions_names, decisions)}

        # Go in the order of the Predictor inputs and get the data either from contexts or decisions
        predictor_inputs = []
        for predictor_input_name in self.predictor.input_names:
            # Ignore the suffix "_input" that comes back from Keras input names
            col = predictor_input_name[:-6]
            if col in self.context_names:
                predictor_inputs.append(context_dict[col])
            elif col in self.decisions_names:
                predictor_inputs.append(decisions_dict[col])
            else:
                raise ValueError("Unknown Predictor input {}".format(col))
        return predictor_inputs

    @staticmethod
    def compute_fitness(predictions):
        """
        predictions contains 3 columns: Cost, Schedule and Quality
        We want to maximize them.
        """
        #cost, schedule, quality = predictions
        #print ("#@!!@!@@! predictions", predictions)
        conversion = predictions
        f = sum([np.argmin(x) for x in conversion])
        print ("~~~ Fitness .. ", f)

        return f
