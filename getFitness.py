import numpy as np
import pandas as pd

from esp_sdk.v1_0.esp_evaluator import EspNNWeightsEvaluator

from xde.data_set_encoder import DataSetEncoder
from xde.surrogate_model import XDESurrogateModel


class Fitness_Evaluator(EspNNWeightsEvaluator):
    """
    A class that computes a fitness for Prescriptor candidates.
    """

    def __init__(self,context_names, actions_names, possible_values_csv, evaluation_samples_df,
                 experiment_params,training_data):
        super().__init__(experiment_params)
        
        self.training_data=training_data
        self.dfps=pd.read_csv(possible_values_csv)
        possible_values_dict = DataSetEncoder.get_possible_values_dict(possible_values_csv)
        
        self.encoded_evaluation_samples_df = DataSetEncoder.encode_df(possible_values_dict, evaluation_samples_df, False)
        
        # Get the app columns from the experiment params (network) and the evaluation samples
        self.contexts, _ = XDESurrogateModel.create_inputs_outputs_df(experiment_params, self.encoded_evaluation_samples_df)
        self.context_names = context_names
        self.actions_names = actions_names
        self.input_names=self.context_names+self.actions_names


    def evaluate_candidate(self, candidate):
        """
        Evaluates a Keras model candidate and returns it fitness.
        :param candidate: a Keras neural network model to evaluate
        :return: a fitness score
        """
        # Generate actions for the evaluation contexts

        actions = candidate.predict(self.contexts)
        new_dec = [] 
        
        for i in actions:
          new_dec.append(i.astype(int))

        # Append the contexts and actions in the order expected by the Predictor
        contexts_and_actions = self.aggregate_predictor_inputs(self.contexts, new_dec)
        
        new_df = pd.DataFrame()
        i = 0
        for col_names in self.input_names:
          new_df[col_names] = contexts_and_actions[i].tolist()
          i=i+1
          
        self.encoded_evaluation_samples_df = self.encoded_evaluation_samples_df.applymap(str)
        new_df = new_df.applymap(str)
        merge_df=pd.merge(new_df,self.encoded_evaluation_samples_df , how='left')

        merge_df["Scores"] = merge_df["Scores"].apply(pd.to_numeric)
        fitness = round(merge_df['Scores'].sum())
        print("************Fitness Score:**********",fitness)
        return fitness

    def aggregate_predictor_inputs(self, contexts, actions):
        """
        Aggregates the contexts and actions in the order expected by predictor.
        :param contexts: the encoded contexts
        :param actions: the encoded actions
        :return: An aggregation of contexts and actions that can be passed to the predictor
        """
        context_dict = {z[0]: z[1] for z in zip(self.context_names, contexts)}
        actions_dict = {z[0]: z[1] for z in zip(self.actions_names, actions)}

        # Go in the order of the Predictor inputs and get the data either from contexts or actions
        predictor_inputs = []

        for predictor_input_name in self.input_names:
            col = predictor_input_name
            if col in self.context_names:
                predictor_inputs.append(context_dict[col])
            elif col in self.actions_names:
                predictor_inputs.append(actions_dict[col])
            else:
                raise ValueError("Unknown Predictor input {}".format(col))
        return predictor_inputs
