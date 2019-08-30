import click
import csv
import json
import pandas as pd
from collections import OrderedDict

from data_util.encoder import EnumStrictEncoder


class DataSetEncoder(object):

    @staticmethod
    def encode(possible_values_dict, data_set_csv, verbose=False):
        with open(data_set_csv) as df_file:
            data_set_df = pd.read_csv(df_file, keep_default_na=False)
            encoded_df = DataSetEncoder.encode_df(possible_values_dict, data_set_df, verbose)
        if verbose:
            print("Done.")
        return encoded_df

    @staticmethod
    def encode_df(column_values_dict, data_set_df, verbose):
        encoder = EnumStrictEncoder()
        data_set_df = data_set_df.copy()
        for column in data_set_df.columns:
            # Encode only columns for which values have been provided
            if column_values_dict.get(column, None):
                if verbose:
                    print("Encoding column '{}'".format(column))
                data_set_df[column] = data_set_df[column].apply(
                    lambda row: encoder.encode(row, column_values_dict[column], verbose=True))
            else:
                if verbose:
                    print("Skipping column '{}'".format(column))
        return data_set_df

    @staticmethod
    def get_possible_values_dict(possible_values_csv):
        """
        Loads the passed CSV file into an ordered dictionary of column to possible values.
        :param possible_values_csv: a CSV file containing a list of columns and for each column its list of possible
        values.
        :return: an ordered dictionary of column to possible values.
        """
        with open(possible_values_csv) as df_file:
            possible_values_df = pd.read_csv(df_file, keep_default_na=False)
            column_values = OrderedDict()
            # Read the possible values for each column
            for column in possible_values_df.columns:
                # Keep the original order.
                values = list(possible_values_df[column].unique())
                # Remove leading and trailing spaces
                values = [x.strip() for x in values]
                # Remove potential trailing blank entry
                if "" in values:
                    values.remove("")
                column_values[column] = values
        return column_values

    @staticmethod
    def decode(encoded_values, columns, column_values):
        encoder = EnumStrictEncoder()
        decoded = []
        for encoded_rows, column in zip(encoded_values, columns):
            decoded_row = []
            for encoded_row in encoded_rows:
                decoded_row.append(encoder.decode(encoded_row.tolist(), column_values[column]))
            decoded.append(decoded_row)
        return decoded

    @staticmethod
    def generate_model_description(pvd, inputs_csv, outputs_csv, nb_hidden_units=3, use_bias=True,
                                   activation_function="tanh", include_evo_description=False):
        with open(inputs_csv, 'r') as inputs_csv_file:
            # Read the CSV and only look at the first row
            reader = csv.reader(inputs_csv_file, delimiter=',')
            input_columns = next(reader)
        with open(outputs_csv, 'r') as outputs_csv_file:
            reader = csv.reader(outputs_csv_file, delimiter=',')
            output_columns = next(reader)
        model_inputs_list = []
        model_outputs_list = []
        for col_name in input_columns:
            if pvd.get(col_name, None):
                model_inputs_list.append({"name": col_name, "size": len(pvd[col_name]), "values": pvd[col_name]})
            else:
                print("Error: could not find input column {} in possible values".format(col_name))
        for col_name in output_columns:
            if pvd.get(col_name, None):
                model_outputs_list.append({"name": col_name, "size": len(pvd[col_name]), "values": pvd[col_name]})
            else:
                print("Error: could not find output column {} in possible values".format(col_name))

        model = {"inputs": model_inputs_list,
                 "nb_hidden_units": nb_hidden_units,
                 "use_bias": use_bias,
                 "activation_function": activation_function,
                 "outputs": model_outputs_list}
        experiment_params = {"network": model}
        if include_evo_description:
            # Some default evolution params
            evolution = {"population_size": 25,
                         "parent_selection": "proportion",
                         "remove_population_pct": 0.8,
                         "mutation_type": "gaussian_noise_percentage",
                         "mutation_probability": 0.1,
                         "mutation_factor": 0.1,
                         "nb_elites": 5}
            experiment_params["evolution"] = evolution

        return experiment_params


@click.group()
def cli():
    pass


@cli.command()
@click.option('--input_data_set_csv', default='tests/fixtures/xde_data_set.csv')
@click.option('--possible_values_csv', default='tests/fixtures/xde_possible_values.csv')
@click.option('--output_data_set_csv', default='tests/fixtures/xde_encoded_data_set.csv')
@click.option('--verbose', default=True)
def encode_data_set(input_data_set_csv, possible_values_csv, output_data_set_csv, verbose):
    """
    Encodes a data set into one hot vectors
    :param input_data_set_csv: the csv file that contains the data set to encode
    :param possible_values_csv: the csv file that contains the list of possible values for each column
    :param output_data_set_csv: the name of the csv file to save the encoded output to
    :param verbose: True to log encoding progress
    :return: nothing
    """
    possible_values_dict = DataSetEncoder.get_possible_values_dict(possible_values_csv)
    df = DataSetEncoder.encode(possible_values_dict, input_data_set_csv, verbose=verbose)
    df.to_csv(output_data_set_csv)


@cli.command()
@click.option('--possible_values_csv')
@click.option('--inputs_csv')
@click.option('--outputs_csv')
@click.option('--output_model_json')
@click.option('--nb_hidden_units', default=3)
@click.option('--use_bias', default=True)
@click.option('--activation', default="tanh")
@click.option('--include_evo', default=False)
def generate_model_description(possible_values_csv, inputs_csv, outputs_csv, output_model_json, nb_hidden_units,
                               use_bias, activation, include_evo):
    """
    Generates a neural network description for the csv file columns
    :param possible_values_csv: a csv file of each column with its possible values
    :param inputs_csv: a csv file containing a list of column names corresponding to the network inputs, separated by
    commas
    :param outputs_csv: a csv file containing a list of column names corresponding to the network outputs (labels),
    separated by commas
    :param output_model_json: the name of the json file to save the model description to
    :param nb_hidden_units: the number of hidden units
    :param use_bias: True to use bias
    :param activation: the type of activation function to use after the hidden layer
    :param include_evo: True to include some default parameters for evolution
    :return: a string that can be used in experiment params to create a neural network
    """
    possible_values_dict = DataSetEncoder.get_possible_values_dict(possible_values_csv)
    model_description = DataSetEncoder.generate_model_description(possible_values_dict,
                                                                  inputs_csv,
                                                                  outputs_csv,
                                                                  nb_hidden_units,
                                                                  use_bias,
                                                                  activation,
                                                                  include_evo)
    model_json = json.dumps(model_description, indent=4, separators=(',', ': '))
    with open(output_model_json, "w") as f:
        f.write(model_json)


if __name__ == '__main__':
    cli()
