import click
import json

import numpy as np
import pandas as pd
from keras.models import Model

from esp_sdk.v1_0.esp_service import EspService


class XDESurrogateModel(object):

    @staticmethod
    def train_from_data(esp_service, experiment_params, training_data_file, batch_size=1, epochs=1000, verbose=1):
        """
        Creates a model using the experiment params, trains it using the data and labels,
        and persists it as an h5 file with the output_name.
        :param esp_service: the ESP service that can be used to create a NN model
        :param experiment_params: the experiment parameters
        :param training_data_file: the .csv file containing the training data and labels
        :param batch_size: number of samples per gradient update
        :param epochs: the number of times to iterate over the training data arrays
        :param verbose: 0, 1, or 2. Keras verbosity mode.
                        0 = silent, 1 = verbose, 2 = one log line per epoch.
        :return: a trained Keras model
        """
        network_params = experiment_params['network']
        output_layers = network_params['outputs']
        output_names = [output_layer['name'] for output_layer in output_layers]

        # Create a model
        model_to_save = esp_service.request_base_model()
        # Compile to avoid warning when loading it
        model_to_save.compile(optimizer='adam',
                              loss='binary_crossentropy',
                              metrics=['accuracy'])
        # Create a new model with the SAME layers up to the argmax lambda layers. Argmax breaks backpropagation.
        outputs = [model_to_save.get_layer(n).output for n in output_names]
        model_to_train = Model(inputs=model_to_save.inputs,
                               outputs=outputs)

        #  Compile it
        model_to_train.compile(optimizer='adam',
                               loss='binary_crossentropy',
                               metrics=['accuracy'])

        # Load the data
        data, labels = XDESurrogateModel.create_inputs_outputs(experiment_params, training_data_file)

        # Train!
        model_to_train.fit(data, labels, epochs=epochs, batch_size=batch_size, verbose=verbose)

        # We've trained the modified model. But we're interested in the
        # original model that has the additional lambda layers.
        # model_to_save.save("{}.h5".format(output_name))
        return model_to_save

    @staticmethod
    def create_inputs_outputs(experiment_params, training_data_csv):
        with open(training_data_csv) as df_file:
            training_df = pd.read_csv(df_file, keep_default_na=False)
            return XDESurrogateModel.create_inputs_outputs_df(experiment_params, training_df)

    @staticmethod
    def create_inputs_outputs_df(experiment_params, training_df):
        input_cols = XDESurrogateModel.gather_input_cols(experiment_params)
        output_cols = XDESurrogateModel.gather_output_cols(experiment_params)

        inputs = XDESurrogateModel._create_df(training_df, input_cols)
        outputs = XDESurrogateModel._create_df(training_df, output_cols)

        return inputs, outputs

    @staticmethod
    def create_inputs_df(experiment_params, training_df):
        input_cols = XDESurrogateModel.gather_input_cols(experiment_params)
        inputs = XDESurrogateModel._create_df(training_df, input_cols)

        return inputs

    @staticmethod
    def _create_df(training_df, columns):
        # Convert the strings to numpy arrays of floats, if needed
        for col in columns:
            if isinstance(training_df[col].iloc[0], str):
                training_df[col] = training_df[col].apply(
                    lambda x: np.array(x.replace("[", "").replace("]", "").split(", ")).astype(float))

        # inputs is 20 rows by 2 cols. Keras wants a list of 1D arrays (2 by 20) so transpose the np array
        inputs = np.array(training_df[columns]).transpose()
        inputs = [np.array(np.array(col).tolist()) for col in inputs]
        return inputs

    @staticmethod
    def gather_input_cols(experiment_params):
        return XDESurrogateModel._gather_cols(experiment_params['network']['inputs'])

    @staticmethod
    def gather_output_cols(experiment_params):
        return XDESurrogateModel._gather_cols(experiment_params['network']['outputs'])

    @staticmethod
    def _gather_cols(layers):
        cols = [layer['name'] for layer in layers]
        return cols


@click.group()
def cli():
    pass


@cli.command()
@click.option('--experiment_params_file')
@click.option('--encoded_training_data_csv')
@click.option('--output_file')
@click.option('--batch_size', default=1)
@click.option('--epochs', default=5000)
@click.option('--verbose', default=0)
def train(experiment_params_file, encoded_training_data_csv, output_file, batch_size, epochs, verbose):
    """
    Creates a model using the experiment params, trains it using the data and labels,
    and persists it as an h5 file with the output_name.
    :param experiment_params_file: the .json file containing the experiment and network description
    :param encoded_training_data_csv: the .csv file containing the encoded training data and labels
    :param output_file: the name of the file to which to save the trained keras network. Should be an .h5 file.
    :param batch_size: number of samples per gradient update
    :param epochs: the number of times to iterate over the training data arrays
    :param verbose: 0, 1, or 2. Keras verbosity mode.
                    0 = silent, 1 = verbose, 2 = one log line per epoch.
    :return: a trained Keras model
    """
    # Get the experiment params
    with open(experiment_params_file) as json_data:
        experiment_params = json.load(json_data)

    esp_service = EspService(experiment_params)
    keras_model = XDESurrogateModel.train_from_data(esp_service,
                                                    experiment_params,
                                                    encoded_training_data_csv,
                                                    batch_size=batch_size,
                                                    epochs=epochs,
                                                    verbose=verbose)
    keras_model.save(output_file)


if __name__ == '__main__':
    cli()
