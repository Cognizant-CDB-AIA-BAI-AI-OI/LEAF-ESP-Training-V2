import json
import os
from functools import lru_cache

import pandas as pd
from keras.models import load_model

from xde.s3utils import S3Util
from xde.data_set_encoder import DataSetEncoder
from xde.surrogate_model import XDESurrogateModel

XDE_MODELS_BUCKET = os.environ.get('XDE_MODELS_BUCKET') or 'xde-models'


@lru_cache(maxsize=250)
def load_xde_model(s3_util, version, model_name, model_definition, possible_values):
    print("Loading model {}/{} defined by {} and {}...".format(version, model_name, model_definition, possible_values))
    keras_model_file = download_file(s3_util, version, model_name)
    keras_model = load_model(keras_model_file)
    json_definition = download_file(s3_util, version, model_definition)
    csv_values = download_file(s3_util, version, possible_values)
    return keras_model, keras_model_file, json_definition, csv_values


def download_file(s3_util, version, file_name):
    # Note: keys are case sensitive in S3
    key = 'models/' + version
    print("Downloading file from S3: {}/{}/{}...".format(XDE_MODELS_BUCKET, key, file_name))
    f = s3_util.download_file(XDE_MODELS_BUCKET,
                              'models/' + version,
                              file_name)
    return f


def keras_predict(request_json, version):
    model_name = "predictor.h5"
    model_definition = "predictor.json"
    possible_values = "possible_values.csv"
    return _access_model(request_json, version, model_name, model_definition, possible_values)


def keras_prescribe(request_json, version):
    model_name = "prescriptor.h5"
    model_definition = "prescriptor.json"
    possible_values = "possible_values.csv"
    return _access_model(request_json, version, model_name, model_definition, possible_values)


def _access_model(request_json, version, model_name, model_definition, possible_values):
    s3_util = S3Util()
    keras_model_file = None
    json_definition = None
    csv_values = None
    try:
        keras_model, keras_model_file, json_definition, csv_values = load_xde_model(s3_util,
                                                                                    version,
                                                                                    model_name,
                                                                                    model_definition,
                                                                                    possible_values)
        column_values = DataSetEncoder.get_possible_values_dict(csv_values)

        request_df = pd.DataFrame.from_records(request_json)
        encoded_request_df = DataSetEncoder.encode_df(column_values, request_df, verbose=False)

        # Use the model to predict outputs
        with open(json_definition) as json_data:
            experiment_params = json.load(json_data)

        surrogate_model = XDESurrogateModel()
        data = surrogate_model.create_inputs_df(experiment_params, encoded_request_df)
        predictions = keras_model.predict(data)

        # Decode and format the predictions
        output_cols = surrogate_model.gather_output_cols(experiment_params)
        decoded_csq = DataSetEncoder.decode(predictions, output_cols, column_values)
        # We have 1 array per output column. Convert them into n rows of output_cols size
        nb_rows = len(decoded_csq[0])
        rows = []
        for row in range(nb_rows):
            rows.append({output_cols[col_idx]: decoded_csq[col_idx][row] for col_idx in range(0, len(output_cols))})
    finally:
        # Make sure we always clean up the locally stored model
        s3_util.remove_temp_file(keras_model_file)
        s3_util.remove_temp_file(json_definition)
        s3_util.remove_temp_file(csv_values)

    return rows
