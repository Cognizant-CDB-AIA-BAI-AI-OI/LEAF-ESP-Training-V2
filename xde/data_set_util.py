import os
import numpy as np
import pandas as pd

from xde.data_set_encoder import DataSetEncoder


class DataSetUtil(object):

    @staticmethod
    def split_train_val(full_data_csv, train_pct):
        """
        Splits the passed csv file into a 'train' Pandas DataFrame and a 'val' one.
        :param full_data_csv: the full data set as a csv file name
        :param train_pct: which percentage of the total data to use for training. The rest is used for validation.
        :return: a Pandas DataFrame containing the training set, and another one containing the validation set
        """
        with open(full_data_csv) as df_file:
            df = pd.read_csv(df_file, keep_default_na=False)

            nb_total_samples = len(df)
            nb_train_samples = int(round(nb_total_samples * train_pct))
            nb_val_samples = nb_total_samples - nb_train_samples
            msk = np.full(nb_total_samples, True)
            msk[:nb_val_samples] = False
            np.random.shuffle(msk)

            train_df = df[msk]
            val_df = df[~msk]

            return train_df, val_df

    @staticmethod
    def persist_data_sets(data_sets, split_path, verbose=True):
        """
        Persists the passed data sets (Pandas DataFrame) to the passed directory.
        :param data_sets: A list of Pandas DataFrames pairs: the 'train' DataFrame and the 'val' DataFrame
        :param split_path: the path to which the files must be written
        :param verbose: True to print debug statements
        :return: nothing
        """
        for i, ds in enumerate(data_sets):
            if not os.path.exists(split_path):
                os.makedirs(split_path)
            train_file_name = "train_set_{}.csv".format(i)
            val_file_name = "val_set_{}.csv".format(i)
            full_train_name = os.path.join(split_path, train_file_name)
            full_val_name = os.path.join(split_path, val_file_name)
            if verbose:
                print("Persisting {}...".format(full_train_name))
                print("Persisting {}...".format(full_val_name))
            ds[0].to_csv(full_train_name, index=False)
            ds[1].to_csv(full_val_name, index=False)
        if verbose:
            print("Done.")

    @staticmethod
    def load_data_sets(nb_runs, split_path, verbose=True):
        """
        Loads data sets from the passed directory
        :param nb_runs: the number of data sets to load
        :param split_path: the path from which the files can be read
        :param verbose: True to print debug statements
        :return: a list of nb_runs data set pairs containing the train set and its associated val set
        """
        df_sets = []
        for i in range(nb_runs):
            train_file_name = "train_set_{}.csv".format(i)
            val_file_name = "val_set_{}.csv".format(i)
            full_train_name = os.path.join(split_path, train_file_name)
            full_val_name = os.path.join(split_path, val_file_name)
            if verbose:
                print("Loading {}...".format(full_val_name))
                print("Loading {}...".format(full_train_name))
            train_df = pd.read_csv(full_train_name, dtype=str)
            val_df = pd.read_csv(full_val_name, dtype=str)
            df_sets.append([train_df, val_df])
        if verbose:
            print("Done.")
        return df_sets

    @staticmethod
    def encode_data_sets(data_sets, possible_values_csv, verbose=True):
        """
        Encodes the passed data sets according to the passed possible values.
        :param data_sets: a list of pairs of Pandas DataFrame (train and val)
        :param possible_values_csv: the name of the csv file containing the possible values for each column
        :param verbose: True to print debug statements
        :return: a list of encoded Pandas DataFrame pairs (train, val)
        """
        encoded_data_sets = []
        possible_values_dict = DataSetEncoder.get_possible_values_dict(possible_values_csv)
        for i, data_set in enumerate(data_sets):
            if verbose:
                print("Encoding data set #{}...".format(i))
            # Turn verbosity only for the first data set to check the columns
            is_detail_verbose = i == 0
            if verbose:
                print("  Encoding Train...")
            train_data_set_encoded = DataSetEncoder.encode_df(possible_values_dict, data_set[0], is_detail_verbose)
            if verbose:
                print("  Encoding Val...")
            val_data_set_encoded = DataSetEncoder.encode_df(possible_values_dict, data_set[1], False)
            encoded_data_sets.append([train_data_set_encoded, val_data_set_encoded])
        return encoded_data_sets
