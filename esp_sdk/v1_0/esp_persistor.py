import csv
import json
import os
import pickle
import statistics
import time

from esp_sdk.v1_0.esp_plotter import EspPlotter
from esp_sdk.v1_0.extension_packaging import ExtensionPackaging

DEFAULT_FITNESS = [{"metric_name": "score", "maximize": True}]


class EspPersistor:
    """
    A class to persist any kind of information from an experiment.
    """

    def __init__(self, experiment_params, evaluator):
        self.experiment_params = experiment_params
        self.extension_packaging = ExtensionPackaging()
        self.save_to_dir = self._generate_persistence_directory()
        # Possible values are all, elites, best, none
        self.candidates_to_persist = self.experiment_params["LEAF"].get("candidates_to_persist", "best").lower()
        if self.candidates_to_persist not in ["all", "elites", "best", "none"]:
            raise ValueError("Unknown value for experiment param [LEAF][candidates_to_persist]: {}".format(
                self.candidates_to_persist))
        self.persist_experiment_params(experiment_params)
        self.evaluator = evaluator

    def _generate_persistence_directory(self):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        persistence_dir = self.experiment_params["LEAF"]["persistence_dir"]
        experiment_id = self.experiment_params["LEAF"]["experiment_id"]
        dirname = os.path.join(persistence_dir, experiment_id)
        version = self.experiment_params["LEAF"]["version"]
        version = version + "_" + timestamp
        dirname = os.path.join(dirname, version)
        os.makedirs(dirname, exist_ok=True)
        return dirname

    def get_persistence_directory(self):
        """
        Returns the name of the directory used for persistence.
        :return: a string
        """
        return self.save_to_dir

    def persist_experiment_params(self, experiment_params):
        """
        Persists the passed experiment parameters.
        :param experiment_params: the experiment parameters to persist
        :return: nothing. Saves a file called `experiment_params.json` to the persistence directory
        """
        filename = os.path.join(self.save_to_dir, 'experiment_params.json')
        with open(filename, 'w') as f:
            f.write(json.dumps(experiment_params, indent=4))

    def persist_response(self, response):
        """
        Persists a generation's information.
        :param response: an evaluated ESP PopulationResponse
        :return: nothing. Saves files to the persistence directory
        """
        gen = response.generation_count
        checkpoint_id = response.checkpoint_id
        candidates_info = self.persist_generation(response)
        self.persist_stats(candidates_info, gen, checkpoint_id)
        self.persist_candidates(candidates_info, gen)
        stats_file = os.path.join(self.save_to_dir, 'experiment_stats.csv')
        title = self.experiment_params["LEAF"]["experiment_id"]
        EspPlotter.plot_stats(stats_file, title)

    def persist_stats(self, candidates_info, generation, checkpoint_id):
        """
        Collects statistics for the passed generation of candidates.
        :param candidates_info: the candidates information
        :param generation: the generation these candidates belong to
        :param checkpoint_id: the checkpoint id corresponding to this generation
        :return: nothing. Saves a file called `experiment_stats.csv` to the persistence directory
        """
        filename = os.path.join(self.save_to_dir, 'experiment_stats.csv')
        file_exists = os.path.exists(filename)

        metrics_stats = {}
        for metric_name in candidates_info[0]["metrics"].keys():
            metric_values = [candidate["metrics"][metric_name] for candidate in candidates_info]
            metrics_stats["max_" + metric_name] = max(metric_values)
            metrics_stats["min_" + metric_name] = min(metric_values)
            metrics_stats["mean_" + metric_name] = statistics.mean(metric_values)
            candidates_info.sort(key=lambda k: k["metrics"][metric_name], reverse=False)
            metrics_stats["cid_min_" + metric_name] = candidates_info[0]["id"]
            metrics_stats["cid_max_" + metric_name] = candidates_info[-1]["id"]

        # 'a+' Opens the file for appending; any data written to the file is automatically added to the end.
        # The file is created if it does not exist.
        with open(filename, 'a+') as stats_file:
            writer = csv.writer(stats_file)
            if not file_exists:
                headers = ["generation", "checkpoint_id"]
                headers.extend(metrics_stats.keys())
                writer.writerow(headers)
            generation_stats = [generation, checkpoint_id]
            generation_stats.extend(metrics_stats.values())
            writer.writerow(generation_stats)

    def persist_generation(self, response):
        """
        Persists the details of a generation to a file.
        :param response: an evaluated ESP PopulationResponse
        :return: nothing. Saves a file called `gen.csv` to the persistence directory (e.g. 1.csv for generation 1)
        """
        gen = response.generation_count
        gen_filename = os.path.join(self.save_to_dir, str(gen) + '.csv')
        # 'w' to truncate the file if it already exists
        candidates_info = []
        with open(gen_filename, 'w') as stats_file:
            writer = csv.writer(stats_file)
            write_header = True
            for candidate in response.population:
                # Candidate's details
                cid = candidate.id
                identity = candidate.identity.decode('UTF-8')
                metrics = self.extension_packaging.from_extension_bytes(candidate.metrics)
                c = {"id": cid,
                     "identity": identity,
                     "metrics": metrics,
                     "model": candidate.interpretation}
                candidates_info.append(c)

                # Write the header if needed
                if write_header:
                    # Unpack the metric names list
                    writer.writerow(["cid", "identity", *metrics.keys()])
                    write_header = False

                # Write a row for this candidate
                row_values = [cid, identity]
                row_values.extend(metrics.values())
                writer.writerow(row_values)
        return candidates_info

    def persist_candidates(self, candidates_info, gen):
        """
        Persists the candidates in the response's population according to the experiment params.
        Can be "all", "elites", "best", "none"
        :param candidates_info: a PopulationResponse containing evaluated candidates
        :param gen: the generation these candidates belong to
        :return: nothing. Saves the candidates to a generation folder in the persistence directory
        """
        if self.candidates_to_persist == "none":
            return

        gen_folder = os.path.join(self.save_to_dir, str(gen))
        os.makedirs(gen_folder, exist_ok=True)
        if self.candidates_to_persist == "all":
            for candidate in candidates_info:
                self.persist_candidate(candidate, gen_folder)
        elif self.candidates_to_persist == "best":
            # Save the best candidate, per objective
            objectives = self.experiment_params['evolution'].get("fitness", DEFAULT_FITNESS)
            for objective in objectives:
                # Sort the candidates to figure out the best one for this objective
                metric_name = objective["metric_name"]
                candidates_info.sort(key=lambda k: k["metrics"][metric_name],
                                     reverse=objective["maximize"])
                # The best candidate for this objective is the first one
                self.persist_candidate(candidates_info[0], gen_folder)
        elif self.candidates_to_persist == "elites":
            # Save the elites, according to the first objective only
            nb_elites = self.experiment_params["evolution"]["nb_elites"]
            objectives = self.experiment_params['evolution'].get("fitness", DEFAULT_FITNESS)
            objective = objectives[0]
            metric_name = objective["metric_name"]
            candidates_info.sort(key=lambda k: k["metrics"][metric_name],
                                 reverse=objective["maximize"])
            for candidate in candidates_info[len(candidates_info) - nb_elites:]:
                self.persist_candidate(candidate, gen_folder)
        else:
            print("Skipping candidates persistence: unknown candidates_to_persist attribute: {}".format(
                self.candidates_to_persist))

    def persist_candidate(self, candidate, gen_folder):
        """
        Persists a candidates to a file
        :param candidate: the candidates to persist
        :param gen_folder: the folder to which to persist it
        :return: nothing. Saves the candidate to a cid.h5 file in generation folder in the persistence directory
        (where cid is the candidate id)
        """
        cid = candidate["id"]
        filename = cid + ".h5"
        filename = os.path.join(gen_folder, filename)
        representation = self.experiment_params["LEAF"]["representation"]
        if representation == "KerasNN":
            self.persist_keras_nn_model(candidate["model"], filename)
        elif representation == "NNWeights":
            self.persist_nn_weights_model(candidate["model"], filename)
        else:
            print("Persistor: Unknown representation: {}".format(representation))

    @staticmethod
    def persist_keras_nn_model(model_bytes, filename):
        """
        Converts the passed bytes to a Keras model and saves it to a file
        :param model_bytes: the bytes corresponding to a Keras model
        :param filename: the name of the file to save to
        :return: nothing
        """
        # Convert the received bytes to a Keras model
        import io
        from keras.models import load_model
        model_file = io.BytesIO(model_bytes)
        keras_model = load_model(model_file)
        # Save the model, without the optimizer (not used)
        keras_model.save(filename, include_optimizer=False)

    def persist_nn_weights_model(self, weights_bytes, filename):
        """
        Creates a model from the passed weight bytes and saves it to a file
        :param weights_bytes: the bytes corresponding to a Keras model weights
        :param filename: the name of the file to save to
        :return: nothing
        """
        indy_weights = pickle.loads(weights_bytes)
        model = self.evaluator.get_keras_model(indy_weights)
        # Save the model, without the optimizer (not used)
        model.save(filename, include_optimizer=False)
