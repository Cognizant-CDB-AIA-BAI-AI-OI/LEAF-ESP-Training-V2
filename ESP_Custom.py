#!/usr/bin/env python
# coding: utf-8

# In[1]:


#All Imports

import copy
import h5py
import io
import json
import numpy as np
import os
import pandas as pd
import pickle
import time
import argparse
from collections import OrderedDict

from keras.models import load_model
from keras.models import clone_model
from google.protobuf.json_format import ParseDict

from esp_sdk.v1_0.esp_persistor import EspPersistor
from esp_sdk.v1_0.esp_service import EspService
from data_util.data_set_encoder import DataSetEncoder
from data_util.data_set_util import DataSetUtil

from getFitness import Fitness_Evaluator 

print ("ALL IMPORTS COMPLETED !!!!!!")


# In[2]:


DATA_PATH = "./data_exe3"
print ("**********OS Current Directory********",os.getcwd())
TRAINING_DATA_CSV = "training_data_withScore.csv"
training_data = pd.read_csv(TRAINING_DATA_CSV, sep=',')
training_data.head(10)


# In[3]:


columnsarr=list(training_data.columns.values)
CAO_variables = [2,2,1]


# In[4]:


#File name definitions

#Context, Actions and Outcomes column names


CONTEXT_CSV = os.path.join(DATA_PATH, "context.csv")
ACTIONS_CSV = os.path.join(DATA_PATH, "actions.csv")
OUTCOMES_CSV = os.path.join(DATA_PATH, "outcomes.csv")
POSSIBLE_VALUES_CSV = os.path.join(DATA_PATH, "possible_values.csv")


# In[5]:


#Evaluation samples: samples we want to use to evaluate a candidate

EVALUATION_SAMPLES_CSV =  os.path.join(DATA_PATH, "splits/70-30/val_set_0.csv")


# In[6]:


PRESCRIPTOR_JSON = os.path.join(DATA_PATH, "prescriptor.json")
PRESCRIPTOR_H5 = os.path.join(DATA_PATH, 'prescriptor.h5')
PREDICTOR_H5 = os.path.join(DATA_PATH, "predictor.h5")


# In[7]:


#Output directory to store training results

TRAINED_PRESCRIPTORS_DIR = "trained_prescriptors/"


# In[8]:


#Context Creation

context=[]
for i in range(0,CAO_variables[0]):
    context.append(columnsarr[i])
context_df=pd.DataFrame(columns=context)
context_names = list(context_df.columns)
print ("# context variables : ", context_names)
context_df = context_df.replace(np.nan, '', regex=True)
context_df.to_csv(CONTEXT_CSV,index=False)


# In[9]:


#Action Creation

action=[]
for i in range(0,CAO_variables[1]):
    action.append(columnsarr[i+CAO_variables[0]])
actions_df=pd.DataFrame(columns=action)
actions_names = list(actions_df.columns)
print ("# Action variables : ", actions_names)
actions_df = actions_df.replace(np.nan, '', regex=True)
actions_df.to_csv(ACTIONS_CSV,index=False)


# In[10]:


#Outcome Creation

outcome=[]
for i in range(0,CAO_variables[2]):
    outcome.append(columnsarr[i+CAO_variables[0]+CAO_variables[1]])
outcome_df=pd.DataFrame(columns=outcome)
outcome_names = list(outcome_df.columns)
print ("# Outcome  variables : ", outcome_names)
outcome_df = outcome_df.replace(np.nan, '', regex=True)
outcome_df.to_csv(OUTCOMES_CSV,index=False)


# In[11]:


#Possible Values

possible_values_list=[]
for i in range(0,sum(CAO_variables)):
    possible_values_list.append(training_data[columnsarr[i]].str.capitalize().unique().flatten())
    
possible_values_df=pd.DataFrame(data={'value':possible_values_list},index=columnsarr[0:sum(CAO_variables)]).value.apply(pd.Series).T
possible_values_df = possible_values_df.replace(np.nan, '', regex=True)
possible_values_df.to_csv(POSSIBLE_VALUES_CSV,index=False)


# In[12]:


#Create Experiment Parameters

possible_values_dict = DataSetEncoder.get_possible_values_dict(POSSIBLE_VALUES_CSV)
experiment_params = DataSetEncoder.generate_model_description(possible_values_dict,
                                                              CONTEXT_CSV,
                                                              ACTIONS_CSV,
                                                              nb_hidden_units=10,
                                                              use_bias=True,
                                                              activation_function="tanh",
                                                              include_evo_description=True)
                                                              
                                                              
#Tweak Experiment Parameters

experiment_params["LEAF"] = {
    "esp_host": "v1.esp.evolution.ml",
    "esp_port": 50051,
    "representation": "NNWeights",
    "experiment_id":"esp-custom",
    "version": "1.0.0",
    "candidates_to_persist": "all",
    "persistence_dir": TRAINED_PRESCRIPTORS_DIR
}
experiment_params["evolution"] = {
    "nb_generations": 5,
    "population_size": 25,
    "nb_elites": 1,
    "remove_population_pct": 0.8,
    "mutation_probability": 0.5,
    "mutation_factor": 0.5,
    "initialization_distribution": "orthogonal",
    "initialization_range": 1
}

# Network

#experiment_params["network"]["nb_hidden_units"] = 25
experiment_params["network"]["activation_function"] = "tanh"


# In[13]:


#Save Experiment Parameters

with open(PRESCRIPTOR_JSON, 'w') as fp:
    json.dump(experiment_params, fp)
    
#load Experiment Parameters
with open(PRESCRIPTOR_JSON) as json_data:
    experiment_params = json.load(json_data)
    
    
evaluator_df = training_data.drop('Conversion Rate', 1).drop_duplicates()
print ("evaluator_df  >>> ", evaluator_df)
print ("len(evaluator_df) ... ", len(evaluator_df))


# In[14]:


#Presciptor Training

def create_evaluator():
    evaluator = Fitness_Evaluator(context_names, actions_names, POSSIBLE_VALUES_CSV, evaluator_df, experiment_params,training_data)
    return evaluator
    
def train(experiment_params, evaluator, checkpoint_id=None):
    esp_service = EspService(experiment_params)
    persistor = EspPersistor(experiment_params, evaluator)
    experiment_id = experiment_params["LEAF"]["experiment_id"]
    timestamp = time.strftime("%Y%m%d-%H%M%S")    
    
    print("Starting training:")
    print("  experiment_id: {}".format(experiment_id))
    print("  checkpoint_id: {}".format(checkpoint_id))
    print("  timestamp: {}".format(timestamp))
    print("  experiment results dir: {}".format(persistor.get_persistence_directory()))

    # Starting point
    if checkpoint_id:
        # We're starting with a check-pointed population.
        print("Asking ESP for a check-pointed population: {}...".format(checkpoint_id))
        response = esp_service.get_previous_population(experiment_id, checkpoint_id)
        current_gen = response.generation_count
    else:
        # We're starting with the 1st generation, which is a 'seed' population because we create it from scratch
        print("Asking ESP for a seed generation...")
        response = esp_service.get_next_population(prev_response=None)
        print("Seed generation received.")
        current_gen = 1
    nb_gen = experiment_params["evolution"]["nb_generations"]
    for gen in range(current_gen, nb_gen + 1):
        print("Evaluating the following PopulationResponse for generation {}...:".format(gen))
        esp_service.print_population_response(response)

        # Evaluate the population. This is going to update the metrics on the candidates contained in the response
        evaluator.evaluate_population(response)
        print("Evaluation done.")

        # Persist the evaluated population
        persistor.persist_response(response)
        print("Generation's info persisted to {}".format(persistor.get_persistence_directory()))

        # Print the candidates and their scores
        esp_service.print_candidates(response)
        
        print("Done with generation {}.".format(gen))
        print("--------\n")
        if gen < nb_gen:
            # Get a new generation from the previous one we've just evaluated
            print("Asking ESP for generation {}...:".format(gen + 1))
            response = esp_service.get_next_population(prev_response=response)
    return persistor.get_persistence_directory()
    


# In[15]:


evaluator = create_evaluator()
experiment_results_dir = train(experiment_params, evaluator)


# In[16]:


#Evolution Summary

stats_file = os.path.join(experiment_results_dir, 'experiment_stats.csv')
with open(stats_file) as csv_file:
    stats_df = pd.read_csv(csv_file, sep=',')
stats_df


# In[17]:


#Evolution Plot

from IPython.display import Image
plot_file = os.path.join(experiment_results_dir, 'experiment_stats.png')
Image(filename=plot_file)


# In[18]:


#Load Best Candidate

last_generation = experiment_params["evolution"]["nb_generations"]
candidate_id_with_max_score = stats_df.iloc[experiment_params["evolution"]["nb_generations"] -1]["cid_max_score"]
max_score = stats_df.iloc[last_generation -1]["max_score"]
print('Best candidate is ',candidate_id_with_max_score,' with a score of ',max_score)


# In[19]:


#Prescribe the actions based on the contexts from the saved model

best_model_filename = os.path.join(experiment_results_dir,
                                   str(last_generation),
                                   candidate_id_with_max_score + ".h5")
best_model = load_model(best_model_filename)
best_model.summary()

windows_user = np.array([[1, 0, 0]])
region = np.array([[1, 0]])

prescription = best_model.predict([region,windows_user])
print("The raw Prescriptor model output is {}".format(prescription))


# In[ ]:




