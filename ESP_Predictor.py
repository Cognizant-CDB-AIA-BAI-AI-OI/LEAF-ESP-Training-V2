#!/usr/bin/env python
# coding: utf-8

# In[1]:


#All Imports

import grpc
import io
import json
import os
import pandas as pd
import time
import numpy as np

from keras.models import Model
from keras.models import load_model

from esp_sdk.v1_0.esp_service import EspService
from data_util.data_set_encoder import DataSetEncoder
from data_util.data_set_util import DataSetUtil
from data_util.surrogate_model import SurrogateModel

print ("All Imports Done..")


# In[2]:


#File Declarations

DATA_DIR = "./data_context5/"
SPLITS_PATH_ROOT = os.path.join(DATA_DIR, 'splits')

#Predictor

PREDICTOR_JSON = os.path.join(DATA_DIR, 'predictor.json')
PREDICTOR_INPUTS_CSV = os.path.join(DATA_DIR, 'predictor_inputs.csv')
PREDICTOR_OUTPUTS_CSV = os.path.join(DATA_DIR, 'predictor_outputs.csv')
PREDICTOR_H5 = os.path.join(DATA_DIR, 'predictor.h5')

#Training data

TRAINING_DATA_CSV = os.path.join(DATA_DIR, "training_data.csv")
TRAINING_DATA_ENCODED_CSV = os.path.join(DATA_DIR, "training_data_encoded.csv")
POSSIBLE_VALUES_CSV = os.path.join(DATA_DIR, 'possible_values.csv')


# In[3]:


#load training data

with open(TRAINING_DATA_CSV) as df_file:
    df = pd.read_csv(df_file, keep_default_na=False)
print (df.head())

TRAIN_PCT = 0.70
VAL_PCT = 1 - TRAIN_PCT
SPLIT_NAME = "{:2.0f}-{:2.0f}".format(TRAIN_PCT * 100, VAL_PCT * 100)
SPLIT_PATH = os.path.join(SPLITS_PATH_ROOT, SPLIT_NAME)

#Number of samples

NB_SAMPLES_TOTAL = 130
NB_SAMPLES_TRAINING = int(round(NB_SAMPLES_TOTAL * TRAIN_PCT))
NB_SAMPLES_VALIDATION = int(round(NB_SAMPLES_TOTAL * VAL_PCT))

NB_RUNS = 1
NB_EPOCHS = 200
BATCH_SIZE = NB_SAMPLES_TRAINING

print("Spliting {:00.0%} training / {:00.0%} validation".format(TRAIN_PCT, VAL_PCT))
print("Nb samples: {}".format(NB_SAMPLES_TOTAL))
print("Nb training samples: {}".format(NB_SAMPLES_TRAINING))
print("Nb validation samples: {}".format(NB_SAMPLES_VALIDATION))

data_sets = []
print("Splitting data in {} train/val sets: {:00.2%}/{:00.2%}".format(NB_RUNS, TRAIN_PCT, VAL_PCT))
print("Creating train set of {} samples, val set of {} samples...".format(NB_SAMPLES_TRAINING, NB_SAMPLES_VALIDATION))


# In[4]:


#Generating Dataset

for i in range(NB_RUNS):
    print("  Generating data set #{}...".format(i + 1))
    start = time.time()
    train_set, val_set = DataSetUtil.split_train_val(TRAINING_DATA_CSV, train_pct=TRAIN_PCT)
    data_sets.append([train_set, val_set])
    end = time.time()
    print("  Dataset generated in {} seconds.".format(end - start))
print("Done.")

DataSetUtil.persist_data_sets(data_sets, SPLIT_PATH)
data_sets = DataSetUtil.load_data_sets(NB_RUNS, SPLIT_PATH)


# In[5]:


#Data Encoding i.e. changing categorized variable to one-hot encoding

train_set = data_sets[0][0]
val_set = data_sets[0][1]

print ("Before Encoding ", data_sets)
encoded_data_sets = DataSetUtil.encode_data_sets(data_sets, POSSIBLE_VALUES_CSV)
print ("Encoded data sets ... done !!" , encoded_data_sets)


# In[6]:


#Get Possible Values Of Variables

possible_values_dict = DataSetEncoder.get_possible_values_dict(POSSIBLE_VALUES_CSV)
possible_values_dict


# In[7]:


#Define Experiment Parameters

experiment_params = DataSetEncoder.generate_model_description(possible_values_dict,
                                                              PREDICTOR_INPUTS_CSV,
                                                              PREDICTOR_OUTPUTS_CSV,
                                                              nb_hidden_units=10,
                                                              use_bias=True,
                                                              activation_function="tanh",
                                                              include_evo_description=False)
experiment_params["LEAF"] = {
    "esp_host": "v1.esp.evolution.ml",
    "esp_port": 50051,
    "representation": "NNWeights",
    "experiment_id":"esp_context5",
    "version": "1.0.0",
    "persistence_dir": "trained_prescriptors/"
}

#Save Experiment Parameters

with open(PREDICTOR_JSON, 'w') as fp:
    json.dump(experiment_params, fp)
experiment_params


# In[8]:


#Load Experiment Parameters

with open(PREDICTOR_JSON) as json_data:
    experiment_params = json.load(json_data)
    
esp_service = EspService(experiment_params)
model_to_save = esp_service.request_base_model()
# Compile the model to avoid warnings when loading it
model_to_save.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])


# In[9]:


#Function Definitions
def create_input_outputs(encoded_data_sets):
   input_output_sets = []
   for i, encoded_ds in enumerate(encoded_data_sets):
       print("Creating input/output for set #{}...".format(i + 1))
       start = time.time()
       train_df = encoded_ds[0]
       val_df = encoded_ds[1]
       train_features, train_labels = SurrogateModel.create_inputs_outputs_df(experiment_params, train_df)
       val_features, val_labels = SurrogateModel.create_inputs_outputs_df(experiment_params, val_df)
       input_output_sets.append([train_features, train_labels, val_features, val_labels])
       end = time.time()
       print("Dataset generated in {} seconds.".format(end - start))
   return input_output_sets


# In[10]:


input_output_sets = create_input_outputs(encoded_data_sets)
input_output_sets


# In[11]:


def print_input_output_info(input_output_sets):
   print("Number of datasets: {}".format(len(data_sets)))
   ds1 = input_output_sets[0]
   print("A data set contains {} sets: train_x, train_y, val_x, val_y".format(len(ds1)))
   ds1_train_x = ds1[0]
   ds1_train_y = ds1[1]
   ds1_val_x = ds1[2]
   ds1_val_y = ds1[3]
   print("train_x contains {} rows x {} columns".format(len(ds1_train_x[0]), len(ds1_train_x)))
   print("train_y contains {} rows x {} columns".format(len(ds1_train_y[0]), len(ds1_train_y)))
   print("val_x contains {} rows x {} columns".format(len(ds1_val_x[0]), len(ds1_val_x)))
   print("val_y contains {} rows x {} columns".format(len(ds1_val_y[0]), len(ds1_val_y)))


# In[12]:


print_input_output_info(input_output_sets)


# In[13]:


#TRAIN
#Model Definition
#Get the Network Parameters

network_params = experiment_params['network']
output_layers = network_params['outputs']
output_names = [output_layer['name'] for output_layer in output_layers]

## create and train model
histories = []
keras_verbose = False
for i in range(NB_RUNS):
   
   print("Training #{}...".format(i+1))
   # Note: using the test for validation. Not enough data to have a proper test set
   train_data, train_labels, val_data, val_labels = input_output_sets[i]
   
   # Create a new model from model_to_save, with the SAME layers up to the argmax lambda layers.
   # Argmax breaks backpropagation.
   outputs = [model_to_save.get_layer(n).output for n in output_names]
   model_to_train = Model(inputs=model_to_save.inputs,
                          outputs=outputs)

   #  Compile it
   model_to_train.compile(optimizer='adam',
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
       
   end = time.time()
   print("    Model created and compiled in {} seconds.".format(end - start))
   
   # Train!
   print("    Training the model...")
   start = time.time()
   history = model_to_train.fit(train_data,
                                train_labels,
                                epochs=NB_EPOCHS,
                                batch_size=BATCH_SIZE,
                                validation_data=(val_data, val_labels),
                                shuffle=True,
#                                  class_weight=class_weights,
                                verbose=keras_verbose)
   end = time.time()
   print("    Model trained in {} seconds.".format(end - start))

   
   histories.append(history)
   print("Finished training #{}...".format(i+1))
print("Done.")


# In[14]:


output_file = PREDICTOR_H5

#Make sure to save the model_to_save and not the model_to_train
#The model to save contains an argmax lambda on its outputs used to convert the outputs to one-hot vectors

model_to_save.save(output_file)
print ("predictor model saved !!")


# In[15]:


#Predict the model for various values of Context and Decision

context_and_actions_header = "Product,Browser,Device Resolution,Region,OS,page color,button color"
context_and_actions_row_1 = "Loan,Chrome,1080x1920 pixels,Asia,MacOS,Grey,Red"
context_and_actions_row = context_and_actions_row_1
context_and_actions_row_csv = context_and_actions_header + "\n" + context_and_actions_row
from io import StringIO
context_and_actions_data = StringIO(context_and_actions_row_csv)
context_and_actions_data_df = pd.read_csv(context_and_actions_data)

#Make sure the column "Production  Defect Range" contains string, not ints

context_and_actions_data_df
print ("@!!@!@ context_and_actions_data_df.. ",context_and_actions_data_df)

#Encode it

encoded_context_and_actions_data_df = DataSetEncoder.encode_df(possible_values_dict, context_and_actions_data_df, verbose=False)
from keras.models import load_model
predictor = load_model(PREDICTOR_H5)

surrogate_model = SurrogateModel()
predictor_inputs_df = surrogate_model.create_inputs_df(experiment_params, encoded_context_and_actions_data_df)
encoded_predictions = predictor.predict(predictor_inputs_df)
print ("$$#$$$$$#$$#$##$#$#$$ encoded_predictions.... ",encoded_predictions)


# In[ ]:




