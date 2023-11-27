#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing process to perform a metric on measurement of fAIr's performance.

Created on Wed 22 Nov 2023

@author: Anna
"""

### Import modules, variables definition, initial checks
import sys
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

import os
print(os.getcwd())
os.environ.update(os.environ)
        # Add a new environment variable to the operating system
os.environ["RAMP_HOME"] = os.getcwd()
# Print the environment variables to verify that the new variable was added
print(os.environ["RAMP_HOME"])
# sys.path.append('../')
sys.path.append('ramp-code/')

import cv2

# initialise keras
os.environ["SM_FRAMEWORK"] = "tf.keras"

import ramp.utils # ??
import hot_fair_utilities
from hot_fair_utilities import preprocess, train, predict, polygonize


# defining path variables
# base_path = f"{os.getcwd()}/ramp-data/sample_2"
base_path = "/Users/azanchetta/fAIr-utilities/ramp-data/test_data" # this path is used in all the rest of the code, so change accordingly
# we should change this variable to be just the home variable, and use after another variable called "city_data" or similar
# need:
#   - a path for input city, preprocessed images will be saved here (city_data)
#   - a path for output model checkpoint (to be saved each time it runs), atm it's overwritten every time the training is run on a data folder (city)
#   - a path for the training outputs, as we shall run in different conditions (batch, epochs, zoom levels, ...) for the same city, and be able to compare results
# should have an automatic way to save the output files which shall be easy to call back
# naming should account for the (3?) variables against which we want to evaluate the model:
# 
# add duration (time) for each city in the for loop

cities_list = ["1_Zanzibar", "2_Kampala"] # will be used to loop into, initially manually inputted, can become a text file

for city in cities_list:
    print(f"Now working on {city} preprocess")
    city_path = f"{base_path}/{city}" # make up string from base_path + city_name_from_list


### Preprocessing
# Should run on a series of cities from the list above
# the preprocessing shouldn't run if images have already been preprocessed
# from hot_fair_utilities import preprocess  # should all the imports be put at beginning?

    model_input_image_path = f"{city_path}/input" # !!! change name here
    preprocess_output=f"{city_path}/preprocessed" # !!! change name here
    preprocess(
                input_path = model_input_image_path,
                output_path = preprocess_output,
                rasterize=True,
                rasterize_options=["binary"],
                georeference_images=True,
            )

### Training
# from hot_fair_utilities import train
    print(f"\n---\n---\nStarting training on {city}\n")
    train_output = f"{city_path}/train" # !!! change name here
    final_accuracy, final_model_path = train(  # !!! final model path has to be changed in the function
        input_path=preprocess_output,
        output_path=train_output,
        epoch_size=2, # need to be able to change also the epoch size?
        batch_size=2, # need to be able to change also the batch size?
        model="ramp",
        model_home=os.environ["RAMP_HOME"]
    )

    print(f"Final accuracy: {final_accuracy} and final model path: {final_model_path}")


### Prediction
# 
# from hot_fair_utilities import predict
    print(f"\n---\n---\nStarting prediction on {city}\n")
    prediction_output = f"{city_path}/prediction/output"   # !!! change file name here
    predict(
        checkpoint_path=final_model_path,
        input_path=f"{city_path}/prediction/input", # the same of above?
        prediction_path=prediction_output,
    )

# from hot_fair_utilities import polygonize
    print(f"\n---\n---\nStarting polygonise result on {city}\n")
    geojson_output= f"{prediction_output}/prediction.geojson"
    polygonize(
        input_path=prediction_output, 
        output_path=geojson_output,
        remove_inputs = True,
    )

### Accuracy
# Add IoU here