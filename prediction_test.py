#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# python3 prediction_test.py cities_list.txt
# python3 metric_test.py -lst cities_list.txt -nep 20 -bch 2 4 8 16

"""
Testing process to generate a metric to measure fAIr's performance, on prediction.

Created on Tue 16 July 2924

@author: AnnaZ
"""

### Import modules, variables definition, initial checks
import sys
sys.path
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

# used later on in the script:
from glob import glob
import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import shutil

import cv2
import warnings
warnings.filterwarnings("ignore")
# initialise keras
os.environ["SM_FRAMEWORK"] = "tf.keras"

# ramp specific
import ramp

# fair_utilities specific
import hot_fair_utilities
# from hot_fair_utilities import preprocess, predict, polygonize
from hot_fair_utilities.training.cleanup import extract_highest_accuracy_model
from hot_fair_utilities import predict
from hot_fair_utilities import polygonize

from ramp.training import (
    callback_constructors,
    loss_constructors,
    metric_constructors,
    model_constructors,
    optimizer_constructors,
)

# for the parser
import argparse




def main():
    parser = argparse.ArgumentParser(
        description=""" 
    Runs training on fAIr fine tuned model using training data as in the inputted names list, n of epochs, batch size;
    performs several metrics using single loss and returns a list of their relative accuracy measure

    Example: python3 prediction_test.py -lst cities_list.txt
    """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-lst",
        "--names_list",
        type=str,
        required=True,
        help=r"Name of txt file containing the list of training datasets",
    )
    
    args = parser.parse_args()

    list_filename =args.names_list

    ### defining path variables
    # base_path = "/Users/azanchetta/fAIr-utilities" # this path is used in all the rest of the code, so change accordingly
    base_path = f"{os.getcwd()}"
    # base_path = "/Users/azanchetta/fAIr_metric/"
    print(f"\n---\nCurrent working directory {base_path}")
    
    #  --- FOR LOCAL MACHINE:
    # metric_path = "/Users/azanchetta/fAIr_metric"
    #  ---

    # check that training-datasets list file exists and is readable
    if not os.path.exists(f'{base_path}/{list_filename}'):
        raise ValueError(f"Can't find file {list_filename}")

    ### generating list of regions (cities) from the input txt file
    # input text file from command line:
    # list_filename = "cities_list.txt"

    #TO_DO: add condition of stopping if filename empty or file doesn't exist
    print(f"---\nI am going to get the names from {list_filename} (name of the list file you provided)")
    # the following is to obtain the list of cities, removing commented lines (starting with "#"):
    with open(list_filename, 'r') as f:
        full_file = f.read()
        # print(full_file)
        full_list = full_file.split('\n') # separating per each new line
        cities_list = []
        for counter in range(len(full_list)):
            line = full_list[counter]
            # print(f"line is {line}")
            if not line.startswith('#'): # this is to avoid commented lines in the input file
                cities_list.append(full_list[counter])
    print(f"---\nList of cities: {cities_list}")

    # path_to_data = f'{base_path}/ramp-data/metric_data' # for metric data path
    
    
    for city in cities_list:
        # print(f"Now working on {city} preprocess")
        
        #  --- FOR LOCAL MACHINE:
        # city_path = f"{metric_path}/training_results/{city}" # make up string from base_path + city_name_from_list
        #  ---
        city_path=f"{base_path}/ramp-data/metric_data/{city}"
    # ## Split training dataset to: training, validation, prediction sets ... it's been already split!!
    #     print(f"---\nSplitting the data training into training, validation and prediction datasets:\n")
    #     from hot_fair_utilities.training.prepare_data import split_training_2_validation_2_prediction
    #     x = split_training_2_validation_2_prediction(preprocess_output, train_output)
    
    ### Prepare data and environment for prediction
        print(f"---\nCity path is {city_path}\n---")
        print(f"---\nPreparing data for city {city}\n---")
        # obtain name of checkpoint file for batch size 8 (the third subfolder in train/model-checkpts/)
        pattern_for_subdirs=f"{city_path}/train/model-checkpts"
        print(f"---\nPattern for subdirs is {pattern_for_subdirs}\n---")
        subfolders_list = [fold for fold in sorted(os.listdir(pattern_for_subdirs)) if not fold.startswith('.')] # this is to avoid hidden folders/files to be listed (names starting with '.')
        print(f"Lis of model checkpoints subfolders: {subfolders_list}")
        #  getting the third one for batch size 8:
        name_Iwant = subfolders_list[2] # 3rd position item!
        print(name_Iwant)
        checkpt_8batch_folder_path=f"{pattern_for_subdirs}/{name_Iwant}"
        model_folder_name=os.listdir(checkpt_8batch_folder_path)[0] # [fold.name for fold in os.scandir(checkpt_8batch_folder_path) if fold.is_dir()]  #os.listdir(checkpt_8batch_folder_path)
        print(f"model folder name {model_folder_name}")
        final_model_path =f"{checkpt_8batch_folder_path}/{model_folder_name}"
        print(f"Final model path: {final_model_path}")
        
        # # --- nairobi poster, temporary!!
        # final_model_path="/home/annazan/fAIr-utilities/ramp-code/ramp/checkpoint.tf"
        # print(f"Final model path: {final_model_path}")
        # # pred_input_path="/home/annazan/fAIr-utilities/ramp-data/metric_data/nairobi5/chips"
        # pred_input_path="/home/annazan/fAIr-utilities/ramp-data/metric_data/nairobi2/"
        # # --- end of nairobi
        
        
        pred_input_path=f"{city_path}/train/pred-chips"
        print(f"Prediction chips (rgb tiles): {pred_input_path}")
        #  --- FOR LOCAL MACHINE:
        # prediction_output = f"{metric_path}/predictions"
        # ---
        prediction_base_output=f"{base_path}/ramp-data/predictions"
        # print(f"Prediction output path: {prediction_base_output}")
        
        # generate folder for each city with predictions:
        city_dir = f"{prediction_base_output}/{city}"
        if os.path.exists(city_dir):
            print(f"City folder {city_dir} already exist, I am removing it and re-creating it")
            shutil.rmtree(city_dir)
            os.makedirs(city_dir)
        # os.mkdir(os.path.join(prediction_base_output,city))
        prediction_output = city_dir
        print(f"prediction output {prediction_output}")
        
### ------- Prediction

        print(f"\n---\n---\nStarting prediction on {city}\n")
        # prediction_output = f"{path_to_output}/{city}/prediction"   # !!! change file name here
        
        predict(
            checkpoint_path=final_model_path,
            input_path=pred_input_path,
            prediction_path=prediction_output
        )

#### ------ Polygonization

        print(f"\n---\n---\nStarting polygonise result of {city}\n")
        # add zoom level information for each prediction tile
        # geojson_output= f"{prediction_output}/prediction.geojson"
        zoom_list=[19,20,21]
        for zoom in zoom_list:
            print(f"input zOOOOOm {zoom}")
            # geojson_temp_output=f"{prediction_output}/prediction_temp{zoom}.geojson"
            geojson_temp_output="temp-labels.geojson"
            geojson_output=f"{prediction_output}/prediction_{zoom}.geojson"
            polygonize(
                input_path=prediction_output, 
                temp_path=geojson_temp_output,
                output_path=geojson_output,
                zoom_levels = zoom,
                remove_inputs = False
                )


if __name__ == "__main__":
    main()