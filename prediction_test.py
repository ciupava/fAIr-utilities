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
    # parser.add_argument(
    #     "-nep",
    #     "--n_of_epochs",
    #     type=int,
    #     required=False,
    #     default=2,
    #     help="Number of epochs to train the model for",
    # )
    # parser.add_argument(
    #     "-bch",
    #     "--batch_size",
    #     type=int,
    #     nargs='+', # https://stackoverflow.com/questions/15753701/how-can-i-pass-a-list-as-a-command-line-argument-with-argparse
    #     required=False,
    #     default=5,
    #     help="Batch size to train model",
    # )
    args = parser.parse_args()

    # importing variables from command line
    # n_of_epochs = args.n_of_epochs
    # # n_of_batches = args.batch_size
    # n_of_batches_array = args.batch_size

    list_filename =args.names_list


    ### defining path variables
    # base_path = "/Users/azanchetta/fAIr-utilities" # this path is used in all the rest of the code, so change accordingly
    base_path = f"{os.getcwd()}"
    # base_path = "/Users/azanchetta/fAIr_metric/"
    print(f"\n---\nCurrent working directory {base_path}")
    metric_path = "/Users/azanchetta/fAIr_metric"

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
        city_path = f"{metric_path}/training_results/{city}" # make up string from base_path + city_name_from_list

    # ## Split training dataset to: training, validation, prediction sets ... it's been already split!!
    #     print(f"---\nSplitting the data training into training, validation and prediction datasets:\n")
    #     from hot_fair_utilities.training.prepare_data import split_training_2_validation_2_prediction
    #     x = split_training_2_validation_2_prediction(preprocess_output, train_output)

    # # ---
    # # TODO: add a call with `extract_highest_accuracy_model` to do cleanup
    # # extract_highest_accuracy_model(model_path)
    #     final_accuracy, final_model_path = extract_highest_accuracy_model(output_path)
    #     print(f'\n-----\nFinal accuracy: {final_accuracy}')
    #     print(f'\n-----\nFinal model path: {final_model_path}')
    #     print('\n-----')
    # # ---
    
    ### Prepare data and environment for prediction
        print(f"---\nCity path is {city_path}\n---")
        print(f"---\nPreparing data for city {city}\n---")
        # obtain name of checkpoint file for batch size 8 (the third subfolder in train/model-checkpts/)
        pattern_for_subdirs=f"{city_path}/train/model-checkpts/"
        print(f"---\nPattern for subdirs is {pattern_for_subdirs}\n---")
        # subfolders_list = [fold.name for fold in os.scandir(pattern_for_subdirs) if fold.is_dir()]
        # print(f"\n---\{subfolders_list}")
        # for subdirrr in glob(pattern_for_subdirs, recursive=True):
        #     print(f"subdir is {subdirrr}")
        # for f in sorted(os.listdir(pattern_for_subdirs)): print(f)
        # subfolders_list = [sorted(os.listdir(pattern_for_subdirs))]
        subfolders_list = [fold for fold in sorted(os.listdir(pattern_for_subdirs)) if not fold.startswith('.')]
        print(f"Lis of model checkpoints subfolders: {subfolders_list}")
        #  getting the third one for batch size 8:
        name_Iwant = subfolders_list[2]
        print(name_Iwant)
        
    # ### Prediction
    # # 

        print(f"\n---\n---\nStarting prediction on {city}\n")
        # prediction_output = f"{path_to_output}/{city}/prediction"   # !!! change file name here

        pred_input_path=f"{city_path}/train/pred-chips"
        prediction_output = "" #### NAME PATH HERE!!!!!!!!!!!!!!!!!!
        predict(
            checkpoint_path=final_model_path,
            # input_path=f"{city_path}/prediction/input", # the same of above?
            input_path=pred_input_path,
            prediction_path=prediction_output
        )

    #### ------ Prediction metrics

    # # from hot_fair_utilities import polygonize
    #     print(f"\n---\n---\nStarting polygonise result on {city}\n")
    #     geojson_output= f"{prediction_output}/prediction.geojson"
    #     polygonize(
    #         input_path=prediction_output, 
    #         output_path=geojson_output,
    #         remove_inputs = False,
    #     )

if __name__ == "__main__":
    main()