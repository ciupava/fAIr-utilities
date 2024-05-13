#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# python3 metric_test.py cities_list.txt
# python3 metric_test.py -lst cities_list.txt -nep 20 -bch 2 4 8 16

"""
Testing process to generate a metric to measure fAIr's performance.

Created on Wed 22 Nov 2023

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
import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

import cv2
import warnings
warnings.filterwarnings("ignore")
# initialise keras
os.environ["SM_FRAMEWORK"] = "tf.keras"

# ramp specific
import ramp

# fair_utilities specific
import hot_fair_utilities
from hot_fair_utilities import preprocess, predict, polygonize
from hot_fair_utilities.training.cleanup import extract_highest_accuracy_model

# for the parser
import argparse

def main():
    parser = argparse.ArgumentParser(
        description=""" 
    Runs training on fAIr fine tuned model using training data as in the inputted names list, n of epochs, batch size;
    performs several metrics using single loss and returns a list of their relative accuracy measure

    Example: python3 metric_test.py -lst cities_list.txt -nep 20 -bch 5
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
    parser.add_argument(
        "-nep",
        "--n_of_epochs",
        type=int,
        required=False,
        default=2,
        help="Number of epochs to train the model for",
    )
    parser.add_argument(
        "-bch",
        "--batch_size",
        type=int,
        nargs='+', # https://stackoverflow.com/questions/15753701/how-can-i-pass-a-list-as-a-command-line-argument-with-argparse
        required=False,
        default=5,
        help="Batch size to train model",
    )
    args = parser.parse_args()

    # importing variables from command line
    n_of_epochs = args.n_of_epochs
    # n_of_batches = args.batch_size
    n_of_batches_array = args.batch_size

    list_filename =args.names_list




    ### defining path variables
    # base_path = "/Users/azanchetta/fAIr-utilities" # this path is used in all the rest of the code, so change accordingly
    base_path = f"{os.getcwd()}"
    print(f"\n---\nCurrent working directory {base_path}")

    # check that training-datasets list file exists and is readable
    if not os.path.exists(f'{base_path}/{list_filename}'):
        raise ValueError(f"Can't find file {list_filename}")

    ### generating list of regions (cities) from the input txt file
    # input text file from command line:
    # list_filename = "cities_list.txt"

    #TO_DO: add condition of stopping if filename empty or file doesn't exist
    print(f"---\nI am going to get the names from {list_filename} (name of the list file you provided)")
    # with open("cities_list.txt", "r") as file:
    #     cities_list = "".join(file.read().split("\n"))
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

    # defining other parameters: (or obtain from config file and loop into these?)
    # n_of_epochs = 2
    # n_of_batches = 2
    print(f'\n---\nNumber of epochs {n_of_epochs} and batch size {n_of_batches_array}\n---')



    # we should change this variable to be just the home variable, and use after another variable called "city_data" or similar
    # need:
    #   - a path for input city, preprocessed images will be saved here (city_data)
    #   - a path for output model checkpoint (to be saved each time it runs), atm it's overwritten every time the training is run on a data folder (city)
    #   - a path for the training outputs, as we shall run in different conditions (batch, epochs, zoom levels, ...) for the same city, and be able to compare results
    # should have an automatic way to save the output files which shall be easy to call back
    # naming should account for the (3?) variables against which we want to evaluate the model:
    # 
    # add duration (time) for each city in the for loop
    # path_to_data = f"{base_path}/ramp-data" # for ramp data path
    path_to_data = f'{base_path}/ramp-data/metric_data' # for metric data path

    # cities_list = ["1_Zanzibar", "2_Kampala"] # will be used to loop into, initially manually inputted, can become a text file
    # with open("cities_list.txt", "r") as file:
    #     cities_list = "".join(file.read().split("\n"))
    for city in cities_list:
        # print(f"Now working on {city} preprocess")
        city_path = f"{path_to_data}/{city}" # make up string from base_path + city_name_from_list

        preprocess_output = city_path # in our case the data are already preprocessed
        train_output = f"{city_path}/train"

        path_to_acc_output = f'{base_path}/outputs'

    # ### Preprocessing
    #  Preprocessing is skipped as the images are already preprocessed.

    ### Prepare data and environment for training
        print(f"---\nPreparing data for city {city}\n---")
    ## Split training dataset to: training, validation, prediction sets
        print(f"---\nSplitting the data training into training, validation and prediction datasets:\n")
        from hot_fair_utilities.training.prepare_data import split_training_2_validation_2_prediction
        x = split_training_2_validation_2_prediction(preprocess_output, train_output)

    ## Import ramp model constructors
        from ramp.training import (
            callback_constructors,
            loss_constructors,
            metric_constructors,
            model_constructors,
            optimizer_constructors,
        )

    ## Import config file
        from hot_fair_utilities.training.run_training import manage_fine_tuning_config
        
        for batch_item in n_of_batches_array:
            n_of_batches = batch_item
            
            output_path = train_output
            epoch_size = n_of_epochs
            batch_size = n_of_batches
            freeze_layers=False
            cfg = manage_fine_tuning_config(
                        output_path, epoch_size, batch_size, freeze_layers
                    )
            # note that the above function is hard-coded on "ramp_config_base.json", change the cfg file name in there if needed

        ### Training
        
            # from hot_fair_utilities import train
            print(f"\n---\nStarting training on {city}\n---")
            
            discard_experiment = False
            if "discard_experiment" in cfg:
                discard_experiment = cfg["discard_experiment"]
            cfg["timestamp"] = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # --- Setting up metrics and loss
            the_metrics = []
            if cfg["metrics"]["use_metrics"]:
                get_metrics_fn_names = cfg["metrics"]["get_metrics_fn_names"]
                get_metrics_fn_parms = cfg["metrics"]["metrics_fn_parms"]
                assert len(get_metrics_fn_names) == len(get_metrics_fn_parms)

                # ---
                # TODO: manually changing here for OHE experiment
                # get_metrics_fn_names = ["get_precision_fn", "get_recall_fn"]
                # get_onehotiou_fn
                # get_metrics_fn_names = [ "get_precision_fn", "get_iou_fn", "get_recall_fn"]
                # get_metrics_fn_parms = [{}, {}, {}]
                # ---
                for get_mf_name, mf_parms in zip(get_metrics_fn_names, get_metrics_fn_parms):
                    
                    get_metric_fn = getattr(metric_constructors, get_mf_name)
                    print(f"Metric constructor function: {get_metric_fn.__name__}")
                    metric_fn = get_metric_fn(mf_parms)
                    the_metrics.append(metric_fn)
                print("--->Note: \n    | the first metric in the above list will be the\n    | one used as benchmark for saving the model check-points\n    | (i.e. the validation accuracy for that metric)")

            # specify a function that will construct the loss function
            get_loss_fn_name = cfg["loss"]["get_loss_fn_name"]

            # ---
            # TODO: manually changing here for OHE experiment
            # get_loss_fn_name = ["get_sparse_categorical_crossentropy_fn", "get_categorical_crossentropy_fn"]
            # get_loss_fn_name = ["get_categorical_crossentropy_fn"]
            # ---
            get_loss_fn = getattr(loss_constructors, get_loss_fn_name)
            # Construct the loss function
            loss_fn = get_loss_fn(cfg)
            print(f"Loss constructor function: {get_loss_fn.__name__}")


        # --- construct optimizer ####
            get_optimizer_fn_name = cfg["optimizer"]["get_optimizer_fn_name"]
            get_optimizer_fn = getattr(optimizer_constructors, get_optimizer_fn_name)

            optimizer = get_optimizer_fn(cfg)

            the_model = None

            # SG: Using the saved model in this cell
            working_ramp_home = os.environ["RAMP_HOME"]
            # load (construct) the model
            model_path = Path(working_ramp_home) / cfg["saved_model"]["saved_model_path"]
            print(f"Model: importing saved model {str(model_path)}")
            the_model = tf.keras.models.load_model(model_path)
            assert (
                the_model is not None
            ), f"the saved model was not constructed: {model_path}"

            if cfg["freeze_layers"]:
                for layer in the_model.layers:
                    layer.trainable = False  # freeze previous layers only update new layers
                    # print("Setting previous model layers traininable : False")


            print("-------")
            print(f'-------{the_metrics}')
            print("-------")

            # If you don't want to save the original state of training, recompile the model.
            the_model.compile(optimizer=optimizer, loss=loss_fn, metrics=[the_metrics])
            # the_model.compile(optimizer=optimizer, loss=loss_fn, metrics=[precision_class_0,precision_class_1])

            # the_model.compile(optimizer = optimizer,
            #    loss=loss_fn,
            #    metrics = [get_iou_coef_fn])
            
        # --- Introducing OHE (One Hot Encoded) for non sparse metrics
            def ohe_batches(batches: tf.data.Dataset, depth=4) -> tf.data.Dataset:
                """For given batches and depth map sparse labels to OHE."""
                return batches.map(lambda x, y: (x, tf.one_hot(y[..., -1], depth, axis=-1)))

        # --- Getting batches ready for the training
            from ramp.training.augmentation_constructors import get_augmentation_fn
            from ramp.utils.misc_ramp_utils import get_num_files
            from ramp.data_mgmt.data_generator import (
                test_batches_from_gtiff_dirs,
                training_batches_from_gtiff_dirs,
            )
            #### define data directories ####
            train_img_dir = Path(working_ramp_home) / cfg["datasets"]["train_img_dir"]
            train_mask_dir = Path(working_ramp_home) / cfg["datasets"]["train_mask_dir"]
            val_img_dir = Path(working_ramp_home) / cfg["datasets"]["val_img_dir"]
            val_mask_dir = Path(working_ramp_home) / cfg["datasets"]["val_mask_dir"]

            #### get the augmentation transform ####
            # aug = None
            if cfg["augmentation"]["use_aug"]:
                aug = get_augmentation_fn(cfg)

            ## RUNTIME Parameters
            batch_size = cfg["batch_size"]
            input_img_shape = cfg["input_img_shape"]
            output_img_shape = cfg["output_img_shape"]

            n_training = get_num_files(train_img_dir, "*.tif")
            n_val = get_num_files(val_img_dir, "*.tif")
            steps_per_epoch = n_training // batch_size
            validation_steps = n_val // batch_size
            # Testing step , not recommended
            if validation_steps <= 0:
                validation_steps = 1

            # add these back to the config
            # in case they are needed by callbacks
            cfg["runtime"] = {}
            cfg["runtime"]["n_training"] = n_training
            cfg["runtime"]["n_val"] = n_val
            cfg["runtime"]["steps_per_epoch"] = steps_per_epoch
            cfg["runtime"]["validation_steps"] = validation_steps

            train_batches = None

            if aug is not None:
                train_batches = training_batches_from_gtiff_dirs(
                    train_img_dir,
                    train_mask_dir,
                    batch_size,
                    input_img_shape,
                    output_img_shape,
                    transforms=aug,
                )
            else:
                train_batches = training_batches_from_gtiff_dirs(
                    train_img_dir, train_mask_dir, batch_size, input_img_shape, output_img_shape
                )
            assert train_batches is not None, "training batches were not constructed"
            print(f"-------\n* train img dir{train_img_dir}\n* train mask dir{train_mask_dir}")
            print(f"* input img shape{input_img_shape}\n* output img shape{output_img_shape}")


            #---
            # TODO: manually changing here for OHE experiment
            train_batches = ohe_batches(train_batches)
            #---

        # --- Getting ready also validation batches:
            # Validation batches
            val_batches = test_batches_from_gtiff_dirs(
                val_img_dir, val_mask_dir, batch_size, input_img_shape, output_img_shape
            )

            #---
            # TODO: manually changing here for OHE experiment
            val_batches = ohe_batches(val_batches)
            #---

            assert val_batches is not None, "validation batches were not constructed"
            print(f"-------\n* val img dir{val_img_dir}\n* val mask dir{val_mask_dir}\n-------")
            print(val_batches)
            print('*\n*\n')

        # --- Set up training callbacks
            callbacks_list = []

            if not discard_experiment:
                # get model checkpoint callback
                if cfg["model_checkpts"]["use_model_checkpts"]:
                    get_model_checkpt_callback_fn_name = cfg["model_checkpts"][
                        "get_model_checkpt_callback_fn_name"
                    ]
                    get_model_checkpt_callback_fn = getattr(
                        callback_constructors, get_model_checkpt_callback_fn_name
                    )
                    callbacks_list.append(get_model_checkpt_callback_fn(cfg))

                # get tensorboard callback
                if cfg["tensorboard"]["use_tb"]:
                    get_tb_callback_fn_name = cfg["tensorboard"]["get_tb_callback_fn_name"]
                    get_tb_callback_fn = getattr(callback_constructors, get_tb_callback_fn_name)
                    callbacks_list.append(get_tb_callback_fn(cfg))

                # get tensorboard model prediction logging callback
                if cfg["prediction_logging"]["use_prediction_logging"]:
                    assert cfg["tensorboard"][
                        "use_tb"
                    ], "Tensorboard logging must be turned on to enable prediction logging"
                    get_prediction_logging_fn_name = cfg["prediction_logging"][
                        "get_prediction_logging_fn_name"
                    ]
                    get_prediction_logging_fn = getattr(
                        callback_constructors, get_prediction_logging_fn_name
                    )
                    callbacks_list.append(get_prediction_logging_fn(the_model, cfg))

            # free up RAM
            tf.keras.backend.clear_session()

            if cfg["early_stopping"]["use_early_stopping"]:
                callbacks_list.append(callback_constructors.get_early_stopping_callback_fn(cfg))

                # get cyclic learning scheduler callback
            if cfg["cyclic_learning_scheduler"]["use_clr"]:
                assert not cfg["early_stopping"][
                    "use_early_stopping"
                ], "cannot use early_stopping with cycling_learning_scheduler"
                get_clr_callback_fn_name = cfg["cyclic_learning_scheduler"][
                    "get_clr_callback_fn_name"
                ]
                get_clr_callback_fn = getattr(callback_constructors, get_clr_callback_fn_name)
                callbacks_list.append(get_clr_callback_fn(cfg))

        # --- import matplotlib.pyplot as plt
            from time import perf_counter

        # --- Main training block
            n_epochs = n_of_epochs
            # # SG: manually make this 10
            # n_epochs = 5
            print(
                f"Starting Training with {n_epochs} epochs , {batch_size} batch size , {steps_per_epoch} steps per epoch , {validation_steps} validation steps......"
            )
            if validation_steps <= 0:
                raise RaiseError(
                    "Not enough data for training, Increase image or Try reducing batchsize/epochs"
                )
            # FIXME : Make checkpoint
            start = perf_counter()
            history = the_model.fit(
                train_batches,
                epochs=n_epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=val_batches,
                validation_steps=validation_steps,
                callbacks=callbacks_list,
            )
            end = perf_counter()
            print(f"Training Finished , Time taken to train : {end-start} seconds")
            print('\n-----\nHistory:')
            print(history.history.keys())
            print('\n-----')

        #### ------ Saving training metrics
            # print(f"Final accuracy: {final_accuracy} and final model path: {final_model_path}")
            # store this output somewhere!!
            accuracy_filename = f'{city}_bch{n_of_batches}_epc{n_of_epochs}.csv'
            accuracy_file_path = f'{path_to_acc_output}/accuracies/{accuracy_filename}'
            accuracy_output = f'{accuracy_file_path}'
            
            print(f'filename {accuracy_filename}')
            print(f'path to acc output {path_to_acc_output}')
            print(f'acc filepath {accuracy_file_path}')
            print(f'\n---\nAccuracy dataframe is being saved at: {accuracy_output}\n---')
            history_df = pd.DataFrame.from_dict(history.history)
            history_df.to_csv(accuracy_output)
            
        # --- Save history
            history_file_name = f'history_{city}_bch{n_of_batches}_epc{n_of_epochs}.npy'
            history_output_path = f'{path_to_acc_output}/accuracies/{history_file_name}'
            np.save(history_output_path,history.history)
            
        # --- Plot and save plot!
            # plot the training and validation accuracy and loss at each epoch
            print("Generating graphs ....")
            graph_file_name = f'graph_{city}_bch{n_of_batches}_epc{n_of_epochs}.png'
            graph_output = f'{path_to_acc_output}/accuracies/{graph_file_name}'

            loss = history.history["loss"]
            val_loss = history_df.loc[:,"val_loss"]
            epochs = list(range(1, len(loss) + 1))

            #  --- Seaborn version ---
            # ---
            # dataframe with only loss - to be able to map it later
            loss_df = pd.DataFrame(data={'epoch': epochs, 'loss': loss, 'val_loss': val_loss})
            # melt the dataframe
            dfm_loss = loss_df.melt('epoch', var_name='col_names', value_name='vals')
            # generate column for type train/valid:
            dfm_loss['type'] = np.where(dfm_loss.col_names.str.contains("val"), "valid", "train")
            # generate column with fancier name of the metric, for plotting
            dfm_loss['metric'] = np.where(dfm_loss.type.str.contains("train"), "Loss", "")
            #  ---
            # convert to long (tidy) form:
            history_df['epoch'] = range(1, len(history_df) + 1) # create column "epoch"
            history_df = history_df.drop('loss', axis = 1) # get rid of column 'loss' to be able to plot it separately
            history_df = history_df.drop('val_loss', axis = 1) # get rid of column 'loss' to be able to plot it separately
            # print(history_df.info())
            # print(history_df.iloc[0:5,:])
            # melt the dataframe
            dfm = history_df.melt('epoch', var_name='col_names', value_name='vals')
            # generate column for type train/valid:
            dfm['type'] = np.where(dfm.col_names.str.contains("val"), "valid", "train") # "valid" where 'val', otherwise "train"
            # generate column with fancier name of the metric, for plotting
            dfm['metric'] = np.where(dfm.col_names.str.contains("precision"), "Precision",
                            np.where(dfm.col_names.str.contains("recall"), "Recall",
                            np.where(dfm.col_names.str.contains("iou"), "IoU",
                            np.where(dfm.col_names.str.contains("categorical"), "Accuracy", 
                            np.where(dfm.col_names.str.contains("f1"), "F1 score",
                            "")))))
            #  ---

            # ---
            palette_div=sns.color_palette("Dark2", 10)
            sns.set_palette(palette_div)
            sns.set_style("whitegrid")
            # sns_plot = sns.lineplot(data=dfm,
            #             x="epoch",
            #             y="vals",
            #             hue='metric',
            #             palette=palette_div,
            #             style='type')
            # sns_plot.set(xlabel='Epochs',
            # ylabel='Accuracy',
            # title='Training/validation accuracies')
            
            # sns_plot.set_ylim(bottom=0, top=1) # this is to avoid Loss values (>1) to alter the graph limits
            # sns_plot.xaxis.set_major_locator(ticker.MultipleLocator(2)) # adding ticks at multiples of 2
            # sns_plot.xaxis.set_major_formatter(ticker.ScalarFormatter())
            # sns_plot.get_figure().savefig(graph_output)
            
            # sns_plot.get_figure().savefig(graph_output)
            # sns_plot.get_figure().show()

            # print(f"Graph generated at : {graph_output}")
            
            # # # clearing up the figure for next plot
            # sns_plot.get_figure().clf()
            fig, ax1 = plt.subplots()
            sns.lineplot(data=dfm_loss,
                        x="epoch",
                        y="vals",
                        style='type',
                        color = 'gray',
                        legend=False)
            ax1.set(xlabel='Epochs',
                    ylabel='Accuracy',
                    title='Training/validation accuracies')
            ax1.xaxis.set_major_locator(ticker.MultipleLocator(1)) # adding ticks at multiples of 2
            ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())
            # ax1.tick_params(axis='y')
            ax1.set_ylabel('Loss', color='tab:grey')
            ax1.grid(True, linestyle=':')
            #  introducing second axis label
            ax2 = ax1.twinx()
            sns.lineplot(data=dfm,
                        x="epoch",
                        y="vals",
                        hue='metric',
                        palette=palette_div,
                        style='type')
            # ax2.tick_params(axis='y')
            ax2.set_ylabel('Accuracy')
            ax2.set_ylim(bottom=0, top=1) # this is to avoid Loss values to alter the graph limits, but doesn't look good
            # 
            plt.savefig(
                f"{graph_output}"
            )
            print(f"Graph generated at : {graph_output}")
            plt.show()
            # # clearing up the figure for next plot to avoid overlapping figures! https://stackoverflow.com/questions/17106288/
            plt.clf()
            plt.cla()
            plt.close()


    # ---
    # TODO: add a call with `extract_highest_accuracy_model` to do cleanup
    # extract_highest_accuracy_model(model_path)
        final_accuracy, final_model_path = extract_highest_accuracy_model(output_path)
        print(f'\n-----\nFinal accuracy: {final_accuracy}')
        print(f'\n-----\nFinal model path: {final_model_path}')
        print('\n-----')
    # ---



    # ### Prediction
    #  will be run in a second moment / separate script ?
    # # 
    # # from hot_fair_utilities import predict
    #     print(f"\n---\n---\nStarting prediction on {city}\n")
    #     # prediction_output = f"{path_to_output}/{city}/prediction"   # !!! change file name here
    #     prediction_output = "" #### NAME PATH HERE!!!!!!!!!!!!!!!!!!
    #     predict(
    #         checkpoint_path=final_model_path,
    #         input_path=f"{city_path}/prediction/input", # the same of above?
    #         prediction_path=prediction_output,
    #     )

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