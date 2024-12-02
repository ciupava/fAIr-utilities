# Standard library imports
import glob
import os
import csv

# Third party imports
import numpy as np
import yaml
from tqdm import tqdm
import shutil
from .utils_anna import convert_tif_to_jpg, write_yolo_file


def yolo_format(
    # input_path="preprocessed/*",
    input_path="",
    csv_path="",
    output_path='',
    city_name=''
):
    """
    Preprocess data for YOLO model training.

    Args:
        input_path (str, optional): The path to the input data folders. Defaults to "preprocessed/*".
        output_path (str, optional): The path to the output YOLO data folders. Defaults to "ramp_data_yolo".
        seed (int, optional): The random seed for data splitting. Defaults to 42.
        train_split (float, optional): The percentage of data to be used for training. Defaults to 0.7.
        val_split (float, optional): The percentage of data to be used for validation. Defaults to 0.15.
        test_split (float, optional): The percentage of data to be used for testing. Defaults to 0.15.

    Returns:
        None
    """

    preprocessed_input = input_path

# Add part to obtain dataset split from the existing files structure

    # Call the find_files function and print the number of found files
    # print(f'---\nTRAIN!!! ----------------\n')
    cwps_train, lwps_train = find_files(preprocessed_input, csv_path, "train")
    # print(f'---\nVAL!!! ----------------\n')
    cwps_val, lwps_val = find_files(preprocessed_input, csv_path, "val")
    # print(f'---\nPRED!!! ----------------\n')
    cwps_pred, lwps_pred = find_files(preprocessed_input, csv_path, "pred")

    print('Found {} chip files'.format(len(cwps_train)+len(cwps_val)+len(cwps_pred)))
    # N = 2
    # test_list_cwps = cwps[:N]
    # print(f"The first {N} elements of list are : {str(test_list_cwps)}")
    print('Found {} label files\n'.format(len(lwps_train)+len(lwps_val)+len(lwps_pred)))
    # test_list_lwps = lwps[:N]
    # print(f"The first {N} elements of list are : {str(test_list_lwps)}")
    # Print message if all filenames match
    # print('All filenames match; each tif has a label!')
    # print(f'\n---\n---cwps is this for train {cwps_train}, \nval {cwps_val}, \npred {cwps_pred}\n')
    # print(f'\n---cwps is this for train {cwps_train}, \nval {cwps_val}, \npred {cwps_pred}\n---')
    # Bringing here the relevant parts from next cell, as we need to keep iterating through the city folders
    train_cwps = cwps_train
    val_cwps = cwps_val
    test_cwps = cwps_pred

    # Output the results to verify
    print(f'\nTrain array size: {len(train_cwps)}')
    print(f'Validation array size: {len(val_cwps)}')
    print(f'Test array size: {len(test_cwps)}\n')

    # Check if the YOLO folder exists, if not create labels, images, and folders
    yolodata_city_withpath=f'{output_path}/{city_name}'
    
    print(f'Checking if Yolo folder data exists for this city...')
    # if not os.path.exists(yolodata_city_withpath):
    #     print('Creating YOLO folders...')
    #     print(f'Name for new Yolo folder that will be created: {yolodata_city_withpath}')
    #     # Create the folder
    #     os.makedirs(yolodata_city_withpath)
    # else:
    #     print('... folder already created')
    if  os.path.exists(yolodata_city_withpath):
        shutil.rmtree(yolodata_city_withpath)

    os.makedirs(yolodata_city_withpath)

    # Write the YOLO label files for the training set
    print('Generating training labels')
    for train_cwp in tqdm(train_cwps):
        # print(f'current train cwp id {train_cwp}')
        write_yolo_file(train_cwp, "train", yolodata_city_withpath)

    # Write the YOLO label files for the validation set
    print('Generating validation labels')
    for val_cwp in tqdm(val_cwps):
        # print(f'current val cwp is {val_cwp}')
        write_yolo_file(val_cwp, "val", yolodata_city_withpath)

    # Write the YOLO label files for the test set
    print('Generating test labels')
    for test_cwp in tqdm(test_cwps):
        write_yolo_file(test_cwp, "test", yolodata_city_withpath)

    # Convert the chip files to JPEG format
    print('Generating training images')
    for train_cwp in tqdm(train_cwps):
        convert_tif_to_jpg(train_cwp, "train", yolodata_city_withpath)

    print('Generating validation images')
    for val_cwp in tqdm(val_cwps):
        convert_tif_to_jpg(val_cwp, "val", yolodata_city_withpath)

    print('Generating test images')
    for test_cwp in tqdm(test_cwps):
        convert_tif_to_jpg(test_cwp, "test", yolodata_city_withpath)



def find_files(data_folders, csv_folder, train_type):
    """
    Find chip (.tif) and label (.geojson) files in the specified folders.

    Returns:
    cwps (list): List of chip filenames with path.
    lwps (list): List of label filenames with path.
    base_folders (list): List of base folder names.
    """

    # Find the folders
    # data_dirs = glob.glob(DATA_FOLDERS)
    # data_dirs = glob.glob(data_folders)
    data_dirs=data_folders
    print(f'\n---\ndata_dirs are {data_dirs}')
    # Create a list to store chip (.tif), mask (.mask.tif), and label (.geojson) filenames with path
    cwps = []
    lwps = []

    csv_file_name = f'fair_split_{train_type}.csv'
    csv_file_path = f'{csv_folder}/{csv_file_name}'
    # print(f'CSV file is {csv_file_name}')
    print(f'CSV path is {csv_file_path}')
    csv_raw_list = []

    with open(csv_file_path, "r") as file_obj:
        # heading = next(file_obj)
        reader_obj = csv.reader(file_obj, delimiter="\t")
        for row in reader_obj:
            # print(f'name of supposed item in csv list: {row}')
            csv_raw_list.append(row)
    print(f'this is the list from the csv file:\n{csv_raw_list}')

    csv_nested_list = []
    csv_nested_list_geojson = []
    for ccc in csv_raw_list:
    #   print(f'\n_______________\nraw list is {ccc}')
      nested = ccc[0]
    #   print(f'nested {nested}\n---')
      name_csv = nested.split('/')[-1]
      name_geojson = name_csv.replace(".tif", ".geojson")
    #   print(f'... and name csv {name_csv} and name geojson {name_geojson}\n_______________\n')
      csv_nested_list.append(name_csv)
      csv_nested_list_geojson.append(name_geojson)
    # print(f'nested list is {csv_nested_list}\nlength of csv files list is {len(csv_nested_list)}\n_______________\n')
    print(f'\nlength of csv files list is {len(csv_nested_list)}\n_______________\n')


    # print(f'________\nDATA DIRS is {data_dirs}\n________')
    
    # for folder in data_dirs:
    # print(f'\n---folder is {folder}')
    # Pattern to match all .tif files in the current folder, including subdirectories
    # tif_pattern = f"{folder}/**/*.tif"
    tif_pattern = f"{data_dirs}/**/*.tif"
    # print(f'\n--- tif pattern is {tif_pattern}')
    # Find all .tif files in the current 'training*' folder and its subdirectories
    found_tif_files = glob.glob(tif_pattern, recursive=True)
    # print(f'\n---found tif files {found_tif_files}')
    print(len(found_tif_files))
    # Filter out .mask.tif files and add the rest to the tif_files list
    # ""
    for file in found_tif_files:
        if not file.endswith('mask.tif'):
            file_name=file.split("/")[-1]
            # print(f'tif file name is {file_name}')
            if file_name in csv_nested_list:
                # print(f'\n it should have this tif name: {file_name}\n---')
                cwps.append(file)
            # cwps.append(file)
    print(f'\nLength of cwps is {len(cwps)}\nThey are these {cwps}')

    # Pattern to match all .geojson files in the current folder, including subdirectories
    # geojson_pattern = f"{folder}/**/*.geojson"
    geojson_pattern = f"{data_dirs}/**/*.geojson"
    # print(f'\n---geojson pattern is {geojson_pattern}')
    # Find all .geojson files
    found_geojson_files = glob.glob(geojson_pattern, recursive=True)
    # print(f'\n---found geojson files {found_geojson_files}')
    print(f'len of geojson files is {len(found_geojson_files)}')
    # Add found .geojson files to the geojson_files list
    for file in found_geojson_files:
        file_name=file.split("/")[-1]
        # print(f'geojson file name is {file_name}')
        if file_name in csv_nested_list_geojson:
            # print(f'\n it should have this geojson name: {file_name}\n---')
            lwps.append(file)

    print(f'\nLength of lwps is {len(lwps)}\nThey are these {lwps}')
    
    
    # for folder in data_dirs:
    #     # Pattern to match all .tif files in the current folder, including subdirectories
    #     tif_pattern = f"{folder}/**/*.tif"
    #     print(f'tif pattern is {tif_pattern}')

    #     # Find all .tif files in the current 'training*' folder and its subdirectories
    #     found_tif_files = glob.glob(tif_pattern, recursive=True)

    #     # Filter out .mask.tif files and add the rest to the tif_files list
    #     for file in found_tif_files:
    #         if file.endswith('.tif'):
    #             print (f'file is: {file}')
    #             cwps.append(file)

    #     # Pattern to match all .geojson files in the current folder, including subdirectories
    #     geojson_pattern = f"{folder}/**/*.geojson"

    #     # Find all .geojson files
    #     found_geojson_files = glob.glob(geojson_pattern, recursive=True)

    #     # Add found .geojson files to the geojson_files list
    #     lwps.extend(found_geojson_files)

    # Sort the lists
    cwps.sort()
    lwps.sort()

    # Assert that the the number files for each type are the same
    assert len(cwps) == len(lwps), "Number of tif files and label files do not match"

    # # Function to check that the filenames match
    # for n, cwp in enumerate(cwps):
    #     c = os.path.basename(cwp).replace('.tif', '')
    #     l = os.path.basename(lwps[n]).replace('.geojson', '')

    #     assert c == l, f"Chip and label filenames do not match: {c} != {l}"

    #     base_folders.append(cwp.split('/')[1])

    return cwps, lwps
