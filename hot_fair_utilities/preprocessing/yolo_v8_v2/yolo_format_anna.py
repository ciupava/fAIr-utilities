# Standard library imports
import glob
import os

# Third party imports
import numpy as np
import yaml
from tqdm import tqdm
import shutil
from .utils import convert_tif_to_jpg, write_yolo_file


def yolo_format(
    input_path="preprocessed/*",
    output_path="ramp_data_yolo",
    city=""
    # seed=42,
    # train_split=0.7,
    # val_split=0.15,
    # test_split=0.15,
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


# Add part to obtain dataset split from the existing files structure

    # Call the find_files function and print the number of found files
    cwps_train, lwps_train = find_files(city_folder_name, "train")
    cwps_val, lwps_val = find_files(city_folder_name, "val")
    cwps_pred, lwps_pred = find_files(city_folder_name, "pred")

    print('Found {} chip files'.format(len(cwps_train)+len(cwps_val)+len(cwps_pred)))
    # N = 2
    # test_list_cwps = cwps[:N]
    # print(f"The first {N} elements of list are : {str(test_list_cwps)}")
    print('Found {} label files\n'.format(len(lwps_train)+len(lwps_val)+len(lwps_pred)))
    # test_list_lwps = lwps[:N]
    # print(f"The first {N} elements of list are : {str(test_list_lwps)}")
    # Print message if all filenames match
    # print('All filenames match; each tif has a label!')

    # Kshitij is skipping this part below, so do I
    # # Call the shapes_data and print the shape of the first chip file
    # shapes_data_train = check_shapes(cwps_train)
    # print(f'Chip shapes with counts are: {shapes_data_train[0]}')
    # shapes_data_val = check_shapes(cwps_val)
    # print(f'Chip shapes with counts are: {shapes_data_val[0]}')
    # shapes_data_pred = check_shapes(cwps_pred)
    # print(f'Chip shapes with counts are: {shapes_data_pred[0]}')

    # Bringing here the relevant parts from next cell, as we need to keep iterating through the city folders
    train_cwps = cwps_train
    val_cwps = cwps_val
    test_cwps = cwps_pred

    # Output the results to verify
    print(f'\nTrain array size: {len(train_cwps)}')
    print(f'Validation array size: {len(val_cwps)}')
    print(f'Test array size: {len(test_cwps)}\n')

    # Check if the YOLO folder exists, if not create labels, images, and folders
    folder_name_for_yolodata_city=f'ramp_data_yolo/{city}'
    yolodata_city_withpath=os.path.join('/content/drive/MyDrive/YOLO_test/data',folder_name_for_yolodata_city)
    # folder_name_for_yolodata='ramp_data_yolo'
    print(f'Name for new Yolo folder that will be created: {yolodata_city_withpath}')
    if not os.path.exists(yolodata_city_withpath):
        print('Creating YOLO folders...')
        # Create the folder
        # os.makedirs(folder_name_for_yolodata_city)

        # Write the YOLO label files for the training set
        print('Generating training labels')
        for train_cwp in tqdm(train_cwps):
            write_yolo_file(train_cwp, city_folder=city, folder_type='train', output_path=output_path)

        # Write the YOLO label files for the validation set
        print('Generating validation labels')
        for val_cwp in tqdm(val_cwps):
            write_yolo_file(val_cwp, city_folder=city, folder_type='val', output_path=output_path)

        # Write the YOLO label files for the test set
        print('Generating test labels')
        for test_cwp in tqdm(test_cwps):
            write_yolo_file(test_cwp, city_folder=city, folder_type='test', output_path=output_path)

        # Convert the chip files to JPEG format
        print('Generating training images')
        for train_cwp in tqdm(train_cwps):
            convert_tif_to_jpg(train_cwp, city_folder=city, folder_type='train', output_path=output_path)

        print('Generating validation images')
        for val_cwp in tqdm(val_cwps):
            convert_tif_to_jpg(val_cwp, city_folder=city, folder_type='val', output_path=output_path)

        print('Generating test images')
        for test_cwp in tqdm(test_cwps):
            convert_tif_to_jpg(test_cwp, city_folder=city, folder_type='test', output_path=output_path)

    else:
        print('Data already converted')

    print(f'\n---\nFinished working on {city}\n---\n\n_____________________________________\n')


def find_files():
    """
    Find chip (.tif) and label (.geojson) files in the specified folders.

    Returns:
    cwps (list): List of chip filenames with path.
    lwps (list): List of label filenames with path.
    base_folders (list): List of base folder names.
    """

    # Find the folders
    data_folders = glob.glob(DATA_FOLDERS)

    # Create a list to store chip (.tif), mask (.mask.tif), and label (.geojson) filenames with path
    cwps = []
    lwps = []

    # Create a list to store the base folder names
    base_folders = []

    for folder in data_folders:
        # Pattern to match all .tif files in the current folder, including subdirectories
        tif_pattern = f"{folder}/**/*.tif"

        # Find all .tif files in the current 'training*' folder and its subdirectories
        found_tif_files = glob.glob(tif_pattern, recursive=True)

        # Filter out .mask.tif files and add the rest to the tif_files list
        for file in found_tif_files:
            if file.endswith('.tif'):
                cwps.append(file)

        # Pattern to match all .geojson files in the current folder, including subdirectories
        geojson_pattern = f"{folder}/**/*.geojson"

        # Find all .geojson files
        found_geojson_files = glob.glob(geojson_pattern, recursive=True)

        # Add found .geojson files to the geojson_files list
        lwps.extend(found_geojson_files)

    # Sort the lists
    cwps.sort()
    lwps.sort()

    # Assert that the the number files for each type are the same
    assert len(cwps) == len(lwps), "Number of tif files and label files do not match"

    # Function to check that the filenames match
    for n, cwp in enumerate(cwps):
        c = os.path.basename(cwp).replace('.tif', '')
        l = os.path.basename(lwps[n]).replace('.geojson', '')

        assert c == l, f"Chip and label filenames do not match: {c} != {l}"

        base_folders.append(cwp.split('/')[1])

    return cwps, lwps
