# Standard library imports
import glob
import os

# Third party imports
import numpy as np
import yaml
from tqdm import tqdm
import shutil
from .utils import convert_tif_to_jpg, write_yolo_file


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
        print(f'folder is {folder}')
        # Pattern to match all .tif files in the current folder, including subdirectories
        tif_pattern = f"{folder}/**/**/**/*.tif"
        print(f'tif pattern is {tif_pattern}')
        print(len(tif_pattern))
        # Find all .tif files in the current 'training*' folder and its subdirectories
        found_tif_files = glob.glob(tif_pattern, recursive=True)
        print(f'found tif files {found_tif_files}')
        print(len(found_tif_files))
        # Filter out .mask.tif files and add the rest to the tif_files list
        for file in found_tif_files:
            if not file.endswith('mask.tif'):
                cwps.append(file)

        # Pattern to match all .geojson files in the current folder, including subdirectories
        geojson_pattern = f"{folder}/**/**/**/*.geojson"
        print(f'geojson pattern is {geojson_pattern}')
        print(len(geojson_pattern))
        # Find all .geojson files
        found_geojson_files = glob.glob(geojson_pattern, recursive=True)
        print(f'found gjson files {found_geojson_files}')
        print(len(found_geojson_files))
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

    return cwps, lwps, base_folders

# Call the function and print the number of found files
cwps, lwps, base_folders = find_files()
print('Found {} chip files'.format(len(cwps)))
print('Found {} label files\n'.format(len(lwps)))

# Print message if all filenames match
print('All filenames match; each tif has a label!')

