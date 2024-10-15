# Standard library imports
import os
from pathlib import Path

from .get_polygons import get_polygons, get_zoom_level
from .merge_polygons import merge_polygons


def polygonize(input_path: str, temp_path: str, output_path: str, zoom_levels, remove_inputs: False) -> None:
    """Polygonize raster tiles from the input path using AutoBFE's algorithm.

    There are two steps:
    1. It polygonizes each of the individual tiles first.
    2. Using GDAL buffering, it merges all the nearby polygons
    so that the split polygons get merged into one.

    Whether the images are georeferenced or not doesn't really matter.
    Even PNG images will do.

    CRS of the resulting GeoJSON file will be EPSG:4326.

    Args:
        input_path: Path of the directory where the image files are stored.
        output_path: Path of the output file.
        remove_inputs: Clears the input image after geojson is produced

    Example::

        polygonize("data/masks_v2/4", "labels.geojson")
    """
    if os.path.exists(output_path):
        os.remove(output_path)
    base_path = Path(output_path).parents[0]
    base_path.mkdir(exist_ok=True, parents=True)
    
    # print(f"input path is {input_path}")
    # zoom_level = get_zoom_level(input_path)
    # unique_zoom_levels = list(set(zoom_levels))
    print(f"___Zoom level is {zoom_levels}")
    # for z in unique_zoom_levels:
    #     print(z)
    get_polygons(input_path, temp_path, zoom_levels, kernel_opening=1)
    # get_polygons(input_path, "temp-labels.geojson", unique_zoom_levels, kernel_opening=1)
    print(f"---\nOutput path is: {output_path}")

    # json_name = "temp-labels.geojson"
    merge_polygons(temp_path, output_path, distance_threshold=0.6)
    os.remove(temp_path)
    print(f"removing {temp_path} here !!!\n---\n")

    if remove_inputs:
        # Get a list of all .tif files in the directory
        tif_files = [f for f in os.listdir(input_path) if f.endswith(".tif")]

        # Loop through the list of .tif files and delete each one
        for file in tif_files:
            file_path = os.path.join(input_path, file)
            os.remove(file_path)
