# Third party imports
import numpy as np
from PIL import Image
from tqdm import tqdm

from .building_footprint import BuildingExtract
from .utils import tiles_from_directory, zoom_from_directory

# def get_polygons(
#     pred_masks_path, polygons_path, kernel_opening=20, simplify_threshold=0.01
# ):
#     """Generate GeoJSON polygons from predicted masks.

#     Args:
#       pred_masks_path: directory where the predicted mask are saved
#       polygons_path: path to GeoJSON file to store features in
#       kernel_opening: the opening morphological operation's kernel size in pixel
#       simplify_threshold: the simplification accuracy as max. percentage of the arc length, in [0, 1]

#     """
#     bldg_extract = BuildingExtract(kernel_opening, simplify_threshold)
#     tiles = list(tiles_from_directory(pred_masks_path))

#     for tile, path in tqdm(tiles, unit="mask"):
#         mask = np.array(Image.open(path).convert("P"), dtype=np.uint8)
#         bldg_extract.extract(tile, mask)
#     print(f'polygons path {polygons_path} \n ----')
#     bldg_extract.save(polygons_path)

def get_polygons(
    pred_masks_path, polygons_path, zoom_level, kernel_opening=20, simplify_threshold=0.01
):
    """Generate GeoJSON polygons from predicted masks.

    Args:
      pred_masks_path: directory where the predicted mask are saved
      polygons_path: path to GeoJSON file to store features in
      kernel_opening: the opening morphological operation's kernel size in pixel
      simplify_threshold: the simplification accuracy as max. percentage of the arc length, in [0, 1]

    """
    # bldg_extract = BuildingExtract(kernel_opening, simplify_threshold)
    # tiles = list(tiles_from_directory(pred_masks_path))
    # # print(f"\n\ntiles are !!!!!!!!!!!!!! {tiles}")
    # for tile, path in tqdm(tiles, unit="mask"):
    #     zoom_from_tile = tile.z
    #     # zoom_from_tile = zoom_from_directory(pred_masks_path)
    #     print(f"zoom from tile is {zoom_from_tile}")
    #     if zoom_from_tile==zoom_level:
    #       mask = np.array(Image.open(path).convert("P"), dtype=np.uint8)
    #       bldg_extract.extract(tile, mask)
    #     # polygons_path = f'temp_labels_{zoom}.geojson'
    #       print("extracted yeah!!!!!!!!!!")
    #       print(f'polygons path {polygons_path} \n ----')
    #       bldg_extract.save(polygons_path)
    bldg_extract = BuildingExtract(kernel_opening, simplify_threshold)
    tiles = list(tiles_from_directory(pred_masks_path))

    print(f"zoom level from input is {zoom_level}")
    for tile, path in tqdm(tiles, unit="mask"):
        zoom_from_tile = tile.z
        print(f"zoom from tile is {zoom_from_tile}")
        if zoom_from_tile==zoom_level:
          mask = np.array(Image.open(path).convert("P"), dtype=np.uint8)
          bldg_extract.extract(tile, mask)
    print(f'polygons path {polygons_path} \n ----')
    bldg_extract.save(polygons_path)

def get_zoom_level(
        pred_masks_path
):
    """Generate GeoJSON polygons from predicted masks.

    Args:
      pred_masks_path: directory where the predicted mask are saved
      polygons_path: path to GeoJSON file to store features in
      kernel_opening: the opening morphological operation's kernel size in pixel
      simplify_threshold: the simplification accuracy as max. percentage of the arc length, in [0, 1]

    """
    zooom = zoom_from_directory(pred_masks_path)
    print(f"\n\n Zoom is {zooom}")
    return zooom
