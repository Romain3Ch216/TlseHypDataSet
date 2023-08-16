import os
import subprocess
import numpy as np


def make_dirs(folders):
    for folder in folders:
        try:
            os.makedirs(folder, exist_ok=True)
        except OSError as exc:
            if exc.errno != os.errno.EEXIST:
                raise
            pass


def data_in_folder(files, folder):
    out = True
    for file in files:
        if file in os.listdir(folder):
            continue
        else:
            return False
    return out


def tile_raster(input_file):
    out_file = input_file[:-3] + 'tif'
    query = "gdal_translate -co TILED=YES -co BLOCKXSIZE=256 -co BLOCKYSIZE=256 " + input_file + " " + out_file
    subprocess.call(query, shell=True)

def convert_lc_map_to_ip_map(pred, permeability):
    pred = np.array(pred).astype(int)
    for class_id in np.unique(pred):
        pred[pred == class_id] = permeability[class_id]
    return pred