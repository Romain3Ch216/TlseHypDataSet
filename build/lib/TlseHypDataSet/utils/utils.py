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


def labels_to_labels(labels, keys_to_values):
    """
    Converts labels to other labels given a label-to-label dictionary
    :param labels: An array of labels encoded by integers
    :param keys_to_values: A dict of labels to labels relation
    :return: An array of labels
    """
    new_labels = np.vectorize(keys_to_values.get)(labels)
    return new_labels
