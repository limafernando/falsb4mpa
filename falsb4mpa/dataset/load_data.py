import numpy as np
import os
from pathlib import Path

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


VALID_DATA_NAMES = [
    "adult-mpa-bin-agg",
    "adult-mpa-bin-wout-agg",
    "adult-mpa-cat-wout-agg",
    "german-mpa-bin-wout-agg",
    "compas-mpa-bin-wout-agg",
    "compas-mpa-cat-wout-agg",
]
VALID_FILE_NAMES = {
    "adult-mpa-bin-agg": "adult_mpa_bin_with_agg",
    "adult-mpa-bin-wout-agg": "adult_mpa_bin_wout_agg",
    "adult-mpa-cat-wout-agg": "adult_mpa_cat_wout_agg",
    "german-mpa-bin-wout-agg": "german_mpa_bin_wout_agg",
    "compas-mpa-bin-wout-agg": "compas_mpa_bin_wout_agg",
    "compas-mpa-cat-wout-agg": "compas_mpa_cat_wout_agg",
}
VALID_FOLDER_NAMES = {
    "adult-mpa-bin-agg": "adult",
    "adult-mpa-bin-wout-agg": "adult",
    "adult-mpa-cat-wout-agg": "adult",
    "german-mpa-bin-wout-agg": "german",
    "compas-mpa-bin-wout-agg": "compas",
    "compas-mpa-cat-wout-agg": "compas",
}
VALID_LEARNING_STEPS = ["train", "valid", "test"]
ACCESS_INDEXES = {
    # dataset-name: [X, Y, A1, A2] - A como subconjunto de X
    "adult-mpa-bin-agg": [slice(-1), -1, 1],  # Em casos com agg tem apenas um A
    "adult-mpa-bin-wout-agg": [slice(-1), -1, 1, 2], # A1 gender A2 race
    "adult-mpa-cat-wout-agg": [slice(-1), -1, 1, slice(2, 7)], # A1 gender A2 race
    "german-mpa-bin-wout-agg": [slice(-1), -1, 0, 1],  # A1 gender A2 age
    "compas-mpa-bin-wout-agg": [slice(-1), -1, 0, 1],  # A1 gender A2 race
    "compas-mpa-cat-wout-agg": [slice(-1), -1, 0, slice(1, 7)],  # A1 gender A2 race
}
DIMENSIONS = {
    # dataset-name: [X, Y, A1, A2]
    "adult-mpa-bin-agg": [116, 1, 1],
    "adult-mpa-bin-wout-agg": [97, 1, 1, 1],
    "adult-mpa-cat-wout-agg": [102, 1, 1, 5],
    "german-mpa-bin-wout-agg": [28, 1, 1, 1],
    "compas-mpa-bin-wout-agg": [11, 1, 1, 1],
    "compas-mpa-cat-wout-agg": [18, 1, 1, 6],
}


def load_data(data_name, learning_step=None, kind="np"):
    """Function to load data.

    Args:
        data_name (str): used to select the correct data file.
        learning_step (str): used to select the correct data for the learning step.

    Returns:
        [type]: [description]
    """
    if not data_name in VALID_DATA_NAMES:
        print("Invalid data name! Input: {} | Valid data names: [{}]", format(VALID_DATA_NAMES))
        return None

    if learning_step is None:
        learning_step = VALID_FILE_NAMES[data_name]

    elif not learning_step in VALID_LEARNING_STEPS:
        print("Invalid data name! Input: {} | Valid steps: [{}]", format(VALID_LEARNING_STEPS))
        return None

    data_folder = select_data_folder(data_name)
    access_indexes = get_access_indexes(data_name)

    if kind == "np":
        # x, y, a = select_data_step_np(learning_step, access_indexes, data_folder, data_name)
        return select_data_step_np(learning_step, access_indexes, data_folder, data_name)
    elif kind == "pd":
        # x, y, a = select_data_step_pd(learning_step, access_indexes, data_folder, data_name)
        return select_data_step_pd(learning_step, access_indexes, data_folder, data_name)


def select_data_folder(data_name):
    return os.path.join(
        ROOT_DIR, Path(f"../../data/processed/{VALID_FOLDER_NAMES[data_name]}".format())
    )


def get_access_indexes(data_name):
    return ACCESS_INDEXES[data_name]


def select_data_step_np(learning_step, access_indexes, data_folder, data_name):
    file = os.path.join(data_folder, Path(f"{learning_step}.csv"))
    data = np.genfromtxt(file, delimiter=",", skip_header=True)[:, 1:]

    num_examples = data.shape[0]
    x = data[:, access_indexes[0]]
    y = data[:, access_indexes[1]].reshape(num_examples, DIMENSIONS[data_name][1])
    a1 = data[:, access_indexes[2]].reshape(num_examples, DIMENSIONS[data_name][2])
    a2 = data[:, access_indexes[3]].reshape(num_examples, DIMENSIONS[data_name][3])

    return x, y, a1, a2


def select_data_step_pd(learning_step, access_indexes, data_folder, data_name):
    file = os.path.join(data_folder, Path(f"{learning_step}.csv"))

    data = pd.read_csv(file)

    return data
