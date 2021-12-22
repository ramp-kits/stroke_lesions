import os
import argparse
from os.path import join
import pathlib
import shutil
import json
from stroke import stroke_config


def bidsify_indi_atlas(atlas_path: str, destination_path: str = "data"):
    """
    Converts the ATLAS dataset distributed by INDI to BIDS.
    Parameters
    ----------
    atlas_path : str
        Path of the "ATLAS_2" directory.
    destination_path : str
        Path for where to store the data. Recommended: data/ relative to the current directory.

    Returns
    -------
    None
    """
    # The relevant data is in the Training directory; the workflow is not set up to use either .csv or
    # data without labels (the Testing directory)
    training_source = join(atlas_path, "Training")

    # Create destination if needed
    dest = pathlib.Path(destination_path)
    training_dest = pathlib.Path(dest).joinpath("train")
    derivatives_dest = training_dest.joinpath("derivatives", stroke_config.training["data_derivatives_names"][0])

    testing_dest = pathlib.Path(dest).joinpath("test")
    derivatives_test_dest = testing_dest.joinpath("derivatives", stroke_config.testing["data_derivatives_names"][0])
    # Get test subjects list
    f = open("data_test_list.txt", "r")
    test_subjects = set(f.read().splitlines())
    f.close()

    if not derivatives_dest.exists():
        derivatives_dest.mkdir(parents=True, exist_ok=True)
    if not derivatives_test_dest.exists():
        derivatives_test_dest.mkdir(parents=True, exist_ok=True)

    # Data is in ATLAS_2/Training/Rxxx/
    # Move out of Rxxx; dataset_description.json is the same across all subjects, so we can just ignore it.
    # If we're on the same filesystem, we can just move the files.
    dev_source = os.stat(atlas_path).st_dev
    dev_dest = os.stat(destination_path).st_dev
    same_fs = dev_source == dev_dest

    if same_fs:
        move_file = os.rename
        move_dir = os.rename
    else:
        move_file = shutil.copy2
        move_dir = shutil.copytree

    # Move files over!
    dataset_description_path = ""
    for r_dir in os.listdir(training_source):
        if r_dir.startswith("."):
            continue  # There are hidden files spread out; we don't need them.
        leading_path = join(training_source, r_dir)
        for sub in os.listdir(leading_path):
            if sub.startswith("."):
                continue  # As above
            path_to_move = join(leading_path, sub)
            if sub in test_subjects:
                destination = join(derivatives_test_dest, sub)
            else:
                destination = join(derivatives_dest, sub)
            if sub == "dataset_description.json":
                dataset_description_path = destination
            if pathlib.Path(path_to_move).is_dir():
                move_dir(path_to_move, destination)
            else:
                move_file(path_to_move, destination)

    # Copy dataset_description.json in test set
    shutil.copy2(dataset_description_path, derivatives_test_dest.joinpath("dataset_description.json"))

    # Write dataset_description.json to top-level training dir
    dataset_desc = {"Name": "ATLAS", "BIDSVersion": "1.6.0", "Authors": ["NPNL"]}
    dataset_desc_path = training_dest.joinpath("dataset_description.json")
    f = open(dataset_desc_path, "w")
    json.dump(dataset_desc, f, separators=(",\n", ":\t"))
    f.close()

    dataset_desc_test_path = testing_dest.joinpath("dataset_description.json")
    f = open(dataset_desc_test_path, "w")
    json.dump(dataset_desc, f, separators=(",\n", ":\t"))
    f.close()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--atlas", help='Path of the "ATLAS_2" directory.', required=True)
    parser.add_argument("-d", "--destination", help="Path for where to store the data.", required=False, default="data")
    pargs = parser.parse_args()

    bidsify_indi_atlas(atlas_path=pargs.atlas, destination_path=pargs.destination)
