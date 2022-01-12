# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path
from shutil import copyfile

import click
import numpy as np

# Libraries for preprocessing
import torch
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    def normalize(tensor):
        mu = torch.mean(tensor)
        std = torch.std(tensor)
        output = (tensor - mu) / std
        return output

    # Find training data
    files = os.listdir(input_filepath)
    data_all = [
        np.load(os.path.join(input_filepath, f))
        for f in files
        if f.endswith(".npz") and "train" in f
    ]

    # Combine .npz files
    merged_data = dict(data_all[0])
    for data in data_all[1:]:
        for k in data.keys():
            merged_data[k] = np.vstack((merged_data[k], dict(data)[k]))
    merged_data["labels"] = np.reshape(
        merged_data["labels"], merged_data["labels"].size
    )

    # TODO: Clean up
    np.savez(os.path.join(output_filepath, "train_data.npz"), **merged_data)

    # Load in the train file
    train = np.load(os.path.join(output_filepath, "train_data.npz"))

    # Now we organize into tenzors and normalize dem
    images_train = normalize(torch.Tensor(train.f.images))
    labels_train = torch.Tensor(train.f.labels).type(torch.LongTensor)

    # Save the individual tensors
    torch.save(images_train, os.path.join(output_filepath, "images_train.pt"))
    torch.save(labels_train, os.path.join(output_filepath, "labels_train.pt"))

    # Pass test data through to output
    copyfile(
        os.path.join(input_filepath, "test.npz"),
        os.path.join(output_filepath, "test.npz"),
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
