# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


# Libraries for preprocessing
import torch
import numpy as np
import os
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Combine .npz files
    data_all = [np.load(os.path.join(input_filepath, 'train_%s.npz' % i)) for i in range(5)]
    merged_data = {}
    for data in data_all:
        [merged_data.update({k: v}) for k, v in data.items()]
    np.savez(os.path.join(output_filepath, 'train_data.npz'), **merged_data) # TODO: Clean up

    # Load in the test and train file
    train = np.load(os.path.join(output_filepath, 'train_data.npz'))  
    test  = np.load(os.path.join(input_filepath, 'test.npz'))

    # Now we organize into tenzors
    images_train = torch.Tensor(train.f.images)
    labels_train = torch.Tensor(train.f.labels).type(torch.LongTensor)

    images_test = torch.Tensor(test.f.images)
    labels_test = torch.Tensor(test.f.labels).type(torch.LongTensor)

    # Save the individual tensors
    torch.save(images_train, os.path.join(output_filepath, 'images_train.pt'))
    torch.save(labels_train, os.path.join(output_filepath, 'labels_train.pt'))
    torch.save(images_test,  os.path.join(output_filepath, 'images_test.pt'))
    torch.save(labels_test,  os.path.join(output_filepath, 'labels_test.pt'))

    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
