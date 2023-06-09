import torch
import pandas as pd
from sklearn.model_selection import train_test_split

from lib.dataset import EnvironmentalDataset
from lib.raster import PatchExtractor
from lib.cnn.models.inception_env import InceptionEnv
from lib.cnn.train import fit
from lib.cnn.predict import predict
from lib.evaluation import evaluate
from lib.metrics import ValidationAccuracyMultipleBySpecies
from lib.metrics import ValidationAccuracyMultiple

import sys
from pathlib2 import Path

# SETTINGS
# files
DATASET_PATH = './data/full_dataset.csv'
RASTER_PATH = './data/rasters_GLC19'
# csv columns
ID = 'id'
LABEL = 'Label'
LATITUDE = 'Latitude'
LONGITUDE = 'Longitude'

# dataset construction
TEST_SIZE = 0.1
VAL_SIZE = 0.1

# environmental patches
PATCH_SIZE = 64

# model params
DROPOUT = 0.7
N_LABELS = 4520

# training
BATCH_SIZE = 32
ITERATIONS = [90, 130, 150, 170, 180]
LOG_MODULO = 32
VAL_MODULO = 1
LR = 0.1
GAMMA = 0.1

# evaluation
METRICS = (ValidationAccuracyMultipleBySpecies([1, 10, 30]), ValidationAccuracyMultiple([1, 10, 30]))

def main():
    # READ DATASET
    df = pd.read_csv(DATASET_PATH, header='infer', sep=';', low_memory=False)

    ids = df[ID].to_numpy()
    labels = df[LABEL].to_numpy()
    positions = df[[LATITUDE, LONGITUDE]].to_numpy()

    # splitting train val test
    train_labels, test_labels, train_positions, test_positions, train_ids, test_ids\
        = train_test_split(labels, positions, ids, test_size=TEST_SIZE, random_state=42)
    train_labels, val_labels, train_positions, val_positions, train_ids, val_ids\
        = train_test_split(train_labels, train_positions, train_ids, test_size=VAL_SIZE, random_state=42)

    # create patch extractor
    extractor = PatchExtractor(RASTER_PATH, size=PATCH_SIZE, verbose=True)
    # add all default rasters
    extractor.add_all()

    # constructing pytorch dataset
    train_set = EnvironmentalDataset(train_labels, train_positions, train_ids, patch_extractor=extractor)
    test_set = EnvironmentalDataset(test_labels, test_positions, test_ids, patch_extractor=extractor)
    validation_set = EnvironmentalDataset(val_labels, val_positions, val_ids, patch_extractor=extractor)

    patch = extractor[(45.17795, 5.7146)]  # generating a patch, just to check and extract input shape
    n_input = patch.shape[0]
    # CONSTRUCT MODEL
    model = InceptionEnv(dropout=DROPOUT, n_labels=N_LABELS, n_input=n_input)

    if torch.cuda.is_available():
        # check if GPU is available
        print("Training on GPU")
        device = torch.device('cuda')
        model.to(device)
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
    else:
        print("Training on CPU")


    # FITTING THE MODEL
    fit(
        model,
        train=train_set, validation=validation_set,
        batch_size=BATCH_SIZE, iterations=ITERATIONS, log_modulo=LOG_MODULO, val_modulo=VAL_MODULO, lr=LR, gamma=GAMMA,
        metrics=METRICS
    )


    # FINAL EVALUATION ON TEST SET
    predictions, labels = predict(model, test_set)
    print(evaluate(predictions, labels, METRICS, final=True))


if __name__ == '__main__':
    main()
    # exit nicely
    print('Done')

    sys.exit(0)