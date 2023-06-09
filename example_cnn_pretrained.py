import torch
import pandas as pd

from lib.dataset import EnvironmentalDataset
from lib.cnn.utils import load_model_state
from lib.raster import PatchExtractor
from lib.cnn.models.inception_env import InceptionEnv
from lib.cnn.predict import predict
from lib.evaluation import evaluate
from lib.metrics import ValidationAccuracyMultipleBySpecies
from lib.metrics import ValidationAccuracyMultiple


# SETTINGS
# files
DATASET_PATH = './data/test_dataset.csv'
RASTER_PATH = './data/rasters_GLC19/'
# csv columns
ID = 'id'
LABEL = 'Label'
LATITUDE = 'Latitude'
LONGITUDE = 'Longitude'

# environmental patches (it must be the same as during the learning process)
PATCH_SIZE = 64

# model params (it must be the same as during the learning process)
DROPOUT = 0.7
N_LABELS = 4520

# evaluation
METRICS = (ValidationAccuracyMultipleBySpecies([1, 10, 30]), ValidationAccuracyMultiple([1, 10, 30]))

def main():
    # READ DATASET
    df = pd.read_csv(DATASET_PATH, header='infer', sep=';', low_memory=False)

    ids = df[ID].to_numpy()
    labels = df[LABEL].to_numpy()
    positions = df[[LATITUDE, LONGITUDE]].to_numpy()

    # create patch extractor
    extractor = PatchExtractor(RASTER_PATH, size=PATCH_SIZE, verbose=True)
    # add all default rasters
    extractor.add_all()

    # constructing pytorch dataset
    test_set = EnvironmentalDataset(labels, positions, ids, patch_extractor=extractor)


    # LOAD MODEL
    model = InceptionEnv(dropout=DROPOUT, n_labels=N_LABELS)
    load_model_state(model, 'pretrained/inception_env_pretrained.torch')
    if torch.cuda.is_available():
        # check if GPU is available
        model.to(torch.cuda.device(0))
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])


    # EVALUATION ON TEST SET
    predictions, labels = predict(model, test_set)
    print(evaluate(predictions, labels, METRICS, final=True))

if __name__ == '__main__':
    main()
    # exit nicely
    print('Done')

    sys.exit(0)