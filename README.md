# CNN SDM

Some recent SDMs use deep neural networks to better address the complexity of ecological niches. Devising SDMs based on neural networks is not new, but earlier models usually integrated a single hidden layer network. Recent advances in deep learning have allowed training much deeper neural networks and acknowledging more complexity in the way environment shapes ecological  niches. Key advantages of deep learning are that (i) it allows characterizing complex structuring of ecological niche depending on multiple environmental factors, (ii) it can learn niche features common to a large number of species, and thus grasp the signatures of common ecological processes and improve SDM predictions across species. 
A specific class of neural networks, named Convolutional Neural Networks (CNN), has very recently been proposed for SDM. A specific property of CNN is that they rely on spatial environmental tensors rather than on local values of environmental factors. These tensors represents the spatial structure of environmental factors around each point. Unlike other SDM approaches, CNN-based SDMs (CNN-SDMs) can represent how the configuration of environmental factors around species populations can affect the distribution of occurrences. They can capture much complex spatial patterns, at multiple scales, thanks to the tens of successive non-linear convolutions. CNNs were originally designed for image classification and proved to outperform any other statistical or machine learning methods in the task of learning complex visual patterns (mainly because of the too large dimensionality of the problem). CNN-SDMs should thus be suited to represent how complex ecological niches and spatial dynamics determine the distribution of many species in a region. Based on simulation results, have shown that CNN-SDMs can capture complex spatial information from environmental rasters and improve predictive performance in SDMs.

## Dependencies

Dependencies for cnn-sdm models:
- pytorch
- scikit-learn
- pandas
- numpy
- rasterio

Additional dependencies for scikit-learn models:
- joblib

Additional dependencies for xgboost models:
- xgboost

## Data

The data is composed of 4 files in the ```./data``` directory.

- occurrences
    - ```full_dataset.csv```
    - ```train_dataset.csv```
    - ```test_dataset.csv```
- rasters
    - ```rasters.zip```

To use the examples given in this repository simply extract the ```./rasters``` folder from the archive.
The ```train_dataset.csv``` and ```test_dataset.csv``` files are the result of a random split of ```full_dataset.csv``` occurrences. Pre-trained models are learned on this separation.
This dataset includes 97683 occurrences from 4520 plant species on the french territory.

## Usage

A full exemple of cnn learning experience is given in ```example_cnn_learning.py```.
A pretrained model is also given (```./pretrained/inception_env_pretrained.torch```) and an exemple code to use it for testing is given in ```example_cnn_pretrained.py```.

### Patch extractor

In ```lib/raster.py``` is given a custom code to extract patches from the environmental raster used there. After it's initialization the PatchExtractor object can extract numpy arrays of desired size for any valid position covered by rasters.

```python
from lib.raster import PatchExtractor

# create extractor
extractor = PatchExtractor("./data/rasters/")
# add all default rasters
extractor.add_all()

# extract patch
lat, lng = 43.6, 3.8
patch = extractor[(lat, lng)]
```

The default extraction size is ```size=64```

### Dataset

In ```lib/dataset.py``` is given an example of a Dataset object (EnvironmentalDataset) extending the pytorch Dataset class. This dataset is a tool to facilitate data managment.
It is initialized with the list of occurrences (positions, labels and ids) and the path to the environmental raster folder.
This Dataset object can return for each occurrence id the pre-formated data for pytorch models (environmental patches in torch tensor format and labels).
The dataset object handles the patch extractor.

read occurrences:
```python
import pandas as pd

# read occurrences
df = pd.read_csv("./data/full_dataset.csv", header='infer', sep=';', low_memory=False)

ids = df["id"].to_numpy()
labels = df["Label"].to_numpy()
positions = df[["Latitude", "Longitude"]].to_numpy()
```

split train/test:
```python
from sklearn.model_selection import train_test_split

# splitting train val test
test_size = 0.1
val_size = 0.1
train_labels, test_labels, train_positions, test_positions, train_ids, test_ids\
    = train_test_split(labels, positions, ids, test_size=test_size)
train_labels, val_labels, train_positions, val_positions, train_ids, val_ids\
    = train_test_split(train_labels, train_positions, train_ids, test_size=val_size)
```

create dataset:
```python
from lib.dataset import EnvironmentalDataset

# create dataset
train_dataset = EnvironmentalDataset(train_labels, train_positions, train_ids, patch_extractor=extractor)

# get data
torch_tensor, label = train_dataset[0]
```



### Convolutional neural networks

The architecture of the CNN used here is inception-v3 (designed by Google) adapted to our data. The description of the pytorch architecture is given at ```lib/cnn/models/inception_env.py```

#### creation

create a model:
```python
from lib.cnn.models.inception_env import InceptionEnv

model = InceptionEnv()
```

 The default model setting is ```n_labels=4520, n_inputs=77, dropout=0.7```
 
 After it's creation, for device with CUDA available, the model can be put on GPU:
 ```python
if torch.cuda.is_available():
    # check if GPU is available
    model.cuda()
```

#### learning

A training procedure is given in the file ```lib/cnn/train.py```.
Fitting the model:
```python
from lib.cnn.train import fit
from lib.metrics import ValidationAccuracyMultipleBySpecies
from lib.metrics import ValidationAccuracyMultiple

# FITTING THE MODEL
fit(
    model,
    train=train_dataset, validation=validation_dataset,
    batch_size=128, iterations=[90, 130, 150, 170, 180], log_modulo=500, val_modulo=5, lr=0.1, gamma=0.1,
    metrics=(ValidationAccuracyMultipleBySpecies([1, 10, 30]), ValidationAccuracyMultiple([1, 10, 30]))
)
```
- ```lr``` is the learning rate (default=0.1)
- ```gamma``` is the coefficient to apply when decreasing the learning rate (default=0.1)
- ```iterations``` is a list of epochs numbers, it indicates when to changes learning rate (default=[90, 130, 150, 170, 180])
- ```log_modulo``` indicates after how many batches the loss is printed (default=500)
- ```val_modulo``` indicates after how many epochs should be done a validation (default=5)

During the learning process given here, at each validation the result of the evaluation on the validation set is printed and the model is saved.
After the end of learning phase it is possible to find the best validation model and therefore, to use it for the final test or predictions.

#### predicting

The prediction code is located in ```lib/cnn/predict.py```.
Predicting on a test set:
```python
from lib.cnn.predict import predict

predictions, labels = predict(model, test_dataset)
```
The fonction ```predict()``` return two numpy array, the first one is a 2D array giving for each row the prediction for the corresponding test occurrence (each column represent the species label)(prediction are in the same order than the occurrences in the dataset).
The second one is a 1D array with the corresponding ground truth.

#### evaluate

To evaluate a prediction in the file ```lib/metrics.py``` is given a bunch of metric functions.
Evaluate with a metric:
```python
from lib.metrics import ValidationAccuracyMultipleBySpecies

# create metric
metric = ValidationAccuracyMultipleBySpecies([1, 10, 30])

# evaluate
score = metric(predictions, labels)
```

#### tensors transformations

A set of transformations fonctions is given in the file ```lib/transformations.py```. To apply these transformation on environmental tensors simply give them to the dataset object.
```python
from lib.transformations import random_rotation

train_dataset = EnvironmentalDataset(train_labels, train_positions, train_ids, patch_extractor=extractor, transform=random_rotation)
```

## Other models

Two examples are given to use scikit-learn (```example_rf.py```) and xgboost (```example_bt.py```) models on this dataset.

