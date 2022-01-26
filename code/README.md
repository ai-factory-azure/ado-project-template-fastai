# Instructions

This is an example demonstrating how to do experimentation during model development.

Use case:
* File-based model training (input to model is just a bunch of files, e.g., images)
* Model is then used for batch scoring (input to batch scoring is also just a bunch of files, e.g., images)

## Run training locally (without AzureML)

```console
conda env create -f src/environment.yml
conda activate file-example
python src/train.py --epochs 1 --data_path data/
```

## Run training on AzureML

This will automatically upload and cache the data from `data/` to AzureML:

```console
az configure --defaults group=mnist workspace=mnistws
az ml job create -f azureml/train-azure.yml --stream
```

### Run training on AzureML using Dataset feature

```console
az configure --defaults group=mnist workspace=mnistws
az ml data create --file azureml/dataset.yml # Create dataset in AzureML
az ml job create -f azureml/train-azure-dataset.yml --stream
```

## Run training locally on your own machine (using AzureML)

```console
az configure --defaults group=ai-factory workspace=ai-factory
az ml job create -f azureml/train-azure.yml --set compute.target=local --stream
```

## Run batch scoring locally for testing

```console
python src/batch.py
```
