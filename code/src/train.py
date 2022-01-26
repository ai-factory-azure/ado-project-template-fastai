import os
import shutil
import argparse
import mlflow.fastai
from mlflow.tracking import MlflowClient

from fastai.vision.all import (
    CategoryBlock,
    DataBlock,
    GrandparentSplitter,
    ImageBlock,
    PILImage,
    URLs,
)
from fastai.vision.all import cnn_learner, get_image_files, parent_label, resnet18, untar_data


def parse_args():
    parser = argparse.ArgumentParser(description="Fasti.ai MNIST example")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs (default: 5). Note it takes about 1 min per epoch")
    parser.add_argument("--data_path", type=str, default='data/', help="Directory path to training data")
    parser.add_argument("--model_path", type=str, default='outputs/', help="Model output directory")
    return parser.parse_args()
    
def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))

def main():
    # Parse command-line arguments
    args = parse_args()

    print(args.data_path)
    os.listdir(os.getcwd())

    # Make sure model output path exists
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Extract training data
    # data_file = os.path.join(args.data_path, 'mnist_tiny.tgz')
    # shutil.unpack_archive(filename=data_file, format="gztar")
    
     # Split data between training and testing
    splitter = GrandparentSplitter(train_name="training", valid_name="testing")

    # Prepare DataBlock which is a generic container to quickly build Datasets and DataLoaders
    mnist = DataBlock(
        blocks=(ImageBlock(PILImage), CategoryBlock),
        get_items=get_image_files,
        splitter=splitter,
        get_y=parent_label,
    )

    # Download, untar the MNIST data set and create DataLoader from DataBlock
    data = mnist.dataloaders(untar_data(URLs.MNIST), bs=256, num_workers=0)


    # Train and fit the Learner model, set output path to args.model_path
    learner = cnn_learner(data, resnet18, path=args.model_path)
    
    # Enable autologging via mlflow
    mlflow.fastai.autolog()

    with mlflow.start_run() as run:
        mlflow.log_metric('example1', 1.23)
        learner.fit(args.epochs, 0.01) # Train and fit with default or supplied command line arguments

    # fetch the auto logged parameters, metrics, and artifacts
    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
    
    # Export model
    learner.export('model.pkl')

if __name__ == "__main__":
    main()