import os
import json
import argparse
import fastai.vision as vis
from fastai.vision import load_learner, open_image
from azureml.core.model import Model

model = None

def init():
    global model
    print("Started batch scoring by running init()")
    
    parser = argparse.ArgumentParser('batch_scoring')
    parser.add_argument('--model_name', type=str, help='Model to use for batch scoring')
    args, _ = parser.parse_known_args()
    
    model_path = Model.get_model_path(args.model_name)
    print(f"Model path: {model_path}")
    model = load_learner(path='', file=model_path)

def run(file_list):
    print(f"Files to process: {file_list}")
    try:
        
        results = []
        for filename in file_list:
            
            image = open_image(filename, convert_mode='L')
            prediction = model.predict(image)
            print(f"Filename: {filename} / Prediction: {prediction}")
            result = {
                'file': os.path.basename(filename),
                'probabilties': prediction[2].numpy().tolist()
            }
            results.append(result)
            print(f"Batch scored: {filename}")
        return results
    except Exception as e:
        error = str(e)
        return error

def test():
    global model
    model = load_learner(path='', file='outputs/model.pkl')
    files = ['mnist_tiny/valid/3/76.png', 'mnist_tiny/valid/3/90.png']
    result = run(files)
    print(result)

if __name__ == "__main__":
    test()