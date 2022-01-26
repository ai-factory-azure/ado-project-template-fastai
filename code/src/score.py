import os
import io
import glob
import json
import fastai.vision as vis
from fastai.vision import load_learner, open_image

from azureml.contrib.services.aml_request import AMLRequest, rawhttp
from azureml.contrib.services.aml_response import AMLResponse

def init():
    global model
    print(f"Model directory {os.getenv('AZUREML_MODEL_DIR')}")
    print(f"Model directory contents: {glob.glob(os.getenv('AZUREML_MODEL_DIR') + '/**')}")
    model = load_learner(path=os.getenv('AZUREML_MODEL_DIR'), file='outputs/model.pkl')

@rawhttp
def run(request):
    if request.method == 'POST':
        body = request.get_data(False)
        response = score(body)
        return AMLResponse(json.dumps(response, indent=2), 200)
    else: 
        return AMLResponse(f"HTTP {request.method} is not supported", 405)
   
def score(data):
    image = open_image(io.BytesIO(data), convert_mode='L')
    prediction = model.predict(image)
    result = {
        'probabilties': prediction[2].numpy().tolist()
    }
    return result

# For local testing
if __name__ == "__main__":
    global model
    model = load_learner(path='outputs', file='model.pkl')
    f = open('mnist_tiny/valid/3/76.png', 'rb').read()
    response = score(f)
    print(response)
    json.dumps(response, indent=2)