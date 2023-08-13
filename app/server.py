from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import torch
from torchvision.transforms import ToTensor
from transformers import ViTModel, ViTFeatureExtractor
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn
import torch.nn.functional as F

import requests
import numpy
from PIL import Image
from torchvision.transforms import ToTensor
import io
from io import BytesIO
import base64

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    # Extract the image from either the URL or the uploaded file
    if 'file' in request.files:
        # The user uploaded a file
        image_file = request.files['file']
        image = Image.open(image_file.stream)
        image = preprocess_image(image)
        encoded_image = None  # No need to send the uploaded image back
    elif 'url' in request.form:
      try:
        # The user provided a URL
        image_url = request.form['url']
        image = download_and_preprocess_image(image_url)

        # Encode the image in base64
        buffered = BytesIO(requests.get(image_url).content)
        encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
      except ValueError as e:
        # The URL is invalid
        return jsonify(error=str(e) + "\n\nTry downloading the image and resubmitting"), 400
    else:
        # No valid input provided
        return jsonify(error='No file or URL provided'), 400
    
    # Predict the image
    probabilities = predict_image(image, model, feature_extractor)

    ai_prob = float(probabilities[0][0])
    human_prob = float(probabilities[0][1])
    return jsonify({
        "human": human_prob,
        "ai": ai_prob,
        "image": f"data:image/jpeg;base64,{encoded_image}" if encoded_image else None
    })

if __name__ == '__main__':

  # Define the model
  class ViTForImageClassification(nn.Module):
    def __init__(self, num_labels=2):
      super(ViTForImageClassification, self).__init__()
      self.vit = ViTModel.from_pretrained('google/vit-large-patch32-384')
      self.dropout = nn.Dropout(0.1)
      self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
      self.num_labels = num_labels

    def forward(self, pixel_values, labels=None):
      outputs = self.vit(pixel_values=pixel_values)
      output = self.dropout(outputs.last_hidden_state[:,0])
      logits = self.classifier(output)
      
      if labels is not None:
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )
      else:
        return logits
  
  # Initialize model and feature extractor
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-large-patch32-384')
  model_path = "./app/static/AID96k_E15_384.pth"
  model = ViTForImageClassification(num_labels=2)
  model.load_state_dict(torch.load(model_path))
  model.to(device)
  model.eval()

  def download_and_preprocess_image(image_url, desired_size=384):
    # Send a GET request to the image URL
    response = requests.get(image_url)
    if response.status_code != 200:
      raise ValueError("Failed to download the image.")

    # Convert bytes to a PIL Image object
    im = Image.open(io.BytesIO(response.content))

    # Resize and pad the image
    old_size = im.size
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    im = im.resize(new_size, Image.ANTIALIAS)

    # Create a new image and paste the resized on it
    new_im = Image.new("RGB", (desired_size, desired_size), "white")
    new_im.paste(im, ((desired_size-new_size[0])//2,
                      (desired_size-new_size[1])//2))

    return new_im

  def preprocess_image(image, desired_size=384):
    im = image

    # Resize and pad the image
    old_size = im.size
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    im = im.resize(new_size)

    # Create a new image and paste the resized on it
    new_im = Image.new("RGB", (desired_size, desired_size), "white")
    new_im.paste(im, ((desired_size-new_size[0])//2, (desired_size-new_size[1])//2))
    return new_im
  
  

  def predict_image(image, model, feature_extractor):
    # Ensure model is in eval mode
    model.eval()

    # Convert image to tensor
    transform = ToTensor()
    input_tensor = transform(image)
    input_tensor = torch.tensor(numpy.array(feature_extractor(input_tensor)['pixel_values']))

    # Move tensors to the right device
    input_tensor = input_tensor.to(device)

    # Forward pass of the image through the model
    output = model(input_tensor)

    # Convert model output to probabilities using softmax
    probabilities = torch.nn.functional.softmax(output, dim=1)

    return probabilities.cpu().detach().numpy()
  
  app.run(debug=True, host='0.0.0.0')