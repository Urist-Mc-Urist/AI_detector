# Human vs AI Art Classifier
The increasing quality of AI-generated art has made it difficult to distinguish from human-created works. Many existing "AI detector" models fall short, mistakenly categorizing high-quality AI art as human-made or associating poorly-drawn images with AI. This project was inspired by the realization that these models were potentially oversimplifying their classification criteria.
High-quality AI art is frequently posted in stable diffusion threads on 4chan. The hypothesis driving this classifier was that by using such top-tier AI-generated images for training, the model could learn to recognize subtle generative artifacts rather than just obvious defects.
However, there are challenges. The model's current training set for human images could benefit from more diversity, such as including classical paintings, abstract art, 3D renders, and various other art forms. Furthermore, the model tends to label unique or abstract images as AI-generated, which may indicate some bias. One significant limitation is the visual transformer architecture's 384x384 resolution cap, leading to the potential loss of critical visual details during compression.
The goal of this project was to test that hypothesis and provide a handy tool to quickly test if a suspicious image is likely AI generated. Remember that the best image classification model in the world is currently between your ears, and to take any outputs from the classifier with a grain of salt. 
## Project Overview:
### Dataset:
- **Total Images**: 110,688
- **Dimensions**: 384x384 (aspect ratio preserved, edges padded with white)
- **Split**: Train (95.9k), Test (14.6k)
### Data Sources:
The dataset consists of both human-generated and AI-generated images. Here's where the data was collected from:
#### Human-Generated:
- [Danbooru.donmai.us](https://danbooru.donmai.us/)
- [idol.sankakucomplex.com](https://idol.sankakucomplex.com/) (NSFW)
- [reddit.com/r/photographs](https://www.reddit.com/r/photographs/)
- [reddit.com/r/itookapicture](https://www.reddit.com/r/itookapicture/)
#### AI-Generated:
- "Stable Diffusion General" (/sdg/) threads on [4chan.org/g](https://www.4chan.org/g), accessed via [desuarchive.org](https://desuarchive.org/)
### Model:

![spreadsheet](https://github.com/Urist-Mc-Urist/AI_detector/assets/80123386/681d6baf-2e5e-4d84-80be-1ed01fce3603)

Built around a pretrained transformer model enhanced with custom layers for classification. It underwent various training epochs with the best checkpoint at epoch 15 showing:
Built around a pretrained transformer model enhanced with custom layers for classification. It underwent 17 total training epochs with the best checkpoint at epoch 15 showing:
- **Accuracy**: 0.8712
- **Recall**: 0.8690
- **F1 Score**: 0.8694
- **AI Detection Accuracy**: 88.00%
- **False Positive Rate**: 15.02%
Additional training data was added to the dataset after epoch 10 so direct comparison to epochs 1-10 isn't necessarily accurate. With additional training after epoch 15, the model was able to achieve a higher AI detection accuracy, but at the expense of a significantly higher false positive rate. A more in-depth training set-up and more training may be able to achieve better results.
### Web App:
<p align="center"> <img width="500" src="https://github.com/Urist-Mc-Urist/AI_detector/assets/80123386/1a3d869c-208a-45cb-83f4-9a703803981f"></p>
An interactive application where users can either upload an image or provide a URL to classify the image as human-generated or AI-generated.
**Features**:
- Local image file upload and URL-based image analysis.
- Result visualization showing the likelihood of each class.
- Simple, user-friendly, and responsive UI.
## Setup & Running the App:
### Prerequisites:
Ensure you have [conda](https://docs.conda.io/en/latest/) installed on your system.
### 1. Model Setup:
Given the size of the model, it is not hosted on this GitHub repository. Before running the app:
- Download the model from [here](https://drive.google.com/file/d/1-BznBiGo-E2p7QdpInbZGEiEhxmWGKwe/view?usp=drive_link).
- Place the downloaded model in the `app/static` folder.
### 2. Installation:
- Navigate to the project directory.
- Run `setup.bat` to install necessary Python libraries.
### 3. Running the App:
- After installation, run `run.bat` which will activate the conda environment and start the server.
- Open a browser and visit `http://localhost:5000/` to interact with the app.
## Insights & Limitations:
- The model is particularly adept at classifying anime art and photographs.
- It might face challenges with abstract art, unique styles, and high-quality AI-generated images.
