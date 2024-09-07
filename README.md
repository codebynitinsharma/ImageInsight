# ImageInsinght

## Overview 
Image captioning is a challenging task in the field of computer vision and natural language processing. The goal of this project is to automatically generate captions that accurately describe the content of an image.
## Data Model:
The link for data is [Data](https://drive.google.com/drive/folders/1eAwZfi882z_sA-PZpSgqgxUz8AJDlk5e?usp=drive_link)

## Model 1

## Technology used
python, pandas,numpy,tensorflow,keras,streamlit.


## Features
- ###  CNN Feature Extraction
   Utilizes the Inception model to extract high-level features from input images.
- ### LSTM Caption Generation:
    Employs a Long Short-Term Memory (LSTM) network to generate captions based on the extracted image features.
- ###  Streamlit Interface:
    Hosted using Streamlit, providing a user-friendly interface for uploading images and viewing generated captions.
- ### Pretrained Models:
   Includes pre-trained weights for both the CNN and LSTM models to facilitate quick deployment and usage.
- ### Trained on Flickr Dataset:
   The models are trained on the Flickr dataset, a widely used benchmark dataset for image captioning tasks.

## Model Architecture

#### The architecture of the image captioning model consists of two main components:

- CNN (Convolutional Neural Network): Extracts high-level features from input images using the Inception model.

- LSTM (Long Short-Term Memory): Generates descriptive captions based on the features extracted by the CNN.

## Installation
- Cloning the repo
  git clone https://github.com/AMANREVANKAR/captionwiz.git
- Activating the virtual environment
  
   run source env/bin/activate
- Running the streamlit file
  
   run streamlit run app.py

## Working
<img width="880" alt="Screenshot 2024-03-21 at 9 26 20 PM" src="https://github.com/AMANREVANKAR/captionwiz/assets/122635887/83e2b474-71a0-4f6d-bedb-710f6a6a517a">

## Model 2

## Model using CNN and Transformer

## Dependencies
•⁠  ⁠Python 3.x
•⁠  ⁠TensorFlow 2.x
•⁠  ⁠Keras
•⁠  ⁠NumPy
•⁠  ⁠OpenCV

 
## Dataset
We used the MS Coco - 2017 dataset for training and evaluation. It contains a large collection of images (around 6 hundred thousand) and corresponding captions.

## Model Architecture
We employ a Transformer model for image captioning. The model consists of three main parts: A pre-trained CNN (inceptionv3 is used for this project) for image extraction, the Transformer Encoder for processing the image features and the caption embeddings, and a Transformer Decoder for generating the output captions.
