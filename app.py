import numpy as np
import pickle
import streamlit as st
import tensorflow
from PIL import Image
import pandas as pd
import cv2
import io

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer


# Add the CSS style for the background image
# st.markdown(
#     """
#     <style>
#     .main {
#         background-image: url("https://img.freepik.com/free-vector/wavy-background-concept_23-2148497712.jpg");
#         background-size: cover;
#         background-repeat: no-repeat;
#         background-position: center;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )



captions_file_path = '/Users/amanrev/Documents/IET/captionwiz/data/captions.txt'
with open(captions_file_path, 'r') as file:
    lines = file.readlines()

captions_dict = {}
for line in lines:
    image_id, caption = line.strip().split(',', 1)
    captions_dict.setdefault(image_id,[] ).append(caption)    

df = pd.DataFrame(list(captions_dict.items()), columns=['images_id', 'captions'])
df = df.iloc[1:]



captions_dict = {}
for line in lines:
    image_id, caption = line.strip().split(',', 1)
    captions_dict.setdefault(image_id, []).append(caption)

for image_id, captions in captions_dict.items():
    captions_dict[image_id] = [caption.strip('[]') for caption in captions]

def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            # take one caption at a time
            caption = captions[i]
            # preprocessing steps
            # convert to lowercase
            caption = caption.lower()
            # delete digits, special chars, etc., 
            caption = caption.replace('[^A-Za-z]', '')
            # delete additional spaces
            caption = caption.replace('\s+', ' ')
            # add start and end tags to the caption
            caption = '<start> ' + " ".join([word for word in caption.split() if len(word)>1]) + ' <end>'
            captions[i] = caption

clean(captions_dict)    

all_captions = []
for key in captions_dict:
    for caption in captions_dict[key]:
        all_captions.append(caption)




loaded_captioning_model=pickle.load(open("/Users/amanrev/Documents/IET/captionwiz/trained_captioning_model.sav",'rb'))







inception_v3 = InceptionV3()
inception_v3 = Model(inputs = inception_v3.inputs, outputs = inception_v3.layers[-2].output)
inception_v3.summary()




tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1







def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None








def predict_caption(model, image, tokenizer, max_length):
 
    in_text = '<start>'
 
    for i in range(max_length):
        print("hello2")
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        print(sequence)

        sequence = pad_sequences([sequence], max_length)
        print( sequence)
        
        yhat = model.predict([image, sequence], verbose=0)
        print(yhat)
      
        yhat = np.argmax(yhat)
        print(yhat)
 
        word = idx_to_word(yhat, tokenizer)
        
        if word is None:
            break
        print(word)
        in_text += " " + word
   
        if word == 'endseq':
            break
      
    return in_text















def get_feature_vector(img):
    img = tensorflow.expand_dims(img, axis=0)
    feature_vector = inception_v3(img)
    feature_vector = np.array(feature_vector)
    print(feature_vector)
    return feature_vector















def generate_caption(image):
    feature=get_feature_vector(image)
    y_pred = predict_caption(loaded_captioning_model,feature , tokenizer, 35) 
    y_pred = y_pred.replace('<start>', '')
    y_pred = y_pred.replace('<end>', '')
    y_pred = y_pred.replace('end', '')
    return y_pred  

# Main function to define the Streamlit app
def main():
    st.title("CaptionWiz: Image Captioning")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
       
        image_bytes = uploaded_file.read()
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        img = tensorflow.io.decode_jpeg(image_bytes, channels=3)
        img = tensorflow.keras.layers.Resizing(299, 299)(img)
        img = img/255
        if st.button("Generate Caption"):
            with st.spinner('Generating caption...'):
                caption = generate_caption(img)
                st.success("Caption: {}".format(caption)) 

# Run the app
if __name__ == "__main__":
    main()
