
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title('Crop Disease Detection')

try:
    model=tf.keras.models.load_model('models/resnet50_model.keras')
except:
    model=None

uploaded=st.file_uploader('Upload Leaf Image',type=['jpg','png','jpeg'])

if uploaded and model:
    img=Image.open(uploaded)
    st.image(img)
    x=np.array(img.resize((224,224)))/255.0
    x=np.expand_dims(x,0)
    pred=model.predict(x)
    st.success(f'Predicted Class Index: {np.argmax(pred)}')
    st.write(f'Confidence: {np.max(pred)*100:.2f}%')
