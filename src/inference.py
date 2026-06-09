
import tensorflow as tf, numpy as np
from PIL import Image

model=tf.keras.models.load_model('models/resnet50_model.keras')

img=Image.open('sample.jpg').resize((224,224))
x=np.array(img)/255.0
x=np.expand_dims(x,0)

pred=model.predict(x)
print('Class:',np.argmax(pred))
print('Confidence:',float(np.max(pred)))
