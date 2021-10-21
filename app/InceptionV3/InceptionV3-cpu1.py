# import packages 
import numpy as np # linear algebra
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD, RMSprop
import tensorflow as tf
import cv2 as cv
import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Sequential
from keras.layers.pooling import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Dropout
from keras.models import model_from_json
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# build model
i_model = InceptionV3(weights= 'imagenet', include_top=False, input_shape=(299, 299, 3))
for layer in i_model.layers:
    layer.trainable = False
model = Sequential()
model.add(i_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(128))
model.add(Dropout(0.2))
model.add(Dense(7, activation = 'softmax'))
model.compile(optimizer = SGD(),
             loss="categorical_crossentropy",
             metrics=["accuracy"])


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")

# %%


# def classify(image_path)
img = load_img(r'46596068_12628.jpg', target_size=(299, 299))
# img = load_img(image_path, target_size=(299, 299))
img = img_to_array(img)
img = img / 255
img = np.expand_dims(img,axis=0)

# 
predictions = loaded_model.predict(img)

result = np.argmax(predictions, axis=1)

if result[0] == 0: 
    ans = 'Balinese'
elif result[0] == 1: 
    ans = 'Bengal'
elif result[0] == 2: 
    ans = 'Devon Rex'
elif result[0] == 3: 
    ans = 'Javanese'
elif result[0] == 4: 
    ans = 'Russian Blue'
elif result[0] == 5: 
    ans = 'Siberian'
else: 
    ans = 'Sphynx'
    

statement = f'\nThe probability of the cat being Balinese is {predictions[0][0]*100:.3f}%.\n\nThe probability of the cat being Bengal is {predictions[0][1]*100:.3f}%.\n\nThe probability of the cat being Devon Rex is {predictions[0][2]*100:.3f}%.\n\nThe probability of the cat being Javanese is {predictions[0][3]*100:.3f}%.\n\nThe probability of the cat being Russian Blue is {predictions[0][4]*100:.3f}%.\n\nThe probability of the cat being Siberian is {predictions[0][5]*100:.3f}%.\n\nThe probability of the cat being Sphynx is {predictions[0][6]*100:.3f}%. \n\nTherefore, this cat is classified as ' + ans + '.'

print(statement)
    


#     Balinese = predictions[0][0]*100
#     Bengal = predictions[0][1]*100
#     Devon_Rex = predictions[0][2]*100

#     # %%
#     Javanese = predictions[0][3]*100

#     # %%
#     Russian_Blue = predictions[0][4]*100

#     # %%
#     Siberian = predictions[0][5]*100

#     # %%
#     Sphynx = predictions[0][6]*100

#     print(Bengal)

    # # %%
# Balinese

# # %%
# Bengal

# # %%
# Devon_Rex

# # %%
# Javanese

# # %%
# Russian_Blue

# # %%
# Siberian

# # # %%
# Sphynx

# %%
# test["Labels"].replace(
#    {'Balinese': 0,
#  'Bengal': 1,
#  'Devon Rex': 2,
#  'Javanese': 3,
#  'Russian Blue': 4,
#  'Siberian': 5,
#  'Sphynx - Hairless Cat': 6}, inplace = True)

# %%


# %%


# %%
