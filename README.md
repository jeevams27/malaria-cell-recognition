# EX06--Deep Neural Network for Malaria Infected Cell Recognition

## AIM:

To develop a deep neural network for Malaria infected cell recognition and to analyze the performance.

## Problem Statement and Dataset:
Malaria dataset of 27,558 cell images with an equal number of parasitized and uninfected cells. A level-set based algorithm was applied to detect and segment the red blood cells. The images were collected and annotated by medical professionals.Here we build a convolutional neural network model that is able to classify the cells.
![image](https://github.com/NITHISH74/malaria-cell-recognition/assets/94164665/31067447-45e0-4100-8412-60ba20457615)

## Neural Network Model:

![image](https://github.com/NITHISH74/malaria-cell-recognition/assets/94164665/4c16630b-3aa1-4d1c-b485-c4458261d575)
## DESIGN STEPS:

### STEP 1:
Import tensorflow and preprocessing libraries.

### STEP 2:
Download and load the dataset folder.

### STEP 3:
Split the training and testing folders.

### STEP 4:
Perform image data generation methods.

### STEP 6:
Build the convolutional neural network model.

### STEP 7:
Train the model with the training data.

### STEP 8:
Plot the performance plot.

### STEP 9:
Evaluate the model with the testing data using probability prediction(uninfected-> prob>0.5,parasitized-> <=0.5).

### STEP 10:
Fit the model and predict the sample input.

## PROGRAM:
```python
### Developed By    : Nithishwar S
### Register Number : 212221230071
```
```python
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf

from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True 
config.log_device_placement = True 
sess = tf.compat.v1.Session(config=config)
set_session(sess)

%matplotlib inline

my_data_dir = 'dataset/cell_images'

os.listdir(my_data_dir)

test_path = my_data_dir+'/test/'
train_path = my_data_dir+'/train/'

os.listdir(train_path)

len(os.listdir(train_path+'/uninfected/'))

len(os.listdir(train_path+'/parasitized/'))

os.listdir(train_path+'/parasitized')[100]

para_img= imread(train_path+
                 '/parasitized/'+
                 os.listdir(train_path+'/parasitized')[100])
                 
plt.imshow(para_img)

dim1 = []
dim2 = []
for image_filename in os.listdir(test_path+'/uninfected'):
    img = imread(test_path+'/uninfected'+'/'+image_filename)
    d1,d2,colors = img.shape
    dim1.append(d1)
    dim2.append(d2)
    
sns.jointplot(x=dim1,y=dim2)

image_shape = (130,130,3)

model = models.Sequential()
model.add(layers.Conv2D(filters=32, kernel_size=(3,3),input_shape=image_shape, activation='relu',))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu',))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu',))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))


model.add(layers.Flatten())


model.add(layers.Dense(128))
model.add(layers.Activation('relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(1))
model.add(layers.Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
              
model.summary()

image_gen = ImageDataGenerator(rotation_range=20, 
                               width_shift_range=0.10, 
                               height_shift_range=0.10, 
                               rescale=1/255, 
                               shear_range=0.1, 
                               zoom_range=0.1, 
                               horizontal_flip=True, 
                               fill_mode='nearest'
                              )
batch_size = 16
train_image_gen = image_gen.flow_from_directory(train_path,
                                               target_size=image_shape[:2],
                                                color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary')
len(train_image_gen.classes)

test_image_gen = image_gen.flow_from_directory(test_path,
                                               target_size=image_shape[:2],
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary',shuffle=False)
train_image_gen.class_indices
results = model.fit(train_image_gen,epochs=4,
                              validation_data=test_image_gen
                             )
losses = pd.DataFrame(model.history.history)
losses[['loss','val_loss']].plot()

model.evaluate(test_image_gen)
pred_probabilities = model.predict(test_image_gen)

test_image_gen.classes
predictions = pred_probabilities > 0.5

print(classification_report(test_image_gen.classes,predictions))
confusion_matrix(test_image_gen.classes,predictions)

import random
import tensorflow as tf
list_dir=["uninfected","parasitized"]
dir_=(random.choice(list_dir))
para_img= imread(train_path+
                 '/'+dir_+'/'+
                 os.listdir(train_path+'/'+dir_)[random.randint(0,100)])
img  = tf.convert_to_tensor(np.asarray(para_img))
img = tf.image.resize(img,(130,130))
img=img.numpy()
pred=bool(model.predict(img.reshape(1,130,130,3))<0.5 )
plt.title("Model prediction: "+("Parasitized" if pred  else "Uninfected")+"\nActual Value: "+str(dir_))
plt.axis("off")
plt.imshow(img)
plt.show()
```

## OUTPUT:

### Training Loss, Validation Loss Vs Iteration Plot
![image](https://github.com/NITHISH74/malaria-cell-recognition/assets/94164665/42513d6f-1c23-448b-b719-b4106bd706f4)

### Classification Report

![image](https://github.com/NITHISH74/malaria-cell-recognition/assets/94164665/5ec447d4-fa8c-4fac-a826-da459e1e2aad)

### Confusion Matrix

![image](https://github.com/NITHISH74/malaria-cell-recognition/assets/94164665/240bd2f8-88f7-44d1-a511-d8f8ed9ba40a)

### New Sample Data Prediction
![image](https://github.com/NITHISH74/malaria-cell-recognition/assets/94164665/16097c7f-39c6-418f-be70-7dadd9f05afc)


## RESULT:
Thus, a deep neural network for Malaria infected cell recognition is developed and the performance is analyzed.
