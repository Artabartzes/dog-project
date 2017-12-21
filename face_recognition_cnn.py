# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 11:24:28 2017

@author: Gary
"""

from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Dense, Activation
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 2)
    return dog_files, dog_targets


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)



# load train, test, and validation datasets
def load_data(path):
    global train_files, train_targets, train_tensors
    global valid_files, valid_targets, valid_tensors
    global test_files, test_targets, test_tensors

    all_files, all_targets = load_dataset(path)

    all_cnt = int(len(all_files)/2)
    train_idx = int(0.5 * all_cnt)
    valid_idx = int(0.3 * all_cnt + train_idx)
    test_idx = int(0.2 * all_cnt + valid_idx)

    #### Look at Image 398 in Shuffle
    np.random.seed(42)
    shuffled = np.random.choice(all_cnt,all_cnt).astype(int)
    train_files   = all_files[shuffled[:train_idx]]
    train_targets = all_targets[shuffled[:train_idx]]

    valid_files   = all_files[shuffled[train_idx:valid_idx]]
    valid_targets = all_targets[shuffled[train_idx:valid_idx]]

    test_files   = all_files[shuffled[valid_idx:test_idx+1]]
    test_targets = all_targets[shuffled[valid_idx:test_idx+1]]

    for i in range(len(valid_files)):
        #print(valid_files[i][8:10],valid_targets[i])
        if (
                (valid_files[i][8:10]=='No') & (valid_targets[i][0]==0) |
                (valid_files[i][8:11]=='Yes') & (valid_targets[i][0]==1)
            ):
            print(valid_files[i],valid_targets[i])

    # pre-process the data for Keras
    train_tensors = paths_to_tensor(train_files).astype('float32')/255
    valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
    test_tensors = paths_to_tensor(test_files).astype('float32')/255
    return

#def define_model():
#global model
#print('define')

load_data('D:\Temp')

epochs = 20
batch_size = 220

model = Sequential()

##56% with 3x3x3x2

### TODO: Define your architecture.
model.add(Conv2D(32, (3, 3), input_shape=(224, 224,3)))
model.add(Activation('relu'))
#model.add(Conv2D(32, (5, 5)))
#model.add(Activation('relu'))
#model.add(Conv2D(32, (5, 5)))
#model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

#model.add(Conv2D(64, (3, 3)))
#model.add(Activation('relu'))
#model.add(Conv2D(64, (3, 3)))
#model.add(Activation('relu'))
#model.add(Conv2D(64, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))


#model.add(Conv2D(128, (3, 3)))
#model.add(Activation('relu'))
#model.add(Conv2D(64*2, (3, 3)))
#model.add(Activation('relu'))
#model.add(Conv2D(64*2, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(GlobalAveragePooling2D())

model.add(Dense(2))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

model.summary()


#model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
##63% accuracy
#return


### TODO: specify the number of epochs that you would like to use to train the model.
#def train_model():
#print('train')
#global epochs, batch_size, model
#global train_files, train_targets, train_tensors
#global valid_files, valid_targets, valid_tensors
#global test_files, test_targets, test_tensors

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.find_faces.hdf5', 
                               verbose=1, save_best_only=True)

## Original fit call before using the ImageDataGenerator
#model.fit(train_tensors, train_targets, 
#          validation_data=(valid_tensors, valid_targets),
#          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=2)

# this is the augmentation configuration I will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

train_generator = train_datagen.flow(train_tensors, train_targets)
validation_generator = train_datagen.flow(valid_tensors, valid_targets)

model.fit_generator(
        train_generator
        , steps_per_epoch=2000 // batch_size
        , epochs=epochs
        , validation_data=validation_generator
        , validation_steps=800 // batch_size
        , callbacks=[checkpointer], verbose=2)

model.load_weights('saved_models/weights.best.find_faces.hdf5')
#return

#def test_model():
#    print('test')
#    global model
#    global test_files, test_targets, test_tensors

# get index of predicted dog breed for each image in test set
face_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# report test accuracy
test_accuracy = 100*np.sum(np.array(face_predictions)==np.argmax(test_targets, axis=1))/len(face_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
#return

#################### Main #####################




