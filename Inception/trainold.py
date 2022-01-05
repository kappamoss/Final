from keras.applications.inception_v3 import InceptionV3
import os
from keras.layers import Flatten, Dense, AveragePooling2D
from keras.models import Model
from tensorflow.keras.optimizers import RMSprop, SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

learning_rate = 0.0001
img_width = 299
img_height = 299
nbr_train_samples = 3019
nbr_validation_samples = 758
nbr_epochs = 45
batch_size = 50

train_data_dir = './train_split'
val_data_dir = './val_split'

Fish = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

InceptionV3_notop = InceptionV3(include_top=False, weights='imagenet',
                    input_tensor=None, input_shape=(299, 299, 3))


output = InceptionV3_notop.get_layer(index = -1).output  
output = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(output)
output = Flatten(name='flatten')(output)
output = Dense(8, activation='softmax', name='predictions')(output)

InceptionV3_model = Model(InceptionV3_notop.input, output)
optimizer = SGD(lr = learning_rate, momentum = 0.9, decay = 1e-6, nesterov = True)

InceptionV3_model.compile(loss='categorical_crossentropy',optimizer = optimizer, metrics=['accuracy'])

best_model_file = "./best.h5"
best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose = 1, save_best_only = True)

earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=2,
                          verbose=1,restore_best_weights=True)




# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        shuffle = True,
        classes = Fish,
        class_mode = 'categorical')

validation_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle = True,
        classes = Fish,
        class_mode = 'categorical')

history =InceptionV3_model.fit_generator(
        train_generator,
        epochs= nbr_epochs,
        validation_data = validation_generator,
        callbacks = [earlystop])

plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
