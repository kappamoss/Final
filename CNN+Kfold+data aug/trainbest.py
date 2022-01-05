import numpy as np
np.random.seed(2000)

import os
import glob
import cv2
import datetime
import pandas as pd
import time
import warnings

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from tensorflow.keras.optimizers import SGD, Adagrad
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import utils
from sklearn.metrics import log_loss
from tensorflow.keras import __version__ as keras_version
from keras.preprocessing.image import ImageDataGenerator

def get_im_cv2(path):
    img = cv2.imread(path)
    resized = cv2.resize(img, (64, 64), cv2.INTER_LINEAR)
    return resized


def load_train():
    X_train = []
    X_train_id = []
    y_train = []
    start_time = time.time()

    print('Read train images')
    folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join('train', fld, '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im_cv2(fl)
            X_train.append(img)
            X_train_id.append(flbase)
            y_train.append(index)

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, y_train, X_train_id


def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
    result1.loc[:, 'image'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result1.to_csv(sub_file, index=False)


def read_and_normalize_train_data():
    train_data, train_target, train_id = load_train()

    print('Convert to numpy...')
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)

    print('Reshape...')
    train_data = train_data.transpose((0, 3, 1, 2))

    print('Convert to float...')
    train_data = train_data.astype('float32')
    train_data = train_data / 255
    train_target = utils.to_categorical(train_target, 8)

    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, train_id


def dict_to_list(d):
    ret = []
    for i in d.items():
        ret.append(i[1])
    return ret


def create_model():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 64, 64), data_format="channels_last"))
    model.add(Conv2D(8, 3, 3, activation='relu', data_format="channels_last", kernel_initializer='random_uniform'))
    model.add(Dropout(0.3))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2, 2), data_format="channels_last"))
    model.add(ZeroPadding2D((1, 1), data_format="channels_last"))
    model.add(Conv2D(16, 3, 3, activation='relu', data_format="channels_last", kernel_initializer='random_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2, 2), data_format="channels_last"))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(96, activation='relu', kernel_initializer='random_uniform'))
    model.add(Dropout(0.3))
    model.add(Dense(24, activation='relu', kernel_initializer='random_uniform'))
    model.add(Dropout(0.3))
    model.add(Dense(8, activation='softmax'))

    """# model = tf.keras.applications.InceptionV3()
    # create the base pre-trained model
    model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 8 classes
    predictions = tf.keras.layers.Dense(8, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=model.input, outputs=predictions)"""

    sgd = SGD(lr=5e-4, decay=decay, momentum=0.89, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    return model


def get_validation_predictions(train_data, predictions_valid):
    pv = []
    for i in range(len(train_data)):
        pv.append(predictions_valid[i])
    return pv


def run_cross_validation_create_models(nfolds=10):
    # input image dimensions
    batch_size = 64
    epochs =42
    random_state = 51
    first_rl = 96

    train_data, train_target, train_id = read_and_normalize_train_data()

    yfull_train = dict()
    kf = KFold(n_splits=nfolds, shuffle=True, random_state=random_state)
    kf.get_n_splits(len(train_id))
    num_fold = 0
    sum_score = 0
    models = []
    for train_index, test_index in kf.split(train_id):
        model = create_model()
        X_train = train_data[train_index]
        Y_train = train_target[train_index]
        X_valid = train_data[test_index]
        Y_valid = train_target[test_index]

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, verbose=0),
        ]
        datagen = ImageDataGenerator( 
            rotation_range = 90,  
            width_shift_range = 0.2, 
            height_shift_range = 0.2, 
            horizontal_flip=True,
            zoom_range = 0.3)

        datagen.fit(X_train)


        class_weights={}
        samples=[ 1375, 160, 93, 53, 372, 239, 140, 587]
        max_sample=np.max(samples)
        print (max_sample)
        for i in range (len(samples)):
            class_weights[i]=max_sample/samples[i]
        for key, value in class_weights.items():
            print ( key, ' : ', value)
  

        history=model.fit(x = datagen.flow(X_train, Y_train, batch_size=64), batch_size=batch_size,epochs=epochs,
              shuffle=True, verbose=2, validation_data=(X_valid, Y_valid),
              #class_weight=class_weights,
              callbacks=callbacks)

        # model.save_weights('checkpoints'+str(num_fold)+'/my_checkpoint')
        model.save_weights('model' + str(num_fold) + ".h5")

        predictions_valid = model.predict(X_valid.astype('float32'), batch_size=batch_size, verbose=2)
        score = log_loss(Y_valid, predictions_valid)
        print('Score log_loss: ', score)
        sum_score += score * len(test_index)

        # Store valid predictions
        for i in range(len(test_index)):
            yfull_train[test_index[i]] = predictions_valid[i]

        models.append(model)

    score = sum_score / len(train_data)
    print("Log_loss train independent avg: ", score)

    info_string = '_' + str(np.round(score, 3)) + '_decay_' + str(decay) + '_flds_' + str(nfolds) + '_eps_' + str(
        epochs) + '_fl_' + str(first_rl)



    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['loss', 'val_loss'], loc='upper left')
    plt.show()

    return info_string, models


def get_config(self):
    config = {
        "n_classes": self.n_classes,
        "s": self.s,
        "m": self.m,
        "easy_margin": self.easy_margin,
        "ls_eps": self.ls_eps
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


if __name__ == '__main__':
    decay = 1e-5
    print('Keras version: {}'.format(keras_version))
    num_folds = 20
    info_string, models = run_cross_validation_create_models(num_folds)