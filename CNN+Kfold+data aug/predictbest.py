import numpy as np
import pandas as pd
np.random.seed(1984)

import os
import glob
import cv2
import time
import warnings
import datetime

warnings.filterwarnings("ignore")
from tensorflow.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from tensorflow.keras.optimizers import SGD, Adagrad

def get_im_cv2(path):
    img = cv2.imread(path)
    resized = cv2.resize(img, (64, 64), cv2.INTER_LINEAR)
    return resized

def load_test(path, X_test, X_test_id):
    print(path)
    files = sorted(glob.glob(path))

    for fl in files:
        flbase = os.path.basename(fl)
        if path[:9] == "test_stg2":
            flbase = "test_stg2/" + flbase
        img = get_im_cv2(fl)
        X_test.append(img)
        X_test_id.append(flbase)

    return X_test, X_test_id


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
    model.add(Dense(96, activation='relu',kernel_initializer='random_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(24, activation='relu',kernel_initializer='random_uniform'))
    model.add(Dropout(0.5))
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

def read_and_normalize_test_data():
    start_time = time.time()
    X_test = []
    X_test_id = []

    for i in range(2):
        path = os.path.join('test_stg' + str(i + 1), '*.jpg')
        test_data, test_id = load_test(path, X_test, X_test_id)

    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.transpose((0, 3, 1, 2))

    test_data = test_data.astype('float32')
    test_data = test_data / 255

    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    print('Read and process test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return test_data, test_id

def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()

def create_submission(predictions, test_id,):
    result1 = pd.DataFrame(predictions, columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
    result1.loc[:, 'image'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    sub_file = 'submission_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result1.to_csv(sub_file, index=False)

def run_cross_validation_process_test(models):
    batch_size = 64
    num_fold = 0
    yfull_test = []
    test_id = []
    nfolds = len(models)

    for i in range(nfolds):
        model = models[i]
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        test_data, test_id = read_and_normalize_test_data()
        test_prediction = model.predict(test_data, batch_size=batch_size, verbose=2)
        yfull_test.append(test_prediction)

    test_res = merge_several_folds_mean(yfull_test, nfolds)
    create_submission(test_res, test_id)

if __name__ == '__main__':
    models = []
    num_folds = 20
    decay = 1e-5
    for i in range(num_folds):
        model = create_model()
        model.load_weights('model'+str(i+1)+'.h5')
        models.append(model)
    run_cross_validation_process_test(models)