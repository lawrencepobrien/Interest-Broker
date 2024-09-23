import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from PIL import Image

class subscriber:
    
    def __init__(self):
        pass

    def read_data(self, trainfilepath, testfilepath):
        train_data = pd.read_csv(trainfilepath)
        test_data = pd.read_csv(testfilepath)
        
        # Data gathering
        train_pics = train_data['Image']
        test_pics = test_data['Image']
        x_train = []
        x_test = []
        
        # Preprocessing
        # x_ arrays store the arrya representation of the PIL image
        for address in train_pics:
            x_train.append(np.asarray(Image.open(address)))
        for address in test_pics:
            x_test.append(np.asarray(Image.open(address)))
            
        # Normalize the dataset (255 being the color values)
        x_train = np.array(x_train) / 255
        x_test = np.array(x_test) / 255
        
        # Reshape the data to store it properly, width=100 height=200
        x_train.reshape(-1, 200, 100, 1)
        x_test.reshape(-1, 200, 100, 1)
        
        y_train1 = train_data['Sub1 Classifier']
        y_test1 = test_data['Sub1 Classifier']
        
        y_train1 = np.array(y_train1)
        y_test1 = np.array(y_test1)
        
        y_train2 = train_data['Sub2 Classifier']
        y_test2 = test_data['Sub2 Classifier']
        
        y_train2 = np.array(y_train2)
        y_test2 = np.array(y_test2)
        
        return x_train, y_train1, y_train2, x_test, y_test1, y_test2
    
    def make_model(self, x_train, y_train1, y_train2):
        # Model construction
        model1 = keras.models.Sequential()
        model1.add(keras.layers.Conv2D(32,3,padding="same", activation="relu", input_shape=(200,100,3)))
        model1.add(keras.layers.MaxPool2D())
        model1.add(keras.layers.Flatten())
        model1.add(keras.layers.Dense(128,activation="relu"))
        model1.add(keras.layers.Dense(2, activation="softmax"))
        
        # Model 2
        model2 = keras.models.Sequential()
        model2.add(keras.layers.Conv2D(32,3,padding="same", activation="relu", input_shape=(200,100,3)))
        model2.add(keras.layers.MaxPool2D())
        model2.add(keras.layers.Flatten())
        model2.add(keras.layers.Dense(128,activation="relu"))
        model2.add(keras.layers.Dense(2, activation="softmax"))
        
        # Model compilation, based on accuracy
        opt = keras.optimizers.Adam(lr=0.000001)
        model1.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
        model2.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
        
        # Fit model to training data
        model1.fit(x_train, y_train1, epochs=6)
        model2.fit(x_train, y_train2, epochs=6)
        
        return model1, model2
    
    def test_model(self, x_test, y_test1, y_test2, model1, model2):
        print()
        print()
        
        # Testing the model
        model1.evaluate(x_test, y_test1)
        model2.evaluate(x_test, y_test2)