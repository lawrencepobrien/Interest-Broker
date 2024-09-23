import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from PIL import Image

class BrokerModel:
    
    def read_data(self):
        train_data = pd.read_csv('../resources/broker_train_data.csv')
        test_data = pd.read_csv('../resources/broker_test_data.csv')
        
        # Data gathering
        train_pics = train_data['Image']
        x_train = []
        
        test_pics = test_data['Image']
        x_test = []
        
        # Preprocessing
        # x_ arrays store the arrya representation of the PIL image
        for address in train_pics:
            img = np.asarray(Image.open(address))
            x_train.append(img)
        for address in test_pics:
            img = np.asarray(Image.open(address))
            x_test.append(img)
            
        # Normalize the dataset (255 being the color values)
        x_train = np.array(x_train) / 255
        x_test = np.array(x_test) / 255
        
        # Reshape the data to store it properly, width=100 height=200
        x_train.reshape(-1, 200, 100, 1)
        x_test.reshape(-1, 200, 100, 1)
        
        y_train = train_data['Classifier']
        y_test = test_data['Classifier']
        
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        
        return x_train, y_train, x_test, y_test
    
    def make_model(self, x_train, y_train):
        # Model construction
        model1 = keras.models.Sequential()
        model1.add(keras.layers.Conv2D(32,3,padding="same", activation="relu", input_shape=(200,100,3)))
        model1.add(keras.layers.BatchNormalization())
        model1.add(keras.layers.MaxPool2D())
        model1.add(keras.layers.BatchNormalization())
        model1.add(keras.layers.MaxPool2D())
        model1.add(keras.layers.Flatten())
        model1.add(keras.layers.Dense(180, activation="relu"))
        model1.add(keras.layers.Dense(100, activation="relu"))
        model1.add(keras.layers.Dense(18, activation="relu"))
        model1.add(keras.layers.Dense(3, activation="softmax"))
        
        # Model compilation, based on accuracy
        opt = keras.optimizers.Adam(lr=0.000001)
        model1.compile(optimizer = opt , loss = tf.keras.losses.SparseCategoricalCrossentropy() , metrics = ['accuracy'])
    
        # Fit model to training data
        model1.fit(x_train, y_train, epochs=30)
        
        return model1
    
    def test_model(self, x_test, y_test, model1):
        print()
        print()
        
        # Testing the model
        model1.evaluate(x_test, y_test)