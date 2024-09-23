from subs import subscriber
import database_gen
from broker import BrokerModel
import time

start_time = time.time()
broker = BrokerModel()
creation_time = time.time() - start_time

x_train, y_train, x_test, y_test = broker.read_data()

start_time = time.time()
model1 = broker.make_model(x_train, y_train)
training_time = time.time() - start_time

start_time = time.time()
broker.test_model(x_test, y_test, model1)
testing_time = time.time() - start_time

start_time = time.time()
model1.predict(x_test)
inference_time = time.time() - start_time

print()
print()
print(creation_time)
print(training_time)
print(testing_time)
print(inference_time)