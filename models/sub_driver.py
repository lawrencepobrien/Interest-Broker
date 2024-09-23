from subs import subscriber
import database_gen
from broker import BrokerModel
import time

start_time = time.time()
sub = subscriber()
creation_time = time.time() - start_time

x_train, y_train1, y_train2, x_test, y_test1, y_test2 = sub.read_data('../resources/sub_train_data.csv', '../resources/sub_test_data.csv')

start_time = time.time()
model1, model2 = sub.make_model(x_train, y_train1, y_train2)
training_time = time.time() - start_time

start_time = time.time()
sub.test_model(x_test, y_test1, y_test2, model1, model2)
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