from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x = np.array([range(1, 101), range(101, 201)])
y = np.array([range(1, 101), range(101, 201)])
print(x)

x = np.transpose(x)
y = np.transpose(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size=0.4)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, random_state=66, test_size=0.5)
'''
model = Sequential()
#ex)model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(8))
model.add(Dense(9))
model.add(Dense(6))
model.add(Dense(1))
'''
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

input = Input(shape=(2,))
hidden1 = Dense(8, activation='elu')(input)
hidden2 = Dense(7, activation='relu')(hidden1)
output = Dense(2, activation='elu')(hidden2)
model = Model(input, output)

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val))

loss, mse = model.evaluate(x_test,y_test, batch_size=1)
print('acc : ', mse)

y_predict = model.predict(x_test)
print(y_predict)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE : ', RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print('R2 : ', r2_y_predict)