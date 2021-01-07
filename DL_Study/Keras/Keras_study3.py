import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

a = np.array(range(1, 11))

size = 5


def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i + size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)


dataset = split_x(a, size)
print('------------------')
print(dataset)

x_train = dataset[:, 0:-1]
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
y_train = dataset[:, -1]

print(x_train)
print(y_train)

model = Sequential()
model.add(LSTM(20, activation='relu', input_shape=(4,1)))
model.add(Dense(10))
model.add(Dense(13))
model.add(Dense(6))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, batch_size=1, epochs=100)

x_input = np.array([35, 36, 37, 38])  # predict용
x_input = x_input.reshape((1, 4, 1))

yhat = model.predict(x_input)
print(yhat)
