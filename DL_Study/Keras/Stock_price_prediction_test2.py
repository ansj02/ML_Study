from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
import numpy as np
import pandas_datareader.data as web
import datetime

start = datetime.datetime(2010, 12, 19)
end = datetime.datetime(2020, 3, 1)

gs = web.DataReader("035720.KS", "yahoo", start, end)

Change_Value=[]
n=20 # n-1일동안 변동치로 n번째일의 변동치 예측

for i in range(len(gs['Close'])) :
    Change_Value.append(gs['Close'][i] - gs['Close'][i-1])

while(len(Change_Value)%n != 0) :
    del Change_Value[0]

for i in range(len(Change_Value)) :
    if Change_Value[i] < -2000 :
        Change_Value[i] = -3
    elif Change_Value[i] < -1000 :
        Change_Value[i] = -2
    elif Change_Value[i] < 0 :
        Change_Value[i] = -1
    elif Change_Value[i] == 0 :
        Change_Value[i] = 0
    elif Change_Value[i] < 1000 :
        Change_Value[i] = 1
    elif Change_Value[i] < 2000 :
        Change_Value[i] = 2
    else :
        Change_Value[i] = -3

'''
for i in range(len(Change_Value)) :
    if Change_Value[i] < 0 :
        Change_Value[i] = -1
    elif Change_Value[i] == 0 :
        Change_Value[i] = 0
    else :
        Change_Value[i] = 1
'''

ComparisonTemp=[]



inputdataTemp=[]

for i in range(len(Change_Value)) :
    if i%n == n-1 :
        ComparisonTemp.append(Change_Value[i])
    else :
        inputdataTemp.append(Change_Value[i])
n=n-1
inputdata = [inputdataTemp[i * n:(i + 1) * n] for i in range((len(inputdataTemp) + n - 1) // n )]
#Comparison = [ComparisonTemp[i * 1:(i + 1) * 1] for i in range((len(ComparisonTemp)) // 1 )]
Comparison = ComparisonTemp

x = np.array(inputdata)
y = np.array(Comparison)
'''
from sklearn.preprocessing import StandardScaler

y = y.reshape(y.shape[0],1)
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
scaler.fit(y)
y = scaler.transform(y)
'''


print(x.shape)
print(y.shape)
x = x.reshape((x.shape[0], x.shape[1], 1))
print(x.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size=0.4)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, random_state=66, test_size=0.5)


model = Sequential()
#ex)model.add(Dense(units=64, activation='relu', input_dim=100))
#model.add(Dense(100, activation='elu', input_dim=n))
model.add(LSTM(10, activation='relu', input_shape=(n,1), return_sequences=True))
model.add(LSTM(18, activation='relu'))
model.add(Dense(14, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='relu'))
model.add(Dense(1))
'''
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
inputs = Input(shape=(n,))
hidden1 = Dense(10, activation='elu')(inputs)
hidden2 = Dense(12, activation='elu')(hidden1)
hidden3 = Dense(8, activation='elu')(hidden2)
hidden4 = Dense(7, activation='elu')(hidden3)
hidden5 = Dense(5, activation='elu')(hidden4)
outputs = Dense(1, activation='elu')(hidden5)
model = Model(inputs = inputs, outputs = outputs)
'''

model.compile(loss='mae', optimizer='adam', metrics=['acc'])

from keras.callbacks import EarlyStopping, TensorBoard

tb_hist = TensorBoard(log_dir='graph', histogram_freq=0, write_grads=True, write_images=True)

early_stopping = EarlyStopping(monitor = 'mae', patience = 10, mode ='max')

hist = model.fit(x_train, y_train, epochs=15, batch_size=1, validation_data=(x_val, y_val), callbacks = [early_stopping, tb_hist])

print(hist.history['loss'])
print(hist.history['acc'])
print(hist.history['val_loss'])
print(hist.history['val_acc'])


loss, mae = model.evaluate(x_test, y_test, batch_size=1)
print('acc : ', mae)

y_predict = model.predict(x_test)
print(y_predict,y_test)

#model.save('testTraningData.h5')