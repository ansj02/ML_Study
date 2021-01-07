import numpy as np
x1 = np.array([range(100), range(311,411), range(100)])
y1 = np.array([range(501, 601), range(711, 811), range(100)])

x2 = np.array([range(100, 200), range(311, 411), range(100, 200)])
y2 = np.array([range(501, 601), range(711, 811), range(100)])

x3 = np.array([range(100, 200), range(311, 411), range(100, 200)])
y3 = np.array([range(501, 601), range(711, 811), range(100)])

y4 = np.array([range(501, 601), range(711, 811), range(100)])

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)
y2 = np.transpose(y2)
x3 = np.transpose(x3)
y3 = np.transpose(y3)
y4 = np.transpose(y4)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.4, random_state=66)
x1_test, x1_val, y1_test, y1_val = train_test_split(x1_test, y1_test, test_size=0.5, random_state=66)

x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.4, random_state=66)
x2_test, x2_val, y2_test, y2_val = train_test_split(x2_test, y2_test, test_size=0.5, random_state=66)

x3_train, x3_test, y3_train, y3_test = train_test_split(x3, y3, test_size=0.4, random_state=66)
x3_test, x3_val, y3_test, y3_val = train_test_split(x3_test, y3_test, test_size=0.5, random_state=66)

y4_train, y4_test = train_test_split(y4, test_size=0.4, random_state=66)
y4_test, y4_val = train_test_split(y4_test, test_size=0.5, random_state=66)

from keras.models import Sequential, Model
from keras.layers import Dense, Input

input1 = Input((3,))
hidden1 = Dense(10, activation='relu')(input1)
hidden2 = Dense(7)(hidden1)
middle1 = Dense(3)(hidden2)

input2 = Input((3,))
hidden1 = Dense(10, activation='relu')(input2)
hidden2 = Dense(7, activation='relu')(hidden1)
middle2 = Dense(3, activation='elu')(hidden2)

input3 = Input((3,))
hidden1 = Dense(10, activation='relu')(input3)
hidden2 = Dense(7, activation='relu')(hidden1)
middle3 = Dense(3, activation='elu')(hidden2)

from keras.layers.merge import concatenate
merge = concatenate([middle1, middle2, middle3])

output1 = Dense(30)(merge)
output1 = Dense(13)(output1)
output1 = Dense(3)(output1)

output2 = Dense(13)(merge)
output2 = Dense(33)(output2)
output2 = Dense(3)(output2)

output3 = Dense(13)(merge)
output3 = Dense(33)(output3)
output3 = Dense(3)(output3)

output4 = Dense(13)(merge)
output4 = Dense(33)(output4)
output4 = Dense(3)(output4)

model = Model(inputs=[input1, input2, input3], outputs=[output1, output2, output3, output4])

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train, y3_train, y4_train], epochs=100, batch_size=1, validation_data=([x1_val, x2_val, x3_val],[y1_val, y2_val, y3_val, y4_val]))

mse = model.evaluate([x1_test, x2_test, x3_test], [y1_test, y2_test, y3_test, y4_test], batch_size=1)

print('loss(mse) : ', mse)
print('loss(mse) : ', mse[0])
print('loss(mse) : ', mse[1])
print('loss(mse) : ', mse[2])
print('loss(mse) : ', mse[3])
print('loss(mse) : ', mse[4])

y1_predict, y2_predict = model.predict([x1_test, x2_test])
print('PREDICT : ', y1_predict, y2_predict)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

RMSE1 = RMSE(y1_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)
print('RMSE1 : ', RMSE1)
print('RMSE2 : ', RMSE2)
print('avgRMSE : ', (RMSE1+RMSE2)/2)

from sklearn.metrics import r2_score
def R2(y_test, y_predict):
    return r2_score(y_test, y_predict)

R2_1 = R2(y1_test, y1_predict)
R2_2 = R2(y2_test, y2_predict)

print('R2_1 :', R2_1)
print('R2_2 :', R2_2)
print('avgR2 :', (R2_1+R2_2)/2)




