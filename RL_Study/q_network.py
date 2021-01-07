import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def one_hot(x):
    return np.identity(16)[x:x+1]

env = gym.make('FrozenLake-v0')

input_size = env.observation_space.n
output_size = env.action_space.n
learning_rate = 0.1

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

X = Input(shape=(input_size,))
hidden1 = Dense(32, activation='elu')(X)
#hidden2 = Dense(8, activation='elu')(hidden1)
#hidden3 = Dense(4, activation='elu')(hidden2)
Qpred = Dense(output_size, activation='elu')(hidden1)

model = Model(X, Qpred)

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

dis = .99
num_episodes = 2000

rList = []

for i in range(num_episodes):
    s = env.reset()
    e = 1. / ((i/50)+10)
    rAll = 0
    done = False
    local_loss = []

    while not done:
        Qs = model.predict(one_hot(s))
        if np.random.rand(1) < e :
            a = env.action_space.sample()
        else :
            a = np.argmax(Qs)

        s1, reward, done, _ = env.step(a)
        if done:
            Qs[0, a] = reward
        else:
            Qs1 = model.predict(one_hot(s1))
            Qs[0, a] = reward + dis * np.max(Qs1)

        model.fit(one_hot(s), Qs, batch_size=1, epochs=1)

        rAll += reward
        s=s1
    rList.append(rAll)
    print(i)

print("Percent of successful episodes:" + str(sum(rList)/num_episodes) + "%")
plt.bar(range(len(rList)), rList, color = "blue")
plt.show()
















