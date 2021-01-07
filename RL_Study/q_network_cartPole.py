import numpy as np
import tensorflow as tf
import gym

env = gym.make('CartPole-v0')
env._max_episode_steps = 10001

learning_rate = 1e-1
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

X = Input(shape=(input_size,))
hidden1 = Dense(15, activation='elu')(X)
hidden2 = Dense(10, activation='elu')(hidden1)
hidden3 = Dense(12, activation='elu')(hidden2)
Qpred = Dense(output_size, activation='elu')(hidden3)
model = Model(X, Qpred)
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

num_episodes = 4000
dis = 0.9
rList = []

for i in range(num_episodes):
    e = 1. / ((i/50)+1)
    rAll = 0
    step_count = 0
    s = env.reset()
    done = False

    while not done:
        step_count += 1
        x = np.reshape(s, [1, input_size])
        Qs = model.predict(x)
        if np.random.rand(1) < e:
            a = env.action_space.sample()
        else:
            a = np.argmax(Qs)

        s1, reward, done, _ = env.step(a)
        if done:
            Qs[0, a] = -100
        else:
            x1 = np.reshape(s1, [1, input_size])

            Qs1 = model.predict(x1)
            Qs[0, a] = reward + dis * np.max(Qs1)

        model.fit(x = x, y = Qs)
        print(step_count)
        s = s1

    rList.append(step_count)
    print("Episode: {} steps: {}".format(i, step_count))

    if len(rList) > 10 and np.mean(rList[-10:]) > 10000:
        break

observation = env.reset()
reward_sum = 0
while True:
    env.render()

    x = np.reshape(observation, [1, input_size])
    Qs = model.predict(x)
    a = np.argmax(Qs)

    observation, reward, done, _ = env.step(a)
    reward_sum += reward
    if done:
        print("Total score: {}".format(reward_sum))
        break










