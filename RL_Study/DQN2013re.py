import numpy as np
import tensorflow as tf
import gym
from collections import deque
import random

env = gym.make('CartPole-v0')
env._max_episode_steps = 10001


input_size = env.observation_space.shape[0]
output_size = env.action_space.n


replay_buffer = deque()
REPLAY_MEMORY = 50000

minibatch_size = 10



from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

X = Input(shape=(input_size,))
hidden1 = Dense(32, activation='elu')(X)
hidden2 = Dense(15, activation='elu')(hidden1)
hidden3 = Dense(8, activation='elu')(hidden2)
Qpred = Dense(output_size, activation='elu')(hidden1)
model = Model(X, Qpred)
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

num_episodes = 4000
dis = 0.7
rList = []

for episode in range(num_episodes):
    e = 1. / ((episode/50)+1)
    rAll = 0
    step_count = 0
    state = env.reset()
    done = False

    while not done:
        step_count += 1

        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            action = np.argmax(model.predict(np.reshape(state, [1, input_size])))

        next_state, reward, done, _ = env.step(action)
        if done:
            reward = -10

        replay_buffer.append((state, action, reward, next_state, done))
        if len(replay_buffer) > REPLAY_MEMORY:
            replay_buffer.popleft()

        state = next_state
        step_count += 1
        if step_count > 10000:
            break

    print("Episode: {} steps: {}".format(episode, step_count))

    if step_count > 10000:
        pass

    if episode % 10 == 1:
        for _ in range(30):
            minibatch = random.sample(replay_buffer, minibatch_size)
            x_stack = np.empty(0).reshape(0, input_size)
            y_stack = np.empty(0).reshape(0, output_size)

            for state, action, reward, next_state, done in minibatch:
                Q = model.predict(np.reshape(state, [1, input_size]))

                if done:
                    Q[0, action] = reward
                else:
                    Q[0, action] = reward + dis*np.max(model.predict(np.reshape(next_state, [1, input_size])))
                x_stack = np.vstack([x_stack, state])
                y_stack = np.vstack([y_stack, Q])

            model.fit(x = x_stack, y = y_stack, epochs = 1, batch_size = 1)

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










