import pandas as pd
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
from gym import spaces
import pickle
from tensorflow.keras.models import load_model
from sklearn.linear_model import LinearRegression
from textblob import TextBlob
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the models
rfr_model = pickle.load(open('rfr_model.pkl', 'rb'))
lstm_model = load_model('lstm_model.h5')
linear_regression_model = pickle.load(open('lr_model.pkl', 'rb'))

# Load the data
data = pd.read_csv('stock_data.csv')

# Convert the 'Date' column to datetime format and then to timestamp
data['Date'] = pd.to_datetime(data['Date'])
data['Date'] = data['Date'].values.astype(float)

class WeightAssignmentEnv(gym.Env):
    def __init__(self, models, data):
        super(WeightAssignmentEnv, self).__init__()
        self.models = models
        self.data = data
        self.reward_range = (-np.inf, np.inf)
        self.action_space = spaces.Box(low=0, high=1, shape=(len(models),))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(data.shape[1],))
        self.current_step = 0

    def step(self, action):
        self.current_step += 1
        if self.current_step >= len(self.data):
            done = True
        else:
            done = False
        obs = self.data.iloc[self.current_step]
        reward = self.calculate_reward(action, obs)
        return obs, reward, done, {}

    def reset(self):
        self.current_step = 0
        return self.data.iloc[self.current_step]

    def calculate_reward(self, action, obs):
        predictions = []
        for model, weight in zip(self.models, action):
            if isinstance(model, LinearRegression):
                prediction = model.predict(obs[0].reshape(-1, 1))
            elif isinstance(model, type(lstm_model)):
                prediction = model.predict(obs.reshape(1, 1, 1))
            else:
                prediction = model.predict(obs.values.reshape(1, -1))
            predictions.append(prediction * weight)
        reward = sum(predictions)
        return reward
    def __init__(self, models, data):
        super(WeightAssignmentEnv, self).__init__()
        self.models = models
        self.data = data
        self.reward_range = (-np.inf, np.inf)
        self.action_space = spaces.Box(low=0, high=1, shape=(len(models),))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(data.shape[1],))
        self.current_step = 0

    def step(self, action):
        self.current_step += 1
        if self.current_step >= len(self.data):
            done = True
        else:
            done = False
        obs = self.data.iloc[self.current_step]
        reward = self.calculate_reward(action, obs)
        return obs, reward, done, {}

    def reset(self):
        self.current_step = 0
        return self.data.iloc[self.current_step]

    def calculate_reward(self, action, obs):
        predictions = []
        for model, weight in zip(self.models, action):
            if isinstance(model, LinearRegression):
                prediction = model.predict(obs[0].reshape(-1, 1))
            elif isinstance(model, type(lstm_model)):
                prediction = model.predict(obs.reshape(1, 1, 1))
            else:
                prediction = model.predict(obs.values.reshape(1, -1))
            predictions.append(prediction * weight)
        reward = sum(predictions)
        return reward
    def __init__(self, models, data):
        super(WeightAssignmentEnv, self).__init__()
        self.models = models
        self.data = data
        self.reward_range = (-np.inf, np.inf)
        self.action_space = spaces.Box(low=0, high=1, shape=(len(models),))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(data.shape[1],))
        self.current_step = 0

    def step(self, action):
        self.current_step += 1
        if self.current_step >= len(self.data):
            done = True
        else:
            done = False
        obs = self.data[self.current_step]
        reward = self.calculate_reward(action, obs)
        return obs, reward, done, {}

    def reset(self):
        self.current_step = 0
        return self.data[self.current_step]

    def calculate_reward(self, action, obs):
        predictions = []
        for model, weight in zip(self.models, action):
            if isinstance(model, LinearRegression):
                prediction = model.predict(obs[0].reshape(-1, 1))
            elif isinstance(model, type(lstm_model)):
                prediction = model.predict(obs.reshape(1, 1, 1))
            else:
                prediction = model.predict(obs.values.reshape(1, -1))
            predictions.append(prediction * weight)
        reward = sum(predictions)
        return reward

# Create the environment
env = DummyVecEnv([lambda: WeightAssignmentEnv([rfr_model, lstm_model, linear_regression_model], data)])

# Initialize the A2C model
model = A2C('MlpPolicy', env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Evaluate the model
obs = env.reset()
done = False
rewards = []
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    rewards.append(reward)
    print(f'Reward: {reward}')

# Plot the rewards
plt.plot(rewards)
plt.xlabel('Step')
plt.ylabel('Reward')
plt.title('Rewards over time')
plt.show()

# Calculate performance metrics
mse = mean_squared_error(data['Close'][1:], rewards)
mae = mean_absolute_error(data['Close'][1:], rewards)
r2 = r2_score(data['Close'][1:], rewards)

print(f'MSE: {mse}')
print(f'MAE: {mae}')
print(f'R2 Score: {r2}')
