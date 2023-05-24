# Reinforcement Learning for Dynamic Model Weight Assignment

This repository contains the code for our research on using reinforcement learning for dynamic model weight assignment in financial markets.

## Overview

We propose a novel approach for assigning weights to different predictive models in a financial market setting. Our approach leverages reinforcement learning, specifically the Advantage Actor-Critic (A2C) algorithm, to dynamically adjust the weights assigned to different models based on their performance.

## Repository Structure

- `rfr.py`: Contains the code for training the Random Forest Regression model.
- `lstm.py`: Contains the code for training the LSTM model.
- `lr.py`: Contains the code for training the Linear Regression model.
- `sentiment_analysis.py`: Contains the code for performing sentiment analysis.
- `dynamic_assignment.py`: Contains the code for the reinforcement learning agent that dynamically assigns weights to the models.

## Installation

To install the necessary dependencies, you need to have the following Python libraries:

- numpy
- pandas
- sklearn
- keras
- tensorflow
- stable_baselines3
- matplotlib
- seaborn
- textblob

You can install these using pip:

```bash
pip install numpy pandas sklearn keras tensorflow stable_baselines3 matplotlib seaborn textblob
