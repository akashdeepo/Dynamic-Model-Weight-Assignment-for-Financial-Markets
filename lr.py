from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import pandas as pd
from datetime import datetime

# Load your data
data = pd.read_csv('stock_data.csv')

# Convert the 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Convert the 'Date' column to ordinal values
data['Date'] = data['Date'].apply(lambda x: x.toordinal())

# Split the data into features and target
X = data.drop('Close', axis=1)
y = data['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
lr_model = LinearRegression()

# Train the model
lr_model.fit(X_train, y_train)

# Save the model
pickle.dump(lr_model, open('lr_model.pkl', 'wb'))
