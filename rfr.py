from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
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
rfr_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rfr_model.fit(X_train, y_train)

# Save the model
pickle.dump(rfr_model, open('rfr_model.pkl', 'wb'))
