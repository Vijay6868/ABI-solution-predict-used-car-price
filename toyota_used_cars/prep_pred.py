import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load and process the data
ca_cars = pd.read_csv('ca_toyota_used_cars.csv')

# Add car age as a feature
current_year = 2024
ca_cars['Car Age'] = current_year - ca_cars['Year']
ca_cars.drop('Year', axis=1, inplace=True)

# Encode the model names
dummies = pd.get_dummies(ca_cars.Model)
df = pd.concat([ca_cars, dummies], axis='columns')
df.drop(['Yaris', 'Model'], axis='columns', inplace=True)

# Define features and target
X = df.drop(['Price'], axis='columns')
y = df.Price

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Save the results for the dashboard
results = X_test.copy()
results['Predicted Price'] = y_pred
results['Actual Price'] = y_test.reset_index(drop=True)

results.to_csv('predicted_prices.csv', index=False)
