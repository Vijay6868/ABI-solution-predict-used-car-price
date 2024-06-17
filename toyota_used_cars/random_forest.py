import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'random_forest_model.pkl')

# Save the dummies columns
dummies_columns = dummies.columns.tolist()
joblib.dump(dummies_columns, 'dummies_columns.pkl')

# Predictions
y_pred = model.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"Random Forest Model:")
print(f"  Mean Absolute Error (MAE): {mae}")
print(f"  Mean Squared Error (MSE): {mse}")
print(f"  Root Mean Squared Error (RMSE): {rmse}")
print(f"  R-squared (R²): {r2}")

# Save the results for the dashboard
X_test['Car ID'] = X_test.index.map(lambda i: f"Car_{i}")
results = X_test.copy()
results['Predicted Price'] = y_pred
results['Actual Price'] = y_test.reset_index(drop=True)

results.to_csv('predicted_prices.csv', index=False)

