import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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

#results.to_csv('predicted_prices.csv', index=False)

# Generate evaluation graphs
plt.figure(figsize=(14, 7))

# Plot Predicted vs Actual Prices
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices')

# Plot Residuals
plt.subplot(1, 2, 2)
sns.histplot(y_test - y_pred, bins=20, kde=True)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residuals Distribution')

plt.tight_layout()
plt.savefig('evaluation_graphs.png')
plt.show()

# Generate pair plot
# Only include numerical columns in the pair plot
pair_plot_columns = ['Price', 'Car Age', 'Mileage']
sns.pairplot(df[pair_plot_columns], kind='scatter')
plt.suptitle('Pair Plot of Features vs Price', y=1.02)
plt.show()

# Feature Importance Plot
importances = model.feature_importances_
features = X.columns
indices = importances.argsort()[::-1]

plt.figure(figsize=(12, 6))
sns.barplot(x=importances[indices], y=features[indices])
plt.title('Feature Importance in Random Forest Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Residuals vs Fitted Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=y_test - y_pred)
plt.axhline(0, linestyle='--', color='r')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')
plt.show()

# Distribution of Predicted vs Actual Prices
plt.figure(figsize=(10, 6))
sns.histplot(y_test, kde=True, color='blue', label='Actual Price')
sns.histplot(y_pred, kde=True, color='orange', label='Predicted Price')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted vs Actual Prices')
plt.legend()
plt.show()
