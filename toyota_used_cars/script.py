import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42)
}

# Train and evaluate models
results = pd.DataFrame()
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    
    print(f"{name}:")
    print(f"  Mean Absolute Error (MAE): {mae}")
    print(f"  Mean Squared Error (MSE): {mse}")
    print(f"  Root Mean Squared Error (RMSE): {rmse}")
    print(f"  R-squared (R²): {r2}\n")
    
    # Save results
    model_results = pd.DataFrame({
        'Model': [name] * len(y_test),
        'Actual Price': y_test,
        'Predicted Price': y_pred,
        'Car ID': X_test.index.map(lambda i: f"Car_{i}")
    })
    results = pd.concat([results, model_results], ignore_index=True)

results.to_csv('predicted_prices.csv', index=False)
