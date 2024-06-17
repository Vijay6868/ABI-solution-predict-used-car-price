import pandas as pd

# Read the original CSV file
df = pd.read_csv('jap_car_dataset.csv')
print(df.head())
df['State'] = df['State'].str.upper()
df['State'] = df['State'].astype(str)
# Filter the DataFrame to keep only Toyota cars
df_toyota = df[df['Make'] == 'Toyota']


# Drop the 'Make' column
df_toyota.drop('Make', axis=1, inplace=True)

# Save the filtered DataFrame to a new CSV file
df_toyota.to_csv('toyota_cars.csv', index=False)

