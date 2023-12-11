from sklearn.linear_model import LinearRegression
import pandas as pd

# Load the weather data
data = pd.read_csv('weather_data.csv')

# Prepare the data
X = data[['temperature', 'humidity', 'wind_speed']]
y = data['weather_condition']

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
new_data = pd.DataFrame([[25, 70, 10]], columns=['temperature', 'humidity', 'wind_speed'])
prediction = model.predict(new_data)

print(prediction)
