from sklearn.linear_model import LinearRegression
import pandas as pd

# Load the weather data
data = pd.read_csv('weather_data.csv')

# Prepare the data
X = data[['temperature', 'humidity', 'wind_speed']]
# Map the weather conditions to integers
weather_conditions = data['weather_condition'].unique()
weather_conditions_dict = {weather_conditions[i]: i for i in range(len(weather_conditions))}
data['weather_condition'] = data['weather_condition'].map(weather_conditions_dict)

# Now 'weather_condition' is a numeric column, and you can use it as your target variable

y = data['weather_condition']

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
new_data = pd.DataFrame([[25, 70, 10]], columns=['temperature', 'humidity', 'wind_speed'])
prediction = model.predict(new_data)

print(prediction)
