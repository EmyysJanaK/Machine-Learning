from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load the weather data
data = pd.read_csv('weather_data.csv')

# Prepare the data
X = data[['temperature', 'humidity', 'wind_speed']]
y = data['weather_condition']

# Encode the categorical labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
new_data = pd.DataFrame([[25, 70, 10]], columns=['temperature', 'humidity', 'wind_speed'])
new_data_encoded = label_encoder.transform(new_data)
prediction = model.predict(new_data_encoded)

print(prediction)
