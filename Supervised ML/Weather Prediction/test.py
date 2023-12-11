from sklearn.model_selection import train_test_split # Splitting the dataset into training and testing sets
from sklearn.linear_model import LogisticRegression # Regression model
from sklearn.preprocessing import LabelEncoder, StandardScaler # Normalizing labels and standardization.
import pandas as pd # For data manipulation and analysis purposes.

# Load the weather data
data = pd.read_csv('weather_data.csv')

# Prepare the data
X = data[['temperature', 'humidity', 'wind_speed']]
y = data['weather_condition']

# Encode the categorical labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LogisticRegression(max_iter=1000) # Train with 1000 iterations.
model.fit(X_train, y_train)


# Make predictions
new_data = pd.DataFrame([[25, 70, 10]], columns=['temperature', 'humidity', 'wind_speed'])
new_data = scaler.transform(new_data)  # scale the new data in the same way as the training data
prediction = model.predict(new_data)

# Convert the encoded prediction back to the original label
prediction_label = label_encoder.inverse_transform(prediction)

print(prediction_label)