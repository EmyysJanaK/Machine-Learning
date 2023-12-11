import pandas as pd #data manipulation and analysis
from sklearn.feature_extraction.text import CountVectorizer #textual data -> numrical data
from sklearn.naive_bayes import MultinomialNB #Multinomial Naive Bayes classifier
from sklearn.model_selection import train_test_split #train_test_split

# Load the dataset and the file contains two columns: "message" and "label".
data = pd.read_csv("spam_dataset.csv")

# Split the dataset into training(80% of the dataset) and testing sets(20% of the dataset)
X_train, X_test, y_train, y_test = train_test_split(data["message"], data["label"], test_size=0.2, train_size=0.8, random_state=42)

# Create a CountVectorizer to convert text into numerical features
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

# Train a Multinomial Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)

# Evaluate the classifier on the testing set
X_test_vectorized = vectorizer.transform(X_test)
accuracy = classifier.score(X_test_vectorized, y_test)
print("Accuracy:", accuracy)
