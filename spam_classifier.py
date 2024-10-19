# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
import joblib

# Load Dataset
data = pd.read_csv('spam.csv', encoding='latin-1')  # Load your dataset
print(data.columns)  # Print the columns to check their names

# Keep only necessary columns
data = data[['label', 'text']]  # Update this line based on actual column names
data.columns = ['label', 'message']  # Rename columns for clarity

# Convert Labels to Binary (spam=1, not spam=0)
data['label'] = data['label'].map({'spam': 1, 'ham': 0})

# Preprocess the Messages (remove stopwords, tokenize)
nltk.download('stopwords')  # Download stopwords for text cleaning
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    tokens = [word for word in text.split() if word not in stop_words]  # Remove stopwords
    return ' '.join(tokens)

data['message'] = data['message'].apply(preprocess_text)  # Apply the cleaning

# Extract Features (Bag of Words using CountVectorizer)
vectorizer = CountVectorizer()  # Transform text data into numerical features
X = vectorizer.fit_transform(data['message'])  # Input features (Bag of Words)
y = data['label']  # Output labels

# Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Naive Bayes Classifier
model = MultinomialNB()  # Create Naive Bayes model
model.fit(X_train, y_train)  # Train the model with training data

# Predict and Evaluate the Model
y_pred = model.predict(X_test)  # Predict using test data
accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
print(f'Accuracy: {accuracy * 100:.2f}%')  # Print accuracy
print(classification_report(y_test, y_pred))  # Print detailed classification report

# Save the Model and Vectorizer (Optional)
joblib.dump(model, 'spam_classifier_model.pkl')  # Save model to a file
joblib.dump(vectorizer, 'vectorizer.pkl')  # Save vectorizer to a file

# Optional: Test with a New Email
new_email = ["Congratulations, you won a lottery!"]  # Test input
new_email_vector = vectorizer.transform(new_email)  # Transform new email to vector
prediction = model.predict(new_email_vector)  # Predict if spam or not
print("Spam" if prediction[0] == 1 else "Not Spam")  # Output prediction result
