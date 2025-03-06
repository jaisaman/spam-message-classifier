import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score



# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Load the dataset
df = pd.read_csv("spam.csv", encoding="latin-1")[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels to numeric values (spam = 1, not spam = 0)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    words = word_tokenize(text)  # Tokenize
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    return " ".join(words)

# Apply preprocessing to each message
df['message'] = df['message'].apply(preprocess_text)

# Convert text to numeric vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Test the model with new messages
def predict_spam(message):
    message = preprocess_text(message)
    message_vec = vectorizer.transform([message])  # Convert the message to a vector
    prediction = model.predict(message_vec)  # Make a prediction
    return "Spam" if prediction[0] == 1 else "Not Spam"

# Test the model with some example messages
print(predict_spam("Congratulations! You've won a free iPhone!"))
print(predict_spam("Elon Musk is Giving Away FREE Bitcoin!"))
