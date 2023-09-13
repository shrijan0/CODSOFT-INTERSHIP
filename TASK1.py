import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

# Function to load and preprocess data and train a classifier
def train_genre_classifier():
    # Load and Parse the Data
    with open('train_data.txt', 'r', encoding='utf-8') as file:
        data_lines = file.readlines()

    data = []
    for line in data_lines:
        parts = line.strip().split(":::")
        if len(parts) == 4:
            movie_title = parts[0].strip()
            release_year = parts[1].strip()
            genre = parts[2].strip()
            plot_summary = parts[3].strip()
            data.append([movie_title, release_year, genre, plot_summary])

    # Create a DataFrame
    df = pd.DataFrame(data, columns=['Movie_Title', 'Release_Year', 'Genre', 'Plot_Summary'])

    # Text Preprocessing (you can customize this further)
    df['Plot_Summary'] = df['Plot_Summary'].str.lower()  # Convert to lowercase

    # Feature Extraction using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = tfidf_vectorizer.fit_transform(df['Plot_Summary'])

    # Label Encoding (if needed)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['Genre'])

    # Train a Naive Bayes classifier
    clf = MultinomialNB()
    clf.fit(X, y)

    return clf, tfidf_vectorizer, label_encoder

# Function to predict genre from a user-provided plot summary
def predict_genre(movie_plot_summary, clf, tfidf_vectorizer, label_encoder):
    # Preprocess the input plot summary
    movie_plot_summary = movie_plot_summary.lower()

    # Transform the plot summary using the TF-IDF vectorizer
    X_input = tfidf_vectorizer.transform([movie_plot_summary])

    # Predict the genre
    predicted_label = clf.predict(X_input)

    # Decode the predicted label back to the genre name
    predicted_genre = label_encoder.inverse_transform(predicted_label)

    return predicted_genre

# Train the genre classifier
classifier, vectorizer, encoder = train_genre_classifier()

# Input from the user
user_input = input("Enter a movie plot summary: ")

# Predict the genre
predicted_genre = predict_genre(user_input, classifier, vectorizer, encoder)
print(f'Predicted Genre: {predicted_genre[0]}')
