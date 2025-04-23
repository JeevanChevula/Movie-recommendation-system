Movie Recommendation System
This project builds a movie recommendation system for Telugu movies based on genre similarity using cosine similarity and TF-IDF vectorization.
Prerequisites

Python 3.x
Google Colab environment
Google Drive access
Required libraries: pandas, scikit-learn

Installation

Install required packages:!pip install pandas scikit-learn


Mount Google Drive:from google.colab import drive
drive.mount('/content/drive')



Dataset

File: TeluguMovies_dataset.csv
Path: /content/drive/MyDrive/Colab Notebooks/Datasets/TeluguMovies_dataset.csv
Columns: Must include Movie and Genre

Usage

Place the dataset in the specified Google Drive path.
Run the script to:
Load and preprocess the dataset.
Clean data by removing rows with missing Genre.
Convert genres to numerical features using TF-IDF.
Compute cosine similarity between movies.
Recommend similar movies based on a given movie name.



Code
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
data_path = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Datasets/TeluguMovies_dataset.csv")

# Clean data
data = data_path.dropna(subset=["Genre"])
data["Genre"] = data["Genre"].str.replace('|', ' ')

# Convert genres to TF-IDF features
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(data["Genre"])

# Calculate cosine similarity
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation function
def recommend_movies(movie_name, data, similarity_matrix):
    if movie_name not in data['Movie'].values:
        return f"Movie '{movie_name}' not found in the dataset. Please search for other movies"
    idx = data[data['Movie'] == movie_name].index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_movies = [data.iloc[i[0]]['Movie'] for i in sim_scores if data.iloc[i[0]]['Movie'] != movie_name][:11]
    return top_movies

# Example usage
movie_name = "Jalsa"
recommended_movies = recommend_movies(movie_name, data, similarity_matrix)
print(f"Movies similar to {movie_name}: {recommended_movies}")

Output
For the input movie Jalsa, the system outputs a list of 11 similar Telugu movies based on genre similarity.
Notes

Ensure the dataset path and column names (Movie, Genre) are correct.
The system recommends up to 11 similar movies, excluding the input movie.
If the movie is not found, an error message is returned.
The dataset must be clean and properly formatted for accurate recommendations.

