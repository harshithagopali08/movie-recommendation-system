import streamlit as st
import pickle
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load movie data
movies = pickle.load(open('movies.pkl', 'rb'))


# Recreate similarity matrix
cv = CountVectorizer(max_features=5000, stop_words='english')

vectors = cv.fit_transform(movies['tags']).toarray()

similarity = cosine_similarity(vectors)


# Streamlit title
st.title("Movie Recommendation System")


# Recommendation function
def recommend(movie):

    movie_index = movies[movies['title'] == movie].index[0]

    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []

    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].title)

    return recommended_movies


# Dropdown
selected_movie = st.selectbox(
    "Select a movie",
    movies['title'].values
)


# Recommend button
if st.button('Recommend'):

    recommendations = recommend(selected_movie)

    for movie in recommendations:
        st.write(movie)
