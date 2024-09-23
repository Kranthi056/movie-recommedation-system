import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.impute import SimpleImputer

# Function to load and preprocess the CSV file
def load_movies():
    df = pd.read_csv('movie_dataset.csv')  # Using the specified CSV file

    # Standardizing column names
    df.columns = [col.capitalize() for col in df.columns]

    # Ensure 'Genres' is a string and flatten if necessary
    df['Genres'] = df['Genres'].apply(lambda x: ', '.join(eval(x)) if isinstance(x, str) and ',' in x else x)
    df['Genres'] = df['Genres'].astype(str)

    # Handling missing values using Imputer for numeric and categorical data
    imputer_numeric = SimpleImputer(strategy='mean')
    imputer_categorical = SimpleImputer(strategy='most_frequent')

    # Impute Vote_average (numeric)
    df['Vote_average'] = imputer_numeric.fit_transform(df[['Vote_average']])

    # Impute categorical columns safely
    for column in ['Genres', 'Title', 'Director', 'Runtime', 'Release_date']:
        if column in df.columns:
            df[column] = imputer_categorical.fit_transform(df[[column]]).ravel()

    return df

# Collaborative filtering model based on 'vote_average'
def build_collaborative_filtering_model(df):
    ratings_matrix = df.pivot_table(index='Title', values='Vote_average', aggfunc=np.mean)
    ratings_matrix = ratings_matrix.fillna(0)

    # Standardize ratings
    scaler = StandardScaler()
    ratings_matrix_scaled = scaler.fit_transform(ratings_matrix)

    # Compute similarity
    movie_similarity = cosine_similarity(ratings_matrix_scaled)

    # Create similarity DataFrame
    movie_similarity_df = pd.DataFrame(movie_similarity, index=ratings_matrix.index, columns=ratings_matrix.index)
    return movie_similarity_df

# Function to get recommendations
def get_movie_recommendations(movie_title, similarity_df, top_n=10):
    if movie_title not in similarity_df.index:
        return []
    similarity_scores = similarity_df[movie_title]
    top_movie_indices = similarity_scores.sort_values(ascending=False)[1:top_n + 1].index
    return top_movie_indices.tolist()

# Loading movie data
movies_df = load_movies()

# Build collaborative filtering model
movie_similarity_df = build_collaborative_filtering_model(movies_df)

# Defining fixed genre options
fixed_genre_options = ['Romance', 'Drama', 'Thriller', 'Comedy', 'Action', 'Horror', 'Adventure']

# CSS for styling the app
st.markdown(
    """
    <style>
    body {
        background-color: #121212;
        color: #ffffff;
    }
    [data-testid="stSidebar"] {
        background-color: #add8e6;
        color: #000000;
        position: fixed;
        width: 300px;
    }
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {
        color: #003366;
    }
    .reset-button {
        background-color: #ff5733;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px;
        cursor: pointer;
        transition: background-color 0.3s ease, box-shadow 0.3s ease;
        box-shadow: 0 0 10px rgba(255, 87, 51, 0.8);
        display: block;
        width: 100%;
    }
    .reset-button:hover {
        background-color: #c70039;
        box-shadow: 0 0 15px rgba(255, 87, 51, 1);
    }
    .movie-box {
        background-color: #2e2e2e;
        border-radius: 10px;
        padding: 15px;
        margin: 10px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
        text-align: left;
        height: 350px;
        width: 100%;
    }
    .movie-title {
        font-size: 20px;
        color: #ffcc00;
    }
    .movie-details {
        color: #d3d3d3;
    }
    .title-main {
        font-size: 36px;
        text-align: center;
        margin: 20px 0;
        color: #ffcc00;
    }
    .stApp {
        background-color: #121212 !important;
    }
    [data-testid="stSidebarResizeHandle"] {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True
)

# Sidebar for user input filters
with st.sidebar:
    st.header('☰ Menu')

    # Genre filter setup
    genre_expander = st.expander("Genre", expanded=False)
    with genre_expander:
        for genre in fixed_genre_options:
            if st.checkbox(genre, value=(st.session_state.get('selected_genre') == genre)):
                st.session_state['selected_genre'] = genre
            else:
                if st.session_state.get('selected_genre') == genre:
                    st.session_state['selected_genre'] = None

    # Director search filter
    director_expander = st.expander("Director", expanded=False)
    with director_expander:
        st.session_state['director_search'] = st.text_input('Search Director', value=st.session_state.get('director_search', ''))

    # Rating filter setup
    st.subheader('✰ Rating')
    rating_checkbox = st.checkbox('Show Ratings')

    if rating_checkbox:
        for rating in range(1, 11):
            if st.checkbox(f'{rating}', key=f'rating_{rating}'):
                if rating not in st.session_state.get('selected_ratings', []):
                    st.session_state['selected_ratings'].append(rating)
            else:
                if rating in st.session_state['selected_ratings']:
                    st.session_state['selected_ratings'].remove(rating)

    # Button to reset filters
    if st.button('🔄 Reset Filters', key='reset_filters', help="Click to reset all filters"):
        st.session_state['selected_genre'] = None
        st.session_state['director_search'] = ''
        st.session_state['selected_ratings'] = []
        st.session_state['movie_search'] = ''
        st.session_state['load_count'] = 9

# Setting up the Streamlit page title
st.markdown('<div class="title-main">🎬 Movie Recommendation System</div>', unsafe_allow_html=True)

# Reset session state variables
if 'load_count' not in st.session_state:
    st.session_state['load_count'] = 9
if 'selected_genre' not in st.session_state:
    st.session_state['selected_genre'] = None
if 'director_search' not in st.session_state:
    st.session_state['director_search'] = ''
if 'selected_ratings' not in st.session_state:
    st.session_state['selected_ratings'] = []
if 'movie_search' not in st.session_state:
    st.session_state['movie_search'] = ''

# Movie name search input
st.session_state['movie_search'] = st.text_input(
    '🔍 Search Movie Name',
    value=st.session_state['movie_search'],
    placeholder="Enter movie name...",
    label_visibility="collapsed"
)

# Filtered movie selection
filtered_movies = movies_df

# Applying filters
if st.session_state['selected_genre']:
    filtered_movies = filtered_movies[filtered_movies['Genres'].apply(lambda x: st.session_state['selected_genre'].lower() in x.lower())]

if st.session_state['director_search']:
    filtered_movies = filtered_movies[filtered_movies['Director'].str.contains(st.session_state['director_search'], case=False)]

if st.session_state['selected_ratings']:
    selected_rating_prefixes = [str(rating) for rating in st.session_state['selected_ratings']]
    filtered_movies = filtered_movies[filtered_movies['Vote_average'].astype(str).str.startswith(tuple(selected_rating_prefixes))]

if st.session_state['movie_search']:
    filtered_movies = filtered_movies[filtered_movies['Title'].str.contains(st.session_state['movie_search'], case=False)]

# Displaying movie recommendations
if st.session_state['movie_search']:
    st.subheader(f"Similar Movies to '{st.session_state['movie_search']}'")
    recommended_movies = get_movie_recommendations(st.session_state['movie_search'], movie_similarity_df)

    if not any(movies_df['Title'].str.contains(st.session_state['movie_search'], case=False)):
        st.write("Not Found")
    elif recommended_movies:
        for movie in recommended_movies:
            st.write(f"- {movie}")
    else:
        st.write("No similar movies found.")

# Displaying filtered movies in a grid
if len(filtered_movies) == 0:
    st.write("No data found.")
else:
    for i in range(0, min(len(filtered_movies), st.session_state['load_count']), 3):
        cols = st.columns(3)
        for col, movie in zip(cols, filtered_movies.iloc[i:i + 3].itertuples()):
            # Check if critical fields are present
            if (pd.isna(movie.Genres) or movie.Genres == '' or
                (movie.Runtime == 0) or (movie.Vote_average == 0)):
                st.write("No data found.")
            else:
                with col:
                    st.markdown(
                        f"<div class='movie-box'>"
                        f"<div class='movie-header'>"
                        f"<span class='movie-title'>{movie.Title}</span>"
                        f"</div>"
                        f"<div class='movie-details'>"
                        f"<span>Director: {movie.Director}</span><br>"
                        f"<span>Runtime: {movie.Runtime} min</span><br>"
                        f"<span>Genre: {movie.Genres}</span><br>"
                        f"<span>Release Date: {movie.Release_date}</span><br>"
                        f"<span>Vote Average: {movie.Vote_average}</span>"
                        f"</div>"
                        "</div>",
                        unsafe_allow_html=True
                    )

# Load More button
if st.session_state['load_count'] < len(filtered_movies):
    if st.button('Load More'):
        st.session_state['load_count'] += 9
