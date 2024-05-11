import streamlit as st
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations

# Load the dataset
movies_df = pd.read_csv('movies.csv', nrows=62424)
tags_df = pd.read_csv('tags.csv', nrows=62424)

# Dropping any unused columns
tags_df = tags_df.drop(['userId', 'timestamp'], axis=1).reset_index(drop=True)

# Removing null values from tag
tags_df['tag'] = tags_df['tag'].apply(lambda x: '' if pd.isnull(x) else x)

# Grouping tags with the same movieId
tags_combined_df = tags_df.groupby('movieId')['tag'].agg(lambda x: '|'.join(x)).reset_index()

# Merging tags with movies
merged = pd.merge(movies_df, tags_combined_df, on='movieId', how='left')

# Fill NaN values in 'tag' column with an empty string
merged['tag'] = merged['tag'].fillna('')

# Removing any movies with no tags
merged = merged[merged['tag'] != ""]

# Removing any movies with no genres
merged = merged[merged['genres'] != "(no genres listed)"]

# One-hot encode genres and tags
genres_one_hot = merged['genres'].str.get_dummies()
tags_one_hot = merged['tag'].str.get_dummies()

# Convert the one-hot encoded dataframes to sparse matrices
genres_sparse = csr_matrix(genres_one_hot.values)
tags_sparse = csr_matrix(tags_one_hot.values)

# Sum one-hot encoded genres and tags
genres_sum = genres_sparse.sum(axis=0)
tags_sum = tags_sparse.sum(axis=0)

# Calculate content-based similarity using genres
genres_matrix = merged['genres'].str.get_dummies(sep='|')
genres_similarity = cosine_similarity(genres_matrix)

# Calculate content-based similarity using tags
tags_matrix = merged['tag'].str.get_dummies(sep='|')
tags_similarity = cosine_similarity(tags_matrix)

# Create a mapping between movie titles and indices
title_to_index = {title: idx for idx, title in enumerate(merged['title'])}

# Calculate diversity
pairwise_combinations = list(combinations(merged['title'], 2))
diversity_scores = []
for movie1, movie2 in pairwise_combinations:
    index1 = title_to_index[movie1]
    index2 = title_to_index[movie2]
    diversity_score = 1 - genres_similarity[index1][index2]
    diversity_scores.append(diversity_score)
diversity = sum(diversity_scores) / len(diversity_scores)

# Calculate feature coverage
total_features = set(merged['genres'].str.split('|').sum() + merged['tag'].str.split('|').sum())
recommended_features = set(merged.loc[1:3, 'genres'].str.split('|').sum() + merged.loc[1:3, 'tag'].str.split('|').sum())
feature_coverage = len(recommended_features) / len(total_features)

# Display Streamlit app
st.title('Movie Recommender System')
user_input = st.text_input('Enter a movie title:')
if user_input:
    movie_index = merged[merged['title'].str.lower() == user_input.lower()].index
    if not movie_index.empty:
        movie_index = movie_index[0]
        if movie_index < len(genres_similarity):
            similar_indices = genres_similarity[movie_index].argsort()[-6:-1][::-1]  # Exclude the movie itself
            similar_movie_ids = genres_one_hot.iloc[similar_indices].index
            similar_movies = merged.loc[similar_movie_ids, 'title']
            st.write('Recommendations for', user_input)
            st.write(similar_movies.tolist())
        else:
            st.write("Movie index out of bounds.")
    else:
        st.write("Movie not found.")

st.write(f'Content-Based Similarity (Genres): \n{genres_similarity}\n')
st.write(f'Content-Based Similarity (Tags): \n{tags_similarity}\n')
st.write(f'Feature Coverage: {feature_coverage}\n')
st.write(f'Diversity: {diversity}\n')

