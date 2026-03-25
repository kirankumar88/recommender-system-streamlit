import streamlit as st
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Animation Movie Recommender", layout="wide")

st.title("Animation Movie Recommendation System")

# Load data
@st.cache_data
def load_data():
    data_path = os.path.join("data", "data.csv")
    df = pd.read_csv(data_path)

    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['episodes'] = pd.to_numeric(df['episodes'], errors='coerce')

    df['rating'] = df['rating'].fillna(0)
    df['episodes'] = df['episodes'].fillna(0)
    df['genre'] = df['genre'].fillna("Unknown")

    df = df.reset_index(drop=True)

    return df

df = load_data()

st.write("Dataset loaded:", df.shape)

# Build similarity
@st.cache_data
def build_similarity(df):
    features = df[['rating', 'episodes']]
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    similarity_matrix = cosine_similarity(features_scaled)
    return similarity_matrix

similarity_matrix = build_similarity(df)

st.write("Similarity matrix shape:", similarity_matrix.shape)

# Dropdown
anime_list = df['name'].dropna().unique()
selected_anime = st.selectbox("Select Animation Movie", anime_list)

st.write("Selected:", selected_anime)

# Recommendation
try:
    idx = df[df['name'] == selected_anime].index[0]
    st.write("Index:", idx)

    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]

    anime_indices = [i[0] for i in sim_scores]
    recommendations = df['name'].iloc[anime_indices]

    st.write("Recommendations:")
    st.write(recommendations)

except Exception as e:
    st.error("Error occurred:")
    st.write(e)