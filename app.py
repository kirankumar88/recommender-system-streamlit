import streamlit as st
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Animation Movie Recommender", layout="wide")

st.title("🎬 Animation Movie Recommendation System")
st.write("Content-Based Recommendation using Cosine Similarity")

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

# Build similarity matrix
@st.cache_data
def build_similarity(df):
    features = df[['rating', 'episodes']]
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    similarity_matrix = cosine_similarity(features_scaled)
    return similarity_matrix

similarity_matrix = build_similarity(df)

# Sidebar
st.sidebar.title("📊 Top Rated Animation Movies")
top_anime = df.sort_values(by="rating", ascending=False).head(5)
st.sidebar.dataframe(top_anime[['name', 'rating']])

# Dropdown using index instead of name
selected_index = st.selectbox(
    "Select Animation Movie",
    df.index,
    format_func=lambda x: df.loc[x, 'name']
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("🎥 Selected Animation Details")
    st.dataframe(df.loc[[selected_index], ['genre', 'rating', 'episodes']])

with col2:
    st.subheader("⭐ Recommended Animation Movies")

    sim_scores = list(enumerate(similarity_matrix[selected_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]

    anime_indices = [i[0] for i in sim_scores]

    for i in anime_indices:
        st.success(df.loc[i, 'name'])