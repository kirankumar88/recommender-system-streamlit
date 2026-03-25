import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Animation_movie_Recommender", layout="wide")

# ---------------------------
# Load Data
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    return df

df = load_data()

# ---------------------------
# Data Cleaning
# ---------------------------
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['episodes'] = pd.to_numeric(df['episodes'], errors='coerce')

df['rating'] = df['rating'].fillna(0)
df['episodes'] = df['episodes'].fillna(0)

# ---------------------------
# Feature Scaling
# ---------------------------
features = df[['rating', 'episodes']]

scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

similarity_matrix = cosine_similarity(features_scaled)

# ---------------------------
# Recommendation Function
# ---------------------------
def recommend(anime_name):
    idx = df[df['name'] == anime_name].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]

    recs = []
    for i in scores:
        recs.append((df.iloc[i[0]]['name'], round(i[1], 3)))
    return recs

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("Animation movie Recommender")
st.sidebar.write("Content-Based Recommendation System")
st.sidebar.write("Using Cosine Similarity")

st.sidebar.subheader("Top Rated Animation movies")
top_anime = df.sort_values(by="rating", ascending=False).head(5)
st.sidebar.write(top_anime[['name', 'rating']])

# ---------------------------
# Main Page
# ---------------------------
st.title("Animation Movie Recommender")
st.write("Select an animation to get similar recommendations")

anime_list = df['name'].dropna().unique()
selected_anime = st.selectbox("Select Animation", anime_list)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Selected Animation Details")
    anime_details = df[df['name'] == selected_anime][['genre', 'rating', 'episodes']]
    st.dataframe(anime_details)

with col2:
    if st.button("Recommend"):
        recommendations = recommend(selected_anime)
        st.subheader("Recommended Animations Based on Similarity")

        for anime, score in recommendations:
            st.success(f"{anime}  | Similarity Score: {score}")