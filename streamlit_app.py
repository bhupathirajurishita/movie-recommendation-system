
import ast
import random
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background: #0b0d0f;
    color: #f5f5f5;
}
.block-container {
    max-width: 900px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 8px;
}
.subtitle {
    text-align: center;
    color: #9aa3ad;
    font-size: 16px;
    margin-bottom: 35px;
}
div[data-baseweb="select"] > div {
    background: #12161a !important;
    border-color: #303840 !important;
    color: white !important;
}
.stButton > button {
    width: 100%;
    height: 50px;
    border-radius: 10px;
    font-weight: 700;
    border: 1px solid #293139;
}
.stButton > button:hover {
    border-color: #21c78a;
}
.movie-card {
    background: #12161a;
    border: 1px solid #293139;
    border-radius: 15px;
    padding: 20px;
    margin-top: 12px;
}
.movie-number {
    color: #21c78a;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1.5px;
}
.movie-title {
    font-size: 21px;
    font-weight: 700;
    margin-top: 5px;
}
.similarity {
    color: #7f8993;
    font-size: 13px;
    margin-top: 7px;
}
.footer {
    text-align: center;
    color: #68727c;
    font-size: 12px;
    margin-top: 45px;
}
</style>
""", unsafe_allow_html=True)


# Public copies of the same TMDB 5000 CSV files.
# This avoids storing the large CSV files in the GitHub repository.
MOVIES_URL = (
    "https://raw.githubusercontent.com/harshitcodes/"
    "tmdb_movie_data_analysis/master/tmdb-5000-movie-dataset/"
    "tmdb_5000_movies.csv"
)
CREDITS_URL = (
    "https://raw.githubusercontent.com/harshitcodes/"
    "tmdb_movie_data_analysis/master/tmdb-5000-movie-dataset/"
    "tmdb_5000_credits.csv"
)


def parse_names(value):
    try:
        return [item["name"] for item in ast.literal_eval(value)]
    except Exception:
        return []


def get_director(value):
    try:
        crew = ast.literal_eval(value)
        for item in crew:
            if item.get("job") == "Director":
                return item.get("name", "")
    except Exception:
        pass
    return ""


@st.cache_data
def load_data():
    local_movies = Path("tmdb_5000_movies.csv")
    local_credits = Path("tmdb_5000_credits.csv")

    if local_movies.exists() and local_credits.exists():
        movies = pd.read_csv(local_movies)
        credits = pd.read_csv(local_credits)
    else:
        movies = pd.read_csv(MOVIES_URL)
        credits = pd.read_csv(CREDITS_URL)

    return movies, credits


@st.cache_resource
def load_recommender():
    movies, credits = load_data()

    movies = movies.merge(
        credits,
        left_on="id",
        right_on="movie_id"
    )

    movies = movies[
        ["title_x", "overview", "genres", "keywords", "cast", "crew"]
    ]

    movies.rename(columns={"title_x": "title"}, inplace=True)

    movies.dropna(inplace=True)

    movies["genres"] = movies["genres"].apply(parse_names)
    movies["keywords"] = movies["keywords"].apply(parse_names)
    movies["cast"] = movies["cast"].apply(
        lambda x: parse_names(x)[:3]
    )
    movies["crew"] = movies["crew"].apply(get_director)

    movies["overview"] = movies["overview"].apply(lambda x: x.split())
    movies["genres"] = movies["genres"].apply(
        lambda x: [i.replace(" ", "") for i in x]
    )
    movies["keywords"] = movies["keywords"].apply(
        lambda x: [i.replace(" ", "") for i in x]
    )
    movies["cast"] = movies["cast"].apply(
        lambda x: [i.replace(" ", "") for i in x]
    )
    movies["crew"] = movies["crew"].apply(
        lambda x: x.replace(" ", "")
    )

    movies["tags"] = (
        movies["overview"]
        + movies["genres"]
        + movies["keywords"]
        + movies["cast"]
        + movies["crew"].apply(lambda x: [x])
    )

    new_df = movies[["title", "tags"]].copy()
    new_df["tags"] = new_df["tags"].apply(lambda x: " ".join(x))
    new_df["tags"] = new_df["tags"].str.lower()

    vectorizer = CountVectorizer(
        max_features=5000,
        stop_words="english"
    )

    vectors = vectorizer.fit_transform(new_df["tags"])
    similarity = cosine_similarity(vectors)

    return new_df, similarity


with st.spinner("🎬 Loading the movie library..."):
    movies, similarity = load_recommender()


st.markdown(
    '<div class="title">🎬 MOVIE MATCH</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Pick a movie. I\'ll find five that match its vibe.</div>',
    unsafe_allow_html=True
)

selected_movie = st.selectbox(
    "Choose a movie",
    movies["title"].tolist()
)

st.write("")

if st.button("✨ Get Recommendations", use_container_width=True):

    movie_index = movies[
        movies["title"] == selected_movie
    ].index[0]

    distances = similarity[movie_index]

    recommendations = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    reactions = [
        "🎬 Your next movie night is sorted.",
        "🍿 These might be your kind of movies.",
        "👀 The algorithm has spoken.",
        "✨ Five movies matching your vibe.",
        "🎥 Consider this your watchlist upgrade."
    ]

    st.markdown(
        f"""
        <div style="
            background:#12161a;
            border:1px solid #293139;
            border-radius:15px;
            padding:18px;
            margin-top:25px;
            text-align:center;
            color:#21c78a;
            font-weight:700;
        ">
            {random.choice(reactions)}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🎞️ Recommended For You")

    for rank, (index, score) in enumerate(recommendations, 1):

        title = movies.iloc[index]["title"]

        st.markdown(
            f"""
            <div class="movie-card">
                <div class="movie-number">MATCH {rank}</div>
                <div class="movie-title">{title}</div>
                <div class="similarity">
                    Similarity score · {score:.3f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown(
    """
    <div class="footer">
        Built with Python · Pandas · Scikit-learn · Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
