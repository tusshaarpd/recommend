import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample Movie Dataset
data = {
    'Movie': ["Inception", "Interstellar", "The Dark Knight", "Memento", "The Prestige"],
    'Description': [
        "A thief who enters the dreams of others to steal secrets.",
        "A group of explorers travel through a wormhole in space.",
        "Batman faces the Joker, a criminal mastermind.",
        "A man with short-term memory loss seeks revenge.",
        "Two magicians engage in a dangerous rivalry."
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Compute TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['Description'])

# Compute Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get recommendations
def get_recommendations(movie, df, cosine_sim):
    if movie not in df['Movie'].values:
        return ["Movie not found"]
    
    idx = df[df['Movie'] == movie].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:4]  # Top 3
    recommendations = [df['Movie'].iloc[i[0]] for i in scores]
    return recommendations

# Streamlit UI
st.title("Movie Recommendation System")
st.write("Select a movie to get similar recommendations.")

selected_movie = st.selectbox("Choose a Movie:", df['Movie'])

if st.button("Recommend"):
    recommendations = get_recommendations(selected_movie, df, cosine_sim)
    st.write("### Recommended Movies:")
    for rec in recommendations:
        st.write(f"- {rec}")
