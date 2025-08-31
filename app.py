import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class IMDBContentBasedRecommendationSystem:
    def __init__(self):
        self.movies_df = None
        self.qualified_movies = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.indices = None

    def clean_title_text(self, text):
        if pd.isna(text):
            return ""
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text))
        cleaned = re.sub(r'\s+', ' ', cleaned).lower().strip()
        return cleaned

    def load_imdb_data(self, file_path):
        self.movies_df = pd.read_csv(file_path, low_memory=False)
        self.movies_df['overview'] = self.movies_df['overview'].fillna('No description available')
        self.movies_df['genre'] = self.movies_df['genre'].fillna('Unknown')
        self.movies_df['crew'] = self.movies_df['crew'].fillna('Unknown')
        self.movies_df['original_title'] = self.movies_df['orig_title'].copy()
        self.movies_df['orig_title'] = self.movies_df['orig_title'].apply(self.clean_title_text)
        self.movies_df = self.movies_df.drop_duplicates(subset=['orig_title']).reset_index(drop=True)
        self.movies_df['enhanced_content'] = (
            self.movies_df['overview'].astype(str) + ' ' +
            self.movies_df['genre'].astype(str).str.replace('|', ' ') + ' ' +
            self.movies_df['crew'].astype(str)
        )
        self.qualified_movies = self.movies_df.copy()

    def build_content_based_system(self):
        working_df = self.qualified_movies.copy()
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        working_df['enhanced_content'] = working_df['enhanced_content'].fillna('')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(working_df['enhanced_content'])
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
        self.indices = pd.Series(working_df.index, index=working_df['orig_title']).drop_duplicates()
        self.qualified_movies = working_df

    def get_content_recommendations(self, title, n=10):
        if title not in self.indices:
            return None, None
        idx = self.indices[title]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
        movie_indices = [i[0] for i in sim_scores]
        movies = self.qualified_movies.iloc[movie_indices].copy()
        movies['similarity'] = [i[1] for i in sim_scores]
        return self.qualified_movies.loc[idx], movies


# ============================
# Streamlit Terminal-style UI
# ============================
def main():
    st.set_page_config(page_title="IMDB Recommender", layout="wide")
    st.title("🎬 IMDB Movie Recommendation (Terminal Style)")

    uploaded_file = st.sidebar.file_uploader("Upload IMDB dataset (CSV)", type="csv")
    if uploaded_file:
        recommender = IMDBContentBasedRecommendationSystem()
        recommender.load_imdb_data(uploaded_file)
        recommender.build_content_based_system()

        movie_title = st.text_input("Enter a movie title (English):")
        n_recs = st.slider("Number of recommendations", 1, 20, 10)

        if st.button("Get Recommendations"):
            cleaned_title = recommender.clean_title_text(movie_title)
            movie_info, recs = recommender.get_content_recommendations(cleaned_title, n=n_recs)

            if movie_info is None:
                st.error(f"❌ No matches found for **{movie_title}**")
            else:
                # Build terminal-style output
                output = []
                output.append(f"🎬 FINDING RECOMMENDATIONS FOR: '{movie_title}'")
                output.append("="*50)
                output.append(f"🎯 Found: {movie_info['names']}")
                output.append(f"📅 Year: {movie_info.get('date_x','Unknown')}")
                output.append(f"🎭 Genre: {movie_info['genre']}")
                output.append(f"⭐ Score: {movie_info['score']}")
                output.append(f"📝 Overview: {str(movie_info['overview'])[:100]}...\n")
                output.append(f"🔥 TOP {n_recs} RECOMMENDATIONS")
                output.append("-"*70)

                for i, (_, rec) in enumerate(recs.iterrows()):
                    output.append(f"{i+1:2d}. {rec['names'][:40]}")
                    output.append(f"    🎯 Similarity: {rec['similarity']:.4f}")
                    output.append(f"    ⭐ Rating: {rec['score']:.2f}")
                    output.append(f"    🎭 Genre: {rec['genre']}\n")

                st.code("\n".join(output), language="text")
    else:
        st.warning("Please upload imdb_movies.csv in the sidebar.")


if __name__ == "__main__":
    main()
