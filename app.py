import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import re
import warnings
warnings.filterwarnings('ignore')

# 🔗 Replace with your GitHub raw CSV URL
GITHUB_CSV_URL = "https://github.com/hongzai-67/Movie-Recommendation/blob/main/imdb_movies.csv"

class MovieRecommender:
    def __init__(self):
        self.df = None
        self.cosine_sim = None
        self.indices = None

    def clean_title(self, text):
        if pd.isna(text):
            return ""
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text))
        return re.sub(r'\s+', ' ', cleaned).lower().strip()

    @st.cache_data
    def load_data(_self, url):
        try:
            df = pd.read_csv(url, low_memory=False)
            
            # Preprocess
            df['overview'] = df['overview'].fillna('No description')
            df['genre'] = df['genre'].fillna('Unknown')
            df['crew'] = df['crew'].fillna('Unknown')
            df['clean_title'] = df['orig_title'].apply(_self.clean_title)
            
            # Create content for similarity
            df['content'] = (df['overview'].astype(str) + ' ' +
                           df['genre'].astype(str).str.replace('|', ' ') + ' ' +
                           df['crew'].astype(str))
            
            # Weighted ratings
            if 'vote_count' not in df.columns:
                df['vote_count'] = (df['revenue'].fillna(0) / 1000000 * 
                                   df['score'].fillna(5) * 
                                   np.random.uniform(50, 500, len(df))).astype(int).clip(lower=1)
            
            avg_rating = df['score'].mean()
            vote_threshold = df['vote_count'].quantile(0.90)
            df['weighted_rating'] = df.apply(
                lambda x: (x['vote_count']/(x['vote_count']+vote_threshold) * x['score']) + 
                         (vote_threshold/(vote_threshold+x['vote_count']) * avg_rating), axis=1)
            
            return df
        except Exception as e:
            st.error(f"Error: {e}")
            return None

    @st.cache_data
    def build_similarity(_self, content_series):
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = tfidf.fit_transform(content_series.fillna(''))
        return linear_kernel(tfidf_matrix, tfidf_matrix)

    def setup(self):
        if GITHUB_CSV_URL == "YOUR_GITHUB_RAW_URL_HERE":
            st.error("🔧 Please set your GitHub URL in the code")
            return False
        
        self.df = self.load_data(GITHUB_CSV_URL)
        if self.df is None:
            return False
        
        self.cosine_sim = self.build_similarity(self.df['content'])
        self.indices = pd.Series(self.df.index, index=self.df['clean_title']).drop_duplicates()
        return True

    def recommend(self, title, n=10):
        clean_title = self.clean_title(title)
        
        if clean_title not in self.indices:
            matches = self.df[self.df['clean_title'].str.contains(clean_title, case=False, na=False)]
            return matches.head(5) if not matches.empty else None, "not_found"
        
        idx = self.indices[clean_title]
        if hasattr(idx, '__iter__'):
            idx = idx.iloc[0]
        
        sim_scores = sorted(enumerate(self.cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:n+1]
        movie_indices = [i[0] for i in sim_scores]
        similarities = [i[1] for i in sim_scores]
        
        recs = self.df.iloc[movie_indices].copy()
        recs['similarity'] = similarities
        return recs, self.df.iloc[idx]

    def search_genre(self, genre, n=10):
        matches = self.df[self.df['genre'].str.contains(genre, case=False, na=False)]
        return matches.nlargest(n, 'weighted_rating') if not matches.empty else None

    def search_crew(self, crew, n=10):
        matches = self.df[self.df['crew'].str.contains(crew, case=False, na=False)]
        return matches.nlargest(n, 'weighted_rating') if not matches.empty else None

@st.cache_resource
def get_system():
    return MovieRecommender()

def main():
    st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")
    st.title("🎬 IMDB Movie Recommender")
    
    system = get_system()
    
    # Auto-setup
    if 'ready' not in st.session_state:
        with st.spinner("Loading..."):
            if system.setup():
                st.session_state.ready = True
                st.success(f"✅ {len(system.df):,} movies loaded!")
            else:
                return

    # Tabs
    tab1, tab2, tab3 = st.tabs(["🎬 Recommendations", "🎭 Genre", "👤 Crew"])

    with tab1:
        movie = st.text_input("Movie title:", placeholder="Enter movie name...")
        n = st.slider("Number of recommendations:", 5, 20, 10)
        
        if movie:
            results, info = system.recommend(movie, n)
            
            if results is not None:
                if isinstance(info, str):
                    st.warning("Similar titles:")
                    for _, m in results.iterrows():
                        st.write(f"• {m['names']}")
                else:
                    st.success(f"**{info['names']}** - {info['score']:.1f}/10")
                    
                    st.subheader("Recommendations:")
                    for i, (_, r) in enumerate(results.iterrows()):
                        sim_pct = r['similarity'] * 100
                        st.write(f"**{i+1}. {r['names']}** ({sim_pct:.1f}% match)")
                        st.write(f"⭐ {r['weighted_rating']:.1f} • {r['genre']}")
                        st.write("---")

    with tab2:
        genres = system.df['genre'].str.split('|').explode().value_counts().head(20).index.tolist()
        genre = st.selectbox("Genre:", genres)
        n = st.slider("Results:", 5, 20, 10, key="genre_n")
        
        if genre:
            results = system.search_genre(genre, n)
            if results is not None:
                st.subheader(f"Top {genre} Movies:")
                for i, (_, m) in enumerate(results.iterrows()):
                    st.write(f"**{i+1}. {m['names']}** - ⭐ {m['weighted_rating']:.1f}")

    with tab3:
        crew = st.text_input("Actor/Director name:", placeholder="e.g., Tom Hanks, Christopher Nolan...")
        n = st.slider("Results:", 5, 20, 10, key="crew_n")
        
        if crew:
            results = system.search_crew(crew, n)
            if results is not None:
                st.subheader(f"Movies with {crew}:")
                for i, (_, m) in enumerate(results.iterrows()):
                    st.write(f"**{i+1}. {m['names']}** - ⭐ {m['weighted_rating']:.1f}")

    # Sidebar info
    if 'ready' in st.session_state:
        st.sidebar.success("✅ System Ready")
        st.sidebar.metric("Movies", f"{len(system.df):,}")
        st.sidebar.metric("Avg Rating", f"{system.df['score'].mean():.1f}")

if __name__ == "__main__":
    main()
