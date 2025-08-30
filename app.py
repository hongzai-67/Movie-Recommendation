import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import re
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Movie Recommender", layout="wide")

class StreamlitIMDBRecommendationSystem:
    def __init__(self):
        self.movies_df = None
        self.qualified_movies = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.indices = None
        self.average_rating = None
        self.vote_threshold = None

    def clean_title_text(self, text):
        if pd.isna(text):
            return ""
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text))
        cleaned = re.sub(r'\s+', ' ', cleaned).lower().strip()
        return cleaned

    @st.cache_data
    def load_data(_self, uploaded_file):
        try:
            df = pd.read_csv(uploaded_file, low_memory=False)
            return df
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None

    def preprocess_data(self):
        self.movies_df['overview'] = self.movies_df['overview'].fillna('No description available')
        self.movies_df['genre'] = self.movies_df['genre'].fillna('Unknown')
        self.movies_df['crew'] = self.movies_df['crew'].fillna('Unknown')
        
        self.movies_df['original_title'] = self.movies_df['orig_title'].copy()
        self.movies_df['orig_title'] = self.movies_df['orig_title'].apply(self.clean_title_text)
        
        before_count = len(self.movies_df)
        self.movies_df = self.movies_df.drop_duplicates(subset=['orig_title'], keep='first').reset_index(drop=True)
        after_count = len(self.movies_df)
        
        self.movies_df['enhanced_content'] = (
            self.movies_df['overview'].astype(str) + ' ' +
            self.movies_df['genre'].astype(str).str.replace('|', ' ') + ' ' +
            self.movies_df['crew'].astype(str)
        )
        
        return before_count - after_count

    def calculate_weighted_ratings(self):
        if 'vote_count' not in self.movies_df.columns:
            self.movies_df['vote_count'] = (
                (self.movies_df['revenue'].fillna(0) / 1000000) *
                (self.movies_df['score'].fillna(5) / 2) *
                np.random.uniform(50, 500, len(self.movies_df))
            ).astype(int)
            self.movies_df['vote_count'] = self.movies_df['vote_count'].clip(lower=1)

        self.average_rating = self.movies_df['score'].mean()
        self.vote_threshold = self.movies_df['vote_count'].quantile(0.90)

        def weighted_rating(x, m=self.vote_threshold, C=self.average_rating):
            v = x['vote_count']
            R = x['score']
            return (v/(v+m) * R) + (m/(m+v) * C)

        self.movies_df['weighted_rating'] = self.movies_df.apply(weighted_rating, axis=1)
        self.qualified_movies = self.movies_df.copy()

    def build_content_system(self):
        working_df = self.qualified_movies.copy()
        
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        working_df['enhanced_content'] = working_df['enhanced_content'].fillna('')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(working_df['enhanced_content'])
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
        self.indices = pd.Series(working_df.index, index=working_df['orig_title']).drop_duplicates()
        self.qualified_movies = working_df

    def get_content_recommendations(self, title, n=10):
        try:
            if title not in self.indices:
                possible_matches = self.qualified_movies[
                    self.qualified_movies['orig_title'].str.contains(title, case=False, na=False)
                ]
                if possible_matches.empty:
                    return None, f"Movie '{title}' not found!"
                title = possible_matches.iloc[0]['orig_title']

            idx = self.indices[title]
            if hasattr(idx, '__iter__') and not isinstance(idx, str):
                idx = idx.iloc[0] if hasattr(idx, 'iloc') else idx[0]

            movie_info = self.qualified_movies.loc[idx]

            sim_scores = list(enumerate(self.cosine_sim[idx]))
            sim_values = [score for _, score in sim_scores]
            min_sim = min(sim_values) if sim_values else 0
            max_sim = max(sim_values) if sim_values else 1

            if max_sim == min_sim:
                scaled_sim_scores = [(i, 0.8) for i, _ in sim_scores]
            else:
                scaled_sim_scores = [
                    (i, 0.8 + (score - min_sim) * (1.0 - 0.8) / (max_sim - min_sim))
                    for i, score in sim_scores
                ]

            scaled_sim_scores = sorted(scaled_sim_scores, key=lambda x: x[1], reverse=True)
            scaled_sim_scores = scaled_sim_scores[1:n+1]

            movie_indices = [i[0] for i in scaled_sim_scores]
            similarity_values = [i[1] for i in scaled_sim_scores]

            recommendations = self.qualified_movies.iloc[movie_indices].copy()
            recommendations['similarity'] = similarity_values
            recommendations['composite_score'] = (
                0.7 * recommendations['similarity'] +
                0.3 * (recommendations['weighted_rating'] / 10.0)
            )

            return movie_info, recommendations

        except Exception as e:
            return None, f"Error: {str(e)}"

    def search_by_genre(self, genre, n=10):
        try:
            genre_matches = self.qualified_movies[
                self.qualified_movies['genre'].str.contains(genre, case=False, na=False)
            ]
            if genre_matches.empty:
                return None
            return genre_matches.nlargest(n, 'weighted_rating')
        except:
            return None

    def search_by_crew(self, crew_name, n=10):
        try:
            crew_matches = self.qualified_movies[
                self.qualified_movies['crew'].str.contains(crew_name, case=False, na=False)
            ]
            if crew_matches.empty:
                return None
            return crew_matches.nlargest(n, 'weighted_rating')
        except:
            return None

    def advanced_search(self, genre=None, crew=None, min_rating=None, max_results=10):
        try:
            results = self.qualified_movies.copy()
            
            if genre:
                results = results[results['genre'].str.contains(genre, case=False, na=False)]
            if crew:
                results = results[results['crew'].str.contains(crew, case=False, na=False)]
            if min_rating:
                results = results[results['weighted_rating'] >= min_rating]
            
            if results.empty:
                return None
            return results.nlargest(max_results, 'weighted_rating')
        except:
            return None

    def get_genre_list(self):
        all_genres = []
        for genres in self.qualified_movies['genre'].dropna():
            all_genres.extend(str(genres).split('|'))
        return pd.Series(all_genres).value_counts().head(20)

    def get_popular_crew(self):
        all_crew = []
        for crew in self.qualified_movies['crew'].dropna():
            crew_members = re.split(r'[,|;]', str(crew))
            for member in crew_members:
                member = member.strip()
                if len(member) > 2:
                    all_crew.append(member)
        return pd.Series(all_crew).value_counts().head(20)

@st.cache_resource
def initialize_system():
    return StreamlitIMDBRecommendationSystem()

def main():
    st.title("Movie Recommendation System")
    
    recommender = initialize_system()
    
    # File upload
    uploaded_file = st.file_uploader("Upload IMDB CSV Dataset", type=['csv'])
    
    if uploaded_file is not None:
        if recommender.movies_df is None:
            with st.spinner("Loading dataset..."):
                recommender.movies_df = recommender.load_data(uploaded_file)
                if recommender.movies_df is not None:
                    recommender.preprocess_data()
                    recommender.calculate_weighted_ratings()
                    recommender.build_content_system()
                    st.success("Dataset loaded successfully!")
        
        if recommender.movies_df is not None:
            # Navigation
            option = st.selectbox(
                "Choose function:",
                ["Movie Recommendations", "Search by Genre", "Search by Crew", "Advanced Search"]
            )
            
            if option == "Movie Recommendations":
                movie_title = st.text_input("Enter movie title:")
                num_recs = st.number_input("Number of recommendations", min_value=1, max_value=20, value=10)
                
                if movie_title:
                    cleaned_title = recommender.clean_title_text(movie_title)
                    movie_info, recommendations = recommender.get_content_recommendations(cleaned_title, num_recs)
                    
                    if movie_info is not None:
                        st.write(f"Found: {movie_info['names']}")
                        st.write(f"Genre: {movie_info['genre']}")
                        st.write(f"Rating: {movie_info['weighted_rating']:.2f}")
                        
                        st.write("Recommendations:")
                        for i, (_, rec) in enumerate(recommendations.iterrows()):
                            st.write(f"{i+1}. {rec['names']} - Similarity: {rec['similarity']:.4f} - Rating: {rec['weighted_rating']:.2f}")
                    else:
                        st.error(recommendations)
            
            elif option == "Search by Genre":
                available_genres = recommender.get_genre_list().index.tolist()
                selected_genre = st.selectbox("Select genre:", [""] + available_genres[:15])
                
                if not selected_genre:
                    selected_genre = st.text_input("Or enter custom genre:")
                
                num_results = st.number_input("Number of results", min_value=1, max_value=20, value=10)
                
                if selected_genre:
                    results = recommender.search_by_genre(selected_genre, num_results)
                    if results is not None and not results.empty:
                        st.write(f"Found {len(results)} movies in '{selected_genre}' genre:")
                        for i, (_, movie) in enumerate(results.iterrows()):
                            st.write(f"{i+1}. {movie['names']} - Rating: {movie['weighted_rating']:.2f}")
                    else:
                        st.warning(f"No movies found for genre '{selected_genre}'")
            
            elif option == "Search by Crew":
                popular_crew = recommender.get_popular_crew().index.tolist()
                selected_crew = st.selectbox("Select crew member:", [""] + popular_crew[:15])
                
                if not selected_crew:
                    selected_crew = st.text_input("Or enter crew member name:")
                
                num_results = st.number_input("Number of results", min_value=1, max_value=20, value=10, key="crew_num")
                
                if selected_crew:
                    results = recommender.search_by_crew(selected_crew, num_results)
                    if results is not None and not results.empty:
                        st.write(f"Found {len(results)} movies with '{selected_crew}':")
                        for i, (_, movie) in enumerate(results.iterrows()):
                            st.write(f"{i+1}. {movie['names']} - Rating: {movie['weighted_rating']:.2f}")
                    else:
                        st.warning(f"No movies found with crew member '{selected_crew}'")
            
            elif option == "Advanced Search":
                genre_filter = st.text_input("Genre (optional):")
                crew_filter = st.text_input("Crew Member (optional):")
                min_rating = st.number_input("Minimum Rating (optional):", min_value=0.0, max_value=10.0, step=0.1, value=None)
                max_results = st.number_input("Max Results", min_value=1, max_value=50, value=10)
                
                if st.button("Search"):
                    results = recommender.advanced_search(
                        genre=genre_filter if genre_filter else None,
                        crew=crew_filter if crew_filter else None,
                        min_rating=min_rating,
                        max_results=max_results
                    )
                    
                    if results is not None and not results.empty:
                        st.write(f"Found {len(results)} movies:")
                        for i, (_, movie) in enumerate(results.iterrows()):
                            st.write(f"{i+1}. {movie['names']} - Rating: {movie['weighted_rating']:.2f}")
                    else:
                        st.warning("No movies match your criteria.")
    
    else:
        st.write("Please upload your imdb_movies.csv file to start using the recommendation system.")

if __name__ == "__main__":
    main()
