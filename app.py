import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


# ====================================================
# Enhanced Recommendation System Class
# ====================================================
class IMDBContentBasedRecommendationSystem:
    def __init__(self):
        self.movies_df = None
        self.qualified_movies = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.indices = None
        self.average_rating = 0
        self.vote_threshold = 0

    def clean_title_text(self, text):
        if pd.isna(text):
            return ""
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text))
        cleaned = re.sub(r'\s+', ' ', cleaned).lower().strip()
        return cleaned

    def load_imdb_data(self, file_path):
        self.movies_df = pd.read_csv(file_path, low_memory=False)

        # Enhanced preprocessing
        self.movies_df['overview'] = self.movies_df['overview'].fillna('No description available')
        self.movies_df['genre'] = self.movies_df['genre'].fillna('Unknown')
        self.movies_df['crew'] = self.movies_df['crew'].fillna('Unknown')
        
        # Add missing columns if they don't exist
        if 'keywords' not in self.movies_df.columns:
            self.movies_df['keywords'] = ""
        if 'tagline' not in self.movies_df.columns:
            self.movies_df['tagline'] = ""

        # Preserve original titles and clean
        self.movies_df['original_title'] = self.movies_df['orig_title'].copy()
        self.movies_df['orig_title'] = self.movies_df['orig_title'].apply(self.clean_title_text)

        # Remove duplicates
        self.movies_df = self.movies_df.drop_duplicates(subset=['orig_title']).reset_index(drop=True)

        # Enhanced content creation
        self.movies_df['enhanced_content'] = (
            self.movies_df['overview'].astype(str) + ' ' +
            self.movies_df['genre'].astype(str).str.replace('|', ' ') + ' ' +
            self.movies_df['crew'].astype(str)
        )

        # Calculate weighted ratings using IMDb formula
        self.calculate_weighted_ratings()

    def calculate_weighted_ratings(self):
        """Calculate weighted ratings using IMDb formula with synthetic vote counts"""
        # Create synthetic vote count based on revenue, score, and randomness
        if 'vote_count' not in self.movies_df.columns:
            self.movies_df['vote_count'] = (
                (self.movies_df['revenue'].fillna(0) / 1_000_000) *
                (self.movies_df['score'].fillna(5) / 2) *
                np.random.uniform(50, 500, len(self.movies_df))
            ).astype(int)
            self.movies_df['vote_count'] = self.movies_df['vote_count'].clip(lower=1)

        # Calculate global statistics
        self.average_rating = self.movies_df['score'].mean()
        self.vote_threshold = self.movies_df['vote_count'].quantile(0.90)
        
        # Apply IMDb weighted rating formula: WR = (v/(v+m) × R) + (m/(m+v) × C)
        def weighted_rating(x, m=self.vote_threshold, C=self.average_rating):
            v = x['vote_count']
            R = x['score']
            return (v/(v+m) * R) + (m/(m+v) * C)
        
        self.movies_df['weighted_rating'] = self.movies_df.apply(weighted_rating, axis=1)
        self.movies_df['weighted'] = self.movies_df['weighted_rating']  # Keep compatibility
        
        self.qualified_movies = self.movies_df.copy()

    def build_content_based_system(self):
        """Build TF-IDF based content similarity system"""
        working_df = self.qualified_movies.copy()
        
        # Enhanced TF-IDF parameters
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        working_df['enhanced_content'] = working_df['enhanced_content'].fillna('')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(working_df['enhanced_content'])
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
        self.indices = pd.Series(working_df.index, index=working_df['orig_title']).drop_duplicates()
        self.qualified_movies = working_df

    def get_similarity_level(self, similarity_score):
        """Convert similarity score to descriptive level"""
        if similarity_score >= 0.87:
            return "🔥 VERY HIGH"
        elif similarity_score >= 0.86:
            return "🟢 HIGH"
        elif similarity_score >= 0.85:
            return "🟡 MODERATE"
        elif similarity_score >= 0.84:
            return "🟠 LOW"
        else:
            return "🔴 VERY LOW"

    def display_centralized_results(self, results_df, search_type="Search", original_query="", n=10):
        """Centralized results display with comprehensive movie information"""
        if results_df.empty:
            return "❌ No results found!"

        display_count = min(n, len(results_df))
        display_results = results_df.head(display_count).copy()

        output = []
        output.append("="*80)
        output.append(f"🎬 {search_type.upper()} RESULTS FOR: '{original_query}'")
        output.append("="*80)
        output.append(f"📊 Showing {display_count} results:")
        output.append("-" * 80)

        for i, (idx, movie) in enumerate(display_results.iterrows()):
            # Main line with rank and movie name
            if i == 0 and search_type == "Content Recommendations":
                main_line = f"🏆 {i+1}. {movie['names'][:45]} ⭐ TOP MATCH!"
            else:
                main_line = f"🎬 {i+1}. {movie['names'][:45]}"

            # Info line with year and rating
            info_parts = []
            try:
                if 'date_x' in movie.index:
                    movie_year = pd.to_datetime(movie['date_x'], errors='coerce')
                    if pd.notna(movie_year):
                        info_parts.append(f"📅 Year: {movie_year.year}")
                    else:
                        info_parts.append("📅 Year: Unknown")
                else:
                    info_parts.append("📅 Year: N/A")
            except:
                info_parts.append("📅 Year: Unknown")

            if 'weighted_rating' in movie.index:
                info_parts.append(f"⭐ Rating: {movie['weighted_rating']:.2f}")
            elif 'score' in movie.index:
                info_parts.append(f"⭐ Score: {movie['score']:.1f}")

            info_line = " ".join(info_parts)
            output.append(main_line)
            output.append(info_line)

            # Similarity for content recommendations
            if 'similarity' in movie.index and search_type == "Content Recommendations":
                similarity_percent = movie['similarity'] * 100
                similarity_level = self.get_similarity_level(movie['similarity'])
                output.append(f"🎯 Similarity: {movie['similarity']:.4f} ({similarity_percent:.1f}%) - {similarity_level}")

            # Additional metadata
            if 'genre' in movie.index:
                genre_display = str(movie['genre'])[:50]
                if len(str(movie['genre'])) > 50:
                    genre_display += "..."
                output.append(f"🎭 Genre: {genre_display}")

            if 'crew' in movie.index:
                crew_display = str(movie['crew'])[:60]
                if len(str(movie['crew'])) > 60:
                    crew_display += "..."
                output.append(f"👥 Crew: {crew_display}")

            if 'orig_lang' in movie.index:
                output.append(f"🗣️ Language: {movie['orig_lang']}")

            if 'country' in movie.index:
                country_display = str(movie['country'])[:30]
                if len(str(movie['country'])) > 30:
                    country_display += "..."
                output.append(f"🌍 Country: {country_display}")

            if 'budget_x' in movie.index and pd.notna(movie['budget_x']) and movie['budget_x'] > 0:
                output.append(f"💰 Budget: ${movie['budget_x']:,}")

            if 'revenue' in movie.index and pd.notna(movie['revenue']) and movie['revenue'] > 0:
                output.append(f"💵 Revenue: ${movie['revenue']:,}")

            output.append("")  # Empty line between movies

        output.append("="*80)
        return "\n".join(output)

    def get_content_recommendations(self, title, n=10):
        """Get content-based recommendations with enhanced similarity scaling"""
        # Fuzzy matching
        if title not in self.indices:
            possible_matches = self.qualified_movies[
                self.qualified_movies['orig_title'].str.contains(title, case=False, na=False)
            ]
            if possible_matches.empty:
                return None, None, None
            return "choose", possible_matches.head(5), None

        # Exact match
        idx = self.indices[title]
        if hasattr(idx, '__iter__') and not isinstance(idx, str):
            idx = idx.iloc[0] if hasattr(idx, 'iloc') else idx[0]
            
        sim_scores = list(enumerate(self.cosine_sim[idx]))

        # Enhanced similarity scaling (0.8-1.0 range like in Colab)
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

        # Sort by similarity (highest first)
        scaled_sim_scores = sorted(scaled_sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
        
        movie_indices = [i[0] for i in scaled_sim_scores]
        similarity_values = [i[1] for i in scaled_sim_scores]
        
        movies = self.qualified_movies.iloc[movie_indices].copy()
        movies['similarity'] = similarity_values

        return "ok", self.qualified_movies.loc[idx], movies

    def get_hybrid_recommendations(self, title, n=10, alpha=0.7):
        """Hybrid recommendations combining content similarity and popularity"""
        # Ensure content system is built
        if self.cosine_sim is None or self.indices is None:
            self.build_content_based_system()

        # Prepare normalized popularity
        if 'popularity_norm' not in self.qualified_movies.columns:
            ratings = self.qualified_movies['weighted_rating'].fillna(self.average_rating)
            min_r, max_r = ratings.min(), ratings.max()
            denom = (max_r - min_r) if max_r != min_r else 1.0
            self.qualified_movies['popularity_norm'] = (ratings - min_r) / denom

        # Check for fuzzy matching
        if title not in self.indices:
            possible_matches = self.qualified_movies[
                self.qualified_movies['orig_title'].str.contains(title, case=False, na=False)
            ]
            if possible_matches.empty:
                return None, None, None
            return "choose", possible_matches.head(5), None

        # Get movie index
        idx = self.indices[title]
        if hasattr(idx, '__iter__') and not isinstance(idx, str):
            idx = idx.iloc[0] if hasattr(idx, 'iloc') else idx[0]

        # Calculate similarities
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = [(i, s) for i, s in sim_scores if i != idx]

        # Build candidate dataframe
        candidate_indices = [i for i, _ in sim_scores]
        candidates_df = self.qualified_movies.iloc[candidate_indices].copy()
        candidates_df['similarity'] = [s for _, s in sim_scores]

        # Hybrid scoring
        candidates_df['hybrid_score'] = (
            alpha * candidates_df['similarity'] + (1 - alpha) * candidates_df['popularity_norm']
        )

        recommendations = candidates_df.sort_values('hybrid_score', ascending=False).head(n)
        
        return "ok", self.qualified_movies.iloc[idx], recommendations

    def search_by_genre(self, genre, n=10, show_details=True):
        """Search movies by genre with centralized display"""
        try:
            genre_matches = self.qualified_movies[
                self.qualified_movies['genre'].str.contains(genre, case=False, na=False)
            ]
            if genre_matches.empty:
                return f"❌ No movies found with genre '{genre}'"
            
            genre_matches = genre_matches.nlargest(n, 'weighted_rating')
            
            if show_details:
                return self.display_centralized_results(genre_matches, "Genre Search", genre, n)
            else:
                return genre_matches
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def search_by_crew(self, crew_name, n=10, show_details=True):
        """Search movies by crew member with centralized display"""
        try:
            crew_matches = self.qualified_movies[
                self.qualified_movies['crew'].str.contains(crew_name, case=False, na=False)
            ]
            if crew_matches.empty:
                return f"❌ No movies found with crew member '{crew_name}'"
            
            crew_matches = crew_matches.nlargest(n, 'weighted_rating')
            
            if show_details:
                return self.display_centralized_results(crew_matches, "Crew Search", crew_name, n)
            else:
                return crew_matches
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def get_top_movies_by_rating(self, n=20, show_details=True):
        """Get top movies by weighted rating"""
        try:
            top_movies = self.qualified_movies.nlargest(n, 'weighted_rating').copy()
            if show_details:
                return self.display_centralized_results(top_movies, "Top Rated Movies", f"Top {n} Movies", n)
            else:
                return top_movies
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def search_by_year(self, year, n=10, show_details=True):
        """Search movies by release year"""
        try:
            self.qualified_movies['year'] = pd.to_datetime(
                self.qualified_movies['date_x'], errors='coerce'
            ).dt.year

            year_matches = self.qualified_movies[self.qualified_movies['year'] == year]
            if year_matches.empty:
                return f"❌ No movies found from year {year}"

            year_matches = year_matches.nlargest(n, 'weighted_rating').copy()

            if show_details:
                return self.display_centralized_results(year_matches, "Year Search", year, n)
            else:
                return year_matches
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def search_by_country(self, country, n=10, show_details=True):
        """Search movies by country"""
        try:
            if 'country' not in self.qualified_movies.columns:
                return "❌ Country information not available in dataset"

            country_matches = self.qualified_movies[
                self.qualified_movies['country'].str.contains(country, case=False, na=False)
            ]
            if country_matches.empty:
                return f"❌ No movies found from country '{country}'"

            country_matches = country_matches.nlargest(n, 'weighted_rating').copy()

            if show_details:
                return self.display_centralized_results(country_matches, "Country Search", country, n)
            else:
                return country_matches
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def search_by_language(self, language, n=10, show_details=True):
        """Search movies by original language"""
        try:
            if 'orig_lang' not in self.qualified_movies.columns:
                return "❌ Language information not available in dataset"

            lang_matches = self.qualified_movies[
                self.qualified_movies['orig_lang'].str.contains(language, case=False, na=False)
            ]
            if lang_matches.empty:
                return f"❌ No movies found in language '{language}'"

            lang_matches = lang_matches.nlargest(n, 'weighted_rating').copy()

            if show_details:
                return self.display_centralized_results(lang_matches, "Language Search", language, n)
            else:
                return lang_matches
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def evaluate_precision_recall_f1(self, k=10, method='content'):
        """Evaluate Precision@k, Recall@k, and F1 using genre-overlap relevance"""
        try:
            precisions, recalls = [], []
            titles = self.qualified_movies['orig_title'].tolist()

            for title in titles[:50]:  # Limit for Streamlit performance
                # Ground-truth relevant set (by genre overlap)
                src = self.qualified_movies[self.qualified_movies['orig_title'] == title]
                if src.empty:
                    continue
                    
                src_genres = set(str(src.iloc[0]['genre']).split('|'))
                relevant_mask = self.qualified_movies['genre'].apply(
                    lambda g: len(src_genres.intersection(set(str(g).split('|')))) > 0
                )
                relevant_indices = set(self.qualified_movies[relevant_mask].index) - set(src.index)

                # Get recommendations
                if method == 'hybrid':
                    status, _, recs = self.get_hybrid_recommendations(title, n=k)
                else:
                    status, _, recs = self.get_content_recommendations(title, n=k)

                if status != "ok" or recs is None or recs.empty:
                    continue

                rec_indices = set(recs.index)
                true_positives = len(rec_indices.intersection(relevant_indices))
                precision = true_positives / max(1, len(rec_indices))
                recall = true_positives / max(1, len(relevant_indices))

                precisions.append(precision)
                recalls.append(recall)

            precision_macro = float(np.mean(precisions)) if precisions else 0.0
            recall_macro = float(np.mean(recalls)) if recalls else 0.0
            f1_macro = 2 * precision_macro * recall_macro / (precision_macro + recall_macro) if (precision_macro + recall_macro) > 0 else 0.0

            return {
                'precision_at_k': precision_macro,
                'recall_at_k': recall_macro,
                'f1_at_k': f1_macro,
                'k': k,
                'method': method
            }
        except Exception as e:
            st.error(f"Error during evaluation: {str(e)}")
            return None

    def create_visualizations(self):
        """Create comprehensive visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Subplot 1: Rating distribution
        axes[0, 0].hist(self.movies_df['score'], bins=30, alpha=0.7, color='skyblue', label='Original')
        axes[0, 0].hist(self.qualified_movies['weighted_rating'], bins=30, alpha=0.7, color='orange', label='Weighted')
        axes[0, 0].set_title('Rating Distribution')
        axes[0, 0].legend()
        
        # Subplot 2: Top genres
        all_genres = []
        for genres in self.qualified_movies['genre'].dropna():
            all_genres.extend(str(genres).split('|'))
        genre_counts = pd.Series(all_genres).value_counts().head(10)
        genre_counts.plot(kind='barh', ax=axes[0, 1], color='coral')
        axes[0, 1].set_title('Top 10 Genres')
        
        # Subplot 3: Year vs Rating
        self.qualified_movies['year'] = pd.to_datetime(self.qualified_movies['date_x'], errors='coerce').dt.year
        valid_years = self.qualified_movies.dropna(subset=['year'])
        if not valid_years.empty:
            axes[1, 0].scatter(valid_years['year'], valid_years['weighted_rating'], alpha=0.6, color='purple')
            axes[1, 0].set_title('Rating vs Release Year')
            axes[1, 0].set_xlabel('Year')
            axes[1, 0].set_ylabel('Rating')
        
        # Subplot 4: Top movies
        top_movies = self.qualified_movies.nlargest(10, 'weighted_rating')
        axes[1, 1].barh(range(len(top_movies)), top_movies['weighted_rating'], color='red', alpha=0.7)
        axes[1, 1].set_yticks(range(len(top_movies)))
        axes[1, 1].set_yticklabels([title[:25] + '...' if len(title) > 25 else title for title in top_movies['names']])
        axes[1, 1].set_title('Top 10 Movies')
        
        plt.tight_layout()
        return fig


# ====================================================
# Enhanced Streamlit UI
# ====================================================
def main():
    st.set_page_config(page_title="Enhanced IMDB Recommender", layout="wide")
    st.title("🎮 ENHANCED INTERACTIVE MOVIE RECOMMENDATION SYSTEM")
    st.markdown("=================================================================")
    st.markdown("✨ **ENHANCED FEATURES**: Improved similarity scaling, hybrid recommendations, and comprehensive evaluations!")
    st.markdown("📊 **DISPLAYS**: Name, Year, Rating, Genre, Crew, Language, Country, Similarity")

    # Sidebar controls
    st.sidebar.title("🎛️ Controls")
    
    # Reset button
    if st.sidebar.button("🔄 Reset All Records", type="secondary"):
        for key in list(st.session_state.keys()):
            if key != 'uploaded_file':
                del st.session_state[key]
        st.success("✅ All search records cleared!")
        st.rerun()

    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload IMDB dataset (CSV)", type="csv")
    if not uploaded_file:
        st.warning("Please upload imdb_movies.csv in the sidebar.")
        return

    # Initialize and cache recommender
    @st.cache_data
    def load_recommender(file):
        recommender = IMDBContentBasedRecommendationSystem()
        recommender.load_imdb_data(file)
        recommender.build_content_based_system()
        return recommender

    with st.spinner("Loading and processing dataset..."):
        recommender = load_recommender(uploaded_file)

    # Enhanced menu without advanced search
    option = st.selectbox("🎯 SEARCH OPTIONS:", [
        "1️⃣ Content-Based Recommendations (by Movie Title)",
        "2️⃣ Hybrid Recommendations (Content + Popularity)",
        "3️⃣ Search by Genre",
        "4️⃣ Search by Crew Member",
        "5️⃣ Search by Year",
        "6️⃣ Search by Country",
        "7️⃣ Search by Language", 
        "8️⃣ Top Rated Movies",
        "9️⃣ System Evaluation & Visualizations"
    ])

    # Main content area
    if option.startswith("1️⃣"):
        st.subheader("🎬 Content-Based Recommendations")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            title = st.text_input("Enter a movie title:")
        with col2:
            n_recs = st.slider("Number of recommendations", 1, 20, 10)

        # Initialize session state for content recommendations
        if 'content_status' not in st.session_state:
            st.session_state.content_status = None
        if 'content_movie_info' not in st.session_state:
            st.session_state.content_movie_info = None
        if 'content_n_recs' not in st.session_state:
            st.session_state.content_n_recs = 10

        if st.button("Get Recommendations", type="primary"):
            if title:
                cleaned_title = recommender.clean_title_text(title)
                status, movie_info, recs = recommender.get_content_recommendations(cleaned_title, n=n_recs)
                
                st.session_state.content_status = status
                st.session_state.content_movie_info = movie_info
                st.session_state.content_n_recs = n_recs
                
                if status == "ok":
                    result = recommender.display_centralized_results(recs, "Content Recommendations", movie_info['names'], n_recs)
                    st.code(result, language="text")
                    # Clear session state after successful result
                    st.session_state.content_status = None
                    st.session_state.content_movie_info = None
                else:
                    st.error("Movie not found!")

        # Handle multiple matches from session state
        if st.session_state.content_status == "choose" and st.session_state.content_movie_info is not None:
            st.warning("🔍 Multiple matches found. Please select:")
            choices = st.session_state.content_movie_info['names'].tolist()
            choice = st.selectbox("Select a movie:", choices, key="content_choice")
            
            if st.button("Confirm Selection", type="secondary"):
                if choice:
                    cleaned_choice = recommender.clean_title_text(choice)
                    n_recs = st.session_state.content_n_recs
                    
                    status, movie_info, recs = recommender.get_content_recommendations(cleaned_choice, n=n_recs)
                    if status == "ok":
                        result = recommender.display_centralized_results(recs, "Content Recommendations", movie_info['names'], n_recs)
                        st.code(result, language="text")
                        
                        # Clear session state after successful result
                        st.session_state.content_status = None
                        st.session_state.content_movie_info = None
                        st.session_state.content_n_recs = 10
                        st.rerun()
                    else:
                        st.error("Error processing selection!")
                else:
                    st.error("Please select a movie from the dropdown!")

    elif option.startswith("2️⃣"):
        st.subheader("🎯 Hybrid Recommendations")
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            title = st.text_input("Enter a movie title:")
        with col2:
            n_recs = st.slider("Number of recommendations", 1, 20, 10)
        with col3:
            alpha = st.slider("Content vs Popularity", 0.0, 1.0, 0.7, help="0=Pure Popularity, 1=Pure Content")

        if st.button("Get Hybrid Recommendations", type="primary"):
            if title:
                cleaned_title = recommender.clean_title_text(title)
                status, movie_info, recs = recommender.get_hybrid_recommendations(cleaned_title, n=n_recs, alpha=alpha)
                
                if status == "choose":
                    st.warning("🔍 Multiple matches found. Please select:")
                    choices = movie_info['names'].tolist()
                    choice = st.selectbox("Select a movie:", choices, key="hybrid_choice")
                    if st.button("Confirm Hybrid Selection"):
                        cleaned_choice = recommender.clean_title_text(choice)
                        status, movie_info, recs = recommender.get_hybrid_recommendations(cleaned_choice, n=n_recs, alpha=alpha)
                        if status == "ok":
                            result = recommender.display_centralized_results(recs, "Hybrid Recommendations", movie_info['names'], n_recs)
                            st.code(result, language="text")
                
                elif status == "ok":
                    result = recommender.display_centralized_results(recs, "Hybrid Recommendations", movie_info['names'], n_recs)
                    st.code(result, language="text")
                else:
                    st.error("Movie not found!")

    elif option.startswith("3️⃣"):
        st.subheader("🎭 Search by Genre")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            genre = st.text_input("Enter a genre:")
        with col2:
            n_results = st.slider("Number of results", 1, 20, 10)
        
        if st.button("Search by Genre", type="primary"):
            if genre:
                result = recommender.search_by_genre(genre, n=n_results, show_details=True)
                if isinstance(result, str):
                    if result.startswith("❌"):
                        st.error(result)
                    else:
                        st.code(result, language="text")

    elif option.startswith("4️⃣"):
        st.subheader("👥 Search by Crew Member")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            crew = st.text_input("Enter crew member name (actor, director, etc.):")
        with col2:
            n_results = st.slider("Number of results", 1, 20, 10, key="crew_slider")
        
        if st.button("Search by Crew", type="primary"):
            if crew:
                result = recommender.search_by_crew(crew, n=n_results, show_details=True)
                if isinstance(result, str):
                    if result.startswith("❌"):
                        st.error(result)
                    else:
                        st.code(result, language="text")

    elif option.startswith("5️⃣"):
        st.subheader("📅 Search by Year")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            year_input = st.number_input("Enter release year:", min_value=1900, max_value=2024, value=2020)
        with col2:
            n_results = st.slider("Number of results", 1, 20, 10, key="year_slider")
        
        if st.button("Search by Year", type="primary"):
            result = recommender.search_by_year(int(year_input), n=n_results, show_details=True)
            if isinstance(result, str):
                if result.startswith("❌"):
                    st.error(result)
                else:
                    st.code(result, language="text")

    elif option.startswith("6️⃣"):
        st.subheader("🌍 Search by Country")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            country = st.text_input("Enter country name:")
        with col2:
            n_results = st.slider("Number of results", 1, 20, 10, key="country_slider")
        
        if st.button("Search by Country", type="primary"):
            if country:
                result = recommender.search_by_country(country, n=n_results, show_details=True)
                if isinstance(result, str):
                    if result.startswith("❌"):
                        st.error(result)
                    else:
                        st.code(result, language="text")

    elif option.startswith("7️⃣"):
        st.subheader("🗣️ Search by Language")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            language = st.text_input("Enter language (e.g., 'en', 'english'):")
        with col2:
            n_results = st.slider("Number of results", 1, 20, 10, key="language_slider")
        
        if st.button("Search by Language", type="primary"):
            if language:
                result = recommender.search_by_language(language, n=n_results, show_details=True)
                if isinstance(result, str):
                    if result.startswith("❌"):
                        st.error(result)
                    else:
                        st.code(result, language="text")

    elif option.startswith("8️⃣"):
        st.subheader("⭐ Top Rated Movies")
        n_results = st.slider("Number of top movies", 1, 50, 20, key="top_slider")
        
        if st.button("Get Top Movies", type="primary"):
            result = recommender.get_top_movies_by_rating(n=n_results, show_details=True)
            if isinstance(result, str):
                if result.startswith("❌"):
                    st.error(result)
                else:
                    st.code(result, language="text")

    elif option.startswith("9️⃣"):
        st.subheader("📊 System Evaluation & Visualizations")
        
        tab1, tab2 = st.tabs(["📈 Visualizations", "🔬 Performance Metrics"])
        
        with tab1:
            if st.button("Generate Visualizations", type="primary"):
                with st.spinner("Creating visualizations..."):
                    fig = recommender.create_visualizations()
                    st.pyplot(fig)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Evaluate Content-Based System", type="secondary"):
                    with st.spinner("Evaluating content-based recommendations..."):
                        metrics = recommender.evaluate_precision_recall_f1(k=10, method='content')
                        if metrics:
                            st.json(metrics)
            
            with col2:
                if st.button("Evaluate Hybrid System", type="secondary"):
                    with st.spinner("Evaluating hybrid recommendations..."):
                        metrics = recommender.evaluate_precision_recall_f1(k=10, method='hybrid')
                        if metrics:
                            st.json(metrics)
            
            if st.button("Dataset Statistics", type="secondary"):
                st.subheader("📊 Dataset Overview")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Movies", len(recommender.qualified_movies))
                    st.metric("Average Rating", f"{recommender.average_rating:.2f}")
                
                with col2:
                    unique_genres = set()
                    for genres in recommender.qualified_movies['genre'].dropna():
                        unique_genres.update(str(genres).split('|'))
                    st.metric("Unique Genres", len(unique_genres))
                    st.metric("Vote Threshold", f"{recommender.vote_threshold:.0f}")
                
                with col3:
                    years = pd.to_datetime(recommender.qualified_movies['date_x'], errors='coerce').dt.year.dropna()
                    if not years.empty:
                        st.metric("Year Range", f"{int(years.min())}-{int(years.max())}")
                    
                    if 'country' in recommender.qualified_movies.columns:
                        unique_countries = recommender.qualified_movies['country'].nunique()
                        st.metric("Countries", unique_countries)

    # Sidebar information
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Dataset Info")
    if 'recommender' in locals():
        st.sidebar.info(f"Movies loaded: {len(recommender.qualified_movies)}")
        st.sidebar.info(f"TF-IDF features: {recommender.tfidf_matrix.shape[1] if recommender.tfidf_matrix is not None else 'Not built'}")


if __name__ == "__main__":
    main()
