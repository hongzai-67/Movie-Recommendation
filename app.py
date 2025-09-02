import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from pathlib import Path


# ====================================================
# Model Training Script (Run this first to create the model)
# ====================================================
def train_and_save_model(csv_file_path, model_save_path="imdb_model.pkl"):
    """
    Train the recommendation model and save it using joblib
    Run this function once with your CSV file to create the model
    """
    print("🚀 Starting model training...")
    
    # Load and preprocess data
    movies_df = pd.read_csv(csv_file_path, low_memory=False)
    
    # Fill missing values
    movies_df['overview'] = movies_df['overview'].fillna('No description available')
    movies_df['genre'] = movies_df['genre'].fillna('Unknown')
    movies_df['crew'] = movies_df['crew'].fillna('Unknown')
    if 'keywords' not in movies_df.columns:
        movies_df['keywords'] = ""
    if 'tagline' not in movies_df.columns:
        movies_df['tagline'] = ""

    # Clean title function
    def clean_title_text(text):
        if pd.isna(text):
            return ""
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text))
        cleaned = re.sub(r'\s+', ' ', cleaned).lower().strip()
        return cleaned

    # Preserve original title and clean
    movies_df['original_title'] = movies_df['orig_title'].copy()
    movies_df['orig_title'] = movies_df['orig_title'].apply(clean_title_text)
    
    # Remove duplicates
    movies_df = movies_df.drop_duplicates(subset=['orig_title']).reset_index(drop=True)
    
    # Create enhanced content
    movies_df['enhanced_content'] = (
        movies_df['overview'].astype(str) + ' ' +
        movies_df['genre'].astype(str).str.replace('|', ' ') + ' ' +
        movies_df['crew'].astype(str)
    )
    
    # Create synthetic vote_count if not exists
    if 'vote_count' not in movies_df.columns:
        movies_df['vote_count'] = (
            (movies_df['revenue'].fillna(0) / 1_000_000) *
            (movies_df['score'].fillna(5) / 2) *
            np.random.uniform(50, 500, len(movies_df))
        ).astype(int)
        movies_df['vote_count'] = movies_df['vote_count'].clip(lower=1)

    # Calculate weighted rating using IMDb formula
    average_rating = movies_df['score'].mean()
    vote_threshold = movies_df['vote_count'].quantile(0.90)
    
    def weighted_rating(x, m=vote_threshold, C=average_rating):
        v = x['vote_count']
        R = x['score']
        return (v/(v+m) * R) + (m/(m+v) * C)
    
    movies_df['weighted_rating'] = movies_df.apply(weighted_rating, axis=1)
    movies_df['weighted'] = movies_df['weighted_rating']
    
    # Build TF-IDF matrix
    print("📊 Building TF-IDF matrix...")
    tfidf_vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )
    
    movies_df['enhanced_content'] = movies_df['enhanced_content'].fillna('')
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['enhanced_content'])
    
    print("🔍 Computing cosine similarity...")
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    # Create indices mapping
    indices = pd.Series(movies_df.index, index=movies_df['orig_title']).drop_duplicates()
    
    # Package everything for saving
    model_data = {
        'movies_df': movies_df,
        'tfidf_vectorizer': tfidf_vectorizer,
        'tfidf_matrix': tfidf_matrix,
        'cosine_sim': cosine_sim,
        'indices': indices,
        'average_rating': average_rating,
        'vote_threshold': vote_threshold,
        'metadata': {
            'total_movies': len(movies_df),
            'trained_date': pd.Timestamp.now(),
            'model_version': '2.0'
        }
    }
    
    # Save the model
    print(f"💾 Saving model to {model_save_path}...")
    joblib.dump(model_data, model_save_path, compress=3)
    
    print(f"✅ Model training complete!")
    print(f"📈 Total movies: {len(movies_df)}")
    print(f"💿 Model saved to: {model_save_path}")
    print(f"📦 File size: {os.path.getsize(model_save_path) / (1024*1024):.1f} MB")
    
    return model_data


# ====================================================
# Enhanced Recommendation System Class
# ====================================================
class IMDBPreTrainedRecommendationSystem:
    def __init__(self, model_path="imdb_model.pkl"):
        self.model_path = model_path
        self.movies_df = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.indices = None
        self.average_rating = 0
        self.vote_threshold = 0
        self.is_loaded = False

    def load_pretrained_model(self):
        """Load the pre-trained model from joblib file"""
        if not os.path.exists(self.model_path):
            return False, f"❌ Model file not found: {self.model_path}"
        
        try:
            print(f"📂 Loading pre-trained model from {self.model_path}...")
            model_data = joblib.load(self.model_path)
            
            # Load all components
            self.movies_df = model_data['movies_df']
            self.tfidf_vectorizer = model_data['tfidf_vectorizer']
            self.tfidf_matrix = model_data['tfidf_matrix']
            self.cosine_sim = model_data['cosine_sim']
            self.indices = model_data['indices']
            self.average_rating = model_data['average_rating']
            self.vote_threshold = model_data['vote_threshold']
            
            self.is_loaded = True
            
            # Print model info
            metadata = model_data.get('metadata', {})
            print(f"✅ Model loaded successfully!")
            print(f"📊 Total movies: {metadata.get('total_movies', len(self.movies_df))}")
            print(f"📅 Trained: {metadata.get('trained_date', 'Unknown')}")
            print(f"🏷️ Version: {metadata.get('model_version', 'Unknown')}")
            
            return True, "Model loaded successfully!"
            
        except Exception as e:
            return False, f"❌ Error loading model: {str(e)}"

    def clean_title_text(self, text):
        if pd.isna(text):
            return ""
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text))
        cleaned = re.sub(r'\s+', ' ', cleaned).lower().strip()
        return cleaned

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
        """
        CENTRALIZED RESULTS DISPLAY - Shows comprehensive movie information in compact format
        """
        if results_df.empty:
            return "❌ No results found!"

        # Ensure we don't exceed available results
        display_count = min(n, len(results_df))
        display_results = results_df.head(display_count).copy()

        output = []
        output.append("="*80)
        output.append(f"🎬 {search_type.upper()} RESULTS FOR: '{original_query}'")
        output.append("="*80)
        output.append(f"📊 Showing {display_count} results:")
        output.append("-" * 80)

        for i, (idx, movie) in enumerate(display_results.iterrows()):
            # Build the main line with rank and movie name
            if i == 0 and search_type == "Content Recommendations":
                main_line = f"🏆 {i+1}. {movie['names'][:45]} ⭐ TOP MATCH!"
            else:
                main_line = f"🎬 {i+1}. {movie['names'][:45]}"

            # Build the info line with year and rating
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

            # Only show similarity if the column exists (in content recommendations)
            if 'similarity' in movie.index and search_type == "Content Recommendations":
                similarity_percent = movie['similarity'] * 100
                similarity_level = self.get_similarity_level(movie['similarity'])
                output.append(f"🎯 Similarity: {movie['similarity']:.4f} ({similarity_percent:.1f}%) - {similarity_level}")

            # Genre
            if 'genre' in movie.index:
                genre_display = str(movie['genre'])[:50]
                if len(str(movie['genre'])) > 50:
                    genre_display += "..."
                output.append(f"🎭 Genre: {genre_display}")

            # Crew
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
        if not self.is_loaded:
            return None, None, "❌ Model not loaded. Please load the model first."
            
        # Fuzzy matching
        if title not in self.indices:
            possible_matches = self.movies_df[
                self.movies_df['orig_title'].str.contains(title, case=False, na=False)
            ]
            if possible_matches.empty:
                return None, None, None
            return "choose", possible_matches.head(5), None

        # Exact matching
        idx = self.indices[title]
        
        # Handle multiple matches properly
        if hasattr(idx, '__iter__') and not isinstance(idx, str):
            idx = idx.iloc[0] if hasattr(idx, 'iloc') else idx[0]
            
        sim_scores = list(enumerate(self.cosine_sim[idx]))

        # Apply similarity scaling (0.8-1.0 range)
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
        
        movies = self.movies_df.iloc[movie_indices].copy()
        movies['similarity'] = similarity_values

        return "ok", self.movies_df.loc[idx], movies

    def search_by_genre(self, genre, n=10, show_details=True):
        """Search movies by genre with CENTRALIZED DISPLAY"""
        if not self.is_loaded:
            return "❌ Model not loaded. Please load the model first."
            
        try:
            genre_matches = self.movies_df[
                self.movies_df['genre'].str.contains(genre, case=False, na=False)
            ]

            if genre_matches.empty:
                return f"❌ No movies found with genre '{genre}'"

            # Sort by weighted rating (highest first)
            genre_matches = genre_matches.nlargest(n, 'weighted_rating')
            
            if show_details:
                return self.display_centralized_results(genre_matches, "Genre Search", genre, n)
            else:
                return genre_matches

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def search_by_crew(self, crew_name, n=10, show_details=True):
        """Search movies by crew member with CENTRALIZED DISPLAY"""
        if not self.is_loaded:
            return "❌ Model not loaded. Please load the model first."
            
        try:
            crew_matches = self.movies_df[
                self.movies_df['crew'].str.contains(crew_name, case=False, na=False)
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

    def advanced_search(self, genre=None, crew=None, min_rating=None, max_results=10, show_details=True):
        """Advanced search with CENTRALIZED DISPLAY"""
        if not self.is_loaded:
            return "❌ Model not loaded. Please load the model first."
            
        try:
            # Build query description
            query_parts = []
            if genre:
                query_parts.append(f"Genre: {genre}")
            if crew:
                query_parts.append(f"Crew: {crew}")
            if min_rating:
                query_parts.append(f"Rating ≥ {min_rating}")

            query_description = " | ".join(query_parts) if query_parts else "All Movies"

            # Start with all movies
            results = self.movies_df.copy()

            # Apply filters
            if genre:
                results = results[results['genre'].str.contains(genre, case=False, na=False)]

            if crew:
                results = results[results['crew'].str.contains(crew, case=False, na=False)]

            if min_rating:
                results = results[results['weighted_rating'] >= min_rating]

            if results.empty:
                return "❌ No movies match your criteria"

            # Sort by weighted rating
            results = results.nlargest(max_results, 'weighted_rating')

            if show_details:
                return self.display_centralized_results(results, "Advanced Search", query_description, max_results)
            else:
                return results

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def get_top_movies_by_rating(self, n=20, show_details=True):
        """Get top movies by weighted rating with centralized display"""
        if not self.is_loaded:
            return "❌ Model not loaded. Please load the model first."
            
        try:
            top_movies = self.movies_df.nlargest(n, 'weighted_rating').copy()

            if show_details:
                return self.display_centralized_results(top_movies, "Top Rated Movies", f"Top {n} Movies", n)
            else:
                return top_movies

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def search_by_year(self, year, n=10, show_details=True):
        """Search movies by release year with centralized display"""
        if not self.is_loaded:
            return "❌ Model not loaded. Please load the model first."
            
        try:
            # Extract year from date
            self.movies_df['year'] = pd.to_datetime(
                self.movies_df['date_x'], errors='coerce'
            ).dt.year

            year_matches = self.movies_df[
                self.movies_df['year'] == year
            ]

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
        """Search movies by country with centralized display"""
        if not self.is_loaded:
            return "❌ Model not loaded. Please load the model first."
            
        try:
            if 'country' not in self.movies_df.columns:
                return "❌ Country information not available in dataset"

            country_matches = self.movies_df[
                self.movies_df['country'].str.contains(country, case=False, na=False)
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
        """Search movies by original language with centralized display"""
        if not self.is_loaded:
            return "❌ Model not loaded. Please load the model first."
            
        try:
            if 'orig_lang' not in self.movies_df.columns:
                return "❌ Language information not available in dataset"

            lang_matches = self.movies_df[
                self.movies_df['orig_lang'].str.contains(language, case=False, na=False)
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

    def get_model_info(self):
        """Get information about the loaded model"""
        if not self.is_loaded:
            return "❌ No model loaded"
        
        info = []
        info.append("📊 MODEL INFORMATION")
        info.append("=" * 50)
        info.append(f"🎬 Total movies: {len(self.movies_df):,}")
        info.append(f"📁 Model file: {self.model_path}")
        info.append(f"💾 File size: {os.path.getsize(self.model_path) / (1024*1024):.1f} MB")
        info.append(f"⭐ Average rating: {self.average_rating:.2f}")
        info.append(f"🗳️ Vote threshold: {self.vote_threshold:.0f}")
        info.append(f"🔍 TF-IDF features: {self.tfidf_matrix.shape[1]:,}")
        info.append(f"📈 Similarity matrix: {self.cosine_sim.shape}")
        return "\n".join(info)


# ====================================================
# Auto-loading Functions
# ====================================================
@st.cache_resource
def load_recommender_automatically():
    """Automatically load the recommender system on startup - cached for performance"""
    recommender = IMDBPreTrainedRecommendationSystem()
    
    # Try to load the model automatically
    success, message = recommender.load_pretrained_model()
    
    if not success:
        # If default model doesn't exist, try common alternative names
        alternative_paths = ["model.pkl", "imdb_recommender.pkl", "movie_model.pkl"]
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                recommender.model_path = alt_path
                success, message = recommender.load_pretrained_model()
                if success:
                    break
    
    return recommender, success, message

# ====================================================
# Enhanced Streamlit UI with Auto-Loading
# ====================================================
def main():
    st.set_page_config(page_title="Ready-to-Use IMDB Recommender", layout="wide")
    st.title("🎬 INSTANT MOVIE RECOMMENDATION SYSTEM")
    st.markdown("=================================================================")
    st.markdown("✨ **READY TO USE** - No setup required, just start searching!")
    st.markdown("📊 **AUTO-LOADED** - Model loads automatically on startup")

    # Auto-load recommender system
    recommender, auto_load_success, auto_load_message = load_recommender_automatically()

    # Show loading status
    if auto_load_success:
        st.success("🚀 **SYSTEM READY!** - Model loaded successfully and ready for use!")
    else:
        st.error("❌ **NO MODEL FOUND** - Please train a model first")
        st.warning(auto_load_message)
        
        # Show training interface if no model exists
        st.markdown("---")
        st.header("🏗️ First Time Setup - Train Your Model")
        
        uploaded_file = st.file_uploader("Upload your IMDB CSV file to get started:", type="csv")
        model_name = st.text_input("Save model as:", value="imdb_model.pkl")
        
        if st.button("🚀 Train Model & Start Using") and uploaded_file:
            with st.spinner("Training your model... This will take a few minutes"):
                try:
                    # Save uploaded file temporarily
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    # Train model
                    train_and_save_model(temp_path, model_name)
                    
                    # Clean up
                    os.remove(temp_path)
                    
                    st.success(f"✅ Model trained successfully! Refreshing page...")
                    st.balloons()
                    
                    # Clear cache and rerun
                    st.cache_resource.clear()
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
        
        return  # Exit early if no model

    # Sidebar with minimal info (optional)
    with st.sidebar:
        st.markdown("### 📊 System Status")
        st.success("✅ Ready to use!")
        
        if st.button("📋 Model Details"):
            st.code(recommender.get_model_info())
        
        # Optional: Advanced training (collapsed by default)
        with st.expander("🔧 Advanced Options"):
            st.markdown("**Retrain with new data:**")
            uploaded_file = st.file_uploader("Upload new CSV", type="csv", key="retrain_csv")
            if st.button("🔄 Retrain Model") and uploaded_file:
                with st.spinner("Retraining..."):
                    try:
                        temp_path = f"temp_{uploaded_file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        train_and_save_model(temp_path, "imdb_model.pkl")
                        os.remove(temp_path)
                        st.cache_resource.clear()
                        st.success("✅ Retrained! Refreshing...")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

    # Main interface - Always visible when model is loaded
    st.markdown("---")
    
    # Enhanced menu with more options
    option = st.radio("🎯 **CHOOSE YOUR SEARCH TYPE:**", [
        "🎬 Movie Title (Get similar movies)",
        "🎭 Genre (Find movies by genre)",
        "👥 Actor/Director (Find movies by crew)",
        "🔍 Advanced Search (Multiple filters)",
        "📅 Release Year (Movies from specific year)",
        "🌍 Country (Movies by country)", 
        "🗣️ Language (Movies by language)",
        "⭐ Top Rated (Highest rated movies)"
    ], horizontal=True)

    # ----------------- Search by Title -----------------
    if "🎬" in option:
        col1, col2 = st.columns([3, 1])
        with col1:
            title = st.text_input("🎬 **Enter movie title:**", placeholder="e.g., Inception, Avatar, Titanic...")
        with col2:
            n_recs = st.selectbox("📊 Results:", [5, 10, 15, 20], index=1)

        if st.button("🚀 Get Recommendations", type="primary") and title:
            with st.spinner("Finding similar movies..."):
                cleaned_title = recommender.clean_title_text(title)
                status, movie_info, recs = recommender.get_content_recommendations(cleaned_title, n=n_recs)
                
                if status == "choose":
                    st.markdown("🔍 **Did you mean one of these movies?**")
                    choices = movie_info['names'].tolist()
                    choice = st.selectbox("Select the correct movie:", choices)
                    if st.button("✅ Confirm Selection"):
                        cleaned_choice = recommender.clean_title_text(choice)
                        status, movie_info, recs = recommender.get_content_recommendations(cleaned_choice, n=n_recs)
                        if status == "ok":
                            formatted_output = recommender.display_centralized_results(
                                recs, "Content Recommendations", f"{movie_info['names']}", n_recs
                            )
                            st.code(formatted_output, language="text")
                
                elif status == "ok":
                    formatted_output = recommender.display_centralized_results(
                        recs, "Content Recommendations", f"{movie_info['names']}", n_recs
                    )
                    st.code(formatted_output, language="text")
                
                else:
                    st.error("❌ Movie not found. Try a different title or check spelling.")

    # ----------------- Search by Genre -----------------
    elif "🎭" in option:
        col1, col2 = st.columns([3, 1])
        with col1:
            genre = st.text_input("🎭 **Enter genre:**", placeholder="e.g., Action, Comedy, Sci-Fi, Horror...")
        with col2:
            n_results = st.selectbox("📊 Results:", [5, 10, 15, 20], index=1)
        
        if st.button("🔍 Search by Genre", type="primary") and genre:
            with st.spinner("Searching movies..."):
                result = recommender.search_by_genre(genre, n=n_results, show_details=True)
                if isinstance(result, str):
                    if result.startswith("❌"):
                        st.error(result)
                    else:
                        st.code(result, language="text")

    # ----------------- Search by Crew -----------------
    elif "👥" in option:
        col1, col2 = st.columns([3, 1])
        with col1:
            crew = st.text_input("👥 **Enter actor or director name:**", placeholder="e.g., Leonardo DiCaprio, Christopher Nolan...")
        with col2:
            n_results = st.selectbox("📊 Results:", [5, 10, 15, 20], index=1)
        
        if st.button("🔍 Search by Crew", type="primary") and crew:
            with st.spinner("Searching movies..."):
                result = recommender.search_by_crew(crew, n=n_results, show_details=True)
                if isinstance(result, str):
                    if result.startswith("❌"):
                        st.error(result)
                    else:
                        st.code(result, language="text")

    # ----------------- Advanced Search -----------------
    elif "🔍" in option:
        st.markdown("🔍 **Advanced Search - Combine multiple criteria:**")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            genre = st.text_input("🎭 Genre:", placeholder="Optional")
        with col2:
            crew = st.text_input("👥 Crew:", placeholder="Optional")
        with col3:
            min_rating = st.number_input("⭐ Min Rating:", min_value=0.0, max_value=10.0, step=0.1, value=0.0)
        with col4:
            n_results = st.selectbox("📊 Max Results:", [5, 10, 15, 20], index=1)
        
        if st.button("🚀 Advanced Search", type="primary"):
            with st.spinner("Searching with filters..."):
                result = recommender.advanced_search(
                    genre=genre if genre else None,
                    crew=crew if crew else None,
                    min_rating=min_rating if min_rating > 0 else None,
                    max_results=n_results,
                    show_details=True
                )
                if isinstance(result, str):
                    if result.startswith("❌"):
                        st.error(result)
                    else:
                        st.code(result, language="text")

    # ----------------- Search by Year -----------------
    elif "📅" in option:
        col1, col2 = st.columns([3, 1])
        with col1:
            year_input = st.number_input("📅 **Enter release year:**", min_value=1900, max_value=2024, value=2020)
        with col2:
            n_results = st.selectbox("📊 Results:", [5, 10, 15, 20], index=1)
        
        if st.button("🔍 Search by Year", type="primary"):
            with st.spinner("Searching movies..."):
                result = recommender.search_by_year(int(year_input), n=n_results, show_details=True)
                if isinstance(result, str):
                    if result.startswith("❌"):
                        st.error(result)
                    else:
                        st.code(result, language="text")

    # ----------------- Search by Country -----------------
    elif "🌍" in option:
        col1, col2 = st.columns([3, 1])
        with col1:
            country = st.text_input("🌍 **Enter country:**", placeholder="e.g., USA, UK, France, Japan...")
        with col2:
            n_results = st.selectbox("📊 Results:", [5, 10, 15, 20], index=1)
        
        if st.button("🔍 Search by Country", type="primary") and country:
            with st.spinner("Searching movies..."):
                result = recommender.search_by_country(country, n=n_results, show_details=True)
                if isinstance(result, str):
                    if result.startswith("❌"):
                        st.error(result)
                    else:
                        st.code(result, language="text")

    # ----------------- Search by Language -----------------
    elif "🗣️" in option:
        col1, col2 = st.columns([3, 1])
        with col1:
            language = st.text_input("🗣️ **Enter language:**", placeholder="e.g., en, english, spanish, french...")
        with col2:
            n_results = st.selectbox("📊 Results:", [5, 10, 15, 20], index=1)
        
        if st.button("🔍 Search by Language", type="primary") and language:
            with st.spinner("Searching movies..."):
                result = recommender.search_by_language(language, n=n_results, show_details=True)
                if isinstance(result, str):
                    if result.startswith("❌"):
                        st.error(result)
                    else:
                        st.code(result, language="text")

    # ----------------- Top Rated Movies -----------------
    elif "⭐" in option:
        n_results = st.selectbox("📊 **How many top movies?**", [10, 20, 30, 50], index=1)
        
        if st.button("🏆 Get Top Movies", type="primary"):
            with st.spinner("Getting top rated movies..."):
                result = recommender.get_top_movies_by_rating(n=n_results, show_details=True)
                if isinstance(result, str):
                    if result.startswith("❌"):
                        st.error(result)
                    else:
                        st.code(result, language="text")

    # Footer with tips
    st.markdown("---")
    st.markdown("💡 **Tips:** Try different movie titles, genres like 'Action' or 'Sci-Fi', actor names, or years to discover great movies!")


# ====================================================
# Usage Instructions and Example
# ====================================================
def create_example_usage():
    """
    Example of how to use this system:
    
    1. FIRST TIME SETUP (Train and save model):
    ```python
    # Run this once to create your model
    from your_script import train_and_save_model
    train_and_save_model('path/to/your/imdb_movies.csv', 'imdb_model.pkl')
    ```
    
    2. NORMAL USAGE (Load and use pre-trained model):
    ```python
    # Run your Streamlit app - it will automatically load the saved model
    streamlit run your_script.py
    ```
    
    3. PROGRAMMATIC USAGE:
    ```python
    # Use the recommender in your own code
    recommender = IMDBPreTrainedRecommendationSystem('imdb_model.pkl')
    success, message = recommender.load_pretrained_model()
    
    if success:
        # Get recommendations
        status, movie_info, recs = recommender.get_content_recommendations('inception', n=10)
        
        # Search by genre
        result = recommender.search_by_genre('action', n=10)
        
        # Advanced search
        result = recommender.advanced_search(genre='sci-fi', min_rating=7.0, max_results=5)
    ```
    """
    pass


if __name__ == "__main__":
    main()
