import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import plotly.express as px
import plotly.graph_objects as go
import re
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="IMDB Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .recommendation-card {
        border: 2px solid #4ECDC4;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

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
        """Clean title text by removing special characters and converting to lowercase."""
        if pd.isna(text):
            return ""
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text))
        cleaned = re.sub(r'\s+', ' ', cleaned).lower().strip()
        return cleaned

    @st.cache_data
    def load_data(_self, uploaded_file):
        """Load and cache the IMDB dataset"""
        try:
            df = pd.read_csv(uploaded_file, low_memory=False)
            return df
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None

    def preprocess_data(self):
        """Clean and preprocess the movie data"""
        # Fill missing values
        self.movies_df['overview'] = self.movies_df['overview'].fillna('No description available')
        self.movies_df['genre'] = self.movies_df['genre'].fillna('Unknown')
        self.movies_df['crew'] = self.movies_df['crew'].fillna('Unknown')

        # Store original title and clean titles
        self.movies_df['original_title'] = self.movies_df['orig_title'].copy()
        self.movies_df['orig_title'] = self.movies_df['orig_title'].apply(self.clean_title_text)

        # Remove duplicates
        before_count = len(self.movies_df)
        self.movies_df = self.movies_df.drop_duplicates(
            subset=['orig_title'], keep='first'
        ).reset_index(drop=True)
        after_count = len(self.movies_df)

        # Create enhanced content features
        self.movies_df['enhanced_content'] = (
            self.movies_df['overview'].astype(str) + ' ' +
            self.movies_df['genre'].astype(str).str.replace('|', ' ') + ' ' +
            self.movies_df['crew'].astype(str)
        )

        return before_count - after_count

    def calculate_weighted_ratings(self):
        """Calculate weighted ratings using IMDB formula"""
        # Create synthetic vote count if not available
        if 'vote_count' not in self.movies_df.columns:
            self.movies_df['vote_count'] = (
                (self.movies_df['revenue'].fillna(0) / 1000000) *
                (self.movies_df['score'].fillna(5) / 2) *
                np.random.uniform(50, 500, len(self.movies_df))
            ).astype(int)
            self.movies_df['vote_count'] = self.movies_df['vote_count'].clip(lower=1)

        # Calculate global statistics
        self.average_rating = self.movies_df['score'].mean()
        self.vote_threshold = self.movies_df['vote_count'].quantile(0.90)

        # Apply weighted rating formula
        def weighted_rating(x, m=self.vote_threshold, C=self.average_rating):
            v = x['vote_count']
            R = x['score']
            return (v/(v+m) * R) + (m/(m+v) * C)

        self.movies_df['weighted_rating'] = self.movies_df.apply(weighted_rating, axis=1)
        self.qualified_movies = self.movies_df.copy()

    def build_content_system(self):
        """Build the content-based filtering system"""
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
        """Get content-based recommendations"""
        try:
            # Handle fuzzy matching
            if title not in self.indices:
                possible_matches = self.qualified_movies[
                    self.qualified_movies['orig_title'].str.contains(title, case=False, na=False)
                ]
                if possible_matches.empty:
                    return None, f"Movie '{title}' not found!"
                
                # Return the first match for simplicity in web interface
                title = possible_matches.iloc[0]['orig_title']

            # Get movie index
            idx = self.indices[title]
            if hasattr(idx, '__iter__') and not isinstance(idx, str):
                idx = idx.iloc[0] if hasattr(idx, 'iloc') else idx[0]

            # Get movie info
            movie_info = self.qualified_movies.loc[idx]

            # Calculate similarities
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

            # Sort and get top recommendations
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
        """Search movies by genre"""
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
        """Search movies by crew member"""
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
        """Advanced search with multiple criteria"""
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
        """Get list of available genres"""
        all_genres = []
        for genres in self.qualified_movies['genre'].dropna():
            all_genres.extend(str(genres).split('|'))
        return pd.Series(all_genres).value_counts().head(20)

    def get_popular_crew(self):
        """Get list of popular crew members"""
        all_crew = []
        for crew in self.qualified_movies['crew'].dropna():
            crew_members = re.split(r'[,|;]', str(crew))
            for member in crew_members:
                member = member.strip()
                if len(member) > 2:
                    all_crew.append(member)
        return pd.Series(all_crew).value_counts().head(20)

# Initialize the recommendation system
@st.cache_resource
def initialize_system():
    return StreamlitIMDBRecommendationSystem()

def main():
    # Header
    st.markdown('<h1 class="main-header">🎬 IMDB Movie Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Enhanced Content-Based Filtering with Weighted Ratings</p>', unsafe_allow_html=True)
    
    # Initialize system
    recommender = initialize_system()
    
    # Sidebar for file upload and navigation
    st.sidebar.title("🎯 Navigation")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload IMDB CSV Dataset", 
        type=['csv'],
        help="Upload your imdb_movies.csv file"
    )
    
    if uploaded_file is not None:
        # Load and process data
        if recommender.movies_df is None:
            with st.spinner("Loading and processing dataset..."):
                recommender.movies_df = recommender.load_data(uploaded_file)
                if recommender.movies_df is not None:
                    duplicates_removed = recommender.preprocess_data()
                    recommender.calculate_weighted_ratings()
                    recommender.build_content_system()
                    
                    st.sidebar.success("✅ Dataset loaded successfully!")
                    st.sidebar.info(f"📊 Total movies: {len(recommender.movies_df)}")
                    st.sidebar.info(f"🗑️ Duplicates removed: {duplicates_removed}")
        
        # Navigation menu
        menu_option = st.sidebar.selectbox(
            "Choose Search Type",
            ["🏠 Dashboard", "🎬 Movie Recommendations", "🎭 Search by Genre", 
             "👥 Search by Crew", "🔍 Advanced Search", "📊 Analytics"]
        )
        
        if recommender.movies_df is not None:
            
            # Dashboard
            if menu_option == "🏠 Dashboard":
                st.markdown("## 📊 Dataset Overview")
                
                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Movies", len(recommender.movies_df))
                with col2:
                    st.metric("Average Rating", f"{recommender.average_rating:.2f}")
                with col3:
                    st.metric("Genres Available", len(recommender.get_genre_list()))
                with col4:
                    st.metric("Vote Threshold", f"{recommender.vote_threshold:.0f}")
                
                # Dataset preview
                st.subheader("📋 Dataset Preview")
                display_cols = ['names', 'genre', 'score', 'weighted_rating']
                if 'overview' in recommender.movies_df.columns:
                    display_cols.append('overview')
                st.dataframe(recommender.movies_df[display_cols].head(10))
                
                # Quick stats
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🎭 Top Genres")
                    genre_counts = recommender.get_genre_list().head(10)
                    fig = px.bar(x=genre_counts.values, y=genre_counts.index, orientation='h',
                               labels={'x': 'Count', 'y': 'Genre'}, title="Most Popular Genres")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("⭐ Rating Distribution")
                    fig = px.histogram(recommender.movies_df, x='weighted_rating', nbins=30,
                                     title="Weighted Rating Distribution")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Movie Recommendations
            elif menu_option == "🎬 Movie Recommendations":
                st.markdown("## 🎬 Get Movie Recommendations")
                
                # Movie search input
                col1, col2 = st.columns([3, 1])
                with col1:
                    movie_title = st.text_input(
                        "Enter a movie title:",
                        placeholder="e.g., The Godfather, Pulp Fiction, Inception..."
                    )
                with col2:
                    num_recs = st.number_input("Number of recommendations", min_value=1, max_value=20, value=10)
                
                if movie_title:
                    cleaned_title = recommender.clean_title_text(movie_title)
                    movie_info, recommendations = recommender.get_content_recommendations(cleaned_title, num_recs)
                    
                    if movie_info is not None:
                        # Display found movie info
                        st.success(f"🎯 Found: {movie_info['names']}")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Original Score", f"{movie_info['score']:.1f}")
                        with col2:
                            st.metric("Weighted Rating", f"{movie_info['weighted_rating']:.2f}")
                        with col3:
                            try:
                                year = pd.to_datetime(movie_info['date_x'], errors='coerce').year
                                st.metric("Year", year if pd.notna(year) else "Unknown")
                            except:
                                st.metric("Year", "Unknown")
                        
                        st.write(f"**Genre:** {movie_info['genre']}")
                        st.write(f"**Overview:** {str(movie_info['overview'])[:300]}...")
                        
                        # Display recommendations
                        st.markdown("### 🔥 Recommendations")
                        
                        for i, (_, rec) in enumerate(recommendations.iterrows()):
                            with st.container():
                                st.markdown(f"""
                                <div class="recommendation-card">
                                    <h4>{'🏆' if i == 0 else '🎬'} {i+1}. {rec['names']}</h4>
                                    <p><strong>Similarity:</strong> {rec['similarity']:.4f} ({rec['similarity']*100:.1f}%)</p>
                                    <p><strong>Rating:</strong> {rec['weighted_rating']:.2f}</p>
                                    <p><strong>Genre:</strong> {rec['genre']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.error(recommendations)
            
            # Search by Genre
            elif menu_option == "🎭 Search by Genre":
                st.markdown("## 🎭 Search Movies by Genre")
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    # Genre input with suggestions
                    available_genres = recommender.get_genre_list().index.tolist()
                    selected_genre = st.selectbox("Select a genre:", [""] + available_genres[:15])
                    if not selected_genre:
                        custom_genre = st.text_input("Or enter a custom genre:")
                        selected_genre = custom_genre
                
                with col2:
                    num_results = st.number_input("Number of results", min_value=1, max_value=20, value=10)
                
                if selected_genre:
                    results = recommender.search_by_genre(selected_genre, num_results)
                    if results is not None and not results.empty:
                        st.success(f"✅ Found {len(results)} movies in '{selected_genre}' genre")
                        
                        # Display results in a clean format
                        for i, (_, movie) in enumerate(results.iterrows()):
                            col1, col2, col3 = st.columns([3, 1, 1])
                            with col1:
                                st.write(f"**{i+1}. {movie['names']}**")
                                st.write(f"Genres: {movie['genre']}")
                            with col2:
                                st.metric("Rating", f"{movie['weighted_rating']:.2f}")
                            with col3:
                                try:
                                    year = pd.to_datetime(movie['date_x'], errors='coerce').year
                                    st.metric("Year", year if pd.notna(year) else "?")
                                except:
                                    st.metric("Year", "?")
                            st.divider()
                    else:
                        st.warning(f"No movies found for genre '{selected_genre}'")
            
            # Search by Crew
            elif menu_option == "👥 Search by Crew":
                st.markdown("## 👥 Search Movies by Crew Member")
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    # Crew input with suggestions
                    popular_crew = recommender.get_popular_crew().index.tolist()
                    selected_crew = st.selectbox("Select a crew member:", [""] + popular_crew[:15])
                    if not selected_crew:
                        custom_crew = st.text_input("Or enter a crew member name:")
                        selected_crew = custom_crew
                
                with col2:
                    num_results = st.number_input("Number of results", min_value=1, max_value=20, value=10, key="crew_num")
                
                if selected_crew:
                    results = recommender.search_by_crew(selected_crew, num_results)
                    if results is not None and not results.empty:
                        st.success(f"✅ Found {len(results)} movies with '{selected_crew}'")
                        
                        for i, (_, movie) in enumerate(results.iterrows()):
                            col1, col2, col3 = st.columns([3, 1, 1])
                            with col1:
                                st.write(f"**{i+1}. {movie['names']}**")
                                st.write(f"Crew: {movie['crew'][:100]}...")
                            with col2:
                                st.metric("Rating", f"{movie['weighted_rating']:.2f}")
                            with col3:
                                try:
                                    year = pd.to_datetime(movie['date_x'], errors='coerce').year
                                    st.metric("Year", year if pd.notna(year) else "?")
                                except:
                                    st.metric("Year", "?")
                            st.divider()
                    else:
                        st.warning(f"No movies found with crew member '{selected_crew}'")
            
            # Advanced Search
            elif menu_option == "🔍 Advanced Search":
                st.markdown("## 🔍 Advanced Multi-Criteria Search")
                
                # Search criteria
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    genre_filter = st.text_input("Genre (optional):", placeholder="e.g., Action, Comedy")
                
                with col2:
                    crew_filter = st.text_input("Crew Member (optional):", placeholder="e.g., Leonardo DiCaprio")
                
                with col3:
                    min_rating = st.number_input("Minimum Rating (optional):", min_value=0.0, max_value=10.0, step=0.1, value=None)
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    max_results = st.number_input("Max Results", min_value=1, max_value=50, value=10)
                with col2:
                    search_button = st.button("🔍 Search Movies", type="primary")
                
                if search_button and (genre_filter or crew_filter or min_rating):
                    results = recommender.advanced_search(
                        genre=genre_filter if genre_filter else None,
                        crew=crew_filter if crew_filter else None,
                        min_rating=min_rating,
                        max_results=max_results
                    )
                    
                    if results is not None and not results.empty:
                        st.success(f"✅ Found {len(results)} movies matching your criteria")
                        
                        # Display results in a table
                        display_df = results[['names', 'weighted_rating', 'genre']].copy()
                        display_df.columns = ['Movie Title', 'Rating', 'Genres']
                        st.dataframe(display_df, use_container_width=True)
                        
                    else:
                        st.warning("No movies match your search criteria. Try adjusting your filters.")
            
            # Analytics
            elif menu_option == "📊 Analytics":
                st.markdown("## 📊 System Analytics & Visualizations")
                
                # Create visualizations
                tab1, tab2, tab3 = st.tabs(["📈 Rating Analysis", "🎭 Genre Analytics", "🌍 Geographic Distribution"])
                
                with tab1:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Rating distribution comparison
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(x=recommender.movies_df['score'], 
                                                 name='Original Ratings', opacity=0.7, nbinsx=30))
                        fig.add_trace(go.Histogram(x=recommender.movies_df['weighted_rating'], 
                                                 name='Weighted Ratings', opacity=0.7, nbinsx=30))
                        fig.update_layout(title="Rating Distribution: Original vs Weighted",
                                        xaxis_title="Rating", yaxis_title="Frequency")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Scatter plot: votes vs ratings
                        fig = px.scatter(recommender.movies_df, x='vote_count', y='weighted_rating',
                                       color='score', title="Vote Count vs Weighted Rating",
                                       labels={'vote_count': 'Vote Count', 'weighted_rating': 'Weighted Rating'})
                        fig.update_xaxes(type="log")
                        st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    # Genre analysis
                    genre_counts = recommender.get_genre_list()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = px.pie(values=genre_counts.values[:8], names=genre_counts.index[:8],
                                   title="Genre Distribution (Top 8)")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Top rated movies by genre
                        top_movies = recommender.qualified_movies.nlargest(15, 'weighted_rating')
                        fig = px.bar(top_movies, x='weighted_rating', y='names',
                                   orientation='h', title="Top 15 Movies by Rating")
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                
                with tab3:
                    if 'country' in recommender.qualified_movies.columns:
                        # Country distribution
                        country_counts = recommender.qualified_movies['country'].value_counts().head(15)
                        fig = px.bar(x=country_counts.index, y=country_counts.values,
                                   title="Movies by Country", labels={'x': 'Country', 'y': 'Count'})
                        fig.update_xaxes(tickangle=45)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Average rating by country
                        avg_rating_by_country = recommender.qualified_movies.groupby('country')['weighted_rating'].mean().sort_values(ascending=False).head(10)
                        fig = px.bar(x=avg_rating_by_country.index, y=avg_rating_by_country.values,
                                   title="Average Rating by Country (Top 10)", 
                                   labels={'x': 'Country', 'y': 'Average Rating'})
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Country data not available in the dataset")
    
    else:
        # Welcome screen
        st.markdown("""
        ## Welcome to the Enhanced IMDB Movie Recommendation System!
        
        **Features:**
        - 🎬 **Content-Based Recommendations** - Find similar movies based on plot, genre, and crew
        - 🎭 **Genre Search** - Discover top-rated movies in specific genres
        - 👥 **Crew Search** - Find movies featuring specific actors or directors
        - 🔍 **Advanced Search** - Combine multiple criteria for precise results
        - 📊 **Analytics Dashboard** - Visualize dataset insights and trends
        - ⚖️ **Weighted Rating System** - Corrects bias using IMDB's formula
        
        **To get started:**
        1. Upload your `imdb_movies.csv` file using the sidebar
        2. Wait for the system to process the data
        3. Choose a search type from the navigation menu
        4. Start discovering amazing movies!
        
        **System Highlights:**
        - Uses TF-IDF vectorization for content analysis
        - Implements cosine similarity for movie matching
        - Applies weighted ratings to handle rating bias
        - Provides interactive visualizations and analytics
        """)
        
        # Add feature highlights
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("🎬 **Content-Based Filtering**\n\nFind movies similar to your favorites based on plot, cast, and genre.")
        with col2:
            st.info("⚖️ **Bias Correction**\n\nUses IMDB's weighted rating formula to provide trustworthy recommendations.")
        with col3:
            st.info("📊 **Advanced Analytics**\n\nExplore dataset insights with interactive visualizations.")

if __name__ == "__main__":
    main()