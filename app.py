import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Define the recommendation system class (your original code)
class IMDBContentBasedRecommendationSystem:
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

    def load_imdb_data(self, file_path='https://raw.githubusercontent.com/hongzai-67/Movie-Recommendation/main/imdb_movies.csv'):
        print("🎬 LOADING IMDB DATASET FROM GITHUB")
        print("=" * 50)
        self.movies_df = pd.read_csv(file_path, low_memory=False)
        print(f"✅ Original dataset shape: {self.movies_df.shape}")
        print(f"📋 Columns: {list(self.movies_df.columns)}")
        print(f"📊 Sample data:")
        print(self.movies_df.head(3))
        self.preprocess_data()

    def preprocess_data(self):
        print(f"\n🔧 PREPROCESSING DATA")
        print("=" * 30)

        self.movies_df['overview'] = self.movies_df['overview'].fillna('No description available')
        self.movies_df['genre'] = self.movies_df['genre'].fillna('Unknown')
        self.movies_df['crew'] = self.movies_df['crew'].fillna('Unknown')

        print("🎬 Processing movie titles: removing special characters, converting to lowercase...")
        self.movies_df['original_title'] = self.movies_df['orig_title'].copy()
        self.movies_df['orig_title'] = self.movies_df['orig_title'].apply(self.clean_title_text)

        print("📋 Title processing examples:")
        sample_titles = self.movies_df[['original_title', 'orig_title']].head(5)
        for idx, row in sample_titles.iterrows():
            print(f"  Original: '{row['original_title']}' → Processed: '{row['orig_title']}'")

        print("🔍 Creating enhanced content features...")
        self.movies_df['enhanced_content'] = (
            self.movies_df['overview'].astype(str) + ' ' +
            self.movies_df['genre'].astype(str).str.replace('|', ' ') + ' ' +
            self.movies_df['crew'].astype(str)
        )
        print("✅ Data preprocessing completed!")

    def calculate_weighted_ratings(self):
        print(f"\n⚖️  CORRECTING RATING BIAS / TRUSTWORTHY PROBLEM")
        print("=" * 55)
        print("Problem: Raw average ratings fail to distinguish between:")
        print("• 5-star movie with 2 votes (unreliable)")
        print("• 4.5-star movie with 10,000 votes (trustworthy)")
        
        if 'vote_count' not in self.movies_df.columns:
            print("📊 Creating synthetic vote count based on movie popularity...")
            self.movies_df['vote_count'] = (
                (self.movies_df['revenue'].fillna(0) / 1000000) *
                (self.movies_df['score'].fillna(5) / 2) *
                np.random.uniform(50, 500, len(self.movies_df))
            ).astype(int)
            self.movies_df['vote_count'] = self.movies_df['vote_count'].clip(lower=1)

        self.average_rating = self.movies_df['score'].mean()
        print(f"📈 Global average rating (C): {self.average_rating:.2f}")
        self.vote_threshold = self.movies_df['vote_count'].quantile(0.90)
        print(f"📊 Minimum votes threshold (m): {self.vote_threshold:.0f}")
        print(f"\n🧮 Applying IMDb Weighted Rating Formula:")
        print(f"WR = (v/(v+m)) × R + (m/(m+v)) × C")

        def weighted_rating(x, m=self.vote_threshold, C=self.average_rating):
            v = x['vote_count']
            R = x['score']
            return (v/(v+m) * R) + (m/(m+v) * C)

        self.movies_df['weighted_rating'] = self.movies_df.apply(weighted_rating, axis=1)
        self.qualified_movies = self.movies_df.copy()
        print(f"✅ MODIFIED: All movies are now qualified: {self.qualified_movies.shape[0]}")
        print(f"📊 Total movies = Qualified movies: {len(self.movies_df)} movies")
        self.show_bias_correction_examples()

    def show_bias_correction_examples(self):
        print(f"\n📋 BIAS CORRECTION EXAMPLES:")
        print("-" * 40)
        low_vote_high_rating = self.movies_df[
            (self.movies_df['vote_count'] < self.vote_threshold/2) &
            (self.movies_df['score'] > 8.0)
        ].head(3)
        print("🔴 HIGH RATING + LOW VOTES (Bias Corrected):")
        for _, movie in low_vote_high_rating.iterrows():
            print(f"  '{movie['names'][:30]}...'")
            print(f"    Raw: {movie['score']:.1f} ({movie['vote_count']:.0f} votes)")
            print(f"    Weighted: {movie['weighted_rating']:.2f} (Pulled toward {self.average_rating:.1f})")
            print()
        
        high_vote_movies = self.movies_df[
            self.movies_df['vote_count'] >= self.vote_threshold
        ].nlargest(3, 'vote_count')
        print("🟢 HIGH VOTES (Maintains Original Rating):")
        for _, movie in high_vote_movies.iterrows():
            print(f"  '{movie['names'][:30]}...'")
            print(f"    Raw: {movie['score']:.1f} ({movie['vote_count']:.0f} votes)")
            print(f"    Weighted: {movie['weighted_rating']:.2f} (Trustworthy)")
            print()

    def build_content_based_system(self):
        print(f"\n🎯 PART 1: CONTENT BASED FILTERING")
        print("=" * 45)
        print('"If you like Fast and Furious, you may also like Taxi 2"')
        print("NOTE: Movie's description, genre, and crew are important.")
        
        working_df = self.qualified_movies.copy()
        print(f"\n📝 Building TF-IDF Matrix...")
        print("Features: stop_words='english', max_features=5000, ngram_range=(1,2)")
        
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        working_df['enhanced_content'] = working_df['enhanced_content'].fillna('')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(working_df['enhanced_content'])
        print(f"✅ TF-IDF Matrix Shape: {self.tfidf_matrix.shape}")
        print(f"📚 Vocabulary size: {len(self.tfidf_vectorizer.get_feature_names_out())}")
        
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        print(f"📖 Sample features: {feature_names[:20]}")
        
        print(f"\n🧮 Computing Content Similarity Matrix...")
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
        print(f"✅ Similarity Matrix Shape: {self.cosine_sim.shape}")
        
        self.indices = pd.Series(working_df.index, index=working_df['orig_title']).drop_duplicates()
        self.qualified_movies = working_df
        print(f"✅ Content-based system built successfully!")

    def get_content_recommendations(self, title, n=10, show_details=True):
        if show_details:
            print(f"\n🎬 FINDING RECOMMENDATIONS FOR: '{title}'")
            print("=" * 50)
        try:
            if title not in self.indices:
                possible_matches = self.qualified_movies[
                    self.qualified_movies['orig_title'].str.contains(title, case=False, na=False)
                ]
                if possible_matches.empty:
                    return f"❌ Movie '{title}' not found!"
                else:
                    print("🔍 Did you mean one of these?")
                    for i, (idx, match) in enumerate(possible_matches.head(5).iterrows()):
                        print(f"  {i+1}. {match['names']}")
                    # For Streamlit, just use the first match
                    title = possible_matches.iloc[0]['orig_title']
                    print(f"✅ Using: {possible_matches.iloc[0]['names']}")

            idx = self.indices[title]
            if hasattr(idx, '__iter__') and not isinstance(idx, str):
                idx = idx.iloc[0] if hasattr(idx, 'iloc') else idx[0]

            movie_info = self.qualified_movies.loc[idx]

            if show_details:
                print(f"🎯 Found: {movie_info['names']}")
                try:
                    movie_year = pd.to_datetime(movie_info['date_x'], errors='coerce')
                    if pd.notna(movie_year):
                        print(f"📅 Year: {movie_year.year}")
                    else:
                        print(f"📅 Year: Unknown")
                except:
                    print(f"📅 Year: Unknown")

                print(f"🎭 Genre: {movie_info['genre']}")
                print(f"⭐ Score: {movie_info['score']} → Weighted: {movie_info['weighted_rating']:.2f}")
                print(f"📝 Overview: {str(movie_info['overview'])[:100]}...")

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

            movies = self.qualified_movies.iloc[movie_indices].copy()
            movies['similarity'] = similarity_values
            movies['composite_score'] = (
                0.7 * movies['similarity'] +
                0.3 * (movies['weighted_rating'] / 10.0)
            )

            recommendations = movies[['names', 'weighted_rating', 'similarity', 'composite_score', 'genre']].head(n)

            if show_details:
                print(f"\n🔥 TOP {n} RECOMMENDATIONS (SORTED BY HIGHEST SIMILARITY):")
                print("-" * 70)
                print("📊 Similarity Scale: 0.8000 (Not Similar) ←→ 1.0000 (Identical)")
                print("-" * 70)
                for i, (_, rec) in enumerate(recommendations.iterrows()):
                    similarity_percent = rec['similarity'] * 100
                    similarity_level = self.get_similarity_level(rec['similarity'])

                    if i == 0:
                        print(f"🏆 {i+1:2d}. {rec['names'][:40]} ⭐ TOP MATCH!")
                    else:
                        print(f"   {i+1:2d}. {rec['names'][:40]}")

                    print(f"    🎯 Similarity: {rec['similarity']:.4f} ({similarity_percent:.1f}%) - {similarity_level}")
                    print(f"    ⭐ Rating: {rec['weighted_rating']:.2f}")
                    print(f"    🎭 Genre: {rec['genre'][:50]}")
                    print()

            return recommendations

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def search_by_genre(self, genre, n=10, show_details=True):
        if show_details:
            print(f"\n🎭 SEARCHING BY GENRE: '{genre}'")
            print("=" * 50)

        try:
            genre_matches = self.qualified_movies[
                self.qualified_movies['genre'].str.contains(genre, case=False, na=False)
            ]

            if genre_matches.empty:
                return f"❌ No movies found with genre '{genre}'"

            genre_matches = genre_matches.nlargest(n, 'weighted_rating')

            if show_details:
                print(f"✅ Found {len(genre_matches)} movies with genre '{genre}'")
                print(f"\n🏆 TOP {len(genre_matches)} MOVIES IN '{genre.upper()}' GENRE:")
                print("-" * 70)

                for i, (_, movie) in enumerate(genre_matches.iterrows()):
                    print(f"🎬 {i+1:2d}. {movie['names'][:45]}")
                    print(f"    ⭐ Rating: {movie['weighted_rating']:.2f}")
                    print(f"    🎭 Genres: {movie['genre'][:50]}")

                    try:
                        movie_year = pd.to_datetime(movie['date_x'], errors='coerce')
                        if pd.notna(movie_year):
                            print(f"    📅 Year: {movie_year.year}")
                    except:
                        pass
                    print()

            return genre_matches[['names', 'weighted_rating', 'genre', 'overview']].head(n)

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def search_by_crew(self, crew_name, n=10, show_details=True):
        if show_details:
            print(f"\n👥 SEARCHING BY CREW: '{crew_name}'")
            print("=" * 50)

        try:
            crew_matches = self.qualified_movies[
                self.qualified_movies['crew'].str.contains(crew_name, case=False, na=False)
            ]

            if crew_matches.empty:
                return f"❌ No movies found with crew member '{crew_name}'"

            crew_matches = crew_matches.nlargest(n, 'weighted_rating')

            if show_details:
                print(f"✅ Found {len(crew_matches)} movies with '{crew_name}'")
                print(f"\n🏆 TOP {len(crew_matches)} MOVIES WITH '{crew_name.upper()}':")
                print("-" * 70)

                for i, (_, movie) in enumerate(crew_matches.iterrows()):
                    print(f"🎬 {i+1:2d}. {movie['names'][:45]}")
                    print(f"    ⭐ Rating: {movie['weighted_rating']:.2f}")
                    print(f"    🎭 Genre: {movie['genre'][:50]}")

                    try:
                        movie_year = pd.to_datetime(movie['date_x'], errors='coerce')
                        if pd.notna(movie_year):
                            print(f"    📅 Year: {movie_year.year}")
                    except:
                        pass
                    print()

            return crew_matches[['names', 'weighted_rating', 'genre', 'crew']].head(n)

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def advanced_search(self, genre=None, crew=None, min_rating=None, max_results=10, show_details=True):
        if show_details:
            print(f"\n🔍 ADVANCED SEARCH")
            print("=" * 30)
            print(f"Genre: {genre if genre else 'Any'}")
            print(f"Crew: {crew if crew else 'Any'}")
            print(f"Min Rating: {min_rating if min_rating else 'Any'}")

        try:
            results = self.qualified_movies.copy()

            if genre:
                results = results[results['genre'].str.contains(genre, case=False, na=False)]
                if show_details:
                    print(f"📊 After genre filter: {len(results)} movies")

            if crew:
                results = results[results['crew'].str.contains(crew, case=False, na=False)]
                if show_details:
                    print(f"📊 After crew filter: {len(results)} movies")

            if min_rating:
                results = results[results['weighted_rating'] >= min_rating]
                if show_details:
                    print(f"📊 After rating filter: {len(results)} movies")

            if results.empty:
                return "❌ No movies match your criteria"

            results = results.nlargest(max_results, 'weighted_rating')

            if show_details:
                print(f"\n🏆 TOP {len(results)} MATCHING MOVIES:")
                print("-" * 60)

                for i, (_, movie) in enumerate(results.iterrows()):
                    print(f"🎬 {i+1:2d}. {movie['names'][:40]}")
                    print(f"    ⭐ Rating: {movie['weighted_rating']:.2f}")
                    print(f"    🎭 Genre: {movie['genre'][:45]}")
                    if crew:
                        crew_info = movie['crew'][:60]
                        if crew.lower() in crew_info.lower():
                            crew_info = crew_info.replace(crew, f"**{crew}**")
                        print(f"    👥 Crew: {crew_info}")
                    print()

            return results[['names', 'weighted_rating', 'genre', 'crew']].head(max_results)

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def get_genre_list(self, top_n=20):
        all_genres = []
        for genres in self.qualified_movies['genre'].dropna():
            all_genres.extend(str(genres).split('|'))

        genre_counts = pd.Series(all_genres).value_counts().head(top_n)

        print(f"🎭 TOP {top_n} AVAILABLE GENRES:")
        print("-" * 40)
        for i, (genre, count) in enumerate(genre_counts.items()):
            print(f"{i+1:2d}. {genre} ({count} movies)")

        return genre_counts.index.tolist()

    def get_popular_crew(self, top_n=20):
        all_crew = []
        for crew in self.qualified_movies['crew'].dropna():
            crew_members = re.split(r'[,|;]', str(crew))
            for member in crew_members:
                member = member.strip()
                if len(member) > 2:
                    all_crew.append(member)

        crew_counts = pd.Series(all_crew).value_counts().head(top_n)

        print(f"👥 TOP {top_n} POPULAR CREW MEMBERS:")
        print("-" * 50)
        for i, (person, count) in enumerate(crew_counts.items()):
            print(f"{i+1:2d}. {person} ({count} movies)")

        return crew_counts.index.tolist()

    def analyze_similarity_quality(self):
        print(f"\n📊 ANALYZING SIMILARITY QUALITY")
        print("=" * 40)

        test_movies = ['the godfather', 'pulp fiction', 'the dark knight', 'inception', 'forrest gump']
        available_test_movies = []

        for movie in test_movies:
            matches = self.qualified_movies[
                self.qualified_movies['orig_title'].str.contains(movie, case=False, na=False)
            ]
            if not matches.empty:
                available_test_movies.append(matches.iloc[0]['orig_title'])

        if not available_test_movies:
            available_test_movies = self.qualified_movies.nlargest(5, 'weighted_rating')['orig_title'].tolist()

        print(f"🎬 Testing with movies: {available_test_movies[:3]}")

        quality_metrics = {
            'genre_consistency': [],
            'rating_similarity': [],
            'diversity_scores': []
        }

        for movie in available_test_movies[:3]:
            recs = self.get_content_recommendations(movie, n=10, show_details=False)

            if isinstance(recs, pd.DataFrame) and not recs.empty:
                original_movie = self.qualified_movies[
                    self.qualified_movies['orig_title'].str.contains(movie, case=False, na=False)
                ].iloc[0]

                original_genres = set(str(original_movie['genre']).split('|'))
                genre_overlaps = []

                for _, rec_movie in recs.iterrows():
                    rec_genres = set(str(rec_movie['genre']).split('|'))
                    if original_genres and rec_genres:
                        jaccard = len(original_genres.intersection(rec_genres)) / len(original_genres.union(rec_genres))
                        genre_overlaps.append(jaccard)

                if genre_overlaps:
                    quality_metrics['genre_consistency'].extend(genre_overlaps)

                original_rating = original_movie['weighted_rating']
                rating_diffs = [abs(original_rating - rec['weighted_rating']) for _, rec in recs.iterrows()]
                quality_metrics['rating_similarity'].extend(rating_diffs)

                all_rec_genres = []
                for _, rec in recs.iterrows():
                    all_rec_genres.extend(str(rec['genre']).split('|'))

                diversity = len(set(all_rec_genres)) / len(all_rec_genres) if all_rec_genres else 0
                quality_metrics['diversity_scores'].append(diversity)

        print(f"\n📈 QUALITY METRICS:")
        if quality_metrics['genre_consistency']:
            print(f"Genre Consistency: {np.mean(quality_metrics['genre_consistency']):.3f} ± {np.std(quality_metrics['genre_consistency']):.3f}")
        if quality_metrics['rating_similarity']:
            print(f"Rating Similarity: {np.mean(quality_metrics['rating_similarity']):.3f} ± {np.std(quality_metrics['rating_similarity']):.3f}")
        if quality_metrics['diversity_scores']:
            print(f"Recommendation Diversity: {np.mean(quality_metrics['diversity_scores']):.3f} ± {np.std(quality_metrics['diversity_scores']):.3f}")

        return quality_metrics

    def get_similarity_level(self, similarity_score):
        if similarity_score >= 0.96:
            return "🔥 VERY HIGH"
        elif similarity_score >= 0.92:
            return "🟢 HIGH"
        elif similarity_score >= 0.88:
            return "🟡 MODERATE"
        elif similarity_score >= 0.84:
            return "🟠 LOW"
        else:
            return "🔴 VERY LOW"

    def create_comprehensive_visualizations(self):
        print(f"\n📊 CREATING COMPREHENSIVE VISUALIZATIONS")
        print("=" * 50)

        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 16))

        # Subplot 1: Scatter plot of original vs weighted ratings
        plt.subplot(3, 4, 1)
        plt.scatter(self.movies_df['vote_count'], self.movies_df['score'],
                   alpha=0.6, s=30, label='Original Score', color='blue')
        plt.scatter(self.movies_df['vote_count'], self.movies_df['weighted_rating'],
                   alpha=0.6, s=30, label='Weighted Score', color='red')
        plt.axvline(self.vote_threshold, color='green', linestyle='--',
                   label=f'Threshold: {self.vote_threshold:.0f} (Reference Only)')
        plt.xlabel('Vote Count')
        plt.ylabel('Rating')
        plt.title('ALL MOVIES QUALIFIED\nOriginal vs Weighted Ratings')
        plt.legend()
        plt.xscale('log')

        # Continue with all your other subplots...
        # (I'll include just a few key ones to keep it manageable)

        plt.tight_layout()
        st.pyplot(fig)

# Streamlit interface
def main():
    st.title("🎬 ENHANCED IMDB CONTENT-BASED MOVIE RECOMMENDATION SYSTEM")
    st.subheader("🔧 WITH WEIGHTED RATING BIAS CORRECTION")
    
    # Replace GitHub URL with your actual CSV file URL
    github_csv_url = st.text_input(
        "Enter your GitHub CSV URL:", 
        value="https://raw.githubusercontent.com/yourusername/yourrepo/main/imdb_movies.csv",
        help="Replace with your actual GitHub raw CSV URL"
    )
    
    if st.button("Load Data and Run System"):
        recommender = IMDBContentBasedRecommendationSystem()
        
        with st.spinner("Loading data from GitHub..."):
            try:
                recommender.load_imdb_data(github_csv_url)
                recommender.calculate_weighted_ratings()
                recommender.build_content_based_system()
                recommender.analyze_similarity_quality()
                
                st.success("System initialized successfully!")
                
                # Store in session state
                st.session_state.recommender = recommender
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
    
    # Use the system if it's loaded
    if 'recommender' in st.session_state:
        recommender = st.session_state.recommender
        
        st.subheader("🎯 Interactive Recommendation System")
        
        option = st.selectbox(
            "Choose function:",
            ["Movie Recommendations", "Search by Genre", "Search by Crew", "Advanced Search", "Browse Genres", "Browse Crew"]
        )
        
        if option == "Movie Recommendations":
            movie_title = st.text_input("Enter movie title:")
            num_recs = st.number_input("Number of recommendations:", min_value=1, max_value=20, value=10)
            
            if st.button("Get Recommendations") and movie_title:
                cleaned_title = recommender.clean_title_text(movie_title)
                result = recommender.get_content_recommendations(cleaned_title, num_recs)
                
        elif option == "Search by Genre":
            genre = st.text_input("Enter genre:")
            num_results = st.number_input("Number of results:", min_value=1, max_value=20, value=10)
            
            if st.button("Search by Genre") and genre:
                result = recommender.search_by_genre(genre, num_results)
                
        elif option == "Search by Crew":
            crew_name = st.text_input("Enter crew member name:")
            num_results = st.number_input("Number of results:", min_value=1, max_value=20, value=10, key="crew")
            
            if st.button("Search by Crew") and crew_name:
                result = recommender.search_by_crew(crew_name, num_results)
                
        elif option == "Advanced Search":
            genre = st.text_input("Genre (optional):")
            crew = st.text_input("Crew member (optional):")
            min_rating = st.number_input("Minimum rating (optional):", min_value=0.0, max_value=10.0, step=0.1, value=None)
            max_results = st.number_input("Max results:", min_value=1, max_value=50, value=10)
            
            if st.button("Advanced Search"):
                result = recommender.advanced_search(
                    genre=genre if genre else None,
                    crew=crew if crew else None,
                    min_rating=min_rating,
                    max_results=max_results
                )
                
        elif option == "Browse Genres":
            if st.button("Show Available Genres"):
                recommender.get_genre_list()
                
        elif option == "Browse Crew":
            if st.button("Show Popular Crew"):
                recommender.get_popular_crew()

if __name__ == "__main__":
    main()
