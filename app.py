import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
warnings.filterwarnings('ignore')

# Define the recommendation system class
class IMDBContentBasedRecommendationSystem:
    def __init__(self):
        # Initialize instance variables to store data and models
        self.movies_df = None  # DataFrame for the IMDb dataset
        self.qualified_movies = None  # DataFrame for processed movies (all movies in this version)
        self.tfidf_vectorizer = None  # TF-IDF vectorizer for text features
        self.tfidf_matrix = None  # Matrix of TF-IDF features
        self.cosine_sim = None  # Cosine similarity matrix
        self.indices = None  # Mapping of movie titles to DataFrame indices
        self.average_rating = None  # Mean rating across all movies
        self.vote_threshold = None  # Vote count threshold (used for reference, not filtering)

    def clean_title_text(self, text):
        """
        Clean title text by removing special characters and converting to lowercase.

        Args:
            text (str): Raw title text

        Returns:
            str: Cleaned title text with words separated by spaces
        """
        if pd.isna(text):
            return ""

        # Convert to string and remove special characters, keep only alphanumeric and spaces
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text))

        # Replace multiple spaces with single space and convert to lowercase
        cleaned = re.sub(r'\s+', ' ', cleaned).lower().strip()

        return cleaned

    def load_imdb_data(self, file_path='imdb_movies.csv'):
        # Load CSV file into a pandas DataFrame
        self.movies_df = pd.read_csv(file_path, low_memory=False)  # low_memory=False handles large datasets
        # Call preprocessing method to clean and prepare data
        self.preprocess_data()

    def preprocess_data(self):
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

    def calculate_weighted_ratings(self):
        self.movies_df['vote_count'] = (
            (self.movies_df['revenue'].fillna(0) / 1_000_000) *
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

    def build_content_based_system(self):
        working_df = self.qualified_movies.copy()
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

    def get_content_recommendations(self, title, n=10):
        if title not in self.indices:
            return pd.DataFrame()

        idx = self.indices[title]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n+1]
        movie_indices = [i[0] for i in sim_scores]
        movies = self.qualified_movies.iloc[movie_indices].copy()
        movies['similarity'] = [i[1] for i in sim_scores]
        movies['composite_score'] = 0.7 * movies['similarity'] + 0.3 * (movies['weighted_rating'] / 10.0)
        return movies[['names', 'weighted_rating', 'similarity', 'composite_score', 'genre']]

def main():
    st.set_page_config(page_title="IMDB Recommendation System", layout="wide")
    st.title("🎬 IMDB Content-Based Movie Recommendation System")

    st.sidebar.header("⚙️ 系统设置")
    uploaded_file = st.sidebar.file_uploader("上传 IMDB 数据集 (CSV)", type="csv")

    if uploaded_file:
        recommender = IMDBContentBasedRecommendationSystem()
        recommender.load_imdb_data(uploaded_file)
        recommender.calculate_weighted_ratings()
        recommender.build_content_based_system()

        st.success("✅ 数据加载成功！可以开始推荐电影啦～")

        # 输入电影名
        movie_title = st.text_input("请输入电影名称（英文）:")
        n_recs = st.slider("推荐数量", 1, 20, 10)

        if st.button("获取推荐结果"):
            cleaned_title = recommender.clean_title_text(movie_title)
            recs = recommender.get_content_recommendations(cleaned_title, n=n_recs)

            if recs.empty:
                st.error(f"没有找到与 **{movie_title}** 匹配的电影，请检查拼写。")
            else:
                st.subheader(f"📌 为你推荐的电影（基于 {movie_title}）")
                st.dataframe(recs)

                # 可视化推荐分布
                st.subheader("📊 推荐电影评分分布")
                fig, ax = plt.subplots()
                sns.barplot(x="weighted_rating", y="names", data=recs, ax=ax, palette="viridis")
                ax.set_xlabel("Weighted Rating")
                ax.set_ylabel("Movie")
                st.pyplot(fig)

    else:
        st.warning("请在左侧上传 IMDb 数据集 CSV 文件。")


if __name__ == "__main__":
    main()
