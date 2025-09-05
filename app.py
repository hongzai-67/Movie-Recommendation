# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from math import sqrt

# ---------------------------
# Recommendation System Class
# ---------------------------
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

    def load_imdb_data(self, file_obj):
        self.movies_df = pd.read_csv(file_obj, low_memory=False)

        if 'overview' not in self.movies_df:
            self.movies_df['overview'] = ''
        if 'genre' not in self.movies_df:
            self.movies_df['genre'] = 'Unknown'
        if 'crew' not in self.movies_df:
            self.movies_df['crew'] = 'Unknown'
        if 'orig_title' not in self.movies_df and 'names' in self.movies_df:
            self.movies_df['orig_title'] = self.movies_df['names']

        self.preprocess_data()
        self.calculate_weighted_ratings()

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

        for col in ['score', 'revenue']:
            if col not in self.movies_df:
                self.movies_df[col] = np.nan

    def calculate_weighted_ratings(self):
        self.movies_df['vote_count'] = (
            (self.movies_df['revenue'].fillna(0) / 1_000_000) *
            (self.movies_df['score'].fillna(5) / 10) *
            np.random.uniform(3, 6, len(self.movies_df))
        ).astype(int)

        self.movies_df['vote_count'] = self.movies_df['vote_count'].clip(lower=1)

        self.average_rating = float(self.movies_df['score'].mean()) if 'score' in self.movies_df else 5.0
        self.vote_threshold = int(self.movies_df['vote_count'].quantile(0.90))

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
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(working_df['enhanced_content'].fillna(''))
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
        self.indices = pd.Series(working_df.index, index=working_df['orig_title']).drop_duplicates()
        self.qualified_movies = working_df

    # ---------------------------
    # Search / Recommendation Methods
    # ---------------------------
    def get_content_recommendations(self, title, n=10):
        if self.indices is None: 
            return None,None,None
        if title not in self.indices:
            possible = self.qualified_movies[self.qualified_movies['orig_title'].str.contains(title, case=False, na=False)]
            if possible.empty: return None,None,None
            return "choose", possible.head(8), None
        idx = self.indices[title]
        if hasattr(idx,'__iter__') and not isinstance(idx,str):
            idx = idx.iloc[0]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores,key=lambda x:x[1],reverse=True)[1:n+1]
        ids = [i for i,_ in sim_scores]; sims=[s for _,s in sim_scores]
        recs = self.qualified_movies.iloc[ids].copy(); recs['similarity']=sims
        return "ok", self.qualified_movies.loc[idx], recs

    def search_by_genre(self, genre, n=10):
        matches = self.qualified_movies[self.qualified_movies['genre'].str.contains(genre, case=False, na=False)]
        return matches.nlargest(n,'weighted_rating')

    def search_by_crew(self, crew, n=10):
        matches = self.qualified_movies[self.qualified_movies['crew'].str.contains(crew, case=False, na=False)]
        return matches.nlargest(n,'weighted_rating')

    def get_top_movies_by_rating(self, n=20):
        return self.qualified_movies.nlargest(n,'weighted_rating')

    def search_by_year(self, year, n=10):
        self.qualified_movies['year']=pd.to_datetime(self.qualified_movies['date_x'],errors='coerce').dt.year
        matches=self.qualified_movies[self.qualified_movies['year']==year]
        return matches.nlargest(n,'weighted_rating')

    def search_by_country(self,country,n=10):
        matches=self.qualified_movies[self.qualified_movies['country'].astype(str).str.contains(country,case=False,na=False)]
        return matches.nlargest(n,'weighted_rating')

    def search_by_language(self,lang,n=10):
        matches=self.qualified_movies[self.qualified_movies['orig_lang'].str.contains(lang,case=False,na=False)]
        return matches.nlargest(n,'weighted_rating')

    # --- hybrid ---
    def get_hybrid_recommendations(self,title,n=10,alpha=0.7):
        if 'popularity_norm' not in self.qualified_movies:
            r=self.qualified_movies['weighted_rating'].fillna(self.average_rating)
            self.qualified_movies['popularity_norm']=(r-r.min())/(r.max()-r.min())
        if title not in self.indices:
            possible=self.qualified_movies[self.qualified_movies['orig_title'].str.contains(title,case=False,na=False)]
            if possible.empty: return None,None,None
            return "choose",possible.head(8),None
        idx=self.indices[title]
        if hasattr(idx,'__iter__') and not isinstance(idx,str): idx=idx.iloc[0]
        sims=list(enumerate(self.cosine_sim[idx])); sims=[(i,s) for i,s in sims if i!=idx]
        ids=[i for i,_ in sims]; simvals=[s for _,s in sims]
        recs=self.qualified_movies.iloc[ids].copy(); recs['similarity']=simvals
        recs['hybrid_score']=alpha*recs['similarity']+(1-alpha)*recs['popularity_norm']
        return "ok", self.qualified_movies.loc[idx], recs.sort_values('hybrid_score',ascending=False).head(n)

    # --- evaluation ---
    def run_all_evaluations(self,k=10,sample_size=100,progress_callback=None):
        results={}
        titles=self.qualified_movies['orig_title'].tolist()[:sample_size]
        precisions=[];recalls=[]
        for idx,title in enumerate(titles):
            status,_,recs=self.get_content_recommendations(title,n=k)
            if recs is None or recs.empty: continue
            src=self.qualified_movies[self.qualified_movies['orig_title']==title]
            src_gen=set(str(src.iloc[0]['genre']).split('|'))
            relevant=self.qualified_movies[self.qualified_movies['genre'].apply(lambda g:len(src_gen & set(str(g).split('|')))>0)]
            rel_idx=set(relevant.index)-set(src.index)
            rec_idx=set(recs.index)
            tp=len(rec_idx & rel_idx)
            precisions.append(tp/max(1,len(rec_idx)))
            recalls.append(tp/max(1,len(rel_idx)))
            if progress_callback: progress_callback(int((idx+1)/len(titles)*50),"Evaluating PRF...")
        prec=np.mean(precisions) if precisions else 0; rec=np.mean(recalls) if recalls else 0
        f1=2*prec*rec/(prec+rec) if prec+rec>0 else 0
        results.update({'precision':prec,'recall':rec,'f1':f1})
        # RMSE
        y_true=[];y_pred=[]
        for idx in range(min(sample_size,len(self.qualified_movies))):
            sims=list(enumerate(self.cosine_sim[idx])); sims=[(i,s) for i,s in sims if i!=idx]
            sims=sorted(sims,key=lambda x:x[1],reverse=True)[:k]
            if not sims: continue
            ids=[i for i,_ in sims]; weights=np.array([s for _,s in sims])
            ratings=self.qualified_movies.iloc[ids]['weighted_rating'].fillna(self.average_rating).values
            pred=(weights@ratings)/weights.sum() if weights.sum()>0 else ratings.mean()
            y_true.append(self.qualified_movies.iloc[idx]['weighted_rating']); y_pred.append(pred)
            if progress_callback: progress_callback(50+int((idx+1)/len(self.qualified_movies)*50),"Evaluating RMSE...")
        if y_true:
            errors=np.array(y_true)-np.array(y_pred)
            mse=np.mean(errors**2); rmse=sqrt(mse)
            results.update({'mse':mse,'rmse':rmse})
        else:
            results.update({'mse':None,'rmse':None})
        return results

# ---------------------------
# Streamlit App
# ---------------------------
def main():
    st.set_page_config(page_title="IMDB Recommender",layout="wide")
    st.title("🎮 Enhanced IMDB Movie Recommendation System")

    uploaded=st.sidebar.file_uploader("Upload IMDB CSV",type="csv")
    if not uploaded:
        st.warning("Upload dataset first.")
        return

    recommender=IMDBContentBasedRecommendationSystem()
    recommender.load_imdb_data(uploaded)
    recommender.build_content_based_system()

    opt=st.radio("Options:",[
        "1️⃣ Title Recommendations",
        "2️⃣ Genre Search",
        "3️⃣ Crew Search",
        "4️⃣ Year Search",
        "5️⃣ Country Search",
        "6️⃣ Language Search",
        "7️⃣ Top Rated",
        "8️⃣ Hybrid Recommendations",
        "9️⃣ Run Evaluation Suite"
    ])

    # Title recs
    if opt.startswith("1️⃣"):
        title=st.text_input("Enter movie title:")
        n=st.slider("Number of recs",1,20,10)
        if st.button("Get Recs"):
            st.session_state.pop('confirmed_title',None)
            status,movie,recs=recommender.get_content_recommendations(recommender.clean_title_text(title),n)
            if status=="choose":
                st.session_state['choices_title']=movie['names'].tolist() if 'names' in movie else movie['original_title'].tolist()
            elif status=="ok":
                st.dataframe(recs[['names','genre','score','similarity']])
        if 'choices_title' in st.session_state:
            choice=st.selectbox("Multiple matches found:",st.session_state['choices_title'])
            if st.button("Confirm Selection"):
                st.session_state['confirmed_title']=choice
                st.session_state.pop('choices_title')
        if 'confirmed_title' in st.session_state:
            status2,movie2,recs2=recommender.get_content_recommendations(recommender.clean_title_text(st.session_state['confirmed_title']),n)
            if status2=="ok":
                st.dataframe(recs2[['names','genre','score','similarity']])

    # Hybrid recs
    elif opt.startswith("8️⃣"):
        title=st.text_input("Enter movie title for hybrid:")
        alpha=st.slider("Alpha weight",0.0,1.0,0.7,0.05)
        n=st.slider("Number hybrid recs",1,20,10)
        if st.button("Get Hybrid Recs"):
            st.session_state.pop('confirmed_hybrid',None)
            status,movie,recs=recommender.get_hybrid_recommendations(recommender.clean_title_text(title),n,alpha)
            if status=="choose":
                st.session_state['choices_hybrid']=movie['names'].tolist() if 'names' in movie else movie['original_title'].tolist()
            elif status=="ok":
                st.dataframe(recs[['names','genre','score','hybrid_score']])
        if 'choices_hybrid' in st.session_state:
            choice=st.selectbox("Multiple matches:",st.session_state['choices_hybrid'])
            if st.button("Confirm Hybrid Selection"):
                st.session_state['confirmed_hybrid']=choice
                st.session_state.pop('choices_hybrid')
        if 'confirmed_hybrid' in st.session_state:
            status2,movie2,recs2=recommender.get_hybrid_recommendations(recommender.clean_title_text(st.session_state['confirmed_hybrid']),n,alpha)
            if status2=="ok":
                st.dataframe(recs2[['names','genre','score','hybrid_score']])

    # Other searches
    elif opt.startswith("2️⃣"):
        g=st.text_input("Genre:")
        if st.button("Search Genre"): st.dataframe(recommender.search_by_genre(g,10))

    elif opt.startswith("3️⃣"):
        c=st.text_input("Crew:")
        if st.button("Search Crew"): st.dataframe(recommender.search_by_crew(c,10))

    elif opt.startswith("4️⃣"):
        y=st.number_input("Year:",1900,2100,2020)
        if st.button("Search Year"): st.dataframe(recommender.search_by_year(y,10))

    elif opt.startswith("5️⃣"):
        co=st.text_input("Country:")
        if st.button("Search Country"): st.dataframe(recommender.search_by_country(co,10))

    elif opt.startswith("6️⃣"):
        l=st.text_input("Language:")
        if st.button("Search Lang"): st.dataframe(recommender.search_by_language(l,10))

    elif opt.startswith("7️⃣"):
        if st.button("Top Rated"): st.dataframe(recommender.get_top_movies_by_rating(20))

    elif opt.startswith("9️⃣"):
        k=st.slider("k",1,30,10); s=st.slider("Sample size",10,500,100,10)
        if st.button("Run Evaluation"):
            prog=st.progress(0); txt=st.empty()
            def cb(p,m): prog.progress(p); txt.text(m)
            results=recommender.run_all_evaluations(k,s,progress_callback=cb)
            st.json(results)

if __name__=="__main__":
    main()
