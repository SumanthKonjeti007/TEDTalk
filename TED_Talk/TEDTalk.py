import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import string
import warnings
from scipy.stats import pearsonr
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Assuming nltk data is already available, or handle its download within the script
warnings.filterwarnings('ignore')

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    df.dropna(subset=['title', 'details', 'main_speaker', 'posted'], inplace=True)
    
    # Split 'posted' date and extract year and month
    splitted = df['posted'].str.split(' ', expand=True)
    df['year'] = splitted[2].astype(int)
    df['month'] = splitted[1]
    
    df['details'] = df['title'] + ' ' + df['details']
    df = df[['main_speaker', 'details', 'url']]
    
    return df

def preprocess_text(df):
    stop_words = set(stopwords.words('english'))
    punctuations_list = string.punctuation

    def remove_stopwords(text):
        return " ".join(word for word in str(text).split() if word.lower() not in stop_words)
    
    def cleaning_punctuations(text):
        signal = str.maketrans('', '', punctuations_list)
        return text.translate(signal)
    
    df['details'] = df['details'].apply(remove_stopwords).apply(cleaning_punctuations)
    return df

class TEDTalkRecommender:
    def __init__(self, data):
        self.data = data
        self.vectorizer = TfidfVectorizer(analyzer='word')
        self.vectorizer.fit(data['details'])
        
    def get_similarities(self, talk_content):
        talk_vector = self.vectorizer.transform(talk_content).toarray()
        cos_sim, pea_sim = [], []

        for _, row in self.data.iterrows():
            details_vector = self.vectorizer.transform([row['details']]).toarray()
            cos_sim.append(cosine_similarity(talk_vector, details_vector)[0][0])
            pea_sim.append(pearsonr(talk_vector.squeeze(), details_vector.squeeze())[0])
        
        return cos_sim, pea_sim
    
    def recommend_talks(self, talk_content):
        self.data['cos_sim'], self.data['pea_sim'] = self.get_similarities(talk_content)
        recommendations = self.data.sort_values(by=['cos_sim', 'pea_sim'], ascending=[False, False])
        return recommendations.head()

def main():
    df = load_and_prepare_data('tedx_dataset.csv')
    df = preprocess_text(df)
    
    recommender = TEDTalkRecommender(df)
    talk_content = ['Time Management and working hard to become successful in life']
    print(recommender.recommend_talks(talk_content))

if __name__ == '__main__':
    main()
