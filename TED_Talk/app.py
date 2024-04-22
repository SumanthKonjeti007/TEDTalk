from flask import Flask, request, render_template
import pandas as pd
from TEDTalk import load_and_prepare_data, preprocess_text, TEDTalkRecommender
import nltk

nltk.download('stopwords')
app = Flask(__name__)

# Load and prepare the TED Talks data
df = load_and_prepare_data('tedx_dataset.csv')
df = preprocess_text(df)
recommender = TEDTalkRecommender(df)

@app.route('/', methods=['GET', 'POST'])
def search():
    results = []
    if request.method == 'POST':
        query = request.form.get('search_query')
        if query:
            talk_content = [query]
            recommendations = recommender.recommend_talks(talk_content)
            for _, row in recommendations.iterrows():
                video_id = extract_youtube_id(row['url'])
                thumbnail_url = f"https://img.youtube.com/vi/{video_id}/0.jpg"
                results.append({
                    'title': row['details'],  # Make sure 'details' contains the correct title
                    'main_speaker': row['main_speaker'],  # Confirm 'main_speaker' is the correct column
                    'url': row['url'],
                    'video_id': video_id,
                    'thumbnail_url': thumbnail_url
                })
    return render_template('index.html', talks=results)

def extract_youtube_id(url):
    if "youtube.com" in url:
        return url.split('v=')[1].split('&')[0]
    return None

if __name__ == '__main__':
    app.run(debug=True)
