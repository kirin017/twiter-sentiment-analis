import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from keras.models import load_model
from ntscraper import Nitter
import re
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense, Input, Dropout
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
# Hàm lấy tweet của user
def create_tweets_dataset(username,no_of_tweets):
    tweets = scraper.get_tweets(username,mode="user",number=no_of_tweets)
    data = {
        'link':[],
        'text':[],
        'user':[],
        'date':[],
        'likes':[],
        'quotes':[],
        'retweets':[],
        'comments':[]
    }

    for tweet in tweets['tweets']:
        data['link'].append(tweet['link'])
        data['text'].append(tweet['text'])
        data['user'].append(tweet['user']['name'])
        data['date'].append(tweet['date'])
        data['likes'].append(tweet['stats']['likes'])
        data['quotes'].append(tweet['stats']['quotes'])
        data['retweets'].append(tweet['stats']['retweets'])
        data['comments'].append(tweet['stats']['comments'])
    df = pd.DataFrame(data)
    df.to_csv(username+"_tweets_data.csv")
    return df

def clean_tweet(tweet):
    return ' '.join(re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|([RT])', ' ', str(tweet).lower()).split())
# model = load_model('best_model.hdf5')
# def analyze_sentiment(tweet):
#     prediction = model.predict(tweet)

#     if prediction > 0.5:
#         return 'Positive'
#     else:
#         return 'Negative'

from textblob import TextBlob
def analyze_sentiment(tweet):
    analysis = TextBlob(tweet)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'
scraper = Nitter(log_level = 1,skip_instance_check= False)

# Tạo giao diện Streamlit
st.title('Twitter Sentiment Analysis')
st.write('Nhập tên người dùng Twitter để phân tích cảm xúc của các bài tweet của họ.')

username = st.text_input('Tên người dùng Twitter')
if username:
    with st.spinner('Đang lấy tweets...'):
        tweets = create_tweets_dataset(username, 20)
        st.success(f'Lấy được {len(tweets)} tweets từ @{username}')
            
        st.write('Phân tích cảm xúc các bài tweet:')
        for tweet in tweets:
            tweet_text = tweet.full_text
            sentiment = analyze_sentiment(tweet_text)
            st.write(f'**Tweet:** {tweet_text}')
            st.write(f'**Sentiment:** {sentiment}')
            st.write('---')

