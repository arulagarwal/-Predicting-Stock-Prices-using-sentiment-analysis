import numpy as np
import pandas as pd
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
import matplotlib.pyplot as mlpt  
import csv
import random
from SentimentAnalyser import get_articles
import unicodedata
import yfinance as yf
from treeinterpreter import treeinterpreter as ti
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report,confusion_matrix

def get_prediction(symbol):
    sia = SentimentIntensityAnalyzer()
    # stock market lexicon
    stock_lex = pd.read_csv('lexicon_data/stock_lex.csv')
    stock_lex['sentiment'] = (stock_lex['Aff_Score'] + stock_lex['Neg_Score'])/2
    stock_lex = dict(zip(stock_lex.Item, stock_lex.sentiment))
    stock_lex = {k:v for k,v in stock_lex.items() if len(k.split(' '))==1}
    stock_lex_scaled = {}
    for k, v in stock_lex.items():
        if v > 0:
            stock_lex_scaled[k] = v / max(stock_lex.values()) * 4
        else:
            stock_lex_scaled[k] = v / min(stock_lex.values()) * -4
    # # # Loughran and McDonald
    positive = []
    with open('lexicon_data/lm_positive.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            positive.append(row[0].strip())
    
    negative = []
    with open('lexicon_data/lm_negative.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            entry = row[0].strip().split(" ")
            if len(entry) > 1:
                negative.extend(entry)
            else:
                negative.append(entry[0])

    final_lex = {}
    final_lex.update({word:2.0 for word in positive})
    final_lex.update({word:-2.0 for word in negative})
    final_lex.update(stock_lex_scaled)
    final_lex.update(sia.lexicon)
    sia.lexicon = final_lex
    keywrds = {"AAPL":"Apple","MSFT":"Microsoft","TSLA":"Tesla"}
    name = keywrds[symbol]
    data = get_articles(name)
    df = pd.DataFrame(columns = ['Date', 'Title'])
    indx=0
    get_news=""
    for i in range(0,len(data)-1):
        get_date=data.date.iloc[i]
        next_date=data.date.iloc[i+1]
        if((str(get_date)!=str(next_date))):
            get_news=get_news+data.title.iloc[i]+" "
        if(str(get_date)!=str(next_date)):
            df.loc[indx]= [get_date, get_news]
            indx=indx+1
            get_news=" "
    a_df = yf.download(symbol, 
                      start=df.Date.iloc[len(df)-1], 
                      end=df.Date.iloc[0], 
                      progress=False)
    aapl_df = a_df.sort_index(ascending=True)
    df['adj_close_price']=""
    indx=0
    for i in range (0,len(df)):
        for j in range (0,len(aapl_df)):
            get_news_date=df.Date.iloc[i]
            get_stock_date=aapl_df.index[j].strftime('%Y-%m-%d')
            if((get_stock_date==get_news_date)):
                df.adj_close_price.iloc[i]=int(aapl_df['Adj Close'][j])
    mean=0
    summ=0
    count=0
    for i in range(0,len(df)):
        if(df.adj_close_price.iloc[i]!=""):
            summ=summ+int(df.adj_close_price.iloc[i])
            count=count+1
    mean=summ/count
    for i in range(0,len(df)):
        if(df.adj_close_price.iloc[i]==""):
            df.adj_close_price.iloc[i]=int(mean)
    df['adj_close_price'] = df['adj_close_price'].apply(np.int64)
    df["Comp"] = ''
    df["Negative"] = ''
    df["Neutral"] = ''
    df["Positive"] = ''
    for indexx, row in df.T.iteritems():
        try:
            sentence_i = unicodedata.normalize('NFKD', df.loc[indexx, 'Title'])
            sentence_sentiment = sia.polarity_scores(sentence_i)
            df.Comp.iloc[indexx]=sentence_sentiment['compound']
            df.Negative.iloc[indexx]=sentence_sentiment['neg']
            df.Neutral.iloc[indexx]=sentence_sentiment['neu']
            df.Positive.iloc[indexx]=sentence_sentiment['pos']
        except TypeError:
            print (stocks_dataf.loc[indexx, 'title'])
            print (indexx)
    df_=df[['Date','adj_close_price','Comp','Negative','Neutral','Positive']].copy()
    df_ = df_[::-1].reset_index()
    train = df_.iloc[0:len(df)-1]
    test = df_.iloc[len(df)-1]
    sentiment_score_list = []
    for date, row in train.T.iteritems():
        sentiment_score = np.asarray([df_.loc[date, 'Comp']])
        sentiment_score_list.append(sentiment_score)
    numpy_df_train = np.asarray(sentiment_score_list)
    sentiment_score_list = []
    sentiment_score = np.asarray(test.Comp)
    sentiment_score_list.append(sentiment_score)
    numpy_df_test = np.asarray(sentiment_score_list)
    y_train = pd.DataFrame(train['adj_close_price'])
    y_test = pd.DataFrame([test.adj_close_price], columns=['adj_close_price'])
    rf = RandomForestRegressor()
    rf.fit(numpy_df_train, y_train)
    numpy_df_test=numpy_df_test.reshape(1, -1)
    prediction, bias, contributions = ti.predict(rf, numpy_df_test)
    return float(prediction[0])