import requests
from bs4 import BeautifulSoup
import pprint
import pandas as pd
import numpy as np
from datetime import date, timedelta

url = 'https://newsapi.org/v2/everything?'
api_key = #Enter API key
def get_articles(name):
  d = date.today() - timedelta(days=30)
  d = d.isoformat()
  param = {'qInTitle' : name, 
          'sources' : 'australian-financial-review,bloomberg,financial-post,the-wall-street-journal',
           'pageSize' : 100,
          'apiKey' : api_key,
          'language' : 'en',
           'sortBy' : 'publishedAt',
          'from' : d
          }
  res = requests.get(url,params = param)
  response = res.json()
  file = response['articles']
  data = []
  df = pd.DataFrame(file)
  df = pd.concat([df.drop(['source'], axis=1), df['source'].apply(pd.Series)], axis=1)
  df = df.drop(['description','urlToImage','content','author','id'], axis = 1)
  column_names = ['publishedAt','name','title','url']
  df = df.reindex(columns=column_names)
  return df
