import pandas as pd 
from TwitterAPI import TwitterAPI, TwitterPager
from datetime import datetime 
from tqdm import tqdm
import time
import re


# Environment Variables: Change them on your end
consumer_key= 'XXXXXXXXXXXXXXXXXXX'
consumer_secret = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
access_token_key= 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
access_token_secret= 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
label = 'extracting tweets' # this label is the name that you have given to the working environment in Twitter Developers

# Extract all tweets related to Singapore's public transport
# Helper functions to change timing from UTC +0000 to UTC+0800
def utc_former(tweet_datetime_raw):
    month_mapper = {
        'Jan':'01',
        'Feb':'02',
        'Mar':'03',
        'Apr':'04',
        'May':'05',
        'Jun':'06',
        'Jul':'07',
        'Aug':'08',
        'Sep':'09',
        'Oct':'10',
        'Nov':'11',
        'Dec':'12'
    }
    lst = tweet_datetime_raw.split()
    newlst = []
    year = lst[5]
    month = month_mapper[lst[1]]
    day = lst[2]
    date = year + '-' + month + '-' + day
    time = lst[3]
    newlst.append(date)
    newlst.append(time)
    return ' '.join(newlst)

def extract_tweets(consumer_key,consumer_secret,access_token_key,access_token_secret,label):
    api = TwitterAPI(consumer_key,consumer_secret,access_token_key,access_token_secret)
    PRODUCT= 'fullarchive'
    LABEL = label
    SEARCH_TERM = '(bus OR smrt OR transport OR mrt OR lrt OR public transport OR sbs OR sbs transit OR transitlink OR lta OR towertransit OR land transport authority) lang:en place:Singapore' # Adding place automatially removes retweets

    pager = TwitterPager(api, 'tweets/search/%s/:%s' % (PRODUCT, LABEL),
                 {'query': SEARCH_TERM,'maxResults':500,'toDate':2020102300001,'fromDate':201501010001}) 
    responseValues = []
    for item in pager.get_iterator():
        if 'text' in item:
            responseValues.append({'created_at': item['created_at'],
                            'tweet_coordinates': item['place'],
                            'text': item['text']})
        elif 'message' in item and item['code'] == 88:
            print ('SUSPEND, RATE LIMIT EXCEEDED: %s\n' % item['message'])
            break
    tweets = pd.DataFrame(responseValues)
    # Changing the timing 
    tweets['created_at'] = tweets['created_at'].apply(utc_former)
    return tweets

# Function to clean the data 
"""
Things to remove:
1) @username 
2) URLs and weird symbols 
3) Indicative Tweets 'I'm at ...'
""" 
def data_processing(tweets):
    tweets = tweets[~tweets['text'].str.contains("I'm at")]
    tweets.reset_index(drop=True,inplace=True)
    pattern_string = "[^a-zA-Z0-9]+"
    url_string = "\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*"
    username_string = "@[A-Za-z0-9]+"
    for i in tqdm(range(len(tweets))):
        tweets.loc[i,'Edited Text'] = re.sub(r"{}".format(username_string)," ",tweets.loc[i,'text'])
        tweets.loc[i,'Edited Text'] = re.sub(r"{}".format(url_string)," ",tweets.loc[i,'Edited Text'])
        tweets.loc[i,'Edited Text'] = re.sub(r"{}".format(pattern_string)," ",tweets.loc[i,'Edited Text'])
    tweets.sort_values(by='created_at',inplace=True)
    tweets.to_csv('twitter_data.csv',inplace=True)
    return tweets

if __name__ == '__main__':
    tweets = extract_tweets(consumer_key,consumer_secret,access_token_key,access_token_secret,label)
    twitter_data = data_processing(tweets)