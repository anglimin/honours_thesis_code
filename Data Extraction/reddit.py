import praw 
import pandas as pd 
import datetime as dt
from tqdm import tqdm
from praw.models import MoreComments
import re

# Environment Variables: Change them on your end
client_id = 'XXXXXXXXXXX'
client_secret = 'XXXXXXXXXXXXXXXXXXXX'
user_agent = 'XXXXXXXXXXXX'
username = 'XXXXXXXXXXX'
password = 'XXXXXXXXXXX'

# Function to extract all relevant comment/reply from reddit
def extract_reddit_comments(client_id, client_secret, user_agent, username, password):
    # Initialise PRAW object and subreddit of r/singapore
    reddit = praw.Reddit(user_agent=user_agent,
                     client_id=client_id, client_secret=client_secret,
                     username=username, password=password)
    singapore = reddit.subreddit('singapore')
    def get_date(created):
        return dt.datetime.fromtimestamp(created)

    def getting_submissions(subreddit,*query):
        topics_dict = { "title":[], 
                    "score":[], 
                    "id":[], "url":[], 
                    "comms_num": [], 
                    "created": [], 
                    "body":[]}
        for search in query:
            for submission in subreddit.search(search):
                topics_dict["title"].append(submission.title) ### title of submission
                topics_dict["score"].append(submission.score)
                topics_dict["id"].append(submission.id) ## reddit id of submission
                topics_dict["url"].append(submission.url)
                topics_dict["comms_num"].append(submission.num_comments)
                topics_dict["created"].append(submission.created)
                topics_dict["body"].append(submission.selftext)
        topics_data = pd.DataFrame(topics_dict)
        _timestamp = topics_data["created"].apply(get_date)
        topics_data = topics_data.assign(timestamp = _timestamp)
        return topics_data
    
    transport_threads = getting_submissions(singapore,'bus','smrt','buses','LRT','MRT','LTA','public transport','land transport authority''sbs','sbs transit','tower transit','transitlink')

    # Drop duplicate threads 
    transport_threads.drop_duplicates(subset=['id'],keep='first',inplace=True)
    transport_threads.reset_index(inplace=True,drop=True)
    
    # Extracting all comments and replies
    new_comments_dict= {
        'submission_id':[],
        'comment':[],
        'comment_timestamp':[]
    }

    submission_ids = list(transport_threads.id)
    for submission_id in tqdm(submission_ids):
        submission = reddit.submission(id=submission_id)
        submission.comments.replace_more(limit=None,threshold=0)
        for comment in submission.comments.list():
            new_comments_dict['comment'].append(comment.body)
            new_comments_dict['comment_timestamp'].append(get_date(comment.created))
            new_comments_dict['submission_id'].append(submission_id)
            reply_lst=[]
            reply_lst.extend(comment.replies)##containing list of reply list
            while reply_lst:
                reply = reply_lst.pop(0)
                new_comments_dict['comment'].append(reply.body)
                new_comments_dict['comment_timestamp'].append(get_date(reply.created))
                new_comments_dict['submission_id'].append(submission_id)
            
    transport_comments=pd.DataFrame(new_comments_dict)
    reddit_data = transport_threads.merge(transport_comments, how='inner',left_on='id',right_on='submission_id')
    return reddit_data


# Function to clean the data from reddit and filter from 2015
def data_processing(reddit_data):

    # remove every duplicated and deleted and removed comments
    reddit_drop = reddit_data.drop_duplicates(subset=['comment'],keep='first')
    reddit_drop = reddit_drop[reddit_drop.comment != '[deleted]']
    reddit_drop = reddit_drop[reddit_drop.comment != '[removed]']
    reddit_drop.reset_index(inplace=True,drop=True)

    """
    Things to remove from the reddit data:
    1) Websites
    2) All non-alphabet and number symbols and line spaces
    3) Remove all empty data after step 1 and 2
    """
    compiler1 = re.compile('http[s]?://\S+')
    compiler2 = re.compile('[^.,a-zA-Z0-9 \n\.]')
    for i in tqdm(range(len(reddit_drop))):
        clean_text1 = re.sub(compiler1,'',reddit_drop.loc[i,'comment'])
        clean_text2= re.sub(compiler2,' ',clean_text1)
        cleaner_comment = clean_text2.replace('\n',' ')
        reddit_drop.loc[i,'comment'] = cleaner_comment
    reddit_drop.to_csv('reddit_data.csv',index=False)
    return reddit_drop

if __name__ == '__main__':
    reddit_data = extract_reddit_comments(client_id,client_secret,user_agent,username,password)
    reddit_data = data_processing(reddit_data)
