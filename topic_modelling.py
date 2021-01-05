import pandas as pd
from tqdm import tqdm
import numpy as np
import nltk
from nltk.corpus import stopwords
import gensim
from gensim.models import LdaModel
from gensim import models, corpora, similarities
from gensim.models import CoherenceModel
import re
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import time
from nltk import FreqDist
from scipy.stats import entropy
import ast
import random
sns.set_style("darkgrid")
from pprint import pprint
import os

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# Enable logging for gensim to keep track of progress
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

# Environment Variables -> Change the mallet path on your end (C Drive)
MALLET_PATH = 'XXXXXXXX'
os.environ.update({'MALLET_HOME': MALLET_PATH})

# Helper functions for tokenizing the raw data
def initial_clean(text):
    """
    Function to clean text of websites, email addresess and any punctuation
    We also lower case the text
    """
    text = re.sub("((\S+)?(http(s)?)(\S+))|((\S+)?(www)(\S+))|((\S+)?(\@)(\S+)?)", " ", text)
    text = re.sub("[^a-zA-Z ]", "", text)
    text = text.lower() # lower case the text
    text = nltk.word_tokenize(text)
    return text

stop_words = stopwords.words('english')
def remove_stop_words(text):
    """
    Function that removes all stopwords from text
    """
    return [word for word in text if word not in stop_words]

wordnet_lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    """
    Function to lemmatize words, for now, only lemmatize verbs,noun, adjective, adverbs 
    """
    try:
        text = [wordnet_lemmatizer.lemmatize(word,pos='v') for word in text]
        text = [wordnet_lemmatizer.lemmatize(word,pos='n') for word in text]
        text = [wordnet_lemmatizer.lemmatize(word,pos='a') for word in text]
        text = [word for word in text if len(word) > 1] # make sure we have no 1 letter words
    except IndexError: # the word "oed" broke this, so needed try except
        pass
    return text


def apply_all(text):
    """
    This function applies all the functions above into one
    """
    return lemmatize_words(remove_stop_words(initial_clean(text)))

def keyword_removal(text):
    """
    Function to remove keywords from the tokenized text list
    """
    keyword_lst = ['bus','smrt','SMRT','buses','LRT','MRT','LTA','mrt','lta','lrt','public transport','land transport authority''sbs','SBS','sbs transit','tower transit','transitlink']
    return list(set(text) - set(keyword_lst))

def clean_data(data):
    if data == 'reddit_data.csv':
        df = pd.read_csv(data)
        df = df[df['comment'].map(type) == str]
        df['tokenized_text'] = df['comment'].apply(apply_all)
    else: # For reading twitter data 
        df = pd.read_csv(data)
        df = df[df['Edited Text'].map(type) == str]
        df['tokenized_text'] = df['Edited Text'].apply(apply_all)
    df.dropna(inplace=True,subset=['tokenized_text'])
    # Shuffle the data
    df = df.sample(frac=1.0)
    df.reset_index(drop=True,inplace=True)
    # Remove search ketwords for the tokenized_text column
    for i in tqdm(range(len(df))):
        df.loc[i,'tokenized_text'] = str(keyword_removal(df.loc[i,'tokenized_text']))
    df.loc[i,'tokenized_text'] = df.apply(lambda x: ast.literal_eval(x.tokenized_text),axis=1)
    # Drop textual documents that are less than 5 length
    df = df[df['tokenized_text'].map(len) >=5 ]
    df = df[df['tokenized_text'].map(type) == list ]
    df.reset_index(drop=True,inplace=True)
    return df

# Function for the LDA-MALLET topic modelling
def lda_modelling(df,output_name):
    """
    Input parameter: 
        1) Tokenised and pre-processed dataframe
        2) Output name for the tagged dataframe
    Outputs: 
        1) Dataframe with tagged topic number for each textual document
        2) Top co-occuring keywords for each topic 
    """
    # Helper Function to derive the topic number with the highest coherence measure
    def mallet_coherence_values(data, limit, start=2, step=1):
        """
        Compute c_v coherence for various number of topics for the mallet model 
        Default alpha is 50/n -> contributing to too much latent topics in one document.
        Testing out with lower alpha and optimized_intervals of 10, allows the dirichlet alpha and (b)eta to be optimized faster
        Parameters:
        ----------
        data: dataframe consisting of the lemmatized and tokenized text
        limit : Max num of topics
        Testing the auto alpha and beta values to let the 'model' learn the hyperparameters from the data
        Returns:
        -------
        model_list : List of LDA topic models
        coherence_values : Coherence values corresponding to the LDA model with respective number of topics
        """
        coherence_values = []
        model_list = []
        dictionary = corpora.Dictionary(data['tokenized_text'])
        corpus = [dictionary.doc2bow(doc) for doc in data['tokenized_text']]
        texts= list(data['tokenized_text'])
        mallet_path = 'C:/new_mallet/mallet-2.0.8/bin/mallet'
        for num_topics in range(start, limit, step):
            model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics,optimize_interval=10, alpha=1,id2word=dictionary,iterations=2000,random_seed=456)
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())

        return model_list, coherence_values
    # Determine the optimal number of topics to trained with 
    def highest_coherence_model(model_list,coherence_values):
        highest_index = coherence_values.index(max(coherence_values))
        print('Model trained with {} topics has the highest coherence score of {}'.format(highest_index+2,coherence_values[highest_index]))
        return model_list[highest_index]

    def get_topic_keywords(mallet_model):
        """
        Purpose:
        Print top 20 co-occuring keyword in each topic to decipher overarching topics
        """
        topic_keywords = mallet_model.show_topics(formatted=False,num_topics=-1,num_words=25)
        topic_dict ={}
        for i in range(len(topic_keywords)):
            keyword_lst = []
            for x in range(len(topic_keywords[i][1])):
                keyword_lst.append(topic_keywords[i][1][x][0])
            if topic_keywords[i][0] not in topic_dict:
                topic_dict[topic_keywords[i][0]] = keyword_lst
        topic_keyword_df = pd.DataFrame(topic_dict)
        topic_column_names = {}
        # Renaming column names
        for num in topic_dict.keys():
            topic_column_names[num] = 'Topic {}'.format(num+1)
        topic_keyword_df.rename(columns=topic_column_names,inplace=True)
        return topic_keyword_df

    # Tagging the dominant topic with the highest topic contribution to each textual document. 
    def format_topics_sentences(ldamodel, df):
        # Init output
        sent_topics_df = pd.DataFrame()
        dictionary = corpora.Dictionary(df['tokenized_text'])
        corpus = [dictionary.doc2bow(doc) for doc in df['tokenized_text']]
        texts = df['tokenized_text']
        # Get main topic in each document
        for i, row in enumerate(ldamodel[corpus]):
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        # Add original text to the end of the output
        contents = pd.Series(texts)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        tagged_df = df.merge(sent_topics_df,how='inner',on='tokenized_text')
        return tagged_df

    # Execution list
    model_list, coherence_values = mallet_coherence_values(data=df, start=2, limit=50, step=1)
    mallet_model = highest_coherence_model(model_list,coherence_values)
    topic_keywords = get_topic_keywords(mallet_model)
    topic_keywords.to_csv('topic_keywords.csv',index=False) # Save the top co-occuring words into a CSV for further semantic analysis
    tagged_df = format_topics_sentences(mallet_model,df)
    tagged_df.to_csv(output_name,index=False)
    return topic_keywords,tagged_df


if __name__ == "__main__":
    # Previously created data from reddit.py and twitter.py
    reddit_data = 'reddit_data.csv'
    twitter_data = 'twitter_data.csv'
    for data in [reddit_data, twitter_data]:
        cleaned_data = clean_data(data)
        if data == reddit_data:
            lda_modelling(cleaned_data,'reddit_lda.csv')
        else:
            lda_modelling(cleaned_data,'twitter_lda.csv')