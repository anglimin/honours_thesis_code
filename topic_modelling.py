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
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
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

def clean_data():
    pass

# Function for the LDA-MALLET topic modelling
def lda_modelling(df):
    """
    Input parameter: Tokenised and pre-processed dataframe
    Outputs: 
        1) Dataframe with tagged topic number for each textual document
        2) Top co-occuring keywords for each topic 
    """
    # Function to derive the topic number with the highest coherence measure
    pass

    # Function to train the model and tag the derived topics



if __name__ == "__main__":
    # for reddit and twitter data
    pass