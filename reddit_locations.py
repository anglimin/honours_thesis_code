import pandas as pd 
import spacy 
from tqdm import tqdm
import os

# Getting the spacy path 
cwd = os.getcwd()
spacy_dir = os.path.join(cwd, 'spacy_sg')

def reddit_locations(reddit_lda):
    # Input: reddit_lda -> 'reddit_lda.csv' path
    reddit = pd.read_csv(reddit_lda)
    reddit['comment_locations'] = ''
    # Initialise trained NER
    nlp = spacy.load(spacy_dir)
    for i in tqdm(range(len(reddit))):
        doc = nlp(reddit.loc[i,'comment'])
        address_lst = []
        for ent in doc.ents:
            address_lst.append(ent.text)
        if len(address_lst) == 0:
            address_str = ''
        else:
            address_str = ','.join(address_lst)
        reddit.loc[i,'comment_locations'] = address_str
    return reddit

if __name__ == '__main__':
    reddit_locations = reddit_locations('reddit_lda.csv')
    reddit_locations.to_csv('reddit_lda_with_locations.csv',index=False)