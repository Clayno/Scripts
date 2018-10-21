# Adapted from https://github.com/tthustla/twitter_sentiment_analysis_part1/blob/master/Capstone_part2.ipynb

import re
import pandas as pd
from bs4 import BeautifulSoup


pat1 = r'\@[A-Za-z0-9]+'
pat2 = r'http(s?)://[A-Za-z0-9\./\?\=]+'
pat3 = r'\#[A-Za-z0-9]+'

def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    stripped = re.sub(pat1, '', souped)
    stripped = re.sub(pat2, '', stripped)
    stripped = re.sub(pat3, '', stripped)
    try:
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped
    return clean

df=pd.read_csv('training140.csv', header=None, encoding = "ISO-8859-1")
label = df[0]
text = df[5]

print "Cleaning and parsing the tweets...\n"
clean_tweet_texts = []
size = len(text)
for i in xrange(0, size):
    if( (i+1)%10000 == 0 ):
        print "Tweets %d of %d has been processed" % (i+1, size)                                                                    
    clean_tweet_texts.append(tweet_cleaner(text[i]))
len(clean_tweet_texts)
	
clean_df = pd.DataFrame(clean_tweet_texts,columns=['text'])
clean_df['label'] = label

clean_df.to_csv('clean_training140.csv',encoding='utf-8')