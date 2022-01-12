#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from wordcloud import WordCloud, STOPWORDS
import string
import spacy


# In[2]:


text=pd.read_csv('F:/Dataset/Elon_musk.csv',encoding='Latin-1')


# In[3]:


text.drop(['Unnamed: 0'],inplace=True,axis=1)


# In[4]:


text


# In[5]:


text=[Text.strip() for Text in text.Text]
text=[Text for Text in text if Text]
text[0:10]


# In[6]:


msg=' '.join(text)


# In[7]:


msg


# In[8]:


from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer(strip_handles=True)
tweets_tokens=tknzr.tokenize(msg)
print(tweets_tokens)


# In[9]:


tweets_tokens_text=' '.join(tweets_tokens)
tweets_tokens_text


# In[10]:


no_punc_text=tweets_tokens_text.translate(str.maketrans('','',string.punctuation))
no_punc_text


# In[11]:


import re
no_url_text=re.sub(r'http\S+', '', no_punc_text)
no_url_text


# In[19]:


from nltk.tokenize import word_tokenize
text_tokens=word_tokenize(no_url_text)
print(text_tokens)


# In[20]:


import nltk
nltk.download('punkt')
nltk.download('stopwords')


# In[21]:


len(text_tokens)


# In[22]:


from nltk.corpus import stopwords
my_stop_words=stopwords.words('english')

sw_list = ['\x92','rt','ye','yeah','haha','Yes','U0001F923','I']
my_stop_words.extend(sw_list)

no_stop_tokens=[word for word in text_tokens if not word in my_stop_words]
print(no_stop_tokens)


# In[23]:


lower_words=[Text.lower() for Text in no_stop_tokens]
print(lower_words[100:200])


# In[25]:


from nltk.stem import PorterStemmer
ps=PorterStemmer()
stemmed_tokens=[ps.stem(word) for word in lower_words]
print(stemmed_tokens[100:200])


# In[31]:


nlp=spacy.load('en_core_web_sm')
doc=nlp(' '.join(lower_words))
print(doc)


# In[32]:


lemmas=[token.lemma_ for token in doc]
print(lemmas)


# In[33]:


clean_tweets=' '.join(lemmas)
clean_tweets


# In[34]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
tweetscv=cv.fit_transform(lemmas)


# In[35]:


print(cv.vocabulary_)


# In[36]:


print(cv.get_feature_names()[100:200])


# In[37]:


print(tweetscv.toarray()[100:200])


# In[38]:


print(tweetscv.toarray().shape)


# In[39]:


cv_ngram_range=CountVectorizer(analyzer='word',ngram_range=(1,3),max_features=100)
bow_matrix_ngram=cv_ngram_range.fit_transform(lemmas)


# In[40]:


print(cv_ngram_range.get_feature_names())
print(bow_matrix_ngram.toarray())


# In[41]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidfv_ngram_max_features=TfidfVectorizer(norm='l2',analyzer='word',ngram_range=(1,3),max_features=500)
tfidf_matix_ngram=tfidfv_ngram_max_features.fit_transform(lemmas)


# In[43]:


print(tfidfv_ngram_max_features.get_feature_names())
print(tfidf_matix_ngram.toarray())


# In[44]:


def plot_cloud(wordcloud):
    plt.figure(figsize=(40,30))
    plt.imshow(wordcloud)
    plt.axis('off')


# In[45]:


STOPWORDS.add('pron')
STOPWORDS.add('rt')
STOPWORDS.add('yeah')
wordcloud=WordCloud(width=3000,height=2000,background_color='black',max_words=50,
                   colormap='Set1',stopwords=STOPWORDS).generate(clean_tweets)
plot_cloud(wordcloud)


# In[46]:


nlp=spacy.load('en_core_web_sm')

one_block=clean_tweets
doc_block=nlp(one_block)
spacy.displacy.render(doc_block,style='ent',jupyter=True)


# In[47]:


for token in doc_block[100:200]:
    print(token,token.pos_)


# In[48]:


nouns_verbs=[token.text for token in doc_block if token.pos_ in ('NOUN','VERB')]
print(nouns_verbs[100:200])


# In[49]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

X=cv.fit_transform(nouns_verbs)
sum_words=X.sum(axis=0)

words_freq=[(word,sum_words[0,idx]) for word,idx in cv.vocabulary_.items()]
words_freq=sorted(words_freq, key=lambda x: x[1], reverse=True)

wd_df=pd.DataFrame(words_freq)
wd_df.columns=['word','count']
wd_df[0:10]


# In[50]:


wd_df[0:10].plot.bar(x='word',figsize=(12,8),title='Top 10 nouns and verbs');


# In[ ]:





# In[ ]:





# In[ ]:




