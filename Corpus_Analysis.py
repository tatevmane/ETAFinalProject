#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[9]:


import pandas as pd
import numpy as np
import seaborn as sns
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
import re
import warnings
warnings.filterwarnings("ignore")


# In[10]:


directory = "./"
data_files = "./files/"
data_out = "./data/"


# ## Importing LIB

# In[11]:


LIB = pd.read_csv("C:/Users/Student/Desktop/UVA/UVA '23 Spring/DS 5001/Final Project/LIB.csv")


# In[12]:


LIB.sample(10)


# ## Sorting into Paragraphs

# In[13]:


PARAS = LIB['text'].str.split("\+\+\+", expand=True).stack()    .to_frame('para_str').sort_index()
PARAS.index.names = ['text_num', 'para_num']
PARAS['para_str'] = PARAS['para_str'].str.replace(r'\n', ' ', regex=True)
PARAS['para_str'] = PARAS['para_str'].str.strip()
PARAS = PARAS[~PARAS['para_str'].str.match(r'^\s*$')]


# In[14]:


PARAS


# ## Sorting into Sentences

# In[15]:


SENTS = PARAS.para_str.apply(lambda x: pd.Series(nltk.sent_tokenize(x)))        .stack()        .to_frame('sent_str')
SENTS.index.names = ['text_num', 'para_num', 'sent_num']


# In[16]:


# standardizing text
SENTS['sent_str'] = SENTS['sent_str'].str.replace(r'\W', ' ').str.lower()
SENTS


# ## Getting Tokens

# In[17]:


keep_whitespace = True
if keep_whitespace:
    TOKENS = SENTS.sent_str            .apply(lambda x: pd.Series(nltk.pos_tag(nltk.word_tokenize(x))))            .stack()            .to_frame('pos_tuple')
else:
    TOKENS = SENTS.sent_str            .apply(lambda x: pd.Series(nltk.pos_tag(nltk.WhitespaceTokenizer().tokenize(x))))            .stack()            .to_frame('pos_tuple')


# In[18]:


TOKENS.index.names = ['text_num', 'para_num', "sent_num","token_num"]
TOKENS


# ## Making the Corups

# In[19]:


CORPUS = TOKENS
CORPUS['pos'] = CORPUS.pos_tuple.apply(lambda x: x[1])
CORPUS['token_str'] = CORPUS.pos_tuple.apply(lambda x: x[0])
CORPUS['term_str'] = CORPUS.token_str.str.lower()


# In[20]:


CORPUS


# In[21]:


CORPUS.reset_index(inplace=True)


# In[22]:


CORPUS['source'] = CORPUS['text_num'].apply(lambda x: 'CNN' if x <= 489 else 'CNBC')


# In[23]:


CORPUS.set_index(['source', 'text_num', 'para_num', 'sent_num', 'token_num'])


# In[28]:


# corpus to csv
CORPUS.to_csv("CORPUS.csv")


# ## Extracting VOCAB

# In[25]:


VOCAB = CORPUS.term_str.value_counts().to_frame('n').sort_index()
VOCAB.index.name = 'term_str'
VOCAB['n_chars'] = VOCAB.index.str.len()
VOCAB['p'] = VOCAB.n / VOCAB.n.sum()
VOCAB['i'] = -np.log2(VOCAB.p)
VOCAB['max_pos'] = CORPUS[['term_str','pos']].value_counts().unstack(fill_value=0).idxmax(1)
VOCAB['n_pos'] = CORPUS[['term_str','pos']].value_counts().unstack().count(1)
VOCAB['cat_pos'] = CORPUS[['term_str','pos']].value_counts().to_frame('n').reset_index()    .groupby('term_str').pos.apply(lambda x: set(x))
sw = pd.DataFrame(nltk.corpus.stopwords.words('english'), columns=['term_str'])
sw = sw.reset_index().set_index('term_str')
sw.columns = ['dummy']
sw.dummy = 1
VOCAB['stop'] = VOCAB.index.map(sw.dummy)
VOCAB['stop'] = VOCAB['stop'].fillna(0).astype('int')
VOCAB = VOCAB.drop('cat_pos', 1) 

stemmer1 = PorterStemmer()
VOCAB['stem_porter'] = VOCAB.apply(lambda x: stemmer1.stem(x.name), 1)

stemmer2 = SnowballStemmer("english")
VOCAB['stem_snowball'] = VOCAB.apply(lambda x: stemmer2.stem(x.name), 1)

stemmer3 = LancasterStemmer()
VOCAB['stem_lancaster'] = VOCAB.apply(lambda x: stemmer3.stem(x.name), 1)

VOCAB.sort_values('p', ascending=False).head(10)


# In[26]:


VOCAB


# In[27]:


# vocab to csv
VOCAB.to_csv("VOCAB.csv")

