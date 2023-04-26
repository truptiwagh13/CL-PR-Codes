
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


corpus=['data science is one of the most important fields of science',
       'this is one of the best data science courses',
       'data scientists analyse data']


# In[3]:


words_set=set()
for doc in corpus:
    words=doc.split(' ')
    words_set = words_set.union(set(words))
    
print('Number of words in the corpus : ',len(words_set))
print('The words in the corpus : \n',words_set)


# In[5]:


n_docs=len(corpus)
n_words_set=len(words_set)
df_tf=pd.DataFrame(np.zeros((n_docs,n_words_set)),columns=words_set)
for i in range(n_docs):
    words=corpus[i].split(' ')
    for w in words:
        df_tf[w][i]=df_tf[w][i]+(1/len(words))
df_tf


# In[6]:


print("IDF of: ")
idf={}
for w in words_set:
    k=0
    
    for i in range(n_docs):
        if w in corpus[i].split():
            k+=1
    idf[w]=np.log10(n_docs/k)
    print(f'{w}:{idf[w]}' )


# In[7]:


df_tf_idf=df_tf.copy()
for w in words_set:
    for i in range(n_docs):
        df_tf_idf[w][i]=df_tf[w][i]*idf[w]
df_tf_idf

