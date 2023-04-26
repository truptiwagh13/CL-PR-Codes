
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[3]:


text=["This is the first the  document.",
      "This document is the second document.",
      "And This is the third one.",
      "Is This the first document?"
      ]


# In[4]:


vectorizer=TfidfVectorizer()


# In[5]:


x=vectorizer.fit_transform(text)
print(x)


# In[6]:


vectorizer.get_feature_names()


# In[8]:


print(x.shape)


# In[10]:


df=pd.DataFrame(x.toarray(),columns=vectorizer.get_feature_names(),index=["D1","D2","D3","D4"])


# In[11]:


df


# In[12]:


print(vectorizer.vocabulary_)


# In[13]:


print(vectorizer.idf_)

