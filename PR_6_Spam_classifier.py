
# coding: utf-8

# In[29]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# In[30]:


spam_df=pd.read_csv('spam.csv',encoding='latin-1')


# In[31]:


spam_df


# In[32]:


spam_df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)


# In[12]:


spam_df


# In[33]:


spam_df.tail()


# In[34]:


spam_df.head(25)


# In[35]:


spam_df.head()


# In[36]:


spam_df.rename(columns={'v1':'Category','v2':'message'},inplace=True)


# In[37]:


spam_df


# In[38]:


spam_df.groupby('Category').describe()


# In[80]:


spam_df['spam']=spam_df['Category'].apply(lambda x:'spam' if x=='spam' else 'not spam')


# In[ ]:





# In[81]:


spam_df


# In[147]:


x_train,x_test,y_train,y_test=train_test_split(spam_df.message, spam_df.spam,test_size=0.11,random_state=42)


# In[148]:


x_train


# In[149]:


x_train.describe()



# In[150]:


cv=CountVectorizer()
x_train_count=cv.fit_transform(x_train.values)


# In[151]:


x_train_count.toarray()


# In[152]:


model=MultinomialNB()
model.fit(x_train_count,y_train)


# In[153]:


email_ham=["URGENT! You have won a 1 week FREE membership in our a$100,000 "]

email_ham_count=cv.transform(email_ham)
model.predict(email_ham_count)


# In[154]:


email_spam=["reward money click"]

email_spam_count=cv.transform(email_spam)
model.predict(email_spam_count)


# In[155]:


email_ham=["Sorry, I'll call later "]

email_ham_count=cv.transform(email_ham)
model.predict(email_ham_count)


# In[156]:


email_spam=["Free entry in 2 a wkly comp to win FA Cup fina...	"]

email_spam_count=cv.transform(email_spam)
model.predict(email_spam_count)


# In[126]:


x_test_count=cv.transform(x_test)
model.score(x_test_count,y_test)

