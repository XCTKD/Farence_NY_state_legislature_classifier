#!/usr/bin/env python
# coding: utf-8

# # Here's what Adam did....

# #### Adam had a LOT of senate legislative data lying around. He tried to implement two machine learninng techniques: Classifers and linear regression.
# 
# #### The first section of this notebook is Adam importing and prepping some of the data before ML usage. Second section is trying to, hopefully somewhat succesfully, use a classifier to predict if a bill has been signed or not based on it's text.
# 
# #### The third section is a botched attempt at using linear regression to predict how long it would take a bill to get to the governor's desk based on it's language.

# #### Import and cleaning. Some column addition and df arranging.

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("2013SenateScraper2.0.csv",sep="|")


# In[3]:


df.columns


# In[4]:


df.signed_by_gov.value_counts()


# In[5]:


df_signed = df[df.signed_by_gov == True]


# In[6]:


df_signed.describe()


# In[7]:


df_signed = df_signed[~df_signed.last_legis_action.str.contains('no floor vote')]
df_signed = df_signed[~df_signed.dtg_date.str.contains('not delivered to governor')]


# In[8]:


df_signed['last_legis_action'] = pd.to_datetime(df_signed['last_legis_action'])
df_signed['dtg_date'] = pd.to_datetime(df_signed['dtg_date'])


# In[9]:


df_signed['total_time'] = df_signed['dtg_date'] - df_signed['last_legis_action']


# In[10]:


df_signed.total_time.append


# In[11]:


df_signed.columns


# # Count words with a vectorizer

# In[12]:


from sklearn.feature_extraction.text import CountVectorizer


# In[13]:


vectorizer = CountVectorizer()
df = df.dropna()
matrix = vectorizer.fit_transform(df.bill_text)
bill_text_vectorized_df = pd.DataFrame(matrix.toarray(),columns=vectorizer.get_feature_names())

bill_text_vectorized_df


# In[14]:


df['is_signed'] = (df['signed_by_gov'] == True).astype(int)
df.head()


# ### Classifier stuff. Random Forest, to be exact

# In[15]:


X = bill_text_vectorized_df

y = df.is_signed


# In[16]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10)
clf.fit(X, y)


# In[17]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[18]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X_train, y_train)


# In[19]:


from sklearn.metrics import confusion_matrix

y_true = y_test
y_pred = clf.predict(X_test)
matrix = confusion_matrix(y_true, y_pred)

label_names = pd.Series(['signed', 'not signed'])
pd.DataFrame(matrix,
     columns='Predicted ' + label_names,
     index='Is ' + label_names)


# # Linear Regression Time!!!! (this didn't work out)

# In[20]:


import statsmodels.formula.api as smf


# In[21]:


# model = smf.ols('total_time ~ bill_text', data=df_signed)
model = smf.ols('total_time ~ bill_text', data=df_signed, bill_text_vectorized_df)
results = model.fit()

results.summary()

