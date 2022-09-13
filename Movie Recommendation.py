#!/usr/bin/env python
# coding: utf-8

# In[122]:


import numpy as np
import pandas as pd
import ast


# In[123]:


movies = pd.read_csv('movies.csv')
credits = pd.read_csv('credits.csv')


# In[124]:


movies = movies.merge(credits, on ='title')


# In[125]:


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[126]:


movies.isnull().sum()


# In[127]:


movies.dropna(inplace=True)


# In[128]:


movies.duplicated().sum()


# In[129]:


movies.iloc[0].genres


# In[131]:


def getName(obj):
    L = []
    for i in ast.literal_eval(obj):       #Converts list of string into integers
        L.append(i['name'])
    return L


# In[132]:


movies['genres'] = movies['genres'].apply(getName)


# In[ ]:





# In[133]:


movies['keywords'] = movies['keywords'].apply(getName)


# In[134]:


def getTop3(obj):
    counter=0
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
        counter+=1
        if(counter==3):
            break
    return L


# In[135]:


movies['cast'].apply(getTop3)


# In[136]:


movies['cast'] = movies['cast'].apply(getTop3)


# In[137]:


def fetchDirector(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if(i['job']=='Director'):
            L.append(i['name'])
            break
    return L


# In[138]:


movies['crew'] = movies['crew'].apply(fetchDirector)


# In[139]:


movies['overview']=movies['overview'].apply(lambda x: x.split())


# In[140]:


movies['genres']=movies['genres'].apply(lambda x: [i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x: [i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x: [i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x: [i.replace(" ","") for i in x])


# In[141]:


movies['tags'] =  movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[142]:


df = movies[['movie_id','title','tags']]


# In[143]:


df['tags'] = df['tags'].apply(lambda x: " ".join(x))


# In[144]:


df['tags']=df['tags'].apply(lambda x : x.lower())


# In[145]:


df


# In[146]:


import nltk


# In[147]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[148]:


def stem(obj):
    y = []
    for i in obj.split():
        y.append(ps.stem(i))
    s = " ".join(y)
    return s


# In[149]:


df['tags'] = df['tags'].apply(stem)


# In[150]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')


# In[151]:


vectors = cv.fit_transform(df['tags']).toarray()


# In[152]:


cv.get_feature_names()


# In[153]:


from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)


# In[158]:


vectors.shape


# In[159]:


def recommend(movie):
    m_index = df[df['title'] == movie].index[0]
    distances = similarity[m_index]
    movie_list = sorted(list(enumerate(distances)),reverse=True, key=lambda x: x[1])[1:10]
    
    for i in movie_list:
        print(df.iloc[i[0]].title)


# In[160]:


recommend('Batman Begins')


# In[161]:


df.to_dict()


# In[162]:


import pickle
pickle.dump(df.to_dict(),open('movie_dict.pkl','wb'))


# In[163]:


pickle.dump(similarity,open('similarity.pkl','wb'))

