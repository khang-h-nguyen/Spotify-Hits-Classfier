#!/usr/bin/env python
# coding: utf-8

# In[12]:


# Preparing the dataset from Spotify API
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import seaborn as sns
import spotipy
import time
from spotipy.oauth2 import SpotifyClientCredentials
client_id = 'f3992d15df3f46f38492dcf60d9fdd50'
client_secret = 'fc265ded75c142e4b986c9c51d9218a9'
client_credentials_manager = SpotifyClientCredentials(client_id, client_secret)
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)


# In[49]:


def getTrackIDs(user, playlist_id):
    ids = []
    playlist = sp.user_playlist(user, playlist_id)
    for item in playlist['tracks']['items']:
        track = item['track']
        ids.append(track['id'])
    return ids
ids1 = getTrackIDs('Spotify', 'spotify:playlist:37i9dQZF1DX4UtSsGT1Sbe')
ids2 = getTrackIDs('Spotify', 'spotify:playlist:4vSTV61efRmetmaoz95Vet')


# In[14]:


def getTrackFeatures(id):
  meta = sp.track(id)
  features = sp.audio_features(id)
  name = meta['name']
  artist = meta['album']['artists'][0]['name']
  length = meta['duration_ms']
  # features
  acousticness = features[0]['acousticness']
  danceability = features[0]['danceability']
  energy = features[0]['energy']
  instrumentalness = features[0]['instrumentalness']
  liveness = features[0]['liveness']
  loudness = features[0]['loudness']
  speechiness = features[0]['speechiness']
  tempo = features[0]['tempo']
  track = [name, artist, length, danceability, acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, tempo]
  return track


# In[50]:


# loop
tracks = []
for i in range(0,90):
  time.sleep(.5)
  track = getTrackFeatures(ids1[i])
  tracks.append(track)
# create dataset
df1 = pd.DataFrame(tracks, columns = ['name', 'artist', 'length', 'danceability', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo'])
df1.to_csv("1980s.csv", sep = ',')


# In[23]:


# loop
tracks = []
for i in range(0,90):
  time.sleep(.5)
  track = getTrackFeatures(ids2[i])
  tracks.append(track)
track
# create dataset
df2 = pd.DataFrame(tracks, columns = ['name', 'artist', 'length', 'danceability', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo'])
df2.to_csv("2020s.csv", sep = ',')


# In[51]:


hits1980s = pd.read_csv("1980s.csv")
hits2020s = pd.read_csv("2020s.csv")
plt.plot(hits1980s.length, color='blue')
plt.plot(hits2020s.length, color='red')
plt.xlabel("song #")
plt.ylabel("length ")
plt.title("length across all songs in both playlists")


# In[58]:


plt.plot(hits1980s.speechiness, color='blue')
plt.plot(hits2020s.speechiness, color='red')
plt.xlabel("song #")
plt.ylabel("length ")
plt.title("length across all songs in both playlists")


# In[59]:


plt.plot(hits1980s.loudness, color='blue')
plt.plot(hits2020s.loudness, color='red')
plt.xlabel("song #")
plt.ylabel("length ")
plt.title("length across all songs in both playlists")


# In[68]:


hits2020v80 = pd.read_csv("20v80.csv")
hits2020v80.head()


# In[69]:


hits2020v80.isnull().values.any()


# In[70]:


hits2020v80.dtypes


# In[71]:


# Normalizing the dataset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(hits2020v80.drop('target', axis=1))


# In[72]:


scaled_features = scaler.transform(hits2020v80.drop('target',axis=1))
scaled_features


# In[73]:


hits2020v80_feat = pd.DataFrame(scaled_features, columns = hits2020v80.columns[:-1])
hits2020v80_feat.head()


# In[74]:


# Splitting dataset into training and testing sets

from sklearn.model_selection import train_test_split
X = hits2020v80_feat
y = hits2020v80['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=30, shuffle=True)


# In[75]:


# Finding the k and training the model
import math
math.sqrt(len(y_test))


# In[122]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=7,p=2,metric='euclidean')
knn.fit(X_train, y_train)


# In[123]:


prediction = knn.predict(X_test)
prediction


# In[124]:


# Evaluating predictions using the Classification Report
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, prediction))


# In[118]:


# Plotting error rate to evaluate alternative K-values for better predictions
error_rate = []
for i in range(1,88):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    prediction_i = knn.predict(X_test)
    error_rate.append(np.mean(prediction_i != y_test))


# In[119]:


plt.plot(error_rate, color='blue')
plt.title('error rate vs. k values')
plt.xlabel('k values')
plt.ylabel('error rate')


# In[120]:


knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)
prediction= knn.predict(X_test)
prediction


# In[121]:


print(classification_report(y_test, prediction))


# In[ ]:




