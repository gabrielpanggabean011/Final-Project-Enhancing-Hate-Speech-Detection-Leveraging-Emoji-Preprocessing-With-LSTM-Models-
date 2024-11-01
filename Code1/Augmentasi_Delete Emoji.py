#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Steps:
# 1) preprocessing
# 2) splitting
# 3) tokenize & padding
# 4) Create model & train
# 5) evaluate
# 
# # Preprocessing (cleaning the datasets):
# 
# 1) remove html entity
# 2) change user tags (@xxx -> user)
# 3) remove urls
# 4) remove unnecessary symbol ('', !, ", ') -> cause a lot of noise in the dataset
# 5) remove stopwords

# # 1| Import libraries

# In[ ]:


import pandas as pd # read the csv
import re # regex to detect username, url, html entity 
import nltk # to use word tokenize (split the sentence into words)
from nltk.corpus import stopwords # to remove the stopwords
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

from keras.utils import to_categorical
from keras import backend as K


# # 2| read the data

# In[ ]:


# Mempersiapkan Data
df_train = pd.read_csv('train_augmentasi.csv')


df_valid = pd.read_csv('validation_augmentasi.csv')


df_test = pd.read_csv('test_augmentasi.csv')



# In[ ]:


data = pd.concat([df_train, df_valid, df_test], ignore_index=True)


# In[ ]:


data.head()


# In[ ]:


# dataset shape to know how many tweets in the datasets
print(f"num of tweets: {data.shape[0]}")

# extract the text and labels
tweet = list(data['text'])
labels = list(data['label_gold'])


# In[ ]:


import pandas as pd

# Misalnya, kita memiliki DataFrame 'data' yang sudah terdefinisi
# data = pd.DataFrame(...)

# Menampilkan jumlah tweet dalam dataset
print(f"num of tweets: {data.shape[0]}")

# Ekstraksi teks dan label
tweet = list(data['text'])
labels = list(data['label_gold'])

# Menghitung jumlah nilai unik dalam kolom 'label_gold'
num_unique_labels = data['label_gold'].nunique()

# Menampilkan jumlah nilai unik dalam kolom 'label_gold'
print(f"Number of unique labels in 'label_gold': {num_unique_labels}")

# Menghitung jumlah teks untuk tiap nilai label
label_counts = data['label_gold'].value_counts()

# Menampilkan jumlah teks untuk tiap nilai label
for label, count in label_counts.items():
    print(f"Label {label}: {count} tweets")


# # 3| functions to clean the data

# In[ ]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import emoji
import re

# Ensure you have downloaded the necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

# Remove emojis
def remove_emojis(text):
    return emoji.replace_emoji(text, replace='')


stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Remove unnecessary symbols and lowercasing
def remove_noise_symbols(raw_text):
    text = raw_text.lower()  # Lowercasing
    text = text.replace('"', '')
    text = text.replace("'", '')
    text = text.replace("!", '')
    text = text.replace("`", '')
    text = text.replace("..", '')
    text = text.replace("_", ' ')
    text = text.replace("-", ' ')
    text = text.replace(":", '')
    text = text.replace("=", '')
    text = text.replace("?", '')
    return text

# Remove stopwords
def remove_stopwords(raw_text):
    tokenize = word_tokenize(raw_text)
    text = [word for word in tokenize if not word.lower() in stop_words]
    text = " ".join(text)
    return text

# Stem words
def stem_words(raw_text):
    tokenize = word_tokenize(raw_text)
    text = [stemmer.stem(word) for word in tokenize]
    text = " ".join(text)
    return text

# Remove hashtags
def remove_hashtags(raw_text):
    return re.sub(r'#\w+', '', raw_text)

# Normalisasi kata
def normalize_words(text):
    normalized_text = text
    normalized_text = normalized_text.replace("thx", "thanks")
    normalized_text = normalized_text.replace("u", "you")
    normalized_text = normalized_text.replace("otw", "on the way")
    normalized_text = normalized_text.replace("r", "are")
    normalized_text = normalized_text.replace("lol", "laughing out loud")
    normalized_text = normalized_text.replace("brb", "be right back")
    normalized_text = normalized_text.replace("gonna", "going to")
    normalized_text = normalized_text.replace("wanna", "want to")
    normalized_text = normalized_text.replace("btw", "by the way")
    normalized_text = normalized_text.replace("idk", "I don't know")
    normalized_text = normalized_text.replace("smh", "shaking my head")
    normalized_text = normalized_text.replace("omg", "oh my god")
    normalized_text = normalized_text.replace("tbh", "to be honest")
    normalized_text = normalized_text.replace("afaik", "as far as I know")
    normalized_text = normalized_text.replace("afk", "away from keyboard")
    normalized_text = normalized_text.replace("ttyl", "talk to you later")
    normalized_text = normalized_text.replace("imho", "in my humble opinion")
    normalized_text = normalized_text.replace("fomo", "fear of missing out")
    normalized_text = normalized_text.replace("ftw", "for the win")
    normalized_text = normalized_text.replace("irl", "in real life")
    normalized_text = normalized_text.replace("tfw", "that feeling when")
    normalized_text = normalized_text.replace("btw", "by the way")
    normalized_text = normalized_text.replace("imo", "in my opinion")
    normalized_text = normalized_text.replace("gtfo", "get the fuck out")
    normalized_text = normalized_text.replace("lmao", "laughing my ass off")
    normalized_text = normalized_text.replace("hmu", "hit me up")
    normalized_text = normalized_text.replace("nvm", "never mind")
    normalized_text = normalized_text.replace("ikr", "I know, right?")
    normalized_text = normalized_text.replace("smh", "shaking my head")
    normalized_text = normalized_text.replace("tldr", "too long; didn't read")
    normalized_text = normalized_text.replace("np", "no problem")
    normalized_text = normalized_text.replace("lol", "laugh out loud")
    normalized_text = normalized_text.replace("smdh", "shaking my damn head")
    # Tambahkan normalisasi kata lainnya di sini sesuai kebutuhan
    
    return normalized_text

# This function preprocesses the dataset by utilizing all the functions above
def preprocess(datas):
    clean = []
    for text in datas:
        # Remove emojis
        text = remove_emojis(text)
        # Remove trailing stuff and lowercasing
        text = remove_noise_symbols(text)
        # Remove stopwords
        text = remove_stopwords(text)
        # Stem words
        text = stem_words(text)
        # Remove hashtags
        text = remove_hashtags(text)
        # Normalisasi kata
        text = normalize_words(text)
        
        clean.append(text)

    return clean

# Contoh penggunaan
cleaned_texts = preprocess(data['text'])
print(cleaned_texts)


# In[ ]:


# call the cleaning function
clean_tweet = preprocess(tweet)


# In[ ]:


# Create a DataFrame with the cleaned tweets and labels
cleaned_data = pd.DataFrame({'text': clean_tweet, 'label_gold': labels})

# Remove rows where the cleaned text is null or empty
cleaned_data = cleaned_data[cleaned_data['text'].str.strip().astype(bool)]

# Extract the cleaned text and labels
clean_tweet = list(cleaned_data['text'])
labels = list(cleaned_data['label_gold'])

# Display the cleaned data
print(f"Number of tweets after cleaning: {cleaned_data.shape[0]}")
print(cleaned_data.head())


# # 4| Splitting the dataset into test and validation

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(clean_tweet, labels, test_size=0.2, random_state=42)


# In[ ]:


## Tokenizing -> basically we use tokenisation for many things, its commonly used for feature extraction in preprocessing. btw idk how it works as feature extraction tho :(
# declare the tokenizer
tokenizer = Tokenizer()
# build the vocabulary based on train dataset
tokenizer.fit_on_texts(X_train)
# tokenize the train and test dataset
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# vocabulary size (num of unique words) -> will be used in embedding layer
vocab_size = len(tokenizer.word_index) + 1


# In[ ]:


## Padding -> to uniform the datas
max_length = max(len(seq) for seq in X_train)

# to test an outlier case (if one of the test dataset has longer length)
for x in X_test:
    if len(x) > max_length:
        print(f"an outlier detected: {x}")

X_train = pad_sequences(X_train, maxlen=max_length)
X_test = pad_sequences(X_test, maxlen=max_length)


# In[ ]:


max_length


# In[ ]:


# create hot_labels (idk whty tapi ini penting, kalo ga bakal error)
y_test = to_categorical(y_test, num_classes=2)
y_train = to_categorical(y_train, num_classes=2)


# In[ ]:


# another look on the number of tweet in test and training data

print(f"num test tweet: {y_test.shape[0]}")
print(f"num train tweet: {y_train.shape[0]}")


# # 5| Building the model

# In[ ]:


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    precisions = precision(y_true, y_pred)
    recalls = recall(y_true, y_pred)
    return 2*((precisions*recalls)/(precisions+recalls+K.epsilon()))


# In[ ]:


from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense




# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam

embedding_dim = 300

def create_model():
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        LSTM(256, dropout=0.3, recurrent_dropout=0.3, return_sequences=True),
        Dropout(0.4),
        LSTM(256, dropout=0.3, recurrent_dropout=0.3, return_sequences=True),
        Dropout(0.4),
        LSTM(128, dropout=0.3, recurrent_dropout=0.3, return_sequences=True),
        Dropout(0.4),
        LSTM(128, dropout=0.3, recurrent_dropout=0.3),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dense(32, activation='relu', kernel_regularizer='l2'),
        Dropout(0.2),
        Dense(2, activation='sigmoid')  # Using sigmoid for binary classification
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy', f1, precision, recall])
    
    return model

# Menampilkan summary model
model = create_model()
model.summary()


# In[ ]:


from sklearn.model_selection import KFold

# Parameter pelatihan
BATCH_SIZE = 32
EPOCHS = 100

# Mendefinisikan KFold dengan 5 split
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# List untuk menyimpan hasil evaluasi dari setiap fold
evaluation_results = []

# Iterasi melalui setiap fold
for fold_num, (train_index, val_index) in enumerate(kfold.split(X_train)):
    print(f"Training on Fold {fold_num+1}...")
    
    # Memisahkan data menjadi data latih dan validasi berdasarkan index
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    
    # Memuat model (asumsi model sudah didefinisikan sebelumnya)
    model = create_model()  # Isi dengan cara memuat model Anda
    
    # Melatih model pada data latih
    model.fit(
        X_train_fold,
        y_train_fold,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val_fold, y_val_fold)
    )
    
    val_evaluation = model.evaluate(X_val_fold, y_val_fold)
    print(f"Validation Evaluation on Fold {fold_num+1}: Loss - {val_evaluation[0]}, Accuracy - {val_evaluation[1]}")
    
    test_evaluation = model.evaluate(X_test, y_test)
    print(f"Test Evaluation after Fold {fold_num+1}: Loss - {test_evaluation[0]}, Accuracy - {test_evaluation[1]}, F1 - {test_evaluation[2]}, Precision - {test_evaluation[3]}, Recall - {test_evaluation[4]}")

# Final evaluation on test data
final_evaluation = model.evaluate(X_test, y_test)
test_loss = final_evaluation[0]
test_accuracy = final_evaluation[1]
test_f1 = final_evaluation[2]
test_precision = final_evaluation[3]
test_recall = final_evaluation[4]

print(f"\nFinal Evaluation on Test Data: Loss - {test_loss}, Accuracy - {test_accuracy}, F1 - {test_f1}, Precision - {test_precision}, Recall - {test_recall}")

# Predicting the labels on the test set
y_pred = model.predict(X_test)
y_pred_binary = np.argmax(y_pred, axis=1)
y_test_binary = np.argmax(y_test, axis=1)

from sklearn.metrics import classification_report

# Generating the classification report
report = classification_report(y_test_binary, y_pred_binary)
print(report)




