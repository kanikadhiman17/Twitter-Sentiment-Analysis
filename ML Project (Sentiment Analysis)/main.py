#!/usr/bin/env python
# coding: utf-8

# # Sentimental Analysis

# ### Dataset `Tweets Airlines`
# 
# **- Listing all the column names**

# In[2]:


import pandas as pd
tweets = pd.read_csv("Tweets.csv")
list(tweets.columns.values)


# **The first five data-set**

# In[3]:


tweets.head()


# ### Listed the number of positive, negative and neutral tweets
# 

# In[4]:


sentiment_counts = tweets.airline_sentiment.value_counts()
print(sentiment_counts)
print("\n")


# ### All the different airlines with all the labels count

# In[5]:


dff = tweets.groupby(["airline", "airline_sentiment" ]).count()["name"]
df_companySentiment = dff.to_frame().reset_index()
df_companySentiment.columns = ["airline", "airline_sentiment", "count"]
df_companySentiment


# ### A plot showing the negativeness in tweets by different airlines

# In[6]:


import matplotlib.pyplot as plt
import matplotlib.style
# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.style
from matplotlib.pyplot import subplots
matplotlib.style.use('ggplot')

df2 = df_companySentiment
df2.index = df2['airline']
del df2['airline']
df2
fig, ax = subplots()
my_colors =['darksalmon', 'papayawhip', 'cornflowerblue']
df2.plot(kind='bar', stacked=True, ax=ax, color=my_colors, figsize=(12, 7), width=0.8)
ax.legend(["Negative", "Neutral", "Positive"])
plt.title("Tweets Sentiments Analysis Airlines, 2017")
plt.show()


# **WordNetLemmatizer-** lemmatizing can often create actual words.
# 
# **stopwords-** Stopwords are the English words which does not add much meaning to a sentence.

# In[13]:


import re, nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()

def normalizer(tweet):
    only_letters = re.sub("[^a-zA-Z]", " ",tweet) 
    tokens = nltk.word_tokenize(only_letters)[2:]
    lower_case = [l.lower() for l in tokens]
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    return lemmas


# In[15]:


normalizer("I recently wrote some texts.")


# In[16]:


pd.set_option('display.max_colwidth', -1) # Setting this so we can see the full content of cells
tweets['normalized_tweet'] = tweets.text.apply(normalizer)
tweets[['text','normalized_tweet']].head()


# In[17]:


from nltk import ngrams
def ngrams(input_list):
    #onegrams = input_list
    bigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:]))]
    trigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:], input_list[2:]))]
    return bigrams+trigrams
tweets['grams'] = tweets.normalized_tweet.apply(ngrams)
tweets[['grams']].head()


# In[18]:


import collections
def count_words(input):
    cnt = collections.Counter()
    for row in input:
        for word in row:
            cnt[word] += 1
    return cnt


# In[19]:


tweets[(tweets.airline_sentiment == 'negative')][['grams']].apply(count_words)['grams'].most_common(20)


# In[21]:


train_neg = tweets[(tweets.airline_sentiment == 'negative')]
train_neg = train_neg['text']
train_pos = tweets[(tweets.airline_sentiment == 'positive')]
train_pos = train_pos['text']

from wordcloud import WordCloud,STOPWORDS

def wordcloud_draw(data, color = 'black'):
    words = ' '.join(data)
    cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and not word.startswith('#')
                                and word != 'RT'
                            ])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color=color,
                      width=2500,
                      height=2000
                     ).generate(cleaned_word)
    plt.figure(1,figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    
print("Negative words")
print("\n")
wordcloud_draw(train_neg)


# In[22]:


tweets[(tweets.airline_sentiment == 'positive')][['grams']].apply(count_words)['grams'].most_common(20)


# In[23]:


print("Positive words")
print("\n")
wordcloud_draw(train_pos, 'white')


# **CountVectorizer -** implements both tokenization and occurrence counting in a single class

# In[26]:


import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer(ngram_range=(1,2))


# In[27]:


vectorized_data = count_vectorizer.fit_transform(tweets.text)
indexed_data = hstack((np.array(range(0,vectorized_data.shape[0]))[:,None], vectorized_data))


# In[28]:


def sentiment2target(sentiment):
    return {
        'negative': 0,
        'neutral': 1,
        'positive' : 2
    }[sentiment]
targets = tweets.airline_sentiment.apply(sentiment2target)


# In[29]:


from sklearn.model_selection import train_test_split
data_train, data_test, targets_train, targets_test = train_test_split(indexed_data, targets, test_size=0.4, random_state=0)
data_train_index = data_train[:,0]
data_train = data_train[:,1:]
data_test_index = data_test[:,0]
data_test = data_test[:,1:]


# **OneVsRestClassifier-** this strategy consists in fitting one classifier per class. For each classifier, the class is fitted against all the other classes.

# In[31]:
print("\n")
print("Training the model, please wait....")
print("\n")
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score

clf = OneVsRestClassifier(svm.SVC(gamma=0.01, C=100., probability=True, class_weight='balanced', kernel='linear'))
# scores = cross_val_score(clf, indexed_data, targets, cv=1)
# scores
clf_output = clf.fit(data_train, targets_train).decision_function(data_test)

print("Model Training completed, Testing accuracy on the Test Data")
print("\n")
# In[32]:


accuracies = clf.score(data_test, targets_test)
print("Accuracy is: ", accuracies)
print("\n")
print("\n")
# In[33]:


sentences = count_vectorizer.transform([
    "What a great airline, the trip was a pleasure!",
    "My issue was quickly resolved after calling customer support. Thanks!",
    "What the hell! My flight was cancelled again. This sucks!",
    "Service was awful. I'll never fly with you again.",
    "You fuckers lost my luggage. Never again!",
    "I have mixed feelings about airlines. I don't know what I think.",
    ""
])
print("Predicting on some sentences")
print("1. What a great airline, the trip was a pleasure!")
print("2. My issue was quickly resolved after calling customer support. Thanks!")
print("3. What the hell! My flight was cancelled again. This sucks!")
print("4. Service was awful. I'll never fly with you again.")
print("5. You fuckers lost my luggage. Never again!")
print("6. I have mixed feelings about airlines. I don't know what I think.")
print("7. ", "an empty sentence")
print("\n")
predictions_sentences = clf.predict_proba(sentences)
print(predictions_sentences)
print("\n")
print("ROC Curve")
# ### ROC Curve

# In[34]:


from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

# Compute macro-average ROC curve and ROC area
y = label_binarize(targets_test, classes=[0, 1, 2])
n_classes = y.shape[1]

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y[:, i], clf_output[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), clf_output.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# In[35]:


from scipy import interp
from itertools import cycle

lw = 2

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
print("\n")
print("thanks")
