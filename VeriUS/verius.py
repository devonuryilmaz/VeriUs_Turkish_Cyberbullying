# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 17:11:17 2018

@author: ONUR
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from nltk.corpus import stopwords 
import re    
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


data = pd.read_csv('turkish cyberbullying.csv')
data.head()

stops = set(stopwords.words('turkish'))
print(stops)

exc_letters_pattern = '[^a-zçğışöü]'


def text_to_wordlist(text, remove_stopwords=False, return_list=False):
    # 1. convert to lower
    text = text.lower()
    # 2. replace special letters
    special_letters = {'î':'i', 'â': 'a'}
    for sp_let, tr_let in special_letters.items():
        text = re.sub(sp_let, tr_let, text)
    # 3. remove non-letters
    text = re.sub(exc_letters_pattern, ' ', text)
    # 4. split
    wordlist = text.split()
    # 5. remove stopwords
    if remove_stopwords:
        wordlist = [w for w in wordlist if w not in stops]
    # 6.
    if return_list:
        return wordlist
    else:
        return ' '.join(wordlist)
    
clean_messages = []
for message in data['message']:
    clean_messages.append(text_to_wordlist(
        message, remove_stopwords=True, return_list=False))    


x_train, x_test, y_train, y_test = train_test_split(
    clean_messages, data['cyberbullying'], test_size=0.33, random_state=1)    

vectorizer = TfidfVectorizer(max_features=5000, stop_words=stops)

train_data_features = vectorizer.fit_transform(x_train)

parameters = {'kernel':['linear','rbf'],
              'C':[0.1,1,10]}

svc = SVC()

clf = GridSearchCV(svc,parameters,cv=5)
clf.fit(train_data_features,y_train)

clf.best_params_

new = SVC(C=1,kernel='linear')
new.fit(train_data_features,y_train)

pred = new.predict(vectorizer.transform(x_test))


print('AUC:', roc_auc_score(y_test,pred))

print(new.predict(vectorizer.transform(["VeriUs'ta staj yapmak çok güzel"])))

print(new.predict(vectorizer.transform(["Sen karaktersizsin"])))










