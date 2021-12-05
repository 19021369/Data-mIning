import numpy as np
import nltk
import re
from sklearn.datasets import load_files
import pickle
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier

my_file = open("topic_detection_train.v1.0.txt", "r")
content = my_file.read()
print(content)

content_list = content.splitlines()
my_file.close()
length = len(content_list)
for i in range(length): 
    content_list[i] = content_list[i].split(" ", 1)

label = []
content = []

for i in range(length):
    label.append(content_list[i][0])
    content.append(content_list[i][1])
   
documents = []
for i in range(0, length):
    # Remove all the special characters
    document = re.sub('\W+', ' ', content[i])
    
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Converting to Lowercase
    document = document.lower()

    documents.append(document)
    
# Converting Text to Numbers
my_file = open("stopword.txt", "r")
stop_word = my_file.read().splitlines()
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=5000, min_df=10, max_df=0.7, stop_words=stop_word)
content = vectorizer.fit_transform(documents).toarray()

# Finding TFIDF
from sklearn.feature_extraction.text import TfidfTransformer
tfidfconverter = TfidfTransformer()
content = tfidfconverter.fit_transform(content).toarray()

# Training and Testing Sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(content, label, test_size=0.2, random_state=0)

# Training Text Classification Model 
classifier = RandomForestClassifier(n_estimators=500, random_state=0)
classifier.fit(x_train, y_train)

# Predicting Topic
y_pred = classifier.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))


test_file = open("topic_detection_test_unlabel.v1.0.txt", "r")
testcontent = test_file.read()
test_list = testcontent.splitlines()
test_file.close()

from sklearn.feature_extraction.text import TfidfVectorizer
tfidfconvertertest = TfidfVectorizer(max_features=5000, min_df=10, max_df=0.7, stop_words=stop_word)
test_list = tfidfconvertertest.fit_transform(documents).toarray()

label_predict = classifier.predict(test_list)

with open("label_predict.txt", "w+") as f:
    for item in label_predict:
        f.write("%s\n" % item)
