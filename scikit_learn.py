import json
import random

class Review:
    def __init__(self, text, rating):
        self.text = text
        self.rating = rating
        self.sentiment = self.get_sentiment()

    def get_sentiment(self):
        if self.rating <= 2.5:
            return "NEGATIVE"
        return "POSITIVE"


class ReviewContainer:
    def __init__(self, reviews):
        self.reviews = reviews

    def distributions(self):
        negative = list(filter(lambda x: x.sentiment == "NEGATIVE", self.reviews))
        positive = list(filter(lambda x: x.sentiment == "POSITIVE", self.reviews))
        equal_positive = positive[:len(negative)]
        self.reviews = negative + equal_positive
        random.shuffle(self.reviews)

    def get_text(self):
        return [data.text for data in self.reviews]

    def get_sentiment(self):
        return [data.sentiment for data in self.reviews]

file_name = 'Books_small_10000.json'
reviews = []
with open(file_name) as f:
    for line in f:
        jf = json.loads(line)
        reviews.append(Review(jf['reviewText'], jf['overall']))

# print(rbox[0].text)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
training, test = train_test_split(reviews, test_size=0.33, random_state=42)
# print(len(training))
# print(len(test))
train_contnr = ReviewContainer(training)
test_contnr = ReviewContainer(test)

train_contnr.distributions()
test_contnr.distributions()

# print(len(test_contnr.reviews))


# train_x = [data.text for data in training]
train_x = train_contnr.get_text()
# train_y = [data.sentiment for data in training]
train_y = train_contnr.get_sentiment()

# test_x = [data.text for data in test]
test_x = test_contnr.get_text()
# test_y = [data.sentiment for data in test]
test_y = test_contnr.get_sentiment()

# print(train_y.count("POSITIVE"))    # Checking distribution
# print(train_y.count("NEGATIVE"))    # Checking distribution

# Bag of words vectorization
# vectorizer = CountVectorizer()
vectorizer = TfidfVectorizer()      # For specific words
train_x_vectors = vectorizer.fit_transform(train_x)

test_x_vectors = vectorizer.transform(test_x)

# Linear SVM
clf_svm = svm.SVC(kernel='linear')
clf_svm.fit(train_x_vectors, train_y)
clf_svm.predict(test_x_vectors[0])

# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
clf_dec = DecisionTreeClassifier()
clf_dec.fit(train_x_vectors, train_y)
clf_dec.predict(test_x_vectors[0])

# GaussianNB
from sklearn.naive_bayes import GaussianNB
clf_gnb = GaussianNB()
# clf_gnb.fit(train_x_vectors, train_y)
# print(clf_gnb.predict(test_x_vectors[0]))

# Logistic Regression
from sklearn.linear_model import LogisticRegression
clf_log = LogisticRegression()
clf_log.fit(train_x_vectors, train_y)
clf_log.predict(test_x_vectors[0])

# Mean Accuracy
print(clf_svm.score(test_x_vectors, test_y))
# print(clf_dec.score(test_x_vectors, test_y))
# print(clf_log.score(test_x_vectors, test_y))

# F1 Score
from sklearn.metrics import f1_score
x = f1_score(test_y, clf_log.predict(test_x_vectors), average=None, labels=["NEGATIVE", "POSITIVE"])
# print(x)

test_set = ['not great', 'good book buy it', 'waste of time']
test_vctzr = vectorizer.transform(test_set)

print(clf_svm.predict(test_vctzr))


# Grid Search
from sklearn.model_selection import GridSearchCV
parameters = {'kernel': ('linear', 'rbf'), 'C': (1, 4, 8, 16, 32)}

svc = svm.SVC()
clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(train_x_vectors, train_y)

# print(clf.score(test_x_vectors, test_y))

# Save & Load Model
import pickle

# with open('sentiment_classifier.pkl', 'wb') as f:
#     pickle.dump(clf, f)

with open('sentiment_classifier.pkl', 'rb') as f:
    loaded_cf = pickle.load(f)

print(test_x[0])
print(loaded_cf.predict(test_x_vectors[0]))