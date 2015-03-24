
import httplib2
from apiclient.discovery import build
import urllib
import json
import csv
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import random
import scipy.stats
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import Imputer

API_KEY = "AIzaSyCtw2c6LVCpUq3xYr4fnMm5fmVhOiyWMvI"
TABLE_ID = "1xTqZaR7mK_e2GwfLlPDRUytXy3S4lzYzwf2G_D7K"

# open the data stored in a file called "data.json"
try:
    fp = open("data/dogs.json")
    dogs = json.load(fp)
    fp = open("data/cats.json")
    cats = json.load(fp)

# but if that file does not exist, download the data from fusion tables
except IOError:
    service = build('fusiontables', 'v1', developerKey=API_KEY)
    query = "SELECT * FROM " + TABLE_ID + " WHERE  AnimalType = 'DOG'"
    dogs = service.query().sql(sql=query).execute()
    fp = open("data/dogs.json", "w+")
    json.dump(dogs, fp)
    query = "SELECT * FROM " + TABLE_ID + " WHERE  AnimalType = 'CAT'"
    cats = service.query().sql(sql=query).execute()
    fp = open("data/cats.json", "w+")
    json.dump(cats, fp)

rows = dogs['rows']  # the actual data

ages = ["Infant - Younger than 6 months", "Youth - Younger than 1 year",
        "Older than 1 year", "Older than 7 years", "Other"]

outcomes = ["Returned to Owner", "Transferred to Rescue Group", "Adopted",
            "Foster", "Euthanized", "Other"]
sort_outs = ["Home", "Euthanized", "Other"]

data = [('Home',        np.array([0.0,0,0,0,0])),
        ('Euthanized',  np.array([0.0,0,0,0,0])),
        ('Other',       np.array([0.0,0,0,0,0]))]

# imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
# imp.transform(rows)

def classify_dog(o,j):
    try:
        o = outcomes.index(o)
        if o == 4: # Re-classify euthanized to correct index
            o = 1
        else:
            o = 0 # We are reclassifying everything but euthanized as 'home'
    except:
        o = 2 # And of course if it is not found... 'other'

    try:
        a = ages.index(j)
    except:
        a = 4 # Unknown ages classified as "other"

    data[o][1][a] += 1 # Another placed doggie

for dog in rows:
    age = dog[7]
    outcome = dog[13]

    classify_dog(outcome,age)

# plot the data to see what it looks like
fig, ax = plt.subplots()
index = np.arange(5)
bar_width = 0.15
opacity = 0.4

rects1 = plt.bar(index,
                 data[0][1], bar_width, alpha=opacity,
                 color='b', label="Home")
rects2 = plt.bar(index+bar_width,
                 data[1][1], bar_width,
                 alpha=opacity, color='r', label='Euth.')
rects3 = plt.bar(index+(bar_width*2),
                 data[2][1], bar_width,
                 alpha=opacity, color='g', label='Other')

plt.ylabel('Number')
plt.title('Number of dogs by age and adoption outcome')
plt.xticks(index + (bar_width), ages)
plt.legend()
plt.tight_layout()
plt.ticklabel_format(labelsize='small')
plt.show()

Observed = np.array([data[0][1], data[1][1], data[2][1]])

X_2, p, dof, expected= scipy.stats.chi2_contingency(Observed)
print "CHI-squared: ", X_2, "p = ", p

########################################
# CHI-squared:  1457.32389168 p =  2.27695285736e-309
########################################


try:
    fp = open("data/random_dogs_and_cats.json")
    all_data = np.array(json.load(fp))

# but if that file does not exist, download the data from fusiontables
except IOError:
    # make an array of all data about cats and dogs
    all_data = cats['rows'] + dogs['rows']
    # randomize it so the cats aren't all first
    np.random.shuffle(all_data)
    fp = open("data/random_dogs_and-cats.json", "w+")
    json.dump(all_data, fp)
    all_data = np.array(all_data)

# features = ['AnimalType', 'IntakeMonth', 'Breed', 'Age', 'Sex',
#             'SpayNeuter', 'Size', 'Color', 'IntakeType']

features = ['AnimalType', 'Breed', 'Age',
            'SpayNeuter', 'Size', 'IntakeType']

cols = dogs['columns']
out = "OutcomeType"

use_data = []

ncols = len(cols)

for i in np.arange(ncols):
    try:
        features.index(cols[i])
        use_data.append(i)

    except ValueError:
        if cols[i] == out:
            out_index = i

X = all_data[:, use_data]
y = all_data[:, out_index]

# Make all the outcomes that are very rare be "Other"
y[y=="No Show"] = "Other"
y[y=="Missing Report Expired"] = "Other"
y[y=="Found Report Expired"] = "Other"
y[y=="Lost Report Expired"] = "Other"
y[y=="Released in Field"] = "Other"
y[y==''] = "Other"
y[y=="Died"] = "Other"
y[y=="Disposal"] = "Other"
y[y=="Missing"] = "Other"
y[y=="Trap Neuter/Spay Released"] = "Other"
y[y=="Transferred to Rescue Group"] = "Other"
y[y==u'Foster']="Other"


y[y=="Returned to Owner"] = "Home"
y[y==u'Adopted']="Home"
y[y==u'Euthanized']="Euthanized"

Outcomes = ["Euth.", "Home", "Other"]

# We'll use the first 20%. This is fine
# to do because we know the data is randomized.
nrows = len(all_data)
percent = len(X)/5
X_opt = X[:percent, :]
y_opt = y[:percent]
# and a train/test set
X_rest = X[percent:, :]
y_rest = y[percent:]

# ======================================================
# use scikit-learn
# ======================================================

# and we need to convert all the data from strings to numeric values
le = preprocessing.LabelEncoder()
labels = []

# collect all the labels. The csv files we are loading
# were generated back in byte 2 and are provided as part
# of this source code. They just contain all possible
# values for each column. We're putting those values all
# in a list now
for name in features:
    csvfile = open('data/{0}.csv'.format(name), 'rb')
    datareader = csv.reader(csvfile, delimiter=',')
    for row in datareader:
        labels.append(row[0])

# make a label for empty values too
labels.append(u'')
le.fit(labels)

# now transform the array to have only numeric values instead
# of strings
X = le.transform(X)

# Lastly we need to split these into a optimization set
# using about 20% of the data
nrows = len(all_data)
percent = len(X)/5

# We'll use the first 20%. This is fine
# to do because we know the data is randomized.
X_opt = X[:percent, :]
y_opt = y[:percent]

# and a train/test set
X_rest = X[percent:, :]
y_rest = y[percent:]

dc = DummyClassifier(strategy='most_frequent', random_state=0)
gnb = GaussianNB()
# you could try other classifiers here
clf = tree.DecisionTreeClassifier(max_depth=5)

# make a set of folds
skf = cross_validation.StratifiedKFold(y_opt, 10)
gnb_acc_scores = []
dc_acc_scores = []
clf_acc_scores = []

# loop through the folds
for train, test in skf:
    # extract the train and test sets
    X_train, X_test = X_opt[train], X_opt[test]
    y_train, y_test = y_opt[train], y_opt[test]

    # train the classifiers
    dc = dc.fit(X_train, y_train)
    gnb = gnb.fit(X_train, y_train)
    clf = clf.fit(X_train, y_train)

    # test the classifiers
    dc_pred = dc.predict(X_test)
    gnb_pred = gnb.predict(X_test)
    clf_pred = clf.predict(X_test)

    # calculate metrics relating how well they did
    dc_accuracy = metrics.accuracy_score(y_test, dc_pred)
    dc_precision, dc_recall, dc_f, dc_support = metrics.precision_recall_fscore_support(y_test, dc_pred)
    gnb_accuracy = metrics.accuracy_score(y_test, gnb_pred)
    gnb_precision, gnb_recall, gnb_f, gnb_support = metrics.precision_recall_fscore_support(y_test, gnb_pred)
    clf_accuracy = metrics.accuracy_score(y_test, clf_pred)
    clf_precision, clf_recall, clf_f, clf_support = metrics.precision_recall_fscore_support(y_test, clf_pred)

    # print the results for this fold
    print "----- Accuracy -----"
    print "Dummy Cl: %.2f" %  dc_accuracy
    print "Naive Ba: %.2f" %  gnb_accuracy
    print "Decis Tr: %.2f" %  clf_accuracy
    print "----- F Score ------ "
    print "Dummy Cl: %s" % dc_f
    print "Naive Ba: %s" % gnb_f
    print "Decis Tr: %s" % clf_f
    print "Precision", "\t".join(list(Outcomes))
    print "Dummy Cl:", "\t".join("%.2f" % score for score in  dc_precision)
    print "Naive Ba:", "\t".join("%.2f" % score for score in  gnb_precision)
    print "Decis Tr:", "\t".join("%.2f" % score for score in  clf_precision)
    print "Recall   ", "\t".join(list(Outcomes))
    print "Dummy Cl:", "\t".join("%.2f" % score for score in  dc_recall)
    print "Naive Ba:", "\t".join("%.2f" % score for score in  gnb_recall)
    print "Decis Tr:", "\t".join("%.2f" % score for score in  clf_recall)

    dc_acc_scores = dc_acc_scores + [dc_accuracy]
    gnb_acc_scores = gnb_acc_scores + [gnb_accuracy]
    clf_acc_scores = clf_acc_scores + [clf_accuracy]

diff = np.mean(dc_acc_scores) - np.mean(gnb_acc_scores)
t, prob = scipy.stats.ttest_rel(dc_acc_scores, gnb_acc_scores)

diff_2 = np.mean(gnb_acc_scores) - np.mean(clf_acc_scores)

t2, prob2 = scipy.stats.ttest_rel(gnb_acc_scores, clf_acc_scores)
tree.export_graphviz(clf, out_file = 'tree.dot', feature_names=features)

print "============================================="
print " Results of optimization "
print "============================================="
print "Dummy Mean accuracy: ", np.mean(dc_acc_scores)
print "Naive Bayes Mean accuracy: ", np.mean(gnb_acc_scores)
print "Decision Tree Mean accuracy: ", np.mean(clf_acc_scores)

print "Accuracy for Dummy Classifier and Naive Bayes differ by {0}; p<{1}".format(diff, prob)
print "Accuracy for Naive Bayes Classifier and Decision Tree differ by {0}; " \
      "p<{" \
      "1}".format(diff_2, prob2)

print "These are good summary scores, but you may also want to"
print "Look at the details of what is going on inside this"
print "Possibly even without 10 fold cross validation"
print "And look at the confusion matrix and other details"
print "Of where mistakes are being made for developing insight"

print "============================================="
print " Final Results "
print "============================================="
print "When you have finished this assignment you should"
print "train a final classifier using the X_rest and y_rest"
print "using 10-fold cross validation"
print "And you should print out some sort of statistics on how it did"