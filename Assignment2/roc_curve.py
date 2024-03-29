# -------------------------------------------------------------------------
# AUTHOR: Akshith Madugula
# FILENAME: roc_curve.py
# SPECIFICATION: plot roc for decision tree.
# FOR: CS 5990 (Advanced Data Mining) - Assignment #2
# TIME SPENT: 20mins
# -----------------------------------------------------------*/

#importing some Python libraries
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import numpy as np
import pandas as pd

# read the dataset cheat_data.csv and prepare the data_training numpy array
# --> add your Python code here
# data_training = ?
path = r"Assignment2\cheat_data.csv"
df = pd.read_csv(path, sep=',', header=0)   #reading a dataset eliminating the header (Pandas library)
data_training = np.array(df.values) #creating a training matrix without the id (NumPy library)

# transform the original training features to numbers and add them to the 5D array X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
# Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [0, 0, 1, 0, 100], ...]]. The feature Marital Status must be one-hot-encoded and Taxable Income must
# be converted to a float.
# --> add your Python code here
X = []
marital_encoding = {"Single": [1, 0, 0], "Married": [0, 1, 0], "Divorced": [0, 0, 1]}
for row in data_training:
  new_row = []
  for ind, val in enumerate(row):
    if ind == 0:  # Refund, converting Yes to 1 and No to 0
      new_row.append(1 if val == "Yes" else 0)
    elif ind == 1:  # Marital Status, one hot encoding-> ['Single', 'Divorced', 'Married']
      new_row.extend(marital_encoding[val])
    elif ind == 2:  # Taxable Income
      new_row.append(float(val.replace("k", "")))  # Convert to float
  X.append(new_row) 

# transform the original training classes to numbers and add them to the vector y. For instance Yes = 1, No = 0, so Y = [1, 1, 0, 0, ...]
# --> add your Python code here
y = [1 if val == "Yes" else 0 for val in data_training[:, 3]]

# split into train/test sets using 30% for test
# --> add your Python code here
trainX, testX, trainy, testy = train_test_split(X, y, test_size = 0.3)

# generate a no skill prediction (random classifier - scores should be all zero)
# --> add your Python code here
ns_probs = [0 for _ in range(len(testy))]  # All zeros for no-skill

# fit a decision tree model by using entropy with max depth = 2
clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=2)
clf = clf.fit(trainX, trainy)

# predict probabilities for all test samples (scores)
dt_probs = clf.predict_proba(testX)

# keep probabilities for the positive outcome only
# --> add your Python code here
dt_probs = dt_probs[:, 1]

# calculate scores by using both classifiers (no skilled and decision tree)
ns_auc = roc_auc_score(testy, ns_probs)
dt_auc = roc_auc_score(testy, dt_probs)

# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Decision Tree: ROC AUC=%.3f' % (dt_auc))

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
dt_fpr, dt_tpr, _ = roc_curve(testy, dt_probs)

# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(dt_fpr, dt_tpr, marker='.', label='Decision Tree')

# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')

# show the legend
pyplot.legend()

# show the plot
pyplot.show()