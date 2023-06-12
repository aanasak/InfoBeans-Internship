# %%
# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

# %%
# reading csv and making a copy to alter
ds = pd.read_csv('IRIS.csv')
df = ds.copy()

# %%
# looking at the dataframe
df.head()
df.shape
df.info()
df.describe()
df.hist(figsize=(20, 15), bins=50)

# %%
# replacing species with ints
df['species'] = df['species'].map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})
# df

#  %%
# looking at dtypes again
# df.info()

# %%
# X and Y vars for train_test_split
X = df.drop(['species'], axis=1)
Y = df['species']
# %%
# splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#  %%
y_train_0 = (y_train == 0)
y_test_0 = (y_test == 0)

# %%
# predicting using gaussian NB
gnb = GaussianNB()
gnb.fit(x_train, y_train_0)
gnb_ans = gnb.predict(x_test)

# %%
# gaussian NB accuracy score
gnb_score = accuracy_score(y_test_0, gnb_ans)
gnb_score


# %%
# gaussian NB precision score (actually correct)
gnb_prec = precision_score(y_test_0, gnb_ans)
gnb_prec

# %%
# gaussian NB recall score (identified correctly)
gnb_rec = recall_score(y_test_0, gnb_ans)
gnb_rec

# %%
# gaussian NB f1 score
gnb_f1 = f1_score(y_test_0, gnb_ans)
gnb_f1

#  %%
# gaussian NB precision recall curve
y_scores = cross_val_predict(gnb, x_train, y_train_0, cv=3)
precisions, recalls, thresholds = precision_recall_curve(y_train_0, y_scores)

# %%
# plotting curve with matplotlib
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel('threshold')
    plt.grid()
    plt.legend()
    
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

# %%
# gaussian NB roc curve
fpr, tpr, thresholds = roc_curve(y_train_0, y_scores)

# %%
# plotting roc curve
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label) 
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False +ve rate')
    plt.ylabel('True +ve rate (recall)')
    plt.grid()
    plt.legend()
    
plot_roc_curve(fpr, tpr)

# %%
# looking at roc auc score
roc_auc_score(y_train_0, y_scores)

# %%
# predicting using decision tree
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
dt_ans = dt.predict(x_test)

# %%
# decision tree accuracy score
dt_score = accuracy_score(y_test, dt_ans)
dt_score
# %%
