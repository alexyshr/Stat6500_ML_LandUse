import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

import os

import time
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import LeaveOneOut as loocv

from plotly import tools
from plotly.offline import plot
import plotly.offline as py
from plotly.graph_objs import Scatter, Layout
import plotly.graph_objs as go
import plotly.figure_factory as ff

# load in training and test data
train = pd.read_csv('C:/Users/kenby/OneDrive/Desktop/Urbanlandcover/training.csv')
test = pd.read_csv('C:/Users/kenby/OneDrive/Desktop/Urbanlandcover/testing.csv')
print("Rows and Columns(Train): ",train.shape)
print("Rows and Columns(Test) : ",test.shape)

# check for missing values although it is clear there are none
train.isnull() .any() .any()

# duplicate function of pandas returns a duplicate row as true and others as false
sum(train.duplicated())

# basic statistical details
fig = train.describe().T
fig = fig.round(5)  # round to 5 decimal places
table = go.Table(
    columnwidth=[0.8]+[0.5]*8,
    header=dict(
        values=['Attribute'] + list(fig.columns),
        line = dict(color='darkslategray'),
        fill = dict(color='royalblue'),
    ),
    cells=dict(
        values=[fig.index] + [fig[k].tolist() for k in fig.columns[:]],
        line = dict(color='darkslategray'),
        fill = dict(color=['paleturquoise', 'white'])
    )
)
plot([table], filename='table-of-data')

# more general data exploration
print(train['class'].value_counts())

f,axes=plt.subplots(1,2,figsize=(20,8))
train['class'].value_counts().plot.pie(autopct='%1.1f%%',ax=axes[0])
axes[0].set_title('Visual for Distribution of Different Classes')
axes[0].set_ylabel('')
sns.countplot('class',data=train,ax=axes[1]) # sns.countplot is used like a histogram but for categorical data
axes[1].set_title('Visual for Distribution of Different Classes')
plt.show()

# Lets take a look at any outliers that could be potential issues
from collections import Counter
def examine_outliers(train_data, n, features):
    outlier_indicator = []
    for out in features:
        Q1 = np.percentile(train_data[out], 25)
        Q3 = np.percentile(train_data[out], 75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR # IQR method of dealing with  outliers, 1 of 2 methods
        outlier_list_out = train_data[
            (train_data[out] < Q1 - outlier_step) | (train_data[out] > Q3 + outlier_step)].index
        outlier_indicator.extend(outlier_list_out)

    outlier_indices = Counter(outlier_indicator)
    multiple_outliers = list(k for k, j in outlier_indices.items() if j > n)

    return multiple_outliers

# find outliers that should be removed
list_atributes = train.drop('class', axis=1).columns
outliers_to_remove = examine_outliers(train, 2, list_atributes)
train.loc[outliers_to_remove]

# lets look at mean values per row and column for train test data
# mean for rows
plt.figure(figsize=(10,5))
features = train.columns.values[1:148]
plt.title("The Distribution of Mean Values Per Row for Train and Test Sets",fontsize=10)
sns.distplot(train[features].mean(axis=1),color="purple", kde=True,bins=50, label='train') # kde is kernel density estimation
sns.distplot(test[features].mean(axis=1),color="green", kde=True,bins=50, label='test')
plt.legend()
plt.show()

# mean for columns
plt.figure(figsize=(16,6))
plt.title("The Distribution of Mean Values Per Column for Train and Test Sets",fontsize=10)
sns.distplot(train[features].mean(axis=0),color="red",kde=True,bins=50, label='train')
sns.distplot(test[features].mean(axis=0),color="blue", kde=True,bins=50, label='test')
plt.legend()
plt.show()

# lets examine correlations between features
# first the categorical variable 'class' needs to be changed to numerical variable so...
group_map = {"grass ":0,"building ":1,'concrete ':2,'tree ':3,'shadow ':4,'pool ':5,'asphalt ':6,'soil ':7,'car ':8}

train['class'] = train['class'].map(group_map)
test['class'] = test['class'].map(group_map)
train['class'].unique()

# now lets look at the correlation bewteen a few variables
sns.pairplot(train, vars=['class', 'BrdIndx','Area','Round','Bright','Compact'], hue='class', palette='deep')
plt.show()

# correlation of features with target
corr = train.corr().abs().unstack().sort_values(kind="quicksort").reset_index()
corr = corr[corr['level_0'] != corr['level_1']]
corr.head()

correlations = corr.loc[corr[0] == 1]
features_to_be_removed = set(list(correlations['level_1']))
correlations.shape

# prepare to try different classification algorithms
X_train = train.drop(['class'], axis=1)
y_train = pd.DataFrame(train['class'].values)
X_test = test.drop(['class'], axis=1)
y_test = test['class']
scaler = StandardScaler() #standardize data values into standard format
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# classification algorithms
classification_choice = [KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier(), GaussianNB(),]
accuracy = {}
accuracy_std = {}
for choice in classification_choice:
    choice.fit(X_train, y_train)
    pred = choice.predict(X_test)
    accuracy[str((str(choice).split('(')[0]))] = accuracy_score(pred, y_test)

for choice in classification_choice:
    choice.fit(X_train_std, y_train)
    prediction = choice.predict(X_test_std)
    accuracy_std[str((str(choice).split('(')[0]))] = accuracy_score(prediction, y_test)

data = accuracy.values()
labels = accuracy.keys()
data_std = accuracy_std.values()
labels_std = accuracy_std.keys()

# plot accuracy for visualization
fig = plt.figure(figsize=(20,5))
plt.subplot(121)
plt.plot([i for i, e in enumerate(data)], data); plt.xticks([i for i, e in enumerate(labels)], [l[:] for l in labels])
plt.title("Accuracy Without Preprocessing",fontsize = 15)
plt.xlabel('Model',fontsize = 12)
plt.xticks(rotation = 75)
plt.ylabel('Accuracy',fontsize = 12)

plt.subplot(122)
plt.plot([i for i, e in enumerate(data_std)], data_std); plt.xticks([i for i, e in enumerate(labels_std)], [l[:] for l in labels_std])
plt.title("Accuracy After Transformation",fontsize = 15)
plt.xlabel('Model',fontsize = 12)
plt.xticks(rotation =75)
plt.show()
# above results show that random forest classifier seems to perform the best

# now lets perform cross validation
n_fold = 5
folds = KFold(n_splits=n_fold, shuffle=True, random_state=1)

# Random Forest Classifier
prediction = np.zeros(len(X_test))
complete_acc = []
out_of_fold = np.zeros(len(X_train))
for fold_n, (train_index, valid_index) in enumerate(folds.split(X_train, y_train)):
    print('Fold', fold_n, 'started at', time.ctime(), end="  ")
    X_train_, X_valid = X_train.iloc[train_index], X_train.iloc[valid_index] #iloc function used to retrieve rows from a data set
    y_train_, y_valid = y_train.iloc[train_index], y_train.iloc[valid_index]

    classifier_randomforest = RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=0) # default estimator, -1 is using all processors, 0 fixes sequence
    classifier_randomforest.fit(X_train_, y_train_)
    out_of_fold[valid_index] = classifier_randomforest.predict(X_train.iloc[valid_index])

    prediction = classifier_randomforest.predict(X_test)
    print("Validation Score: ", accuracy_score(y_test, prediction))
    complete_acc.append(accuracy_score(y_test, prediction))
print("CV score".format(accuracy_score(y_train, out_of_fold)))
print("Mean Testing Score: ", np.mean(complete_acc))

# will take a look at hyperparameters for random forest and decision tree later in assignment
#PCA analysis
scaler = StandardScaler()
scaled_training = scaler.fit_transform(X_train)
PCA_xtrain = PCA().fit_transform(scaled_training)
plt.scatter(PCA_xtrain[:, 0], PCA_xtrain[:, 1], c=train['class'], cmap="viridis")
plt.axis('off')
plt.colorbar()
plt.show()






