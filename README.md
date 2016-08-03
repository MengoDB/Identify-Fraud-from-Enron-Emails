Enron Email project
==============


### Overview

This project is to find out people of fraud interest from Enron Email and financial data. I used several machine learning models to identify people of interest and compared with the people who has committed fraud, improving the accuracy and capibility of the prediction models. Throught this project, I got to understand the process of investigating a topic by exploring data and building machine learning models.


### Data Exploration

## Number of data points

146

## Features

The dataset includes two categories of feature, namely financial features and email features.

Financial features:  'salary', 'deferral\_payments', 'total\_payments', 'loan\_advances', 'bonus',
                       'restricted\_stock\_deferred', 'deferred\_income', 'total\_stock\_value', 'expenses',
                       'exercised\_stock\_options', 'other', 'long\_term\_incentive', 'restricted\_stock', 'director\_fees'.

Email features: 'to\_messages', 'from\_poi\_to\_this\_person', 'from\_messages', 'from\_this\_person\_to\_poi',
                       'shared\_receipt\_with\_poi', 'fraction\_to\_poi', 'fraction\_from\_poi'.

Among these features, 'fraction\_to\_poi' and 'fraction\_from\_poi' are generated from existed email related variables. 'fraction\_to\_poi' equals 'from\_this\_person\_to\_poi' divided by 'from\_messages'. 'fraction\_from\_poi' equals 'from\_poi\_to\_this\_person' divided by 'to\_messages'. These added feature could interpret the relationship between persons and poi better than existed variables. Otherwise, people with small total number of mails will not show up through our model. So I added these two new features and expected them to represent email features.

## POI

POI is the lable in the dataset represents people who has committed fraud. There are 18 data points with POI=1.

## Outliers

As I checked the dataset, I removed two outliers.

One is 'TOTAL'. Because this data point is the total number of all the other data points. The other one is 'THE TRAVEL AGENCY IN THE PARK'. I removed it because this is a data point about an agency while other data points are for investigated people.

## Missing values

With removing outliers and add features, I count the missing values for each feature. The result is as the follows.

'bonus': 63,
'deferral\_payments': 106,
'deferred\_income': 96,
'director\_fees': 128,
'exercised\_stock\_options': 43,
'expenses': 50,
'fraction\_from\_poi': 0,
'fraction\_to\_poi': 0,
'from\_messages': 58,
'from\_poi\_to\_this\_person': 58,
'from\_this\_person\_to\_poi': 58,
'loan\_advances': 141,
'long\_term\_incentive': 79,
'other': 53,
'poi': 0,
'restricted\_stock': 35,
'restricted\_stock\_deferred': 127,
'salary': 50,
'shared\_receipt\_with\_poi': 58,
'to\_messages': 58,
'total\_payments': 21,
'total\_stock\_value': 19

### Feature Selection


In order to select features, I used Decision Tree as a sample model for multiple iterations. 

DecisionTreeClassifier(compute\_importances=None, criterion='gini',
            max\_depth=None, max\_features=None, max\_leaf\_nodes=None,
            min\_density=None, min\_samples\_leaf=1, min\_samples\_split=2,
            random\_state=None, splitter='best')
  Accuracy: 0.81827 Precision: 0.31346  Recall: 0.30500 F1: 0.30917 F2: 0.30666
  Total predictions: 15000  True positives:  610  False positives: 1336 False negatives: 1390 True negatives: 11664

salary :  0.0
deferral\_payments :  0.0
total\_payments :  0.0833333333333
loan\_advances :  0.0
bonus :  0.0
restricted\_stock\_deferred :  0.0
deferred\_income :  0.0
total\_stock\_value :  0.0907738095238
expenses :  0.187971552257
exercised\_stock\_options :  0.296613945578
other :  0.0
long\_term\_incentive :  0.0
restricted\_stock :  0.0331909937888
director\_fees :  0.0
to\_messages :  0.0
from\_poi\_to\_this\_person :  0.047619047619
from\_messages :  0.0860119047619
from\_this\_person\_to\_poi :  0.0
shared\_receipt\_with\_poi :  0.047619047619
fraction\_to\_poi :  0.126866365519
fraction\_from\_poi :  0.0

For first iteration above, we input all features into the model. 9 features in the model are important (importance > 0).  They are: 'total\_payments', 'total\_stock\_value', 'expenses', 'exercised\_stock\_options', 'restricted\_stock', 'from\_poi\_to\_this\_person', 'from\_messages','shared\_receipt\_with\_poi','fraction\_to\_poi'.



DecisionTreeClassifier(compute\_importances=None, criterion='gini',
            max\_depth=None, max\_features=None, max\_leaf\_nodes=None,
            min\_density=None, min\_samples\_leaf=1, min\_samples\_split=2,
            random\_state=None, splitter='best')
  Accuracy: 0.83173 Precision: 0.35131  Recall: 0.30950 F1: 0.32908 F2: 0.31705
  Total predictions: 15000  True positives:  619  False positives: 1143 False negatives: 1381 True negatives: 11857

total\_payments :  0.0833333333333
total\_stock\_value :  0.0302197802198
expenses :  0.123685837972
exercised\_stock\_options :  0.357167974882
restricted\_stock :  0.0331909937888
from\_poi\_to\_this\_person :  0.0
from\_messages :  0.0860119047619
shared\_receipt\_with\_poi :  0.111904761905
fraction\_to\_poi :  0.174485413138


Using 9 important features into the Decision Tree model, second iteration is shown above. Except the feature 'from\_poi\_to\_this\_person' of which importance equals to 0, other features are important.


DecisionTreeClassifier(compute\_importances=None, criterion='gini',
            max\_depth=None, max\_features=None, max\_leaf\_nodes=None,
            min\_density=None, min\_samples\_leaf=1, min\_samples\_split=2,
            random\_state=None, splitter='best')
  Accuracy: 0.82607 Precision: 0.31886  Recall: 0.26800 F1: 0.29123 F2: 0.27683
  Total predictions: 15000  True positives:  536  False positives: 1145 False negatives: 1464 True negatives: 11855

total\_payments :  0.047619047619
total\_stock\_value :  0.0302197802198
expenses :  0.159400123686
exercised\_stock\_options :  0.357167974882
restricted\_stock :  0.0808100414079
from\_messages :  0.0860119047619
shared\_receipt\_with\_poi :  0.111904761905
fraction\_to\_poi :  0.126866365519


With 8 important features from the second iteration, the third iteration above run the decision tree model again, proving all the eight features selected are important.

As a result, I selected 8 features to build my machine learning models. They are: 'total\_payments', 'total\_stock\_value', 'expenses', 'exercised\_stock\_options', 'restricted\_stock', 'shared\_receipt\_with\_poi', 'from\_messages','fraction\_to\_poi'.


## Algorithm

### Decision Tree

```python
from sklearn import tree
clf = tree.DecisionTreeClassifier()
```

DecisionTreeClassifier(compute\_importances=None, criterion='gini',
            max\_depth=None, max\_features=None, max\_leaf\_nodes=None,
            min\_density=None, min\_samples\_leaf=1, min\_samples\_split=2,
            random\_state=None, splitter='best')
  Accuracy: 0.82607 Precision: 0.31886  Recall: 0.26800 F1: 0.29123 F2: 0.27683
  Total predictions: 15000  True positives:  536  False positives: 1145 False negatives: 1464 True negatives: 11855

total\_payments :  0.047619047619
total\_stock\_value :  0.0302197802198
expenses :  0.159400123686
exercised\_stock\_options :  0.357167974882
restricted\_stock :  0.0808100414079
from\_messages :  0.0860119047619
shared\_receipt\_with\_poi :  0.111904761905
fraction\_to\_poi :  0.126866365519

### Gaussian Naive Bayes

```python
from sklearn.naive\_bayes import GaussianNB
clf = GaussianNB()
```

GaussianNB()
  Accuracy: 0.84807 Precision: 0.38867  Recall: 0.24350 F1: 0.29942 F2: 0.26316
  Total predictions: 15000  True positives:  487  False positives:  766 False negatives: 1513 True negatives: 12234

### K-Nearest Neighbors

```python
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

knn = KNeighborsClassifier()
estimators = [('scale', StandardScaler()), ('knn', knn)]
clf = Pipeline(estimators)
```
Pipeline(steps=[('scale', StandardScaler(copy=True, with\_mean=True, with\_std=True)), ('knn', KNeighborsClassifier(algorithm='auto', leaf\_size=30, metric='minkowski',
           metric\_params=None, n\_neighbors=5, p=2, weights='uniform'))])
  Accuracy: 0.86280 Precision: 0.43318  Recall: 0.09400 F1: 0.15448 F2: 0.11145
  Total predictions: 15000  True positives:  188  False positives:  246 False negatives: 1812 True negatives: 12754

### Random Forest

```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
```
RandomForestClassifier(bootstrap=True, compute\_importances=None,
            criterion='gini', max\_depth=None, max\_features='auto',
            max\_leaf\_nodes=None, min\_density=None, min\_samples\_leaf=1,
            min\_samples\_split=2, n\_estimators=10, n\_jobs=1,
            oob\_score=False, random\_state=None, verbose=0)
  Accuracy: 0.85427 Precision: 0.35514  Recall: 0.11400 F1: 0.17260 F2: 0.13191
  Total predictions: 15000  True positives:  228  False positives:  414 False negatives: 1772 True negatives: 12586

total\_payments :  0.132165501822
total\_stock\_value :  0.0842622479555
expenses :  0.12935687862
exercised\_stock\_options :  0.191842369768
restricted\_stock :  0.11368767859
from\_messages :  0.0333290762057
shared\_receipt\_with\_poi :  0.162363356029
fraction\_to\_poi :  0.15299289101


## Algorithm Tuning

I used grid search to automatically tune the KNN model and Decision Tree model.

### Decision Tree Tuning

```python
from sklearn.grid\_search import GridSearchCV
from sklearn import tree

tree\_clf = tree.DecisionTreeClassifier()
parameters = {'criterion': ('gini', 'entropy'),
              'splitter': ('best', 'random')}
clf = GridSearchCV(tree\_clf, parameters, scoring='recall')
```



GridSearchCV(cv=None,
       estimator=DecisionTreeClassifier(compute\_importances=None, criterion='gini',
            max\_depth=None, max\_features=None, max\_leaf\_nodes=None,
            min\_density=None, min\_samples\_leaf=1, min\_samples\_split=2,
            random\_state=None, splitter='best'),
       fit\_params={}, iid=True, loss\_func=None, n\_jobs=1,
       param\_grid={'splitter': ('best', 'random'), 'criterion': ('gini', 'entropy')},
       pre\_dispatch='2*n\_jobs', refit=True, score\_func=None,
       scoring='recall', verbose=0)
  Accuracy: 0.82360 Precision: 0.32709  Recall: 0.30550 F1: 0.31593 F2: 0.30959
  Total predictions: 15000  True positives:  611  False positives: 1257 False negatives: 1389 True negatives: 11743

#### best\_params
{'splitter': 'random', 'criterion': 'entropy'}
DecisionTreeClassifier(compute\_importances=None, criterion='entropy',
            max\_depth=None, max\_features=None, max\_leaf\_nodes=None,
            min\_density=None, min\_samples\_leaf=1, min\_samples\_split=2,
            random\_state=None, splitter='random')
  Accuracy: 0.82680 Precision: 0.34667  Recall: 0.33800 F1: 0.34228 F2: 0.33970
  Total predictions: 15000  True positives:  676  False positives: 1274 False negatives: 1324 True negatives: 11726

total\_payments :  0.0973186272037
total\_stock\_value :  0.0625
expenses :  0.111317475172
exercised\_stock\_options :  0.18530624806
restricted\_stock :  0.0808444264639
from\_messages :  0.0363532823221
shared\_receipt\_with\_poi :  0.13678033831
fraction\_to\_poi :  0.289579602468

This part is to tune original Decision Tree model on the parameters of criterion and splitter. From this tuning process, the model is optimized with splitter as 'random' and criterion as 'entropy'. The Recall of this model is improved to 0.338 and Precision to 0.347.



### KNN Tuning

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.grid\_search import GridSearchCV

knn = KNeighborsClassifier()

estimators = [('scale', StandardScaler()), ('knn', knn)]
pipeline = Pipeline(estimators)
parameters = {'knn\_\_n\_neighbors': [1, 8],
              'knn\_\_algorithm': ('ball\_tree', 'kd\_tree', 'brute', 'auto')}
clf = GridSearchCV(pipeline, parameters, scoring='recall')

```

GridSearchCV(cv=None,
       estimator=Pipeline(steps=[('scale', StandardScaler(copy=True, with\_mean=True, with\_std=True)), ('knn', KNeighborsClassifier(algorithm='auto', leaf\_size=30, metric='minkowski',
           metric\_params=None, n\_neighbors=5, p=2, weights='uniform'))]),
       fit\_params={}, iid=True, loss\_func=None, n\_jobs=1,
       param\_grid={'knn\_\_algorithm': ('ball\_tree', 'kd\_tree', 'brute', 'auto'), 'knn\_\_n\_neighbors': [1, 8]},
       pre\_dispatch='2*n\_jobs', refit=True, score\_func=None,
       scoring='recall', verbose=0)
  Accuracy: 0.85347 Precision: 0.43210  Recall: 0.31500 F1: 0.36437 F2: 0.33305
  Total predictions: 15000  True positives:  630  False positives:  828 False negatives: 1370 True negatives: 12172

#### best\_params
{'knn\_\_algorithm': 'ball\_tree', 'knn\_\_n\_neighbors': 1}
Pipeline(steps=[('scale', StandardScaler(copy=True, with\_mean=True, with\_std=True)), ('knn', KNeighborsClassifier(algorithm='ball\_tree', leaf\_size=30, metric='minkowski',
           metric\_params=None, n\_neighbors=1, p=2, weights='uniform'))])
  Accuracy: 0.85347 Precision: 0.43210  Recall: 0.31500 F1: 0.36437 F2: 0.33305
  Total predictions: 15000  True positives:  630  False positives:  828 False negatives: 1370 True negatives: 12172


This part is to tune original KNN model on the parameters of knn\_algorithm and knn\_n\_neighbors. From this tuning process, the model is optimized with knn\_algorithm as 'ball\_tree' and knn\_n\_neighbors as 1. The Recall of this model is improved to 0.315 and Precision to 0.432.


## Result


Algorithm | Precision | recall
--- | --- | ---
Decision Tree (Tuned) | 0.347 | 0.338
KNN (Tuned) | 0.432 | 0.315
Gaussian Naive Bayes | 0.389 | 0.244
Random Forest | 0.355 | 0.114

From this result, I select Decision Tree as the final algorithm considering performance both on precision and recall.

The specific algorithm shows below.



DecisionTreeClassifier(compute\_importances=None, criterion='entropy',
            max\_depth=None, max\_features=None, max\_leaf\_nodes=None,
            min\_density=None, min\_samples\_leaf=1, min\_samples\_split=2,
            random\_state=None, splitter='random')
  Accuracy: 0.82680 Precision: 0.34667  Recall: 0.33800 F1: 0.34228 F2: 0.33970
  Total predictions: 15000  True positives:  676  False positives: 1274 False negatives: 1324 True negatives: 11726

total\_payments :  0.0973186272037
total\_stock\_value :  0.0625
expenses :  0.111317475172
exercised\_stock\_options :  0.18530624806
restricted\_stock :  0.0808444264639
from\_messages :  0.0363532823221
shared\_receipt\_with\_poi :  0.13678033831
fraction\_to\_poi :  0.289579602468



## Validation

For the chosen algorithm, we need to validate it to see how well the algorithm generalizes beyond the training dataset. A classic mistake we might make is to use same dataset for training and testing.

The whole dataset we have includes only 146 data points, which is very small. So I chose stratified shuffle split cross validation to validate the selected algorithm.

## Evaluation

### Precision

0.338

Precision, referred as positive predictive value, here indicated that 34% of people who are predicted as poi are truly people of interests.

### Recall

0.347

Recall, referred as true positive rate or sensitivity, here indicated that among people of interests, 35% of them are correctly predicted via our final algorithm.

















