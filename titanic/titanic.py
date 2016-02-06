"""
-------------------------------------------------------------------
Title: Titanic.py
Description:This script's purpose is to train a predictive model
            leveraging the Titanic data from Kaggle.

Author: Nate Bitting
Email: nate.bitting@gmail.com
Twitter: @nbitting
LinkedIn: http://www.linkedin.com/in/natebitting/en
Kaggle: http://www.kaggle.com/users/294671/nate-bitting
-------------------------------------------------------------------
"""

# import the necessary modules used in this program
from sklearn import linear_model # logistic regression
from sklearn import svm # support vector machines
from sklearn.naive_bayes import GaussianNB # Naive Bayes
from sklearn import tree # decision tree
from sklearn.metrics import confusion_matrix  # evaluating predictive accuracy
from sklearn.metrics import accuracy_score  # proportion correctly predicted
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve # import ROC curve
from sklearn.metrics import auc # import AUC
from sklearn.cross_validation import cross_val_score  # cross-validation
from sklearn.cross_validation import train_test_split # used for creating training/test sets
import numpy as np # used for arrays
# import matplotlib.pyplot as plt # matplotlib
import pandas as pd  # pandas for data frame operations
# from sklearn.externals.six import StringIO
from sklearn.ensemble import RandomForestClassifier # random forest classifier
from sklearn.neighbors import KNeighborsClassifier # knn classfier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier

# create a data frame of the titanic training data set
train = pd.read_csv('train.csv')

test = pd.read_csv('test.csv')

#variables to store FPR and TPR of each model
fpr_master = []
tpr_master = []
model = []
aucs = []

# create mappings for sex into binary variable
convert_to_binary = {'male': 1, 'female': 0}
sex = np.array(train['Sex'].map(convert_to_binary))
sex_t = np.array(test['Sex'].map(convert_to_binary))

# create dummy variables for categorical attributes
dummies = ['Pclass', 'Embarked']
for val in dummies:
    dummies = pd.get_dummies(train[val], prefix=val)
    train = train.join(dummies)

dummies = ['Pclass', 'Embarked']
for val in dummies:
    dummies = pd.get_dummies(test[val], prefix=val)
    test = test.join(dummies)

# create new field for Cabins
train['cabin2'] = train['Cabin'].str[:1]
dummies = pd.get_dummies(train['cabin2'], prefix='Cabin')
train = train.join(dummies)

# create new field for Cabins
test['cabin2'] = test['Cabin'].str[:1]
dummies = pd.get_dummies(test['cabin2'], prefix='Cabin')
test = test.join(dummies)


# create dummy variables for Titles for training set
train['mr'] = np.where(train['Name'].str.contains('Mr.'), 1, 0)
train['mrs'] = np.where(train['Name'].str.contains('Mrs.'), 1, 0)
train['miss'] = np.where(train['Name'].str.contains('Miss.'), 1, 0)
train['master'] = np.where(train['Name'].str.contains('Master.'), 1, 0)
train['rev'] = np.where(train['Name'].str.contains('Rev.'), 1, 0)

mean_mr = np.mean(train[train['mr'] == 1]['Age'])
mean_mrs = np.mean(train[train['mrs'] == 1]['Age'])
mean_miss = np.mean(train[train['miss'] == 1]['Age'])
mean_master = np.mean(train[train['master'] == 1]['Age'])
mean_rev = np.mean(train[train['rev'] == 1]['Age'])

train.ix[(train['mr'] == 1) & (pd.isnull(train['Age'])), 'Age'] = mean_mr
train.ix[(train['mrs'] == 1) & (pd.isnull(train['Age'])), 'Age'] = mean_mrs
train.ix[(train['miss'] == 1) & (pd.isnull(train['Age'])), 'Age'] = mean_miss
train.ix[(train['master'] == 1) & (pd.isnull(train['Age']))]['Age'] = mean_master
train.ix[(train['rev'] == 1) & (pd.isnull(train['Age'])), 'Age'] = mean_rev
train.ix[pd.isnull(train['Age']), 'Age'] = train['Age'].mean()
train.ix[pd.isnull(train['SibSp']), 'SibSp'] = 0
train.ix[pd.isnull(train['Parch']), 'Parch'] = 0
train['family_size'] = train['Parch'] + train['SibSp']
train['log_fare'] = np.log(train.Fare)
train.ix[(train['log_fare'] == np.inf), 'log_fare'] = -1
train.ix[(train['log_fare'] == -np.inf), 'log_fare'] = -1
train['fare_per_person'] = train['Fare'] / train['family_size']
train.ix[(train['fare_per_person'] == np.inf), 'fare_per_person'] = train['Fare']
train['class_fare'] = train.Pclass * train.Fare


# assign predictor variables to numpy arrays for training model
pclass1 = np.array(train.Pclass_1)
pclass2 = np.array(train.Pclass_2)
sibsp = np.array(train.SibSp)
pclass_pow = np.array(train.Pclass**2)
age = np.array(train['Age'])
fare = np.array(train.Fare)
log_fare = np.array(train.log_fare)
class_fare = np.array(train.class_fare)
fare_per_person = np.array(train.fare_per_person)
family_size = np.array(train.family_size + 1)
mr = np.array(train.mr)
mrs = np.array(train.mrs)
miss = np.array(train.miss)
master = np.array(train.master)

# create dummy variables for Titles for training set
test['mr'] = np.where(test['Name'].str.contains('Mr.'), 1, 0)
test['mrs'] = np.where(test['Name'].str.contains('Mrs.'), 1, 0)
test['miss'] = np.where(test['Name'].str.contains('Miss.'), 1, 0)
test['master'] = np.where(test['Name'].str.contains('Master.'), 1, 0)
test['rev'] = np.where(test['Name'].str.contains('Rev.'), 1, 0)

mean_mr_t = np.mean(test[test['mr'] == 1]['Age'])
mean_mrs_t = np.mean(test[test['mrs'] == 1]['Age'])
mean_miss_t = np.mean(test[test['miss'] == 1]['Age'])
mean_master_t = np.mean(test[test['master'] == 1]['Age'])
mean_rev_t = np.mean(test[test['rev'] == 1]['Age'])

test.ix[(test['mr'] == 1) & (pd.isnull(test['Age'])), 'Age'] = mean_mr_t
test.ix[(test['mrs'] == 1) & (pd.isnull(test['Age'])), 'Age'] = mean_mrs_t
test.ix[(test['miss'] == 1) & (pd.isnull(test['Age'])), 'Age'] = mean_miss_t
test.ix[(test['master'] == 1) & (pd.isnull(test['Age'])), 'Age'] = mean_master_t
test.ix[(test['rev'] == 1) & (pd.isnull(test['Age'])), 'Age'] = mean_rev_t
test.ix[pd.isnull(test['Age']), 'Age'] = test['Age'].mean()
test.ix[pd.isnull(test['Fare']), 'Fare'] = test['Fare'].mean()
test.ix[pd.isnull(test['SibSp']), 'SibSp'] = 0
test.ix[pd.isnull(test['Parch']), 'Parch'] = 0
test['family_size'] = test['Parch'] + test['SibSp']
test['log_fare'] = np.log(test.Fare)
test.ix[(test['log_fare'] == np.inf), 'log_fare'] = -1
test.ix[(test['log_fare'] == -np.inf), 'log_fare'] = -1
test['fare_per_person'] = test['Fare'] / test['family_size']
test.ix[(test['fare_per_person'] == np.inf), 'fare_per_person'] = train.Fare
test['class_fare'] = test.Pclass * test.Fare


# assign predictor variables to numpy arrays for test data
pclass1_t = np.array(test.Pclass_1)
pclass2_t = np.array(test.Pclass_2)
pclass_pow_t = np.array(test.Pclass**2)
age_t = np.array(test.Age)
family_size_t = np.array(test.family_size + 1)
fare_t = np.array(test.Fare)
log_fare_t = np.array(test.log_fare)
class_fare_t = np.array(test.class_fare)
fare_per_person_t = np.array(test.fare_per_person)
id_t = np.array(test.PassengerId)
mr_t = np.array(test.mr)
mrs_t = np.array(test.mrs)
miss_t = np.array(test.miss)
master_t = np.array(test.master)


# create predictor and response variables
X = np.array([sex, age, pclass1, pclass2, log_fare, family_size, mr, mrs, miss, master, fare_per_person, class_fare, pclass_pow]).T
y = train['Survived']

x_test = np.array([sex_t, age_t, pclass1_t, pclass2_t, log_fare_t, family_size_t, mr_t, mrs_t, miss_t, master_t,
                   fare_per_person_t, class_fare_t, pclass_pow_t]).T

# # create your training and test data sets
#x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=9999)


# -------------------------------------------------------------------
# Create the Decision Tree Model
# -------------------------------------------------------------------

# # fit the linear svc model
# myTree = tree.DecisionTreeClassifier(random_state=9999, criterion='entropy', max_depth=8, min_samples_leaf=5)
# tree_model_fit = myTree.fit(X, y)

# # predicted class in training data only
# y_pred = tree_model_fit.predict(x_test)
# kaggle_array = np.array([id_t, y_pred]).T
# kaggle_df = pd.DataFrame(kaggle_array)
# kaggle_df.columns = ['PassengerId', 'Survived']
# kaggle_df.to_csv('submission.csv', index=False)


# -------------------------------------------------------------------
# Create the Random Forest Model
# -------------------------------------------------------------------

# fit the linear svc model
clf = RandomForestClassifier(n_estimators=1000, max_depth=5, min_samples_split=1, min_samples_leaf=1, max_features='auto',
                             bootstrap=False, oob_score=False, n_jobs=1, random_state=7777)
rf = clf.fit(X, y)

# predicted class in training data only
y_pred = rf.predict(x_test)
kaggle_array = np.array([id_t, y_pred]).T
kaggle_df = pd.DataFrame(kaggle_array)
kaggle_df.columns = ['PassengerId', 'Survived']
kaggle_df.to_csv('submission.csv', index=False)
# print 'Accuracy Score:', round(accuracy_score(y_test, y_pred), 6)
# cv_results_rf = cross_val_score(rf, x_train, y_train, cv=10)
# print 'CV Results:', round(cv_results_rf.mean(), 6), '\n'

# -------------------------------------------------------------------
# Create the Linear SVC Model
# -------------------------------------------------------------------

# # fit the linear svc model
# SVM = svm.SVC(probability=True)
# svm_model_fit = SVM.fit(X, y)
#
# y_pred = svm_model_fit.predict(x_test)
# kaggle_array = np.array([id_t, y_pred]).T
# kaggle_df = pd.DataFrame(kaggle_array)
# kaggle_df.columns = ['PassengerId', 'Survived']
# kaggle_df.to_csv('submission.csv', index=False)


# # how about multi-fold cross-validation with 5 folds
# cv_results_svc = cross_val_score(SVM, x_train, y_train, cv=5)
# print 'CV Results:',round(cv_results_svc.mean(), 3), '\n'  # cross-validation average accuracy
#
# # -------------------------------------------------------------------
# # Create the Logistic Regression Model
# # -------------------------------------------------------------------
#
# # fit a logistic regression model
# logreg = linear_model.LogisticRegression(C=1e5, class_weight='auto')
# log_model_fit = logreg.fit(x_train, y_train)
#
# # predicted class in training data only
# y_pred = log_model_fit.predict(x_test)
# print 'Logistic Regression Results:'
# print'Confusion Matrix:\n', confusion_matrix(y_test, y_pred)
# print 'Accuracy Score:', round(accuracy_score(y_test, y_pred), 3)
#
# # calculate the TPR/FPR for Logistic Regression
# y_test_prob = log_model_fit.predict_proba(x_test)
# fpr, tpr, thresholds = roc_curve(y_test, y_test_prob[:, 1])
# roc_auc = auc(fpr, tpr)
# print "AUC : %f" % roc_auc
# fpr_master.append(fpr)
# tpr_master.append(tpr)
# model.append('LR')
# aucs.append(roc_auc)
#
# # how about multi-fold cross-validation with 5 folds
# cv_results_log = cross_val_score(logreg, x_train, y_train, cv=5)
# print 'CV Results:',round(cv_results_log.mean(), 3), '\n'  # cross-validation average accuracy
#
# # -------------------------------------------------------------------
# # Create the Naive Bayes Model
# # -------------------------------------------------------------------
#
# # fit the linear svc model
# gnb = GaussianNB()
# gnb_model_fit = gnb.fit(x_train, y_train)
#
# # predicted class in training data only
# y_pred = gnb_model_fit.predict(x_test)
# print 'Gaussian Naive Bayes Results:'
# print 'Confusion Matrix:\n',confusion_matrix(y_test, y_pred)
# print 'Accuracy Score:',round(accuracy_score(y_test, y_pred), 3)
#
# # calculate the TPR/FPR for NB
# y_test_prob = gnb_model_fit.predict_proba(x_test)
# fpr, tpr, thresholds = roc_curve(y_test, y_test_prob[:, 1])
# roc_auc = auc(fpr, tpr)
# print "AUC : %f" % roc_auc
# fpr_master.append(fpr)
# tpr_master.append(tpr)
# model.append('NB')
# aucs.append(roc_auc)
#
# # how about multi-fold cross-validation with 5 folds
# cv_results_gnb = cross_val_score(gnb, x_train, y_train, cv=5)
# print 'CV Results:',round(cv_results_gnb.mean(),3)  # cross-validation average accuracy


# # -------------------------------------------------------------------
# # Create the Random Forest Model
# # -------------------------------------------------------------------
#
# # fit the RF model
# clf = RandomForestClassifier(n_estimators=100)
# rf = clf.fit(x_train, y_train)
#
# # predicted class in training data only
# y_pred = rf.predict(x_test)
# print '\nRandom Forest Results:'
# print 'Confusion Matrix:\n',confusion_matrix(y_test, y_pred)
# print 'Accuracy Score:',round(accuracy_score(y_test, y_pred), 3)
#
# # calculate the TPR/FPR for NB
# y_test_prob = rf.predict_proba(x_test)
# fpr, tpr, thresholds = roc_curve(y_test, y_test_prob[:, 1])
# roc_auc = auc(fpr, tpr)
# print "AUC : %f" % roc_auc
# fpr_master.append(fpr)
# tpr_master.append(tpr)
# model.append('RF')
# aucs.append(roc_auc)
#
# # how about multi-fold cross-validation with 5 folds
# cv_results_rf = cross_val_score(clf, x_train, y_train, cv=5)
# print 'CV Results:', round(cv_results_rf.mean(), 3)  # cross-validation average accuracy
#
#
# # -------------------------------------------------------------------
# # Create the K-Nearest Neighbors Model
# # -------------------------------------------------------------------
#
# # fit the KNN model
# knn = KNeighborsClassifier()
# knn_model = knn.fit(x_train, y_train)
#
# # predicted class in training data only
# y_pred = knn_model.predict(x_test)
# print '\nK-Nearest Neighbors Results:'
# print 'Confusion Matrix:\n', confusion_matrix(y_test, y_pred)
# print 'Accuracy Score:', round(accuracy_score(y_test, y_pred), 3)
#
# # calculate the TPR/FPR for NB
# y_test_prob = knn_model.predict_proba(x_test)
# fpr, tpr, thresholds = roc_curve(y_test, y_test_prob[:, 1])
# roc_auc = auc(fpr, tpr)
# print "AUC : %f" % roc_auc
# fpr_master.append(fpr)
# tpr_master.append(tpr)
# model.append('KNN')
# aucs.append(roc_auc)
#
# # how about multi-fold cross-validation with 5 folds
# cv_results_knn = cross_val_score(knn, x_train, y_train, cv=5)
# print 'CV Results:', round(cv_results_knn.mean(), 3)  # cross-validation average accuracy
#
#
# # -------------------------------------------------------------------
# # Create the Gradient Boosting Model
# # -------------------------------------------------------------------
#
# # fit the GridSearch model
# grad = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
# grad_model = grad.fit(x_train, y_train)
#
# # predicted class in training data only
# y_pred = grad_model.predict(x_test)
# print '\nGradient Boosting Results:'
# print 'Confusion Matrix:\n', confusion_matrix(y_test, y_pred)
# print 'Accuracy Score:', round(accuracy_score(y_test, y_pred), 3)
#
# # calculate the TPR/FPR for NB
# y_test_prob = grad_model.predict_proba(x_test)
# fpr, tpr, thresholds = roc_curve(y_test, y_test_prob[:, 1])
# roc_auc = auc(fpr, tpr)
# print "AUC : %f" % roc_auc
# fpr_master.append(fpr)
# tpr_master.append(tpr)
# model.append('GRAD')
# aucs.append(roc_auc)
#
# # how about multi-fold cross-validation with 5 folds
# cv_results_grad = cross_val_score(grad, x_train, y_train, cv=5)
# print 'CV Results:', round(cv_results_grad.mean(), 3)  # cross-validation average accuracy
#
#
# # -------------------------------------------------------------------
# # Create the Extremely Randomized Trees Model
# # -------------------------------------------------------------------
#
# # fit the RF model
# clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)
# extra = clf.fit(x_train, y_train)
#
# # predicted class in training data only
# y_pred = extra.predict(x_test)
# print '\nExtremely Randomized Trees Results:'
# print 'Confusion Matrix:\n',confusion_matrix(y_test, y_pred)
# print 'Accuracy Score:',round(accuracy_score(y_test, y_pred), 3)
#
# # calculate the TPR/FPR for NB
# y_test_prob = extra.predict_proba(x_test)
# fpr, tpr, thresholds = roc_curve(y_test, y_test_prob[:, 1])
# roc_auc = auc(fpr, tpr)
# print "AUC : %f" % roc_auc
# fpr_master.append(fpr)
# tpr_master.append(tpr)
# model.append('RF')
# aucs.append(roc_auc)
#
# # how about multi-fold cross-validation with 5 folds
# cv_results_extra = cross_val_score(clf, x_train, y_train, cv=5)
# print 'CV Results:', round(cv_results_extra.mean(), 3)  # cross-validation average accuracy