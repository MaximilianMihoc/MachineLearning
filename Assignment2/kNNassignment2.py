from pandas import DataFrame
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import pandas as pd
import csv
from sklearn import tree
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, precision_score
from sklearn import preprocessing

#get train data from file
# read feature names from the featurenames file and place them in a list.
featureNames = [line.rstrip() for line in open('./data/featurenames.txt', 'r')]
# remove empty elements from the list, if any
featureNames = [f for f in featureNames if f != '']
#print(featureNames)

Location = r'./data/trainingset.txt'
campaign_df = pd.read_csv(Location, names=featureNames)


####################################
# Extract Target Feature
###################################
# **sci-kit** expects that the descriptive features and target features 
# are passed to the model training functions as separate parameters. 
# so the first step in data preprocessins is to extract the 
# target feature values into a separate variable
targetLabels = campaign_df['target']
#print(targetLabels[0])

####################################
# Extract Numeric Descriptive Features
###################################
# We want to do some preprocessing on the categorical data so 
# We first extract the numeric_features into a separate data structure
#numeric_features = ['age','balance','day','duration','campaign','pdays','previous']
numeric_features = ['age','balance','day','campaign','pdays','previous']
numeric_dfs = campaign_df[numeric_features]
numeric_dfs.head()

####################################
# Extract Categorical Descriptive Features
###################################
cat_dfs = campaign_df.drop(numeric_features + ['target'] + ['id'] ,axis=1)

####################################
# Remove missing values and apply one-hot encoding
###################################
#handle missing values
#If the data has missing values, they will become NaNs in the Numpy arrays generated by the vectorizor so lets get rid of them
cat_dfs.replace('?','NA')
cat_dfs.fillna( 'NA', inplace = True )

#transpose into array of dictionaries (one dict per instance) of feature:level pairs
cat_dfs = cat_dfs.T.to_dict().values()
#convert to numeric encoding
vectorizer = DictVectorizer( sparse = False )
vec_cat_dfs = vectorizer.fit_transform(cat_dfs) 

encoding_dictionary = vectorizer.vocabulary_

########################################################
# Merge Categorical and Numeric Descriptive Features
########################################################
train_dfs = np.hstack((numeric_dfs.as_matrix(), vec_cat_dfs ))


#--------------------------------------------
# Hold-out Test Set + Confusion Matrix
#--------------------------------------------
print("-------------------------------------------------")
print("Accuracy and Confusion Matrix on Hold-out Testset")
print("-------------------------------------------------")

#define a decision tree model using entropy based information gain

train_dfs = preprocessing.normalize(train_dfs)

#Split the data: 60% training : 40% test set
instances_train, instances_test, target_train, target_test = cross_validation.train_test_split(train_dfs, targetLabels, test_size=0.4, random_state=0)

knn = KNeighborsClassifier(n_neighbors=5)

#k_range = range(1, 61)
#k_scores = []
#for k in k_range:
#    knn = KNeighborsClassifier(n_neighbors=k)
#    scores = cross_validation.cross_val_score(knn, instances_train, target_train, cv=10, scoring='accuracy')
#    k_scores.append(scores.mean())
#print (k_scores)

#fit the model using just the test set
knn.fit(instances_train, target_train)
#Use the model to make predictions for the test set queries
predictions = knn.predict(instances_test)
#Output the accuracy score of the model on the test set
print("Accuracy= " + str(accuracy_score(target_test, predictions, normalize=True)))
#Output the confusion matrix on the test set
confusionMatrix = confusion_matrix(target_test, predictions)
print(confusionMatrix)
print("\n\n")

target_names = ['type A', 'type B']
clsReport = classification_report(target_test, predictions, target_names= target_names)
print ("ClassificationReport:\n" + str(clsReport))

precisionScore = precision_score(target_test, predictions, average=None)

print("AverageClassAccuracy = " + str(sum(precisionScore) / float(len(precisionScore))))

#Draw the confusion matrix
import matplotlib.pyplot as plt
# matplotlib inline
# Show confusion matrix in a separate window
plt.matshow(confusionMatrix)
#plt.plot(confusionMatrix)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

#--------------------------------------------
# Cross-validation to Compare to Models
#--------------------------------------------
print("------------------------")
print("Cross-validation Results")
print("------------------------")

#run a 10 fold cross validation on this model using the full census data
scores=cross_validation.cross_val_score(knn, instances_train, target_train, cv=10)
#the cross validaton function returns an accuracy score for each fold
print("Entropy based Model:")
print("Score by fold: " + str(scores))
#we can output the mean accuracy score and standard deviation as follows:
print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("\n\n")

#for a comparison we will do the same experiment using a decision tree that uses the Gini impurity metric
decTreeModel3 = tree.DecisionTreeClassifier(criterion='gini')
scores=cross_validation.cross_val_score(decTreeModel3, instances_train, target_train, cv=10)
print("Gini based Model:")
print("Score by fold: " + str(scores))
print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))