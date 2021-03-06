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
numeric_features = ['age','balance','day','duration','campaign','pdays','previous']
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

#################################################################################
decTreeModel2 = RandomForestClassifier(criterion='entropy')
#Split the data: 60% training : 40% test set
instances_train, instances_test, target_train, target_test = cross_validation.train_test_split(train_dfs, targetLabels, test_size=0.4, random_state=0)
#fit the model using just the test set
decTreeModel2.fit(instances_train, target_train)