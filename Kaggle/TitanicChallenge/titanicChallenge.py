import imp
from sklearn import preprocessing
import pandas as pd

# get train data from file
# read feature names from the featurenames file and place them in a list.
featureNames = [line.rstrip() for line in open('featurenames.txt', 'r')]
# remove empty elements from the list, if any
featureNames = [f for f in featureNames if f != '']
# print(featureNames)

Location = r'train.csv'
campaign_df = pd.read_csv(Location, names=featureNames)

targetLabels = campaign_df['Survived']
# print(targetLabels[0])

####################################
# Extract Numeric Descriptive Features
###################################
# We want to do some preprocessing on the categorical data so
# We first extract the numeric_features into a separate data structure
numeric_features = ['Survived','Pclass','Age','SibSp','Parch','Fare']
numeric_dfs = campaign_df[numeric_features]
numeric_dfs.head()

