#---------------------------------------
#Student 1: Maximilian Mihoc, C12728559
#Student 2: Tomas Higgins, C12733411 
#---------------------------------------

import numpy as np
import pandas as pd
from pandas import DataFrame

from sklearn import tree, model_selection, preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.kernel_approximation import RBFSampler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier, VotingClassifier

def main():
	#get train data from file
	# read feature names from the featurenames file and place them in a list.
	featureNames = [line.rstrip() for line in open('./data/featurenames.txt', 'r')]
	# remove empty elements from the list, if any
	featureNames = [f for f in featureNames if f != '']
	#print(featureNames)

	Location = r'./data/trainingset.txt'
	campaign_df = pd.read_csv(Location, names=featureNames)

	#extract train target feature labels
	targetLabels = campaign_df['target']
	train_dfs = preprocessAndCleanData(campaign_df)

	#classifier = RandomForestClassifier(criterion='entropy')
	#classifier = QuadraticDiscriminantAnalysis()
	#classifier = LogisticRegression(random_state=1)
	#classifier = tree.DecisionTreeClassifier(criterion='entropy')
	#classifier = AdaBoostClassifier()
	#classifier = GaussianNB()
	#classifier = GradientBoostingClassifier()
	#classifier = BaggingClassifier()

	clf1 = LogisticRegression(random_state=1)
	clf2 = RandomForestClassifier(random_state=1)
	clf3 = GradientBoostingClassifier()
	clf4 = AdaBoostClassifier()
	classifier = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3) , ('abc', clf4)], voting='hard')

	# Evaluation
	#-------------------------------------------------------------------------------------------------
	#Split the data: 60% training : 40% test set
	instances_train, instances_test, target_train, target_test = cross_validation.train_test_split(train_dfs, targetLabels, test_size=0.4, random_state=0)

	#fit the model using just the test set
	classifier.fit(instances_train, target_train)

	#Use the model to make predictions for the test set queries
	predictions = classifier.predict(instances_test)

	#Output the accuracy score of the model on the test set
	print("Accuracy= " + str(accuracy_score(target_test, predictions, normalize=True)))

	#Output the confusion matrix on the test set
	confusionMatrix = confusion_matrix(target_test, predictions)
	print(confusionMatrix)
	print("\n\n")

	target_names = ['Type A', 'Type B']
	clsReport = classification_report(target_test, predictions, target_names= target_names)
	print ("ClassificationReport:\n" + str(clsReport))

	precisionScore = precision_score(target_test, predictions, average=None)
	print("AverageClassAccuracy = " + str(sum(precisionScore) / float(len(precisionScore))))
	print("HarmonicMean = " + str( 2 / (1/precisionScore[0] + 1/precisionScore[1]) ))

	#--------------------------------------------
	# Cross-validation to Compare to Models
	#--------------------------------------------
	print("------------------------")
	print("Cross-validation Results")
	print("------------------------")

	#run a 10 fold cross validation on this model using the full census data
	scores=cross_validation.cross_val_score(classifier, instances_train, target_train, cv=10)
	#the cross validaton function returns an accuracy score for each fold
	print("Entropy based Model:")
	print("Score by fold: " + str(scores))
	#we can output the mean accuracy score and standard deviation as follows:
	print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

	#-------------------------------------------------------------------------------------------------

	#--------------------------------------------
	# Predictions
	#--------------------------------------------

	# fit the classifier for all training data
	classifier.fit(train_dfs, targetLabels)

	# get the queries data from file
	Location = r'./data/queries.txt'
	tempQueries_df = pd.read_csv(Location, names=featureNames)

	#get queries IDs for later usage
	queryIds = tempQueries_df['id']

	#preprocess and clean data for queries
	queries_df = preprocessAndCleanData(tempQueries_df)
	predictions = classifier.predict(queries_df)

	#-------------------------------------------
	# Print predictions to file
	#-------------------------------------------

	# open file to write Predictions
	predictionFile = open('./solutions/C12728559+C12733411.txt', 'w')

	for qId, prediction in zip(queryIds, predictions):
		predictionFile.write(qId + "," + prediction + "\n")

# function to preprocess and clean train and query data before usage
def preprocessAndCleanData(dataFrame):
    #extract the numeric_features into a separate data structure
	numeric_features = ['age','day','duration','balance','campaign','pdays','previous']
	numeric_dfs = dataFrame[numeric_features]
	#print (numeric_dfs)

	#normalize numeric features of the dataframe
	#numeric_dfs = dataFrame[numeric_features].apply( lambda x: (x - x.min()) / (x.max() - x.min()) )
	#print (numeric_dfs)

	# extract categorical features 
	# Remove id instances from training dataset
	cat_dfs = dataFrame.drop(numeric_features + ['target'] + ['id'], axis=1)
	#print (cat_dfs)

	# remove missing values if they exist.
	cat_dfs.replace('?','NA')
	cat_dfs.fillna( 'NA', inplace = True )

	#transpose into array of dictionaries (one dict per instance) of feature:level pairs
	cat_dfs = cat_dfs.T.to_dict().values()

	#Convert to numeric encoding using one-hot-encoding
	vectorizer = DictVectorizer( sparse = False )
	vec_cat_dfs = vectorizer.fit_transform(cat_dfs) 
	encoding_dictionary = vectorizer.vocabulary_

	# Merge Categorical and Numeric Descriptive Features
	temp_dfs = np.hstack((numeric_dfs.as_matrix(), vec_cat_dfs ))

	return temp_dfs

main()