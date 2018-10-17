# Student number : C12728559
# Student Name   : Maximilian Mihoc

import pandas as pd
import csv
from scipy import stats

# read feature names from the featurenames file and place them in a list.
featureNames = [line.rstrip() for line in open('./data/featurenames.txt', 'r')]
# remove empty elements from the list, if any
featureNames = [f for f in featureNames if f != '']
print(featureNames)

Location = r'./data/queries.txt'
df = pd.read_csv(Location, names=featureNames)

###########################################
###### CONTINUOUS FEATURES ################
# create a function to calculate values for the Continuous Features
# create a data frame for the continuous features and then write that on a file

contFeatureNames = ['Feature', 'Count', '% Miss.', 'Card.', 'Min', '1st Qrt.', 'Mean', 'Median', '3rd Qrt.', 'Max', 'Std. Dev.']

#count Missing values
def findMissingValues(fList):
	count = 0;
	for val in fList:
		if val == ' ?':
			count += 1
	return count

# function to calculate values for each dataframe column/feature that represents a continuous feature
def builtContinuousFeatureTable(fName, dfColumn):
	contTableLine = []
	contTableLine.append(fName)
	contTableLine.append(len(dfColumn))
	contTableLine.append(round(findMissingValues(dfColumn)*100/len(dfColumn), 1)) 
	# calculate cardinality of feature by counting the unique values in the feature 
	# and tacking out the missing values represented as ' ?'
	tempList = dfColumn.unique().tolist()
	#print(tempList)
	if ' ?' in tempList:
		tempList.remove(' ?')
	contTableLine.append(len(tempList)) #cardinality
	contTableLine.append(dfColumn.min())
	contTableLine.append(dfColumn.quantile(.25)) # 1st Qrt
	contTableLine.append(round(dfColumn.mean(), 1)) 
	contTableLine.append(dfColumn.median()) #median
	contTableLine.append(dfColumn.quantile(.75)) # 3rd Qrt
	contTableLine.append(dfColumn.max()) 
	contTableLine.append(round(dfColumn.std(), 1))

	return contTableLine

contFeatureData = [ 
builtContinuousFeatureTable('AGE', df['age']),
builtContinuousFeatureTable('BALANCE', df['balance']),
builtContinuousFeatureTable('day', df['day']),
builtContinuousFeatureTable('duration', df['duration']),
builtContinuousFeatureTable('campaign', df['campaign']),
builtContinuousFeatureTable('pdays', df['pdays']),
builtContinuousFeatureTable('previous', df['previous'])
]
continuousFeatureDataFrame = pd.DataFrame(data = contFeatureData, columns = contFeatureNames)
continuousFeatureDataFrame.to_csv('./data/c12728559CONT.csv', index=False, header=True)

###########################################
###### CATEGORICAL FEATURES ###############
# create a function to calculate values for the Categorical Features
# create a data frame for the continuous features and then write that on a file

catFeatureNames = ['Feature', 'Count', '% Miss.', 'Card.', 'Mode', 'Mode Freq.', 'Mode %', '2nd Mode', '2nd Mode Freq.', '2nd Mode %']

def countModeFrequency(fList, modeTerm):
	count = 0;
	for val in fList:
		if val == modeTerm:
			count += 1
	return count

def builtCardinalityFeatureTable(fName, dfColumn):
	catTableLine = []
	catTableLine.append(fName)
	catTableLine.append(len(dfColumn))
	missingValuesCount = findMissingValues(dfColumn)
	catTableLine.append(round(missingValuesCount*100/len(dfColumn), 1))
	# calculate cardinality of feature by counting the unique values in the feature 
	# and tacking out the missing values represented as ' ?'
	tempList = dfColumn.unique().tolist()
	#print(tempList)
	if ' ?' in tempList:
		tempList.remove(' ?')
	catTableLine.append(len(tempList)) #cardinality

	# First Mode columns calculation
	firstMode = dfColumn.mode().iloc[0]
	catTableLine.append(firstMode) # write first mode to file
	# calculate and write first mode freq.
	firstModeFreq = countModeFrequency(dfColumn, firstMode)
	catTableLine.append(firstModeFreq) 

	# find first Mode %
	catTableLine.append( round((firstModeFreq * 100) / (len(dfColumn) - missingValuesCount), 1) )

	# Second Mode Columns Calculation
	# copy dataframe into a new dataframe
	dfColumnCopy = dfColumn.copy()
	# remove all rows that contain the first mode
	dfColumnCopy = dfColumnCopy[dfColumnCopy != firstMode]
	# find the mode of the new dataframe, this should give me the second mode
	secondMode = dfColumnCopy.mode().iloc[0]
	catTableLine.append(secondMode)
	secondModeFreq = countModeFrequency(dfColumnCopy, secondMode)
	catTableLine.append(secondModeFreq)

	catTableLine.append( round((secondModeFreq * 100) / (len(dfColumn) - missingValuesCount), 1) )
	return catTableLine
	
catFeatureData = [builtCardinalityFeatureTable('job', df['job']),
builtCardinalityFeatureTable('marital', df['marital']),
builtCardinalityFeatureTable('education', df['education']),
builtCardinalityFeatureTable('default', df['default']),
builtCardinalityFeatureTable('housing', df['housing']),
builtCardinalityFeatureTable('loan', df['loan']),
builtCardinalityFeatureTable('contact', df['contact']),
builtCardinalityFeatureTable('month', df['month']),
builtCardinalityFeatureTable('poutcome', df['poutcome'])]
categoricalFeatureDataFrame = pd.DataFrame(data = catFeatureData, columns = catFeatureNames)
categoricalFeatureDataFrame.to_csv('./data/c12728559CAT.csv', index=False, header=True)


