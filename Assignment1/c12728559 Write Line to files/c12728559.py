# Student number : C12728559
# Student Name   : Maximilian Mihoc

import pandas as pd
import csv
from scipy import stats

# read feature names from the featurenames file and place them in a list.
featureNames = [line.rstrip() for line in open('./data/featurenames.txt', 'r')]
# remove empty elements from the list, if any
featureNames = [f for f in featureNames if f != '']
#print(featureNames)

Location = r'./data/DataSet.txt'
df = pd.read_csv(Location, names=featureNames)

###########################################
###### CONTINUOUS FEATURES ################

# open file to write Continuous Features
contFeatureFile = open('./data/c12728559CONT.csv', 'w', newline='')
outputWriterContFile = csv.writer(contFeatureFile)
outputWriterContFile.writerow(['Feature', 'Count', '% Miss.', 'Card.', 'Min', '1st Qrt.', 'Mean', 'Median', '3rd Qrt.', 'Max', 'Std. Dev.'])

#count Missing values
def findMissingValues(fList):
	count = 0;
	for val in fList:
		if val == ' ?':
			count += 1
	return count

# function to write data to file for each dataframe column/feature that represents
# a continuous feature
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

	#write line to file
	outputWriterContFile.writerow(contTableLine)

# Call the function for each Continuous Feature.
builtContinuousFeatureTable('AGE', df['age'])
builtContinuousFeatureTable('FNLWGT', df['fnlwgt'])
builtContinuousFeatureTable('CAPITAL_GAIN', df['capital-gain'])
builtContinuousFeatureTable('CAPITAL_LOSS', df['capital-loss'])
builtContinuousFeatureTable('HOURS_PER_WEEK', df['hours-per-week'])
contFeatureFile.close()

###########################################
###### CATEGORICAL FEATURES ###############
# open file to write Categorical Features
catFeatureFile = open('./data/c12728559CAT.csv', 'w', newline='')
outputWriterCatFile = csv.writer(catFeatureFile)
outputWriterCatFile.writerow(['Feature', 'Count', '% Miss.', 'Card.', 'Mode', 'Mode Freq.', 'Mode %', '2nd Mode', '2nd Mode Freq.', '2nd Mode %'])

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

	# the next line can be used to verify the mode order and frequency
	#print(dfColumn.value_counts())
	#write line to file
	outputWriterCatFile.writerow(catTableLine)

builtCardinalityFeatureTable('WORKCLASS', df['workclass'])
builtCardinalityFeatureTable('EDUCATION', df['education'])
builtCardinalityFeatureTable('EDUCATION-NUM', df['education-num'])
builtCardinalityFeatureTable('MARITAL-STATUS', df['marital-status'])
builtCardinalityFeatureTable('OCCUPATION', df['occupation'])
builtCardinalityFeatureTable('RELATIONSHIP', df['relationship'])
builtCardinalityFeatureTable('RACE', df['race'])
builtCardinalityFeatureTable('SEX', df['sex'])
builtCardinalityFeatureTable('NATIVE-COUNTRY', df['native-country'])
builtCardinalityFeatureTable('TARGET', df['target'])
catFeatureFile.close()