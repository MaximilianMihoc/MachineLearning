import math

class decisionnode:
  def __init__(self,col=-1,value=None,results=None,tb=None,fb=None):
    self.col=col
    self.value=value
    self.results=results
    self.tb=tb
    self.fb=fb


# Divides a set on a specific column. Can handle numeric
# or nominal values
def divideset(rows,column,value):
   # Make a function that tells us if a row is in
   # the first group (true) or the second group (false)
   split_function=None
   if isinstance(value,int) or isinstance(value,float):
      split_function=lambda row:row[column]>=value
   else:
      split_function=lambda row:row[column]==value

   # Divide the rows into two sets and return them
   set1=[row for row in rows if split_function(row)]
   set2=[row for row in rows if not split_function(row)]
   return (set1,set2)

# Create counts of possible results (the last column of
# each row is the result)
def uniquecounts(rows):
   results={}
   for row in rows:
      # The result is the last column
      r=row[len(row)-1]
      if r not in results: results[r]=0
      results[r]+=1
   return results

# Entropy is the sum of p(x)log(p(x)) across all
# the different possible results
def entropy(rows):
   from math import log
   log2=lambda x:log(x)/log(2)
   results=uniquecounts(rows)
   # Now calculate the entropy
   ent=0.0
   for r in results.keys():
      p=float(results[r])/len(rows)
      ent=ent-p*log2(p)
   return ent


def buildtree(rows,scoref=entropy):
  if len(rows)==0: return decisionnode()

  current_score=scoref(rows)

  # Set up some variables to track the best criteria
  best_gain=0.0
  best_criteria=None
  best_sets=None

  # (1.2) Divide up the group by the possible values of
  # each attribute
  column_count=len(rows[0])-1
  for col in range(0,column_count):
    # (1.2.1) Generate the list of different values in
    # this column
    column_values={}
    for row in rows:
       column_values[row[col]]=1
    # (1.2.2) Now try dividing the rows up for each value
    # in this column
    for value in column_values.keys():
      (set1,set2)=divideset(rows,col,value)

      # (1.3) Calculate the Information gain of the new groups.
      # Information gain is the difference between the
      # current entropy and the weighted-average entropy of
      # the two groups.
      p=float(len(set1))/len(rows)
      gain=current_score-p*scoref(set1)-(1-p)*scoref(set2)

      # (1.4) Select the attribute with the highest
      # information gain as the one to split on
      if gain>best_gain and len(set1)>0 and len(set2)>0:
        best_gain=gain
        best_criteria=(col,value)
        best_sets=(set1,set2)

  # (2) Create the branches
  # -----------------------
  # (2.1) Determines if the branch can be divided further
  # or if it has reached a solid conclusion.
  # A branch stops dividing when the information gain
  # from splitting a node is not more than zero.
  if best_gain>0:
    # (2.2) If one of the new branches can be divided,
    # the algorithm creates two branches corresponding
    # to true or false for the splitting condition.
    # It does this by recursively calling the algorithm for
    # each branch
    trueBranch=buildtree(best_sets[0],scoref)
    falseBranch=buildtree(best_sets[1],scoref)
    # The results of the calls on each subset are attached to the
    # True and False branches of the nodes, eventually constructing
    # the entire tree.
    return decisionnode(col=best_criteria[0],value=best_criteria[1],
                        tb=trueBranch,fb=falseBranch)
  else:
    # If the best pair of subsets doesn't have a lower
    # weighted-average entropy than the current set, that
    # branch ends and the counts of all the possible
    # outcomes  are stored.
    return decisionnode(results=uniquecounts(rows))

#
# Reading Data from a File:
#
# Whichever classifier we use we need to:
# get the data set into a data structure.
#
# The following code will open the file,
# parse it line by line and return a 2D
# list structure containing the
# data set.

def getInputData(datafile):
  data=[line.split(',') for line in open(datafile, 'r')]
  #print(data[0])
  for i in range(0,len(data)):
      # convert the numeric data type from
      # strings to floats and ints.
      # print(data[i][0] + " | " + data[i][1] + " | " + data[i][2] )
       data[i][0] = int(data[i][0])
      # strip the \n characters from the
      # right side of the target variable
       data[i][2] = data[i][2].rstrip('\n')
  return data

def classify(observation,tree):
  if tree.results!=None:
    return tree.results
  else:
    v=observation[tree.col]
    branch=None
    if isinstance(v,int) or isinstance(v,float):
      if v>=tree.value: branch=tree.tb
      else: branch=tree.fb
    else:
      if v==tree.value: branch=tree.tb
      else: branch=tree.fb
    return classify(observation,branch)

def convertDecTreeOutput2Label(output):
    maxcount = 0
    maxlabel = ''
    for k in output.keys():
        if output[k]>maxcount:
            maxcount = output[k]
            maxlabel = k
    return maxlabel




# Note: we need to cast some of the data 
# from strings to numeric values
#

def main():
      data = getInputData('trainingset.txt')
      tests = getInputData('queries.txt')

      queries = []
      for i in range(len(tests)):
          q = tests[i][:-1]
          queries.append(q)

      tree=buildtree(data)
      predictions = []
      for i in range(len(queries)):
          o = classify(queries[i],tree)
          p = convertDecTreeOutput2Label(o)
          predictions.append(p)

      correct = 0
      for i in range(len(predictions)):
          print("Test query: " + str(tests[i]) + ", Predicted target: " + predictions[i])
          if predictions[i] == tests[i][-1]:
              correct += 1
      print("Percentage correct: " + str("%.4f" % round(float(correct)/len(predictions),4)))

#      print('ai2.txt = ')
#      print(data)
#      print('test_set.txt =')
#      print(test)

      print ("finsihed processing!")

main()
