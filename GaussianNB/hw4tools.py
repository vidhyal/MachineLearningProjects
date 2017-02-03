
"""
Python file provided by the TA for hw3, modified according to needs to process input data for hw4

"""

import numpy as np

def LoadSpamData(filename = "spambase.data"):
  unprocessed_data_file = file(filename,'r')
  unprocessed_data = unprocessed_data_file.readlines()  #read the file line by line
  labels = []
  features = []

  for line in unprocessed_data:
    feature_vector = []

    csvalues = line.split(',')   #get all the comma separated values

    #Iterate across all the features (last element is the label)
    for element in csvalues[:-1]:
      feature_vector.append(float(element))

    #Add the new feature vector to the features list and new label to label_list
    features.append(feature_vector)
    labels.append(int(csvalues[-1]))

  #return List of features with its list of corresponding labels
  return features, labels


def SeparateDataSets(features, labels):

  # Assume input features and labels are in order and all examples with label 1 appear before those with label 0
  
  count_0 = labels.count(0)
  count_1 = labels.count(1)
  if (count_0 %2) == 0:
      train_count_0 = count_0/2
      test_count_0 = train_count_0+ count_0/2
  else:
      train_count_0 = count_0//2
      test_count_0 = train_count_0+ count_0//2 +1
  if (count_1 %2) == 0:
      train_count_1 = count_1/2
      test_count_1 = train_count_1+ count_1/2
  else:
      train_count_1 = count_1//2
      test_count_1 = train_count_1+ count_1//2 +1
  
  "return separate positive and negative training sets and a test set containing both positive and negative examples"
  
  return features[:train_count_1], features[-train_count_0:],labels[:train_count_1],  labels[-train_count_0:], features[train_count_1:test_count_1]+features[-test_count_0:-train_count_0],labels[train_count_1:test_count_1] + labels[-test_count_0:-train_count_0]
  


  

def FormatData(features, labels, filename):
  # final data format :  label 0:feature0, 1:feature1, 2:feature2, etc...
  if len(features) != len(labels):
    raise Exception("Number of samples and labels must match")
  dat_file = file(filename,'w')
  for s in range(len(features)):

    if labels[s]==0:
      line="-1 "
    else:
      line="1 "

    for f in range(len(features[s])):
      line +="%i:%f " % (f+1 , features[s][f])
    line += "\n"
    dat_file.write(line)
  dat_file.close()


def PrintToFile(param, name, filename):
  data_file = file(filename,'a')
  data_file.write(name+"\n")
  for s in range(len(param)):
    data_file.write('%18s ' %param[s])
  data_file.write('\n')
  data_file.close()
  
  


def main():

  features, labels = LoadSpamData()
  train_features_1,train_features_0, train_labels_1, train_labels_0, test_features, test_labels = SeparateDataSets(features, labels)
  train_features_0, train_labels_0 = np.asarray(train_features_0), np.asarray(train_labels_0) " convert to array"

  train_features_1, train_labels_1 = np.asarray(train_features_1), np.asarray(train_labels_1)
  total = len(train_labels_0) +len(train_labels_1)
  neg = len(train_labels_0)
  pos = len(train_labels_1)
  test_features, test_labels = np.asarray(test_features), np.asarray(test_labels)

  means_0 = np.mean(train_features_0, axis = 0)
  means_1 = np.mean(train_features_1, axis = 0)
  stdDev_0 = np.std(train_features_0, axis = 0)
  stdDev_1 = np.std(train_features_1, axis = 0)
  PrintToFile(means_0, "means_0", "Gaussian_parameters.txt")
  PrintToFile(stdDev_0, "stdDev_0", "Gaussian_parameters.txt")
  PrintToFile(means_1, "means_1", "Gaussian_parameters.txt")
  
  PrintToFile(stdDev_1, "stdDev_1", "Gaussian_parameters.txt")
  writeProb = file("prob.txt",'w')
  writeProb.write("total=%5d\npos=%5d\nneg=%5d" %(total,pos, neg))
  writeProb.close()
  FormatData(test_features, test_labels,"test_features.data")

if __name__ == "__main__":
  main()


