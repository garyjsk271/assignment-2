#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#reading the data in a csv file
X_total = []
Y_total = []
with open('binary_points.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
         X_total.append([float(row[0]), float(row[1]) ])
         Y_total.append(1.0 if row[2] == '+' else 0.0)


#loop your data to allow each instance to be your test set
test_index = 0
nWrongPredictions = 0
for i, instance in enumerate(db):

    #add the training features to the 2D array X removing the instance that will be used for testing in this iteration. For instance, X = [[1, 3], [2, 1,], ...]]. Convert each feature value to
    # float to avoid warning messages
    X = list(X_total)
    del X[i]

    #transform the original training classes to numbers and add to the vector Y removing the instance that will be used for testing in this iteration. For instance, Y = [1, 2, ,...]. Convert each
    #  feature value to float to avoid warning messages
    Y = list(Y_total)
    del Y[i]

    #store the test sample of this iteration in the vector testSample
    testSample = [X_total[i][0], X_total[i][1], Y_total[i]]

    #fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #use your test sample in this iteration to make the class prediction. For instance:
    class_predicted = clf.predict([[testSample[0], testSample[1]]])[0]

    #compare the prediction with the true label of the test instance to start calculating the error rate.
    if class_predicted != testSample[2]:
        nWrongPredictions += 1

#print the error rate
error_rate = nWrongPredictions/len(X_total)
print("LOOCV error rate: {}".format(error_rate))






