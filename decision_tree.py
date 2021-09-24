#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']
testfile = 'contact_lens_test.csv'


for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append(row)

    #transform the original training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    ageToInt =  { 
        "Young" : 1,
        "Prepresbyopic" : 2, 
        "Presbyopic" : 3 
    }

    spectaclePrescriptionToInt = { 
        "Myope" : 1, 
        "Hypermetrope" : 2 
    }

    astigmatismToInt = { 
        "Yes" : 1, 
        "No" : 2 
    }

    tearProductionRateToInt = { 
        "Reduced" : 1, 
        "Normal" : 2 
    }
    
    for data in dbTraining:
        data[0] = ageToInt[data[0]]
        data[1] = spectaclePrescriptionToInt[data[1]]
        data[2] = astigmatismToInt[data[2]]
        data[3] = tearProductionRateToInt[data[3]]
        X.append(data[:len(data) - 1] )

    #transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    for data in dbTraining:
        Y.append(1 if data[-1] == "Yes" else 2)

    #loop your training and test tasks 10 times here
    lowest_accuracy = 1.0
    for i in range (10):

        #fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
        clf = clf.fit(X, Y)

        #read the test data and add this data to dbTest
        dbTest = []
        with open(testfile, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i > 0: #skipping the header
                    dbTest.append(row)
       
        nCorrectPredictions = 0 
        for data in dbTest:
            #transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
            #class_predicted = clf.predict([[3, 1, 2, 1]])[0]           -> [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            data[0] = ageToInt[data[0]]
            data[1] = spectaclePrescriptionToInt[data[1]]
            data[2] = astigmatismToInt[data[2]]
            data[3] = tearProductionRateToInt[data[3]]
            data[4] = astigmatismToInt[data[4]]
           
            class_predicted = clf.predict([[data[0], data[1], data[2], data[3]]])[0]
           
            #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            if class_predicted == data[4]:
                nCorrectPredictions += 1

        #find the lowest accuracy of this model during the 10 runs (training and test set)
        accuracy = nCorrectPredictions / len(dbTest)
        lowest_accuracy = min(lowest_accuracy, accuracy)
        
    #print the lowest accuracy of this model during the 10 runs (training and test set) and save it.
    #your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    print("final accuracy when training on {file}: {acc}".format(file = ds, acc = lowest_accuracy) )



