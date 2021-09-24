#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

db_training = []
db_test = []
test_predictions = []
X = []
Y = []
training_file = 'weather_training.csv'
test_file = 'weather_test.csv'

#reading the training data
with open(training_file, 'r') as csvfile:
   reader = csv.reader(csvfile)
   for i, row in enumerate(reader):
      if i > 0: #skipping the header
          db_training.append(row)
          X.append([row[1], row[2], row[3], row[4]])
          Y.append(row[5])
#transform the original training features to numbers and add to the 4D array X. For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
outlookToInt = {
    'Sunny' : 1,
    'Overcast' : 2,
    'Rain' : 3
}

temperatureToInt = {
    'Hot' : 1,
    'Mild' : 2,
    'Cool' : 3
}

humidityToInt = {
    'High' : 1,
    'Normal' : 2
}

windToInt = {
    'Weak' : 1,
    'Strong' : 2
}

for data in X:
    data[0] = outlookToInt[data[0]]
    data[1] = temperatureToInt[data[1]]
    data[2] = humidityToInt[data[2]]
    data[3] = windToInt[data[3]]

#transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
for data in Y:
    data = 1 if data == 'Yes' else 0

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the data in a csv file
with open(test_file, 'r') as csvfile:
   reader = csv.reader(csvfile)
   for i, row in enumerate(reader):
      if i > 0: #skipping the header
          db_test.append([row[1], row[2], row[3], row[4]])
          test_predictions.append(row)

#transform test data to integer values.
for data in db_test:
    data[0] = outlookToInt[data[0]]
    data[1] = temperatureToInt[data[1]]
    data[2] = humidityToInt[data[2]]
    data[3] = windToInt[data[3]]

#printing the header os the solution
print ("Day".ljust(15) + "Outlook".ljust(15) + "Temperature".ljust(15) + "Humidity".ljust(15) + "Wind".ljust(15) + "PlayTennis".ljust(15) + "Confidence".ljust(15))

#use your test samples to make probabilistic predictions.
for i in range(len(db_test)):
    predicted = clf.predict_proba([[db_test[i][0], db_test[i][1], db_test[i][2], db_test[i][3]]])[0]
    if predicted[0] >= 0.75:
        test_predictions[i][5] = 'No'
        test_predictions[i].append(str(predicted[0])[:4] )
    elif predicted[1] >= 0.75:
        test_predictions[i][5] = 'Yes'
        test_predictions[i].append(str(predicted[1])[:4] )
    else:
        test_predictions[i].append('?')

#print results
for data in test_predictions:
    print (data[0].ljust(15) + data[1].ljust(15) + data[2].ljust(15) + data[3].ljust(15) + data[4].ljust(15) + data[5].ljust(15) + data[6])
    


