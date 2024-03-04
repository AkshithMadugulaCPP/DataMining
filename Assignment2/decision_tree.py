# -------------------------------------------------------------------------
# AUTHOR: Akshith Madugula
# FILENAME: decision_tree
# SPECIFICATION: decision tree with gini
# FOR: CS 5990 (Advanced Data Mining) - Assignment #2
# TIME SPENT: 30mins
# -----------------------------------------------------------*/

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataSets = ['cheat_training_1.csv', 'cheat_training_2.csv']

def update_row(row):
    new_row = []
    for ind, val in enumerate(row):
        if ind == 0:  # Refund, converting Yes to 1 and No to 0
            new_row.append(1 if val == "Yes" else 0)
        elif ind == 1:  # Marital Status, one hot encoding-> ['Single', 'Divorced', 'Married'] 
            new_row.extend(marital_encoding[val])
        elif ind == 2:  # Taxable Income, converting to Float
            new_row.append(float(val.replace("k", "")))
    return new_row


for ds in dataSets:

    X = []
    Y = []

    df = pd.read_csv('Assignment2/'+ds, sep=',', header=0)   #reading a dataset eliminating the header (Pandas library)
    data_training = np.array(df.values)[:,1:] #creating a training matrix without the id (NumPy library)

    #transform the original training features to numbers and add them to the 5D array X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
    #Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [2, 0, 1, 0, 100], ...]]. The feature Marital Status must be one-hot-encoded and Taxable Income must
    #be converted to a float.
    X = []
    marital_encoding = {"Single": [1, 0, 0], "Married": [0, 1, 0], "Divorced": [0, 0, 1]}
    for row in data_training:
        X.append(update_row(row))        
    #transform the original training classes to numbers and add them to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    Y = [1 if val == "Yes" else 2 for val in data_training[:, 3]]

    accuracy_list = []
    #loop your training and test tasks 10 times here
    for i in range (10):

        #fitting the decision tree to the data by using Gini index and no max_depth
        clf = tree.DecisionTreeClassifier(criterion = 'gini', max_depth=None)
        clf = clf.fit(X, Y)
 
        #plotting the decision tree
        tree.plot_tree(clf, feature_names=['Refund', 'Single', 'Divorced', 'Married', 'Taxable Income'], class_names=['Yes','No'], filled=True, rounded=True)
        plt.show()
 
        #read the test data and add this data to data_test NumPy
        #--> add your Python code here
        cheat_test = pd.read_csv('Assignment2/cheat_test.csv', sep=',', header=0)
        data_test = np.array(cheat_test.values)[:,1:]

        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for data in data_test:
            #transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
            #class_predicted = clf.predict([[1, 0, 1, 0, 115]])[0], where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            updated_row = update_row(data)
            predicted_class = clf.predict([updated_row])[0]
    
            #compare the prediction with the true label (located at data[3]) of the test instance to start calculating the model accuracy.
            actual_class = 2
            if data[-1] == "Yes":
                actual_class = 1
            
            if actual_class == predicted_class:
                if predicted_class == 1:
                    tp+=1
                else:
                    tn+=1
            else:
                if predicted_class ==1:
                    fp+=1
                else:
                    fn+=1
        accuray = (tp+tn)/(tp+tn+fp+fn)
        accuracy_list.append(accuray)
        # find the average accuracy of this model during the 10 runs (training and test set)
    final_accuracy = np.average(accuracy_list)

    #print the accuracy of this model during the 10 runs (training and test set).
    #your output should be something like that: final accuracy when training on cheat_training_1.csv: 0.2
    print(f"final accuracy when training on {ds}: {final_accuracy}")



