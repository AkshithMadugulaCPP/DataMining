#-------------------------------------------------------------------------
# AUTHOR: Akshith Madugula
# FILENAME: Knn
# SPECIFICATION: KNN to classify temp
# FOR: CS 5990- Assignment #3
# TIME SPENT: 45
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
#11 classes after discretization
classes = [i for i in range(-22, 40, 6)]

#update the training class values according to the discretization (11 values only)
def discretize(instance):
    try:
        prev=-100
        for c in classes:
            if prev < instance["Temperature (C)"] <= c:
                instance["Temperature (C)"] = c
            prev = c
        if instance["Temperature (C)"] > c:
            instance["Temperature (C)"] = c
    except:
        print("unable to discretize", instance)
    return instance

#defining the hyperparameter values of KNN
k_values = [i for i in range(1, 20)]
p_values = [1, 2]
w_values = ['uniform', 'distance']

#reading the training data
training_path = r"Assignment3\weather_training.csv"
data_training = pd.read_csv(training_path, sep=',', header=0)   #reading a dataset eliminating the header (Pandas library)

discreted_training_data = data_training.apply(discretize, axis = 1)

# The columns that we will be making predictions with.
y_training = np.array(discreted_training_data["Temperature (C)"]).astype(dtype='int')
X_training = np.array(discreted_training_data.drop(["Temperature (C)","Formatted Date"], axis=1).values)

#reading the test data
#hint: to convert values to float while reading them -> np.array(df.values)[:,-1].astype('f')
testing_path = r"Assignment3\weather_test.csv"
test_df = pd.read_csv(testing_path, sep=',', header=0)

#update the test class values according to the discretization (11 values only)
discreted_testing_data = test_df.apply(discretize, axis=1)
y_test = discreted_testing_data["Temperature (C)"].astype(dtype='int')
X_test = discreted_testing_data.drop(["Temperature (C)","Formatted Date"], axis=1).values

#loop over the hyperparameter values (k, p, and w) ok KNN
highest_accuracy = 0
for k in k_values:
    for p in p_values:
        for w in w_values:
            current_accuracy = 0
            #fitting the knn to the data
            clf = KNeighborsClassifier(n_neighbors=k, p=p, weights=w)
            clf = clf.fit(X_training, y_training)

            #make the KNN prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously, use zip()
            #Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
            #to make a prediction do: clf.predict([x_testSample])
            #the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values.
            #to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
            for test_x , test_y in zip(X_test,y_test):
                preditcted = clf.predict(np.array([test_x]))

                # given that the prediction should be considered correct if the output value is [-15%,+15%] from the real output values.
                approx = (abs(preditcted[0] - test_y)/test_y)*100
                if -15<=approx<=15:
                    current_accuracy+=1

            #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
            #with the KNN hyperparameters. Example: "Highest KNN accuracy so far: 0.92, Parameters: k=1, p=2, w= 'uniform'"
            current_accuracy /= len(y_test)
            if highest_accuracy < current_accuracy:
                highest_accuracy = current_accuracy
                print(f"Highest KNN accuracy so far: {current_accuracy}, Parameters: k={k}, p={p}, w= '{w}'")
