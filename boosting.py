from dataset_details import dataset_read_boosting
from prediction import *
import math
import subprocess
import sys
import os
import shutil

# function for boosting

def boosting_tree(maxdepth,numtrees,datapath):

    # Initialize empty dataframes and list
    all_prediction = pd.DataFrame()
    weights_list = []
    alpha_weight = []
    final_prediction = []
    test_prediction = pd.DataFrame()
    train_temp, test_temp, depth_temp = dataset_read_boosting("firstread", datapath)

    # for loop for number of trees we want to use
    for i in range(int(numtrees)):
        if i == 0:

            # load the data and assign the weights to observations
            train, test, depth = dataset_read_boosting(maxdepth,datapath)
            weights = 1 / len(train)
            train['weight'] = weights
            weights_list.append(train.weight)
            train.to_csv(str(datapath + "agaricuslepiotatrain1.csv"), index=False)
            currDir = os.path.dirname(os.path.realpath('__file__'))
            if not os.path.exists('temp'):
                os.makedirs('temp')
            else:
                shutil.rmtree('temp')
                os.makedirs('temp')

            # call the subprocess for the decision tree with the weights assigned
            theproc = subprocess.Popen([sys.executable, "decision_tree_boosting.py",str(maxdepth),datapath])
            theproc.communicate()
            os.chdir(os.getcwd() + "/temp/")
            predicted = pd.read_csv("predicted_train.csv")
            predicted_test = pd.read_csv("predicted_test.csv")
            os.chdir("..")

            # append the predicted labels in one dataframe
            all_prediction[str(i)] = predicted["predictedValue"]
            test_prediction[str(i)] = predicted_test["predictedValue"]

            # calculate error,alpha and assign weights
            error = 1 - tree_accuracy(predicted)
            alpha = (1 / 2) * math.log((1 - error) / error)
            alpha_weight.append(alpha)
            old_weights = weights_list.pop()
            new_weights = calculate_weights(old_weights, predicted, alpha)
            train["weight"] = new_weights
            weights_list.append(train.weight)
            train.to_csv(str(datapath + "agaricuslepiotatrain1.csv"), index=False)

        else:
            # call the subprocess for the decision tree with the weights assigned
            theproc = subprocess.Popen([sys.executable, "decision_tree_boosting.py",str(maxdepth),datapath])
            theproc.communicate()
            os.chdir(os.getcwd() + "/temp/")
            predicted = pd.read_csv("predicted_train.csv")
            predicted_test = pd.read_csv("predicted_test.csv")
            os.chdir("..")

            # append the predicted labels in one dataframe
            all_prediction[str(i)] = predicted["predictedValue"]
            test_prediction[str(i)] = predicted_test["predictedValue"]

            # calculate error,alpha and assign weights
            error = 1 - tree_accuracy(predicted)
            alpha = (1 / 2) * math.log((1 - error) / error)
            alpha_weight.append(alpha)
            old_weights = weights_list.pop()
            new_weights = calculate_weights(old_weights, predicted, alpha)
            train["weight"] = new_weights
            weights_list.append(train.weight)
            train.to_csv(str(datapath + "agaricuslepiotatrain1.csv"), index=False)

    # replace 0 with -1 so we can calculate the final label based on the alpha
    test_prediction = test_prediction.replace(to_replace=0, value=-1)
    test_prediction = test_prediction * alpha_weight
    total = test_prediction.sum(1)

    for each in total:
        if each >= 0:
            final_prediction.append(1)
        else:
            final_prediction.append(0)
    predicted_test["test_label"] = test["class"]
    predicted_test["predictedValue"] = final_prediction
    tree_accuracy_final(predicted_test)
    train_temp.to_csv(str(datapath + "agaricuslepiotatrain1.csv"), index=False)
    test_temp.to_csv(str(datapath + "agaricuslepiotatest1.csv"), index=False)
    shutil.rmtree('temp')
    shutil.rmtree('__pycache__')
    return

# this function calculates the new weights based on the last decision tree cycle
def calculate_weights(old_weights, predicted, alpha):
    weights = []
    for each in range(len(old_weights)):
        flag = predicted.loc[each, "predictedValue"] == predicted.loc[each, "test_label"]

        if flag:
            weights.append(old_weights[each] * math.exp(-alpha))
        else:
            weights.append(old_weights[each] * math.exp(alpha))

    return weights


# final tree accuracy with the confusion matrix
def tree_accuracy_final(predicted_value):
    positive = predicted_value[predicted_value["predictedValue"] == 1]
    negative = predicted_value[predicted_value["predictedValue"] == 0]
    True_positive = len(positive[(positive["predictedValue"] == positive["test_label"])])
    True_negative = len(negative[(negative["predictedValue"] == negative["test_label"])])
    False_positive = len(positive[(positive["predictedValue"] != positive["test_label"])])
    False_negative = len(negative[(negative["predictedValue"] != negative["test_label"])])
    accuracy = (True_positive + True_negative) / (len(predicted_value))
    print("Confusion matrix of final boosting....")
    print("                Predicted:Negative(0)   Predicted: Positive(1)")
    print("Actual: Negative           {}            {}".format(True_negative, False_positive))
    print("Actual: Positive           {}            {}".format(False_negative, True_positive))
    print("Accuracy of the  Boosting : {}".format(accuracy))
    print("Misclassification error of the Boosting : {}".format(1 - accuracy))
    return accuracy