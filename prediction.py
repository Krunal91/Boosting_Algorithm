import pandas as pd
import numpy as np
import math


# predict the input test record from previous train model
def predict(record, summary_train_model, parent, childNode):
    nodeDetails = summary_train_model.loc[
        summary_train_model['NodeNum'] == childNode, ["NodeType", "SplitColumn", "SplitValue", "Child0", "Child1",
                                                      "Parent", "ClassLable"]]
    # stopping condition. When LN is found, return the control.
    if (str(nodeDetails.iloc[0, 0]) == 'LN'):
        return nodeDetails.iloc[0, 6]
    else:
        splitColumn = nodeDetails.iloc[0, 1]
        splitValue = nodeDetails.iloc[0, 2]
        childNode = np.nan

        if (record[splitColumn] == splitValue):
            childNode = nodeDetails.iloc[0, 3]

        else:
            childNode = nodeDetails.iloc[0, 4]

        parent = nodeDetails.iloc[0, 5]
        return predict(record, summary_train_model, parent, childNode)


# predict the input test record from previous train model
def predict_value(summary_train_model, test):

    summary_train_model = summary_train_model
    predictedValueDf = pd.DataFrame(np.nan, index=[0], columns=['predictedValue'])
    for n in range(len(test)):
        predClassLable = predict(test.iloc[n, :], summary_train_model, 0, 0)
        predictedValueDf = predictedValueDf.append(pd.Series([predClassLable], index=['predictedValue']),
                                                   ignore_index=True)
    predictedValueDf = predictedValueDf.iloc[1:, :]
    return predictedValueDf


## Accuracy of the decision tree
def tree_accuracy(predicted_value):
    positive = predicted_value[predicted_value["predictedValue"] == 1]
    negative = predicted_value[predicted_value["predictedValue"] == 0]
    True_positive = len(positive[(positive["predictedValue"] == positive["test_label"])])
    True_negative = len(negative[(negative["predictedValue"] == negative["test_label"])])
    False_positive = len(positive[(positive["predictedValue"] != positive["test_label"])])
    False_negative = len(negative[(negative["predictedValue"] != negative["test_label"])])
    accuracy = (True_positive + True_negative) / (len(predicted_value))
    # print("Confusion matrix of a bag....")
    # print("                Predicted:Negative(0)   Predicted: Positive(1)")
    # print("Actual: Negative           {}            {}".format(True_negative, False_positive))
    # print("Actual: Positive           {}            {}".format(False_negative, True_positive))
    # print("Accuracy of the Decision Tree : {}".format(accuracy))
    # print("Misclassification error of the Decision Tree : {}".format(1 - accuracy))
    return accuracy