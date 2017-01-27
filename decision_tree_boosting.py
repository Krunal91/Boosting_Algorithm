from prediction import *
from dataset_details import dataset_read_boosting
from information_gain import *
import os
import sys

# Load training data and assign column names
train, test, depth = dataset_read_boosting(sys.argv[1],sys.argv[2])
# generate unique sequence to give unique id to each node of decision tree
seqValue = 0


def sequenceGenerator():
    global seqValue
    seqValue = seqValue + 2
    return seqValue


# recursively call this function to generate the tree
def grow_tree(dataset, nodeOwnNum, parent, child0, child1, depth1, depth2):
    global summary_train_model
    global seqValue
    global tree_depth
    seqValue = seqValue + 2

    identical_feature = dataset.iloc[:, 1:].drop_duplicates().shape[0]
    datasetClases = dataset.iloc[:, 0].unique()

    # stop developing tree if dataset target variable has only one class or all the dataset points have same value
    if (len(datasetClases) == 1 or identical_feature == 1 or depth1 >= tree_depth or depth2 >= tree_depth):

        if (depth1 >= tree_depth):
            summary_train_model = summary_train_model.append(pd.Series([nodeOwnNum, np.nan, np.nan, np.argmax(
                [len(dataset[dataset['class'] == 0]['class']), len(dataset[dataset['class'] == 1]['class'])]), "LN",
                                                                        parent, np.nan, np.nan], index=cols),
                                                             ignore_index=True)
        elif (depth1 >= tree_depth):
            summary_train_model = summary_train_model.append(pd.Series([nodeOwnNum, np.nan, np.nan, np.argmax(
                [len(dataset[dataset['class'] == 0]['class']), len(dataset[dataset['class'] == 1]['class'])]), "LN",
                                                                        parent, np.nan, np.nan], index=cols),
                                                             ignore_index=True)
        else:
            summary_train_model = summary_train_model.append(
                pd.Series([nodeOwnNum, np.nan, np.nan, dataset.iloc[0, 0], "LN", parent, np.nan, np.nan], index=cols),
                ignore_index=True)

    else:
        datasetSplitSummary = information_gain_boosting(dataset)  # returns best split
        datasetColName = datasetSplitSummary[0]
        datasetSplitValue = datasetSplitSummary[1]
        summary_train_model = summary_train_model.append(
            pd.Series([nodeOwnNum, datasetColName, datasetSplitValue, np.nan, "IN", parent, child0, child1],
                      index=cols), ignore_index=True)

        split0 = dataset[dataset[datasetColName] == datasetSplitValue]
        split1 = dataset[dataset[datasetColName] != datasetSplitValue]

        if (split0.shape[0] != 0):
            depth1 = depth1 + 1
            grow_tree(split0, child0, nodeOwnNum, sequenceGenerator(), sequenceGenerator(), depth1, depth2)

        if (split1.shape[0] != 0):
            depth2 = depth2 + 1
            grow_tree(split1, child1, nodeOwnNum, sequenceGenerator(), sequenceGenerator(), depth1, depth2)
    parent = parent + 1


## Initialize the depth list. If no input from user, default depth from 1,2,4,6..,16 would be performed
total_accuracy = []
if (depth[0] == -1):
    depth = [1] + list(range(2, 17, 2))

# iterate this loop for each depth
for i in depth:
    #print("---------------------------------------Depth: " + str(i) + "-------------------------------------------")

    # initialize parameters
    cols = ['NodeNum', 'SplitColumn', 'SplitValue', 'ClassLable', 'NodeType', 'Parent', 'Child0', 'Child1']
    summary_train_model = pd.DataFrame(np.nan, index=[0], columns=cols)
    nodeOwnNum = 0
    child0 = 1
    child1 = 2
    parent = 0
    depth1 = 0
    depth2 = 0
    tree_depth = i

    # grow the tree
    print("Growing Tree....")
    grow_tree(train, nodeOwnNum, parent, child0, child1, depth1, depth2)
    summary_train_model = summary_train_model.iloc[1:, :]
    print("Tree has been grown successfully. Now Predicting....")

    # predict the class of train record
    predictedValues = predict_value(summary_train_model, train)

    predictedValues = predictedValues.set_index([list(range(len(train)))])
    predictedValues["test_label"] = train["class"]
    os.chdir(os.getcwd() + "/temp/")
    predictedValues.to_csv("predicted_train.csv",index = False)

    # test record
    predictedValues = predict_value(summary_train_model, test)
    predictedValues = predictedValues.set_index([list(range(len(test)))])
    predictedValues["test_label"] = test["class"]
    predictedValues.to_csv("predicted_test.csv", index=False)
    os.chdir("..")