'''
This script should have read, write and delete permission.
'''
import pandas as pd
from os import path
import math
import numpy as np


# read data sets
def dataset_read_bagging(mxdpth,datapath):
    # user parameter editing section started
    depth = int(mxdpth)

    if (datapath[-1] != "/"):
        datapath = datapath + "/"

    train_path = datapath + "agaricuslepiotatrain1.csv"
    train_separator = ","

    test_path = datapath + "agaricuslepiotatest1.csv"
    test_separator = ","

    maxdepth_parameters = [depth]  # if given -1, default decision tree with the depth 1,2,4,...16 build, else given depth size tree build
   # user parameter editing section ended. Don't edit anything further


    train = pd.read_table(train_path,
                          sep=train_separator)

    test = pd.read_table(test_path,
                         sep=test_separator)
    #test = test.drop([0, 8],1)  # put column number inside [ ],if you don't want to include columns in model building, else delete this line
    #test.columns = col_names

    if('bruises?-no' in train.columns.values):
        train = train.drop(['bruises?-no'], axis=1)
        test = test.drop(['bruises?-no'], axis=1)

    if('weight' in train.columns.values):
        train = train.drop(['weight'], axis=1)

    if ('bruises?-bruises' in train.columns.values):
        train = train.rename(columns={"bruises?-bruises": "class"})
        test = test.rename(columns={"bruises?-bruises": "class"})

        class_loc = train.columns.get_loc('class')
        cols = train.columns.tolist()

        class_loc_test = test.columns.get_loc('class')
        cols_test = test.columns.tolist()

        cols[0], cols[class_loc] = cols[class_loc], cols[0]
        cols_test[0], cols_test[class_loc_test] = cols_test[class_loc_test], cols_test[0]

        train = train[cols]
        test = test[cols_test]

    col_names = train.columns.values

    #take bootstrap resamples
    train.reset_index()
    train = train.iloc[list(np.random.choice(len(train), len(train))), :]

    ## Convert the columns datatype into category
    for i in range(len(col_names)-1):
        train[col_names[i]] = train[col_names[i]].astype("category")
        test[col_names[i]] = test[col_names[i]].astype("category")
    return train, test, maxdepth_parameters


# read data sets
def dataset_read_boosting(mxdpth,datapath):
    # user parameter editing section started


    # use below if you are reading train data from local file
    if (datapath[-1] != "/"):
        datapath = datapath + "/"

    train_path = datapath + "agaricuslepiotatrain1.csv"
    train_separator = ","

    # use below if you are reading test data from local file
    test_path = datapath + "agaricuslepiotatest1.csv"
    test_separator = ","

      # if given -1, default decision tree with the depth 1,2,4,...16 build, else given depth size tree build

    # maxdepth_parameters = [1]  # if given -1, default decision tree with the depth 1,2,4,...16 build, else given depth size tree build
    # col_names = ["class", "a1", "a2", "a3", "a4", "a5",
    #            "a6"]  # change the column name in this format. 'class' attribute should be first column of dataset with name 'class'
    # user parameter editing section ended. Don't edit anything further

    train = pd.read_table(train_path, sep=train_separator)
    test = pd.read_table(test_path, sep=test_separator)

    if(mxdpth =="firstread"):
        return train, test, "1"

    maxdepth_parameters = [int(mxdpth)]

    if ('bruises?-no' in train.columns.values):
        train = train.drop(['bruises?-no'], axis=1)

    if ('bruises?-no' in test.columns.values):
        test = test.drop(['bruises?-no'], axis=1)

    if ('bruises?-bruises' in train.columns.values):
        train = train.rename(columns={"bruises?-bruises": "class"})

    if ('bruises?-bruises' in test.columns.values):
        test = test.rename(columns={"bruises?-bruises": "class"})

    if ('class' in train.columns.values):
        class_loc = train.columns.get_loc('class')
        cols = train.columns.tolist()
        cols[0], cols[class_loc] = cols[class_loc], cols[0]
        train = train[cols]

    if ('class' in test.columns.values):
        class_loc_test = test.columns.get_loc('class')
        cols_test = test.columns.tolist()
        cols_test[0], cols_test[class_loc_test] = cols_test[class_loc_test], cols_test[0]
        test = test[cols_test]

    col_names = train.columns.values
    # train = train.drop([0, 8],1)  # put column number inside [ ],if you don't want to include columns in model building, else delete this line


    ## Convert the columns datatype into category
    for i in range((len(col_names) - 1)):
        train[col_names[i]] = train[col_names[i]].astype("category")
        test[col_names[i]] = test[col_names[i]].astype("category")
    return train, test, maxdepth_parameters