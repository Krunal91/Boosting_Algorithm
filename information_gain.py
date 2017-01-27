import pandas as pd
import math
import numpy as np


def information_gain(train_data):
    names = train_data.columns[1:len(train_data.columns)]
    entropy = []
    feature_value = []
    for columns in names:
        levels = train_data[columns].unique()
        info_list = []
        for category in levels:
            info = main_entropy(train_data, columns, category)
            info_list.append(info)
        info_array = np.asarray(info_list)
        feature_val = levels[np.argmin(info_array)]
        entropy.append(min(info_list))
        feature_value.append(feature_val)

    best_split = np.asarray(entropy).argmin()

    feature_name = names[best_split]
    level_name = feature_value[best_split]
    return feature_name, level_name


def main_entropy(train_data, columns, category):
    left = train_data.loc[train_data[columns] == category, :]
    right = train_data.loc[train_data[columns] != category, :]

    ## for left part
    pos_le = len(left.loc[left["class"] == 1, :])
    neg_le = len(left) - pos_le
    total_le = pos_le + neg_le

    if pos_le != 0:
        pos_ex = pos_le / total_le * (math.log2(pos_le / total_le))
    else:
        pos_ex = -0
    if neg_le != 0:
        neg_ex = neg_le / total_le * (math.log2(neg_le / total_le))
    else:
        neg_ex = -0

    entropy_left = -(pos_ex + neg_ex)

    # for right part
    pos_re = len(right.loc[right["class"] == 1, :])
    neg_re = len(right) - pos_re
    total_re = pos_re + neg_re

    if pos_re != 0:
        pos_right = pos_re / total_re * (math.log2(pos_re / total_re))
    else:
        pos_right = -0
    if neg_re != 0:
        neg_right = neg_re / total_re * (math.log2(neg_re / total_re))
    else:
        neg_right = -0

    entropy_right = -(pos_right + neg_right)

    ## Weighted entropy
    total = total_re + total_le
    exp_entropy = ((pos_le + neg_le) / total) * entropy_left + ((pos_re + neg_re) / total) * entropy_right

    return exp_entropy


'''Boosting information gains'''

def information_gain_boosting(train_data):
    names = train_data.columns[1:(len(train_data.columns)-1)]
    entropy = []
    feature_value = []
    for columns in names:
        levels = train_data[columns].unique()
        info_list = []
        for category in levels:
            info = main_entropy_boosting(train_data, columns, category)
            info_list.append(info)
        info_array = np.asarray(info_list)
        feature_val = levels[np.argmin(info_array)]
        entropy.append(min(info_list))
        feature_value.append(feature_val)

    best_split = np.asarray(entropy).argmin()

    feature_name = names[best_split]
    level_name = feature_value[best_split]
    return feature_name, level_name


def main_entropy_boosting(train_data, columns, category):
    left = train_data.loc[train_data[columns] == category, :]
    right = train_data.loc[train_data[columns] != category, :]

    ## for left part
    pos_le = len(left.loc[left["class"] == 1, :])
    neg_le = len(left) - pos_le
    total_le = pos_le + neg_le

    if pos_le != 0:
        pos_ex = pos_le / total_le * (math.log2(pos_le / total_le))
    else:
        pos_ex = -0
    if neg_le != 0:
        neg_ex = neg_le / total_le * (math.log2(neg_le / total_le))
    else:
        neg_ex = -0

    entropy_left = -(pos_ex + neg_ex)

    # for right part
    pos_re = len(right.loc[right["class"] == 1, :])
    neg_re = len(right) - pos_re
    total_re = pos_re + neg_re

    if pos_re != 0:
        pos_right = pos_re / total_re * (math.log2(pos_re / total_re))
    else:
        pos_right = -0
    if neg_re != 0:
        neg_right = neg_re / total_re * (math.log2(neg_re / total_re))
    else:
        neg_right = -0

    entropy_right = -(pos_right + neg_right)

    ## Weighted entropy
    total = total_re + total_le
    exp_entropy = ((pos_le + neg_le) / total) * entropy_left + ((pos_re + neg_re) / total) * entropy_right

    return exp_entropy



