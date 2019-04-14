# -*-coding:utf8-*-
"""
author:jacquelin
date:2019
test the performance of the model in the test data for gbdt and gbdt_lr
"""
from __future__ import division
import numpy as np
import xgboost as xgb
import train as TA
import util.get_feature_num as GF
from scipy.sparse import csc_matrix
import math
import sys



def get_test_data(test_file,feature_num_file):
    '''

    :param test_file:
    :param feature_num_file:
    :return:
    '''
    total_feature_num = GF.get_feature_num (feature_num_file)
    test_label = np.genfromtxt(test_file, dtype= np.float32, delimiter=",", usecols= -1)
    feature_list = range(total_feature_num)
    test_feature = np.genfromtxt(test_file, dtype= np.float32, delimiter=",", usecols= feature_list)
    return test_feature, test_label


def predict_by_tree(test_feature,tree_model):
    '''
    predict by xgboost model
    '''
    predict_list=tree_model.predict(xgb.DMatrix(test_feature))
    return predict_list


def predict_by_lr_gbdt(test_feature,mix_tree_model,mix_lr_coef,tree_info):
    '''
    predict by mix model
    '''
    tree_leaf=mix_tree_model.predict(xgb.DMatrix(test_feature),pred_leaf=True)
    (tree_depth,tree_num,step_size)=tree_info
    total_feature_list=TA.get_gbdt_and_lr_feature(tree_leaf,tree_depth=tree_depth,tree_num=tree_num)
    result_list=np.dot(csc_matrix(mix_lr_coef),total_feature_list.tocsc().T).toarray()[0]
    sigmoid_ufunc=np.frompyfunc(sigmoid,1,1)
    return sigmoid_ufunc(result_list)


def sigmoid(x):
    """
    sigmoid function
    """
    return 1/(1+math.exp(-x))


def get_auc(predict_list, test_label):
    """
    Args:
        predict_list: model predict score list
        test_label: label of  test data
    auc = (sum(pos_index)-pos_num(pos_num + 1)/2)/pos_num*neg_num
    """
    total_list = []
    for index in range(len(predict_list)):
        predict_score = predict_list[index]
        label = test_label[index]
        total_list.append((label, predict_score))
    sorted_total_list = sorted(total_list, key = lambda ele:ele[1])
    neg_num = 0
    pos_num = 0
    count = 1
    total_pos_index = 0
    for zuhe in sorted_total_list:
        label, predict_score = zuhe
        if label == 0:
            neg_num += 1
        else:
            pos_num += 1
            total_pos_index += count
        count += 1
    auc_score = (total_pos_index - (pos_num)*(pos_num + 1)/2) / (pos_num*neg_num)
    print ("auc:%.5f" %(auc_score))


def get_accuary(predict_list, test_label):
    """
    Args:
        predict_list: model predict score list
        test_label: label of test data
    """
    score_thr = 0.5
    right_num = 0
    for index in range(len(predict_list)):
        predict_score = predict_list[index]
        if predict_score >= score_thr:
            predict_label = 1
        else:
            predict_label = 0
        if predict_label == test_label[index]:
            right_num += 1
    total_num = len(predict_list)
    accuary_score = right_num/total_num
    print ("accuary:%.5f" %(accuary_score))



def run_check_core(test_feature,test_label,model,score_func):
    '''
    xgboost模型在测试集上的核心
    :param test_feature:
    :param test_label:
    :param tree_model:
    :param score_func: predict_by_tree
    '''
    predict_list=score_func(test_feature,model)
    get_auc(predict_list,test_label)
    get_accuary(predict_list,test_label)



def run_check(test_file,tree_model_file,feature_num_file):
    '''
    :param test_file:
    :param tree_model_file:
    :param feature_num_file:
    '''
    test_feature,test_label=get_test_data(test_file,feature_num_file)
    tree_model=xgb.Booster(model_file=tree_model_file)
    run_check_core(test_feature,test_label,tree_model,predict_by_tree)


def run_check_lr_gbdt_core(test_feature,test_label,mix_tree_model,mix_lr_coef,tree_info,score_func):
    '''
    得出auc评价分数的运行脚本
    :param test_feature:
    :param test_label:
    :param mix_tree_model:
    :param mix_lr_coef:
    :param tree_info:
    :param score_func:
    :return:
    '''
    predict_list=score_func(test_feature,mix_tree_model,mix_lr_coef,tree_info)
    get_auc(predict_list,test_label)
    get_accuary(predict_list,test_label)



def run_check_lr_gbdt(test_file,tree_mix_model_file,lr_coef_mix_model_file,feature_num_file):

    '''
    :param test_file:
    :param tree_mix_model:
    :param mix_lr_coef:
    :param tree_info:
    :return:
    '''
    test_feature,test_label=get_test_data(test_file,feature_num_file)
    mix_tree_model=xgb.Booster(model_file=tree_mix_model_file)
    mix_lr_coef=np.genfromtxt(lr_coef_mix_model_file,dtype=np.float32,delimiter=',')
    tree_info=TA.get_mix_model_tree_info()
    run_check_lr_gbdt_core(test_feature,test_label,mix_tree_model,mix_lr_coef,tree_info,predict_by_lr_gbdt)


if __name__=='__main__':
    #run_check('../data/test_file','../data/xgb.model','../data/feature_num_file')
    #run_check_lr_gbdt('../data/test_file','../data/xgb_mix_model','../data/lr_coef_mix_model','../data/feature_num_file')
    if len(sys.argv) == 4:
        test_file=sys.argv[1]
        tree_model=sys.argv[2]
        feature_num_file=sys.argv[3]
        run_check(test_file,tree_model,feature_num_file)
    elif len(sys.argv) == 5:
        test_file=sys.argv[1]
        tree_mix_model=sys.argv[2]
        lr_coef_mix_model=sys.argv[3]
        feature_num_file=sys.argv[4]
        run_check_lr_gbdt(test_file,tree_mix_model,lr_coef_mix_model,feature_num_file)
    else:
        print ("check gbdt model usage: python xx.py test_file  tree_model feature_num_file")
        print ("check lr_gbdt model usage: python xx.py test_file tree_mix_model lr_coef_mix_model feature_num_file")
        sys.exit()